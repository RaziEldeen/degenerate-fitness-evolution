"""
Validate theoretical claims in the Langevin note (section 1.2 / 5.2).

The note's Euler-Maruyama update is:

    x_{t+1} = x_t + η ∇f(x_t) + √(2η) σ ξ_t,   ξ_t ~ N(0, I)

For the 2D landscape f(x, y) = F_max - ½ x² g(y), integrating out the fast x
process gives an effective slow dynamics on y. The note's predictions are:

    P*(y) ∝ √(2 - η g(y)) / √g(y)            (finite-η)
        vs. 1/√g(y)                          (continuous limit; Cauchy when g=(1+y²)²)

    ⟨ẏ⟩ = -σ² g'(y) / [g(y)(2 - η g(y))]    (finite-η)
        vs. -σ² g'(y) / [2 g(y)]             (continuous)

    ⟨x²⟩|y = 2σ² / [g(y)(2 - η g(y))]       (finite-η)
        vs. σ² / g(y)                        (continuous)

Figures:
    1. Smooth g(y) = (1+y²)² — stationary distribution, effective drift, OU variance.
    2. Singular g(y) = y² — non-normalizability and ⟨y²⟩(t) drift.
    3. Effective potentials V_eff(y) = -ln P*(y), theory only.
"""

import argparse
from functools import partial

import jax

# Enable float64 for numerical stability — singular landscape can produce
# very large excursions in x near y=0.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from objective_function import create_landscape_2d, create_smooth_landscape_2d
from plot_utils import (
    setup_style,
    save_figure,
    style_axis,
    create_figure,
    add_subplot_labels,
)
from langevin_tss_validity import diagnose_langevin_tss_validity


# Canonical parameters from the note (section 5.2 quantile table)
SIGMA = 0.05
ETA = 0.01
N_WALKERS = 1500
N_STEPS = 500_000
STRIDE = 100  # subsampling interval for trajectory storage

QUANTILE_LEVELS = (0.50, 0.75, 0.90, 0.95, 0.99)


# =============================================================================
# Euler-Maruyama simulator
# =============================================================================

def make_em_runner(grad_func, eta, sigma, stride, num_chunks):
    """JIT-compiled Euler-Maruyama runner.

    Implements x_{t+1} = x_t + η ∇f(x_t) + √(2η) σ ξ. Runs `num_chunks * stride`
    steps per walker, returning positions sampled every `stride` steps.

    Returns a callable (keys, initial_population) -> (final_population, trajectories)
    where `trajectories` has shape (n_walkers, num_chunks + 1, 2).
    """
    noise_scale = jnp.sqrt(2.0 * eta) * sigma

    def step(state, _):
        pos, k = state
        k, sub = jax.random.split(k)
        xi = jax.random.normal(sub, shape=(2,), dtype=pos.dtype)
        new_pos = pos + eta * grad_func(pos) + noise_scale * xi
        return (new_pos, k), None

    def chunk(state, _):
        new_state, _ = jax.lax.scan(step, state, None, length=stride)
        return new_state, new_state[0]

    def run_one(k, init):
        (final_pos, _), traj = jax.lax.scan(chunk, (init, k), None, length=num_chunks)
        full = jnp.concatenate([init[None, :], traj], axis=0)
        return final_pos, full

    return jax.jit(jax.vmap(run_one))


# =============================================================================
# Theory
# =============================================================================

def smooth_g(y):
    return (1.0 + y**2)**2


def smooth_g_prime(y):
    return 4.0 * y * (1.0 + y**2)


def singular_g(y):
    return y**2


def singular_g_prime(y):
    return 2.0 * y


def y_cut_smooth(eta):
    """Smallest y > 0 where η · g(y) = 2, for g(y) = (1+y²)²."""
    return float(np.sqrt(np.sqrt(2.0 / eta) - 1.0))


def y_cut_singular(eta):
    """Smallest y > 0 where η · g(y) = 2, for g(y) = y²."""
    return float(np.sqrt(2.0 / eta))


def stationary_pdf_finite_unnorm(y, eta, g_func):
    """Finite-η stationary PDF (unnormalized): √(2 - η g(y)) / √g(y)."""
    g = g_func(y)
    inside = np.maximum(2.0 - eta * g, 0.0)
    return np.sqrt(inside) / np.sqrt(g)


def stationary_pdf_cauchy(y):
    """Continuous-limit Cauchy density 1/[π(1+y²)] for g(y) = (1+y²)²."""
    return 1.0 / (np.pi * (1.0 + y**2))


def normalize_pdf(y_grid, pdf_unnorm):
    """Normalize PDF on a uniform y_grid by trapezoidal integration."""
    # NumPy 2.0 renamed np.trapz → np.trapezoid; fall back for older versions.
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return pdf_unnorm / trap(pdf_unnorm, y_grid)


def quantiles_of_abs_y(y_grid, pdf_normalized, p_levels):
    """Compute |Y| quantiles given a symmetric PDF.

    Assumes pdf_normalized integrates to 1 over y_grid and is symmetric.
    """
    mask = y_grid >= 0
    y_pos = y_grid[mask]
    pdf_pos = pdf_normalized[mask]
    # P(|y| ≤ a) = 2 · ∫_0^a P*(y) dy
    cdf = np.concatenate([
        [0.0],
        2.0 * np.cumsum(0.5 * (pdf_pos[:-1] + pdf_pos[1:]) * np.diff(y_pos)),
    ])
    cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
    return np.interp(p_levels, cdf, y_pos)


def cauchy_quantiles_of_abs_y(p_levels):
    """Closed-form |Y| quantiles under Cauchy 1/[π(1+y²)]: F⁻¹(p) = tan(p π/2)."""
    return np.tan(np.asarray(p_levels) * np.pi / 2.0)


# =============================================================================
# Figure 1: Smooth landscape — stationary, drift, OU variance
# =============================================================================

def figure_1_smooth_landscape():
    """Three-panel: P*(y), ⟨ẏ⟩(y), ⟨x²⟩(y) for the smooth landscape."""
    print("\n" + "=" * 70)
    print("Figure 1: Smooth landscape g(y) = (1+y²)²")
    print("=" * 70)

    F_MAX = 5.0
    _, _, grad_func = create_smooth_landscape_2d(F_MAX)
    Y_cut = y_cut_smooth(ETA)
    print(f"  Parameters: σ={SIGMA}, η={ETA}, N={N_WALKERS}, T={N_STEPS}, stride={STRIDE}")
    print(f"  Y_cut (finite-η support boundary) = {Y_cut:.4f}")

    # Initialization: y uniform on (-2, 2) within the bulk; x = 0.
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    y0 = jax.random.uniform(init_key, (N_WALKERS,), minval=-2.0, maxval=2.0)
    x0 = jnp.zeros(N_WALKERS)
    initial_pop = jnp.stack([x0, y0], axis=1)

    num_chunks = N_STEPS // STRIDE
    print(f"  Running EM ({num_chunks} chunks × {STRIDE} steps)…")
    keys = jax.random.split(key, N_WALKERS)
    runner = make_em_runner(grad_func, ETA, SIGMA, STRIDE, num_chunks)
    _, trajectories = runner(keys, initial_pop)
    trajectories.block_until_ready()
    print(f"  Trajectory tensor: {trajectories.shape}, dtype={trajectories.dtype}")

    # 25% burn-in
    n_burn = num_chunks // 4
    print(f"  Burn-in: first {n_burn} chunks ({n_burn * STRIDE} steps)")
    traj_ss = np.asarray(trajectories[:, n_burn:, :])

    # Pool samples for (a) histogram
    y_samples = traj_ss[..., 1].ravel()
    finite_mask = np.isfinite(y_samples)
    n_inf = (~finite_mask).sum()
    if n_inf:
        print(f"  WARNING: {n_inf} non-finite y samples discarded")
    y_samples = y_samples[finite_mask]
    print(f"  Pooled samples (after burn-in): {y_samples.size:,}")

    # For drift and ⟨x²⟩ binning, use chunk-to-chunk pairs.
    y_pre = traj_ss[:, :-1, 1].ravel()
    y_post = traj_ss[:, 1:, 1].ravel()
    x_pre = traj_ss[:, :-1, 0].ravel()
    pair_mask = np.isfinite(y_pre) & np.isfinite(y_post) & np.isfinite(x_pre)
    y_pre = y_pre[pair_mask]
    y_post = y_post[pair_mask]
    x_pre = x_pre[pair_mask]
    drift_per_unit_time = (y_post - y_pre) / (STRIDE * ETA)
    x_sq = x_pre**2

    # === Theory grids ===
    y_grid = np.linspace(-Y_cut * 0.9999, Y_cut * 0.9999, 4000)
    pdf_finite_norm = normalize_pdf(y_grid, stationary_pdf_finite_unnorm(y_grid, ETA, smooth_g))
    pdf_cauchy = stationary_pdf_cauchy(y_grid)

    # Fine grid for quantiles
    y_grid_q = np.linspace(-Y_cut * 0.9999, Y_cut * 0.9999, 200_001)
    pdf_finite_q = normalize_pdf(y_grid_q, stationary_pdf_finite_unnorm(y_grid_q, ETA, smooth_g))
    q_theory = quantiles_of_abs_y(y_grid_q, pdf_finite_q, list(QUANTILE_LEVELS))
    q_cauchy = cauchy_quantiles_of_abs_y(list(QUANTILE_LEVELS))
    q_emp = np.quantile(np.abs(y_samples), QUANTILE_LEVELS)

    print("\n  Quantile comparison for |y|:")
    print(f"  {'p':>5}  {'Empirical':>10}  {'Finite-η':>10}  {'Cauchy':>10}  "
          f"{'Δ_finη':>10}  {'Δ_cauchy':>10}")
    for p, qe, qt, qc in zip(QUANTILE_LEVELS, q_emp, q_theory, q_cauchy):
        rel_finite = 100.0 * (qe - qt) / qt
        rel_cauchy = 100.0 * (qe - qc) / qc
        print(f"  {p:>5.2f}  {qe:>10.4f}  {qt:>10.4f}  {qc:>10.4f}  "
              f"{rel_finite:>+9.2f}%  {rel_cauchy:>+9.2f}%")

    # === Binned drift (panel b) ===
    n_bins = 25
    bin_edges = np.linspace(-3.2, 3.2, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(y_pre, bin_edges) - 1

    drift_mean = np.full(n_bins, np.nan)
    drift_sem = np.full(n_bins, np.nan)
    x2_mean = np.full(n_bins, np.nan)
    x2_sem = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = (bin_idx == b)
        n = int(sel.sum())
        if n >= 200:
            d = drift_per_unit_time[sel]
            drift_mean[b] = d.mean()
            drift_sem[b] = d.std(ddof=1) / np.sqrt(n)
            x2 = x_sq[sel]
            x2_mean[b] = x2.mean()
            x2_sem[b] = x2.std(ddof=1) / np.sqrt(n)

    drift_finite_th = -SIGMA**2 * smooth_g_prime(bin_centers) / (
        smooth_g(bin_centers) * (2.0 - ETA * smooth_g(bin_centers))
    )
    drift_continuous_th = -SIGMA**2 * smooth_g_prime(bin_centers) / (2.0 * smooth_g(bin_centers))

    x2_finite_th = 2.0 * SIGMA**2 / (
        smooth_g(bin_centers) * (2.0 - ETA * smooth_g(bin_centers))
    )
    x2_continuous_th = SIGMA**2 / smooth_g(bin_centers)

    drift_res = drift_mean - drift_finite_th
    x2_res = x2_mean - x2_finite_th
    print("\n  Drift binned residual stats (empirical − finite-η theory):")
    print(f"    mean |residual|  = {np.nanmean(np.abs(drift_res)):.4e}")
    print(f"    max  |residual|  = {np.nanmax(np.abs(drift_res)):.4e}")
    print(f"    typical |drift| at |y|=2.0: {abs(drift_finite_th[np.argmin(np.abs(bin_centers - 2.0))]):.4e}")
    print("\n  ⟨x²⟩ binned residual stats (empirical − finite-η theory):")
    print(f"    mean |residual|  = {np.nanmean(np.abs(x2_res)):.4e}")
    print(f"    max  |residual|  = {np.nanmax(np.abs(x2_res)):.4e}")
    print(f"    typical ⟨x²⟩ at |y|=2.0: {x2_finite_th[np.argmin(np.abs(bin_centers - 2.0))]:.4e}")

    # === Plot ===
    fig, axes = create_figure(n_cols=3, width_per_panel=4.6, height_per_panel=4.2)

    # (a) Stationary distribution
    ax = axes[0]
    ax.hist(y_samples, bins=120, range=(-3.6, 3.6), density=True,
            color='#8c564b', alpha=0.45, label='Empirical (EM)', edgecolor='none')
    ax.plot(y_grid, pdf_finite_norm, color='black', lw=2.0, label='Finite-η theory')
    ax.plot(y_grid, pdf_cauchy, color='#d62728', lw=2.0, ls='--',
            label=r'Cauchy  $1/[\pi(1+y^{2})]$')
    for sign in (-1, 1):
        ax.axvline(sign * Y_cut, color='gray', ls=':', lw=0.9)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel('Density')
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(bottom=0)
    ax.set_title('Stationary distribution')
    ax.legend(loc='upper right', fontsize=9)
    style_axis(ax)

    # (b) Drift
    ax = axes[1]
    ax.errorbar(bin_centers, drift_mean, yerr=drift_sem, fmt='o',
                color='#8c564b', ms=4, lw=0.8, capsize=2, label='Empirical')
    ax.plot(bin_centers, drift_finite_th, color='black', lw=2.0, label='Finite-η theory')
    ax.plot(bin_centers, drift_continuous_th, color='#d62728', lw=2.0, ls='--',
            label='Continuous theory')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$\langle \dot y \rangle$')
    ax.set_title('Effective drift')
    ax.legend(loc='best', fontsize=9)
    style_axis(ax)

    # (c) ⟨x²⟩ at fixed y
    ax = axes[2]
    ax.errorbar(bin_centers, x2_mean, yerr=x2_sem, fmt='o',
                color='#8c564b', ms=4, lw=0.8, capsize=2, label='Empirical')
    ax.plot(bin_centers, x2_finite_th, color='black', lw=2.0, label='Finite-η theory')
    ax.plot(bin_centers, x2_continuous_th, color='#d62728', lw=2.0, ls='--',
            label='Continuous theory')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$\langle x^{2} \rangle\,\vert\,y$')
    ax.set_yscale('log')
    ax.set_title('OU variance at fixed $y$')
    ax.legend(loc='best', fontsize=9)
    style_axis(ax)

    add_subplot_labels(axes)
    fig.tight_layout()
    save_figure(fig, 'langevin_validation_smooth', format='jpeg')
    plt.close(fig)


# =============================================================================
# Figure 2: Singular landscape — TSS two-phase ⟨x²⟩ pattern
# =============================================================================
#
# Convention note (important): this figure uses the analysis.py /
# langevin_tss_validity.py convention where *x* is the slow/TSS coord
# and *y* is the fast coord — opposite to the rest of this file. The
# landscape f = F_max − ½ x² y² is symmetric in (x, y), so the choice
# is purely a labeling one; we follow the archived diagnostic so the
# figure can be compared 1:1 to the older `langevin_tss_validity_*.jpeg`
# plots committed under archive/older_langevin/. The §4.1 ⟨y²⟩ claim
# in the note's text corresponds to ⟨x²⟩ here.
# =============================================================================

def make_reduced_u_runner(sigma, dt, stride, num_chunks):
    """JIT-compiled EM runner for u(t) = y(t)² on the reduced 1D SDE.

    The adiabatically-reduced slow-y SDE for g(y) = y² is
        dy = -(σ²/y) dt + √(2)·σ dW.
    Itô's lemma gives d(y²) = 2√2·σ·y dW, i.e. a *driftless* martingale
        du = 2·sign(y)·√(2σ²·u) dW           (sign-conditional)
    whose unsigned form is the Bessel-squared SDE  du = 2√(2σ²·u) dW
    with u ≥ 0. Integrating directly in u removes the singular -σ²/y drift
    that destabilizes the y-form's EM near y≈0, and the martingale property
    is preserved in expectation per step: E[u_{n+1}|u_n] = u_n.

    Absorbing boundary at u=0: u_{n+1} ← max(0, u_n + step). This matches
    the underlying BES(0) process, which hits 0 in finite time a.s. and is
    absorbed there. By optional stopping, E[u(t)] = u₀ for all t even after
    some walkers have been absorbed (the surviving population concentrates
    on larger u to compensate for the zero contribution of absorbed ones —
    this is the §4.1 picture of "median collapses, heavy tails grow").
    """
    noise_coef_sq = 2.0 * sigma * sigma * dt

    def step(state, _):
        u, k = state
        k, sub = jax.random.split(k)
        xi = jax.random.normal(sub, shape=(), dtype=u.dtype)
        step_u = 2.0 * jnp.sqrt(noise_coef_sq * jnp.maximum(u, 0.0)) * xi
        new_u = jnp.maximum(0.0, u + step_u)  # absorbing at 0
        return (new_u, k), None

    def chunk(state, _):
        new_state, _ = jax.lax.scan(step, state, None, length=stride)
        return new_state, new_state[0]

    def run_one(k, init_u):
        (final_u, _), traj = jax.lax.scan(chunk, (init_u, k), None, length=num_chunks)
        full = jnp.concatenate([init_u[None], traj])
        return final_u, full

    return jax.jit(jax.vmap(run_one))


def figure_2_singular_landscape():
    """TSS validity diagnostic on f = F_max − ½ x² y². Two panels:

      (a) |x| distribution snapshots vs t — the §4.1 picture (median collapse
          while heavy tails grow), from the η=0.10 run.
      (b) Empirical ⟨x²⟩_pop(τ) over continuous time τ = η·t for
          η ∈ {0.01, 0.05, 0.10}. The §4.1 Itô martingale claim says ⟨x²⟩
          should stop decreasing once the fast mode has equilibrated; we
          read the plateau directly from the simulation (mean over the
          second half of each run).

    Initial distribution: (x₀, y₀) ~ Uniform([-3, 3]²), so ⟨x²⟩(0) = 3.
    The deterministic limit conserves x² − y², driving each walker to
    either x≈0 (collapse) or y≈0 (the y=0 manifold); for x₀, y₀ ~ U(-3, 3)
    iid the deterministic phase-1 prediction is
        ⟨x²⟩_plateau, det = ½·E[|x₀² − y₀²|] = 3/2.
    Noise + finite-η give η-dependent corrections.

    Uses `langevin_tss_validity.diagnose_langevin_tss_validity` (brought
    in from archive/older_langevin/).
    """
    print("\n" + "=" * 70)
    print("Figure 2: TSS validity on f = F_max − ½ x² y²")
    print("=" * 70)

    F_MAX = 10.0
    _, _, grad_func = create_landscape_2d(F_MAX)

    INIT_BOUND = 3.0
    POP = 1000

    # ---- Panel (b): vary σ at fixed η = 0.10 (the archive's params) ----
    ETA_B = 0.10
    N_ITER_B = 10_000  # matches the archived langevin_tss_validity_*.jpeg
    sigma_values = (0.02, 0.05, 0.10)

    print(f"  Panel (b): η = {ETA_B}, n_iter = {N_ITER_B}, init_bound = {INIT_BOUND}, "
          f"pop = {POP}, τ_end = {ETA_B * N_ITER_B}")
    print(f"  σ values = {sigma_values}")

    results_b = {}
    for i, sigma_b in enumerate(sigma_values):
        key = jax.random.PRNGKey(100 + i)
        out = diagnose_langevin_tss_validity(
            key=key,
            num_iterations=N_ITER_B,
            population_size=POP,
            mutation_std=sigma_b,
            learning_rate=ETA_B,
            grad_func=grad_func,
            init_bound=INIT_BOUND,
            num_snapshots=101,
        )
        burn = N_ITER_B // 2
        plateau_emp = float(np.mean(out['stats']['mean_x_sq'][burn:]))
        plateau_std = float(np.std(out['stats']['mean_x_sq'][burn:]))
        results_b[sigma_b] = {'out': out, 'plateau': plateau_emp,
                              'plateau_std': plateau_std}
        print(f"\n  [σ={sigma_b}]  ⟨x²⟩(0)={out['stats']['mean_x_sq'][0]:.4f}  "
              f"⟨x²⟩(τ_end)={out['stats']['mean_x_sq'][-1]:.4f}  "
              f"plateau (mean of t>n_iter/2) = {plateau_emp:.4f} ± {plateau_std:.4f}")

    # ---- Panel (a): separate run at η = 0.05, σ = 0.05, τ_end = 1000 ----
    ETA_A = 0.05
    SIGMA_A = 0.05
    N_ITER_A = int(round(1000 / ETA_A))  # 20_000
    print(f"\n  Panel (a): η = {ETA_A}, σ = {SIGMA_A}, n_iter = {N_ITER_A}")
    out_a = diagnose_langevin_tss_validity(
        key=jax.random.PRNGKey(200),
        num_iterations=N_ITER_A,
        population_size=POP,
        mutation_std=SIGMA_A,
        learning_rate=ETA_A,
        grad_func=grad_func,
        init_bound=INIT_BOUND,
        num_snapshots=101,
    )

    # ============ Plot ============
    fig, axes = create_figure(n_cols=2, width_per_panel=5.4, height_per_panel=4.2)

    # ---- (a) |x| histograms at τ ∈ {10, 100, 1000} from the η=0.05 run ----
    ax = axes[0]
    # Snapshots at τ=10, 100, 1000 → iterations 200, 2000, 20000.
    snapshot_steps = (int(10 / ETA_A), int(100 / ETA_A), N_ITER_A)
    snapshot_colors = ['#1f77b4', '#2ca02c', '#d62728']
    log_bins = np.logspace(-3, 1.5, 50)
    for t, color in zip(snapshot_steps, snapshot_colors):
        pop = out_a['snapshots'][t]
        abs_x = np.abs(pop[:, 0])
        abs_x = abs_x[abs_x > 0]
        tau_label = t * ETA_A
        ax.hist(abs_x, bins=log_bins, density=True, histtype='step',
                color=color, lw=2.0,
                label=rf'$t={t:,}$  ($\tau={tau_label:.0f}$)')
    # Initial U(0, 3) density reference (initial pop is U(-3, 3) per axis)
    ax.plot([1e-3, 3.0], [1.0/3.0, 1.0/3.0], color='gray', lw=1.5, ls=':',
            label=r'Initial $|x_0|\sim U(0,3)$ (density $=1/3$)')
    # P*(x) ∝ 1/|x| slope reference
    ax.plot(log_bins, log_bins[10] / log_bins,
            color='black', lw=1.5, ls='--',
            label=r'$\propto 1/|x|$ (TSS quasi-stationary)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$|x|$')
    ax.set_ylabel('Density')
    ax.set_title(rf'Distribution of $|x|$ vs $t$  (2D EM, $\eta={ETA_A}$, $\sigma={SIGMA_A}$)')
    ax.legend(loc='lower left', fontsize=8.5)
    style_axis(ax)

    # ---- (b) ⟨x²⟩_pop(τ) at η=0.10 for σ ∈ {0.02, 0.05, 0.10} ----
    ax = axes[1]
    color_map = {0.02: '#1f77b4', 0.05: '#2ca02c', 0.10: '#d62728'}
    for sigma_b in sigma_values:
        r = results_b[sigma_b]
        out = r['out']
        ax.plot(out['tau'], out['stats']['mean_x_sq'], color=color_map[sigma_b], lw=2.0,
                label=rf'$\sigma = {sigma_b}$  '
                      rf'(plateau $\approx {r["plateau"]:.3f}\pm{r["plateau_std"]:.3f}$)')

    ax.set_xlabel(r'continuous time  $\tau = \eta\,t$')
    ax.set_ylabel(r'$\langle x^{2} \rangle_{\mathrm{pop}}$')
    ax.set_yscale('log')
    ax.set_title(
        rf'$\langle x^{{2}}\rangle(\tau)$ at $\eta={ETA_B}$ '
        rf'(archive params): $\sigma$-sweep'
    )
    ax.legend(loc='best', fontsize=8.5)
    style_axis(ax)

    add_subplot_labels(axes)
    fig.tight_layout()
    save_figure(fig, 'langevin_validation_singular', format='jpeg')
    plt.close(fig)


# =============================================================================
# Figure 3: Effective potentials (theory only)
# =============================================================================

def figure_3_potentials():
    """V_eff(y) = -ln P*(y) for both landscapes — theory only, no simulation."""
    print("\n" + "=" * 70)
    print("Figure 3: Effective potentials V_eff(y) = -ln P*(y)")
    print("=" * 70)

    Y_cut_s = y_cut_smooth(ETA)
    Y_cut_d = y_cut_singular(ETA)
    print(f"  Smooth   Y_cut = {Y_cut_s:.4f}")
    print(f"  Singular Y_cut = {Y_cut_d:.4f}")

    fig, axes = create_figure(n_cols=2, width_per_panel=5.0, height_per_panel=4.2)

    # (a) Smooth landscape
    ax = axes[0]
    y_s = np.linspace(-Y_cut_s * 0.998, Y_cut_s * 0.998, 4000)
    g_s = smooth_g(y_s)
    V_cont_s = 0.5 * np.log(g_s)
    V_fin_s = 0.5 * np.log(g_s) - 0.5 * np.log(np.maximum(2.0 - ETA * g_s, 1e-12))
    ax.plot(y_s, V_cont_s, color='#d62728', lw=2.0, ls='--', label='Continuous')
    ax.plot(y_s, V_fin_s, color='black', lw=2.0, label='Finite-η')
    for sign in (-1, 1):
        ax.axvline(sign * Y_cut_s, color='gray', ls=':', lw=0.9)
    ax.text(Y_cut_s, ax.get_ylim()[0], f'  $Y_{{cut}}={Y_cut_s:.2f}$',
            va='bottom', ha='left', fontsize=9, color='gray')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$V_{\mathrm{eff}}(y)$')
    ax.set_title(r'Smooth $g(y) = (1+y^{2})^{2}$')
    ax.legend(loc='upper center', fontsize=9)
    style_axis(ax)

    # (b) Singular landscape
    ax = axes[1]
    y_d_pos = np.linspace(0.02, Y_cut_d * 0.998, 4000)
    g_d = singular_g(y_d_pos)
    V_cont_d = 0.5 * np.log(g_d)
    V_fin_d = 0.5 * np.log(g_d) - 0.5 * np.log(np.maximum(2.0 - ETA * g_d, 1e-12))
    # Plot symmetrically (V_eff is even)
    for sign, lbl_cont, lbl_fin in ((-1, None, None), (1, 'Continuous', 'Finite-η')):
        ax.plot(sign * y_d_pos, V_cont_d, color='#d62728', lw=2.0, ls='--', label=lbl_cont)
        ax.plot(sign * y_d_pos, V_fin_d, color='black', lw=2.0, label=lbl_fin)
    for sign in (-1, 1):
        ax.axvline(sign * Y_cut_d, color='gray', ls=':', lw=0.9)
    ax.text(Y_cut_d, ax.get_ylim()[0], f'  $Y_{{cut}}={Y_cut_d:.2f}$',
            va='bottom', ha='left', fontsize=9, color='gray')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$V_{\mathrm{eff}}(y)$')
    ax.set_title(r'Singular $g(y) = y^{2}$')
    ax.legend(loc='upper center', fontsize=9)
    style_axis(ax)

    add_subplot_labels(axes)
    fig.tight_layout()
    save_figure(fig, 'langevin_validation_potentials', format='jpeg')
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

FIGURES = {
    '1': ('Figure 1: Smooth landscape — stationary, drift, OU variance', figure_1_smooth_landscape),
    '2': ('Figure 2: Singular landscape — non-normalizability, ⟨y²⟩(t)', figure_2_singular_landscape),
    '3': ('Figure 3: Effective potentials (theory only)', figure_3_potentials),
}


def main():
    parser = argparse.ArgumentParser(
        description='Validate theoretical claims in the Langevin note.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python validate_langevin_note.py --fig 1\n'
            '  python validate_langevin_note.py --fig all\n'
        ),
    )
    parser.add_argument('--fig', '-f', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Which figure to produce (default: all).')
    args = parser.parse_args()

    setup_style()

    if args.fig == 'all':
        for key in ('1', '2', '3'):
            desc, func = FIGURES[key]
            print(desc)
            func()
    else:
        desc, func = FIGURES[args.fig]
        print(desc)
        func()

    print("\nDone.")


if __name__ == '__main__':
    main()
