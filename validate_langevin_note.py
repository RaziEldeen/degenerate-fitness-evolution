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
# Figure 2: Singular landscape — non-normalizability and ⟨y²⟩(t)
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
    """Two-panel: |y| histograms over time, and ⟨y²⟩(t) with three curves.

    Initial distribution y₀ ~ Uniform(-3, 3) gives ⟨y²⟩(0) = 3 — the
    conservation value of the continuous-limit Itô martingale (note §4.1).
    Three curves on the second panel:
      • Continuous limit (1D reduced SDE, du = 2√(2σ²·u) dW in u=y² coords)
        — expected to stay at ⟨y²⟩=3 (martingale).
      • Full 2D EM at η = 0.01 — expected slight decrease via the §4.2
        bracket 2σ²[1 − 2⟨1/(2−ηy²)⟩] < 0.
      • Full 2D EM at η = 0.05 — same effect, more pronounced.

    Panel (a) uses the 1D reduced SDE with |y₀| ~ Uniform(0, 3) (the same
    initial second moment) to illustrate the §4.1 picture of median
    collapsing toward 0 while heavy tails grow, with ⟨y²⟩ conserved.
    """
    print("\n" + "=" * 70)
    print("Figure 2: Singular landscape g(y) = y²")
    print("=" * 70)

    N = 10_000
    a_init, b_init = -3.0, 3.0
    y2_conserved = (a_init**2 + a_init * b_init + b_init**2) / 3.0  # = 3.0

    T_cont = 1000.0
    dt_1d = 0.01
    eta_values = (0.01, 0.05)

    print(f"  Initial distribution: y₀ ~ Uniform({a_init}, {b_init})")
    print(f"  Conservation value:   ⟨y²⟩(0) = (a² + ab + b²)/3 = {y2_conserved:.4f}")
    print(f"  Common horizon:       T_cont = {T_cont}")

    # ----------- Continuous limit: 1D reduced SDE in u = y² ----------------
    n_steps_1d = int(T_cont / dt_1d)
    stride_1d = 100
    num_chunks_1d = n_steps_1d // stride_1d

    key = jax.random.PRNGKey(1)
    key, k0, k1 = jax.random.split(key, 3)
    y0_1d = jax.random.uniform(
        k0, (N,), minval=a_init, maxval=b_init, dtype=jnp.float64
    )
    initial_u = y0_1d**2

    keys_1d = jax.random.split(k1, N)
    runner_1d = make_reduced_u_runner(SIGMA, dt_1d, stride_1d, num_chunks_1d)
    _, u_traj = runner_1d(keys_1d, initial_u)
    u_traj.block_until_ready()
    u_traj = np.asarray(u_traj)

    u_safe = np.where(np.isfinite(u_traj), u_traj, np.nan)
    y2_mean_1d = np.nanmean(u_safe, axis=0)
    t_axis_1d = np.arange(num_chunks_1d + 1) * stride_1d * dt_1d

    delta_1d = y2_mean_1d[-1] - y2_mean_1d[0]
    print(f"\n  [Continuous limit, dt={dt_1d}, n_steps={n_steps_1d}]")
    print(f"    ⟨y²⟩(0) = {y2_mean_1d[0]:.4f}  ⟨y²⟩(T) = {y2_mean_1d[-1]:.4f}  "
          f"Δ = {delta_1d:+.4f} ({100*delta_1d/y2_mean_1d[0]:+.2f} %)")

    # ----------- Finite-η: full 2D EM ----------------
    F_MAX = 5.0
    _, _, grad_func_2d = create_landscape_2d(F_MAX)

    results_2d = {}
    for i, eta in enumerate(eta_values):
        n_steps = int(T_cont / eta)
        stride = max(1, n_steps // 1000)
        num_chunks = n_steps // stride

        key_eta = jax.random.PRNGKey(100 + i)
        k_init, k_sim = jax.random.split(key_eta)
        y0_2d = jax.random.uniform(
            k_init, (N,), minval=a_init, maxval=b_init, dtype=jnp.float64
        )
        x0_2d = jnp.zeros(N, dtype=jnp.float64)
        initial_pop = jnp.stack([x0_2d, y0_2d], axis=1)

        keys_2d = jax.random.split(k_sim, N)
        runner_2d = make_em_runner(grad_func_2d, eta, SIGMA, stride, num_chunks)
        _, trajs = runner_2d(keys_2d, initial_pop)
        trajs.block_until_ready()

        y_traj = np.asarray(trajs)[..., 1]
        y2 = np.where(np.isfinite(y_traj), y_traj**2, np.nan)
        y2_mean = np.nanmean(y2, axis=0)
        t_axis = np.arange(num_chunks + 1) * stride * eta

        results_2d[eta] = (t_axis, y2_mean)
        delta = y2_mean[-1] - y2_mean[0]
        print(f"\n  [2D EM η={eta}, n_steps={n_steps}]")
        print(f"    ⟨y²⟩(0) = {y2_mean[0]:.4f}  ⟨y²⟩(T) = {y2_mean[-1]:.4f}  "
              f"Δ = {delta:+.4f} ({100*delta/y2_mean[0]:+.2f} %)")

    # ----------- Panel (a) snapshots from 1D reduced sim ----------------
    snapshot_steps = (1_000, 10_000, 100_000)
    snapshot_chunks = tuple(t // stride_1d for t in snapshot_steps)
    abs_y_traj = np.sqrt(np.maximum(u_traj, 0.0))
    snapshots = {t: abs_y_traj[:, c] for t, c in zip(snapshot_steps, snapshot_chunks)}
    print()
    for t, ys in snapshots.items():
        finite = np.isfinite(ys)
        print(f"    snapshot t={t:>6d}: median|y|={np.median(ys[finite]):.4f}, "
              f"p95|y|={np.quantile(ys[finite], 0.95):.4f}")

    # ============ Plot ============
    fig, axes = create_figure(n_cols=2, width_per_panel=5.4, height_per_panel=4.2)

    # (a) |y| histograms — 1D reduced sim, |y₀| ∼ U(0, 3)
    ax = axes[0]
    snapshot_colors = ['#1f77b4', '#2ca02c', '#d62728']
    log_bins = np.logspace(-3, 2, 60)
    for (t, ys), color in zip(snapshots.items(), snapshot_colors):
        abs_y = ys[np.isfinite(ys)]
        abs_y = abs_y[abs_y > 0]
        ax.hist(abs_y, bins=log_bins, density=True, histtype='step',
                color=color, lw=2.0, label=f't = {t:,}')

    # Initial uniform reference: U(0, 3) has density 1/3 on the support.
    ax.plot([1e-3, 3.0], [1.0/3.0, 1.0/3.0], color='gray', lw=1.5, ls=':',
            label=r'Initial $|y_0|\sim U(0,3)$ (density $=1/3$)')

    # Continuous-limit prediction P*(y) ∝ 1/|y|, anchored near y=1.
    ax.plot(log_bins, 1.0 / log_bins * log_bins[10],
            color='black', lw=1.5, ls='--',
            label=r'Continuous  $\propto 1/|y|$ (non-normalizable)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$|y|$')
    ax.set_ylabel('Density')
    ax.set_title(r'Distribution of $|y|$ vs $t$ (reduced 1D SDE)')
    ax.legend(loc='lower left', fontsize=8.5)
    style_axis(ax)

    # (b) ⟨y²⟩(t) — three curves overlaid
    ax = axes[1]
    ax.plot(t_axis_1d, y2_mean_1d, color='#1f77b4', lw=2.0,
            label=r'Continuous limit (1D SDE)')
    color_map = {0.01: '#2ca02c', 0.05: '#d62728'}
    for eta in eta_values:
        t_axis, y2_mean = results_2d[eta]
        ax.plot(t_axis, y2_mean, color=color_map[eta], lw=2.0,
                label=rf'2D EM, $\eta = {eta}$')
    ax.axhline(y2_conserved, color='black', lw=1.5, ls='--',
               label=rf'$\langle y^{{2}}\rangle_{{0}} = 3$ (martingale value)')
    ax.set_xlabel(r'continuous time  $t$')
    ax.set_ylabel(r'$\langle y^{2} \rangle$')
    ax.set_title(r'$\langle y^{2} \rangle(t)$: continuous-limit vs finite-$\eta$')
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
