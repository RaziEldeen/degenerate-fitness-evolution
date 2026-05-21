"""
Analysis functions for comparing evolutionary and gradient-based optimization algorithms.

This module contains wrapper functions that run multiple algorithms and return 
structured results for comparison. These functions do NOT contain plotting logic;
use the corresponding plot_* functions from plot_utils.py to visualize results.

Main functions:
- compare_mutation_std: Compare ED across different mutation rates
- compare_ed_gd_sgd: Compare ED, GLD, and SGD final populations
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Dict, Tuple, List, Optional

from algorithms import (
    simulate_evolution,
    simulate_gradient_descent_population,
)


# =============================================================================
# Internal: full-trajectory population simulator for Langevin / noisy-gradient.
# Used by `compare_langevin_to_gradient_flow` to compute per-step ensemble
# statistics. Kept private so it can be JIT-compiled with grad_func/noise_type
# as static arguments.
# =============================================================================

@partial(jax.jit, static_argnames=['num_iterations', 'grad_func', 'noise_type'])
def _run_population_trajectories(
    key: jax.Array,
    initial_population: jax.Array,
    num_iterations: int,
    learning_rate: float,
    mutation_std: float,
    grad_func,
    noise_type: str = 'langevin',
) -> jax.Array:
    """
    Run an ensemble of independent gradient/Langevin trajectories on a shared
    fitness landscape and return the full trajectory tensor.

    Returns:
        Array of shape (num_iterations + 1, population_size, dim) including
        the initial population as the first entry.
    """
    sigma = mutation_std

    def step(carry, _):
        positions, key = carry
        key, nkey = jax.random.split(key)
        N, D = positions.shape
        # noise ~ N(0, sigma^2 I), independently per individual and dim
        noise = jax.random.normal(nkey, (N, D)) * sigma
        grads = jax.vmap(grad_func)(positions)
        if noise_type == 'langevin':
            new_pos = positions + learning_rate * grads + jnp.sqrt(2.0 * learning_rate) * noise
        else:  # 'additive' (noisy-gradient)
            new_pos = positions + learning_rate * (grads + noise)
        return (new_pos, key), new_pos

    init = (initial_population, key)
    _, traj = jax.lax.scan(step, init, None, length=num_iterations)
    full = jnp.concatenate([initial_population[None], traj], axis=0)
    return full


def _trajectory_summary_stats(traj: np.ndarray) -> Dict:
    """
    Compute per-iteration ensemble statistics from a (T+1, N, 2) trajectory tensor.
    """
    abs_x = np.abs(traj[:, :, 0])
    x_sq = traj[:, :, 0] ** 2
    y_sq = traj[:, :, 1] ** 2

    return {
        'median_abs_x': np.median(abs_x, axis=1),
        'p25_abs_x': np.percentile(abs_x, 25, axis=1),
        'p75_abs_x': np.percentile(abs_x, 75, axis=1),
        'mean_abs_x': np.mean(abs_x, axis=1),
        'median_x_sq': np.median(x_sq, axis=1),
        'p25_x_sq': np.percentile(x_sq, 25, axis=1),
        'p75_x_sq': np.percentile(x_sq, 75, axis=1),
        'mean_x_sq': np.mean(x_sq, axis=1),
        'mean_y_sq': np.mean(y_sq, axis=1),
        'median_y_sq': np.median(y_sq, axis=1),
    }


def compare_langevin_to_gradient_flow(
    key: jax.Array,
    x0: float,
    num_iterations: int,
    population_size: int,
    mutation_std: float,
    learning_rate: float,
    fitness_function,
    grad_func,
    hessian_func,
    include_noisy_gradient: bool = True,
    num_snapshots: int = 5,
    return_full_trajectory: bool = False,
) -> Dict:
    r"""
    Compare an Euler-Maruyama Langevin simulation on the degenerate landscape
    f(x, y) = F_max - 0.5 * x^2 * y^2 to the timescale-separated gradient-flow
    theory derived in the supplementary section
    "Timescale-separated regime".

    The theory eliminates the fast (sharp) coordinate y adiabatically at fixed
    slow x, giving the quasi-stationary variance V_y(x) = sigma^2 / x^2 and
    the slow-mode drift dx/d tau = -sigma^2 / x, i.e. gradient flow on the
    entropic potential F_eff(x) = sigma^2 ln|x|. The deterministic solution is
        x(tau)^2 = x_0^2 - 2 sigma^2 tau,         tau = learning_rate * t,
    valid while |x| >> sqrt(sigma) (eq. (eq:TS_condition) in the supplement).

    For the "noisy-gradient" discretization the analogous drift is
        dx/d tau = -learning_rate * sigma^2 / (2 x),
    so the same calculation predicts
        x(tau)^2 = x_0^2 - learning_rate * sigma^2 * tau,
    which vanishes as learning_rate -> 0.

    Args:
        key: JAX random key.
        x0: Initial slow-coordinate value. Population starts at (x0, 0).
        num_iterations: Number of discrete-time steps.
        population_size: Ensemble size.
        mutation_std: Noise standard deviation sigma (variance sigma^2 per
            coordinate per step before any sqrt(2 eta) factor).
        learning_rate: Step size eta.
        fitness_function, grad_func, hessian_func: JAX callables; only
            grad_func is needed for the dynamics.
        include_noisy_gradient: Also run the noisy-gradient variant for
            comparison.
        num_snapshots: Number of evenly spaced 2D population snapshots to
            return (including initial and final).
        return_full_trajectory: If True, include full (T+1, N, 2) trajectory
            tensors in the result. Default False to keep memory small.

    Returns:
        Dict with keys:
            'params'            -- echo of the input parameters
            'iterations', 'tau' -- shape (T+1,) discrete and continuous time
            'langevin'          -- ensemble statistics from Langevin run
            'noisy_gradient'    -- ditto for noisy-gradient (or None)
            'theory'            -- gradient-flow predictions for both variants
            'snapshots'         -- {'langevin': {time_index: pop, ...}, ...}
    """
    initial_population = jnp.zeros((population_size, 2)).at[:, 0].set(x0)

    key, lan_key, ng_key = jax.random.split(key, 3)

    print("=" * 60)
    print("Langevin vs. gradient-flow theory comparison")
    print("=" * 60)
    print(f"Parameters: x0={x0}, sigma={mutation_std}, eta={learning_rate}")
    print(f"            pop_size={population_size}, iterations={num_iterations}")
    sigma2 = float(mutation_std) ** 2
    print(f"  TSS validity: |x| >> sqrt(sigma) = {np.sqrt(mutation_std):.3f}")
    t_star_lan = (x0 ** 2) / (2.0 * sigma2 * learning_rate)
    print(f"  Langevin gradient-flow hits x=0 at iteration ~ {t_star_lan:.0f}")
    if include_noisy_gradient:
        t_star_ng = (x0 ** 2) / (learning_rate * sigma2 * learning_rate)
        print(f"  Noisy-gradient gradient-flow hits x=0 at iteration ~ {t_star_ng:.0f}")

    print("\nRunning Langevin (Euler-Maruyama) trajectories...")
    lan_traj = _run_population_trajectories(
        lan_key, initial_population, num_iterations,
        learning_rate, mutation_std, grad_func, noise_type='langevin',
    )
    lan_traj.block_until_ready()
    lan_traj_np = np.asarray(lan_traj)
    lan_stats = _trajectory_summary_stats(lan_traj_np)

    if include_noisy_gradient:
        print("Running noisy-gradient trajectories...")
        ng_traj = _run_population_trajectories(
            ng_key, initial_population, num_iterations,
            learning_rate, mutation_std, grad_func, noise_type='additive',
        )
        ng_traj.block_until_ready()
        ng_traj_np = np.asarray(ng_traj)
        ng_stats = _trajectory_summary_stats(ng_traj_np)
    else:
        ng_traj_np = None
        ng_stats = None

    iterations = np.arange(num_iterations + 1)
    tau = iterations * learning_rate

    # Theoretical gradient-flow predictions (clipped at 0 to avoid sqrt of negatives)
    x_lan_sq = np.maximum(x0 ** 2 - 2.0 * sigma2 * tau, 0.0)
    x_ng_sq = np.maximum(x0 ** 2 - learning_rate * sigma2 * tau, 0.0)

    # Quasi-stationary V_y(x) using the deterministic theory trajectory
    safe_x_lan_sq = np.where(x_lan_sq > 1e-8, x_lan_sq, np.nan)
    Vy_theory_lan = sigma2 / safe_x_lan_sq

    # Snapshot indices: evenly spaced including endpoints.
    snap_idx = np.linspace(0, num_iterations, num_snapshots).astype(int).tolist()
    snapshots = {
        'langevin': {int(i): lan_traj_np[int(i)] for i in snap_idx},
    }
    if ng_traj_np is not None:
        snapshots['noisy_gradient'] = {int(i): ng_traj_np[int(i)] for i in snap_idx}

    # ---- Conditional E[y^2 | x] from pooled (x_t, y_t^2) across the TSS-valid
    # window. The theory predicts V_y(x) = sigma^2 / x^2 (continuous limit) /
    # sigma^2 / [x^2 (1 - eta x^2 / 2)] (finite-eta correction).
    # Pool over a window where the gradient flow has not yet hit zero.
    tss_end = int(min(num_iterations, 0.9 * t_star_lan / max(learning_rate, 1e-12)))
    # Discard the first few iterations so y has had time to equilibrate.
    tss_start = min(num_iterations, max(20, int(0.02 * num_iterations)))
    if tss_end <= tss_start:
        tss_end = tss_start + max(1, num_iterations // 4)
    pooled = lan_traj_np[tss_start:tss_end]                # (W, N, 2)
    x_pool = pooled[..., 0].reshape(-1)
    y_pool = pooled[..., 1].reshape(-1)
    abs_x_pool = np.abs(x_pool)

    # Logarithmic bins in |x| from sqrt(sigma) to x0
    x_lo = max(1.5 * np.sqrt(mutation_std), 5e-2)
    x_hi = max(x0 * 1.05, x_lo * 2.0)
    edges = np.geomspace(x_lo, x_hi, 20)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cond_mean = np.full_like(centers, np.nan)
    cond_count = np.zeros_like(centers, dtype=int)
    for i in range(len(centers)):
        mask = (abs_x_pool >= edges[i]) & (abs_x_pool < edges[i + 1])
        cond_count[i] = int(mask.sum())
        if cond_count[i] >= 50:
            cond_mean[i] = float(np.mean(y_pool[mask] ** 2))
    conditional_Vy = {
        'x_bin_centers': centers,
        'mean_y_sq': cond_mean,
        'counts': cond_count,
        'theory_continuous': sigma2 / centers ** 2,
        'theory_finite_eta': 2.0 * sigma2 / (centers ** 2 * (2.0 - learning_rate * centers ** 2)),
        'tss_window': (tss_start, tss_end),
    }

    print("\nDone.")
    print(f"  Langevin  median |x| at t={num_iterations}: {lan_stats['median_abs_x'][-1]:.4f}")
    if ng_stats is not None:
        print(f"  Noisy-grad median |x| at t={num_iterations}: {ng_stats['median_abs_x'][-1]:.4f}")
    print(f"  Theory (Langevin)        |x| at t={num_iterations}: {np.sqrt(x_lan_sq[-1]):.4f}")

    result = {
        'params': {
            'x0': float(x0),
            'mutation_std': float(mutation_std),
            'sigma2': sigma2,
            'learning_rate': float(learning_rate),
            'population_size': int(population_size),
            'num_iterations': int(num_iterations),
        },
        'iterations': iterations,
        'tau': tau,
        'langevin': lan_stats,
        'noisy_gradient': ng_stats,
        'theory': {
            'langevin_x_sq': x_lan_sq,
            'langevin_x': np.sqrt(x_lan_sq),
            'noisy_gradient_x_sq': x_ng_sq,
            'noisy_gradient_x': np.sqrt(x_ng_sq),
            'Vy_quasi_stationary_theory_x': Vy_theory_lan,
        },
        'snapshots': snapshots,
        'conditional_Vy': conditional_Vy,
    }
    if return_full_trajectory:
        result['_full_trajectory'] = {
            'langevin': lan_traj_np,
            'noisy_gradient': ng_traj_np,
        }
    return result


def diagnose_langevin_tss_validity(
    key: jax.Array,
    num_iterations: int,
    population_size: int,
    mutation_std: float,
    learning_rate: float,
    fitness_function,
    grad_func,
    hessian_func,
    init_bound: float = 3.0,
    initial_population: Optional[jax.Array] = None,
    num_snapshots: int = 5,
    drift_bins: int = 24,
    min_count_per_bin: int = 200,
) -> Dict:
    r"""
    Diagnose where the timescale-separated (TSS) gradient-flow theory holds
    for the Euler-Maruyama Langevin dynamics on the degenerate landscape
    f(x, y) = F_max - 0.5 * x^2 * y^2, and quantify why the empirical final
    population spreads along the manifold rather than collapsing to the
    origin (the apparent contradiction with the gradient-flow prediction).

    The diagnostic addresses three things:

    (1) TSS validity boundary in (x, y).
        For y-elimination at fixed x: tau_y / tau_x ~ sigma^2 / x^4.
        TSS holds when |x| >> sqrt(sigma); analogously |y| >> sqrt(sigma).
        We test this empirically by binning per-step increments
        Delta x = x_{t+1} - x_t over many trajectories AND times, and
        comparing the conditional mean to the TSS prediction
        <Delta x | |x|> = -eta * sigma^2 / x.

    (2) Conditional fast-mode variance V_y(x) = sigma^2 / x^2.
        Same binning yields E[y^2 | |x|], compared to sigma^2 / x^2.

    (3) The "no-collapse" puzzle.
        In the TSS continuum limit, dx = -(sigma^2/x) dt + sqrt(2)*sigma*dW_x
        (the slow x has its own diffusion). Ito on x^2:
            d(x^2) = 2x dx + (dx)^2 = -2 sigma^2 dt + 2 sigma^2 dt + noise
                   = 2 sqrt(2) sigma x dW_x,
        so <x^2> is a *martingale*: the entropic drift exactly cancels the
        x-diffusion and the ensemble mean of x^2 is conserved while the
        median collapses. The population therefore spreads along the
        manifold instead of shrinking. We check this by tracking
        <x^2>_pop and median(x^2) over time.

    Args:
        key, num_iterations, population_size, mutation_std, learning_rate:
            Standard simulation knobs. mutation_std == sigma; learning_rate ==
            eta in the equations above.
        fitness_function, grad_func, hessian_func: JAX callables; only
            grad_func is needed for the dynamics.
        init_bound: Half-side of the uniform initial box [-init_bound,
            init_bound]^2. Ignored if initial_population is provided.
        initial_population: Optional explicit initial population (N, 2). If
            None, sample uniformly inside [-init_bound, init_bound]^2.
        num_snapshots: Number of evenly spaced 2D snapshots to record.
        drift_bins, min_count_per_bin: Binning parameters for the conditional
            <Delta x | |x|> and E[y^2 | |x|] tests.

    Returns:
        Dict with simulation statistics, theory predictions, snapshots, and
        per-bin conditional means.
    """
    sigma = float(mutation_std)
    sigma2 = sigma ** 2
    eta = float(learning_rate)

    key, init_key, sim_key = jax.random.split(key, 3)
    if initial_population is None:
        initial_population = jax.random.uniform(
            init_key,
            shape=(population_size, 2),
            minval=-init_bound,
            maxval=init_bound,
        )

    print("=" * 60)
    print("Langevin TSS-validity diagnostic on f = F_max - 0.5 x^2 y^2")
    print("=" * 60)
    print(f"  sigma={sigma}, eta={eta}, pop={population_size}, T={num_iterations}")
    print(f"  initial-box half-side = {init_bound}")
    sqrt_sigma = float(np.sqrt(sigma))
    print(f"  TSS validity: |x|, |y| >> sqrt(sigma) = {sqrt_sigma:.3f}")
    init_pop_np = np.asarray(initial_population)
    init_xsq_mean = float(np.mean(init_pop_np[:, 0] ** 2))
    init_ysq_mean = float(np.mean(init_pop_np[:, 1] ** 2))
    print(f"  initial <x^2>={init_xsq_mean:.3f}, <y^2>={init_ysq_mean:.3f}")

    print("\nRunning Langevin (Euler-Maruyama) trajectories...")
    traj = _run_population_trajectories(
        sim_key, initial_population, num_iterations,
        learning_rate, mutation_std, grad_func, noise_type='langevin',
    )
    traj.block_until_ready()
    traj = np.asarray(traj)  # (T+1, N, 2)
    stats = _trajectory_summary_stats(traj)

    iterations = np.arange(num_iterations + 1)
    tau = iterations * eta

    # ---- Conditional <Delta x | |x|> and E[y^2 | |x|] in the TSS regime ----
    # The TSS theory only applies once trajectories are *on* the manifold
    # (y has equilibrated to V_y(x) = sigma^2/x^2 << 1). The Fig 4-style
    # uniform-in-[-L, L]^2 initialisation places most trajectories OFF the
    # manifold; during the transient slide-onto-manifold, the bare gradient
    # -eta x y^2 dominates and the conditional drift is much larger than
    # the TSS prediction. We therefore restrict the pooled sample to:
    #   (i) the second half of iterations (post-burn-in onto the manifold),
    #   (ii) points near the y = 0 manifold (|y| < |x|, i.e. closer to
    #       the x-axis than the x = 0 axis -- this picks the regime in
    #       which y is the fast / sharp coordinate).
    # The same window is used for both the drift and the V_y(x) checks.
    burn_in = num_iterations // 2
    xs = traj[burn_in:-1 if num_iterations > burn_in else None, :, 0].reshape(-1)
    ys = traj[burn_in:-1 if num_iterations > burn_in else None, :, 1].reshape(-1)
    dxs = (traj[burn_in + 1:, :, 0] - traj[burn_in:-1, :, 0]).reshape(-1)
    abs_x = np.abs(xs)
    abs_y = np.abs(ys)
    on_manifold_mask = abs_y < abs_x  # nearer to the y = 0 axis
    xs = xs[on_manifold_mask]
    ys = ys[on_manifold_mask]
    dxs = dxs[on_manifold_mask]
    abs_x = abs_x[on_manifold_mask]

    x_lo = max(0.5 * sqrt_sigma, 1e-2)
    x_hi = max(init_bound * 1.05, x_lo * 4.0)
    edges = np.geomspace(x_lo, x_hi, drift_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cond_drift = np.full_like(centers, np.nan)
    cond_y_sq = np.full_like(centers, np.nan)
    counts = np.zeros_like(centers, dtype=int)
    for i in range(len(centers)):
        mask = (abs_x >= edges[i]) & (abs_x < edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i] >= min_count_per_bin:
            # Sign-symmetrize: report -sgn(x) * <Delta x | |x|>, the
            # signed drift TOWARD the origin. TSS theory predicts
            # this equals eta * sigma^2 / |x|.
            cond_drift[i] = float(np.mean(np.sign(xs[mask]) * dxs[mask]))
            cond_y_sq[i] = float(np.mean(ys[mask] ** 2))

    # Theoretical predictions per bin
    drift_theory_continuous = -eta * sigma2 / centers
    Vy_theory_continuous = sigma2 / centers ** 2
    Vy_theory_finite_eta = 2.0 * sigma2 / (centers ** 2 * (2.0 - eta * centers ** 2))

    # ---- Snapshot indices ----
    snap_idx = np.linspace(0, num_iterations, num_snapshots).astype(int).tolist()
    snapshots = {int(i): traj[int(i)] for i in snap_idx}

    # ---- Theoretical no-collapse martingale prediction (TSS Ito limit) ----
    # In the TSS continuum limit, d<x^2>/dt = 0, so <x^2> stays at its
    # initial value across the entire run.
    martingale_x_sq = np.full_like(tau, init_xsq_mean)

    print(f"\nFinal: <x^2>={stats['mean_x_sq'][-1]:.3f},  median(x^2)={stats['median_x_sq'][-1]:.3f}")
    print(f"       <y^2>={stats['mean_y_sq'][-1]:.3f},  median(y^2)={stats['median_y_sq'][-1]:.3f}")
    print(f"       Itô-martingale prediction <x^2>_t = {init_xsq_mean:.3f} for all t")

    return {
        'params': {
            'mutation_std': sigma,
            'sigma2': sigma2,
            'learning_rate': eta,
            'population_size': int(population_size),
            'num_iterations': int(num_iterations),
            'init_bound': float(init_bound),
            'sqrt_sigma': sqrt_sigma,
            'init_x_sq_mean': init_xsq_mean,
            'init_y_sq_mean': init_ysq_mean,
        },
        'iterations': iterations,
        'tau': tau,
        'stats': stats,
        'martingale_x_sq': martingale_x_sq,
        'conditional': {
            'x_bin_centers': centers,
            'x_bin_edges': edges,
            'mean_dx': cond_drift,
            'mean_y_sq': cond_y_sq,
            'counts': counts,
            'drift_theory_continuous': drift_theory_continuous,
            'Vy_theory_continuous': Vy_theory_continuous,
            'Vy_theory_finite_eta': Vy_theory_finite_eta,
            'burn_in_iter': int(burn_in),
            'on_manifold_filter': '|y| < |x|',
        },
        'snapshots': snapshots,
        'final_population': traj[-1],
    }


def compare_mutation_std(
    key: jax.Array,
    initial_mean: jax.Array,
    num_iterations: int,
    population_size: int,
    num_select: int,
    mutation_std_values: List[float],
    fitness_function,
    hessian_func,
    selection_method: str = 'linear',
    beta: float = 1.0,
    track_hessian: bool = True,
    M: int = 1
) -> Dict:
    """
    Compare evolutionary dynamics across different mutation standard deviations.
    
    Runs evolution with each mutation rate and collects trajectories and statistics
    for comparison. This function only runs the simulations and returns data;
    use plot_mutation_std_comparison() from plot_utils to visualize.
    
    Args:
        key: JAX random key
        initial_mean: Initial mean position for the population
        num_iterations: Number of evolution iterations
        population_size: Size of the population
        num_select: Number of individuals to select
        mutation_std_values: List of mutation standard deviations to compare
        fitness_function: JAX-compatible fitness function
        hessian_func: JAX-compatible Hessian function
        selection_method: Selection method ('linear', 'proportional', 'softmax')
        beta: Selection intensity parameter
        track_hessian: Whether to track Hessian statistics
        M: Number of independent runs to average
    
    Returns:
        Dictionary containing:
            - 'results': Dict mapping mutation_std -> (statistics, final_population)
            - 'params': Dict of simulation parameters
            - 'initial_mean': The initial mean position used
    """
    results = {}
    
    for mutation_std in mutation_std_values:
        # Split key for this run
        key, subkey = jax.random.split(key)
        
        # Run evolution
        avg_stats, example_stats, final_pop = simulate_evolution(
            key=subkey,
            initial_mean=initial_mean,
            num_iterations=num_iterations,
            population_size=population_size,
            sigma=mutation_std,
            num_select=num_select,
            fitness_function=fitness_function,
            hessian_func=hessian_func,
            selection_method=selection_method,
            beta=beta,
            track_hessian=track_hessian,
            M=M
        )
        
        # Ensure computation is complete
        final_pop.block_until_ready()
        
        results[mutation_std] = {
            'statistics': avg_stats,  # Use averaged statistics for comparisons
            'example_statistics': example_stats,
            'final_population': final_pop
        }
    
    # Create initial population sample for visualization (using middle mutation rate)
    middle_idx = len(mutation_std_values) // 2
    key, init_key = jax.random.split(key)
    initial_cov = jnp.eye(len(initial_mean)) * 0.01
    initial_population = jax.random.multivariate_normal(
        init_key, initial_mean, initial_cov, (population_size,)
    )
    
    return {
        'results': results,
        'initial_population': initial_population,
        'initial_mean': initial_mean,
        'params': {
            'num_iterations': num_iterations,
            'population_size': population_size,
            'num_select': num_select,
            'mutation_std_values': mutation_std_values,
            'selection_method': selection_method,
            'beta': beta,
            'M': M
        }
    }


def compare_ed_gd_sgd(
    key: jax.Array,
    initial_mean: jax.Array,
    num_iterations: int,
    population_size: int,
    mutation_std: float,
    beta: float,
    learning_rate: float,
    fitness_function,
    grad_func,
    hessian_func,
    initial_population: jax.Array = None,
    M: int = 1,
    gld_noise_type: str = 'additive',
) -> Dict:
    """
    Compare final populations of ED, GD, and SGD algorithms.
    
    Runs all three algorithms with equivalent parameters and returns
    the final populations for comparison. This function only runs the 
    simulations; use plot_ed_gd_sgd_populations() from plot_utils to visualize.
    
    ED starts from a uniform box, GD and SGD populations also start from
    uniformly distributed points within the same box.
    
    Args:
        key: JAX random key
        initial_mean: Initial mean position for ED (2D)
        num_iterations: Number of iterations to run
        population_size: Size of population for all methods
        mutation_std: Mutation standard deviation (σ)
        beta: Selection intensity / learning rate
        fitness_function: Fitness function
        grad_func: Gradient function
        hessian_func: Hessian function
        initial_population: Optional initial population array
        M: Number of independent ED runs for averaging (default 1)
        gld_noise_type: GLD discretization to use:
            'additive' (default): noisy-gradient dynamics
                x_{t+1} = x_t + eta * (grad F(x_t) + sigma * xi_t)
            'langevin': Euler-Maruyama Langevin dynamics
                x_{t+1} = x_t + eta * grad F(x_t) + sqrt(2 * eta) * sigma * xi_t
    
    Returns:
        Dictionary containing:
            - 'populations': Dict with 'ED', 'GLD', 'SGD' final populations
            - 'example_statistics': Dict with single-run statistics for trajectory plotting
            - 'average_statistics': Dict with averaged statistics (M runs for ED, population avg for GLD/SGD)
            - 'params': Dict of simulation parameters
    """
    dim = len(initial_mean)
    sigma_squared = mutation_std**2  # Variance
    
    # Split keys for each method
    key, ed_key, ed_init_key, gd_key, gd_init_key, sgd_key, sgd_init_key = jax.random.split(key, 7)
    
    print("=" * 60)
    print("Running ED, GD, and SGD Population Comparison")
    print("=" * 60)
    print(f"Parameters: pop_size={population_size}, iterations={num_iterations}")
    print(f"            mutation_std={mutation_std}, beta={beta}")
    
    # --- Run ED ---
    print(f"\nRunning Evolutionary Dynamics (ED) with M={M} runs...")
    if initial_population is None:
        initial_population = jnp.tile(initial_mean, (population_size, 1))

    ed_avg_stats, ed_example_stats, ed_final_pop = simulate_evolution(
        key=ed_key,
        initial_mean=initial_mean,
        num_iterations=num_iterations,
        population_size=population_size,
        sigma=sigma_squared,
        num_select=population_size,
        fitness_function=fitness_function,
        hessian_func=hessian_func,
        selection_method='linear',
        beta=beta,
        track_hessian=True,
        M=M,
        initial_population=initial_population
    )
    ed_final_pop.block_until_ready()
    print(f"   ED final mean: ({jnp.mean(ed_final_pop[:, 0]):.4f}, {jnp.mean(ed_final_pop[:, 1]):.4f})")
    
    # --- Run GD Population ---
    if gld_noise_type == 'langevin':
        print("\nRunning Gradient Descent Population (GLD, Euler-Maruyama Langevin)...")
    else:
        print("\nRunning Gradient Descent Population (GLD, noisy-gradient)...")
    Sigma_matrix = jnp.eye(dim) * sigma_squared
    
    
    gd_final_pop, gd_avg_stats, gd_example_stats = simulate_gradient_descent_population(
        key=gd_key,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        Sigma=Sigma_matrix,
        fitness_function=fitness_function,
        grad_func=grad_func,
        hessian_func=hessian_func,
        noise_type=gld_noise_type,
        initial_population=initial_population
    )
    gd_final_pop.block_until_ready()
    print(f"   GLD final mean: ({jnp.mean(gd_final_pop[:, 0]):.4f}, {jnp.mean(gd_final_pop[:, 1]):.4f})")
    
    # --- Run SGD Population ---
    print("\nRunning Shift Gradient Descent Population (SGD)...")
    

    sgd_final_pop, sgd_avg_stats, sgd_example_stats = simulate_gradient_descent_population(
        key=sgd_key,
        initial_population=initial_population,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        Sigma=Sigma_matrix,
        fitness_function=fitness_function,
        grad_func=grad_func,
        hessian_func=hessian_func,
        noise_type='shift'
    )
    sgd_final_pop.block_until_ready()
    print(f"   SGD final mean: ({jnp.mean(sgd_final_pop[:, 0]):.4f}, {jnp.mean(sgd_final_pop[:, 1]):.4f})")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Final Population Statistics")
    print("=" * 60)
    
    populations = {
        'ED': ed_final_pop,
        'GLD': gd_final_pop,
        'SGD': sgd_final_pop
    }
    
    # ED now also returns separate example and average statistics when M>1
    # For GLD/SGD we have separate example and average from population
    example_statistics = {
        'ED': ed_example_stats,
        'GLD': gd_example_stats,
        'SGD': sgd_example_stats
    }
    
    average_statistics = {
        'ED': ed_avg_stats,
        'GLD': gd_avg_stats,
        'SGD': sgd_avg_stats
    }
    
    for name, pop in populations.items():
        mean = jnp.mean(pop, axis=0)
        cov = jnp.cov(pop.T)
        fitnesses = jax.vmap(fitness_function)(pop)
        
        print(f"\n{name}:")
        print(f"  Mean: ({mean[0]:.4f}, {mean[1]:.4f})")
        print(f"  Covariance:")
        print(f"    σ_xx = {cov[0,0]:.6f}")
        print(f"    σ_yy = {cov[1,1]:.6f}")
        print(f"    σ_xy = {cov[0,1]:.6f}")
        print(f"  Mean Fitness: {jnp.mean(fitnesses):.4f}")
    
    return {
        'populations': populations,
        'example_statistics': example_statistics,
        'average_statistics': average_statistics,
        'params': {
            'num_iterations': num_iterations,
            'population_size': population_size,
            'mutation_std': mutation_std,
            'beta': beta,
            'learning_rate': learning_rate,
            'initial_mean': initial_mean,
            'initial_population': initial_population,
            'M': M,
            'gld_noise_type': gld_noise_type,
        }
    }

