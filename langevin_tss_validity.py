"""
Langevin TSS (timescale-separated) validity diagnostic for f(x,y)=F_max-½x²y².

Brought in from archive/older_langevin/src/analysis.py (the uncommitted
work-in-progress on the main worktree) so figure 2 of the Langevin note
can reuse the two-phase ⟨x²⟩ pattern. Only the three functions feeding
that figure are pulled in here (the diagnose entrypoint plus its two
private helpers), so this module does not depend on the rest of
analysis.py / algorithms.py.

Convention here (matches the archived code): on f(x,y)=F_max-½x²y², for
walkers that fall onto the y=0 manifold (|y|<|x|) the *slow* coord is x
and the *fast* coord is y. The companion note `figures/langevin_note.md`
uses the opposite convention (y slow, x fast), so when this module's
results feed the note's Figure 2 the roles of x and y in the labels
must be swapped — the population statistics are symmetric in (x,y)
because the landscape is symmetric in (x,y).
"""

from functools import partial
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


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
    """Run an ensemble of independent Langevin/noisy-gradient trajectories.

    Returns the full (num_iterations + 1, population_size, dim) trajectory
    tensor (initial state at index 0).
    """
    sigma = mutation_std

    def step(carry, _):
        positions, key = carry
        key, nkey = jax.random.split(key)
        N, D = positions.shape
        noise = jax.random.normal(nkey, (N, D)) * sigma
        grads = jax.vmap(grad_func)(positions)
        if noise_type == 'langevin':
            new_pos = positions + learning_rate * grads + jnp.sqrt(2.0 * learning_rate) * noise
        else:  # 'additive' / noisy-gradient
            new_pos = positions + learning_rate * (grads + noise)
        return (new_pos, key), new_pos

    init = (initial_population, key)
    _, traj = jax.lax.scan(step, init, None, length=num_iterations)
    full = jnp.concatenate([initial_population[None], traj], axis=0)
    return full


def _trajectory_summary_stats(traj: np.ndarray) -> Dict:
    """Per-iteration ensemble stats from a (T+1, N, 2) trajectory tensor."""
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


def diagnose_langevin_tss_validity(
    key: jax.Array,
    num_iterations: int,
    population_size: int,
    mutation_std: float,
    learning_rate: float,
    grad_func,
    init_bound: float = 3.0,
    initial_population: Optional[jax.Array] = None,
    num_snapshots: int = 5,
) -> Dict:
    """Diagnose TSS validity and the two-phase ⟨x²⟩ pattern.

    Runs Langevin EM on f = F_max − ½·x²·y² and returns per-iteration
    population stats plus 2D snapshots. The slow second moment ⟨x²⟩(t)
    (= ⟨y²⟩(t) by symmetry) exhibits a two-phase pattern:
      (1) off-manifold transient: ⟨x²⟩ drops as walkers slide toward
          the y=0 (or x=0) axes under the bare gradient −η·x·y².
      (2) on-manifold TSS regime: the fast mode has equilibrated to
          σ²/x², the entropic drift −σ²/x cancels the slow-mode
          diffusion in mean-square, and ⟨x²⟩ stops decreasing
          (Itô martingale, note §4.1).

    Parameters
    ----------
    key, num_iterations, population_size, mutation_std, learning_rate
        Standard knobs. mutation_std=σ, learning_rate=η.
    grad_func
        JAX-compatible ∇f(position). For f=F_max−½x²y² this is
        (−x·y², −x²·y).
    init_bound
        Half-side of the uniform initial box [−init_bound, init_bound]²
        sampled from for the initial population (when initial_population
        is None).

    Returns
    -------
    Dict with keys
        params       — echo of inputs plus init_x_sq_mean, init_y_sq_mean
        iterations   — (T+1,) discrete time
        tau          — (T+1,) continuous time τ = η·t
        stats        — output of _trajectory_summary_stats
        snapshots    — {iteration_index: (N, 2)} at num_snapshots evenly
                       spaced times including 0 and T
        final_population — (N, 2)
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

    init_pop_np = np.asarray(initial_population)
    init_xsq_mean = float(np.mean(init_pop_np[:, 0] ** 2))
    init_ysq_mean = float(np.mean(init_pop_np[:, 1] ** 2))

    traj = _run_population_trajectories(
        sim_key, initial_population, num_iterations,
        learning_rate, mutation_std, grad_func, noise_type='langevin',
    )
    traj.block_until_ready()
    traj_np = np.asarray(traj)
    stats = _trajectory_summary_stats(traj_np)

    iterations = np.arange(num_iterations + 1)
    tau = iterations * eta

    snap_idx = np.linspace(0, num_iterations, num_snapshots).astype(int).tolist()
    snapshots = {int(i): traj_np[int(i)] for i in snap_idx}

    return {
        'params': {
            'mutation_std': sigma,
            'sigma2': sigma2,
            'learning_rate': eta,
            'population_size': int(population_size),
            'num_iterations': int(num_iterations),
            'init_bound': float(init_bound),
            'init_x_sq_mean': init_xsq_mean,
            'init_y_sq_mean': init_ysq_mean,
        },
        'iterations': iterations,
        'tau': tau,
        'stats': stats,
        'snapshots': snapshots,
        'final_population': traj_np[-1],
    }
