"""
Example scripts for reproducing figures from:

    "Evolution on degenerate fitness landscapes is not neutral:
     curvature drives directional drift"

This script demonstrates the main algorithms:
1. Evolutionary Dynamics (ED)
2. Gradient Langevin Dynamics (GLD) / Stochastic Gradient Descent (SGD)
3. Natural Evolution Strategy (NES)
4. Theoretical dynamics predictions

Usage:
    python run_examples.py --example 1      # Run specific example
    python run_examples.py --all            # Run all examples
    python run_examples.py --list           # List available examples
    python run_examples.py                  # Interactive mode
"""

import argparse
import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

# Import core algorithms
from algorithms import (
    run_evolution,
    run_evolution_with_snapshots,
    run_gradient_descent_jax,
    run_full_natural_gradient_es,
    run_multiplicative_natural_gradient_es,
    run_exponential_natural_gradient_es,
    run_manifold_dynamics_theoretical,
)

# Import analysis functions
from analysis import (
    run_mutation_std_comparison,
    run_ed_gd_sgd_comparison,
)

# Import plotting utilities
from plot_utils import (
    setup_style,
    save_figure,
    plot_comparison,
    plot_ed_vs_full_nes_summary,
    plot_dynamics_theory_vs_empirical,
    plot_theory_ellipses_on_landscape,
    plot_mutation_std_comparison,
    plot_high_dim_sigma_comparison,
    plot_ed_gd_sgd_populations,
    plot_ed_gd_sgd_trajectories,
    plot_ed_gd_sgd_avg_dynamics_and_populations,
    plot_mu_theory_and_mutation_std,
    plot_curvature_and_fitness_histograms,
    get_colors_for_values,
    compute_covariance_hessian_alignment,
    plot_covariance_alignment_over_time,
    plot_projected_variance_by_eigenvector,
    plot_variance_eigenvalue_anticorrelation,
    plot_alignment_and_anticorrelation_summary,
)

# Import objective functions
from objective_function import create_jax_landscape_2d, create_landscape_jax


def run_theory_ellipse_example():
    """
    Example 1: Plot theoretical distribution ellipses on fitness landscape.
    
    Demonstrates the theoretical dynamics on the degenerate optimal manifold (y=0).
    """
    print("\n" + "=" * 60)
    print("Example 1: Theoretical Ellipses on Fitness Landscape")
    print("=" * 60)
    
    # Create fitness function
    F_MAX = 5.0
    fitness_fn, hessian_fn, grad_fn = create_jax_landscape_2d(Fmax=F_MAX)
    
    # Run theoretical dynamics
    theory_stats = run_manifold_dynamics_theoretical(
        initial_mu=3.0,       # Initial x-position on y=0 manifold
        initial_a=0.05,       # Initial σ_x²
        initial_b=0.05,       # Initial σ_y²
        num_iterations=100,
        beta=1.0,
        m_x=0.01,             # Mutation variance in x
        m_y=0.01              # Mutation variance in y
    )
    
    # Plot ellipses on landscape
    fig, ax = plot_theory_ellipses_on_landscape(
        theory_statistics=theory_stats,
        fitness_function=fitness_fn,
        snapshot_times=[0, 10, 50],
        params={'beta': 0.1, 'mutation_std': 0.05, 'n_sigma': 2.0},
        save_fig=True
    )
    
    print("[Done] Theory ellipse plot generated!")
    return theory_stats


def run_ed_vs_full_nes_example():
    """
    Example 2: Compare ED with Full Natural Gradient ES.
    
    Runs both algorithms and generates a comprehensive comparison plot.
    """
    print("\n" + "=" * 60)
    print("Example 2: ED vs Full NES Comparison")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 300
    MUTATION_STD = 0.05
    BETA = 0.1
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 1.0])
    
    SNAPSHOT_TIMES = [1, 100, NUM_ITERATIONS]
    
    # Run ED with snapshots
    key, ed_key = jax.random.split(key)
    print(f"Running ED with population snapshots at t={SNAPSHOT_TIMES}...")
    ed_stats, ed_final_pop, ed_snapshots = run_evolution_with_snapshots(
        key=ed_key,
        initial_mean=initial_mean,
        num_iterations=max(SNAPSHOT_TIMES),
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        snapshot_times=SNAPSHOT_TIMES,
        selection_method='linear',
        beta=BETA,
        track_hessian=True
    )
    
    # Run Full NES
    print("Running Full NES for comparison...")
    full_nes_stats = run_full_natural_gradient_es(
        initial_mean=initial_mean,
        initial_std=MUTATION_STD,
        mutation_std=MUTATION_STD,
        num_iterations=max(SNAPSHOT_TIMES),
        eta_mu=BETA,
        eta_sigma=BETA,
        f_max=F_MAX
    )
    
    # Parameters for plotting
    plot_params = {
        'num_iterations': max(SNAPSHOT_TIMES),
        'population_size': POPULATION_SIZE,
        'mutation_std': MUTATION_STD,
        'beta': BETA
    }
    
    # Generate summary plot
    print("Generating advanced summary plot...")
    plot_ed_vs_full_nes_summary(
        ed_populations=ed_snapshots,
        ed_statistics=ed_stats,
        full_nes_statistics=full_nes_stats,
        fitness_function=fitness_func,
        snapshot_times=SNAPSHOT_TIMES,
        params=plot_params,
        add_ellipse=True,
        save_fig=True
    )
    
    print("[Done] ED vs Full NES comparison plot generated!")
    return ed_stats, full_nes_stats


def run_theory_vs_empirical_example():
    """
    Example 3: Compare theoretical dynamics with empirical ED and Full NES.
    
    Verifies that the theoretical update rules match empirical simulations.
    """
    print("\n" + "=" * 60)
    print("Example 3: Theoretical vs Empirical Dynamics")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 100
    MUTATION_STD = 0.05
    BETA = 0.1
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    
    # Setup
    key = jax.random.PRNGKey(30)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.5, 0.0])
    
    # Run ED
    key, ed_key = jax.random.split(key)
    print("Running ED simulation...")
    ed_stats, _, _ = run_evolution(
        key=ed_key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        selection_method='linear',
        beta=BETA,
        track_hessian=True,
        M=1
    )
    
    # Run Full NES
    print("Running Full NES simulation...")
    full_nes_stats = run_full_natural_gradient_es(
        initial_mean=initial_mean,
        initial_std=MUTATION_STD,
        mutation_std=MUTATION_STD,
        num_iterations=NUM_ITERATIONS*10,
        eta_mu=BETA,
        eta_sigma=BETA,
        f_max=F_MAX
    )
    
    # Run theoretical dynamics
    print("Computing theoretical trajectory...")
    initial_mu = float(initial_mean[0])
    theory_stats = run_manifold_dynamics_theoretical(
        initial_mu=initial_mu,
        initial_a=SIGMA,
        initial_b=SIGMA,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        m_x=SIGMA,
        m_y=SIGMA
    )
    print("Computing extended theoretical trajectory for panel (c)...")
    theory_stats_long = run_manifold_dynamics_theoretical(
        initial_mu=initial_mu,
        initial_a=SIGMA,
        initial_b=SIGMA,
        num_iterations=NUM_ITERATIONS*10,
        beta=BETA,
        m_x=SIGMA,
        m_y=SIGMA
    )
    
    # Print comparison
    theory_time, theory_mu, theory_a, theory_b = theory_stats
    ed_time, ed_pos, ed_fitness, ed_hessian, ed_cov = ed_stats
    _, nes_traj, _, _, _, nes_cov = full_nes_stats
    
    print(f"\nFinal values at t={NUM_ITERATIONS}:")
    print(f"  Theory:   μ={theory_mu[-1]:.4f}, a={theory_a[-1]:.6f}, b={theory_b[-1]:.6f}")
    print(f"  ED:       μ={ed_pos[-1, 0]:.4f}, a={ed_cov[-1][0, 0]:.6f}, b={ed_cov[-1][1, 1]:.6f}")
    print(f"  Full NES: μ={nes_traj[-1, 0]:.4f}, a={nes_cov[-1][0, 0]:.6f}, b={nes_cov[-1][1, 1]:.6f}")
    
    # Plot comparison
    plot_params = {
        'num_iterations': NUM_ITERATIONS,
        'population_size': POPULATION_SIZE,
        'mutation_std': MUTATION_STD,
        'beta': BETA
    }
    
    plot_dynamics_theory_vs_empirical(
        ed_statistics=ed_stats,
        theory_statistics=theory_stats,
        full_nes_statistics=None,
        fitness_function=fitness_func,
        params=plot_params,
        save_fig=True,
        theory_statistics_for_ellipses=theory_stats_long,
        snapshot_times=[0, NUM_ITERATIONS, NUM_ITERATIONS*10]
    )
    
    print("[Done] Theory vs empirical comparison plot generated!")
    return ed_stats, theory_stats


def run_mutation_std_example():
    """
    Example 4: Compare different mutation standard deviations.
    
    Shows how mutation rate affects population evolution.
    """
    print("\n" + "=" * 60)
    print("Example 4: Mutation Standard Deviation Comparison")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 400
    BETA = 0.1
    F_MAX = 5.0
    
    mutation_std_values = [0.01, 0.1, 0.5]
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 1.0])
    
    print(f"Running ED with mutation rates: {mutation_std_values}")
    
    # Run comparison
    comparison_results = run_mutation_std_comparison(
        key=key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        num_select=POPULATION_SIZE,
        mutation_std_values=mutation_std_values,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        selection_method='linear',
        beta=BETA,
        track_hessian=True,
        M=1
    )
    
    # Plot results
    plot_mutation_std_comparison(
        comparison_results=comparison_results,
        fitness_function=fitness_func,
        save_fig=True
    )
    
    print("[Done] Mutation std comparison plot generated!")
    return comparison_results


def run_ed_gd_sgd_example():
    """
    Example 5: Compare ED, GLD, and SGD final populations.
    
    Shows how the three algorithms produce different steady-state distributions.
    """
    print("\n" + "=" * 60)
    print("Example 5: ED vs GLD vs SGD Population Comparison")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 1000
    NUM_ITERATIONS = 10000
    MUTATION_STD = 0.05

    BETA = 0.1
    F_MAX = 10.0
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 0.0])
    initial_population = jax.random.uniform(key, shape=(POPULATION_SIZE, 2), minval=-3.0, maxval=3.0)
    
    # Run comparison
    comparison_results = run_ed_gd_sgd_comparison(
        key=key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        mutation_std=MUTATION_STD,
        beta=BETA,
        learning_rate=BETA,
        fitness_function=fitness_func,
        grad_func=grad_func,
        hessian_func=hessian_func,
        initial_population=initial_population
    )
    
    # Plot populations
    plot_ed_gd_sgd_populations(
        comparison_results=comparison_results,
        fitness_function=fitness_func,
        F_MAX=F_MAX,
        save_fig=True
    )
    
    print("[Done] ED vs GLD vs SGD population plot generated!")
    return comparison_results


def run_steady_state_histogram_example():
    """
    Example 13: Steady-state curvature + fitness histograms (large population).
    
    Runs ED with a large population and plots steady-state distributions.
    """
    print("\n" + "=" * 60)
    print("Example 13: Steady-State Histograms (Large Population)")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 100000
    NUM_ITERATIONS = 1000
    MUTATION_STD = 0.1
    BETA = 0.1
    F_MAX = 10.0
    SIGMA = MUTATION_STD**2
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, _ = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 0.0])
    
    # Run ED (JAX-based)
    print("Running ED for steady-state distribution...")
    _, _, ed_final_pop = run_evolution(
        key=key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        selection_method='linear',
        beta=BETA,
        track_hessian=True,
        M=1
    )
    
    # Curvature and fitness histograms
    ed_hessians = jax.vmap(hessian_func)(ed_final_pop)
    ed_eigvals = jnp.linalg.eigvalsh(ed_hessians)
    ed_curvatures = jnp.sum(jnp.abs(ed_eigvals), axis=1)
    ed_fitnesses = jax.vmap(fitness_func)(ed_final_pop)
    fig = plot_curvature_and_fitness_histograms(
        final_population_curvatures=np.asarray(jax.device_get(ed_curvatures)),
        final_population_fitnesses=np.asarray(jax.device_get(ed_fitnesses)),
        bins=50
    )
    save_figure(fig, f"steady_state_histograms_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_mut{MUTATION_STD:.3f}", format='png')
    print("[Done] Steady-state histograms generated!")
    return ed_final_pop


def run_high_dim_steady_state_histogram_example():
    """
    Example 14: High-D steady-state curvature + fitness histograms.
    
    Runs ED on a high-D landscape and plots steady-state distributions.
    """
    print("\n" + "=" * 60)
    print("Example 14: High-D Steady-State Histograms")
    print("=" * 60)
    
    # High-D landscape parameters
    NF = 20  # flat directions (x)
    NS = 10  # sharp directions (y)
    F_MAX = 10.0
    
    # ED parameters
    POPULATION_SIZE = 100000
    NUM_ITERATIONS = 1000
    MUTATION_STD = 0.1
    BETA = 0.1
    SIGMA = MUTATION_STD**2
    BATCH_SIZE = 2000
    
    # Setup
    key = jax.random.PRNGKey(123)
    fitness_func, hessian_func, _ = create_landscape_jax(Fmax=F_MAX, NS=NS, NF=NF, key=key)
    initial_x = jnp.ones(NF) * 0.5
    initial_y = jnp.ones(NS) * 0.5
    initial_mean = jnp.concatenate([initial_x, initial_y])
    
    print(f"Running high-D ED (D={NF + NS}) for steady-state histograms...")
    _, _, ed_final_pop = run_evolution(
        key=key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        selection_method='linear',
        beta=BETA,
        track_hessian=True,
        M=1
    )
    
    # Compute curvature and fitness in batches to reduce memory pressure
    num_batches = int(np.ceil(POPULATION_SIZE / BATCH_SIZE))
    curvatures_list = []
    fitnesses_list = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, POPULATION_SIZE)
        batch = ed_final_pop[start:end]
        
        batch_hessians = jax.vmap(hessian_func)(batch)
        batch_eigvals = jnp.linalg.eigvalsh(batch_hessians)
        batch_curvatures = jnp.sum(jnp.abs(batch_eigvals), axis=1)
        batch_fitnesses = jax.vmap(fitness_func)(batch)
        
        curvatures_list.append(jax.device_get(batch_curvatures))
        fitnesses_list.append(jax.device_get(batch_fitnesses))
    
    curvatures = np.concatenate(curvatures_list, axis=0)
    fitnesses = np.concatenate(fitnesses_list, axis=0)
    
    fig = plot_curvature_and_fitness_histograms(
        final_population_curvatures=curvatures,
        final_population_fitnesses=fitnesses,
        bins=50
    )
    save_figure(fig, f"high_dim_steady_state_histograms_D{NF + NS}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_mut{MUTATION_STD:.3f}", format='png')
    
    print("[Done] High-D steady-state histograms generated!")
    return ed_final_pop


def run_ed_gd_sgd_trajectory_example():
    """
    Example 6: Compare ED, GLD, and SGD trajectories.
    
    Shows fitness, curvature, and trajectory for each algorithm.
    Displays both a single example trajectory and the average trajectory.
    """
    print("\n" + "=" * 60)
    print("Example 6: ED vs GLD vs SGD Trajectory Comparison")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 1000
    NUM_ITERATIONS = 300
    MUTATION_STD = 0.5
    BETA = 0.01
    F_MAX = 5.0
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 1.0])
    
    # Run comparison
    comparison_results = run_ed_gd_sgd_comparison(
        key=key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        mutation_std=MUTATION_STD,
        beta=BETA,
        learning_rate=BETA/MUTATION_STD**2,
        fitness_function=fitness_func,
        grad_func=grad_func,
        hessian_func=hessian_func,
        M=POPULATION_SIZE
    )
    
    # Plot trajectories (fitness, curvature, trajectory)
    plot_ed_gd_sgd_trajectories(
        comparison_results=comparison_results,
        fitness_function=fitness_func,
        F_MAX=F_MAX,
        save_fig=True
    )
    
    print("[Done] ED vs GLD vs SGD trajectory plot generated!")
    return comparison_results


def run_ed_gd_sgd_avg_dynamics_and_populations_example():
    """
    Example 8: Average dynamics (from example 6) + final populations (example 5).
    
    Top row: average trajectory/curvature/fitness. Bottom row: final populations.
    """
    print("\n" + "=" * 60)
    print("Example 8: ED vs GLD vs SGD Avg Dynamics + Populations")
    print("=" * 60)
    
    # Parameters for average dynamics (aligned with example 6 for runtime)
    POP_SIZE_AVG = 1000
    NUM_ITER_AVG = 300
    MUT_STD_AVG = 0.05
    BETA_AVG = 0.01
    F_MAX = 5.0
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 1.0])
    
    # Run comparison for average dynamics (top row)
    avg_results = run_ed_gd_sgd_comparison(
        key=key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITER_AVG,
        population_size=POP_SIZE_AVG,
        mutation_std=MUT_STD_AVG,
        beta=BETA_AVG,
        learning_rate=BETA_AVG,
        fitness_function=fitness_func,
        grad_func=grad_func,
        hessian_func=hessian_func,
        M=POP_SIZE_AVG
    )
    
    # Parameters for final populations (aligned with example 5)
    POP_SIZE_POP = 1000
    NUM_ITER_POP = 10000
    MUT_STD_POP = 0.05
    BETA_POP = 0.1
    
    # Run comparison for final populations (bottom row)
    key, pop_key = jax.random.split(key)
    pop_results = run_ed_gd_sgd_comparison(
        key=pop_key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITER_POP,
        population_size=POP_SIZE_POP,
        mutation_std=MUT_STD_POP,
        beta=BETA_POP,
        learning_rate=BETA_POP,
        fitness_function=fitness_func,
        grad_func=grad_func,
        hessian_func=hessian_func,
        initial_population=jax.random.uniform(
            pop_key, shape=(POP_SIZE_POP, 2), minval=-3.0, maxval=3.0
        )
    )
    
    # Plot combined figure
    plot_ed_gd_sgd_avg_dynamics_and_populations(
        avg_dynamics_results=avg_results,
        population_results=pop_results,
        fitness_function=fitness_func,
        F_MAX=F_MAX,
        save_fig=True
    )
    
    print("[Done] Avg dynamics + population plot generated!")
    return avg_results, pop_results


def run_high_dim_ed_example():
    """
    Example 9: High-dimensional ED with NF flat directions and NS sharp ones.
    
    Runs ED on a high-D landscape and plots fitness and curvature vs time.
    """
    print("\n" + "=" * 60)
    print("Example 9: High-D ED (NF flat, NS sharp)")
    print("=" * 60)
    
    # High-D landscape parameters
    NF = 20  # flat directions (x)
    NS = 10  # sharp directions (y)
    F_MAX = 10.0
    
    # ED parameters
    POPULATION_SIZE = 5000
    NUM_ITERATIONS = 100
    BETA = 0.1
    MUTATION_STD_VALUES = [0.01, 0.05, 0.1]
    SIGMA_VALUES = [mutation_std**2 for mutation_std in MUTATION_STD_VALUES]
    
    # Setup
    key = jax.random.PRNGKey(123)
    fitness_func, hessian_func, _ = create_landscape_jax(Fmax=F_MAX, NS=NS, NF=NF, key=key)
    initial_x = jnp.ones(NF) * 0.5
    initial_y = jnp.ones(NS) * 0.5
    initial_mean = jnp.concatenate([initial_x, initial_y])
    
    print(f"Running ED in {NF + NS}D (NF={NF}, NS={NS})...")
    print(f"Comparing sigma values: {MUTATION_STD_VALUES}")
    sigma_colors = get_colors_for_values(MUTATION_STD_VALUES)
    sigma_color_map = {}
    sigma_results = {}
    
    for mutation_std, sigma_var, color in zip(MUTATION_STD_VALUES, SIGMA_VALUES, sigma_colors):
        key, sigma_key = jax.random.split(key)
        label = f"σ = {mutation_std:.2f}"
        avg_stats, _, _ = run_evolution(
            key=sigma_key,
            initial_mean=initial_mean,
            num_iterations=NUM_ITERATIONS,
            population_size=POPULATION_SIZE,
            sigma=sigma_var,
            num_select=POPULATION_SIZE,
            fitness_function=fitness_func,
            hessian_func=hessian_func,
            selection_method='linear',
            beta=BETA,
            track_hessian=True,
            M=1
        )
        sigma_results[label] = avg_stats
        sigma_color_map[label] = color
    
    snapshot_times = list(range(NUM_ITERATIONS + 1))
    sigma_idx = len(SIGMA_VALUES) // 2
    key, snapshot_key = jax.random.split(key)
    ed_stats, _, pop_snapshots = run_evolution_with_snapshots(
        key=snapshot_key,
        initial_mean=initial_mean,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        sigma=SIGMA_VALUES[sigma_idx],
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        snapshot_times=snapshot_times,
        selection_method='linear',
        beta=BETA,
        track_hessian=True
    )
    
    focus_label = f"σ = {MUTATION_STD_VALUES[sigma_idx]:.2f}"
    fig = plot_high_dim_sigma_comparison(
        results=sigma_results,
        hessian_func=hessian_func,
        color_map=sigma_color_map,
        flat_k=NF,
        sharp_k=NS,
        focus_label=focus_label
    )
    save_figure(fig, f"high_dim_sigma_comparison_NF{NF}_NS{NS}_iter{NUM_ITERATIONS}", format='png')
    # Compute and plot covariance alignment with Hessian eigenvectors
    print("Computing covariance-Hessian alignment...")
    alignment_data = compute_covariance_hessian_alignment(
        ed_statistics=ed_stats,
        hessian_func=hessian_func,
        flat_k=NF,   # Flat directions
        sharp_k=NS,  # Sharp directions
        population_snapshots=pop_snapshots
    )
    
    plot_alignment_and_anticorrelation_summary(
        alignment_data=alignment_data,
        save_fig=True,
        filename_prefix=f"alignment_anticorrelation_NF{NF}_NS{NS}_iter{NUM_ITERATIONS}"
    )

    
    # plot_projected_variance_by_eigenvector(
    #     alignment_data=alignment_data,
    #     snapshot_times=[0, NUM_ITERATIONS // 2, NUM_ITERATIONS],
    #     save_fig=True,
    #     filename_prefix=f"projected_variance_NF{NF}_NS{NS}_iter{NUM_ITERATIONS}"
    # )
    
    # plot_variance_eigenvalue_anticorrelation(
    #     alignment_data=alignment_data,
    #     snapshot_times=[0, 5, 10],
    #     save_fig=True,
    #     filename_prefix=f"variance_eigenvalue_anticorrelation_NF{NF}_NS{NS}_iter{NUM_ITERATIONS}"
    # )
    
    print("[Done] High-D ED fitness, curvature, and covariance alignment plots generated!")
    return ed_stats, alignment_data


def run_ed_proportional_selection_example():
    """
    Example 10: ED with proportional selection.
    
    Demonstrates evolutionary dynamics using proportional (fitness-weighted) selection
    instead of linear selection. Compares proportional vs linear selection dynamics.
    """
    print("\n" + "=" * 60)
    print("Example 10: ED with Proportional Selection")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 200
    MUTATION_STD = 0.05
    BETA = 0.1
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 0.0])
    
    SNAPSHOT_TIMES = [1, 50, 100, NUM_ITERATIONS]
    
    # Run ED with proportional selection
    key, prop_key = jax.random.split(key)
    print("Running ED with proportional selection...")
    prop_stats, prop_final_pop, prop_snapshots = run_evolution_with_snapshots(
        key=prop_key,
        initial_mean=initial_mean,
        num_iterations=max(SNAPSHOT_TIMES),
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        snapshot_times=SNAPSHOT_TIMES,
        selection_method='proportional',
        beta=BETA,
        track_hessian=True
    )
    
    # Run ED with linear selection for comparison
    key, lin_key = jax.random.split(key)
    print("Running ED with linear selection for comparison...")
    lin_stats, lin_final_pop, lin_snapshots = run_evolution_with_snapshots(
        key=lin_key,
        initial_mean=initial_mean,
        num_iterations=max(SNAPSHOT_TIMES),
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        snapshot_times=SNAPSHOT_TIMES,
        selection_method='linear',
        beta=BETA,
        track_hessian=True
    )
    
    # Extract statistics
    prop_time, prop_pos, prop_fitness, prop_hessian, prop_cov = prop_stats
    lin_time, lin_pos, lin_fitness, lin_hessian, lin_cov = lin_stats
    
    # Print comparison
    print(f"\nFinal values at t={NUM_ITERATIONS}:")
    print(f"  Proportional: μ=({prop_pos[-1, 0]:.4f}, {prop_pos[-1, 1]:.4f}), fitness={prop_fitness[-1]:.4f}")
    print(f"  Linear:       μ=({lin_pos[-1, 0]:.4f}, {lin_pos[-1, 1]:.4f}), fitness={lin_fitness[-1]:.4f}")
    
    # Plot comparison
    plot_comparison(
        results={
            "ED (Proportional)": prop_stats,
            "ED (Linear)": lin_stats
        },
        title="Proportional vs Linear Selection",
        plot_fitness=True,
        plot_contours=True,
        plot_curvature=True,
        fitness_function=fitness_func
    )
    
    print("[Done] Proportional selection comparison plot generated!")
    return prop_stats, lin_stats


def run_combined_mu_mutation_example():
    """
    Example 7: Combine μ_t vs time with mutation std trajectory and curvature.
    
    Uses μ_t from theory vs empirical dynamics (start at (2,0)) and
    trajectory/curvature from mutation std comparison (start at (2,1)).
    """
    print("\n" + "=" * 60)
    print("Example 7: Combined μ_t + Mutation Std Trajectory/Curvature")
    print("=" * 60)
    
    # Shared setup
    F_MAX = 5.0
    fitness_func, hessian_func, _ = create_jax_landscape_2d(F_MAX)
    key = jax.random.PRNGKey(30)
    
    # --- μ_t vs time (theory vs empirical) ---
    POPULATION_SIZE_MU = 10000
    NUM_ITERATIONS_MU = 100
    BETA_MU = 0.1
    
    initial_mean_mu = jnp.array([2.0, 0.0])
    
    # --- Mutation std comparison (trajectory + curvature) ---
    POPULATION_SIZE_MUT = 10000
    NUM_ITERATIONS_MUT = 400
    BETA_MUT = 0.1
    mutation_std_values = [0.01, 0.1, 0.5]
    mu_sigmas = mutation_std_values
    
    initial_mean_mut = jnp.array([2.0, 1.0])
    
    key, mut_key = jax.random.split(key)
    print(f"Running mutation std comparison: {mutation_std_values}...")
    mutation_results = run_mutation_std_comparison(
        key=mut_key,
        initial_mean=initial_mean_mut,
        num_iterations=NUM_ITERATIONS_MUT,
        population_size=POPULATION_SIZE_MUT,
        num_select=POPULATION_SIZE_MUT,
        mutation_std_values=mutation_std_values,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        selection_method='linear',
        beta=BETA_MUT,
        track_hessian=True,
        M=1
    )
    
    # --- μ_t vs time (theory vs empirical) ---
    ed_stats_list = []
    theory_stats_list = []
    for mu_sigma in mu_sigmas:
        sigma_var = mu_sigma**2
        key, ed_key = jax.random.split(key)
        print(f"Running ED for μ_t dynamics (σ={mu_sigma:.2f})...")
        ed_stats, _, _ = run_evolution(
            key=ed_key,
            initial_mean=initial_mean_mu,
            num_iterations=NUM_ITERATIONS_MU,
            population_size=POPULATION_SIZE_MU,
            sigma=sigma_var,
            num_select=POPULATION_SIZE_MU,
            fitness_function=fitness_func,
            hessian_func=hessian_func,
            selection_method='linear',
            beta=BETA_MU,
            track_hessian=True,
            M=1
        )
        print(f"Computing theoretical μ_t trajectory (σ={mu_sigma:.2f})...")
        theory_stats = run_manifold_dynamics_theoretical(
            initial_mu=float(initial_mean_mu[0]),
            initial_a=sigma_var,
            initial_b=sigma_var,
            num_iterations=NUM_ITERATIONS_MU,
            beta=BETA_MU,
            m_x=sigma_var,
            m_y=sigma_var
        )
        ed_stats_list.append(ed_stats)
        theory_stats_list.append(theory_stats)
    
    # Plot combined figure
    plot_params = {
        'mu': {
            'num_iterations': NUM_ITERATIONS_MU,
            'population_size': POPULATION_SIZE_MU,
        }
    }
    
    plot_mu_theory_and_mutation_std(
        ed_statistics=ed_stats_list,
        theory_statistics=theory_stats_list,
        full_nes_statistics=None,
        mutation_comparison_results=mutation_results,
        fitness_function=fitness_func,
        params=plot_params,
        mu_sigmas=mu_sigmas,
        save_fig=True
    )
    
    print("[Done] Combined plot generated!")
    return ed_stats, theory_stats, mutation_results


def run_ed_vs_multiplicative_nes_example():
    """
    Example 11: Compare ED (proportional selection) with Multiplicative NES on log-fitness.
    
    Demonstrates that ED with multiplicative/proportional selection performs 
    natural gradient ascent on log-expected fitness, not regular fitness.
    """
    print("\n" + "=" * 60)
    print("Example 11: ED (Multiplicative) vs NES on Log-Fitness")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 300
    MUTATION_STD = 0.05
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 1.0])
    
    SNAPSHOT_TIMES = [1, 100, NUM_ITERATIONS]
    
    # Run ED with multiplicative selection (fitness-proportional)
    key, prop_key = jax.random.split(key)
    print("Running ED with multiplicative selection...")
    ed_stats, ed_final_pop, ed_snapshots = run_evolution_with_snapshots(
        key=prop_key,
        initial_mean=initial_mean,
        num_iterations=max(SNAPSHOT_TIMES),
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        snapshot_times=SNAPSHOT_TIMES,
        selection_method='multiplicative',
        beta=1.0,  # Not used for multiplicative
        track_hessian=True
    )
    
    # Run Multiplicative NES (on log-fitness)
    print("Running Multiplicative NES (log-fitness)...")
    mult_nes_stats = run_multiplicative_natural_gradient_es(
        initial_mean=initial_mean,
        initial_std=MUTATION_STD,
        mutation_std=MUTATION_STD,
        num_iterations=max(SNAPSHOT_TIMES),
        eta_mu=1.0,  # Multiplicative selection has effective learning rate 1
        eta_sigma=1.0,
        f_max=F_MAX
    )
    
    # Extract statistics for comparison
    ed_time, ed_pos, ed_fitness, ed_hessian, ed_cov = ed_stats
    nes_time, nes_pos, nes_fitness, nes_hessian, nes_post_cov, nes_cov = mult_nes_stats
    
    # Print comparison
    print(f"\nFinal values at t={NUM_ITERATIONS}:")
    print(f"  ED (Proportional):    μ=({ed_pos[-1, 0]:.4f}, {ed_pos[-1, 1]:.4f}), fitness={ed_fitness[-1]:.4f}")
    print(f"  Multiplicative NES:   μ=({nes_pos[-1, 0]:.4f}, {nes_pos[-1, 1]:.4f}), fitness={nes_fitness[-1]:.4f}")
    
    # Parameters for plotting
    plot_params = {
        'num_iterations': max(SNAPSHOT_TIMES),
        'population_size': POPULATION_SIZE,
        'mutation_std': MUTATION_STD,
    }
    
    # Generate comparison plot (reuse same plotting function as Example 2)
    print("Generating comparison plot...")
    plot_ed_vs_full_nes_summary(
        ed_populations=ed_snapshots,
        ed_statistics=ed_stats,
        full_nes_statistics=mult_nes_stats,
        fitness_function=fitness_func,
        snapshot_times=SNAPSHOT_TIMES,
        params=plot_params,
        add_ellipse=True,
        save_fig=True
    )
    
    print("[Done] ED vs Multiplicative NES comparison plot generated!")
    return ed_stats, mult_nes_stats


def run_ed_vs_exponential_nes_example():
    """
    Example 12: Compare ED (Boltzmann/exponential selection) with Exponential NES on Free Energy.
    
    Demonstrates that ED with exponential/Boltzmann selection W(x) = exp(F(x)/T)
    performs natural gradient ascent on the Free Energy E = ln ⟨exp(F/T)⟩.
    
    The temperature parameter T controls selection pressure:
    - T → ∞: Uniform selection (exploration)
    - T → 0: Greedy selection (exploitation)
    """
    print("\n" + "=" * 60)
    print("Example 12: ED (Boltzmann) vs NES on Free Energy")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 300
    MUTATION_STD = 0.05
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    TEMPERATURE = 1.0  # Boltzmann temperature
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 1.0])
    
    SNAPSHOT_TIMES = [1, 100, NUM_ITERATIONS]
    
    # Run ED with Boltzmann (exponential) selection
    key, boltz_key = jax.random.split(key)
    print(f"Running ED with Boltzmann selection (T={TEMPERATURE})...")
    ed_stats, ed_final_pop, ed_snapshots = run_evolution_with_snapshots(
        key=boltz_key,
        initial_mean=initial_mean,
        num_iterations=max(SNAPSHOT_TIMES),
        population_size=POPULATION_SIZE,
        sigma=SIGMA,
        num_select=POPULATION_SIZE,
        fitness_function=fitness_func,
        hessian_func=hessian_func,
        snapshot_times=SNAPSHOT_TIMES,
        selection_method='boltzmann',  # or 'exponential'
        beta=TEMPERATURE,  # Temperature parameter (reusing beta for softmax_temperature)
        track_hessian=True
    )
    
    # Run Exponential NES (on Free Energy)
    key, nes_key = jax.random.split(key)
    print(f"Running Exponential NES (Free Energy, T={TEMPERATURE})...")
    exp_nes_stats = run_exponential_natural_gradient_es(
        initial_mean=initial_mean,
        initial_std=MUTATION_STD,
        mutation_std=MUTATION_STD,
        num_iterations=max(SNAPSHOT_TIMES),
        temperature=TEMPERATURE,
        eta_mu=1.0,
        eta_sigma=1.0,
        f_max=F_MAX,
        n_samples=10000,
        key=nes_key
    )
    
    # Extract statistics for comparison
    ed_time, ed_pos, ed_fitness, ed_hessian, ed_cov = ed_stats
    nes_time, nes_pos, nes_fitness, nes_hessian, nes_post_cov, nes_cov = exp_nes_stats
    
    # Print comparison
    print(f"\nFinal values at t={NUM_ITERATIONS}:")
    print(f"  ED (Boltzmann T={TEMPERATURE}):  μ=({ed_pos[-1, 0]:.4f}, {ed_pos[-1, 1]:.4f}), fitness={ed_fitness[-1]:.4f}")
    print(f"  Exponential NES:                  μ=({nes_pos[-1, 0]:.4f}, {nes_pos[-1, 1]:.4f}), fitness={nes_fitness[-1]:.4f}")
    
    # Parameters for plotting
    plot_params = {
        'num_iterations': max(SNAPSHOT_TIMES),
        'population_size': POPULATION_SIZE,
        'mutation_std': MUTATION_STD,
        'temperature': TEMPERATURE,
    }
    
    # Generate comparison plot
    print("Generating comparison plot...")
    plot_ed_vs_full_nes_summary(
        ed_populations=ed_snapshots,
        ed_statistics=ed_stats,
        full_nes_statistics=exp_nes_stats,
        fitness_function=fitness_func,
        snapshot_times=SNAPSHOT_TIMES,
        params=plot_params,
        add_ellipse=True,
        save_fig=True,
        filename_suffix='_boltzmann'
    )
    
    print("[Done] ED vs Exponential NES comparison plot generated!")
    return ed_stats, exp_nes_stats


def run_ed_vs_exponential_nes_temperature_sweep():
    """
    Example 12b: Compare ED (Boltzmann) vs Exponential NES across different temperatures.
    
    Shows how temperature affects the selection pressure and convergence.
    """
    print("\n" + "=" * 60)
    print("Example 12b: Temperature Sweep for Boltzmann Selection")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 150
    MUTATION_STD = 0.05
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    
    # Temperature values to compare
    TEMPERATURES = [0.5, 1.0, 2.0, 5.0]
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_jax_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.0, 1.0])
    
    results = []
    
    for temp in TEMPERATURES:
        print(f"\nRunning with Temperature T={temp}...")
        
        key, ed_key, nes_key = jax.random.split(key, 3)
        
        # Run ED with Boltzmann selection
        ed_avg_stats, ed_ex_stats, ed_final_pop = run_evolution(
            key=ed_key,
            initial_mean=initial_mean,
            num_iterations=NUM_ITERATIONS,
            population_size=POPULATION_SIZE,
            sigma=SIGMA,
            num_select=POPULATION_SIZE,
            fitness_function=fitness_func,
            hessian_func=hessian_func,
            selection_method='boltzmann',
            softmax_temperature=temp,
            track_hessian=True,
            M=1
        )
        
        # Run Exponential NES
        exp_nes_stats = run_exponential_natural_gradient_es(
            initial_mean=initial_mean,
            initial_std=MUTATION_STD,
            mutation_std=MUTATION_STD,
            num_iterations=NUM_ITERATIONS,
            temperature=temp,
            eta_mu=1.0,
            eta_sigma=1.0,
            f_max=F_MAX,
            n_samples=10000,
            key=nes_key
        )
        
        ed_time, ed_pos, ed_fitness, ed_hessian, ed_cov = ed_avg_stats
        nes_time, nes_pos, nes_fitness, nes_hessian, nes_post_cov, nes_cov = exp_nes_stats
        
        print(f"  ED final:  μ=({ed_pos[-1, 0]:.4f}, {ed_pos[-1, 1]:.4f}), fitness={ed_fitness[-1]:.4f}")
        print(f"  NES final: μ=({nes_pos[-1, 0]:.4f}, {nes_pos[-1, 1]:.4f}), fitness={nes_fitness[-1]:.4f}")
        
        results.append({
            'temperature': temp,
            'ed_stats': ed_avg_stats,
            'nes_stats': exp_nes_stats
        })
    
    # Create comparison figure
    print("\nGenerating temperature comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(TEMPERATURES)))
    
    for idx, (result, color) in enumerate(zip(results, colors)):
        temp = result['temperature']
        ed_time, ed_pos, ed_fitness, _, _ = result['ed_stats']
        nes_time, nes_pos, nes_fitness, _, _, _ = result['nes_stats']
        
        # Plot x trajectory
        axes[0, 0].plot(ed_time, ed_pos[:, 0], '-', color=color, alpha=0.8, 
                       label=f'ED T={temp}')
        axes[0, 0].plot(nes_time, nes_pos[:, 0], '--', color=color, alpha=0.8,
                       label=f'NES T={temp}')
    
        # Plot y trajectory
        axes[0, 1].plot(ed_time, ed_pos[:, 1], '-', color=color, alpha=0.8)
        axes[0, 1].plot(nes_time, nes_pos[:, 1], '--', color=color, alpha=0.8)
        
        # Plot fitness
        axes[1, 0].plot(ed_time, ed_fitness, '-', color=color, alpha=0.8)
        axes[1, 0].plot(nes_time, nes_fitness, '--', color=color, alpha=0.8)
        
        # Plot 2D trajectory
        axes[1, 1].plot(ed_pos[:, 0], ed_pos[:, 1], '-', color=color, alpha=0.8,
                       label=f'T={temp}')
        axes[1, 1].plot(nes_pos[:, 0], nes_pos[:, 1], '--', color=color, alpha=0.8)
    
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('x position')
    axes[0, 0].set_title('X Trajectory (solid=ED, dashed=NES)')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('y position')
    axes[0, 1].set_title('Y Trajectory')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Fitness')
    axes[1, 0].set_title('Fitness Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('2D Trajectories')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    plt.suptitle(f'Boltzmann Selection: Temperature Comparison\n'
                f'Pop={POPULATION_SIZE}, Iterations={NUM_ITERATIONS}, σ_mut={MUTATION_STD}',
                fontsize=12)
    plt.tight_layout()
    
    filename = f'boltzmann_temperature_sweep_iter{NUM_ITERATIONS}_pop{POPULATION_SIZE}'
    save_figure(fig, filename, format='jpeg')
    plt.close()
    
    print("[Done] Temperature sweep comparison complete!")
    return results


# Map example numbers to their functions and descriptions
EXAMPLES = {
    '1': ('Theory ellipses on landscape', run_theory_ellipse_example),
    '2': ('ED vs Full NES comparison', run_ed_vs_full_nes_example),
    '3': ('Theory vs empirical dynamics', run_theory_vs_empirical_example),
    '4': ('Mutation std comparison', run_mutation_std_example),
    '5': ('ED vs GLD vs SGD populations', run_ed_gd_sgd_example),
    '6': ('ED vs GLD vs SGD trajectories', run_ed_gd_sgd_trajectory_example),
    '7': ('Combined mu_t + mutation std', run_combined_mu_mutation_example),
    '8': ('Avg dynamics + populations', run_ed_gd_sgd_avg_dynamics_and_populations_example),
    '9': ('High-D ED fitness/curvature', run_high_dim_ed_example),
    '10': ('ED with proportional selection', run_ed_proportional_selection_example),
    '11': ('ED (Multiplicative) vs NES', run_ed_vs_multiplicative_nes_example),
    '12': ('ED (Boltzmann) vs NES', run_ed_vs_exponential_nes_example),
    '12b': ('Boltzmann temperature sweep', run_ed_vs_exponential_nes_temperature_sweep),
    '13': ('Steady-state histograms', run_steady_state_histogram_example),
    '14': ('High-D steady-state histograms', run_high_dim_steady_state_histogram_example),
}


def list_examples():
    """Print available examples."""
    print("\nAvailable examples:")
    print("-" * 50)
    for key, (desc, _) in EXAMPLES.items():
        print(f"  {key:4s}  {desc}")
    print("-" * 50)


def run_example(choice):
    """Run a single example by its key."""
    if choice not in EXAMPLES:
        print(f"Invalid example: {choice}")
        list_examples()
        return False
    
    desc, func = EXAMPLES[choice]
    print(f"\nRunning Example {choice}: {desc}")
    func()
    return True


def run_all_examples():
    """Run all examples sequentially."""
    for key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']:
        run_example(key)


def interactive_mode():
    """Run in interactive mode with menu."""
    list_examples()
    print("  a     Run all examples")
    
    choice = input("\nEnter choice: ").strip().lower()
    
    if choice == 'a':
        run_all_examples()
    else:
        run_example(choice)


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description='Evolutionary Dynamics Examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py --example 1     Run example 1
  python run_examples.py --example 3     Run example 3
  python run_examples.py --all           Run all examples
  python run_examples.py --list          List available examples
  python run_examples.py                 Interactive mode
        """
    )
    parser.add_argument('--example', '-e', type=str, 
                        help='Example number to run (1-14, or 12b)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run all examples')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available examples')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Evolutionary Dynamics - Example Scripts")
    print("=" * 60)
    
    # Setup style
    setup_style()
    
    if args.list:
        list_examples()
    elif args.all:
        run_all_examples()
    elif args.example:
        run_example(args.example)
    else:
        # Interactive mode
        interactive_mode()
    
    print("\n" + "=" * 60)
    print("Completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

