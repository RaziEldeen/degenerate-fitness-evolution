"""
Reproduce figures from:

    "Evolution on degenerate fitness landscapes is not neutral:
     curvature drives directional drift"

This script reproduces figures from the paper using:
- Evolutionary Dynamics (ED)
- Gradient Langevin Dynamics (GLD) / Stochastic Gradient Descent (SGD)
- Natural Evolution Strategy (NES)
- Theoretical dynamics predictions

Usage:
    python reproduce_figures.py --fig 2      # Reproduce Figure 2 (main text)
    python reproduce_figures.py --fig S1     # Reproduce Supplementary Figure 1
    python reproduce_figures.py --main       # All main text figures (2-4)
    python reproduce_figures.py --supp       # All supplementary figures (S1-S6)
    python reproduce_figures.py --all        # All figures
    python reproduce_figures.py --list       # List available figures
    python reproduce_figures.py              # Interactive mode
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
    plot_ed_vs_full_nes_summary,
    plot_dynamics_theory_vs_empirical,
    plot_high_dim_sigma_comparison,
    plot_ed_gd_sgd_populations,
    plot_mu_theory_and_mutation_std,
    plot_curvature_and_fitness_histograms,
    get_colors_for_values,
    compute_covariance_hessian_alignment,
    plot_alignment_and_anticorrelation_summary,
)

# Import objective functions
from objective_function import create_landscape_2d, create_landscape


def run_ed_vs_full_nes():
    """
    Figure S1 (Supplementary): ED vs Full Natural Gradient ES.
    
    Compares Evolutionary Dynamics with Full NES and generates a comprehensive
    comparison plot showing population snapshots and trajectory evolution.
    """
    print("\n" + "=" * 60)
    print("Figure S1: ED vs Full NES Comparison")
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
    fitness_func, hessian_func, grad_func = create_landscape_2d(F_MAX)
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


def run_theory_vs_empirical():
    """
    Figure 3 (Main Text): Theoretical vs Empirical Dynamics.
    
    Compares theoretical dynamics predictions with empirical ED simulations,
    verifying that the theoretical update rules match empirical behavior.
    """
    print("\n" + "=" * 60)
    print("Figure 3: Theoretical vs Empirical Dynamics")
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
    fitness_func, hessian_func, grad_func = create_landscape_2d(F_MAX)
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


def run_ed_gd_sgd():
    """
    Figure 4 (Main Text): ED vs GLD vs SGD Populations.
    
    Compares final population distributions from Evolutionary Dynamics,
    Gradient Langevin Dynamics, and Stochastic Gradient Descent, showing
    how each algorithm produces different steady-state distributions.
    """
    print("\n" + "=" * 60)
    print("Figure 4: ED vs GLD vs SGD Population Comparison")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 1000
    NUM_ITERATIONS = 10000
    MUTATION_STD = 0.05

    BETA = 0.1
    F_MAX = 10.0
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_landscape_2d(F_MAX)
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


def run_steady_state_histogram():
    """
    Figure S4 (Supplementary): Steady-State Histograms.
    
    Runs ED with a large population and plots steady-state distributions
    of curvature and fitness values.
    """
    print("\n" + "=" * 60)
    print("Figure S4: Steady-State Histograms (Large Population)")
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
    fitness_func, hessian_func, _ = create_landscape_2d(F_MAX)
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


def run_high_dim_steady_state_histogram():
    """
    Figure S6 (Supplementary): High-D Steady-State Histograms.
    
    Runs ED on a high-dimensional landscape and plots steady-state
    distributions of curvature and fitness values.
    """
    print("\n" + "=" * 60)
    print("Figure S6: High-D Steady-State Histograms")
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
    fitness_func, hessian_func, _ = create_landscape(Fmax=F_MAX, NS=NS, NF=NF, key=key)
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


def run_high_dim_ed():
    """
    Figure S5 (Supplementary): High-D Evolutionary Dynamics.
    
    Runs ED on a high-dimensional landscape with NF flat directions and
    NS sharp directions, plotting fitness, curvature, and covariance alignment.
    """
    print("\n" + "=" * 60)
    print("Figure S5: High-D ED (NF flat, NS sharp)")
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
    fitness_func, hessian_func, _ = create_landscape(Fmax=F_MAX, NS=NS, NF=NF, key=key)
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
    
    print("[Done] High-D ED fitness, curvature, and covariance alignment plots generated!")
    return ed_stats, alignment_data


def run_combined_mu_mutation():
    """
    Figure 2 (Main Text): Mean Dynamics and Mutation Effects.
    
    Combines mean position dynamics (μ_t vs time) with trajectory and
    curvature evolution for different mutation standard deviations.
    """
    print("\n" + "=" * 60)
    print("Figure 2: Mean Dynamics and Mutation Effects")
    print("=" * 60)
    
    # Shared setup
    F_MAX = 5.0
    fitness_func, hessian_func, _ = create_landscape_2d(F_MAX)
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


def run_ed_vs_multiplicative_nes():
    """
    Figure S2 (Supplementary): ED (Multiplicative) vs NES on Log-Fitness.
    
    Compares ED with multiplicative selection to NES, demonstrating that
    multiplicative selection performs natural gradient ascent on log-expected
    fitness rather than regular fitness.
    """
    print("\n" + "=" * 60)
    print("Figure S2: ED (Multiplicative) vs NES on Log-Fitness")
    print("=" * 60)
    
    # Parameters
    POPULATION_SIZE = 10000
    NUM_ITERATIONS = 300
    MUTATION_STD = 0.05
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    
    # Setup
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, grad_func = create_landscape_2d(F_MAX)
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


def run_ed_vs_exponential_nes():
    """
    Figure S3 (Supplementary): ED (Boltzmann) vs NES on Free Energy.
    
    Compares ED with Boltzmann/exponential selection W(x) = exp(F(x)/T) to NES,
    demonstrating natural gradient ascent on the Free Energy E = ln ⟨exp(F/T)⟩.
    The temperature parameter T controls selection pressure.
    """
    print("\n" + "=" * 60)
    print("Figure S3: ED (Boltzmann) vs NES on Free Energy")
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
    fitness_func, hessian_func, grad_func = create_landscape_2d(F_MAX)
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


# Map figure numbers to their functions and descriptions
FIGURES = {
    # Main text figures
    '2': ('Fig 2: Mean dynamics and mutation effects', run_combined_mu_mutation),
    '3': ('Fig 3: Theory vs empirical dynamics', run_theory_vs_empirical),
    '4': ('Fig 4: ED vs GLD vs SGD populations', run_ed_gd_sgd),
    # Supplementary figures
    'S1': ('Fig S1: ED vs Full NES', run_ed_vs_full_nes),
    'S2': ('Fig S2: ED (Multiplicative) vs NES', run_ed_vs_multiplicative_nes),
    'S3': ('Fig S3: ED (Boltzmann) vs NES', run_ed_vs_exponential_nes),
    'S4': ('Fig S4: Steady-state histograms', run_steady_state_histogram),
    'S5': ('Fig S5: High-D ED dynamics', run_high_dim_ed),
    'S6': ('Fig S6: High-D steady-state histograms', run_high_dim_steady_state_histogram),
}

# Keys for different figure groups
MAIN_FIGURES = ['2', '3', '4']
SUPP_FIGURES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
ALL_FIGURES = MAIN_FIGURES + SUPP_FIGURES


def list_figures():
    """Print available figures."""
    print("\nAvailable figures:")
    print("-" * 60)
    print("Main Text:")
    for key in MAIN_FIGURES:
        desc, _ = FIGURES[key]
        print(f"  {key:4s}  {desc}")
    print("\nSupplementary:")
    for key in SUPP_FIGURES:
        desc, _ = FIGURES[key]
        print(f"  {key:4s}  {desc}")
    print("-" * 60)


def run_figure(choice):
    """Run a single figure by its key."""
    if choice not in FIGURES:
        print(f"Invalid figure: {choice}")
        list_figures()
        return False
    
    desc, func = FIGURES[choice]
    print(f"\nReproducing {desc}")
    func()
    return True


def run_main_figures():
    """Reproduce all main text figures (2-4)."""
    for key in MAIN_FIGURES:
        run_figure(key)


def run_supp_figures():
    """Reproduce all supplementary figures (S1-S6)."""
    for key in SUPP_FIGURES:
        run_figure(key)


def run_all_figures():
    """Reproduce all figures sequentially."""
    for key in ALL_FIGURES:
        run_figure(key)


def interactive_mode():
    """Run in interactive mode with menu."""
    list_figures()
    print("\n  main  Reproduce all main text figures")
    print("  supp  Reproduce all supplementary figures")
    print("  all   Reproduce all figures")
    
    choice = input("\nEnter choice: ").strip().lower()
    
    if choice == 'all':
        run_all_figures()
    elif choice == 'main':
        run_main_figures()
    elif choice == 'supp':
        run_supp_figures()
    else:
        run_figure(choice.upper() if choice.startswith('s') else choice)


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description='Reproduce figures from the paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reproduce_figures.py --fig 2      Reproduce Figure 2 (main text)
  python reproduce_figures.py --fig S1     Reproduce Supplementary Figure 1
  python reproduce_figures.py --main       Reproduce all main text figures (2-4)
  python reproduce_figures.py --supp       Reproduce all supplementary figures (S1-S6)
  python reproduce_figures.py --all        Reproduce all figures
  python reproduce_figures.py --list       List available figures
  python reproduce_figures.py              Interactive mode
        """
    )
    parser.add_argument('--fig', '-f', type=str, 
                        help='Figure to reproduce (2, 3, 4, S1-S6)')
    parser.add_argument('--main', action='store_true',
                        help='Reproduce all main text figures (2-4)')
    parser.add_argument('--supp', action='store_true',
                        help='Reproduce all supplementary figures (S1-S6)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Reproduce all figures')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available figures')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Reproduce Paper Figures")
    print("=" * 60)
    
    # Setup style
    setup_style()
    
    if args.list:
        list_figures()
    elif args.all:
        run_all_figures()
    elif args.main:
        run_main_figures()
    elif args.supp:
        run_supp_figures()
    elif args.fig:
        # Normalize figure key (e.g., 's1' -> 'S1')
        fig_key = args.fig.upper() if args.fig.lower().startswith('s') else args.fig
        run_figure(fig_key)
    else:
        # Interactive mode
        interactive_mode()
    
    print("\n" + "=" * 60)
    print("Completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

