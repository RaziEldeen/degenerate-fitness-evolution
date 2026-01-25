"""
Analysis functions for comparing evolutionary and gradient-based optimization algorithms.

This module contains wrapper functions that run multiple algorithms and return 
structured results for comparison. These functions do NOT contain plotting logic;
use the corresponding plot_* functions from plot_utils.py to visualize results.

Main functions:
- run_mutation_std_comparison: Compare ED across different mutation rates
- run_ed_gd_sgd_comparison: Compare ED, GLD, and SGD final populations
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List, Optional

from algorithms import (
    run_evolution,
    run_gradient_descent_population,
)


def run_mutation_std_comparison(
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
        avg_stats, example_stats, final_pop = run_evolution(
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


def run_ed_gd_sgd_comparison(
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

    ed_avg_stats, ed_example_stats, ed_final_pop = run_evolution(
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
    print("\nRunning Gradient Descent Population (GLD)...")
    Sigma_matrix = jnp.eye(dim) * sigma_squared
    
    
    gd_final_pop, gd_avg_stats, gd_example_stats = run_gradient_descent_population(
        key=gd_key,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        Sigma=Sigma_matrix,
        fitness_function=fitness_function,
        grad_func=grad_func,
        hessian_func=hessian_func,
        noise_type='additive',
        initial_population=initial_population
    )
    gd_final_pop.block_until_ready()
    print(f"   GLD final mean: ({jnp.mean(gd_final_pop[:, 0]):.4f}, {jnp.mean(gd_final_pop[:, 1]):.4f})")
    
    # --- Run SGD Population ---
    print("\nRunning Shift Gradient Descent Population (SGD)...")
    

    sgd_final_pop, sgd_avg_stats, sgd_example_stats = run_gradient_descent_population(
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
            'M': M
        }
    }

