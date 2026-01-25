"""
Core dynamics algorithms for evolutionary and gradient-based optimization.

This module contains pure algorithm implementations with no plotting dependencies.
All functions are JAX-compatible and many are JIT-compiled for performance.

Main algorithms:
- Evolutionary Dynamics (ED): `run_evolution`, `run_evolution_with_snapshots`
- Gradient Descent: `run_gradient_descent_jax`, `run_gradient_descent_population`
- Theoretical dynamics: `run_manifold_dynamics_theoretical`
- Full Natural Gradient ES: `run_full_natural_gradient_es`
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

# Import expected gradient/hessian functions for Full NES
from objective_function import (
    compute_expected_gradient_jax,
    compute_expected_hessian_jax,
    compute_expected_fitness_jax,
    compute_log_expected_fitness_gradient_jax,
    compute_log_expected_fitness_hessian_jax,
    compute_free_energy_all_jax
)


# =============================================================================
# SELECTION FUNCTIONS
# =============================================================================

def select_linear(key: jax.Array, population: jax.Array, fitnesses: jax.Array, 
                  num_select: int, beta: float = 1.0) -> jax.Array:
    """
    JAX-compatible linear selection.
    
    Args:
        key: JAX random key
        population: Population array of shape (N, D)
        fitnesses: Fitness values of shape (N,)
        num_select: Number of individuals to select
        beta: Selection intensity parameter
        
    Returns:
        Selected population of shape (num_select, D)
    """
    mean_fitness = jnp.mean(fitnesses)
    # Fitness can be negative, so we must shift to ensure probabilities are non-negative
    probabilities = 1 + beta * (fitnesses - mean_fitness)
    probabilities = jnp.maximum(probabilities, 0)
    prob_sum = jnp.sum(probabilities)
    probabilities = jnp.where(
        prob_sum > 0,
        probabilities / prob_sum,
        jnp.ones_like(probabilities) / len(probabilities)
    )
    selected_indices = jax.random.choice(key, len(population), shape=(num_select,), 
                                         p=probabilities, replace=True)
    return population[selected_indices]


def select_proportional(key: jax.Array, population: jax.Array, fitnesses: jax.Array, 
                        num_select: int) -> jax.Array:
    """
    JAX-compatible proportional selection.
    
    Args:
        key: JAX random key
        population: Population array of shape (N, D)
        fitnesses: Fitness values of shape (N,)
        num_select: Number of individuals to select
        
    Returns:
        Selected population of shape (num_select, D)
    """
    # Shift fitnesses to be non-negative for proportional selection
    shifted_fitnesses = fitnesses - jnp.min(fitnesses)
    probabilities = shifted_fitnesses
    prob_sum = jnp.sum(probabilities)

    # Normalize, or use uniform if sum is zero
    probabilities = jnp.where(
        prob_sum > 0,
        probabilities / prob_sum,
        jnp.ones_like(probabilities) / len(probabilities)
    )
    
    selected_indices = jax.random.choice(key, len(population), shape=(num_select,), 
                                         p=probabilities, replace=True)
    return population[selected_indices]


def select_multiplicative(
    key: jax.Array,
    population: jax.Array,
    fitnesses: jax.Array,
    num_select: int
) -> jax.Array:
    """
    Multiplicative selection: p_i ∝ f_i / ⟨f⟩ (requires nonnegative fitness).
    """
    nonnegative_fitness = jnp.maximum(fitnesses, 0.0)
    mean_fitness = jnp.mean(nonnegative_fitness)
    safe_mean = jnp.maximum(mean_fitness, 1e-12)
    probabilities = nonnegative_fitness / safe_mean
    prob_sum = jnp.sum(probabilities)
    probabilities = jnp.where(
        prob_sum > 0,
        probabilities / prob_sum,
        jnp.ones_like(probabilities) / len(probabilities)
    )
    selected_indices = jax.random.choice(
        key, len(population), shape=(num_select,), p=probabilities, replace=True
    )
    return population[selected_indices]


def select_softmax(key: jax.Array, population: jax.Array, fitnesses: jax.Array, 
                   num_select: int, temperature: float = 1.0) -> jax.Array:
    """
    Softmax (exponential) selection.
    
    Args:
        key: JAX random key
        population: Population array of shape (N, D)
        fitnesses: Fitness values of shape (N,)
        num_select: Number of individuals to select
        temperature: Temperature parameter (higher = more uniform)
        
    Returns:
        Selected population of shape (num_select, D)
    """
    # Temperature of 0 would cause division by zero, handle it.
    # High temperature -> more uniform probabilities. Low temperature -> more greedy.
    safe_temperature = jnp.maximum(temperature, 1e-6)
    probabilities = jax.nn.softmax(fitnesses / safe_temperature)
    
    selected_indices = jax.random.choice(key, len(population), shape=(num_select,), 
                                         p=probabilities, replace=True)
    return population[selected_indices]


def select_boltzmann(key: jax.Array, population: jax.Array, fitnesses: jax.Array, 
                     num_select: int, temperature: float = 1.0) -> jax.Array:
    """
    Boltzmann (exponential) selection with weight W(x) = exp(F(x)/T).
    
    This implements the exponential selection operator from the general framework:
        S·p(x,t) = W(x)/⟨W(x)⟩ · p(x,t)  where W(x) = exp(F(x)/T)
    
    The selection probability is proportional to the Boltzmann weight:
        p_i ∝ exp(F_i / T)
    
    This is equivalent to softmax selection but explicitly named for the 
    theoretical connection to the exponential selection operator.
    
    Args:
        key: JAX random key
        population: Population array of shape (N, D)
        fitnesses: Fitness values of shape (N,)
        num_select: Number of individuals to select
        temperature: Temperature parameter T (higher = more uniform, lower = more greedy)
        
    Returns:
        Selected population of shape (num_select, D)
    """
    safe_temperature = jnp.maximum(temperature, 1e-6)
    # Boltzmann weights: W_i = exp(F_i / T), normalized via softmax
    probabilities = jax.nn.softmax(fitnesses / safe_temperature)
    
    selected_indices = jax.random.choice(key, len(population), shape=(num_select,), 
                                         p=probabilities, replace=True)
    return population[selected_indices]


# =============================================================================
# EVOLUTIONARY DYNAMICS
# =============================================================================

def run_evolution(
    key: jax.Array,
    initial_mean: jax.Array,
    num_iterations: int,
    population_size: int,
    sigma: float,
    num_select: int,
    fitness_function,
    hessian_func,
    selection_method: str = 'linear',
    beta: float = 1.0,
    softmax_temperature: float = 1.0,
    reset_to_mean_after_selection: bool = False,
    track_hessian: bool = True,
    M: int = 1,
    initial_population: jax.Array = None
):
    """
    High-performance, JAX-based evolution algorithm.
    Can run M independent runs and average the results.
    
    Args:
        key: JAX random key
        initial_mean: Initial mean position for the population
        num_iterations: Number of evolution iterations
        population_size: Size of the population
        sigma: Mutation variance (sigma^2)
        num_select: Number of individuals to select
        fitness_function: JAX-compatible fitness function
        hessian_func: JAX-compatible Hessian function
        selection_method: Selection method ('linear', 'proportional', 'softmax')
        beta: Selection intensity parameter
        softmax_temperature: Temperature for softmax selection
        reset_to_mean_after_selection: If True, reset population to mean after selection
        track_hessian: Whether to track Hessian statistics
        M: Number of independent runs to average over
        initial_population: Optional pre-sampled initial population
        
    Returns:
        avg_statistics: Tuple of (time_points, position_history, fitness_history, 
                       hessian_traces, covariance_matrices) averaged over M runs
        example_statistics: Tuple with same structure from the first run (for trajectory visualization)
        final_population: Final population array from the first run
    """
    return _run_evolution_jit(
        key, initial_mean, num_iterations, population_size, sigma, num_select,
        fitness_function, hessian_func, selection_method, beta, softmax_temperature,
        reset_to_mean_after_selection, track_hessian, M, initial_population
    )


@partial(jax.jit, static_argnames=[
    'num_iterations', 'population_size', 'sigma', 'num_select',
    'fitness_function', 'hessian_func', 'selection_method', 'beta',
    'reset_to_mean_after_selection', 'track_hessian', 'M'
])
def _run_evolution_jit(
    key: jax.Array,
    initial_mean: jax.Array,
    num_iterations: int,
    population_size: int,
    sigma: float,
    num_select: int,
    fitness_function,
    hessian_func,
    selection_method: str = 'linear',
    beta: float = 1.0,
    softmax_temperature: float = 1.0,
    reset_to_mean_after_selection: bool = False,
    track_hessian: bool = True,
    M: int = 1,
    initial_population: jax.Array = None
):
    """Internal JIT-compiled implementation."""
    
    if M > 1:
        keys = jax.random.split(key, M)
        # Vmap the single run function over the keys
        vmapped_run = jax.vmap(_run_evolution_single, in_axes=(
            0, None, None, None, None, None, None, None, None, None, None, None, None, None
        ))
        statistics_tuple, final_populations = vmapped_run(
            keys, initial_mean, num_iterations, population_size, sigma, num_select,
            fitness_function, hessian_func, selection_method, beta, softmax_temperature,
            reset_to_mean_after_selection, track_hessian, initial_population
        )
        
        # Unpack the stacked statistics
        (all_time_points_stacked, position_history_stacked, 
         fitness_history_stacked, hessian_trace_history_stacked,
         covariance_matrices_stacked) = statistics_tuple
        
        # Average the statistics across all runs
        avg_position_history = jnp.mean(position_history_stacked, axis=0)
        avg_covariance_matrices = jnp.mean(covariance_matrices_stacked, axis=0)
        
        # Calculate fitness and hessian trace for the average trajectory
        avg_fitness_history = jax.vmap(fitness_function)(avg_position_history)
        avg_hessian_trace_history = jax.vmap(lambda p: jnp.trace(hessian_func(p)))(avg_position_history) if track_hessian else jnp.full(num_iterations + 1, jnp.nan)
        
        avg_statistics = (
            all_time_points_stacked[0],  # Time is the same for all runs
            avg_position_history,
            avg_fitness_history,
            avg_hessian_trace_history,
            avg_covariance_matrices
        )
        
        # Example statistics from the first run
        example_statistics = (
            all_time_points_stacked[0],
            position_history_stacked[0],
            fitness_history_stacked[0],
            hessian_trace_history_stacked[0],
            covariance_matrices_stacked[0]
        )
        
        # For the final population, we'll return the first run's final population
        # (not averaged, so it represents a real population state)
        example_final_population = final_populations[0]
        
        return avg_statistics, example_statistics, example_final_population
    
    else:
        stats, final_pop = _run_evolution_single(
            key, initial_mean, num_iterations, population_size, sigma, num_select,
            fitness_function, hessian_func, selection_method, beta, softmax_temperature,
            reset_to_mean_after_selection, track_hessian, initial_population
        )
        # When M=1, example and average are the same
        return stats, stats, final_pop


def _run_evolution_single(
    key: jax.Array,
    initial_mean: jax.Array,
    num_iterations: int,
    population_size: int,
    sigma: float,
    num_select: int,
    fitness_function,
    hessian_func,
    selection_method: str = 'linear',
    beta: float = 1.0,
    softmax_temperature: float = 1.0,
    reset_to_mean_after_selection: bool = False,
    track_hessian: bool = True,
    initial_population: jax.Array = None
):
    """Helper function for a single run of evolution."""
    dimension = len(initial_mean)
    key, subkey = jax.random.split(key)
    initial_cov = jnp.eye(dimension) * sigma**2

    # Use provided initial_population if available, otherwise sample
    if initial_population is not None:
        population = initial_population
    else:
        population = jax.lax.cond(
            reset_to_mean_after_selection,
            lambda: jnp.tile(initial_mean, (population_size, 1)),
            lambda: jax.random.multivariate_normal(subkey, initial_mean, initial_cov, (population_size,))
        )

    initial_avg_pos = jnp.mean(population, axis=0)
    initial_avg_fitness = fitness_function(initial_avg_pos)

    def compute_hessian_stats(pos):
        H = hessian_func(pos)
        return jnp.trace(H), jnp.linalg.eigvalsh(H)
    
    def dummy_hessian_stats(_):
        return jnp.nan, jnp.full(dimension, jnp.nan)

    initial_hessian_trace, initial_hessian_eigenvals = jax.lax.cond(
        track_hessian, compute_hessian_stats, dummy_hessian_stats, initial_avg_pos
    )

    def evolution_step(state, iter_idx):
        population, key = state
        key, mutate_key, select_key = jax.random.split(key, 3)

        # Get covariance matrix for this iteration
        cov_matrix = jax.lax.cond(
            iter_idx > 0,
            lambda: jnp.cov(population.T),
            lambda: initial_cov
        )
        post_mutation_cov = cov_matrix + jnp.eye(dimension) * sigma
        
        mutated_population = jax.random.multivariate_normal(
            mutate_key, population, jnp.eye(dimension) * sigma, shape=(population_size,)
        )
        fitnesses = jax.vmap(fitness_function)(mutated_population)
        
        if selection_method == 'linear':
            selected_population = select_linear(select_key, mutated_population, fitnesses, num_select, beta)
        elif selection_method == 'proportional':
            selected_population = select_proportional(select_key, mutated_population, fitnesses, num_select)
        elif selection_method == 'softmax':
            selected_population = select_softmax(select_key, mutated_population, fitnesses, num_select, temperature=softmax_temperature)
        elif selection_method == 'multiplicative':
            selected_population = select_multiplicative(select_key, mutated_population, fitnesses, num_select)
        elif selection_method == 'boltzmann' or selection_method == 'exponential':
            selected_population = select_boltzmann(select_key, mutated_population, fitnesses, num_select, temperature=softmax_temperature)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

        next_population = jax.lax.cond(
            reset_to_mean_after_selection,
            lambda pop: jnp.tile(jnp.mean(pop, axis=0), (population_size, 1)),
            lambda pop: pop,
            selected_population
        )

        avg_position = jnp.mean(next_population, axis=0)
        avg_fitness = fitness_function(avg_position)
        hessian_trace, hessian_eigenvals = jax.lax.cond(
            track_hessian, compute_hessian_stats, dummy_hessian_stats, avg_position
        )

        new_state = (next_population, key)
        stats = (avg_position, avg_fitness, hessian_trace, hessian_eigenvals, post_mutation_cov)
        return new_state, stats

    initial_scan_state = (population, key)
    final_state, stacked_stats = jax.lax.scan(evolution_step, initial_scan_state, jnp.arange(num_iterations))

    (mean_pos_hist, avg_fit_hist, hess_trace_hist, hess_eigval_hist, cov_mat_hist) = stacked_stats

    # Combine initial state with history
    all_time_points = jnp.arange(num_iterations + 1)
    full_pos_history = jnp.concatenate([jnp.expand_dims(initial_avg_pos, 0), mean_pos_hist])
    average_fitness = jnp.concatenate([jnp.array([initial_avg_fitness]), avg_fit_hist])
    hessian_traces = jnp.concatenate([jnp.array([initial_hessian_trace]), hess_trace_hist])
    covariance_matrices = jnp.concatenate([jnp.expand_dims(initial_cov, axis=0), cov_mat_hist])

    statistics = (all_time_points, full_pos_history, average_fitness, hessian_traces, covariance_matrices)
    final_population = final_state[0]
    return statistics, final_population


def run_evolution_with_snapshots(
    key: jax.Array,
    initial_mean: jax.Array,
    num_iterations: int,
    population_size: int,
    sigma: float,
    num_select: int,
    fitness_function,
    hessian_func,
    snapshot_times: list,
    selection_method: str = 'linear',
    beta: float = 1.0,
    track_hessian: bool = True
):
    """
    Run evolution and save population snapshots at specified time points.
    
    Args:
        key: JAX random key
        initial_mean: Initial mean position for the population
        num_iterations: Number of evolution iterations
        population_size: Size of the population
        sigma: Mutation variance (sigma^2)
        num_select: Number of individuals to select
        fitness_function: JAX-compatible fitness function
        hessian_func: JAX-compatible Hessian function
        snapshot_times: List of time points at which to save population snapshots
        selection_method: Selection method ('linear', 'proportional', 'softmax')
        beta: Selection intensity parameter
        track_hessian: Whether to track Hessian statistics
    
    Returns:
        statistics: Tuple of (time_points, position_history, fitness_history, 
                   hessian_traces, covariance_matrices)
        final_population: Final population array
        snapshots: Dictionary mapping time -> population array
    """
    dimension = len(initial_mean)
    key, subkey = jax.random.split(key)
    initial_cov = jnp.eye(dimension) * sigma
    
    # Initialize population from initial mean with initial covariance
    population = jax.random.multivariate_normal(subkey, initial_mean, initial_cov, (population_size,))
    
    initial_avg_pos = jnp.mean(population, axis=0)
    initial_avg_fitness = fitness_function(initial_avg_pos)
    initial_hessian_trace = jnp.trace(hessian_func(initial_avg_pos)) if track_hessian else jnp.nan
    
    # Storage for results
    position_history = [initial_avg_pos]
    fitness_history = [initial_avg_fitness]
    hessian_trace_history = [initial_hessian_trace]
    covariance_history = [initial_cov]
    snapshots = {}
    
    # Save initial snapshot if requested
    if 0 in snapshot_times:
        snapshots[0] = population.copy()
    
    for t in range(1, num_iterations + 1):
        key, mutate_key, select_key = jax.random.split(key, 3)
        
        # Get covariance matrix from current population
        cov_matrix = jnp.cov(population.T) if t > 1 else initial_cov
        post_mutation_cov = cov_matrix + jnp.eye(dimension) * sigma
        
        # Mutation step
        mutated_population = jax.random.multivariate_normal(
            mutate_key, population, jnp.eye(dimension) * sigma, shape=(population_size,)
        )
        fitnesses = jax.vmap(fitness_function)(mutated_population)
        
        # Selection step
        if selection_method == 'linear':
            selected_population = select_linear(select_key, mutated_population, fitnesses, num_select, beta)
        elif selection_method == 'proportional':
            selected_population = select_proportional(select_key, mutated_population, fitnesses, num_select)
        elif selection_method == 'softmax':
            selected_population = select_softmax(select_key, mutated_population, fitnesses, num_select)
        elif selection_method == 'multiplicative':
            selected_population = select_multiplicative(select_key, mutated_population, fitnesses, num_select)
        elif selection_method == 'boltzmann' or selection_method == 'exponential':
            selected_population = select_boltzmann(select_key, mutated_population, fitnesses, num_select)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        population = selected_population
        
        # Record statistics
        avg_position = jnp.mean(population, axis=0)
        avg_fitness = fitness_function(avg_position)
        hessian_trace = jnp.trace(hessian_func(avg_position)) if track_hessian else jnp.nan
        
        position_history.append(avg_position)
        fitness_history.append(avg_fitness)
        hessian_trace_history.append(hessian_trace)
        covariance_history.append(post_mutation_cov)
        
        # Save snapshot if this is a requested time point
        if t in snapshot_times:
            snapshots[t] = population.copy()
    
    # Convert lists to arrays
    all_time_points = jnp.arange(num_iterations + 1)
    position_history = jnp.stack(position_history)
    fitness_history = jnp.array(fitness_history)
    hessian_trace_history = jnp.array(hessian_trace_history)
    covariance_history = jnp.stack(covariance_history)
    
    statistics = (all_time_points, position_history, fitness_history, hessian_trace_history, covariance_history)
    return statistics, population, snapshots


# =============================================================================
# GRADIENT DESCENT
# =============================================================================

def run_gradient_descent_jax(
    key: jax.Array,
    initial_position: jax.Array,
    num_iterations: int,
    learning_rate: float,
    Sigma,  # Can be scalar, vector, or matrix
    fitness_function,
    grad_func,
    hessian_func,
    noise_type='additive',
    M: int = 1,
    average_gradients: bool = True
):
    """
    High-performance, JAX-based gradient ascent with optional noise.
    
    Args:
        key: JAX random key
        initial_position: Starting position
        num_iterations: Number of gradient descent steps
        learning_rate: Learning rate
        Sigma: Noise covariance - can be scalar (isotropic), vector (diagonal), or matrix
        fitness_function: Fitness function
        grad_func: Gradient function
        hessian_func: Hessian function
        noise_type: 'additive' for GLD, 'shift' for SGD
        M: Number of samples/runs
        average_gradients: If True, average M gradient estimates at each step.
                          If False, run M independent trajectories and average results.
                          
    Returns:
        final_position: Final position
        statistics: Tuple of (time_points, position_history, fitness_history, hessian_trace_history)
    """
    # Convert Sigma to proper matrix form before JIT compilation
    if isinstance(Sigma, (int, float)):  # Scalar
        Sigma_matrix = jnp.eye(len(initial_position)) * Sigma
    elif jnp.ndim(Sigma) == 0:  # JAX scalar
        Sigma_matrix = jnp.eye(len(initial_position)) * Sigma
    elif jnp.ndim(Sigma) == 1:  # Vector (diagonal elements)
        Sigma_matrix = jnp.diag(Sigma)
    else:  # Already a matrix
        Sigma_matrix = Sigma
    
    # Call the JIT-compiled internal function
    return _run_gradient_descent_jax_jit(
        key, initial_position, num_iterations, learning_rate, Sigma_matrix,
        fitness_function, grad_func, hessian_func, noise_type, M, average_gradients
    )


@partial(jax.jit, static_argnames=[
    'num_iterations', 'fitness_function', 'grad_func', 'hessian_func',
    'noise_type', 'M', 'average_gradients'
])
def _run_gradient_descent_jax_jit(
    key: jax.Array,
    initial_position: jax.Array,
    num_iterations: int,
    learning_rate: float,
    Sigma: jax.Array,
    fitness_function,
    grad_func,
    hessian_func,
    noise_type='additive',
    M: int = 1,
    average_gradients: bool = True
):
    """Internal JIT-compiled implementation."""
    
    if average_gradients or M == 1:
        # Use gradient averaging within a single trajectory
        return _run_gradient_descent_jax_single(
            key, initial_position, num_iterations, learning_rate, Sigma,
            fitness_function, grad_func, hessian_func, noise_type, M
        )
    else:
        # Run M independent trajectories and average the results
        keys = jax.random.split(key, M)
        
        # Vmap the single run function over the keys (with M=1 for each)
        vmapped_run = jax.vmap(
            lambda k: _run_gradient_descent_jax_single(
                k, initial_position, num_iterations, learning_rate, Sigma,
                fitness_function, grad_func, hessian_func, noise_type, 1
            )
        )
        
        final_positions, statistics_tuple = vmapped_run(keys)
        
        # Unpack and average the statistics
        (all_time_points_stacked, position_history_stacked, 
         fitness_history_stacked, hessian_trace_history_stacked) = statistics_tuple
        
        avg_position_history = jnp.mean(position_history_stacked, axis=0)
        
        # Calculate fitness and hessian trace for the average trajectory
        avg_fitness_history = jax.vmap(fitness_function)(avg_position_history)
        avg_hessian_trace_history = jax.vmap(lambda p: jnp.trace(hessian_func(p)))(avg_position_history)

        avg_statistics = (
            all_time_points_stacked[0],  # Time is the same for all runs
            avg_position_history,
            avg_fitness_history,
            avg_hessian_trace_history
        )
        
        avg_final_position = jnp.mean(final_positions, axis=0)
        
        return avg_final_position, avg_statistics


def _run_gradient_descent_jax_single(
    key: jax.Array,
    initial_position: jax.Array,
    num_iterations: int,
    learning_rate: float,
    Sigma: jax.Array,
    fitness_function,
    grad_func,
    hessian_func,
    noise_type='additive',
    M: int = 1
):
    """Helper function for gradient descent with optional mini-batch gradient averaging."""

    # Calculate initial stats
    initial_fitness = fitness_function(initial_position)
    initial_hessian = hessian_func(initial_position)
    initial_hessian_trace = jnp.trace(initial_hessian)

    def gd_step(state, _):
        position, key = state
        
        if M > 1:
            # Generate M noise samples
            key, noise_key = jax.random.split(key)
            noise_keys = jax.random.split(noise_key, M)
            
            # Generate M noise samples
            noises = jax.vmap(lambda k: jax.random.multivariate_normal(
                k, jnp.zeros_like(position), Sigma
            ))(noise_keys)
            
            # Compute M gradient estimates
            if noise_type == 'shift':
                # For shift noise, evaluate gradient at position + noise
                gradient_estimates = jax.vmap(lambda noise: grad_func(position + noise))(noises)
            else:
                # For additive noise, add noise to the gradient
                base_grad = grad_func(position)
                gradient_estimates = jax.vmap(lambda noise: base_grad + noise)(noises)
            
            # Average the gradient estimates
            averaged_gradient = jnp.mean(gradient_estimates, axis=0)
            
            # Update position with averaged gradient
            next_position = position + learning_rate * averaged_gradient
        else:
            # Original single-sample implementation
            key, noise_key = jax.random.split(key)
            noise = jax.random.multivariate_normal(noise_key, jnp.zeros_like(position), Sigma)
            
            if noise_type == 'shift':
                next_position = position + learning_rate * grad_func(position + noise)
            else:
                next_position = position + learning_rate * (grad_func(position) + noise)

        # Calculate stats for the new position
        fitness = fitness_function(next_position)
        hessian_trace = jnp.trace(hessian_func(next_position))
        stats = (next_position, fitness, hessian_trace)
        new_state = (next_position, key)
        return new_state, stats

    # Run the scan
    initial_scan_state = (initial_position, key)
    final_state, stacked_stats = jax.lax.scan(gd_step, initial_scan_state, None, length=num_iterations)
    (pos_hist, fitness_hist, hessian_trace_hist) = stacked_stats
    final_position, _ = final_state

    # Combine initial state with history
    all_time_points = jnp.arange(num_iterations + 1)
    position_history = jnp.concatenate([jnp.expand_dims(initial_position, 0), pos_hist])
    fitness_history = jnp.concatenate([jnp.array([initial_fitness]), fitness_hist])
    hessian_trace_history = jnp.concatenate([jnp.array([initial_hessian_trace]), hessian_trace_hist])

    statistics = (all_time_points, position_history, fitness_history, hessian_trace_history)
    return final_position, statistics


@partial(jax.jit, static_argnames=[
    'num_iterations', 'fitness_function', 'grad_func', 
    'hessian_func', 'noise_type'
])
def run_gradient_descent_population(
    key: jax.Array,
    initial_population: jax.Array,
    num_iterations: int,
    learning_rate: float,
    Sigma: jax.Array,
    fitness_function,
    grad_func,
    hessian_func,
    noise_type: str = 'additive'
) -> Tuple[jax.Array, Tuple]:
    """
    Run a population of independent gradient descent trajectories.
    
    Each individual in the population starts from the provided initial position
    and evolves independently following gradient descent with noise.
    
    Args:
        key: JAX random key
        initial_population: Initial population array of shape (population_size, dim)
        num_iterations: Number of gradient descent steps
        learning_rate: Learning rate for gradient updates
        Sigma: Noise covariance matrix
        fitness_function: Fitness function
        grad_func: Gradient function
        hessian_func: Hessian function
        noise_type: 'additive' for GD, 'shift' for SGD
    
    Returns:
        final_population: Array of final positions (population_size, dim)
        avg_statistics: Tuple of (time_points, mean_trajectory, mean_fitness, mean_hessian_trace)
        example_statistics: Tuple of (time_points, example_trajectory, example_fitness, example_hessian_trace)
    """
    population_size = initial_population.shape[0]
    dim = initial_population.shape[1]
    
    # Split keys for trajectory evolution
    trajectory_keys = jax.random.split(key, population_size)
    
    def run_single_trajectory(carry):
        traj_key, init_pos = carry
        
        def gd_step(state, _):
            position, key = state
            key, noise_key = jax.random.split(key)
            noise = jax.random.multivariate_normal(noise_key, jnp.zeros(dim), Sigma)
            
            if noise_type == 'shift':
                next_position = position + learning_rate * grad_func(position + noise)
            else:  # additive
                next_position = position + learning_rate * (grad_func(position) + noise)
            
            new_state = (next_position, key)
            return new_state, next_position
        
        initial_state = (init_pos, traj_key)
        final_state, trajectory = jax.lax.scan(gd_step, initial_state, None, length=num_iterations)
        final_position = final_state[0]
        
        return final_position, trajectory
    
    # Run all trajectories in parallel using vmap
    final_positions, all_trajectories = jax.vmap(run_single_trajectory)((trajectory_keys, initial_population))
    
    # Compute statistics (mean trajectory)
    mean_trajectory = jnp.mean(all_trajectories, axis=0)
    initial_mean_pos = jnp.mean(initial_population, axis=0)
    full_mean_trajectory = jnp.concatenate([jnp.expand_dims(initial_mean_pos, 0), mean_trajectory])
    
    # Compute fitness and hessian trace for mean trajectory
    mean_fitness = jax.vmap(fitness_function)(full_mean_trajectory)
    mean_hessian_trace = jax.vmap(lambda p: jnp.trace(hessian_func(p)))(full_mean_trajectory)
    
    time_points = jnp.arange(num_iterations + 1)
    avg_statistics = (time_points, full_mean_trajectory, mean_fitness, mean_hessian_trace)
    
    # Also compute one example trajectory (first one)
    example_trajectory = all_trajectories[0]
    full_example_trajectory = jnp.concatenate([jnp.expand_dims(initial_population[0], 0), example_trajectory])
    example_fitness = jax.vmap(fitness_function)(full_example_trajectory)
    example_hessian_trace = jax.vmap(lambda p: jnp.trace(hessian_func(p)))(full_example_trajectory)
    example_statistics = (time_points, full_example_trajectory, example_fitness, example_hessian_trace)
    
    return final_positions, avg_statistics, example_statistics


# =============================================================================
# THEORETICAL DYNAMICS
# =============================================================================

@partial(jax.jit, static_argnames=['num_iterations'])
def run_manifold_dynamics_theoretical(
    initial_mu: float,
    initial_a: float,
    initial_b: float,
    num_iterations: int,
    beta: float,
    m_x: float,
    m_y: float
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Calculate theoretical dynamics on the degenerate optimal manifold (y=0).
    
    Implements the theoretical update rules:
        μ_{t+1} = μ_t(1 - β a_t b_t)
        a_{t+1} = a_t(1 - β a_t b_t) + m_x
        b_{t+1} = b_t(1 - β b_t(a_t + μ_t²)) + m_y
    
    where:
        - μ_t is the x-coordinate of the mean (on y=0 manifold)
        - a_t is the variance in x direction (σ_x²)
        - b_t is the variance in y direction (σ_y²)
        - m_x, m_y are mutation variances
        - β is the selection strength / learning rate
    
    Args:
        initial_mu: Initial mean x-position
        initial_a: Initial variance in x direction
        initial_b: Initial variance in y direction
        num_iterations: Number of iterations
        beta: Selection strength / learning rate
        m_x: Mutation variance in x direction
        m_y: Mutation variance in y direction
    
    Returns:
        time_points: Array of time indices
        mu_history: Mean trajectory over time
        a_history: Variance in x direction over time
        b_history: Variance in y direction over time
    """
    # Initial conditions
    mu_0 = jnp.float64(initial_mu)
    a_0 = jnp.float64(initial_a)
    b_0 = jnp.float64(initial_b)
    
    def dynamics_step(state, _):
        mu, a, b = state
        
        # Update rules from theory
        mu_new = mu * (1 - beta * a * b)
        a_new = a * (1 - beta * a * b) + m_x
        b_new = b * (1 - beta * b * (a + mu**2)) + m_y
        
        new_state = (mu_new, a_new, b_new)
        return new_state, (mu_new, a_new, b_new)
    
    # Run the dynamics using scan
    initial_state = (mu_0, a_0, b_0)
    final_state, history = jax.lax.scan(dynamics_step, initial_state, None, length=num_iterations)
    
    mu_hist, a_hist, b_hist = history
    
    # Prepend initial conditions
    time_points = jnp.arange(num_iterations + 1)
    mu_history = jnp.concatenate([jnp.array([mu_0]), mu_hist])
    a_history = jnp.concatenate([jnp.array([a_0]), a_hist])
    b_history = jnp.concatenate([jnp.array([b_0]), b_hist])
    
    return time_points, mu_history, a_history, b_history


# =============================================================================

# NATURAL GRADIENT ES CORE
# =============================================================================

@partial(jax.jit, static_argnames=['num_iterations', 'grad_fn', 'hess_fn', 'fitness_fn', 'trace_fn', 'clip_covariance'])
def _run_natural_gradient_es_core(
    initial_mean: jax.Array,
    initial_std: float,
    mutation_std: float,
    num_iterations: int,
    eta_mu: float,
    eta_sigma: float,
    grad_fn,
    hess_fn,
    fitness_fn,
    trace_fn,
    min_variance: float,
    max_variance: float,
    clip_covariance: bool
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shared Natural Gradient ES loop for regular and log-fitness objectives."""
    dim = len(initial_mean)
    mu = initial_mean.astype(jnp.float64)
    initial_cov = jnp.eye(dim) * initial_std**2
    mutation_cov = jnp.eye(dim) * mutation_std**2
    
    # Calculate initial statistics
    initial_fitness = fitness_fn(mu, initial_cov)
    initial_hessian_trace = trace_fn(mu, initial_cov)
    
    def natural_gradient_step(state, _):
        mu, cov = state

        grad = grad_fn(mu, cov)
        hess = hess_fn(mu, cov)
        
        # Natural gradient update for mean: μ += η_μ · Σ · ∇
        natural_grad_mu = cov @ grad
        mu_new = mu + eta_mu * natural_grad_mu
        
        # Natural gradient update for covariance: Σ += η_Σ · Σ · H · Σ
        natural_grad_cov = cov @ hess @ cov
        cov_new = cov + eta_sigma * natural_grad_cov
        
        if clip_covariance:
            cov_new = jnp.clip(cov_new, min_variance, max_variance)
        
        expected_fitness = fitness_fn(mu_new, cov_new)
        hessian_trace = trace_fn(mu_new, cov_new)
        post_mutation_cov = cov_new + mutation_cov
        new_state = (mu_new, post_mutation_cov)
        stats = (mu_new, cov, post_mutation_cov, expected_fitness, hessian_trace)
        return new_state, stats
    
    initial_state = (mu, initial_cov)
    _, stacked_stats = jax.lax.scan(
        natural_gradient_step, initial_state, None, length=num_iterations
    )
    
    mu_hist, post_mut_cov_hist, cov_hist, fitness_hist, hessian_trace_hist = stacked_stats
    
    all_time_points = jnp.arange(num_iterations + 1)
    trajectory = jnp.concatenate([jnp.expand_dims(mu, 0), mu_hist])
    post_mutation_covariance_history = jnp.concatenate([jnp.expand_dims(initial_cov, 0), post_mut_cov_hist])
    covariance_history = jnp.concatenate([jnp.expand_dims(initial_cov, 0), cov_hist])
    fitness_history = jnp.concatenate([jnp.array([initial_fitness]), fitness_hist])
    hessian_trace_history = jnp.concatenate([jnp.array([initial_hessian_trace]), hessian_trace_hist])
    
    statistics = (all_time_points, trajectory, fitness_history, hessian_trace_history, 
                  post_mutation_covariance_history, covariance_history)
    return statistics


# =============================================================================
# FULL NATURAL GRADIENT ES
# =============================================================================

def run_full_natural_gradient_es(
    initial_mean: jax.Array,
    initial_std: float,
    mutation_std: float,
    num_iterations: int,
    eta_mu: float = 0.01,
    eta_sigma: float = 0.01,
    f_max: float = 5.0,
    min_variance: float = 1e-6,
    max_variance: float = 10.0
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Run Full Natural Gradient Evolution Strategy that optimizes BOTH mean and covariance.
    """
    def grad_fn(mu, cov):
        return compute_expected_gradient_jax(mu, cov)
    
    def hess_fn(mu, cov):
        return compute_expected_hessian_jax(mu, cov)
    
    def fitness_fn(mu, cov):
        return compute_expected_fitness_jax(mu, cov, f_max)
    
    def trace_fn(mu, cov):
        return jnp.trace(compute_expected_hessian_jax(mu, cov))
    
    return _run_natural_gradient_es_core(
        initial_mean, initial_std, mutation_std, num_iterations,
        eta_mu, eta_sigma, grad_fn, hess_fn, fitness_fn, trace_fn,
        min_variance, max_variance, False
    )


# =============================================================================
# MULTIPLICATIVE SELECTION NES (Log-Fitness)
# =============================================================================

def run_multiplicative_natural_gradient_es(
    initial_mean: jax.Array,
    initial_std: float,
    mutation_std: float,
    num_iterations: int,
    eta_mu: float = 1.0,
    eta_sigma: float = 1.0,
    f_max: float = 5.0,
    min_variance: float = 1e-6,
    max_variance: float = 10.0
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Run Multiplicative Natural Gradient Evolution Strategy on LOG-FITNESS.
    
    This implements the theoretical dynamics for multiplicative selection:
    
    Mean update (natural gradient on log-expected fitness):
        μ_{t+1} = μ_t + Σ · ∇ ln ⟨F⟩
        
    Covariance update (natural gradient on log-expected fitness):
        Σ_{t+1} = Σ_t + Σ · ∇² ln ⟨F⟩ · Σ + Σ_M
        
    Note: For multiplicative selection, the effective learning rate is 1
    (not β as in linear selection), since the normalization by ⟨F⟩ is built in.
    
    Args:
        initial_mean: Starting mean position (2D)
        initial_std: Initial standard deviation for covariance
        mutation_std: Mutation standard deviation added each iteration
        num_iterations: Number of optimization steps
        eta_mu: Learning rate for mean update (typically 1 for multiplicative)
        eta_sigma: Learning rate for covariance update (typically 1 for multiplicative)
        f_max: Maximum fitness value for the landscape
        min_variance: Minimum allowed variance (for numerical stability)
        max_variance: Maximum allowed variance
        
    Returns:
        statistics tuple: (time_points, trajectory, fitness_history, hessian_trace_history, 
                          post_mutation_covariance_history, covariance_history)
    """
    def grad_fn(mu, cov):
        return compute_log_expected_fitness_gradient_jax(mu, cov, f_max)
    
    def hess_fn(mu, cov):
        return compute_log_expected_fitness_hessian_jax(mu, cov, f_max)
    
    def fitness_fn(mu, cov):
        return compute_expected_fitness_jax(mu, cov, f_max)
    
    def trace_fn(mu, cov):
        return jnp.trace(compute_log_expected_fitness_hessian_jax(mu, cov, f_max))
    
    return _run_natural_gradient_es_core(
        initial_mean, initial_std, mutation_std, num_iterations,
        eta_mu, eta_sigma, grad_fn, hess_fn, fitness_fn, trace_fn,
        min_variance, max_variance, True
    )


# =============================================================================
# EXPONENTIAL (BOLTZMANN) SELECTION NES (Free Energy)
# =============================================================================

def run_exponential_natural_gradient_es(
    initial_mean: jax.Array,
    initial_std: float,
    mutation_std: float,
    num_iterations: int,
    temperature: float = 1.0,
    eta_mu: float = 1.0,
    eta_sigma: float = 1.0,
    f_max: float = 5.0,
    min_variance: float = 1e-6,
    max_variance: float = 10.0,
    n_samples: int = 10000,
    key: jax.Array = None
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Run Exponential (Boltzmann) Natural Gradient Evolution Strategy on FREE ENERGY.
    
    This implements the theoretical dynamics for exponential selection with weight
    function W(x) = exp(F(x)/T), following the general framework:
    
    The "Free Energy" objective is: E_t = ln ⟨exp(F/T)⟩_t
    
    Mean update (natural gradient on Free Energy):
        μ_{t+1} = μ_t + η_μ · Σ · ∇E
        
    where ∇E = (1/T) ⟨∇F⟩_tilted (expectation under tilted distribution)
        
    Covariance update (natural gradient on Free Energy):
        Σ_{t+1} = Σ_t + η_Σ · Σ · ∇²E · Σ + Σ_M
        
    where ∇²E = (1/T) ⟨H⟩_tilted + (1/T²) Cov_tilted(∇F)
    
    The tilted distribution has density p_tilted(x) ∝ exp(F(x)/T) · p(x).
    
    Physical interpretation:
    - T → ∞ (high temperature): Uniform selection, exploration-dominated
    - T → 0 (low temperature): Greedy selection, exploitation-dominated
    - T = 1: Balanced exponential selection
    
    Args:
        initial_mean: Starting mean position (2D)
        initial_std: Initial standard deviation for covariance
        mutation_std: Mutation standard deviation added each iteration
        num_iterations: Number of optimization steps
        temperature: Temperature parameter T for Boltzmann weighting
        eta_mu: Learning rate for mean update
        eta_sigma: Learning rate for covariance update
        f_max: Maximum fitness value for the landscape
        min_variance: Minimum allowed variance (for numerical stability)
        max_variance: Maximum allowed variance
        n_samples: Number of Monte Carlo samples for estimating expectations
        key: JAX random key (if None, uses a fixed seed)
        
    Returns:
        statistics tuple: (time_points, trajectory, fitness_history, hessian_trace_history, 
                          post_mutation_covariance_history, covariance_history)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    dim = len(initial_mean)
    mu = initial_mean.astype(jnp.float64)
    initial_cov = jnp.eye(dim) * initial_std**2
    mutation_cov = jnp.eye(dim) * mutation_std**2
    
    # Calculate initial statistics
    initial_fitness = compute_expected_fitness_jax(mu, initial_cov, f_max)
    _, _, initial_hess = compute_free_energy_all_jax(
        mu, initial_cov, temperature, f_max, key, n_samples
    )
    initial_hessian_trace = jnp.trace(initial_hess)
    
    def exponential_natural_gradient_step(state, iter_key):
        mu, cov = state
        
        # Compute Free Energy gradient and Hessian using Monte Carlo
        _, grad, hess = compute_free_energy_all_jax(
            mu, cov, temperature, f_max, iter_key, n_samples
        )
        
        # Natural gradient update for mean: μ += η_μ · Σ · ∇E
        natural_grad_mu = cov @ grad
        mu_new = mu + eta_mu * natural_grad_mu
        
        # Natural gradient update for covariance: Σ += η_Σ · Σ · ∇²E · Σ
        natural_grad_cov = cov @ hess @ cov
        cov_new = cov + eta_sigma * natural_grad_cov
        
        # Clip covariance for stability
        cov_new = jnp.clip(cov_new, min_variance, max_variance)
        
        expected_fitness = compute_expected_fitness_jax(mu_new, cov_new, f_max)
        hessian_trace = jnp.trace(hess)
        post_mutation_cov = cov_new + mutation_cov
        
        new_state = (mu_new, post_mutation_cov)
        stats = (mu_new, cov, post_mutation_cov, expected_fitness, hessian_trace)
        return new_state, stats
    
    # Generate keys for each iteration
    iter_keys = jax.random.split(key, num_iterations)
    
    initial_state = (mu, initial_cov)
    _, stacked_stats = jax.lax.scan(
        exponential_natural_gradient_step, initial_state, iter_keys
    )
    
    mu_hist, cov_hist, post_mut_cov_hist, fitness_hist, hessian_trace_hist = stacked_stats
    
    all_time_points = jnp.arange(num_iterations + 1)
    trajectory = jnp.concatenate([jnp.expand_dims(mu, 0), mu_hist])
    post_mutation_covariance_history = jnp.concatenate([jnp.expand_dims(initial_cov, 0), post_mut_cov_hist])
    covariance_history = jnp.concatenate([jnp.expand_dims(initial_cov, 0), cov_hist])
    fitness_history = jnp.concatenate([jnp.array([initial_fitness]), fitness_hist])
    hessian_trace_history = jnp.concatenate([jnp.array([initial_hessian_trace]), hessian_trace_hist])
    
    statistics = (all_time_points, trajectory, fitness_history, hessian_trace_history, 
                  post_mutation_covariance_history, covariance_history)
    return statistics



