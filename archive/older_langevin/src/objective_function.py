"""
Objective functions and expected value computations for evolutionary dynamics.

This module provides:
- JAX-based fitness landscape functions
- Expected gradient, Hessian, and fitness computations under Gaussian distributions
- Free energy computations for Boltzmann/exponential selection
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


def create_landscape(Fmax: float, NS: int, NF: int, key: jax.Array, is_toy: bool = False):
    """Creates JAX-based functions for the loss, gradient, and Hessian."""
    if is_toy and NS == 1 and NF == 1:
        return create_landscape_2d(Fmax)
    
    key, λ0_key, Λ_key = jax.random.split(key, 3)
    λ0s = jax.random.uniform(λ0_key, shape=(NS,), minval=0.01, maxval=1.0)

    def create_spd_matrix(ev_key, mat):
        ev = jax.random.uniform(ev_key, shape=(NF,), minval=0.01, maxval=10.0)
        Q, _ = jnp.linalg.qr(mat)
        return Q @ jnp.diag(ev) @ Q.T

    random_matrices = jax.random.normal(Λ_key, shape=(NS, NF, NF))
    Λ_keys = jax.random.split(key, NS)
    Λs = jax.vmap(create_spd_matrix)(Λ_keys, random_matrices)

    def fitness_fn(params, λ0s, Λs):
        x, y = params[:NF], params[NF:]
        def single_term(i, carry):
            quadratic_form = x.T @ Λs[i] @ x
            return carry + (λ0s[i] + quadratic_form) * y[i]**2
        total_loss = jax.lax.fori_loop(0, NS, single_term, 0.0)
        return Fmax - total_loss * 0.5

    # Note: We are maximizing fitness, which is defined directly by the landscape function.
    # The gradient of the fitness is what we need for gradient ascent.
    fitness_function = jax.jit(partial(fitness_fn, λ0s=λ0s, Λs=Λs))
    hessian_fn = jax.jit(jax.hessian(partial(fitness_fn, λ0s=λ0s, Λs=Λs)))
    grad_fn = jax.jit(jax.grad(partial(fitness_fn, λ0s=λ0s, Λs=Λs)))

    return fitness_function, hessian_fn, grad_fn


def create_landscape_2d(Fmax=1.0):
    """2D fitness: f(x, y) = Fmax - 0.5 * x^2 * y^2"""
    
    def loss_fn(params):
        x, y = params
        return Fmax - 0.5 * x**2 * y**2

    fitness_function = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.grad(loss_fn))
    hessian_fn = jax.jit(jax.hessian(loss_fn))
    
    return fitness_function, hessian_fn, grad_fn


def create_landscape_2d_confining(Fmax: float = 1.0):
    r"""
    2D fitness: f(x, y) = Fmax - 0.5 * x^2 * (1 + y^2)^2.

    The optimal manifold is the entire y-axis (x = 0). The fast (sharp)
    direction is x everywhere, with curvature |H_xx| = (1 + y^2)^2 that grows
    with |y| (the opposite of the x^2 y^2 case, where the sharp curvature is
    largest near the origin of the manifold).

    The Euler-Maruyama Langevin timescale-separated theory yields:
        V_x(y) = sigma^2 / (1 + y^2)^2          (quasi-stationary fast variance)
        d<y>/dtau = -2 sigma^2 y / (1 + y^2)    (slow-mode entropic drift)
    which is gradient flow on the *confining* entropic potential
        F_eff(y) = sigma^2 ln(1 + y^2).
    The stationary distribution along the manifold is therefore the
    standard Cauchy:
        p^*(y) = (1/pi) / (1 + y^2).
    Median |y| = 1; E[y^2] is divergent (heavy Cauchy tails).
    """

    def loss_fn(params):
        x, y = params
        return Fmax - 0.5 * x ** 2 * (1.0 + y ** 2) ** 2

    fitness_function = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.grad(loss_fn))
    hessian_fn = jax.jit(jax.hessian(loss_fn))

    return fitness_function, hessian_fn, grad_fn


# =============================================================================
# JAX-based expected gradient and Hessian functions
# =============================================================================

@jax.jit
def compute_expected_gradient(mu: jax.Array, cov_matrix: jax.Array) -> jax.Array:
    """
    Compute the expected gradient ⟨∇f⟩ of the fitness function f(x,y) = f_max - x²y²/2
    under a Gaussian distribution N(μ, Σ).
    
    For this specific fitness function:
    - ∂f/∂x = -xy²  →  ⟨∂f/∂x⟩ = -μ_x(μ_y² + σ_y²) - 2 μ_y σ_xy
    - ∂f/∂y = -x²y  →  ⟨∂f/∂y⟩ = -μ_y(μ_x² + σ_x²) - 2 μ_x σ_xy
    
    Args:
        mu: Mean position array [mu_x, mu_y]
        cov_matrix: 2x2 covariance matrix
        
    Returns:
        2D gradient vector
    """
    mu_x, mu_y = mu[0], mu[1]
    sigma_x_sq = cov_matrix[0, 0]
    sigma_y_sq = cov_matrix[1, 1]
    sigma_xy = cov_matrix[0, 1]
    
    grad_x = -mu_x * (mu_y**2 + sigma_y_sq) - 2.0 * mu_y * sigma_xy
    grad_y = -mu_y * (mu_x**2 + sigma_x_sq) - 2.0 * mu_x * sigma_xy
    
    return jnp.array([grad_x, grad_y])


@jax.jit
def compute_expected_hessian(mu: jax.Array, cov_matrix: jax.Array) -> jax.Array:
    """
    Compute the expected Hessian ⟨H⟩ of the fitness function f(x,y) = f_max - x²y²/2
    under a Gaussian distribution N(μ, Σ).
    
    For this specific fitness function:
    - ∂²f/∂x² = -y²  →  ⟨∂²f/∂x²⟩ = -(μ_y² + σ_y²)
    - ∂²f/∂y² = -x²  →  ⟨∂²f/∂y²⟩ = -(μ_x² + σ_x²)
    - ∂²f/∂x∂y = -2xy →  ⟨∂²f/∂x∂y⟩ = -2(μ_x·μ_y + Σ_xy)
    
    Args:
        mu: Mean position array [mu_x, mu_y]
        cov_matrix: 2x2 covariance matrix
        
    Returns:
        2x2 expected Hessian matrix
    """
    mu_x, mu_y = mu[0], mu[1]
    sigma_x_sq = cov_matrix[0, 0]
    sigma_y_sq = cov_matrix[1, 1]
    sigma_xy = cov_matrix[0, 1]
    
    H_xx = -(mu_y**2 + sigma_y_sq)
    H_yy = -(mu_x**2 + sigma_x_sq)
    H_xy = -2 * (mu_x * mu_y + sigma_xy)
    
    return jnp.array([[H_xx, H_xy], [H_xy, H_yy]])


@jax.jit
def compute_expected_fitness(mu: jax.Array, cov_matrix: jax.Array, f_max: float = 5.0) -> jax.Array:
    """
    Compute the expected fitness ⟨f⟩ of the fitness function f(x,y) = f_max - x²y²/2
    under a Gaussian distribution N(μ, Σ).
    
    For this specific fitness function:
    ⟨x²y²⟩ = (μ_x² + σ_x²)(μ_y² + σ_y²) + 2σ_xy² + 4μ_x μ_y σ_xy
    so ⟨f⟩ = f_max - 0.5 * ⟨x²y²⟩
    
    Args:
        mu: Mean position array [mu_x, mu_y]
        cov_matrix: 2x2 covariance matrix
        f_max: Maximum fitness value
        
    Returns:
        Expected fitness scalar
    """
    mu_x, mu_y = mu[0], mu[1]
    sigma_x_sq = cov_matrix[0, 0]
    sigma_y_sq = cov_matrix[1, 1]
    sigma_xy = cov_matrix[0, 1]
    
    expected_x2y2 = (
        (mu_x**2 + sigma_x_sq) * (mu_y**2 + sigma_y_sq)
        + 2.0 * sigma_xy**2
        + 4.0 * mu_x * mu_y * sigma_xy
    )
    
    return f_max - 0.5 * expected_x2y2


@jax.jit
def compute_log_expected_fitness_gradient(mu: jax.Array, cov_matrix: jax.Array, f_max: float = 5.0) -> jax.Array:
    """
    Compute the gradient of log-expected fitness: ∇ ln ⟨F⟩ = ∇⟨F⟩ / ⟨F⟩
    
    This is the gradient used in multiplicative selection dynamics, where
    the mean update is: μ_{t+1} = μ_t + Σ · ∇ ln ⟨F⟩
    
    Args:
        mu: Mean position array [mu_x, mu_y]
        cov_matrix: 2x2 covariance matrix
        f_max: Maximum fitness value
        
    Returns:
        2D gradient vector of log-expected fitness
    """
    expected_F = compute_expected_fitness(mu, cov_matrix, f_max)
    grad_F = compute_expected_gradient(mu, cov_matrix)
    # Avoid division by zero
    safe_expected_F = jnp.maximum(expected_F, 1e-10)
    return grad_F / safe_expected_F


@jax.jit
def compute_log_expected_fitness_hessian(mu: jax.Array, cov_matrix: jax.Array, f_max: float = 5.0) -> jax.Array:
    """
    Compute the Hessian of log-expected fitness: ∇² ln ⟨F⟩ = ⟨H⟩/⟨F⟩ - (∇⟨F⟩)(∇⟨F⟩)ᵀ / ⟨F⟩²
    
    This is the Hessian used in multiplicative selection covariance dynamics, where
    the covariance update is: Σ_{t+1} = Σ_t + Σ · ∇² ln ⟨F⟩ · Σ + Σ_M
    
    Args:
        mu: Mean position array [mu_x, mu_y]
        cov_matrix: 2x2 covariance matrix
        f_max: Maximum fitness value
        
    Returns:
        2x2 Hessian matrix of log-expected fitness
    """
    expected_F = compute_expected_fitness(mu, cov_matrix, f_max)
    grad_F = compute_expected_gradient(mu, cov_matrix)
    hess_F = compute_expected_hessian(mu, cov_matrix)
    
    # Avoid division by zero
    safe_expected_F = jnp.maximum(expected_F, 1e-10)
    
    # ∇² ln ⟨F⟩ = ⟨H⟩/⟨F⟩ - (∇⟨F⟩)(∇⟨F⟩)ᵀ / ⟨F⟩²
    return hess_F / safe_expected_F - jnp.outer(grad_F, grad_F) / (safe_expected_F ** 2)


# =============================================================================
# EXPONENTIAL (BOLTZMANN) SELECTION FUNCTIONS
# =============================================================================
# These implement the general selection operator framework with W(x) = exp(F(x)/T)
# where T is the temperature parameter.
#
# For exponential selection:
#   - Free Energy: E_t = ln ⟨exp(F/T)⟩
#   - Mean update: μ_{t+1} = μ_t + Σ · ∇E
#   - Covariance update: Σ_{t+1} = Σ_t + Σ · ∇²E · Σ + Σ_M
#
# The Hessian of Free Energy is:
#   ∇²E = (1/T)⟨H⟩_tilted + (1/T²)Cov_tilted(∇F)
# =============================================================================

def _fitness_fn_2d(x: jax.Array, f_max: float) -> jax.Array:
    """Compute fitness f(x,y) = f_max - x²y²/2"""
    return f_max - 0.5 * x[0]**2 * x[1]**2


def _grad_fitness_fn_2d(x: jax.Array) -> jax.Array:
    """Compute gradient of fitness ∇f = [-xy², -x²y]"""
    return jnp.array([-x[0] * x[1]**2, -x[0]**2 * x[1]])


def _hessian_fitness_fn_2d(x: jax.Array) -> jax.Array:
    """Compute Hessian of fitness"""
    return jnp.array([
        [-x[1]**2, -2*x[0]*x[1]],
        [-2*x[0]*x[1], -x[0]**2]
    ])


@partial(jax.jit, static_argnames=['n_samples'])
def compute_free_energy(
    mu: jax.Array, 
    cov_matrix: jax.Array, 
    temperature: float,
    f_max: float = 5.0,
    key: jax.Array = None,
    n_samples: int = 10000
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute Free Energy, its gradient, and Hessian in one pass (more efficient).
    
    Args:
        mu: Mean position array [mu_x, mu_y]
        cov_matrix: 2x2 covariance matrix
        temperature: Temperature parameter T
        f_max: Maximum fitness value
        key: JAX random key for sampling
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Tuple of (free_energy, gradient, hessian)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Sample from the Gaussian distribution
    samples = jax.random.multivariate_normal(key, mu, cov_matrix, shape=(n_samples,))
    
    # Compute fitness values, gradients, and Hessians
    fitnesses = jax.vmap(lambda x: _fitness_fn_2d(x, f_max))(samples)
    gradients = jax.vmap(_grad_fitness_fn_2d)(samples)
    hessians = jax.vmap(_hessian_fitness_fn_2d)(samples)
    
    T = temperature
    log_weights = fitnesses / T
    
    # Free Energy using log-sum-exp trick
    max_log_w = jnp.max(log_weights)
    free_energy = max_log_w + jnp.log(jnp.mean(jnp.exp(log_weights - max_log_w)))
    
    # Normalized weights
    weights = jax.nn.softmax(log_weights)
    
    # Tilted expectations
    tilted_hess = jnp.sum(weights[:, None, None] * hessians, axis=0)
    tilted_grad = jnp.sum(weights[:, None] * gradients, axis=0)
    outer_products = jax.vmap(lambda g: jnp.outer(g, g))(gradients)
    tilted_outer = jnp.sum(weights[:, None, None] * outer_products, axis=0)
    
    # Gradient: ∇E = (1/T) ⟨∇F⟩_tilted
    gradient = tilted_grad / T
    
    # Hessian: ∇²E = (1/T) ⟨H⟩_tilted + (1/T²) Cov_tilted(∇F)
    tilted_cov_grad = tilted_outer - jnp.outer(tilted_grad, tilted_grad)
    hessian = tilted_hess / T + tilted_cov_grad / (T**2)
    
    return free_energy, gradient, hessian
