
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from algorithms import simulate_evolution, simulate_theoretical_manifold_dynamics
from objective_function import create_landscape_2d
from plot_utils import (
    setup_style,
    save_figure,
    style_axis,
    create_figure,
    adjust_layout,
    get_colors_for_values,
    get_algorithm_color,
    AlgorithmEnum,
    COLORS,
)


def _initialize_population(key, population_size, mu_0, sigma_init):
    dim = len(mu_0)
    init_cov = jnp.eye(dim) * (sigma_init ** 2)
    return jax.random.multivariate_normal(key, mu_0, init_cov, (population_size,))


def _estimate_steady_state_variances(mu_0, mutation_variance, beta, num_iterations):
    time, mu_hist, a_hist, b_hist = simulate_theoretical_manifold_dynamics(
        initial_mu=float(mu_0),
        initial_a=mutation_variance,
        initial_b=mutation_variance,
        num_iterations=num_iterations,
        beta=beta,
        m_x=mutation_variance,
        m_y=mutation_variance
    )
    tail = max(5, int(0.1 * len(time)))
    a_star = float(jnp.mean(a_hist[-tail:]))
    b_star = float(jnp.mean(b_hist[-tail:]))
    return a_star, b_star


def _compute_drift_metrics(mu_0, final_positions):
    displacements = [pos[0] - mu_0[0] for pos in final_positions]
    signed_drift = [-d * np.sign(mu_0[0]) for d in displacements]
    mean_drift = float(np.mean(signed_drift))
    var_drift = float(np.var(signed_drift))
    return signed_drift, mean_drift, var_drift


def _compute_dimensionless_D(N, mu_0, beta, a_star, b_star):
    denom = (mu_0 * beta * b_star) ** 2 * a_star
    if denom <= 0:
        return 0.0
    return float(N * denom)


def run_curvature_drift_transition():
    print("Running curvature drift vs genetic drift transition test...")
    setup_style()

    # Fixed parameters
    mu_0 = jnp.array([5.0, 0.0])
    beta = 0.1
    mutation_variance = 0.01
    num_iterations = 1000
    num_replicates = 100
    sigma_init = 0.1
    population_sizes = [2, 5, 10, 20, 50, 100, 200, 500]

    # Setup landscape
    fitness_func, hessian_func, _ = create_landscape_2d()

    # Estimate steady-state variances for theoretical N*
    a_star, b_star = _estimate_steady_state_variances(
        mu_0[0], mutation_variance, beta, num_iterations
    )
    critical_N = 1.0 / max((mu_0[0] * beta * b_star) ** 2 * a_star, 1e-12)

    # Run empirical simulations
    key = jax.random.PRNGKey(42)
    results = {}
    D_values = {}

    for N in population_sizes:
        final_positions = []
        for _ in range(num_replicates):
            key, init_key, run_key = jax.random.split(key, 3)
            initial_population = _initialize_population(init_key, N, mu_0, sigma_init)
            stats, _, _ = simulate_evolution(
                key=run_key,
                initial_mean=mu_0,
                num_iterations=num_iterations,
                population_size=N,
                sigma=mutation_variance,
                num_select=N,
                fitness_function=fitness_func,
                hessian_func=hessian_func,
                selection_method='linear',
                beta=beta,
                track_hessian=False,
                M=1,
                initial_population=initial_population
            )
            _, pos_hist, _, _, _ = stats
            final_positions.append(np.array(pos_hist[-1]))

        signed_drift, mean_drift, var_drift = _compute_drift_metrics(mu_0, final_positions)
        results[N] = {
            "mean_drift": mean_drift,
            "var_drift": var_drift,
            "signed_drifts": signed_drift
        }
        D_values[N] = _compute_dimensionless_D(N, mu_0[0], beta, a_star, b_star)

    # Primary figure: mean drift and CV vs N
    fig, axes = create_figure(n_cols=2, width_per_panel=4.5, height_per_panel=4)
    ax_mean, ax_cv = axes

    Ns = np.array(population_sizes, dtype=float)
    mean_drifts = np.array([results[N]["mean_drift"] for N in population_sizes])
    var_drifts = np.array([results[N]["var_drift"] for N in population_sizes])
    cv = np.sqrt(var_drifts) / np.maximum(np.abs(mean_drifts), 1e-8)

    ax_mean.plot(Ns, mean_drifts, marker='o', linewidth=1.5)
    #ax_mean.axvline(critical_N, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_mean.set_xscale('log')
    ax_mean.set_xlabel("Population size N")
    ax_mean.set_ylabel(r"Mean drift $\langle D \rangle$")
    #ax_mean.set_title("Mean drift toward flat region")
    style_axis(ax_mean)

    ax_cv.plot(Ns, cv, marker='o', linewidth=1.5, color="#8c564b")
    #ax_cv.axvline(critical_N, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_cv.set_xscale('log')
    ax_cv.set_xlabel("Population size N")
    ax_cv.set_ylabel("CV")
    #ax_cv.set_title("Noise-to-signal ratio")
    style_axis(ax_cv)

    adjust_layout(fig, left=0.12, right=0.95, top=0.88, bottom=0.15, wspace=0.35)
    save_figure(fig, "curvature_drift_transition.png")

    # Secondary test: vary mu_0 at fixed N
    fixed_N = 500
    mu_0_values = [0.5, 1, 2, 5, 10, 20]
    mu_results = {}
    mu_D_values = {}

    for mu_val in mu_0_values:
        mu_vec = jnp.array([float(mu_val), 0.0])
        a_mu, b_mu = _estimate_steady_state_variances(mu_val, mutation_variance, beta, num_iterations)
        mu_D_values[mu_val] = _compute_dimensionless_D(fixed_N, mu_val, beta, a_mu, b_mu)
        final_positions = []

        for _ in range(num_replicates):
            key, init_key, run_key = jax.random.split(key, 3)
            initial_population = _initialize_population(init_key, fixed_N, mu_vec, sigma_init)
            stats, _, _ = simulate_evolution(
                key=run_key,
                initial_mean=mu_vec,
                num_iterations=num_iterations,
                population_size=fixed_N,
                sigma=mutation_variance,
                num_select=fixed_N,
                fitness_function=fitness_func,
                hessian_func=hessian_func,
                selection_method='linear',
                beta=beta,
                track_hessian=False,
                M=1,
                initial_population=initial_population
            )
            _, pos_hist, _, _, _ = stats
            final_positions.append(np.array(pos_hist[-1]))

        signed_drift, mean_drift, var_drift = _compute_drift_metrics(mu_vec, final_positions)
        mu_results[mu_val] = {
            "mean_drift": mean_drift,
            "var_drift": var_drift,
            "signed_drifts": signed_drift
        }

    fig_mu, axes_mu = create_figure(n_cols=2, width_per_panel=4.5, height_per_panel=4)
    ax_mu_mean, ax_mu_cv = axes_mu

    mu_vals = np.array(mu_0_values, dtype=float)
    mu_mean_drifts = np.array([mu_results[m]["mean_drift"] for m in mu_0_values])
    mu_var_drifts = np.array([mu_results[m]["var_drift"] for m in mu_0_values])
    mu_cv = np.sqrt(mu_var_drifts) / np.maximum(np.abs(mu_mean_drifts), 1e-8)

    ax_mu_mean.plot(mu_vals, mu_mean_drifts, marker='o', linewidth=1.5)
    ax_mu_mean.set_xscale('log')
    ax_mu_mean.set_xlabel(r"Initial $|\mu_0|$")
    ax_mu_mean.set_ylabel(r"Mean drift $\langle D \rangle$")
    ax_mu_mean.set_title("Drift vs starting distance")
    style_axis(ax_mu_mean)

    ax_mu_cv.plot(mu_vals, mu_cv, marker='o', linewidth=1.5, color="#8c564b")
    ax_mu_cv.set_xscale('log')
    ax_mu_cv.set_xlabel(r"Initial $|\mu_0|$")
    ax_mu_cv.set_ylabel("CV")
    ax_mu_cv.set_title("Noise-to-signal ratio")
    style_axis(ax_mu_cv)

    adjust_layout(fig_mu, left=0.12, right=0.95, top=0.88, bottom=0.15, wspace=0.35)
    save_figure(fig_mu, "curvature_drift_mu0_sweep.png")

    # Data collapse plot across both sweeps
    max_mean_drift = max(np.max(mean_drifts), np.max(mu_mean_drifts), 1e-8)
    normalized_N_drift = mean_drifts / max_mean_drift
    normalized_mu_drift = mu_mean_drifts / max_mean_drift

    fig_collapse, ax_collapse = create_figure(n_cols=1, width_per_panel=4.5, height_per_panel=4)
    ax_collapse = ax_collapse if isinstance(ax_collapse, plt.Axes) else ax_collapse[0]

    D_N = np.array([D_values[N] for N in population_sizes])
    D_mu = np.array([mu_D_values[m] for m in mu_0_values])

    ax_collapse.scatter(D_N, normalized_N_drift, label="N sweep", s=40, alpha=0.85)
    ax_collapse.scatter(D_mu, normalized_mu_drift, label=r"$\mu_0$ sweep", s=40, alpha=0.85)
    ax_collapse.axvline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_collapse.set_xscale('log')
    ax_collapse.set_xlabel(r"Dimensionless drift $\mathcal{D}$")
    ax_collapse.set_ylabel(r"$\langle D \rangle / D_{\mathrm{det}}$")
    ax_collapse.set_title("Data collapse on drift control parameter")
    ax_collapse.legend(frameon=False)
    style_axis(ax_collapse)

    adjust_layout(fig_collapse, left=0.15, right=0.95, top=0.88, bottom=0.15, wspace=0.3)
    save_figure(fig_collapse, "curvature_drift_data_collapse.png")

    print("Done! Generated plots:")
    print("- curvature_drift_transition.png")
    print("- curvature_drift_mu0_sweep.png")
    print("- curvature_drift_data_collapse.png")

def verify_regime_of_dominance():
    print("Verifying 'Regime of Dominance'...")
    setup_style()
    
    # Parameters from Fig 3
    NUM_ITERATIONS = 200  # Increased slightly to see long term behavior
    MUTATION_STD = 0.05
    BETA = 0.1
    F_MAX = 5.0
    SIGMA = MUTATION_STD**2
    
    # Population sizes to test (log scale)
    # Ranging from very small (high noise) to very large (deterministic)
    population_sizes = [50, 200, 1000, 5000, 25000, 100000]
    
    # Setup landscape
    key = jax.random.PRNGKey(42)
    fitness_func, hessian_func, _ = create_landscape_2d(F_MAX)
    initial_mean = jnp.array([2.5, 0.0]) # Starting far out on x-axis
    
    # 1. Run Theoretical Dynamics (Infinite Population Limit)
    print("Running Theoretical Dynamics...")
    initial_mu = float(initial_mean[0])
    theory_stats = simulate_theoretical_manifold_dynamics(
        initial_mu=initial_mu,
        initial_a=SIGMA,
        initial_b=SIGMA,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        m_x=SIGMA,
        m_y=SIGMA
    )
    theory_time, theory_mu, theory_a, theory_b = theory_stats
    
    # 2. Run Empirical Simulations for different N
    empirical_results = {}
    
    for N in population_sizes:
        print(f"Running simulation for N={N}...")
        key, subkey = jax.random.split(key)
        
        # We run multiple replicates to visualize the spread/noise
        num_replicates = 50
        replicate_trajectories = []
        
        for i in range(num_replicates):
            key, rep_key = jax.random.split(key)
            ed_stats, _, _ = simulate_evolution(
                key=rep_key,
                initial_mean=initial_mean,
                num_iterations=NUM_ITERATIONS,
                population_size=N,
                sigma=SIGMA,
                num_select=N,
                fitness_function=fitness_func,
                hessian_func=hessian_func,
                selection_method='linear',
                beta=BETA,
                track_hessian=False, # Speed up
                M=1
            )
            _, pos_hist, _, _, _ = ed_stats
            replicate_trajectories.append(pos_hist[:, 0]) # Keep x-coordinate (mu)
            
        empirical_results[N] = replicate_trajectories

    # 3. Analyze and Plot
    
    # Plot 1: Trajectories vs Time for different N
    fig, ax = create_figure(n_cols=1, width_per_panel=7.0, height_per_panel=4.2)
    if isinstance(ax, np.ndarray):
        ax = ax.flat[0]
    
    # Plot Theory
    theory_color = get_algorithm_color(AlgorithmEnum.SHIFT_MODEL_THEORETICAL.value)
    empirical_colors = get_colors_for_values(population_sizes, use_odd_indices=False)
    ax.plot(theory_time, theory_mu, color=theory_color, linestyle='--', linewidth=2.0, label='Theory (N -> inf)')
    
    for idx, N in enumerate(population_sizes):
        trajs = empirical_results[N]
        # Plot mean of replicates
        mean_traj = np.mean(trajs, axis=0)
        # Plot standard deviation as shaded region? Or just individual lines?
        # Let's plot the first replicate as solid line, others transparent
        ax.plot(theory_time, mean_traj, color=empirical_colors[idx], linewidth=1.5, label=f'N={N}')
        
        # Plot individual replicates to show noise
        # for traj in trajs:
        #     ax.plot(theory_time, traj, color=colors[idx], alpha=0.2, linewidth=1)
            
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Position (μ)')
    ax.set_title('Curvature Drift vs Population Size')
    style_axis(ax)
    ax.legend(frameon=False, ncol=2)
    adjust_layout(fig, left=0.1, right=0.96, top=0.88, bottom=0.16, wspace=0.35)
    
    save_figure(fig, 'dominance_trajectories.png')
    
    # Plot 2: Drift Intensity Analysis
    # We want to check the condition: N > 1 / ((mu * beta * b)^2 * a)
    # We can calculate the "Critical N" over time for the theoretical trajectory
    
    critical_N_over_time = []
    drift_intensity_over_time = [] # For a fixed reference N (e.g. 1000)
    
    for t in range(len(theory_time)):
        mu_t = theory_mu[t]
        a_t = theory_a[t]
        b_t = theory_b[t]
        
        # Avoid division by zero
        if mu_t < 1e-6:
            mu_t = 1e-6
            
        # Theoretical Drift Velocity V_drift = mu * beta * a * b
        # (Note: formula in prompt is mu * beta * a * b, let's verify magnitude)
        # Actually prompt says V_drift approx - mu * beta * a * b
        v_drift = abs(mu_t * BETA * a_t * b_t)
        
        # Noise level for N=1: V_noise_1 = sqrt(a_t)
        # Condition: V_drift > sqrt(a_t / N)
        # V_drift^2 > a_t / N
        # N > a_t / V_drift^2
        # N > a_t / (mu * beta * a * b)^2
        # N > 1 / (mu^2 * beta^2 * a * b^2)
        
        denom = (mu_t * BETA * b_t)**2 * a_t
        if denom > 0:
            crit_N = 1.0 / denom
        else:
            crit_N = float('inf')
            
        critical_N_over_time.append(crit_N)
        
    fig2, ax2 = create_figure(n_cols=1, width_per_panel=7.0, height_per_panel=4.2)
    if isinstance(ax2, np.ndarray):
        ax2 = ax2.flat[0]
    threshold_color = get_algorithm_color(AlgorithmEnum.EVOLUTION.value)
    ax2.plot(theory_time, critical_N_over_time, color=threshold_color, linewidth=1.8, label='Critical N threshold')
    ax2.set_yscale('log')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Critical Population Size (N*)')
    ax2.set_title('Regime of Dominance Threshold\n(N > N* for Drift to Dominate)')
    
    # Add horizontal lines for the N we tested
    for N in population_sizes:
        ax2.axhline(y=N, color=COLORS[7], linestyle='--', alpha=0.4, linewidth=0.9)
        ax2.text(0, N, f' N={N}', verticalalignment='bottom', color=COLORS[7], fontsize=8)
        
    style_axis(ax2)
    ax2.legend(frameon=False)
    adjust_layout(fig2, left=0.1, right=0.96, top=0.88, bottom=0.16, wspace=0.35)
    save_figure(fig2, 'dominance_threshold.png')
    
    # Plot 3: Empirical "Velocity" vs N at a specific snapshot (e.g. t=50)
    # We want to see if the actual drift matches theory for large N and deviates for small N
    
    snapshot_t = 50
    theoretical_velocity = -(theory_mu[snapshot_t+1] - theory_mu[snapshot_t])
    
    observed_velocities = []
    observed_noise = []
    
    for N in population_sizes:
        trajs = empirical_results[N]
        # Calculate velocity around snapshot_t
        # Averaging over a small window to reduce noise
        vels = []
        for traj in trajs:
            v = -(traj[snapshot_t+5] - traj[snapshot_t-5]) / 10.0
            vels.append(v)
        observed_velocities.append(np.mean(vels))
        observed_noise.append(np.std(vels))
        
    fig3, ax3 = create_figure(n_cols=1, width_per_panel=5.8, height_per_panel=4.2)
    if isinstance(ax3, np.ndarray):
        ax3 = ax3.flat[0]
    empirical_color = get_algorithm_color(AlgorithmEnum.EVOLUTION.value)
    reference_color = get_algorithm_color(AlgorithmEnum.SHIFT_MODEL_THEORETICAL.value)
    ax3.errorbar(
        population_sizes,
        observed_velocities,
        yerr=observed_noise,
        fmt='o-',
        color=empirical_color,
        capsize=4,
        linewidth=1.5,
        label='Empirical Velocity'
    )
    ax3.axhline(
        y=theoretical_velocity,
        color=reference_color,
        linestyle='--',
        linewidth=1.5,
        label='Theoretical Drift Velocity'
    )
    ax3.set_xscale('log')
    ax3.set_xlabel('Population Size (N)')
    ax3.set_ylabel('Drift Velocity')
    ax3.set_title(f'Drift Velocity at t={snapshot_t}')
    style_axis(ax3)
    ax3.legend(frameon=False)
    adjust_layout(fig3, left=0.12, right=0.96, top=0.88, bottom=0.16, wspace=0.3)
    save_figure(fig3, 'dominance_velocity_scaling.png')

    print("Done! Generated plots: dominance_trajectories.png, dominance_threshold.png, dominance_velocity_scaling.png")

if __name__ == "__main__":
    run_curvature_drift_transition()
