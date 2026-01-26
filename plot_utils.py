"""
Plotting utilities with consistent styling for evolutionary dynamics figures.

This module provides a unified styling system for all matplotlib plots,
ensuring visual consistency across figures. The style is based on 
scientific publication standards with clean, minimal aesthetics.

Usage:
    from plot_utils import (
        setup_style, style_axis, create_figure, 
        get_algorithm_color, AlgorithmEnum, COLORS
    )
    
    # Setup global style (call once at start)
    setup_style()
    
    # Create a styled figure
    fig, axes = create_figure(n_cols=3)
    
    # Style individual axes
    for ax in axes:
        style_axis(ax)
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import ticker
import numpy as np
from enum import StrEnum
from typing import Optional, Dict, Tuple, List, Union
import seaborn as sns

# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================

class AlgorithmEnum(StrEnum):
    """Enum for algorithm names used in plots."""
    EVOLUTION = 'ED'
    GRADIENT_ASCENT = 'GLD'
    SHIFT_GRADIENT_ASCENT = 'SGD'
    NES = 'NES'
    NES_ANALYTICAL = 'NES Analytical'
    NES_NUMERICAL = 'NES Numerical'
    GLD_THEORETICAL = 'GLD (Theory)'
    SHIFT_MODEL_THEORETICAL = 'Shift Model (Theory)'
    FULL_NES = 'Full NES'


# Base color palette (tab10)
COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-green
    '#17becf',  # Cyan
]

# Algorithm-specific color mapping for consistency across all plots
ALGORITHM_COLORS = {
    AlgorithmEnum.EVOLUTION.value: '#d62728',              # Red
    AlgorithmEnum.GRADIENT_ASCENT.value: '#8c564b',        # Brown
    AlgorithmEnum.SHIFT_GRADIENT_ASCENT.value: '#1f77b4',  # Blue
    AlgorithmEnum.NES.value: '#2ca02c',                    # Green
    AlgorithmEnum.NES_ANALYTICAL.value: '#2ca02c',         # Green
    AlgorithmEnum.NES_NUMERICAL.value: '#2ca02c',          # Green
    AlgorithmEnum.GLD_THEORETICAL.value: '#8c564b',        # Brown (same as GLD)
    AlgorithmEnum.SHIFT_MODEL_THEORETICAL.value: '#1f77b4', # Blue (same as SGD)
    AlgorithmEnum.FULL_NES.value: '#9467bd',               # Purple
}

# Default styling parameters
DEFAULT_STYLE = {
    'figure_width_per_panel': 4,
    'figure_height': 4,
    'linewidth': 1.5,
    'marker_size': 10,
    'scatter_alpha': 0.22,
    'scatter_size': 10.0,
    'contour_levels': 21,
    'contour_alpha': 0.3,
    'contour_linewidth': 0.5,
    'spine_linewidth': 0.5,
    'tick_length': 3,
    'tick_width': 0.5,
    'label_fontsize': 12,
    'tick_fontsize': 10,
    'legend_fontsize': 10,
    'subplot_label_fontsize': 14,
}


# =============================================================================
# STYLE SETUP
# =============================================================================

def setup_style():
    """
    Configure matplotlib rcParams for consistent, publication-quality figures.
    Call this once at the start of your script.
    """
    plt.rcParams.update({
        # Font settings
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        
        # DPI settings
        "figure.dpi": 300,
        "savefig.dpi": 300,
        
        # Line widths
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        
        # Legend
        "legend.frameon": False,
        
        # Colors
        "axes.edgecolor": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })


def get_algorithm_color(algorithm_name: str) -> str:
    """
    Get consistent color for an algorithm.
    
    Args:
        algorithm_name: Name of the algorithm (should match AlgorithmEnum values)
        
    Returns:
        Hex color string
    """
    return ALGORITHM_COLORS.get(algorithm_name, '#7f7f7f')  # Default gray


def get_colors_for_values(values: list, use_odd_indices: bool = True) -> list:
    """
    Get a list of distinct colors for a set of values.
    
    Args:
        values: List of values to assign colors to
        use_odd_indices: If True, uses odd-indexed colors (more distinct)
        
    Returns:
        List of hex color strings
    """
    if use_odd_indices:
        # Pick every other color (odd indices) for better distinction
        base_colors = [COLORS[2 * i + 1] for i in range(min(len(values), 5))]
        if len(values) > 5:
            base_colors.extend(COLORS[:len(values) - 5])
        return base_colors[:len(values)]
    else:
        return COLORS[:len(values)]


# =============================================================================
# AXIS & FIGURE STYLING
# =============================================================================

def style_axis(ax: plt.Axes, 
               remove_top_right: bool = True,
               spine_linewidth: float = 0.5,
               tick_direction: str = 'out',
               tick_length: float = 3,
               tick_width: float = 0.5,
               tick_labelsize: float = 10,
               background_color: str = 'white') -> None:
    """
    Apply consistent styling to a matplotlib axis.
    
    This is the core styling function - use it on every axis for consistency.
    
    Args:
        ax: Matplotlib Axes object to style
        remove_top_right: Whether to remove top and right spines
        spine_linewidth: Width of visible spines
        tick_direction: Direction of ticks ('in', 'out', 'inout')
        tick_length: Length of tick marks
        tick_width: Width of tick marks
        tick_labelsize: Font size for tick labels
        background_color: Background color for the axes
    """
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Set spine width for remaining spines
    ax.spines['bottom'].set_linewidth(spine_linewidth)
    ax.spines['left'].set_linewidth(spine_linewidth)
    
    # Configure ticks
    ax.tick_params(
        direction=tick_direction,
        length=tick_length,
        width=tick_width,
        labelsize=tick_labelsize
    )
    
    # Set background
    ax.set_facecolor(background_color)


def create_figure(n_cols: int = 1, 
                  n_rows: int = 1,
                  width_per_panel: float = 4,
                  height_per_panel: float = 4,
                  **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create a figure with consistent sizing.
    
    Args:
        n_cols: Number of columns
        n_rows: Number of rows
        width_per_panel: Width of each panel in inches
        height_per_panel: Height of each panel in inches
        **kwargs: Additional arguments passed to plt.subplots
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(width_per_panel * n_cols, height_per_panel * n_rows),
        **kwargs
    )
    fig.patch.set_facecolor('white')
    return fig, axes


def add_subplot_labels(axes: Union[plt.Axes, np.ndarray, list],
                       labels: Optional[list] = None,
                       x_offset: float = -0.1,
                       y_offset: float = 1.2,
                       fontsize: float = 14) -> None:
    """
    Add subplot labels (a), (b), (c), etc. to axes.
    
    Args:
        axes: Single axis or array/list of axes
        labels: Custom labels. If None, uses (a), (b), (c), ...
        x_offset: Horizontal offset in axes coordinates
        y_offset: Vertical offset in axes coordinates
        fontsize: Font size for labels
    """
    # Handle single axis
    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    if labels is None:
        labels = [f'({chr(ord("a") + i)})' for i in range(len(axes))]
    
    for ax, label in zip(axes, labels):
        ax.text(x_offset, y_offset, label,
                transform=ax.transAxes,
                fontsize=fontsize,
                va='top', ha='right')


def set_equal_aspect(ax: plt.Axes, ratio_based: bool = False) -> None:
    """
    Set aspect ratio for an axis.
    
    Args:
        ax: Matplotlib Axes object
        ratio_based: If True, uses data ratio. If False, sets equal aspect.
    """
    if ratio_based:
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    else:
        ax.set_aspect('equal', adjustable='box')


def adjust_layout(fig: plt.Figure,
                  left: float = 0.15,
                  right: float = 0.85,
                  top: float = 0.9,
                  bottom: float = 0.15,
                  wspace: float = 0.5,
                  hspace: float = 0.35) -> None:
    """
    Adjust subplot layout with consistent spacing.
    
    Args:
        fig: Matplotlib Figure object
        left, right, top, bottom: Margins
        wspace: Width spacing between subplots
        hspace: Height spacing between subplots
    """
    plt.subplots_adjust(
        left=left, right=right,
        top=top, bottom=bottom,
        wspace=wspace, hspace=hspace
    )


# =============================================================================
# LEGEND HELPERS
# =============================================================================

def add_legend(fig: plt.Figure,
               handles: list,
               loc: str = 'upper center',
               ncol: int = 3,
               fontsize: float = 10,
               bbox_to_anchor: Tuple[float, float] = (0.5, 0.88),
               frameon: bool = False) -> None:
    """
    Add a legend to the figure with consistent styling.
    
    Args:
        fig: Matplotlib Figure object
        handles: List of legend handles
        loc: Legend location
        ncol: Number of columns
        fontsize: Font size
        bbox_to_anchor: Anchor position
        frameon: Whether to show legend frame
    """
    fig.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon
    )


# =============================================================================
# CONTOUR PLOT HELPERS
# =============================================================================

def add_fitness_contours(ax: plt.Axes,
                         fitness_function,
                         xlim: Tuple[float, float] = (-3, 3),
                         ylim: Tuple[float, float] = (-3, 3),
                         resolution: int = 200,
                         levels: int = 21,
                         vmin: float = 0,
                         vmax: float = 10,
                         color: str = 'black',
                         linewidth: float = 0.5,
                         alpha: float = 0.3) -> None:
    """
    Add fitness landscape contour lines to an axis.
    
    Args:
        ax: Matplotlib Axes object
        fitness_function: Function that takes (N, 2) array and returns (N,) array
        xlim: Tuple of (xmin, xmax)
        ylim: Tuple of (ymin, ymax)
        resolution: Grid resolution
        levels: Number of contour levels
        vmin, vmax: Value range for contours
        color: Contour line color
        linewidth: Contour line width
        alpha: Contour transparency
    """
    try:
        import jax.numpy as jnp
        import jax
        
        x = jnp.linspace(xlim[0], xlim[1], resolution)
        y = jnp.linspace(ylim[0], ylim[1], resolution)
        X, Y = jnp.meshgrid(x, y)
        grid_points = jnp.stack([X, Y], axis=-1)
        Z = jax.vmap(jax.vmap(fitness_function))(grid_points)
        
        # Convert to numpy for matplotlib
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    except ImportError:
        # Fallback to numpy if JAX not available
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = fitness_function(np.array([X[i, j], Y[i, j]]))
    
    ax.contour(X, Y, Z, 
               levels=np.linspace(vmin, vmax, levels),
               colors=color, 
               linewidths=linewidth, 
               alpha=alpha)


# =============================================================================
# SCATTER PLOT HELPERS
# =============================================================================

def scatter_population(ax: plt.Axes,
                       population: np.ndarray,
                       color: str = '#d62728',
                       alpha: float = 0.22,
                       size: float = 10.0,
                       zorder: int = 3,
                       **kwargs) -> None:
    """
    Scatter plot a population with consistent styling.
    
    Args:
        ax: Matplotlib Axes object
        population: Array of shape (N, 2)
        color: Point color
        alpha: Transparency
        size: Point size
        zorder: Z-order for layering
        **kwargs: Additional scatter kwargs
    """
    ax.scatter(
        population[:, 0], population[:, 1],
        color=color, alpha=alpha, s=size,
        linewidths=0, zorder=zorder,
        **kwargs
    )


def scatter_point(ax: plt.Axes,
                  x: float, y: float,
                  color: str = '#8c564b',
                  size: float = 50,
                  marker: str = '.',
                  zorder: int = 11,
                  **kwargs) -> None:
    """
    Plot a single point (e.g., mean position) with consistent styling.
    
    Args:
        ax: Matplotlib Axes object
        x, y: Point coordinates
        color: Point color
        size: Point size
        marker: Marker style
        zorder: Z-order for layering
        **kwargs: Additional scatter kwargs
    """
    ax.scatter(x, y, color=color, s=size, marker=marker, zorder=zorder, **kwargs)


# =============================================================================
# ELLIPSE HELPERS
# =============================================================================

def add_covariance_ellipse(ax: plt.Axes,
                           center: Tuple[float, float],
                           cov_matrix: np.ndarray,
                           n_sigma: float = 3.0,
                           color: str = '#8c564b',
                           linewidth: float = 0.9,
                           linestyle: str = '-',
                           alpha: float = 1.0,
                           fill: bool = False,
                           zorder: int = 10) -> None:
    """
    Add a covariance ellipse to an axis.
    
    Args:
        ax: Matplotlib Axes object
        center: (x, y) center of ellipse
        cov_matrix: 2x2 covariance matrix
        n_sigma: Number of standard deviations for ellipse size
        color: Ellipse color
        linewidth: Line width
        linestyle: Line style
        alpha: Transparency
        fill: Whether to fill the ellipse
        zorder: Z-order for layering
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute ellipse dimensions
    width = 2 * n_sigma * np.sqrt(eigenvalues[0])
    height = 2 * n_sigma * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    ellipse = mpatches.Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color if fill else 'none',
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(ellipse)


# =============================================================================
# TRAJECTORY PLOT HELPERS
# =============================================================================

def plot_trajectory(ax: plt.Axes,
                    positions: np.ndarray,
                    color: str = '#d62728',
                    linestyle: str = '-',
                    linewidth: float = 1.5,
                    alpha: float = 0.9,
                    label: Optional[str] = None,
                    time_gradient: bool = False) -> None:
    """
    Plot a trajectory with consistent styling.
    
    Args:
        ax: Matplotlib Axes object
        positions: Array of shape (T, 2) for 2D trajectory, or (T, D) for higher D
        color: Line color
        linestyle: Line style
        linewidth: Line width
        alpha: Transparency
        label: Optional label for legend
        time_gradient: If True, fade color over time
    """
    if time_gradient and len(positions) > 1:
        # Plot segments with fading color
        import matplotlib.colors as mcolors
        
        def adjust_brightness(color_hex, factor):
            rgb = mcolors.to_rgb(color_hex)
            return tuple(min(1.0, c + (1 - c) * factor) for c in rgb)
        
        for i in range(len(positions) - 1):
            brightness = i / len(positions)
            segment_color = adjust_brightness(color, brightness)
            ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                   color=segment_color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha)
    else:
        ax.plot(positions[:, 0], positions[:, 1],
               color=color, linestyle=linestyle,
               linewidth=linewidth, alpha=alpha, label=label)


def plot_timeseries(ax: plt.Axes,
                    time: np.ndarray,
                    values: np.ndarray,
                    color: str = '#d62728',
                    linestyle: str = '-',
                    linewidth: float = 1.5,
                    alpha: float = 0.9,
                    label: Optional[str] = None) -> None:
    """
    Plot a time series with consistent styling.
    
    Args:
        ax: Matplotlib Axes object
        time: Array of time points
        values: Array of values
        color: Line color
        linestyle: Line style
        linewidth: Line width
        alpha: Transparency
        label: Optional label for legend
    """
    ax.plot(time, values,
            color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha, label=label)


# =============================================================================
# KDE PLOT HELPERS
# =============================================================================

def plot_kde(ax: plt.Axes,
             data: np.ndarray,
             color: str = '#d62728',
             alpha: float = 0.5,
             fill: bool = True,
             label: Optional[str] = None) -> None:
    """
    Plot a kernel density estimate with consistent styling.
    
    Args:
        ax: Matplotlib Axes object
        data: 1D array of data points
        color: Line/fill color
        alpha: Fill transparency
        fill: Whether to fill under the curve
        label: Optional label for legend
    """
    sns.kdeplot(
        data=data,
        color=color,
        alpha=alpha,
        ax=ax,
        fill=fill,
        label=label
    )


def plot_gaussian(ax: plt.Axes,
                  mean: float,
                  std: float,
                  x_range: Tuple[float, float] = (-3, 3),
                  n_points: int = 500,
                  color: str = '#8c564b',
                  linestyle: str = '-',
                  linewidth: float = 1,
                  alpha: float = 0.8,
                  fill: bool = True,
                  fill_alpha: float = 0.2,
                  label: Optional[str] = None) -> None:
    """
    Plot a Gaussian distribution with consistent styling.
    
    Args:
        ax: Matplotlib Axes object
        mean: Gaussian mean
        std: Gaussian standard deviation
        x_range: Range for x-axis
        n_points: Number of points to plot
        color: Line color
        linestyle: Line style
        linewidth: Line width
        alpha: Line transparency
        fill: Whether to fill under curve
        fill_alpha: Fill transparency
        label: Optional label for legend
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    ax.plot(x, pdf, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha, label=label)
    
    if fill:
        ax.fill_between(x, pdf, color=color, alpha=fill_alpha)


# =============================================================================
# SAVE HELPERS
# =============================================================================

# Dedicated folder for saving figures
FIGURES_OUTPUT_DIR = "figures"


def save_figure(fig: plt.Figure,
                filename: str,
                dpi: int = 300,
                format: str = 'jpeg',
                bbox_inches: str = 'tight',
                verbose: bool = True) -> None:
    """
    Save a figure with consistent settings to the dedicated figures folder.
    
    Args:
        fig: Matplotlib Figure object
        filename: Output filename (extension will be added if missing)
        dpi: Resolution
        format: Output format ('jpeg', 'png', 'pdf', 'svg')
        bbox_inches: Bounding box setting
        verbose: Whether to print confirmation
    """
    import os
    
    # Ensure the output directory exists
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
    
    # Ensure correct extension
    if not filename.endswith(f'.{format}'):
        filename = f"{filename}.{format}"
    
    # Prepend the output directory to the filename
    filepath = os.path.join(FIGURES_OUTPUT_DIR, filename)
    
    fig.savefig(filepath, dpi=dpi, format=format, bbox_inches=bbox_inches)
    
    if verbose:
        print(f"Figure saved as {filepath}")


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def plot_high_dim_sigma_comparison(
    results: Dict[str, Tuple],
    hessian_func,
    color_map: Optional[Dict[str, str]] = None,
    flat_k: int = 0,
    sharp_k: int = 0,
    focus_label: Optional[str] = None
) -> plt.Figure:
    """
    Plot high-D ED comparison across sigma values.
    
    Panels:
    - (a) Fitness over time
    - (b) Curvature (|trace(H)|) over time
    - (c) All |eigenvalue| traces for middle sigma, grouped by flat vs sharp
    """
    try:
        import jax.numpy as jnp
        import jax
    except ImportError:
        raise ImportError("JAX is required for this function")
    
    fig, axes = create_figure(n_cols=3)
    ax_fitness, ax_curvature, ax_eigs = axes
    
    legend_elements = []
    focus_stats = None
    focus_time = None
    
    for label, stats in results.items():
        time_data, pos_history, fitness_data, hessian_trace_data = stats[:4]
        color = color_map.get(label) if color_map is not None else get_algorithm_color(label)
        
        plot_timeseries(ax_fitness, np.array(time_data), np.array(fitness_data), color=color)
        
        curvature = np.abs(np.array(hessian_trace_data))
        plot_timeseries(ax_curvature, np.array(time_data), curvature, color=color)
        
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))
        
        if focus_label is not None and label == focus_label:
            focus_stats = stats
            focus_time = time_data
    
    if focus_stats is None:
        labels = list(results.keys())
        if labels:
            focus_label = labels[len(labels) // 2]
            focus_stats = results[focus_label]
            focus_time = focus_stats[0]
    
    if focus_stats is not None:
        _, pos_history, _, _ = focus_stats[:4]
        try:
            eigvals = jax.vmap(lambda p: jnp.linalg.eigvalsh(hessian_func(p)))(pos_history)
            eigvals_abs = np.abs(np.array(eigvals))
        except Exception:
            eigvals_abs = []
            for p in np.array(pos_history):
                eigvals_abs.append(np.abs(np.linalg.eigvalsh(np.array(hessian_func(p)))))
            eigvals_abs = np.array(eigvals_abs)
        
        eigvals_abs_sorted = np.sort(eigvals_abs, axis=1)
        _, dim = eigvals_abs_sorted.shape
        flat_k = max(0, min(flat_k, dim))
        sharp_k = max(0, min(sharp_k, dim - flat_k))
        
        red = COLORS[3]
        time_np = np.array(focus_time)
        
        for idx in range(flat_k):
            ax_eigs.plot(
                time_np, eigvals_abs_sorted[:, idx],
                color=red, linestyle='--', alpha=0.3, linewidth=1.0
            )
        for idx in range(dim - sharp_k, dim):
            ax_eigs.plot(
                time_np, eigvals_abs_sorted[:, idx],
                color=red, linestyle='-', alpha=0.5, linewidth=1.0
            )
    
    
    ax_fitness.set_xlabel("Generation")
    ax_fitness.set_ylabel("Fitness")
    style_axis(ax_fitness)
    set_equal_aspect(ax_fitness, ratio_based=True)
    
    ax_curvature.set_xlabel("Generation")
    ax_curvature.set_ylabel("Curvature")
    style_axis(ax_curvature)
    set_equal_aspect(ax_curvature, ratio_based=True)
    
    ax_eigs.set_xlabel("Generation")
    ax_eigs.set_ylabel(r"$|\lambda|$")
    style_axis(ax_eigs)
    set_equal_aspect(ax_eigs, ratio_based=True)
    ax_eigs.legend(
        handles=[
            Line2D([0], [0], color=COLORS[3], linestyle='--', label='flat'),
            Line2D([0], [0], color=COLORS[3], linestyle='-', label='sharp')
        ],
        frameon=False,
        fontsize=8,
        loc='upper right'
    )
    
    if len(legend_elements) > 1:
        add_legend(fig, legend_elements, ncol=min(len(legend_elements), 4))
    
    add_subplot_labels(axes)
    adjust_layout(fig, left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.4)
    plt.show()
    
    return fig


# =============================================================================
# HISTOGRAM PLOTS
# =============================================================================

def plot_curvature_and_fitness_histograms(
    final_population_curvatures: np.ndarray,
    final_population_fitnesses: np.ndarray,
    bins: int = 50
):
    """
    Plot histograms of both curvature and fitness for a population using semi-log scale.
    
    Args:
        final_population_curvatures: Array of curvature values
        final_population_fitnesses: Array of fitness values
        bins: Number of histogram bins
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot curvature histogram
    ax1.hist(final_population_curvatures, bins=bins, density=True, alpha=0.7,
             color='#d62728', edgecolor='black')
    ax1.set_yscale('log')

    # Plot fitness histogram
    ax2.hist(final_population_fitnesses, bins=bins, density=True, alpha=0.7,
             color='#d62728', edgecolor='black')
    ax2.set_yscale('log')

    # Customize both axes
    for ax in [ax1, ax2]:
        style_axis(ax)
        ax.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation())
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    ax1.set_ylabel('log(Density)', fontsize=14)
    ax1.set_xlabel('Curvature', fontsize=14)
    ax2.set_xlabel('Fitness', fontsize=14)

    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=14)
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=14)

    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15, wspace=0.4)
    plt.show()
    
    return fig


# =============================================================================
# EIGENVALUE PLOTS
# =============================================================================
# ED VS NES SUMMARY PLOTS
# =============================================================================

def plot_ed_vs_full_nes_summary(
    ed_populations: Dict[int, np.ndarray],
    ed_statistics: Tuple,
    full_nes_statistics: Tuple,
    fitness_function,
    snapshot_times: List[int],
    params: Dict,
    add_ellipse: bool = True,
    save_fig: bool = True,
    filename_suffix: str = ''
):
    """
    Create advanced summary plot comparing ED and Full NES algorithms.
    
    Args:
        ed_populations: Dictionary mapping time -> ED population array
        ed_statistics: Tuple from run_evolution (time, positions, fitness, hessian, covariance)
        full_nes_statistics: Tuple from run_full_natural_gradient_es
        fitness_function: Fitness function for contour plots
        snapshot_times: List of time points to show in scatter plots
        params: Dictionary with parameters (num_iterations, population_size, mutation_std, etc.)
        add_ellipse: Whether to add covariance ellipses for Full NES
        save_fig: Whether to save the figure to file
    """
    import matplotlib.colors
    
    fig = plt.figure(figsize=(12, 8))
    
    # Create GridSpec layout
    outer_gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], 
                                 hspace=0.5, top=0.88, bottom=0.12, left=0.12, right=0.88)
    
    top_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[0], 
                                            height_ratios=[4, 1], 
                                            hspace=0, wspace=0.3)
    
    bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], wspace=0.4)

    # Unpack statistics
    ed_time_points, ed_pos_history, ed_fitness_history, ed_hessian_traces, ed_cov_matrices = ed_statistics
    nes_time_points, nes_trajectory, nes_fitness_history, nes_hessian_traces, nes_post_mut_cov, nes_cov_history = full_nes_statistics
    
    # Colors
    ed_color = '#d62728'
    nes_color = '#8c564b'
    
    # Styling parameters
    ellipse_sigma = float(params.get("ellipse_sigma", 3.0))
    ellipse_lw = float(params.get("ellipse_linewidth", 0.9))
    ed_point_alpha = float(params.get("ed_point_alpha", 0.22))
    ed_point_size = float(params.get("ed_point_size", 10.0))
    nes_point_size = float(params.get("nes_point_size", 50.0))

    # Create fitness landscape grid
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    
    try:
        import jax
        import jax.numpy as jnp
        grid_points = jnp.stack([jnp.array(X), jnp.array(Y)], axis=-1)
        Z = np.array(jax.vmap(jax.vmap(fitness_function))(grid_points))
    except ImportError:
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = fitness_function(np.array([X[i, j], Y[i, j]]))

    # Plot scatter plots and KDE for each snapshot time
    for i, t in enumerate(snapshot_times):
        ax_scatter = fig.add_subplot(top_gs[0, i])
        
        ax_scatter.contour(X, Y, Z, levels=np.linspace(0, 10, 21), 
                          colors='black', linewidths=0.5, alpha=0.3)
        
        # Plot ED population
        if t in ed_populations:
            ed_pop = np.array(ed_populations[t])
            ax_scatter.scatter(ed_pop[:, 0], ed_pop[:, 1],
                color=ed_color, alpha=ed_point_alpha, s=ed_point_size, linewidths=0, zorder=3)
        
        # Plot Full NES position
        nes_idx = min(t, len(nes_trajectory) - 1)
        ax_scatter.scatter(nes_trajectory[nes_idx, 0], nes_trajectory[nes_idx, 1], 
                          color=nes_color, s=nes_point_size, marker='.', zorder=11)

        # Add ellipse
        if add_ellipse and t < len(nes_post_mut_cov):
            cov_matrix = np.array(nes_post_mut_cov[nes_idx])
            add_covariance_ellipse(
                ax_scatter,
                center=(float(nes_trajectory[nes_idx, 0]), float(nes_trajectory[nes_idx, 1])),
                cov_matrix=cov_matrix,
                n_sigma=ellipse_sigma,
                color=nes_color,
                linewidth=ellipse_lw
            )

        if i == 0:
            ax_scatter.set_ylabel('$y$')
        
        ax_scatter.set_xticklabels([])
        if i != 0:
            ax_scatter.set_yticklabels([])
        ax_scatter.set_xlim(-3, 3)
        ax_scatter.set_ylim(-3, 3)

        # KDE plot
        ax_hist = fig.add_subplot(top_gs[1, i], sharex=ax_scatter)

        if t in ed_populations:
            ed_pop = np.array(ed_populations[t])
            sns.kdeplot(data=ed_pop[:, 0], color=ed_color, alpha=0.5, ax=ax_hist, fill=True, label='ED')
        
        nes_mean_x = float(nes_trajectory[nes_idx, 0])
        nes_cov_matrix = np.array(nes_post_mut_cov[nes_idx])
        nes_std_x = np.sqrt(nes_cov_matrix[0, 0])
        
        x_range = np.linspace(-3, 3, 500)
        nes_pdf = (1 / (nes_std_x * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - nes_mean_x) / nes_std_x) ** 2)
        ax_hist.plot(x_range, nes_pdf, color=nes_color, linestyle='-', linewidth=1, alpha=0.8)
        ax_hist.fill_between(x_range, nes_pdf, color=nes_color, alpha=0.2)
        
        ax_hist.set_xlabel('$x$')
        ax_hist.set_xlim(-3, 3)
        ax_hist.set_yticks([])
        ax_hist.set_ylabel('')

    # Bottom row: Trajectory, Curvature, Fitness
    ax_xy = fig.add_subplot(bottom_gs[0, 0])
    
    def adjust_brightness(color_hex, factor):
        color = matplotlib.colors.to_rgb(color_hex)
        return tuple(min(1.0, c + (1 - c) * factor) for c in color)
    
    ed_pos_np = np.array(ed_pos_history)
    nes_traj_np = np.array(nes_trajectory)
    
    max_time = max(len(ed_pos_np), len(nes_traj_np))
    ed_colors = [adjust_brightness(ed_color, t/max_time) for t in range(len(ed_pos_np))]
    nes_colors = [adjust_brightness(nes_color, t/max_time) for t in range(len(nes_traj_np))]
    
    for j in range(len(ed_pos_np)-1):
        ax_xy.plot(ed_pos_np[j:j+2, 0], ed_pos_np[j:j+2, 1], color=ed_colors[j], 
                 linestyle='--', linewidth=1.5)
    
    for j in range(len(nes_traj_np)-1):
        ax_xy.plot(nes_traj_np[j:j+2, 0], nes_traj_np[j:j+2, 1], color=nes_colors[j], 
                    linestyle='-', linewidth=1.5)
    
    ax_xy.contour(X, Y, Z, levels=np.linspace(0, 10, 21), colors='black', linewidths=0.5, alpha=0.3)
    ax_xy.set_xlim(0, 2)
    ax_xy.set_ylim(-1, 1)
    ax_xy.set_xlabel('$x$')
    ax_xy.set_ylabel('$y$')

    # L2 Norm plot
    ax_norm = fig.add_subplot(bottom_gs[0, 1])
    ed_l2_norm = np.linalg.norm(ed_pos_np, axis=1)
    nes_l2_norm = np.linalg.norm(nes_traj_np, axis=1)
    
    ax_norm.plot(np.array(ed_time_points), ed_l2_norm, color=ed_color, linestyle='--', linewidth=1.5)
    ax_norm.plot(np.array(nes_time_points), nes_l2_norm, color=nes_color, linestyle='-', linewidth=1.5)
    ax_norm.set_xlabel('Time')
    ax_norm.set_ylabel('Curvature')

    # Fitness plot
    ax_fitness = fig.add_subplot(bottom_gs[0, 2])
    ax_fitness.plot(np.array(ed_time_points), np.array(ed_fitness_history), color=ed_color, linewidth=1.5)
    ax_fitness.plot(np.array(nes_time_points), np.array(nes_fitness_history), color=nes_color, linewidth=1.5)
    ax_fitness.set_xlabel('Time')
    ax_fitness.set_ylabel('Fitness')
    ax_fitness.set_xlim(0, params.get('num_iterations', 100))

    # Style all axes
    all_axes = fig.get_axes()
    for k, ax in enumerate(all_axes):
        style_axis(ax)
        if k in [0, 2, 4, 6]:
            ax.set_aspect('equal', adjustable='box')
        elif k in [1, 3, 5]:
            ax.set_box_aspect(0.25)
        else:
            ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    # Add subplot labels
    subplot_labels = {0: '(a)', 2: '(b)', 4: '(c)', 6: '(d)', 7: '(e)', 8: '(f)'}
    for k, label in subplot_labels.items():
        if k < len(all_axes):
            all_axes[k].text(-0.1, 1.25, label, transform=all_axes[k].transAxes, 
                            fontsize=14, va='top', ha='right')
    
    fig.patch.set_facecolor('white')

    if save_fig:
        base_filename = f"ed_vs_full_nes_summary_iter{params.get('num_iterations', 100)}_pop{params.get('population_size', 1000)}_mut{params.get('mutation_std', 0.1):.3f}{filename_suffix}"
        save_figure(fig, base_filename, format='jpeg')
    
    plt.show()
    
    return fig


# =============================================================================
# DYNAMICS THEORY VS EMPIRICAL PLOTS
# =============================================================================

def plot_dynamics_theory_vs_empirical(
    ed_statistics: Tuple,
    theory_statistics: Tuple,
    full_nes_statistics: Tuple,
    fitness_function,
    params: Dict,
    save_fig: bool = True,
    snapshot_times: Optional[List[int]] = None,
    theory_statistics_for_ellipses: Optional[Tuple] = None
):
    """
    Plot comparison of theoretical predictions vs empirical dynamics.
    
    Panels:
      (a) a_t and b_t over time (same plot)
      (b) a_t and b_t vs μ_t
      (c) theory ellipses on fitness landscape
    
    Args:
        ed_statistics: Tuple from run_evolution (time, positions, fitness, hessian, covariance)
        full_nes_statistics: Tuple from run_full_natural_gradient_es
        theory_statistics: Tuple from run_manifold_dynamics_theoretical (time, mu, a, b)
        fitness_function: Fitness function for contour plots
        params: Dictionary with simulation parameters
        save_fig: Whether to save the figure
        snapshot_times: Optional list of time indices for theory ellipses
        theory_statistics_for_ellipses: Optional theory stats for panel (c)
    """
    # Unpack statistics
    ed_time, ed_pos, ed_fitness, ed_hessian, ed_cov = ed_statistics
    if full_nes_statistics is not None:
        nes_time, nes_traj, nes_fitness, nes_hessian, nes_post_mut_cov, nes_cov = full_nes_statistics
    else:
        nes_time = None
        nes_traj = None
        nes_fitness = None
        nes_hessian = None
        nes_post_mut_cov = None
        nes_cov = None
    theory_time, theory_mu, theory_a, theory_b = theory_statistics
    
    # Extract empirical values
    ed_mu = np.array(ed_pos)[:, 0]
    ed_a = np.array([cov[0, 0] for cov in ed_cov])
    ed_b = np.array([cov[1, 1] for cov in ed_cov])
    
    if nes_time is not None:
        nes_mu = np.array(nes_traj)[:, 0]
    else:
        nes_mu = None
    if nes_time is not None:
        nes_a = np.array([cov[0, 0] for cov in nes_cov])
    else:
        nes_a = None
    if nes_time is not None:
        nes_b = np.array([cov[1, 1] for cov in nes_cov])
    else:
        nes_b = None
    
    # Colors and styles
    a_color = COLORS[0]
    b_color = COLORS[1]
    ed_style = '-'
    nes_style = '-.'
    theory_style = '--'
    
    # Create figure
    fig, axes = create_figure(n_cols=3)
    
    # Plot (a): a_t and b_t over time
    axes[0].plot(ed_time[:len(ed_a)], ed_a, color=a_color, linewidth=1.5, linestyle=ed_style, alpha=0.9)
    axes[0].plot(ed_time[:len(ed_b)], ed_b, color=b_color, linewidth=1.5, linestyle=ed_style, alpha=0.9)
    if nes_time is not None:
        axes[0].plot(nes_time[:len(nes_a)], nes_a, color=a_color, linewidth=1.5, linestyle=nes_style, alpha=0.9)
        axes[0].plot(nes_time[:len(nes_b)], nes_b, color=b_color, linewidth=1.5, linestyle=nes_style, alpha=0.9)
    axes[0].plot(theory_time, theory_a, color=a_color, linewidth=2, linestyle=theory_style, alpha=0.9)
    axes[0].plot(theory_time, theory_b, color=b_color, linewidth=2, linestyle=theory_style, alpha=0.9)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel(r'$a_t, b_t$')
    set_equal_aspect(axes[0], ratio_based=True)
    style_axis(axes[0])
    
    # Plot (b): a_t and b_t vs μ_t
    axes[1].plot(ed_mu[:len(ed_a)], ed_a, color=a_color, linewidth=1.5, linestyle=ed_style, alpha=0.9)
    axes[1].plot(ed_mu[:len(ed_b)], ed_b, color=b_color, linewidth=1.5, linestyle=ed_style, alpha=0.9)
    if nes_time is not None:
        axes[1].plot(nes_mu[:len(nes_a)], nes_a, color=a_color, linewidth=1.5, linestyle=nes_style, alpha=0.9)
        axes[1].plot(nes_mu[:len(nes_b)], nes_b, color=b_color, linewidth=1.5, linestyle=nes_style, alpha=0.9)
    axes[1].plot(theory_mu, theory_a, color=a_color, linewidth=2, linestyle=theory_style, alpha=0.9)
    axes[1].plot(theory_mu, theory_b, color=b_color, linewidth=2, linestyle=theory_style, alpha=0.9)
    axes[1].set_xlabel(r'$\mu_t$')
    axes[1].set_ylabel(r'$a_t, b_t$')
    set_equal_aspect(axes[1], ratio_based=True)
    style_axis(axes[1])
    
    # Plot (c): Theory ellipses on landscape
    ellipse_stats = theory_statistics_for_ellipses or theory_statistics
    ellipse_time, ellipse_mu, ellipse_a, ellipse_b = ellipse_stats
    if snapshot_times is None:
        max_t = len(ellipse_time) - 1
        mid_t = int(max_t / 2)
        if max_t >= 2:
            if mid_t <= 0:
                mid_t = 1
            elif mid_t >= max_t:
                mid_t = max_t - 1
        snapshot_times = [0, mid_t, max_t]
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(snapshot_times) - 1)) if len(snapshot_times) > 1 else cmap(0.5)
              for i in range(len(snapshot_times))]
    
    add_fitness_contours(axes[2], fitness_function, xlim=(-4, 4), ylim=(-4, 4),
                         resolution=200, levels=21, vmin=0, vmax=10)
    for i, t in enumerate(snapshot_times):
        t = min(t, len(ellipse_mu) - 1)
        mu_t = ellipse_mu[t]
        a_t = ellipse_a[t]
        b_t = ellipse_b[t]
        center = (mu_t, 0.0)
        cov_matrix = np.array([[a_t, 0.0], [0.0, b_t]])
        add_covariance_ellipse(
            axes[2], center, cov_matrix, n_sigma=2.0,
            color=colors[i], linewidth=1.5, alpha=0.85, fill=False, zorder=10 + i
        )
    axes[2].axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.5, zorder=5)
    axes[2].axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5, zorder=5)
    axes[2].set_xlabel(r'$x$')
    axes[2].set_ylabel(r'$y$')
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    set_equal_aspect(axes[2])
    style_axis(axes[2], remove_top_right=True)
    
    # Legend intentionally omitted; handled in caption.

    add_subplot_labels(axes)
    
    adjust_layout(fig, left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.4)
    plt.show()
    
    if save_fig:
        beta = params.get('beta', 1.0)
        mutation_std = params.get('mutation_std', 0.1)
        num_iter = params.get('num_iterations', 100)
        pop_size = params.get('population_size', 1000)
        filename = (
            f"dynamics_theory_vs_empirical_iter{num_iter}_pop{pop_size}_"
            f"mut{mutation_std:.3f}_beta{beta:.2f}"
        )
        save_figure(fig, filename, format='jpeg')
    
    return fig, axes


def plot_mu_theory_and_mutation_std(
    ed_statistics: Union[Tuple, List[Tuple]],
    theory_statistics: Union[Tuple, List[Tuple]],
    full_nes_statistics: Optional[Tuple],
    mutation_comparison_results: Dict,
    fitness_function,
    params: Optional[Dict] = None,
    mu_sigmas: Optional[List[float]] = None,
    save_fig: bool = True
):
    """
    Plot μ_t vs time (theory vs empirical) alongside mutation std trajectory and curvature.
    
    Args:
        ed_statistics: Tuple or list of tuples from run_evolution
        theory_statistics: Tuple or list of tuples from run_manifold_dynamics_theoretical
        full_nes_statistics: Optional tuple from run_full_natural_gradient_es
        mutation_comparison_results: Dict from run_mutation_std_comparison
        fitness_function: Fitness function for contour plots
        params: Optional dict for naming metadata
        mu_sigmas: Optional list of sigma values for μ_t series coloring
        save_fig: Whether to save figure
    """
    # Normalize μ_t series inputs
    if isinstance(ed_statistics, list):
        ed_series = ed_statistics
        theory_series = theory_statistics if isinstance(theory_statistics, list) else [theory_statistics] * len(ed_series)
    else:
        ed_series = [ed_statistics]
        theory_series = [theory_statistics]
    
    if mu_sigmas is not None and len(mu_sigmas) != len(ed_series):
        raise ValueError("mu_sigmas length must match number of μ_t series.")
    
    mu_colors = get_colors_for_values(mu_sigmas) if mu_sigmas else None
    
    # Unpack mutation std comparison
    results = mutation_comparison_results['results']
    mutation_params = mutation_comparison_results['params']
    mutation_std_values = mutation_params['mutation_std_values']
    
    colors = get_colors_for_values(mutation_std_values)
    legend_elements = []
    
    # Create figure
    fig, axes = create_figure(n_cols=3)
    ax_trajectory, ax_curvature, ax_mu = axes
    
    # Plot μ_t vs time (theory vs empirical)
    mu_legend = []
    for idx, (ed_stats, th_stats) in enumerate(zip(ed_series, theory_series)):
        ed_time, ed_pos, _, _, _ = ed_stats
        theory_time, theory_mu, _, _ = th_stats
        ed_mu = np.array(ed_pos)[:, 0]
        
        color = mu_colors[idx] if mu_colors is not None else ALGORITHM_COLORS[AlgorithmEnum.EVOLUTION.value]
        sigma_label = f"σ = {mu_sigmas[idx]:.2f}" if mu_sigmas is not None else None
        
        ax_mu.plot(ed_time, ed_mu, color=color, linewidth=1.5, alpha=0.9)
        ax_mu.plot(theory_time, theory_mu, color=color, linewidth=2, linestyle='--', alpha=0.9)
        
        if sigma_label is not None:
            mu_legend.append(Line2D([0], [0], color=color, lw=1.5, linestyle='-',
                                    label=f'ED {sigma_label}'))
            mu_legend.append(Line2D([0], [0], color=color, lw=2, linestyle='--',
                                    label=f'Theory {sigma_label}'))
    ax_mu.set_xlabel('Generation')
    ax_mu.set_ylabel(r'$\mu_t$')
    style_axis(ax_mu)
    set_equal_aspect(ax_mu, ratio_based=True)
    
    # Legend intentionally omitted; handled in caption.
    
    # Trajectory and curvature plots (mutation std comparison)
    add_fitness_contours(
        ax_trajectory,
        fitness_function,
        xlim=(0, 2),
        ylim=(-1, 1),
        levels=41
    )
    
    for i, mutation_std in enumerate(mutation_std_values):
        color = colors[i]
        result = results[mutation_std]
        evo_stats = result['statistics']
        
        time_data, pos_history, _, _, _ = evo_stats
        time_data = np.array(time_data)
        pos_history = np.array(pos_history)
        
        import jax.numpy as jnp
        curvature_data = jnp.linalg.norm(pos_history, axis=1)
        
        plot_trajectory(ax_trajectory, pos_history, color=color)
        scatter_point(ax_trajectory, float(pos_history[-1, 0]), float(pos_history[-1, 1]),
                     color=color, size=20, zorder=3)
        plot_timeseries(ax_curvature, time_data, np.array(curvature_data), color=color)
        
        legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='-',
                                      label=f'σ = {mutation_std:.2f}', markersize=5))
    
    ax_trajectory.set_xlabel('x')
    ax_trajectory.set_ylabel('y')
    ax_trajectory.set_xlim(0, 2)
    ax_trajectory.set_ylim(-1, 1)
    style_axis(ax_trajectory)
    set_equal_aspect(ax_trajectory)
    
    ax_curvature.set_xlabel('Generation')
    ax_curvature.set_ylabel('Curvature')
    style_axis(ax_curvature)
    set_equal_aspect(ax_curvature, ratio_based=True)
    
    add_subplot_labels(axes)
    
    if len(mutation_std_values) > 1:
        ncols = min(len(mutation_std_values), 4)
        add_legend(fig, legend_elements, ncol=ncols, bbox_to_anchor=(0.5, 0.92))
    
    adjust_layout(fig, left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.4)
    plt.show()
    
    if save_fig:
        mu_params = (params or {}).get('mu', {})
        mu_iter = mu_params.get('num_iterations', len(ed_series[0][0]) - 1)
        mu_pop = mu_params.get('population_size', 0)
        mut_iter = mutation_params.get('num_iterations', 0)
        mut_pop = mutation_params.get('population_size', 0)
        if mu_sigmas:
            mu_sigma_label = f"{mu_sigmas[0]:.3f}to{mu_sigmas[-1]:.3f}"
        else:
            mu_sigma_label = "single"
        base_filename = (
            "combined_mu_mutation_std_"
            f"muIter{mu_iter}_muPop{mu_pop}_muSigma{mu_sigma_label}_"
            f"mutIter{mut_iter}_mutPop{mut_pop}"
        )
        save_figure(fig, base_filename, format='jpeg')
    
    return fig, axes


# =============================================================================
# ANALYSIS RESULT PLOTS
# =============================================================================

def plot_ed_gd_sgd_populations(
    comparison_results: Dict,
    fitness_function,
    F_MAX,
    save_fig: bool = True
):
    """
    Plot results from run_ed_gd_sgd_comparison.
    
    Args:
        comparison_results: Dictionary returned by run_ed_gd_sgd_comparison
        fitness_function: Fitness function for contour plots
        save_fig: Whether to save the figure
    """
    populations = comparison_results['populations']
    params = comparison_results['params']
    
    setup_style()
    fig, axes = create_figure(n_cols=3)
    
    ed_color = get_algorithm_color(AlgorithmEnum.EVOLUTION)
    gd_color = get_algorithm_color(AlgorithmEnum.GRADIENT_ASCENT)
    sgd_color = get_algorithm_color(AlgorithmEnum.SHIFT_GRADIENT_ASCENT)
    
    xlim = (-3, 3)
    ylim = (-3, 3)
    
    methods = [
        (AlgorithmEnum.EVOLUTION, populations['ED'], ed_color),
        (AlgorithmEnum.GRADIENT_ASCENT, populations['GLD'], gd_color),
        (AlgorithmEnum.SHIFT_GRADIENT_ASCENT, populations['SGD'], sgd_color)
    ]
    
    for i, (name, pop, color) in enumerate(methods):
        ax = axes[i]
        
        add_fitness_contours(ax, fitness_function, xlim, ylim, vmin=0, vmax=F_MAX)
        scatter_population(ax, np.array(pop), color=color)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x", fontsize=12)
        if i == 0:
            ax.set_ylabel("y", fontsize=12)
        
        style_axis(ax)
        set_equal_aspect(ax)
    
    add_subplot_labels(axes)
    adjust_layout(fig)
    
    plt.show()
    
    if save_fig:
        num_iterations = params['num_iterations']
        population_size = params['population_size']
        mutation_std = params['mutation_std']
        beta = params['beta']
        filename = f"ed_gd_sgd_populations_iter{num_iterations}_pop{population_size}_mut{mutation_std:.3f}_beta{beta:.2f}.jpeg"
        save_figure(fig, filename)
    
    return fig


# =============================================================================
# COVARIANCE-HESSIAN ALIGNMENT ANALYSIS
# =============================================================================

def compute_covariance_hessian_alignment(
    ed_statistics: Tuple,
    hessian_func,
    flat_k: int = None,
    sharp_k: int = None,
    population_snapshots: Optional[Dict[int, np.ndarray]] = None
) -> Dict:
    """
    Compute covariance projections onto Hessian eigenvectors over time.
    
    Projects the population covariance matrix onto the Hessian eigenbasis
    at each time step to analyze variance along flat vs sharp directions.
    
    Args:
        ed_statistics: Tuple of (time, pos_history, fitness, hessian_trace, cov_matrices)
        hessian_func: JAX-compatible Hessian function
        flat_k: Number of flattest directions to group (smallest |eigenvalue|).
                Defaults to half the dimension.
        sharp_k: Number of sharpest directions to group (largest |eigenvalue|).
                 Defaults to half the dimension.
        population_snapshots: Optional dict mapping time -> population array.
                              If provided, uses the average Hessian over the
                              population at each time instead of the Hessian
                              at the population mean.
    
    Returns:
        Dictionary containing:
            - 'time': Time points array
            - 'hessian_eigenvalues': Array of shape (T, D) with sorted eigenvalues per time
            - 'projected_variance': Array of shape (T, D) with variance along each eigenvector
            - 'flat_variance': Mean variance in flat directions over time (T,)
            - 'sharp_variance': Mean variance in sharp directions over time (T,)
            - 'flat_k': Number of flat directions used
            - 'sharp_k': Number of sharp directions used
    """
    import jax.numpy as jnp
    
    time_points, pos_history, _, _, cov_matrices = ed_statistics
    
    T = len(time_points)
    D = pos_history.shape[1]
    
    if flat_k is None:
        flat_k = D // 2
    if sharp_k is None:
        sharp_k = D // 2
    
    hessian_eigenvalues = np.zeros((T, D))
    projected_variance = np.zeros((T, D))
    
    for t in range(T):
        pos = pos_history[t]
        cov = cov_matrices[t]
        
        # Compute Hessian at current time
        if population_snapshots is not None:
            time_key = int(time_points[t])
            pop = population_snapshots.get(time_key)
        else:
            pop = None
        
        if pop is not None:
            # Average Hessian over population
            try:
                import jax
                import jax.numpy as jnp
                pop_jnp = jnp.array(pop)
                H_pop = jax.vmap(hessian_func)(pop_jnp)
                H = np.array(jnp.mean(H_pop, axis=0))
            except ImportError:
                H_sum = np.zeros((D, D))
                for i in range(len(pop)):
                    H_sum += np.array(hessian_func(pop[i]))
                H = H_sum / max(len(pop), 1)
        else:
            # Hessian at mean position (fallback)
            H = np.array(hessian_func(pos))
        
        # Eigen-decomposition of Hessian
        eig_vals, eig_vecs = np.linalg.eigh(H)
        
        # Sort by absolute eigenvalue (ascending: flat first, sharp last)
        sort_idx = np.argsort(np.abs(eig_vals))
        eig_vals_sorted = eig_vals[sort_idx]
        eig_vecs_sorted = eig_vecs[:, sort_idx]
        
        hessian_eigenvalues[t] = eig_vals_sorted
        
        # Project covariance onto Hessian eigenbasis: diag(V^T C V)
        cov_np = np.array(cov)
        cov_projected = eig_vecs_sorted.T @ cov_np @ eig_vecs_sorted
        projected_variance[t] = np.diag(cov_projected)
    
    # Compute mean variance for flat vs sharp groups
    flat_variance = np.mean(projected_variance[:, :flat_k], axis=1)
    sharp_variance = np.mean(projected_variance[:, -sharp_k:], axis=1)
    
    return {
        'time': np.array(time_points),
        'hessian_eigenvalues': hessian_eigenvalues,
        'projected_variance': projected_variance,
        'flat_variance': flat_variance,
        'sharp_variance': sharp_variance,
        'flat_k': flat_k,
        'sharp_k': sharp_k
    }


def plot_alignment_and_anticorrelation_summary(
    alignment_data: Dict,
    save_fig: bool = False,
    filename_prefix: str = "alignment_anticorrelation_summary"
) -> plt.Figure:
    """
    Combine alignment and anticorrelation summaries.
    
    Panels:
    - (a) Projected variance in flat vs sharp subspaces
    - (b) Pearson correlation of log |eigenvalue| vs log variance over time
    - (c) Final-time binned |eigenvalue| vs average variance (log-log)
    """
    time = np.array(alignment_data['time'])
    projected_var = np.array(alignment_data['projected_variance'])
    hess_eigs = np.array(alignment_data['hessian_eigenvalues'])
    flat_var = np.array(alignment_data['flat_variance'])
    sharp_var = np.array(alignment_data['sharp_variance'])
    flat_k = alignment_data['flat_k']
    sharp_k = alignment_data['sharp_k']
    
    T, D = projected_var.shape
    
    fig, axes = create_figure(n_cols=3)
    ax_var, ax_corr, ax_binned = axes
    
    # Panel (a): Variance in flat vs sharp subspaces
    flat_color = COLORS[0]
    sharp_color = COLORS[3]
    plot_timeseries(ax_var, time, flat_var, color=flat_color)
    plot_timeseries(ax_var, time, sharp_var, color=sharp_color)
    ax_var.set_xlabel('Generation')
    ax_var.set_ylabel(r'$\langle v \rangle$')

    style_axis(ax_var)
    set_equal_aspect(ax_var, ratio_based=True)
    
    legend_elements = [
        Line2D([0], [0], color=flat_color, lw=2, label=f'Flat ({flat_k} dirs)'),
        Line2D([0], [0], color=sharp_color, lw=2, label=f'Sharp ({sharp_k} dirs)'),
    ]
    add_legend(fig, legend_elements, ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    # Panel (b): Pearson correlation over time (log-log)
    corrs = np.zeros(T)
    for t in range(T):
        eigs = np.abs(hess_eigs[t])
        vars_ = projected_var[t]
        mask = (eigs > 0) & (vars_ > 0)
        if np.sum(mask) < 2:
            corrs[t] = np.nan
            continue
        x = np.log10(eigs[mask])
        y = np.log10(vars_[mask])
        if np.std(x) == 0 or np.std(y) == 0:
            corrs[t] = np.nan
        else:
            corrs[t] = np.corrcoef(x, y)[0, 1]
    
    ax_corr.plot(time, corrs, color=COLORS[7], linewidth=1.8)
    ax_corr.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax_corr.set_xlabel('Generation')
    ax_corr.set_ylabel(r'$R\!\left(\log|\lambda|, \log v\right)$')
    style_axis(ax_corr)
    set_equal_aspect(ax_corr, ratio_based=True)
    
    # Panel (c): Final-time binned trend (log-log)
    eigs_final = np.abs(hess_eigs[-1])
    vars_final = projected_var[-1]
    mask = (eigs_final > 0) & (vars_final > 0)
    eigs_final = eigs_final[mask]
    vars_final = vars_final[mask]
    
    num_bins = min(10, max(3, D // 3))
    bin_edges = np.quantile(eigs_final, np.linspace(0, 1, num_bins + 1))
    bin_centers = []
    bin_means = []
    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (eigs_final >= lo) & (eigs_final <= hi if b == num_bins - 1 else eigs_final < hi)
        if np.any(in_bin):
            bin_centers.append(np.sqrt(lo * hi))
            bin_means.append(np.mean(vars_final[in_bin]))
    
    if bin_centers:
        ax_binned.plot(bin_centers, bin_means, marker='o', color=COLORS[2], linewidth=1.5)
    ax_binned.set_xscale('log')
    ax_binned.set_yscale('log')
    ax_binned.set_xlabel(r'$|\lambda_i|$ (binned)')
    ax_binned.set_ylabel(r'$\langle v_i \rangle$')
    ax_binned.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax_binned.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True))
    ax_binned.yaxis.set_minor_formatter(ticker.NullFormatter())
    style_axis(ax_binned)
    set_equal_aspect(ax_binned, ratio_based=True)
    
    add_subplot_labels(axes)
    adjust_layout(fig, left=0.1, right=0.95, top=0.86, bottom=0.15, wspace=0.55)
    plt.show()
    
    if save_fig:
        fname = f"{filename_prefix}_D{D}_flat{flat_k}_sharp{sharp_k}.jpeg"
        save_figure(fig, fname)
    
    return fig


# =============================================================================
# INITIALIZATION
# =============================================================================

# Apply style on import
setup_style()

