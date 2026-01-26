# Evolution on Degenerate Fitness Landscapes

Code for the paper:

**"Evolution on degenerate fitness landscapes is not neutral: curvature drives directional drift"**

## Overview

This repository contains implementations of evolutionary dynamics algorithms for studying population evolution on fitness landscapes with degenerate optima. The code demonstrates that curvature effects lead to directional drift rather than neutral evolution.

### Main Algorithms

- **Evolutionary Dynamics (ED)**: Population-based evolution with selection and mutation
- **Gradient Langevin Dynamics (GLD)**: Gradient ascent with additive noise
- **Stochastic Gradient Descent (SGD)**: Gradient ascent with shifted noise
- **Natural Evolution Strategy (NES)**: Natural gradient optimization on expected fitness

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- JAX (with jaxlib)
- NumPy
- Matplotlib
- Seaborn

## Reproducing Paper Figures

### Reproduce Specific Figures

```bash
python reproduce_figures.py --fig 2      # Figure 2 (main text)
python reproduce_figures.py --fig 3      # Figure 3 (main text)
python reproduce_figures.py --fig S1     # Supplementary Figure 1
```

### Reproduce All Main Text Figures

```bash
python reproduce_figures.py --main
```

### Reproduce All Supplementary Figures

```bash
python reproduce_figures.py --supp
```

### Reproduce All Figures

```bash
python reproduce_figures.py --all
```

### List Available Figures

```bash
python reproduce_figures.py --list
```

### Interactive Mode

```bash
python reproduce_figures.py
```

## Module Structure

| File | Description |
|------|-------------|
| `reproduce_figures.py` | Main script to reproduce paper figures |
| `algorithms.py` | Core algorithm implementations (ED, GLD, SGD, NES) |
| `objective_function.py` | Fitness landscape definitions |
| `analysis.py` | Comparison and analysis wrappers |
| `plot_utils.py` | Plotting utilities with consistent styling |

## Output

Generated figures are saved to the `figures/` directory.

## Figure Reference

### Main Text

| Figure | Description |
|--------|-------------|
| 2 | Mean dynamics and mutation effects |
| 3 | Theory vs empirical dynamics |
| 4 | ED vs GLD vs SGD populations |

### Supplementary

| Figure | Description |
|--------|-------------|
| S1 | ED vs Full NES comparison |
| S2 | ED (Multiplicative) vs NES on log-fitness |
| S3 | ED (Boltzmann) vs NES on free energy |
| S4 | Steady-state curvature and fitness histograms |
| S5 | High-dimensional ED with flat/sharp directions |
| S6 | High-dimensional steady-state histograms |
