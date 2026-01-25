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

## Usage

### Run Specific Examples

```bash
python run_examples.py --example 1    # Theory ellipses on landscape
python run_examples.py --example 3    # Theory vs empirical dynamics
python run_examples.py --example 9    # High-dimensional ED
```

### Run All Examples

```bash
python run_examples.py --all
```

### List Available Examples

```bash
python run_examples.py --list
```

### Interactive Mode

```bash
python run_examples.py
```

## Module Structure

| File | Description |
|------|-------------|
| `run_examples.py` | Main script with example experiments |
| `algorithms.py` | Core algorithm implementations (ED, GLD, SGD, NES) |
| `objective_function.py` | Fitness landscape definitions |
| `analysis.py` | Comparison and analysis wrappers |
| `plot_utils.py` | Plotting utilities with consistent styling |

## Output

Generated figures are saved to the `figures/` directory in JPEG format.

## Examples

| # | Description |
|---|-------------|
| 1 | Theory ellipses on fitness landscape |
| 2 | ED vs Full NES comparison |
| 3 | Theory vs empirical dynamics |
| 4 | Mutation rate comparison |
| 5 | ED vs GLD vs SGD final populations |
| 6 | ED vs GLD vs SGD trajectories |
| 7 | Combined mean dynamics and mutation effects |
| 8 | Average dynamics and population distributions |
| 9 | High-dimensional ED with flat/sharp directions |
| 10 | ED with proportional selection |
| 11 | ED (Multiplicative) vs NES on log-fitness |
| 12 | ED (Boltzmann) vs NES on free energy |
| 13 | Steady-state curvature and fitness histograms |
| 14 | High-dimensional steady-state histograms |
