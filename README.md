# Evolution on Degenerate Fitness Landscapes

Code for the paper:

**"Evolution on degenerate fitness landscapes is not neutral: curvature drives directional drift"**

## Overview

This repository contains implementations of evolutionary dynamics algorithms for studying population evolution on fitness landscapes with degenerate optima. The code demonstrates that curvature effects lead to directional drift rather than neutral evolution.

### Main Algorithms

- **Evolutionary Dynamics (ED)** — population-based evolution with selection and mutation
- **Gradient Langevin Dynamics (GLD)** — gradient ascent with additive noise
- **Stochastic Gradient Descent (SGD)** — gradient ascent with shifted noise
- **Natural Evolution Strategy (NES)** — natural gradient optimization on expected fitness

## Installation

Tested with Python 3.9–3.12. We recommend using a fresh virtual environment.

```bash
# Clone and enter the repo
git clone <repo-url> EDEgeneracy
cd EDEgeneracy

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Every figure runs on CPU; no GPU is needed.

## Reproducing Paper Figures

All figures are produced by `reproduce_figures.py`. Generated images are written to `figures/`.

```bash
python reproduce_figures.py --list           # List available figures
python reproduce_figures.py --fig 2          # A single main-text figure
python reproduce_figures.py --fig S1         # A single supplementary figure
python reproduce_figures.py --main           # All main-text figures (2–4)
python reproduce_figures.py --supp           # All supplementary figures (S1–S6)
python reproduce_figures.py --all            # Everything
python reproduce_figures.py                  # Interactive menu
```

## Module Structure

| File | Description |
|------|-------------|
| `reproduce_figures.py` | Top-level script that drives every figure in the paper |
| `algorithms.py` | Core algorithm implementations (ED, GLD, SGD, NES, theoretical dynamics) |
| `objective_function.py` | Fitness landscape definitions (2D and high-D) |
| `analysis.py` | Comparison and analysis wrappers used by `reproduce_figures.py` |
| `plot_utils.py` | Plotting utilities with consistent styling |
| `verify_dominance.py` | Curvature-drift vs genetic-drift transition analysis (supplementary S1) |

## Figure Reference

Numbering follows the current state of the manuscript and supplementary.

### Main Text

| Figure | Description |
|--------|-------------|
| 2 | Mean dynamics and mutation effects |
| 3 | Theory vs empirical dynamics |
| 4 | ED vs GLD vs SGD steady-state populations |

### Supplementary

| Figure | Description |
|--------|-------------|
| S1 | Finite-population transition: curvature-driven vs noise-dominated dynamics (N sweep) |
| S2 | ED (linear selection) vs Natural Evolution Strategy |
| S3 | ED (multiplicative selection) vs NES on log-fitness |
| S4 | ED (Boltzmann selection) vs NES on free energy |
| S5 | High-dimensional ED — mutation strength comparison |
| S6 | High-dimensional ED — variance/curvature alignment |

S5 and S6 share the same generator (`generate_high_dim_ed_figure`) but a `figures` argument routes each FIGURES entry to only the simulations it needs, so calling `--fig S5` or `--fig S6` independently is cheap.
