# Older Langevin / Ito / TSS work — archive snapshot

Snapshot of uncommitted work-in-progress from the **main worktree**
(`/Users/razifachareldeen/EDEgeneracy`) at the time the `note/langevin-validation`
branch was being prepared. Surfaced here so it can be compared against the
new note + Figure 2 work on this branch without disturbing the main worktree's
in-progress state.

## What's here

### `figures/` — 13 jpegs

- `langevin_tss_validity_sigma0.050_eta0.100_box3.0_pop1000_iter10000.jpeg`
- `langevin_tss_confining_sigma0.050_eta0.100_box3.0_pop1000_iter10000.jpeg`
- `langevin_confining_tss_sigma0.050_eta0.010_box3.0_pop1500_iter100000.jpeg`
- `langevin_confining_tss_sigma0.050_eta0.010_box3.0_pop1500_iter500000.jpeg`
- `langevin_confining_finite_eta_sigma0.050_eta0.010_pop1500_iter500000.jpeg`
- `langevin_persistent_excursions_sigma0.050_eta0.100_iter10000.jpeg`
- `langevin_vs_gradflow_x02.00_sigma0.100_eta0.050_pop2000_iter5000.jpeg`
- `curvature_drift_data_collapse.png.jpeg`
- `curvature_drift_mu0_sweep.png.jpeg`
- `curvature_drift_transition.png.jpeg`
- `ed_gd_sgd_populations_iter100000_pop1000_mut0.050_beta0.10_gld-langevin.jpeg`
- `ed_gd_sgd_populations_iter10000_pop1000_mut0.050_beta0.10.jpeg`
- `ed_gd_sgd_populations_iter10000_pop1000_mut0.050_beta0.10_gld-langevin.jpeg`

The `langevin_tss_*` files are the most relevant for the §4.1 / TSS
discussion — they're what was meant by "the earlier plot we had".

### `src/` — modified versions of the project source files

These are the files as they sat in the main worktree (with uncommitted edits)
when the figures above were generated. They contain the analysis/plot functions
that produced the figures.

- `analysis.py` — adds `compare_langevin_to_gradient_flow`,
  `diagnose_langevin_tss_validity`, `_run_population_trajectories`,
  `_trajectory_summary_stats`, `compare_mutation_std`, `compare_ed_gd_sgd`.
- `plot_utils.py` — adds `plot_langevin_vs_gradient_flow`,
  `plot_langevin_tss_validity`.
- `reproduce_figures.py` — renames `run_*` → `generate_*`.
- `algorithms.py` — renames `run_*` → `simulate_*`.
- `objective_function.py` — additions (note: differs from this branch's
  version, which adds `create_smooth_landscape_2d` for the note's Figure 1).

## Why archived, not merged

The modified source files conflict topically with the note's `validate_langevin_note.py`
(separate validation entrypoint), and the rename of `run_*` → `simulate_*` /
`generate_*` is a sweeping refactor that should be reviewed deliberately rather
than merged silently. Putting everything here lets the user diff at their
leisure.
