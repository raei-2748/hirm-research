# HIRM Research Repository

**HIRM (Hedging Invariant Risk Minimization)** is a research framework for learning hedging policies whose risk gradients align across market regimes. The repository collects the end-to-end code used for the accompanying paper: data loaders, objectives, experiment runners, diagnostics, and analysis notebooks.

## Why this repository?
- Provide a clean, publication-ready snapshot of the HIRM experiments.
- Offer reproducible baselines, ablations, and the full experiment suite with consistent CLI tooling.
- Enable easy onboarding for collaborators, students, and reviewers via concise docs and Colab notebooks.

## Quick installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
For GPU-enabled Colab sessions use `pip install -r requirements-colab.txt`.

## Quick tiny demo
Run a single-seed smoke test on synthetic data:
```bash
python scripts/run_smoke_test.py --config configs/experiments/smoke_test.yaml --device cpu --results-dir results/smoke_demo
```
Diagnostics and checkpoints will appear under `results/smoke_demo/`.

## Reproducing the full suite (reduced)
Run the publication grid in reduced mode for quick validation:
```bash
python scripts/run_full_experiment_suite.py \
  --config configs/experiments/full_experiment_suite.yaml \
  --datasets synthetic_heston,real_spy \
  --methods erm_baseline,hirm_full \
  --seeds 0 \
  --device cpu \
  --reduced \
  --results-dir results/full_suite_reduced
```
Follow with analysis scripts (see `RUNS.md`) to regenerate summary tables.

## Colab notebooks
- `notebooks/hirm_tiny_demo.ipynb`: minimal synthetic walkthrough with plots.
- `notebooks/hirm_phase9_reduced.ipynb`: reduced full-suite grid and figure regeneration (will be renamed in history notes).

## Repository layout
- `configs/`: experiment, environment, model, and objective configs.
- `scripts/`: standardized CLI runners for baselines, ablations, full suite, diagnostics, and analysis.
- `analysis/`: offline analysis helpers and plotting utilities.
- `docs/`: developer notes, configuration format, diagnostics overview, and historical phase artifacts.
- `tests/`: fast pytest suite covering configuration loading, smoke runs, and diagnostic metrics.

## Citing
See `CITATION.cff` for citation metadata.
