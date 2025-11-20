# Developer Notes

This document consolidates the historical `DEV_NOTES` and `DEVELOPER_NOTES` files into a single reference for maintainers.

## Contribution Pointers
- **New datasets**: implement a loader in `hirm/experiments/datasets.py` using `register_dataset`, then add configuration defaults under `configs/envs/`. Update experiment configs to reference the dataset name.
- **New objectives**: register a class in `hirm/objectives` with `@register_objective` and expose it via `hirm/experiments/methods.py` so the grid runner can pick it up.
- **Result layout**: grid experiments write to `results/<suite>/<dataset>/<method>/seed_<seed>`; keep this structure for compatibility with the analysis scripts.
- **Known limitations**: smoke tests use CPU-only runs with reduced epochs; full paper sweeps should target GPU for throughput.

## Recent Updates
- Added a `--results-dir` flag to the grid runner to allow Colab-friendly output roots.
- Improved `scripts/run_tiny_experiment.py` with logging, seeding controls, and run metadata.
- Added Colab notebooks: `notebooks/hirm_tiny_demo.ipynb` for a quick synthetic run and `notebooks/hirm_phase9_reduced.ipynb` (soon to be renamed) for a reduced grid plus plots.
- Documented canonical commands in `RUNS.md` and linked them from the README.

## Remaining Limitations / TODOs
- Full experiment suites remain computationally heavy; expect long runtimes on CPU-only environments.
- SPY data must still be provided at `data/raw/spy.csv` for real-data experiments.
- Logging in legacy scripts is minimal; consider a shared logging helper if deeper verbosity is needed.

## Environment Assumptions
- Python 3.10 with the pinned requirements is available; PyTorch CUDA wheels are installed automatically on Colab via `requirements-colab.txt`.
- Scripts should be run from the repository root unless a specific working directory is provided.
