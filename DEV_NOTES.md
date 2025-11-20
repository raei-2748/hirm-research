# DEV_NOTES

## Summary of changes
- Added a results-dir flag to `scripts/run_experiment_grid.py` so Phase 9 grids can target custom output roots (useful for Colab runs).
- Improved `scripts/run_tiny_experiment.py` with logging, seeding controls, and run metadata written to a dedicated results directory.
- Created Colab-friendly notebooks: `notebooks/hirm_tiny_demo.ipynb` for a quick synthetic run and `notebooks/hirm_phase9_reduced.ipynb` for a reduced Phase 9 grid plus plots.
- Documented canonical commands in `RUNS.md` and linked them from the README.

## Remaining limitations / TODOs
- Full Phase 9 grids remain computationally heavy; expect long runtimes on CPU-only environments.
- SPY data must still be provided at `data/raw/spy.csv` for real-data experiments.
- The logging in legacy scripts is minimal; consider a shared logging helper if deeper verbosity is needed.

## Assumptions
- Python 3.10 with the pinned requirements is available; PyTorch CUDA wheels are installed automatically on Colab via `requirements-colab.txt`.
- Users run scripts from the repository root so relative paths in configs remain valid.
