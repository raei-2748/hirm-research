# Cleanup Summary

## Repository restructuring
- Moved legacy status and phase documents into `docs/history/` and consolidated developer notes into `docs/developer_notes.md`.
- Added documentation scaffolding (`docs/config_format.md`, `docs/design_history.md`, `docs/diagnostics_overview.md`) to clarify configuration, history, and diagnostics.
- Archived historical experiment configs under `docs/history/phase_configs/` and introduced semantically named configs in `configs/experiments/`.

## Naming updates
- Renamed experiment suites: Phase 7 → `baseline_benchmark`, Phase 8 → `ablation_study`, Phase 9 → `full_experiment_suite`.
- Updated primary scripts to `run_baseline_benchmark.py`, `run_ablation_study.py`, `run_full_experiment_suite.py`, and `run_smoke_test.py` with consistent CLI flags.
- Updated tests and run guides to reference the new names and paths.

## Remaining limitations
- Colab notebooks still carry legacy filenames; they continue to run but should be renamed in a follow-up pass.
- Some historical documentation under `docs/history/` preserves original phase wording for archival purposes.
