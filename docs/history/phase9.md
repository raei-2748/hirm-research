# Phase 9 (full experiment suite)

Implemented a paper-ready pipeline with a reproducible grid runner, fresh smoke and unit tests, and an analysis script that summarizes results into tables and figures. The suite is now exposed via `scripts/run_full_experiment_suite.py` with `configs/experiments/full_experiment_suite.yaml`; outputs default to `results/full_experiment_suite/`. After runs, execute `python analysis/phase9_analysis.py --root_dir results/full_experiment_suite` to rebuild tables. Remaining TODOs: expand figure styling and add more exhaustive hyper-parameter sweeps for publication.
