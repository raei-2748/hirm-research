# Configuration format

Experiments load YAML files via `hirm.utils.config.load_config`. Key sections:
- `defaults`: base includes and environment/model presets under `configs/envs/` and `configs/models/`.
- `env`: environment-specific hyperparameters such as `feature_dim` and `action_dim`.
- `training`: training hyperparameters (`max_epochs`, `batch_size`, `lr`, `early_stop_patience`, `deterministic`).
- `objective`: objective name and invariance settings when relevant.
- `diagnostics`: optional settings for invariance (ISI/IG), robustness (WG/VR), efficiency (ER/TR), and crisis CVaR.
- `experiment_grid`: optional convenience block listing methods, datasets, and seeds for grid runners.

Paths such as dataset locations can be overridden at the CLI level via `--config` plus `--results-dir` and dataset arguments. Scripts always dump the resolved config to each run directory for reproducibility.
