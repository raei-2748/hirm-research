# Developer Notes

- **New datasets**: implement a loader in `hirm/experiments/datasets.py` using `register_dataset`, then add config defaults under `configs/envs/`. Update experiment configs to reference the dataset name.
- **New objectives**: register a class in `hirm/objectives` with `@register_objective` and expose it via `hirm/experiments/methods.py` so the grid runner can pick it up.
- **Result layout**: phase 9 experiments write to `results/phase9/{dataset}/{method}/seed_{seed}`; keep this structure for compatibility with the analysis scripts.
- **Known limitations**: smoke tests use CPU-only runs with reduced epochs; full paper sweeps should target GPU for throughput.
