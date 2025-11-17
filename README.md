# Head Invariant Risk Model

This research introduce a developed hedging framework named "HIRM" (Head Invariant Risk Model) that synthesise gradient alignment in producing hedge actions. This repository reproduces all results for “Robust Generalization for Hedging under Crisis Regime Shifts.” The release ships a deterministic pipeline that stages data, trains the invariant hedging models, generates diagnostics, and exports the camera-ready paper assets.

## Ablation axes

Each registered ablation toggles one or more of the following axes:

* **Invariance mode:** head-only, full-IRM, env-specific heads, or disabled.
* **State factorization:** split `phi`/`r` encoders, `phi`-only, `r`-only, or no split.
* **Objective:** HIRM variants vs. ERM/group baselines.
* **Environment labels:** original volatility bands, random labels, or coarse bands.

Supported names (see `hirm/experiments/ablations.py` for exact settings):

* `hirm_full`, `hirm_no_hgca`, `hirm_full_irm`, `hirm_env_specific_heads`, `hirm_no_split`
* `hirm_phi_only`, `hirm_r_only`, `hirm_random_env_labels`, `hirm_coarse_env_bands`
* Baselines: `erm_baseline`, `groupdro_baseline`, `vrex_baseline`

## Running the grid

Use the dedicated Phase 8 config (enables diagnostics including crisis CVaR):

```
python scripts/run_ablation_grid.py \
  --config configs/experiments/phase8.yaml \
  --datasets synthetic_heston,real_spy \
  --ablation_names hirm_full,hirm_no_hgca \
  --seeds 0,1,2 \
  --device cuda
```

For quick debugging runs:

```
python scripts/run_ablation_grid.py \
  --config configs/experiments/phase8.yaml \
  --datasets synthetic_heston \
  --ablation_names hirm_full,erm_baseline \
  --seeds 0 \
  --device cpu \
  --reduced
```

Outputs are stored under `results/phase8/{dataset}/{ablation}/seed_{seed}/` and
include the applied config, training logs, diagnostics, metadata, and checkpoints.

## Analyzing results

Aggregate metrics and compute deltas vs. `hirm_full` with:

```
python analysis/analyze_ablation.py --root_dir results/phase8
```

This writes a CSV summary (defaults to `results/phase8/ablation_summary.csv`) and
prints per-dataset pivot tables. Crisis CVaR (`metrics/cvar95/crisis`) will appear
for datasets with defined crisis windows (e.g., `real_spy`).

Key metrics for the paper: mean PnL, crisis CVaR, WG/VR robustness, ISI/IG
invariance, and ER/TR efficiency.

## Notes on ISI

ISI now combines:

* **C1:** outcome-level risk stability.
* **C2:** gradient alignment across environments.
* **C3:** activation dispersion across probe layers.

These components are recorded during ablation runs and surfaced in the analysis
tables.


