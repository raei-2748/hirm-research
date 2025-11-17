# Phase 6 – Diagnostics

This repo now ships a full I–R–E (Invariance–Robustness–Efficiency) diagnostic
suite aligned with the HIRM paper.  The diagnostics live under
`hirm.diagnostics` and expose pure functional APIs so that experiments, scripts
and notebooks can import the metrics directly.

## Metrics

| Axis | Metric | Definition |
| ---- | ------ | ---------- |
| Invariance | ISI | Internal Stability Index composed of C1 (risk variance), C2 (head gradient alignment) and C3 (structural dispersion). See `hirm/diagnostics/invariance.py`.|
| Invariance | IG | Outcome-level invariance gap computed over held-out env risks.|
| Robustness | WG | Worst-case generalization computed via a CVaR-style surrogate.|
| Robustness | VR | Volatility ratio of rolling risk statistics.|
| Crisis | crisis_cvar | CVaR of losses evaluated on a dedicated crisis or stress split.|
| Efficiency | ER | Expected return divided by CVaR tail risk. Supports `mode="returns"` and `mode="loss"`.|
| Efficiency | TR | Turnover ratio measuring action smoothness.|

Each metric is implemented as a pure function documented with references to the corresponding equation or section in the paper.

## Running diagnostics

1. Train your models using the existing Phase 5 pipeline or the orchestration stub in this repo.
2. Run the diagnostics script on a resolved config and checkpoint:

```bash
python scripts/run_diagnostics.py \
  --config configs/experiments/synth_hirm.yaml \
  --checkpoint outputs/synth_hirm/best_model.pt \
  --results-dir outputs/phase6
```

The script respects the `diagnostics.enabled` flag from `configs/base.yaml`. Use
`--force` to override the flag when sweeping configs. For quick end-to-end
iterations you can also launch `scripts/run_experiment_and_diagnostics.py`,
which trains a small synthetic model, writes a checkpoint, and immediately runs
the diagnostics in one pass.

Both entry points replay the saved model on held-out probe batches and write a
JSONL file with all metrics to the requested directory. Set
`diagnostics.enabled=false` in a config to skip these diagnostics during sweeps;
pass `--force` to the scripts to override the flag when needed.

Each JSONL line now includes richer metadata for downstream analytics:

```json
{
  "experiment_id": "synth_hirm",
  "model_name": "invariant_policy",
  "method": "HIRM",
  "seed": 7,
  "checkpoint": "outputs/synth_hirm/best_model.pt",
  "env_config": "synthetic_volatility_bands",
  "dataset_name": "synthetic_sandbox",
  "splits": {
    "train": "synthetic_train",
    "test": "synthetic_test",
    "crisis": "synthetic_crisis"
  },
  "metrics": {
    "ISI": 0.91,
    "ISI_C1": 0.88,
    "ISI_C1_trimmed": 0.88,
    "ISI_C2": 0.84,
    "ISI_C2_trimmed": 0.92,
    "ISI_C3": 0.90,
    "ISI_C3_trimmed": 0.94,
    "IG": 0.04,
    "WG": 0.23,
    "VR": 0.15,
    "ER": 1.12,
    "TR": 0.42,
    "crisis_cvar": 0.37
  }
}
```

### ISI trimming

`compute_isi` retains the full list of pairwise gradient alignments (C2) and
layer dispersion statistics (C3).  The trimming fraction is applied symmetrically
to those lists before aggregating, and the normalized alpha vector is applied to
the trimmed component scores when forming the final ISI.  Both raw and trimmed
versions are returned for C1--C3 so the effect of trimming is fully observable
even though trimming is a no-op for C1.

### ER conventions

`compute_er` exposes a `mode` flag.  `mode="loss"` negates returns, computes the
CVaR of the resulting losses, and divides the mean return by the downside risk.
`mode="returns"` keeps the PnL series in return space while still measuring the
lower tail via CVaR for continuity with historical runs.  The configuration flows
from `diagnostics.er.mode` through `scripts/run_diagnostics.py` and into the metric
computation.

### Crisis CVaR

`configs/base.yaml` adds a `data.splits.crisis` section and matching diagnostics
controls.  When enabled, `scripts/run_diagnostics.py` evaluates the policy on
the crisis split, converts the resulting PnL series into losses, and records the
`crisis_cvar` metric using the same CVaR utility as the efficiency axis.  These
metrics can be correlated with ISI and IG via the reporting helpers.

## Summaries and reporting

Use the reporting helper to aggregate and visualize diagnostics.  You can pass
`--group-by` multiple times (or provide several columns) to control aggregation keys
and optionally request plots:

```bash
python scripts/summarize_diagnostics.py \
  --results-dir outputs/phase6 \
  --out outputs/phase6/summary.json \
  --plot-dir outputs/phase6/plots \
  --group-by method
```

The summary JSON contains per-method (or per-group) means/stdevs and the
correlations between ISI, IG and the optional crisis CVaR metrics.  When crisis
metrics are available, `plot_isi_vs_crisis_cvar` produces a quick-look scatter
plot alongside the standard ISI-vs-IG visualization.

## Configuration knobs

`configs/base.yaml` defines defaults for all diagnostic hyperparameters.  Each
experiment inherits the defaults and can override the `diagnostics` block to
select probe layers, tau thresholds, trimming, etc.
