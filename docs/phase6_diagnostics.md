# Phase 6 – Diagnostics

This repo now ships a full I–R–E (Invariance–Robustness–Efficiency) diagnostic
suite aligned with the HIRM paper.  The diagnostics live under
`hirm.diagnostics` and expose pure functional APIs so that experiments, scripts
and notebooks can import the metrics directly.

## Metrics

| Axis | Metric | Definition |
| ---- | ------ | ---------- |
| Invariance | ISI | Internal Stability Index composed of C1 (risk variance), C2 (head gradient alignment) and C3 (structural dispersion). See `hirm/diagnostics/invariance.py`.
| Invariance | IG | Outcome-level invariance gap computed over held-out env risks.
| Robustness | WG | Worst-case generalization computed via a CVaR-style surrogate.
| Robustness | VR | Volatility ratio of rolling risk statistics.
| Efficiency | ER | Expected return divided by CVaR tail risk.
| Efficiency | TR | Turnover ratio measuring action smoothness.

Each metric is implemented as a pure function documented with references to the
corresponding equation or section in the paper.

## Running diagnostics

1. Train your models using the existing Phase 5 pipeline.
2. Run the diagnostics script on a resolved config and checkpoint:

```bash
python scripts/run_diagnostics.py \
  --config configs/experiments/synth_hirm.yaml \
  --checkpoint outputs/synth_hirm/best_model.pt \
  --results-dir outputs/phase6
```

The script loads the config, replays the saved model on held-out probe batches
and writes a JSONL file with all metrics to the requested directory.

Each JSONL line looks like:

```json
{
  "experiment_id": "synth_hirm",
  "model_name": "invariant_policy",
  "seed": 7,
  "checkpoint": "outputs/synth_hirm/best_model.pt",
  "metrics": {
    "ISI": 0.91,
    "ISI_C1": 0.88,
    "IG": 0.04,
    "WG": 0.23,
    "VR": 0.15,
    "ER": 1.12,
    "TR": 0.42
  }
}
```

## Summaries and reporting

Use the reporting helper to aggregate and visualize diagnostics:

```bash
python scripts/summarize_diagnostics.py \
  --results-dir outputs/phase6 \
  --out outputs/phase6/summary.json \
  --plot-dir outputs/phase6/plots
```

The summary JSON contains per-method means/stdevs and the correlations between
ISI, IG and crisis CVaR mentioned in the paper.  The optional plot directory
stores quick-look scatter plots.

## Configuration knobs

`configs/base.yaml` defines defaults for all diagnostic hyperparameters.  Each
experiment inherits the defaults and can override the `diagnostics` block to
select probe layers, tau thresholds, trimming, etc.
