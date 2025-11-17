# Phase 8 Ablations

This phase introduces a registry-driven ablation suite for HIRM and baselines.

## Running the grid

Use `scripts/run_ablation_grid.py` to enumerate datasets, ablations, and seeds.

```bash
python scripts/run_ablation_grid.py \
  --config configs/base.yaml \
  --datasets synthetic_heston,real_spy \
  --ablation_names hirm_full,hirm_no_hgca \
  --seeds 0,1,2 \
  --device cpu
```

Outputs are stored under `results/phase8/{dataset}/{ablation}/seed_{seed}/` and include
checkpoints, configs, training logs, diagnostics, and metadata. Use `--reduced` to
run a smoke-test grid with two seeds.

## Analyzing results

Aggregate metrics with:

```bash
python analysis/analyze_ablation.py --root results/phase8 --output results/phase8/ablation_summary.csv
```

The analysis script computes mean/std/quantiles across seeds and deltas relative to
`hirm_full`. Per-dataset tables are printed to stdout.

## Configuration

Ablations are defined in `hirm/experiments/ablations.py` and can be referenced via
`--ablation_names` or by setting `ablation` fields in configs. Each ablation wires
invariance mode, state factorization, objective type, and environment label scheme
into the experiment configuration to ensure reproducibility.
