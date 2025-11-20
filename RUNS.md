# Canonical run commands

All commands assume you are in the repository root with Python 3.10+ in an active virtual environment and dependencies installed via `pip install -r requirements.txt && pip install -e .`.

## Smoke tests
- Tiny synthetic smoke (CPU friendly)
  ```bash
  python scripts/run_smoke_test.py --config configs/experiments/smoke_test.yaml --device cpu --results-dir results/smoke_demo
  ```
- Pytest smoke
  ```bash
  pytest tests/test_full_suite_smoke.py -q
  ```

## Baseline benchmark
- Reduced baseline grid
  ```bash
  python scripts/run_baseline_benchmark.py \
    --config configs/experiments/baseline_benchmark.yaml \
    --datasets synthetic_heston \
    --methods erm,hirm \
    --seeds 0 \
    --device cpu \
    --results-dir results/baseline_reduced
  ```

## Ablation study
- Reduced ablation grid
  ```bash
  python scripts/run_ablation_study.py \
    --config configs/experiments/ablation_study.yaml \
    --datasets synthetic_heston \
    --ablation_names hirm_full,erm_baseline \
    --seeds 0 \
    --device cpu \
    --reduced \
    --results-dir results/ablation_reduced
  ```

## Full experiment suite
- Reduced publication grid
  ```bash
  python scripts/run_full_experiment_suite.py \
    --config configs/experiments/full_experiment_suite.yaml \
    --datasets synthetic_heston,real_spy \
    --methods erm_baseline,hirm_full \
    --seeds 0 \
    --device cpu \
    --reduced \
    --results-dir results/full_suite_reduced
  ```
- Full grid (slow)
  ```bash
  python scripts/run_full_experiment_suite.py \
    --config configs/experiments/full_experiment_suite.yaml \
    --device cuda:0 \
    --results-dir results/full_experiment_suite
  ```

## Analysis utilities
- Baseline summary
  ```bash
  python scripts/summarize_baseline_results.py --root results/baseline_reduced --out analysis_outputs/baseline
  ```
- Ablation analysis
  ```bash
  python analysis/analyze_ablation.py --root_dir results/ablation_reduced --output_dir analysis_outputs/ablation
  ```
- Full suite aggregation and plots
  ```bash
  python analysis/phase9_analysis.py --root_dir results/full_suite_reduced --output_dir analysis_outputs/full_suite_reduced
  ```

## Colab notebooks
- `notebooks/hirm_tiny_demo.ipynb` – one-click tiny synthetic run and plot.
- `notebooks/hirm_phase9_reduced.ipynb` – reduced full-suite grid with diagnostics and plots (rename pending in history notes).
