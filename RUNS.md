# Canonical run commands

This cheat sheet lists the most common end-to-end commands for the HIRM project.
All commands assume you are in the repository root with a Python 3.10+ virtual environment activated and dependencies installed via `pip install -r requirements.txt && pip install -e .`.

## Smoke tests

- **Tiny synthetic demo (CPU friendly)**
  ```bash
  python scripts/run_tiny_experiment.py --device cpu --results-dir results/tiny_demo
  ```

- **Pytest smoke for phase 9**
  ```bash
  pytest tests/test_phase9_smoke.py -q
  ```

## Single runs

- **Synthetic HIRM run with diagnostics**
  ```bash
  python scripts/run_experiment_and_diagnostics.py \
    --config configs/experiments/phase9.yaml \
    --device cuda:0 --results-dir outputs/synthetic_demo
  ```

- **Diagnostics on an existing checkpoint**
  ```bash
  python scripts/run_diagnostics.py \
    --config configs/experiments/phase9.yaml \
    --checkpoint outputs/synthetic_demo/checkpoints/model_final.pt \
    --device cpu --results-dir outputs/synthetic_demo/diagnostics
  ```

## Grid experiments

- **Phase 7 benchmark grid**
  ```bash
  python scripts/run_grid.py --config configs/experiments/phase7.yaml --device cuda:0
  ```

- **Phase 8 ablations (reduced)**
  ```bash
  python scripts/run_ablation_grid.py \
    --config configs/experiments/phase8.yaml \
    --reduced --device cpu
  ```

- **Phase 9 paper-aligned grid (reduced for Colab)**
  ```bash
  python scripts/run_experiment_grid.py \
    --config configs/experiments/phase9.yaml \
    --datasets synthetic_heston,real_spy \
    --methods erm_baseline,hirm_full \
    --seeds 0 --reduced --device cuda:0 \
    --results-dir results/phase9_reduced
  ```

- **Full Phase 9 grid (slow, multiple seeds)**
  ```bash
  python scripts/run_experiment_grid.py \
    --config configs/experiments/phase9.yaml \
    --device cuda:0 --results-dir results/phase9
  ```

## Analysis utilities

- **Phase 7 summary**
  ```bash
  python scripts/summarize_phase7_results.py --root results/phase7 --out analysis_outputs/phase7
  ```

- **Phase 8 ablation analysis**
  ```bash
  python analysis/analyze_ablation.py --root_dir results/phase8 --output_dir analysis_outputs/phase8
  ```

- **Phase 9 aggregation and plots**
  ```bash
  python analysis/phase9_analysis.py --root_dir results/phase9_reduced --output_dir analysis_outputs/phase9_reduced
  ```

## Colab notebooks

- `notebooks/hirm_tiny_demo.ipynb` – one-click tiny synthetic run and plot.
- `notebooks/hirm_phase9_reduced.ipynb` – reduced paper grid with diagnostics and plots.
