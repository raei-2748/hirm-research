
# HIRM: Hedging with Invariant Risk Minimization

*Note: Future versions may adopt the project name **Praesidium**, based on Causal Distilled Structural Invariance.*

<p align="left">

  <img src="https://img.shields.io/badge/version-v1.0.0-blue.svg">
  <img src="https://img.shields.io/badge/license-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red">
  <img src="https://img.shields.io/badge/Reproducible-Yes-brightgreen">
  <img src="https://img.shields.io/badge/Colab-demo-yellow">
  <img src="https://img.shields.io/badge/data-SPY%20%2B%20Synthetic-orange">

</p>

This repository contains the reference implementation of **HIRM (Hedging with Invariant Risk Minimization)**, a decision-level robustness method for dynamic hedging under market regime shifts. It supports synthetic environments and real SPY experiments, includes strong baselines, and provides diagnostics and reproducibility pipelines.

Most hedge models fail when volatility, liquidity, or correlations shift abruptly. HIRM improves robustness by stabilizing the **hedge decision rule** itself across environments, rather than enforcing invariance on representations or losses. The architecture separates:

- a **representation module** that adapts to market structure  
- a **hedge head** whose gradients are regularized across regimes  

This design aims to produce more stable and interpretable hedges in stress periods while preserving flexibility in normal conditions.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate         # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
````

For Colab, see the notebooks in `notebooks/`.

---

## Quickstart

### 1. Smoke test

Minimal end-to-end check:

```bash
python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml \
  --device cpu
```

### 2. Mini synthetic demo

Single-seed synthetic Heston run comparing HIRM vs ERM:

```bash
python scripts/run_full_experiment_suite.py \
  --datasets synthetic_heston \
  --methods hirm_full,erm_baseline \
  --seeds 0
```

### 3. Mini SPY demo

Requires:

```text
data/processed/spy_prices.csv
data/processed/spy_regimes.txt
```

Then run:

```bash
python scripts/run_full_experiment_suite.py \
  --datasets real_spy \
  --methods hirm_full,erm_baseline \
  --seeds 0
```

Additional commands are listed in **RUNS.md**.

---

## Experiments

Experiment families are organized by purpose:

* **baseline_benchmark**
  ERM, GroupDRO, V-REx, IRM, HIRM

* **ablation_study**
  Effects of removing or modifying HIRM components

* **full_experiment_suite**
  Synthetic + SPY experiments (publication-style grid)

* **smoke_test**
  Minimal system validation

Each experiment has:

* a config file under `configs/experiments/`
* a corresponding runner in `scripts/`

---

## Diagnostics

HIRM provides diagnostics along three axes: **Invariance**, **Robustness**, and **Efficiency**.

**Invariance**

* **ISI (Invariant Signal Index)**
  Composite measure of internal invariance (loss, gradient, representation stability)

* **IG (Invariance Gap)**
  Dispersion of risk across test environments

**Robustness**

* **Worst-group CVaR**
  Tail risk in the worst regime

* **Crisis CVaR**
  Tail risk restricted to crisis windows

* **Volatility ratios**
  Temporal stability of risk over time

**Efficiency**

* **Return-risk efficiency**
  Expected return per unit tail risk

* **Turnover**
  Trading intensity / implementation cost proxy

Run diagnostics on any checkpoint:

```bash
python scripts/run_diagnostics.py \
  --checkpoint <path> \
  --results-dir results/diagnostics
```

---

## Data

### Synthetic

* Generated at runtime (no external files required)
* Heston-style dynamics with optional jump and liquidity stress
* Volatility-band regimes: Low / Medium / High / Crisis

### Real SPY

Requires:

```text
data/processed/spy_prices.csv
data/processed/spy_regimes.txt
```

* SPY daily closes and realized volatility
* Regimes defined by 20-day realized volatility bands
* Training: low/medium-volatility periods (2017–2019)
* Testing: Feb–Mar 2018, Mar 2020, 2022 sell-off

See `docs/data_preparation.md` for detailed instructions.

---

## Repository Structure

```text
hirm/             # Core library: models, environments, objectives, training, diagnostics
configs/          # Experiment, model, objective, and dataset configs
scripts/          # Runners for training grids, diagnostics, and utilities
analysis/         # Aggregation and plotting utilities
notebooks/        # Colab-friendly demos
docs/             # Documentation and design notes
results/          # Created at runtime (gitignored)
```

---

## Reproducibility

Each run stores:

* random seed
* full config snapshot
* command-line arguments
* timestamps and device information
* git commit hash

Reproduction scripts live under `scripts/`. A typical pattern is:

```bash
python scripts/run_full_experiment_suite.py \
  --config <config_path> \
  --results-dir <results_dir> \
  [other flags...]
```

---

## Canonical Run Commands

All commands assume you are in the repo root with dependencies installed.

### Smoke tests

```bash
python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml \
  --device cpu \
  --results-dir results/smoke_demo

pytest tests/test_full_suite_smoke.py -q
```

### Baseline benchmark (reduced)

```bash
python scripts/run_baseline_benchmark.py \
  --config configs/experiments/baseline_benchmark.yaml \
  --datasets synthetic_heston \
  --methods erm,hirm \
  --seeds 0 \
  --device cpu \
  --results-dir results/baseline_reduced
```

### Mini ablation run

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

### Full experiment suite (reduced)

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

### Full publication grid (slow)

```bash
python scripts/run_full_experiment_suite.py \
  --config configs/experiments/full_experiment_suite.yaml \
  --device cuda:0 \
  --results-dir results/full_experiment_suite
```

### Analysis utilities

```bash
python scripts/summarize_baseline_results.py \
  --root results/baseline_reduced \
  --out analysis_outputs/baseline

python analysis/analyze_ablation.py \
  --root_dir results/ablation_reduced \
  --output_dir analysis_outputs/ablation

python analysis/phase9_analysis.py \
  --root_dir results/full_suite_reduced \
  --output_dir analysis_outputs/full_suite_reduced
```

### Colab notebooks

* `notebooks/hirm_tiny_demo.ipynb`
  Tiny synthetic run with basic plots

* `notebooks/hirm_phase9_reduced.ipynb`
  Reduced full-suite diagnostics and plots

---

## Citation

If you use this code, please cite:

```bibtex
@article{hirm2025,
  title   = {Hedging with Invariant Risk Minimization},
  author  = {Authors},
  journal = {Working Paper},
  year    = {2025}
}
```

---

## License

Released under the MIT License. See `LICENSE` for details.

---

## Contributions

Contributions are welcome.

For substantial changes (new environments, objectives, or large refactors), please open an issue first so scope and design can be discussed before implementation.

```
```
