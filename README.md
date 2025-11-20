## HIRM: Hedging with Invariant Risk Minimization
Note: Maybe renamed to Praesidium, Causal Distilled Structural Invariance for future developments.

<p align="center">

  <img src="https://img.shields.io/badge/version-v1.0.0-blue.svg">
  <img src="https://img.shields.io/badge/license-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red">
  <img src="https://img.shields.io/badge/Reproducible-Yes-brightgreen">
  <img src="https://img.shields.io/badge/Colab-demo-yellow">
  <img src="https://img.shields.io/badge/data-SPY%20%2B%20Synthetic-orange">


This repository provides the reference implementation of **HIRM (Hedging with Invariant Risk Minimization)**, a decision-level robustness method for dynamic hedging under market regime shifts. It includes synthetic and real SPY experiments, baselines, diagnostics, and reproducible training pipelines.

---

## 1. What is HIRM?

Most hedge models break when volatility, liquidity, or correlations shift. HIRM improves robustness by stabilizing the **hedge decision rule** across environments rather than trying to learn fully invariant features.

The model separates:
- a **representation** that adapts to market conditions  
- a **hedge head** whose gradients are regularized across regimes  

This leads to more stable actions in stress periods while preserving flexibility in normal markets.

---

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
````

For Colab, see `notebooks/`.

---

## 3. Quickstart

### Tiny smoke test

```bash
python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml \
  --device cpu
```

### Small synthetic demo (HIRM vs ERM)

```bash
python scripts/run_full_experiment_suite.py \
  --datasets synthetic_heston \
  --methods hirm_full,erm_baseline \
  --seeds 0
```

### Reduced SPY demo

Requires `data/processed/spy_prices.csv` and `spy_regimes.txt`.

```bash
python scripts/run_full_experiment_suite.py \
  --datasets real_spy \
  --methods hirm_full,erm_baseline \
  --seeds 0
```

Full commands and grids are documented in `RUNS.md`.

---

## 4. Experiments

Experiment families are organized semantically:

* **baseline_benchmark** – ERM, GroupDRO, V-REx, IRM, HIRM comparisons
* **ablation_study** – removing or altering components of HIRM
* **full_experiment_suite** – synthetic and SPY experiments used in the paper
* **smoke_test** – minimal sanity check

Each has a config in `configs/experiments/` and a runner in `scripts/`.

---

## 5. Diagnostics

HIRM includes quantitative diagnostics for:

* **Invariance:** ISI, IG
* **Robustness:** worst-group CVaR, crisis CVaR, volatility ratios
* **Efficiency:** return-risk efficiency, turnover

Run diagnostics on any checkpoint:

```bash
python scripts/run_diagnostics.py \
  --checkpoint <path> \
  --results-dir results/diagnostics
```

---

## 6. Data

### Synthetic

Generated at runtime; no external files needed.

### Real SPY

Requires preprocessed files:

```
data/processed/spy_prices.csv
data/processed/spy_regimes.txt
```

Format details in `docs/data_preparation.md`.

---

## 7. Repository Structure

```
hirm/             # core library
configs/          # experiment configs
scripts/          # runners and diagnostics
analysis/         # result aggregation + plots
notebooks/        # Colab demos
docs/             # documentation and design notes
results/          # generated at runtime (gitignored)
```

---

## 8. Reproducibility

Each run stores:

* seed
* config copy
* arguments
* timestamps
* device info
* git commit hash

A reproduction script for the full suite is provided in `scripts/`.
