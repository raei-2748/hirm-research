# Hedging with Invariant Risk Minimization

*Note: Future versions may adopt the project name **Praesidium** (Causal Distilled Structural Invariance).*

<p align="left">

  <img src="https://img.shields.io/badge/version-v1.0.0-blue.svg">
  <img src="https://img.shields.io/badge/license-MIT-green.svg">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red">
  <img src="https://img.shields.io/badge/Reproducible-Yes-brightgreen">
  <img src="https://img.shields.io/badge/Colab-demo-yellow">
  <img src="https://img.shields.io/badge/data-SPY%20%2B%20Synthetic-orange">

</p>

This repository provides the reference implementation of **HIRM (Hedging with Invariant Risk Minimization)**, a decision-level robustness method for dynamic hedging under regime shifts.

Instead of enforcing invariance on features or losses, HIRM regularizes the **hedge decision rule** across environments. The architecture separates:

- a **representation module** that adapts to market structure  
- a **hedge head** regularized to remain stable across regimes  

The codebase includes synthetic environments, SPY experiments, standard baselines, and diagnostics.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate         # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
````

For Colab usage, see `notebooks/`.

---

## Quickstart

### Smoke test

```bash
python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml \
  --device cpu
```

### Synthetic demo

```bash
python scripts/run_full_experiment_suite.py \
  --datasets synthetic_heston \
  --methods hirm_full,erm_baseline \
  --seeds 0
```

### SPY demo

Requires:

```text
data/processed/spy_prices.csv
data/processed/spy_regimes.txt
```

Then:

```bash
python scripts/run_full_experiment_suite.py \
  --datasets real_spy \
  --methods hirm_full,erm_baseline \
  --seeds 0
```

Additional commands are listed in `RUNS.md`.

---

## Experiments

Experiment groups:

* **baseline_benchmark**: ERM, GroupDRO, V-REx, IRM, HIRM
* **ablation_study**: HIRM component analysis
* **full_experiment_suite**: combined synthetic and SPY grid
* **smoke_test**: minimal end to end check

Configs live in `configs/experiments/`, with matching runners under `scripts/`.

---

## Diagnostics

Diagnostics cover:

* **Invariance**: ISI, Invariance Gap
* **Robustness**: worst group CVaR, crisis CVaR, volatility ratios
* **Efficiency**: return risk efficiency, turnover

Example:

```bash
python scripts/run_diagnostics.py \
  --checkpoint <path> \
  --results-dir results/diagnostics
```

---

## Data

* **Synthetic**: generated at runtime, no external files required
* **SPY**: prices and regimes under `data/processed/`

See `docs/data_preparation.md` for details.

---

## Repository Structure

```text
hirm/             core library: models, envs, objectives, training, diagnostics
configs/          experiment and model configs
scripts/          training, grids, diagnostics, utilities
analysis/         aggregation and plotting
notebooks/        Colab demos
docs/             documentation and design notes
results/          run outputs (gitignored)
```

---

## Reproducibility

Each run saves:

* config snapshot
* seed and CLI arguments
* timestamps and device info
* git commit hash

Standard patterns are provided in `scripts/` and `RUNS.md`.
