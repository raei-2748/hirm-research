## Praesidium: Causally Distilled Structural Invariance

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![reproducibility](https://img.shields.io/badge/Reproducible-Yes-brightgreen)]()
[![python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![pytorch](https://img.shields.io/badge/PyTorch-2.0+-red)]()
[![version](https://img.shields.io/badge/version-v1.0.0-blue.svg)]()


Note: Planning to rename from HIRM to Praesidium for future extensions

</p>

This repository contains the reference implementation of **HIRM (Hedging with Invariant Risk Minimization)**, a decision level invariance approach for robust portfolio hedging under regime shifts. It is designed to be:

- research grade  
- reproducible  
- minimal and readable  
- easy to run on local machines and Google Colab  

The code supports synthetic model markets and real data experiments and provides diagnostics that link invariance properties to out of sample robustness.

---

## 1. Overview

### 1.1 Problem

Dynamic hedging strategies are often tuned on stable market regimes. When volatility, liquidity, or correlation structures shift, these strategies can fail exactly when protection is needed most. Traditional hedging pipelines and standard robust machine learning methods tend to regularize features or losses in ways that do not directly control the **hedge decision rule** itself.

HIRM aims to stabilize the **mapping from core risk factors to hedge actions** across environments, rather than trying to find fully invariant features or simply optimizing worst case losses.

### 1.2 Method in one paragraph

HIRM builds on the idea of invariant risk minimization but moves the invariance target from the representation level to the decision level. The model has a representation layer that reads prices, returns, Greeks or proxies, liquidity and inventory. The hedge head maps a small set of core risk factors to hedge actions. HIRM penalizes variation in the gradient of the risk objective with respect to the hedge head across training environments. This encourages a hedge rule that behaves consistently across regimes while still allowing the upstream representation to adapt to changing volatility and liquidity.

### 1.3 Main contributions in this repo

- A modular implementation of HIRM and baselines for dynamic hedging:
  - ERM  
  - GroupDRO  
  - V-REx  
  - IRM style baselines  
  - Rule based baselines  
- Synthetic environments with controlled regime shifts for stress testing.  
- Real data experiments based on SPY with volatility driven regimes.  
- A suite of diagnostics that quantify:
  - invariance of the decision rule  
  - robustness under regime shifts  
  - efficiency of risk taking vs protection  
- Ready to run experiment grids and analysis scripts that reproduce the main results.

---

## 2. Installation

### 2.1 Requirements

- Python 3.10 or 3.11  
- Recommended: virtual environment  
- PyTorch 2.0 or newer with CPU or CUDA support  

### 2.2 Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

pytest -q                        # optional but recommended
If you are running on Google Colab, see the notebooks in notebooks/ and the Colab section below.

3. Quickstart
3.1 Tiny synthetic smoke test
Run a fast end to end training loop on a small synthetic environment. This is the simplest way to confirm the installation and pipeline.

bash
Copy code
python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml \
  --device cpu
This will:

construct a small synthetic dataset

build a minimal model

run a short training loop

write metrics and a checkpoint under results/smoke_test/...

3.2 Minimal HIRM experiment on synthetic data
Run HIRM on a synthetic environment with clear regime shifts:

bash
Copy code
python scripts/run_full_experiment_suite.py \
  --config configs/experiments/full_experiment_suite.yaml \
  --datasets synthetic_heston \
  --methods hirm_full,erm_baseline \
  --seeds 0 \
  --device cpu \
  --results-dir results/phase9_synthetic_demo
Then run the analysis script:

bash
Copy code
python analysis/phase9_analysis.py \
  --root results/phase9_synthetic_demo \
  --output analysis/phase9_synthetic_demo
3.3 HIRM on SPY (reduced demo)
If you have prepared SPY data in data/processed, you can run a reduced real data experiment:

bash
Copy code
python scripts/run_full_experiment_suite.py \
  --config configs/experiments/full_experiment_suite.yaml \
  --datasets real_spy \
  --methods hirm_full,erm_baseline \
  --seeds 0 \
  --device cuda:0 \
  --results-dir results/phase9_spy_demo
Then analyze:

bash
Copy code
python analysis/phase9_analysis.py \
  --root results/phase9_spy_demo \
  --output analysis/phase9_spy_demo
For full experiment commands, refer to RUNS.md.

4. Experiments
The repository organizes experiments by purpose rather than by internal phase numbers. The main experiment families are:

4.1 Baseline benchmark
Compare HIRM to standard methods on synthetic and real environments.

Config:

text
Copy code
configs/experiments/baseline_benchmark.yaml
Runner:

bash
Copy code
python scripts/run_baseline_benchmark.py \
  --config configs/experiments/baseline_benchmark.yaml \
  --device cuda:0 \
  --results-dir results/baseline_benchmark
Typical methods:

erm_baseline

groupdro_baseline

vrex_baseline

irm_baseline

hirm_full

4.2 Ablation study
Measure how performance and diagnostics change when HIRM components are removed or altered, such as:

no decision level invariance penalty

no representation factorization into mechanistic and adaptive parts

different environment partitions

Config:

text
Copy code
configs/experiments/ablation_study.yaml
Runner:

bash
Copy code
python scripts/run_ablation_study.py \
  --config configs/experiments/ablation_study.yaml \
  --device cuda:0 \
  --results-dir results/ablation_study
Analysis:

bash
Copy code
python analysis/analyze_ablation.py \
  --root results/ablation_study \
  --output analysis/ablation_summary
4.3 Full experiment suite
This is the main research grid used for the paper. It typically covers:

Synthetic Heston type environments

Real SPY episodes with volatility based regimes

Multiple seeds and methods

Config:

text
Copy code
configs/experiments/full_experiment_suite.yaml
Runner:

bash
Copy code
python scripts/run_full_experiment_suite.py \
  --config configs/experiments/full_experiment_suite.yaml \
  --datasets synthetic_heston,real_spy \
  --methods erm_baseline,hirm_full,irm_baseline,groupdro_baseline,vrex_baseline \
  --seeds 0,1,2 \
  --device cuda:0 \
  --results-dir results/full_suite
Analysis:

bash
Copy code
python analysis/phase9_analysis.py \
  --root results/full_suite \
  --output analysis/phase9_full
5. Diagnostics
HIRM is evaluated not only by returns and risk but also by diagnostics that probe invariance and robustness.

5.1 Invariance diagnostics
Computed in hirm/diagnostics/invariance.py:

ISI: Invariant Signal Index, a measure of how stable the head gradients are across environments.

IG: Invariance Gap, the dispersion of gradients or decisions across regimes.

5.2 Robustness diagnostics
Computed in hirm/diagnostics/robustness.py and hirm/diagnostics/crisis.py:

WG: worst group risk, typically worst environment CVaR.

VR: volatility and variance ratios that compare regime risk to baseline.

Crisis CVaR: loss distribution conditioned on known crisis windows.

5.3 Efficiency diagnostics
Computed in hirm/diagnostics/efficiency.py:

ER: efficiency of risk taking (return per unit of downside risk).

TR: turnover related measures that capture implementation cost.

5.4 Running diagnostics on a checkpoint
You can run diagnostics on a specific trained model:

bash
Copy code
python scripts/run_diagnostics.py \
  --config configs/experiments/full_experiment_suite.yaml \
  --checkpoint results/full_suite/dataset=synthetic_heston/method=hirm_full/seed=0/checkpoint.pt \
  --results-dir results/diagnostics/hirm_synth_seed0 \
  --device cpu
Diagnostics will be saved as JSON and CSV files suitable for the analysis scripts.

6. Data
6.1 Synthetic environments
Synthetic environments live under hirm/envs/synthetic/. They allow:

controlled volatility dynamics

jump and regime features

reproducible hedging episodes

They are configured via the experiment configs and generated at runtime. No additional files are required beyond the repo and Python dependencies.

6.2 Real SPY data
Real data experiments use SPY prices and volatility based regimes. The implementation expects preprocessed files, for example:

data/processed/spy_prices.csv

data/processed/spy_regimes.txt

Due to data licensing constraints the repository does not distribute vendor data. You must supply or construct compatible datasets. See docs/data_preparation.md for details on formats and scripts if present.

7. Repository structure
A simplified view of the repository:

text
Copy code
hirm/                     # Core library
  data/                   # Data loading and preprocessing
  envs/                   # Synthetic and real market environments
  episodes/               # Episode generation utilities
  state/                  # State and feature construction
  models/                 # Policy architectures and encoders
  objectives/             # ERM, GroupDRO, V-REx, IRM, HIRM losses
  training/               # Training loops and utilities
  diagnostics/            # Invariance and robustness diagnostics
  experiments/            # Dataset and method registries
  utils/                  # Shared utilities

configs/
  envs/                   # Environment specific configs
  models/                 # Model hyperparameters
  objectives/             # Objective hyperparameters
  experiments/            # High level experiment configs

scripts/
  run_smoke_test.py
  run_baseline_benchmark.py
  run_ablation_study.py
  run_full_experiment_suite.py
  run_diagnostics.py
  summarize_phase7_results.py
  run_ablation_grid.py

analysis/
  phase9_analysis.py      # Full suite analysis
  analyze_ablation.py     # Ablation analysis
  ...                     # Additional plotting and reporting

notebooks/
  hirm_tiny_demo.ipynb
  hirm_full_suite_reduced.ipynb

docs/
  developer_notes.md
  diagnostics_overview.md
  config_format.md
  data_preparation.md
  history/
    phase7.md
    phase8.md
    phase9.md

results/                  # Created at runtime, not tracked by git
For a more complete description see docs/developer_notes.md.

8. Reproducibility
The repository is designed to support reproducible experiments.

All main scripts accept a --seed or --seeds flag.

A global seeding utility is called at the start of each run.

Experiment configs are stored as YAML files and copied into each run directory.

Each run directory records:

the config used

the random seeds

command line arguments

timestamps

device information

commit hash if available

A convenience script may be provided as:

bash
Copy code
bash scripts/reproduce_full_suite.sh
which runs the main reduced grids and analysis pipelines end to end. Details are described in RUNS.md.

9. Colab usage
The notebooks/ directory contains Colab ready notebooks:

hirm_tiny_demo.ipynb

installs the package

runs a tiny synthetic experiment

visualizes hedge behavior vs unhedged portfolio

hirm_full_suite_reduced.ipynb

runs a reduced version of the full experiment suite

reproduces key plots used in the paper

To use Colab:

Open the notebook in Colab.

Run the setup cell that installs dependencies and the package, for example:

python
Copy code
!pip install -q -e .
Follow the instructions inside the notebook.

