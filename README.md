## Praesidium: Causally Distilled Structural Invariance

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Note: Planning to rename from HIRM to Praesidium for future extensions

This repository implements **HIRM** (Hedging Invariant Risk Minmization), a portfolio hedging framework that learns representations and hedge policies whose **risk gradients are aligned across market regimes**. The codebase supports synthetic stress tests, real SPY data, standard baselines (ERM, IRM, GroupDRO, VREx), diagnostics for invariance, robustness and efficiency, and a full ablation suite for the HIRM architecture.

Maybe renamed to Praesidium in the future.

The code is organized as a sequence of “phases” that mirror the research pipeline:

- Phase 1 to 3: environments, state construction, data integrity  
- Phase 4 to 7: model objectives, benchmark grid, diagnostics  
- Phase 8: ablation registry, experiment grid, and I–R–E analysis

---

## Main features

**Models and objectives**

- HIRM objective with head only gradient alignment penalty  
- Baselines: ERM, IRM, GroupDRO, VREx  
- Flexible representation layer with Phi / r factorization and variants

**Environments and data**

- Synthetic Heston and Heston plus Merton jump regimes  
- Real SPY historical replay with volatility based regime labelling  
- Explicit regime bands: low, medium, high, crisis

**Diagnostics (I–R–E)**

- Invariance: ISI (with components C1, C2, C3) and IG  
- Robustness: crisis CVaR, worst environment generalization (WG), volatility ratio (VR), drawdown  
- Efficiency: mean PnL, efficiency ratio (ER), turnover (TR)

**Experiment management**

- Phase 7 grid runner for benchmarks on methods × datasets × seeds  
- Phase 8 ablation grid runner and ablation registry  
- Analysis script that aggregates results into tables and deltas vs full HIRM

---

## Repository layout

```text
hirm/
  data/             # CSV and episode loaders (SPY, synthetic episodes)
  diagnostics/      # ISI, IG, robustness, efficiency and reporting utilities
  envs/             # Synthetic and real environments and regime labelling
  episodes/         # Episode container and generators
  experiments/      # Dataset and method registries, ablation registry
  models/           # InvariantPolicy and supporting modules
  objectives/       # ERM, IRM, GroupDRO, VREx, HIRM objectives and risk functions
  state/            # Feature engineering, preprocessing, train/val/test splits
  training/         # Training loops and utilities
  utils/            # Config handling, seeding, logging, math helpers

analysis/
  analyze_ablation.py    # Phase 8 ablation result aggregation

configs/
  base.yaml              # Shared configuration
  envs/                  # Environment specific configs
  experiments/           # Phase 7, Phase 8 and tiny test configs

scripts/
  run_tiny_experiment.py           # Minimal end to end training demo
  run_grid.py                      # Phase 7 benchmark grid runner
  run_ablation_grid.py             # Phase 8 ablation grid runner
  run_experiment_and_diagnostics.py
  run_diagnostics.py
  summarize_diagnostics.py
  summarize_phase7_results.py

tests/
  test_phase1_smoke.py
  test_phase7_smoke.py
  test_phase8_smoke.py
  ... plus unit tests for envs, features, objectives and diagnostics

README_phase8.md      # Phase specific notes for ablation work
LICENSE               # MIT License
pytest.ini
````

---

## Installation

### 1. Set up a Python environment

Recommended Python version: **3.10**. Versions 3.11–3.12 work if you keep NumPy below 2.0, but 3.10 is the most stable across local and Colab.

```bash
git clone <your-repo-url> hirm-research
cd hirm-research

python -m venv .venv
source .venv/bin/activate      # on macOS or Linux
# .venv\Scripts\activate       # on Windows PowerShell
```

### 2. Install dependencies and the package

Use the pinned requirements to avoid the NumPy/PyTorch incompatibility seen on Python 3.12:

```bash
pip install -r requirements.txt
pip install -e .
```

This installs NumPy 1.26.x, Torch 2.4.1 (CPU build by default), pandas, matplotlib, and the testing stack. The same dependency set is declared in `pyproject.toml`, so `pip install -e .` will work on its own if you prefer.

---

## Data

### Synthetic data

Synthetic environments (Heston and Merton jumps) are generated on the fly and do not require external files.

### SPY data

Real SPY experiments expect a CSV at:

```text
data/raw/spy.csv
```

The loader is intentionally flexible:

* It uses `csv.DictReader`
* Looks for a date column such as `Date` or `Timestamp`
* Looks for a price column among `Adj Close`, `Adj_Close`, `AdjClose`, `Close`, `price`

A minimal CSV example:

```text
Date,Adj Close
2010-01-04,111.92
2010-01-05,112.37
...
```

Place the file at `data/raw/spy.csv` before running SPY experiments.

---

## Quick start

### 1. Run the tiniest possible experiment

This script uses synthetic data and a small model, useful to check that everything imports and runs.

```bash
python scripts/run_tiny_experiment.py
```

You should see a short training log and no errors.

---

### 2. Run the Phase 8 smoke test (recommended first run)

This is a small ablation grid on synthetic data only, designed to be light enough to run on a laptop CPU.

```bash
python scripts/run_ablation_grid.py \
  --config configs/experiments/phase8.yaml \
  --datasets synthetic_heston \
  --ablation_names hirm_full,hirm_no_hgca \
  --seeds 0 \
  --device cpu \
  --reduced
```

This will create a directory such as:

```text
results/phase8/synthetic_heston/hirm_full/seed_0/
  config.yaml
  diagnostics.jsonl
  train_logs.jsonl
  metadata.json
  checkpoint.pt
```

Then run the analysis script:

```bash
python analysis/analyze_ablation.py --root_dir results/phase8
```

This aggregates metrics across runs and writes summary CSV files with mean, standard deviation and deltas vs `hirm_full`.

---

## Run Phase 8 in Colab

The Colab workflow mirrors local usage but pins GPU-friendly wheels and NumPy 1.26 to avoid the `np.number`/`np.object_` import error seen with NumPy 2.x.

1. In **Runtime → Change runtime type**, choose **Python 3.10** and **GPU**.
2. Clone the repo and check out the Phase 8 branch:

   ```bash
   !git clone https://github.com/raei-2748/hirm-research.git
   %cd hirm-research
   !git checkout phase8-ablation
   ```

3. Install dependencies and the package (CUDA 12.1 Torch wheels via the extra index):

   ```bash
   !pip install -r requirements-colab.txt
   !pip install -e .
   ```

4. Run the smoke test:

   ```bash
   !pytest tests/test_phase8_smoke.py -vv
   ```

5. Run a reduced ablation on GPU:

   ```bash
   !python scripts/run_ablation_grid.py \
       --config configs/experiments/phase8.yaml \
       --datasets synthetic_heston \
       --ablation_names hirm_full,erm_baseline \
       --seeds 0 \
       --device cuda \
       --reduced
   ```

The `requirements-colab.txt` file locks NumPy to `<2.0` and installs the CUDA 12.1 Torch wheels to ensure compatibility in fresh Colab runtimes.

---

## Running full experiments

### Phase 7 benchmark grid

The Phase 7 grid compares ERM, IRM, GroupDRO, VREx and HIRM across datasets and seeds.

Example command:

```bash
python scripts/run_grid.py \
  --config configs/experiments/phase7.yaml \
  --datasets synthetic_heston,real_spy \
  --methods erm,irm,groupdro,vrex,hirm \
  --seeds 0,1,2 \
  --device cuda   # use cpu if no GPU is available
```

Results are written under:

```text
results/phase7/<dataset>/<method>/seed_<k>/
```

You can summarize them with:

```bash
python scripts/summarize_phase7_results.py --root_dir results/phase7
```

### Phase 8 ablation grid

The Phase 8 grid sweeps over the ablation registry, for example:

* `hirm_full`
* `hirm_no_hgca`
* `hirm_full_irm`
* `hirm_env_specific_heads`
* `hirm_no_split`
* `hirm_phi_only`
* `hirm_r_only`
* `hirm_random_env_labels`
* `hirm_coarse_env_bands`
* plus baselines such as `erm_baseline`, `groupdro_baseline`, `vrex_baseline`

A typical command:

```bash
python scripts/run_ablation_grid.py \
  --config configs/experiments/phase8.yaml \
  --datasets synthetic_heston,real_spy \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --device cuda
```

You can restrict ablations via `--ablation_names` and use `--reduced` for a cheaper test run.

Once the grid finishes, run:

```bash
python analysis/analyze_ablation.py --root_dir results/phase8
```

This produces tables of:

* Crisis CVaR
* WG, VR
* ISI, IG
* Mean PnL, ER, TR
* Deltas relative to `hirm_full` for each dataset and ablation

These tables are the raw material for the ablation section of the paper.

---

## Running tests

To check that everything is wired correctly:

```bash
pytest -q
```

Key tests:

* `tests/test_phase1_smoke.py` checks basic environment and config plumbing
* `tests/test_phase7_smoke.py` runs a small Phase 7 grid and diagnostics
* `tests/test_phase8_smoke.py` runs a reduced ablation grid and the analysis script
* Other tests cover envs, state features, objectives and diagnostics

---

## Performance notes

* Synthetic experiments are light and can run on CPU
* Real SPY experiments and the full Phase 8 ablation grid are computationally heavier, especially with ISI and IG diagnostics
* For serious multi seed grids, a GPU (T4, V100, A40, A100, similar) is strongly preferred

---

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

```
```

