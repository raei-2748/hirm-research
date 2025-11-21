# Robust Generalization for Hedging under Crisis Regime Shifts
<img width="875" height="178" alt="Screenshot 2025-11-21 at 9 41 38 PM" src="https://github.com/user-attachments/assets/b655545d-fdf2-4bad-8551-b6fcf4927e65" />

This repository contains the reference implementation for **HIRM (Hedging with Invariant Risk Minimization)** as the core engine of our architecture system **Praesidium**.

HIRM addresses the failure of standard Deep Hedging models during market crises. While standard ERM models overfit to low-volatility regime shortcuts, HIRM enforces **decision-level invariance**: it constrains the *hedge ratio's sensitivity to risk* to be stable across market environments, while allowing the internal representation to remain regime-adaptive.

## Key Features

*   **Objective:** Head Gradient Cosine Alignment (HGCA) penalty (Section 4.2).
*   **Architecture:** Disentangled Representation ($\phi$) and Decision Head ($\psi$).
*   **Diagnostics:** 
    *   **ISI (Internal Stability Index):** Measures gradient and feature stability.
    *   **WG (Worst-Group Risk):** CVaR-95 under the worst realized regime.
    *   **ER/TR:** Efficiency and Turnover ratios.

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Benchmark (Table 2 in Paper)
Compare ERM, GroupDRO, V-REx, and HIRM on Synthetic Heston data.
```bash
python scripts/run_grid.py \
  --config configs/experiments/baseline_benchmark.yaml \
  --mode benchmark \
  --datasets synthetic_heston \
  --device cuda:0
```

### 3. Run Real-World SPY Analysis
Requires `data/processed/spy_prices.csv`. 
```bash
python scripts/run_grid.py \
  --config configs/experiments/baseline_benchmark.yaml \
  --mode benchmark \
  --datasets real_spy
```

### 4. Diagnostics & Plotting
Summarize results into the I-R-E (Invariance-Robustness-Efficiency) tables.
```bash
python scripts/summarize_diagnostics.py \
  --results-dir results/custom \
  --out results/summary.json
```

## Repository Structure

*   `hirm/objectives/`: Implementation of ERM, IRM, GroupDRO, VREx, and HIRM.
*   `hirm/diagnostics/`: Invariance (ISI, IG), Robustness (WG, VR), Efficiency (ER, TR).
*   `hirm/envs/`: Heston/Merton synthetic generators and SPY data loaders.
*   `configs/`: Experiment hyperparameters aligned with the paper.


### Summary of Actions Taken

1.  **Engineered a `train_and_evaluate` engine** to replace 1000+ lines of copy-pasted script logic.
2.  **Unified the CLI** into `run_grid.py` with clear modes (`benchmark` vs `ablation`).
3.  **Aligned Code with Math:** The `HIRMObjective` now explicitly comments on the normalization and cosine operations derived in the paper.
4.  **Cleaned Artifacts:** Removed "Phase 7/8/9" references from the user-facing documentation.
