"""Minimal end-to-end smoke test for ERM and HIRM.

The script spins up tiny runs on a synthetic environment to ensure that
checkpoints, logs, and diagnostics artifacts are written without error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirm.engine import train_and_evaluate
from hirm.experiments.registry import ExperimentRunConfig
from hirm.utils.config import ConfigNode, load_config, to_plain_dict


def _maybe_shrink_training(cfg: ConfigNode, *, max_epochs: int = 3, batch_size: int = 16) -> ConfigNode:
    cfg = ConfigNode(to_plain_dict(cfg))
    if hasattr(cfg, "training"):
        cfg.training.max_epochs = max_epochs
        cfg.training.batch_size = batch_size
        cfg.training.early_stop_patience = min(getattr(cfg.training, "early_stop_patience", max_epochs), max_epochs)
        cfg.training.episodes_per_env = min(getattr(cfg.training, "episodes_per_env", 2), 2)
        cfg.training.num_steps = min(getattr(cfg.training, "num_steps", 3), 5)
    if hasattr(cfg, "diagnostics"):
        cfg.diagnostics.enabled = True
        if hasattr(cfg.diagnostics, "crisis"):
            cfg.diagnostics.crisis.enabled = True
    return cfg


def _ensure_outputs(run_dir: Path, required: Iterable[str]) -> None:
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts in {run_dir}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/baseline_benchmark.yaml")
    parser.add_argument("--results-dir", default="results/smoke_suite")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    smoke_cfg = _maybe_shrink_training(cfg)
    methods = ["erm", "hirm"]
    dataset = getattr(smoke_cfg, "datasets", ["synthetic_heston"])[0]
    seeds = getattr(smoke_cfg, "seeds", [0])

    for method in methods:
        for seed in seeds[:1]:
            run_cfg = ExperimentRunConfig(
                dataset=dataset,
                method=method,
                seed=int(seed),
                config=smoke_cfg,
                device=device,
                ablation=None,
            )
            out_dir = results_root / dataset / method / f"seed_{seed}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            train_and_evaluate(run_cfg, out_dir, force_rerun=True)
            _ensure_outputs(
                out_dir,
                required=["train_logs.jsonl", "diagnostics.jsonl", "checkpoint.pt", "config.json", "metadata.json"],
            )
            metadata = json.loads((out_dir / "metadata.json").read_text())
            metadata["smoke"] = True
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
