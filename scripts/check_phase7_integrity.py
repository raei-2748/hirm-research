"""Validate that Phase 7 results directories are complete."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

from hirm.utils.config import load_config

REQUIRED_FILES = {
    "train_logs.jsonl",
    "checkpoint.pt",
    "diagnostics.jsonl",
    "config.yaml",
    "metadata.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def _resolve_items(cfg) -> Tuple[Iterable[str], Iterable[str], Iterable[int]]:
    methods = getattr(cfg, "methods", [])
    datasets = getattr(cfg, "datasets", [])
    seeds = getattr(cfg, "seeds", [0])
    return methods, datasets, seeds


def check_run(dataset: str, method: str, seed: int) -> Tuple[bool, Dict[str, bool]]:
    base = Path("results") / dataset / method / f"seed_{seed}"
    status = {}
    for fname in REQUIRED_FILES:
        status[fname] = (base / fname).exists()
    return all(status.values()), status


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    methods, datasets, seeds = _resolve_items(cfg)
    all_ok = True
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                ok, details = check_run(dataset, method, seed)
                prefix = "[OK]" if ok else "[MISSING]"
                print(f"{prefix} {dataset}/{method}/seed_{seed}")
                if not ok:
                    all_ok = False
                    missing = [k for k, v in details.items() if not v]
                    print(f"  Missing: {', '.join(missing)}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
