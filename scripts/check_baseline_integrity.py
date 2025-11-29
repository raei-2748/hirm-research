"""Validate that benchmark results directories are complete."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirm.utils.config import load_config

REQUIRED_FILES = {
    "train_logs.jsonl",
    "checkpoint.pt",
    "diagnostics.jsonl",
    "metadata.json",
}

CONFIG_CANDIDATES = ("config.json", "config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results-root", type=str, default="results")
    return parser.parse_args()


def _resolve_items(cfg) -> Tuple[Iterable[str], Iterable[str], Iterable[int]]:
    def _resolve_list(value):
        if isinstance(value, str):
            cleaned = value.strip("[]\"'")
            return [v.strip().strip("\"'") for v in cleaned.split(",") if v.strip()]
        return list(value)

    methods = _resolve_list(getattr(cfg, "methods", []))
    datasets = _resolve_list(getattr(cfg, "datasets", []))
    seeds = [int(s) for s in _resolve_list(getattr(cfg, "seeds", [0]))]
    return methods, datasets, seeds


def check_run(dataset: str, method: str, seed: int, root: Path) -> Tuple[bool, Dict[str, bool]]:
    base = root / dataset / method / f"seed_{seed}"
    status = {}
    for fname in REQUIRED_FILES:
        status[fname] = (base / fname).exists()
    status["config_snapshot"] = any((base / candidate).exists() for candidate in CONFIG_CANDIDATES)
    return all(status.values()), status


def main() -> None:
    import traceback
    try:
        args = parse_args()
        cfg = load_config(args.config)
        methods, datasets, seeds = _resolve_items(cfg)
        root = Path(args.results_root)
        all_ok = True
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    ok, details = check_run(dataset, method, seed, root)
                    prefix = "[OK]" if ok else "[MISSING]"
                    print(f"{prefix} {dataset}/{method}/seed_{seed}")
                if not ok:
                    all_ok = False
                    missing = [k for k, v in details.items() if not v]
                    print(f"  Missing: {', '.join(missing)}")
        if not all_ok:
            print("Integrity check failed: Missing required files", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Integrity check failed with exception: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
