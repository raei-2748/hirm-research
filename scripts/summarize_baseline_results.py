"""Summarize baseline benchmark diagnostics into compact CSV/JSON tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple

import csv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirm.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results-root", type=str, default="results")
    return parser.parse_args()


def _load_diagnostics(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _aggregate(metrics: List[float]) -> Tuple[float, float]:
    if not metrics:
        return 0.0, 0.0
    if len(metrics) == 1:
        return metrics[0], 0.0
    return float(mean(metrics)), float(pstdev(metrics))


def summarize_dataset(dataset: str, methods: Iterable[str], seeds: Iterable[int], root: Path):
    summary_records = []
    summary_dict = []
    for method in methods:
        entries: List[Dict] = []
        for seed in seeds:
            diag_path = root / dataset / method / f"seed_{seed}" / "diagnostics.jsonl"
            if not diag_path.exists():
                print(f"DEBUG: Missing {diag_path}")
                continue
            entries.extend(list(_load_diagnostics(diag_path)))
        if not entries:
            continue
        isi_vals = [float(e.get("isi", 0.0)) for e in entries]
        ig_vals = [float(e.get("ig", 0.0)) for e in entries]
        wg_vals = [float(e.get("wg", 0.0)) for e in entries]
        vr_vals = [float(e.get("vr", 0.0)) for e in entries]
        er_vals = [float(e.get("er", 0.0)) for e in entries]
        tr_vals = [float(e.get("tr", 0.0)) for e in entries]
        crisis_vals: List[float] = []
        for e in entries:
            env_metrics = e.get("env_metrics", {}) or {}
            for env, metrics in env_metrics.items():
                if "crisis" in env:
                    crisis_vals.append(float(metrics.get("cvar95", metrics.get("risk", 0.0))))
        mean_crisis, std_crisis = _aggregate(crisis_vals)
        record = {
            "dataset": dataset,
            "method": method,
            "num_seeds": len(entries),
            "mean_isi": _aggregate(isi_vals)[0],
            "std_isi": _aggregate(isi_vals)[1],
            "mean_ig": _aggregate(ig_vals)[0],
            "std_ig": _aggregate(ig_vals)[1],
            "mean_wg": _aggregate(wg_vals)[0],
            "std_wg": _aggregate(wg_vals)[1],
            "mean_vr": _aggregate(vr_vals)[0],
            "std_vr": _aggregate(vr_vals)[1],
            "mean_er": _aggregate(er_vals)[0],
            "std_er": _aggregate(er_vals)[1],
            "mean_tr": _aggregate(tr_vals)[0],
            "std_tr": _aggregate(tr_vals)[1],
            "mean_crisis_cvar95": mean_crisis,
            "std_crisis_cvar95": std_crisis,
        }
        summary_records.append(record)
        summary_dict.append(record)
    return summary_records, summary_dict


def _resolve_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip("[]\"'")
        return [v.strip().strip("\"'") for v in cleaned.split(",") if v.strip()]
    return list(value)


def write_summary(dataset: str, records: List[Dict], root: Path):
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"{dataset}_summary.csv"
    json_path = summary_dir / f"{dataset}_summary.json"

    if records:
        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
    else:
        # Create empty files with proper structure
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            # Write a header with expected columns
            writer = csv.DictWriter(csvfile, fieldnames=["dataset", "method", "num_seeds"])
            writer.writeheader()
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump([], handle, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    methods = _resolve_list(getattr(cfg, "methods", []))
    datasets = _resolve_list(getattr(cfg, "datasets", []))
    seeds = [int(s) for s in _resolve_list(getattr(cfg, "seeds", [0]))]
    root = Path(args.results_root)

    for dataset in datasets:
        records, summary_dict = summarize_dataset(dataset, methods, seeds, root)
        write_summary(dataset, records, root)
        print(f"Summarized {dataset}: {len(records)} method entries")


if __name__ == "__main__":
    main()
