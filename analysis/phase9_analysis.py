"""Simple aggregation script for Phase 9 results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_COLUMNS = [
    "dataset",
    "method",
    "seed",
    "crisis_cvar",
    "WG",
    "VR",
    "ER",
    "TR",
    "ISI",
    "IG",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_records(root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for jsonl in root.rglob("diagnostics.jsonl"):
        rows = _load_jsonl(jsonl)
        for row in rows:
            row.setdefault("dataset", jsonl.parent.parent.name)
            row.setdefault("method", jsonl.parent.name)
            row.setdefault("seed", jsonl.parent.name.split("_")[-1])
            records.append(row)
    if not records:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    df = pd.DataFrame.from_records(records)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [col for col in DEFAULT_COLUMNS if col in df.columns and col not in {"dataset", "method", "seed"}]
    grouped = df.groupby(["dataset", "method"])
    summary = grouped[numeric_cols].mean().reset_index()
    return summary


def plot_cvar(summary: pd.DataFrame, output_dir: Path) -> None:
    if "crisis_cvar" not in summary.columns:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, sub in summary.groupby("dataset"):
        plt.figure(figsize=(6, 4))
        plt.bar(sub["method"], sub["crisis_cvar"])
        plt.ylabel("Crisis CVaR")
        plt.title(f"{dataset} crisis CVaR")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_cvar.png", dpi=200)
        plt.close()


def plot_isi_vs_ig(summary: pd.DataFrame, output_dir: Path) -> None:
    if not {"ISI", "IG"}.issubset(summary.columns):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    for method, sub in summary.groupby("method"):
        plt.scatter(sub["ISI"], sub["IG"], label=method)
    plt.xlabel("ISI")
    plt.ylabel("IG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "isi_vs_ig.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", type=str, default="results/full_experiment_suite")
    parser.add_argument("--output_dir", type=str, default="analysis_outputs/full_experiment_suite")
    args = parser.parse_args()

    root = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_records(root)
    summary = summarize(df)
    summary.to_csv(output_dir / "summary.csv", index=False)
    summary.to_json(output_dir / "summary.json", orient="records", indent=2)

    plot_cvar(summary, output_dir)
    plot_isi_vs_ig(summary, output_dir)
    print(f"Wrote summary for {len(summary)} rows to {output_dir}")


if __name__ == "__main__":
    main()
