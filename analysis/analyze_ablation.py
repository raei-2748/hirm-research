"""Aggregate Phase 8 ablation results into summary tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_ablation_results(root_dir: str) -> pd.DataFrame:
    root = Path(root_dir)
    records: List[dict] = []
    for dataset_dir in root.glob("*/"):
        dataset = dataset_dir.name.rstrip("/")
        for ablation_dir in dataset_dir.glob("*/"):
            ablation_name = ablation_dir.name.rstrip("/")
            for seed_dir in ablation_dir.glob("seed_*/"):
                seed_str = seed_dir.name.replace("seed_", "")
                try:
                    seed = int(seed_str)
                except ValueError:
                    continue
                diag_path = seed_dir / "diagnostics.jsonl"
                if not diag_path.exists():
                    continue
                diag_rows = _load_jsonl(diag_path)
                for row in diag_rows:
                    for key, value in row.items():
                        if key in {"dataset", "method", "seed", "ablation_name"}:
                            continue
                        if isinstance(value, dict):
                            for subkey, subval in value.items():
                                records.append(
                                    {
                                        "dataset": dataset,
                                        "ablation_name": ablation_name,
                                        "seed": seed,
                                        "metric_name": f"{key}/{subkey}",
                                        "value": subval,
                                    }
                                )
                        else:
                            records.append(
                                {
                                    "dataset": dataset,
                                    "ablation_name": ablation_name,
                                    "seed": seed,
                                    "metric_name": key,
                                    "value": value,
                                }
                            )
    return pd.DataFrame.from_records(records)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "ablation_name", "metric_name"])
    summary = grouped["value"].agg([
        ("mean", "mean"),
        ("std", "std"),
        ("median", "median"),
        ("q10", lambda x: x.quantile(0.1)),
        ("q90", lambda x: x.quantile(0.9)),
    ])
    return summary.reset_index()


def add_deltas(summary: pd.DataFrame, baseline: str = "hirm_full") -> pd.DataFrame:
    baseline_rows = summary[summary["ablation_name"] == baseline]
    merged = summary.merge(
        baseline_rows,
        on=["dataset", "metric_name"],
        suffixes=("", "_baseline"),
        how="left",
    )
    merged["delta_mean_vs_hirm"] = merged["mean"] - merged["mean_baseline"]
    return merged


def pivot_table(summary: pd.DataFrame, metrics: List[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for dataset, df_ds in summary.groupby("dataset"):
        pivot = df_ds[df_ds["metric_name"].isin(metrics)].pivot_table(
            index="ablation_name", columns="metric_name", values="mean"
        )
        tables[dataset] = pivot
    return tables


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="results/phase8", help="Path to ablation outputs")
    parser.add_argument("--root_dir", type=str, help="Alias for --root", dest="root_alias")
    parser.add_argument("--metrics", type=str, default="metrics/cvar95/crisis,metrics/wg,metrics/isi/global,metrics/ig/global,metrics/pnl/mean,metrics/er,metrics/tr")
    parser.add_argument("--output", type=str, default="results/phase8/ablation_summary.csv")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    root_dir = args.root_alias or args.root
    df = load_ablation_results(root_dir)
    summary = summarize_metrics(df)
    summary_with_delta = add_deltas(summary)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_with_delta.to_csv(out_path, index=False)

    tables = pivot_table(summary_with_delta, metrics)
    for dataset, table in tables.items():
        print(f"\n=== {dataset} ===")
        print(table)


if __name__ == "__main__":
    main()
