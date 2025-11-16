"""Summarize diagnostics JSONL files into tables and correlations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hirm.diagnostics.reporting import (
    compute_diagnostics_correlations,
    load_diagnostics_results,
    plot_isi_vs_crisis_cvar,
    plot_isi_vs_ig,
    summarize_diagnostics_by_method,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        help="Column to group by when aggregating metrics (default: method if available)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    df = load_diagnostics_results([str(path) for path in jsonl_files])
    correlations = compute_diagnostics_correlations(df)
    summary = summarize_diagnostics_by_method(df, group_by=args.group_by)
    output = {
        "num_records": int(len(df)),
        "correlations": correlations,
        "summary": summary.to_dict(orient="records") if not summary.empty else [],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary_path = out_path.with_suffix(".summary.csv")
    if not summary.empty:
        summary.to_csv(summary_path, index=False)
    if args.plot_dir and not df.empty:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_isi_vs_ig(df, str(plot_dir / "isi_vs_ig.png"))
        if "metrics.crisis_cvar" in df.columns:
            plot_isi_vs_crisis_cvar(df, str(plot_dir / "isi_vs_cvar.png"))


if __name__ == "__main__":
    main()
