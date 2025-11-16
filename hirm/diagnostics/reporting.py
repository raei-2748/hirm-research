"""Utilities for summarizing and plotting diagnostics outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency guard
        raise ImportError("pandas is required for diagnostics reporting")
    return pd


def load_diagnostics_results(paths: List[str]) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Load diagnostics JSONL files into a dataframe."""

    pandas = _require_pandas()
    records: List[Dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        files: Iterable[Path]
        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
        else:
            files = [path]
        for file_path in files:
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    metrics = record.pop("metrics", {})
                    for key, value in metrics.items():
                        record[f"metrics.{key}"] = value
                    records.append(record)
    if not records:
        return pandas.DataFrame()
    return pandas.DataFrame.from_records(records)


def compute_diagnostics_correlations(df) -> Dict[str, float]:  # type: ignore[no-untyped-def]
    """Compute pairwise correlations between canonical metrics."""

    pandas = _require_pandas()
    if df.empty:
        return {}
    required = {
        "metrics.ISI": None,
        "metrics.IG": None,
        "metrics.crisis_cvar": None,
    }
    available = {col for col in df.columns if col in required}
    results: Dict[str, float] = {}
    if {"metrics.ISI", "metrics.IG"}.issubset(available):
        results["corr(ISI, IG)"] = float(df["metrics.ISI"].corr(df["metrics.IG"]))
    if {"metrics.ISI", "metrics.crisis_cvar"}.issubset(available):
        results["corr(ISI, crisis_cvar)"] = float(
            df["metrics.ISI"].corr(df["metrics.crisis_cvar"])
        )
    if {"metrics.IG", "metrics.crisis_cvar"}.issubset(available):
        results["corr(IG, crisis_cvar)"] = float(
            df["metrics.IG"].corr(df["metrics.crisis_cvar"])
        )
    return results


def summarize_diagnostics_by_method(df) -> "pd.DataFrame":  # type: ignore[name-defined, no-untyped-def]
    """Group diagnostics by method/model and report mean/std."""

    pandas = _require_pandas()
    if df.empty:
        return pandas.DataFrame()
    group_key = "method" if "method" in df.columns else "model_name"
    if group_key not in df.columns:
        raise KeyError("Diagnostics results must contain 'method' or 'model_name'")
    metric_cols = [col for col in df.columns if col.startswith("metrics.")]
    grouped = df.groupby(group_key)[metric_cols]
    summary = grouped.agg(["mean", "std"])
    summary.columns = ["{}__{}".format(metric, agg) for metric, agg in summary.columns]
    return summary.reset_index()


def plot_isi_vs_ig(df, out_path: str) -> None:  # type: ignore[no-untyped-def]
    """Create a scatter plot of ISI vs IG."""

    if plt is None:  # pragma: no cover - dependency guard
        raise ImportError("matplotlib is required for plotting")
    if df.empty:
        raise ValueError("Dataframe is empty")
    fig, ax = plt.subplots()
    ax.scatter(df["metrics.ISI"], df["metrics.IG"], c="tab:blue")
    ax.set_xlabel("Internal Stability Index (ISI)")
    ax.set_ylabel("Invariance Gap (IG)")
    ax.set_title("ISI vs IG")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_isi_vs_crisis_cvar(df, out_path: str) -> None:  # type: ignore[no-untyped-def]
    """Scatter plot for ISI vs crisis CVaR risk."""

    if plt is None:  # pragma: no cover - dependency guard
        raise ImportError("matplotlib is required for plotting")
    if df.empty:
        raise ValueError("Dataframe is empty")
    fig, ax = plt.subplots()
    ax.scatter(df["metrics.ISI"], df["metrics.crisis_cvar"], c="tab:red")
    ax.set_xlabel("Internal Stability Index (ISI)")
    ax.set_ylabel("Crisis CVaR")
    ax.set_title("ISI vs Crisis CVaR")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "load_diagnostics_results",
    "compute_diagnostics_correlations",
    "summarize_diagnostics_by_method",
    "plot_isi_vs_ig",
    "plot_isi_vs_crisis_cvar",
]
