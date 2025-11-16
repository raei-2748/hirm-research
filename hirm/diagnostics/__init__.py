"""Diagnostic metrics for the HIRM I--R--E analysis."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .efficiency import compute_er, compute_tr
from .invariance import compute_ig, compute_isi
from .robustness import compute_vr, compute_wg


__all__ = [
    "compute_invariance_metrics",
    "compute_robustness_metrics",
    "compute_efficiency_metrics",
    "compute_all_diagnostics",
    "compute_isi",
    "compute_ig",
    "compute_wg",
    "compute_vr",
    "compute_er",
    "compute_tr",
]


def compute_invariance_metrics(
    *,
    isi_inputs: Mapping[str, Any],
    ig_inputs: Mapping[str, Any],
) -> Dict[str, float]:
    """Compute the invariance axis metrics (Section 5 of the paper)."""

    metrics: Dict[str, float] = {}
    metrics.update(compute_isi(**isi_inputs))
    metrics.update(compute_ig(**ig_inputs))
    return metrics


def compute_robustness_metrics(
    *,
    wg_inputs: Mapping[str, Any],
    vr_inputs: Mapping[str, Any],
) -> Dict[str, float]:
    """Compute robustness metrics (Worst-case generalization + volatility ratio)."""

    metrics: Dict[str, float] = {}
    metrics.update(compute_wg(**wg_inputs))
    metrics.update(compute_vr(**vr_inputs))
    return metrics


def compute_efficiency_metrics(
    *,
    er_inputs: Mapping[str, Any],
    tr_inputs: Mapping[str, Any],
) -> Dict[str, float]:
    """Compute efficiency metrics (Expected return-risk and turnover ratio)."""

    metrics: Dict[str, float] = {}
    metrics.update(compute_er(**er_inputs))
    metrics.update(compute_tr(**tr_inputs))
    return metrics


def compute_all_diagnostics(
    invariance_inputs: Mapping[str, Mapping[str, Any]],
    robustness_inputs: Mapping[str, Mapping[str, Any]],
    efficiency_inputs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    """Compute and merge all diagnostics into a single metric dictionary."""

    diagnostics: Dict[str, float] = {}
    diagnostics.update(compute_invariance_metrics(**invariance_inputs))
    diagnostics.update(compute_robustness_metrics(**robustness_inputs))
    diagnostics.update(compute_efficiency_metrics(**efficiency_inputs))
    return diagnostics
