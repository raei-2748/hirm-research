"""Diagnostics namespace."""
from hirm.diagnostics.efficiency import compute_efficiency_metrics
from hirm.diagnostics.ig import compute_ig
from hirm.diagnostics.isi import compute_isi
from hirm.diagnostics.robustness import compute_robustness_metrics

__all__ = [
    "compute_efficiency_metrics",
    "compute_ig",
    "compute_isi",
    "compute_robustness_metrics",
]
