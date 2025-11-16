"""Crisis diagnostics helpers."""
from __future__ import annotations

from typing import Dict, Sequence

from hirm.utils.math import cvar


def compute_crisis_cvar(pnl_time_series: Sequence[float], alpha: float) -> Dict[str, float]:
    """Compute CVaR over crisis-period losses from a PnL series."""

    values = [float(v) for v in pnl_time_series]
    if not values:
        raise ValueError("pnl_time_series cannot be empty")
    losses = [-v for v in values]
    crisis_value = max(0.0, float(cvar(losses, alpha=alpha)))
    return {"crisis_cvar": crisis_value}


__all__ = ["compute_crisis_cvar"]
