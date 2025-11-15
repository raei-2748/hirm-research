"""Regime helpers."""
# Volatility thresholds mirror the HIRM paper's 20-day realized volatility bands
# for mapping continuous volatility into discrete stress regimes.
from __future__ import annotations

from typing import Dict

import numpy as np

REGIME_NAMES: Dict[int, str] = {
    0: "low",
    1: "medium",
    2: "high",
    3: "crisis",
}


def map_vol_to_regime(vol: float) -> int:
    """Map annualized realized volatility to discrete regime labels."""

    if not np.isfinite(vol):
        vol = 0.0
    vol = max(0.0, float(vol))
    if vol < 0.10:
        return 0
    if vol < 0.25:
        return 1
    if vol < 0.40:
        return 2
    return 3


__all__ = ["map_vol_to_regime", "REGIME_NAMES"]
