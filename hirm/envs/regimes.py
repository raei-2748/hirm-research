"""Utilities for constructing volatility-band regimes."""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class VolatilityBand:
    name: str
    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value < self.upper


DEFAULT_BANDS: Sequence[VolatilityBand] = (
    VolatilityBand("low", 0.0, 0.10),
    VolatilityBand("medium", 0.10, 0.25),
    VolatilityBand("high", 0.25, 0.40),
    VolatilityBand("crisis", 0.40, float("inf")),
)


def compute_realized_volatility(
    returns: Sequence[float], window: int = 20, annualization: int = 252
) -> List[float]:
    """Return a rolling realized volatility series."""

    realized: List[float] = []
    for idx in range(len(returns)):
        if idx + 1 < window:
            realized.append(float("nan"))
            continue
        window_slice = returns[idx + 1 - window : idx + 1]
        mean = sum(window_slice) / window
        variance = sum((val - mean) ** 2 for val in window_slice) / max(window - 1, 1)
        realized.append(math.sqrt(variance) * math.sqrt(annualization))
    return realized


def assign_volatility_bands(
    realized_vol: Sequence[float], bands: Sequence[VolatilityBand] = DEFAULT_BANDS
) -> List[str]:
    """Map realized volatility to band names."""

    labels: List[str] = []
    for value in realized_vol:
        label = "unknown"
        if math.isnan(value):
            labels.append(label)
            continue
        for band in bands:
            if band.contains(value):
                label = band.name
                break
        labels.append(label)
    return labels


def tag_regimes(
    dates: Sequence[dt.date],
    returns: Sequence[float],
    window: int = 20,
    bands: Sequence[VolatilityBand] = DEFAULT_BANDS,
) -> Dict[str, List[str]]:
    """Assign volatility bands to each date and split.

    Returns
    -------
    dict
        Mapping with keys ``"dates"`` and ``"labels"`` for downstream use.
    """

    realized = compute_realized_volatility(returns, window=window)
    labels = assign_volatility_bands(realized, bands=bands)
    return {"dates": list(dates), "labels": labels}


__all__ = [
    "VolatilityBand",
    "DEFAULT_BANDS",
    "compute_realized_volatility",
    "assign_volatility_bands",
    "tag_regimes",
]
