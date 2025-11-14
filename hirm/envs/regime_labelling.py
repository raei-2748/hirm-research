"""Utilities for labeling market regimes based on realized volatility."""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

DEFAULT_REGIME_THRESHOLDS: Mapping[str, float] = {
    "low": 0.10,
    "medium": 0.25,
    "high": 0.40,
}

REGIME_TO_ID: Mapping[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "crisis": 3,
}

ID_TO_REGIME: Dict[int, str] = {value: key for key, value in REGIME_TO_ID.items()}


def compute_realized_vol(prices: Sequence[float], window: int = 20, trading_days: int = 252) -> List[float]:
    """Compute annualized realized volatility using rolling log returns."""

    values = [float(price) for price in prices]
    if len(values) <= window:
        raise ValueError("prices length must exceed rolling window")

    log_prices = [math.log(price) for price in values]
    log_returns = [log_prices[idx + 1] - log_prices[idx] for idx in range(len(log_prices) - 1)]
    realized_vol: List[float] = [float("nan")] * len(values)

    for idx in range(window - 1, len(log_returns)):
        window_slice = log_returns[idx - window + 1 : idx + 1]
        mean = sum(window_slice) / len(window_slice)
        variance = sum((val - mean) ** 2 for val in window_slice) / len(window_slice)
        realized_vol[idx + 1] = math.sqrt(variance) * math.sqrt(trading_days)

    # Forward fill NaNs so downstream logic receives aligned arrays.
    first_valid_idx = next((i for i, val in enumerate(realized_vol) if not math.isnan(val)), None)
    if first_valid_idx is None:
        raise RuntimeError("Unable to compute realized volatility for the provided prices")
    first_value = realized_vol[first_valid_idx]
    for idx in range(first_valid_idx):
        realized_vol[idx] = first_value
    for idx, value in enumerate(realized_vol):
        if math.isnan(value):
            realized_vol[idx] = first_value
    return realized_vol


def label_regimes(
    realized_vol: Sequence[float],
    thresholds: Mapping[str, float] | None = None,
) -> List[int]:
    """Assign discrete regime labels based on realized volatility bands."""

    values = [float(vol) for vol in realized_vol]
    final_thresholds: Dict[str, float] = dict(DEFAULT_REGIME_THRESHOLDS)
    if thresholds:
        final_thresholds.update(thresholds)

    low = final_thresholds["low"]
    medium = final_thresholds["medium"]
    high = final_thresholds["high"]

    labels: List[int] = []
    for value in values:
        if value < low:
            labels.append(REGIME_TO_ID["low"])
        elif value < medium:
            labels.append(REGIME_TO_ID["medium"])
        elif value < high:
            labels.append(REGIME_TO_ID["high"])
        else:
            labels.append(REGIME_TO_ID["crisis"])
    return labels


__all__ = [
    "DEFAULT_REGIME_THRESHOLDS",
    "REGIME_TO_ID",
    "ID_TO_REGIME",
    "compute_realized_vol",
    "label_regimes",
]
