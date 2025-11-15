"""Realized volatility and regime labelling utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .regimes import map_vol_to_regime


def realized_vol_20(returns: np.ndarray) -> np.ndarray:
    """Compute trailing 20-day realized volatility (annualized)."""

    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValueError("returns must be 1D")
    n = arr.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    vols = np.empty(n, dtype=float)
    for idx in range(n):
        start = max(0, idx - 19)
        window = arr[start : idx + 1]
        if window.size == 0:
            vols[idx] = 0.0
        else:
            vols[idx] = np.sqrt(252.0) * float(np.std(window, ddof=0))
    return vols


def _vol_series_from_prices(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be 1D")
    if prices.size < 2:
        return np.zeros(prices.size, dtype=float)
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    vol_returns = realized_vol_20(returns)
    vol_prices = np.empty(prices.size, dtype=float)
    vol_prices[0] = vol_returns[0] if vol_returns.size > 0 else 0.0
    vol_prices[1:] = vol_returns
    return vol_prices


def label_series_with_regimes(prices: np.ndarray) -> np.ndarray:
    """Label a price series with discrete volatility regimes."""

    prices_arr = np.asarray(prices, dtype=float)
    if prices_arr.ndim != 1:
        raise ValueError("prices must be 1D")
    if prices_arr.size == 0:
        return np.empty(0, dtype=int)
    vols = _vol_series_from_prices(prices_arr)
    regimes = np.empty_like(vols, dtype=int)
    for idx, vol in enumerate(vols):
        regimes[idx] = map_vol_to_regime(float(vol))
    return regimes


def price_series_volatility(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return returns-aligned and price-aligned volatility arrays."""

    prices_arr = np.asarray(prices, dtype=float)
    if prices_arr.size < 2:
        return np.zeros(prices_arr.size, dtype=float), np.zeros(prices_arr.size, dtype=float)
    log_prices = np.log(prices_arr)
    returns = np.diff(log_prices)
    vol_returns = realized_vol_20(returns)
    vol_prices = np.empty(prices_arr.size, dtype=float)
    if vol_returns.size == 0:
        vol_prices.fill(0.0)
    else:
        vol_prices[0] = vol_returns[0]
        vol_prices[1:] = vol_returns
    return vol_returns, vol_prices


__all__ = [
    "realized_vol_20",
    "label_series_with_regimes",
    "price_series_volatility",
]
