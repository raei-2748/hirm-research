"""Feature computation for HIRM state representation."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from hirm.episodes.episode import Episode


_PHI_DEFAULTS = {
    "realized_vol_window": 20,
    "momentum_window": 20,
    "cvar_window": 20,
    "cvar_alpha": 0.05,
    "annualization_factor": math.sqrt(252.0),
}

_R_DEFAULTS = {
    "include_time_fraction": True,
}


def _get_config(config: Dict[str, Any] | None, defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    if config:
        merged.update(config)
    return merged


def _to_list(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        raw = values.tolist()
        if isinstance(raw, list):
            return [float(x) for x in raw]
    if isinstance(values, list):
        return [float(x) for x in values]
    return [float(x) for x in list(values)]


def _running_max(values: List[float]) -> List[float]:
    peaks: List[float] = []
    current = float("-inf")
    for value in values:
        current = value if value > current else current
        peaks.append(current)
    return peaks


def _previous_return_series(returns: List[float]) -> List[float]:
    if not returns:
        return []
    prev = [0.0]
    prev.extend(returns[:-1])
    return prev


def _realized_volatility_series(
    returns: List[float], window: int, annualization: float
) -> List[float]:
    vols: List[float] = []
    for idx in range(len(returns)):
        start = max(0, idx - window + 1)
        window_values = returns[start : idx + 1]
        if not window_values:
            vols.append(0.0)
            continue
        mean = sum(window_values) / len(window_values)
        variance = sum((val - mean) ** 2 for val in window_values) / len(window_values)
        vols.append(annualization * math.sqrt(variance))
    return vols


def _drawdown_series(prices: List[float], length: int) -> List[float]:
    if len(prices) < length:
        raise ValueError("prices must have at least `length` elements")
    price_slice = prices[:length]
    peaks = _running_max(price_slice)
    drawdowns: List[float] = []
    for price, peak in zip(price_slice, peaks):
        if peak <= 0:
            drawdowns.append(0.0)
        else:
            drawdowns.append(1.0 - price / peak)
    return drawdowns


def _momentum_series(prices: List[float], length: int, window: int) -> List[float]:
    if len(prices) < length:
        raise ValueError("prices must have at least `length` elements")
    price_slice = prices[:length]
    logs = [math.log(price) for price in price_slice]
    momentum: List[float] = []
    for idx in range(length):
        start = max(0, idx - window + 1)
        momentum.append(logs[idx] - logs[start])
    return momentum


def _quantile(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    position = (len(sorted_vals) - 1) * alpha
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_vals[lower]
    weight = position - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _cvar_series(returns: List[float], window: int, alpha: float) -> List[float]:
    cvar: List[float] = []
    for idx in range(len(returns)):
        start = max(0, idx - window + 1)
        window_values = returns[start : idx + 1]
        if not window_values:
            cvar.append(0.0)
            continue
        var = _quantile(window_values, alpha)
        tail = [val for val in window_values if val <= var]
        if tail:
            cvar.append(sum(tail) / len(tail))
        else:
            cvar.append(var)
    return cvar


def _stack_columns(columns: List[List[float]]) -> np.ndarray:
    if not columns or not columns[0]:
        return np.asarray([], dtype=float)
    rows = [list(values) for values in zip(*columns)]
    return np.asarray(rows, dtype="object")


def compute_phi_features(episode: Episode, config: Dict[str, Any] | None = None) -> np.ndarray:
    """Compute invariant ``Phi`` features for ``episode``."""

    cfg = _get_config(config, _PHI_DEFAULTS)
    returns = _to_list(episode.returns)
    prices = _to_list(episode.prices)
    length = len(returns)
    prev_returns = _previous_return_series(returns)
    realized_vol = _realized_volatility_series(
        returns,
        window=int(cfg["realized_vol_window"]),
        annualization=float(cfg["annualization_factor"]),
    )
    drawdown = _drawdown_series(prices, length)
    momentum = _momentum_series(prices, length, window=int(cfg["momentum_window"]))
    cvar = _cvar_series(
        returns,
        window=int(cfg["cvar_window"]),
        alpha=float(cfg["cvar_alpha"]),
    )
    inventory = [0.0 for _ in range(length)]
    phi = _stack_columns(
        [prev_returns, realized_vol, drawdown, momentum, cvar, inventory]
    )
    return phi


def compute_r_features(
    episode: Episode, env_id: int, config: Dict[str, Any] | None = None
) -> np.ndarray:
    """Compute context ``r`` features for ``episode``."""

    cfg = _get_config(config, _R_DEFAULTS)
    returns = _to_list(episode.returns)
    length = len(returns)
    if "regimes" not in episode.metadata:
        raise KeyError("Episode metadata must include 'regimes' aligned with prices")
    regimes = _to_list(episode.metadata["regimes"])
    if len(regimes) < length:
        raise ValueError("regime metadata must be at least as long as the episode length")
    regime_feature = [float(val) for val in regimes[:length]]
    env_feature = [float(env_id) for _ in range(length)]
    features = [regime_feature, env_feature]
    if cfg.get("include_time_fraction", True):
        if length <= 1:
            time_feature = [0.0 for _ in range(length)]
        else:
            time_feature = [idx / (length - 1) for idx in range(length)]
        features.append(time_feature)
    r = _stack_columns(features)
    return r


def compute_all_features(
    episode: Episode,
    env_id: int,
    config: Dict[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    """Compute both ``Phi`` and ``r`` features for ``episode``."""

    config = config or {}
    phi_config = config.get("phi") if isinstance(config, dict) else None
    r_config = config.get("r") if isinstance(config, dict) else None
    phi = compute_phi_features(episode, phi_config)
    r = compute_r_features(episode, env_id, r_config)
    if phi.shape[0] != r.shape[0]:
        raise ValueError("Phi and r feature lengths must match")
    return {"phi": phi, "r": r}


__all__ = ["compute_phi_features", "compute_r_features", "compute_all_features"]
