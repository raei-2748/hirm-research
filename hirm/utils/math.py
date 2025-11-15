"""Basic math helpers for Phase 1 that avoid heavy dependencies."""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence


def softplus(x: float) -> float:
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def stable_logsumexp(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("values cannot be empty")
    max_val = max(values_list)
    shifted_sum = sum(math.exp(v - max_val) for v in values_list)
    return math.log(shifted_sum) + max_val


def clip(values: Iterable[float], minimum: float, maximum: float) -> List[float]:
    return [clip_scalar(v, minimum, maximum) for v in values]


def clip_scalar(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, value)))


def rolling_window(series: Sequence[float], window: int) -> List[List[float]]:
    """Return rolling windows over ``series`` with the given size."""

    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(series):
        return []
    return [list(series[idx : idx + window]) for idx in range(len(series) - window + 1)]


def safe_log(x: float, eps: float = 1e-12) -> float:
    """Compute ``log(max(x, eps))`` to avoid -inf."""

    return math.log(max(x, eps))


def safe_div(num: float, denom: float, eps: float = 1e-12) -> float:
    """Safely divide ``num`` by ``denom`` with epsilon regularization."""

    adjusted = denom if abs(denom) > eps else math.copysign(eps, denom if denom != 0 else 1.0)
    return num / adjusted


def _flatten_nested(value: Any) -> List[float]:
    if hasattr(value, "detach"):
        return _flatten_nested(value.detach())
    if hasattr(value, "cpu"):
        return _flatten_nested(value.cpu())
    if hasattr(value, "numpy"):
        try:
            arr = value.numpy()
            return [float(v) for v in arr.reshape(-1)]  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback path
            pass
    if hasattr(value, "ravel"):
        try:
            flattened = value.ravel()
            return [float(v) for v in flattened]
        except Exception:  # pragma: no cover - fallback path
            pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # pragma: no cover - fallback path
            pass
    if isinstance(value, (list, tuple)):
        flattened: List[float] = []
        for item in value:
            flattened.extend(_flatten_nested(item))
        return flattened
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def flatten_gradients(param_list: Iterable[object]) -> List[float]:
    """Flatten gradient buffers from an iterable of parameters."""

    flattened: List[float] = []
    for param in param_list:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        flattened.extend(_flatten_nested(grad))
    return flattened


def cvar(values: Sequence[float], alpha: float = 0.95) -> float:
    """Compute a simple Conditional Value at Risk over ``values``."""

    if not values:
        raise ValueError("values cannot be empty")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    sorted_values = sorted(values)
    tail_fraction = max(1, int(math.ceil(len(sorted_values) * (1 - alpha))))
    tail = sorted_values[:tail_fraction]
    return sum(tail) / len(tail)


__all__ = [
    "cvar",
    "clip",
    "clip_scalar",
    "flatten_gradients",
    "rolling_window",
    "safe_div",
    "safe_log",
    "softplus",
    "stable_logsumexp",
]
