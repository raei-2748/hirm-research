"""Basic math helpers for Phase 1 that avoid heavy dependencies."""
from __future__ import annotations

import math
from typing import Iterable, List


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


__all__ = ["softplus", "stable_logsumexp", "clip", "clip_scalar"]
