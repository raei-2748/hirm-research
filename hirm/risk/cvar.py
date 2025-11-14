"""Conditional Value-at-Risk implemented with pure Python."""
from __future__ import annotations

import math
from typing import Iterable, List


def cvar_loss(episode_pnl: Iterable[float], alpha: float = 0.95) -> float:
    pnl = list(float(value) for value in episode_pnl)
    losses = [-value for value in pnl]
    losses.sort(reverse=True)
    tail_fraction = max(1, math.ceil((1 - alpha) * len(losses)))
    tail = losses[:tail_fraction]
    return sum(tail) / len(tail)


__all__ = ["cvar_loss"]
