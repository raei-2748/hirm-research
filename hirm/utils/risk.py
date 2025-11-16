"""Risk utility helpers."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def compute_cvar(losses: Tensor | Iterable[float], alpha: float = 0.95) -> Tensor:
    """Compute Conditional Value-at-Risk for ``losses``.

    Args:
        losses: 1D tensor of per-sample losses (higher means worse).
        alpha: CVaR tail probability.
    """

    values = losses if isinstance(losses, Tensor) else torch.as_tensor(losses)
    if values.numel() == 0:
        raise ValueError("losses tensor must be non-empty")
    flattened = values.reshape(-1)
    sorted_losses, _ = torch.sort(flattened)
    tail_count = max(1, int((1 - alpha) * sorted_losses.numel()))
    tail = sorted_losses[-tail_count:]
    return tail.mean()


__all__ = ["compute_cvar"]
