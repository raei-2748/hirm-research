"""Minimal training step helpers."""
from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor


def train_step(
    model,
    objective,
    optimizer,
    batch: Dict[str, Tensor],
    env_ids: Tensor,
    risk_fn,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Run a single optimization step for ``model`` using ``objective``."""

    optimizer.zero_grad()
    loss, logs = objective(model, batch, env_ids, risk_fn)
    loss.backward()
    optimizer.step()
    return loss, logs


__all__ = ["train_step"]
