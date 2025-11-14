"""Efficiency diagnostics placeholder."""
from __future__ import annotations

from typing import Dict

from hirm.envs.episodes import EnvBatch
from hirm.models.policy import Policy


def compute_efficiency_metrics(
    policy: Policy, env_batches: Dict[str, EnvBatch]
) -> Dict[str, float]:
    """Return simple exposure statistics."""

    if not env_batches:
        return {"avg_position": 0.0}
    exposures = []
    for batch in env_batches.values():
        states = batch.states
        avg_pos = states.mean().item()
        exposures.append(avg_pos)
    return {"avg_position": float(sum(exposures) / len(exposures))}


__all__ = ["compute_efficiency_metrics"]
