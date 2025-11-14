"""Outcome dispersion diagnostics placeholder."""
from __future__ import annotations

from typing import Dict

from hirm.envs.episodes import EnvBatch
from hirm.models.policy import Policy


def compute_ig(policy: Policy, env_batches: Dict[str, EnvBatch]) -> Dict[str, float]:
    """Return max risk gap across environments."""

    if not env_batches:
        return {"ig": 0.0}
    from hirm.objectives.utils import risk_on_env_batch

    risks = [risk_on_env_batch(policy, batch).detach().item() for batch in env_batches.values()]
    return {"ig": float(max(risks) - min(risks))}


__all__ = ["compute_ig"]
