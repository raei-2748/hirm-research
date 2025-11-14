"""Robustness diagnostics placeholder."""
from __future__ import annotations

from typing import Dict

from hirm.envs.episodes import EnvBatch
from hirm.models.policy import Policy


def compute_robustness_metrics(
    policy: Policy, env_batches: Dict[str, EnvBatch]
) -> Dict[str, float]:
    """Return simple worst/best risk summary."""

    if not env_batches:
        return {"worst_risk": 0.0, "best_risk": 0.0}
    from hirm.objectives.utils import risk_on_env_batch

    risks = [risk_on_env_batch(policy, batch).detach().item() for batch in env_batches.values()]
    return {"worst_risk": float(max(risks)), "best_risk": float(min(risks))}


__all__ = ["compute_robustness_metrics"]
