"""Internal invariance diagnostics stubs."""
from __future__ import annotations

from typing import Dict

from hirm.envs.episodes import EnvBatch
from hirm.models.policy import Policy


def compute_isi(policy: Policy, env_batches: Dict[str, EnvBatch]) -> Dict[str, float]:
    """Compute a placeholder invariance score.

    Later phases will replace this with the full statistic.  For now we report
    the maximum pairwise risk gap as a coarse stability indicator.
    """

    if not env_batches:
        return {"isi": 0.0}
    risks = []
    for batch in env_batches.values():
        # defer import to avoid circular dependency
        from hirm.objectives.utils import risk_on_env_batch

        risks.append(risk_on_env_batch(policy, batch).detach().item())
    isi = max(risks) - min(risks)
    return {"isi": float(isi)}


__all__ = ["compute_isi"]
