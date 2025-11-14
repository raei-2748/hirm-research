"""Implementation of the HIRM objective using finite differences."""
from __future__ import annotations

import itertools
from typing import Dict

from hirm.envs.episodes import EnvBatch
from hirm.models.policy import Policy
from hirm.objectives.utils import compute_head_gradients, risk_on_env_batch


def hirm_loss(
    policy: Policy,
    env_batches: Dict[str, EnvBatch],
    lambda_invariance: float,
    alpha: float = 0.95,
) -> float:
    def loss_fn(policy: Policy, batch: EnvBatch) -> float:
        return risk_on_env_batch(policy, batch, alpha=alpha)

    risks = []
    gradients = []
    for batch in env_batches.values():
        loss, grad = compute_head_gradients(policy, loss_fn, batch)
        risks.append(loss)
        gradients.append(grad)
    mean_risk = sum(risks) / len(risks)
    if len(gradients) < 2:
        return mean_risk
    penalty_terms = []
    for g_i, g_j in itertools.combinations(gradients, 2):
        norm_i = sum(value * value for value in g_i) ** 0.5
        norm_j = sum(value * value for value in g_j) ** 0.5
        if norm_i == 0 or norm_j == 0:
            continue
        dot = sum(a * b for a, b in zip(g_i, g_j))
        cosine = dot / (norm_i * norm_j)
        penalty_terms.append(1.0 - cosine)
    penalty = sum(penalty_terms) / len(penalty_terms) if penalty_terms else 0.0
    return mean_risk + lambda_invariance * penalty


__all__ = ["hirm_loss"]
