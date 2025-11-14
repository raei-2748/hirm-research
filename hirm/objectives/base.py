"""Objective implementations using NumPy losses."""
from __future__ import annotations

import abc
from typing import Any, Dict, Mapping

import statistics

from hirm.envs.episodes import EnvBatch
from hirm.models.policy import Policy
from hirm.objectives.utils import risk_on_env_batch


class Objective(abc.ABC):
    def __init__(self, alpha: float = 0.95) -> None:
        self.alpha = alpha

    def risk(self, policy: Policy, env_batch: EnvBatch) -> float:
        return risk_on_env_batch(policy, env_batch, alpha=self.alpha)

    @abc.abstractmethod
    def __call__(
        self, policy: Policy, env_batches: Mapping[str, EnvBatch]
    ) -> float:
        raise NotImplementedError


class ERMObjective(Objective):
    def __call__(
        self, policy: Policy, env_batches: Mapping[str, EnvBatch]
    ) -> float:
        risks = [self.risk(policy, batch) for batch in env_batches.values()]
        return statistics.fmean(risks)


class GroupDROObjective(Objective):
    def __call__(
        self, policy: Policy, env_batches: Mapping[str, EnvBatch]
    ) -> float:
        risks = [self.risk(policy, batch) for batch in env_batches.values()]
        return max(risks)


class VRExObjective(Objective):
    def __init__(self, alpha: float = 0.95, penalty_weight: float = 1.0) -> None:
        super().__init__(alpha=alpha)
        self.penalty_weight = penalty_weight

    def __call__(
        self, policy: Policy, env_batches: Mapping[str, EnvBatch]
    ) -> float:
        risks = [self.risk(policy, batch) for batch in env_batches.values()]
        mean_risk = statistics.fmean(risks)
        variance = statistics.fmean([(risk - mean_risk) ** 2 for risk in risks])
        return mean_risk + self.penalty_weight * variance


class IRMv1Objective(Objective):
    def __init__(self, alpha: float = 0.95, penalty_weight: float = 1.0) -> None:
        super().__init__(alpha=alpha)
        self.penalty_weight = penalty_weight

    def __call__(
        self, policy: Policy, env_batches: Mapping[str, EnvBatch]
    ) -> float:
        risks = [self.risk(policy, batch) for batch in env_batches.values()]
        mean_risk = statistics.fmean(risks)
        penalty = statistics.fmean([(risk - mean_risk) ** 2 for risk in risks])
        return mean_risk + self.penalty_weight * penalty


class HIRMObjective(Objective):
    def __init__(self, alpha: float = 0.95, lambda_invariance: float = 1.0) -> None:
        super().__init__(alpha=alpha)
        self.lambda_invariance = lambda_invariance

    def __call__(
        self, policy: Policy, env_batches: Mapping[str, EnvBatch]
    ) -> float:
        from hirm.objectives.hirm import hirm_loss

        return hirm_loss(
            policy,
            env_batches,
            lambda_invariance=self.lambda_invariance,
            alpha=self.alpha,
        )


OBJECTIVE_REGISTRY = {
    "erm": ERMObjective,
    "groupdro": GroupDROObjective,
    "vrex": VRExObjective,
    "irmv1": IRMv1Objective,
    "hirm": HIRMObjective,
}


def build_objective(config: Dict[str, Any]) -> Objective:
    name = config.get("name", "erm")
    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(f"Unknown objective '{name}'")
    cls = OBJECTIVE_REGISTRY[name]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    return cls(**kwargs)


__all__ = [
    "Objective",
    "ERMObjective",
    "GroupDROObjective",
    "VRExObjective",
    "IRMv1Objective",
    "HIRMObjective",
    "build_objective",
]
