"""GroupDRO objective implementation."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from hirm.objectives.common import compute_env_risks


class GroupDROObjective:
    """Group DRO objective minimizing the worst-environment risk."""

    def __init__(self, cfg: Any) -> None:
        self.step_size = float(getattr(cfg, "step_size", 0.05))
        self.min_weight = float(getattr(cfg, "min_weight", 1e-3))
        self.env_weights: Dict[int, float] = {}

    def __call__(
        self,
        policy,
        batch: Dict[str, Tensor],
        env_ids: Tensor,
        risk_fn,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        env_risks, pnl, _, _ = compute_env_risks(policy, batch, env_ids, risk_fn)
        self._update_weights(env_risks)
        loss = sum(self.env_weights[env] * risk for env, risk in env_risks.items())
        loss_tensor = loss if isinstance(loss, Tensor) else torch.tensor(loss)
        logs: Dict[str, Tensor] = {"loss": loss_tensor.detach(), "pnl/mean": pnl.mean().detach()}
        for env, risk in env_risks.items():
            logs[f"risk/env_{env}"] = risk.detach()
            logs[f"groupdro/weight_env_{env}"] = torch.tensor(
                self.env_weights[env]
            )
        return loss_tensor, logs

    def _update_weights(self, env_risks: Dict[int, Tensor]) -> None:
        if not self.env_weights:
            uniform = 1.0 / len(env_risks)
            self.env_weights = {env: uniform for env in env_risks}
        for env, risk in env_risks.items():
            detached = float(risk.detach())
            weight = self.env_weights.get(env, self.min_weight)
            weight *= math.exp(detached * self.step_size)
            self.env_weights[env] = max(weight, self.min_weight)
        total = sum(self.env_weights.values())
        if total <= 0:
            total = 1.0
        for env in list(self.env_weights.keys()):
            self.env_weights[env] = self.env_weights[env] / total


__all__ = ["GroupDROObjective"]
