"""Variance Risk Extrapolation objective."""
from __future__ import annotations

from typing import Any, Dict

import torch

from hirm.objectives.base import BaseObjective, register_objective


@register_objective("vrex")
class VREXObjective(BaseObjective):
    """V-REx objective penalizing variance of environment risks."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        self.beta = float(getattr(self.obj_cfg, "beta", getattr(self.obj_cfg, "penalty_weight", 1.0)))

    def compute_loss(
        self,
        env_risks: Dict[str, torch.Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del model, batch
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        variance = ((risks - mean_risk) ** 2).mean()
        loss = mean_risk + self.beta * variance
        self._log(
            extra_state,
            {
                "train/risk/mean": mean_risk.detach(),
                "train/objective/variance_penalty": variance.detach(),
                "train/objective/vrex_beta": torch.tensor(self.beta),
            },
        )
        return loss


__all__ = ["VREXObjective"]
