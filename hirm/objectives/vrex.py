"""Variance Risk Extrapolation objective."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from .base import BaseObjective, register_objective


@register_objective("vrex")
class VREXObjective(BaseObjective):
    """Penalize risk variance across environments."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        obj_cfg = self.objective_cfg
        self.beta = float(getattr(obj_cfg, "beta", getattr(obj_cfg, "penalty_weight", 1.0)))

    def compute_loss(
        self,
        env_risks: Dict[str, Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> Tensor:
        if not env_risks:
            raise ValueError("env_risks must be non-empty for V-REx")
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        variance = ((risks - mean_risk) ** 2).mean()
        loss = mean_risk + self.beta * variance
        self.log_metrics(
            {
                "train/objective/mean_risk": mean_risk,
                "train/objective/variance_penalty": variance,
                "train/objective/vrex_beta": torch.tensor(self.beta),
            }
        )
        return loss


__all__ = ["VREXObjective"]
