"""Empirical Risk Minimization objective."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from .base import BaseObjective, register_objective


@register_objective("erm")
class ERMObjective(BaseObjective):
    """ERM baseline minimizing the mean of per-environment risks."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)

    def compute_loss(
        self,
        env_risks: Dict[str, Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> Tensor:
        if not env_risks:
            raise ValueError("env_risks must be non-empty for ERM")
        risks = torch.stack(list(env_risks.values()))
        loss = risks.mean()
        self.log_metrics({"train/objective/mean_risk": loss})
        return loss


__all__ = ["ERMObjective"]
