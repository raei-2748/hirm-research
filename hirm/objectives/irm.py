"""IRMv1 objective implementation."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from .base import BaseObjective, register_objective
from .common import flatten_head_gradients


@register_objective("irmv1")
class IRMv1Objective(BaseObjective):
    """Penalize environment-specific gradients of the head parameters."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        obj_cfg = self.objective_cfg
        self.lambda_irm = float(getattr(obj_cfg, "lambda_irm", getattr(obj_cfg, "penalty_weight", 1.0)))

    def compute_loss(
        self,
        env_risks: Dict[str, Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> Tensor:
        if not env_risks:
            raise ValueError("env_risks must be non-empty for IRMv1")
        head_params = self.get_head_parameters(model)
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        penalty = torch.zeros(1, device=mean_risk.device)
        for risk in env_risks.values():
            grads = torch.autograd.grad(
                risk,
                head_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            flat = flatten_head_gradients(grads)
            penalty = penalty + (flat ** 2).sum()
        total_loss = mean_risk + self.lambda_irm * penalty
        self.log_metrics(
            {
                "train/objective/mean_risk": mean_risk,
                "train/objective/irm_penalty": penalty,
                "train/objective/irm_lambda": torch.tensor(self.lambda_irm),
            }
        )
        return total_loss


__all__ = ["IRMv1Objective"]
