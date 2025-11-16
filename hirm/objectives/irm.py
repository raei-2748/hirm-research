"""IRMv1 objective based on head gradient penalties."""
from __future__ import annotations

from typing import Any, Dict, Iterable

import torch

from hirm.objectives.base import BaseObjective, register_objective
from hirm.objectives.common import flatten_head_gradients


def _get_head_parameters(model) -> Iterable[torch.nn.Parameter]:
    if hasattr(model, "head_parameters"):
        return list(model.head_parameters())
    if hasattr(model, "head"):
        return list(model.head.parameters())
    raise ValueError("Model must expose head parameters for IRMv1")


@register_objective("irmv1")
class IRMv1Objective(BaseObjective):
    """Classic IRMv1 using gradients of each env risk w.r.t. the head."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        self.lambda_irm = float(getattr(self.obj_cfg, "lambda_irm", getattr(self.obj_cfg, "penalty_weight", 1.0)))

    def compute_loss(
        self,
        env_risks: Dict[str, torch.Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del batch
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        head_params = _get_head_parameters(model)
        if not head_params:
            raise ValueError("IRMv1 requires at least one head parameter")
        penalty = torch.zeros(1, device=mean_risk.device)
        for risk in env_risks.values():
            grads = torch.autograd.grad(
                risk,
                head_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            grad_vec = flatten_head_gradients(grads)
            penalty = penalty + (grad_vec ** 2).sum()
        total = mean_risk + self.lambda_irm * penalty
        self._log(
            extra_state,
            {
                "train/risk/mean": mean_risk.detach(),
                "train/objective/irm_penalty": penalty.detach(),
                "train/objective/irm_lambda": torch.tensor(self.lambda_irm),
            },
        )
        return total


__all__ = ["IRMv1Objective"]
