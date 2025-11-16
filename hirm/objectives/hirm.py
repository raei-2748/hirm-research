"""Head-Invariant Risk Minimization objective."""
from __future__ import annotations

import itertools
from typing import Any, Dict

import torch
from torch import Tensor

from .base import BaseObjective, register_objective
from .common import flatten_head_gradients


@register_objective("hirm")
class HIRMObjective(BaseObjective):
    """Encourage alignment of head gradients across environments."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        obj_cfg = self.objective_cfg
        self.lambda_hirm = float(getattr(obj_cfg, "lambda_hirm", getattr(obj_cfg, "lambda_invariance", 1.0)))
        self.eps = float(getattr(obj_cfg, "eps", 1e-8))

    def compute_loss(
        self,
        env_risks: Dict[str, Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> Tensor:
        if not env_risks:
            raise ValueError("env_risks must be non-empty for HIRM")
        head_params = self.get_head_parameters(model)
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        grad_vectors = self._compute_gradients(env_risks, head_params)
        if len(grad_vectors) < 2:
            penalty = torch.zeros(1, device=mean_risk.device)
            alignment = torch.ones(1, device=mean_risk.device)
        else:
            penalty = self._dispersion_penalty(grad_vectors)
            alignment = self._alignment_metric(grad_vectors)
        total_loss = mean_risk + self.lambda_hirm * penalty
        self.log_metrics(
            {
                "train/objective/mean_risk": mean_risk,
                "train/objective/hirm_penalty": penalty,
                "train/objective/hirm_lambda": torch.tensor(self.lambda_hirm),
                "train/hirm/alignment": alignment,
            }
        )
        if extra_state is not None:
            extra_state["hirm_alignment"] = alignment.detach()
        return total_loss

    def _compute_gradients(
        self,
        env_risks: Dict[str, Tensor],
        head_params,
    ) -> list[Tensor]:
        grad_vectors: list[Tensor] = []
        for risk in env_risks.values():
            grads = torch.autograd.grad(
                risk,
                head_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            flat = flatten_head_gradients(grads)
            norm = flat.norm(p=2)
            grad_vectors.append(flat / (norm + self.eps))
        return grad_vectors

    def _dispersion_penalty(self, grads: list[Tensor]) -> Tensor:
        stacked = torch.stack(grads)
        mean = stacked.mean(dim=0, keepdim=True)
        return ((stacked - mean) ** 2).mean()

    def _alignment_metric(self, grads: list[Tensor]) -> Tensor:
        if len(grads) < 2:
            return torch.ones(1, device=grads[0].device)
        cosines: list[Tensor] = []
        for g1, g2 in itertools.combinations(grads, 2):
            cosines.append((g1 * g2).sum())
        return torch.stack(cosines).mean()


__all__ = ["HIRMObjective"]
