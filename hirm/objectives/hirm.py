"""Head Invariant Risk Minimization (HIRM) objective.

Implements the Head Gradient Cosine Alignment (HGCA) penalty described in
Section 4.2 of "Robust Generalization for Hedging under Crisis Regime Shifts".
"""
from __future__ import annotations

import itertools
from typing import Any, Dict

import torch

from hirm.objectives.base import BaseObjective, register_objective
from hirm.objectives.common import flatten_head_gradients


def _get_parameters(model, mode: str):
    if mode == "head_only":
        if hasattr(model, "head_parameters"):
            return list(model.head_parameters())
        if hasattr(model, "head"):
            return list(model.head.parameters())
        raise ValueError("Model must expose head parameters for HIRM")
    if mode in {"full", "full_irm"}:
        return list(model.parameters())
    if mode in {"none", "env_specific_heads"}:
        return []
    raise ValueError(f"Unsupported invariance_mode '{mode}'")


@register_objective("hirm")
class HIRMObjective(BaseObjective):
    """Decision-head invariance objective grounded in Sections 4.2 and 4.4.

    Loss = Mean(Risk_e) + λ * Dispersion(Grads_e)

    Dispersion corresponds to the cosine-based gradient disagreement term
    (Eq. 11), enforcing alignment of normalized head gradients so invariance
    focuses on directional mechanism rather than scale.
    """

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        # Section 4.4: λ controls strength of the head alignment constraint.
        self.lambda_hirm = float(getattr(self.obj_cfg, "lambda_hirm", getattr(self.obj_cfg, "lambda_invariance", 1.0)))
        self.eps = float(getattr(self.obj_cfg, "eps", 1e-8))
        self.invariance_mode = str(getattr(self.obj_cfg, "invariance_mode", "head_only"))

    def compute_loss(
        self,
        env_risks: Dict[str, torch.Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Return the HIRM objective aligned with the paper's derivation.

        Section 4.2 defines the HGCA penalty on per-environment gradients, and
        Section 4.4 normalizes those gradients before computing cosine
        dispersion so the constraint targets the hedge rule's direction.
        """

        del batch
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        head_params = _get_parameters(model, self.invariance_mode)
        if not head_params or self.invariance_mode == "none":
            self._log(
                extra_state,
                {
                    "train/risk/mean": mean_risk.detach(),
                    "train/objective/hirm_penalty": torch.tensor(0.0, device=mean_risk.device),
                    "train/objective/hirm_lambda": torch.tensor(self.lambda_hirm),
                    "train/hirm/alignment": torch.tensor(0.0, device=mean_risk.device),
                },
            )
            return mean_risk

        gradients: Dict[str, torch.Tensor] = {}
        for env_name, risk in env_risks.items():
            grads = torch.autograd.grad(
                risk,
                head_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            flat = flatten_head_gradients(grads)
            gradients[env_name] = flat / (flat.norm() + self.eps)

        dispersion = self._gradient_dispersion(gradients)
        alignment = 1.0 - dispersion
        total = mean_risk + self.lambda_hirm * dispersion

        alignment_detached = alignment.detach()
        self._log(
            extra_state,
            {
                "train/risk/mean": mean_risk.detach(),
                "train/objective/hirm_penalty": dispersion.detach(),
                "train/objective/hirm_lambda": torch.tensor(self.lambda_hirm),
                "train/hirm/alignment": alignment_detached,
            },
        )
        if extra_state is not None:
            extra_state["hirm_alignment"] = alignment_detached
        return total

    def _gradient_dispersion(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        env_names = list(gradients.keys())
        if len(env_names) <= 1:
            return torch.zeros(1, device=self.device)
        cos_values = []
        for env_a, env_b in itertools.combinations(env_names, 2):
            cos = torch.sum(gradients[env_a] * gradients[env_b])
            cos_values.append(cos)
        cos_tensor = torch.stack(cos_values)
        mean_cos = cos_tensor.mean()
        return 1.0 - mean_cos


__all__ = ["HIRMObjective"]
