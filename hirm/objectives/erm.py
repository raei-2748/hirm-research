"""Empirical Risk Minimization objective."""
from __future__ import annotations

from typing import Any, Dict

import torch

from hirm.objectives.base import BaseObjective, register_objective


@register_objective("erm")
class ERMObjective(BaseObjective):
    """ERM baseline minimizing the mean of per-environment risks."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)

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
        self._log(extra_state, {"train/risk/mean": mean_risk.detach()})
        return mean_risk


__all__ = ["ERMObjective"]
