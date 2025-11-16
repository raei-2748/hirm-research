"""Group Distributionally Robust Optimization objective."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from .base import BaseObjective, register_objective


@register_objective("group_dro")
class GroupDROObjective(BaseObjective):
    """Minimize the worst-case environment risk or a smoothed variant."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        obj_cfg = self.objective_cfg
        self.smooth = bool(getattr(obj_cfg, "group_dro_smooth", False))
        self.temperature = float(getattr(obj_cfg, "group_dro_temperature", 1.0))

    def compute_loss(
        self,
        env_risks: Dict[str, Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> Tensor:
        if not env_risks:
            raise ValueError("env_risks must be non-empty for GroupDRO")
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        if not self.smooth:
            loss = risks.max()
            self.log_metrics(
                {
                    "train/objective/groupdro/max_risk": loss,
                    "train/objective/mean_risk": mean_risk,
                }
            )
            return loss
        weights = torch.softmax(self.temperature * risks, dim=0)
        loss = (weights * risks).sum()
        logs = {
            "train/objective/groupdro/weighted_risk": loss,
            "train/objective/groupdro/temperature": torch.tensor(self.temperature),
            "train/objective/mean_risk": mean_risk,
        }
        for (env_name, risk), weight in zip(env_risks.items(), weights):
            logs[f"train/objective/groupdro/weight/{env_name}"] = weight
            logs[f"train/objective/groupdro/risk/{env_name}"] = risk
        self.log_metrics(logs)
        return loss


__all__ = ["GroupDROObjective"]
