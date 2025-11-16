"""Group Distributionally Robust Optimization objective."""
from __future__ import annotations

from typing import Any, Dict

import torch

from hirm.objectives.base import BaseObjective, register_objective


@register_objective("group_dro")
class GroupDROObjective(BaseObjective):
    """GroupDRO objective using either hard max or a smoothed softmax."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        super().__init__(cfg, device)
        self.smooth = bool(getattr(self.obj_cfg, "group_dro_smooth", False))
        self.temperature = float(getattr(self.obj_cfg, "group_dro_temperature", 1.0))

    def compute_loss(
        self,
        env_risks: Dict[str, torch.Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del model, batch
        risks = torch.stack(list(env_risks.values()))
        if not self.smooth or len(risks) == 1:
            loss = risks.max()
            self._log(
                extra_state,
                {"train/objective/groupdro_temperature": torch.tensor(self.temperature)},
            )
            return loss
        weights = torch.softmax(self.temperature * risks, dim=0)
        loss = (weights * risks).sum()
        self._log(
            extra_state,
            {
                "train/objective/groupdro_temperature": torch.tensor(self.temperature),
                "train/objective/groupdro_weights": weights.detach(),
            },
        )
        return loss


__all__ = ["GroupDROObjective"]
