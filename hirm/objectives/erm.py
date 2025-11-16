"""Empirical Risk Minimization objective."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from hirm.objectives.common import compute_env_risks


class ERMObjective:
    """ERM baseline minimizing the mean environment risk from the paper."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def __call__(
        self,
        policy,
        batch: Dict[str, Tensor],
        env_ids: Tensor,
        risk_fn,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        env_risks, pnl, _, _ = compute_env_risks(policy, batch, env_ids, risk_fn)
        risks = torch.stack(list(env_risks.values()))
        loss = risks.mean()
        logs: Dict[str, Tensor] = {"loss": loss.detach(), "risk/mean": loss.detach()}
        for env, risk in env_risks.items():
            logs[f"risk/env_{env}"] = risk.detach()
        logs["pnl/mean"] = pnl.mean().detach()
        return loss, logs


__all__ = ["ERMObjective"]
