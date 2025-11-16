"""IRMv1 objective and penalty."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from hirm.objectives.common import compute_env_risks, compute_pnl_from_actions


class IRMv1Objective:
    """Classic IRMv1 objective enforcing invariant optimal heads."""

    def __init__(self, cfg: Any) -> None:
        self.penalty_weight = float(getattr(cfg, "penalty_weight", 100.0))

    def __call__(
        self,
        policy,
        batch: Dict[str, Tensor],
        env_ids: Tensor,
        risk_fn,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        env_risks, pnl, actions, env_tensor = compute_env_risks(
            policy, batch, env_ids, risk_fn
        )
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        penalty = self._irm_penalty(batch, actions, env_tensor, risk_fn)
        loss = mean_risk + self.penalty_weight * penalty
        logs: Dict[str, Tensor] = {
            "loss": loss.detach(),
            "risk/mean": mean_risk.detach(),
            "irm/penalty": penalty.detach(),
            "irm/lambda": torch.tensor(self.penalty_weight),
            "pnl/mean": pnl.mean().detach(),
        }
        for env, risk in env_risks.items():
            logs[f"risk/env_{env}"] = risk.detach()
        return loss, logs

    def _irm_penalty(
        self,
        batch: Dict[str, Tensor],
        actions: Tensor,
        env_tensor: Tensor,
        risk_fn,
    ) -> Tensor:
        scale = torch.tensor(1.0, device=actions.device, requires_grad=True)
        penalty = torch.zeros(1, device=actions.device)
        unique_envs = torch.unique(env_tensor, sorted=True)
        for env in unique_envs:
            mask = env_tensor == env
            if not mask.any():
                continue
            scaled_actions = actions[mask] * scale
            env_pnl = compute_pnl_from_actions(batch, scaled_actions, mask=mask)
            env_risk = risk_fn(env_pnl)
            grad = torch.autograd.grad(env_risk, scale, create_graph=True)[0]
            penalty = penalty + grad.pow(2)
        return penalty


__all__ = ["IRMv1Objective"]
