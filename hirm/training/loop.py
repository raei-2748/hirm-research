"""Minimal training step helpers."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from hirm.objectives.base import BaseObjective
from hirm.objectives.common import compute_env_risks


def _format_env_name(env: int | str) -> str:
    if isinstance(env, str):
        return env
    return f"env_{env}"


def train_step(
    model,
    objective: BaseObjective,
    optimizer,
    batch: Dict[str, Tensor],
    env_ids: Tensor,
    risk_fn,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Run a single optimization step for ``model`` using ``objective``."""

    optimizer.zero_grad(set_to_none=True)
    env_risks_int, pnl, actions, env_tensor = compute_env_risks(model, batch, env_ids, risk_fn)
    env_risks = {_format_env_name(env): risk for env, risk in env_risks_int.items()}
    extra_state: Dict[str, Any] = {
        "objective_logs": {},
        "pnl": pnl,
        "actions": actions,
        "env_tensor": env_tensor,
        "risk_fn": risk_fn,
    }
    loss = objective.compute_loss(env_risks=env_risks, model=model, batch=batch, extra_state=extra_state)
    loss.backward()
    optimizer.step()
    logs: Dict[str, Tensor] = {"train/loss": loss.detach()}
    risks_tensor = torch.stack(list(env_risks.values()))
    logs["train/risk/mean"] = risks_tensor.mean().detach()
    for env_name, risk in env_risks.items():
        logs[f"train/env/{env_name}/risk"] = risk.detach()
    pnl_detached = pnl.detach()
    logs["train/pnl/mean"] = pnl_detached.mean()
    logs["train/pnl/cvar95"] = torch.quantile(pnl_detached, 0.05)
    if actions is not None:
        logs["train/turnover"] = actions.detach().abs().mean()
    logs.update(extra_state.get("objective_logs", {}))
    return loss, logs


__all__ = ["train_step"]
