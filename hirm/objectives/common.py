"""Shared helpers for objective computation."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor


def _to_tensor(value: Tensor | Iterable[float]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float32)


def concat_state(batch: Dict[str, Tensor | Iterable[float]]) -> Tensor:
    """Concatenate ``Phi`` and ``r`` features into a single tensor."""

    for key in ("features", "x", "state"):
        if key in batch:
            return _to_tensor(batch[key])
    if "phi" not in batch or "r" not in batch:
        raise KeyError("Batch must include either 'features' or both 'phi' and 'r'")
    phi = _to_tensor(batch["phi"])
    r = _to_tensor(batch["r"])
    if phi.shape[0] != r.shape[0]:
        raise ValueError("Phi and r must have matching batch dimensions")
    return torch.cat([phi, r], dim=-1)


def compute_actions_and_pnl(
    policy,
    batch: Dict[str, Tensor | Iterable[float]],
    env_ids: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run the policy and compute hedged PnL for ``batch``."""

    features = concat_state(batch)
    actions = policy(features, env_ids=env_ids)
    pnl = _compute_pnl(batch, actions)
    return actions, pnl


def compute_pnl_from_actions(
    batch: Dict[str, Tensor | Iterable[float]],
    actions: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute hedged PnL for ``actions`` without re-running the policy."""

    if mask is not None:
        batch = _slice_batch(batch, mask)
    return _compute_pnl(batch, actions)


def _slice_batch(
    batch: Dict[str, Tensor | Iterable[float]], mask: Tensor
) -> Dict[str, Tensor | Iterable[float]]:
    sliced: Dict[str, Tensor | Iterable[float]] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            sliced[key] = value[mask]
        else:
            sliced[key] = value
    return sliced


def _compute_pnl(batch: Dict[str, Tensor | Iterable[float]], actions: Tensor) -> Tensor:
    if "pnl" in batch:
        return _to_tensor(batch["pnl"])
    if "hedge_returns" not in batch:
        raise KeyError("Batch must provide 'hedge_returns' when 'pnl' is absent")
    hedge = _to_tensor(batch["hedge_returns"])
    if hedge.shape[0] != actions.shape[0]:
        raise ValueError("hedge_returns must align with the batch dimension")
    if hedge.shape[1] != actions.shape[1]:
        raise ValueError("hedge_returns must match action_dim")
    base = _to_tensor(batch.get("base_pnl", torch.zeros(actions.shape[0], device=actions.device)))
    if base.shape[0] != actions.shape[0]:
        raise ValueError("base_pnl must align with the batch dimension")
    # ``base_pnl`` and the hedge returns correspond to the per-timestep PnL
    # observations currently available.  They approximate the per-episode
    # PnL random variables used in the paper's risk definitions.
    return base + torch.sum(actions * hedge, dim=-1)


def compute_env_risks(
    policy,
    batch: Dict[str, Tensor | Iterable[float]],
    env_ids: Tensor,
    risk_fn,
) -> Tuple[Dict[int, Tensor], Tensor, Tensor, Tensor]:
    """Return per-environment risks, PnL, policy actions, and env ids.

    In the paper the risk functional is applied to per-episode PnL.  The
    current prototype often feeds per-timestep or flattened PnL vectors;
    the same coherent risk functional ``risk_fn`` is used regardless of
    granularity until full episode rollouts are available.
    """

    env_tensor = _to_tensor(env_ids).long()
    if env_tensor.ndim != 1:
        raise ValueError("env_ids must be a 1D tensor")
    actions, pnl = compute_actions_and_pnl(policy, batch, env_tensor)
    unique_envs = torch.unique(env_tensor, sorted=True)
    risks: Dict[int, Tensor] = {}
    for env in unique_envs.tolist():
        mask = env_tensor == env
        env_pnl = pnl[mask]
        if env_pnl.numel() == 0:
            continue
        risks[int(env)] = risk_fn(env_pnl)
    if not risks:
        raise ValueError("At least one environment must be present in the batch")
    return risks, pnl, actions, env_tensor


def flatten_head_gradients(grads: Iterable[Tensor | None]) -> Tensor:
    """Flatten gradient tensors, replacing ``None`` entries with zeros."""

    flat: list[Tensor] = []
    device = None
    for grad in grads:
        if grad is None:
            continue
        if device is None:
            device = grad.device
        flat.append(grad.reshape(-1))
    if not flat:
        return torch.zeros(1, device=device or torch.device("cpu"))
    return torch.cat(flat)


__all__ = [
    "compute_actions_and_pnl",
    "compute_env_risks",
    "compute_pnl_from_actions",
    "concat_state",
    "flatten_head_gradients",
]
