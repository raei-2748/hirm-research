"""Synthetic dataset helpers used by demo scripts and tests."""
from __future__ import annotations

from typing import Dict

import torch

Batch = Dict[str, torch.Tensor]


def build_synthetic_dataset(
    num_samples: int,
    feature_dim: int,
    action_dim: int,
    num_envs: int,
    generator: torch.Generator,
) -> Batch:
    env_ids = torch.arange(num_samples) % max(2, num_envs)
    perm = torch.randperm(num_samples, generator=generator)
    env_ids = env_ids[perm]
    features = torch.randn(num_samples, feature_dim, generator=generator)
    hedge_returns = 0.05 * torch.randn(num_samples, action_dim, generator=generator)
    env_effect = torch.linspace(-0.2, 0.2, steps=max(2, num_envs))
    base_signal = env_effect[env_ids]
    base_noise = 0.01 * torch.randn(num_samples, generator=generator)
    base_pnl = base_signal + base_noise
    return {
        "features": features,
        "hedge_returns": hedge_returns,
        "base_pnl": base_pnl,
        "env_ids": env_ids.long(),
    }


def sample_batch(dataset: Batch, indices: torch.Tensor) -> Batch:
    return {key: value[indices] for key, value in dataset.items()}


__all__ = ["build_synthetic_dataset", "sample_batch"]
