from __future__ import annotations

import torch

from hirm.experiments.datasets import get_dataset_builder
from hirm.utils.config import load_config


def test_synthetic_hard_split_shapes() -> None:
    cfg = load_config("configs/experiments/hard_benchmark.yaml")
    builder = get_dataset_builder("synthetic_hard")
    train_ds = builder(cfg, "train", seed=0)
    test_ds = builder(cfg, "test", seed=0)

    assert train_ds.environments, "expected train environments"
    assert test_ds.environments, "expected test environments"
    for env in train_ds.environments.values():
        assert {"features", "hedge_returns", "base_pnl", "env_ids"}.issubset(env.keys())
        assert env["features"].shape[1] == cfg.env.feature_dim
        assert env["hedge_returns"].shape[1] == cfg.env.action_dim

    crisis_envs = [name for name in test_ds.environments if "crisis" in name.lower()]
    assert crisis_envs, "crisis window missing"
    for name in crisis_envs:
        data = test_ds.environments[name]
        assert torch.isfinite(data["base_pnl"]).all()
