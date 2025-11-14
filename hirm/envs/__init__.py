"""Environment registry."""
from __future__ import annotations

from typing import Any, Dict, Type

from hirm.envs.base import BaseEnv
from hirm.envs.spy_real_env import SpyRealEnv
from hirm.envs.synthetic_heston_env import SyntheticHestonEnv

ENV_REGISTRY: Dict[str, Type[BaseEnv]] = {
    "spy_real": SpyRealEnv,
    "synthetic_heston": SyntheticHestonEnv,
}


def build_env(config: Dict[str, Any]) -> BaseEnv:
    name = config.get("name")
    if name not in ENV_REGISTRY:
        raise KeyError(f"Unknown environment {name}")
    env_cls = ENV_REGISTRY[name]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    return env_cls(**kwargs)


__all__ = ["build_env", "ENV_REGISTRY"]
