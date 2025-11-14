"""Environment registry."""
from __future__ import annotations

from typing import Any, Dict, Type

from hirm.envs.base import Environment
from hirm.envs.spy_real_env import SpyRealEnv
from hirm.envs.synthetic_heston_env import SyntheticHestonEnv

ENV_REGISTRY: Dict[str, Type[Environment]] = {
    "spy_real": SpyRealEnv,
    "synthetic_heston": SyntheticHestonEnv,
}

RESERVED_ENV_KEYS = {"split", "episodes"}


def build_env(config: Dict[str, Any]) -> Environment:
    name = config.get("name")
    if name not in ENV_REGISTRY:
        raise KeyError(f"Unknown environment {name}")
    env_cls = ENV_REGISTRY[name]
    ignored = {"name"} | RESERVED_ENV_KEYS
    kwargs = {k: v for k, v in config.items() if k not in ignored}
    return env_cls(**kwargs)


__all__ = ["build_env", "ENV_REGISTRY"]
