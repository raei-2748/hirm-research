"""Environment registry."""
from __future__ import annotations

from typing import Any, Dict, Type

from hirm.envs.base import Environment
from hirm.envs.real import RealVolatilityBandEnv
from hirm.envs.synthetic import SyntheticVolatilityBandEnv

ENV_REGISTRY: Dict[str, Type[Environment]] = {
    "real_volatility_bands": RealVolatilityBandEnv,
    "synthetic_volatility_bands": SyntheticVolatilityBandEnv,
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
