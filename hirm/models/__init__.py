"""Model registry for policy construction."""
from __future__ import annotations

from typing import Any, Dict, Type

from hirm.models.policy import MLPPolicy, Policy

MODEL_REGISTRY: Dict[str, Type[Policy]] = {
    "mlp": MLPPolicy,
}


def build_policy(config: Dict[str, Any]) -> Policy:
    name = config.get("name", "mlp")
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown policy '{name}'")
    cls = MODEL_REGISTRY[name]
    kwargs = {k: v for k, v in config.items() if k != "name"}
    return cls(**kwargs)


__all__ = ["build_policy", "MODEL_REGISTRY", "Policy", "MLPPolicy"]
