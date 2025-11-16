"""Unified objective base classes and registry utilities."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import torch


class BaseObjective:
    """Abstract base class for all training objectives.

    Concrete subclasses are responsible for implementing ``compute_loss``
    which returns the scalar loss that should be backpropagated during the
    optimization step.  All objectives operate on pre-computed
    per-environment risks so that the surrounding training loop can log
    metrics consistently.
    """

    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        # Convenience handle to the ``objective`` section regardless of
        # whether we were given the root config or the objective sub-config.
        self.obj_cfg = getattr(cfg, "objective", cfg)

    # NOTE: subclasses must override ``compute_loss``
    def compute_loss(
        self,
        env_risks: Dict[str, torch.Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    # Helper to collect logs inside ``extra_state`` without forcing callers
    # to check for ``None``.
    def _log(self, extra_state: Optional[Dict[str, Any]], values: Dict[str, Any]) -> None:
        if extra_state is None:
            return
        logs = extra_state.setdefault("objective_logs", {})
        processed: Dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.detach()
            else:
                processed[key] = value
        logs.update(processed)


OBJECTIVE_REGISTRY: Dict[str, Type[BaseObjective]] = {}


def register_objective(name: str) -> Callable[[Type[BaseObjective]], Type[BaseObjective]]:
    """Decorator used to register objective classes by name."""

    key = name.lower()

    def decorator(cls: Type[BaseObjective]) -> Type[BaseObjective]:
        OBJECTIVE_REGISTRY[key] = cls
        return cls

    return decorator


def build_objective(cfg: Any, device: torch.device) -> BaseObjective:
    """Instantiate an objective from the configuration tree."""

    cfg_objective = getattr(cfg, "objective", cfg)
    name = getattr(cfg_objective, "name", None)
    if not name:
        raise ValueError("Objective config must provide a 'name' field")
    key = str(name).lower()
    if key not in OBJECTIVE_REGISTRY:
        available = ", ".join(sorted(OBJECTIVE_REGISTRY))
        raise ValueError(f"Unknown objective '{name}'. Available: {available}")
    objective_cls = OBJECTIVE_REGISTRY[key]
    return objective_cls(cfg, device)


__all__ = ["BaseObjective", "build_objective", "register_objective", "OBJECTIVE_REGISTRY"]
