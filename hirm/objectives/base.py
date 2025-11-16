"""Base objective abstractions and registry utilities."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
from torch import Tensor


class BaseObjective:
    """Shared interface for all Phase-5 objectives."""

    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.objective_cfg = getattr(cfg, "objective", cfg)
        self._latest_logs: Dict[str, Tensor] = {}

    def compute_loss(
        self,
        env_risks: Dict[str, Tensor],
        model,
        batch: Dict[str, Any],
        extra_state: Dict[str, Any] | None = None,
    ) -> Tensor:
        """Return the scalar loss for a batch.

        Sub-classes must override this method.  ``env_risks`` contains one
        differentiable scalar per environment so that the objective can compose
        mean/variance penalties.  ``extra_state`` may include auxiliary tensors
        such as the per-sample PnL or cached environment ids.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Logging helpers
    def log_metrics(self, metrics: Mapping[str, Tensor | float | int]) -> None:
        """Record the latest objective-specific metrics for logging."""

        logs: Dict[str, Tensor] = {}
        for key, value in metrics.items():
            if isinstance(value, Tensor):
                logs[key] = value.detach()
            else:
                logs[key] = torch.as_tensor(value, device=self.device)
        self._latest_logs = logs

    def clear_logs(self) -> None:
        self._latest_logs = {}

    def get_latest_logs(self) -> Dict[str, Tensor]:
        return dict(self._latest_logs)

    # ------------------------------------------------------------------
    def get_head_parameters(self, model) -> list[Tensor]:
        """Return the model head parameters, raising if missing."""

        if not hasattr(model, "head_parameters"):
            raise ValueError("Model must implement 'head_parameters'")
        params = list(model.head_parameters())  # type: ignore[attr-defined]
        if not params:
            raise ValueError("head_parameters returned an empty iterable")
        return params


OBJECTIVE_REGISTRY: Dict[str, type[BaseObjective]] = {}


def register_objective(name: str):
    """Decorator used by objective implementations to register themselves."""

    def decorator(cls: type[BaseObjective]) -> type[BaseObjective]:
        key = name.lower()
        if key in OBJECTIVE_REGISTRY and OBJECTIVE_REGISTRY[key] is not cls:
            raise ValueError(f"Objective '{name}' already registered")
        OBJECTIVE_REGISTRY[key] = cls
        return cls

    return decorator


def build_objective(cfg: Any, device: torch.device) -> BaseObjective:
    """Instantiate an objective from the config registry."""

    cfg_obj = getattr(cfg, "objective", cfg)
    name = getattr(cfg_obj, "name", None)
    if not name:
        raise ValueError("Objective config must include a 'name'")
    key = str(name).lower()
    if key not in OBJECTIVE_REGISTRY:
        raise ValueError(f"Unknown objective '{name}'")
    cls = OBJECTIVE_REGISTRY[key]
    return cls(cfg, device)


__all__ = [
    "BaseObjective",
    "build_objective",
    "register_objective",
]

