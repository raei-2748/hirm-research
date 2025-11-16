"""Model factory utilities."""
from __future__ import annotations

from typing import Any

from hirm.models.policy import InvariantPolicy


def build_model(cfg_model: Any, input_dim: int, action_dim: int) -> InvariantPolicy:
    """Instantiate the configured policy model.

    Parameters
    ----------
    cfg_model:
        Configuration node describing the architecture.
    input_dim:
        Dimensionality of the concatenated ``(Phi, r)`` features.
    action_dim:
        Number of hedge actions to output per timestep.
    """

    name = getattr(cfg_model, "name", "invariant_policy")
    if name not in {"mlp", "mlp_small", "invariant_policy", "invariant_mlp"}:
        raise ValueError(f"Unsupported model name '{name}'")
    return InvariantPolicy(cfg_model, input_dim=input_dim, action_dim=action_dim)


__all__ = ["build_model", "InvariantPolicy"]
