"""Model factory utilities."""
from __future__ import annotations

from typing import Any

from hirm.models.policy import InvariantPolicy


def build_model(cfg_model: Any, input_dim: int, action_dim: int) -> InvariantPolicy:
    """Instantiate the invariant policy described in the HIRM paper.

    Parameters
    ----------
    cfg_model:
        Configuration node describing the architecture.
    input_dim:
        Dimensionality of the concatenated ``(Phi, r)`` features.
    action_dim:
        Number of hedge actions to output per timestep.
    """
    if cfg_model is None:
        cfg_model = {}

    # Minimal defaults so smoke-test configs without a model block still work.
    def _ensure(cfg, key, value):
        if isinstance(cfg, dict):
            cfg.setdefault(key, value)
        elif not hasattr(cfg, key):
            setattr(cfg, key, value)

    _ensure(cfg_model, "name", "invariant_policy")
    _ensure(cfg_model, "state_factorization", "phi_only")
    _ensure(cfg_model, "representation", {"hidden_dims": [16], "activation": "relu"})
    _ensure(cfg_model, "head", {"hidden_dims": [8], "activation": "relu"})
    _ensure(cfg_model, "r_network", {})
    if getattr(cfg_model, "state_factorization", "phi_only") == "phi_only":
        if isinstance(cfg_model, dict):
            cfg_model["r_network"].setdefault("hidden_dims", [])
        else:
            cfg_model.r_network.hidden_dims = []

    name = getattr(cfg_model, "name", "invariant_policy")
    if name not in {"mlp", "mlp_small", "invariant_policy", "invariant_mlp"}:
        raise ValueError(f"Unsupported model name '{name}'")
    return InvariantPolicy(cfg_model, input_dim=input_dim, action_dim=action_dim)


__all__ = ["build_model", "InvariantPolicy"]
