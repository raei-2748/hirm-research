"""Ablation registry and configuration helpers for phase 8."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from hirm.utils.config import ConfigNode, to_plain_dict


@dataclass
class AblationConfig:
    name: str
    method: str
    invariance_mode: str = "head_only"
    state_factorization: str = "phi_r"
    objective_type: str = "cvar"
    env_label_scheme: str = "vol_bands"
    extra_hparams: Dict[str, Any] = field(default_factory=dict)

    def to_config_node(self) -> ConfigNode:
        return ConfigNode(
            {
                "name": self.name,
                "method": self.method,
                "invariance_mode": self.invariance_mode,
                "state_factorization": self.state_factorization,
                "objective_type": self.objective_type,
                "env_label_scheme": self.env_label_scheme,
                "extra_hparams": to_plain_dict(self.extra_hparams),
            }
        )


_ABLATION_REGISTRY: Dict[str, AblationConfig] = {}


def register_ablation(config: AblationConfig) -> None:
    key = config.name
    _ABLATION_REGISTRY[key] = config


# Presets required by the specification.
register_ablation(
    AblationConfig(
        name="hirm_full",
        method="hirm",
        invariance_mode="head_only",
        state_factorization="phi_r",
        objective_type="cvar",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="hirm_no_hgca",
        method="hirm",
        invariance_mode="none",
        state_factorization="phi_r",
        objective_type="cvar",
        env_label_scheme="vol_bands",
        extra_hparams={"lambda_invariance": 0.0},
    )
)
register_ablation(
    AblationConfig(
        name="hirm_full_irm",
        method="hirm",
        invariance_mode="full_irm",
        state_factorization="phi_r",
        objective_type="cvar",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="hirm_env_specific_heads",
        method="hirm",
        invariance_mode="env_specific_heads",
        state_factorization="phi_r",
        objective_type="cvar",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="hirm_no_split",
        method="hirm",
        invariance_mode="head_only",
        state_factorization="no_split",
        objective_type="cvar",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="hirm_phi_only",
        method="hirm",
        invariance_mode="head_only",
        state_factorization="phi_only",
        objective_type="cvar",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="hirm_r_only",
        method="hirm",
        invariance_mode="head_only",
        state_factorization="r_only",
        objective_type="cvar",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="hirm_random_env_labels",
        method="hirm",
        invariance_mode="head_only",
        state_factorization="phi_r",
        objective_type="cvar",
        env_label_scheme="random",
    )
)
register_ablation(
    AblationConfig(
        name="hirm_coarse_env_bands",
        method="hirm",
        invariance_mode="head_only",
        state_factorization="phi_r",
        objective_type="cvar",
        env_label_scheme="coarse_bands",
    )
)
register_ablation(
    AblationConfig(
        name="erm_baseline",
        method="erm",
        invariance_mode="none",
        state_factorization="no_split",
        objective_type="mean",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="groupdro_baseline",
        method="groupdro",
        invariance_mode="none",
        state_factorization="no_split",
        objective_type="cvar",
        env_label_scheme="vol_bands",
    )
)
register_ablation(
    AblationConfig(
        name="vrex_baseline",
        method="vrex",
        invariance_mode="none",
        state_factorization="no_split",
        objective_type="variance_like",
        env_label_scheme="vol_bands",
    )
)


def get_ablation_config(name: str | None) -> Optional[AblationConfig]:
    if name is None:
        return None
    if name not in _ABLATION_REGISTRY:
        available = ", ".join(sorted(_ABLATION_REGISTRY))
        raise KeyError(f"Unknown ablation '{name}'. Available: {available}")
    return _ABLATION_REGISTRY[name]


def list_ablations() -> list[str]:
    return sorted(_ABLATION_REGISTRY)


def apply_ablation_to_config(base_cfg: ConfigNode, ablation: AblationConfig) -> ConfigNode:
    """Return a ConfigNode with the ablation choices baked in."""

    cfg = ConfigNode(to_plain_dict(base_cfg))
    cfg.ablation = ablation.to_config_node()

    # Method/objective wiring.
    if "objective" not in cfg:
        cfg["objective"] = ConfigNode({})
    cfg.objective.name = ablation.method
    cfg.objective.invariance_mode = ablation.invariance_mode
    if ablation.objective_type:
        cfg.objective.risk_name = ablation.objective_type
    if ablation.extra_hparams:
        for key, value in ablation.extra_hparams.items():
            setattr(cfg.objective, key, value)

    # Model wiring.
    if "model" not in cfg:
        cfg["model"] = ConfigNode({})
    cfg.model.state_factorization = ablation.state_factorization
    cfg.model.invariance_mode = ablation.invariance_mode

    # Dataset/env wiring.
    if "env" not in cfg:
        cfg["env"] = ConfigNode({})
    cfg.env.label_scheme = ablation.env_label_scheme

    return cfg


__all__ = [
    "AblationConfig",
    "get_ablation_config",
    "list_ablations",
    "apply_ablation_to_config",
]
