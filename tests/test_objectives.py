"""Unit tests for objective math and gradients."""
from __future__ import annotations

from typing import Dict

import pytest

torch = pytest.importorskip("torch")

from hirm.models import build_model
from hirm.objectives import build_objective
from hirm.objectives.common import compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import ConfigNode


def _model_cfg() -> ConfigNode:
    return ConfigNode(
        {
            "name": "invariant_mlp",
            "representation": {"hidden_dims": [8], "activation": "relu"},
            "head": {"hidden_dims": [4], "activation": "relu"},
        }
    )


def _make_batch(batch_size: int, feature_dim: int, action_dim: int) -> Dict[str, torch.Tensor]:
    env_ids = torch.tensor([(idx % 2) for idx in range(batch_size)], dtype=torch.long)
    features = torch.randn(batch_size, feature_dim)
    hedge_returns = torch.randn(batch_size, action_dim)
    base_pnl = torch.randn(batch_size)
    return {
        "features": features,
        "hedge_returns": hedge_returns,
        "base_pnl": base_pnl,
        "env_ids": env_ids,
    }


def _cfg_with_objective(name: str, **kwargs) -> ConfigNode:
    payload = {"objective": {"name": name}}
    payload["objective"].update(kwargs)
    return ConfigNode(payload)


def test_erm_returns_mean_risk() -> None:
    cfg = _cfg_with_objective("erm")
    objective = build_objective(cfg, device=torch.device("cpu"))
    env_risks = {
        "env_a": torch.tensor(1.0, requires_grad=True),
        "env_b": torch.tensor(3.0, requires_grad=True),
    }
    loss = objective.compute_loss(env_risks, model=None, batch={}, extra_state=None)
    assert torch.isclose(loss, torch.tensor(2.0))


def test_groupdro_max_risk() -> None:
    cfg = _cfg_with_objective("group_dro", group_dro_smooth=False)
    objective = build_objective(cfg, device=torch.device("cpu"))
    env_risks = {
        "e1": torch.tensor(0.5, requires_grad=True),
        "e2": torch.tensor(1.5, requires_grad=True),
    }
    loss = objective.compute_loss(env_risks, model=None, batch={}, extra_state=None)
    assert torch.isclose(loss, torch.tensor(1.5))


def test_vrex_mean_plus_variance() -> None:
    cfg = _cfg_with_objective("vrex", beta=1.0)
    objective = build_objective(cfg, device=torch.device("cpu"))
    env_risks = {
        "e1": torch.tensor(1.0, requires_grad=True),
        "e2": torch.tensor(3.0, requires_grad=True),
    }
    loss = objective.compute_loss(env_risks, model=None, batch={}, extra_state=None)
    assert torch.isclose(loss, torch.tensor(3.0))


def _build_real_components(name: str):  # type: ignore[no-untyped-def]
    feature_dim = 6
    action_dim = 2
    batch = _make_batch(batch_size=8, feature_dim=feature_dim, action_dim=action_dim)
    env_ids = batch["env_ids"]
    cfg_model = _model_cfg()
    model = build_model(cfg_model, input_dim=feature_dim, action_dim=action_dim)
    cfg = _cfg_with_objective(name)
    objective = build_objective(cfg, device=torch.device("cpu"))
    risk_fn = build_risk_function(cfg.objective)
    env_risks_raw, pnl, actions, env_tensor = compute_env_risks(model, batch, env_ids, risk_fn)
    env_risks = {f"env_{env}": risk for env, risk in env_risks_raw.items()}
    extra_state = {
        "pnl": pnl,
        "actions": actions,
        "env_tensor": env_tensor,
        "risk_fn": risk_fn,
    }
    return model, objective, env_risks, batch, extra_state


def test_irmv1_and_hirm_produce_gradients() -> None:
    for name in ("irmv1", "hirm"):
        model, objective, env_risks, batch, extra = _build_real_components(name)
        loss = objective.compute_loss(env_risks, model, batch, extra_state=extra)
        assert loss.dim() == 0
        assert loss.requires_grad
        loss.backward()
        grad_norm = sum(
            param.grad.norm().item()
            for param in model.parameters()
            if param.grad is not None
        )
        assert grad_norm > 0.0
