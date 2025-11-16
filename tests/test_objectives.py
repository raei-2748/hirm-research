from __future__ import annotations

from typing import Dict

import pytest

from hirm.utils.config import ConfigNode

torch = pytest.importorskip("torch")


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


def _objective_cfg(name: str) -> ConfigNode:
    payload: Dict[str, float | str] = {"name": name, "alpha": 0.7}
    if name == "groupdro":
        payload.update({"step_size": 0.1, "min_weight": 0.01})
    if name == "vrex":
        payload.update({"penalty_weight": 2.0})
    if name == "irmv1":
        payload.update({"penalty_weight": 10.0})
    if name == "hirm":
        payload.update({"lambda_invariance": 0.5})
    return ConfigNode(payload)


def _build_components(name: str):  # type: ignore[no-untyped-def]
    feature_dim = 6
    action_dim = 2
    batch = _make_batch(batch_size=8, feature_dim=feature_dim, action_dim=action_dim)
    env_ids = batch["env_ids"]
    from hirm.models import build_model
    from hirm.objectives import build_objective
    from hirm.objectives.risk import build_risk_function

    cfg_model = _model_cfg()
    model = build_model(cfg_model, input_dim=feature_dim, action_dim=action_dim)
    cfg_obj = _objective_cfg(name)
    objective = build_objective(cfg_obj)
    risk_fn = build_risk_function(cfg_obj)
    return model, objective, risk_fn, batch, env_ids


def test_objectives_produce_finite_gradients() -> None:
    objective_names = ["erm", "groupdro", "vrex", "irmv1", "hirm"]
    for name in objective_names:
        model, objective, risk_fn, batch, env_ids = _build_components(name)
        model.zero_grad()
        loss, logs = objective(model, batch, env_ids, risk_fn)
        assert loss.dim() == 0
        assert loss.requires_grad
        assert logs
        loss.backward()
        grad_norm = sum(param.grad.norm().item() for param in model.parameters() if param.grad is not None)
        assert grad_norm > 0.0


def test_irmv1_loss_is_scalar() -> None:
    model, objective, risk_fn, batch, env_ids = _build_components("irmv1")
    model.zero_grad()
    loss, _ = objective(model, batch, env_ids, risk_fn)
    assert loss.dim() == 0
    assert loss.requires_grad
    loss.backward()
    grad_norm = sum(
        param.grad.norm().item()
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_norm > 0.0


def test_hirm_invariance_depends_on_head_gradients() -> None:
    model, objective, risk_fn, batch, env_ids = _build_components("hirm")
    for param in model.representation_parameters():
        param.requires_grad_(False)
    model.zero_grad()
    loss, _ = objective(model, batch, env_ids, risk_fn)
    loss.backward()
    head_grad_norms = [
        param.grad.norm().item()
        for param in model.head_parameters()
        if param.grad is not None
    ]
    assert head_grad_norms, "Head parameters should receive gradients"
    assert any(norm > 0.0 for norm in head_grad_norms)
    rep_grads = [param.grad for param in model.representation_parameters()]
    assert all(grad is None for grad in rep_grads)
