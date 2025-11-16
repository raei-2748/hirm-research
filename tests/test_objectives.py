from __future__ import annotations

from typing import Dict

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from hirm.objectives import build_objective
from hirm.utils.config import ConfigNode


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.head.weight, 1.0)

    def head_parameters(self):  # pragma: no cover - passthrough wrapper
        return self.head.parameters()


def _objective_cfg(name: str, **overrides) -> ConfigNode:
    payload = {"name": name}
    payload.update(overrides)
    return ConfigNode({"objective": payload})


def _env_risks(model: DummyModel) -> Dict[str, torch.Tensor]:
    weight = next(model.head.parameters()).reshape(-1)[0]
    return {
        "env_a": weight * torch.tensor(1.0),
        "env_b": weight * torch.tensor(3.0),
    }


def test_erm_matches_mean_risk() -> None:
    model = DummyModel()
    env_risks = _env_risks(model)
    cfg = _objective_cfg("erm")
    objective = build_objective(cfg, device=torch.device("cpu"))
    loss = objective.compute_loss(env_risks, model, batch={}, extra_state={})
    assert torch.isclose(loss, torch.tensor(2.0))


def test_groupdro_uses_worst_environment() -> None:
    model = DummyModel()
    env_risks = _env_risks(model)
    cfg = _objective_cfg("group_dro", group_dro_smooth=False)
    objective = build_objective(cfg, device=torch.device("cpu"))
    loss = objective.compute_loss(env_risks, model, batch={}, extra_state={})
    assert torch.isclose(loss, torch.tensor(3.0))


def test_vrex_penalty_matches_variance() -> None:
    model = DummyModel()
    env_risks = _env_risks(model)
    cfg = _objective_cfg("vrex", beta=1.0)
    objective = build_objective(cfg, device=torch.device("cpu"))
    loss = objective.compute_loss(env_risks, model, batch={}, extra_state={})
    assert torch.isclose(loss, torch.tensor(3.0))
    cfg_erm = _objective_cfg("vrex", beta=0.0)
    objective_erm = build_objective(cfg_erm, device=torch.device("cpu"))
    loss_erm = objective_erm.compute_loss(env_risks, model, batch={}, extra_state={})
    assert torch.isclose(loss_erm, torch.tensor(2.0))


@pytest.mark.parametrize("name", ["irmv1", "hirm"])
def test_gradient_based_objectives_backprop(name: str) -> None:
    model = DummyModel()
    env_risks = _env_risks(model)
    cfg = _objective_cfg(name)
    objective = build_objective(cfg, device=torch.device("cpu"))
    extra_state: Dict[str, torch.Tensor] = {}
    loss = objective.compute_loss(env_risks, model, batch={}, extra_state=extra_state)
    loss.backward()
    grads = [param.grad for param in model.head.parameters() if param.grad is not None]
    assert grads, "Head parameters must receive gradients"
    assert any(torch.any(g != 0) for g in grads)
