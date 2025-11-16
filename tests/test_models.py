from __future__ import annotations

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


def test_invariant_policy_shapes_and_parameter_groups() -> None:
    cfg = _model_cfg()
    from hirm.models import build_model

    model = build_model(cfg, input_dim=6, action_dim=2)
    x = torch.randn(5, 6)
    output = model(x)
    assert output.shape == (5, 2)

    phi_params = list(model.representation_parameters())
    psi_params = list(model.head_parameters())
    assert phi_params and psi_params

    phi_ids = {id(param) for param in phi_params}
    psi_ids = {id(param) for param in psi_params}
    assert phi_ids.isdisjoint(psi_ids)

    combined = phi_ids | psi_ids
    all_ids = {id(param) for param in model.parameters()}
    assert combined == all_ids
