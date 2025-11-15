from __future__ import annotations

from hirm.utils.config import ConfigNode, load_config


def test_config_merging() -> None:
    config = load_config("configs/experiments/tiny_test.yaml")
    assert isinstance(config, ConfigNode)
    assert config.env.name == "synthetic_volatility_bands"
    assert config.model.input_dim == 2
    assert config.objective.lambda_invariance == 1.0
    assert config["training"]["num_steps"] == config.training.num_steps
