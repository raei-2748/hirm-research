from __future__ import annotations

from hirm.utils.config import load_config


def test_config_merging() -> None:
    config = load_config("configs/experiments/tiny_test.yaml")
    assert config["env"]["name"] == "spy_real"
    assert config["env"]["episode_length"] == 8
    assert config["model"]["representation_dim"] == 8
    assert config["objective"]["risk_level"] == 0.95
