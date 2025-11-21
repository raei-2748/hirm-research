from __future__ import annotations

import pytest

from hirm.utils.config import load_config


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/experiments/baseline_benchmark.yaml",
        "configs/experiments/hard_benchmark.yaml",
    ],
)
def test_experiment_configs_parse_lists(config_path: str) -> None:
    cfg = load_config(config_path)
    assert isinstance(cfg.methods, list)
    assert isinstance(cfg.datasets, list)
    assert isinstance(cfg.seeds, list)
    assert isinstance(cfg.diagnostics.isi.probe_layers, list)
    assert all(isinstance(x, float) for x in cfg.diagnostics.isi.alpha_components)
    if hasattr(cfg, "real_spy"):
        crisis = getattr(cfg.real_spy.data, "crisis_windows", {})
        if isinstance(crisis, dict):
            for window in crisis.values():
                assert isinstance(window, list)
