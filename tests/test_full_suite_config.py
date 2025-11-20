from hirm.utils.config import load_config


def test_full_suite_configs_load():
    cfg = load_config("configs/experiments/full_experiment_suite.yaml")
    assert hasattr(cfg, "env")
    assert hasattr(cfg, "training")
    assert getattr(cfg, "seeds", None) is not None
    assert getattr(cfg, "experiment_grid", None) is not None

    # Ensure dataset-specific overrides remain accessible
    assert hasattr(cfg, "real_spy")
    assert "prices_path" in cfg.real_spy["data"]
