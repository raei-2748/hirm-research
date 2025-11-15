from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from hirm.utils.config import load_config
from hirm.utils.logging import make_logger
from hirm.utils.math import cvar, flatten_gradients, rolling_window, safe_div, safe_log
from hirm.utils.seed import set_seed
from hirm.utils.serialization import (
    load_checkpoint,
    load_model,
    load_optimizer,
    save_checkpoint,
    save_model,
    save_optimizer,
)

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


class DummyModel:
    def __init__(self, weight: float = 0.0) -> None:
        self.weight = weight

    def state_dict(self) -> dict[str, float]:
        return {"weight": self.weight}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.weight = state["weight"]


class DummyOptimizer:
    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def state_dict(self) -> dict[str, float]:
        return {"lr": self.lr}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.lr = state["lr"]


class DummyParam:
    def __init__(self, grad) -> None:  # type: ignore[no-untyped-def]
        self.grad = grad


def test_phase1_stack(tmp_path: Path) -> None:
    cfg = load_config("configs/experiments/tiny_test.yaml")
    assert cfg.experiment.name == "tiny_test"
    assert cfg.training.num_steps == cfg["training"]["num_steps"]

    set_seed(123)
    first = random.random()
    set_seed(123)
    assert random.random() == first

    if np is not None:
        set_seed(42)
        first_np = np.random.rand()
        set_seed(42)
        assert np.random.rand() == first_np

    if torch is not None:
        set_seed(11)
        first_torch = float(torch.rand(1))
        set_seed(11)
        assert float(torch.rand(1)) == pytest.approx(first_torch)

    logger = make_logger(cfg.experiment.name, cfg, base_dir=tmp_path)
    logger.info("initializing")
    logger.log({"step": 1, "loss": 0.25})
    logger.close()
    run_dirs = list(tmp_path.iterdir())
    assert run_dirs, "logger should create a run directory"
    run_dir = run_dirs[0]
    assert (run_dir / "logs.txt").exists()
    metrics_path = run_dir / "metrics.jsonl"
    contents = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents and json.loads(contents[0])["loss"] == 0.25
    config_path = run_dir / "config_resolved.yaml"
    assert config_path.exists()

    model_path = run_dir / "model.pt"
    save_model(DummyModel(0.5), model_path)
    loaded_model = load_model(DummyModel, model_path)
    assert isinstance(loaded_model, DummyModel)
    assert loaded_model.weight == pytest.approx(0.5)

    opt_path = run_dir / "optimizer.pt"
    save_optimizer(DummyOptimizer(0.01), opt_path)
    loaded_opt = load_optimizer(DummyOptimizer, opt_path)
    assert loaded_opt.lr == pytest.approx(0.01)

    checkpoint_path = run_dir / "checkpoint.bin"
    save_checkpoint({"step": 5}, checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["step"] == 5

    windows = rolling_window([1.0, 2.0, 3.0, 4.0], window=2)
    assert windows == [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    assert safe_log(1e-8) > -math.inf
    assert safe_div(1.0, 0.0) == pytest.approx(1.0 / 1e-12, rel=1e-6)

    params = [DummyParam([1.0, 2.0]), DummyParam([[3.0], [4.0]])]
    grads = flatten_gradients(params)
    assert grads == [1.0, 2.0, 3.0, 4.0]

    cvar_value = cvar([-1.0, -2.0, -3.0, 0.0], alpha=0.75)
    assert cvar_value <= -1.0
