from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from hirm.engine import _compute_diagnostics
from hirm.experiments.datasets import ExperimentDataset, get_dataset_builder
from hirm.experiments.methods import get_method_builder
from hirm.experiments.registry import ExperimentRunConfig
from hirm.utils.config import ConfigNode, load_config


def _shrink_training(cfg: ConfigNode) -> ConfigNode:
    cfg = ConfigNode({**cfg.to_dict()})
    cfg.training.max_epochs = 1
    cfg.training.batch_size = 8
    cfg.training.early_stop_patience = 1
    return cfg


def _build_small_dataset(cfg: ConfigNode) -> tuple[ExperimentDataset, ExperimentDataset]:
    builder = get_dataset_builder("synthetic_heston")
    train_ds = builder(cfg, "train", seed=0)
    val_ds = builder(cfg, "val", seed=0)
    return train_ds, val_ds


def test_registered_methods_smoke(tmp_path: Path) -> None:
    cfg = _shrink_training(load_config("configs/experiments/baseline_benchmark.yaml"))
    train_ds, val_ds = _build_small_dataset(cfg)
    device = torch.device("cpu")

    for method in ["erm", "irm", "groupdro", "vrex", "hirm", "risk_parity"]:
        run_cfg = ExperimentRunConfig(
            dataset="synthetic_heston",
            method=method,
            seed=0,
            config=cfg,
            device=device,
        )
        trainer = get_method_builder(method)(run_cfg)
        trainer.set_datasets(train=train_ds, val=val_ds)
        logs = trainer.train()
        assert logs, f"logs missing for {method}"
        ckpt_path = tmp_path / f"{method}.pt"
        trainer.save(str(ckpt_path))
        assert ckpt_path.exists()
        diagnostics = _compute_diagnostics(trainer, train_ds, val_ds, run_cfg)
        assert isinstance(diagnostics, dict)
        assert "metrics/er" in diagnostics
