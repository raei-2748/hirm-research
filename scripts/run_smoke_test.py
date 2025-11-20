"""Run a minimal YAML-driven training loop for HIRM objectives.

This entrypoint is optimized for quick Colab smoke tests. It logs to both
stdout and a JSONL file under ``results/tiny_experiment`` and stores a copy
of the config that was used.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirm.data.synthetic import build_synthetic_dataset, sample_batch
from hirm.models import build_model
from hirm.objectives import build_objective
from hirm.objectives.risk import build_risk_function
from hirm.training import train_step
from hirm.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/smoke_test.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the number of optimization steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size used for synthetic data",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/tiny_experiment",
        help="Directory to store logs and metadata",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the seed from the config")
    return parser.parse_args()


def _get_attr(node, key: str, default):  # type: ignore[no-untyped-def]
    if node is None:
        return default
    if hasattr(node, key):
        return getattr(node, key)
    if isinstance(node, dict):
        return node.get(key, default)
    return default


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hirm.tiny")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    seed = int(args.seed if args.seed is not None else getattr(cfg, "seed", 0))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / f"run_{timestamp}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    logger = _setup_logger(log_path)

    feature_dim = int(getattr(getattr(cfg, "env", None), "feature_dim", 6) or 6)
    action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)

    train_cfg = getattr(cfg, "training", None)
    num_steps = args.num_steps or int(_get_attr(train_cfg, "num_steps", 50))
    batch_size = args.batch_size or int(_get_attr(train_cfg, "batch_size", 32))
    lr = float(_get_attr(train_cfg, "lr", 1e-3))
    dataset_size = int(_get_attr(train_cfg, "dataset_size", 256))
    num_envs = int(_get_attr(train_cfg, "num_envs", 2))

    dataset = build_synthetic_dataset(
        num_samples=dataset_size,
        feature_dim=feature_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        generator=generator,
    )
    dataset = {key: value.to(device) for key, value in dataset.items()}

    model = build_model(cfg.model, input_dim=feature_dim, action_dim=action_dim).to(device)
    objective = build_objective(cfg, device=device)
    risk_fn = build_risk_function(cfg.objective)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metadata = {
        "config": str(Path(args.config).resolve()),
        "seed": seed,
        "device": str(device),
        "num_steps": num_steps,
        "batch_size": batch_size,
        "lr": lr,
        "timestamp": timestamp,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if Path(args.config).exists():
        (run_dir / Path(args.config).name).write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    logger.info(
        "Starting tiny experiment with %s steps, batch_size=%s, lr=%.5f on device=%s",
        num_steps,
        batch_size,
        lr,
        device,
    )

    with log_path.open("a", encoding="utf-8") as log_handle:
        for step in range(num_steps):
            indices = torch.randint(dataset_size, (batch_size,), generator=generator).to(device)
            batch = sample_batch(dataset, indices)
            env_ids = batch["env_ids"]
            model.train()
            loss, logs = train_step(model, objective, optimizer, batch, env_ids, risk_fn)
            if step % 10 == 0 or step == num_steps - 1:
                risk_mean = logs.get("train/risk/mean", torch.tensor(float("nan")))
                logger.info(
                    "step=%04d loss=%.4f risk_mean=%.4f",
                    step,
                    loss.item(),
                    float(risk_mean),
                )
                log_record = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "risk_mean": float(risk_mean.detach().cpu()),
                }
                log_handle.write(json.dumps(log_record) + "\n")

    logger.info("Finished tiny experiment. Logs written to %s", run_dir)


if __name__ == "__main__":
    main()
