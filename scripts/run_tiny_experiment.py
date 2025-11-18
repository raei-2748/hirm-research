"""Run a minimal YAML-driven training loop for HIRM objectives."""
from __future__ import annotations

import argparse
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

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
        default="configs/experiments/tiny_test.yaml",
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
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    return parser.parse_args()
def _get_attr(node, key: str, default):  # type: ignore[no-untyped-def]
    if node is None:
        return default
    if hasattr(node, key):
        return getattr(node, key)
    if isinstance(node, dict):
        return node.get(key, default)
    return default


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    seed = int(getattr(cfg, "seed", 0))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

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

    print(
        f"Starting tiny experiment with {num_steps} steps, batch_size={batch_size}, lr={lr}"
    )
    for step in range(num_steps):
        indices = torch.randint(dataset_size, (batch_size,), generator=generator).to(device)
        batch = sample_batch(dataset, indices)
        env_ids = batch["env_ids"]
        model.train()
        loss, logs = train_step(model, objective, optimizer, batch, env_ids, risk_fn)
        if step % 10 == 0 or step == num_steps - 1:
            risk_mean = logs.get("train/risk/mean", torch.tensor(float("nan")))
            print(f"step={step:04d} loss={loss.item():.4f} risk_mean={float(risk_mean):.4f}")

    print("Finished tiny experiment.")


if __name__ == "__main__":
    main()
