"""Run a tiny synthetic demo highlighting the different objectives."""
from __future__ import annotations

import argparse
from typing import List

import torch

from hirm.data.synthetic import build_synthetic_dataset, sample_batch
from hirm.models import build_model
from hirm.objectives import build_objective
from hirm.objectives.risk import build_risk_function
from hirm.training import train_step
from hirm.utils.config import load_config


def _default_configs() -> List[str]:
    return [
        "configs/experiments/synth_erm.yaml",
        "configs/experiments/synth_groupdro.yaml",
        "configs/experiments/synth_vrex.yaml",
        "configs/experiments/synth_irm.yaml",
        "configs/experiments/synth_hirm.yaml",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=_default_configs(),
        help="Experiment config files to evaluate",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--num-steps", type=int, default=None, help="Override number of steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def _get_training_param(node, key: str, default: int | float) -> int | float:
    if node is None:
        return default
    if hasattr(node, key):
        return getattr(node, key)
    if isinstance(node, dict):
        return node.get(key, default)
    return default


def run_single_demo(cfg_path: str, device: torch.device, num_steps: int | None, batch_size: int | None) -> None:
    cfg = load_config(cfg_path)
    training = getattr(cfg, "training", None)
    steps = int(num_steps or _get_training_param(training, "num_steps", 40))
    batch = int(batch_size or _get_training_param(training, "batch_size", 32))
    dataset_size = int(_get_training_param(training, "dataset_size", 256))
    num_envs = int(_get_training_param(training, "num_envs", 2))
    lr = float(_get_training_param(training, "lr", 1e-3))
    feature_dim = int(getattr(getattr(cfg, "env", None), "feature_dim", 6) or 6)
    action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)

    generator = torch.Generator().manual_seed(int(getattr(cfg, "seed", 0)))
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

    final_logs = {}
    for step in range(steps):
        indices = torch.randint(dataset_size, (batch,), generator=generator).to(device)
        batch_data = sample_batch(dataset, indices)
        env_ids = batch_data["env_ids"]
        model.train()
        _, logs = train_step(model, objective, optimizer, batch_data, env_ids, risk_fn)
        final_logs = logs

    env_metrics = {key: float(value) for key, value in final_logs.items() if key.startswith("train/env/")}
    alignment = final_logs.get("train/hirm/alignment")
    penalty_keys = [key for key in final_logs if key.startswith("train/objective/")]
    print(f"=== {cfg.experiment.name} ({cfg_path}) ===")
    print(f"  final loss: {float(final_logs.get('train/loss', torch.tensor(float('nan')))):.4f}")
    for key, value in sorted(env_metrics.items()):
        print(f"  {key}: {value:.4f}")
    for key in sorted(penalty_keys):
        value = final_logs[key]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                display = f"{float(value):.4f}"
            else:
                display = value.detach().cpu().tolist()
        else:
            display = value
        print(f"  {key}: {display}")
    if alignment is not None:
        print(f"  train/hirm/alignment: {float(alignment):.4f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for cfg_path in args.configs:
        run_single_demo(cfg_path, device=device, num_steps=args.num_steps, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
