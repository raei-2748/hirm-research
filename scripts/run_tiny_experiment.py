"""Run a minimal YAML-driven training loop for HIRM objectives."""
from __future__ import annotations

import argparse
from typing import Dict

import torch

from hirm.models import build_model
from hirm.objectives import build_objective
from hirm.objectives.common import compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import load_config

Batch = Dict[str, torch.Tensor]


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


def build_synthetic_dataset(
    num_samples: int,
    feature_dim: int,
    action_dim: int,
    num_envs: int,
    generator: torch.Generator,
) -> Batch:
    env_ids = torch.arange(num_samples) % max(2, num_envs)
    perm = torch.randperm(num_samples, generator=generator)
    env_ids = env_ids[perm]
    features = torch.randn(num_samples, feature_dim, generator=generator)
    hedge_returns = 0.05 * torch.randn(num_samples, action_dim, generator=generator)
    env_effect = torch.linspace(-0.2, 0.2, steps=max(2, num_envs))
    base_signal = env_effect[env_ids]
    base_noise = 0.01 * torch.randn(num_samples, generator=generator)
    base_pnl = base_signal + base_noise
    return {
        "features": features,
        "hedge_returns": hedge_returns,
        "base_pnl": base_pnl,
        "env_ids": env_ids.long(),
    }


def sample_batch(dataset: Batch, indices: torch.Tensor) -> Batch:
    return {key: value[indices] for key, value in dataset.items()}


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
        indices = torch.randint(dataset_size, (batch_size,), generator=generator)
        batch = sample_batch(dataset, indices)
        env_ids = batch["env_ids"]
        optimizer.zero_grad(set_to_none=True)
        model.train()
        env_risks_raw, pnl, actions, env_tensor = compute_env_risks(
            model, batch, env_ids, risk_fn
        )
        env_risks = {f"env_{env}": risk for env, risk in env_risks_raw.items()}
        extra_state = {
            "pnl": pnl,
            "actions": actions,
            "env_tensor": env_tensor,
            "risk_fn": risk_fn,
        }
        loss = objective.compute_loss(env_risks, model, batch, extra_state=extra_state)
        loss.backward()
        optimizer.step()
        logs = {
            "train/loss": loss.detach(),
            "train/pnl/mean": pnl.mean().detach(),
        }
        for env_name, risk in env_risks.items():
            logs[f"train/env/{env_name}/risk"] = risk.detach()
        if "hirm_alignment" in extra_state:
            logs["train/hirm/alignment"] = extra_state["hirm_alignment"]
        logs.update(objective.get_latest_logs())
        if step % 10 == 0 or step == num_steps - 1:
            mean_risk = logs.get("train/objective/mean_risk", torch.tensor(float("nan")))
            print(
                f"step={step:04d} loss={loss.item():.4f} risk_mean={float(mean_risk):.4f}"
            )

    print("Finished tiny experiment.")


if __name__ == "__main__":
    main()
