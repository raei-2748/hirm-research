"""Synthetic demo comparing objectives on a toy dataset."""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch

from hirm.models import build_model
from hirm.objectives import build_objective
from hirm.objectives.common import compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import ConfigNode

Batch = Dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40, help="Number of updates per objective")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for the toy data")
    parser.add_argument("--dataset-size", type=int, default=512, help="Synthetic dataset size")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
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


def _objective_cfg(name: str, overrides: Dict[str, float] | None = None) -> ConfigNode:
    payload = {"objective": {"name": name, "alpha": 0.95}}
    if overrides:
        payload["objective"].update(overrides)
    return ConfigNode(payload)


OBJECTIVE_SWEEP: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("erm", {}),
    ("group_dro", {"group_dro_smooth": False}),
    ("vrex", {"beta": 1.0}),
    ("irmv1", {"lambda_irm": 10.0}),
    ("hirm", {"lambda_hirm": 2.0}),
)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    feature_dim = 6
    action_dim = 2
    dataset = build_synthetic_dataset(
        num_samples=args.dataset_size,
        feature_dim=feature_dim,
        action_dim=action_dim,
        num_envs=2,
        generator=generator,
    )
    dataset = {key: value.to(device) for key, value in dataset.items()}
    indices = torch.arange(args.dataset_size, device=device)

    for name, overrides in OBJECTIVE_SWEEP:
        cfg = _objective_cfg(name, overrides)
        model_cfg = ConfigNode(
            {
                "name": "invariant_mlp",
                "representation": {"hidden_dims": [16], "activation": "relu"},
                "head": {"hidden_dims": [8], "activation": "relu"},
            }
        )
        model = build_model(model_cfg, input_dim=feature_dim, action_dim=action_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        objective = build_objective(cfg, device=device)
        risk_fn = build_risk_function(cfg.objective)

        print(f"\n=== Objective: {name} ===")
        for step in range(args.steps):
            step_indices = indices[torch.randint(args.dataset_size, (args.batch_size,), generator=generator)]
            batch = sample_batch(dataset, step_indices)
            env_ids = batch["env_ids"]
            optimizer.zero_grad(set_to_none=True)
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
            if step % 10 == 0 or step == args.steps - 1:
                mean_risk = objective.get_latest_logs().get(
                    "train/objective/mean_risk", torch.tensor(float("nan"))
                )
                print(
                    f"step={step:03d} loss={loss.item():.4f} mean_risk={float(mean_risk):.4f}"
                )

        final_logs = objective.get_latest_logs()
        for key, value in final_logs.items():
            print(f"  {key}: {float(value):.4f}")


if __name__ == "__main__":
    main()
