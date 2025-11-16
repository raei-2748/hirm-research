"""Train an experiment and optionally run diagnostics in one pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hirm.data.synthetic import build_synthetic_dataset, sample_batch
from hirm.models import build_model
from hirm.objectives import build_objective
from hirm.objectives.risk import build_risk_function
from hirm.training import train_step
from hirm.utils.config import load_config
from scripts.run_diagnostics import run_diagnostics_from_config


def _get_attr(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-dir", type=str, default="outputs/experiments")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force-diagnostics", action="store_true")
    parser.add_argument("--run-diagnostics", dest="run_diagnostics", action="store_true")
    parser.add_argument("--skip-diagnostics", dest="run_diagnostics", action="store_false")
    parser.set_defaults(run_diagnostics=True)
    return parser.parse_args()


def _train_or_load_checkpoint(
    cfg,
    device: torch.device,
    results_dir: Path,
    model_name: str,
    checkpoint_override: str | None,
    num_steps_override: int | None,
    batch_size_override: int | None,
) -> str:
    if checkpoint_override is not None:
        return checkpoint_override

    seed = int(getattr(cfg, "seed", 0))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    feature_dim = int(getattr(getattr(cfg, "env", None), "feature_dim", 6) or 6)
    action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)
    train_cfg = getattr(cfg, "training", None)
    dataset_size = int(_get_attr(train_cfg, "dataset_size", 256))
    num_envs = int(_get_attr(train_cfg, "num_envs", 2))
    batch_size = batch_size_override or int(_get_attr(train_cfg, "batch_size", 32))
    num_steps = num_steps_override or int(_get_attr(train_cfg, "num_steps", 50))
    lr = float(_get_attr(train_cfg, "lr", 1e-3))

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

    for step in range(num_steps):
        indices = torch.randint(dataset_size, (batch_size,), generator=generator).to(device)
        batch = sample_batch(dataset, indices)
        env_ids = batch["env_ids"]
        model.train()
        train_step(model, objective, optimizer, batch, env_ids, risk_fn)

    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{model_name}_final.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metadata = {
        "experiment_id": cfg.experiment.name,
        "model_name": model_name,
        "seed": seed,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "checkpoint": str(checkpoint_path),
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(checkpoint_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model_name = args.model_name or getattr(cfg.model, "name", "model")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _train_or_load_checkpoint(
        cfg,
        device,
        results_dir,
        model_name,
        args.checkpoint,
        args.num_steps,
        args.batch_size,
    )

    if args.run_diagnostics:
        diag_cfg = getattr(cfg, "diagnostics", {})
        enabled = bool(getattr(diag_cfg, "enabled", True))
        if enabled or args.force_diagnostics:
            run_diagnostics_from_config(
                cfg,
                checkpoint=checkpoint_path,
                results_dir=str(results_dir / "diagnostics"),
                device=args.device,
                model_name=model_name,
                force=args.force_diagnostics,
            )
        else:
            print(
                "Diagnostics disabled via config; skipping. "
                "Use --force-diagnostics to override."
            )


if __name__ == "__main__":
    main()
