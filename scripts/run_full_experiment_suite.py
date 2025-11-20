"""Full experiment suite runner.

This entrypoint mirrors the publication grid: datasets × methods × seeds with
standard diagnostics and a reproducible results layout.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from hirm.diagnostics import compute_all_diagnostics, compute_crisis_cvar
from hirm.diagnostics.invariance_helpers import collect_invariance_signals
from hirm.experiments.datasets import ExperimentDataset, get_dataset_builder
from hirm.experiments.methods import get_method_builder
from hirm.experiments.registry import ExperimentRunConfig
from hirm.objectives.common import concat_state, compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import ConfigNode, load_config, to_plain_dict


METHOD_ALIASES = {
    "erm_baseline": "erm",
    "groupdro_baseline": "groupdro",
    "vrex_baseline": "vrex",
    "irm_baseline": "irm",
    "hirm_full": "hirm",
    "hirm_full_irm": "hirm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default=None, help="Comma separated method names")
    parser.add_argument("--datasets", type=str, default=None, help="Comma separated dataset names")
    parser.add_argument("--seeds", type=str, default=None, help="Comma separated integer seeds")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers (unused for synthetic data)")
    parser.add_argument("--force_rerun", type=int, default=0, help="Force rerun even if results exist")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string, e.g. cpu or cuda:0")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/full_experiment_suite",
        help="Root directory to store checkpoints, logs, and diagnostics",
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Run a reduced grid for smoke testing (single seed, short training)",
    )
    return parser.parse_args()


def _set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic))
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _combine_environments(dataset: ExperimentDataset, device: torch.device):
    batches = []
    env_ids = []
    for _, data in dataset.environments.items():
        batches.append({k: v.to(device) for k, v in data.items()})
        env_ids.append(data["env_ids"].to(device))
    merged: dict[str, torch.Tensor] = {}
    for key in batches[0].keys():
        merged[key] = torch.cat([b[key] for b in batches], dim=0)
    merged["env_ids"] = torch.cat(env_ids, dim=0)
    return merged, merged["env_ids"]


def _env_level_metrics(model, risk_fn, dataset: ExperimentDataset, device: torch.device):
    metrics: dict[str, dict[str, float]] = {}
    for name, data in dataset.environments.items():
        batch = {k: v.to(device) for k, v in data.items()}
        env_ids = batch["env_ids"]
        env_risks, pnl, actions, env_tensor = compute_env_risks(model, batch, env_ids, risk_fn)
        risk_value = float(list(env_risks.values())[0].detach().cpu()) if env_risks else 0.0
        pnl_arr = pnl.detach().cpu()
        metrics[name] = {
            "risk": risk_value,
            "pnl_mean": float(pnl_arr.mean()) if pnl_arr.numel() else 0.0,
            "pnl_cvar95": float(torch.quantile(pnl_arr, 0.05)) if pnl_arr.numel() else 0.0,
            "turnover": float(actions.abs().mean().detach().cpu()) if actions is not None else 0.0,
            "env_id": int(env_tensor[0].item()) if env_tensor.numel() else 0,
        }
    return metrics


def _run_diagnostics(trainer, train_data: ExperimentDataset, test_data: ExperimentDataset, cfg: ConfigNode, device: torch.device):
    diag_cfg = getattr(cfg, "diagnostics", {})
    isi_cfg = getattr(diag_cfg, "isi", {})

    raw_alpha = isi_cfg.get("alpha_components", [1.0, 1.0, 1.0])
    if isinstance(raw_alpha, (int, float, str)):
        try:
            val = float(raw_alpha)
        except Exception:
            val = 1.0
        alpha_components = [val, val, val]
    else:
        alpha_components = list(raw_alpha)
    if len(alpha_components) == 1:
        alpha_components = [alpha_components[0]] * 3
    elif len(alpha_components) == 2:
        alpha_components = [alpha_components[0], alpha_components[1], alpha_components[1]]
    elif len(alpha_components) > 3:
        alpha_components = alpha_components[:3]

    train_batch, train_env_ids = _combine_environments(train_data, device)
    test_batch, test_env_ids = _combine_environments(test_data, device)

    env_risks, train_pnl, _, _ = compute_env_risks(trainer.model, train_batch, train_env_ids, trainer.risk_fn)
    test_env_risks, eval_pnl, eval_actions, _ = compute_env_risks(trainer.model, test_batch, test_env_ids, trainer.risk_fn)

    head_gradients, layer_activations = collect_invariance_signals(
        trainer.model,
        train_data.environments,
        getattr(trainer.model, "invariance_mode", "head_only"),
        device,
        trainer.risk_fn,
        max_samples_per_env=int(getattr(isi_cfg, "max_samples_per_env", 256)),
    )

    robustness_inputs = {
        "wg_inputs": {
            "env_risks": {f"env_{k}": float(v.detach().cpu()) for k, v in test_env_risks.items()},
            "alpha": float(getattr(getattr(diag_cfg, "wg", {}), "alpha", 0.05)),
            "num_grid": int(getattr(getattr(diag_cfg, "wg", {}), "num_grid", 1000)),
        },
        "vr_inputs": {
            "risk_time_series": (-train_pnl.detach().cpu()).tolist(),
            "eps": float(getattr(getattr(diag_cfg, "vr", {}), "eps", 1e-8)),
        },
    }

    efficiency_inputs = {
        "er_inputs": {
            "returns_time_series": eval_pnl.detach().cpu().tolist(),
            "cvar_alpha": float(getattr(getattr(diag_cfg, "er", {}), "cvar_alpha", 0.05)),
            "eps": float(getattr(getattr(diag_cfg, "er", {}), "eps", 1e-8)),
            "mode": str(getattr(getattr(diag_cfg, "er", {}), "mode", "loss")),
        },
        "tr_inputs": {
            "actions": eval_actions.detach().cpu().tolist(),
            "eps": float(getattr(getattr(diag_cfg, "tr", {}), "eps", 1e-8)),
            "treat_first_step": str(getattr(getattr(diag_cfg, "tr", {}), "treat_first_step", "drop")),
        },
    }

    invariance_inputs = {
        "isi_inputs": {
            "env_risks": {f"env_{k}": float(v.detach().cpu()) for k, v in env_risks.items()},
            "head_gradients": {k: v.detach().cpu() for k, v in head_gradients.items()},
            "layer_activations": layer_activations,
            "tau_R": float(isi_cfg.get("tau_R", 0.05)),
            "tau_C": float(isi_cfg.get("tau_C", 1.0)),
            "alpha_components": alpha_components,
            "eps": float(isi_cfg.get("eps", 1e-8)),
            "cov_regularizer": float(isi_cfg.get("cov_regularizer", 1e-4)),
            "grad_norm_eps": float(isi_cfg.get("grad_norm_eps", 1e-12)),
            "trim_fraction": float(isi_cfg.get("trim_fraction", 0.1)),
        },
        "ig_inputs": {
            "test_env_risks": {f"env_{k}": float(v.detach().cpu()) for k, v in test_env_risks.items()},
            "tau_IG": float(isi_cfg.get("tau_IG", 0.05)),
            "eps": float(isi_cfg.get("eps", 1e-8)),
        },
    }

    diagnostics = compute_all_diagnostics(
        invariance_inputs=invariance_inputs,
        robustness_inputs=robustness_inputs,
        efficiency_inputs=efficiency_inputs,
    )

    diagnostics.update(
        {
            "metrics/isi/global": float(diagnostics.get("isi", 0.0)),
            "metrics/ig/global": float(diagnostics.get("ig", 0.0)),
            "metrics/wg": float(diagnostics.get("wg", 0.0)),
            "metrics/vr": float(diagnostics.get("vr", 0.0)),
            "metrics/er": float(diagnostics.get("er", 0.0)),
            "metrics/tr": float(diagnostics.get("tr", 0.0)),
            "metrics/pnl/mean": float(eval_pnl.mean().detach().cpu()),
            "metrics/pnl/cvar95": float(torch.quantile(eval_pnl.detach().cpu(), 0.05)),
        }
    )

    env_metrics = _env_level_metrics(trainer.model, trainer.risk_fn, test_data, device)
    diagnostics["env_metrics"] = env_metrics

    crisis_cfg = getattr(diag_cfg, "crisis", {})
    if crisis_cfg.get("enabled", False):
        risk_fn = build_risk_function(cfg.objective)
        env_pnls = {}
        for name, data in test_data.environments.items():
            batch = {k: v.to(device) for k, v in data.items()}
            env_ids = batch["env_ids"]
            _, pnl, _, _ = compute_env_risks(trainer.model, batch, env_ids, risk_fn)
            env_pnls[name] = pnl.detach().cpu()
        crisis_alpha = float(crisis_cfg.get("cvar_alpha", 0.05))
        crisis_metrics: dict[str, float] = {}
        for name, pnl_tensor in env_pnls.items():
            if "crisis" in name or name.startswith("20"):
                cvar_val = float(torch.quantile(pnl_tensor, crisis_alpha))
                crisis_metrics[name] = cvar_val
        diagnostics["crisis_cvar"] = crisis_metrics

    return diagnostics


def _resolve_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def run_single_experiment(
    dataset_name: str,
    method_name: str,
    seed: int,
    base_cfg: ConfigNode,
    device: torch.device,
    force: bool,
    results_root: Path,
    method_key: str | None = None,
) -> None:
    method_for_builder = method_key or method_name
    out_dir = results_root / dataset_name / method_name / f"seed_{seed}"
    checkpoint_path = out_dir / "checkpoint.pt"
    diagnostics_path = out_dir / "diagnostics.jsonl"
    train_log_path = out_dir / "train_logs.jsonl"
    metadata_path = out_dir / "metadata.json"
    config_path = out_dir / "config.json"

    if not force and checkpoint_path.exists() and diagnostics_path.exists():
        print(
            f"[FULL_SUITE] Skipping completed run dataset={dataset_name} "
            f"method={method_name} seed={seed}"
        )
        return

    cfg = ConfigNode(to_plain_dict(base_cfg))
    cfg.seed = seed
    _ensure_dir(out_dir)
    config_path.write_text(json.dumps(to_plain_dict(cfg), indent=2), encoding="utf-8")

    dataset_builder = get_dataset_builder(dataset_name)
    dataset_train = dataset_builder(cfg, split="train", seed=seed)
    dataset_val = dataset_builder(cfg, split="val", seed=seed + 13)
    dataset_test = dataset_builder(cfg, split="test", seed=seed + 31)

    if not getattr(dataset_val, "environments", None):
        print(f"[WARN] Validation split empty for dataset={dataset_name}, using train split as validation.")
        dataset_val = dataset_train

    run_cfg = ExperimentRunConfig(
        dataset=dataset_name,
        method=method_for_builder,
        seed=seed,
        config=cfg,
        device=device,
    )
    trainer_builder = get_method_builder(method_for_builder)
    trainer = trainer_builder(run_cfg)
    trainer.set_datasets(train=dataset_train, val=dataset_val)

    start = time.time()
    logs = trainer.train()
    trainer.save(str(checkpoint_path))
    diagnostics = _run_diagnostics(trainer, dataset_train, dataset_test, cfg, device)
    elapsed = time.time() - start

    _write_jsonl(train_log_path, logs)
    _write_jsonl(
        diagnostics_path,
        [
            {
                "method": method_name,
                "dataset": dataset_name,
                "seed": seed,
                **diagnostics,
            }
        ],
    )
    metadata = {
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "timestamp": time.time(),
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "training_time_seconds": elapsed,
        "library_versions": {"torch": torch.__version__},
        "command": " ".join(sys.argv),
        "device": str(device),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[FULL_SUITE] Finished dataset={dataset_name} method={method_name} seed={seed}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    grid_cfg = getattr(cfg, "experiment_grid", {})
    default_methods = grid_cfg.get("methods", ["erm", "hirm"])
    default_datasets = grid_cfg.get("datasets", ["synthetic_heston", "real_spy"])
    default_seeds = grid_cfg.get("seeds", list(range(3 if not args.reduced else 1)))

    raw_methods = _resolve_list(args.methods) or default_methods
    methods = [(m, METHOD_ALIASES.get(m, m)) for m in raw_methods]
    datasets = _resolve_list(args.datasets) or default_datasets
    seeds = [int(s) for s in (_resolve_list(args.seeds) or default_seeds)]
    device = torch.device(args.device)
    force = bool(args.force_rerun)
    deterministic = bool(getattr(getattr(cfg, "training", {}), "deterministic", False))
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for display_method, method_key in methods:
            for seed in seeds:
                print(
                    f"[FULL_SUITE] Starting dataset={dataset} "
                    f"method={display_method} seed={seed}"
                )
                _set_seed(seed, deterministic=deterministic)
                run_single_experiment(
                    dataset,
                    display_method,
                    seed,
                    cfg,
                    device,
                    force,
                    results_root=results_root,
                    method_key=method_key,
                )


if __name__ == "__main__":
    main()
