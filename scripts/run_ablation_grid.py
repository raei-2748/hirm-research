"""Phase 8 ablation grid runner."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from hirm.diagnostics import compute_all_diagnostics, compute_crisis_cvar
from hirm.experiments.ablations import apply_ablation_to_config, get_ablation_config, list_ablations
from hirm.experiments.datasets import ExperimentDataset, get_dataset_builder
from hirm.experiments.methods import get_method_builder
from hirm.experiments.registry import ExperimentRunConfig
from hirm.objectives.common import concat_state, compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import ConfigNode, load_config, to_plain_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation_names", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_rerun", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reduced", action="store_true", help="Run a reduced grid for smoke testing")
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
    train_batch, train_env_ids = _combine_environments(train_data, device)
    test_batch, test_env_ids = _combine_environments(test_data, device)

    env_risks_tensor, train_pnl, _, _ = compute_env_risks(trainer.model, train_batch, train_env_ids, trainer.risk_fn)
    env_risks = {f"env_{env}": float(risk.detach().cpu()) for env, risk in env_risks_tensor.items()}

    test_env_risks_tensor, eval_pnl, eval_actions, eval_env_tensor = compute_env_risks(
        trainer.model, test_batch, test_env_ids, trainer.risk_fn
    )
    test_env_risks = {f"env_{env}": float(risk.detach().cpu()) for env, risk in test_env_risks_tensor.items()}

    robustness_inputs = {
        "wg_inputs": {
            "env_risks": test_env_risks,
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
            "env_risks": env_risks,
            "head_gradients": {},
            "layer_activations": {},
            "tau_R": float(isi_cfg.get("tau_R", 0.05)),
            "tau_C": float(isi_cfg.get("tau_C", 1.0)),
            "alpha_components": list(isi_cfg.get("alpha_components", [1.0, 1.0, 1.0])),
            "eps": float(isi_cfg.get("eps", 1e-8)),
            "cov_regularizer": float(isi_cfg.get("cov_regularizer", 1e-4)),
            "grad_norm_eps": float(isi_cfg.get("grad_norm_eps", 1e-12)),
            "trim_fraction": float(isi_cfg.get("trim_fraction", 0.1)),
        },
        "ig_inputs": {
            "test_env_risks": test_env_risks,
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
                result = compute_crisis_cvar(pnl_time_series=pnl_tensor.tolist(), alpha=crisis_alpha)
                crisis_metrics[name] = float(result.get("crisis_cvar", 0.0))
        if crisis_metrics:
            diagnostics["crisis_cvar"] = crisis_metrics
            if "crisis" in crisis_metrics:
                diagnostics["metrics/cvar95/crisis"] = crisis_metrics.get("crisis", 0.0)
    return diagnostics


def _resolve_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def run_single_ablation(dataset_name: str, ablation_name: str, seed: int, base_cfg: ConfigNode, device: torch.device, force: bool):
    ablation = get_ablation_config(ablation_name)
    if ablation is None:
        raise ValueError(f"Unknown ablation {ablation_name}")
    out_dir = Path("results") / "phase8" / dataset_name / ablation.name / f"seed_{seed}"
    _ensure_dir(out_dir)
    checkpoint_path = out_dir / "checkpoint.pt"
    diagnostics_path = out_dir / "diagnostics.jsonl"
    train_log_path = out_dir / "train_logs.jsonl"
    config_path = out_dir / "config.yaml"
    metadata_path = out_dir / "metadata.json"

    if not force and checkpoint_path.exists() and diagnostics_path.exists():
        print(f"[ABLATION] Skipping completed run dataset={dataset_name} ablation={ablation_name} seed={seed}")
        return

    cfg = apply_ablation_to_config(base_cfg, ablation)
    cfg.seed = seed
    _ensure_dir(out_dir)
    config_path.write_text(json.dumps(to_plain_dict(cfg), indent=2), encoding="utf-8")

    dataset_builder = get_dataset_builder(dataset_name)
    dataset_train = dataset_builder(cfg, split="train", seed=seed)
    dataset_val = dataset_builder(cfg, split="val", seed=seed + 13)
    dataset_test = dataset_builder(cfg, split="test", seed=seed + 31)

    run_cfg = ExperimentRunConfig(
        dataset=dataset_name,
        method=ablation.method,
        seed=seed,
        config=cfg,
        device=device,
        ablation=ablation,
    )
    trainer_builder = get_method_builder(ablation.method)
    trainer = trainer_builder(run_cfg)
    trainer.set_datasets(train=dataset_train, val=dataset_val)

    start = time.time()
    logs = trainer.train()
    trainer.save(str(checkpoint_path))
    diagnostics = _run_diagnostics(trainer, dataset_train, dataset_test, cfg, device)
    elapsed = time.time() - start

    _write_jsonl(train_log_path, logs)
    _write_jsonl(diagnostics_path, [
        {
            "method": ablation.method,
            "dataset": dataset_name,
            "seed": seed,
            "ablation_name": ablation.name,
            **diagnostics,
        }
    ])
    metadata = {
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "timestamp": time.time(),
        "dataset": dataset_name,
        "method": ablation.method,
        "seed": seed,
        "ablation_name": ablation.name,
        "training_time_seconds": elapsed,
        "library_versions": {"torch": torch.__version__},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[ABLATION] Finished dataset={dataset_name} ablation={ablation_name} seed={seed}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ablations = _resolve_list(args.ablation_names) or list_ablations()
    datasets = _resolve_list(args.datasets) or ["synthetic_heston", "real_spy"]
    seeds = [int(s) for s in (_resolve_list(args.seeds) or list(range(10 if not args.reduced else 2)))]
    device = torch.device(args.device)
    force = bool(args.force_rerun)
    deterministic = bool(getattr(getattr(cfg, "training", {}), "deterministic", False))

    for dataset in datasets:
        for ablation_name in ablations:
            for seed in seeds:
                print(f"[ABLATION] Starting dataset={dataset} ablation={ablation_name} seed={seed}")
                _set_seed(seed, deterministic=deterministic)
                run_single_ablation(dataset, ablation_name, seed, cfg, device, force)


if __name__ == "__main__":
    main()
