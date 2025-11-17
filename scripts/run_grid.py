"""Grid runner for Phase 7 benchmark experiments."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from hirm.diagnostics import compute_all_diagnostics, compute_crisis_cvar
from hirm.objectives.common import concat_state, compute_env_risks
from hirm.utils.config import ConfigNode, load_config, to_plain_dict
from hirm.experiments.datasets import ExperimentDataset, get_dataset_builder
from hirm.experiments.methods import get_method_builder
from hirm.experiments.registry import ExperimentRunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_rerun", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
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


def _collect_layer_activations(model, batch: Mapping[str, torch.Tensor], env_ids: torch.Tensor, layer_names: Iterable[str]):
    hooks = []
    captured: Dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    for name in layer_names:
        module = modules.get(name)
        if module is None:
            continue

        def _hook(_, __, output, layer=name):  # type: ignore[override]
            captured[layer] = output.detach().cpu()

        hooks.append(module.register_forward_hook(_hook))
    with torch.no_grad():
        features = concat_state(batch)
        model(features, env_ids=env_ids)
    for handle in hooks:
        handle.remove()
    env_tensor = env_ids.detach().cpu()
    unique_envs = torch.unique(env_tensor)
    layer_acts: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer, tensor in captured.items():
        env_map: Dict[str, torch.Tensor] = {}
        for env in unique_envs.tolist():
            mask = env_tensor == env
            if not mask.any():
                continue
            env_map[f"env_{int(env)}"] = tensor[mask].detach().cpu().tolist()
        layer_acts[layer] = env_map
    return layer_acts


def _collect_head_gradients(model, env_risks: Mapping[int, torch.Tensor]) -> Dict[str, list[float]]:
    head_params = list(model.head_parameters()) if hasattr(model, "head_parameters") else []
    if not head_params:
        return {}
    gradients: Dict[str, list[float]] = {}
    env_items = list(env_risks.items())
    for idx, (env, risk) in enumerate(env_items):
        model.zero_grad(set_to_none=True)
        retain_graph = idx < len(env_items) - 1
        risk.backward(retain_graph=retain_graph)
        flat: list[torch.Tensor] = []
        for param in head_params:
            if param.grad is None:
                continue
            flat.append(param.grad.detach().reshape(-1))
        if flat:
            gradients[f"env_{int(env)}"] = torch.cat(flat).detach().cpu().tolist()
    model.zero_grad(set_to_none=True)
    return gradients


def _combine_environments(dataset: ExperimentDataset, device: torch.device) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    batches = []
    env_ids = []
    for name, data in dataset.environments.items():
        batches.append({k: v.to(device) for k, v in data.items()})
        env_ids.append(data["env_ids"].to(device))
    merged: Dict[str, torch.Tensor] = {}
    for key in batches[0].keys():
        merged[key] = torch.cat([b[key] for b in batches], dim=0)
    merged["env_ids"] = torch.cat(env_ids, dim=0)
    return merged, merged["env_ids"]


def _env_level_metrics(model, risk_fn, dataset: ExperimentDataset, device: torch.device):
    metrics: Dict[str, Dict[str, float]] = {}
    for name, data in dataset.environments.items():
        batch = {k: v.to(device) for k, v in data.items()}
        env_ids = batch["env_ids"]
        env_risks, pnl, actions, _ = compute_env_risks(model, batch, env_ids, risk_fn)
        risk_value = float(list(env_risks.values())[0].detach().cpu()) if env_risks else 0.0
        pnl_arr = pnl.detach().cpu()
        mean_pnl = float(pnl_arr.mean())
        cvar95 = float(torch.quantile(pnl_arr, 0.05)) if pnl_arr.numel() > 0 else 0.0
        max_drawdown = float(torch.min(torch.cumsum(pnl_arr, dim=0)).item()) if pnl_arr.numel() > 0 else 0.0
        metrics[name] = {
            "cvar95": cvar95,
            "mean_pnl": mean_pnl,
            "max_drawdown": max_drawdown,
            "risk": risk_value,
            "turnover": float(actions.abs().mean().detach().cpu()) if actions is not None else 0.0,
        }
    return metrics


def _compute_env_pnls(model, risk_fn, dataset: ExperimentDataset, device: torch.device) -> Dict[str, torch.Tensor]:
    pnl_series: Dict[str, torch.Tensor] = {}
    for name, data in dataset.environments.items():
        batch = {k: v.to(device) for k, v in data.items()}
        env_ids = batch["env_ids"]
        _, pnl, _, _ = compute_env_risks(model, batch, env_ids, risk_fn)
        pnl_series[name] = pnl.detach().cpu()
    return pnl_series


def run_diagnostics(trainer, train_data: ExperimentDataset, test_data: ExperimentDataset, cfg: ConfigNode, device: torch.device):
    diag_cfg = getattr(cfg, "diagnostics", {})
    isi_cfg = getattr(diag_cfg, "isi", {})
    train_batch, train_env_ids = _combine_environments(train_data, device)
    test_batch, test_env_ids = _combine_environments(test_data, device)

    env_risks_tensor, train_pnl, _, _ = compute_env_risks(trainer.model, train_batch, train_env_ids, trainer.risk_fn)
    env_risks = {f"env_{env}": float(risk.detach().cpu()) for env, risk in env_risks_tensor.items()}
    head_gradients = _collect_head_gradients(trainer.model, env_risks_tensor)
    layer_activations = _collect_layer_activations(
        trainer.model,
        train_batch,
        train_env_ids,
        isi_cfg.get("probe_layers", ["representation", "head"]),
    )

    test_env_risks_tensor, eval_pnl, eval_actions, _ = compute_env_risks(trainer.model, test_batch, test_env_ids, trainer.risk_fn)
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
            "head_gradients": head_gradients,
            "layer_activations": layer_activations,
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
    env_metrics = _env_level_metrics(trainer.model, trainer.risk_fn, test_data, device)
    diagnostics["env_metrics"] = env_metrics

    # Prefixed metrics for downstream logging
    prefixed: Dict[str, float] = {}
    if "isi" in diagnostics:
        prefixed["metrics/isi/global"] = float(diagnostics.get("isi", 0.0))
    if "ig" in diagnostics:
        prefixed["metrics/ig/global"] = float(diagnostics.get("ig", 0.0))
    if "wg" in diagnostics:
        prefixed["metrics/wg"] = float(diagnostics.get("wg", 0.0))
    if "vr" in diagnostics:
        prefixed["metrics/vr"] = float(diagnostics.get("vr", 0.0))
    if "er" in diagnostics:
        prefixed["metrics/er"] = float(diagnostics.get("er", 0.0))
    if "tr" in diagnostics:
        prefixed["metrics/tr"] = float(diagnostics.get("tr", 0.0))
    diagnostics.update(prefixed)

    crisis_cfg = getattr(diag_cfg, "crisis", {})
    if crisis_cfg.get("enabled", False):
        env_pnls = _compute_env_pnls(trainer.model, trainer.risk_fn, test_data, device)
        crisis_alpha = float(crisis_cfg.get("cvar_alpha", 0.05))
        crisis_metrics: Dict[str, float] = {}
        for name, pnl_tensor in env_pnls.items():
            if "crisis" in name or name.startswith("20"):
                result = compute_crisis_cvar(
                    pnl_time_series=pnl_tensor.tolist(),
                    alpha=crisis_alpha,
                )
                crisis_metrics[name] = float(result.get("crisis_cvar", 0.0))
        if crisis_metrics:
            diagnostics["crisis_cvar"] = crisis_metrics
            if "crisis" in crisis_metrics:
                diagnostics.setdefault("crisis_cvar_aggregate", crisis_metrics.get("crisis", 0.0))
    return diagnostics


def _resolve_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def run_single_experiment(dataset_name: str, method_name: str, seed: int, base_cfg: ConfigNode, device: torch.device, force: bool):
    out_dir = Path("results") / dataset_name / method_name / f"seed_{seed}"
    _ensure_dir(out_dir)
    checkpoint_path = out_dir / "checkpoint.pt"
    diagnostics_path = out_dir / "diagnostics.jsonl"
    train_log_path = out_dir / "train_logs.jsonl"
    config_path = out_dir / "config.yaml"
    metadata_path = out_dir / "metadata.json"

    if not force and checkpoint_path.exists() and diagnostics_path.exists():
        print(f"[GRID] Skipping completed run dataset={dataset_name} method={method_name} seed={seed}")
        return

    cfg = ConfigNode(to_plain_dict(base_cfg))
    if "objective" not in cfg:
        cfg["objective"] = ConfigNode({"name": method_name})
    else:
        cfg.objective.name = method_name
    if "model" not in cfg:
        cfg["model"] = ConfigNode({"name": "invariant_policy"})
    if "env" not in cfg:
        cfg["env"] = ConfigNode({"feature_dim": 6, "action_dim": 2})
    cfg.seed = seed
    _ensure_dir(out_dir)
    config_path.write_text(json.dumps(to_plain_dict(cfg), indent=2), encoding="utf-8")

    dataset_builder = get_dataset_builder(dataset_name)
    dataset_train = dataset_builder(cfg, split="train", seed=seed)
    dataset_val = dataset_builder(cfg, split="val", seed=seed + 13)
    dataset_test = dataset_builder(cfg, split="test", seed=seed + 31)

    run_cfg = ExperimentRunConfig(
        dataset=dataset_name,
        method=method_name,
        seed=seed,
        config=cfg,
        device=device,
    )
    trainer_builder = get_method_builder(method_name)
    trainer = trainer_builder(run_cfg)
    trainer.set_datasets(train=dataset_train, val=dataset_val)

    start = time.time()
    logs = trainer.train()
    trainer.save(str(checkpoint_path))
    diagnostics = run_diagnostics(trainer, dataset_train, dataset_test, cfg, device)
    elapsed = time.time() - start

    _write_jsonl(train_log_path, logs)
    _write_jsonl(diagnostics_path, [
        {
            "method": method_name,
            "dataset": dataset_name,
            "seed": seed,
            **diagnostics,
        }
    ])
    metadata = {
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "timestamp": time.time(),
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "training_time_seconds": elapsed,
        "library_versions": {"torch": torch.__version__},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[GRID] Finished dataset={dataset_name} method={method_name} seed={seed}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    methods = _resolve_list(args.methods) or list(getattr(cfg, "methods", []))
    datasets = _resolve_list(args.datasets) or list(getattr(cfg, "datasets", []))
    seeds = [int(s) for s in (_resolve_list(args.seeds) or list(getattr(cfg, "seeds", [0])))]
    device = torch.device(args.device)
    force = bool(args.force_rerun)
    deterministic = bool(getattr(getattr(cfg, "training", {}), "deterministic", False))

    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                print(f"[GRID] Starting dataset={dataset} method={method} seed={seed}")
                _set_seed(seed, deterministic=deterministic)
                run_single_experiment(dataset, method, seed, cfg, device, force)


if __name__ == "__main__":
    main()
