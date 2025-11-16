"""Compute full I-R-E diagnostics for a saved experiment checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch

from hirm.data.synthetic import build_synthetic_dataset
from hirm.diagnostics import compute_all_diagnostics
from hirm.objectives.common import concat_state, compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.utils.config import load_config
from hirm.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--results-dir", type=str, default="outputs/diagnostics")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default=None)
    return parser.parse_args()


def _format_env(env: int) -> str:
    return f"env_{env}"


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def _collect_layer_activations(
    model,
    batch: Mapping[str, torch.Tensor],
    env_ids: torch.Tensor,
    layer_names: Iterable[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
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
            env_map[_format_env(int(env))] = tensor[mask].detach().cpu().tolist()
        layer_acts[layer] = env_map
    return layer_acts


def _collect_head_gradients(model, env_risks: Mapping[int, torch.Tensor]) -> Dict[str, list[float]]:
    head_params = list(model.head_parameters())
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
            gradients[_format_env(int(env))] = torch.cat(flat).detach().cpu().tolist()
    model.zero_grad(set_to_none=True)
    return gradients


def _build_dataset(cfg, generator: torch.Generator, num_samples: int):  # type: ignore[no-untyped-def]
    feature_dim = int(getattr(getattr(cfg, "env", None), "feature_dim", 6) or 6)
    action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)
    num_envs = int(getattr(getattr(cfg, "training", None), "num_envs", 2) or 2)
    dataset = build_synthetic_dataset(
        num_samples=num_samples,
        feature_dim=feature_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        generator=generator,
    )
    return dataset, feature_dim, action_dim


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(getattr(cfg, "seed", 0))
    torch.manual_seed(seed)
    device = torch.device(args.device)
    train_cfg = getattr(cfg, "training", None)
    dataset_size = int(getattr(train_cfg, "dataset_size", 512) or 512)
    generator = torch.Generator().manual_seed(seed)
    dataset, feature_dim, action_dim = _build_dataset(cfg, generator, dataset_size)
    batch = _to_device(dataset, device)
    eval_generator = torch.Generator().manual_seed(seed + 1)
    eval_dataset, _, _ = _build_dataset(cfg, eval_generator, dataset_size)
    eval_batch = _to_device(eval_dataset, device)

    model = build_model(cfg.model, input_dim=feature_dim, action_dim=action_dim).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict):
            model.load_state_dict(state)
    model.eval()

    risk_fn = build_risk_function(cfg.objective)
    env_risks_tensor, pnl, actions, _ = compute_env_risks(
        model, batch, batch["env_ids"], risk_fn
    )
    env_risks = {_format_env(env): float(risk.detach().cpu()) for env, risk in env_risks_tensor.items()}
    head_gradients = _collect_head_gradients(model, env_risks_tensor)
    diag_cfg = getattr(cfg, "diagnostics", {})
    isi_cfg = getattr(diag_cfg, "isi", {})
    probe_layers = isi_cfg.get("probe_layers", ["representation", "head"])
    layer_activations = _collect_layer_activations(model, batch, batch["env_ids"], probe_layers)

    test_env_risks_tensor, eval_pnl, eval_actions, _ = compute_env_risks(
        model, eval_batch, eval_batch["env_ids"], risk_fn
    )
    test_env_risks = {
        _format_env(env): float(risk.detach().cpu()) for env, risk in test_env_risks_tensor.items()
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
        },
        "ig_inputs": {
            "test_env_risks": test_env_risks,
            "tau_IG": float(isi_cfg.get("tau_IG", 0.05)),
            "eps": float(isi_cfg.get("eps", 1e-8)),
        },
    }

    diag_wg = getattr(diag_cfg, "wg", {})
    diag_vr = getattr(diag_cfg, "vr", {})
    diag_er = getattr(diag_cfg, "er", {})
    diag_tr = getattr(diag_cfg, "tr", {})

    robustness_inputs = {
        "wg_inputs": {
            "env_risks": test_env_risks,
            "alpha": float(diag_wg.get("alpha", 0.05)),
            "num_grid": int(diag_wg.get("num_grid", 1000)),
        },
        "vr_inputs": {
            "risk_time_series": (-pnl.detach().cpu()).tolist(),
            "eps": float(diag_vr.get("eps", 1e-8)),
        },
    }

    efficiency_inputs = {
        "er_inputs": {
            "returns_time_series": eval_pnl.detach().cpu().tolist(),
            "cvar_alpha": float(diag_er.get("cvar_alpha", 0.05)),
            "eps": float(diag_er.get("eps", 1e-8)),
        },
        "tr_inputs": {
            "actions": eval_actions.detach().cpu().tolist(),
            "eps": float(diag_tr.get("eps", 1e-8)),
            "treat_first_step": str(diag_tr.get("treat_first_step", "drop")),
        },
    }

    metrics = compute_all_diagnostics(
        invariance_inputs=invariance_inputs,
        robustness_inputs=robustness_inputs,
        efficiency_inputs=efficiency_inputs,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment_id": cfg.experiment.name,
        "model_name": args.model_name or getattr(cfg.model, "name", "model"),
        "seed": seed,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
    }
    out_path = results_dir / "diagnostics.jsonl"
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    pretty = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"Diagnostics for {record['model_name']}: {pretty}")


if __name__ == "__main__":
    main()
