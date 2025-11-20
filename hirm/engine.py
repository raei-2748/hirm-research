"""Core execution engine for HIRM experiments.

Encapsulates the training loop, checkpointing, and I-R-E diagnostic evaluation
for benchmark, ablation, and paper_grid experiment grids.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch

from hirm.diagnostics import compute_all_diagnostics, compute_crisis_cvar
from hirm.diagnostics.invariance_helpers import collect_invariance_signals
from hirm.experiments.datasets import ExperimentDataset, get_dataset_builder
from hirm.experiments.methods import get_method_builder
from hirm.experiments.registry import ExperimentRunConfig
from hirm.objectives.common import compute_env_risks
from hirm.utils.config import ConfigNode, to_plain_dict
from hirm.utils.seed import set_seed as apply_seed


def train_and_evaluate(
    run_cfg: ExperimentRunConfig,
    output_dir: Path,
    force_rerun: bool = False,
) -> None:
    """Execute a full experiment lifecycle: Train -> Save -> Diagnose -> Log."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ckpt": output_dir / "checkpoint.pt",
        "diag": output_dir / "diagnostics.jsonl",
        "logs": output_dir / "train_logs.jsonl",
        "meta": output_dir / "metadata.json",
        "conf": output_dir / "config.json",
    }

    if not force_rerun and paths["ckpt"].exists() and paths["diag"].exists():
        print(f"[SKIP] {run_cfg.dataset}/{run_cfg.method}/seed_{run_cfg.seed} exists.")
        return

    apply_seed(run_cfg.seed, deterministic=_get_training_cfg(run_cfg.config).get("deterministic", False))
    paths["conf"].write_text(json.dumps(to_plain_dict(run_cfg.config), indent=2))

    train_ds, val_ds, test_ds = _load_datasets(run_cfg)
    if not getattr(val_ds, "environments", None):
        val_ds = train_ds

    trainer = get_method_builder(run_cfg.method)(run_cfg)
    trainer.set_datasets(train=train_ds, val=val_ds)

    start_time = time.time()
    logs = trainer.train()
    trainer.save(str(paths["ckpt"]))
    training_time = time.time() - start_time

    diagnostics = _compute_diagnostics(trainer, train_ds, test_ds, run_cfg)

    _write_jsonl(paths["logs"], logs)
    _write_jsonl(
        paths["diag"],
        [
            {
                "method": run_cfg.method,
                "dataset": run_cfg.dataset,
                "seed": run_cfg.seed,
                "ablation": run_cfg.ablation.name if run_cfg.ablation else None,
                **diagnostics,
            }
        ],
    )

    paths["meta"].write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "duration": training_time,
                "device": str(run_cfg.device),
                "commit": os.popen("git rev-parse HEAD").read().strip(),
            },
            indent=2,
        )
    )

    print(
        f"[DONE] {run_cfg.dataset}/{run_cfg.method}/seed_{run_cfg.seed} "
        f"(ISI={diagnostics.get('metrics/isi/global', 0):.2f}, "
        f"Crisis={diagnostics.get('metrics/cvar95/crisis', 0):.2f})"
    )


def _get_training_cfg(cfg: ConfigNode | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg, ConfigNode):
        cfg = to_plain_dict(cfg)
    return cfg.get("training", {}) if isinstance(cfg, dict) else {}


def _load_datasets(run_cfg: ExperimentRunConfig) -> tuple[ExperimentDataset, ExperimentDataset, ExperimentDataset]:
    ds_builder = get_dataset_builder(run_cfg.dataset)
    train_ds = ds_builder(run_cfg.config, split="train", seed=run_cfg.seed)
    val_ds = ds_builder(run_cfg.config, split="val", seed=run_cfg.seed + 13)
    test_ds = ds_builder(run_cfg.config, split="test", seed=run_cfg.seed + 31)
    return train_ds, val_ds, test_ds


def _merge_environments(dataset: ExperimentDataset, device: torch.device) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    batches = [{k: v.to(device) for k, v in data.items()} for data in dataset.environments.values()]
    merged: Dict[str, torch.Tensor] = {}
    for key in batches[0].keys():
        merged[key] = torch.cat([b[key] for b in batches], dim=0)
    return merged, merged["env_ids"]


def _compute_diagnostics(trainer, train_ds: ExperimentDataset, test_ds: ExperimentDataset, run_cfg: ExperimentRunConfig) -> Dict[str, float]:
    """Compute invariance, robustness, and efficiency diagnostics (paper Sections 4--6)."""

    device = run_cfg.device
    model = trainer.model
    model.eval()

    train_batch, train_env_ids = _merge_environments(train_ds, device)
    test_batch, test_env_ids = _merge_environments(test_ds, device)

    train_env_risks, train_pnl, _, _ = compute_env_risks(model, train_batch, train_env_ids, trainer.risk_fn)
    test_env_risks, test_pnl, test_actions, _ = compute_env_risks(model, test_batch, test_env_ids, trainer.risk_fn)

    isi_cfg = run_cfg.config.diagnostics.isi
    invariance_mode = getattr(model, "invariance_mode", "head_only")
    head_grads, layer_acts = collect_invariance_signals(
        model,
        train_ds.environments,
        invariance_mode,
        device,
        trainer.risk_fn,
        max_samples_per_env=int(getattr(isi_cfg, "max_samples_per_env", 256)),
    )

    diag_cfg = run_cfg.config.diagnostics
    diagnostics_inputs = {
        "invariance_inputs": {
            "isi_inputs": {
                "env_risks": {f"env_{k}": v.detach().cpu().item() for k, v in train_env_risks.items()},
                "head_gradients": head_grads,
                "layer_activations": layer_acts,
                "tau_R": float(isi_cfg.tau_R),
                "tau_C": float(isi_cfg.tau_C),
                "alpha_components": [float(x) for x in isi_cfg.alpha_components],
                "eps": float(isi_cfg.eps),
                "cov_regularizer": float(isi_cfg.cov_regularizer),
            },
            "ig_inputs": {
                "test_env_risks": {f"env_{k}": v.detach().cpu().item() for k, v in test_env_risks.items()},
                "tau_IG": float(isi_cfg.tau_IG),
                "eps": float(isi_cfg.eps),
            },
        },
        "robustness_inputs": {
            "wg_inputs": {
                "env_risks": {f"env_{k}": v.detach().cpu().item() for k, v in test_env_risks.items()},
                "alpha": float(diag_cfg.wg.alpha),
                "num_grid": int(diag_cfg.wg.num_grid),
            },
            "vr_inputs": {
                "risk_time_series": (-train_pnl.detach().cpu()).tolist(),
                "eps": float(diag_cfg.vr.eps),
            },
        },
        "efficiency_inputs": {
            "er_inputs": {
                "returns_time_series": test_pnl.detach().cpu().tolist(),
                "cvar_alpha": float(diag_cfg.er.cvar_alpha),
                "mode": diag_cfg.er.mode,
                "eps": float(diag_cfg.er.eps),
            },
            "tr_inputs": {
                "actions": test_actions.detach().cpu().tolist(),
                "treat_first_step": diag_cfg.tr.treat_first_step,
                "eps": float(diag_cfg.tr.eps),
            },
        },
    }

    raw_metrics = compute_all_diagnostics(**diagnostics_inputs)
    metrics = {
        "isi": raw_metrics.get("ISI", 0.0),
        "ig": raw_metrics.get("IG", 0.0),
        "wg": raw_metrics.get("WG", 0.0),
        "vr": raw_metrics.get("VR", 0.0),
        "er": raw_metrics.get("ER", 0.0),
        "tr": raw_metrics.get("TR", 0.0),
        "metrics/isi/global": raw_metrics.get("ISI", 0.0),
        "metrics/ig/global": raw_metrics.get("IG", 0.0),
        "metrics/wg": raw_metrics.get("WG", 0.0),
        "metrics/vr": raw_metrics.get("VR", 0.0),
        "metrics/er": raw_metrics.get("ER", 0.0),
        "metrics/tr": raw_metrics.get("TR", 0.0),
        **raw_metrics,
    }

    if diag_cfg.crisis.enabled:
        crisis_metrics = _compute_crisis_metrics(test_ds, model, trainer.risk_fn, device, float(diag_cfg.crisis.cvar_alpha))
        metrics.update(crisis_metrics)

    return metrics


def _compute_crisis_metrics(test_ds: ExperimentDataset, model, risk_fn, device: torch.device, alpha: float) -> Dict[str, float]:
    pnl_time_series = []
    for name, data in test_ds.environments.items():
        if "crisis" not in name.lower() and not name.startswith("20"):
            continue
        batch = {k: v.to(device) for k, v in data.items()}
        _, pnl, _, _ = compute_env_risks(model, batch, batch["env_ids"], risk_fn)
        pnl_time_series.extend(pnl.detach().cpu().tolist())

    if not pnl_time_series:
        return {}

    crisis = compute_crisis_cvar(pnl_time_series=pnl_time_series, alpha=alpha)
    return {"metrics/cvar95/crisis": crisis.get("crisis_cvar", 0.0)}


def _write_jsonl(path: Path, data):
    with path.open("w", encoding="utf-8") as handle:
        for entry in data:
            handle.write(json.dumps(entry) + "\n")


__all__ = ["train_and_evaluate"]
