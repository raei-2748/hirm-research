"""Method registry that wraps model/objective construction."""
from __future__ import annotations

import copy
from typing import Callable, Dict, Iterable

import torch
from torch import nn

from hirm.models import build_model
from hirm.objectives import build_objective
from hirm.objectives.common import compute_env_risks
from hirm.objectives.risk import build_risk_function
from hirm.training import train_step
from hirm.utils.config import ConfigNode

from .registry import ExperimentRunConfig, Trainer
from .datasets import ExperimentDataset

MethodBuilder = Callable[[ExperimentRunConfig], Trainer]


_METHOD_REGISTRY: Dict[str, MethodBuilder] = {}


def register_method(name: str) -> Callable[[MethodBuilder], MethodBuilder]:
    key = name.lower()

    def decorator(fn: MethodBuilder) -> MethodBuilder:
        _METHOD_REGISTRY[key] = fn
        return fn

    return decorator


def get_method_builder(name: str) -> MethodBuilder:
    key = name.lower()
    if key not in _METHOD_REGISTRY:
        available = ", ".join(sorted(_METHOD_REGISTRY))
        raise KeyError(f"Unknown method '{name}'. Available: {available}")
    return _METHOD_REGISTRY[key]


def list_methods() -> Iterable[str]:
    return sorted(_METHOD_REGISTRY)


class BasicTrainer(Trainer):
    """Lightweight trainer that reuses the Phase 5 objective logic."""

    def __init__(self, run_cfg: ExperimentRunConfig) -> None:
        self.run_cfg = run_cfg
        cfg = run_cfg.config
        device = run_cfg.device
        feature_dim = int(getattr(getattr(cfg, "env", None), "feature_dim", 6) or 6)
        action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)
        self.model = build_model(cfg.model, input_dim=feature_dim, action_dim=action_dim).to(device)
        self.objective = build_objective(cfg, device=device)
        self.risk_fn = build_risk_function(cfg.objective)
        lr = float(getattr(getattr(cfg, "training", None), "lr", 1e-3) or 1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_data: ExperimentDataset | None = None
        self.val_data: ExperimentDataset | None = None

    def set_datasets(self, *, train: ExperimentDataset, val: ExperimentDataset) -> None:
        self.train_data = train.to_device(self.run_cfg.device)
        self.val_data = val.to_device(self.run_cfg.device)

    def _sample_batch(self, envs: Dict[str, Dict[str, torch.Tensor]], batch_size: int) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        per_env = max(1, batch_size // max(1, len(envs)))
        batches = []
        env_ids = []
        for name, data in envs.items():
            count = min(per_env, data["features"].shape[0])
            idx = torch.randint(data["features"].shape[0], (count,), generator=torch.Generator().manual_seed(self.run_cfg.seed))
            batch = {k: v[idx] for k, v in data.items()}
            batches.append(batch)
            env_ids.append(batch["env_ids"])
        merged: Dict[str, torch.Tensor] = {}
        for key in batches[0].keys():
            merged[key] = torch.cat([b[key] for b in batches], dim=0)
        env_tensor = torch.cat(env_ids, dim=0)
        merged["env_ids"] = env_tensor
        return merged, env_tensor

    def train(self) -> list[dict]:
        if self.train_data is None or self.val_data is None:
            raise RuntimeError("Datasets must be set before training")
        cfg = self.run_cfg.config
        train_cfg = getattr(cfg, "training", ConfigNode({}))
        max_epochs = int(getattr(train_cfg, "max_epochs", getattr(train_cfg, "num_steps", 50)) or 50)
        batch_size = int(getattr(train_cfg, "batch_size", 32) or 32)
        patience = int(getattr(train_cfg, "early_stop_patience", 20) or 20)
        logs: list[dict] = []
        best_val = float("inf")
        best_state = None
        wait = 0
        for epoch in range(1, max_epochs + 1):
            batch, env_ids = self._sample_batch(self.train_data.environments, batch_size)
            self.model.train()
            _, train_logs = train_step(self.model, self.objective, self.optimizer, batch, env_ids, self.risk_fn)

            with torch.no_grad():
                self.model.eval()
                val_batch, val_envs = self._sample_batch(self.val_data.environments, batch_size)
                env_risks, pnl, actions, env_tensor = self.risk_eval(val_batch, val_envs)
                mean_val = torch.stack(list(env_risks.values())).mean().item()
            log_entry = {
                "epoch": epoch,
                "method": self.run_cfg.method,
                "dataset": self.run_cfg.dataset,
                "seed": self.run_cfg.seed,
                "train_cvar95": float(train_logs.get("train/pnl/cvar95", train_logs.get("train/risk/mean", 0.0))),
                "train_mean_pnl": float(train_logs.get("train/pnl/mean", 0.0)),
                "train_turnover": float(train_logs.get("train/turnover", 0.0)),
                "val_cvar95": float(mean_val),
                "val_mean_pnl": float(pnl.mean().item()),
                "val_turnover": float(actions.abs().mean().item()) if actions is not None else 0.0,
            }
            for key, value in train_logs.items():
                log_entry[key] = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            if actions is not None:
                for env in torch.unique(env_tensor).tolist():
                    mask = env_tensor == env
                    env_pnl = pnl[mask]
                    if env_pnl.numel() == 0:
                        continue
                    log_entry[f"val_cvar95_env_{int(env)}"] = float(torch.quantile(env_pnl, 0.05))
            logs.append(log_entry)

            if mean_val < best_val:
                best_val = mean_val
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return logs

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def risk_eval(self, batch: Dict[str, torch.Tensor], env_ids: torch.Tensor):
        return self._compute_env_risks(batch, env_ids)

    def _compute_env_risks(self, batch: Dict[str, torch.Tensor], env_ids: torch.Tensor):
        return compute_env_risks(self.model, batch, env_ids, self.risk_fn)


class RuleBasedPolicy(nn.Module):
    """Deterministic policy that scales hedge notional by realized volatility."""

    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

    def forward(self, x, env_ids=None):  # type: ignore[override]
        vol = x[..., 0:1]
        hedge = torch.clamp(vol, 0.0, 2.0)
        if self.action_dim == 1:
            return hedge
        anti = -hedge
        if self.action_dim > 2:
            pad = torch.zeros_like(x[..., : self.action_dim - 2])
            return torch.cat([hedge, anti, pad], dim=-1)
        return torch.cat([hedge, anti], dim=-1)


class RuleBasedTrainer(Trainer):
    """Non-learned baseline that mimics a simple risk-parity hedge rule."""

    def __init__(self, run_cfg: ExperimentRunConfig) -> None:
        self.run_cfg = run_cfg
        cfg = run_cfg.config
        device = run_cfg.device
        feature_dim = int(getattr(getattr(cfg, "env", None), "feature_dim", 6) or 6)
        action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)
        self.model = RuleBasedPolicy(feature_dim, action_dim).to(device)
        # Use ERM objective purely for metric computation; optimizer unused.
        cfg.objective.name = "erm"
        self.objective = build_objective(cfg, device=device)
        self.risk_fn = build_risk_function(cfg.objective)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)
        self.train_data: ExperimentDataset | None = None
        self.val_data: ExperimentDataset | None = None

    def set_datasets(self, *, train: ExperimentDataset, val: ExperimentDataset) -> None:
        self.train_data = train.to_device(self.run_cfg.device)
        self.val_data = val.to_device(self.run_cfg.device)

    def _sample_batch(self, envs: Dict[str, Dict[str, torch.Tensor]], batch_size: int) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        per_env = max(1, batch_size // max(1, len(envs)))
        batches = []
        env_ids = []
        for _, data in envs.items():
            count = min(per_env, data["features"].shape[0])
            idx = torch.randint(data["features"].shape[0], (count,), generator=torch.Generator().manual_seed(self.run_cfg.seed))
            batch = {k: v[idx] for k, v in data.items()}
            batches.append(batch)
            env_ids.append(batch["env_ids"])
        merged: Dict[str, torch.Tensor] = {}
        for key in batches[0].keys():
            merged[key] = torch.cat([b[key] for b in batches], dim=0)
        env_tensor = torch.cat(env_ids, dim=0)
        merged["env_ids"] = env_tensor
        return merged, env_tensor

    def train(self) -> list[dict]:
        if self.train_data is None or self.val_data is None:
            raise RuntimeError("Datasets must be set before training")
        cfg = self.run_cfg.config
        train_cfg = getattr(cfg, "training", ConfigNode({}))
        max_epochs = int(getattr(train_cfg, "max_epochs", 5) or 5)
        batch_size = int(getattr(train_cfg, "batch_size", 32) or 32)
        logs: list[dict] = []
        for epoch in range(1, max_epochs + 1):
            batch, env_ids = self._sample_batch(self.train_data.environments, batch_size)
            self.model.eval()
            with torch.no_grad():
                env_risks, pnl, actions, env_tensor = compute_env_risks(self.model, batch, env_ids, self.risk_fn)
                val_batch, val_envs = self._sample_batch(self.val_data.environments, batch_size)
                val_risks, val_pnl, val_actions, val_env_tensor = compute_env_risks(self.model, val_batch, val_envs, self.risk_fn)
            log_entry = {
                "epoch": epoch,
                "method": self.run_cfg.method,
                "dataset": self.run_cfg.dataset,
                "seed": self.run_cfg.seed,
                "train_cvar95": float(torch.quantile(pnl, 0.05)),
                "train_mean_pnl": float(pnl.mean()),
                "train_turnover": float(actions.abs().mean()),
                "val_cvar95": float(torch.stack(list(val_risks.values())).mean().item()),
                "val_mean_pnl": float(val_pnl.mean()),
                "val_turnover": float(val_actions.abs().mean()),
            }
            for env in torch.unique(env_tensor).tolist():
                env_mask = env_tensor == env
                env_pnl = pnl[env_mask]
                if env_pnl.numel() > 0:
                    log_entry[f"train_cvar95_env_{int(env)}"] = float(torch.quantile(env_pnl, 0.05))
            for env in torch.unique(val_env_tensor).tolist():
                env_mask = val_env_tensor == env
                env_pnl = val_pnl[env_mask]
                if env_pnl.numel() > 0:
                    log_entry[f"val_cvar95_env_{int(env)}"] = float(torch.quantile(env_pnl, 0.05))
            logs.append(log_entry)
        return logs

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def risk_eval(self, batch: Dict[str, torch.Tensor], env_ids: torch.Tensor):
        return compute_env_risks(self.model, batch, env_ids, self.risk_fn)


@register_method("erm")
def build_erm_trainer(run_cfg: ExperimentRunConfig) -> Trainer:
    return BasicTrainer(run_cfg)


@register_method("irm")
def build_irm_trainer(run_cfg: ExperimentRunConfig) -> Trainer:
    run_cfg.config.objective.name = "irm"
    return BasicTrainer(run_cfg)


@register_method("groupdro")
def build_groupdro_trainer(run_cfg: ExperimentRunConfig) -> Trainer:
    run_cfg.config.objective.name = "group_dro"
    return BasicTrainer(run_cfg)


@register_method("vrex")
def build_vrex_trainer(run_cfg: ExperimentRunConfig) -> Trainer:
    run_cfg.config.objective.name = "vrex"
    return BasicTrainer(run_cfg)


@register_method("hirm")
def build_hirm_trainer(run_cfg: ExperimentRunConfig) -> Trainer:
    run_cfg.config.objective.name = "hirm"
    return BasicTrainer(run_cfg)


@register_method("risk_parity")
def build_risk_parity_trainer(run_cfg: ExperimentRunConfig) -> Trainer:
    return RuleBasedTrainer(run_cfg)


__all__ = [
    "register_method",
    "get_method_builder",
    "list_methods",
    "BasicTrainer",
]
