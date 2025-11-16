"""Dataset registry for reproducible Phase 7 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

import torch

from hirm.data.synthetic import build_synthetic_dataset

DatasetBuilder = Callable[[Mapping[str, object], str, int], "ExperimentDataset"]


_DATASET_REGISTRY: Dict[str, DatasetBuilder] = {}


def register_dataset(name: str) -> Callable[[DatasetBuilder], DatasetBuilder]:
    key = name.lower()

    def decorator(fn: DatasetBuilder) -> DatasetBuilder:
        _DATASET_REGISTRY[key] = fn
        return fn

    return decorator


def get_dataset_builder(name: str) -> DatasetBuilder:
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASET_REGISTRY[key]


def list_datasets() -> Iterable[str]:
    return sorted(_DATASET_REGISTRY)


@dataclass
class ExperimentDataset:
    """Container for train/val/test environments."""

    environments: Dict[str, Dict[str, torch.Tensor]]

    def to_device(self, device: torch.device) -> "ExperimentDataset":
        return ExperimentDataset(
            environments={
                name: {k: v.to(device) for k, v in batch.items()}
                for name, batch in self.environments.items()
            }
        )


def _build_env_batch(
    *,
    num_samples: int,
    feature_dim: int,
    action_dim: int,
    env_id: int,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    data = build_synthetic_dataset(
        num_samples=num_samples,
        feature_dim=feature_dim,
        action_dim=action_dim,
        num_envs=1,
        generator=generator,
    )
    env_ids = torch.full_like(data["env_ids"], env_id)
    data["env_ids"] = env_ids
    return data


def _band_splits(cfg: Mapping[str, object], split: str) -> Dict[str, int]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else getattr(cfg, "data", {})
    splits = getattr(data_cfg, "splits", None) if not isinstance(data_cfg, dict) else data_cfg.get("splits", {})
    split_cfg = splits.get(split, {}) if isinstance(splits, dict) else {}
    default = int(getattr(split_cfg, "num_samples", getattr(split_cfg, "dataset_size", 512)) or 512)
    return {
        "low": default,
        "medium": default,
        "high": default,
        "crisis": default,
    }


@register_dataset("synthetic_heston")
def build_synthetic_heston_dataset(cfg: Mapping[str, object], split: str, seed: int) -> ExperimentDataset:
    feature_dim = int(getattr(getattr(cfg, "env", None), "feature_dim", 6) or 6)
    action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)
    sizes = _band_splits(cfg, split)
    envs: Dict[str, Dict[str, torch.Tensor]] = {}
    generator = torch.Generator().manual_seed(seed + (0 if split == "train" else 1))
    for idx, (name, num_samples) in enumerate(sizes.items()):
        envs[name] = _build_env_batch(
            num_samples=int(num_samples),
            feature_dim=feature_dim,
            action_dim=action_dim,
            env_id=idx,
            generator=generator,
        )
    return ExperimentDataset(environments=envs)


@register_dataset("real_spy")
def build_real_spy_dataset(cfg: Mapping[str, object], split: str, seed: int) -> ExperimentDataset:
    # Placeholder implementation using the synthetic generator to maintain determinism.
    # The registry interface makes it easy to swap in the real SPY loader later.
    return build_synthetic_heston_dataset(cfg, split, seed + 100)


__all__ = [
    "ExperimentDataset",
    "DatasetBuilder",
    "register_dataset",
    "get_dataset_builder",
    "list_datasets",
    "build_synthetic_heston_dataset",
    "build_real_spy_dataset",
]
