"""Dataset registry for reproducible Phase 7 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from hirm.data.loader import load_raw_spy
from hirm.data.synthetic import build_synthetic_dataset
from hirm.envs.regime_labelling import price_series_volatility
from hirm.envs.regimes import REGIME_NAMES, map_vol_to_regime

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


def _parse_date(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(str(value))


def _load_spy_series(cfg: Mapping[str, object]):
    env_cfg = getattr(cfg, "real_spy", {}) if not isinstance(cfg, Mapping) else cfg.get("real_spy", {})
    data_cfg = env_cfg.get("data", {}) if isinstance(env_cfg, Mapping) else getattr(env_cfg, "data", {})
    prices_path = data_cfg.get("prices_path", "data/processed/spy_prices.csv")
    regimes_path = data_cfg.get("regimes_path", "data/processed/spy_regimes.txt")
    horizon = int(data_cfg.get("horizon", 60))

    raw = load_raw_spy(prices_path)
    dates = [row[0] for row in raw]
    prices = np.asarray([row[1] for row in raw], dtype=float)
    _, price_vol = price_series_volatility(prices)
    regimes: np.ndarray
    path = None
    try:
        path = regimes_path
        with open(regimes_path, "r", encoding="utf-8") as handle:
            regimes = np.asarray([int(line.strip()) for line in handle if line.strip() != ""], dtype=int)
    except FileNotFoundError:
        regimes = np.asarray([map_vol_to_regime(float(v)) for v in price_vol], dtype=int)
    if regimes.shape[0] != prices.shape[0]:
        regimes = np.resize(regimes, prices.shape[0])
    returns = np.diff(np.log(prices))
    return {
        "dates": dates,
        "prices": prices,
        "price_vol": price_vol,
        "regimes": regimes,
        "returns": returns,
        "horizon": horizon,
    }


def _episode_start_indices(dates: Sequence[datetime], horizon: int, start: datetime, end: datetime) -> list[int]:
    bound_start = _parse_date(start)
    bound_end = _parse_date(end)
    max_start = len(dates) - (horizon + 1)
    idxs: list[int] = []
    for idx in range(max_start + 1):
        if bound_start <= dates[idx] <= bound_end:
            idxs.append(idx)
    return idxs


def _regime_ids(values: Sequence[int | str]) -> list[int]:
    ids: list[int] = []
    for value in values:
        if isinstance(value, str):
            match = [rid for rid, name in REGIME_NAMES.items() if name == value]
            ids.extend(match or [int(value) if str(value).isdigit() else 0])
        else:
            ids.append(int(value))
    return ids


def _build_spy_environment(
    *,
    name: str,
    env_id: int,
    indices: Sequence[int],
    prices: np.ndarray,
    returns: np.ndarray,
    price_vol: np.ndarray,
    regimes: np.ndarray,
    horizon: int,
    action_dim: int,
    generator: torch.Generator,
    max_samples: int | None = None,
) -> Dict[str, torch.Tensor]:
    if not indices:
        return {}
    idx_array = torch.tensor(indices)
    perm = torch.randperm(idx_array.shape[0], generator=generator)
    idx_array = idx_array[perm]
    if max_samples is not None:
        idx_array = idx_array[: int(max_samples)]

    features: list[list[float]] = []
    hedge_returns: list[list[float]] = []
    base_pnl: list[float] = []
    env_ids: list[int] = []

    for start_idx in idx_array.tolist():
        for step in range(horizon):
            day = start_idx + step
            if day >= returns.shape[0]:
                break
            ret = float(returns[day])
            vol = float(price_vol[day])
            regime = int(regimes[day]) if day < regimes.shape[0] else 0
            one_hot = [1.0 if regime == rid else 0.0 for rid in sorted(REGIME_NAMES.keys())]
            t_frac = float(step) / float(horizon)
            features.append([vol, *one_hot, t_frac])
            hedge_row = [ret]
            if action_dim > 1:
                hedge_row.append(-ret)
                if action_dim > 2:
                    hedge_row.extend([0.0] * (action_dim - 2))
            hedge_returns.append(hedge_row[:action_dim])
            base_pnl.append(ret)
            env_ids.append(env_id)

    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "hedge_returns": torch.tensor(hedge_returns, dtype=torch.float32),
        "base_pnl": torch.tensor(base_pnl, dtype=torch.float32),
        "env_ids": torch.tensor(env_ids, dtype=torch.long),
    }


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
    env_settings = _load_spy_series(cfg)
    action_dim = int(getattr(getattr(cfg, "env", None), "action_dim", 2) or 2)
    band_sizes = _band_splits(cfg, split)
    envs: Dict[str, Dict[str, torch.Tensor]] = {}
    generator = torch.Generator().manual_seed(seed + (0 if split == "train" else 1))

    data_cfg = getattr(cfg, "real_spy", {}) if not isinstance(cfg, Mapping) else cfg.get("real_spy", {})
    split_cfg = data_cfg.get("splits", {}) if isinstance(data_cfg, Mapping) else getattr(data_cfg, "splits", {})
    crisis_windows = data_cfg.get("crisis_windows", {}) if isinstance(data_cfg, Mapping) else getattr(data_cfg, "crisis_windows", {})

    def _range_for_split(name: str, default_start: str, default_end: str):
        cfg_entry = split_cfg.get(name, {}) if isinstance(split_cfg, dict) else {}
        start = cfg_entry.get("start", default_start)
        end = cfg_entry.get("end", default_end)
        regimes = cfg_entry.get("regimes", [])
        return start, end, regimes

    prices = env_settings["prices"]
    returns = env_settings["returns"]
    price_vol = env_settings["price_vol"]
    regimes = env_settings["regimes"]
    dates = env_settings["dates"]
    horizon = int(env_settings["horizon"])
    env_id_counter = 0

    def _build_band(name: str, regime_filter: Sequence[int], start: str, end: str):
        nonlocal env_id_counter
        start_indices = _episode_start_indices(dates, horizon, _parse_date(start), _parse_date(end))
        start_indices = [i for i in start_indices if regimes[i] in regime_filter]
        batch = _build_spy_environment(
            name=name,
            env_id=env_id_counter,
            indices=start_indices,
            prices=prices,
            returns=returns,
            price_vol=price_vol,
            regimes=regimes,
            horizon=horizon,
            action_dim=action_dim,
            generator=generator,
            max_samples=band_sizes.get(name),
        )
        if batch:
            envs[name] = batch
            env_id_counter += 1

    if split == "train":
        start, end, regimes_cfg = _range_for_split("train", "2016-01-01", "2019-12-31")
        allowed = _regime_ids(regimes_cfg) or [0, 1]
        _build_band("low", [0], start, end)
        _build_band("medium", [1], start, end)
        if not envs:
            _build_band("train", allowed, start, end)
    elif split == "val":
        start, end, regimes_cfg = _range_for_split("val", "2018-09-01", "2018-12-31")
        allowed = _regime_ids(regimes_cfg) or [2]
        _build_band("high", allowed, start, end)
    else:
        start, end, regimes_cfg = _range_for_split("test", "2018-01-01", "2023-12-31")
        allowed = _regime_ids(regimes_cfg) or [0, 1, 2, 3]
        _build_band("crisis", [3], start, end)
        _build_band("high", [2], start, end)
        crisis_items = crisis_windows.items() if isinstance(crisis_windows, Mapping) else []
        for crisis_name, (c_start, c_end) in crisis_items:
            window_indices = _episode_start_indices(dates, horizon, _parse_date(c_start), _parse_date(c_end))
            window_indices = [i for i in window_indices if regimes[i] in allowed]
            batch = _build_spy_environment(
                name=crisis_name,
                env_id=env_id_counter,
                indices=window_indices,
                prices=prices,
                returns=returns,
                price_vol=price_vol,
                regimes=regimes,
                horizon=horizon,
                action_dim=action_dim,
                generator=generator,
                max_samples=band_sizes.get("crisis"),
            )
            if batch:
                envs[crisis_name] = batch
                env_id_counter += 1
        if not envs:
            _build_band("test", allowed, start, end)

    return ExperimentDataset(environments=envs)


__all__ = [
    "ExperimentDataset",
    "DatasetBuilder",
    "register_dataset",
    "get_dataset_builder",
    "list_datasets",
    "build_synthetic_heston_dataset",
    "build_real_spy_dataset",
]
