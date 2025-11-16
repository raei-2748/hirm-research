"""Feature preprocessing pipeline for Phase 3."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hirm.episodes.episode import Episode
from hirm.state.features import compute_all_features
from hirm.state.splits import create_episode_splits
from hirm.utils.serialization import load_checkpoint, save_checkpoint


class FeatureScaler:
    """Standardization helper for ``Phi`` features."""

    def __init__(self) -> None:
        self.mean_: List[float] | None = None
        self.std_: List[float] | None = None

    @staticmethod
    def _rows(phi: np.ndarray) -> List[List[float]]:
        if hasattr(phi, "tolist"):
            data = phi.tolist()
        else:
            data = list(phi)
        if not data:
            return []
        if isinstance(data[0], list):
            return [[float(value) for value in row] for row in data]
        return [[float(value) for value in data]]

    def fit(self, phi_list: List[np.ndarray]) -> "FeatureScaler":
        if not phi_list:
            raise ValueError("phi_list must contain at least one episode")
        totals: List[float] | None = None
        totals_sq: List[float] | None = None
        count = 0
        for phi in phi_list:
            rows = self._rows(phi)
            if not rows:
                continue
            if totals is None:
                totals = [0.0] * len(rows[0])
                totals_sq = [0.0] * len(rows[0])
            for row in rows:
                for idx, value in enumerate(row):
                    totals[idx] += value
                    totals_sq[idx] += value * value
            count += len(rows)
        if totals is None or totals_sq is None or count == 0:
            raise ValueError("Phi arrays must contain at least one value")
        means = [total / count for total in totals]
        variances = [max(total_sq / count - mean ** 2, 0.0) for total_sq, mean in zip(totals_sq, means)]
        std = [max(math.sqrt(var), 1e-8) for var in variances]
        self.mean_ = means
        self.std_ = std
        return self

    def transform(self, phi: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform")
        rows = self._rows(phi)
        if not rows:
            return np.asarray([], dtype=float)
        transformed = []
        for row in rows:
            transformed.append(
                [
                    (value - self.mean_[idx]) / self.std_[idx]
                    for idx, value in enumerate(row)
                ]
            )
        return np.asarray(transformed, dtype="object")

    def fit_transform(self, phi_list: List[np.ndarray]) -> List[np.ndarray]:
        self.fit(phi_list)
        return [self.transform(phi) for phi in phi_list]

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before saving")
        save_checkpoint({"mean": self.mean_, "std": self.std_}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        payload = load_checkpoint(path)
        scaler = cls()
        scaler.mean_ = [float(x) for x in payload["mean"]]
        scaler.std_ = [float(x) for x in payload["std"]]
        return scaler


def _ensure_output_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _serialize_arrays(arrays: List[np.ndarray]) -> List[Any]:
    serialized: List[Any] = []
    for arr in arrays:
        if hasattr(arr, "tolist"):
            serialized.append(arr.tolist())
        else:
            serialized.append(list(arr))
    return serialized


def _save_split(split_data: Dict[str, List[np.ndarray]], split_name: str, output_dir: Path) -> None:
    file_path = output_dir / f"{split_name}_features.pkl"
    payload = {
        "phi": _serialize_arrays(split_data["phi"]),
        "r": _serialize_arrays(split_data["r"]),
    }
    save_checkpoint(payload, file_path)


def preprocess_episodes(
    episodes: List[Episode],
    env_ids: List[int],
    config: Dict[str, Any] | None = None,
    split_seed: int = 123,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Compute features, scale ``Phi`` using train statistics, and persist outputs."""

    if len(episodes) != len(env_ids):
        raise ValueError("episodes and env_ids must have the same length")
    num_episodes = len(episodes)
    config = config or {}
    split_config = config.get("splits", {})
    splits = create_episode_splits(
        num_episodes,
        train_frac=split_config.get("train_frac", 0.7),
        val_frac=split_config.get("val_frac", 0.15),
        test_frac=split_config.get("test_frac", 0.15),
        seed=split_seed,
    )
    output: Dict[str, Dict[str, List[np.ndarray]]] = {
        "train": {"phi": [], "r": []},
        "val": {"phi": [], "r": []},
        "test": {"phi": [], "r": []},
    }
    episode_features: List[Dict[str, np.ndarray]] = []
    for episode, env_id in zip(episodes, env_ids):
        episode_features.append(
            compute_all_features(episode, env_id, config)
        )
    index_to_split = {}
    for split_name, indices in splits.items():
        for idx in indices:
            index_to_split[idx] = split_name
    for idx, feat in enumerate(episode_features):
        split_name = index_to_split.get(idx, "train")
        output[split_name]["phi"].append(feat["phi"])
        output[split_name]["r"].append(feat["r"])
    train_phi = output["train"]["phi"]
    if not train_phi:
        raise ValueError("Train split must contain at least one episode")
    scaler = FeatureScaler()
    scaler.fit(train_phi)
    for split_name in output:
        output[split_name]["phi"] = [scaler.transform(phi) for phi in output[split_name]["phi"]]
    output_dir = _ensure_output_dir(config.get("output_dir", "data/processed/features"))
    scaler.save(output_dir / "phi_scaler.pkl")
    for split_name, split_data in output.items():
        _save_split(split_data, split_name, output_dir)
    return output


__all__ = ["FeatureScaler", "preprocess_episodes"]
