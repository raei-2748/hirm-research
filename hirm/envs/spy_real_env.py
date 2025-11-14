"""Skeleton environment for SPY data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from hirm.envs.base import BaseEnv
from hirm.state import preprocess


class SpyRealEnv(BaseEnv):
    """Environment backed by processed SPY observations."""

    def __init__(
        self,
        data_path: str | Path,
        state_dim: int,
        action_dim: int,
        episode_length: int,
        feature_columns: Sequence[str],
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, episode_length=episode_length)
        arrays = preprocess.load_processed_features(data_path)
        self.feature_columns = list(feature_columns)
        if len(self.feature_columns) != self.state_dim:
            raise ValueError("state_dim must match number of feature columns")
        self.features: List[List[float]] = []
        for idx in range(len(arrays["return"])):
            row = [float(arrays[col][idx]) for col in self.feature_columns]
            self.features.append(row)
        self.returns = [float(v) for v in arrays["return"]]
        self.cursor = 0
        if len(self.features) < self.episode_length:
            raise ValueError("Not enough data for requested episode length")

    def _initial_state(self) -> List[float]:
        self.cursor = 0
        return list(self.features[self.cursor])

    def _transition(self, action: Sequence[float]) -> tuple[List[float], float, Dict[str, float]]:
        squared = sum(float(a) ** 2 for a in action)
        norm = squared ** 0.5
        reward = float(self.returns[self.cursor]) - 0.01 * norm
        info: Dict[str, float] = {
            "returns": float(self.returns[self.cursor]),
            "realized_vol": float(self.features[self.cursor][1]),
            "inventory": float(self.features[self.cursor][-1]),
        }
        state = list(self.features[self.cursor])
        self.cursor = min(self.cursor + 1, len(self.features) - 1)
        return state, reward, info


__all__ = ["SpyRealEnv"]
