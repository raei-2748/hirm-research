"""Synthetic environment placeholder."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from hirm.envs.base import BaseEnv


class SyntheticHestonEnv(BaseEnv):
    """Brownian motion placeholder for later stochastic models."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        episode_length: int,
        drift: float,
        volatility: float,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, episode_length=episode_length)
        self.drift = drift
        self.volatility = volatility
        self.rng = random.Random(seed)
        self.inventory = 0.0

    def _initial_state(self) -> List[float]:
        self.inventory = 0.0
        return [0.0 for _ in range(self.state_dim)]

    def _transition(self, action: Sequence[float]) -> tuple[List[float], float, Dict[str, float]]:
        noise = [self.rng.gauss(self.drift, self.volatility) for _ in range(self.state_dim)]
        state = [self._state[idx] + float(noise[idx]) for idx in range(self.state_dim)]
        self.inventory += float(sum(action))
        reward = float(state[0]) - 0.001 * self.inventory
        mean_noise = sum(noise) / len(noise)
        variance = sum((value - mean_noise) ** 2 for value in noise) / max(len(noise), 1)
        realized_vol = variance ** 0.5
        info: Dict[str, float] = {
            "returns": float(state[0]),
            "realized_vol": float(realized_vol),
            "inventory": float(self.inventory),
        }
        return state, reward, info


__all__ = ["SyntheticHestonEnv"]
