"""Synthetic volatility-band environments."""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping

from hirm.envs.base import Episode, Environment


class SyntheticVolatilityBandEnv(Environment):
    """Simple GBM-based synthetic environment with volatility bands."""

    DEFAULT_SPLITS = {
        "train": ("train_low", "train_medium"),
        "val": ("val_high",),
        "test": ("test_high", "test_crisis"),
    }

    REGIME_PARAMS = {
        "low": {"drift": 0.05, "vol": 0.10},
        "medium": {"drift": 0.05, "vol": 0.20},
        "high": {"drift": 0.03, "vol": 0.30},
        "crisis": {"drift": -0.10, "vol": 0.50},
    }

    def __init__(
        self,
        horizon: int = 60,
        start_price: float = 100.0,
        dt: float = 1.0 / 252,
        split_env_ids: Mapping[str, Iterable[str]] | None = None,
        seed: int = 0,
        transaction_cost: float = 0.0005,
    ) -> None:
        self._horizon = int(horizon)
        self._start_price = float(start_price)
        self._dt = float(dt)
        self._rng = random.Random(seed)
        self._transaction_cost = float(transaction_cost)
        if split_env_ids is None:
            split_env_ids = self.DEFAULT_SPLITS
        self._split_env_ids = {
            split: list(values)
            for split, values in split_env_ids.items()
        }
        super().__init__(split_env_ids=self._split_env_ids)

    def _env_regime(self, env_id: str) -> str:
        for regime in self.REGIME_PARAMS:
            if regime in env_id:
                return regime
        raise ValueError(f"Cannot infer regime from env_id '{env_id}'")

    def _simulate_episode(self, env_id: str, split: str) -> Episode:
        regime = self._env_regime(env_id)
        params = self.REGIME_PARAMS[regime]
        drift = params["drift"]
        vol = params["vol"]
        prices: List[float] = [self._start_price]
        states: List[List[float]] = []
        price = self._start_price
        for _ in range(self._horizon):
            shock = self._rng.gauss(0.0, math.sqrt(self._dt))
            price *= math.exp((drift - 0.5 * vol**2) * self._dt + vol * shock)
            prices.append(price)
            states.append([price, vol])
        meta = {
            "split": split,
            "regime": regime,
            "transaction_cost": self._transaction_cost,
            "liability": prices[-1],
        }
        return Episode(
            prices=prices,
            states=states,
            pnl=0.0,
            env_id=env_id,
            meta=meta,
        )

    def sample_episode(self, split: str = "train", env_id: str | None = None) -> Episode:
        env_ids = self._split_env_ids.get(split)
        if not env_ids:
            raise ValueError(f"Split '{split}' has no environments")
        if env_id is None:
            env_id = self._rng.choice(env_ids)
        if env_id not in env_ids:
            raise ValueError(f"env_id '{env_id}' not part of split '{split}'")
        return self._simulate_episode(env_id, split)


__all__ = ["SyntheticVolatilityBandEnv"]
