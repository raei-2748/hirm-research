"""Synthetic environment backed by simplified Heston dynamics."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Sequence

from hirm.envs.base import Episode, Environment
from hirm.envs.regime_labelling import DEFAULT_REGIME_THRESHOLDS, compute_realized_vol, label_regimes

EnvConfig = Mapping[str, float | str]


class SyntheticHestonEnv(Environment):
    """Generate Heston price paths grouped into volatility regimes."""

    def __init__(
        self,
        horizon: int = 60,
        dt: float = 1.0 / 252.0,
        start_price: float = 100.0,
        n_paths_per_env: int = 64,
        thresholds: Mapping[str, float] | None = None,
        split_configs: Mapping[str, Sequence[EnvConfig]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.start_price = start_price
        self.n_paths_per_env = n_paths_per_env
        self.thresholds = dict(DEFAULT_REGIME_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.rng = random.Random(seed)

        if split_configs is None:
            split_configs = self._default_split_configs()
        self._split_configs = split_configs
        super().__init__(
            split_env_ids={split: [cfg["env_id"] for cfg in configs] for split, configs in split_configs.items()}
        )
        self._episode_pools: Dict[str, Dict[str, List[Episode]]] = {
            split: {env_cfg["env_id"]: [] for env_cfg in configs}
            for split, configs in split_configs.items()
        }
        self._populate_episode_pools()

    def _default_split_configs(self) -> Dict[str, Sequence[EnvConfig]]:
        return {
            "train": [
                {
                    "env_id": "train_low",
                    "mu": 0.03,
                    "kappa": 2.0,
                    "theta": 0.02,
                    "sigma": 0.3,
                    "rho": -0.4,
                    "v0": 0.02,
                },
                {
                    "env_id": "train_medium",
                    "mu": 0.05,
                    "kappa": 1.5,
                    "theta": 0.04,
                    "sigma": 0.4,
                    "rho": -0.5,
                    "v0": 0.04,
                },
            ],
            "val": [
                {
                    "env_id": "val_high",
                    "mu": 0.00,
                    "kappa": 1.8,
                    "theta": 0.06,
                    "sigma": 0.5,
                    "rho": -0.6,
                    "v0": 0.05,
                }
            ],
            "test": [
                {
                    "env_id": "test_crisis",
                    "mu": -0.02,
                    "kappa": 1.2,
                    "theta": 0.09,
                    "sigma": 0.7,
                    "rho": -0.7,
                    "v0": 0.08,
                }
            ],
        }

    def _populate_episode_pools(self) -> None:
        for split, configs in self._split_configs.items():
            for config in configs:
                env_id = str(config["env_id"])
                for _ in range(self.n_paths_per_env):
                    prices = self._simulate_path(config)
                    window = min(20, max(2, len(prices) - 1))
                    realized_vol = compute_realized_vol(prices, window=window)
                    regimes = label_regimes(realized_vol, thresholds=self.thresholds)
                    episode = Episode(
                        prices=prices,
                        states=[[] for _ in range(len(prices))],
                        pnl=0.0,
                        env_id=env_id,
                        meta={
                            "split": split,
                            "regimes": regimes,
                            "parameters": dict(config),
                            "realized_vol": realized_vol,
                        },
                    )
                    self._episode_pools[split][env_id].append(episode)

    def _simulate_path(self, params: EnvConfig) -> List[float]:
        s_path: List[float] = [self.start_price]
        v_path: List[float] = [float(params["v0"])]
        for _ in range(self.horizon):
            z1 = self.rng.gauss(0.0, 1.0)
            z2 = self.rng.gauss(0.0, 1.0)
            rho = float(params["rho"])
            kappa = float(params["kappa"])
            theta = float(params["theta"])
            sigma = float(params["sigma"])
            mu = float(params["mu"])

            v_prev = max(v_path[-1], 0.0)
            dw_v = math.sqrt(self.dt) * z1
            dw_s = math.sqrt(self.dt) * (rho * z1 + math.sqrt(max(1 - rho**2, 1e-8)) * z2)
            v_next = v_prev + kappa * (theta - v_prev) * self.dt + sigma * math.sqrt(max(v_prev, 1e-8)) * dw_v
            v_next = max(v_next, 1e-8)
            drift = (mu - 0.5 * v_prev) * self.dt
            diffusion = math.sqrt(max(v_prev, 1e-8)) * dw_s
            s_next = s_path[-1] * math.exp(drift + diffusion)
            s_path.append(max(s_next, 1e-8))
            v_path.append(v_next)
        return s_path

    def sample_episode(self, split: str = "train") -> Episode:
        env_ids = self.available_env_ids(split)
        if not env_ids:
            raise ValueError(f"Split '{split}' does not have any configured environments")
        env_id = env_ids[self.rng.randrange(len(env_ids))]
        candidates = self._episode_pools[split][env_id]
        if not candidates:
            raise RuntimeError(f"No precomputed episodes for env_id={env_id}")
        episode = candidates[self.rng.randrange(len(candidates))]
        return self._clone_episode(episode)


__all__ = ["SyntheticHestonEnv"]
