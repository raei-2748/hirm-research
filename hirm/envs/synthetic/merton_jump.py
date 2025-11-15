"""Merton jump-diffusion environment (simplified)."""
# Provides a jump-diffusion stress-testing alternative aligned with the HIRM
# paper's synthetic regime descriptions.
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..base import Env
from ..regimes import map_vol_to_regime


class MertonJumpEnv(Env):
    """Simplified GBM with deterministic Poisson jumps."""

    def __init__(
        self,
        s0: float = 100.0,
        mu: float = 0.03,
        sigma: float = 0.2,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.02,
        jump_std: float = 0.05,
        horizon: int = 252,
        dt: float = 1.0 / 252.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(horizon=horizon, seed=seed)
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.jump_intensity = float(jump_intensity)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)
        self.dt = float(dt)
        self._price = self.s0
        self._t = 0
        self._returns: list[float] = []

    def seed(self, seed: int) -> None:
        super().seed(seed)
        self._price = self.s0
        self._t = 0
        self._returns = []

    def reset(self) -> Dict[str, Any]:
        self._price = self.s0
        self._t = 0
        self._returns = []
        obs = {"price": self._price, "t": self._t}
        info = {
            "t": self._t,
            "price": self._price,
            "realized_vol_20": 0.0,
            "regime": -1,
        }
        return {"obs": obs, "info": info}

    def _realized_vol(self) -> float:
        if not self._returns:
            return 0.0
        arr = np.array(self._returns[-20:], dtype=float)
        return np.sqrt(252.0) * float(np.std(arr, ddof=0))

    def step(self, action: float | None = None) -> Dict[str, Any]:
        del action
        if self._t >= self.horizon:
            raise RuntimeError("Episode finished")
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * self.rng.standard_normal()
        num_jumps = self.rng.poisson(self.jump_intensity * self.dt)
        jump_component = 0.0
        if num_jumps > 0:
            jumps = self.rng.normal(self.jump_mean, self.jump_std, size=num_jumps)
            jump_component = np.sum(jumps)
        log_return = drift + diffusion + jump_component
        reward = log_return
        self._returns.append(reward)
        self._price *= float(np.exp(log_return))
        self._t += 1
        vol = self._realized_vol()
        regime = map_vol_to_regime(vol if np.isfinite(vol) else 0.0)
        obs = {"price": self._price, "t": self._t}
        info = {
            "t": self._t,
            "price": self._price,
            "realized_vol_20": vol,
            "regime": regime,
            "return_type": "log",
        }
        done = self._t >= self.horizon
        return {"obs": obs, "reward": reward, "price": self._price, "done": done, "info": info}


__all__ = ["MertonJumpEnv"]
