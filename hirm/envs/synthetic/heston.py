"""Heston stochastic volatility environment."""
# Implements the synthetic stochastic-volatility process described in the HIRM
# paper, enabling deterministic experiments with controllable risk regimes.
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..base import Env
from ..regimes import map_vol_to_regime


class HestonEnv(Env):
    """Discrete-time Heston model with deterministic seeding."""

    def __init__(
        self,
        s0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.0,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.5,
        rho: float = -0.7,
        horizon: int = 252,
        dt: float = 1.0 / 252.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(horizon=horizon, seed=seed)
        self.s0 = float(s0)
        self.v0 = float(v0)
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.dt = float(dt)
        self._price = self.s0
        self._var = self.v0
        self._t = 0
        self._returns: list[float] = []

    def seed(self, seed: int) -> None:
        super().seed(seed)
        self._price = self.s0
        self._var = self.v0
        self._t = 0
        self._returns = []

    def reset(self) -> Dict[str, Any]:
        self._price = self.s0
        self._var = max(self.v0, 1e-8)
        self._t = 0
        self._returns = []
        obs = {
            "price": self._price,
            "instantaneous_var": self._var,
            "t": self._t,
        }
        info = {
            "t": self._t,
            "price": self._price,
            "instantaneous_var": self._var,
            "realized_vol_20": 0.0,
            "regime": -1,
        }
        return {"obs": obs, "info": info}

    def _compute_realized_vol(self) -> float:
        if not self._returns:
            return 0.0
        window = self._returns[-20:]
        arr = np.array(window, dtype=float)
        vol = np.sqrt(252.0) * float(np.std(arr, ddof=0))
        return vol

    def step(self, action: float | None = None) -> Dict[str, Any]:
        del action
        if self._t >= self.horizon:
            raise RuntimeError("Episode already finished; call reset().")
        z1 = self.rng.standard_normal()
        z2 = self.rng.standard_normal()
        z2 = self.rho * z1 + np.sqrt(max(0.0, 1.0 - self.rho**2)) * z2
        v_t = max(self._var, 0.0)
        sqrt_v = np.sqrt(v_t)
        v_next = v_t + self.kappa * (self.theta - v_t) * self.dt
        v_next += self.sigma * sqrt_v * np.sqrt(self.dt) * z2
        v_next = float(max(v_next, 1e-8))
        log_return = (
            (self.mu - 0.5 * v_t) * self.dt + sqrt_v * np.sqrt(self.dt) * z1
        )
        s_next = float(self._price * np.exp(log_return))
        reward = log_return
        self._returns.append(reward)
        vol = self._compute_realized_vol()
        regime = map_vol_to_regime(vol if np.isfinite(vol) else 0.0)
        self._price = s_next
        self._var = v_next
        self._t += 1
        obs = {
            "price": self._price,
            "instantaneous_var": self._var,
            "t": self._t,
        }
        info = {
            "t": self._t,
            "price": self._price,
            "instantaneous_var": self._var,
            "realized_vol_20": vol,
            "regime": regime,
            "return_type": "log",
        }
        done = self._t >= self.horizon
        return {
            "obs": obs,
            "reward": reward,
            "price": self._price,
            "done": done,
            "info": info,
        }


__all__ = ["HestonEnv"]
