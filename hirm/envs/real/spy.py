"""SPY historical replay environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ...data.loader import load_raw_spy
from ..base import Env
from ..regime_labelling import label_series_with_regimes, price_series_volatility


@dataclass
class _EpisodeWindow:
    start: int
    end: int


class SPYEnv(Env):
    """Environment that replays historical SPY data."""

    def __init__(
        self,
        csv_path: str = "data/raw/spy.csv",
        horizon: int = 252,
        start_index: Optional[int] = None,
        price_col: Optional[str] = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(horizon=horizon, seed=seed)
        self.csv_path = csv_path
        self.price_col = price_col
        self._dates, self._prices = self._load_prices()
        _, price_vol = price_series_volatility(self._prices)
        self._price_vol = price_vol
        self._regimes = label_series_with_regimes(self._prices)
        self._returns = np.diff(np.log(self._prices))
        if self._returns.size == 0:
            raise ValueError("SPY price series must contain at least two entries")
        self._start_override = start_index
        self._window: _EpisodeWindow | None = None
        self._t = 0

    def _load_prices(self):
        rows = load_raw_spy(self.csv_path)
        dates = [row[0] for row in rows]
        prices = np.asarray([row[1] for row in rows], dtype=float)
        return dates, prices

    def seed(self, seed: int) -> None:
        super().seed(seed)
        self._window = None
        self._t = 0

    def _sample_window(self) -> _EpisodeWindow:
        max_start = self._prices.size - (self.horizon + 1)
        if max_start < 0:
            raise ValueError("Horizon longer than available data")
        if self._start_override is not None:
            start = int(self._start_override)
        else:
            start = int(self.rng.integers(0, max_start + 1))
        end = start + self.horizon
        return _EpisodeWindow(start=start, end=end)

    def reset(self) -> Dict[str, Dict[str, float]]:
        self._window = self._sample_window()
        self._t = 0
        start = self._window.start
        end = self._window.end
        self._episode_prices = self._prices[start : end + 1]
        self._episode_vol = self._price_vol[start : end + 1]
        self._episode_regimes = self._regimes[start : end + 1]
        self._episode_dates = self._dates[start : end + 1]
        self._episode_returns = np.diff(np.log(self._episode_prices))
        obs = {
            "price": float(self._episode_prices[0]),
            "t": 0,
        }
        info = {
            "t": 0,
            "price": float(self._episode_prices[0]),
            "realized_vol_20": float(self._episode_vol[0]),
            "regime": int(self._episode_regimes[0]),
            "date": self._episode_dates[0],
        }
        return {"obs": obs, "info": info}

    def step(self, action: float | None = None):
        del action
        if self._window is None:
            raise RuntimeError("Call reset() before stepping")
        if self._t >= self.horizon:
            raise RuntimeError("Episode finished; reset required")
        reward = float(self._episode_returns[self._t])
        self._t += 1
        price = float(self._episode_prices[self._t])
        done = self._t >= self.horizon
        obs = {"price": price, "t": self._t}
        info = {
            "t": self._t,
            "price": price,
            "realized_vol_20": float(self._episode_vol[self._t]),
            "regime": int(self._episode_regimes[self._t]),
            "date": self._episode_dates[self._t],
        }
        return {
            "obs": obs,
            "reward": reward,
            "price": price,
            "done": done,
            "info": info,
        }


__all__ = ["SPYEnv"]
