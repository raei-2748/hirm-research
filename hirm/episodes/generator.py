"""Episode generation utilities."""
from __future__ import annotations

from typing import List

import numpy as np

from hirm.envs.base import Env
from hirm.episodes.episode import Episode


def generate_episodes(env: Env, num_episodes: int, horizon: int, seed: int) -> List[Episode]:
    """Generate ``num_episodes`` deterministic episodes from ``env``."""

    env.seed(seed)
    episodes: List[Episode] = []
    for _ in range(num_episodes):
        reset_out = env.reset()
        info = reset_out.get("info", {})
        price_list = [float(info.get("price", reset_out["obs"].get("price", 0.0)))]
        regimes = [int(info.get("regime", -1))]
        vol = [float(info.get("realized_vol_20", np.nan))]
        times = [int(info.get("t", 0))]
        returns = []
        for _step in range(horizon):
            result = env.step(0.0)
            returns.append(float(result.get("reward", 0.0)))
            price_list.append(float(result["price"]))
            info = result.get("info", {})
            regimes.append(int(info.get("regime", -1)))
            vol.append(float(info.get("realized_vol_20", np.nan)))
            times.append(int(info.get("t", times[-1] + 1)))
            if result.get("done"):
                break
        prices_arr = np.asarray(price_list, dtype=float)
        returns_arr = np.asarray(returns, dtype=float)
        metadata = {
            "regimes": np.asarray(regimes[: prices_arr.size], dtype=int),
            "vol_20": np.asarray(vol[: prices_arr.size], dtype=float),
            "t": np.asarray(times[: prices_arr.size], dtype=int),
        }
        episodes.append(Episode(prices=prices_arr, returns=returns_arr, metadata=metadata))
    return episodes


__all__ = ["generate_episodes"]
