"""Real SPY volatility-band environment."""
from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path
from typing import Dict, List, Tuple

from hirm.envs.base import Episode, Environment
from hirm.envs.regimes import assign_volatility_bands, compute_realized_volatility


class RealVolatilityBandEnv(Environment):
    """Environment built from historical SPY prices."""

    def __init__(
        self,
        data_path: str | Path,
        horizon: int = 60,
        transaction_cost: float = 0.0005,
        seed: int = 0,
    ) -> None:
        self._data_path = Path(data_path)
        if not self._data_path.exists():
            raise FileNotFoundError(self._data_path)
        self._horizon = int(horizon)
        self._transaction_cost = float(transaction_cost)
        self._rng = random.Random(seed)
        dates, prices = self._load_prices()
        self._dates = dates
        self._prices = prices
        returns = [0.0]
        for idx in range(1, len(prices)):
            prev = prices[idx - 1]
            curr = prices[idx]
            returns.append((curr - prev) / prev)
        realized = compute_realized_volatility(returns)
        self._realized_vol = realized
        self._regimes = assign_volatility_bands(realized)
        self._episodes_by_env = self._prepare_episodes()
        split_env_ids = {
            split: list(sorted(env_dict.keys()))
            for split, env_dict in self._episodes_by_env.items()
        }
        super().__init__(split_env_ids=split_env_ids)

    def _load_prices(self) -> Tuple[List[dt.date], List[float]]:
        dates: List[dt.date] = []
        prices: List[float] = []
        with self._data_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "date" not in reader.fieldnames or "price" not in reader.fieldnames:
                raise ValueError("CSV must contain 'date' and 'price' columns")
            for row in reader:
                dates.append(dt.date.fromisoformat(row["date"]))
                prices.append(float(row["price"]))
        return dates, prices

    @staticmethod
    def _split_for_date(date_value: dt.date) -> str | None:
        if dt.date(2018, 9, 1) <= date_value <= dt.date(2018, 12, 31):
            return "val"
        if dt.date(2018, 2, 1) <= date_value <= dt.date(2018, 3, 31):
            return "test"
        if dt.date(2020, 3, 1) <= date_value <= dt.date(2020, 6, 30):
            return "test"
        if dt.date(2022, 1, 1) <= date_value <= dt.date(2022, 7, 31):
            return "test"
        if dt.date(2017, 1, 1) <= date_value <= dt.date(2019, 12, 31):
            return "train"
        return None

    def _prepare_episodes(self) -> Dict[str, Dict[str, List[Episode]]]:
        episodes: Dict[str, Dict[str, List[Episode]]] = {"train": {}, "val": {}, "test": {}}
        for start_idx in range(0, len(self._prices) - self._horizon - 1):
            split = self._split_for_date(self._dates[start_idx])
            regime = self._regimes[start_idx]
            if split is None or regime == "unknown":
                continue
            if split == "train" and regime not in {"low", "medium"}:
                continue
            if split == "val" and regime != "high":
                continue
            if split == "test" and regime not in {"high", "crisis"}:
                continue
            prices = self._prices[start_idx : start_idx + self._horizon + 1]
            realized_window = self._realized_vol[start_idx : start_idx + self._horizon]
            if len(realized_window) < self._horizon:
                continue
            if any(val != val for val in realized_window):  # filter NaNs
                continue
            states = [
                [prices[idx], realized_window[idx]]
                for idx in range(self._horizon)
            ]
            env_id = f"{split}_{regime}"
            meta = {
                "split": split,
                "regime": regime,
                "start_date": self._dates[start_idx].isoformat(),
                "liability": prices[-1],
                "transaction_cost": self._transaction_cost,
            }
            episode = Episode(
                prices=list(prices),
                states=states,
                pnl=0.0,
                env_id=env_id,
                meta=meta,
            )
            episodes.setdefault(split, {}).setdefault(env_id, []).append(episode)
        return episodes

    def sample_episode(self, split: str = "train", env_id: str | None = None) -> Episode:
        env_dict = self._episodes_by_env.get(split)
        if not env_dict:
            raise ValueError(f"No episodes for split '{split}'")
        env_ids = list(env_dict.keys())
        if env_id is None:
            env_id = self._rng.choice(env_ids)
        if env_id not in env_dict:
            raise ValueError(f"Unknown env_id '{env_id}' for split '{split}'")
        return self._rng.choice(env_dict[env_id])


__all__ = ["RealVolatilityBandEnv"]
