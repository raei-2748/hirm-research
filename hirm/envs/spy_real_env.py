"""Environment that serves pre-processed SPY historical windows."""
from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from hirm.envs.base import Episode, Environment
from hirm.envs.regime_labelling import (
    DEFAULT_REGIME_THRESHOLDS,
    REGIME_TO_ID,
    compute_realized_vol,
    label_regimes,
)


class SpyRealEnv(Environment):
    """Construct episodes from (placeholder) SPY historical data."""

    TRAIN_WINDOWS: Sequence[Dict[str, object]] = (
        {"env_id": "train_low", "start": "2017-01-01", "end": "2018-06-30", "regimes": {"low", "medium"}},
        {"env_id": "train_medium", "start": "2018-07-01", "end": "2019-12-31", "regimes": {"low", "medium"}},
    )
    VAL_WINDOWS: Sequence[Dict[str, object]] = (
        {"env_id": "val_2018_high", "start": "2018-09-01", "end": "2018-12-31", "regimes": {"high"}},
    )
    TEST_WINDOWS: Sequence[Dict[str, object]] = (
        {"env_id": "test_2018_crisis", "start": "2018-01-01", "end": "2018-03-31", "regimes": {"crisis"}},
        {"env_id": "test_2020_covid", "start": "2020-02-01", "end": "2020-06-30", "regimes": {"crisis"}},
        {"env_id": "test_2022_selloff", "start": "2022-01-01", "end": "2022-06-30", "regimes": {"crisis"}},
    )

    def __init__(
        self,
        data_path: str | Path = Path("data/processed/spy_prices.csv"),
        horizon: int = 60,
        thresholds: Mapping[str, float] | None = None,
        regime_override: Sequence[int] | None = None,
        regime_path: str | Path | None = None,
        realized_vol_path: str | Path | None = None,
        window: int = 20,
        seed: int | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.horizon = horizon
        self.thresholds = dict(DEFAULT_REGIME_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.window = window
        self.rng = random.Random(seed)

        prices, dates = self._load_prices_and_dates(self.data_path)
        self.prices = prices
        self.dates = dates

        if regime_override is not None:
            regimes = [int(value) for value in regime_override]
            if len(regimes) != len(prices):
                raise ValueError("regime_override must align with price series")
        elif regime_path is not None:
            regimes = self._load_regime_file(Path(regime_path))
            if len(regimes) != len(prices):
                raise ValueError("regime_path must align with price series")
        else:
            realized_vol = self._load_or_compute_realized_vol(prices, realized_vol_path)
            regimes = label_regimes(realized_vol, thresholds=self.thresholds)
        self.regimes = regimes

        split_env_ids = {
            "train": [cfg["env_id"] for cfg in self.TRAIN_WINDOWS],
            "val": [cfg["env_id"] for cfg in self.VAL_WINDOWS],
            "test": [cfg["env_id"] for cfg in self.TEST_WINDOWS],
        }
        super().__init__(split_env_ids=split_env_ids)

        self._episode_pools: Dict[str, Dict[str, List[Episode]]] = {
            split: {env_id: [] for env_id in env_ids}
            for split, env_ids in split_env_ids.items()
        }
        self._build_episode_pools()

    def _load_prices_and_dates(self, data_path: Path) -> tuple[List[float], List[dt.date]]:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file {data_path} not found")
        prices: List[float] = []
        dates: List[dt.date] = []
        with data_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "date" not in reader.fieldnames:
                raise ValueError("CSV must include a 'date' column")
            price_column = "price" if "price" in reader.fieldnames else None
            return_column = "return" if "return" in reader.fieldnames else None
            if price_column is None and return_column is None:
                raise ValueError("CSV must include either 'price' or 'return' column")
            running_price = 100.0
            for row in reader:
                dates.append(self._parse_date(row["date"]))
                if price_column is not None:
                    running_price = float(row[price_column])
                else:
                    running_price *= 1 + float(row[return_column])
                prices.append(running_price)
        return prices, dates

    def _parse_date(self, value: str) -> dt.date:
        return dt.date.fromisoformat(value)

    def _load_or_compute_realized_vol(
        self, prices: List[float], realized_vol_path: str | Path | None
    ) -> List[float]:
        if realized_vol_path:
            path = Path(realized_vol_path)
            if path.exists():
                values = [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(values) == len(prices):
                    return values
        return compute_realized_vol(prices, window=self.window)

    def _load_regime_file(self, path: Path) -> List[int]:
        values: List[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    values.append(int(float(stripped)))
        return values

    def _window_to_indices(self, start: str, end: str) -> List[int]:
        start_dt = dt.date.fromisoformat(start)
        end_dt = dt.date.fromisoformat(end)
        return [idx for idx, date in enumerate(self.dates) if start_dt <= date <= end_dt]

    def _build_episode_pools(self) -> None:
        split_to_windows = {
            "train": self.TRAIN_WINDOWS,
            "val": self.VAL_WINDOWS,
            "test": self.TEST_WINDOWS,
        }
        for split, windows in split_to_windows.items():
            for window_cfg in windows:
                env_id = str(window_cfg["env_id"])
                allowed_regimes = {REGIME_TO_ID[name] for name in window_cfg["regimes"]}
                indices = self._window_to_indices(window_cfg["start"], window_cfg["end"])
                if len(indices) <= self.horizon:
                    continue
                min_idx = indices[0]
                max_idx = indices[-1]
                start_bound = dt.date.fromisoformat(window_cfg["start"])
                end_bound = dt.date.fromisoformat(window_cfg["end"])
                for start_idx in range(min_idx, max_idx - self.horizon + 1):
                    end_idx = start_idx + self.horizon
                    if end_idx > max_idx:
                        break
                    start_date = self.dates[start_idx]
                    end_date = self.dates[end_idx]
                    if not (start_bound <= start_date and end_date <= end_bound):
                        continue
                    window_regimes = self.regimes[start_idx : end_idx + 1]
                    if not all(regime in allowed_regimes for regime in window_regimes):
                        continue
                    prices = self.prices[start_idx : end_idx + 1]
                    episode = Episode(
                        prices=prices,
                        states=[[] for _ in range(len(prices))],
                        pnl=0.0,
                        env_id=env_id,
                        meta={
                            "split": split,
                            "dates": [date.isoformat() for date in self.dates[start_idx : end_idx + 1]],
                            "regimes": window_regimes,
                        },
                    )
                    self._episode_pools[split][env_id].append(episode)
                if not self._episode_pools[split][env_id]:
                    raise RuntimeError(
                        f"No episodes created for {env_id}. Ensure data covers the requested window and regimes."
                    )

    def sample_episode(self, split: str = "train") -> Episode:
        env_ids = self.available_env_ids(split)
        if not env_ids:
            raise ValueError(f"Split '{split}' does not have any configured environments")
        env_id = env_ids[self.rng.randrange(len(env_ids))]
        candidates = self._episode_pools[split][env_id]
        if not candidates:
            raise RuntimeError(f"No cached episodes for env_id={env_id}")
        episode = candidates[self.rng.randrange(len(candidates))]
        return self._clone_episode(episode)


__all__ = ["SpyRealEnv"]
