from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

from hirm.envs.regime_labelling import DEFAULT_REGIME_THRESHOLDS, REGIME_TO_ID, label_regimes
from hirm.envs.spy_real_env import SpyRealEnv
from hirm.envs.synthetic_heston_env import SyntheticHestonEnv


def test_regime_labelling_matches_thresholds() -> None:
    realized_vol = [0.05, 0.12, 0.3, 0.55]
    labels = label_regimes(realized_vol, thresholds=DEFAULT_REGIME_THRESHOLDS)
    expected = [
        REGIME_TO_ID["low"],
        REGIME_TO_ID["medium"],
        REGIME_TO_ID["high"],
        REGIME_TO_ID["crisis"],
    ]
    assert labels == expected


def test_synthetic_env_sampling_split_consistency() -> None:
    env = SyntheticHestonEnv(horizon=10, n_paths_per_env=4, seed=123)
    episode = env.sample_episode("train")
    assert len(episode.prices) == 11
    assert len(episode.states) == 11

    val_episodes = env.sample_episodes(3, split="val")
    assert len(val_episodes) == 3
    for ep in val_episodes:
        assert ep.env_id in env.split_env_ids["val"]


def _create_dummy_spy_dataset(tmp_path: Path) -> tuple[Path, list[int]]:
    start = dt.date(2017, 1, 1)
    total_days = (dt.date(2023, 7, 1) - start).days
    dates = [start + dt.timedelta(days=offset) for offset in range(total_days)]
    prices: list[float] = []
    level = 100.0
    for idx in range(total_days):
        daily_return = 0.0002 + 0.01 * math.sin(idx / 25)
        level *= math.exp(daily_return)
        prices.append(level)

    regimes = [REGIME_TO_ID["low"] for _ in dates]
    for idx, current in enumerate(dates):
        if dt.date(2018, 4, 1) <= current < dt.date(2019, 12, 31):
            regimes[idx] = REGIME_TO_ID["medium"]
        if dt.date(2018, 9, 1) <= current < dt.date(2018, 12, 31):
            regimes[idx] = REGIME_TO_ID["high"]
        if dt.date(2018, 1, 1) <= current < dt.date(2018, 3, 31):
            regimes[idx] = REGIME_TO_ID["crisis"]
        if dt.date(2020, 2, 1) <= current < dt.date(2020, 7, 1):
            regimes[idx] = REGIME_TO_ID["crisis"]
        if dt.date(2022, 1, 1) <= current < dt.date(2022, 7, 1):
            regimes[idx] = REGIME_TO_ID["crisis"]

    data_path = tmp_path / "spy_mock.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "price"])
        for date_value, price in zip(dates, prices):
            writer.writerow([date_value.isoformat(), f"{price:.6f}"])
    return data_path, regimes


def test_spy_env_can_sample_all_splits(tmp_path: Path) -> None:
    data_path, regimes = _create_dummy_spy_dataset(tmp_path)
    env = SpyRealEnv(data_path=data_path, horizon=20, regime_override=regimes, seed=0)

    for split in ("train", "val", "test"):
        episode = env.sample_episode(split)
        assert len(episode.prices) == 21
        assert len(episode.states) == 21
        assert episode.meta["split"] == split
