"""Data loading stubs for Phase 1."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def load_raw_equity_data(path: str | Path) -> Dict[str, List[object]]:
    rows = _read_csv(path)
    dates = [datetime.fromisoformat(row["date"]) for row in rows]
    returns = [float(row["return"]) for row in rows]
    mean_return = sum(returns) / max(len(returns), 1)
    normalized = [value - mean_return for value in returns]
    return {"date": dates, "return": normalized}


def load_processed_features(path: str | Path) -> Dict[str, List[object]]:
    rows = _read_csv(path)
    data: Dict[str, List[float]] = {"return": [], "realized_vol": [], "liquidity": [], "inventory": []}
    dates: List[datetime] = []
    for row in rows:
        dates.append(datetime.fromisoformat(row["date"]))
        for key in data:
            data[key].append(float(row[key]))
    arrays: Dict[str, List[float]] = {"date": dates}
    arrays.update(data)
    return arrays


def load_regime_file(path: str | Path) -> List[float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def merge_by_date(primary: Dict[str, List[object]], secondary: Dict[str, List[object]]) -> Dict[str, List[object]]:
    secondary_dates = set(secondary["date"])
    merged: Dict[str, List[float]] = {key: [] for key in primary}
    for idx, date in enumerate(primary["date"]):
        if date in secondary_dates:
            for key in primary:
                merged[key].append(primary[key][idx])
    return merged


def create_episode_splits(num_steps: int, episode_length: int) -> List[Tuple[int, int]]:
    splits: List[Tuple[int, int]] = []
    for start in range(0, num_steps, episode_length):
        end = min(start + episode_length, num_steps)
        splits.append((start, end))
    return splits


def build_feature_matrix(arrays: Dict[str, List[float]], columns: Sequence[str]) -> List[List[float]]:
    matrix: List[List[float]] = []
    num_rows = len(arrays[columns[0]])
    for row in range(num_rows):
        matrix.append([float(arrays[col][row]) for col in columns])
    return matrix


__all__ = [
    "load_raw_equity_data",
    "load_processed_features",
    "load_regime_file",
    "merge_by_date",
    "create_episode_splits",
    "build_feature_matrix",
]
