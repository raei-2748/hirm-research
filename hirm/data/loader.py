"""Data loading helpers without external dependencies."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from hirm.episodes.episode import Episode

DatePriceSeries = List[Tuple[datetime, float]]


def load_raw_spy(csv_path: str = "data/raw/spy.csv") -> DatePriceSeries:
    """Load SPY CSV data as a list of (date, price) tuples sorted by date."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"SPY CSV not found: {csv_path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        price_col = None
        for candidate in ("Adj Close", "Adj_Close", "AdjClose", "Close", "price"):
            if candidate in headers:
                price_col = candidate
                break
        if price_col is None:
            raise ValueError("Could not locate price column in CSV")
        date_col = "Date" if "Date" in headers else headers[0]
        rows: DatePriceSeries = []
        for row in reader:
            date_str = row.get(date_col, "")
            price_str = row.get(price_col, "")
            if not date_str or not price_str:
                continue
            try:
                date = datetime.fromisoformat(date_str)
            except ValueError:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            price = float(price_str)
            rows.append((date, price))
    rows.sort(key=lambda item: item[0])
    return rows


def load_episode(path: str) -> Episode:
    """Load a serialized Episode object."""

    return Episode.load(path)


def load_episode_list(paths: Sequence[str]) -> List[Episode]:
    """Load multiple episodes from disk."""

    return [load_episode(p) for p in paths]


def load_episodes_from_dir(dir_path: str) -> List[Episode]:
    """Load all episode files from a directory (sorted)."""

    directory = Path(dir_path)
    if not directory.exists():
        raise FileNotFoundError(dir_path)
    files = sorted(p for p in directory.glob("*.pkl"))
    return [Episode.load(str(p)) for p in files]


__all__ = [
    "load_raw_spy",
    "load_episode",
    "load_episode_list",
    "load_episodes_from_dir",
]
