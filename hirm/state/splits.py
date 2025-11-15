"""Utilities for deterministic episode splits."""
from __future__ import annotations

import math
import random
from typing import Dict, List


def _compute_split_counts(num_episodes: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    total_frac = train_frac + val_frac + test_frac
    if not math.isclose(total_frac, 1.0):
        train_frac /= total_frac
        val_frac /= total_frac
        test_frac /= total_frac
    train_count = int(round(num_episodes * train_frac))
    val_count = int(round(num_episodes * val_frac))
    test_count = num_episodes - train_count - val_count
    # Adjust if rounding produced negative or overflow
    if test_count < 0:
        test_count = 0
    if train_count + val_count + test_count != num_episodes:
        # Fall back to floor for train/val and assign remainder to test
        train_count = int(num_episodes * train_frac)
        val_count = int(num_episodes * val_frac)
        test_count = num_episodes - train_count - val_count
    return train_count, val_count, test_count


def create_episode_splits(
    num_episodes: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 123,
) -> Dict[str, List[int]]:
    """Split ``num_episodes`` deterministically into train/val/test indices."""

    if num_episodes < 0:
        raise ValueError("num_episodes must be non-negative")
    rng = random.Random(seed)
    indices = list(range(num_episodes))
    rng.shuffle(indices)
    train_count, val_count, test_count = _compute_split_counts(
        num_episodes, train_frac, val_frac, test_frac
    )
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count : train_count + val_count + test_count]
    return {
        "train": list(train_idx),
        "val": list(val_idx),
        "test": list(test_idx),
    }


__all__ = ["create_episode_splits"]
