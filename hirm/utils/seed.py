"""Reproducibility helpers."""
from __future__ import annotations

import random
from typing import Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def seed_from_config(config: dict, key: str = "seed") -> Optional[int]:
    value = config.get(key)
    if isinstance(value, int):
        set_seed(value)
        return value
    return None


__all__ = ["set_seed", "seed_from_config"]
