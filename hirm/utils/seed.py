"""Reproducibility helpers."""
from __future__ import annotations

import random
from typing import Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:  # pragma: no cover - torch version differences
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_from_config(config: dict, key: str = "seed") -> Optional[int]:
    value = config.get(key)
    if isinstance(value, int):
        set_seed(value)
        return value
    return None


__all__ = ["set_seed", "seed_from_config"]
