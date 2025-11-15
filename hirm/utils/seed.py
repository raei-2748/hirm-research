"""Reproducibility helpers for Phase 1."""
from __future__ import annotations

import random
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _seed_torch(seed: int, deterministic: bool) -> None:
    if torch is None:
        return
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "manual_seed_all", None)):
        torch.cuda.manual_seed_all(seed)  # type: ignore[call-arg]
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)  # type: ignore[call-arg]
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            setattr(cudnn, "deterministic", True)
            setattr(cudnn, "benchmark", False)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs."""

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    _seed_torch(seed, deterministic)


def seed_from_config(
    config: dict[str, Any],
    key: str = "seed",
    deterministic_key: str = "deterministic",
) -> Optional[int]:
    """Seed from configuration dictionaries if possible."""

    value = config.get(key)
    if isinstance(value, int):
        deterministic = bool(config.get(deterministic_key, False))
        set_seed(value, deterministic)
        return value
    return None


__all__ = ["seed_from_config", "set_seed"]
