"""Base environment interface for Phase 2."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class Env:
    """Minimal deterministic environment API."""

    def __init__(self, horizon: int = 252, seed: int | None = None) -> None:
        self._horizon = int(horizon)
        self._seed = None
        self.rng = np.random.default_rng()
        if seed is not None:
            self.seed(seed)

    def reset(self) -> Dict[str, Any]:  # pragma: no cover - interface only
        """Reset environment state. Implemented by subclasses."""
        raise NotImplementedError

    def step(self, action: float) -> Dict[str, Any]:  # pragma: no cover - interface only
        """Advance environment by one step. Implemented by subclasses."""
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        """Seed the RNG for deterministic behavior."""

        self._seed = int(seed)
        self.rng = np.random.default_rng(self._seed)

    @property
    def horizon(self) -> int:
        """Maximum number of steps in an episode."""

        return self._horizon


__all__ = ["Env"]
