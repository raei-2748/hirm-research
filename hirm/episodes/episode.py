"""Episode container."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from hirm.utils.serialization import load_checkpoint, save_checkpoint


class Episode:
    """Serialization-friendly container for a single episode."""

    def __init__(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        metadata: Dict[str, np.ndarray],
    ) -> None:
        self.prices = np.asarray(prices, dtype=float)
        self.returns = np.asarray(returns, dtype=float)
        self.metadata: Dict[str, np.ndarray] = {
            key: np.asarray(value)
            for key, value in metadata.items()
        }
        if self.returns.ndim != 1:
            raise ValueError("returns must be 1D")
        if self.prices.ndim != 1:
            raise ValueError("prices must be 1D")
        if self.returns.size + 1 not in {self.prices.size, self.returns.size}:
            raise ValueError("Inconsistent prices/returns lengths")

    @property
    def length(self) -> int:
        """Length of the episode (number of returns)."""

        return self.returns.size

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary for persistence."""

        return {
            "prices": self.prices,
            "returns": self.returns,
            "metadata": {k: np.asarray(v) for k, v in self.metadata.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Episode":
        """Deserialize from ``payload``."""

        return cls(
            prices=np.asarray(payload["prices"]),
            returns=np.asarray(payload["returns"]),
            metadata={k: np.asarray(v) for k, v in payload["metadata"].items()},
        )

    def save(self, path: str) -> None:
        """Persist the episode to disk."""

        save_checkpoint(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> "Episode":
        """Load an episode from disk."""

        payload = load_checkpoint(path)
        return cls.from_dict(dict(payload))


__all__ = ["Episode"]
