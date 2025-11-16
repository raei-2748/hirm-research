"""Shared experiment registry dataclasses and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from hirm.utils.config import ConfigNode


@dataclass
class ExperimentRunConfig:
    dataset: str
    method: str
    seed: int
    config: ConfigNode
    device: torch.device


class Trainer(Protocol):
    model: object

    def set_datasets(self, *, train, val) -> None:  # pragma: no cover - protocol
        ...

    def train(self) -> list[dict]:  # pragma: no cover - protocol
        ...

    def save(self, path: str) -> None:  # pragma: no cover - protocol
        ...

    def risk_eval(self, batch, env_ids):  # pragma: no cover - protocol
        ...


__all__ = ["ExperimentRunConfig", "Trainer"]
