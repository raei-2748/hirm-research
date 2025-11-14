"""Environment abstractions."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class Transition:
    state: List[float]
    reward: float
    info: Dict[str, float]
    done: bool


class BaseEnv(abc.ABC):
    """Abstract environment for hedging experiments."""

    def __init__(self, state_dim: int, action_dim: int, episode_length: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.current_step = 0
        self.metrics: Dict[str, List[float]] = {
            "returns": [],
            "realized_vol": [],
            "inventory": [],
        }
        self._state = [0.0 for _ in range(self.state_dim)]

    def reset(self) -> List[float]:
        self.current_step = 0
        for values in self.metrics.values():
            values.clear()
        self._state = self._initial_state()
        return list(self._state)

    def step(self, action: Sequence[float]) -> Transition:
        if len(action) != self.action_dim:
            raise ValueError(
                f"Action length {len(action)} incompatible with action_dim {self.action_dim}"
            )
        if self.current_step >= self.episode_length:
            raise RuntimeError("Episode already finished. Call reset().")
        state, reward, info = self._transition(action)
        self._state = state
        self.current_step += 1
        done = self.current_step >= self.episode_length
        self._track_metrics(info)
        return Transition(state=list(state), reward=reward, info=info, done=done)

    def _track_metrics(self, info: Dict[str, float]) -> None:
        for key in self.metrics:
            if key in info:
                self.metrics[key].append(float(info[key]))

    @abc.abstractmethod
    def _initial_state(self) -> List[float]:
        ...

    @abc.abstractmethod
    def _transition(self, action: Sequence[float]) -> Tuple[List[float], float, Dict[str, float]]:
        ...


__all__ = ["BaseEnv", "Transition"]
