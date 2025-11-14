"""Environment abstractions for hedging research episodes.

This module defines lightweight dataclasses and base interfaces that all
multi-regime environments must implement.  The environments correspond to the
setups described in ``Invariant_Hedging_Research_Paper.md`` where an agent is
trained and evaluated on episodes sampled from different market regimes (e.g.
SPY historical windows or synthetic Heston worlds).  The base classes are kept
framework agnostic on purpose: they only know how to describe and return
episodes, without imposing any learning logic or reward definitions.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Sequence


@dataclass
class Episode:
    """Container for a hedging episode.

    Attributes
    ----------
    prices:
        Underlying price path of length ``T + 1`` for a ``T`` step episode.  The
        path can optionally include extra columns (e.g. multi-asset settings).
    states:
        Feature array aligned with ``prices``.  Feature engineering happens in
        later phases, but the placeholder allows downstream code to consume a
        consistent structure.
    pnl:
        Episode level profit-and-loss for a given strategy.  The environments in
        this phase typically set it to ``0.0`` because hedging logic is not yet
        plugged in, but the field exists so future components can reuse the
        dataclass without changes.
    env_id:
        Identifier describing which environment/regime bucket produced the
        episode (e.g. ``"train_low_vol"`` or ``"covid_crisis"``).
    meta:
        Free-form metadata such as dates, realized volatility, regime labels, or
        simulation parameters.
    """

    prices: List[float]
    states: List[List[float]]
    pnl: float
    env_id: str | int
    meta: Dict[str, Any] = field(default_factory=dict)


class Environment(abc.ABC):
    """Abstract base class for hedging environments.

    Concrete subclasses generate price episodes that belong to specific regime
    buckets.  Consumers interact with the environment purely through episode
    sampling – there is no notion of an online ``reset``/``step`` API at this
    layer.  This keeps the module decoupled from any reinforcement learning
    framework and mirrors the offline construction described in the research
    paper.
    """

    def __init__(self, split_env_ids: Mapping[str, Sequence[str]] | None = None) -> None:
        self._split_env_ids: Dict[str, List[str]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        if split_env_ids:
            for split, env_ids in split_env_ids.items():
                if split not in self._split_env_ids:
                    raise ValueError(f"Unknown split '{split}'. Expected train/val/test.")
                self._split_env_ids[split] = list(env_ids)

    @property
    def split_env_ids(self) -> Mapping[str, Sequence[str]]:
        """Return the environment identifiers available for each split."""

        return {split: tuple(env_ids) for split, env_ids in self._split_env_ids.items()}

    def available_env_ids(self, split: str) -> Sequence[str]:
        if split not in self._split_env_ids:
            raise ValueError(f"Unknown split '{split}'. Expected train/val/test.")
        return tuple(self._split_env_ids[split])

    @abc.abstractmethod
    def sample_episode(self, split: str = "train", env_id: str | None = None) -> Episode:
        """Return a single episode drawn from the requested split.

        Parameters
        ----------
        split:
            Data split to sample from (``"train"``, ``"val"`` or ``"test"``).
        env_id:
            Optional environment identifier within the split.  When ``None`` a
            random environment from the split is selected.  Subclasses that do
            not differentiate between environments can ignore the argument.
        """

    def sample_episodes(
        self, n: int, split: str = "train", env_id: str | None = None
    ) -> List[Episode]:
        """Return ``n`` i.i.d. episodes for the requested split."""

        if n <= 0:
            raise ValueError("n must be positive")
        return [self.sample_episode(split=split, env_id=env_id) for _ in range(n)]

    # The helper lives on the base class so subclasses can share the cloning
    # logic and avoid exposing mutable references to cached arrays.
    @staticmethod
    def _clone_episode(episode: Episode) -> Episode:
        return replace(
            episode,
            prices=list(episode.prices),
            states=[list(row) for row in episode.states],
            meta=dict(episode.meta),
        )


__all__ = ["Episode", "Environment"]
