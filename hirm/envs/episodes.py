"""Episodic hedging utilities implemented with Python lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from hirm.envs.base import Episode
from hirm.models.policy import Policy


@dataclass
class EnvBatch:
    env_id: str
    split: str
    prices: List[List[float]]  # (batch, T + 1)
    states: List[List[List[float]]]  # (batch, T, feature_dim)
    liabilities: List[float]
    transaction_cost: float = 0.0
    meta: List[Dict[str, Any]] | None = None

    @property
    def batch_size(self) -> int:
        return len(self.states)

    @property
    def horizon(self) -> int:
        return len(self.states[0]) if self.states else 0

    @classmethod
    def from_episodes(
        cls,
        episodes: Sequence[Episode],
        env_id: str,
        split: str,
    ) -> "EnvBatch":
        if not episodes:
            raise ValueError("episodes must be non-empty")
        prices = [list(map(float, ep.prices)) for ep in episodes]
        states = [
            [list(map(float, step)) for step in ep.states]
            for ep in episodes
        ]
        liabilities = [float(ep.meta.get("liability", ep.prices[-1])) for ep in episodes]
        transaction_cost = float(episodes[0].meta.get("transaction_cost", 0.0))
        return cls(
            env_id=env_id,
            split=split,
            prices=prices,
            states=states,
            liabilities=liabilities,
            transaction_cost=transaction_cost,
            meta=[ep.meta for ep in episodes],
        )


def _apply_policy(policy: Policy, states: List[List[float]]) -> List[List[float]]:
    return policy(states)


def simulate_policy_on_env_batch(policy: Policy, env_batch: EnvBatch) -> List[float]:
    pnl_values: List[float] = []
    for episode_idx in range(env_batch.batch_size):
        states = env_batch.states[episode_idx]
        prices = env_batch.prices[episode_idx]
        liabilities = env_batch.liabilities[episode_idx]
        actions = _apply_policy(policy, states)
        price_diffs = [prices[t + 1] - prices[t] for t in range(len(states))]
        pnl_positions = 0.0
        for t, diff in enumerate(price_diffs):
            pnl_positions += sum(actions[t][k] * diff for k in range(len(actions[t])))
        delta_q: List[float] = []
        prev = 0.0
        for action in actions:
            current = sum(action)
            delta_q.append(current - prev)
            prev = current
        costs = env_batch.transaction_cost * sum(abs(val) for val in delta_q)
        pnl = pnl_positions - costs - liabilities
        pnl_values.append(pnl)
    return pnl_values


__all__ = ["EnvBatch", "simulate_policy_on_env_batch"]
