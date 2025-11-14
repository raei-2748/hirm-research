from __future__ import annotations

from hirm.envs.base import Episode
from hirm.envs.episodes import EnvBatch, simulate_policy_on_env_batch
from hirm.models.policy import MLPPolicy


def _make_episode(regime: str) -> Episode:
    prices = [100.0, 101.0, 102.5]
    states = [[100.0, 0.1], [101.0, 0.1]]
    meta = {"regime": regime, "liability": 102.5, "transaction_cost": 0.0}
    return Episode(prices=prices, states=states, pnl=0.0, env_id=f"train_{regime}", meta=meta)


def test_env_batch_shapes() -> None:
    batch = EnvBatch.from_episodes([
        _make_episode("low"),
        _make_episode("medium"),
    ], env_id="train_low", split="train")
    assert len(batch.prices) == 2
    assert len(batch.states[0]) == 2


def test_simulation_pnl_sign() -> None:
    batch = EnvBatch.from_episodes([_make_episode("low")], env_id="train_low", split="train")
    policy = MLPPolicy(input_dim=2, hidden_dims=[4], head_dim=1)
    pnl = simulate_policy_on_env_batch(policy, batch)
    assert len(pnl) == 1
