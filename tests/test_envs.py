from __future__ import annotations

from hirm.envs.spy_real_env import SpyRealEnv
from hirm.envs.synthetic_heston_env import SyntheticHestonEnv


def test_synthetic_env_step() -> None:
    env = SyntheticHestonEnv(
        state_dim=4,
        action_dim=2,
        episode_length=5,
        drift=0.0,
        volatility=0.01,
        seed=0,
    )
    state = env.reset()
    assert len(state) == 4
    action = [0.0 for _ in range(env.action_dim)]
    transition = env.step(action)
    assert len(transition.state) == 4
    assert isinstance(transition.reward, float)
    assert not transition.done


def test_spy_env_shapes() -> None:
    env = SpyRealEnv(
        data_path="data/processed/spy_features.csv",
        state_dim=4,
        action_dim=1,
        episode_length=5,
        feature_columns=["return", "realized_vol", "liquidity", "inventory"],
    )
    state = env.reset()
    assert len(state) == 4
    transition = env.step([0.0])
    assert len(transition.state) == 4
    assert "returns" in transition.info
