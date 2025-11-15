import numpy as np

from hirm.envs.synthetic.heston import HestonEnv
from hirm.envs.synthetic.merton_jump import MertonJumpEnv


def rollout(env):
    env.seed(123)
    env.reset()
    prices = []
    rewards = []
    for _ in range(5):
        out = env.step(0.0)
        prices.append(out["price"])
        rewards.append(out["reward"])
    return np.array(prices), np.array(rewards)


def test_heston_env_deterministic():
    env1 = HestonEnv(horizon=10)
    env2 = HestonEnv(horizon=10)
    p1, r1 = rollout(env1)
    p2, r2 = rollout(env2)
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(r1, r2)


def test_merton_env_runs():
    env = MertonJumpEnv(horizon=10)
    env.seed(42)
    env.reset()
    for _ in range(10):
        out = env.step(0.0)
        assert np.isfinite(out["price"])
        assert np.isfinite(out["reward"])
        if out["done"]:
            break
