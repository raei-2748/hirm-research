import numpy as np

from hirm.envs.real.spy import SPYEnv


def test_spy_env_episode():
    env = SPYEnv(horizon=50, seed=123)
    env.seed(123)
    reset = env.reset()
    first_price = reset["obs"]["price"]
    assert first_price > 0
    prices = [first_price]
    regimes = [reset["info"]["regime"]]
    vols = [reset["info"]["realized_vol_20"]]
    for _ in range(50):
        step = env.step(0.0)
        prices.append(step["price"])
        regimes.append(step["info"]["regime"])
        vols.append(step["info"]["realized_vol_20"])
        if step["done"]:
            break
    assert len(prices) >= 2
    assert all(np.isfinite(v) for v in prices)
    assert all(r in {0, 1, 2, 3} for r in regimes)
    assert len(vols) == len(prices)
    env.seed(123)
    env.reset()
    step1 = env.step(0.0)
    env.seed(123)
    env.reset()
    step2 = env.step(0.0)
    np.testing.assert_allclose(step1["price"], step2["price"])
