import numpy as np

from hirm.envs.base import Env


class DummyEnv(Env):
    def __init__(self):
        super().__init__(horizon=5)
        self._price = 1.0
        self._t = 0

    def reset(self):
        self._price = 1.0
        self._t = 0
        obs = {"price": self._price, "t": self._t}
        info = {"t": self._t, "price": self._price}
        return {"obs": obs, "info": info}

    def step(self, action):
        del action
        self._t += 1
        self._price += 1.0
        done = self._t >= self.horizon
        obs = {"price": self._price, "t": self._t}
        info = {"t": self._t, "price": self._price}
        return {
            "obs": obs,
            "reward": 1.0,
            "price": self._price,
            "done": done,
            "info": info,
        }


def test_env_interface_and_determinism():
    env1 = DummyEnv()
    env2 = DummyEnv()
    env1.seed(123)
    env2.seed(123)
    traj1 = []
    traj2 = []
    obs1 = env1.reset()
    obs2 = env2.reset()
    traj1.append(obs1["obs"]["price"])
    traj2.append(obs2["obs"]["price"])
    for _ in range(env1.horizon):
        step1 = env1.step(0.0)
        step2 = env2.step(0.0)
        traj1.append(step1["price"])
        traj2.append(step2["price"])
    np.testing.assert_allclose(traj1, traj2)
