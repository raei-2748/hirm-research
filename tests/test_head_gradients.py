from __future__ import annotations

from hirm.envs.base import Episode
from hirm.envs.episodes import EnvBatch
from hirm.models.policy import MLPPolicy
from hirm.objectives.utils import compute_head_gradients, risk_on_env_batch


def _batch() -> EnvBatch:
    episode = Episode(
        prices=[100.0, 101.0],
        states=[[0.0, 0.1]],
        pnl=0.0,
        env_id="train_low",
        meta={"liability": 101.0, "transaction_cost": 0.0},
    )
    return EnvBatch.from_episodes([episode], env_id="train_low", split="train")


def test_head_gradient_nonzero() -> None:
    policy = MLPPolicy(input_dim=2, hidden_dims=[4], head_dim=1)
    batch = _batch()
    loss_fn = lambda p, b: risk_on_env_batch(p, b)
    loss, grad = compute_head_gradients(policy, loss_fn, batch)
    assert isinstance(loss, float)
    expected = sum(len(param) if not param or not isinstance(param[0], list) else len(param) * len(param[0]) for param in policy.parameters_head())
    assert len(grad) == expected
