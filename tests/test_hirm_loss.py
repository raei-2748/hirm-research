from __future__ import annotations

from hirm.envs.base import Episode
from hirm.envs.episodes import EnvBatch
from hirm.models.policy import MLPPolicy
from hirm.objectives.hirm import hirm_loss


def _env_batches() -> dict[str, EnvBatch]:
    episode = Episode(
        prices=[100.0, 101.0],
        states=[[1.0, 0.1]],
        pnl=0.0,
        env_id="train_low",
        meta={"liability": 101.0, "transaction_cost": 0.0},
    )
    return {
        "train_low": EnvBatch.from_episodes([episode], env_id="train_low", split="train"),
        "train_low_copy": EnvBatch.from_episodes([episode], env_id="train_low", split="train"),
    }


def test_hirm_reduces_to_erm_when_lambda_zero() -> None:
    policy = MLPPolicy(input_dim=2, hidden_dims=[4], head_dim=1)
    batches = _env_batches()
    loss = hirm_loss(policy, batches, lambda_invariance=0.0)
    assert abs(loss) < 1e6


def test_hirm_penalty_zero_identical_gradients() -> None:
    policy = MLPPolicy(input_dim=2, hidden_dims=[4], head_dim=1)
    batches = _env_batches()
    loss_lambda_zero = hirm_loss(policy, batches, lambda_invariance=0.0)
    loss_lambda_one = hirm_loss(policy, batches, lambda_invariance=1.0)
    assert abs(loss_lambda_zero - loss_lambda_one) < 1e-6
