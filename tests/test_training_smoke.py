from __future__ import annotations

from hirm.train.trainer import Trainer


def _base_trainer(objective_config: dict[str, object]) -> Trainer:
    env_config = {
        "name": "synthetic_volatility_bands",
        "horizon": 10,
        "start_price": 100.0,
        "seed": 0,
    }
    model_config = {"name": "mlp", "input_dim": 2, "hidden_dims": [4], "head_dim": 1}
    training_config = {"episodes_per_env": 1, "num_steps": 1, "eval_episodes_per_env": 1}
    return Trainer(env_config, model_config, objective_config, training_config)


def test_trainer_runs_erm() -> None:
    trainer = _base_trainer({"name": "erm", "alpha": 0.95})
    metrics = trainer.train()
    assert "train_loss_step" in metrics


def test_trainer_runs_hirm() -> None:
    trainer = _base_trainer({"name": "hirm", "alpha": 0.95, "lambda_invariance": 0.1})
    metrics = trainer.train()
    assert "val_risk" in metrics
