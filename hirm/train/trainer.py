"""Config-driven training loop with finite-difference gradients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

from hirm.envs import build_env
from hirm.envs.episodes import EnvBatch
from hirm.models import build_policy
from hirm.objectives import Objective, build_objective


@dataclass
class TrainerConfig:
    episodes_per_env: int = 4
    num_steps: int = 10
    lr: float = 1e-3
    eval_episodes_per_env: int = 2


class Trainer:
    def __init__(
        self,
        env_config: Dict[str, object],
        model_config: Dict[str, object],
        objective_config: Dict[str, object],
        training_config: Mapping[str, object],
    ) -> None:
        self.env = build_env(env_config)
        self.policy = build_policy(model_config)
        self.objective: Objective = build_objective(objective_config)
        self.config = TrainerConfig(
            episodes_per_env=int(training_config.get("episodes_per_env", 4)),
            num_steps=int(training_config.get("num_steps", 10)),
            lr=float(training_config.get("lr", 1e-3)),
            eval_episodes_per_env=int(training_config.get("eval_episodes_per_env", 2)),
        )

    def _sample_env_batches(
        self, split: str, episodes_per_env: int
    ) -> Dict[str, EnvBatch]:
        env_batches: Dict[str, EnvBatch] = {}
        for env_id in self.env.available_env_ids(split):
            episodes = self.env.sample_episodes(episodes_per_env, split=split, env_id=env_id)
            env_batches[env_id] = EnvBatch.from_episodes(episodes, env_id=env_id, split=split)
        return env_batches

    def _finite_difference(
        self, env_batches: Mapping[str, EnvBatch]
    ) -> tuple[list[np.ndarray], float]:
        params = self.policy.parameters_all()
        grads = []
        base_loss = self.objective(self.policy, env_batches)
        epsilon = 1e-4
        for param in params:
            if param and isinstance(param[0], list):
                grad_matrix = [[0.0 for _ in row] for row in param]
                for i in range(len(param)):
                    for j in range(len(param[i])):
                        original = param[i][j]
                        param[i][j] = original + epsilon
                        perturbed = self.objective(self.policy, env_batches)
                        grad_matrix[i][j] = (perturbed - base_loss) / epsilon
                        param[i][j] = original
                grads.append(grad_matrix)
            else:
                grad_vector = [0.0 for _ in param]
                for i in range(len(param)):
                    original = param[i]
                    param[i] = original + epsilon
                    perturbed = self.objective(self.policy, env_batches)
                    grad_vector[i] = (perturbed - base_loss) / epsilon
                    param[i] = original
                grads.append(grad_vector)
        return grads, base_loss

    def _apply_gradients(self, grads: list[List]) -> None:
        for param, grad in zip(self.policy.parameters_all(), grads):
            if param and isinstance(param[0], list):
                for i in range(len(param)):
                    for j in range(len(param[i])):
                        param[i][j] -= self.config.lr * grad[i][j]
            else:
                for i in range(len(param)):
                    param[i] -= self.config.lr * grad[i]

    def _evaluate(self, split: str) -> float | None:
        env_ids = self.env.available_env_ids(split)
        if not env_ids:
            return None
        env_batches = self._sample_env_batches(split, self.config.eval_episodes_per_env)
        risks = [self.objective.risk(self.policy, batch) for batch in env_batches.values()]
        return sum(risks) / len(risks)

    def train(self) -> Dict[str, float | None]:
        metrics: Dict[str, float | None] = {}
        for _ in range(self.config.num_steps):
            env_batches = self._sample_env_batches("train", self.config.episodes_per_env)
            grads, loss = self._finite_difference(env_batches)
            self._apply_gradients(grads)
            metrics["train_loss_step"] = loss
        metrics["val_risk"] = self._evaluate("val")
        metrics["test_risk"] = self._evaluate("test")
        return metrics


__all__ = ["Trainer", "TrainerConfig"]
