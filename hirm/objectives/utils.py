"""Objective helper utilities using finite differences and Python lists."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

from hirm.envs.episodes import EnvBatch, simulate_policy_on_env_batch
from hirm.models.policy import Policy
from hirm.risk.cvar import cvar_loss


LossFn = Callable[[Policy, EnvBatch], float]


def risk_on_env_batch(policy: Policy, env_batch: EnvBatch, alpha: float = 0.95) -> float:
    pnls = simulate_policy_on_env_batch(policy, env_batch)
    return cvar_loss(pnls, alpha=alpha)


def _iterate_parameter(param: List) -> Iterable[tuple[List, int, int | None]]:
    if param and isinstance(param[0], list):
        for i in range(len(param)):
            for j in range(len(param[i])):
                yield param[i], j, None
    else:
        for i in range(len(param)):
            yield param, i, None


def _finite_difference(
    params: Sequence[List],
    loss_fn: Callable[[], float],
    epsilon: float = 1e-4,
) -> tuple[List[List], float]:
    grads = []
    base_loss = loss_fn()
    for param in params:
        if param and isinstance(param[0], list):
            grad_matrix = [[0.0 for _ in row] for row in param]
            for i in range(len(param)):
                for j in range(len(param[i])):
                    original = param[i][j]
                    param[i][j] = original + epsilon
                    perturbed = loss_fn()
                    grad_matrix[i][j] = (perturbed - base_loss) / epsilon
                    param[i][j] = original
            grads.append(grad_matrix)
        else:
            grad_vector = [0.0 for _ in param]
            for i in range(len(param)):
                original = param[i]
                param[i] = original + epsilon
                perturbed = loss_fn()
                grad_vector[i] = (perturbed - base_loss) / epsilon
                param[i] = original
            grads.append(grad_vector)
    return grads, base_loss


def compute_head_gradients(
    policy: Policy,
    loss_fn: LossFn,
    env_batch: EnvBatch,
) -> tuple[float, List[float]]:
    def evaluate() -> float:
        return loss_fn(policy, env_batch)

    head_params = policy.parameters_head()
    grads, base_loss = _finite_difference(head_params, evaluate)
    flat_grad: List[float] = []
    for grad in grads:
        if grad and isinstance(grad[0], list):
            for row in grad:
                flat_grad.extend(row)
        else:
            flat_grad.extend(grad)
    norm = sum(value * value for value in flat_grad) ** 0.5
    if norm > 0:
        flat_grad = [value / norm for value in flat_grad]
    return base_loss, flat_grad


__all__ = ["risk_on_env_batch", "compute_head_gradients"]
