"""Efficiency diagnostics (Section 5.3)."""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from hirm.utils.math import cvar


def compute_er(
    returns_time_series: Sequence[float],
    cvar_alpha: float,
    eps: float,
    mode: str = "loss",
) -> Dict[str, float]:
    """Compute the expected return / tail risk ratio (Eq. 10)."""

    if mode not in {"loss", "returns"}:
        raise ValueError("mode must be 'loss' or 'returns'")

    values = [float(v) for v in returns_time_series]
    if not values:
        raise ValueError("returns_time_series cannot be empty")
    mean_return = sum(values) / len(values)
    tail_risk_returns = float(cvar(values, alpha=cvar_alpha))
    if mode == "returns":
        tail_risk = tail_risk_returns
    else:
        # Interpret losses as ``-returns`` and capture the upper tail via sign flip.
        tail_risk = max(0.0, -tail_risk_returns)
    er = mean_return / (tail_risk + eps)
    return {"ER": er}


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover
            pass
    return value


def _ensure_3d(actions: Any) -> list[list[list[float]]]:
    array = _to_list(actions)
    if not isinstance(array, list):
        raise ValueError("actions must be array-like")
    if array and isinstance(array[0], list) and array[0] and isinstance(array[0][0], list):
        return [[[float(x) for x in vec] for vec in seq] for seq in array]
    if array and isinstance(array[0], list):
        return [[[float(x) for x in vec] for vec in array]]
    return [[[float(x)] for x in array]]


def _vector_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def compute_tr(
    actions,
    eps: float,
    treat_first_step: str = "drop",
) -> Dict[str, float]:
    """Compute the turnover ratio E[||a_t - a_{t-1}||] / E[||a_t||]."""

    batch_actions = _ensure_3d(actions)
    batch = len(batch_actions)
    time = len(batch_actions[0])
    if treat_first_step not in {"drop", "zero"}:
        raise ValueError("treat_first_step must be 'drop' or 'zero'")
    if time <= 1:
        return {"TR": 0.0}
    diffs: list[list[list[float]]] = []
    if treat_first_step == "drop":
        for seq in batch_actions:
            diffs.append([
                [seq[t][d] - seq[t - 1][d] for d in range(len(seq[t]))]
                for t in range(1, len(seq))
            ])
    else:
        zeros = [0.0 for _ in range(len(batch_actions[0][0]))]
        for seq in batch_actions:
            prev = zeros
            seq_diffs = []
            for vec in seq:
                seq_diffs.append([vec[d] - prev[d] for d in range(len(vec))])
                prev = vec
            diffs.append(seq_diffs)
    diff_norms = []
    action_norms = []
    for seq_diff, seq in zip(diffs, batch_actions):
        diff_norms.extend(_vector_norm(vec) for vec in seq_diff)
        action_norms.extend(_vector_norm(vec) for vec in seq)
    mean_diff = sum(diff_norms) / len(diff_norms)
    mean_action = sum(action_norms) / len(action_norms)
    tr = mean_diff / (mean_action + eps)
    return {"TR": tr}


__all__ = ["compute_er", "compute_tr"]
