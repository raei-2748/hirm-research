from __future__ import annotations

import pytest

from hirm.diagnostics import compute_efficiency_metrics
from hirm.diagnostics.efficiency import compute_er, compute_tr


def test_er_rewards_higher_mean_returns() -> None:
    low_risk = compute_er([0.2, 0.2, 0.2, 0.2], cvar_alpha=0.5, eps=1e-6)["ER"]
    high_risk = compute_er([0.5, -0.5, 0.5, -0.5], cvar_alpha=0.5, eps=1e-6)["ER"]
    assert low_risk > high_risk


def test_tr_zero_for_constant_actions() -> None:
    actions = [[1.0, 1.0] for _ in range(5)]
    metrics = compute_tr(actions, eps=1e-6)
    assert metrics["TR"] == pytest.approx(0.0, abs=1e-9)


def test_tr_increases_with_oscillation() -> None:
    actions = [[(-1) ** t, (-1) ** (t + 1)] for t in range(10)]
    metrics = compute_tr(actions, eps=1e-6)
    assert metrics["TR"] > 1.0


def test_compute_efficiency_metrics() -> None:
    er_inputs = {"returns_time_series": [0.1, 0.2, 0.3], "cvar_alpha": 0.5, "eps": 1e-6}
    tr_inputs = {"actions": [[1.0, 1.0] for _ in range(3)], "eps": 1e-6}
    metrics = compute_efficiency_metrics(er_inputs=er_inputs, tr_inputs=tr_inputs)
    assert set(metrics) == {"ER", "TR"}
