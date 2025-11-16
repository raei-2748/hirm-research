from __future__ import annotations

import pytest

from hirm.diagnostics import compute_robustness_metrics
from hirm.diagnostics.robustness import compute_vr, compute_wg


def test_wg_single_environment_equals_risk() -> None:
    risks = {"env_0": 0.15}
    metrics = compute_wg(env_risks=risks, alpha=0.1)
    assert metrics["WG"] == pytest.approx(0.15, rel=1e-6)


def test_wg_interpolates_between_risks() -> None:
    risks = {"env_0": 0.1, "env_1": 0.2}
    metrics = compute_wg(env_risks=risks, alpha=0.05)
    assert 0.15 < metrics["WG"] < 0.21


def test_vr_is_zero_for_constant_series() -> None:
    metrics = compute_vr(risk_time_series=[0.2] * 10, eps=1e-8)
    assert metrics["VR"] == pytest.approx(0.0, abs=1e-9)


def test_vr_grows_with_variance() -> None:
    low = compute_vr(risk_time_series=[1.0] * 20, eps=1e-8)["VR"]
    high = compute_vr(risk_time_series=[0.1 + 0.05 * i for i in range(20)], eps=1e-8)["VR"]
    assert high > low


def test_compute_robustness_metrics() -> None:
    wg_inputs = {"env_risks": {"env_0": 0.1, "env_1": 0.2}, "alpha": 0.1}
    vr_inputs = {"risk_time_series": [0.5, 0.6, 0.7], "eps": 1e-6}
    metrics = compute_robustness_metrics(wg_inputs=wg_inputs, vr_inputs=vr_inputs)
    assert set(metrics) == {"WG", "VR"}
