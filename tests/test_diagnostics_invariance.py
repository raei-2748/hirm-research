from __future__ import annotations

import math
import random

import math

import pytest

from hirm.diagnostics import compute_invariance_metrics
from hirm.diagnostics.invariance import compute_ig, compute_isi


@pytest.fixture()
def base_inputs() -> dict:
    env_risks = {"env_0": 0.1, "env_1": 0.1}
    head_gradients = {
        "env_0": [1.0] * 4,
        "env_1": [1.0] * 4,
    }
    layer_activations = {
        "representation": {
            "env_0": [[1.0, 1.0] for _ in range(8)],
            "env_1": [[1.0, 1.0] for _ in range(8)],
        }
    }
    cfg = dict(
        env_risks=env_risks,
        head_gradients=head_gradients,
        layer_activations=layer_activations,
        tau_R=0.1,
        tau_C=1.0,
        alpha_components=[1.0, 1.0, 1.0],
        eps=1e-8,
        cov_regularizer=1e-4,
        trim_fraction=0.1,
    )
    return cfg


def test_isi_perfect_invariance(base_inputs) -> None:
    metrics = compute_isi(**base_inputs)
    assert metrics["ISI"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["ISI_C1"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["ISI_C2"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["ISI_C3"] == pytest.approx(1.0, rel=1e-3)


def test_isi_penalizes_variance(base_inputs) -> None:
    unstable = base_inputs.copy()
    unstable["env_risks"] = {"env_0": 0.1, "env_1": 0.5}
    metrics_stable = compute_isi(**base_inputs)
    metrics_unstable = compute_isi(**unstable)
    assert metrics_unstable["ISI_C1"] < metrics_stable["ISI_C1"]


def test_c2_drops_with_misaligned_gradients(base_inputs) -> None:
    noisy = base_inputs.copy()
    noisy["head_gradients"] = {
        "env_0": [1.0, 0.0, 0.0, 0.0],
        "env_1": [0.0, 1.0, 0.0, 0.0],
    }
    metrics = compute_isi(**noisy)
    assert metrics["ISI_C2"] < 0.75


def test_c3_detects_covariance_shift(base_inputs) -> None:
    shifted = base_inputs.copy()
    shifted["layer_activations"] = {
        "representation": {
            "env_0": [[1.0, 1.0] for _ in range(10)],
            "env_1": [[1.0 + 0.1 * t, 10.0 + 0.2 * t] for t in range(10)],
        }
    }
    metrics = compute_isi(**shifted)
    assert metrics["ISI_C3"] < 1.0


def test_isi_handles_small_gradients_without_nan(base_inputs) -> None:
    tiny = base_inputs.copy()
    tiny["head_gradients"] = {"env_0": [0.0] * 4, "env_1": [0.0] * 4}
    metrics = compute_isi(**tiny)
    assert math.isfinite(metrics["ISI_C2"])


def test_c2_trimming_dampens_outlier(base_inputs) -> None:
    trim_case = base_inputs.copy()
    trim_case["trim_fraction"] = 0.2
    trim_case["env_risks"] = {f"env_{idx}": 0.1 for idx in range(5)}
    aligned = [1.0, 0.0, 0.0, 0.0]
    inverted = [-1.0, 0.0, 0.0, 0.0]
    trim_case["head_gradients"] = {f"env_{idx}": aligned[:] for idx in range(5)}
    trim_case["head_gradients"]["env_4"] = inverted
    trim_case["layer_activations"] = {
        "representation": {
            f"env_{idx}": [[1.0, 1.0] for _ in range(6)] for idx in range(5)
        }
    }
    metrics = compute_isi(**trim_case)
    assert metrics["ISI_C2_trimmed"] > metrics["ISI_C2"]


def test_c3_trimming_handles_dispersion_outlier(base_inputs) -> None:
    trim_case = base_inputs.copy()
    trim_case["trim_fraction"] = 0.25
    trim_case["env_risks"] = {f"env_{idx}": 0.1 for idx in range(4)}
    layer = {}
    for idx in range(4):
        if idx == 3:
            layer[f"env_{idx}"] = [[10.0 * t, -10.0 * t] for t in range(1, 10)]
        else:
            layer[f"env_{idx}"] = [[1.0 + 0.01 * t, 1.0 + 0.02 * t] for t in range(1, 10)]
    trim_case["layer_activations"] = {"representation": layer}
    metrics = compute_isi(**trim_case)
    assert metrics["ISI_C3_trimmed"] > metrics["ISI_C3"]


def test_isi_uses_trimmed_components(base_inputs) -> None:
    inputs = base_inputs.copy()
    inputs["trim_fraction"] = 0.2
    metrics = compute_isi(**inputs)
    alphas = inputs["alpha_components"]
    expected = (
        alphas[0] * metrics["ISI_C1_trimmed"]
        + alphas[1] * metrics["ISI_C2_trimmed"]
        + alphas[2] * metrics["ISI_C3_trimmed"]
    ) / sum(alphas)
    assert metrics["ISI"] == pytest.approx(expected, rel=1e-6)


def test_ig_reflects_gap() -> None:
    risks = {"env_0": 0.2, "env_1": 0.5, "env_2": 0.25}
    metrics = compute_ig(test_env_risks=risks, tau_IG=0.1, eps=1e-6)
    assert metrics["IG"] == pytest.approx(0.3, rel=1e-6)


def test_compute_invariance_metrics_merges_outputs(base_inputs) -> None:
    ig_inputs = {"test_env_risks": base_inputs["env_risks"], "tau_IG": 0.1, "eps": 1e-8}
    metrics = compute_invariance_metrics(
        isi_inputs=base_inputs,
        ig_inputs=ig_inputs,
    )
    assert "ISI" in metrics and "IG" in metrics
