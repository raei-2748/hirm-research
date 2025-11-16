from __future__ import annotations

import math
import random

import pytest

from hirm.diagnostics import compute_all_diagnostics


def _axis_inputs(env_risks, head_gradients, layer_acts, test_env_risks, risk_series, returns, actions):
    invariance_inputs = {
        "isi_inputs": {
            "env_risks": env_risks,
            "head_gradients": head_gradients,
            "layer_activations": layer_acts,
            "tau_R": 0.1,
            "tau_C": 1.0,
            "alpha_components": [1.0, 1.0, 1.0],
            "eps": 1e-8,
            "cov_regularizer": 1e-4,
        },
        "ig_inputs": {
            "test_env_risks": test_env_risks,
            "tau_IG": 0.05,
            "eps": 1e-8,
        },
    }
    robustness_inputs = {
        "wg_inputs": {"env_risks": test_env_risks, "alpha": 0.05},
        "vr_inputs": {"risk_time_series": risk_series, "eps": 1e-8},
    }
    efficiency_inputs = {
        "er_inputs": {
            "returns_time_series": returns,
            "cvar_alpha": 0.1,
            "eps": 1e-8,
        },
        "tr_inputs": {
            "actions": actions,
            "eps": 1e-8,
        },
    }
    return invariance_inputs, robustness_inputs, efficiency_inputs


def test_end_to_end_hirm_beats_erm_on_invariance() -> None:
    env_ids = ["env_0", "env_1", "env_2"]
    erm_env_risks = {env: risk for env, risk in zip(env_ids, [0.3, 0.6, 0.4])}
    hirm_env_risks = {env: risk for env, risk in zip(env_ids, [0.35, 0.4, 0.38])}
    rng = random.Random(0)
    erm_head = {env: [rng.random() for _ in range(4)] for env in env_ids}
    hirm_head = {env: [1.0 for _ in range(4)] for env in env_ids}
    erm_layers = {
        "representation": {env: [[rng.random(), rng.random()] for _ in range(12)] for env in env_ids}
    }
    hirm_layers = {
        "representation": {
            env: [[1.0 + 0.01 * rng.random(), 1.0 + 0.01 * rng.random()] for _ in range(12)]
            for env in env_ids
        }
    }
    test_env_risks_erm = {env: risk for env, risk in zip(env_ids, [0.28, 0.65, 0.33])}
    test_env_risks_hirm = {env: risk for env, risk in zip(env_ids, [0.32, 0.38, 0.36])}
    erm_risk_series = [0.2 + (0.8 - 0.2) * i / 49 for i in range(50)]
    hirm_risk_series = [0.3 + (0.45 - 0.3) * i / 49 for i in range(50)]
    erm_returns = [0.1 + (0.4 - 0.1) * i / 49 for i in range(50)]
    hirm_returns = [0.12 + (0.35 - 0.12) * i / 49 for i in range(50)]
    erm_actions = [
        [math.sin(t), math.cos(t)]
        for t in [3.14 * i / 29 for i in range(30)]
    ]
    hirm_actions = [[0.5 * val for val in action] for action in erm_actions]

    erm_inputs = _axis_inputs(
        erm_env_risks,
        erm_head,
        erm_layers,
        test_env_risks_erm,
        erm_risk_series,
        erm_returns,
        erm_actions,
    )
    hirm_inputs = _axis_inputs(
        hirm_env_risks,
        hirm_head,
        hirm_layers,
        test_env_risks_hirm,
        hirm_risk_series,
        hirm_returns,
        hirm_actions,
    )

    erm_metrics = compute_all_diagnostics(*erm_inputs)
    hirm_metrics = compute_all_diagnostics(*hirm_inputs)

    assert hirm_metrics["ISI"] > erm_metrics["ISI"]
    assert hirm_metrics["IG"] < erm_metrics["IG"]
    assert hirm_metrics["VR"] < erm_metrics["VR"]
    assert hirm_metrics["TR"] < erm_metrics["TR"]
    assert hirm_metrics["ER"] == pytest.approx(hirm_metrics["ER"], rel=1e-6)
