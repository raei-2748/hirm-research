import math

import torch

from hirm.diagnostics.efficiency import compute_er, compute_tr
from hirm.diagnostics.invariance import compute_ig, compute_isi
from hirm.diagnostics.robustness import compute_vr, compute_wg
from hirm.utils.math import cvar


def test_metrics_handle_small_arrays():
    wg = compute_wg({"a": 0.1, "b": 0.2}, alpha=0.2, num_grid=5)
    assert wg["WG"] > 0

    vr = compute_vr([0.0, 0.1, 0.2], eps=1e-6)
    assert math.isfinite(vr["VR"])

    er = compute_er([0.1, 0.2, -0.05], cvar_alpha=0.5, eps=1e-6, mode="returns")
    assert math.isfinite(er["ER"])

    tr = compute_tr([[1.0, 2.0, 3.0]], eps=1e-6, treat_first_step="zero")
    assert tr["TR"] >= 0

    ig = compute_ig({"env_0": 0.1, "env_1": 0.15}, tau_IG=0.05, eps=1e-6)
    assert ig["IG"] > 0

    head_grads = {"env_0": [1.0, 0.0], "env_1": [0.5, 0.5]}
    layer_acts = {"representation": {"env_0": torch.ones(2, 2), "env_1": torch.zeros(2, 2)}}
    isi = compute_isi(
        env_risks={"env_0": 0.1, "env_1": 0.2},
        head_gradients=head_grads,
        layer_activations=layer_acts,
        tau_R=0.2,
        tau_C=1.0,
        alpha_components=[1.0, 1.0, 1.0],
        eps=1e-6,
        cov_regularizer=1e-4,
    )
    assert 0.0 <= isi["ISI"] <= 1.0


def test_cvar_edge_cases_do_not_crash():
    assert cvar([0.0, 0.0], alpha=0.5) == 0.0
    values = [0.1 for _ in range(3)]
    assert cvar(values, alpha=0.5) <= max(values)
