from __future__ import annotations

import random

from hirm.models.policy import MLPPolicy


def test_policy_partition() -> None:
    policy = MLPPolicy(input_dim=2, hidden_dims=[8], head_dim=1)
    rep_params = list(policy.parameters_representation())
    head_params = list(policy.parameters_head())
    assert rep_params
    assert head_params
    rep_ids = {id(p) for p in rep_params}
    head_ids = {id(p) for p in head_params}
    assert rep_ids.isdisjoint(head_ids)


def test_policy_forward_shapes() -> None:
    policy = MLPPolicy(input_dim=2, hidden_dims=[4], head_dim=1)
    random.seed(0)
    batch = [[random.random(), random.random()] for _ in range(3)]
    out = policy(batch)
    assert len(out) == 3 and len(out[0]) == 1
    seq = [
        [[random.random(), random.random()] for _ in range(5)]
        for _ in range(2)
    ]
    seq_out = policy(seq)
    assert len(seq_out) == 2 and len(seq_out[0]) == 5
