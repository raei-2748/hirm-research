import types

import torch

from hirm.objectives.hirm import HIRMObjective


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.head(x)

    def head_parameters(self):
        return self.head.parameters()


def _make_cfg(lambda_hirm: float):
    obj = types.SimpleNamespace(name="hirm", lambda_hirm=lambda_hirm, invariance_mode="head_only", eps=1e-8)
    return types.SimpleNamespace(objective=obj)


def test_hirm_penalty_normalizes_gradients():
    model = _TinyModel()
    cfg = _make_cfg(lambda_hirm=1.0)
    objective = HIRMObjective(cfg, device=torch.device("cpu"))

    x_a = torch.tensor([[1.0, 0.0]], requires_grad=False)
    x_b = torch.tensor([[0.5, 1.0]], requires_grad=False)
    env_risks = {
        "env_a": (model(x_a) ** 2).mean(),
        "env_b": (model(x_b) ** 2).mean(),
    }
    scaled_env_risks = {k: v * 5.0 for k, v in env_risks.items()}

    logs_one: dict = {}
    logs_two: dict = {}
    _ = objective.compute_loss(env_risks, model, batch={}, extra_state=logs_one)
    _ = objective.compute_loss(scaled_env_risks, model, batch={}, extra_state=logs_two)

    penalty_one = logs_one["objective_logs"]["train/objective/hirm_penalty"]
    penalty_two = logs_two["objective_logs"]["train/objective/hirm_penalty"]
    assert torch.allclose(penalty_one, penalty_two, atol=1e-5), "Penalty should be scale-invariant after normalization"


def test_hirm_penalty_weight_increases_loss():
    model = _TinyModel()
    env_risks = {
        "env_a": (model(torch.tensor([[1.0, -0.5]])).sum()),
        "env_b": (model(torch.tensor([[0.2, 0.4]])).sum()),
    }

    cfg_low = _make_cfg(lambda_hirm=0.1)
    cfg_high = _make_cfg(lambda_hirm=5.0)

    obj_low = HIRMObjective(cfg_low, device=torch.device("cpu"))
    obj_high = HIRMObjective(cfg_high, device=torch.device("cpu"))

    loss_low = obj_low.compute_loss(env_risks, model, batch={})
    loss_high = obj_high.compute_loss(env_risks, model, batch={})

    assert loss_high > loss_low, "Higher lambda_hirm should raise the total loss"
