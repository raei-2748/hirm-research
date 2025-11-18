import torch
from torch import nn

from hirm.diagnostics.invariance import compute_isi
from hirm.diagnostics.invariance_helpers import collect_invariance_signals


class TinyModel(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.representation = nn.Linear(input_dim, 4)
        self.head = nn.Linear(4, action_dim)
        self.invariance_mode = "head_only"

    def head_parameters(self):
        return self.head.parameters()

    def forward(self, x, env_ids=None):  # type: ignore[override]
        h = torch.relu(self.representation(x))
        return self.head(h)


def _make_env(env_id: int) -> dict[str, torch.Tensor]:
    batch = 5
    features = torch.randn(batch, 3)
    hedge_returns = torch.randn(batch, 2)
    base_pnl = torch.randn(batch)
    env_ids = torch.full((batch,), env_id, dtype=torch.long)
    return {
        "features": features,
        "hedge_returns": hedge_returns,
        "base_pnl": base_pnl,
        "env_ids": env_ids,
    }


def test_collect_invariance_signals_and_compute_isi():
    model = TinyModel(input_dim=3, action_dim=2)

    train_envs = {"env_a": _make_env(0), "env_b": _make_env(1)}

    risk_fn = lambda pnl: -pnl.mean()
    head_grads, activations = collect_invariance_signals(
        model, train_envs, model.invariance_mode, torch.device("cpu"), risk_fn, max_samples_per_env=4
    )

    assert head_grads, "Head gradients should not be empty"
    assert activations, "Activations should not be empty"

    env_risks = {"env_0": 0.1, "env_1": 0.2}
    isi = compute_isi(
        env_risks=env_risks,
        head_gradients=head_grads,
        layer_activations=activations,
        tau_R=0.05,
        tau_C=1.0,
        alpha_components=[1.0, 1.0, 1.0],
        eps=1e-8,
        cov_regularizer=1e-4,
    )

    for key in ["ISI", "ISI_C1", "ISI_C2", "ISI_C3"]:
        assert key in isi
