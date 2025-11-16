"""Policy models implementing the HIRM representation/head decomposition."""
from __future__ import annotations

from typing import Iterable, Sequence

from torch import Tensor, nn

from hirm.models.modules import build_mlp, resolve_activation


class InvariantPolicy(nn.Module):
    """Policy with a learnable representation ``f_phi`` and head ``w_psi``."""

    def __init__(
        self,
        cfg_model,
        input_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        if self.input_dim <= 0 or self.action_dim <= 0:
            raise ValueError("input_dim and action_dim must be positive")
        repr_cfg = _resolve_block(cfg_model, "representation")
        head_cfg = _resolve_block(cfg_model, "head")
        repr_hidden = repr_cfg.get("hidden_dims") or cfg_model.get("hidden_dims")
        if not repr_hidden:
            raise ValueError("Representation hidden_dims must be specified")
        repr_dropout = float(repr_cfg.get("dropout", cfg_model.get("dropout", 0.0)))
        repr_act = repr_cfg.get("activation") or cfg_model.get("activation", "relu")
        representation, rep_dim = build_mlp(
            input_dim=self.input_dim,
            hidden_dims=_ensure_sequence(repr_hidden),
            activation=repr_act,
            dropout=repr_dropout,
        )
        head_hidden = head_cfg.get("hidden_dims") or cfg_model.get("head_hidden_dims", [])
        head_dropout = float(head_cfg.get("dropout", cfg_model.get("head_dropout", 0.0)))
        head_act = head_cfg.get("activation") or cfg_model.get(
            "head_activation", repr_act
        )
        self.representation = representation
        self.head = _build_head(
            rep_dim,
            self.action_dim,
            _ensure_sequence(head_hidden),
            activation=head_act,
            dropout=head_dropout,
        )
        if not any(param.requires_grad for param in self.representation.parameters()):
            raise ValueError("Representation must have trainable parameters")
        if not any(param.requires_grad for param in self.head.parameters()):
            raise ValueError("Head must have trainable parameters")

    def forward(self, x: Tensor, env_ids: Tensor | None = None) -> Tensor:
        """Compute hedge actions for ``x``."""

        del env_ids
        if x.dim() != 2:
            x = x.view(x.shape[0], -1)
        h = self.representation(x)
        actions = self.head(h)
        return actions

    def representation_parameters(self) -> Iterable[nn.Parameter]:
        """Return an iterator over representation parameters (``phi``)."""

        return self.representation.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        """Return an iterator over head parameters (``psi``)."""

        return self.head.parameters()


def _build_head(
    input_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int],
    activation: str | None,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    act_module = resolve_activation(activation)
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, int(dim)))
        layers.append(act_module.__class__())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = int(dim)
    layers.append(nn.Linear(prev, action_dim))
    return nn.Sequential(*layers)


def _ensure_sequence(values) -> Sequence[int]:  # type: ignore[no-untyped-def]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return [int(v) for v in values]
    return [int(values)]


def _resolve_block(cfg_model, key: str):  # type: ignore[no-untyped-def]
    if hasattr(cfg_model, key):
        return getattr(cfg_model, key)
    return cfg_model.get(key, {}) if hasattr(cfg_model, "get") else {}


__all__ = ["InvariantPolicy"]
