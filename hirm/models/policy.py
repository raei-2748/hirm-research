"""Policy models implementing the HIRM representation/head decomposition."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from hirm.models.modules import build_mlp, resolve_activation


class EnvSpecificHead(nn.Module):
    """Collection of per-environment heads with lazy construction."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        head_hidden: Sequence[int],
        activation: str | None,
        dropout: float,
        num_envs: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.activation = activation
        self.dropout = dropout
        self.head_hidden = list(head_hidden)
        self.num_envs = max(1, int(num_envs))
        self.heads = nn.ModuleDict(
            {str(env): _build_head(input_dim, action_dim, head_hidden, activation, dropout) for env in range(self.num_envs)}
        )

    def forward(self, h: Tensor, env_ids: Tensor | None) -> Tensor:  # type: ignore[override]
        if env_ids is None:
            env_ids = h.new_zeros(h.shape[0], dtype=torch.long)  # type: ignore[attr-defined]
        env_tensor = env_ids.view(-1).long()
        outputs = h.new_zeros((h.shape[0], self.action_dim))
        for env in torch.unique(env_tensor):  # type: ignore[attr-defined]
            head = self.heads.get(str(int(env.item())))
            if head is None:
                head = _build_head(self.input_dim, self.action_dim, self.head_hidden, self.activation, self.dropout)
                self.heads[str(int(env.item()))] = head
            mask = env_tensor == env
            outputs[mask] = head(h[mask])
        return outputs

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return super().parameters(recurse=recurse)


class InvariantPolicy(nn.Module):
    """Two-stage policy :math:`\pi_\theta(x) = w_\psi(f_\phi(x))` with ablations."""

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
        self.state_factorization = getattr(cfg_model, "state_factorization", "phi_r")
        self.invariance_mode = getattr(cfg_model, "invariance_mode", "head_only")

        repr_cfg = _resolve_block(cfg_model, "representation")
        head_cfg = _resolve_block(cfg_model, "head")
        phi_cfg = _resolve_block(cfg_model, "phi")
        r_cfg = _resolve_block(cfg_model, "r")
        repr_hidden = repr_cfg.get("hidden_dims") or cfg_model.get("hidden_dims")
        phi_hidden = phi_cfg.get("hidden_dims") or repr_hidden
        if self.state_factorization == "phi_only":
            r_hidden = []
        else:
            r_hidden = r_cfg.get("hidden_dims") or repr_hidden
        if not repr_hidden and self.state_factorization == "no_split":
            raise ValueError("Representation hidden_dims must be specified")

        repr_dropout = float(repr_cfg.get("dropout", cfg_model.get("dropout", 0.0)))
        repr_act = repr_cfg.get("activation") or cfg_model.get("activation", "relu")
        head_hidden = head_cfg.get("hidden_dims") or cfg_model.get("head_hidden_dims", [])
        head_dropout = float(head_cfg.get("dropout", cfg_model.get("head_dropout", 0.0)))
        head_act = head_cfg.get("activation") or cfg_model.get(
            "head_activation", repr_act
        )
        env_head_cfg = head_cfg.get("env_specific", {}) if hasattr(head_cfg, "get") else {}
        num_env_heads = env_head_cfg.get("num_envs", cfg_model.get("num_envs", 4)) if hasattr(env_head_cfg, "get") else 4

        self.representation: nn.Module | None = None
        self.phi_encoder: nn.Module | None = None
        self.r_encoder: nn.Module | None = None
        rep_dim = 0

        if self.state_factorization == "no_split":
            self.representation, rep_dim = build_mlp(
                input_dim=self.input_dim,
                hidden_dims=_ensure_sequence(repr_hidden),
                activation=repr_act,
                dropout=repr_dropout,
            )
        else:
            if phi_hidden:
                self.phi_encoder, phi_dim = build_mlp(
                    input_dim=self.input_dim,
                    hidden_dims=_ensure_sequence(phi_hidden),
                    activation=phi_cfg.get("activation", repr_act),
                    dropout=float(phi_cfg.get("dropout", repr_dropout)),
                )
            else:
                self.phi_encoder, phi_dim = None, 0
            if r_hidden:
                self.r_encoder, r_dim = build_mlp(
                    input_dim=self.input_dim,
                    hidden_dims=_ensure_sequence(r_hidden),
                    activation=r_cfg.get("activation", repr_act),
                    dropout=float(r_cfg.get("dropout", repr_dropout)),
                )
            else:
                self.r_encoder, r_dim = None, 0
            rep_dim = 0
            if self.state_factorization in {"phi_r", "phi_only"} and self.phi_encoder is not None:
                rep_dim += phi_dim
            if self.state_factorization in {"phi_r", "r_only"} and self.r_encoder is not None:
                rep_dim += r_dim
            if rep_dim == 0:
                raise ValueError("At least one representation path must be active")

        if self.invariance_mode == "env_specific_heads":
            self.head: nn.Module = EnvSpecificHead(
                rep_dim,
                self.action_dim,
                _ensure_sequence(head_hidden),
                activation=head_act,
                dropout=head_dropout,
                num_envs=num_env_heads,
            )
        else:
            self.head = _build_head(
                rep_dim,
                self.action_dim,
                _ensure_sequence(head_hidden),
                activation=head_act,
                dropout=head_dropout,
            )
        self._rep_dim = rep_dim

    def forward(self, x: Tensor, env_ids: Tensor | None = None) -> Tensor:
        """Compute hedge actions for ``x`` with configurable factorization."""

        if x.dim() != 2:
            x = x.view(x.shape[0], -1)
        if self.state_factorization == "no_split":
            assert self.representation is not None
            h = self.representation(x)
        else:
            reps: list[Tensor] = []
            if self.state_factorization in {"phi_r", "phi_only"} and self.phi_encoder is not None:
                reps.append(self.phi_encoder(x))
            if self.state_factorization in {"phi_r", "r_only"} and self.r_encoder is not None:
                reps.append(self.r_encoder(x))
            if not reps:
                raise RuntimeError("No active representation paths")
            h = reps[0] if len(reps) == 1 else torch.cat(reps, dim=-1)
        if self.invariance_mode == "env_specific_heads":
            actions = self.head(h, env_ids)
        else:
            actions = self.head(h)
        return actions

    def representation_parameters(self) -> Iterable[nn.Parameter]:
        if self.state_factorization == "no_split":
            assert self.representation is not None
            return self.representation.parameters()
        params: list[nn.Parameter] = []
        if self.phi_encoder is not None and self.state_factorization in {"phi_r", "phi_only"}:
            params.extend(list(self.phi_encoder.parameters()))
        if self.r_encoder is not None and self.state_factorization in {"phi_r", "r_only"}:
            params.extend(list(self.r_encoder.parameters()))
        return params

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()


def _build_head(
    input_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int],
    activation: str | None,
    dropout: float,
) -> nn.Sequential:
    """Construct the head network without forcing hidden layers.

    We keep this helper separate from ``build_mlp`` so that the head can be a
    single linear layer (``hidden_dims`` empty) without adding additional
    special-casing to the shared module builder.
    """

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
