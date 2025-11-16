"""Reusable neural network building blocks for HIRM models."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from torch import nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


def resolve_activation(name: str | None) -> nn.Module:
    """Return an activation module for ``name`` (defaults to ReLU)."""

    if not name:
        return nn.ReLU()
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'")
    return _ACTIVATIONS[key]()


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: str | None = "relu",
    dropout: float = 0.0,
) -> tuple[nn.Sequential, int]:
    """Construct an MLP and return it along with the output dimension."""

    layers: List[nn.Module] = []
    prev_dim = int(input_dim)
    if prev_dim <= 0:
        raise ValueError("input_dim must be positive")
    hidden_dims = [int(dim) for dim in hidden_dims]
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")
    act = resolve_activation(activation)
    for idx, dim in enumerate(hidden_dims):
        if dim <= 0:
            raise ValueError("hidden_dims entries must be positive")
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(act.__class__())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = dim
    return nn.Sequential(*layers), prev_dim


def count_parameters(modules: Iterable[nn.Module]) -> int:
    """Count the total number of parameters across ``modules``."""

    total = 0
    for module in modules:
        total += sum(param.numel() for param in module.parameters())
    return total


__all__ = ["build_mlp", "count_parameters", "resolve_activation"]
