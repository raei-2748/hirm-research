"""Policy abstractions implemented with pure Python lists."""
from __future__ import annotations

import math
import random
from typing import List, Sequence


class Policy:
    def forward_representation(self, x: List[float]) -> List[float]:
        raise NotImplementedError

    def forward_head(self, z: List[float]) -> List[float]:
        raise NotImplementedError

    def forward(self, x: List) -> List:
        if not x:
            return []
        # Detect shape
        if isinstance(x[0], list) and x and isinstance(x[0][0], list):
            return [[self._forward_single(vector) for vector in sequence] for sequence in x]
        return [self._forward_single(vector) for vector in x]

    def __call__(self, x: List) -> List:
        return self.forward(x)

    def _forward_single(self, features: List[float]) -> List[float]:
        rep = self.forward_representation(features)
        return self.forward_head(rep)

    def parameters_representation(self) -> List[List[List[float]] | List[float]]:
        raise NotImplementedError

    def parameters_head(self) -> List[List[List[float]] | List[float]]:
        raise NotImplementedError

    def parameters_all(self) -> List[List[List[float]] | List[float]]:
        return self.parameters_representation() + self.parameters_head()


class MLPPolicy(Policy):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        head_dim: int = 1,
        activation: str = "relu",
        seed: int = 0,
    ) -> None:
        self._activation_name = activation
        hidden_dims = tuple(hidden_dims or (32, 32))
        self._rng = random.Random(seed)
        self._representation_layers: List[dict[str, List]] = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            weight = [[self._rng.gauss(0.0, 0.1) for _ in range(prev_dim)] for _ in range(hidden)]
            bias = [0.0 for _ in range(hidden)]
            self._representation_layers.append({"weight": weight, "bias": bias})
            prev_dim = hidden
        self._head_weight = [[self._rng.gauss(0.0, 0.1) for _ in range(prev_dim)] for _ in range(head_dim)]
        self._head_bias = [0.0 for _ in range(head_dim)]

    def _activation(self, value: float) -> float:
        if self._activation_name == "relu":
            return max(0.0, value)
        if self._activation_name == "gelu":
            return 0.5 * value * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
        raise ValueError(f"Unknown activation {self._activation_name}")

    def forward_representation(self, x: List[float]) -> List[float]:  # type: ignore[override]
        out = x
        for layer in self._representation_layers:
            transformed: List[float] = []
            for i, weights in enumerate(layer["weight"]):
                total = sum(weights[j] * out[j] for j in range(len(weights))) + layer["bias"][i]
                transformed.append(self._activation(total))
            out = transformed
        return out

    def forward_head(self, z: List[float]) -> List[float]:  # type: ignore[override]
        outputs = []
        for i, weights in enumerate(self._head_weight):
            total = sum(weights[j] * z[j] for j in range(len(weights))) + self._head_bias[i]
            outputs.append(total)
        return outputs

    def parameters_representation(self) -> List[List[List[float]] | List[float]]:  # type: ignore[override]
        params: List[List[List[float]] | List[float]] = []
        for layer in self._representation_layers:
            params.extend([layer["weight"], layer["bias"]])
        return params

    def parameters_head(self) -> List[List[List[float]] | List[float]]:  # type: ignore[override]
        return [self._head_weight, self._head_bias]


__all__ = ["Policy", "MLPPolicy"]
