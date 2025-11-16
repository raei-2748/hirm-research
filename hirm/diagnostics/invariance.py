"""Invariance diagnostics (Section 5.1 of the HIRM paper)."""
from __future__ import annotations

import math
import itertools
from typing import Any, Dict, Mapping, Sequence


def _to_flat_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_to_flat_list(item))
        return result
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def _to_matrix(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # pragma: no cover
            pass
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [[float(x) for x in row] for row in value]
    flat = _to_flat_list(value)
    return [[float(x)] for x in flat]


def _trimmed_mean(values: Sequence[float], proportion_to_cut: float = 0.1) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    trim = int(math.floor(n * proportion_to_cut))
    sliced = sorted_vals[trim : n - trim] if n - 2 * trim > 0 else sorted_vals
    return float(sum(sliced) / len(sliced))


def _vector_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _pairwise_cosine(vectors: Sequence[list[float]]) -> Sequence[float]:
    if len(vectors) < 2:
        return [1.0]
    cosines = []
    for vec_a, vec_b in itertools.combinations(vectors, 2):
        denom = _vector_norm(vec_a) * _vector_norm(vec_b)
        if denom <= 0:
            cosines.append(0.0)
        else:
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            cosines.append(max(-1.0, min(1.0, dot / denom)))
    return cosines


def _covariance(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return [[0.0]]
    if len(matrix[0]) == 0:
        return [[0.0]]
    n = len(matrix)
    dim = len(matrix[0])
    means = [sum(row[i] for row in matrix) / n for i in range(dim)]
    cov = [[0.0 for _ in range(dim)] for _ in range(dim)]
    if n <= 1:
        return cov
    for row in matrix:
        centered = [row[i] - means[i] for i in range(dim)]
        for i in range(dim):
            for j in range(dim):
                cov[i][j] += centered[i] * centered[j]
    scale = 1.0 / (n - 1)
    for i in range(dim):
        for j in range(dim):
            cov[i][j] *= scale
    return cov


def _add_identity(matrix: list[list[float]], value: float) -> list[list[float]]:
    dim = len(matrix)
    result = []
    for i in range(dim):
        row = matrix[i][:]
        if i < len(row):
            row[i] += value
        result.append(row)
    return result


def _matrix_trace(matrix: list[list[float]]) -> float:
    return sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0]))))


def _identity(dim: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]


def _matrix_inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    aug = [row[:] + ident_row[:] for row, ident_row in zip(matrix, _identity(n))]
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if abs(aug[row][col]) > 1e-12:
                pivot = row
                break
        if pivot is None:
            return _identity(n)
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        factor = 1.0 / pivot_val
        aug[col] = [v * factor for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            coeff = aug[row][col]
            aug[row] = [curr - coeff * base for curr, base in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def compute_isi(
    env_risks: Mapping[str, float],
    head_gradients: Mapping[str, Any],
    layer_activations: Mapping[str, Mapping[str, Any]],
    tau_R: float,
    tau_C: float,
    alpha_components: Sequence[float],
    eps: float,
    cov_regularizer: float,
    grad_norm_eps: float = 1e-12,
    trim_fraction: float = 0.1,
) -> Dict[str, float]:
    """Compute the Internal Stability Index (Eq. 7)."""

    if len(alpha_components) != 3:
        raise ValueError("alpha_components must contain three weights")
    risk_values = [float(v) for v in env_risks.values()]
    if len(risk_values) > 1:
        mean_risk = sum(risk_values) / len(risk_values)
        variance = sum((v - mean_risk) ** 2 for v in risk_values) / (len(risk_values) - 1)
    else:
        variance = 0.0
    c1_raw = 1.0 - min(1.0, variance / (tau_R + eps))
    c1_raw = max(0.0, min(1.0, c1_raw))

    normalized_grads: list[list[float]] = []
    for env_id in env_risks:
        grad = head_gradients.get(env_id)
        if grad is None:
            continue
        vec = _to_flat_list(grad)
        if not vec:
            continue
        norm = _vector_norm(vec)
        if norm <= grad_norm_eps:
            continue
        normalized_grads.append([v / (norm + grad_norm_eps) for v in vec])
    if normalized_grads:
        cosines = _pairwise_cosine(normalized_grads)
        c2_raw = float(sum((1.0 + c) / 2.0 for c in cosines) / len(cosines))
    else:
        c2_raw = 0.0
    c2_raw = max(0.0, min(1.0, c2_raw))

    dispersion_values: list[float] = []
    for env_map in layer_activations.values():
        covariances = []
        for env_id in env_risks:
            if env_id not in env_map:
                continue
            acts = _to_matrix(env_map[env_id])
            covariances.append(_covariance(acts))
        if not covariances:
            continue
        dim = len(covariances[0])
        avg_cov = [[0.0 for _ in range(dim)] for _ in range(dim)]
        for cov in covariances:
            for i in range(dim):
                for j in range(dim):
                    avg_cov[i][j] += cov[i][j] / len(covariances)
        ref_cov = _add_identity(avg_cov, cov_regularizer)
        ref_inv = _matrix_inverse(ref_cov)
        for cov in covariances:
            cov_reg = _add_identity(cov, cov_regularizer)
            trace_val = _matrix_trace(_matrix_multiply(ref_inv, cov_reg))
            dispersion_values.append(abs(trace_val - dim))
    if dispersion_values:
        mean_disp = sum(dispersion_values) / len(dispersion_values)
        c3_raw = 1.0 - min(1.0, mean_disp / (tau_C + eps))
    else:
        c3_raw = 1.0
    c3_raw = max(0.0, min(1.0, c3_raw))

    c1_trim = _trimmed_mean([c1_raw], trim_fraction)
    c2_trim = _trimmed_mean([c2_raw], trim_fraction)
    c3_trim = _trimmed_mean([c3_raw], trim_fraction)

    total_alpha = max(sum(alpha_components), eps)
    isi = (
        alpha_components[0] * c1_trim
        + alpha_components[1] * c2_trim
        + alpha_components[2] * c3_trim
    ) / total_alpha
    isi = max(0.0, min(1.0, isi))

    return {
        "ISI": isi,
        "ISI_C1": c1_raw,
        "ISI_C2": c2_raw,
        "ISI_C3": c3_raw,
        "ISI_C1_trimmed": c1_trim,
        "ISI_C2_trimmed": c2_trim,
        "ISI_C3_trimmed": c3_trim,
    }


def _matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            for j in range(cols):
                result[i][j] += a[i][k] * b[k][j]
    return result


def compute_ig(
    test_env_risks: Mapping[str, float],
    tau_IG: float,
    eps: float,
) -> Dict[str, float]:
    """Compute the outcome-level invariance gap (Eq. 8)."""

    values = [float(v) for v in test_env_risks.values()]
    if not values:
        raise ValueError("test_env_risks cannot be empty")
    ig = max(values) - min(values)
    ig_norm = ig / (tau_IG + eps)
    return {"IG": ig, "IG_norm": ig_norm}


__all__ = ["compute_isi", "compute_ig"]
