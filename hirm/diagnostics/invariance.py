"""Invariance diagnostics (Section 5.1 of the HIRM paper)."""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Mapping, Sequence


try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


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
    proportion = float(max(0.0, min(0.5, proportion_to_cut)))
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    trim = int(math.floor(n * proportion))
    if trim == 0 or 2 * trim >= n:
        sliced = sorted_vals
    else:
        sliced = sorted_vals[trim : n - trim]
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


def _covariance_torch(matrix: list[list[float]]):
    if torch is None:  # pragma: no cover - guard
        raise RuntimeError("Torch is not available")
    if not matrix:
        return torch.zeros((1, 1), dtype=torch.float64)
    tensor = torch.tensor(matrix, dtype=torch.float64)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[0] <= 1:
        dim = tensor.shape[1]
        return torch.zeros((dim, dim), dtype=torch.float64)
    centered = tensor - tensor.mean(dim=0, keepdim=True)
    return centered.t().mm(centered) / (tensor.shape[0] - 1)


def _robust_inverse_torch(matrix, cov_regularizer: float):
    if torch is None:  # pragma: no cover - guard
        raise RuntimeError("Torch is not available")
    if matrix.numel() == 0:
        return matrix
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
    attempt = matrix.clone()
    jitter = max(cov_regularizer, 0.0)
    for _ in range(5):
        try:
            return torch.linalg.inv(attempt)
        except RuntimeError:
            jitter = jitter * 10.0 if jitter > 0 else 1e-6
            attempt = matrix + jitter * identity
    return torch.linalg.pinv(attempt)


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

    # Force alpha_components to be a clean float list
    try:
        alpha_components = [float(a) for a in alpha_components]
    except Exception:
        # Fallback if any element is not convertible
        alpha_components = [1.0, 1.0, 1.0]

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
        c2_values = [float((1.0 + c) / 2.0) for c in cosines]
        c2_raw = float(sum(c2_values) / len(c2_values))
        c2_trim = _trimmed_mean(c2_values, trim_fraction)
    else:
        c2_values = []
        c2_raw = 0.0
        c2_trim = 0.0
    c2_raw = max(0.0, min(1.0, c2_raw))
    c2_trim = max(0.0, min(1.0, c2_trim))

    dispersion_values: list[float] = []
    use_torch = torch is not None
    for env_map in layer_activations.values():
        if use_torch:
            dispersion_values.extend(
                _layer_dispersion_torch(env_map, env_risks, cov_regularizer)
            )
        else:
            dispersion_values.extend(
                _layer_dispersion_python(env_map, env_risks, cov_regularizer)
            )
    if dispersion_values:
        raw_disp = sum(dispersion_values) / len(dispersion_values)
        trimmed_disp = _trimmed_mean(dispersion_values, trim_fraction)
        c3_raw = 1.0 - min(1.0, raw_disp / (tau_C + eps))
        c3_trim = 1.0 - min(1.0, trimmed_disp / (tau_C + eps))
    else:
        c3_raw = 1.0
        c3_trim = 1.0
    c3_raw = max(0.0, min(1.0, c3_raw))
    c3_trim = max(0.0, min(1.0, c3_trim))

    # C1 is a single statistic; trimming is a no-op but returned for API symmetry.
    c1_trim = c1_raw

    total_alpha = float(sum(alpha_components))
    if total_alpha <= eps:
        normalized_alphas = [1.0 / 3.0] * 3
    else:
        normalized_alphas = [float(w) / total_alpha for w in alpha_components]
    isi = (
        normalized_alphas[0] * c1_trim
        + normalized_alphas[1] * c2_trim
        + normalized_alphas[2] * c3_trim
    )
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
def _layer_dispersion_torch(
    env_map: Mapping[str, Any],
    env_risks: Mapping[str, float],
    cov_regularizer: float,
) -> list[float]:
    if torch is None:
        return []
    covariances = []
    for env_id in env_risks:
        if env_id not in env_map:
            continue
        acts = _to_matrix(env_map[env_id])
        covariances.append(_covariance_torch(acts))
    if not covariances:
        return []
    dim = covariances[0].shape[0]
    avg_cov = torch.stack(covariances, dim=0).mean(dim=0)
    ref_cov = avg_cov + cov_regularizer * torch.eye(dim, dtype=avg_cov.dtype)
    ref_inv = _robust_inverse_torch(ref_cov, cov_regularizer)
    eye = torch.eye(dim, dtype=avg_cov.dtype)
    values: list[float] = []
    for cov in covariances:
        cov_reg = cov + cov_regularizer * eye
        trace_val = float(torch.trace(ref_inv @ cov_reg).item())
        values.append(abs(trace_val - dim))
    return values


def _layer_dispersion_python(
    env_map: Mapping[str, Any],
    env_risks: Mapping[str, float],
    cov_regularizer: float,
) -> list[float]:
    covariances = []
    for env_id in env_risks:
        if env_id not in env_map:
            continue
        acts = _to_matrix(env_map[env_id])
        cov = _covariance_list(acts)
        covariances.append(cov)
    if not covariances:
        return []
    dim = len(covariances[0])
    avg_cov = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for cov in covariances:
        for i in range(dim):
            for j in range(dim):
                avg_cov[i][j] += cov[i][j] / len(covariances)
    ref_cov = _add_identity(avg_cov, cov_regularizer)
    ref_inv = _robust_inverse_list(ref_cov, cov_regularizer)
    values: list[float] = []
    for cov in covariances:
        cov_reg = _add_identity(cov, cov_regularizer)
        product = _matrix_multiply(ref_inv, cov_reg)
        trace_val = _matrix_trace(product)
        values.append(abs(trace_val - dim))
    return values


def _covariance_list(matrix: list[list[float]]) -> list[list[float]]:
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


def _cholesky_decomposition(matrix: list[list[float]]) -> list[list[float]]:
    dim = len(matrix)
    lower = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(i + 1):
            s = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                value = matrix[i][i] - s
                if value <= 0.0:
                    raise ValueError("Matrix is not positive definite")
                lower[i][j] = math.sqrt(value)
            else:
                if abs(lower[j][j]) <= 1e-12:
                    raise ValueError("Singular matrix")
                lower[i][j] = (matrix[i][j] - s) / lower[j][j]
    return lower


def _forward_substitution(lower: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    dim = len(lower)
    cols = len(rhs[0])
    result = [[0.0 for _ in range(cols)] for _ in range(dim)]
    for i in range(dim):
        for col in range(cols):
            value = rhs[i][col] - sum(lower[i][k] * result[k][col] for k in range(i))
            result[i][col] = value / lower[i][i]
    return result


def _backward_substitution(lower: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    dim = len(lower)
    cols = len(rhs[0])
    result = [[0.0 for _ in range(cols)] for _ in range(dim)]
    for i in range(dim - 1, -1, -1):
        for col in range(cols):
            value = rhs[i][col] - sum(lower[k][i] * result[k][col] for k in range(i + 1, dim))
            result[i][col] = value / lower[i][i]
    return result


def _solve_spd(matrix: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    lower = _cholesky_decomposition(matrix)
    y = _forward_substitution(lower, rhs)
    x = _backward_substitution(lower, y)
    return x


def _robust_inverse_list(matrix: list[list[float]], cov_regularizer: float) -> list[list[float]]:
    dim = len(matrix)
    identity = _identity(dim)
    attempt = [row[:] for row in matrix]
    jitter = max(cov_regularizer, 0.0)
    for _ in range(5):
        try:
            return _solve_spd(attempt, identity)
        except ValueError:
            jitter = jitter * 10.0 if jitter > 0 else 1e-6
            attempt = _add_identity(matrix, jitter)
    return identity


