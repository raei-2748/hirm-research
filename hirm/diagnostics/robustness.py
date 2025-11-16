"""Robustness diagnostics (Section 5.2)."""
from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


def compute_wg(
    env_risks: Mapping[str, float],
    alpha: float,
    tau_min: float | None = None,
    tau_max: float | None = None,
    num_grid: int = 1000,
) -> Dict[str, float]:
    """Compute the worst-case generalization metric (CVaR surrogate)."""

    risks = [float(v) for v in env_risks.values()]
    if not risks:
        raise ValueError("env_risks cannot be empty")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    min_risk = min(risks)
    max_risk = max(risks)
    spread = max(1e-8, max_risk - min_risk)
    if tau_min is None:
        tau_min = min_risk - 0.25 * spread
    if tau_max is None:
        tau_max = max_risk + 0.25 * spread
    if tau_max <= tau_min:
        tau_max = tau_min + spread
    step = (tau_max - tau_min) / max(num_grid - 1, 1)
    wg_value = math.inf
    for idx in range(num_grid):
        tau = tau_min + idx * step
        positive_part = [(risk - tau) if risk > tau else 0.0 for risk in risks]
        candidate = tau + (sum(positive_part) / len(risks)) / alpha
        wg_value = min(wg_value, candidate)
    return {"WG": float(wg_value)}


def compute_vr(
    risk_time_series: Sequence[float],
    eps: float,
) -> Dict[str, float]:
    """Compute the volatility ratio (std / mean) from Section 5.2."""

    series = [float(v) for v in risk_time_series]
    if not series:
        raise ValueError("risk_time_series cannot be empty")
    mean = sum(series) / len(series)
    if len(series) > 1:
        variance = sum((x - mean) ** 2 for x in series) / (len(series) - 1)
    else:
        variance = 0.0
    std = math.sqrt(max(variance, 0.0))
    vr = std / (mean + eps)
    return {"VR": vr}


__all__ = ["compute_wg", "compute_vr"]
