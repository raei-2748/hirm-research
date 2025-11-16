"""Risk function builders."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor


RiskFn = Callable[[Tensor], Tensor]


def build_risk_function(cfg_objective) -> RiskFn:  # type: ignore[no-untyped-def]
    """Instantiate the coherent risk functional described in the paper."""

    risk_cfg = getattr(cfg_objective, "risk", None)
    name = None
    alpha = None
    if risk_cfg is not None:
        name = getattr(risk_cfg, "name", None)
        alpha = getattr(risk_cfg, "alpha", None)
    if name is None:
        name = getattr(cfg_objective, "risk_name", None)
    if alpha is None:
        alpha = getattr(cfg_objective, "alpha", 0.95)
    name = (name or "cvar").lower()
    if name not in {"cvar", "conditional_value_at_risk"}:
        raise ValueError(f"Unsupported risk function '{name}'")
    return make_cvar(alpha=float(alpha))


def make_cvar(alpha: float = 0.95) -> RiskFn:
    """Return a differentiable Conditional Value-at-Risk closure.

    We follow the standard PnL convention where positive values are good and
    negative values are bad.  Risk is therefore the *expected loss* in the
    worst ``(1 - alpha)`` fraction of outcomes, so larger values correspond to
    worse risk.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    def _cvar(pnl: Tensor) -> Tensor:
        if pnl.numel() == 0:
            raise ValueError("pnl tensor must be non-empty")

        losses = -pnl.reshape(-1)
        sorted_losses, _ = torch.sort(losses)
        n = sorted_losses.numel()
        tail_fraction = max(1, int(math.ceil(n * (1.0 - alpha))))
        tail = sorted_losses[-tail_fraction:]
        return tail.mean()

    return _cvar


__all__ = ["build_risk_function", "make_cvar", "RiskFn"]
