from __future__ import annotations

from hirm.risk.cvar import cvar_loss


def test_cvar_monotone() -> None:
    pnl = [1.0, -1.0, -2.0, -3.0]
    cvar_90 = cvar_loss(pnl, alpha=0.9)
    cvar_50 = cvar_loss(pnl, alpha=0.5)
    assert cvar_90 >= cvar_50


def test_cvar_handles_single_episode() -> None:
    pnl = [0.5]
    loss = cvar_loss(pnl)
    assert abs(loss + 0.5) < 1e-6
