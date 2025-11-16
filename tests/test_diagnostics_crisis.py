from __future__ import annotations

import pytest

from hirm.diagnostics.crisis import compute_crisis_cvar


def test_compute_crisis_cvar_uses_losses() -> None:
    pnl = [0.1, -0.2, 0.05, -0.1]
    metrics = compute_crisis_cvar(pnl, alpha=0.5)
    assert "crisis_cvar" in metrics
    assert metrics["crisis_cvar"] >= 0


def test_compute_crisis_cvar_requires_values() -> None:
    with pytest.raises(ValueError):
        compute_crisis_cvar([], alpha=0.9)
