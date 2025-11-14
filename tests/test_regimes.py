from __future__ import annotations

import datetime as dt

from hirm.envs.regimes import assign_volatility_bands, compute_realized_volatility


def test_volatility_band_thresholds() -> None:
    realized = [0.05, 0.15, 0.3, 0.5]
    labels = assign_volatility_bands(realized)
    assert labels == ["low", "medium", "high", "crisis"]


def test_realized_volatility_window() -> None:
    returns = [0.01] * 40
    realized = compute_realized_volatility(returns, window=20)
    assert len(realized) == len(returns)
    assert realized[19] > 0.0
