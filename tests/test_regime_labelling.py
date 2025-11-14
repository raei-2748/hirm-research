from __future__ import annotations

from hirm.envs.regime_labelling import (
    DEFAULT_REGIME_THRESHOLDS,
    compute_realized_vol,
    label_regimes,
)


def test_regime_labels_align_with_prices() -> None:
    prices = [100 + idx * 0.5 for idx in range(50)]
    realized_vol = compute_realized_vol(prices, window=5)
    assert len(realized_vol) == len(prices)
    labels = label_regimes(realized_vol, thresholds=DEFAULT_REGIME_THRESHOLDS)
    assert len(labels) == len(prices)
    assert set(labels).issubset({0, 1, 2, 3})
