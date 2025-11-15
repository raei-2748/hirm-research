import numpy as np

from hirm.envs.regime_labelling import label_series_with_regimes, realized_vol_20
from hirm.envs.regimes import map_vol_to_regime


def test_map_vol_to_regime_boundaries():
    assert map_vol_to_regime(0.05) == 0
    assert map_vol_to_regime(0.10) == 1
    assert map_vol_to_regime(0.249) == 1
    assert map_vol_to_regime(0.25) == 2
    assert map_vol_to_regime(0.4) == 3


def test_realized_vol_rolling_window():
    returns = np.array([0.01] * 25)
    vols = realized_vol_20(returns)
    assert vols.shape == returns.shape
    assert all(v >= 0 for v in vols)


def test_label_series_length_matches():
    prices = np.linspace(100, 120, 40)
    regimes = label_series_with_regimes(prices)
    assert regimes.shape[0] == prices.shape[0]
    assert set(regimes).issubset({0, 1, 2, 3})
