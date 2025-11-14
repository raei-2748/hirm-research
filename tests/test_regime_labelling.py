from __future__ import annotations

from hirm.envs.regime_labelling import label_regimes


def test_regime_labels(tmp_path) -> None:
    returns = [(-0.01 + idx * 0.0005) for idx in range(40)]
    labels = label_regimes(returns, window=5, save=True, save_path=tmp_path)
    assert len(labels) == len(returns)
    assert set(labels).issubset({0, 1, 2})
    saved = tmp_path / "latest_regimes.json"
    assert saved.exists()
