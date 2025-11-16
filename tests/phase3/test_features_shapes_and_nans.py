import math

import numpy as np

from hirm.episodes.episode import Episode
from hirm.state.features import compute_all_features


def _build_episode() -> Episode:
    prices = np.array([100.0, 101.0, 103.0, 102.0, 104.0, 105.0], dtype=float)
    returns = np.diff(np.log(prices))
    metadata = {"regimes": np.zeros(len(prices))}
    return Episode(prices=prices, returns=returns, metadata=metadata)


def _has_nan(array) -> bool:
    if hasattr(array, "tolist"):
        data = array.tolist()
    else:
        data = list(array)

    def _recursive(values) -> bool:
        if isinstance(values, list):
            return any(_recursive(v) for v in values)
        try:
            return math.isnan(float(values))
        except Exception:
            return False

    return _recursive(data)


def test_feature_shapes_and_nans() -> None:
    episode = _build_episode()
    features = compute_all_features(episode, env_id=1, config=None)
    phi = features["phi"]
    r = features["r"]
    assert phi.shape[0] == episode.length
    assert r.shape[0] == episode.length
    assert not _has_nan(phi)
    assert not _has_nan(r)
