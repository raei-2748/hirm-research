import numpy as np

from hirm.episodes.episode import Episode


def test_episode_roundtrip(tmp_path):
    prices = np.array([1.0, 1.1, 1.2])
    returns = np.array([0.1, 0.09])
    metadata = {"regimes": np.array([0, 1, 1]), "vol_20": np.array([0.1, 0.2, 0.3])}
    episode = Episode(prices, returns, metadata)
    assert episode.length == 2
    payload = episode.to_dict()
    restored = Episode.from_dict(payload)
    np.testing.assert_allclose(restored.prices, prices)
    np.testing.assert_allclose(restored.returns, returns)
    path = tmp_path / "episode.pkl"
    episode.save(path)
    loaded = Episode.load(path)
    np.testing.assert_allclose(loaded.prices, prices)
    np.testing.assert_allclose(loaded.returns, returns)
    np.testing.assert_allclose(loaded.metadata["regimes"], metadata["regimes"])
