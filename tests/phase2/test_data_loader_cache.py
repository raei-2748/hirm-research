from pathlib import Path

from hirm.data.cache import maybe_cache
from hirm.data.loader import load_episode, load_episode_list, load_episodes_from_dir
from hirm.episodes.episode import Episode


def test_maybe_cache(tmp_path):
    calls = {"n": 0}

    def generator():
        calls["n"] += 1
        return {"value": 42}

    cache_path = tmp_path / "cache.pkl"
    maybe_cache(cache_path, generator)
    maybe_cache(cache_path, generator)
    assert calls["n"] == 1


def test_episode_loaders(tmp_path):
    prices = [1.0, 1.1, 1.2]
    returns = [0.1, 0.09]
    metadata = {"regimes": [0, 1, 1]}
    episode = Episode(prices, returns, metadata)
    path = tmp_path / "ep.pkl"
    episode.save(path)
    loaded = load_episode(path)
    assert loaded.length == episode.length
    paths = [path]
    episodes = load_episode_list(paths)
    assert len(episodes) == 1
    dir_path = tmp_path / "eps"
    dir_path.mkdir()
    episode.save(dir_path / "ep1.pkl")
    episode.save(dir_path / "ep2.pkl")
    ep_list = load_episodes_from_dir(dir_path)
    assert len(ep_list) == 2
