import numpy as np

from hirm.episodes.episode import Episode
from hirm.state.preprocess import preprocess_episodes


def _build_episode(seed: int, length: int = 6) -> Episode:
    rng = np.random.default_rng(seed)
    returns = np.asarray(rng.normal(scale=0.01, size=length), dtype=float)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    metadata = {"regimes": np.zeros(len(prices))}
    return Episode(prices=np.array(prices, dtype=float), returns=returns, metadata=metadata)


def _extract_arrays(output):
    extracted = {}
    for split_name, data in output.items():
        extracted[split_name] = {
            "phi": [arr.tolist() if hasattr(arr, "tolist") else list(arr) for arr in data["phi"]],
            "r": [arr.tolist() if hasattr(arr, "tolist") else list(arr) for arr in data["r"]],
        }
    return extracted


def _assert_nested_allclose(a, b, atol: float = 1e-8) -> None:
    assert len(a) == len(b)
    for row_a, row_b in zip(a, b):
        assert len(row_a) == len(row_b)
        for val_a, val_b in zip(row_a, row_b):
            assert abs(val_a - val_b) <= atol


def test_preprocess_is_deterministic(tmp_path) -> None:
    episodes = [_build_episode(seed) for seed in range(4)]
    env_ids = [0, 1, 0, 1]
    config = {"output_dir": tmp_path / "features"}

    first = preprocess_episodes(episodes, env_ids, config=config, split_seed=7)
    second = preprocess_episodes(episodes, env_ids, config=config, split_seed=7)

    extracted_first = _extract_arrays(first)
    extracted_second = _extract_arrays(second)
    for split in extracted_first:
        assert len(extracted_first[split]["phi"]) == len(extracted_second[split]["phi"])
        for arr_a, arr_b in zip(extracted_first[split]["phi"], extracted_second[split]["phi"]):
            _assert_nested_allclose(arr_a, arr_b)
        for arr_a, arr_b in zip(extracted_first[split]["r"], extracted_second[split]["r"]):
            _assert_nested_allclose(arr_a, arr_b)
