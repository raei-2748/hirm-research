import numpy as np

from hirm.episodes.episode import Episode
from hirm.state.features import compute_phi_features


def _prices_from_returns(start_price: float, returns: np.ndarray) -> np.ndarray:
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices, dtype=float)


def _episode_from_returns(returns: np.ndarray) -> Episode:
    prices = _prices_from_returns(100.0, returns)
    metadata = {"regimes": np.zeros(len(prices))}
    return Episode(prices=prices, returns=returns, metadata=metadata)


def _rows(array: np.ndarray) -> list[list[float]]:
    data = array.tolist()
    if not data:
        return []
    if isinstance(data[0], list):
        return data
    return [data]


def test_features_do_not_look_ahead() -> None:
    base_returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01], dtype=float)
    episode_a = _episode_from_returns(base_returns)
    phi_a = compute_phi_features(episode_a)

    modified_returns = base_returns.tolist()
    modified_returns[-1] = -0.2
    modified_returns = np.asarray(modified_returns, dtype=float)
    episode_b = _episode_from_returns(modified_returns)
    phi_b = compute_phi_features(episode_b)

    rows_a = _rows(phi_a)
    rows_b = _rows(phi_b)
    assert rows_a[:-1] == rows_b[:-1]
