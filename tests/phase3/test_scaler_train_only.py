import math
import random
from typing import List

import numpy as np

from hirm.state.preprocess import FeatureScaler


def _random_matrix(rng: random.Random, rows: int, cols: int, loc: float, scale: float) -> np.ndarray:
    data = [
        [loc + scale * rng.gauss(0.0, 1.0) for _ in range(cols)]
        for _ in range(rows)
    ]
    return np.asarray(data, dtype="object")


def _flatten_rows(arrays: List[np.ndarray]) -> List[List[float]]:
    rows: list[list[float]] = []
    for arr in arrays:
        data = arr.tolist()
        if not data:
            continue
        if isinstance(data[0], list):
            rows.extend(data)
        else:
            rows.append([float(value) for value in data])
    return rows


def test_feature_scaler_uses_train_only_statistics() -> None:
    rng = random.Random(0)
    train_phi = [_random_matrix(rng, 16, 4, loc=5.0, scale=2.0) for _ in range(3)]
    val_phi = [_random_matrix(rng, 12, 4, loc=-3.0, scale=1.0) for _ in range(2)]

    scaler = FeatureScaler()
    scaler.fit(train_phi)

    transformed_train = [scaler.transform(phi) for phi in train_phi]
    train_rows = _flatten_rows(transformed_train)
    num_rows = len(train_rows)
    num_cols = len(train_rows[0])
    train_means = [sum(row[col] for row in train_rows) / num_rows for col in range(num_cols)]
    train_stds = [
        math.sqrt(
            sum((row[col] - train_means[col]) ** 2 for row in train_rows) / num_rows
        )
        for col in range(num_cols)
    ]
    assert all(abs(mean) < 1e-6 for mean in train_means)
    assert all(abs(std - 1.0) < 1e-6 for std in train_stds)

    transformed_val = [scaler.transform(phi) for phi in val_phi]
    val_rows = _flatten_rows(transformed_val)
    val_means = [sum(row[col] for row in val_rows) / len(val_rows) for col in range(num_cols)]
    assert all(abs(mean) > 0.5 for mean in val_means)
