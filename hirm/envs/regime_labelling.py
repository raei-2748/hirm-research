"""Minimal realized-volatility regime labelling."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence


def label_regimes(
    returns: Sequence[float],
    window: int = 20,
    save: bool = True,
    save_path: Optional[str | Path] = None,
) -> List[int]:
    values = list(returns)
    if len(values) < window:
        raise ValueError("returns length must exceed rolling window")
    rolling_std: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_slice = values[start : idx + 1]
        mean = sum(window_slice) / len(window_slice)
        variance = sum((val - mean) ** 2 for val in window_slice) / len(window_slice)
        rolling_std.append(variance ** 0.5)
    tail = rolling_std[window - 1 :]
    sorted_tail = sorted(tail)
    low_idx = int(len(sorted_tail) / 3)
    high_idx = int(2 * len(sorted_tail) / 3)
    low_threshold = sorted_tail[low_idx]
    high_threshold = sorted_tail[high_idx]
    labels: List[int] = []
    for value in rolling_std:
        if value >= high_threshold:
            labels.append(2)
        elif value >= low_threshold:
            labels.append(1)
        else:
            labels.append(0)
    if save:
        base_dir = Path(save_path) if save_path else Path("data/processed/regimes")
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / "latest_regimes.json"
        output_path.write_text("\n".join(str(label) for label in labels), encoding="utf-8")
    return labels


__all__ = ["label_regimes"]
