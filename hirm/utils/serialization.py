"""Safe serialization helpers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dict(data: Dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    _ensure_parent(file_path)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def load_dict(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_array(array: Iterable[float], path: str | Path) -> None:
    file_path = Path(path)
    _ensure_parent(file_path)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(list(array), handle)


def load_array(path: str | Path) -> List[float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(float(v) for v in data)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    _ensure_parent(file_path)
    with file_path.open("wb") as handle:
        pickle.dump(state, handle)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


__all__ = [
    "save_dict",
    "load_dict",
    "save_array",
    "load_array",
    "save_checkpoint",
    "load_checkpoint",
]
