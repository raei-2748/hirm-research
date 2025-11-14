"""Safe serialization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # Optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch may not be installed
    torch = None


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
    if torch is None:
        raise RuntimeError("PyTorch is required for checkpoint serialization")
    file_path = Path(path)
    _ensure_parent(file_path)
    torch.save(state, file_path)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for checkpoint serialization")
    return torch.load(Path(path), map_location="cpu")


__all__ = [
    "save_dict",
    "load_dict",
    "save_array",
    "load_array",
    "save_checkpoint",
    "load_checkpoint",
]
