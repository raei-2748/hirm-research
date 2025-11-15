"""Serialization helpers for Phase 1 infrastructure."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _torch_available() -> bool:
    return torch is not None


def save_model(model: Any, path: str | Path) -> None:
    """Persist a model or its state dict."""

    file_path = Path(path)
    _ensure_parent(file_path)
    state = model.state_dict() if hasattr(model, "state_dict") else model
    if _torch_available():  # pragma: no cover - executed when torch present
        torch.save(state, file_path)
    else:
        with file_path.open("wb") as handle:
            pickle.dump(state, handle)


def load_model(model_class: type, path: str | Path, *args: Any, **kwargs: Any) -> Any:
    """Load model parameters into a new instance of ``model_class``."""

    file_path = Path(path)
    if _torch_available():  # pragma: no cover - executed when torch present
        state = torch.load(file_path, map_location="cpu")  # type: ignore[arg-type]
    else:
        with file_path.open("rb") as handle:
            state = pickle.load(handle)
    model = model_class(*args, **kwargs)
    if hasattr(model, "load_state_dict") and isinstance(state, Mapping):
        model.load_state_dict(state)  # type: ignore[call-arg]
        return model
    if isinstance(state, model_class):  # pragma: no cover - defensive
        return state
    return model


def save_optimizer(optimizer: Any, path: str | Path) -> None:
    """Persist optimizer state to disk."""

    file_path = Path(path)
    _ensure_parent(file_path)
    state = optimizer.state_dict() if hasattr(optimizer, "state_dict") else optimizer
    if _torch_available():  # pragma: no cover - executed when torch present
        torch.save(state, file_path)
    else:
        with file_path.open("wb") as handle:
            pickle.dump(state, handle)


def load_optimizer(
    optimizer_class: type,
    path: str | Path,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Load optimizer state into a new instance."""

    file_path = Path(path)
    if _torch_available():  # pragma: no cover - executed when torch present
        state = torch.load(file_path, map_location="cpu")  # type: ignore[arg-type]
    else:
        with file_path.open("rb") as handle:
            state = pickle.load(handle)
    optimizer = optimizer_class(*args, **kwargs)
    if hasattr(optimizer, "load_state_dict") and isinstance(state, Mapping):
        optimizer.load_state_dict(state)  # type: ignore[call-arg]
    return optimizer


def save_checkpoint(state: Mapping[str, Any], path: str | Path) -> None:
    """Persist a training checkpoint dictionary."""

    file_path = Path(path)
    _ensure_parent(file_path)
    payload = dict(state)
    if _torch_available():  # pragma: no cover - executed when torch present
        torch.save(payload, file_path)
    else:
        with file_path.open("wb") as handle:
            pickle.dump(payload, handle)


def load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Load a checkpoint dictionary."""

    file_path = Path(path)
    if _torch_available():  # pragma: no cover - executed when torch present
        return torch.load(file_path, map_location="cpu")  # type: ignore[arg-type]
    with file_path.open("rb") as handle:
        return pickle.load(handle)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(ch in text for ch in ":#") or text.strip() != text or " " in text:
        escaped = text.replace("\"", "\\\"")
        return f'"{escaped}"'
    return text


def _yaml_lines(data: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(data, Mapping):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(value)}")
        return lines
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return lines
    return [f"{prefix}{_format_scalar(data)}"]


def write_yaml(data: Any, path: str | Path) -> None:
    """Write data to disk using a simple YAML emitter."""

    file_path = Path(path)
    _ensure_parent(file_path)
    lines = _yaml_lines(data)
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "load_checkpoint",
    "load_model",
    "load_optimizer",
    "save_checkpoint",
    "save_model",
    "save_optimizer",
    "write_yaml",
]
