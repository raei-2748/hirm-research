"""Simple caching helper."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from hirm.utils.serialization import load_checkpoint, save_checkpoint


def maybe_cache(path: str, generator_fn: Callable[[], Any]) -> Any:
    """Cache the output of ``generator_fn`` to ``path``."""

    file_path = Path(path)
    if file_path.exists():
        payload = load_checkpoint(file_path)
        return payload.get("value", payload)
    result = generator_fn()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint({"value": result}, file_path)
    return result


__all__ = ["maybe_cache"]
