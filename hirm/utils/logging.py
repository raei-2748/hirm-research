"""Minimal experiment logger."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TextIO


class ExperimentLogger:
    """Simple file-backed logger with timestamps."""

    def __init__(self, experiment_name: str, output_dir: str | Path = "outputs") -> None:
        self.experiment_name = experiment_name
        self.base_dir = Path(output_dir) / experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.base_dir / "log.txt"
        self._handle: TextIO = self.log_path.open("a", encoding="utf-8")

    def _write(self, level: str, message: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        formatted = f"[{timestamp}] {level.upper():<7} {message}\n"
        self._handle.write(formatted)
        self._handle.flush()

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()


__all__ = ["ExperimentLogger"]
