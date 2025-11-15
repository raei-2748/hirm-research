"""Phase 1 logging utilities."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, TextIO

from .config import to_plain_dict
from .serialization import write_yaml


class Phase1Logger:
    """Unified logging interface for experiments and smoke tests."""

    def __init__(
        self,
        experiment_name: str,
        cfg: Mapping[str, Any] | Any,
        base_dir: str | Path = "runs",
    ) -> None:
        sanitized = experiment_name.replace(" ", "_") or "experiment"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"{timestamp}_{sanitized}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle: TextIO = (self.run_dir / "logs.txt").open(
            "a", encoding="utf-8"
        )
        self._metrics_handle: TextIO = (self.run_dir / "metrics.jsonl").open(
            "a", encoding="utf-8"
        )
        config_path = self.run_dir / "config_resolved.yaml"
        write_yaml(to_plain_dict(cfg), config_path)

    def _write_message(self, level: str, message: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        formatted = f"[{timestamp}] {level:<7} {message}\n"
        self._log_handle.write(formatted)
        self._log_handle.flush()

    def info(self, message: str) -> None:
        self._write_message("INFO", message)

    def warning(self, message: str) -> None:
        self._write_message("WARNING", message)

    def error(self, message: str) -> None:
        self._write_message("ERROR", message)

    def log(self, metrics: Mapping[str, Any]) -> None:
        entry = dict(metrics)
        entry.setdefault("timestamp", datetime.utcnow().isoformat())
        self._metrics_handle.write(json.dumps(entry, sort_keys=True) + "\n")
        self._metrics_handle.flush()

    def close(self) -> None:
        if not self._log_handle.closed:
            self._log_handle.close()
        if not self._metrics_handle.closed:
            self._metrics_handle.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()


def make_logger(
    experiment_name: str,
    cfg: Mapping[str, Any] | Any,
    base_dir: str | Path = "runs",
) -> Phase1Logger:
    """Factory to build the Phase 1 logger."""

    return Phase1Logger(experiment_name, cfg, base_dir=base_dir)


__all__ = ["Phase1Logger", "make_logger"]
