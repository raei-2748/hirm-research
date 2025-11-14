"""Configuration loading utilities for HIRM Phase 1."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        stripped = value.strip("'\"")
        return stripped


def _parse_list(lines: List[str], start: int, indent: int) -> Tuple[List[Any], int]:
    items: List[Any] = []
    idx = start
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent or not stripped.startswith("- "):
            break
        value = stripped[2:].strip()
        idx += 1
        if value:
            items.append(_parse_scalar(value))
        else:
            nested, idx = _parse_block(lines, idx, indent + 2)
            items.append(nested)
    return items, idx


def _parse_block(lines: List[str], start: int, indent: int) -> Tuple[Dict[str, Any], int]:
    data: Dict[str, Any] = {}
    idx = start
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if ":" not in stripped:
            idx += 1
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        idx += 1
        if value:
            data[key] = _parse_scalar(value)
        else:
            if idx < len(lines):
                next_line = lines[idx]
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                next_stripped = next_line.strip()
                if next_stripped.startswith("- ") and next_indent >= indent + 2:
                    lst, idx = _parse_list(lines, idx, indent + 2)
                    data[key] = lst
                else:
                    nested, idx = _parse_block(lines, idx, indent + 2)
                    data[key] = nested
            else:
                data[key] = {}
    return data, idx


def _read_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    parsed, _ = _parse_block(lines, 0, 0)
    return parsed


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_subconfig(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in config:
        return config
    section = config[key]
    if not isinstance(section, dict):
        return config
    config_path = section.get("config_path")
    if config_path is None:
        return config
    repo_root = _resolve_repo_root()
    file_path = (repo_root / config_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Config path {file_path} does not exist")
    file_config = _read_yaml(file_path)
    override = {k: v for k, v in section.items() if k != "config_path"}
    config[key] = _merge_dicts(file_config, override)
    return config


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load the final experiment configuration.

    Parameters
    ----------
    path:
        Path to the experiment configuration file.
    """

    repo_root = _resolve_repo_root()
    experiment_path = (repo_root / path).resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config {experiment_path} not found")

    base_path = repo_root / "configs" / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError("Base configuration is missing")

    base_config = _read_yaml(base_path)
    experiment_config = _read_yaml(experiment_path)
    merged = _merge_dicts(base_config, experiment_config)

    for section in ("env", "model", "objective"):
        merged = _resolve_subconfig(merged, section)

    merged.setdefault("experiment", {})
    merged["experiment"].setdefault("name", experiment_path.stem)
    return merged


__all__ = ["load_config"]
