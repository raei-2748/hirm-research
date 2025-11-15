"""Configuration loading utilities for the Phase 1 foundation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class ConfigNode(dict):
    """A dict wrapper that exposes dot-notation access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise AttributeError(key) from exc
        return _wrap_value(value)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            return super().__setattr__(key, value)
        self[key] = value

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        value = super().__getitem__(key)
        return _wrap_value(value)

    def to_dict(self) -> Dict[str, Any]:
        return _unwrap_value(self)


def _wrap_value(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, ConfigNode):
        return ConfigNode(value)
    if isinstance(value, list):
        return [
            _wrap_value(item) if isinstance(item, (dict, list)) else item
            for item in value
        ]
    return value


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return {k: _unwrap_value(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _unwrap_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap_value(v) for v in value]
    return value


def to_plain_dict(config: Any) -> Any:
    """Convert a possibly wrapped config into built-in containers."""

    return _unwrap_value(config)


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
            if ":" in value:
                key, remainder = value.split(":", 1)
                key = key.strip()
                remainder = remainder.strip()
                if remainder:
                    items.append({key: _parse_scalar(remainder)})
                else:
                    nested, idx = _parse_block(lines, idx, indent + 2)
                    items.append({key: nested})
            else:
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


def _candidate_paths(repo_root: Path, entry: str) -> Iterable[Path]:
    relative = Path(entry)
    if not relative.suffix:
        relative = relative.with_suffix(".yaml")
    yield repo_root / relative
    yield repo_root / "configs" / relative


def _candidate_group_paths(repo_root: Path, group: str, name: str) -> Iterable[Path]:
    clean_name = name if Path(name).suffix else f"{name}.yaml"
    group_dir = Path(group)
    plural_dir = Path(f"{group}s")
    for candidate in (
        repo_root / "configs" / group_dir / clean_name,
        repo_root / "configs" / plural_dir / clean_name,
        repo_root / "configs" / group_dir / name,
        repo_root / "configs" / plural_dir / name,
        repo_root / name,
    ):
        yield candidate


def _load_defaults(repo_root: Path, defaults: Iterable[Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for entry in defaults:
        block: Dict[str, Any]
        if isinstance(entry, str):
            if entry in {"_self_", "..."}:
                continue
            block = _load_default_block(repo_root, entry)
        elif isinstance(entry, dict) and len(entry) == 1:
            (group, name), = entry.items()
            if isinstance(name, str):
                block = _load_group_block(repo_root, group, name)
            else:
                raise TypeError("Default entries must map to string values")
        else:
            raise TypeError("Defaults must be strings or single-entry dicts")
        merged = _merge_dicts(merged, block)
    return merged


def _load_default_block(repo_root: Path, entry: str) -> Dict[str, Any]:
    for candidate in _candidate_paths(repo_root, entry):
        if candidate.exists():
            return _read_yaml(candidate)
    raise FileNotFoundError(f"Unable to resolve default entry '{entry}'")


def _load_group_block(repo_root: Path, group: str, name: str) -> Dict[str, Any]:
    for candidate in _candidate_group_paths(repo_root, group, name):
        if candidate.exists():
            return {group: _read_yaml(candidate)}
    raise FileNotFoundError(f"Unable to resolve default '{group}: {name}'")


def load_config(path: str | Path) -> ConfigNode:
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
    defaults = experiment_config.pop("defaults", [])

    merged = _merge_dicts(base_config, _load_defaults(repo_root, defaults))
    merged = _merge_dicts(merged, experiment_config)

    for section in ("env", "model", "objective"):
        merged = _resolve_subconfig(merged, section)

    merged.setdefault("experiment", {})
    merged["experiment"].setdefault("name", experiment_path.stem)
    return ConfigNode(merged)


__all__ = ["ConfigNode", "load_config", "to_plain_dict"]
