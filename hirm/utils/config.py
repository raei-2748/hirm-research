"""Configuration loading utilities for the Phase 1 foundation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


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


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Parse a YAML file into native Python objects.

    The previous implementation used a bespoke, line-oriented parser that
    treated inline sequences and mappings as raw strings (for example,
    ``[1, 2, 3]`` became the literal string "[1, 2, 3]"). This caused the merged
    configuration to surface stringified lists and dictionaries, which broke
    downstream consumers expecting real Python containers. Delegating parsing to
    ``yaml.safe_load`` restores correct typing for lists, dictionaries, and
    numeric values while still maintaining safe loading semantics.
    """

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parsed or {}


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
