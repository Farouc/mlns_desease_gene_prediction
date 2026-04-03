"""Configuration utilities for experiment management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed config dictionary.

    Raises:
        ConfigError: If the file does not exist or cannot be parsed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML at {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML object must be a mapping in {config_path}")
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary.

    Args:
        base: Base dictionary.
        updates: Update dictionary.

    Returns:
        Updated dictionary.
    """
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Parse key=value CLI overrides with dot notation.

    Args:
        overrides: List of strings like `model.alpha=0.7`.

    Returns:
        Nested dictionary suitable for deep merge.

    Raises:
        ConfigError: If override format is invalid.
    """
    parsed: dict[str, Any] = {}
    if not overrides:
        return parsed

    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}'. Expected key=value format.")
        key, raw_value = item.split("=", 1)
        key_parts = key.strip().split(".")
        if not all(key_parts):
            raise ConfigError(f"Invalid override key '{key}'.")

        value: Any
        if raw_value.lower() in {"true", "false"}:
            value = raw_value.lower() == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value

        cursor = parsed
        for part in key_parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            if not isinstance(cursor[part], dict):
                raise ConfigError(f"Override key conflict at '{part}' in '{item}'.")
            cursor = cursor[part]
        cursor[key_parts[-1]] = value

    return parsed


def resolve_path(base_dir: str | Path, maybe_relative: str | Path) -> Path:
    """Resolve a potentially relative path against a base directory."""
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return Path(base_dir) / path
