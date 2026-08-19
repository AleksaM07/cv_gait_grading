"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML configuration file when PyYAML is available."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML configs. Install project dependencies "
            "or pass defaults through the CLI."
        ) from exc
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    return data
