"""YAML/dict-based configuration management.

Provides a lightweight config system that loads YAML files and supports
hierarchical overrides without requiring Hydra at runtime.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to the ``.yaml`` configuration file.

    Returns
    -------
    dict[str, Any]
        The parsed configuration as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Handle 'defaults' key for config inheritance
    if "defaults" in config:
        base_configs = config.pop("defaults")
        merged = {}
        for base_name in base_configs:
            base_path = path.parent / f"{base_name}.yaml"
            if base_path.exists():
                base_cfg = load_config(base_path)
                merged = merge_configs(merged, base_cfg)
        config = merge_configs(merged, config)

    return config


def merge_configs(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge *override* into *base* (non-destructive).

    Parameters
    ----------
    base : dict
        Base configuration.
    override : dict
        Override configuration; its values take precedence.

    Returns
    -------
    dict
        Merged configuration (new dict, inputs unchanged).
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def config_to_flat(
    config: dict[str, Any],
    sep: str = ".",
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten a nested config dict into dot-separated keys.

    Parameters
    ----------
    config : dict
        Nested configuration.
    sep : str
        Separator between levels.
    prefix : str
        Key prefix (for recursion).

    Returns
    -------
    dict[str, Any]
        Flat dictionary.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(config_to_flat(value, sep=sep, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def override_from_dotlist(
    config: dict[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """Apply command-line style overrides to a config.

    Parameters
    ----------
    config : dict
        Base configuration (modified in-place).
    overrides : list[str]
        Overrides in ``key=value`` or ``key.subkey=value`` format.

    Returns
    -------
    dict
        The modified configuration.

    Example
    -------
    >>> cfg = {"model": {"lr": 0.01}}
    >>> override_from_dotlist(cfg, ["model.lr=0.001"])
    {'model': {'lr': 0.001}}
    """
    config = copy.deepcopy(config)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value_str = item.split("=", 1)

        # Try to parse value as Python literal
        value: Any
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError:
            value = value_str

        # Navigate to the right nesting level
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    return config
