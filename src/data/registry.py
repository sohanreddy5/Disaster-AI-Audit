"""Dataset registry: centralised access to all benchmark datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Type

from src.data.base import DisasterDataset


_REGISTRY: dict[str, Type[DisasterDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class under *name*.

    Usage::

        @register_dataset("crisismmd")
        class CrisisMMDDataset(DisasterDataset):
            ...
    """

    def decorator(cls: Type[DisasterDataset]) -> Type[DisasterDataset]:
        _REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_dataset(
    name: str,
    root: str | Path | None = None,
    **kwargs,
) -> DisasterDataset:
    """Instantiate a dataset by name.

    Parameters
    ----------
    name : str
        Registered dataset name (case-insensitive).
    root : str | Path | None
        Root directory. If ``None``, the dataset class must handle defaults.
    **kwargs
        Additional keyword arguments passed to the dataset constructor.

    Returns
    -------
    DisasterDataset

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered: {list(_REGISTRY)}"
        )
    cls = _REGISTRY[key]
    if root is not None:
        return cls(root=root, **kwargs)
    # Try without root – some datasets (e.g. HuggingFace-based) don't need one
    return cls(**kwargs)


def list_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    return sorted(_REGISTRY.keys())


def _auto_register() -> None:
    """Import all dataset modules so their @register_dataset decorators run."""
    # These imports trigger the registration side-effects
    import src.data.crisismmd  # noqa: F401
    import src.data.humaid  # noqa: F401
    import src.data.medic  # noqa: F401
    import src.data.xbd  # noqa: F401
    import src.data.bright  # noqa: F401


_auto_register()
