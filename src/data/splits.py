"""Split generators for disaster AI benchmark.

Supports:
- Random split (stratified by label)
- Leave-One-Event-Out (LOEO)
- Disaster-type-held-out
- Region-held-out
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class SplitResult:
    """Holds train/test indices and metadata for one split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    split_name: str
    held_out_value: str | None = None

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_test(self) -> int:
        return len(self.test_indices)

    def __repr__(self) -> str:
        return (
            f"SplitResult(name='{self.split_name}', "
            f"held_out='{self.held_out_value}', "
            f"n_train={self.n_train}, n_test={self.n_test})"
        )


def random_split(
    n_samples: int,
    labels: np.ndarray,
    test_fraction: float = 0.2,
    n_splits: int = 1,
    seed: int = 42,
) -> list[SplitResult]:
    """Generate random stratified train/test splits.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    labels : np.ndarray
        Label array of shape ``(n_samples,)`` for stratification.
    test_fraction : float
        Fraction of data to hold out for testing.
    n_splits : int
        Number of independent random splits to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[SplitResult]
        One :class:`SplitResult` per split.
    """
    if n_samples != len(labels):
        raise ValueError(
            f"n_samples ({n_samples}) != len(labels) ({len(labels)})"
        )

    n_folds = max(2, int(round(1.0 / test_fraction)))
    results: list[SplitResult] = []

    for split_idx in range(n_splits):
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed + split_idx
        )
        # Take the first fold as our split
        train_idx, test_idx = next(skf.split(np.arange(n_samples), labels))
        results.append(
            SplitResult(
                train_indices=train_idx,
                test_indices=test_idx,
                split_name=f"random_split_{split_idx}",
                held_out_value=None,
            )
        )

    return results


def leave_one_event_out(
    event_ids: Sequence[str],
    labels: np.ndarray | None = None,
    min_test_samples: int = 5,
) -> list[SplitResult]:
    """Leave-One-Event-Out (LOEO) cross-validation.

    Each unique event is held out as the test set exactly once.

    Parameters
    ----------
    event_ids : Sequence[str]
        Per-sample event identifier, length ``N``.
    labels : np.ndarray | None
        Per-sample labels (unused, accepted for API consistency).
    min_test_samples : int
        Skip events with fewer than this many samples.

    Returns
    -------
    list[SplitResult]
        One :class:`SplitResult` per held-out event.
    """
    event_arr = np.array(event_ids)
    unique_events = sorted(set(event_ids))
    all_indices = np.arange(len(event_ids))
    results: list[SplitResult] = []

    for event in unique_events:
        test_mask = event_arr == event
        if test_mask.sum() < min_test_samples:
            continue
        train_idx = all_indices[~test_mask]
        test_idx = all_indices[test_mask]
        results.append(
            SplitResult(
                train_indices=train_idx,
                test_indices=test_idx,
                split_name=f"loeo_{event}",
                held_out_value=event,
            )
        )

    return results


def group_held_out(
    group_labels: Sequence[str],
    labels: np.ndarray | None = None,
    group_type: str = "group",
    min_test_samples: int = 5,
) -> list[SplitResult]:
    """Hold out one group at a time (generalizes LOEO to arbitrary groupings).

    Parameters
    ----------
    group_labels : Sequence[str]
        Per-sample group identifier (event, disaster_type, region, etc.).
    labels : np.ndarray | None
        Per-sample labels (unused, accepted for API consistency).
    group_type : str
        Human-readable name for the grouping variable (for split names).
    min_test_samples : int
        Skip groups with fewer than this many samples.

    Returns
    -------
    list[SplitResult]
        One :class:`SplitResult` per held-out group.
    """
    group_arr = np.array(group_labels)
    unique_groups = sorted(set(group_labels))
    all_indices = np.arange(len(group_labels))
    results: list[SplitResult] = []

    for group in unique_groups:
        test_mask = group_arr == group
        if test_mask.sum() < min_test_samples:
            continue
        train_idx = all_indices[~test_mask]
        test_idx = all_indices[test_mask]
        results.append(
            SplitResult(
                train_indices=train_idx,
                test_indices=test_idx,
                split_name=f"{group_type}_held_out_{group}",
                held_out_value=group,
            )
        )

    return results


def disaster_type_held_out(
    disaster_types: Sequence[str],
    labels: np.ndarray | None = None,
    min_test_samples: int = 5,
) -> list[SplitResult]:
    """Hold out one disaster type at a time.

    Parameters
    ----------
    disaster_types : Sequence[str]
        Per-sample disaster type string.
    labels : np.ndarray | None
        Unused; for API consistency.
    min_test_samples : int
        Minimum number of test samples per split.

    Returns
    -------
    list[SplitResult]
    """
    return group_held_out(
        disaster_types,
        labels=labels,
        group_type="disaster_type",
        min_test_samples=min_test_samples,
    )


def region_held_out(
    regions: Sequence[str],
    labels: np.ndarray | None = None,
    min_test_samples: int = 5,
) -> list[SplitResult]:
    """Hold out one geographic region at a time.

    Parameters
    ----------
    regions : Sequence[str]
        Per-sample region string.
    labels : np.ndarray | None
        Unused; for API consistency.
    min_test_samples : int
        Minimum number of test samples per split.

    Returns
    -------
    list[SplitResult]
    """
    return group_held_out(
        regions,
        labels=labels,
        group_type="region",
        min_test_samples=min_test_samples,
    )


# Aliases for consistent naming
leave_one_disaster_type_out = disaster_type_held_out
leave_one_region_out = region_held_out



def compound_shift_split(
    event_ids: Sequence[str],
    disaster_types: Sequence[str],
    regions: Sequence[str],
    labels: np.ndarray | None = None,
    min_test_samples: int = 5,
) -> list[SplitResult]:
    """Compound shift: hold out (disaster_type, region) pairs.

    This tests generalization under simultaneous shifts in both disaster
    type and geographic region.

    Parameters
    ----------
    event_ids : Sequence[str]
        Per-sample event id (unused directly, for reference).
    disaster_types : Sequence[str]
        Per-sample disaster type.
    regions : Sequence[str]
        Per-sample region.
    labels : np.ndarray | None
        Unused; for API consistency.
    min_test_samples : int
        Minimum test set size.

    Returns
    -------
    list[SplitResult]
    """
    compound = [f"{dt}__{reg}" for dt, reg in zip(disaster_types, regions)]
    return group_held_out(
        compound,
        labels=labels,
        group_type="compound",
        min_test_samples=min_test_samples,
    )


# -- Convenience: get split by name --------------------------------------

SPLIT_REGISTRY: dict[str, type] = {}

_SPLIT_FUNCTIONS = {
    "random": random_split,
    "loeo": leave_one_event_out,
    "event_held_out": leave_one_event_out,
    "disaster_type_held_out": disaster_type_held_out,
    "region_held_out": region_held_out,
    "compound_shift": compound_shift_split,
}


def get_split_fn(name: str):
    """Look up a split function by name.

    Parameters
    ----------
    name : str
        One of: ``'random'``, ``'loeo'``/``'event_held_out'``,
        ``'disaster_type_held_out'``, ``'region_held_out'``,
        ``'compound_shift'``.

    Returns
    -------
    Callable
        The split generator function.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _SPLIT_FUNCTIONS:
        raise KeyError(
            f"Unknown split '{name}'. Available: {list(_SPLIT_FUNCTIONS)}"
        )
    return _SPLIT_FUNCTIONS[name]
