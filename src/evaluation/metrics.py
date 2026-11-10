"""Standard classification metrics with per-event disaggregation.

All public functions accept plain numpy arrays and return either scalar values
or a :class:`pandas.DataFrame` for tabular per-event results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


def accuracy(y_true: NDArray[np.integer], y_pred: NDArray[np.integer]) -> float:
    """Compute overall classification accuracy.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
        Ground-truth labels.
    y_pred : ndarray of shape ``(n_samples,)``
        Predicted labels.

    Returns
    -------
    float
        Fraction of correctly classified samples.
    """
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: NDArray[np.integer], y_pred: NDArray[np.integer]) -> float:
    """Compute macro-averaged F1 score.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
    y_pred : ndarray of shape ``(n_samples,)``

    Returns
    -------
    float
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def balanced_accuracy(
    y_true: NDArray[np.integer], y_pred: NDArray[np.integer]
) -> float:
    """Compute balanced accuracy (macro-averaged recall).

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
    y_pred : ndarray of shape ``(n_samples,)``

    Returns
    -------
    float
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def per_event_metrics(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    event_ids: NDArray[np.str_],
) -> pd.DataFrame:
    """Compute accuracy, macro-F1, and balanced accuracy per event.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
        Ground-truth labels.
    y_pred : ndarray of shape ``(n_samples,)``
        Predicted labels.
    event_ids : ndarray of shape ``(n_samples,)``
        Event identifiers aligning with *y_true* / *y_pred*.

    Returns
    -------
    pd.DataFrame
        One row per unique event, with columns:

        * ``event_id`` — the event identifier
        * ``n_samples`` — number of samples in the event
        * ``accuracy``
        * ``macro_f1``
        * ``balanced_accuracy``
    """
    unique_events = np.unique(event_ids)
    rows: list[dict[str, object]] = []

    for eid in unique_events:
        mask = event_ids == eid
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append(
            {
                "event_id": eid,
                "n_samples": int(mask.sum()),
                "accuracy": accuracy(yt, yp),
                "macro_f1": macro_f1(yt, yp),
                "balanced_accuracy": balanced_accuracy(yt, yp),
            }
        )

    return pd.DataFrame(rows)
