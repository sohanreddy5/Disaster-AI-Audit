"""Selective prediction (abstention) utilities.

Functions compute risk-coverage trade-offs: given a model's predictions and
associated confidence scores, measure how accuracy changes as low-confidence
predictions are withheld.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def risk_coverage_curve(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    confidences: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the risk-coverage curve.

    Samples are sorted by decreasing confidence.  At each coverage level
    (fraction of samples retained), the *risk* is the error rate among
    retained samples.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
        Ground-truth labels.
    y_pred : ndarray of shape ``(n_samples,)``
        Predicted labels.
    confidences : ndarray of shape ``(n_samples,)``
        Confidence scores (higher = more confident).

    Returns
    -------
    coverages : ndarray of shape ``(n_samples,)``
        Coverage values from ``1/n`` to ``1.0``.
    risks : ndarray of shape ``(n_samples,)``
        Risk (error rate) at each coverage level.
    """
    n = len(y_true)
    # Sort by confidence descending (most confident first)
    order = np.argsort(-np.asarray(confidences, dtype=np.float64))
    errors = (np.asarray(y_true)[order] != np.asarray(y_pred)[order]).astype(np.float64)

    # Cumulative error / cumulative count
    cum_errors = np.cumsum(errors)
    counts = np.arange(1, n + 1, dtype=np.float64)

    risks = cum_errors / counts
    coverages = counts / n

    return coverages, risks


def aurc(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    confidences: NDArray[np.floating],
) -> float:
    """Area Under the Risk-Coverage curve (AURC).

    Lower is better: a perfect selective classifier has AURC = 0.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
    y_pred : ndarray of shape ``(n_samples,)``
    confidences : ndarray of shape ``(n_samples,)``

    Returns
    -------
    float
        AURC value computed via the trapezoidal rule.
    """
    coverages, risks = risk_coverage_curve(y_true, y_pred, confidences)
    # Trapezoidal integration over [0, 1] coverage range (NumPy 2.x compatible)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is None:
        from scipy.integrate import trapezoid as trapz_fn
    return float(trapz_fn(risks, coverages))



def selective_accuracy_at_coverage(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    confidences: NDArray[np.floating],
    coverage: float = 0.8,
) -> float:
    """Accuracy when retaining only the top-*coverage* fraction of predictions.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
    y_pred : ndarray of shape ``(n_samples,)``
    confidences : ndarray of shape ``(n_samples,)``
    coverage : float
        Fraction of samples to retain, in ``(0, 1]`` (default ``0.8``).

    Returns
    -------
    float
        Classification accuracy on the retained subset.

    Raises
    ------
    ValueError
        If *coverage* is not in ``(0, 1]``.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    n = len(y_true)
    k = max(1, int(np.ceil(coverage * n)))

    order = np.argsort(-np.asarray(confidences, dtype=np.float64))
    selected = order[:k]

    correct = (np.asarray(y_true)[selected] == np.asarray(y_pred)[selected]).sum()
    return float(correct / k)
