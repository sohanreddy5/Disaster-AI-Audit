"""Calibration diagnostics: ECE, NLL, and reliability-diagram data.

Functions operate on numpy arrays and are model-agnostic: pass predicted
probabilities alongside ground-truth labels.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray


def expected_calibration_error(
    y_true: NDArray[np.integer],
    y_prob: NDArray[np.floating],
    n_bins: int = 15,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Uses uniform-width confidence bins.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
        Ground-truth labels (integer class indices).
    y_prob : ndarray of shape ``(n_samples,)`` or ``(n_samples, n_classes)``
        If 2-D, the predicted probability for the *predicted* class is extracted
        automatically (i.e. ``max`` over the class axis).  If 1-D, treated as
        the predicted confidence directly.
    n_bins : int
        Number of equal-width bins in ``[0, 1]`` (default ``15``).

    Returns
    -------
    float
        Weighted-average absolute calibration gap across bins.
    """
    confidences, accuracies = _extract_conf_acc(y_true, y_prob)
    bin_accs, bin_confs, bin_counts = _bin_data(confidences, accuracies, n_bins)

    total = confidences.shape[0]
    ece = 0.0
    for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
        if count == 0:
            continue
        ece += (count / total) * abs(acc - conf)
    return float(ece)


def negative_log_likelihood(
    y_true: NDArray[np.integer],
    y_prob: NDArray[np.floating],
    eps: float = 1e-15,
) -> float:
    """Compute the mean negative log-likelihood.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
        Ground-truth class indices.
    y_prob : ndarray of shape ``(n_samples, n_classes)``
        Predicted class probabilities (rows should sum to 1).
    eps : float
        Clipping floor to prevent ``log(0)``.

    Returns
    -------
    float
        Mean NLL across all samples.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:
        raise ValueError(
            "negative_log_likelihood requires 2-D y_prob of shape (n_samples, n_classes)."
        )
    y_true = np.asarray(y_true, dtype=np.intp)
    n = len(y_true)
    # Select predicted probability for the true class
    probs_true = y_prob[np.arange(n), y_true]
    probs_true = np.clip(probs_true, eps, 1.0)
    return float(-np.log(probs_true).mean())


def reliability_diagram_data(
    y_true: NDArray[np.integer],
    y_prob: NDArray[np.floating],
    n_bins: int = 15,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Return binned data for plotting a reliability diagram.

    Parameters
    ----------
    y_true : ndarray of shape ``(n_samples,)``
    y_prob : ndarray of shape ``(n_samples,)`` or ``(n_samples, n_classes)``
    n_bins : int

    Returns
    -------
    bin_accs : ndarray of shape ``(n_bins,)``
        Mean accuracy within each confidence bin.
    bin_confs : ndarray of shape ``(n_bins,)``
        Mean confidence within each bin.
    bin_counts : ndarray of shape ``(n_bins,)``
        Number of samples falling in each bin.
    """
    confidences, accuracies = _extract_conf_acc(y_true, y_prob)
    return _bin_data(confidences, accuracies, n_bins)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _extract_conf_acc(
    y_true: NDArray[np.integer],
    y_prob: Union[NDArray[np.floating], NDArray[np.floating]],
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Extract per-sample confidence and binary accuracy arrays."""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true = np.asarray(y_true)

    if y_prob.ndim == 2:
        predictions = y_prob.argmax(axis=1)
        confidences = y_prob.max(axis=1)
    else:
        # 1-D: assume confidences for a binary problem or pre-extracted
        confidences = y_prob
        predictions = (y_prob >= 0.5).astype(np.intp)

    accuracies = (predictions == y_true).astype(np.intp)
    return confidences, accuracies


def _bin_data(
    confidences: NDArray[np.floating],
    accuracies: NDArray[np.integer],
    n_bins: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Bin confidences and accuracies into *n_bins* equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins, dtype=np.float64)
    bin_confs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            # Last bin is inclusive on the right
            mask = (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()

    return bin_accs, bin_confs, bin_counts
