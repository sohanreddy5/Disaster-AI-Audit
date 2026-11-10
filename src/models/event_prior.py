"""Event-prior baseline that predicts using only event-level class frequencies.

This serves as a *no-feature* baseline: given only the ``event_id`` of a test
sample, the model predicts the most frequent class observed for that event
during training.  For unseen events it falls back to the global prior.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class EventPriorBaseline:
    """Predict labels from event-level class frequencies (no features).

    Parameters
    ----------
    smoothing : float
        Laplace smoothing added to every class count (default ``1.0``).
    random_state : int | None
        Seed used when breaking ties randomly.

    Examples
    --------
    >>> import numpy as np
    >>> model = EventPriorBaseline()
    >>> event_ids = np.array(["eq1", "eq1", "eq1", "fl1", "fl1"])
    >>> labels = np.array([0, 0, 1, 1, 1])
    >>> model.fit(event_ids, labels)  # doctest: +ELLIPSIS
    <...EventPriorBaseline object at 0x...>
    >>> model.predict(np.array(["eq1", "fl1"]))
    array([0, 1])
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.smoothing = smoothing
        self.random_state = random_state

        # Populated after fit()
        self._classes: Optional[NDArray[np.integer]] = None
        self._event_priors: dict[str, NDArray[np.floating]] = {}
        self._global_prior: Optional[NDArray[np.floating]] = None
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        event_ids: NDArray[np.str_],
        labels: NDArray[np.integer],
    ) -> "EventPriorBaseline":
        """Compute per-event class distributions.

        Parameters
        ----------
        event_ids : ndarray of shape ``(n_samples,)``
            Event identifier for each training sample.
        labels : ndarray of shape ``(n_samples,)``
            Integer class labels.

        Returns
        -------
        self
        """
        self._classes = np.unique(labels)
        n_classes = len(self._classes)
        class_to_idx = {int(c): i for i, c in enumerate(self._classes)}

        # Global counts
        global_counts = np.full(n_classes, self.smoothing, dtype=np.float64)
        for lab in labels:
            global_counts[class_to_idx[int(lab)]] += 1.0
        self._global_prior = global_counts / global_counts.sum()

        # Per-event counts
        event_counts: dict[str, NDArray[np.float64]] = {}
        for eid, lab in zip(event_ids, labels):
            eid_str = str(eid)
            if eid_str not in event_counts:
                event_counts[eid_str] = np.full(n_classes, self.smoothing, dtype=np.float64)
            event_counts[eid_str][class_to_idx[int(lab)]] += 1.0

        self._event_priors = {
            eid: counts / counts.sum() for eid, counts in event_counts.items()
        }

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _get_prior(self, event_id: str) -> NDArray[np.floating]:
        """Return the prior distribution for *event_id*, or the global prior."""
        if event_id in self._event_priors:
            return self._event_priors[event_id]
        assert self._global_prior is not None
        return self._global_prior

    def predict(self, event_ids: NDArray[np.str_]) -> NDArray[np.integer]:
        """Predict the most frequent class per event.

        Parameters
        ----------
        event_ids : ndarray of shape ``(n_samples,)``
            Event identifier for each test sample.

        Returns
        -------
        ndarray of shape ``(n_samples,)``
            Predicted class labels.
        """
        if self._classes is None:
            raise RuntimeError("EventPriorBaseline has not been fitted yet.")

        predictions = np.empty(len(event_ids), dtype=self._classes.dtype)
        for i, eid in enumerate(event_ids):
            prior = self._get_prior(str(eid))
            predictions[i] = self._classes[int(np.argmax(prior))]
        return predictions

    def predict_proba(self, event_ids: NDArray[np.str_]) -> NDArray[np.floating]:
        """Return the event-level class distribution for each sample.

        Parameters
        ----------
        event_ids : ndarray of shape ``(n_samples,)``
            Event identifier for each test sample.

        Returns
        -------
        ndarray of shape ``(n_samples, n_classes)``
            Smoothed class probabilities.
        """
        if self._classes is None:
            raise RuntimeError("EventPriorBaseline has not been fitted yet.")

        n_classes = len(self._classes)
        probs = np.empty((len(event_ids), n_classes), dtype=np.float64)
        for i, eid in enumerate(event_ids):
            probs[i] = self._get_prior(str(eid))
        return probs


# Convenient alias
EventPrior = EventPriorBaseline

