"""Power analysis for event-held-out evaluation.

Answers the question: *how many held-out events do we need for stable
conclusions?*  Subsamples events, recomputes bootstrap CIs and ranking
stability, and reports how they change with the number of events.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def subsample_events_power(
    all_event_scores: dict[str, NDArray[np.floating]],
    k_range: list[int],
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Subsample *k* events and measure CI width and ranking stability.

    For each value of *k* in *k_range* and for each bootstrap replicate:

    1. Draw *k* events (without replacement) from the full set.
    2. Compute a cluster-bootstrap CI for each method's mean score.
    3. Record the CI width and the method ranking at that subsample.

    Rank stability is measured as the mean Kendall τ between each
    subsampled ranking and the full-data ranking.

    Parameters
    ----------
    all_event_scores : dict[str, ndarray]
        Mapping from method name to an array of per-event scores (same
        length, one entry per event).
    k_range : list[int]
        Values of *k* (number of held-out events) to evaluate.
    n_boot : int
        Number of subsampling replicates per *k* (default ``500``).
    alpha : float
        CI significance level (default ``0.05``).
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``mean_ci_width``, ``rank_stability``.

        * ``mean_ci_width`` — average 95 % CI width across methods, averaged
          over bootstrap replicates.
        * ``rank_stability`` — mean Kendall τ between the subsampled ranking
          and the full-data ranking, averaged over replicates.
    """
    rng = np.random.default_rng(random_state)
    methods = sorted(all_event_scores.keys())
    n_methods = len(methods)

    # Full-data reference scores and ranking
    full_scores = np.array(
        [all_event_scores[m].mean() for m in methods], dtype=np.float64
    )
    full_ranks = _rank_array(full_scores)
    n_events_total = len(next(iter(all_event_scores.values())))

    rows: list[dict[str, object]] = []

    for k in k_range:
        if k < 2:
            raise ValueError(f"k must be >= 2 for meaningful CIs, got {k}")
        if k > n_events_total:
            raise ValueError(
                f"k={k} exceeds the total number of events ({n_events_total})."
            )

        ci_widths_accum: list[float] = []
        taus_accum: list[float] = []

        for _ in range(n_boot):
            idx = rng.choice(n_events_total, size=k, replace=False)

            # Mean score per method on the subsample
            sub_means = np.array(
                [all_event_scores[m][idx].mean() for m in methods],
                dtype=np.float64,
            )

            # Bootstrap CI width on the subsample (inner bootstrap)
            widths = _bootstrap_ci_widths(
                {m: all_event_scores[m][idx] for m in methods},
                methods=methods,
                n_inner=200,
                alpha=alpha,
                rng=rng,
            )
            ci_widths_accum.append(float(np.mean(widths)))

            # Rank stability
            sub_ranks = _rank_array(sub_means)
            tau, _ = stats.kendalltau(full_ranks, sub_ranks)
            taus_accum.append(float(tau) if not np.isnan(tau) else 0.0)

        rows.append(
            {
                "k": k,
                "mean_ci_width": float(np.mean(ci_widths_accum)),
                "rank_stability": float(np.mean(taus_accum)),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _rank_array(values: NDArray[np.floating]) -> NDArray[np.integer]:
    """Rank values in descending order (best = rank 1)."""
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _bootstrap_ci_widths(
    event_scores: dict[str, NDArray[np.floating]],
    methods: list[str],
    n_inner: int,
    alpha: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Return the CI width for each method from a cluster bootstrap."""
    k = len(next(iter(event_scores.values())))
    widths = np.empty(len(methods), dtype=np.float64)

    for mi, m in enumerate(methods):
        scores = event_scores[m]
        boot_means = np.empty(n_inner, dtype=np.float64)
        for b in range(n_inner):
            idx = rng.integers(0, k, size=k)
            boot_means[b] = scores[idx].mean()
        lo = np.percentile(boot_means, 100 * alpha / 2)
        hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
        widths[mi] = hi - lo

    return widths
