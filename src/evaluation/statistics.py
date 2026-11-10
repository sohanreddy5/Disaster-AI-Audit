"""Statistical testing utilities for event-held-out evaluation.

Provides cluster-aware bootstrap confidence intervals (where clusters are
disaster events), paired hypothesis tests with family-wise error-rate
control, and rank-stability diagnostics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


def cluster_bootstrap_ci(
    scores: NDArray[np.floating],
    event_ids: NDArray[np.str_],
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> tuple[float, float, float]:
    """Cluster bootstrap confidence interval for a metric.

    Resamples at the *event* level (not individual samples) to respect the
    clustered structure of disaster-event data, then returns the mean metric
    and a ``(1 - alpha)`` confidence interval.

    Parameters
    ----------
    scores : ndarray of shape ``(n_events,)``
        Per-event metric values (e.g. per-event accuracy).
    event_ids : ndarray of shape ``(n_events,)``
        Event identifiers corresponding to *scores*.  Unique events are used
        as resampling clusters.
    n_boot : int
        Number of bootstrap replicates (default ``2000``).
    alpha : float
        Significance level (default ``0.05`` → 95 % CI).
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    mean : float
        Point estimate (mean of *scores*).
    ci_lower : float
        Lower bound of the CI.
    ci_upper : float
        Upper bound of the CI.
    """
    rng = np.random.default_rng(random_state)
    unique_events = np.unique(event_ids)
    n_events = len(unique_events)

    # Build mapping: event -> score
    event_to_score = dict(zip(event_ids, scores))
    event_scores = np.array([event_to_score[e] for e in unique_events])

    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_events, size=n_events)
        boot_means[b] = event_scores[idx].mean()

    mean = float(event_scores.mean())
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, ci_lower, ci_upper


def paired_wilcoxon_with_holm(
    results_dict: dict[str, NDArray[np.floating]],
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

    Compares every pair of methods on their per-event scores, applies the
    Holm step-down correction for multiple comparisons, and returns a tidy
    DataFrame.

    Parameters
    ----------
    results_dict : dict[str, ndarray]
        Mapping from method name to an array of per-event scores.  All
        arrays must have the same length (same set of events).

    Returns
    -------
    pd.DataFrame
        Columns: ``method_a``, ``method_b``, ``statistic``, ``p_value``,
        ``p_adjusted`` (Holm), ``significant`` (at α = 0.05).

    Raises
    ------
    ValueError
        If arrays have inconsistent lengths.
    """
    methods = sorted(results_dict.keys())
    n_methods = len(methods)

    # Validate consistent lengths
    lengths = {m: len(v) for m, v in results_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent score array lengths: {lengths}")

    rows: list[dict[str, object]] = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            a, b = methods[i], methods[j]
            diff = results_dict[a] - results_dict[b]
            # Skip if all differences are zero (Wilcoxon would fail)
            if np.all(diff == 0):
                rows.append(
                    {
                        "method_a": a,
                        "method_b": b,
                        "statistic": 0.0,
                        "p_value": 1.0,
                    }
                )
                continue
            stat, p = stats.wilcoxon(results_dict[a], results_dict[b])
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "statistic": float(stat),
                    "p_value": float(p),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df["p_adjusted"] = pd.Series(dtype=float)
        df["significant"] = pd.Series(dtype=bool)
        return df

    # Holm-Bonferroni step-down correction
    df = df.sort_values("p_value").reset_index(drop=True)
    m = len(df)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for k in range(m):
        corrected = df.loc[k, "p_value"] * (m - k)
        running_max = max(running_max, corrected)
        adjusted[k] = min(running_max, 1.0)

    df["p_adjusted"] = adjusted
    df["significant"] = df["p_adjusted"] < 0.05
    return df.sort_values(["method_a", "method_b"]).reset_index(drop=True)


def kendall_tau_rank_change(
    random_ranks: NDArray[np.integer],
    loeo_ranks: NDArray[np.integer],
) -> tuple[float, float]:
    """Kendall τ correlation between two model rankings.

    Measures how much the relative ordering of methods changes when moving
    from random splits to leave-one-event-out (LOEO) evaluation.

    Parameters
    ----------
    random_ranks : ndarray of shape ``(n_methods,)``
        Ranks of methods under the random-split protocol (1 = best).
    loeo_ranks : ndarray of shape ``(n_methods,)``
        Ranks of the same methods under LOEO.

    Returns
    -------
    tau : float
        Kendall τ statistic in ``[-1, 1]``.
    p_value : float
        Two-sided p-value for testing ``τ = 0``.
    """
    result = stats.kendalltau(random_ranks, loeo_ranks)
    return float(result.statistic), float(result.pvalue)
