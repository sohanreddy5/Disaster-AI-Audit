"""Evaluation package for disaster AI audit benchmark.

Provides metrics, calibration analysis, selective prediction (abstention),
statistical testing, and power-analysis utilities for event-held-out
evaluation of multimodal disaster AI systems.
"""

from src.evaluation.metrics import accuracy, macro_f1, balanced_accuracy, per_event_metrics
from src.evaluation.calibration import (
    expected_calibration_error,
    negative_log_likelihood,
    reliability_diagram_data,
)
from src.evaluation.abstention import (
    risk_coverage_curve,
    aurc,
    selective_accuracy_at_coverage,
)
from src.evaluation.statistics import (
    cluster_bootstrap_ci,
    paired_wilcoxon_with_holm,
    kendall_tau_rank_change,
)
from src.evaluation.power_analysis import subsample_events_power

__all__ = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "per_event_metrics",
    "expected_calibration_error",
    "negative_log_likelihood",
    "reliability_diagram_data",
    "risk_coverage_curve",
    "aurc",
    "selective_accuracy_at_coverage",
    "cluster_bootstrap_ci",
    "paired_wilcoxon_with_holm",
    "kendall_tau_rank_change",
    "subsample_events_power",
]
