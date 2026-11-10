"""Label noise detection and confident learning audit for disaster datasets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class LabelNoiseAuditor:
    """Audits disaster dataset labels using Confident Learning (cleanlab).

    Identifies:
        - Out-of-fold predicted probabilities for every sample
        - Estimated label issue indices and noise rate
        - Per-class noise prevalence (e.g. informative vs uninformative ambiguity)
    """

    def __init__(self, n_folds: int = 5, seed: int = 42) -> None:
        self.n_folds = n_folds
        self.seed = seed

    def compute_out_of_fold_probs(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Compute out-of-fold predicted class probabilities using cross-validated logistic regression."""
        n_samples = len(labels)
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        oof_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for train_idx, val_idx in skf.split(features, labels):
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=self.seed)
            clf.fit(features[train_idx], labels[train_idx])
            probs = clf.predict_proba(features[val_idx])
            oof_probs[val_idx] = probs

        return oof_probs

    def audit_label_issues(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_ids: Sequence[str] | None = None,
        event_ids: Sequence[str] | None = None,
    ) -> dict[str, any]:
        """Run confident learning to identify label noise and issues."""
        oof_probs = self.compute_out_of_fold_probs(features, labels)

        # Use cleanlab to find label issues
        issue_indices = find_label_issues(
            labels=labels,
            pred_probs=oof_probs,
            return_indices_ranked_by="self_confidence",
        )

        quality_scores = get_label_quality_scores(
            labels=labels,
            pred_probs=oof_probs,
        )

        n_samples = len(labels)
        noise_rate = len(issue_indices) / max(n_samples, 1)

        flagged_samples = []
        for idx in issue_indices[:100]:
            item = {
                "sample_idx": int(idx),
                "given_label": int(labels[idx]),
                "predicted_label": int(np.argmax(oof_probs[idx])),
                "confidence_in_given": float(oof_probs[idx, labels[idx]]),
                "confidence_in_pred": float(np.max(oof_probs[idx])),
                "quality_score": float(quality_scores[idx]),
            }
            if sample_ids:
                item["sample_id"] = sample_ids[idx]
            if event_ids:
                item["event_id"] = event_ids[idx]
            flagged_samples.append(item)

        return {
            "total_samples": n_samples,
            "detected_issues_count": len(issue_indices),
            "estimated_noise_rate": float(noise_rate),
            "flagged_samples": flagged_samples,
            "mean_quality_score": float(np.mean(quality_scores)),
        }
