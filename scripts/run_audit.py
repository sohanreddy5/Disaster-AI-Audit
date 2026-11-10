#!/usr/bin/env python3
"""Run data contamination, near-duplicate, and label noise audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.duplicates import DuplicateAuditor
from src.audit.label_noise import LabelNoiseAuditor
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit disaster datasets for leakage and label noise.")
    parser.add_argument("--features-npz", type=str, default=None, help="Path to precomputed features .npz file")
    parser.add_argument("--output-dir", type=str, default="./outputs/audit", help="Directory to save audit findings")
    parser.add_argument("--dummy", action="store_true", help="Run audit on synthetic dataset")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of dummy samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Disaster AI Dataset Audit...")

    if args.features_npz:
        feat_path = Path(args.features_npz)
        if not feat_path.is_absolute():
            feat_path = PROJECT_ROOT / feat_path
        logger.info(f"Loading real features from {feat_path}")
        data = np.load(feat_path)
        features = data["features"]
        labels = data["labels"]
        event_ids = data["event_ids"].tolist()
        sample_ids = [f"sample_{i}" for i in range(len(labels))]
        split_assignments = ["train" if i % 2 == 0 else "test" for i in range(len(labels))]
        logger.info(f"Loaded {len(labels)} real samples across {len(set(event_ids))} events")
    elif args.dummy:
        rng = np.random.RandomState(42)
        n = args.n_samples
        dim = 512
        features = rng.randn(n, dim).astype(np.float32)
        # Inject exact/near duplicates to test audit sensitivity
        features[10:15] = features[0:5] + rng.randn(5, dim) * 0.001
        labels = rng.choice([0, 1], size=n)
        # Inject label noise
        labels[0:5] = 1 - labels[0:5]
        sample_ids = [f"sample_{i:04d}" for i in range(n)]
        event_ids = [f"event_{i % 5}" for i in range(n)]
        split_assignments = ["train" if i % 2 == 0 else "test" for i in range(n)]
    else:
        logger.error("Please supply precomputed features via --features-npz or pass --dummy.")
        return


    # 1. Near-duplicate and leakage audit
    logger.info("Auditing feature-space near duplicates and cross-split leakage...")
    dup_auditor = DuplicateAuditor(cosine_threshold=0.90)
    dup_results = dup_auditor.audit_embedding_duplicates(
        embeddings=features,
        sample_ids=sample_ids,
        event_ids=event_ids,
        split_assignments=split_assignments,
    )
    logger.info(f"Detected {dup_results['high_similarity_pairs_count']} high-similarity pairs")
    logger.info(f"Detected {dup_results['cross_split_contamination_count']} cross-split contamination cases")

    # 2. Confident learning label noise audit
    logger.info("Auditing label noise with Confident Learning (cleanlab)...")
    noise_auditor = LabelNoiseAuditor(n_folds=5, seed=42)
    noise_results = noise_auditor.audit_label_issues(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        event_ids=event_ids,
    )
    logger.info(f"Estimated noise rate: {noise_results['estimated_noise_rate'] * 100:.2f}%")
    logger.info(f"Detected {noise_results['detected_issues_count']} label issues")

    # Save summary
    audit_summary = {
        "duplicates": dup_results,
        "label_noise": noise_results,
    }
    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Audit report saved to {out_dir / 'audit_summary.json'}")


if __name__ == "__main__":
    main()
