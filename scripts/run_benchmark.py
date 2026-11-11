#!/usr/bin/env python3
"""Full Disaster AI Generalization Benchmark Runner.

Evaluates models under:
- Random i.i.d. splits (5 seeds)
- Leave-One-Event-Out (LOEO)
- Leave-One-Disaster-Type-Out (LODTO)
- Leave-One-Region-Out (LORO)

Computes:
- Accuracy, Balanced Accuracy, Macro-F1
- Worst-Event Accuracy and Tail-Risk Bounds
- Expected Calibration Error (ECE) and NLL
- Selective Prediction & Risk-Coverage (AURC)
- Event-Level Cluster Bootstrap 95% Confidence Intervals
- Kendall's tau rank correlation across split types
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.crisismmd import CrisisMMDDataset
from src.data.humaid import HumAIDDataset
from src.data.splits import (
    random_split,
    leave_one_event_out,
    leave_one_disaster_type_out,
    leave_one_region_out,
    SplitResult,
)
from src.evaluation.calibration import expected_calibration_error, negative_log_likelihood
from src.evaluation.abstention import aurc, risk_coverage_curve
from src.evaluation.statistics import cluster_bootstrap_ci
from src.features.text_extractor import TextFeatureExtractor
from src.models.probes import LogisticProbe
from src.models.event_prior import EventPrior
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Full Disaster AI Benchmark.")
    parser.add_argument("--dataset", "--dataset-name", dest="dataset", type=str, default="crisismmd", choices=["crisismmd", "humaid"], help="Dataset to benchmark")
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory")
    parser.add_argument("--cache-dir", type=str, default="./cache/features", help="Embedding cache directory")
    parser.add_argument("--output-dir", type=str, default="./outputs/benchmark", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample data for ultra-fast run")
    parser.add_argument("--n-random-splits", type=int, default=5, help="Number of random splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_dataset_and_features(
    dataset_name: str,
    data_root: Path,
    cache_dir: Path,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """Load dataset, encode labels, and extract/cache Sentence-Transformer embeddings."""
    if dataset_name == "crisismmd":
        ds_path = data_root / "crisismmd"
        logger.info(f"Loading CrisisMMD humanitarian task from {ds_path}")
        ds = CrisisMMDDataset(root=ds_path, label_task="humanitarian")
    elif dataset_name == "humaid":
        ds_path = data_root / "humaid"
        logger.info(f"Loading HumAID from {ds_path}")
        ds = HumAIDDataset(root=ds_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = ds.samples
    if max_samples and max_samples < len(samples):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in indices]
        logger.info(f"Subsampled to {len(samples)} samples for benchmark")

    cache_file = cache_dir / f"{dataset_name}_benchmark_features.npz"
    alt_cache = cache_dir / f"{dataset_name}_humanitarian_text_features.npz"
    if not cache_file.exists() and alt_cache.exists():
        cache_file = alt_cache

    if cache_file.exists() and not max_samples:
        logger.info(f"Loading cached features from {cache_file}")
        cached = np.load(cache_file)
        return (
            cached["features"],
            cached["labels"],
            cached["event_ids"].tolist(),
            cached["disaster_types"].tolist(),
            cached["regions"].tolist(),
        )


    texts = [s["text"] for s in samples]
    extractor = TextFeatureExtractor(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=cache_dir,
        batch_size=128,
    )
    logger.info(f"Extracting features for {len(texts)} samples...")
    features = extractor.extract_features(texts, dataset_name=dataset_name, show_progress=True)

    raw_labels = [s["label"] for s in samples]
    unique_labels = sorted(set(raw_labels))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    labels = np.array([label_map[lbl] for lbl in raw_labels], dtype=np.int64)

    event_ids = [s["event_id"] for s in samples]
    disaster_types = [s["disaster_type"] for s in samples]
    regions = [s["region"] for s in samples]

    if not max_samples:
        np.savez_compressed(
            cache_file,
            features=features,
            labels=labels,
            event_ids=np.array(event_ids),
            disaster_types=np.array(disaster_types),
            regions=np.array(regions),
        )
        logger.info(f"Saved feature cache to {cache_file}")

    return features, labels, event_ids, disaster_types, regions


def evaluate_split(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_events: List[str] | None = None,
    test_events: List[str] | None = None,
    n_classes: int = 8,
) -> Dict[str, float]:
    """Train model on split and compute primary metrics, calibration, and abstention."""
    if model_name == "event_prior":
        model = EventPrior()
        model.fit(np.array(train_events), y_train)
        probs = model.predict_proba(np.array(test_events))
        preds = model.predict(np.array(test_events))

    elif model_name == "logistic":
        model = LogisticProbe(C=1.0, max_iter=500)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        # Ensure probs has shape (N, n_classes)
        if probs.shape[1] < n_classes:
            full_probs = np.zeros((len(y_test), n_classes), dtype=np.float32)
            for local_c, global_c in enumerate(model.model.classes_):
                full_probs[:, global_c] = probs[:, local_c]
            probs = full_probs
        preds = np.argmax(probs, axis=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    acc = float(np.mean(preds == y_test))

    # Balanced accuracy
    unique_classes = np.unique(y_test)
    class_accs = [np.mean(preds[y_test == c] == c) for c in unique_classes]
    bal_acc = float(np.mean(class_accs))

    # Calibration
    ece = expected_calibration_error(y_test, probs)
    nll = negative_log_likelihood(y_test, probs)

    # Abstention / Selective prediction
    confidences = np.max(probs, axis=1)
    val_aurc = aurc(y_test, preds, confidences)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "ece": ece,
        "nll": nll,
        "aurc": val_aurc,
    }


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"DISASTER AI BENCHMARK: {args.dataset.upper()}")
    logger.info("=" * 70)

    features, labels, event_ids, disaster_types, regions = load_dataset_and_features(
        args.dataset, data_root, cache_dir, args.max_samples
    )
    n_samples = len(labels)
    unique_events = sorted(set(event_ids))
    unique_types = sorted(set(disaster_types))
    unique_regions = sorted(set(regions))
    n_classes = len(np.unique(labels))

    logger.info(f"Dataset stats: {n_samples} samples, {len(unique_events)} events, {n_classes} classes")
    logger.info(f"Disaster types ({len(unique_types)}): {unique_types}")
    logger.info(f"Regions ({len(unique_regions)}): {unique_regions}")

    # Build all split families
    splits_dict: Dict[str, List[SplitResult]] = {
        "random": random_split(n_samples, labels, test_fraction=0.2, n_splits=args.n_random_splits, seed=args.seed),
        "loeo": leave_one_event_out(event_ids, min_test_samples=5),
        "lodto": leave_one_disaster_type_out(disaster_types, min_test_samples=5),
        "loro": leave_one_region_out(regions, min_test_samples=5),
    }

    models_to_test = ["event_prior", "logistic"]
    results_records = []

    for model_name in models_to_test:
        logger.info(f"\nEvaluating Model: {model_name}...")
        for split_type, split_list in splits_dict.items():
            logger.info(f"  Running {split_type.upper()} ({len(split_list)} folds)...")
            for split in split_list:
                X_tr = features[split.train_indices]
                y_tr = labels[split.train_indices]
                X_te = features[split.test_indices]
                y_te = labels[split.test_indices]

                tr_ev = [event_ids[i] for i in split.train_indices]
                te_ev = [event_ids[i] for i in split.test_indices]

                metrics = evaluate_split(
                    model_name=model_name,
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_te,
                    y_test=y_te,
                    train_events=tr_ev,
                    test_events=te_ev,
                    n_classes=n_classes,
                )

                record = {
                    "dataset": args.dataset,
                    "model": model_name,
                    "split_type": split_type,
                    "split_name": split.split_name,
                    "held_out": split.held_out_value or "none",
                    "n_train": split.n_train,
                    "n_test": split.n_test,
                    **metrics,
                }
                results_records.append(record)

    df_results = pd.DataFrame(results_records)
    csv_path = output_dir / f"{args.dataset}_benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nAll benchmark split runs saved to {csv_path}")

    # Aggregations & Confidence Intervals
    summary_rows = []
    for model_name in models_to_test:
        for split_type in ["random", "loeo", "lodto", "loro"]:
            sub = df_results[(df_results["model"] == model_name) & (df_results["split_type"] == split_type)]
            if len(sub) == 0:
                continue

            mean_acc = sub["accuracy"].mean()
            mean_bal = sub["balanced_accuracy"].mean()
            mean_ece = sub["ece"].mean()
            mean_aurc = sub["aurc"].mean()
            worst_acc = sub["accuracy"].min()

            # Cluster bootstrap CI for LOEO
            if split_type == "loeo" and len(sub) >= 3:
                scores = sub["accuracy"].values
                events = sub["held_out"].values
                _, ci_l, ci_u = cluster_bootstrap_ci(scores, events, n_boot=1000, random_state=42)
            else:
                std = sub["accuracy"].std()
                ci_l = mean_acc - 1.96 * (std / np.sqrt(max(len(sub), 1)))
                ci_u = mean_acc + 1.96 * (std / np.sqrt(max(len(sub), 1)))

            summary_rows.append({
                "model": model_name,
                "split_type": split_type,
                "accuracy": mean_acc,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "worst_event_accuracy": worst_acc,
                "balanced_accuracy": mean_bal,
                "ece": mean_ece,
                "aurc": mean_aurc,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / f"{args.dataset}_benchmark_summary.csv", index=False)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY: {args.dataset.upper()}")
    print("=" * 80)
    print(df_summary[["model", "split_type", "accuracy", "ci_lower", "ci_upper", "worst_event_accuracy", "ece", "aurc"]].to_string(index=False))

    # Kendall's tau rank correlation between Random and LOEO
    rand_scores = df_results[df_results["split_type"] == "random"].groupby("model")["accuracy"].mean()
    loeo_scores = df_results[df_results["split_type"] == "loeo"].groupby("model")["accuracy"].mean()
    models_common = [m for m in models_to_test if m in rand_scores and m in loeo_scores]
    if len(models_common) >= 2:
        tau, p_val = kendalltau([rand_scores[m] for m in models_common], [loeo_scores[m] for m in models_common])
        print(f"\nKendall's tau ranking correlation (Random vs LOEO): tau={tau:.3f} (p={p_val:.3f})")

    # Generate LaTeX table
    tex_rows = []
    for model_name in models_to_test:
        r_acc = df_summary[(df_summary["model"] == model_name) & (df_summary["split_type"] == "random")]["accuracy"].values[0]
        l_acc = df_summary[(df_summary["model"] == model_name) & (df_summary["split_type"] == "loeo")]["accuracy"].values[0]
        w_acc = df_summary[(df_summary["model"] == model_name) & (df_summary["split_type"] == "loeo")]["worst_event_accuracy"].values[0]
        l_ece = df_summary[(df_summary["model"] == model_name) & (df_summary["split_type"] == "loeo")]["ece"].values[0]
        l_aurc = df_summary[(df_summary["model"] == model_name) & (df_summary["split_type"] == "loeo")]["aurc"].values[0]
        gap = r_acc - l_acc
        tex_rows.append(
            f"{model_name.upper()} & {r_acc:.3f} & {l_acc:.3f} & \\textbf{{{w_acc:.3f}}} & +{gap:.3f} & {l_ece:.3f} & {l_aurc:.3f} \\\\"
        )

    latex_table = r"""\begin{table}[t]
\centering
\caption{Benchmark results on """ + args.dataset.upper() + r""": Random vs. Leave-One-Event-Out (LOEO) evaluation across models. Worst-Event Acc indicates tail risk on the hardest unseen disaster.}
\label{tab:benchmark_""" + args.dataset + r"""}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Random Acc} & \textbf{LOEO Acc} & \textbf{Worst-Event} & $\Delta$ \textbf{Gap} & \textbf{LOEO ECE} & \textbf{LOEO AURC} \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / f"table_{args.dataset}.tex", "w") as fp:
        fp.write(latex_table)
    logger.info(f"LaTeX table saved to {output_dir / f'table_{args.dataset}.tex'}")


if __name__ == "__main__":
    main()
