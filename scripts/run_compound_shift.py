#!/usr/bin/env python3
"""Compound Shift Benchmark on Multimodal Disaster AI.

Evaluates model robustness under simultaneous compound shifts:
1. Joint Semantic + Geographic Shift: Leave-(Disaster-Type x Region)-Out
2. Modality / Feature Stress Shift: Event Shift (LOEO) under varying noise/missingness
3. Selective Prediction & Uncertainty: Risk-Coverage (AURC) and Calibration (ECE)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import compound_shift_split, leave_one_event_out
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.abstention import aurc
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def run_joint_shift_benchmark(
    features: np.ndarray,
    labels: np.ndarray,
    event_ids: List[str],
    disaster_types: List[str],
    regions: List[str],
    dataset_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Benchmark under joint (Disaster Type x Region) compound shift."""
    logger.info("=" * 70)
    logger.info(f"JOINT (TYPE x REGION) COMPOUND SHIFT: {dataset_name.upper()}")
    logger.info("=" * 70)

    splits = compound_shift_split(
        event_ids=event_ids,
        disaster_types=disaster_types,
        regions=regions,
        labels=labels,
        min_test_samples=50,
    )
    logger.info(f"Generated {len(splits)} compound (Type x Region) held-out splits.")

    results = []
    for split in splits:
        held_out_pair = split.held_out_value
        tr_idx = np.array(split.train_indices)
        te_idx = np.array(split.test_indices)

        # Skip if single class in test or train
        if len(set(labels[tr_idx])) < 2 or len(set(labels[te_idx])) < 2:
            continue

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(features[tr_idx], labels[tr_idx])

        probs = clf.predict_proba(features[te_idx])
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(labels[te_idx], preds)
        macro_f1 = f1_score(labels[te_idx], preds, average="macro", zero_division=0)
        ece = expected_calibration_error(labels[te_idx], probs)
        confidences = np.max(probs, axis=1)
        aurc_val = aurc(labels[te_idx], preds, confidences)

        results.append({
            "held_out_pair": held_out_pair,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "ece": ece,
            "aurc": aurc_val,
        })
        logger.info(
            f"Held-out '{held_out_pair:25s}' (N={len(te_idx):5d}) -> "
            f"Acc: {acc * 100:.2f}% | F1: {macro_f1:.4f} | ECE: {ece:.4f} | AURC: {aurc_val:.4f}"
        )

    df_joint = pd.DataFrame(results)
    csv_path = output_dir / f"{dataset_name}_compound_joint_shift_results.csv"
    df_joint.to_csv(csv_path, index=False)
    logger.info(f"Saved joint compound shift results to {csv_path}")
    return df_joint


def run_modality_stress_benchmark(
    features: np.ndarray,
    labels: np.ndarray,
    event_ids: List[str],
    dataset_name: str,
    output_dir: Path,
    noise_levels: List[float] = [0.0, 0.1, 0.25, 0.5, 0.75],
    dropout_rates: List[float] = [0.0, 0.1, 0.25, 0.5],
) -> pd.DataFrame:
    """Benchmark LOEO shift under simultaneous feature corruption / missingness."""
    logger.info("\n" + "=" * 70)
    logger.info(f"COMPOUND STRESS SHIFT (EVENT SHIFT + SENSOR DEGRADATION): {dataset_name.upper()}")
    logger.info("=" * 70)

    folds = leave_one_event_out(event_ids, min_test_samples=100)
    results = []

    for fold in folds:
        ev = fold.held_out_value
        tr_idx = np.array(fold.train_indices)
        te_idx = np.array(fold.test_indices)

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(features[tr_idx], labels[tr_idx])

        X_test_clean = features[te_idx]
        y_test = labels[te_idx]

        for sigma in noise_levels:
            for d_rate in dropout_rates:
                np.random.seed(42)
                # Feature corruption: Gaussian sensor noise + random modality dropout
                noise = np.random.normal(0, sigma, size=X_test_clean.shape) if sigma > 0 else 0.0
                mask = np.random.binomial(1, 1.0 - d_rate, size=X_test_clean.shape) if d_rate > 0 else 1.0

                X_corrupted = (X_test_clean + noise) * mask

                probs = clf.predict_proba(X_corrupted)
                preds = np.argmax(probs, axis=1)

                acc = accuracy_score(y_test, preds)
                ece = expected_calibration_error(y_test, probs)
                conf = np.max(probs, axis=1)
                aurc_val = aurc(y_test, preds, conf)

                results.append({
                    "event": ev,
                    "noise_sigma": sigma,
                    "dropout_rate": d_rate,
                    "accuracy": acc,
                    "ece": ece,
                    "aurc": aurc_val,
                })

    df_stress = pd.DataFrame(results)
    csv_path = output_dir / f"{dataset_name}_compound_stress_results.csv"
    df_stress.to_csv(csv_path, index=False)
    logger.info(f"Saved stress compound shift results to {csv_path}")

    # Aggregated summary
    summary = df_stress.groupby(["noise_sigma", "dropout_rate"]).agg({
        "accuracy": ["mean", "min", "std"],
        "ece": ["mean"],
        "aurc": ["mean"],
    }).reset_index()

    logger.info("\n" + "=" * 80)
    logger.info(f"COMPOUND STRESS SHIFT SUMMARY (LOEO + CORRUPTION) on {dataset_name.upper()}")
    logger.info("=" * 80)
    for _, row in summary.iterrows():
        sig = row["noise_sigma"].values[0] if hasattr(row["noise_sigma"], "values") else row["noise_sigma"]
        drp = row["dropout_rate"].values[0] if hasattr(row["dropout_rate"], "values") else row["dropout_rate"]
        m_acc = row["accuracy"]["mean"] * 100
        w_acc = row["accuracy"]["min"] * 100
        m_ece = row["ece"]["mean"]
        m_aurc = row["aurc"]["mean"]
        logger.info(
            f"Noise sigma={sig:.2f}, Dropout={drp:.2f} -> "
            f"Mean Acc: {m_acc:.2f}% | Worst Event: {w_acc:.2f}% | ECE: {m_ece:.4f} | AURC: {m_aurc:.4f}"
        )

    return df_stress


def generate_latex_table(df_joint: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    """Generate LaTeX table for joint compound shift."""
    tex_rows = []
    for _, row in df_joint.sort_values(by="accuracy", ascending=False).iterrows():
        pair_clean = row["held_out_pair"].replace("__", " $\\times$ ").replace("_", " ").title()
        tex_rows.append(
            f"{pair_clean} & {int(row['n_test'])} & "
            f"{row['accuracy'] * 100:.2f}\\% & "
            f"{row['macro_f1']:.3f} & "
            f"{row['ece']:.3f} & "
            f"{row['aurc']:.3f} \\\\"
        )

    mean_acc = df_joint["accuracy"].mean() * 100
    mean_f1 = df_joint["macro_f1"].mean()
    mean_ece = df_joint["ece"].mean()
    mean_aurc = df_joint["aurc"].mean()

    table_tex = r"""\begin{table}[t]
\centering
\caption{Model evaluation under Joint Compound Shift (Leave-(Disaster Type $\times$ Geographic Region)-Out) on """ + dataset_name.upper() + r""". Shows severe performance degradation when models face novel disaster categories in unobserved geographical settings.}
\label{tab:compound_shift_""" + dataset_name + r"""}
\begin{tabular}{lccccc}
\toprule
\textbf{Held-Out Compound Domain} & \textbf{Test $N$} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{ECE} & \textbf{AURC} \\
\midrule
""" + "\n".join(tex_rows) + r"""
\midrule
\textbf{Macro Average} & -- & \textbf{""" + f"{mean_acc:.2f}" + r"""\%} & \textbf{""" + f"{mean_f1:.3f}" + r"""} & \textbf{""" + f"{mean_ece:.3f}" + r"""} & \textbf{""" + f"{mean_aurc:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / f"table_compound_shift_{dataset_name}.tex", "w") as fp:
        fp.write(table_tex)
    logger.info(f"Saved LaTeX table to {output_dir / f'table_compound_shift_{dataset_name}.tex'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compound shift benchmark.")
    parser.add_argument("--features-npz", type=str, required=True, help="Features NPZ path")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name")
    parser.add_argument("--output-dir", type=str, default="./outputs/compound_shift", help="Output directory")
    args = parser.parse_args()

    setup_logging(level="INFO")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.features_npz)
    if not npz_path.is_absolute():
        npz_path = PROJECT_ROOT / npz_path

    data = np.load(npz_path)
    features = data["features"]
    labels = data["labels"]
    event_ids = data["event_ids"].tolist()
    disaster_types = data["disaster_types"].tolist()
    regions = data["regions"].tolist()

    df_joint = run_joint_shift_benchmark(
        features=features,
        labels=labels,
        event_ids=event_ids,
        disaster_types=disaster_types,
        regions=regions,
        dataset_name=args.dataset_name,
        output_dir=out_dir,
    )
    generate_latex_table(df_joint, args.dataset_name, out_dir)

    run_modality_stress_benchmark(
        features=features,
        labels=labels,
        event_ids=event_ids,
        dataset_name=args.dataset_name,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
