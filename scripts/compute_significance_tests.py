#!/usr/bin/env python3
"""Compute statistical significance tests (Paired Wilcoxon & Student's t-test) across LOEO folds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def analyze_dataset_dg(csv_path: Path, dataset_name: str) -> None:
    if not csv_path.exists():
        logger.warning(f"File {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    logger.info("=" * 70)
    logger.info(f"STATISTICAL SIGNIFICANCE TESTS: {dataset_name.upper()}")
    logger.info("=" * 70)

    methods = df["method"].unique()
    erm_df = df[df["method"] == "ERM"].sort_values(by="held_out_event")
    erm_accs = erm_df["accuracy"].values
    erm_eces = erm_df["ece"].values
    erm_aurcs = erm_df["aurc"].values

    for m in methods:
        if m == "ERM":
            continue

        m_df = df[df["method"] == m].sort_values(by="held_out_event")
        m_accs = m_df["accuracy"].values
        m_eces = m_df["ece"].values
        m_aurcs = m_df["aurc"].values

        # Paired Wilcoxon signed-rank test
        diff_acc = m_accs - erm_accs
        if np.all(diff_acc == 0):
            w_stat_acc, p_val_acc = 0.0, 1.0
        else:
            w_stat_acc, p_val_acc = stats.wilcoxon(m_accs, erm_accs)

        # Paired t-test
        t_stat_acc, p_val_t = stats.ttest_rel(m_accs, erm_accs)

        # ECE comparison
        diff_ece = m_eces - erm_eces
        if np.all(diff_ece == 0):
            w_stat_ece, p_val_ece = 0.0, 1.0
        else:
            w_stat_ece, p_val_ece = stats.wilcoxon(m_eces, erm_eces)

        logger.info(f"\n--- {m} vs ERM ---")
        logger.info(f"  Mean Acc: {m_accs.mean()*100:.2f}% vs ERM {erm_accs.mean()*100:.2f}% (Diff: {diff_acc.mean()*100:+.2f} pp)")
        logger.info(f"  Wilcoxon Signed-Rank Test: stat={w_stat_acc:.2f}, p-value={p_val_acc:.4f}")
        logger.info(f"  Paired t-Test: t={t_stat_acc:.2f}, p-value={p_val_t:.4f}")
        logger.info(f"  Mean ECE: {m_eces.mean():.4f} vs ERM {erm_eces.mean():.4f} (p={p_val_ece:.4f})")
        logger.info(f"  Worst-Event Acc: {m_accs.min()*100:.2f}% vs ERM {erm_accs.min()*100:.2f}%")


def main() -> None:
    setup_logging(level="INFO")
    analyze_dataset_dg(
        PROJECT_ROOT / "outputs" / "dg_benchmark" / "crisismmd_dg_benchmark_results.csv",
        "CrisisMMD",
    )
    analyze_dataset_dg(
        PROJECT_ROOT / "outputs" / "dg_benchmark" / "humaid_dg_benchmark_results.csv",
        "HumAID",
    )


if __name__ == "__main__":
    main()
