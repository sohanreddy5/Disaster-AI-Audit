#!/usr/bin/env python3
"""Run empirical power analysis across held-out event sample sizes (k).

Generates:
1. CI Width vs. k curve (how confidence interval width narrows as more events are added)
2. Ranking Stability vs. k curve (Kendall's tau vs. full benchmark ranking)
3. High-resolution publication figure in outputs/power_analysis/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.power_analysis import subsample_events_power
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Power Analysis on LOEO Benchmark Results.")
    parser.add_argument("--results-csv", type=str, required=True, help="Path to benchmark_results.csv containing LOEO fold results")
    parser.add_argument("--output-dir", type=str, default="./outputs/power_analysis", help="Output directory")
    parser.add_argument("--n-boot", type=int, default=500, help="Number of bootstrap resamples per k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    results_p = Path(args.results_csv)
    if not results_p.is_absolute():
        results_p = PROJECT_ROOT / results_p

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_p.exists():
        logger.error(f"Results file not found: {results_p}")
        return

    df = pd.read_csv(results_p)
    loeo_df = df[df["split_type"] == "loeo"]

    models = sorted(loeo_df["model"].unique())
    events = sorted(loeo_df["held_out"].unique())
    n_total_events = len(events)
    logger.info(f"Loaded LOEO results: {len(loeo_df)} rows, {len(models)} models, {n_total_events} events")

    # Build event score matrix for each model
    all_event_scores = {}
    for m in models:
        m_df = loeo_df[loeo_df["model"] == m].set_index("held_out")
        scores = np.array([m_df.loc[e, "accuracy"] if e in m_df.index else 0.0 for e in events])
        all_event_scores[m] = scores

    # Determine k_range
    if n_total_events <= 5:
        k_range = list(range(2, n_total_events))
    elif n_total_events <= 10:
        k_range = [3, 4, 5, 6, 7, n_total_events - 1]
    else:
        k_range = [3, 5, 7, 10, 12, 15, min(18, n_total_events - 1)]
    k_range = [k for k in k_range if 2 <= k < n_total_events]

    logger.info(f"Evaluating event sample sizes k ∈ {k_range} (n_boot={args.n_boot})...")
    df_power = subsample_events_power(
        all_event_scores=all_event_scores,
        k_range=k_range,
        n_boot=args.n_boot,
        random_state=args.seed,
    )

    power_csv = out_dir / "power_analysis_curve.csv"
    df_power.to_csv(power_csv, index=False)
    logger.info(f"Saved power analysis metrics to {power_csv}")

    # Plot figure
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

    color_ci = "#1f77b4"
    color_rank = "#2ca02c"

    ax1.set_xlabel("Number of Held-Out Events ($k$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("95% Bootstrap CI Width ($\Delta$ Acc)", color=color_ci, fontsize=12, fontweight="bold")
    line1 = ax1.plot(df_power["k"], df_power["mean_ci_width"], marker="o", color=color_ci, linewidth=2.5, label="Mean 95% CI Width")
    ax1.tick_params(axis="y", labelcolor=color_ci)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Ranking Stability (Kendall's $\\tau$ vs. Full)", color=color_rank, fontsize=12, fontweight="bold")
    line2 = ax2.plot(df_power["k"], df_power["rank_stability"], marker="s", color=color_rank, linewidth=2.5, linestyle="--", label="Ranking Stability ($\\tau$)")
    ax2.tick_params(axis="y", labelcolor=color_rank)
    ax2.set_ylim(-0.05, 1.05)

    # Threshold indicator at 3pp CI width
    ax1.axhline(0.05, color="red", linestyle=":", alpha=0.7, label="5 pp Precision Bound")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title(f"Benchmark Statistical Power Analysis ({results_p.stem})", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()

    fig_path = out_dir / "power_analysis_curve.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Power analysis figure saved to {fig_path}")


if __name__ == "__main__":
    main()
