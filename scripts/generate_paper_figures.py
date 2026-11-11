#!/usr/bin/env python3
"""Generate publication-ready figures for the disaster AI audit paper.

Produces:
1. Fig 1: Split Inflation and Worst-Event Tail Collapse (Macro vs Pooled, LORO, LODTO, Worst Event)
2. Fig 2: Near-Duplicate Leakage Analysis (Association Gap vs Net Test Set Effect vs Causal Deduplication)
3. Fig 3: Empirical & Analytic Resampling Precision (Bootstrap vs Analytic vs FPC curves, k = 15 benchmark)
4. Fig 4: Compound Stress Response Heatmap on HumAID (Gaussian Noise sigma + NSR vs Feature Dropout)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Publication style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
})

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_split_inflation() -> None:
    """Fig 1: Compare Random vs LOEO vs LODTO vs LORO vs Worst Event with Macro and Pooled metrics."""
    data = [
        # CrisisMMD
        {"Dataset": "CrisisMMD", "Protocol": "Random (Pooled)", "Accuracy": 55.12, "Type": "Random"},
        {"Dataset": "CrisisMMD", "Protocol": "Random (Macro)", "Accuracy": 54.78, "Type": "Random"},
        {"Dataset": "CrisisMMD", "Protocol": "LORO (Region)", "Accuracy": 52.66, "Type": "OOD"},
        {"Dataset": "CrisisMMD", "Protocol": "LOEO (Pooled)", "Accuracy": 53.24, "Type": "OOD"},
        {"Dataset": "CrisisMMD", "Protocol": "LOEO (Macro)", "Accuracy": 52.08, "Type": "OOD"},
        {"Dataset": "CrisisMMD", "Protocol": "LODTO (Type)", "Accuracy": 51.24, "Type": "OOD"},
        {"Dataset": "CrisisMMD", "Protocol": "Worst Event", "Accuracy": 39.33, "Type": "Worst"},
        # HumAID
        {"Dataset": "HumAID", "Protocol": "Random (Pooled)", "Accuracy": 72.69, "Type": "Random"},
        {"Dataset": "HumAID", "Protocol": "Random (Macro)", "Accuracy": 74.93, "Type": "Random"},
        {"Dataset": "HumAID", "Protocol": "LORO (Region)*", "Accuracy": 73.36, "Type": "OOD"},
        {"Dataset": "HumAID", "Protocol": "LOEO (Pooled)", "Accuracy": 69.60, "Type": "OOD"},
        {"Dataset": "HumAID", "Protocol": "LOEO (Macro)", "Accuracy": 71.69, "Type": "OOD"},
        {"Dataset": "HumAID", "Protocol": "LODTO (Type)", "Accuracy": 69.73, "Type": "OOD"},
        {"Dataset": "HumAID", "Protocol": "Worst Event", "Accuracy": 51.48, "Type": "Worst"},
    ]
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    palette = {
        "Random (Pooled)": "#1f77b4",
        "Random (Macro)": "#aec7e8",
        "LORO (Region)": "#2ca02c",
        "LORO (Region)*": "#2ca02c",
        "LOEO (Pooled)": "#ff7f0e",
        "LOEO (Macro)": "#ffbb78",
        "LODTO (Type)": "#9467bd",
        "Worst Event": "#d62728",
    }

    sns.barplot(
        data=df,
        x="Dataset",
        y="Accuracy",
        hue="Protocol",
        palette=palette,
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    ax.set_ylim(30, 82)
    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax.set_xlabel("Harmonized Crisis Benchmark", fontweight="bold")
    ax.set_title("OOD Evaluation Reveals Tail Collapse and Split Sensitivity", fontweight="bold", pad=12)
    ax.legend(title="Evaluation Protocol", frameon=True, loc="upper right", ncol=2)

    # Annotate Paired Shift Drops on Worst Events
    ax.annotate("Worst: California Wildfires (39.3%)\nPaired drop: -5.54 pp vs random\n(Max paired drop: Iraq-Iran -6.97 pp)",
                xy=(-0.05, 39.5), xytext=(-0.35, 43.5),
                arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1.2, headwidth=5),
                fontsize=8.5, fontweight="semibold", color="#a50f15", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fee5d9", ec="#d62728", lw=0.8))

    ax.annotate("Worst: Maryland Floods (51.5%)\nPaired drop: -15.52 pp vs random\n(Prior KL divergence: 0.78)",
                xy=(0.95, 51.8), xytext=(0.60, 56.5),
                arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1.2, headwidth=5),
                fontsize=8.5, fontweight="semibold", color="#a50f15", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fee5d9", ec="#d62728", lw=0.8))

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig1_split_inflation.pdf"
    out_png = FIG_DIR / "fig1_split_inflation.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {out_pdf} and {out_png}")


def plot_leakage_impact() -> None:
    """Fig 2: Dissecting Near-Duplicate Leakage: Slice Association vs Net Test Inflation vs Causal Purging (5-Seed Means)."""
    data = [
        # CrisisMMD (5-seed means: 16.3% leaked at tau=0.90, N=588.6 / 3,617)
        {"Dataset": "CrisisMMD", "Metric": "Full Test (Random)", "Accuracy": 55.12, "Annotation": "N=3,617"},
        {"Dataset": "CrisisMMD", "Metric": "Leaked Slice (tau>=0.90)", "Accuracy": 59.64, "Annotation": "N=588.6 (16.3%)"},
        {"Dataset": "CrisisMMD", "Metric": "Unleaked Clean Slice", "Accuracy": 54.24, "Annotation": "N=3,028.4 (83.7%)"},
        {"Dataset": "CrisisMMD", "Metric": "Causally Purged Model", "Accuracy": 54.43, "Annotation": "Net Drop: -0.69 pp"},
        # HumAID (5-seed means: 4.41% leaked at tau=0.90, N=674.2 / 15,297)
        {"Dataset": "HumAID", "Metric": "Full Test (Random)", "Accuracy": 72.69, "Annotation": "N=15,297"},
        {"Dataset": "HumAID", "Metric": "Leaked Slice (tau>=0.90)", "Accuracy": 87.47, "Annotation": "N=674.2 (4.4%)"},
        {"Dataset": "HumAID", "Metric": "Unleaked Clean Slice", "Accuracy": 72.01, "Annotation": "N=14,622.8 (95.6%)"},
        {"Dataset": "HumAID", "Metric": "Causally Purged Model", "Accuracy": 72.69, "Annotation": "Net Drop: 0.00 pp"},
    ]
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    palette = {
        "Full Test (Random)": "#1f77b4",
        "Leaked Slice (tau>=0.90)": "#d62728",
        "Unleaked Clean Slice": "#2ca02c",
        "Causally Purged Model": "#9467bd",
    }

    sns.barplot(
        data=df,
        x="Dataset",
        y="Accuracy",
        hue="Metric",
        palette=palette,
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    ax.set_ylim(45, 95)
    ax.set_ylabel("Classification Accuracy (%) [5-Seed Mean]", fontweight="bold")
    ax.set_xlabel("Benchmark Dataset", fontweight="bold")
    ax.set_title("Near-Duplicate Leakage: Association vs. Net Benchmark Inflation", fontweight="bold", pad=12)
    ax.legend(title="Evaluation Cohort / Condition", frameon=True, loc="upper left")

    # Annotations highlighting sample sizes and true net effects
    ax.annotate("CrisisMMD (tau >= 0.90):\n• Leaked slice advantage: +5.40 pp (N=588.6)\n• Net test effect: +0.88 pp\n• Causal drop when purged: -0.69 pp\n[17.5% of dataset are duplicate retweets]",
                xy=(-0.05, 60.5), xytext=(-0.35, 71.0),
                arrowprops=dict(facecolor="#333333", shrink=0.05, width=1.0, headwidth=4),
                fontsize=8.5, color="#111111", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", lw=0.8))

    ax.annotate("HumAID (tau >= 0.90):\n• Leaked slice advantage: +15.46 pp (N=674.2)\n• Net test effect: +0.68 pp\n• Causal drop when purged: 0.00 pp\n[Note: at tau>=0.98, N=6.2 samples (0.04%)]",
                xy=(0.95, 87.5), xytext=(0.60, 68.0),
                arrowprops=dict(facecolor="#333333", shrink=0.05, width=1.0, headwidth=4),
                fontsize=8.5, color="#111111", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", lw=0.8))

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig2_leakage_impact.pdf"
    out_png = FIG_DIR / "fig2_leakage_impact.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {out_pdf} and {out_png}")


def plot_power_analysis() -> None:
    """Fig 3: Empirical Resampling Precision vs. Number of Held-Out Events."""
    csv_path = PROJECT_ROOT / "outputs" / "power_analysis" / "power_analysis_curve.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 3: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Empirical values from HumAID event LOEO results: SD across 19 events = 0.0768 (7.68 pp)
    sd_events = 0.0768
    N_total = 19
    k_dense = np.linspace(2, 18.5, 100)

    # Analytic curve (infinite superpopulation assumption): 2 * 1.96 * SD / sqrt(k)
    ci_analytic = (2 * 1.96 * sd_events / np.sqrt(k_dense)) * 100

    # Finite Population Correction (FPC) curve: 2 * 1.96 * SD / sqrt(k) * sqrt((N - k) / (N - 1))
    fpc_factor = np.sqrt(np.maximum(0, (N_total - k_dense) / (N_total - 1)))
    ci_fpc = ci_analytic * fpc_factor

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=300)

    # Plot lines
    ax.plot(df["k"], df["mean_ci_width"] * 100, marker="o", color="#08519c", linewidth=2.2, label="Empirical Cluster Bootstrap (without repl.)")
    ax.plot(k_dense, ci_analytic, linestyle="--", color="#e6550d", linewidth=1.8, label="Analytic Superpopulation (2 × 1.96 × SD / $\\sqrt{k}$)")
    ax.plot(k_dense, ci_fpc, linestyle="-.", color="#756bb1", linewidth=1.8, label="Analytic with FPC (Finite Population Correction, N=19)")

    # Thresholds and bounds
    ax.axhline(y=11.0, color="#de2d26", linestyle=":", linewidth=1.5, label="Target Precision Bound (11.0 pp CI Width, ±5.5 pp)")
    ax.axvline(x=15, color="#31a354", linestyle="--", linewidth=1.5, label="Precision Benchmark ($k = 15$ events)")

    ax.set_xlabel("Number of Held-Out Events ($k$)", fontweight="bold")
    ax.set_ylabel("95% Confidence Interval Width (pp)", fontweight="bold")
    ax.set_title("Resampling Precision: Held-Out Event Sample Size vs. 95% CI Width", fontweight="bold", pad=12)
    ax.set_xticks(range(2, 20, 2))
    ax.set_ylim(0, 25)
    ax.legend(frameon=True, loc="upper right")

    # Callout annotation
    ax.annotate("At $k=15$:\n• Empirical CI width: 10.96 pp (half-width ±5.48 pp)\n• Superpopulation width: 7.77 pp\n• FPC-adjusted width: 3.66 pp",
                xy=(15, 10.96), xytext=(9.5, 3.5),
                arrowprops=dict(facecolor="#08519c", shrink=0.05, width=1.2, headwidth=5),
                fontsize=8.5, color="#08519c",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eff3ff", ec="#08519c", lw=0.8))

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig3_power_analysis.pdf"
    out_png = FIG_DIR / "fig3_power_analysis.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {out_pdf} and {out_png}")


def plot_compound_stress_heatmap() -> None:
    """Fig 4: Heatmap of Accuracy under Event Shift + Gaussian Noise + Feature Dropout on HumAID."""
    csv_path = PROJECT_ROOT / "outputs" / "compound_shift" / "humaid_compound_stress_results.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 4: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Filter to representative subset of noise sigmas for clear visual presentation
    target_sigmas = [0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50]
    df_sub = df[df["noise_sigma"].isin(target_sigmas)]

    # Compute mean accuracy across folds (converted to %)
    pivot = df_sub.groupby(["noise_sigma", "dropout_rate"])["accuracy"].mean().unstack() * 100

    # Format y-index with Noise-to-Signal Ratio (NSR)
    def label_row(sigma: float) -> str:
        nsr = np.sqrt(384 * (sigma ** 2))
        return f"{sigma:.3f} (NSR: {nsr:.2f})" if sigma > 0 else f"{sigma:.3f} (Clean)"

    pivot.index = [label_row(s) for s in pivot.index]

    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=300)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd_r",
        cbar_kws={"label": "LOEO Accuracy (%) [19 Folds Mean]"},
        linewidths=0.6,
        ax=ax,
    )
    ax.set_title("HumAID Compound Stress: Event Shift + Noise (NSR) vs Dropout", fontweight="bold", pad=12)
    ax.set_xlabel("Feature Dimension Dropout Fraction", fontweight="bold")
    ax.set_ylabel("Sensor Noise $\\sigma$ and Noise-to-Signal Ratio (NSR)", fontweight="bold")

    plt.tight_layout()
    out_pdf = FIG_DIR / "fig4_compound_stress_heatmap.pdf"
    out_png = FIG_DIR / "fig4_compound_stress_heatmap.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {out_pdf} and {out_png}")


def main() -> None:
    plot_split_inflation()
    plot_leakage_impact()
    plot_power_analysis()
    plot_compound_stress_heatmap()
    print("All updated publication figures successfully generated!")


if __name__ == "__main__":
    main()
