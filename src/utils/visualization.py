"""Visualization utilities for disaster AI audit.

Functions for generating publication-quality plots:
- Per-event accuracy bar charts
- Heatmaps (event × model accuracy)
- Reliability diagrams (calibration)
- Risk-coverage curves (selective prediction)
- Rank-change visualisations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colour palette
PALETTE = sns.color_palette("Set2", 10)


def plot_per_event_accuracy(
    event_names: Sequence[str],
    accuracies: Sequence[float],
    title: str = "Per-Event Accuracy",
    ylabel: str = "Accuracy",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    color: str | None = None,
    sort: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of per-event accuracy.

    Parameters
    ----------
    event_names : Sequence[str]
        Event identifiers.
    accuracies : Sequence[float]
        Accuracy for each event.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str | Path | None
        If given, save the figure to this path.
    ax : plt.Axes | None
        Existing axes to draw on; creates a new figure if ``None``.
    color : str | None
        Bar colour.
    sort : bool
        Sort events by accuracy (ascending).

    Returns
    -------
    plt.Figure
    """
    names = list(event_names)
    accs = list(accuracies)

    if sort:
        order = np.argsort(accs)
        names = [names[i] for i in order]
        accs = [accs[i] for i in order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    else:
        fig = ax.figure

    bars = ax.barh(names, accs, color=color or PALETTE[0], edgecolor="white")
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1.05)

    # Annotate values
    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path))
    return fig


def plot_accuracy_heatmap(
    df: pd.DataFrame,
    title: str = "Accuracy Heatmap (Event × Model)",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "YlOrRd",
    annot: bool = True,
    fmt: str = ".2f",
) -> plt.Figure:
    """Heatmap of accuracy values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rows = events, columns = models, values = accuracy.
    title : str
        Plot title.
    save_path : str | Path | None
        If given, save the figure.
    figsize : tuple[int, int]
        Figure size.
    cmap : str
        Colormap name.
    annot : bool
        Annotate cells with values.
    fmt : str
        Number format for annotations.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_ylabel("Event")
    ax.set_xlabel("Model")
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path))
    return fig


def plot_reliability_diagram(
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_counts: np.ndarray,
    title: str = "Reliability Diagram",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (6, 6),
) -> plt.Figure:
    """Reliability diagram (calibration plot).

    Parameters
    ----------
    bin_accuracies : np.ndarray
        Mean accuracy in each confidence bin.
    bin_confidences : np.ndarray
        Mean confidence in each bin.
    bin_counts : np.ndarray
        Number of samples in each bin.
    title : str
        Plot title.
    save_path : str | Path | None
        If given, save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    n_bins = len(bin_accuracies)
    bin_width = 1.0 / n_bins

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Gap bars
    ax1.bar(
        bin_confidences,
        bin_accuracies,
        width=bin_width * 0.8,
        color=PALETTE[0],
        edgecolor="white",
        alpha=0.8,
        label="Model",
    )

    # Gap fill
    for bc, ba in zip(bin_confidences, bin_accuracies):
        if ba < bc:
            ax1.bar(
                bc, bc - ba, bottom=ba, width=bin_width * 0.8,
                color="salmon", alpha=0.3,
            )

    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of counts
    ax2.bar(
        bin_confidences,
        bin_counts,
        width=bin_width * 0.8,
        color=PALETTE[1],
        edgecolor="white",
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path))
    return fig


def plot_risk_coverage(
    coverages: np.ndarray,
    risks: np.ndarray,
    title: str = "Risk-Coverage Curve",
    aurc: float | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (7, 5),
) -> plt.Figure:
    """Risk-coverage curve for selective prediction.

    Parameters
    ----------
    coverages : np.ndarray
        Coverage values (fraction of samples retained).
    risks : np.ndarray
        Risk (error rate) at each coverage level.
    title : str
        Plot title.
    aurc : float | None
        Pre-computed AURC to display in legend.
    save_path : str | Path | None
        If given, save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    label = "Model"
    if aurc is not None:
        label += f" (AURC={aurc:.4f})"

    ax.plot(coverages, risks, "-", color=PALETTE[0], linewidth=2, label=label)
    ax.fill_between(coverages, risks, alpha=0.15, color=PALETTE[0])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path))
    return fig


def plot_rank_changes(
    model_names: Sequence[str],
    random_ranks: Sequence[int],
    loeo_ranks: Sequence[int],
    title: str = "Model Rank Change: Random → LOEO",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Bump chart showing how model rankings change between split strategies.

    Parameters
    ----------
    model_names : Sequence[str]
        Model names.
    random_ranks : Sequence[int]
        Ranks under random split (1 = best).
    loeo_ranks : Sequence[int]
        Ranks under LOEO split.
    title : str
        Plot title.
    save_path : str | Path | None
        If given, save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = len(model_names)
    for i in range(n):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            [0, 1],
            [random_ranks[i], loeo_ranks[i]],
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=model_names[i],
        )
        # Labels
        ax.text(-0.05, random_ranks[i], model_names[i], ha="right", va="center", fontsize=9)
        ax.text(1.05, loeo_ranks[i], model_names[i], ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0.5, n + 0.5)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random Split", "LOEO Split"])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path))
    return fig


def plot_multi_model_bar(
    results_df: pd.DataFrame,
    x: str = "model",
    y: str = "accuracy",
    hue: str = "split",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Grouped bar chart comparing models across split strategies.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form DataFrame with columns for model, metric, and split.
    x, y, hue : str
        Column names for the x-axis, y-axis, and grouping variable.
    title : str
        Plot title.
    save_path : str | Path | None
        If given, save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=results_df, x=x, y=y, hue=hue, ax=ax, palette="Set2")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=hue.capitalize())
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path))
    return fig
