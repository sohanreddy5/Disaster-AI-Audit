#!/usr/bin/env python3
"""Generate publication-ready LaTeX tables from disaster AI benchmark results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for the paper.")
    parser.add_argument("--results-csv", type=str, default="./outputs/go_no_go/go_no_go_results.csv")
    parser.add_argument("--output-tex", type=str, default="./docs/table_split_inflation.tex")
    return parser.parse_args()


def generate_inflation_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table comparing Random vs LOEO accuracy and F1."""
    grouped = df.groupby(["model", "split_name"]).agg({"accuracy": "mean", "macro_f1": "mean"}).reset_index()

    models = df["model"].unique()
    rows = []

    for m in models:
        m_df = df[df["model"] == m]
        rand_acc = m_df[m_df["split_name"].str.startswith("random")]["accuracy"].mean()
        rand_f1 = m_df[m_df["split_name"].str.startswith("random")]["macro_f1"].mean()

        loeo_df = m_df[m_df["split_name"].str.startswith("loeo")]
        loeo_acc = loeo_df["accuracy"].mean()
        loeo_f1 = loeo_df["macro_f1"].mean()
        worst_acc = loeo_df["accuracy"].min()

        gap_acc = rand_acc - loeo_acc
        gap_f1 = rand_f1 - loeo_f1

        rows.append(
            f"{m.upper()} & {rand_acc:.3f} & {loeo_acc:.3f} & \\textbf{{{worst_acc:.3f}}} & "
            f"+{gap_acc:.3f} & {rand_f1:.3f} & {loeo_f1:.3f} & +{gap_f1:.3f} \\\\"
        )

    table = r"""\begin{table*}[t]
\centering
\small
\caption{Comparison of classification performance under Random Split vs. Leave-One-Event-Out (LOEO) evaluation. $\Delta$ denotes the score overestimation (inflation) caused by random event leakage.}
\label{tab:random_vs_loeo}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \multicolumn{4}{c}{\textbf{Accuracy}} & \multicolumn{3}{c}{\textbf{Macro-F1}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-8}
 & \textbf{Random} & \textbf{LOEO} & \textbf{Worst-Event} & $\Delta$ \textbf{Gap} & \textbf{Random} & \textbf{LOEO} & $\Delta$ \textbf{Gap} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    return table


def main() -> None:
    args = parse_args()
    csv_p = Path(args.results_csv)
    if not csv_p.is_absolute():
        csv_p = PROJECT_ROOT / csv_p

    out_p = Path(args.output_tex)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not csv_p.exists():
        print(f"Results CSV not found at {csv_p}. Run scripts/run_go_no_go.py first.")
        return

    df = pd.read_csv(csv_p)
    latex = generate_inflation_table(df)

    with open(out_p, "w") as f:
        f.write(latex)

    print(f"LaTeX table generated at {out_p}")
    print("\nPreview:\n" + latex)


if __name__ == "__main__":
    main()
