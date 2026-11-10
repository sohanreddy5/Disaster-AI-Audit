#!/usr/bin/env python3
"""Domain Generalization Benchmark on Disaster AI: ERM vs GroupDRO vs IRM vs CORAL.

Evaluates whether domain generalization algorithms improve out-of-distribution
generalization on unseen disaster events (LOEO) compared to standard ERM.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import leave_one_event_out, SplitResult
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.abstention import aurc
from src.models.domain_generalization import GroupDROTrainer, IRMTrainer, CORALTrainer
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class LinearClassifier(nn.Module):
    """Simple linear probe on frozen embeddings."""

    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ERMTrainer:
    """Standard Empirical Risk Minimization baseline."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = torch.device(device)

    def fit(self, train_loader: DataLoader) -> "ERMTrainer":
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for x, y, _ in train_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.long)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain Generalization Benchmark on Disaster AI.")
    parser.add_argument("--features-npz", type=str, required=True, help="Path to precomputed features .npz")
    parser.add_argument("--dataset-name", type=str, default="crisismmd", help="Dataset name")
    parser.add_argument("--output-dir", type=str, default="./outputs/dg_benchmark", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = Path(args.features_npz)
    if not npz_path.is_absolute():
        npz_path = PROJECT_ROOT / npz_path

    logger.info("=" * 70)
    logger.info(f"DOMAIN GENERALIZATION BENCHMARK: {args.dataset_name.upper()}")
    logger.info("=" * 70)

    data = np.load(npz_path)
    features = data["features"]
    labels = data["labels"]
    event_ids = data["event_ids"].tolist()

    n_samples, feat_dim = features.shape
    unique_events = sorted(set(event_ids))
    unique_classes = sorted(set(labels))
    n_classes = len(unique_classes)
    n_events = len(unique_events)

    logger.info(f"Samples: {n_samples}, Features: {feat_dim}, Classes: {n_classes}, Events: {n_events}")

    # Map events to integer IDs [0, n_events - 1]
    event_to_idx = {e: i for i, e in enumerate(unique_events)}
    group_indices = np.array([event_to_idx[e] for e in event_ids], dtype=np.int64)

    # Generate LOEO folds
    folds = leave_one_event_out(event_ids, min_test_samples=10)
    logger.info(f"Evaluating across {len(folds)} Leave-One-Event-Out folds...")

    methods = ["ERM", "GroupDRO", "IRM", "CORAL"]
    fold_results = []

    for fold_i, fold in enumerate(folds):
        held_out_event = fold.held_out_value
        logger.info(f"\n--- Fold [{fold_i + 1}/{len(folds)}]: Held-out Event '{held_out_event}' (Test size: {fold.n_test}) ---")

        tr_idx = fold.train_indices
        te_idx = fold.test_indices

        X_train, y_train, g_train = features[tr_idx], labels[tr_idx], group_indices[tr_idx]
        X_test, y_test = features[te_idx], labels[te_idx]

        # Re-index training groups to contiguous [0, n_train_groups)
        tr_unique_g = sorted(set(g_train.tolist()))
        g_map = {old: new for new, old in enumerate(tr_unique_g)}
        g_train_mapped = np.array([g_map[g] for g in g_train], dtype=np.int64)
        n_train_groups = len(tr_unique_g)

        # PyTorch Dataset
        ds_tr = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(g_train_mapped, dtype=torch.long),
        )
        train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)

        for m_name in methods:
            torch.manual_seed(42 + fold_i)
            base_model = LinearClassifier(feat_dim, n_classes)

            if m_name == "ERM":
                trainer = ERMTrainer(base_model, epochs=args.epochs, device=args.device)
            elif m_name == "GroupDRO":
                trainer = GroupDROTrainer(base_model, n_groups=n_train_groups, epochs=args.epochs, device=args.device)
            elif m_name == "IRM":
                trainer = IRMTrainer(base_model, n_groups=n_train_groups, epochs=args.epochs, irm_lambda=10.0, device=args.device)
            elif m_name == "CORAL":
                trainer = CORALTrainer(base_model, n_groups=n_train_groups, epochs=args.epochs, coral_lambda=1.0, device=args.device)
            else:
                continue

            trainer.fit(train_loader)
            probs = trainer.predict_proba(X_test)
            preds = np.argmax(probs, axis=1)

            acc = float(np.mean(preds == y_test))
            ece = expected_calibration_error(y_test, probs)
            confidences = np.max(probs, axis=1)
            val_aurc = aurc(y_test, preds, confidences)

            logger.info(f"  {m_name:10s} -> Acc: {acc:.4f} | ECE: {ece:.4f} | AURC: {val_aurc:.4f}")

            fold_results.append({
                "method": m_name,
                "held_out_event": held_out_event,
                "accuracy": acc,
                "ece": ece,
                "aurc": val_aurc,
            })

    df_res = pd.DataFrame(fold_results)
    csv_path = out_dir / f"{args.dataset_name}_dg_benchmark_results.csv"
    df_res.to_csv(csv_path, index=False)
    logger.info(f"\nFold results saved to {csv_path}")

    # Summary table
    summary_list = []
    for m in methods:
        sub = df_res[df_res["method"] == m]
        mean_acc = sub["accuracy"].mean()
        worst_acc = sub["accuracy"].min()
        std_acc = sub["accuracy"].std()
        mean_ece = sub["ece"].mean()
        mean_aurc = sub["aurc"].mean()

        summary_list.append({
            "Method": m,
            "Mean LOEO Acc": mean_acc,
            "Worst-Event Acc": worst_acc,
            "Std Acc": std_acc,
            "Mean ECE": mean_ece,
            "Mean AURC": mean_aurc,
        })

    df_summary = pd.DataFrame(summary_list)
    print("\n" + "=" * 80)
    print("DOMAIN GENERALIZATION BENCHMARK SUMMARY (LOEO Out-of-Distribution)")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # LaTeX table
    tex_rows = []
    for _, row in df_summary.iterrows():
        tex_rows.append(
            f"{row['Method']} & {row['Mean LOEO Acc']:.3f} $\\pm$ {row['Std Acc']:.3f} & \\textbf{{{row['Worst-Event Acc']:.3f}}} & {row['Mean ECE']:.3f} & {row['Mean AURC']:.3f} \\\\"
        )

    latex_table = r"""\begin{table}[t]
\centering
\caption{Comparison of Domain Generalization Algorithms under Leave-One-Event-Out (LOEO) evaluation on """ + args.dataset_name.upper() + r""". Worst-Event Acc indicates performance on the single hardest unseen disaster.}
\label{tab:dg_benchmark_""" + args.dataset_name + r"""}
\begin{tabular}{lcccc}
\toprule
\textbf{Algorithm} & \textbf{Mean LOEO Acc} & \textbf{Worst-Event Acc} & \textbf{LOEO ECE} & \textbf{LOEO AURC} \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    with open(out_dir / f"table_dg_{args.dataset_name}.tex", "w") as fp:
        fp.write(latex_table)
    logger.info(f"LaTeX table saved to {out_dir / f'table_dg_{args.dataset_name}.tex'}")


if __name__ == "__main__":
    main()
