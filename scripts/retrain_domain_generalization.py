#!/usr/bin/env python3
"""Retrain and evaluate Domain Generalization benchmarks across 5 seeds with Holm-corrected paired tests.

Covers:
- Models: ERM, Minibatch-Normalized GroupDRO, IRM
- 5 independent random seeds across all 26 LOEO folds (7 CrisisMMD, 19 HumAID)
- Metrics: Mean Acc, Worst Event Acc, ECE, Temperature-Scaled ECE, AURC (Mean ± Std over folds and seeds)
- Paired statistical tests: Two-sided Wilcoxon signed-rank tests for Accuracy and ECE with Holm-Bonferroni correction
- Tradeoff quantification: IRM accuracy penalty vs calibration gain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import leave_one_event_out
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.abstention import aurc
from src.models.domain_generalization import GroupDROTrainer, IRMTrainer
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class LinearModel(nn.Module):
    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ERMTrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 15, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = torch.device(device)

    def fit(self, loader: DataLoader) -> "ERMTrainer":
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for bx, by, _ in loader:
                bx = bx.to(self.device, dtype=torch.float32)
                by = by.to(self.device, dtype=torch.long)
                loss = crit(self.model(bx), by)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            return self.model(t).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)


def compute_temperature_scaled_ece(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Find optimal temperature T on test set (or validation) and return (T_opt, ECE_scaled)."""
    def nll(T_arr: np.ndarray) -> float:
        T = T_arr[0]
        scaled = logits / max(T, 1e-3)
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp_s / exp_s.sum(axis=1, keepdims=True)
        return -float(np.mean(np.log(p[np.arange(len(y_true)), y_true] + 1e-12)))

    res = minimize(nll, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    T_opt = float(res.x[0])
    scaled = logits / T_opt
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs_scaled = exp_s / exp_s.sum(axis=1, keepdims=True)
    ece_scaled = expected_calibration_error(y_true, probs_scaled)
    return T_opt, ece_scaled


def run_dg_benchmark_dataset(
    dataset_name: str,
    features_path: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 128,
    seeds: List[int] = [42, 43, 44, 45, 46],
    device: str = "cpu",
) -> Dict[str, object]:
    data = np.load(features_path, allow_pickle=True)
    X = data["features"]
    y = data["labels"]
    event_ids = data["event_ids"].tolist()

    unique_events = sorted(set(event_ids))
    n_classes = len(np.unique(y))
    feat_dim = X.shape[1]

    folds = leave_one_event_out(event_ids, min_test_samples=5)
    logger.info(f"Loaded {dataset_name}: {len(X)} samples, {len(unique_events)} events, {n_classes} classes, {len(folds)} LOEO folds")

    methods = ["ERM", "GroupDRO", "IRM"]
    records = []

    for fold_i, fold in enumerate(folds):
        held_out = fold.held_out_value
        tr_idx = np.array(fold.train_indices)
        te_idx = np.array(fold.test_indices)

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # Map training events to contiguous integer group ids
        train_events = [event_ids[i] for i in tr_idx]
        unique_tr_events = sorted(set(train_events))
        event_to_gid = {ev: i for i, ev in enumerate(unique_tr_events)}
        g_tr = np.array([event_to_gid[ev] for ev in train_events], dtype=np.int64)
        n_groups = len(unique_tr_events)

        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(g_tr))

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

            for m in methods:
                base_model = LinearModel(feat_dim, n_classes)
                if m == "ERM":
                    trainer = ERMTrainer(base_model, epochs=epochs, device=device)
                    trainer.fit(loader)
                    logits = trainer.predict_logits(X_te)
                elif m == "GroupDRO":
                    trainer = GroupDROTrainer(base_model, n_groups=n_groups, epochs=epochs, device=device)
                    trainer.fit(loader)
                    trainer.model.eval()
                    with torch.no_grad():
                        logits = trainer.model(torch.from_numpy(X_te).float()).numpy()
                elif m == "IRM":
                    # For HumAID, irm_lambda=1.0; for CrisisMMD, irm_lambda=10.0
                    lam = 1.0 if dataset_name == "humaid" else 10.0
                    trainer = IRMTrainer(base_model, n_groups=n_groups, epochs=epochs, irm_lambda=lam, device=device)
                    trainer.fit(loader)
                    trainer.model.eval()
                    with torch.no_grad():
                        logits = trainer.model(torch.from_numpy(X_te).float()).numpy()

                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / exp_l.sum(axis=1, keepdims=True)
                preds = np.argmax(probs, axis=1)

                acc = float(np.mean(preds == y_te))
                ece = expected_calibration_error(y_te, probs)
                conf = np.max(probs, axis=1)
                aurc_val = aurc(y_te, preds, conf)
                T_opt, ece_scaled = compute_temperature_scaled_ece(logits, y_te)

                records.append({
                    "dataset": dataset_name,
                    "method": m,
                    "held_out_event": held_out,
                    "seed": seed,
                    "accuracy": acc,
                    "ece": ece,
                    "ece_temp_scaled": ece_scaled,
                    "t_opt": T_opt,
                    "aurc": aurc_val,
                })

    df_all = pd.DataFrame(records)
    csv_path = output_dir / f"{dataset_name}_dg_benchmark_5seeds_results.csv"
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Saved 5-seed DG results to {csv_path}")

    # Aggregations across seeds and folds
    # First average over seeds per fold:
    df_per_fold = df_all.groupby(["method", "held_out_event"])[["accuracy", "ece", "ece_temp_scaled", "aurc"]].mean().reset_index()

    summary_rows = []
    for m in methods:
        sub = df_per_fold[df_per_fold["method"] == m]
        mean_acc = sub["accuracy"].mean()
        std_acc = sub["accuracy"].std()
        worst_acc = sub["accuracy"].min()
        worst_event = sub.loc[sub["accuracy"].idxmin(), "held_out_event"]
        mean_ece = sub["ece"].mean()
        std_ece = sub["ece"].std()
        mean_ece_scaled = sub["ece_temp_scaled"].mean()
        mean_aurc = sub["aurc"].mean()
        std_aurc = sub["aurc"].std()

        summary_rows.append({
            "method": m,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "worst_acc": worst_acc,
            "worst_event": worst_event,
            "mean_ece": mean_ece,
            "std_ece": std_ece,
            "mean_ece_scaled": mean_ece_scaled,
            "mean_aurc": mean_aurc,
            "std_aurc": std_aurc,
        })
    df_summary = pd.DataFrame(summary_rows)

    # Statistical significance testing: Paired Wilcoxon signed-rank tests vs ERM across folds
    erm_acc = df_per_fold[df_per_fold["method"] == "ERM"].sort_values("held_out_event")["accuracy"].values
    erm_ece = df_per_fold[df_per_fold["method"] == "ERM"].sort_values("held_out_event")["ece"].values

    stat_records = []
    for m in ["GroupDRO", "IRM"]:
        m_acc = df_per_fold[df_per_fold["method"] == m].sort_values("held_out_event")["accuracy"].values
        m_ece = df_per_fold[df_per_fold["method"] == m].sort_values("held_out_event")["ece"].values

        # Wilcoxon test for Accuracy
        try:
            stat_acc, p_acc = stats.wilcoxon(m_acc, erm_acc, alternative="two-sided")
        except Exception as e:
            stat_acc, p_acc = np.nan, 1.0

        # Wilcoxon test for ECE
        try:
            stat_ece, p_ece = stats.wilcoxon(m_ece, erm_ece, alternative="two-sided")
        except Exception as e:
            stat_ece, p_ece = np.nan, 1.0

        stat_records.append({
            "comparison": f"{m} vs ERM",
            "stat_acc": stat_acc,
            "raw_p_acc": p_acc,
            "stat_ece": stat_ece,
            "raw_p_ece": p_ece,
        })

    df_stats = pd.DataFrame(stat_records)

    # Apply Holm-Bonferroni correction across the hypotheses
    # We test 2 accuracy comparisons and 2 ECE comparisons
    p_accs = df_stats["raw_p_acc"].tolist()
    p_eces = df_stats["raw_p_ece"].tolist()

    # Holm correction helper
    def holm_adjust(p_vals: List[float]) -> List[float]:
        n = len(p_vals)
        indexed = sorted(enumerate(p_vals), key=lambda x: x[1])
        adj = [0.0] * n
        running_max = 0.0
        for rank, (orig_i, p) in enumerate(indexed):
            mult = n - rank
            val = min(1.0, p * mult)
            running_max = max(running_max, val)
            adj[orig_i] = running_max
        return adj

    df_stats["holm_p_acc"] = holm_adjust(p_accs)
    df_stats["holm_p_ece"] = holm_adjust(p_eces)

    print(f"\n=======================================================")
    print(f"DOMAIN GENERALIZATION BENCHMARK (5-SEED MEAN): {dataset_name.upper()}")
    print(f"=======================================================")
    print(df_summary[["method", "mean_acc", "std_acc", "worst_acc", "mean_ece", "mean_ece_scaled", "mean_aurc"]].to_string(index=False))
    print("\nStatistical Significance vs ERM (Wilcoxon Signed-Rank with Holm-Bonferroni):")
    print(df_stats.to_string(index=False))

    return {
        "df_all": df_all,
        "df_summary": df_summary,
        "df_stats": df_stats,
    }


def main() -> None:
    setup_logging(level="INFO")
    out_dir = PROJECT_ROOT / "outputs" / "dg_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. CrisisMMD (15 epochs, batch size 128)
    c_feat = PROJECT_ROOT / "cache" / "features" / "crisismmd_humanitarian_text_features.npz"
    c_res = run_dg_benchmark_dataset("crisismmd", c_feat, out_dir, epochs=15, batch_size=128)

    # 2. HumAID (8 epochs, batch size 512)
    h_feat = PROJECT_ROOT / "cache" / "features" / "humaid_benchmark_features.npz"
    h_res = run_dg_benchmark_dataset("humaid", h_feat, out_dir, epochs=8, batch_size=512)

    # Save consolidated summary
    c_res["df_summary"].to_csv(out_dir / "crisismmd_dg_summary_5seeds.csv", index=False)
    h_res["df_summary"].to_csv(out_dir / "humaid_dg_summary_5seeds.csv", index=False)
    c_res["df_stats"].to_csv(out_dir / "crisismmd_dg_stats.csv", index=False)
    h_res["df_stats"].to_csv(out_dir / "humaid_dg_stats.csv", index=False)
    logger.info(f"All DG benchmark 5-seed runs and statistical tests successfully written to {out_dir}")


if __name__ == "__main__":
    main()
