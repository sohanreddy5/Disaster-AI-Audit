#!/usr/bin/env python3
"""Compute Paired Shift Gaps, Event Diagnostics, Causal Deduplication, and Event Prior Analysis.

Covers:
1. Paired shift gap per event: Acc_random(e) vs Acc_loeo(e) and Delta(e)
2. Macro-over-events vs pooled metrics for Random, LOEO, LORO, LODTO
3. Per-fold Model vs Event-Prior comparison
4. Diagnostics: Correlation of Delta(e) with N_e, C_e, and KL(P_e || P_train)
5. Causal deduplication: Retrain on deduplicated train set and evaluate on clean test set
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q) with Laplace-style smoothing."""
    p_smooth = p + eps
    p_smooth = p_smooth / p_smooth.sum()
    q_smooth = q + eps
    q_smooth = q_smooth / q_smooth.sum()
    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))


def analyze_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    events: np.ndarray,
    regions: np.ndarray,
    types: np.ndarray,
    n_seeds: int = 5,
    seed_base: int = 42,
) -> Dict[str, object]:
    unique_events = sorted(np.unique(events))
    n_classes = len(np.unique(y))
    N = len(y)

    print(f"\n==================================================")
    print(f"ANALYSIS: {dataset_name.upper()} (N={N}, {len(unique_events)} events, {n_classes} classes)")
    print(f"==================================================")

    # 1. Evaluate Random Split across n_seeds, tracking per-event accuracy on test set
    random_event_accs = {e: [] for e in unique_events}
    random_pooled_accs = []

    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_base + s)
        train_idx, test_idx = next(skf.split(np.arange(N), y))

        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        y_test = y[test_idx]
        e_test = events[test_idx]

        random_pooled_accs.append(float(np.mean(preds == y_test)))

        for e in unique_events:
            mask_e = e_test == e
            if mask_e.sum() > 0:
                acc_e = float(np.mean(preds[mask_e] == y_test[mask_e]))
                random_event_accs[e].append(acc_e)

    mean_random_per_event = {e: float(np.mean(random_event_accs[e])) for e in unique_events}
    std_random_per_event = {e: float(np.std(random_event_accs[e])) for e in unique_events}
    macro_random_acc = float(np.mean(list(mean_random_per_event.values())))
    pooled_random_acc = float(np.mean(random_pooled_accs))

    # 2. Evaluate LOEO per event and Event Prior per event
    loeo_results = {}
    event_prior_results = {}
    loeo_preds_all = np.zeros(N, dtype=np.int64)

    for e in unique_events:
        test_mask = events == e
        train_mask = ~test_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        # Fit model
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        preds_loeo = clf.predict(X_te)
        acc_loeo = float(np.mean(preds_loeo == y_te))
        loeo_results[e] = acc_loeo
        loeo_preds_all[test_mask] = preds_loeo

        # Event prior: training event prior mixture
        # Compute mean class distribution across training events
        train_events = events[train_mask]
        p_c_list = []
        for tr_e in np.unique(train_events):
            m_tre = train_events == tr_e
            counts = np.bincount(y_tr[m_tre], minlength=n_classes)
            p_c_list.append(counts / counts.sum())
        avg_prior = np.mean(p_c_list, axis=0)
        pred_prior = int(np.argmax(avg_prior))
        acc_prior = float(np.mean(y_te == pred_prior))
        event_prior_results[e] = acc_prior

    macro_loeo_acc = float(np.mean(list(loeo_results.values())))
    pooled_loeo_acc = float(np.mean(loeo_preds_all == y))
    macro_prior_acc = float(np.mean(list(event_prior_results.values())))

    # 3. Paired shift gaps & event diagnostics
    event_records = []
    # Overall class counts across dataset
    overall_class_dist = np.bincount(y, minlength=n_classes) / N

    for e in unique_events:
        mask_e = events == e
        n_e = int(mask_e.sum())
        y_e = y[mask_e]
        classes_present = len(np.unique(y_e))
        dist_e = np.bincount(y_e, minlength=n_classes) / n_e

        # Training mix excluding e
        mask_other = ~mask_e
        dist_train = np.bincount(y[mask_other], minlength=n_classes) / mask_other.sum()

        kl_div = compute_kl_divergence(dist_e, dist_train)

        acc_rand = mean_random_per_event[e]
        acc_loeo = loeo_results[e]
        acc_prior = event_prior_results[e]
        paired_gap = acc_rand - acc_loeo
        model_vs_prior = acc_loeo - acc_prior

        event_records.append({
            "event": e,
            "n_samples": n_e,
            "classes_present": classes_present,
            "kl_divergence": kl_div,
            "random_acc": acc_rand,
            "random_acc_std": std_random_per_event[e],
            "loeo_acc": acc_loeo,
            "paired_gap": paired_gap,
            "event_prior_acc": acc_prior,
            "model_vs_prior": model_vs_prior,
        })

    df_events = pd.DataFrame(event_records)

    # Correlations between paired_gap and diagnostics
    corr_size_p, p_size_p = stats.pearsonr(df_events["paired_gap"], df_events["n_samples"])
    corr_size_s, p_size_s = stats.spearmanr(df_events["paired_gap"], df_events["n_samples"])

    corr_kl_p, p_kl_p = stats.pearsonr(df_events["paired_gap"], df_events["kl_divergence"])
    corr_kl_s, p_kl_s = stats.spearmanr(df_events["paired_gap"], df_events["kl_divergence"])

    corr_c_p, p_c_p = stats.pearsonr(df_events["paired_gap"], df_events["classes_present"])
    corr_c_s, p_c_s = stats.spearmanr(df_events["paired_gap"], df_events["classes_present"])

    diagnostics = {
        "corr_size_pearson": (corr_size_p, p_size_p),
        "corr_size_spearman": (corr_size_s, p_size_s),
        "corr_kl_pearson": (corr_kl_p, p_kl_p),
        "corr_kl_spearman": (corr_kl_s, p_kl_s),
        "corr_classes_pearson": (corr_c_p, p_c_p),
        "corr_classes_spearman": (corr_c_s, p_c_s),
    }

    print(f"Summary Metrics:")
    print(f"  Random Split: Macro={macro_random_acc*100:.2f}%, Pooled={pooled_random_acc*100:.2f}%")
    print(f"  LOEO:         Macro={macro_loeo_acc*100:.2f}%, Pooled={pooled_loeo_acc*100:.2f}%")
    print(f"  Event Prior:  Macro={macro_prior_acc*100:.2f}%")
    print(f"  Paired Shift Gap: Mean={df_events['paired_gap'].mean()*100:.2f} pp, Min={df_events['paired_gap'].min()*100:.2f} pp, Max={df_events['paired_gap'].max()*100:.2f} pp (Event: {df_events.loc[df_events['paired_gap'].idxmax(), 'event']})")
    print(f"\nCorrelations with Paired Gap (Delta):")
    print(f"  KL Divergence: r={corr_kl_p:.3f} (p={p_kl_p:.4f}), rho={corr_kl_s:.3f} (p={p_kl_s:.4f})")
    print(f"  Sample Size:   r={corr_size_p:.3f} (p={p_size_p:.4f}), rho={corr_size_s:.3f} (p={p_size_s:.4f})")
    print(f"  Classes Count: r={corr_c_p:.3f} (p={p_c_p:.4f}), rho={corr_c_s:.3f} (p={p_c_s:.4f})")

    # Check model vs prior:
    beaten = df_events[df_events["model_vs_prior"] > 0]
    unbeaten = df_events[df_events["model_vs_prior"] <= 0]
    print(f"  Model beats Event Prior on {len(beaten)}/{len(df_events)} events.")
    if len(unbeaten) > 0:
        print(f"  Events where model fails to beat Event Prior:")
        for _, r in unbeaten.iterrows():
            print(f"    - {r['event']}: Model {r['loeo_acc']*100:.2f}% vs Prior {r['event_prior_acc']*100:.2f}% (Diff: {r['model_vs_prior']*100:.2f} pp)")

    return {
        "dataset_name": dataset_name,
        "macro_random_acc": macro_random_acc,
        "pooled_random_acc": pooled_random_acc,
        "macro_loeo_acc": macro_loeo_acc,
        "pooled_loeo_acc": pooled_loeo_acc,
        "macro_prior_acc": macro_prior_acc,
        "df_events": df_events,
        "diagnostics": diagnostics,
    }


def causal_deduplication_audit(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    events: np.ndarray,
    seed: int = 42,
    tau_thresholds: Tuple[float, ...] = (0.90, 0.95, 0.98),
) -> Dict[str, object]:
    """Causal test of deduplication on random split:
    Train on standard train set vs train set with near-duplicates purged.
    Evaluate on clean test instances and overall test instances.
    """
    N = len(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(skf.split(np.arange(N), y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Batch cosine similarity: X is L2-normalized so cos(u, v) = u @ v.T
    # Compute max cosine similarity for test samples against train
    # Process in chunks to maintain low memory
    chunk_size = 2000
    max_sims = np.zeros(len(test_idx), dtype=np.float32)
    for i in range(0, len(test_idx), chunk_size):
        chunk = X_test[i : i + chunk_size]
        sims = chunk @ X_train.T
        max_sims[i : i + chunk_size] = np.max(sims, axis=1)

    # Baseline standard model
    clf_std = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    clf_std.fit(X_train, y_train)
    preds_std = clf_std.predict(X_test)
    acc_std_overall = float(np.mean(preds_std == y_test))

    results = []
    for tau in tau_thresholds:
        leaked_mask = max_sims >= tau
        clean_mask = ~leaked_mask
        n_leaked = int(leaked_mask.sum())
        n_clean = int(clean_mask.sum())
        leak_share = n_leaked / len(test_idx)

        acc_leaked = float(np.mean(preds_std[leaked_mask] == y_test[leaked_mask])) if n_leaked > 0 else 0.0
        acc_clean = float(np.mean(preds_std[clean_mask] == y_test[clean_mask]))
        assoc_gap = acc_leaked - acc_clean

        # Net effect of leakage on overall test accuracy:
        # acc_overall = leak_share * acc_leaked + (1 - leak_share) * acc_clean
        # net effect = acc_overall - acc_clean = leak_share * (acc_leaked - acc_clean)
        net_effect = leak_share * assoc_gap

        # Causal experiment: Find train instances that are near-duplicates of the training set itself
        # Or remove train instances that leak into test (purged training)
        # Purged train: remove train samples that have sim >= tau with ANY test sample
        # We can find which train samples leak:
        leaked_train_mask = np.zeros(len(train_idx), dtype=bool)
        for i in range(0, len(train_idx), chunk_size):
            chunk = X_train[i : i + chunk_size]
            sims = chunk @ X_test.T
            leaked_train_mask[i : i + chunk_size] = np.max(sims, axis=1) >= tau

        X_train_purged = X_train[~leaked_train_mask]
        y_train_purged = y_train[~leaked_train_mask]

        if len(np.unique(y_train_purged)) == len(np.unique(y_train)) and len(y_train_purged) > 100:
            clf_purged = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            clf_purged.fit(X_train_purged, y_train_purged)
            preds_purged = clf_purged.predict(X_test)
            acc_purged_overall = float(np.mean(preds_purged == y_test))
            acc_purged_clean = float(np.mean(preds_purged[clean_mask] == y_test[clean_mask]))
            causal_drop = acc_std_overall - acc_purged_overall
        else:
            acc_purged_overall = float("nan")
            acc_purged_clean = float("nan")
            causal_drop = float("nan")

        results.append({
            "tau": tau,
            "leak_share": leak_share,
            "n_leaked": n_leaked,
            "n_clean": n_clean,
            "acc_leaked": acc_leaked,
            "acc_clean": acc_clean,
            "assoc_gap": assoc_gap,
            "net_effect": net_effect,
            "acc_std_overall": acc_std_overall,
            "acc_purged_overall": acc_purged_overall,
            "causal_drop": causal_drop,
        })

    df_leak = pd.DataFrame(results)
    print(f"\nDeduplication & Leakage Audit for {dataset_name.upper()}:")
    print(df_leak[["tau", "leak_share", "n_leaked", "acc_leaked", "acc_clean", "assoc_gap", "net_effect", "causal_drop"]].to_string(index=False))
    return {"df_leak": df_leak}


def main() -> None:
    # 1. CrisisMMD
    c_data = np.load(PROJECT_ROOT / "cache" / "features" / "crisismmd_humanitarian_text_features.npz", allow_pickle=True)
    c_res = analyze_dataset("crisismmd", c_data["features"], c_data["labels"], c_data["event_ids"], c_data["regions"], c_data["disaster_types"])
    c_leak = causal_deduplication_audit("crisismmd", c_data["features"], c_data["labels"], c_data["event_ids"])

    # 2. HumAID
    h_data = np.load(PROJECT_ROOT / "cache" / "features" / "humaid_benchmark_features.npz", allow_pickle=True)
    h_res = analyze_dataset("humaid", h_data["features"], h_data["labels"], h_data["event_ids"], h_data["regions"], h_data["disaster_types"])
    h_leak = causal_deduplication_audit("humaid", h_data["features"], h_data["labels"], h_data["event_ids"])

    # Save diagnostics and paired gaps
    out_dir = PROJECT_ROOT / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    c_res["df_events"].to_csv(out_dir / "crisismmd_paired_shift_events.csv", index=False)
    h_res["df_events"].to_csv(out_dir / "humaid_paired_shift_events.csv", index=False)
    c_leak["df_leak"].to_csv(out_dir / "crisismmd_causal_dedup.csv", index=False)
    h_leak["df_leak"].to_csv(out_dir / "humaid_causal_dedup.csv", index=False)
    print(f"\nAll paired shift, event diagnostics, and causal dedup saved to {out_dir}")

if __name__ == "__main__":
    main()
