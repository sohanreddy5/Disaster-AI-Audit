#!/usr/bin/env python3
"""Go/No-Go Experiment: Week 1 Quick Validation.

Tests whether LOEO (Leave-One-Event-Out) reveals a meaningful accuracy gap
compared to random splits on CrisisMMD, using frozen CLIP features and a
logistic regression probe.

Usage:
    python scripts/run_go_no_go.py                     # with real data
    python scripts/run_go_no_go.py --dummy              # with synthetic data
    python scripts/run_go_no_go.py --config configs/go_no_go.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import kendalltau

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logging, get_logger
from src.utils.config import load_config
from src.data.splits import random_split, leave_one_event_out, SplitResult


logger = get_logger(__name__)


# ── Dummy data generation ─────────────────────────────────────────────────

CRISISMMD_EVENTS = [
    ("california_wildfires", "wildfire", "north_america", 2018),
    ("hurricane_harvey", "hurricane", "north_america", 2017),
    ("hurricane_irma", "hurricane", "caribbean", 2017),
    ("hurricane_maria", "hurricane", "caribbean", 2017),
    ("iraq_iran_earthquake", "earthquake", "middle_east", 2017),
    ("mexico_earthquake", "earthquake", "central_america", 2017),
    ("srilanka_floods", "flood", "south_asia", 2017),
]


def generate_dummy_data(
    n_per_event: int = 100,
    feature_dim: int = 512,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Generate synthetic data mimicking CrisisMMD structure.

    Each event gets its own cluster centre so LOEO should reveal a gap.

    Parameters
    ----------
    n_per_event : int
        Number of samples per event.
    feature_dim : int
        Dimensionality of feature vectors.
    n_classes : int
        Number of classes.
    seed : int
        Random seed.

    Returns
    -------
    features : np.ndarray, shape (N, feature_dim)
    labels : np.ndarray, shape (N,)
    event_ids : list[str]
    disaster_types : list[str]
    regions : list[str]
    """
    rng = np.random.RandomState(seed)
    n_events = len(CRISISMMD_EVENTS)
    total = n_per_event * n_events

    features_list = []
    labels_list = []
    event_ids = []
    disaster_types = []
    regions = []

    for idx, (event_name, dtype, region, year) in enumerate(CRISISMMD_EVENTS):
        # Each event has a distinct cluster centre
        centre = rng.randn(feature_dim) * 0.5
        # Vary class balance per event to make LOEO harder
        class_probs = rng.dirichlet(np.ones(n_classes) * 2.0)

        for _ in range(n_per_event):
            feat = centre + rng.randn(feature_dim) * 0.3
            label = rng.choice(n_classes, p=class_probs)
            # Inject event-specific signal into features
            feat[:10] += idx * 0.2
            # Class-conditional shift
            feat[10:20] += label * 0.3

            features_list.append(feat)
            labels_list.append(label)
            event_ids.append(event_name)
            disaster_types.append(dtype)
            regions.append(region)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    # Normalise features (CLIP-like)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)

    return features, labels, event_ids, disaster_types, regions


# ── Real data loading (CLIP features) ─────────────────────────────────────

def load_real_data(
    data_root: str | Path,
    cache_dir: str | Path,
    task: str = "humanitarian",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Load CrisisMMD and extract text features.

    Parameters
    ----------
    data_root : str | Path
        Root of CrisisMMD dataset.
    cache_dir : str | Path
        Directory for caching extracted features.
    task : str
        'informative' or 'humanitarian'.
    model_name : str
        Model variant.
    pretrained : str
        Pretrained weights source.

    Returns
    -------
    Same as generate_dummy_data.
    """
    from src.data.crisismmd import CrisisMMDDataset
    from src.features.text_extractor import TextFeatureExtractor

    dataset = CrisisMMDDataset(root=data_root, label_task=task)
    logger.info(f"Loaded CrisisMMD ({task}): {len(dataset)} samples, {len(dataset.event_ids)} events")

    # Extract or load cached embeddings
    cache_path = Path(cache_dir) / f"crisismmd_{task}_text_features.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        features = data["features"]
        labels = data["labels"]
        event_ids = data["event_ids"].tolist()
        disaster_types = data["disaster_types"].tolist()
        regions = data["regions"].tolist()
    else:
        logger.info(f"Extracting sentence embeddings for {len(dataset)} tweets ({task})...")
        extractor = TextFeatureExtractor(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir,
            batch_size=128,
        )
        texts = [s["text"] for s in dataset.samples]
        features = extractor.extract_features(texts, dataset_name=f"crisismmd_{task}", show_progress=True)

        raw_labels = [s["label"] for s in dataset.samples]
        unique_labels = sorted(set(raw_labels))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        logger.info(f"Label map ({len(unique_labels)} classes): {label_map}")

        labels = np.array([label_map[s["label"]] for s in dataset.samples], dtype=np.int64)
        event_ids = [s["event_id"] for s in dataset.samples]
        disaster_types = [s["disaster_type"] for s in dataset.samples]
        regions = [s["region"] for s in dataset.samples]

        np.savez_compressed(
            cache_path,
            features=features,
            labels=labels,
            event_ids=np.array(event_ids),
            disaster_types=np.array(disaster_types),
            regions=np.array(regions),
        )
        logger.info(f"Cached extracted features to {cache_path}")

    return features, labels, event_ids, disaster_types, regions




# ── Evaluation helpers ────────────────────────────────────────────────────

def evaluate_split(
    features: np.ndarray,
    labels: np.ndarray,
    split: SplitResult,
    model_name: str = "logistic",
    C: float = 1.0,
) -> dict:
    """Train a probe on one split and return metrics.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (N, D).
    labels : np.ndarray
        Label array (N,).
    split : SplitResult
        Train/test split.
    model_name : str
        'logistic' or 'mlp'.
    C : float
        Regularisation for logistic regression.

    Returns
    -------
    dict with accuracy, macro_f1, balanced_accuracy, model_name, split_name
    """
    X_train = features[split.train_indices]
    y_train = labels[split.train_indices]
    X_test = features[split.test_indices]
    y_test = labels[split.test_indices]

    if model_name == "logistic":
        clf = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs"
        )

    elif model_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "model": model_name,
        "split_name": split.split_name,
        "held_out_event": split.held_out_value,
        "n_train": split.n_train,
        "n_test": split.n_test,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    }


# ── Main go/no-go experiment ─────────────────────────────────────────────

def run_go_no_go(
    features: np.ndarray,
    labels: np.ndarray,
    event_ids: list[str],
    disaster_types: list[str],
    regions: list[str],
    n_random_splits: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the go/no-go experiment.

    Compares random-split vs LOEO performance for logistic and MLP probes.

    Parameters
    ----------
    features, labels, event_ids, disaster_types, regions
        Data from generate_dummy_data or load_real_data.
    n_random_splits : int
        Number of random splits to average over.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Results table with one row per (model, split) combination.
    """
    n_samples = len(labels)
    all_results: list[dict] = []

    model_configs = [
        ("logistic", {}),
        ("mlp", {}),
    ]

    for model_name, kwargs in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        # ── Random splits ──
        logger.info(f"\n--- Random Splits ({n_random_splits}x) ---")
        random_splits = random_split(
            n_samples, labels,
            test_fraction=0.2,
            n_splits=n_random_splits,
            seed=seed,
        )

        random_results = []
        for split in random_splits:
            result = evaluate_split(features, labels, split, model_name)
            random_results.append(result)
            logger.info(
                f"  {split.split_name}: "
                f"acc={result['accuracy']:.4f}, "
                f"f1={result['macro_f1']:.4f}"
            )

        # Average random performance
        avg_random = {
            "model": model_name,
            "split_name": "random_avg",
            "held_out_event": None,
            "n_train": int(np.mean([r["n_train"] for r in random_results])),
            "n_test": int(np.mean([r["n_test"] for r in random_results])),
            "accuracy": np.mean([r["accuracy"] for r in random_results]),
            "macro_f1": np.mean([r["macro_f1"] for r in random_results]),
            "balanced_accuracy": np.mean(
                [r["balanced_accuracy"] for r in random_results]
            ),
        }
        all_results.append(avg_random)
        all_results.extend(random_results)

        # ── LOEO splits ──
        logger.info(f"\n--- LOEO Splits ---")
        loeo_splits = leave_one_event_out(event_ids, labels, min_test_samples=5)
        logger.info(f"  {len(loeo_splits)} events held out")

        loeo_results = []
        for split in loeo_splits:
            result = evaluate_split(features, labels, split, model_name)
            loeo_results.append(result)
            logger.info(
                f"  {split.split_name}: "
                f"acc={result['accuracy']:.4f}, "
                f"f1={result['macro_f1']:.4f}, "
                f"n_test={result['n_test']}"
            )

        # Average LOEO performance
        avg_loeo = {
            "model": model_name,
            "split_name": "loeo_avg",
            "held_out_event": None,
            "n_train": int(np.mean([r["n_train"] for r in loeo_results])),
            "n_test": int(np.mean([r["n_test"] for r in loeo_results])),
            "accuracy": np.mean([r["accuracy"] for r in loeo_results]),
            "macro_f1": np.mean([r["macro_f1"] for r in loeo_results]),
            "balanced_accuracy": np.mean(
                [r["balanced_accuracy"] for r in loeo_results]
            ),
        }
        all_results.append(avg_loeo)
        all_results.extend(loeo_results)

    return pd.DataFrame(all_results)


def print_go_no_go_summary(results_df: pd.DataFrame) -> None:
    """Print a formatted summary of the go/no-go results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from :func:`run_go_no_go`.
    """
    print("\n" + "=" * 70)
    print("GO / NO-GO SUMMARY")
    print("=" * 70)

    models = results_df["model"].unique()

    # Collect for rank analysis
    model_random_accs = {}
    model_loeo_accs = {}

    for model in models:
        model_df = results_df[results_df["model"] == model]
        random_row = model_df[model_df["split_name"] == "random_avg"]
        loeo_row = model_df[model_df["split_name"] == "loeo_avg"]

        if random_row.empty or loeo_row.empty:
            continue

        random_acc = random_row["accuracy"].values[0]
        loeo_acc = loeo_row["accuracy"].values[0]
        gap = random_acc - loeo_acc

        random_f1 = random_row["macro_f1"].values[0]
        loeo_f1 = loeo_row["macro_f1"].values[0]

        model_random_accs[model] = random_acc
        model_loeo_accs[model] = loeo_acc

        print(f"\n  Model: {model}")
        print(f"  {'Metric':<20} {'Random':<12} {'LOEO':<12} {'Gap':<12}")
        print(f"  {'-'*56}")
        print(f"  {'Accuracy':<20} {random_acc:<12.4f} {loeo_acc:<12.4f} {gap:<12.4f}")
        print(f"  {'Macro F1':<20} {random_f1:<12.4f} {loeo_f1:<12.4f} {random_f1-loeo_f1:<12.4f}")

        # Per-event breakdown
        loeo_events = model_df[
            model_df["split_name"].str.startswith("loeo_")
            & (model_df["split_name"] != "loeo_avg")
        ]
        if not loeo_events.empty:
            print(f"\n  Per-event LOEO accuracy:")
            for _, row in loeo_events.iterrows():
                bar = "█" * int(row["accuracy"] * 40)
                print(f"    {row['held_out_event']:<30} {row['accuracy']:.4f} {bar}")

            std = loeo_events["accuracy"].std()
            min_acc = loeo_events["accuracy"].min()
            max_acc = loeo_events["accuracy"].max()
            print(f"\n    Std:  {std:.4f}")
            print(f"    Min:  {min_acc:.4f}  ({loeo_events.loc[loeo_events['accuracy'].idxmin(), 'held_out_event']})")
            print(f"    Max:  {max_acc:.4f}  ({loeo_events.loc[loeo_events['accuracy'].idxmax(), 'held_out_event']})")
            print(f"    Range: {max_acc - min_acc:.4f}")

    # Rank analysis
    if len(model_random_accs) > 1:
        print(f"\n{'='*70}")
        print("RANK ANALYSIS")
        print(f"{'='*70}")

        # Compute ranks (higher accuracy = rank 1)
        random_ranked = sorted(model_random_accs.items(), key=lambda x: -x[1])
        loeo_ranked = sorted(model_loeo_accs.items(), key=lambda x: -x[1])

        random_ranks = {name: i + 1 for i, (name, _) in enumerate(random_ranked)}
        loeo_ranks = {name: i + 1 for i, (name, _) in enumerate(loeo_ranked)}

        print(f"\n  {'Model':<20} {'Random Rank':<15} {'LOEO Rank':<15} {'Change':<10}")
        print(f"  {'-'*60}")
        for model in models:
            if model in random_ranks:
                rr = random_ranks[model]
                lr = loeo_ranks[model]
                change = rr - lr
                arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
                print(f"  {model:<20} {rr:<15} {lr:<15} {arrow} {abs(change)}")

        # Kendall's tau
        if len(models) >= 3:
            r_ranks = [random_ranks[m] for m in models if m in random_ranks]
            l_ranks = [loeo_ranks[m] for m in models if m in loeo_ranks]
            tau, p_value = kendalltau(r_ranks, l_ranks)
            print(f"\n  Kendall's τ = {tau:.4f} (p = {p_value:.4f})")
            if tau < 0.7:
                print("  ⚠️  Rankings are UNSTABLE across split strategies!")
            else:
                print("  ✓  Rankings are relatively stable.")

    # Go/No-Go decision
    print(f"\n{'='*70}")
    print("DECISION")
    print(f"{'='*70}")

    any_gap = False
    for model in models:
        model_df = results_df[results_df["model"] == model]
        random_row = model_df[model_df["split_name"] == "random_avg"]
        loeo_row = model_df[model_df["split_name"] == "loeo_avg"]
        if not random_row.empty and not loeo_row.empty:
            gap = random_row["accuracy"].values[0] - loeo_row["accuracy"].values[0]
            if gap > 0.03:  # 3% threshold
                any_gap = True

    if any_gap:
        print("\n  ✅ GO: Meaningful accuracy gap detected between Random and LOEO.")
        print("  The event-held-out evaluation reveals generalization issues")
        print("  that random splits miss. Proceed with full benchmark.\n")
    else:
        print("\n  ⚠️  MARGINAL: Gap is small. Consider:")
        print("  - More events / datasets")
        print("  - Compound shift evaluation")
        print("  - Check if task is too easy for CLIP features\n")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Go/No-Go: Random vs LOEO split comparison on CrisisMMD"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use synthetic dummy data instead of real CrisisMMD",
    )
    parser.add_argument(
        "--n-per-event",
        type=int,
        default=200,
        help="Samples per event for dummy data (default: 200)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="humanitarian",
        choices=["informative", "humanitarian"],
        help="Label task to evaluate ('informative' or 'humanitarian')",
    )
    parser.add_argument(
        "--data-root",

        type=str,
        default="./data/crisismmd",
        help="Root directory for CrisisMMD dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/features",
        help="Feature cache directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/go_no_go",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-random-splits",
        type=int,
        default=3,
        help="Number of random splits to average over",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the go/no-go experiment."""
    args = parse_args()
    setup_logging(level="INFO")

    logger.info("=" * 60)
    logger.info("Go/No-Go Experiment: Random vs LOEO Split Comparison")
    logger.info("=" * 60)

    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    # Create output directory
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    start_time = time.time()

    # Load data
    if args.dummy:
        logger.info(f"Generating dummy data: {args.n_per_event} samples/event")
        features, labels, event_ids, disaster_types, regions = generate_dummy_data(
            n_per_event=args.n_per_event,
            seed=args.seed,
        )
        logger.info(
            f"Generated {len(labels)} samples, "
            f"{len(set(event_ids))} events, "
            f"{len(set(labels))} classes"
        )
    else:
        logger.info(f"Loading real data from {data_root}")
        try:
            features, labels, event_ids, disaster_types, regions = load_real_data(
                data_root=data_root,
                cache_dir=cache_dir,
                task=args.task,
            )

        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            logger.info("Falling back to dummy data. Use --dummy to suppress this warning.")
            features, labels, event_ids, disaster_types, regions = generate_dummy_data(
                n_per_event=args.n_per_event,
                seed=args.seed,
            )


    logger.info(f"Data shape: {features.shape}")
    logger.info(f"Events: {sorted(set(event_ids))}")
    logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Run experiment
    results_df = run_go_no_go(
        features=features,
        labels=labels,
        event_ids=event_ids,
        disaster_types=disaster_types,
        regions=regions,
        n_random_splits=args.n_random_splits,
        seed=args.seed,
    )

    elapsed = time.time() - start_time
    logger.info(f"\nExperiment completed in {elapsed:.1f}s")

    # Save results
    results_path = output_dir / "go_no_go_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    # Save as JSON too
    summary = {
        "experiment": "go_no_go",
        "dummy_data": args.dummy,
        "n_samples": len(labels),
        "n_events": len(set(event_ids)),
        "elapsed_seconds": elapsed,
        "results": results_df[
            results_df["split_name"].isin(["random_avg", "loeo_avg"])
        ].to_dict(orient="records"),
    }
    json_path = output_dir / "go_no_go_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print_go_no_go_summary(results_df)

    # Visualization (if matplotlib available)
    try:
        from src.utils.visualization import (
            plot_per_event_accuracy,
            plot_rank_changes,
        )

        # Per-event LOEO accuracy for each model
        for model_name in results_df["model"].unique():
            model_df = results_df[results_df["model"] == model_name]
            loeo_events = model_df[
                model_df["split_name"].str.startswith("loeo_")
                & (model_df["split_name"] != "loeo_avg")
            ]
            if not loeo_events.empty:
                fig = plot_per_event_accuracy(
                    event_names=loeo_events["held_out_event"].tolist(),
                    accuracies=loeo_events["accuracy"].tolist(),
                    title=f"Per-Event LOEO Accuracy ({model_name})",
                    save_path=output_dir / f"per_event_accuracy_{model_name}.png",
                )
                plt_imported = True

        logger.info(f"Plots saved to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


if __name__ == "__main__":
    main()
