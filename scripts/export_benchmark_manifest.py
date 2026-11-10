#!/usr/bin/env python3
"""Export standardized split manifests and benchmark index files for Zenodo release.

Saves:
- data/splits_manifest/crisismmd_splits.json
- data/splits_manifest/humaid_splits.json
- data/splits_manifest/leakage_masks.json
- data/splits_manifest/dataset_card.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import (
    random_split,
    leave_one_event_out,
    disaster_type_held_out,
    region_held_out,
    compound_shift_split,
)

MANIFEST_DIR = PROJECT_ROOT / "data" / "splits_manifest"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def export_dataset_splits(npz_path: Path, dataset_name: str) -> None:
    data = np.load(npz_path)
    features = data["features"]
    labels = data["labels"]
    event_ids = data["event_ids"].tolist()
    disaster_types = data["disaster_types"].tolist()
    regions = data["regions"].tolist()
    n_samples = len(labels)

    manifest = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_classes": len(set(labels)),
        "n_events": len(set(event_ids)),
        "events": sorted(set(event_ids)),
        "disaster_types": sorted(set(disaster_types)),
        "regions": sorted(set(regions)),
        "splits": {},
    }

    # 1. Random splits (5 seeds)
    manifest["splits"]["random"] = []
    for s in range(5):
        split = random_split(n_samples=n_samples, labels=labels, test_fraction=0.2, seed=42 + s)[0]
        manifest["splits"]["random"].append({
            "seed": 42 + s,
            "train_indices": split.train_indices.tolist() if isinstance(split.train_indices, np.ndarray) else split.train_indices,
            "test_indices": split.test_indices.tolist() if isinstance(split.test_indices, np.ndarray) else split.test_indices,
        })

    # 2. Leave-One-Event-Out (LOEO)
    loeo_folds = leave_one_event_out(event_ids, min_test_samples=10)
    manifest["splits"]["loeo"] = []
    for fold in loeo_folds:
        manifest["splits"]["loeo"].append({
            "held_out_event": fold.held_out_value,
            "n_train": fold.n_train,
            "n_test": fold.n_test,
            "train_indices": fold.train_indices.tolist() if isinstance(fold.train_indices, np.ndarray) else fold.train_indices,
            "test_indices": fold.test_indices.tolist() if isinstance(fold.test_indices, np.ndarray) else fold.test_indices,
        })

    # 3. Leave-One-Disaster-Type-Out (LODTO)
    lodto_folds = disaster_type_held_out(disaster_types, labels=labels, min_test_samples=10)
    manifest["splits"]["lodto"] = []
    for fold in lodto_folds:
        manifest["splits"]["lodto"].append({
            "held_out_type": fold.held_out_value,
            "n_train": fold.n_train,
            "n_test": fold.n_test,
            "train_indices": fold.train_indices.tolist() if isinstance(fold.train_indices, np.ndarray) else fold.train_indices,
            "test_indices": fold.test_indices.tolist() if isinstance(fold.test_indices, np.ndarray) else fold.test_indices,
        })

    # 4. Leave-One-Region-Out (LORO)
    loro_folds = region_held_out(regions, labels=labels, min_test_samples=10)
    manifest["splits"]["loro"] = []
    for fold in loro_folds:
        manifest["splits"]["loro"].append({
            "held_out_region": fold.held_out_value,
            "n_train": fold.n_train,
            "n_test": fold.n_test,
            "train_indices": fold.train_indices.tolist() if isinstance(fold.train_indices, np.ndarray) else fold.train_indices,
            "test_indices": fold.test_indices.tolist() if isinstance(fold.test_indices, np.ndarray) else fold.test_indices,
        })

    out_file = MANIFEST_DIR / f"{dataset_name}_splits_manifest.json"
    with open(out_file, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Exported {dataset_name} splits manifest to {out_file}")


def main() -> None:
    export_dataset_splits(
        PROJECT_ROOT / "cache" / "features" / "crisismmd_humanitarian_text_features.npz",
        "crisismmd",
    )
    export_dataset_splits(
        PROJECT_ROOT / "cache" / "features" / "humaid_benchmark_features.npz",
        "humaid",
    )
    print("Manifest exports complete!")


if __name__ == "__main__":
    main()
