#!/usr/bin/env python3
"""Comprehensive test suite verifying the entire benchmark, audit, and evaluation engine."""

import json
import unittest
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import (
    random_split,
    leave_one_event_out,
    disaster_type_held_out,
    region_held_out,
    compound_shift_split,
)
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.abstention import aurc
from src.evaluation.statistics import cluster_bootstrap_ci
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.domain_generalization import GroupDROTrainer


class TestBenchmarkEngine(unittest.TestCase):
    """Unit and regression tests for disaster AI benchmark suite."""

    def test_random_split_properties(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
        n = len(labels)
        splits = random_split(n, labels, test_fraction=0.2, seed=42)
        self.assertEqual(len(splits), 1)
        sp = splits[0]
        self.assertEqual(len(set(sp.train_indices) & set(sp.test_indices)), 0)
        self.assertEqual(len(sp.train_indices) + len(sp.test_indices), n)

    def test_leave_one_event_out(self):
        events = ["event_A"] * 20 + ["event_B"] * 30 + ["event_C"] * 25
        folds = leave_one_event_out(events, min_test_samples=10)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(len(set(fold.train_indices) & set(fold.test_indices)), 0)
            self.assertTrue(fold.held_out_value in ["event_A", "event_B", "event_C"])

    def test_compound_shift(self):
        types = ["flood"] * 20 + ["earthquake"] * 20 + ["wildfire"] * 20
        regions = ["asia"] * 20 + ["americas"] * 20 + ["europe"] * 20
        events = ["e1"] * 20 + ["e2"] * 20 + ["e3"] * 20
        splits = compound_shift_split(events, types, regions, min_test_samples=10)
        self.assertEqual(len(splits), 3)

    def test_ece_perfect_and_worst(self):
        # Perfect calibration: predicted 1.0 for true label 1
        y_true = np.array([0, 1, 0, 1])
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        ece_perf = expected_calibration_error(y_true, probs)
        self.assertAlmostEqual(ece_perf, 0.0, places=4)

        # Inverted confidence: high confidence on wrong class
        probs_inv = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        ece_worst = expected_calibration_error(y_true, probs_inv)
        self.assertGreater(ece_worst, 0.5)

    def test_aurc_bounds(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        conf = np.array([0.9, 0.8, 0.85, 0.95])
        val_aurc = aurc(y_true, y_pred, conf)
        self.assertAlmostEqual(val_aurc, 0.0, places=4)

    def test_cluster_bootstrap(self):
        per_event_scores = np.array([0.55, 0.62, 0.48, 0.70, 0.65, 0.58, 0.50])
        event_ids = np.array([f"event_{i}" for i in range(7)])
        mean_val, ci_low, ci_high = cluster_bootstrap_ci(per_event_scores, event_ids, n_boot=100)
        self.assertLessEqual(ci_low, ci_high)
        self.assertGreaterEqual(mean_val, ci_low)
        self.assertLessEqual(mean_val, ci_high)

    def test_manifest_files_exist_and_valid(self):
        manifest_dir = PROJECT_ROOT / "data" / "splits_manifest"
        for name in ["crisismmd_splits_manifest.json", "humaid_splits_manifest.json"]:
            path = manifest_dir / name
            self.assertTrue(path.exists(), f"Missing manifest {path}")
            with open(path) as f:
                data = json.load(f)
                self.assertIn("splits", data)
                self.assertIn("loeo", data["splits"])


    def test_group_dro_trainer(self):
        # 5 groups, 10 samples total, batches of size 4 (so batches have missing groups)
        x = torch.randn(20, 16)
        y = torch.randint(0, 3, (20,))
        g = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2)
        dataset = TensorDataset(x, y, g)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = nn.Linear(16, 3)
        trainer = GroupDROTrainer(model, n_groups=5, lr=0.01, epochs=2)
        trainer.fit(loader)
        preds = trainer.predict(x.numpy())
        self.assertEqual(len(preds), 20)
        self.assertAlmostEqual(trainer._group_weights.sum().item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
