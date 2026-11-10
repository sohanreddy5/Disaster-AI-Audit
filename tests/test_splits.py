"""Unit tests for dataset split generators."""

import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import (
    random_split,
    leave_one_event_out,
    leave_one_disaster_type_out,
    leave_one_region_out,
)



class TestSplits(unittest.TestCase):
    def test_random_split_sizes(self):
        n_samples = 100
        labels = np.array([0, 1] * 50)
        splits = random_split(n_samples=n_samples, labels=labels, test_fraction=0.2, seed=42)
        res = splits[0]
        self.assertEqual(len(res.train_indices), 80)
        self.assertEqual(len(res.test_indices), 20)
        self.assertEqual(len(set(res.train_indices).intersection(set(res.test_indices))), 0)

    def test_loeo_disjoint_events(self):
        event_ids = ["event_A"] * 30 + ["event_B"] * 40 + ["event_C"] * 30
        folds = leave_one_event_out(event_ids)

        self.assertEqual(len(folds), 3)
        for fold in folds:
            held_out = fold.split_name.replace("loeo_", "")
            train_events = {event_ids[i] for i in fold.train_indices}
            test_events = {event_ids[i] for i in fold.test_indices}

            self.assertNotIn(held_out, train_events)
            self.assertIn(held_out, test_events)
            self.assertEqual(len(test_events), 1)
            self.assertEqual(len(train_events.intersection(test_events)), 0)

    def test_lodto_disjoint_types(self):
        disaster_types = ["flood"] * 25 + ["earthquake"] * 35 + ["wildfire"] * 40
        folds = leave_one_disaster_type_out(disaster_types)

        self.assertEqual(len(folds), 3)
        for fold in folds:
            train_types = {disaster_types[i] for i in fold.train_indices}
            test_types = {disaster_types[i] for i in fold.test_indices}
            self.assertEqual(len(train_types.intersection(test_types)), 0)



if __name__ == "__main__":
    unittest.main()

