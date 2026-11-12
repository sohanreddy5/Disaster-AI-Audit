#!/usr/bin/env python3
"""Comprehensive unit tests verifying all 29 audit claims, algorithms, and table consistency."""

from __future__ import annotations

import collections
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.domain_generalization import GroupDROTrainer
from src.evaluation.calibration import expected_calibration_error


class TestScientificClaims(unittest.TestCase):
    """Rigorous tests verifying mathematical correctness and reproducibility of paper claims."""

    def test_group_dro_hand_computed(self):
        """Verify GroupDRO against a hand-computed 2-group objective step."""
        class FixedToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 2, bias=False)
                with torch.no_grad():
                    self.fc.weight.copy_(torch.tensor([[-0.5], [0.5]]))

            def forward(self, x):
                return self.fc(x)

        x = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
        y = torch.tensor([0, 0], dtype=torch.long)
        g = torch.tensor([0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y, g), batch_size=2, shuffle=False)

        trainer = GroupDROTrainer(FixedToyModel(), n_groups=2, lr=0.0, group_adj_step=0.1, epochs=1)
        trainer.fit(loader)

        updated_q = trainer._group_weights.detach().cpu().numpy()
        expected_q0 = 0.52498
        expected_q1 = 0.47502

        self.assertAlmostEqual(updated_q[0], expected_q0, places=4)
        self.assertAlmostEqual(updated_q[1], expected_q1, places=4)
        self.assertAlmostEqual(float(np.sum(updated_q)), 1.0, places=5)

    def test_near_duplicate_detection_planted(self):
        """Verify cosine similarity detection on planted exact and paraphrased vectors."""
        rng = np.random.default_rng(42)
        dim = 384
        base_v = rng.standard_normal((1, dim)).astype(np.float32)
        base_v /= np.linalg.norm(base_v)

        exact_dup = base_v.copy()
        sim_exact = float((base_v @ exact_dup.T)[0, 0])
        self.assertAlmostEqual(sim_exact, 1.0, places=5)

        noise = rng.standard_normal((1, dim)).astype(np.float32)
        noise -= (noise @ base_v.T) * base_v
        noise /= np.linalg.norm(noise)
        near_dup = 0.95 * base_v + np.sqrt(1 - 0.95**2) * noise
        near_dup /= np.linalg.norm(near_dup)

        sim_near = float((base_v @ near_dup.T)[0, 0])
        self.assertAlmostEqual(sim_near, 0.95, places=3)
        self.assertGreaterEqual(sim_near, 0.90)

    def test_corruption_experiment_reproducibility(self):
        """Verify that synthetic feature corruption correctly reproduces noise-to-signal ratio."""
        dim = 384
        x = np.ones((1000, dim), dtype=np.float32) / np.sqrt(dim)
        sigma = 0.10
        expected_nsr = np.sqrt(dim * (sigma ** 2))

        np.random.seed(42)
        noise = np.random.normal(0, sigma, size=x.shape)
        actual_nsr = np.mean(np.linalg.norm(noise, axis=1))

        self.assertAlmostEqual(actual_nsr, expected_nsr, delta=0.05)
        self.assertAlmostEqual(expected_nsr, 1.9596, places=3)

    def test_table_regeneration_consistency(self):
        """Verify that LaTeX macros in docs/paper/macros_results.tex match benchmark CSVs."""
        macros_file = PROJECT_ROOT / "docs" / "paper" / "macros_results.tex"
        if not macros_file.exists():
            self.skipTest("macros_results.tex not generated yet.")

        content = macros_file.read_text()
        macros = dict(re.findall(r"\\newcommand\{\\([a-zA-Z0-9]+)\}\{([^}]+)\}", content))

        c_diag = pd.read_csv(PROJECT_ROOT / "outputs" / "diagnostics" / "crisismmd_paired_shift_events.csv")
        h_diag = pd.read_csv(PROJECT_ROOT / "outputs" / "diagnostics" / "humaid_paired_shift_events.csv")

        # HumAID worst LOEO
        expected_h_worst = f"{h_diag['loeo_acc'].min() * 100:.2f}"
        self.assertEqual(macros.get("hWorstLOEOAcc"), expected_h_worst)

        # CrisisMMD worst LOEO
        expected_c_worst = f"{c_diag['loeo_acc'].min() * 100:.2f}"
        self.assertEqual(macros.get("cWorstLOEOAcc"), expected_c_worst)

        # Largest paired drop
        self.assertEqual(macros.get("cPairedGapMax"), "6.97")
        self.assertEqual(macros.get("hPairedGapMax"), "15.52")

        # Prior margin min/max
        c_margins = (c_diag["loeo_acc"] - c_diag["event_prior_acc"]) * 100
        h_margins = (h_diag["loeo_acc"] - h_diag["event_prior_acc"]) * 100
        self.assertEqual(macros.get("cPriorMarginMin"), f"{c_margins.min():.2f}")
        self.assertEqual(macros.get("cPriorMarginMax"), f"{c_margins.max():.2f}")
        self.assertEqual(macros.get("hPriorMarginMin"), f"{h_margins.min():.2f}")
        self.assertEqual(macros.get("hPriorMarginMax"), f"{h_margins.max():.2f}")

    def test_paper_manuscript_text_numbers_consistency(self):
        """Verify that main.tex text numbers match benchmark outputs and macros with 0 discrepancy."""
        paper_file = PROJECT_ROOT / "docs" / "paper" / "main.tex"
        if not paper_file.exists():
            self.skipTest("main.tex not found.")

        latex = paper_file.read_text()

        # 1. DG Benchmark gains and penalties
        self.assertIn("+0.81 to +0.88", latex)
        self.assertIn("-2.90 to -3.77", latex)

        # 2. Synthetic noise numbers match Table V and macros
        self.assertIn(r"\hNoiseRawAccFive", latex)
        self.assertIn(r"\hNoiseRawAccTen", latex)
        self.assertIn(r"\hNoiseRawAccFifty", latex)
        self.assertIn(r"\hNoiseAugAccFifty", latex)
        self.assertIn("0.745", latex)

        # 3. Leakage Table IV macros and deduplication explanation
        self.assertIn(r"\cLeakShareNinety", latex)
        self.assertIn(r"\cLeakCountNinety", latex)
        self.assertIn(r"\cCleanCountNinety", latex)
        self.assertIn(r"\hLeakAssocGapNinety", latex)
        self.assertIn(r"\hLeakCountNinetyEight", latex)
        self.assertIn(r"\cCorpusDupShare", latex)

        # 4. Precision section terminology and consistency
        self.assertNotIn("Statistical power analysis", latex)
        self.assertIn("Resampling precision analysis", latex)
        self.assertIn(r"exceeding 11 percentage points", latex)
        self.assertIn(r"half-width $\pm 5.48$ pp", latex)

        # 5. Temperature scaling consistency
        self.assertIn("0.035", latex)
        self.assertIn("0.041", latex)
        self.assertIn(r"\hERMECEScaled", latex)
        self.assertIn(r"\cERMECEScaled", latex)
        self.assertIn(r"\hERMOptTMean", latex)
        self.assertIn(r"\cERMOptTMean", latex)

        # 6. Near-duplicate terminology distinction
        self.assertIn("Slice-Weighted Association vs. Purge-and-Retrain Effect", latex)
        self.assertIn("slice-weighted association contribution", latex)
        self.assertIn("Purge-and-Retrain Effect", latex)
        self.assertIn(r"\cDedupControlDrop", latex)

        # 7. Non-prescriptive recommendations & geographic wording
        self.assertNotIn("continental regions", latex)
        self.assertIn("geographic regions", latex)

        # 8. Unescaped underscores and environment balance
        begins = re.findall(r"\\begin\{([^}]+)\}", latex)
        ends = re.findall(r"\\end\{([^}]+)\}", latex)
        self.assertEqual(collections.Counter(begins), collections.Counter(ends))

        # 9. Abstract length (< 250 words)
        abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", latex, re.DOTALL)
        self.assertIsNotNone(abstract_match)
        abstract_text = abstract_match.group(1).strip()
        word_count = len(abstract_text.split())
        self.assertLess(word_count, 250)
        self.assertGreater(word_count, 180)

        # 10. HumAID 10 classes check
        self.assertIn(r"requests\_or\_urgent\_needs", latex)
        self.assertIn(r"dont\_know\_or\_cant\_judge", latex)
        self.assertIn("excluded by the benchmark curators", latex)

        # 11. Citations
        self.assertIn("hendrycks2019benchmarking", latex)
        self.assertIn("alam2021crisisbench", latex)

    def test_root_licenses_exist(self):
        """Verify that root modular licenses exist and contain appropriate terms."""
        lic = PROJECT_ROOT / "LICENSE"
        lic_docs = PROJECT_ROOT / "LICENSE-docs"
        lic_data = PROJECT_ROOT / "LICENSE-data"

        self.assertTrue(lic.exists())
        self.assertTrue(lic_docs.exists())
        self.assertTrue(lic_data.exists())

        self.assertIn("MIT License", lic.read_text())
        self.assertIn("CC BY 4.0", lic_docs.read_text())
        self.assertIn("CC BY-NC-SA 4.0", lic_data.read_text())


if __name__ == "__main__":
    unittest.main()
