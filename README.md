# Beyond the Random Split: An Empirical Audit and Out-of-Distribution Benchmark of Multimodal Disaster AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A standardized benchmark, empirical audit suite, and evaluation protocol for out-of-distribution (OOD) generalization in multimodal disaster AI across **94,566 samples and 26 real-world global disaster events**.

---

## 🌟 Overview & Key Discoveries

Most published machine learning models for disaster triage and damage assessment are evaluated using **uniform random train/test splits**. Because crisis data is clustered by events, random splits allow near-duplicate images and tweets from the *same crisis* to leak into test sets.

This audit provides empirical proof of four major systemic failures:
1. **Severe Tail Collapse (15.8 to 21.2 pp Drops):** Random splits conceal catastrophic out-of-distribution failure, with accuracy on the worst-case unseen disaster dropping to **39.33%** on CrisisMMD and **51.48%** on HumAID (with the hardest single-event ERM fold reaching **47.71%**).
2. **Train/Test Leakage Advantage (+5.9 to +15.5 pp):** Near-duplicate samples crossing random train/test partitions enjoy an artificial memorization boost of up to **+15.46 pp** ($\tau \ge 0.90$) and **+15.09 pp** ($\tau \ge 0.98$).
3. **Synthetic Feature Corruption & Calibration Crash:** Simulated feature perturbations ($\sigma=0.10$ to $0.50$) cause accuracy to plummet from **71.69% to 17.19%** and ECE to surge from **0.039 to 0.745** (collapsing to **13.91%** accuracy, **0.797** ECE, and **0.852** AURC under extreme joint degradation with $\sigma=0.75, \delta=0.50$).
4. **Empirical Precision Bound ($k \ge 15$):** Holding out fewer than 15 disaster events results in an empirical 95% confidence interval width exceeding $\pm 10$ percentage points, demonstrating the necessity of large multi-event benchmarks.

---

## 📊 Summary of Benchmark Results

### 1. Split Protocol Comparison
| Split Protocol | CrisisMMD (18,082 samples) | HumAID (76,484 samples) | Finding |
| :--- | :---: | :---: | :--- |
| **Random Split** | 55.12% $\pm$ 0.4% | 72.69% $\pm$ 0.2% | Standard baseline (artificially inflated) |
| **Region-Held-Out (LORO)** | 52.74% $\pm$ 3.8% | 73.41% $\pm$ 4.5% | Geographic distribution shift |
| **Event-Held-Out (LOEO)** | 52.08% $\pm$ 12.9% | 71.69% $\pm$ 8.9% | Operational deployment setting |
| **Worst-Event (LOEO Tail)** | **39.33% (-15.8 pp)** | **51.48% (-21.2 pp)** | Catastrophic failure on hard disasters |
| **Event-Prior Anchor** | **38.67%** | **32.21%** | Zero-feature baseline exploiting label priors |

### 2. Domain Generalization Algorithms (26 LOEO Folds)
| Algorithm | CrisisMMD Mean LOEO | CrisisMMD Worst Event | HumAID Mean LOEO | HumAID Worst Event | HumAID Mean ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ERM** | **50.51%** | 37.35% | **67.81%** | 47.71% | 0.1609 |
| **GroupDRO** | 50.26% | **38.96% (+1.61 pp)** | 67.14% | **48.11% (+0.40 pp)** | 0.1952 |
| **IRM** | 49.45% | 37.02% | 64.14% | 47.30% | **0.0816 (-49% ECE)** |
| **CORAL** | 50.51% | 37.35% | 67.81% | 47.71% | 0.1609 |

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/sohanreddy5/Disaster-AI-Audit.git
cd Disaster-AI-Audit

# Create environment and install dependencies
uv venv .venv --python python3.11
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce All Experiments

```bash
# 1. Run Unit Tests (Split integrity, zero-leakage, calibration, and models)
python tests/test_splits.py
python tests/test_full_suite.py

# 2. Run Comprehensive Multi-Split Benchmark (Random, LORO, LOEO, LODTO)
python scripts/run_benchmark.py --dataset crisismmd
python scripts/run_benchmark.py --dataset humaid

# 3. Run Near-Duplicate Leakage Impact Experiment
python scripts/run_leakage_impact.py --features-npz cache/features/crisismmd_humanitarian_text_features.npz --dataset-name crisismmd
python scripts/run_leakage_impact.py --features-npz cache/features/humaid_benchmark_features.npz --dataset-name humaid

# 4. Run Domain Generalization Benchmark (ERM vs GroupDRO vs IRM vs CORAL)
python scripts/run_domain_generalization.py --features-npz cache/features/crisismmd_humanitarian_text_features.npz --dataset-name crisismmd
python scripts/run_domain_generalization.py --features-npz cache/features/humaid_benchmark_features.npz --dataset-name humaid

# 5. Run Compound Shift & Sensor Degradation Stress Test
python scripts/run_compound_shift.py --features-npz cache/features/crisismmd_humanitarian_text_features.npz --dataset-name crisismmd
python scripts/run_compound_shift.py --features-npz cache/features/humaid_benchmark_features.npz --dataset-name humaid

# 6. Run Empirical Resampling and Precision Analysis
python scripts/run_power_analysis.py --results-csv outputs/benchmark/humaid_benchmark_results.csv

# 7. Generate All Publication Figures (PDF & 300 DPI PNG)
python scripts/generate_paper_figures.py

# 8. Export Zenodo Manifests
python scripts/export_benchmark_manifest.py
```

---

## 📁 Repository Structure

```
disaster-ai-audit/
├── src/
│   ├── data/                 # Dataset loaders (CrisisMMD, HumAID) and split engines
│   ├── features/             # OpenCLIP, SigLIP, and MiniLM feature extractors
│   ├── models/               # Linear/MLP probes, EventPrior, GroupDRO, IRM, CORAL
│   ├── evaluation/           # Calibration (ECE), Abstention (AURC), Cluster Bootstrap
│   └── audit/                # Duplicate detection (pHash/Dense) and Cleanlab label noise
├── scripts/                  # Standalone executable benchmark and audit scripts
├── outputs/
│   ├── benchmark/            # Multi-split CSV results and LaTeX tables
│   ├── leakage_impact/       # Leakage inflation CSVs and LaTeX tables
│   ├── dg_benchmark/         # Domain generalization 26-fold results and tables
│   ├── compound_shift/       # Compound shift CSVs and LaTeX tables
│   ├── power_analysis/       # Subsampling power curve and CSVs
│   └── figures/              # Publication figures (PDF and high-res PNG)
├── docs/
│   └── paper/                # Full academic paper manuscript (LaTeX main.tex and Markdown)
├── data/
│   └── splits_manifest/      # Zenodo-ready split index manifests (JSON)
├── tests/                    # Unittest suite for split verification
└── requirements.txt          # Python dependencies
```

---

## 📜 Citation

If you use this benchmark, split manifests, or evaluation harness, please cite:

```bibtex
@article{mandala2026beyond,
  title={Beyond the Random Split: An Empirical Audit and Out-of-Distribution Benchmark of Multimodal Disaster AI},
  author={Mandala, Sohan Reddy},
  journal={IEEE Transactions on Geoscience and Remote Sensing (Under Review)},
  year={2026}
}
```
