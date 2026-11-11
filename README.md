# Beyond the Random Split: An Empirical Audit and Out-of-Distribution Benchmark of Text-Based Disaster AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A standardized benchmark, empirical audit suite, and evaluation protocol for out-of-distribution (OOD) generalization in disaster AI across **94,566 labeled observations and 26 evaluation folds (22 unique physical disasters)**.

---

## 🌟 Overview & Key Findings

Most published machine learning models for disaster triage and crisis classification are evaluated using **uniform random train/test splits**. Because crisis data is clustered by disaster events, random splits allow observations from every disaster incident to appear in both training and test partitions, evaluating in-distribution interpolation rather than true generalization to novel crisis incidents.

This empirical audit investigates 94,566 observations from CrisisMMD and HumAID across 26 event folds, establishing:
1. **Paired Shift Gaps on Unseen Events:** While macro LOEO accuracy drops modestly on average (2.70 percentage points on CrisisMMD and 3.24 percentage points on HumAID relative to random splits), evaluating individual target events reveals significant performance vulnerability: paired accuracy on the hardest held-out event drops by **5.54 percentage points** on CrisisMMD (California Wildfires) and **15.52 points** on HumAID (Maryland Floods) relative to event-matched random test instances.
2. **Diagnosis of Out-of-Distribution Degradation:** Class-prior divergence $D_{\text{KL}}(P_e \| P_{\text{train}})$ is strongly associated with out-of-distribution performance drop ($r = 0.563, p = 0.0121$), representing its strongest observed predictor. Furthermore, an input-free label-prior baseline achieves **38.67%** (CrisisMMD) and **32.21%** (HumAID) accuracy without consuming input features, demonstrating that random-split figures heavily reflect base rates.
3. **Near-Duplicate Leakage vs. Net Benchmark Inflation:** While test instances with training-set near-duplicates exhibit an association accuracy advantage of up to **+15.46 pp** ($\tau \ge 0.90$), their low prevalence bounds the net test-set effect to +0.01 to +0.92 pp. Causally purging near-duplicates from training data causes an overall test accuracy change of only **0.69 pp** on CrisisMMD and **0.00 pp** on HumAID.
4. **Domain Generalization & The IRM Tradeoff (5-Seed Benchmark):** Minibatch-normalized GroupDRO provides modest worst-event protection (**+0.81 to +0.88 pp**), whereas IRM incurs a statistically significant **-2.90 to -3.77 pp accuracy penalty** ($p \le 0.031$) in exchange for reduced Expected Calibration Error (ECE 0.081 vs 0.161). However, applying post-hoc temperature scaling ($T \approx 0.64$) to ERM achieves superior calibration (**ECE 0.024--0.041**) with zero accuracy loss.
5. **Synthetic Feature Corruption & Norm Inflation:** Unnormalized Gaussian perturbations scale feature norms by up to $10\times$ ($\text{NSR} = \sigma \sqrt{384} \approx 9.80$ at $\sigma=0.50$), inflating unconstrained logits and causing ECE to surge to **0.745**. Re-normalizing features preserves calibration ($\text{ECE} \le 0.052$ for $\sigma \le 0.05$).
6. **Resampling Precision Analysis ($k \ge 15$ Events):** In this benchmark, at least 15 held-out disaster events were required to achieve an empirical 95% confidence interval width below 11 percentage points (a half-width of $\pm 5.5$ percentage points). Small-scale benchmarks evaluating 2--3 events produce 95% confidence interval widths exceeding 18 percentage points ($\pm 9$ pp).

---

## 📊 Summary of Benchmark Results

### 1. Multi-Split Protocol Comparison
| Split Protocol | CrisisMMD (18,082 samples) | HumAID (76,484 samples) | Evaluation Context |
| :--- | :---: | :---: | :--- |
| **Random Split (Pooled)** | 55.12% $\pm$ 0.6% | 72.69% $\pm$ 0.3% | Uniform stratified partition (5 seeds) |
| **Random Split (Macro-Events)** | 54.78% $\pm$ 11.2% | 74.93% $\pm$ 5.7% | Unweighted event macro-average |
| **Region-Held-Out (LORO)** | 52.66% $\pm$ 13.9% | 73.36% $\pm$ 7.2% | Continental geographic transfer |
| **Event-Held-Out (LOEO, Pooled)** | 53.24% | 69.60% | Pooled sample-weighted transfer |
| **Event-Held-Out (LOEO, Macro)** | 52.08% $\pm$ 12.8% | 71.69% $\pm$ 7.7% | Target event deployment setting |
| **Worst-Event (LOEO Tail)** | **39.33% (-5.54 pp paired)** | **51.48% (-15.52 pp paired)** | Hardest unseen disaster incident |
| **Event-Prior Anchor** | **38.67%** | **32.21%** | Zero-feature historical base-rate prior |

### 2. Domain Generalization Algorithms (26 LOEO Folds, 5 Seeds)
| Algorithm | CrisisMMD Mean LOEO | CrisisMMD Worst Event | HumAID Mean LOEO | HumAID Worst Event | HumAID ECE | Scaled ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ERM** | **51.41% $\pm$ 11.6%** | 38.77% | **67.75% $\pm$ 8.9%** | 46.90% | 0.1608 | **0.024** |
| **GroupDRO** | 51.27% $\pm$ 11.2% | **39.65% (+0.88 pp)** | 67.14% $\pm$ 8.4% | **47.71% (+0.81 pp)** | 0.1956 | 0.022 |
| **IRM** | 48.51% $\pm$ 11.6% | 37.86% | 63.98% $\pm$ 9.7% | 45.97% | **0.0806** | 0.021 |

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/sohanreddy5/Disaster-AI-Audit.git
cd Disaster-AI-Audit

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Verification Unit Tests
```bash
# Run unit tests validating split integrity, claims, and calibration
python -m unittest discover -s tests
python -m unittest tests/test_claims.py
```

### 3. Reproduce All Experiments & Generate Figures
```bash
# Master reproduction script
bash scripts/reproduce_all.sh

# Or run individual audit components:
python scripts/run_benchmark.py --dataset crisismmd
python scripts/run_benchmark.py --dataset humaid
python scripts/retrain_domain_generalization.py
python scripts/run_compound_shift.py
python scripts/run_power_analysis.py
python scripts/generate_paper_figures.py
python scripts/generate_tables_and_macros.py
```

---

## 📁 Repository Structure

```
disaster-ai-audit/
├── LICENSE                    # MIT (Code), CC BY 4.0 (Docs), CC BY-NC-SA 4.0 (Data)
├── requirements.txt           # Dependency specifications
├── src/
│   ├── data/                  # Dataset loaders & split generators (LOEO, LORO, LODTO)
│   ├── features/              # MiniLM text & OpenCLIP feature extractors
│   ├── models/                # Convex probes, GroupDRO, IRM, EventPrior
│   └── evaluation/            # Calibration (ECE), selective prediction (AURC), power
├── scripts/
│   ├── reproduce_all.sh       # Master 8-step reproduction pipeline
│   ├── run_benchmark.py       # Multi-split benchmark harness
│   ├── retrain_domain_generalization.py # 5-seed DG retrain with Holm tests
│   ├── run_compound_shift.py  # Fine-grained noise & NSR stress testing
│   ├── compute_paired_shift_and_diagnostics.py # Paired gaps & causal deduplication
│   ├── generate_tables_and_macros.py # LaTeX macro generator
│   └── generate_paper_figures.py # Publication figures (PDF & PNG)
├── tests/
│   ├── test_claims.py         # Unit tests validating mathematical claims
│   ├── test_splits.py         # Leakage-free partition unit tests
│   └── test_full_suite.py     # End-to-end component tests
└── docs/paper/
    ├── main.tex               # IEEEtran journal camera-ready LaTeX source
    ├── macros_results.tex     # Auto-generated semantic result macros
    └── figures/               # Generated publication vector graphics (Fig 1-4)
```

---

## 📜 Citation & License

If you use this benchmark, audit harness, or split manifests in your research, please cite:
```bibtex
@article{mandala2026beyond,
  title={Beyond the Random Split: An Empirical Audit and Out-of-Distribution Benchmark of Text-Based Disaster AI},
  author={Mandala, Sohan Reddy},
  journal={arXiv preprint},
  year={2026}
}
```

- **Software:** MIT License
- **Manuscript & Documentation:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Derived Manifests & Features:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
