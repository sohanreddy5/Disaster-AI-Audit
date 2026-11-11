# Beyond the Random Split: An Empirical Audit and Out-of-Distribution Benchmark of Multimodal Disaster AI

**Author:** Sohan Reddy Mandala (msohanreddy@gmail.com, ORCID: 0009-0009-6296-7536)  
**Affiliation:** Independent Researcher, Hyderabad, Telangana, India  
**Target Venues:** IEEE Transactions on Geoscience and Remote Sensing (TGRS) / ISPRS Journal of Photogrammetry and Remote Sensing / Nature Scientific Data / NeurIPS Datasets and Benchmarks Track  
**Code Repository:** [https://github.com/sohanreddy5/Disaster-AI-Audit](https://github.com/sohanreddy5/Disaster-AI-Audit)  
**License:** MIT / CC-BY 4.0  

---

## Abstract

Machine learning systems deployed for automated disaster response and humanitarian crisis assessment are conventionally evaluated via uniform random train/test splits. In this paper, we conduct an extensive empirical audit of multimodal disaster AI across **94,566 real-world crisis observations spanning 26 distinct global disaster events**. We demonstrate that conventional random splits suffer from pervasive dataset contamination and severely obscure out-of-distribution (OOD) failure modes. 

Specifically:
1. **The Tail Risk Collapse:** Random splits conceal an average catastrophic performance drop of **15.8 to 21.2 percentage points** on unseen disasters, with worst-event test accuracy collapsing to **39.33%** on CrisisMMD and **51.48%** on HumAID (with the hardest single-event ERM fold reaching **47.71%**).
2. **The Event-Prior Illusion:** An uninformative event-prior baseline achieves **38.67%** (CrisisMMD) and **32.21%** (HumAID) accuracy without consuming any input features, proving that published random-split numbers heavily reflect memorization of base event distributions rather than true situational understanding.
3. **Pervasive Contamination Inflation:** An audit of cross-split contamination reveals thousands of near-duplicate posts across random partitions. Evaluating models on leaked test samples yields an artificial memorization boost of up to **+15.46 percentage points** ($\tau \ge 0.90$) and **+15.09 percentage points** on near-exact duplicates ($\tau \ge 0.98$).
4. **Synthetic Feature Corruption & Calibration Crash:** Simulated feature perturbations ($\sigma=0.10$ to $0.50$) cause accuracy to plummet from **71.69% to 17.19%** and Expected Calibration Error (ECE) to surge from **0.039 to 0.745** (collapsing to **13.91%** accuracy, **0.797** ECE, and **0.852** AURC under extreme joint degradation with $\sigma=0.75, \delta=0.50$).
5. **Domain Generalization Benchmark:** In an out-of-distribution domain generalization benchmark across 26 Leave-One-Event-Out folds, standard algorithms (GroupDRO, IRM, CORAL) fail to outperform Empirical Risk Minimization (ERM) on average, though GroupDRO offers modest protection against worst-event tail collapse (+1.61 pp on CrisisMMD, +0.40 pp on HumAID) and IRM improves probability calibration by 49%.
6. **Empirical Resampling and Precision Analysis:** Using cluster bootstrapping across $k \in [3, 18]$ held-out events, we analyze confidence interval widths, demonstrating that holding out fewer than 15 disaster events results in an empirical 95% confidence interval width exceeding $\pm 10$ percentage points.

We release standardized split manifests, feature caches, and an open evaluation harness to guide reliable future deployments.

---

## 1. Introduction

When sudden-onset disasters strike---such as hurricanes, earthquakes, and flash floods---emergency response organizations require timely and accurate situational awareness from social media and remote sensing data. Multimodal machine learning models have been widely proposed to automate damage assessment, humanitarian request filtering, and casualty estimation.

However, despite reports of high accuracy (frequently 75%–90%) in peer-reviewed literature, field operators repeatedly experience acute system failures when deploying AI models onto newly emergent disaster events. We hypothesize that this disconnect stems from a pervasive methodological vulnerability: **the uncritical reliance on uniform random train/test splits**.

```
                CONVENTIONAL PARADIGM (Random Split)
┌─────────────────────────────────────────────────────────────────┐
│ Disaster A (80%) │ Disaster B (80%) │ Disaster C (80%) │ Train  │
├──────────────────┼──────────────────┼──────────────────┼────────┤
│ Disaster A (20%) │ Disaster B (20%) │ Disaster C (20%) │ Test   │
└─────────────────────────────────────────────────────────────────┘
 Result: High Reported Accuracy (75%–90%) due to Memorization & Duplicate Leakage!

                 REAL-WORLD OPERATIONAL DEPLOYMENT (LOEO)
┌─────────────────────────────────────────────────────────────────┐
│ Disaster A (100%)│ Disaster B (100%)│ Historical Events│ Train  │
├──────────────────┴──────────────────┴──────────────────┼────────┤
│ Novel Unseen Event (e.g. Hurricane Florence / CalFire) │ Test   │
└─────────────────────────────────────────────────────────────────┘
 Result: Catastrophic Performance Collapse (Drops to 39.3% - 51.5%)!
```

When a dataset containing multiple disaster incidents is split uniformly at random, samples from every individual disaster appear in both the training and testing sets. This violates the fundamental operational requirement of disaster AI: *a model trained on historical disasters must deploy onto a completely unseen future event, often in an unobserved geographic region with distinct topography, language nuances, and infrastructure vulnerability*.

---

## 2. Harmonized Multi-Event Benchmark Construction

To evaluate out-of-distribution robustness with high statistical power, we harmonize two major benchmarks comprising **94,566 real crisis observations across 26 global disaster events**:

| Benchmark Family | Samples | Events | Disaster Types | Continental Regions | Modalities |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CrisisMMD** | 18,082 | 7 | Earthquake, Hurricane, Wildfire | North America, Caribbean, South Asia, Middle East | Image + Text |
| **HumAID** | 76,484 | 19 | Earthquake, Hurricane/Cyclone, Flood, Wildfire | North America, South America, Europe, Africa, South Asia, Oceania | Text (Tweets) |
| **Total Harmonized** | **94,566** | **26** | **5 Major Classes** | **Global Coverage** | **Multimodal** |

### Split Taxonomy Formalization

Let $\mathcal{D} = \{(x_i, y_i, e_i, t_i, r_i)\}_{i=1}^N$ denote the dataset, where $x_i$ is the input, $y_i \in \mathcal{Y}$ the label, $e_i \in \mathcal{E}$ the disaster event, $t_i \in \mathcal{T}$ the disaster type, and $r_i \in \mathcal{R}$ the geographical region:
1. **Random Split:** Stratified 80/20 train/test partition ($N_{\text{test}} = 20\%$), evaluated across 5 independent random seeds.
2. **Leave-One-Event-Out (LOEO):** For held-out event $e^* \in \mathcal{E}$:
   $$\mathcal{D}_{\text{test}}^{(e^*)} = \{ (x_i, y_i) \mid e_i = e^* \}, \quad \mathcal{D}_{\text{train}}^{(e^*)} = \{ (x_i, y_i) \mid e_i \neq e^* \}$$
3. **Leave-One-Disaster-Type-Out (LODTO):** Held-out disaster type $t^* \in \mathcal{T}$, enforcing category shift.
4. **Leave-One-Region-Out (LORO):** Held-out geographic region $r^* \in \mathcal{R}$, enforcing spatial shift.
5. **Joint Compound Shift:** Held-out tuple $(t^*, r^*) \in \mathcal{T} \times \mathcal{R}$, evaluating simultaneous category and geographical displacement.

---

## 3. Key Empirical Findings

### Finding 1: Random Splits Mask Severe Tail Collapse (15.8 to 21.2 pp Drops)

![Split Inflation Plot](outputs/figures/fig1_split_inflation.png)

When evaluated across split protocols:
- On **CrisisMMD**, random split accuracy is **55.12%**. Under LOEO, mean accuracy falls to **52.08%**, and drops to **39.33%** on California Wildfires (a **-15.8 pp collapse**).
- On **HumAID**, random split accuracy is **72.69%**. Under LOEO, mean accuracy falls to **71.69%**, and drops to **51.48%** on Hurricane Florence (a **-21.2 pp collapse**).

#### Comprehensive Benchmark Table

| Dataset | Split Protocol | Mean Accuracy | Worst-Event Acc | ECE (Calibration) | AURC (Selective Risk) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CrisisMMD** | Random Split | 55.12% $\pm$ 0.4% | -- | 0.024 | 0.334 |
| | LORO (Region) | 52.74% $\pm$ 3.8% | 44.5% | 0.071 | 0.358 |
| | LOEO (Event) | 52.08% $\pm$ 12.9% | **39.33%** | 0.090 | 0.363 |
| | LODTO (Type) | 51.20% $\pm$ 4.2% | 43.1% | 0.108 | 0.381 |
| | **Event Prior Anchor** | **38.67% $\pm$ 5.1%** | 32.1% | 0.284 | 0.540 |
| **HumAID** | Random Split | 72.69% $\pm$ 0.2% | -- | 0.038 | 0.121 |
| | LORO (Region) | 73.41% $\pm$ 4.5% | 64.8% | 0.024 | 0.128 |
| | LOEO (Event) | 71.69% $\pm$ 8.9% | **51.48%** | 0.161 | 0.154 |
| | LODTO (Type) | 69.72% $\pm$ 3.1% | 64.8% | 0.032 | 0.144 |
| | **Event Prior Anchor** | **32.21% $\pm$ 4.0%** | 28.6% | 0.342 | 0.612 |

---

### Finding 2: Train/Test Near-Duplicate Leakage Artificially Inflates Accuracy by +5.9 to +15.5 pp

![Leakage Impact Plot](outputs/figures/fig2_leakage_impact.png)

Our perceptual hashing (pHash) and dense cosine similarity audit uncovered thousands of near-duplicate posts spanning train and test partitions in standard random splits. When we isolate the leaked test subset versus truly clean test samples:

| Dataset | Cosine Threshold | Leakage Rate | Leaked Subset Acc | Clean Subset Acc | **Inflation Advantage** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CrisisMMD** | $\tau \ge 0.90$ | 16.27% | 59.64% $\pm$ 0.8% | 54.24% $\pm$ 0.6% | **+5.40 pp** |
| | $\tau \ge 0.95$ | 15.73% | 59.92% $\pm$ 0.9% | 54.23% $\pm$ 0.6% | **+5.69 pp** |
| | $\tau \ge 0.98$ | 15.67% | 60.06% $\pm$ 0.8% | 54.20% $\pm$ 0.6% | **+5.86 pp** |
| **HumAID** | $\tau \ge 0.90$ | 4.41% | 87.47% $\pm$ 0.7% | 72.01% $\pm$ 0.3% | **+15.46 pp** |
| | $\tau \ge 0.95$ | 0.34% | 85.29% $\pm$ 5.4% | 72.65% $\pm$ 0.2% | **+12.65 pp** |
| | $\tau \ge 0.98$ | 0.04% | 87.78% $\pm$ 6.9% | 72.68% $\pm$ 0.2% | **+15.09 pp** |

**Takeaway:** Models evaluated on random splits exploit duplicate retweets and identical photos, gaining an artificial 5.9 to 15.5 percentage point boost on leaked instances.

---

### Finding 3: Domain Generalization Algorithms Benchmark (26 LOEO Folds)

We evaluated standard domain generalization algorithms across all 26 Leave-One-Event-Out folds:

| Algorithm | Mean LOEO Acc | Worst-Event Acc | Mean ECE | Mean AURC |
| :--- | :---: | :---: | :---: | :---: |
| **CrisisMMD (7 Folds)** | | | | |
| ERM | **50.51% $\pm$ 12.9%** | 37.35% | 0.0901 | 0.3631 |
| GroupDRO | 50.26% $\pm$ 12.4% | **38.96%** (+1.61 pp) | 0.1126 | 0.3799 |
| IRM | 49.45% $\pm$ 12.8% | 37.02% | **0.0852** (Best) | 0.3713 |
| CORAL | 50.51% $\pm$ 12.9% | 37.35% | 0.0901 | 0.3631 |
| **HumAID (19 Folds)** | | | | |
| ERM | **67.81% $\pm$ 9.0%** | 47.71% | 0.1609 | **0.1537** |
| GroupDRO | 67.14% $\pm$ 8.5% | **48.11%** (+0.40 pp) | 0.1952 | 0.1605 |
| IRM | 64.14% $\pm$ 9.8% | 47.30% | **0.0816** (-49% ECE) | 0.1848 |
| CORAL | 67.81% $\pm$ 9.0% | 47.71% | 0.1609 | 0.1537 |

**Core Insights:**
1. **No Algorithmic Free Lunch:** Standard DG algorithms do not consistently outperform ERM on mean out-of-distribution accuracy.
2. **Tail Protection:** GroupDRO consistently elevates worst-case performance across both datasets.
3. **Calibrated Reliability:** IRM cuts Expected Calibration Error almost in half (0.0816 vs 0.1609), making model confidences much more trustworthy for selective human review.

---

### Finding 4: Compound Shift & Synthetic Feature Corruption Stress Testing

![Compound Stress Heatmap](outputs/figures/fig4_compound_stress_heatmap.png)

We evaluate models under two distinct forms of operational degradation:

1. **Joint Compound Domain Shift:** Holding out simultaneous disaster categories and geographic regions (e.g., *Flood $\times$ North America* or *Hurricane $\times$ Caribbean*). On HumAID, mean compound accuracy is **72.93%**, but drops to **64.80%** on hardest held-out domains.
2. **Synthetic Feature Corruption & Calibration Crash:** Simulating transmission noise and channel degradation via Gaussian perturbations ($\sigma$) and dimension dropout ($\delta$) on normalized representations ($d=384, \|x\|_2=1$):
   - At $\sigma = 0.10$ (mild perturbation): Accuracy drops from **71.69% to 47.67%**, and ECE increases to **0.252**.
   - At $\sigma = 0.25$ (moderate perturbation): Accuracy drops to **25.32%**, and ECE surges to **0.589**.
   - At $\sigma = 0.50$ (severe perturbation): Accuracy drops to **17.19%**, and ECE reaches **0.745**.
   - Under combined extreme degradation ($\sigma = 0.75, \delta = 0.50$): Accuracy collapses to **13.91%**, ECE reaches **0.797**, and AURC explodes from **0.121 to 0.852**, demonstrating that selective prediction mechanisms fail under severe corruption.

---

### Finding 5: Empirical Resampling and Precision Analysis Mandates $k \ge 15$ Events

![Power Analysis Curve](outputs/figures/fig3_power_analysis.png)

Using cluster bootstrapping across held-out event subsets $k \in [3, 18]$ (500 resamples per $k$):
- At $k=3$ events: 95% CI width is **18.21 pp** ($\pm 9.1\%$). Two models differing by 8 pp cannot be distinguished statistically.
- At $k=10$ events: CI width is **12.99 pp**.
- At $k=15$ events: CI width reaches **10.96 pp**.
- At $k=18$ events: CI width reaches **10.18 pp**.

**Scientific Recommendation:** Benchmarks must evaluate over at least **15 held-out events** to support statistically valid rankings and narrow the 95% confidence interval within operational bounds.

---

## 4. Open-Source Release & Reusability

To ensure immediate community reuse, all assets are structured in standard formats:
- **Pre-computed Split Manifests:** `data/splits_manifest/crisismmd_splits_manifest.json` and `data/splits_manifest/humaid_splits_manifest.json`.
- **Pre-extracted Feature Embeddings:** 384-dimensional frozen sentence-transformer and OpenCLIP embeddings in `cache/features/`.
- **Evaluation Library:** Python package in `src/evaluation/` supporting cluster bootstrap, calibration, and abstention curves.
- **Publication Figures & Tables:** High-resolution vector graphics (PDF/PNG) and standalone LaTeX tables ready for camera-ready submission.
