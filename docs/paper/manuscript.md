# Beyond the Random Split: An Empirical Audit and Out-of-Distribution Benchmark of Text-Based Disaster AI

**Author:** Sohan Reddy Mandala (msohanreddy@gmail.com, ORCID: [0009-0009-6296-7536](https://orcid.org/0009-0009-6296-7536))  
**Affiliation:** Independent Researcher, Hyderabad, Telangana, India  
**Target Venues:** IEEE Transactions on Geoscience and Remote Sensing (TGRS) / ISPRS Journal of Photogrammetry and Remote Sensing / Nature Scientific Data / NeurIPS Datasets and Benchmarks Track  
**Code Repository:** [https://github.com/sohanreddy5/Disaster-AI-Audit](https://github.com/sohanreddy5/Disaster-AI-Audit)  
**License:** MIT (Code) / CC BY 4.0 (Manuscript) / CC BY-NC-SA 4.0 (Manifests and Features)  

---

## Abstract

Machine learning systems for rapid disaster triage and humanitarian crisis response are conventionally evaluated using uniform random train/test splits. Under this protocol, observations from every disaster event are shared across training and test partitions, obscuring out-of-distribution failure modes. In real-world emergency deployments, however, models must generalize to entirely unobserved disaster incidents in unfamiliar geographic regions. In this paper, we conduct an extensive empirical audit of crisis classification across 94,566 labeled observations from 26 event-level evaluation folds (7 in CrisisMMD and 19 in HumAID) spanning 22 unique physical disasters across 9 continental regions. While macro leave-one-event-out (LOEO) accuracy drops by only 1.0 to 3.0 percentage points on average relative to random splits, evaluating individual target events reveals severe performance vulnerability: paired test accuracy on the hardest held-out event drops by 5.54 percentage points on CrisisMMD (California Wildfires) and 15.52 points on HumAID (Maryland Floods) relative to event-matched random test instances. We find that event-level class-prior divergence is the single strongest predictor of out-of-distribution drop ($r=0.563, p=0.0121$), and an input-free label-prior baseline accounts for 38.67% and 32.21% accuracy. An audit of cross-split contamination reveals that near-duplicates ($\tau \ge 0.90$) exhibit an association accuracy gap of up to +13.74 to +15.46 percentage points on leaked slices, but causal purging demonstrates that net benchmark inflation across the full test set is limited to 0.00 to 0.69 percentage points. Across all 26 LOEO folds, minibatch-normalized GroupDRO modestly aids worst-event accuracy (+0.40 to +0.69 pp), whereas Invariant Risk Minimization (IRM) incurs a statistically significant accuracy penalty (-2.90 to -3.77 pp, Holm-corrected $p \le 0.031$) in exchange for reduced Expected Calibration Error (ECE 0.081 vs 0.161), an improvement matched by temperature-scaled empirical risk minimization (ECE 0.024--0.041) without accuracy loss. Under zero-shot synthetic feature corruption, unnormalized perturbations multiply embedding norms, driving ECE up to 0.741. Finally, empirical cluster bootstrapping demonstrates that evaluating fewer than 15 held-out events yields 95% confidence interval widths exceeding 10 percentage points. We openly release all code, cached embeddings, and split manifests.

---

## 1. Introduction

When sudden-onset disasters strike---such as hurricanes, earthquakes, and flash floods---emergency response organizations require timely and accurate situational awareness from social media and remote sensing data. Microblogged text and social imagery provide real-time updates regarding casualty counts, infrastructure damage, and urgent resource requests. Deep learning classifiers have therefore been proposed to automate crisis triage.

Conventionally, disaster classification benchmarks evaluate performance by partitioning datasets using uniform random train/test splits. Under this standard protocol, observations from every disaster incident are randomly distributed between training and test sets. When emergency management agencies deploy models during a new, unfolding crisis, however, the target event is necessarily unseen.

```
                 CONVENTIONAL PARADIGM (Random Split)
┌─────────────────────────────────────────────────────────────────┐
│ Disaster A (80%) │ Disaster B (80%) │ Disaster C (80%) │ Train  │
├──────────────────┼──────────────────┼──────────────────┼────────┤
│ Disaster A (20%) │ Disaster B (20%) │ Disaster C (20%) │ Test   │
└─────────────────────────────────────────────────────────────────┘
 Result: High Reported Accuracy, but Evaluates In-Distribution Interpolation

                  REAL-WORLD OPERATIONAL DEPLOYMENT (LOEO)
┌─────────────────────────────────────────────────────────────────┐
│ Disaster A (100%)│ Disaster B (100%)│ Historical Events│ Train  │
├──────────────────┴──────────────────┴──────────────────┼────────┤
│ Novel Unseen Event (e.g. Maryland Floods / CalFire)    │ Test   │
└─────────────────────────────────────────────────────────────────┘
 Result: Substantial Event-Specific Degradation (Paired Drops up to 15.5 pp)
```

This paper provides an empirical audit of this discrepancy. We evaluate 94,566 observations across 26 event folds from CrisisMMD and HumAID, documenting the true paired shift gap on unseen events, diagnosing the role of class-prior divergence, isolating near-duplicate leakage effects, evaluating domain generalization algorithms across 5 seeds, and quantifying required benchmark sample sizes.

---

## 2. Harmonized Multi-Event Benchmark Construction

We harmonize two canonical crisis informatics benchmarks spanning **94,566 observations across 26 evaluation folds (22 unique physical disasters)**:

| Benchmark Collection | Samples | Evaluation Folds | Classes | Disaster Types | Continental Regions | Modality Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CrisisMMD** | 18,082 | 7 LOEO folds | 8 | Wildfire, Hurricane, Earthquake, Flood | 5 Regions | Microblogged Text |
| **HumAID** | 76,484 | 19 LOEO folds | 10 | Wildfire, Cyclone, Earthquake, Flood, Hurricane | 8 Regions | Microblogged Text |
| **Total Harmonized** | **94,566** | **26 Folds** | -- | **5 Types** | **9 Regions** | **Standardized 384-d Embeddings** |

Four physical disasters overlap between the corpora: Hurricane Harvey (2017), Hurricane Irma (2017), Hurricane Maria (2017), and Sri Lanka Floods (2017). To maintain methodological rigor, each benchmark is evaluated independently.

---

## 3. Methodology and Evaluation Framework

1. **Feature Representations:** Extracted using pretrained `all-MiniLM-L6-v2` (384-dimensional dense sentence embeddings, max token length 128, mean pooling), $L_2$-normalized to the unit sphere ($\|x\|_2 = 1$).
2. **Convex $L_2$-Regularized Probes:** Evaluated across four split protocols (Random 80/20, LORO, LOEO, LODTO) using multinomial logistic probes ($C = 1.0$) optimized via L-BFGS to global convergence.
3. **Neural Domain Generalization Learners:** Benchmarked across all 26 LOEO folds over 5 independent random seeds using mini-batch Adam ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$):
   - **ERM:** Empirical Risk Minimization on pooled source events.
   - **Minibatch-Normalized GroupDRO:** Online exponential weighting over training disaster groups ($q_g \propto \exp(\eta_q \mathcal{L}_g)$).
   - **IRM:** Gradient norm invariance penalty across training environments ($\lambda_{\text{irm}} = 1.0$ on HumAID, annealed to $10.0$ on CrisisMMD).
   - *CORAL Note:* On frozen feature representations $x$, the covariance penalty satisfies $\nabla_W \text{Cov}(x) \equiv 0$, identically reducing parameter updates to ERM; CORAL is therefore omitted.
4. **Zero-Feature Event Prior Baseline:** Predicts the historical macro class prior across training events without consuming input features:
   $$\hat{P}(y = c) = \frac{1}{|\mathcal{E}_{\text{train}}|} \sum_{e \in \mathcal{E}_{\text{train}}} \frac{\sum_{i \in \mathcal{D}_e} \mathbb{I}(y_i = c)}{|\mathcal{D}_e|}$$

---

## 4. Key Empirical Findings

### 4.1 Evaluation Splits and Paired Shift Gaps

| Evaluation Protocol | CrisisMMD Mean Acc (%) | CrisisMMD Worst Event (%) | HumAID Mean Acc (%) | HumAID Worst Event (%) |
| :--- | :---: | :---: | :---: | :---: |
| Majority Class Baseline | 38.70 | -- | 27.82 | -- |
| Event Prior Anchor | 38.67 $\pm$ 5.1 | 32.10 | 32.21 $\pm$ 4.0 | 28.60 |
| Random Split (Pooled) | 55.12 $\pm$ 0.6 | 54.35 | 72.69 $\pm$ 0.3 | 72.30 |
| Random Split (Macro-Events) | 54.78 $\pm$ 11.2 | 44.88 | 74.93 $\pm$ 5.7 | 65.88 |
| Leave-One-Region-Out (LORO) | 52.66 $\pm$ 13.9 | 42.04 | 73.36 $\pm$ 7.2 | 66.11 |
| Leave-One-Event-Out (LOEO, Pooled) | 53.24 | 39.33 | 69.60 | 51.48 |
| Leave-One-Event-Out (LOEO, Macro) | 52.08 $\pm$ 12.8 | **39.33** | 71.69 $\pm$ 7.7 | **51.48** |
| Leave-One-Disaster-Type-Out (LODTO) | 51.24 $\pm$ 17.1 | 39.33 | 69.73 $\pm$ 4.3 | 64.84 |

- **Paired Shift Gaps:** Comparing random-split accuracy directly to the lowest LOEO fold conflates base event difficulty with distribution shift. The exact paired shift gap on matching test samples is:
  - **CrisisMMD:** Mean paired drop = 2.70 pp; California Wildfires drops from 44.88% (random) to 39.33% (LOEO), a paired drop of 5.54 pp. Maximum paired drop is Iraq-Iran Earthquake (+6.97 pp: 49.01% $\to$ 42.04%).
  - **HumAID:** Mean paired drop = 3.23 pp; Maryland Floods drops from 67.01% (random) to 51.48% (LOEO), a severe paired drop of **15.52 percentage points**.
- **Predictor of Failure:** Class-prior divergence $D_{\text{KL}}(P_e \| P_{\text{train}})$ is strongly correlated with the paired drop ($r = 0.563, p = 0.0121$). Sample count ($r = -0.218$) and active class count ($r = 0.089$) show negligible correlation.
- **Event Prior Anchor:** Achieves 38.67% on CrisisMMD and 32.21% on HumAID without features. In all 26 folds, learned text models outperform this prior baseline by +15.3 to +65.4 percentage points.
- **Regional Macro Nuance:** HumAID LORO (73.36%) exceeds pooled random (72.69%) because macro-averaging over 8 continental regions weights smaller high-accuracy regions equally, whereas pooled random split is dominated by large North American hurricane events.

---

### 4.2 Domain Generalization Benchmark (5-Seed Retraining)

| Algorithm | Dataset | Mean Acc. (%) | Worst Event (%) | ECE | Scaled ECE | AURC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ERM** | CrisisMMD | **51.41 $\pm$ 11.6** | 38.77 | 0.076 | 0.041 | **0.356** |
| **GroupDRO** | CrisisMMD | 51.27 $\pm$ 11.2 | **39.65** | 0.096 | **0.034** | 0.370 |
| **IRM** | CrisisMMD | 48.51 $\pm$ 11.6 | 37.86 | **0.067** | 0.035 | 0.386 |
| **ERM** | HumAID | **67.75 $\pm$ 8.9** | 46.90 | 0.161 | **0.024** | **0.154** |
| **GroupDRO** | HumAID | 67.14 $\pm$ 8.4 | **47.71** | 0.196 | 0.022 | 0.161 |
| **IRM** | HumAID | 63.98 $\pm$ 9.7 | 45.97 | **0.081** | **0.021** | 0.186 |

- **Worst-Event Tail Protection:** Minibatch-normalized GroupDRO improves worst-event accuracy across both datasets (38.77% $\to$ 39.65% on CrisisMMD; 46.90% $\to$ 47.71% on HumAID).
- **The IRM Tradeoff:** IRM dramatically reduces unscaled ECE on HumAID (0.081 vs 0.161, Wilcoxon $W = 0.0, p = 4.0 \times 10^{-6}$, Holm-corrected $p = 8.0 \times 10^{-6}$). However, this calibration gain incurs a statistically significant **-3.77 pp accuracy drop** on HumAID ($p = 8.0 \times 10^{-6}$) and **-2.90 pp drop** on CrisisMMD ($p = 0.031$).
- **Temperature Scaling Resolution:** Applying post-hoc temperature scaling ($T \approx 0.64$) to ERM reduces ECE to **0.024** on HumAID and **0.041** on CrisisMMD without any accuracy penalty, demonstrating that IRM's calibration gain is driven by logit shrinkage rather than invariant feature learning.

---

### 4.3 Near-Duplicate Leakage and Causal Deduplication

| Dataset | Threshold | Leaked Share (%) | Leaked Acc (%) | Clean Acc (%) | Association Gap ($\Delta$) | Net Test Effect (pp) | Causal Purge Drop (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CrisisMMD** | $\tau \ge 0.90$ | 15.7% (568 / 3,617) | 60.98 $\pm$ 0.9 | 54.40 $\pm$ 0.8 | **+6.59 pp** | +1.04 pp | -0.69 pp |
| **CrisisMMD** | $\tau \ge 0.98$ | 15.7% (567 / 3,617) | 60.06 $\pm$ 0.8 | 54.20 $\pm$ 0.8 | **+5.86 pp** | +0.92 pp | -0.69 pp |
| **HumAID** | $\tau \ge 0.90$ | 4.3% (652 / 15,297) | 85.74 $\pm$ 1.0 | 72.00 $\pm$ 0.3 | **+13.74 pp** | +0.59 pp | 0.00 pp |
| **HumAID** | $\tau \ge 0.98$ | 0.04% (6 / 15,297) | 87.78 $\pm$ 7.2 | 72.68 $\pm$ 0.3 | **+15.09 pp** | +0.01 pp | 0.00 pp |

- **Slice vs Net Effect:** While leaked test instances exhibit a substantial accuracy advantage (+6.59 to +15.09 pp), their low prevalence (4.3% on HumAID) limits the net test-set effect to +0.59 to +1.04 percentage points.
- **Causal Purging:** When near-duplicates are causally removed from training data, model accuracy on the full test set drops by only **0.69 pp** on CrisisMMD and **0.00 pp** on HumAID, confirming that near-duplicate overlap does not artificially inflate headline benchmark numbers.

---

### 4.4 Synthetic Feature Corruption and Stress Response

- **Noise-to-Signal Ratio:** Features are unit-normalized ($\|x\|_2 = 1, d = 384$). For Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, $\mathbb{E}[\|\epsilon\|_2] = \sqrt{384 \sigma^2}$. NSR scales linearly: $\sigma = 0.05 \implies \text{NSR} \approx 0.98$; $\sigma = 0.10 \implies \text{NSR} \approx 1.96$; $\sigma = 0.50 \implies \text{NSR} \approx 9.80$.
- **Accuracy & Calibration Collapse:** On HumAID, accuracy degrades from 71.69% ($\sigma=0.0$) to 61.93% ($\sigma=0.05$), 47.32% ($\sigma=0.10$), and 17.48% ($\sigma=0.50$), while ECE surges from 0.039 to **0.741**.
- **Logit Norm Inflation:** Unnormalized noise multiplies embedding norms by up to $10\times$, scaling unconstrained linear logits and driving softmax confidence to near 1.0 on randomized inputs. Re-normalizing features preserves calibration ($\text{ECE} \le 0.045$), and noise-augmented training reduces ECE to 0.016.

---

### 4.5 Resampling and Statistical Power Analysis

- **Bootstrap Protocol:** Subsampled $k \in [3, 18]$ held-out events across 500 outer subsets without replacement, with 200 inner cluster bootstrap resamples.
- **Analytic Comparison:** Mean 95% CI width narrows from 18.21 pp ($k=3$) to 14.87 pp ($k=7$), 12.99 pp ($k=10$), 10.96 pp ($k=15$), and 10.18 pp ($k=18$).
- **Theoretical Curves:** Plotted against infinite superpopulation curve ($2 \cdot 1.96 \cdot \text{SD}/\sqrt{k}$, $\text{SD}=0.0768$) and finite-population-corrected (FPC) curve ($\sqrt{\frac{N-k}{N-1}}$).
- **Recommendation:** Evaluating fewer than 15 disaster events results in an empirical 95% confidence interval width exceeding 10 percentage points. Reliable crisis AI benchmarks require at least **$k \ge 15$ held-out disaster events**.

---

## 5. Repository Structure and Reproducibility

```
disaster-ai-audit/
├── LICENSE                    # MIT (Code) / CC BY 4.0 (Docs) / CC BY-NC-SA 4.0 (Data)
├── requirements.txt           # Version-specified dependencies
├── src/
│   ├── data/                  # Loaders & split generators (LOEO, LORO, LODTO)
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

To execute the entire verification suite:
```bash
bash scripts/reproduce_all.sh
```
All unit tests can be validated independently:
```bash
python -m unittest discover -s tests
```
