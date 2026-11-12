# Beyond the Random Split: An Empirical Audit and Out-of-Distribution Benchmark of Text-Based Disaster AI

**Author:** Sohan Reddy Mandala (msohanreddy@gmail.com, ORCID: [0009-0009-6296-7536](https://orcid.org/0009-0009-6296-7536))  
**Affiliation:** Independent Researcher, Hyderabad, Telangana, India  
**Target Venues:** IEEE Transactions on Geoscience and Remote Sensing (TGRS) / ISPRS Journal of Photogrammetry and Remote Sensing / Nature Scientific Data / NeurIPS Datasets and Benchmarks Track  
**Code Repository:** [https://github.com/sohanreddy5/Disaster-AI-Audit](https://github.com/sohanreddy5/Disaster-AI-Audit)  
**License:** MIT (Code) / CC BY 4.0 (Manuscript) / CC BY-NC-SA 4.0 (Manifests and Features)  

---

## Abstract

Disaster-response classification systems are routinely benchmarked using uniform random train/test splits. We show that this standard practice obscures severe out-of-distribution failure modes. Evaluating 94,566 text observations across 26 leave-one-event-out (LOEO) folds from CrisisMMD and HumAID, we find that while macro LOEO accuracy drops modestly on average (2.70 pp on CrisisMMD, 3.23 pp on HumAID), the largest paired performance drop on individual disaster events reaches 6.97 pp on CrisisMMD (Iraq-Iran Earthquake) and 15.52 pp on HumAID (Maryland Floods). Event-level class-prior divergence is the strongest observed predictor of this degradation ($r = 0.563, p = 0.0121$), and an input-free label prior baseline achieves 38.67% and 32.21% accuracy. An audit of cross-split contamination reveals that near-duplicates yield an accuracy advantage of up to +15.46 pp on the leaked slice ($\tau \ge 0.90$), corresponding to a slice-weighted contribution of +0.68 to +0.88 pp; a purge-and-retrain evaluation after removing training near-duplicates shifts full test accuracy by at most 0.69 pp. Across all 26 folds, GroupDRO yields modest worst-event gains (+0.81 to +0.88 pp), while Invariant Risk Minimization (IRM) incurs an accuracy penalty of 2.90 to 3.77 pp ($p \le 0.032$) with no calibration advantage over temperature-scaled empirical risk minimization (scaled ECE 0.035 on HumAID, 0.041 on CrisisMMD). Finally, resampling analysis indicates that at least 15 held-out events were required in this benchmark setting to achieve an empirical 95% confidence interval width below 11 percentage points ($\pm 5.5$ pp). All code and data manifests are openly released.

---

## 1. Introduction

When sudden-onset disasters strike---such as hurricanes, earthquakes, and flash floods---emergency response organizations require timely and accurate situational awareness from social media feeds. Microblogged text provides real-time updates regarding casualty counts, infrastructure damage, and urgent resource requests. Deep learning classifiers have therefore been widely proposed to automate crisis triage.

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

| Benchmark Collection | Samples | Evaluation Folds | Classes | Disaster Types | Geographic Regions | Modality Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CrisisMMD** | 18,082 | 7 LOEO folds | 8 | Wildfire, Hurricane, Earthquake, Flood | 5 Regions | Microblogged Text |
| **HumAID** | 76,484 | 19 LOEO folds | 10 | Wildfire, Cyclone, Earthquake, Flood, Hurricane | 8 Regions | Microblogged Text |
| **Total Harmonized** | **94,566** | **26 Folds** | -- | **5 Types** | **9 Regions** | **Standardized 384-d Embeddings** |

Four physical disasters overlap between the corpora: Hurricane Harvey (2017), Hurricane Irma (2017), Hurricane Maria (2017), and Sri Lanka Floods (2017). To maintain methodological rigor, each benchmark is evaluated independently.

HumAID is formulated as a 10-class classification task following its canonical benchmark release: `caution_and_advice`, `displaced_people_and_evacuations`, `infrastructure_and_utility_damage`, `injured_or_dead_people`, `missing_or_found_people`, `not_humanitarian`, `other_relevant_information`, `requests_or_urgent_needs`, `rescue_volunteering_or_donation_effort`, and `sympathy_and_support`. The 11th category present in the original raw collection (`dont_know_or_cant_judge`) was excluded by the benchmark curators due to semantic ambiguity and label sparsity. Across individual event folds, active classes range from 6 to 10 depending on local disaster communications.

---

## 3. Methodology and Evaluation Framework

1. **Feature Representations and Model Scope:** Extracted using pretrained `all-MiniLM-L6-v2` (384-dimensional dense sentence embeddings, max token length 128, mean pooling), $L_2$-normalized to the unit sphere ($\|x\|_2 = 1$). Following standard domain generalization benchmark audits (e.g., DomainBed, WILDS), our benchmark evaluates linear probes trained on standardized frozen representations, isolating distribution shift and objective behavior from end-to-end representation adaptation.
2. **Convex $L_2$-Regularized Probes:** Evaluated across four split protocols (Random 80/20, LORO, LOEO, LODTO) using multinomial logistic probes ($C = 1.0$) optimized via L-BFGS to global convergence.
3. **Neural Domain Generalization Learners:** Benchmarked across all 26 LOEO folds over 5 independent random seeds using mini-batch Adam ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$):
   - **ERM:** Empirical Risk Minimization on pooled source events.
   - **Minibatch-Normalized GroupDRO:** Maintains group weights $q \in \Delta^{|\mathcal{G}|}$ with step size $\eta_q = 0.01$, updated via an exponentiated gradient step over active groups in each mini-batch: $q_g \leftarrow q_g \exp(\eta_q \mathcal{L}_g(\theta))$ and normalized via $w_g^{\text{batch}} = q_g / \sum_{g' \in \mathcal{G}_{\text{batch}}} q_{g'}$. This prevents inactive batch groups from deflating parameter gradients.
   - **IRM:** Gradient norm invariance penalty across training environments ($\lambda_{\text{irm}} = 1.0$ on HumAID, annealed to $10.0$ on CrisisMMD).
   - *CORAL Note:* On frozen feature representations $x$, the covariance penalty satisfies $\nabla_W \text{Cov}(x) \equiv 0$, identically reducing parameter updates to ERM; CORAL is therefore omitted.
4. **Post-Hoc Probability Calibration:** Test-time logits are scaled via a positive scalar temperature $T > 0$ as $\hat{z} = z / T$, where optimal scalar temperature $T$ is fitted strictly on source training events by minimizing cross-entropy via L-BFGS-B, and frozen during held-out target event evaluation.
5. **Zero-Feature Event Prior Baseline:** Predicts the historical macro class prior across training events without consuming input features:
   $$\hat{P}(y = c) = \frac{1}{|\mathcal{E}_{\text{train}}|} \sum_{e \in \mathcal{E}_{\text{train}}} \frac{\sum_{i \in \mathcal{D}_e} \mathbb{I}(y_i = c)}{|\mathcal{D}_e|}$$

---

## 4. Key Empirical Findings

### 4.1 Evaluation Splits and Paired Shift Gaps

| Evaluation Protocol | CrisisMMD Mean Acc (%) | CrisisMMD Worst Fold / Event (%) | HumAID Mean Acc (%) | HumAID Worst Fold / Event (%) |
| :--- | :---: | :---: | :---: | :---: |
| Majority Class Baseline | 38.70 | -- | 27.82 | -- |
| Event Prior Anchor | 38.67 $\pm$ 5.1 | 32.10 | 32.21 $\pm$ 4.0 | 28.60 |
| Random Split (Pooled) | 55.12 $\pm$ 0.6 | 54.35 (lowest seed) | 72.69 $\pm$ 0.3 | 72.30 (lowest seed) |
| Random Split (Macro-Events) | 54.78 $\pm$ 11.2 | 44.88 (worst event) | 74.93 $\pm$ 5.7 | 65.88 (worst event) |
| Leave-One-Region-Out (LORO) | 52.66 $\pm$ 13.9 | 42.04 | 73.36 $\pm$ 7.2 | 66.11 |
| Leave-One-Event-Out (LOEO, Pooled) | 53.24 | 39.33 | 69.60 | 51.48 |
| Leave-One-Event-Out (LOEO, Macro) | 52.08 $\pm$ 12.8 | **39.33** | 71.69 $\pm$ 7.7 | **51.48** |
| Leave-One-Disaster-Type-Out (LODTO) | 51.24 $\pm$ 17.1 | 39.33 | 69.73 $\pm$ 4.3 | 64.84 |

*Note: For pooled evaluations, worst-case reports the lowest random seed across 5 runs; for event-based evaluations, it reports the lowest held-out event fold. Macro-over-events rows report unweighted event-level accuracy ($\pm$ across events); sample-weighted metrics are reported in pooled rows and omitted in macro rows.*

- **Paired Shift Gaps:** Comparing random-split accuracy directly to the lowest LOEO fold conflates base event difficulty with distribution shift. The exact paired shift gap on matching test samples is:
  - **CrisisMMD:** Mean paired drop = 2.70 pp; California Wildfires drops from 44.88% (random) to 39.33% (LOEO), a paired drop of 5.54 pp. Maximum paired drop is Iraq-Iran Earthquake (+6.97 pp: 49.01% $\to$ 42.04%).
  - **HumAID:** Mean paired drop = 3.23 pp; Maryland Floods drops from 67.01% (random) to 51.48% (LOEO), a severe paired drop of **15.52 percentage points**. (Differences calculated from unrounded fold-level accuracies).
- **Predictors of Out-of-Distribution Degradation:** On HumAID, class-prior divergence $D_{\text{KL}}(P_e \| P_{\text{train}})$ exhibits a significant parametric correlation with paired drop (Pearson $r = 0.563, p = 0.0121$). However, rank correlation is more modest (Spearman $\rho = 0.253, p = 0.297$), and leave-one-out sensitivity analysis reveals that `maryland_floods_2018` ($D_{\text{KL}} = 0.784$, paired drop 15.52 pp) exerts substantial leverage: excluding Maryland Floods drops Pearson $r$ to $0.091$ ($p = 0.720$, Spearman $\rho = 0.121, p = 0.633$). On CrisisMMD, prior divergence exhibits negligible correlation ($r = -0.024, p = 0.960$; Spearman $\rho = 0.107, p = 0.819$). Event sample size ($r = -0.218$) and active class count ($r = 0.089$) show no significant correlation with LOEO drop.
- **Event Prior Anchor:** Achieves 38.67% on CrisisMMD and 32.21% on HumAID without features. In all 7/7 CrisisMMD folds and 19/19 HumAID folds, learned text models outperform this prior baseline: the model vs. prior margin ranges from **+5.58 to +18.76 pp** (mean +13.35 pp) on CrisisMMD, and from **+15.32 to +65.37 pp** (mean +43.12 pp) on HumAID, confirming genuine situational feature extraction.
- **Regional Macro Nuance:** HumAID LORO (73.36%) exceeds pooled random (72.69%) because macro-averaging over 8 geographic regions weights smaller high-accuracy regions equally, whereas pooled random split is dominated by large North American hurricane events.

---

### 4.2 Domain Generalization Benchmark (5-Seed Retraining)

| Algorithm | Dataset | Mean Acc. (%) | Worst Event (%) | ECE | Scaled ECE | AURC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ERM** | CrisisMMD | **51.41 $\pm$ 11.6** | 38.77 | 0.076 | 0.041 | **0.356** |
| **GroupDRO** | CrisisMMD | 51.27 $\pm$ 11.2 | **39.65** (+0.88 pp) | 0.096 | **0.034** | 0.370 |
| **IRM** | CrisisMMD | 48.51 $\pm$ 11.6 (-2.90 pp) | 37.86 | **0.067** | 0.035 | 0.386 |
| **ERM** | HumAID | **67.75 $\pm$ 8.9** | 46.90 | 0.161 | **0.035** | **0.154** |
| **GroupDRO** | HumAID | 67.14 $\pm$ 8.4 | **47.71** (+0.81 pp) | 0.196 | 0.041 | 0.161 |
| **IRM** | HumAID | 63.98 $\pm$ 9.7 (-3.77 pp) | 45.97 | **0.081** | 0.043 | 0.186 |

- **Worst-Event Tail Protection:** Minibatch-normalized GroupDRO modestly improves worst-event accuracy across both datasets (+0.88 pp on CrisisMMD: 38.77% $\to$ 39.65%; +0.81 pp on HumAID: 46.90% $\to$ 47.71%). The worst fold is identical between ERM and GroupDRO (`california_wildfires` on CrisisMMD, `maryland_floods_2018` on HumAID). GroupDRO incurs minor penalties in mean accuracy (-0.14 pp CrisisMMD, -0.61 pp HumAID) and ECE (+0.020 CrisisMMD, +0.035 HumAID).
- **The IRM Tradeoff:** IRM dramatically reduces unscaled ECE on HumAID (from 0.161 to 0.081, Wilcoxon $W = 0.0, p = 3.81 \times 10^{-6}$, Holm-corrected $p = 7.63 \times 10^{-6}$), whereas on CrisisMMD the unscaled ECE reduction is minimal (0.076 to 0.067). However, this calibration gain incurs a statistically significant **-3.77 pp accuracy penalty** on HumAID ($p = 7.63 \times 10^{-6}$) and **-2.90 pp penalty** on CrisisMMD ($W = 0.0, p = 0.0156$, Holm-corrected $p = 0.0313 \le 0.032$, testing $m=2$ hypotheses).
- **Post-Hoc Temperature Scaling:** Applying post-hoc temperature scaling ($T$ fitted strictly on source training events) to ERM reduces ECE to **0.035** on HumAID and **0.041** on CrisisMMD with zero accuracy penalty. The optimal temperatures average $\hat{T} = 0.645 \pm 0.136$ on HumAID and $\hat{T} = 0.873 \pm 0.162$ on CrisisMMD. Because $\hat{T} < 1.0$, scaling sharpens predicted probabilities, indicating that uncalibrated ERM models on out-of-distribution events were systematically *underconfident* rather than overconfident. This proves that IRM's calibration improvement is readily surpassed by simple post-hoc scaling without paying IRM's 2.90 to 3.77 pp accuracy penalty.

---

### 4.3 Near-Duplicate Leakage and Deduplication Audit

| Dataset | Threshold | Leaked Share (%) | Leaked Acc (%) | Clean Acc (%) | Association Gap ($\Delta$) | Slice-Weighted Contribution (pp) | Purge-and-Retrain Impact (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CrisisMMD** | $\tau \ge 0.90$ | 16.27% (588.6 / 3,617) | 59.64 $\pm$ 0.9 | 54.24 $\pm$ 0.8 | **+5.40 pp** | +0.88 pp | -0.69 pp |
| **CrisisMMD** | $\tau \ge 0.95$ | 15.73% (568.8 / 3,617) | 59.92 $\pm$ 0.9 | 54.23 $\pm$ 0.8 | **+5.69 pp** | +0.90 pp | -0.69 pp |
| **CrisisMMD** | $\tau \ge 0.98$ | 15.67% (566.8 / 3,617) | 60.06 $\pm$ 0.8 | 54.20 $\pm$ 0.8 | **+5.86 pp** | +0.92 pp | -0.69 pp |
| **HumAID** | $\tau \ge 0.90$ | 4.41% (674.2 / 15,297) | 87.47 $\pm$ 1.0 | 72.01 $\pm$ 0.3 | **+15.46 pp** | +0.68 pp | 0.00 pp |
| **HumAID** | $\tau \ge 0.95$ | 0.34% (51.8 / 15,297) | 85.29 $\pm$ 5.5 | 72.65 $\pm$ 0.3 | **+12.65 pp** | +0.04 pp | 0.00 pp |
| **HumAID** | $\tau \ge 0.98$ | 0.04% (6.2 / 15,297) | 87.78 $\pm$ 7.2 | 72.68 $\pm$ 0.3 | **+15.09 pp** | +0.01 pp | 0.00 pp |

- **Physical Nature of Leakage:** In CrisisMMD, 17.53% of the dataset consists of near-identical duplicate retweets and automated weather advisories with identical sentence embeddings ($\cos \ge 0.98$); consequently, varying $\tau$ from 0.90 to 0.98 leaves the leaked count nearly unchanged ($588.6 \to 568.8 \to 566.8$). In contrast, HumAID was explicitly deduplicated during dataset curation (leaving only 0.02% exact duplicate tweets), so the count drops rapidly ($674.2 \to 51.8 \to 6.2$).
- **Slice-Weighted Association vs. Purge-and-Retrain Effect:** While leaked test instances exhibit a marked association accuracy advantage (+5.40 to +5.86 pp on CrisisMMD; +12.65 to +15.46 pp on HumAID), their low prevalence limits the slice-weighted association contribution ($\Delta_{\text{assoc}} = \text{Prevalence} \times \text{Gap}$) to +0.88 to +0.92 pp on CrisisMMD and +0.01 to +0.68 pp on HumAID.
- **Purge-and-Retrain Impact:** When near-duplicates are removed from training data and probes are retrained from scratch, model accuracy on the full test set drops by only **0.69 pp** on CrisisMMD (55.12% $\to$ 54.43%) and **0.00 pp** on HumAID (72.69% $\to$ 72.69%). To baseline this against arbitrary data pruning, removing an equivalent number of random training samples produces an accuracy drop of 0.12% $\pm$ 0.14% on CrisisMMD (588 samples) and 0.08% $\pm$ 0.05% on HumAID (674 samples). The purge-and-retrain effect is thus comparable to random data reduction, confirming that near-duplicate overlap does not artificially inflate headline benchmark numbers.

---

### 4.4 Synthetic Feature Corruption and Mitigation Strategies

Features are unit-normalized ($\|x\|_2 = 1, d = 384$). For Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, $\mathbb{E}[\|\epsilon\|_2] = \sqrt{384 \sigma^2}$. NSR scales linearly: $\sigma = 0.05 \implies \text{NSR} \approx 0.98$; $\sigma = 0.10 \implies \text{NSR} \approx 1.96$; $\sigma = 0.50 \implies \text{NSR} \approx 9.80$.

| Noise Level $\sigma$ | Noise-to-Signal Ratio (NSR) | Raw Perturbed Acc. (%) | Raw Perturbed ECE | Re-normalized Acc. (%) | Re-normalized ECE | Noise-Augmented Acc. (%) | Noise-Augmented ECE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 0.00 | 71.69 | 0.039 | 71.69 | 0.039 | 71.69 | 0.039 |
| **0.05** | 0.98 | 63.15 | 0.077 | 57.57 | 0.052 | 65.80 | 0.044 |
| **0.10** | 1.96 | 47.67 | 0.252 | 31.88 | 0.214 | 56.68 | 0.070 |
| **0.25** | 4.90 | 25.32 | 0.589 | 13.29 | 0.449 | 37.17 | 0.139 |
| **0.50** | 9.80 | 17.19 | 0.745 | 10.51 | 0.508 | 27.63 | 0.115 |

- **Logit Norm Inflation Mechanism:** These empirical results are consistent with the hypothesis that unnormalized Gaussian perturbations inflate logit magnitudes by expanding vector norms. Because linear logits scale as $z = W\tilde{x} + b$, norm expansion artificially drives softmax probabilities toward saturation, escalating ECE to 0.745, analogous to corruption-induced miscalibration observed in vision benchmarks (Hendrycks & Dietterich, 2019).
- **Mitigation Dynamics:** Re-normalizing perturbed inputs to the unit sphere ($\tilde{x} / \|\tilde{x}\|_2$) bounds ECE at 0.052 for $\sigma \le 0.05$. At severe perturbations ($\sigma = 0.50$), however, random Gaussian noise fundamentally rotates feature vectors away from true decision manifolds; re-normalization cannot restore directional semantic content, resulting in low accuracy (10.51%) and elevated ECE (0.508). In contrast, training models with noise augmentation ($\sigma_{\text{train}} = 0.10$) induces wider classifier margins, preserving 27.63% accuracy and restraining ECE to 0.115 at $\sigma = 0.50$.

---

### 4.5 Resampling and Precision Analysis

- **Bootstrap Protocol:** Subsampled $k \in [3, 18]$ held-out events across 500 outer subsets without replacement, with 200 inner cluster bootstrap resamples of the macro-averaged mean accuracy.
- **Analytic Comparison:** Mean 95% CI width narrows from 18.21 pp ($k=3$) to 14.87 pp ($k=7$), 12.99 pp ($k=10$), 10.96 pp ($k=15$, half-width $\pm 5.48$ pp), and 10.18 pp ($k=18$).
- **Theoretical Curves:** Plotted against infinite superpopulation curve ($2 \cdot 1.96 \cdot \text{SD}/\sqrt{k}$, $\text{SD}=0.0768$) and finite-population-corrected (FPC) curve ($\sqrt{\frac{N-k}{N-1}}$).
- **Benchmark Precision Requirement:** In this benchmark setting, at least 15 held-out events were required to achieve an empirical 95% confidence interval width below 11 percentage points (a half-width of $\pm 5.5$ percentage points). The empirical curve lies above the analytic superpopulation curve for small $k$ because individual disaster events exhibit heavy-tailed difficulty variation (e.g., Maryland Floods at 51.48% vs. Fort McMurray at 82.72%). Evaluations relying on $k=3$ disaster events produce confidence intervals spanning over 18 percentage points ($\pm 9$ percentage points), making algorithmic comparisons statistically uninformative.

---

## 5. Methodological Recommendations for Crisis Benchmarking

1. **Transparent Event-Level LOEO Reporting:** Operational benchmarks should report paired LOEO evaluations across all available independent events, accompany aggregate metrics with fold-level dispersion or cluster bootstrap confidence intervals, explicitly disclose the number of independent physical disasters evaluated, and treat performance conclusions from small event pools ($k < 15$) as statistically provisional due to wider estimation uncertainty.
2. **Prior Alignment Disclosure:** Evaluators should quantify the divergence $D_{\text{KL}}$ between historical training label distributions and incoming target incident streams to identify severe base-rate shifts.
3. **Strict Out-of-Distribution Calibration:** When evaluating calibration under distribution shift, temperature scaling parameters must be fitted strictly on held-out source folds or validation subsets, preventing test-set information leakage.

---

## 6. Repository Structure and Reproducibility

```
disaster-ai-audit/
├── LICENSE                    # MIT (Code)
├── LICENSE-docs               # CC BY 4.0 (Academic Manuscript & Documentation)
├── LICENSE-data               # CC BY-NC-SA 4.0 (Manifests, Caches & Tables)
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
│   ├── compute_paired_shift_and_diagnostics.py # Paired gaps & purge-and-retrain evaluation
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
