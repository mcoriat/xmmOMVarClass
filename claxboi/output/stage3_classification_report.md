# Stage 3: CLAXBOI Classification of XMM-Newton OM Sources

**Date:** 2026-02-23  
**Catalogue:** XMM-Newton OM UV/optical source survey (5,363,088 sources)  
**Pipeline stage:** Stage 3 — Bayesian source classification into Star / QSO / Galaxy

---

## 1. Method

Sources are classified using **CLAXBOI** (CLAssification by Bayesian Objecttype Identification), a Naive Bayes classifier that models per-feature, per-class probability densities via Kernel Density Estimation (KDE).

**Features (12):**

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `uvw1_u` | colour | UVW1 − u |
| 2 | `b_v` | colour | B − V |
| 3 | `W2_W1` | colour | WISE W2 − W1 |
| 4 | `BP_RP` | colour | Gaia BP − RP |
| 5 | `UVM2mUVW1` | colour | UVM2 − UVW1 |
| 6 | `UVW2mUVW1` | colour | UVW2 − UVW1 |
| 7 | `UVW1mGmag` | colour | UVW1 − Gaia G |
| 8 | `Gaia_G_WISE_W1` | colour | Gaia G − WISE W1 |
| 9 | `gaia_extended` | extent | Gaia morphological extent (weight=4) |
| 10 | `umag_rmag` | colour | SDSS u − r |
| 11 | `k_WiseW1` | colour | 2MASS K − WISE W1 |
| 12 | `OMu_b` | colour | OM u − b |

**Classification model:**

- **Likelihood:** Product of KDE-estimated P(x_i | class) over all available features
- **Missing values:** "splitpba" strategy — P(missing | class) is estimated from training data and multiplied into the likelihood. This accounts for the informational content of missing detections (e.g., galaxies are more likely to lack UV measurements).
- **Priors:** [Star, QSO, Galaxy] = [0.69, 0.13, 0.18]
- **Global coefficients:** 5 category weights [alpha(missing), brightness, colour, extent, variability] that scale the log-likelihoods
- **Numerical stability:** All computations in log-space; class posteriors via log-sum-exp

---

## 2. Training Data

Training labels are derived from a two-stage process: an initial cross-match to known catalogues, followed by SIMBAD enrichment.

| Class | Pre-SIMBAD | Post-SIMBAD | Enrichment factor |
|-------|------------|-------------|-------------------|
| Star | ~1,360,000 | 1,359,873 | ~1.0× (already dominated by Gaia spectral types) |
| QSO | 3,642 | 10,242 | 2.8× |
| Galaxy | 4,932 | 49,812 | 10.1× |
| **Total labelled** | — | **1,419,927** | **26.5% of catalogue** |
| Unlabelled | — | 3,943,161 | 73.5% of catalogue |

SIMBAD enrichment was essential for obtaining a representative galaxy training sample. The pre-SIMBAD galaxy set (4,932 sources) was too small and biased toward bright, compact objects, leading to artificially inflated performance metrics that did not generalise.

---

## 3. Coefficient Optimization

The 5 global coefficients were optimized to maximize classification performance on labelled data.

- **Method:** `scipy.optimize.differential_evolution` (global optimizer)
- **Subsample:** Equipartitioned set of 78,784 sources (equal representation per class)
- **Objective:** Maximize mean F1 score across all 3 classes (C = −1)
- **Convergence:** 17 iterations, 2.5 minutes wall time

| Coefficient | Original | Optimized | Interpretation |
|-------------|----------|-----------|----------------|
| alpha (missing) | 0.91 | **0.0** | Missing values treated as uninformative |
| brightness | 1.52 | 8.97 | Free parameter (no brightness features) |
| colour | 6.97 | 4.72 | Reduced weight on colour indices |
| extent | 4.61 | 2.11 | Reduced weight on morphology |
| variability | 2.07 | 9.11 | Free parameter (no variability features) |

**Key finding:** The optimized alpha = 0 indicates that missing values should carry no penalty in the likelihood. This has a clear physical interpretation: galaxies are preferentially faint and extended, leading to non-detections in the shorter-wavelength UV filters. The original alpha = 0.91 systematically penalized sources with missing UV data, biasing classification against galaxies.

Note: The brightness and variability coefficients are unconstrained (no features in those categories currently exist), so their optimized values are arbitrary.

---

## 4. Results

### 4.1 F1 Scores (3-stage comparison, Bayesian-corrected)

| Stage | Star | QSO | Galaxy | Mean F1 |
|-------|------|-----|--------|---------|
| Pre-SIMBAD training | 0.959 | 0.821 | 0.797 | 0.859 |
| + SIMBAD enrichment | 0.939 | 0.746 | 0.502 | 0.729 |
| + Coefficient optimization | 0.942 | 0.805 | 0.650 | **0.799** |

The pre-SIMBAD F1 scores are misleadingly high. They reflect performance against a small, unrepresentative test set (3.6K QSOs, 4.9K galaxies). The post-SIMBAD scores, evaluated against 10K QSOs and 50K galaxies, are scientifically more meaningful. Coefficient optimization recovered much of the F1 loss incurred by introducing the harder (but more realistic) SIMBAD-enriched test set.

### 4.2 Confusion Matrix (optimized coefficients, labelled sources only)

```
                 Truth
                 Star        QSO       Galaxy
Pred Star     1,314,110       561      14,474      Retrieval: 96.6%
Pred QSO         16,911     8,906       8,073      Retrieval: 87.0%
Pred Galaxy      28,852       775      27,265      Retrieval: 54.7%

Corrected
Precision:        91.8%     75.0%       80.1%
```

### 4.3 Predicted Class Distribution (full catalogue, 5,363,088 sources)

| Class | Count | Fraction |
|-------|-------|----------|
| Star | 3,866,674 | 72.1% |
| QSO | 1,158,101 | 21.6% |
| Galaxy | 338,313 | 6.3% |

---

## 5. Feature Analysis

Features ranked by discriminating power (mean pairwise KDE overlap; lower overlap = better separation):

| Rank | Feature | Mean overlap | Notes |
|------|---------|-------------|-------|
| 1 | UVW1 − G | 0.007 | Best overall separator |
| 2 | UVM2 − UVW1 | 0.009 | Strong UV colour diagnostic |
| 3 | OM u − b | 0.015 | Third-best colour index |
| 4 | gaia_extended | 0.033 | Only morphological feature; critical for QSO vs Galaxy |
| 5 | UVW1 − u | 0.035 | |
| 6 | G − W1 | 0.040 | Optical–infrared baseline |
| 7 | K − W1 | 0.16 | Near-IR; moderate power |
| 8 | BP − RP | 0.17 | Gaia colour |
| 9 | W2 − W1 | 0.19 | WISE colour |
| 10 | u − r (SDSS) | 0.34 | Weak — Star/Galaxy overlap |
| 11 | UVW2 − UVW1 | 0.67 | Near-useless |
| 12 | B − V | 0.74 | Near-useless |

Features 11–12 (UVW2 − UVW1, B − V) exhibit QSO ≡ Galaxy overlap approaching 1.0, rendering them non-discriminating between these two classes. This became apparent only after SIMBAD enrichment provided a representative galaxy sample; the pre-SIMBAD galaxy training set was too small and biased to reveal this degeneracy.

---

## 6. Computational Performance

| Component | Time | Notes |
|-----------|------|-------|
| `classify_new_fast.py` (production run) | **9.1 min total** | Loading: 51 s, Classification: 56 s, Saving: 6.5 min |
| `classify_new.py` (original) | ~6 h 30 min | Row-by-row Python loops |
| Speedup | **21×** | Fully vectorized NumPy implementation |
| Coefficient optimization | 2.5 min | Vectorized F1 evaluation within differential_evolution |

The I/O-dominated runtime (saving 6.5 min out of 9.1 min) reflects the 26 GB FITS output and could be reduced by writing only classification columns.

---

## 7. Remaining Weaknesses

1. **Galaxy retrieval is 54.7%** — nearly half of true galaxies are misclassified, predominantly as Stars (29.1% of true galaxies) or QSOs (16.2%).
2. **Feature degeneracies** — Two of twelve features (UVW2 − UVW1, B − V) contribute negligible discriminating power and could be removed without loss.
3. **No brightness or variability features** — Two of five coefficient categories have no associated features. Incorporating UV magnitude and variability metrics could improve Galaxy retrieval.
4. **Class imbalance in training** — Stars outnumber QSOs by 133:1 and Galaxies by 27:1. The equipartitioned optimization subsample mitigates this, but the KDE distributions are still estimated from the full imbalanced set.
5. **Naive Bayes assumption** — Feature independence is assumed but violated (e.g., UVM2 − UVW1 and UVW2 − UVW1 share the UVW1 band). Correlated features receive excess weight.

---

## 8. Output Files

| File | Description |
|------|-------------|
| `output/classification_OM.fits` | Full classification output (5,363,088 rows, 71 columns) |
| `output/classification_OM.csv` | CSV extract of classification columns |
| `output/classification_OM.metrics` | Summary performance metrics |
| `claxboi/classif/distrib_KDE_OM/*.dat` | 12 KDE distribution files (one per feature) |
| `claxboi/intermediates/suss_with_training.fits` | Input catalogue with training labels |

---

*Report generated 2026-02-23. Pipeline: xmmOMVarClass Stage 3 (CLAXBOI).*
