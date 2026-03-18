# xmmOMVarClass Pipeline Runbook

## Overview
Pipeline for building a single-source catalogue from the XMM-OM SUSS 6.1 survey,
cross-matched with Gaia DR3, with proper-motion correction and variability statistics.

**Input**: XMM-OM-SUSS6.1.fits (9.9M rows), Gaia DR3 high-PM catalogue
**Final output**: `output/sussxgaiadr3_ep2000_singlerecs_stg2_merged.fits` (5.36M unique sources, 239 columns)

**Python environment**: `/Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3`
(required for sklearn, scipy, astropy — system python3 lacks sklearn)

**STILTS**: `/opt/homebrew/bin/stilts`

---

## Stage 1b: Gaia DR3 Cross-Match

### What it does
Matches SUSS 6.1 sources to Gaia DR3 by propagating Gaia proper motions to each
observation epoch (0.1-year steps from 2005.0 to 2024.0), finding positional matches
within 0.33 arcsec radius. Uses HEALPix tiling for spatial acceleration.

### Command
```bash
cd /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass
/Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 -c "
from single_source import match2gaiadr3_positions2Epoch2000 as m2g
m2g.main()
"
```

### Stage 1b-ii: Finalise & Merge
```bash
/Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 -c "
from single_source import match2gaiadr3_positions2Epoch2000 as m2g
m2g.finalise()
"
```
Then copy:
```bash
cp output/matched2gaiadr3.fits output/sussxgaiadr3_ep2000.fits
```

### Output
- `output/sussxgaiadr3_ep2000.fits` — 8.4GB, 7,722,233 rows, 256 columns

### Estimated time
~8 min with parallelization (180 epochs), ~15 min for finalise/merge

---

## Stage 2.1: Single-Source Catalogue Construction

### What it does
Groups the multi-observation catalogue by IAUNAME to produce one row per unique source.
For each source, computes median magnitudes, chi-squared variability statistics, skewness,
minimum magnitude, and observation counts across all 6 UV/optical bands (UVW2, UVM2, UVW1, U, B, V).
Concatenates OBSID lists, epoch lists, and SRCNUM lists into pipe-delimited strings.

Uses an optimised `mainsub_fast()` function (4.2x speedup over original):
- Skips expensive `ma.asarray()` for single-observation sources (88% of data)
- Moves `ma.asarray()` outside the band loop for multi-obs sources
- Guards dead code behind `fix_duplicate` flag

### Command
```bash
cd /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass
/Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 -u -c "
from single_source import single_source_cat_v2 as ssc
ssc.mainsub_fast(ssc.chunk)
"
```

### Output
- `output/sussxgaiadr3_ep2000_singlerecs.csv` — 10GB, 5,364,037 rows (+1 header), 239 columns

### Estimated time
~87 min (~1030 sources/sec)

### Post-processing: CSV to FITS conversion
```bash
/opt/homebrew/bin/stilts tpipe \
  in=output/sussxgaiadr3_ep2000_singlerecs.csv ifmt=csv \
  out=output/sussxgaiadr3_ep2000_singlerecs.fits ofmt=fits-basic
```
**Time**: ~2 min. Output: 26GB FITS (large due to fixed-width string columns OBSIDS/EPOCHS/SRCNUMS)

---

## Stage 2.2: High Proper-Motion Source Merging

### What it does
Identifies sources with PM > 20 mas/yr that appear under different IAUNAMEs due to
positional shift between epochs. Uses DBSCAN clustering on proper-motion vectors,
then spatial proximity checks (BallTree, haversine metric) to group sources that are
the same physical object observed at different sky positions.

Steps:
1. STILTS extracts low-PM sources (PM ≤ 20 or null) — passed through unchanged
2. STILTS extracts high-PM sources (PM > 20) — 85,862 sources
3. Flags bad-PM sources (PM/error < 10) — 372 flagged
4. DBSCAN clusters remaining 85,490 sources by PM similarity
5. Spatial grouping within PM clusters identifies 1,875 sources → 926 merged records
6. Merges low-PM FITS + high-PM FITS into final output

### Command
```bash
cd /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass
/Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 -u -c "
from single_source import single_source_cat_mergepm_v2 as pm
pm.mainsub2(pm.chunk)
"
```

**Important**: Must use the shared venv (needs sklearn). If intermediate files
`_stg2a.fits` and `_hipm.fits` already exist, they are reused automatically.

### Output
- `output/sussxgaiadr3_ep2000_singlerecs_stg2_merged.fits` — 26GB, 5,363,088 rows, 239 columns

### Intermediate files
- `output/sussxgaiadr3_ep2000_singlerecs_stg2a.fits` — low-PM sources (5,278,175 rows)
- `output/sussxgaiadr3_ep2000_singlerecs_hipm.fits` — high-PM sources (85,862 rows)
- `output/sussxgaiadr3_ep2000_singlerecs_stg2b.csv` — high-PM processed output (84,913 rows)
- `output/sussxgaiadr3_ep2000_singlerecs_stg2b.fits` — high-PM FITS conversion

### Estimated time
- STILTS low-PM extraction: ~9 min
- STILTS high-PM extraction: ~7.5 min
- DBSCAN clustering + merging: ~1 min
- STILTS final merge: ~5 min
- **Total: ~23 min** (if run from scratch; ~6 min if STILTS extractions cached)

---

## Stage 3: CLAXBOI Bayesian Classification

### What it does
Classifies all 5.36M sources into 3 classes (Star, QSO, Galaxy) using CLAXBOI — a
Bayesian Naive Bayes classifier with KDE-estimated per-class per-feature likelihoods.

The CLAXBOI pipeline has its own preparation steps before classification:
1. **External cross-matches** (`om_crossmatch.py`): Queries CDS VizieR for AllWISE,
   SDSS DR16, 2MASS, and PanSTARRS photometry. These provide the infrared/optical
   colours used as classification features. The AllWISE query returns W1–W4 magnitudes
   + errors + AllWISE designation (3″ radius, best match).
2. **Feature computation** (`om_compute_features.py`): Computes 12 colour/extent
   features from the cross-matched photometry
3. **Training label preparation**: Builds labels from Gaia spectral types + SIMBAD
4. **KDE distributions**: Estimates P(feature|class) for 12 features × 3 classes using sklearn KernelDensity
5. **Coefficient optimization** (optional): Tunes 5 global weights via differential_evolution
6. **Classification**: Computes posterior P(class|features) in log-space, assigns most probable class

Uses `classify_new_fast.py` — a fully vectorized numpy replacement achieving 21× speedup over the original.

### Prerequisites
- Stage 2.2 output: `output/sussxgaiadr3_ep2000_singlerecs_stg2_merged.fits`
- SIMBAD-enriched training labels in `claxboi/intermediates/suss_with_training.fits`
- Feature config: `claxboi/intermediates/suss_with_training.in`

### Step 3.1: Build KDE distributions + classify
```bash
cd /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass/claxboi
PYTHONUNBUFFERED=1 /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 \
  classify_new_fast.py configfile.ini
```

Config settings for first run:
```yaml
compute_distrib: 1    # Build KDE distributions
optimize_coeffs: 0    # Skip optimization for now
save: 1               # Write output catalogue
```

### Step 3.2: Optimize global coefficients (optional)
```bash
cd /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass/claxboi
PYTHONUNBUFFERED=1 /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 \
  classify_new_fast.py configfile.ini
```

Config settings for optimization:
```yaml
compute_distrib: 0    # Reuse existing distributions
optimize_coeffs: 1    # Run differential_evolution
C: -1                 # Optimize mean F1 across all classes
save: 0               # Don't save catalogue (just optimize)
```

The optimizer auto-updates `configfile.ini` with optimal coefficients and sets
`optimize_coeffs: 0, save: 1` for the final classification run.

### Step 3.3: Final classification with optimized coefficients
```bash
cd /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass/claxboi
PYTHONUNBUFFERED=1 /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 \
  classify_new_fast.py configfile.ini
```

Config is already set to `optimize_coeffs: 0, save: 1` from Step 3.2.

### Configuration
Classification config: `claxboi/configfile.ini`
```yaml
filename: 'intermediates/suss_with_training.fits'
fileout: 'output/classification_OM.fits'
dirref: 'classif/distrib_KDE_OM/'
categories: ['brightness', 'colour', 'extent', 'variability']
global_coeffs: [0.0, 8.97, 4.72, 2.11, 9.11]    # optimized
classnames: ['Star', 'QSO', 'Galaxy']
trueprop: [0.69, 0.13, 0.18]
```

12 features (11 colour + 1 extent):
`uvw1_u, b_v, W2_W1, BP_RP, UVM2mUVW1, UVW2mUVW1, UVW1mGmag,
Gaia_G_WISE_W1, gaia_extended, umag_rmag, k_WiseW1, OMu_b`

### Output
- `output/classification_OM.fits` — 5,363,088 rows, 71 columns
- `output/classification_OM.csv` — classification columns only
- `output/classification_OM.metrics` — summary metrics
- `claxboi/classif/distrib_KDE_OM/*.dat` — 12 KDE distribution files (99 bins each)

### Results (optimized coefficients)
```
Coefficients: [0.0, 8.97, 4.72, 2.11, 9.11]
F1 scores: Star 0.942, QSO 0.805, Galaxy 0.650 (mean: 0.799)

Predicted counts:
  Star:   3,866,674 (72.1%)
  QSO:    1,158,101 (21.6%)
  Galaxy:   338,313 (6.3%)
```

### Estimated time
- Step 3.1 (distributions + classify): ~18 min
- Step 3.2 (optimization): ~2.5 min
- Step 3.3 (final classify): ~9 min
- **Total: ~30 min** (Steps 3.2 + 3.3 only if optimizing)

### Notes
- `PYTHONUNBUFFERED=1` or `python3 -u` required to see progress in real-time when
  redirecting to a log file (`> logfile.out 2>&1`)
- Optimization log written to `intermediates/suss_with_training_optimsteps.dat`
- The `classify_new_fast.py` is a drop-in replacement for `classify_new.py` (21× faster,
  bit-identical results). Key optimizations: direct FITS I/O, vectorized numpy classification,
  no CSV intermediary files.
- Alpha coefficient → 0 means missing values are treated as uninformative (P^0 = 1).
  This was critical for Galaxy classification — galaxies are faint/extended and often lack
  UV detections, so the old alpha=0.91 systematically misclassified them.

---

## Stage 4: 5XMM OM Assembly Table

### What it does
Produces a lightweight FITS table (~2.4 GB, 68 columns) for inclusion in the 5XMM
catalogue. Extracts and renames selected columns from the 26 GB pipeline output,
runs a new AllWISE cross-match (W1–W4 + errors), computes match probabilities for
Gaia and WISE associations, and derives additional columns.

Steps:
1. **STILTS column extraction**: Single-pass streaming read of the 26 GB stg2_merged
   file to extract ~56 columns into a ~7 GB intermediate file
2. **SRCNUM extraction**: Parses the first integer from the SRCNUMS string column
3. **AllWISE CDS cross-match**: Queries VizieR `II/328/allwise` via STILTS `cdsskymatch`
   (radius=3″, best match), returning W1–W4 magnitudes, errors, and AllWISE designation.
   Deduplicates on AllWISE name, then merges back onto 5.36M sources via `tmatch2`
4. **Match probabilities**: Likelihood Ratio method for both Gaia↔OM and WISE↔OM:
   `LR = exp(-r²/(2σ²)) / (2πσ²ρ)`, `P(match) = LR / (LR + 1)`
   - Gaia: uses `angDist` (separation) and `POSERR` (OM positional uncertainty),
     background density ρ ≈ 4.2×10⁻² src/arcsec²
   - WISE: uses STILTS `Separation`, combined σ = √(POSERR² + 0.5²),
     background density ρ ≈ 1.7×10⁻² src/arcsec²
5. **Derived columns**: PARALLAX_OVER_ERROR, CHI2RED pass-through, CHISQ_DOF = NOBS−1
   (set to NaN for the 926 mismatchedPM=3 sources where Stage 2 recomputed χ²)
6. **Assembly + write**: Builds the final table with standardised column names

### Command
```bash
cd /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass
/Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3 \
  assemble_5xmm_om_table.py [--resume] [--skip-wise]
```

Flags:
- `--resume`: Reuse intermediate files (STILTS extraction, AllWISE query/merge)
  if they already exist. Safe to restart after interruption.
- `--skip-wise`: Skip AllWISE cross-match entirely (WISE columns filled with NaN)
- `--output PATH`: Override default output path

### Prerequisites
- Stage 2.2 output: `output/sussxgaiadr3_ep2000_singlerecs_stg2_merged.fits`
- Stage 3 output: `claxboi/output/classification_OM.fits`
- Slim catalogue: `claxboi/intermediates/suss_slim.fits` (used as input for CDS query)

### AllWISE cross-match details

The AllWISE cross-match is originally performed during Stage 3 preparation
(`claxboi/om_crossmatch.py`, Step 1 of the CLAXBOI pipeline), which queries
AllWISE, SDSS DR16, 2MASS, and PanSTARRS via CDS to build external photometry
for classification features. However, the initial run only kept W1mag and W2mag.
The pipeline code (`om_crossmatch.py`) has since been updated to keep all 4 bands
plus errors, but the current `suss_with_extphot.fits` intermediate still reflects
the old query. This stage re-queries CDS to obtain the full set:

| Column | Description |
|--------|-------------|
| AllWISE | AllWISE designation (e.g. J000000.00+000000.0) |
| W1mag, W2mag, W3mag, W4mag | WISE magnitudes (3.4, 4.6, 12, 22 μm) |
| e_W1mag, e_W2mag, e_W3mag, e_W4mag | Magnitude errors |

The query uses `suss_slim.fits` (5.36M unique sources, 24 columns) as input.
CDS `cdsskymatch` with `find=best` returns only the single closest AllWISE match
within 3″. After deduplication on AllWISE name (removing ~43k duplicates),
sources are merged back by sky position via STILTS `tmatch2`.

Match rate: ~29% (1.56M / 5.36M) — many UV-selected OM sources lack infrared
counterparts, which is expected.

### Output columns (68 total)

| Group | Columns | Count |
|-------|---------|-------|
| OM photometry | OM_{band}_AB_MAG, _ERR, _QUALITY_FLAG, _EXTENDED_FLAG, _CHISQ, _CHISQ_DOF, _CHI2RED × 6 bands | 42 |
| OM source ID | OM_SRCID | 1 |
| WISE photometry | WISE_NAME, W1–W4 MAG + ERR, WISE_MATCH_PROB | 10 |
| Gaia DR3 | SOURCE_ID, PARALLAX, _ERROR, _OVER_ERROR, PM_RA, PM_DEC, GMAG, BPMAG, RPMAG, MATCH_PROB, DIST | 11 |
| Classification | CLASSOPT_CLASS, PROB_STAR, PROB_AGN, PROB_GALAXY | 4 |

### CHISQ_DOF caveat
For 926 sources with `mismatchedPM = 3` (high-PM sources merged in Stage 2.2),
the χ² was recomputed from PM-group medians but `NOBS` stores the accumulated
total from Stage 1. `CHISQ_DOF` is set to NaN for these sources. The `CHI2RED`
column (= χ²/nObs as computed by the pipeline) is always available as an alternative.

### Intermediate files
- `output/intermediates/stg2_slim_extract.fits` — ~7 GB, 56 columns from stg2_merged
- `output/intermediates/xmatch_allwise_trimmed.fits` — deduplicated CDS output
- `output/intermediates/xmatch_allwise_merged.fits` — AllWISE merged with suss_slim

### Output
- `output/5xmm_om_assembly.fits` — 5,363,088 rows, 68 columns, 2.39 GB

### Estimated time
- STILTS extraction: ~40 sec
- CDS AllWISE query: ~2.5 min (may vary with CDS load)
- Assembly + write: ~30 sec
- **Total: ~4 min** (with `--resume` after first run: ~1 min)

---

## Planned Pipeline Modifications

The following changes are planned for future pipeline runs to avoid the need
for the Stage 4 patch script:

### 1. Expanded WISE columns (`claxboi/om_crossmatch.py`) — DONE
- ~~Add W3mag, W4mag, e_W1mag–e_W4mag to `keep_cols`~~ — already updated in code
- Keep `AllWISE` designation column (currently dropped during dedup)
- Keep `Separation` column (currently dropped — needed for match probability)
- **Note**: Code is updated but current `suss_with_extphot.fits` still reflects
  the old query. Will take effect on next full pipeline re-run.

### 2. Keep Gaia match separation (`single_source/stilts_fin_match2gaia.sh`)
- Remove `Separation` from the `delcols` command at line 27 (needed for Gaia match prob)
- Remove `parallax_over_error` from `delcols` at line 16 (needed for output)

### 3. Optional fixed-width columns (`single_source/single_source_cat_v2.py`)
- Add `SRCNUM_INT` integer column alongside the existing SRCNUMS string
- Make OBSIDS, EPOCHS, SRCNUMS fixed-width string columns optional via config flag
  (these account for 19.1 GB of the 26 GB output file)

---

## Directory Structure

```
xmmOMVarClass/
├── README.md, PIPELINE.md, .gitignore, requirements.txt
├── assemble_5xmm_om_table.py     # Stage 4: 5XMM assembly (patch script)
├── data/                          # Input catalogues
│   ├── XMM-OM-SUSS6.1.fits       #   SUSS 6.1 (9.9M rows)
│   ├── XMM-OM-SUSS6.1ep.fits     #   Epoch version
│   ├── gaiadr3_pmgt25.fits        #   Gaia DR3 high-PM sources
│   └── summary.fits               #   SUSS summary
├── single_source/                 # Stages 1b, 2.1, 2.2
│   ├── config.ini                 #   Pipeline configuration
│   ├── match2gaiadr3_positions2Epoch2000.py  # Stage 1b
│   ├── single_source_cat_v2.py    #   Stage 2.1
│   ├── single_source_cat_mergepm_v2.py      # Stage 2.2
│   └── augmentation/              #   Cross-match shell scripts
├── claxboi/                       # Stage 3: Bayesian classification
│   ├── classify_new_fast.py       #   Vectorized classifier (21× faster)
│   ├── configfile.ini             #   Classification config
│   ├── classif/distrib_KDE_OM/    #   KDE distributions (12 features)
│   ├── intermediates/             #   Working data
│   └── output/                    #   Classification outputs
├── variability/                   # Variability analysis module
│   └── xmm2athena.py             #   Main variability code
├── output/                        # Final pipeline outputs
│   ├── sussxgaiadr3_ep2000_singlerecs_stg2_merged.fits  # Stage 2.2 (26 GB)
│   ├── 5xmm_om_assembly.fits     #   Stage 4 output (2.4 GB)
│   └── intermediates/             #   Stage 4 working files
└── tests/                         # Test data and utilities
```

## Configuration

Pipeline configuration in `single_source/config.ini`:

```ini
[DEFAULT]
ROOT = /Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/xmmOMVarClass
TOPCATPATH = /opt/homebrew/bin
suss_dir = data/
suss_summary = summary.fits

[match2gaia]     # Stage 1b settings
[single_source]  # Stage 2.1 settings
[mergepm]        # Stage 2.2 settings
```

Classification configuration in `claxboi/configfile.ini` (see Stage 3 section).

## Row Count Summary

| Stage | Rows | Columns | Size | Description |
|-------|------|---------|------|-------------|
| Input SUSS 6.1 | 9,938,983 | — | — | Raw observations |
| After Gaia match (Stage 1b) | 7,722,233 | 256 | 8.4 GB | Matched to Gaia DR3 |
| Unique sources (Stage 2.1) | 5,364,037 | 239 | 26 GB | One row per IAUNAME |
| After PM merge (Stage 2.2) | 5,363,088 | 239 | 26 GB | High-PM duplicates merged |
| After classification (Stage 3) | 5,363,088 | 71 | 2.7 GB | + classification columns |
| 5XMM assembly (Stage 4) | 5,363,088 | 68 | 2.4 GB | Lightweight output for 5XMM |

## Known Issues / Notes
- The 26GB FITS files are large due to STILTS storing OBSIDS/EPOCHS/SRCNUMS as
  fixed-width strings (up to 1308 chars). These columns contain concatenated lists
  for multi-observation sources.
- System python3 (`/usr/bin/python3`) lacks sklearn — always use the shared venv.
- Stage 2.2 `fileio()` function is now dead code (replaced by STILTS pre-extraction
  in `mainsub2()`). Kept for backward compatibility.
- The `stats()` function has a latent bug: default `err=[None]` would crash if
  called without the `err` kwarg (`.all()` on a plain list). In practice, it's
  always called with `err=errx`.
- **CHISQ_DOF unreliable for 926 mismatchedPM=3 sources**: When Stage 2.2 merges
  high-PM groups, it recomputes χ² from PM-group medians but stores the accumulated
  `NOBS` from Stage 1 (sum of per-group observation counts). This means
  `DOF = NOBS - 1` overestimates the true degrees of freedom for these sources.
  The assembly script (Stage 4) sets DOF = NaN for them. `CHI2RED` (= χ²/nObs as
  computed by the pipeline) remains valid and is included as a separate column.
- **AllWISE CDS match rate is ~29%**: The pipeline cross-match via `suss_slim.fits`
  (unique sources) yields 1.56M matches out of 5.36M. This is expected — many
  UV-selected OM sources are too faint in the infrared for AllWISE detection.
