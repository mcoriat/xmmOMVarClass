#!/usr/bin/env bash
# ==========================================================================
#  CLAXBOI OM Classification Pipeline — Production Run Script
#
#  This script runs the full 3-step classification pipeline on the
#  5,363,088-source SUSS slim catalogue.
#
#  Prerequisites:
#    - Python 3.9+ with: astropy, numpy, scipy, sklearn, tqdm, pyyaml
#    - STILTS on PATH (or set STILTS=/path/to/stilts)
#    - Input file: intermediates/suss_slim.fits (24 cols, 5.36M rows, 1GB)
#
#  Usage:
#    cd /path/to/claxboi/
#    bash run_pipeline.sh            # run all steps
#    bash run_pipeline.sh --step 2   # run from step 2 onwards
#    bash run_pipeline.sh --step 3   # run only step 3 (classification)
#
#  Estimated runtimes (40 cores, 315GB RAM):
#    Step 1: 12-24h (CDS network queries — run overnight)
#    Step 2:  6-12h (SIMBAD queries — run overnight)
#    Step 3:  ~10min (classification — vectorized numpy, single core)
#    Total:  ~24-48h across 2-3 days (dominated by network I/O)
# ==========================================================================

set -euo pipefail

# ---- Configuration -------------------------------------------------------
PYTHON="${PYTHON:-/Users/mcoriat/Desktop/XMM-SSC/5XMM/Classification/venv/bin/python3}"
STILTS_CMD="${STILTS:-stilts}"
START_STEP="${1:-1}"

# Parse --step N argument
if [[ "${1:-}" == "--step" ]]; then
    START_STEP="${2:-1}"
fi

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASEDIR"

LOG="pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# ---- Pre-flight checks ---------------------------------------------------
log "=== CLAXBOI OM Pipeline — Production Run ==="
log "Python:  $($PYTHON --version 2>&1)"
log "STILTS:  $($STILTS_CMD -version 2>&1 | head -1 || echo 'not found')"
log "Working: $BASEDIR"
log "Start:   step $START_STEP"
log ""

# Create output directories if they don't exist
mkdir -p intermediates classif/distrib_KDE_OM output

# Check required files
if [[ ! -f intermediates/suss_slim.fits ]]; then
    log "ERROR: intermediates/suss_slim.fits not found!"
    log "  Please copy the slim catalogue to this location first."
    exit 1
fi

# Check Python dependencies
$PYTHON -c "import astropy, numpy, scipy, sklearn, tqdm, yaml" 2>/dev/null || {
    log "ERROR: Missing Python dependencies. Need: astropy numpy scipy sklearn tqdm pyyaml"
    exit 1
}

# Ensure the .in file is in the right place for classify_new.py
if [[ ! -f intermediates/suss_with_training.in ]]; then
    cp suss_with_training.in intermediates/suss_with_training.in
fi

# ---- Step 1: CDS Cross-matching ------------------------------------------
if [[ $START_STEP -le 1 ]]; then
    log "============================================"
    log "STEP 1: CDS Cross-matching (4 catalogues)"
    log "  This queries VizieR remotely — expect 12-24h"
    log "============================================"

    $PYTHON om_crossmatch.py --resume 2>&1 | tee -a "$LOG"

    if [[ ! -f intermediates/suss_with_extphot.fits ]]; then
        log "ERROR: Step 1 failed — output not produced"
        exit 1
    fi
    log "Step 1 COMPLETE: intermediates/suss_with_extphot.fits"
    log ""
fi

# ---- Step 2: Feature computation + training labels -----------------------
if [[ $START_STEP -le 2 ]]; then
    log "============================================"
    log "STEP 2: Feature computation + training labels"
    log "  Includes SIMBAD cross-match (~6-12h)"
    log "============================================"

    $PYTHON om_compute_features.py 2>&1 | tee -a "$LOG"

    if [[ ! -f intermediates/suss_with_training.fits ]]; then
        log "ERROR: Step 2 failed — output not produced"
        exit 1
    fi
    log "Step 2 COMPLETE: intermediates/suss_with_training.fits"
    log ""
fi

# ---- Step 3: Bayesian classification -------------------------------------
if [[ $START_STEP -le 3 ]]; then
    log "============================================"
    log "STEP 3: Bayesian KDE classification"
    log "  12 features, 3 classes (Star/QSO/Galaxy)"
    log "============================================"

    PYTHONUNBUFFERED=1 $PYTHON classify_new_fast.py configfile.ini 2>&1 | tee -a "$LOG"

    if [[ ! -f output/classification_OM.fits ]]; then
        log "ERROR: Step 3 failed — output not produced"
        exit 1
    fi
    log "Step 3 COMPLETE: output/classification_OM.fits"
    log ""
fi

# ---- Summary --------------------------------------------------------------
log "============================================"
log "PIPELINE COMPLETE"
log "============================================"
log ""
log "Output files:"
[[ -f output/classification_OM.fits ]] && \
    log "  output/classification_OM.fits  (classified catalogue)"
[[ -f output/classification_OM.metrics ]] && \
    log "  output/classification_OM.metrics  (F1 scores + confusion matrix)"
[[ -f output/classification_OM.csv ]] && \
    log "  output/classification_OM.csv  (CSV version)"
log ""
log "KDE distributions: classif/distrib_KDE_OM/*.dat (12 files)"
log "Full log: $LOG"
