#!/usr/bin/env python3
"""
Assemble the lightweight 5XMM OM table from existing pipeline outputs.

Reads:
  - output/sussxgaiadr3_ep2000_singlerecs_stg2_merged.fits  (26 GB)
  - claxboi/output/classification_OM.fits                    (2.7 GB)
  - claxboi/intermediates/suss_slim.fits                     (1 GB, for WISE CDS query)

Uses STILTS for fast column extraction from the 26 GB file (single-pass),
then Python for computed columns and final assembly.

Produces:
  - output/5xmm_om_assembly.fits  (~2-3 GB, ~66 columns)

The AllWISE CDS cross-match (Step 2) is the bottleneck (~6-12 h).
Use --resume to reuse the intermediate AllWISE file if it already exists.

Usage:
    cd /path/to/xmmOMVarClass
    python3 assemble_5xmm_om_table.py [--resume] [--skip-wise]
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from astropy.table import Table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASEDIR = os.path.dirname(os.path.abspath(__file__))

STG2_FILE = os.path.join(
    BASEDIR, "output/sussxgaiadr3_ep2000_singlerecs_stg2_merged.fits")
CLASS_FILE = os.path.join(BASEDIR, "claxboi/output/classification_OM.fits")
SLIM_FILE = os.path.join(BASEDIR, "claxboi/intermediates/suss_slim.fits")
OUTPUT_FILE = os.path.join(BASEDIR, "output/5xmm_om_assembly.fits")

# Intermediates directory
INTERMEDIATES = os.path.join(BASEDIR, "output/intermediates")

# AllWISE intermediate files
WISE_RAW = os.path.join(INTERMEDIATES, "xmatch_allwise_raw.fits")
WISE_TRIMMED = os.path.join(INTERMEDIATES, "xmatch_allwise_trimmed.fits")
WISE_MERGED = os.path.join(INTERMEDIATES, "xmatch_allwise_merged.fits")

# STILTS-extracted slim version of stg2_merged
STG2_SLIM = os.path.join(INTERMEDIATES, "stg2_slim_extract.fits")

STILTS = "stilts"
WISE_RADIUS = 3  # arcsec

BANDS = ["UVW2", "UVM2", "UVW1", "U", "B", "V"]

# Background source densities (sources / arcsec²)
RHO_GAIA = 1.8e9 / (41253.0 * 3600.0**2)   # ~4.2e-2
RHO_WISE = 747e6 / (41253.0 * 3600.0**2)    # ~1.7e-2
SIGMA_WISE = 0.5  # WISE typical positional uncertainty (arcsec)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def elapsed(t0):
    dt = time.time() - t0
    h, rem = divmod(int(dt), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_stilts(cmd, description=""):
    """Run a STILTS command, printing the invocation."""
    log(f"  [{description}] Running STILTS...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ERROR: {result.stderr}")
        raise RuntimeError(f"STILTS {description} failed")
    log(f"  [{description}] Done in {elapsed(t0)}")


# ---------------------------------------------------------------------------
# Step 0: Extract needed columns from stg2_merged via STILTS
# ---------------------------------------------------------------------------

def extract_stg2_columns(resume=False):
    """
    Use STILTS tpipe to extract only the needed columns from the 26 GB
    stg2_merged file in a single streaming pass.  Result: ~2 GB file.
    """
    if resume and os.path.isfile(STG2_SLIM):
        log("  Reusing existing stg2 slim extract (--resume)")
        return

    log("Step 0: Extracting needed columns from stg2_merged via STILTS...")

    # Build the list of columns to keep
    keep_cols = ["IAUNAME", "SRCNUMS", "POSERR", "angDist"]
    for band in BANDS:
        keep_cols.extend([
            f"{band}_AB_MAG", f"{band}_AB_MAG_ERR",
            f"{band}_QUALITY_FLAG", f"{band}_EXTENDED_FLAG",
            f"{band}_CHISQ", f"{band}_NOBS",
        ])
    keep_cols.extend([
        "Source", "Plx", "e_Plx", "pmRA", "pmDE",
        "Gmag", "BPmag", "RPmag", "Dist",
    ])

    keepcols_expr = " ".join(keep_cols)
    cmd = [
        STILTS, "tpipe",
        f"in={STG2_FILE}", "ifmt=fits",
        f"out={STG2_SLIM}", "ofmt=fits",
        f"cmd=keepcols \"{keepcols_expr}\"",
    ]
    run_stilts(cmd, description="extract stg2 columns")

    fsize = os.path.getsize(STG2_SLIM) / 1e9
    log(f"  Extracted to {STG2_SLIM} ({fsize:.2f} GB)")


# ---------------------------------------------------------------------------
# Step 1: Extract SRCNUM integer from SRCNUMS string
# ---------------------------------------------------------------------------

def extract_srcnum(srcnums_col):
    """Extract the (unique) integer SRCNUM from the SRCNUMS string column."""
    log("Step 1: Extracting SRCNUM from SRCNUMS column...")
    t0 = time.time()

    n = len(srcnums_col)
    srcnum = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        s = srcnums_col[i]
        if isinstance(s, bytes):
            s = s.decode("ascii", errors="ignore")
        s = str(s).strip()
        if s and s != "--":
            srcnum[i] = int(s.split("_")[0])

    valid = np.sum(srcnum >= 0)
    log(f"  SRCNUM extracted: {valid:,}/{n:,} valid  ({elapsed(t0)})")
    return srcnum


# ---------------------------------------------------------------------------
# Step 2: AllWISE expanded cross-match via STILTS
# ---------------------------------------------------------------------------

def run_wise_crossmatch(resume=False):
    """
    Query AllWISE via CDS cdsskymatch to get W1-W4 + errors + designation.
    Returns the path to the merged result file aligned with suss_slim rows.
    """
    log("Step 2: AllWISE expanded cross-match")

    os.makedirs(INTERMEDIATES, exist_ok=True)

    # --- 2a. CDS query ---
    if resume and os.path.isfile(WISE_TRIMMED):
        log("  Reusing existing trimmed AllWISE file (--resume)")
    else:
        if resume and os.path.isfile(WISE_RAW):
            log("  Reusing existing raw CDS output (--resume)")
        else:
            log("  Querying CDS AllWISE (this will take several hours)...")
            t0 = time.time()
            cmd = [
                STILTS, "cdsskymatch",
                f"in={SLIM_FILE}",
                f"out={WISE_RAW}",
                "ra=RA", "dec=DEC",
                "cdstable=II/328/allwise",
                "find=best",
                f"radius={WISE_RADIUS}",
            ]
            log(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            log(f"  CDS query complete ({elapsed(t0)})")

        # --- 2b. Deduplicate + trim columns ---
        log("  Deduplicating and trimming columns...")
        t0 = time.time()
        raw = Table.read(WISE_RAW)
        n_before = len(raw)

        # cdsskymatch prepends input columns (24 from suss_slim)
        n_input_cols = 24
        cat_colnames = raw.colnames[n_input_cols:]

        ra_xm = raw["RA"].data.copy()
        dec_xm = raw["DEC"].data.copy()
        raw = raw[cat_colnames]

        # Deduplicate on AllWISE designation
        if "AllWISE" in raw.colnames:
            _, unique_idx = np.unique(raw["AllWISE"], return_index=True)
            unique_idx.sort()
            n_after = len(unique_idx)
            raw = raw[unique_idx]
            ra_xm = ra_xm[unique_idx]
            dec_xm = dec_xm[unique_idx]
            log(f"  Dedup: {n_before:,} -> {n_after:,} "
                f"({n_before - n_after:,} duplicates removed)")

        keep = ["AllWISE", "W1mag", "W2mag", "W3mag", "W4mag",
                "e_W1mag", "e_W2mag", "e_W3mag", "e_W4mag", "Separation"]
        available = [c for c in keep if c in raw.colnames]
        missing = set(keep) - set(available)
        if missing:
            log(f"  WARNING: columns not found: {missing}")
        trimmed = raw[available]
        trimmed["RA_xm"] = ra_xm
        trimmed["DEC_xm"] = dec_xm
        trimmed.write(WISE_TRIMMED, overwrite=True)
        log(f"  Trimmed to {len(available)} columns, "
            f"{len(trimmed):,} rows ({elapsed(t0)})")

        if os.path.isfile(WISE_RAW):
            os.remove(WISE_RAW)
            log("  Removed raw CDS file")

    # --- 2c. Merge back onto suss_slim by sky match ---
    if resume and os.path.isfile(WISE_MERGED):
        log("  Reusing existing merged AllWISE file (--resume)")
    else:
        log("  Merging AllWISE onto suss_slim via STILTS tmatch2...")
        t0 = time.time()
        cmd = [
            STILTS, "tmatch2",
            f"in1={SLIM_FILE}",
            f"in2={WISE_TRIMMED}",
            "matcher=sky", f"params={WISE_RADIUS}",
            "values1=RA DEC",
            "values2=RA_xm DEC_xm",
            "find=best1", "join=all1",
            f"out={WISE_MERGED}",
        ]
        subprocess.run(cmd, check=True)

        # Clean up join metadata columns via STILTS
        cmd_clean = [
            STILTS, "tpipe",
            f"in={WISE_MERGED}", "ifmt=fits",
            f"out={WISE_MERGED}", "ofmt=fits",
            "cmd=delcols \"RA_xm DEC_xm GroupSize GroupID\"",
        ]
        # GroupSize/GroupID may not exist; ignore errors
        subprocess.run(cmd_clean, capture_output=True)

        tab = Table.read(WISE_MERGED)
        n_matched = np.sum(np.isfinite(
            tab["W1mag"].data.astype(float))) if "W1mag" in tab.colnames else 0
        log(f"  Merged: {n_matched:,}/{len(tab):,} matched ({elapsed(t0)})")

    return WISE_MERGED


# ---------------------------------------------------------------------------
# Step 3: Compute match probabilities (Likelihood Ratio)
# ---------------------------------------------------------------------------

def compute_match_probability(separation, sigma, rho):
    """
    Likelihood Ratio match probability.

    LR = exp(-r²/(2σ²)) / (2πσ²ρ)
    reliability = LR / (LR + 1)

    Parameters
    ----------
    separation : array, arcsec
    sigma : array or scalar, arcsec (combined positional uncertainty)
    rho : float, sources/arcsec² (background source density)

    Returns
    -------
    reliability : array, P(real match) in [0, 1], NaN where unmatched.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.exp(-separation**2 / (2.0 * sigma**2)) / (
            2.0 * np.pi * sigma**2 * rho)
        reliability = lr / (lr + 1.0)
    reliability[~np.isfinite(separation)] = np.nan
    return reliability


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble lightweight 5XMM OM table from pipeline outputs")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse intermediate files if they exist")
    parser.add_argument(
        "--skip-wise", action="store_true",
        help="Skip AllWISE cross-match (WISE columns will be NaN)")
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help=f"Output FITS file (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    log("=" * 60)
    log("5XMM OM Assembly Table")
    log("=" * 60)
    t_start = time.time()

    # --- Check input files ---
    for label, path in [("stg2_merged", STG2_FILE),
                        ("classification", CLASS_FILE),
                        ("suss_slim", SLIM_FILE)]:
        if not os.path.isfile(path):
            log(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    os.makedirs(INTERMEDIATES, exist_ok=True)

    # === Step 0: Extract columns from 26 GB file via STILTS ===
    extract_stg2_columns(resume=args.resume)

    # --- Load the slim extract (~2 GB) ---
    log("Loading stg2 slim extract...")
    t0 = time.time()
    stg2 = Table.read(STG2_SLIM)
    nrows = len(stg2)
    log(f"  {nrows:,} rows, {len(stg2.colnames)} columns ({elapsed(t0)})")

    # --- Load classification table ---
    log("Loading classification_OM.fits...")
    t0 = time.time()
    class_tab = Table.read(CLASS_FILE)
    log(f"  {len(class_tab):,} rows, {len(class_tab.colnames)} columns "
        f"({elapsed(t0)})")

    # --- Verify row alignment ---
    log("Verifying row alignment (IAUNAME)...")

    def _to_u22(col):
        return np.char.strip(np.array(col, dtype="U22"))

    s1 = _to_u22(stg2["IAUNAME"][:1000])
    s2 = _to_u22(class_tab["IAUNAME"][:1000])
    assert np.all(s1 == s2), "IAUNAME mismatch in first 1000 rows!"
    s1 = _to_u22(stg2["IAUNAME"][-1000:])
    s2 = _to_u22(class_tab["IAUNAME"][-1000:])
    assert np.all(s1 == s2), "IAUNAME mismatch in last 1000 rows!"
    log("  Row alignment verified")

    # === Step 1: SRCNUM ===
    srcnum = extract_srcnum(stg2["SRCNUMS"])

    # === Step 2: AllWISE cross-match ===
    wise_tab = None
    if not args.skip_wise:
        wise_path = run_wise_crossmatch(resume=args.resume)
        wise_tab = Table.read(wise_path)
        assert len(wise_tab) == nrows, \
            f"WISE row count mismatch: {len(wise_tab)} vs {nrows}"

    # === Step 3: Match probabilities ===
    log("Step 3: Computing match probabilities (LR method)...")

    angdist = np.array(stg2["angDist"], dtype=np.float64)
    poserr = np.array(stg2["POSERR"], dtype=np.float64)

    gaia_prob = compute_match_probability(angdist, poserr, RHO_GAIA)
    n_gaia = np.sum(np.isfinite(gaia_prob))
    log(f"  GAIA_MATCH_PROB: {n_gaia:,} valid, "
        f"median={np.nanmedian(gaia_prob):.4f}")

    wise_prob = np.full(nrows, np.nan)
    if wise_tab is not None and "Separation" in wise_tab.colnames:
        wise_sep = np.array(wise_tab["Separation"], dtype=np.float64)
        sigma_wise = np.sqrt(poserr**2 + SIGMA_WISE**2)
        wise_prob = compute_match_probability(wise_sep, sigma_wise, RHO_WISE)
        n_wise = np.sum(np.isfinite(wise_prob))
        log(f"  WISE_MATCH_PROB: {n_wise:,} valid, "
            f"median={np.nanmedian(wise_prob):.4f}")

    # === Step 4: Derived columns ===
    log("Step 4: Computing derived columns...")

    plx = np.array(stg2["Plx"], dtype=np.float64)
    e_plx = np.array(stg2["e_Plx"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        parallax_over_error = np.where(e_plx > 0, plx / e_plx, np.nan)

    dist = np.array(stg2["Dist"], dtype=np.float64)

    chisq_dof = {}
    for band in BANDS:
        nobs = np.array(stg2[f"{band}_NOBS"], dtype=np.float64)
        chisq_dof[band] = np.where(nobs > 1, nobs - 1, np.nan)

    # === Step 5: Assemble output table ===
    log("Step 5: Assembling output table...")
    t0 = time.time()

    out = Table()

    # -- OM_SRCID --
    out["OM_SRCID"] = srcnum
    out["OM_SRCID"].description = \
        "OM source ID in SUSS catalogue (=SRCNUM)"

    # -- OM band columns --
    for band in BANDS:
        out[f"OM_{band}_AB_MAG"] = np.array(
            stg2[f"{band}_AB_MAG"], dtype=np.float64)
        out[f"OM_{band}_AB_MAG"].description = \
            f"OM {band} band magnitude (AB system)"

        out[f"OM_{band}_AB_MAG_ERR"] = np.array(
            stg2[f"{band}_AB_MAG_ERR"], dtype=np.float64)
        out[f"OM_{band}_AB_MAG_ERR"].description = \
            f"OM {band} band magnitude error (AB system)"

        out[f"OM_{band}_QUALITY_FLAG"] = np.array(
            stg2[f"{band}_QUALITY_FLAG"], dtype=np.int16)
        out[f"OM_{band}_QUALITY_FLAG"].description = \
            f"OM {band} band source quality flag"

        out[f"OM_{band}_EXTENDED_FLAG"] = np.array(
            stg2[f"{band}_EXTENDED_FLAG"], dtype=np.int16)
        out[f"OM_{band}_EXTENDED_FLAG"].description = \
            f"OM {band} band source extent flag"

        out[f"OM_{band}_CHISQ"] = np.array(
            stg2[f"{band}_CHISQ"], dtype=np.float64)
        out[f"OM_{band}_CHISQ"].description = \
            f"OM {band} band variability chi-squared"

        out[f"OM_{band}_CHISQ_DOF"] = chisq_dof[band]
        out[f"OM_{band}_CHISQ_DOF"].description = \
            f"OM {band} band chi-squared degrees of freedom"

    # -- WISE columns --
    wise_col_map = [
        ("AllWISE", "WISE_NAME", "AllWISE catalog name"),
        ("W1mag", "WISE_W1MAG", "WISE W1 magnitude"),
        ("W2mag", "WISE_W2MAG", "WISE W2 magnitude"),
        ("W3mag", "WISE_W3MAG", "WISE W3 magnitude"),
        ("W4mag", "WISE_W4MAG", "WISE W4 magnitude"),
        ("e_W1mag", "WISE_W1MAG_ERR", "WISE W1 magnitude error"),
        ("e_W2mag", "WISE_W2MAG_ERR", "WISE W2 magnitude error"),
        ("e_W3mag", "WISE_W3MAG_ERR", "WISE W3 magnitude error"),
        ("e_W4mag", "WISE_W4MAG_ERR", "WISE W4 magnitude error"),
    ]
    for src_col, out_col, desc in wise_col_map:
        if wise_tab is not None and src_col in wise_tab.colnames:
            out[out_col] = wise_tab[src_col]
        elif "NAME" in out_col:
            out[out_col] = np.full(nrows, "", dtype="U22")
        else:
            out[out_col] = np.full(nrows, np.nan, dtype=np.float64)
        out[out_col].description = desc

    out["WISE_MATCH_PROB"] = wise_prob
    out["WISE_MATCH_PROB"].description = \
        "WISE-OM match probability (LR method)"

    # -- Gaia columns --
    out["GAIADR3_SOURCE_ID"] = np.array(stg2["Source"], dtype=np.int64)
    out["GAIADR3_SOURCE_ID"].description = "Gaia DR3 source ID"

    out["GAIADR3_PARALLAX"] = plx
    out["GAIADR3_PARALLAX"].description = "Gaia DR3 parallax (mas)"

    out["GAIADR3_PARALLAX_ERROR"] = e_plx
    out["GAIADR3_PARALLAX_ERROR"].description = \
        "Gaia DR3 parallax error (mas)"

    out["GAIADR3_PARALLAX_OVER_ERROR"] = parallax_over_error
    out["GAIADR3_PARALLAX_OVER_ERROR"].description = \
        "Gaia DR3 parallax / parallax_error"

    out["GAIADR3_PM_RA"] = np.array(stg2["pmRA"], dtype=np.float64)
    out["GAIADR3_PM_RA"].description = \
        "Gaia DR3 proper motion RA (mas/yr)"

    out["GAIADR3_PM_DEC"] = np.array(stg2["pmDE"], dtype=np.float64)
    out["GAIADR3_PM_DEC"].description = \
        "Gaia DR3 proper motion Dec (mas/yr)"

    out["GAIADR3_GMAG"] = np.array(stg2["Gmag"], dtype=np.float64)
    out["GAIADR3_GMAG"].description = "Gaia DR3 G magnitude"

    out["GAIADR3_BPMAG"] = np.array(stg2["BPmag"], dtype=np.float64)
    out["GAIADR3_BPMAG"].description = "Gaia DR3 BP magnitude"

    out["GAIADR3_RPMAG"] = np.array(stg2["RPmag"], dtype=np.float64)
    out["GAIADR3_RPMAG"].description = "Gaia DR3 RP magnitude"

    out["GAIA_MATCH_PROB"] = gaia_prob
    out["GAIA_MATCH_PROB"].description = \
        "Gaia-OM match probability (LR method)"

    out["GAIADR3_DIST"] = dist
    out["GAIADR3_DIST"].description = \
        "Gaia DR3 distance (pc, Bailer-Jones)"

    # -- Classification columns --
    out["CLASSOPT_CLASS"] = np.array(
        class_tab["prediction"], dtype=np.int16)
    out["CLASSOPT_CLASS"].description = \
        "Optical classification (0=Star, 1=QSO, 2=Galaxy)"

    out["CLASSOPT_PROB_STAR"] = np.array(
        class_tab["PbaC0"], dtype=np.float64)
    out["CLASSOPT_PROB_STAR"].description = "Posterior probability: Star"

    out["CLASSOPT_PROB_AGN"] = np.array(
        class_tab["PbaC1"], dtype=np.float64)
    out["CLASSOPT_PROB_AGN"].description = "Posterior probability: AGN/QSO"

    out["CLASSOPT_PROB_GALAXY"] = np.array(
        class_tab["PbaC2"], dtype=np.float64)
    out["CLASSOPT_PROB_GALAXY"].description = \
        "Posterior probability: Galaxy"

    log(f"  Assembled: {len(out):,} rows, {len(out.colnames)} columns "
        f"({elapsed(t0)})")

    # === Step 6: Write output ===
    log(f"Step 6: Writing {args.output}...")
    t0 = time.time()
    out.write(args.output, format="fits", overwrite=True)
    fsize = os.path.getsize(args.output) / 1e9
    log(f"  Written: {fsize:.2f} GB ({elapsed(t0)})")

    # --- Summary ---
    log("")
    log("=" * 60)
    log("ASSEMBLY COMPLETE")
    log("=" * 60)
    log(f"Output: {args.output}")
    log(f"  {len(out):,} rows, {len(out.colnames)} columns, {fsize:.2f} GB")
    log(f"Total time: {elapsed(t_start)}")
    log("")
    log("Columns:")
    for col in out.colnames:
        desc = out[col].description or ""
        log(f"  {col}: {out[col].dtype}  -- {desc}")


if __name__ == "__main__":
    main()
