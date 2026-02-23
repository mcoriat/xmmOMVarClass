#!/usr/bin/env python3
"""
Step 1: Cross-match OM sources against CDS VizieR catalogues.

Queries AllWISE, SDSS DR16, 2MASS, and PanSTARRS DR1 via STILTS cdsskymatch.
Merges results onto the slim catalogue to provide external photometry
for CLAXBOI classification features.

Usage:
    cd claxboi
    python3 om_crossmatch.py [--resume] [--parallel N]
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
INTERMEDIATES = os.path.join(BASEDIR, "intermediates")

INPUT_FILE = os.path.join(INTERMEDIATES, "suss_slim.fits")
OUTPUT_FILE = os.path.join(INTERMEDIATES, "suss_with_extphot.fits")

STILTS = "stilts"          # assumes stilts is on PATH
MATCH_RADIUS = 3           # arcsec

# Number of columns in the input slim catalogue (used to strip duplicated
# input columns from cdsskymatch output).
N_INPUT_COLS = 24

# Each catalogue entry: (short name, VizieR table, columns to keep,
#                        dedup column)
CATALOGUES = [
    {
        "name":       "AllWISE",
        "vizier":     "II/328/allwise",
        "keep_cols":  ["W1mag", "W2mag"],
        "dedup_col":  "AllWISE",
    },
    {
        "name":       "SDSS_DR16",
        "vizier":     "V/154/sdss16",
        # CDS renames gmag -> gmag_cds (clashes with input Gmag).
        # spCl is the VizieR column name (capital C).
        "keep_cols":  ["umag", "rmag", "gmag_cds", "spCl", "Q"],
        "dedup_col":  "objID",
        "rename":     {"gmag_cds": "gmag_sdss", "spCl": "spcl"},
    },
    {
        "name":       "2MASS",
        "vizier":     "II/246/out",
        "keep_cols":  ["Kmag"],
        # CDS returns the column as "2MASS" (starts with digit).
        # We use the column index $25 for STILTS dedup expression.
        "dedup_col":  "2MASS",
    },
    {
        "name":       "PanSTARRS_DR1",
        "vizier":     "II/349/ps1",
        # CDS renames gmag -> gmag_cds (clashes with input).
        "keep_cols":  ["gmag_cds", "rmag"],
        "dedup_col":  "objID",
        "rename":     {"gmag_cds": "gPSFMag", "rmag": "rPSFMag"},
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def elapsed(t0):
    """Return a human-readable elapsed time string."""
    dt = time.time() - t0
    h, rem = divmod(int(dt), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def run_stilts(cmd, description=""):
    """Run a STILTS command, printing the invocation and checking for errors."""
    print(f"  [{description}] Running: {' '.join(cmd)}")
    sys.stdout.flush()
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR running STILTS ({description}):")
        print(exc.stderr)
        raise
    print(f"  [{description}] Done in {elapsed(t0)}")
    sys.stdout.flush()


def raw_path(cat_name):
    return os.path.join(INTERMEDIATES, f"xmatch_raw_{cat_name}.fits")


def deduped_path(cat_name):
    return os.path.join(INTERMEDIATES, f"xmatch_dedup_{cat_name}.fits")


def trimmed_path(cat_name):
    return os.path.join(INTERMEDIATES, f"xmatch_{cat_name}.fits")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def query_catalogue(cat, resume=False):
    """
    Run cdsskymatch + dedup + column trim for one external catalogue.

    Parameters
    ----------
    cat : dict
        Catalogue definition from CATALOGUES.
    resume : bool
        If True, skip this catalogue when its trimmed output already exists.

    Returns
    -------
    str or None
        Path to the trimmed result file, or None on failure.
    """
    name = cat["name"]
    out_trimmed = trimmed_path(name)

    if resume and os.path.isfile(out_trimmed):
        n = len(Table.read(out_trimmed))
        print(f"[{name}] Skipping (resume mode) -- {n:,} rows in {out_trimmed}")
        return out_trimmed

    print(f"\n{'='*60}")
    print(f"[{name}]  VizieR table = {cat['vizier']}")
    print(f"{'='*60}")
    t0 = time.time()

    out_raw = raw_path(name)
    out_dedup = deduped_path(name)

    # --- 1. CDS sky match -------------------------------------------------
    if resume and os.path.isfile(out_raw):
        print(f"  [{name}] Raw file exists, skipping cdsskymatch")
    else:
        cmd = [
            STILTS, "cdsskymatch",
            f"in={INPUT_FILE}",
            f"out={out_raw}",
            "ra=RA", "dec=DEC",
            f"cdstable={cat['vizier']}",
            "find=best",
            f"radius={MATCH_RADIUS}",
        ]
        try:
            run_stilts(cmd, description=f"{name} cdsskymatch")
        except subprocess.CalledProcessError:
            print(f"  [{name}] cdsskymatch FAILED -- skipping this catalogue")
            return None

    # --- 2. Deduplicate on source ID --------------------------------------
    # Use astropy instead of tmatch1: STILTS tmatch1 fails when all values
    # are unique (no matches found), and column names starting with digits
    # (like "2MASS") can't be used in STILTS expressions.
    print(f"  [{name}] Deduplicating on {cat['dedup_col']}...")
    sys.stdout.flush()
    result = Table.read(out_raw)
    n_before = len(result)
    dedup_col = cat["dedup_col"]
    if dedup_col in result.colnames:
        # Keep first occurrence of each unique value in the dedup column
        _, unique_idx = np.unique(result[dedup_col], return_index=True)
        unique_idx.sort()  # preserve original row order
        result = result[unique_idx]
    n_after = len(result)
    n_dups = n_before - n_after
    print(f"  [{name}] Dedup: {n_before:,} -> {n_after:,} ({n_dups:,} duplicates removed)")
    sys.stdout.flush()

    # --- 3. Trim to needed columns ----------------------------------------

    # Save RA, DEC from the prepended input columns before stripping them.
    ra_xm = result["RA"].data.copy()
    dec_xm = result["DEC"].data.copy()

    # cdsskymatch prepends all input columns; drop them to keep only the
    # catalogue columns plus the match metadata.
    cat_cols = result.colnames[N_INPUT_COLS:]
    result = result[cat_cols]

    # Build the list of columns to keep (dedup ID + science columns)
    keep = [cat["dedup_col"]] + cat["keep_cols"]
    available = [c for c in keep if c in result.colnames]
    missing = set(keep) - set(available)
    if missing:
        print(f"  [{name}] WARNING: columns not found in result: {missing}")
    result = result[available]

    # Apply any renames (e.g. gmag_cds -> gmag_sdss)
    for old, new in cat.get("rename", {}).items():
        if old in result.colnames:
            result.rename_column(old, new)

    # Add RA, DEC for the later sky-match join
    result["RA_xm"] = ra_xm
    result["DEC_xm"] = dec_xm

    result.write(out_trimmed, overwrite=True)
    print(f"  [{name}] Trimmed to {len(available)} columns, "
          f"{len(result):,} rows  (elapsed {elapsed(t0)})")

    # Clean up raw intermediate file
    if os.path.isfile(out_raw):
        os.remove(out_raw)

    return out_trimmed


def merge_all(cat_results):
    """
    Sequentially join each catalogue result onto the slim catalogue
    using STILTS tmatch2 (sky match, keep all rows from table 1).
    Then compute the merged g-band magnitude gmag_M.

    Parameters
    ----------
    cat_results : list of (dict, str)
        Pairs of (catalogue definition, path to trimmed result).
    """
    print(f"\n{'='*60}")
    print("Merging all catalogues onto slim catalogue")
    print(f"{'='*60}")
    t0 = time.time()

    current = INPUT_FILE

    for i, (cat, result_path) in enumerate(cat_results):
        name = cat["name"]
        step_out = os.path.join(
            INTERMEDIATES, f"suss_merge_step{i+1}_{name}.fits"
        )

        cmd = [
            STILTS, "tmatch2",
            f"in1={current}",
            f"in2={result_path}",
            "matcher=sky", "params=3",
            "values1=RA DEC",
            "values2=RA_xm DEC_xm",
            "find=best1", "join=all1",
            f"out={step_out}",
        ]
        try:
            run_stilts(cmd, description=f"merge {name}")
        except subprocess.CalledProcessError:
            print(f"  merge {name} FAILED -- continuing without it")
            continue

        # Remove the temporary RA_xm/DEC_xm columns and catalogue dedup ID
        tab = Table.read(step_out)
        for dropcol in ["RA_xm", "DEC_xm", cat["dedup_col"],
                        "Separation", "GroupSize", "GroupID"]:
            if dropcol in tab.colnames:
                tab.remove_column(dropcol)
        tab.write(step_out, overwrite=True)

        # Resolve through rename mapping to find the actual column name
        check_col = cat["keep_cols"][0]
        renames = cat.get("rename", {})
        check_col = renames.get(check_col, check_col)
        n_matched = np.sum(np.isfinite(tab[check_col].data
                                        .astype(float)))
        n_total = len(tab)
        pct = 100.0 * n_matched / n_total
        print(f"  [{name}] {n_matched:,}/{n_total:,} matched ({pct:.1f}%)")

        # Clean up previous step file (but never delete the original input)
        if current != INPUT_FILE and os.path.isfile(current):
            os.remove(current)
        current = step_out

    # --- Compute gmag_M (merged g-band) -----------------------------------
    print("\nComputing gmag_M (merged g-band: SDSS priority, PanSTARRS fallback)")
    tab = Table.read(current)

    gmag_sdss = np.full(len(tab), np.nan)
    gPSFMag = np.full(len(tab), np.nan)

    if "gmag_sdss" in tab.colnames:
        gmag_sdss = np.array(tab["gmag_sdss"], dtype=float)
    if "gPSFMag" in tab.colnames:
        gPSFMag = np.array(tab["gPSFMag"], dtype=float)

    gmag_M = np.where(np.isfinite(gmag_sdss), gmag_sdss, gPSFMag)
    tab["gmag_M"] = gmag_M

    n_sdss = np.sum(np.isfinite(gmag_sdss))
    n_ps = np.sum(np.isfinite(gPSFMag) & ~np.isfinite(gmag_sdss))
    n_tot = np.sum(np.isfinite(gmag_M))
    print(f"  gmag_M filled: {n_tot:,} total "
          f"({n_sdss:,} from SDSS, {n_ps:,} from PanSTARRS)")

    # Write final output
    tab.write(OUTPUT_FILE, overwrite=True)
    print(f"\nFinal catalogue written to {OUTPUT_FILE}")
    print(f"  {len(tab):,} rows, {len(tab.colnames)} columns")
    print(f"  Columns: {tab.colnames}")
    print(f"  Total merge time: {elapsed(t0)}")

    # Clean up last step file
    if current != INPUT_FILE and os.path.isfile(current):
        os.remove(current)

    # Clean up trimmed catalogue files
    for cat, _ in cat_results:
        p = trimmed_path(cat["name"])
        if os.path.isfile(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-match OM slim catalogue against CDS VizieR "
                    "catalogues for external photometry."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip catalogues whose intermediate files already exist."
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of CDS queries to run in parallel (default: 1). "
             "Note: CDS may throttle parallel requests."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Override input file (default: intermediates/suss_slim.fits)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output file (default: intermediates/suss_with_extphot.fits)."
    )
    args = parser.parse_args()

    # Allow CLI override of input/output paths
    global INPUT_FILE, OUTPUT_FILE
    if args.input:
        INPUT_FILE = args.input
    if args.output:
        OUTPUT_FILE = args.output

    if not os.path.isfile(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # Read the input once to confirm its shape
    slim = Table.read(INPUT_FILE)
    print(f"Input: {INPUT_FILE}")
    print(f"  {len(slim):,} rows, {len(slim.colnames)} columns")
    print(f"  Columns: {slim.colnames}")
    del slim

    t_start = time.time()
    cat_results = []

    if args.parallel > 1:
        # Parallel execution using concurrent.futures
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for cat in CATALOGUES:
                fut = pool.submit(query_catalogue, cat, resume=args.resume)
                futures[fut] = cat
            for fut in as_completed(futures):
                cat = futures[fut]
                result_path = fut.result()
                if result_path is not None:
                    cat_results.append((cat, result_path))
        # Restore catalogue order for deterministic merging
        order = {c["name"]: i for i, c in enumerate(CATALOGUES)}
        cat_results.sort(key=lambda x: order[x[0]["name"]])
    else:
        for cat in CATALOGUES:
            result_path = query_catalogue(cat, resume=args.resume)
            if result_path is not None:
                cat_results.append((cat, result_path))

    if not cat_results:
        print("\nERROR: No catalogues were successfully queried.")
        sys.exit(1)

    print(f"\nCatalogue queries completed in {elapsed(t_start)}")
    print(f"  Successful: {len(cat_results)}/{len(CATALOGUES)}")

    merge_all(cat_results)
    print(f"\nTotal runtime: {elapsed(t_start)}")


if __name__ == "__main__":
    main()
