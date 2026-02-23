#!/usr/bin/env python3
"""
Step 2: Compute classification features and assign training labels.

Computes 12 color/morphology features from OM, Gaia, and external photometry.
Assigns training class labels:
  - Class 0 (Star): Gaia proper motion S/N > 30
  - Class 1 (QSO): SDSS spectroscopic QSO (spcl=='QSO', Q==3) + SIMBAD AGN types
  - Class 2 (Galaxy): SDSS spectroscopic galaxy (spcl=='GALAXY', Q==3) + SIMBAD galaxy types

Usage:
    cd claxboi
    python3 om_compute_features.py [--skip-simbad]
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from astropy.table import Table


# ---------------------------------------------------------------------------
#  SIMBAD type lists (consistent with classification/CLAXBOI auto_classes.py)
# ---------------------------------------------------------------------------
SIMCAT_QSO = ['AGN', 'Seyfert_1', 'Seyfert_2', 'BLLac', 'Blazar', 'QSO']
SIMCAT_GAL = ['Galaxy', 'GinCl', 'GinGroup', 'BClG', 'GrG', 'PairG', 'IG']

STILTS = os.environ.get('STILTS', 'stilts')  # override via env var if needed


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------
def _safe_col(table, name):
    """Return a float64 copy of *name*, or NaN array if the column is absent."""
    if name in table.colnames:
        return np.array(table[name], dtype=np.float64)
    print(f"  WARNING: column '{name}' not found -- filling with NaN")
    return np.full(len(table), np.nan)


def _safe_str_col(table, name):
    """Return a string array for *name*, or empty strings if absent."""
    if name in table.colnames:
        raw = table[name]
        # FITS string columns may be bytes -- decode if necessary
        try:
            return np.array([s.decode() if isinstance(s, bytes) else str(s)
                             for s in raw])
        except Exception:
            return np.array([str(s) for s in raw])
    print(f"  WARNING: column '{name}' not found -- filling with ''")
    return np.full(len(table), '', dtype='U32')


# ---------------------------------------------------------------------------
#  Part 2a: Compute the 12 classification features
# ---------------------------------------------------------------------------
def compute_features(t):
    """Add 12 color / morphology feature columns to *t* (in-place)."""

    print("\n--- Computing 12 classification features ---")
    n = len(t)

    # 1. uvw1_u  =  UVW1_AB_MAG - U_AB_MAG
    print("  [1/12] uvw1_u")
    t['uvw1_u'] = _safe_col(t, 'UVW1_AB_MAG') - _safe_col(t, 'U_AB_MAG')

    # 2. b_v  =  B_AB_MAG - V_AB_MAG
    print("  [2/12] b_v")
    t['b_v'] = _safe_col(t, 'B_AB_MAG') - _safe_col(t, 'V_AB_MAG')

    # 3. W2_W1  =  W2mag - W1mag  (AllWISE)
    print("  [3/12] W2_W1")
    t['W2_W1'] = _safe_col(t, 'W2mag') - _safe_col(t, 'W1mag')

    # 4. BP_RP  -- rename Gaia BP-RP to avoid hyphens
    print("  [4/12] BP_RP")
    if 'BP-RP' in t.colnames:
        t['BP_RP'] = np.array(t['BP-RP'], dtype=np.float64)
    else:
        t['BP_RP'] = _safe_col(t, 'BP_RP')   # may already exist

    # 5. UVM2mUVW1  =  UVM2_AB_MAG - UVW1_AB_MAG
    print("  [5/12] UVM2mUVW1")
    t['UVM2mUVW1'] = _safe_col(t, 'UVM2_AB_MAG') - _safe_col(t, 'UVW1_AB_MAG')

    # 6. UVW2mUVW1  =  UVW2_AB_MAG - UVW1_AB_MAG
    print("  [6/12] UVW2mUVW1")
    t['UVW2mUVW1'] = _safe_col(t, 'UVW2_AB_MAG') - _safe_col(t, 'UVW1_AB_MAG')

    # 7. UVW1mGmag  =  UVW1_AB_MAG - Gmag
    print("  [7/12] UVW1mGmag")
    t['UVW1mGmag'] = _safe_col(t, 'UVW1_AB_MAG') - _safe_col(t, 'Gmag')

    # 8. Gaia_G_WISE_W1  (conditional -- xmm2athena.py lines 3089-3094)
    print("  [8/12] Gaia_G_WISE_W1")
    Gmag = _safe_col(t, 'Gmag')
    W1mag = _safe_col(t, 'W1mag')
    BII = _safe_col(t, 'BII')

    has_gaia = np.isfinite(Gmag)
    has_w1 = np.isfinite(W1mag)
    high_lat = np.abs(BII) > 10

    gw = np.full(n, np.nan)
    # Case 1: no Gaia, has WISE, high galactic latitude, W1<16
    mask1 = ~has_gaia & has_w1 & high_lat & (W1mag < 16)
    gw[mask1] = 20.0 - W1mag[mask1]
    # Case 2: has Gaia, no WISE, bright Gaia (G < 15.6)
    mask2 = has_gaia & ~has_w1 & (Gmag < 15.6)
    gw[mask2] = Gmag[mask2] - 17.1
    # Case 3: has both
    mask3 = has_gaia & has_w1
    gw[mask3] = Gmag[mask3] - W1mag[mask3]

    t['Gaia_G_WISE_W1'] = gw

    # 9. gaia_extended  (star/galaxy separator -- xmm2athena.py lines 3098-3102)
    print("  [9/12] gaia_extended")
    gmag_M = _safe_col(t, 'gmag_M')
    has_gmag_M = np.isfinite(gmag_M) & (gmag_M > 0)

    ge = np.full(n, np.nan)
    # Case 1: no Gaia, has ground-based g, high latitude, g<19
    mask1 = ~has_gaia & has_gmag_M & high_lat & (gmag_M < 19)
    ge[mask1] = 20.0 - gmag_M[mask1]
    # Case 2: has both Gaia and ground-based g
    mask2 = has_gaia & has_gmag_M
    ge[mask2] = Gmag[mask2] - gmag_M[mask2]
    # Case 3: has Gaia, no ground-based g -- use V band
    V_AB = _safe_col(t, 'V_AB_MAG')
    has_v = np.isfinite(V_AB)
    mask3 = has_gaia & ~has_gmag_M & has_v
    ge[mask3] = Gmag[mask3] - V_AB[mask3]

    t['gaia_extended'] = ge

    # 10. umag_rmag  =  umag - rmag  (SDSS)
    print("  [10/12] umag_rmag")
    t['umag_rmag'] = _safe_col(t, 'umag') - _safe_col(t, 'rmag')

    # 11. k_WiseW1  =  Kmag - W1mag  (2MASS - WISE)
    print("  [11/12] k_WiseW1")
    t['k_WiseW1'] = _safe_col(t, 'Kmag') - _safe_col(t, 'W1mag')

    # 12. OMu_b  =  U_AB_MAG - B_AB_MAG
    print("  [12/12] OMu_b")
    t['OMu_b'] = _safe_col(t, 'U_AB_MAG') - _safe_col(t, 'B_AB_MAG')

    # Quick coverage summary
    features = ['uvw1_u', 'b_v', 'W2_W1', 'BP_RP', 'UVM2mUVW1', 'UVW2mUVW1',
                'UVW1mGmag', 'Gaia_G_WISE_W1', 'gaia_extended', 'umag_rmag',
                'k_WiseW1', 'OMu_b']
    print("\nFeature coverage (finite values):")
    for f in features:
        nfin = np.sum(np.isfinite(np.array(t[f], dtype=np.float64)))
        print(f"  {f:20s}  {nfin:>10,d} / {n:,d}  ({100*nfin/n:5.1f}%)")

    return t


# ---------------------------------------------------------------------------
#  Part 2b: Assign training class labels
# ---------------------------------------------------------------------------
def assign_labels_spectroscopic(t):
    """Assign Star / QSO / Galaxy flags from Gaia PM and SDSS spectra."""

    print("\n--- Assigning spectroscopic training labels ---")

    # --- STARS: Gaia proper-motion S/N > 30 ---
    pmRA = _safe_col(t, 'pmRA')
    pmDE = _safe_col(t, 'pmDE')
    e_pmRA = _safe_col(t, 'e_pmRA')
    e_pmDE = _safe_col(t, 'e_pmDE')

    pm_total = np.sqrt(pmRA**2 + pmDE**2)
    pm_err = np.sqrt(e_pmRA**2 + e_pmDE**2)
    pm_snr = np.where(pm_err > 0, pm_total / pm_err, 0.0)

    is_star = pm_snr > 30
    t['isStar'] = is_star.astype(int)
    print(f"  Stars  (PM S/N > 30):  {np.sum(is_star):,d}")

    # --- QSOs: SDSS spectroscopic (spcl == 'QSO', Q == 3) ---
    spcl = _safe_str_col(t, 'spcl')
    Q = _safe_col(t, 'Q')

    # Strip whitespace for robust comparison
    spcl_stripped = np.array([s.strip() for s in spcl])

    is_qso_sdss = (spcl_stripped == 'QSO') & (Q == 3)
    t['isQSO'] = is_qso_sdss.astype(int)
    print(f"  QSOs   (SDSS spec):    {np.sum(is_qso_sdss):,d}")

    # --- Galaxies: SDSS spectroscopic (spcl == 'GALAXY', Q == 3) ---
    is_gal_sdss = (spcl_stripped == 'GALAXY') & (Q == 3)
    t['isGalaxy'] = is_gal_sdss.astype(int)
    print(f"  Galaxies (SDSS spec):  {np.sum(is_gal_sdss):,d}")

    return t


def enrich_simbad(t, input_file):
    """Cross-match against SIMBAD via STILTS cdsskymatch and fold matches
    into the isQSO / isGalaxy flags.

    Parameters
    ----------
    t : astropy.table.Table
        Must already contain 'isQSO' and 'isGalaxy' columns.
    input_file : str
        Path to the FITS file to feed to STILTS (needs RA, DEC columns).
    """
    print("\n--- SIMBAD enrichment via STILTS cdsskymatch ---")

    simbad_fits = os.path.join(os.path.dirname(input_file), 'simbad_result.fits')

    # Determine RA/DEC column names (SUSS convention)
    ra_col = 'RA' if 'RA' in t.colnames else 'ra2000Ep'
    dec_col = 'DEC' if 'DEC' in t.colnames else 'dec2000Ep'

    # Determine a unique ID column for matching back
    id_col = None
    for candidate in ['IAUNAME', 'iauname', 'SRCID']:
        if candidate in t.colnames:
            id_col = candidate
            break
    if id_col is None:
        print("  ERROR: no suitable ID column found for SIMBAD matching.")
        return t

    cmd = [
        STILTS, 'cdsskymatch',
        f'in={input_file}', f'ra={ra_col}', f'dec={dec_col}',
        'cdstable=simbad', 'find=best',
        f'out={simbad_fits}', 'radius=3',
    ]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"  ERROR: STILTS not found at {STILTS}. Skipping SIMBAD enrichment.")
        return t
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: STILTS failed: {exc.stderr}")
        return t

    elapsed = time.time() - t0
    print(f"  STILTS cdsskymatch completed in {elapsed:.0f}s")

    if not os.path.exists(simbad_fits):
        print("  WARNING: SIMBAD result file not found. Skipping enrichment.")
        return t

    simbad = Table.read(simbad_fits)
    print(f"  SIMBAD matches: {len(simbad):,d}")

    if 'main_type' not in simbad.colnames:
        print("  WARNING: 'main_type' column missing from SIMBAD result.")
        return t

    # Decode main_type if needed
    sim_types = _safe_str_col(simbad, 'main_type')
    sim_types = np.array([s.strip() for s in sim_types])

    # Build lookup: ID -> index in main table
    main_ids = _safe_str_col(t, id_col)
    main_ids = np.array([s.strip() for s in main_ids])
    id_to_idx = {name: idx for idx, name in enumerate(main_ids)}

    sim_ids = _safe_str_col(simbad, id_col)
    sim_ids = np.array([s.strip() for s in sim_ids])

    n_qso_added = 0
    n_gal_added = 0

    for i, (sid, stype) in enumerate(zip(sim_ids, sim_types)):
        idx = id_to_idx.get(sid)
        if idx is None:
            continue
        if stype in SIMCAT_QSO and t['isQSO'][idx] == 0:
            t['isQSO'][idx] = 1
            n_qso_added += 1
        elif stype in SIMCAT_GAL and t['isGalaxy'][idx] == 0:
            t['isGalaxy'][idx] = 1
            n_gal_added += 1

    print(f"  SIMBAD additions: +{n_qso_added:,d} QSO, +{n_gal_added:,d} Galaxy")

    # Clean up temporary file
    try:
        os.remove(simbad_fits)
    except OSError:
        pass

    return t


def assign_final_classes(t):
    """Set the 'class' column: 0=Star, 1=QSO, 2=Galaxy. Conflicts -> NaN."""

    print("\n--- Assigning final class labels ---")

    t['class'] = np.nan

    is_star = np.array(t['isStar'], dtype=int)
    is_qso = np.array(t['isQSO'], dtype=int)
    is_gal = np.array(t['isGalaxy'], dtype=int)

    n_types = (is_star > 0).astype(int) + (is_qso > 0).astype(int) + (is_gal > 0).astype(int)
    conflicts = n_types > 1

    # Stars: class 0 (no conflict)
    t['class'][(is_star > 0) & ~conflicts] = 0
    # QSOs: class 1 (not star, no conflict)
    t['class'][(is_qso > 0) & (is_star == 0) & ~conflicts] = 1
    # Galaxies: class 2 (not star, not QSO, no conflict)
    t['class'][(is_gal > 0) & (is_star == 0) & (is_qso == 0) & ~conflicts] = 2
    # Conflicts: explicit NaN
    t['class'][conflicts] = np.nan

    # Summary
    cls = np.array(t['class'], dtype=np.float64)
    n_unclassified = np.sum(np.isnan(cls)) - np.sum(conflicts)

    print("\nTraining sample summary:")
    print(f"  Stars (class 0):    {np.sum(cls == 0):>10,d}")
    print(f"  QSOs (class 1):     {np.sum(cls == 1):>10,d}")
    print(f"  Galaxies (class 2): {np.sum(cls == 2):>10,d}")
    print(f"  Conflicts (NaN):    {np.sum(conflicts):>10,d}")
    print(f"  Unclassified:       {n_unclassified:>10,d}")

    return t


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Compute 12 CLAXBOI features and assign training labels.')
    parser.add_argument('--input', default='intermediates/suss_with_extphot.fits',
                        help='Input FITS table (default: intermediates/suss_with_extphot.fits)')
    parser.add_argument('--output', default='intermediates/suss_with_training.fits',
                        help='Output FITS table (default: intermediates/suss_with_training.fits)')
    parser.add_argument('--skip-simbad', action='store_true',
                        help='Skip the SIMBAD cross-match enrichment step')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    if not os.path.isfile(input_file):
        sys.exit(f"ERROR: input file not found: {input_file}")

    print(f"Reading {input_file} ...")
    t0 = time.time()
    t = Table.read(input_file)
    print(f"  Loaded {len(t):,d} rows x {len(t.colnames)} columns "
          f"in {time.time()-t0:.1f}s")

    # Part 2a -- features
    t = compute_features(t)

    # Part 2b -- training labels
    t = assign_labels_spectroscopic(t)

    if not args.skip_simbad:
        t = enrich_simbad(t, input_file)
    else:
        print("\n  (SIMBAD enrichment skipped)")

    t = assign_final_classes(t)

    # Write output
    print(f"\nWriting {output_file} ...")
    t0 = time.time()
    t.write(output_file, format='fits', overwrite=True)
    print(f"  Written {len(t):,d} rows x {len(t.colnames)} columns "
          f"in {time.time()-t0:.1f}s")
    print("Done.")


if __name__ == '__main__':
    main()
