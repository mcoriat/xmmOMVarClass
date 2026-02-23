#!/bin/bash
#
# Usage: stilts_fin_match2gaia.sh <matchout> <nomatch> <outfile>
#
# Requires: stilts in PATH (installed via homebrew at /opt/homebrew/bin/stilts)
#
echo "finalise.match2Gaia.sh: rematch all"
IN1=${1:-matchout_all.fits}
IN2=${2:-nomatch_all.fits}
OUT=${3:-sussxgaiadr3_ep2000.fits}
#
echo first file
echo
stilts cdsskymatch cdstable=I/355/gaiadr3 \
 find=each in=$IN1  ifmt=fits  ra=ra2000Ep dec=dec2000Ep radius=3 out=gaia_pm_match.fits \
 ocmd='delcols "ra_error dec_error parallax_over_error healpix raObs decObs obsEp 2000Ep" '\
 ocmd='addcol mismatchedPM "((pm_in - pm_cds > 5)?1:0)" '
echo
echo second file
echo
stilts cdsskymatch cdstable=I/355/gaiadr3 find=each in=$IN2 ifmt=fits \
 ra=ra2000Ep dec=dec2000Ep radius=3 out=gaia_no_pm_match.fits \
 ocmd='addcol mismatchedPM "0" ' ocmd='delcols "2000Ep"'
echo
echo make file 2 match preparing
stilts tpipe in=gaia_pm_match.fits ifmt=fits  out=tmp_gaia_pm_match.fits ofmt=fits \
 cmd='delcols "designation source_id ra2016gaia dec2016gaia parallax pm_in pmra_in pmra_error pmdec pmdec_error Separation phot_g_mean_mag bp_rp bp_g g_rp non_single_star"' \
 cmd='colmeta -name PM PM_cds' \
 cmd='colmeta -name pmRA pmRA_cds'
echo
echo tcat them
stilts tcat in='tmp_gaia_pm_match.fits gaia_no_pm_match.fits' ifmt=fits out=$OUT ofmt=fits
echo
