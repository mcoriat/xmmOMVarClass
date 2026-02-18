#!/bin/bash
#
# How to use:
#  addEpoch2Source.sh XMM-OM-SUSS6.1.fits <output_file_name>
# where the input is the SUSS or UVOTSSC file with two extensions #1 and #2 for the
#   SRCLIST and SUMMARY
#
# The OUTPUT is just the SRCLIST with added column "obs_epoch"
#
# Requires: stilts in PATH (installed via homebrew at /opt/homebrew/bin/stilts)
#
echo "addEpoch2Source.sh"
#
# Step 1: Extract SUMMARY (#2), compute obs_epoch from MJD, keep only OBSID + obs_epoch
stilts tpipe \
  in="$1#2" ifmt=fits out=tmp2.fits ofmt=fits-basic \
  cmd='addcol -units y obs_epoch mjdToDecYear(0.5*(MJD_START+MJD_END));keepcols "OBSID obs_epoch"'
#
if [ "$2" != "" ]; then
   OUT=$2
else
   OUT="withObsEpoch_$1"
fi
#
# Step 2: Join obs_epoch to SRCLIST (#1) by matching on OBSID
stilts tmatch2 \
  in1="$1#1" ifmt1=fits in2=tmp2.fits ifmt2=fits \
  out=tmp1.fits ofmt=fits-basic \
  matcher=exact values1="OBSID" values2="OBSID" \
  join=all1 find=best1 fixcols=dups suffix1= suffix2=_2
#
# Step 3: Remove extra columns from the join
stilts tpipe \
  in=tmp1.fits ifmt=fits out="$OUT" ofmt=fits-basic \
  cmd="delcols 'OBSID_2 GroupID GroupSize'"
#
# recombining with the SUMMARY extension from the SUSS has not been done
# e.g., fappend $1#2 $OUT
# clean up
#
rm -f tmp1.fits tmp2.fits
