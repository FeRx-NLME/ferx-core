#!/bin/bash
# why.sh <bench model> <data>: which guards fail (or where recorded paths diverge) on a fit.
r=/scratch/mu/ad/kj/why_run
rm -rf "$r"; mkdir -p "$r"
cp "/scratch/speed/bench/$1.ferx" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 timeout 1800 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "/scratch/speed/bench/$2.csv" --clean > out.txt 2> err.txt
echo "rc=$? $(grep -m1 Elapsed out.txt)"
grep -oE "kjit: compiled.*" err.txt | cut -c1-200
grep -E "guard failed|paths diverge|loop .*failed|record-time" err.txt | sed -E 's/=[-0-9.e+NaIinf]+/=V/g; s/#[0-9]+/#N/' | sort | uniq -c | sort -rn | head -12 | cut -c1-220
