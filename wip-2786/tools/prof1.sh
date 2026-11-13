#!/bin/bash
# prof1.sh <example> <data>: one fit with the compiled path on: elapsed, counters, compile time.
r=$(mktemp -d /scratch/mu/ad/kj/pf.XXXX)
cp "/root/package/ferx-core/.worktrees/lanes/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 timeout 900 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "/root/package/ferx-core/.worktrees/lanes/data/$2" > out.txt 2> err.txt
echo "rc=$? $(grep -m1 OFV out.txt) $(grep -m1 Elapsed out.txt)"
grep -oE "kjit: compiled.*" err.txt | cut -c1-200
grep -E "^ +se:" m-fit.yaml | head -4 | tr -s ' ' | tr '\n' ' '; echo
cd /; rm -rf "$r"
