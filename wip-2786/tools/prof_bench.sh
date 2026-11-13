#!/bin/bash
# prof_bench.sh <model> <data>: one benchmark-set fit with the compiled path on: elapsed,
# counters, the costliest compiles and the churning guard sites.
r=$(mktemp -d /scratch/mu/ad/kj/pb.XXXX)
cp "/scratch/speed/bench/$1.ferx" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 timeout 1800 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "/scratch/speed/bench/$2.csv" --clean > out.txt 2> err.txt
echo "rc=$? $(grep -m1 OFV out.txt) $(grep -m1 Elapsed out.txt)"
grep -oE "kjit: compiled.*" err.txt | cut -c1-220
grep -E "guard flip|br_value pin|failed validation|not recorded" err.txt | sed -E 's/ -?[0-9]+(\.[0-9]+)?(e-?[0-9]+)? / V /g' | sort | uniq -c | sort -rn | head -8 | cut -c1-200
cd /; rm -rf "$r"
