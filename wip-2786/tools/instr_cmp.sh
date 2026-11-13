#!/bin/bash
# instr_cmp.sh <example> <data>: instructions and cycles, compiled path off vs on (one core).
W=/root/package/ferx-core/.worktrees/lanes
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
r=$(mktemp -d /scratch/mu/ad/kj/ic.XXXX)
cp "$M" "$r/m.ferx"
cd "$r" || exit 1
for mode in off on; do
  if [ "$mode" = on ]; then export FERX_KJIT=1; else unset FERX_KJIT; fi
  perf stat -x, -e instructions,cycles -o stat.txt taskset -c 2 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > out.txt 2>&1
  printf "%s %s  %s\n" "$mode" "$(grep -m1 -E '^OFV' out.txt)" "$(awk -F, '/instructions|cycles/ {printf "%s=%.3fG ", $3, $1/1e9}' stat.txt)"
done
cd /scratch/mu/ad/kj && rm -rf "$r"
