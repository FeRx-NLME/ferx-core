#!/bin/bash
# time_cmp.sh <example> <data>: wall time of the fit, compiled path off vs on, interleaved x3, one core.
B=${B:-/scratch/mu/ad/ferx-kjit-inc}
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/tc.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
for i in 1 2 3; do
  for mode in off on; do
    if [ "$mode" = on ]; then export FERX_KJIT=1; else unset FERX_KJIT; fi; [ -n "$FAST" ] && [ "$mode" = on ] && export FERX_KJIT_FAST=1
    s=$(date +%s.%N)
    taskset -c 2 "$B" m.ferx --data "$2" --threads 1 > /dev/null 2> err.txt
    e=$(date +%s.%N)
    printf "%s %s %.2f s  %s\n" "$1" "$mode" "$(python3 -c "print($e - $s)")" "$(grep -oE 'compile [0-9.]+ ms' err.txt)"
  done
done
cd /scratch/mu/ad/kj && rm -rf "$r"
