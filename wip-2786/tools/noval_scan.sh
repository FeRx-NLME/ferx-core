#!/bin/bash
# noval_scan.sh: user-space instructions, compiled path off / on / on without the value path.
# (built by the caller)
W=/root/package/ferx-core/.worktrees/lanes
D=$W/data
B=/scratch/mu/ad/ferx-kjit-inc
for m in "warfarin.ferx warfarin.csv" "two_cpt_oral_cov.ferx two_cpt_oral_cov.csv" "mm_oral.ferx mm_oral.csv" "warfarin_ode.ferx warfarin.csv" "two_cpt_oral_ode.ferx two_cpt_oral_cov.csv" "three_cpt_iv_ode.ferx three_cpt_iv.csv"; do
  set -- $m
  r=$(mktemp -d /scratch/mu/ad/kj/ts.XXXX)
  cp "$W/examples/$1" "$r/m.ferx"
  cd "$r" || exit 1
  line="$1:"
  for mode in off on noval; do
    unset FERX_KJIT FERX_KJIT_NOVAL
    case $mode in on) export FERX_KJIT=1;; noval) export FERX_KJIT=1 FERX_KJIT_NOVAL=1;; esac
    perf stat -x, -e instructions:u -o stat.txt taskset -c 3 "$B" m.ferx --data "$D/$2" --threads 1 > /dev/null 2>&1
    line="$line $mode=$(awk -F, '/instructions/ {printf "%.3f", $1/1e9}' stat.txt)"
  done
  echo "$line"
  cd /
  rm -rf "$r"
done
