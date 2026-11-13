#!/bin/bash
# (built by the caller)
D=/root/package/ferx-core/.worktrees/lanes/data
K=/scratch/mu/ad/kj
for m in "warfarin.ferx warfarin.csv" "two_cpt_oral_cov.ferx two_cpt_oral_cov.csv" "mm_oral.ferx mm_oral.csv" "warfarin_ode.ferx warfarin.csv"; do
  set -- $m
  echo "### $1"
  bash $K/instr3.sh "$1" "$D/$2"
done
