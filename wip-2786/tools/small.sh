#!/bin/bash
# small.sh: wall time of the Markov/CTMM examples, merge-base reference vs this branch (best of 3).
W=/root/package/ferx-core/.worktrees/lanes
for pair in "ctmm_2state.ferx ctmm_2state.csv" "mctmm_3state.ferx mctmm_3state.csv" "three_cpt_iv_ode.ferx three_cpt_iv.csv"; do
  set -- $pair
  for tag in main on; do
    best=999999
    for i in 1 2 3; do
      r=$(mktemp -d /scratch/mu/ad/kj/sm.XXXX)
      cp "$W/examples/$1" "$r/m.ferx"
      if [ $tag = on ]; then b=/scratch/mu/ad/ferx-kjit-inc; export FERX_KJIT=1; else b=/scratch/mu/ad/ferx-main-feat; unset FERX_KJIT; fi
      s=$(date +%s%N)
      (cd "$r" && timeout 600 "$b" m.ferx --data "$W/data/$2" > out.txt 2> err.txt)
      e=$(( ($(date +%s%N) - s) / 1000000 ))
      [ $e -lt $best ] && best=$e
      o=$(grep -m1 OFV "$r/out.txt")
      rm -rf "$r"
    done
    echo "$1 $tag ${best}ms $o"
  done
done
