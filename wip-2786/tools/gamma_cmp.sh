#!/bin/bash
# gamma_cmp.sh: transit / IG fits (incomplete-gamma primitive) with the compiled path off and on.
W=/root/package/ferx-core/.worktrees/lanes
for pair in "one_cpt_transit.ferx $W/data/transit_oral.csv" \
            "two_cpt_transit.ferx $W/data/transit_oral.csv" \
            "one_cpt_ig.ferx $W/data/ig_indomain_oral.csv"; do
  set -- $pair
  echo "### $1"
  bash /scratch/mu/ad/kj/fitcmp.sh "$1" "$2"
done
