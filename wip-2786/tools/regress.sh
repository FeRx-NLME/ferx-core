#!/bin/bash
# regress.sh: full fits, compiled path off vs on, across model families.
W=/root/package/ferx-core/.worktrees/lanes
D=$W/data
S=/scratch/speed/bench
export B=/scratch/mu/ad/ferx-kjit-inc
run() { echo "### $1"; bash /scratch/mu/ad/kj/fitcmp.sh "$1" "$2"; }
run warfarin.ferx $D/warfarin.csv
run two_cpt_oral_cov.ferx $D/two_cpt_oral_cov.csv
run warfarin_iov.ferx $D/warfarin.csv
run one_cpt_transit.ferx $D/transit_oral.csv
run one_cpt_ig.ferx $D/ig_indomain_oral.csv
run oral_with_lagtime.ferx $D/one_cpt_oral_lagtime.csv
run warfarin_ode.ferx $D/warfarin.csv
run two_cpt_oral_ode.ferx $D/two_cpt_oral_cov.csv
run three_cpt_iv_ode.ferx $D/three_cpt_iv.csv
run mm_oral.ferx $D/mm_oral.csv
run mm_iv.ferx $D/mm_iv.csv
