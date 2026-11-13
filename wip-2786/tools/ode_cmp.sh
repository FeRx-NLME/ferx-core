#!/bin/bash
# ode_cmp.sh: ODE example fits with the compiled path off and on.
W=/root/package/ferx-core/.worktrees/lanes
PAIRS=${PAIRS:-warfarin_ode.ferx:warfarin.csv two_cpt_iv_ode.ferx:two_cpt_iv.csv two_cpt_oral_ode.ferx:two_cpt_oral_cov.csv three_cpt_iv_ode.ferx:three_cpt_iv.csv}
for pair in $PAIRS; do
  m=${pair%%:*}; d=${pair#*:}
  echo "### $m"
  bash /scratch/mu/ad/kj/fitcmp.sh "$m" "$W/data/$d"
done
