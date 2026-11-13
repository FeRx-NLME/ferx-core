#!/bin/bash
# exit1.sh: run the models that exit with 1 in the sweep, compiled path off and on, and show
# their errors.
W=/root/package/ferx-core/.worktrees/lanes
for m in one_cpt_oral_baseline warfarin_derived warfarin_derived_pkpd warfarin_scaled warfarin_update; do
  r=$(mktemp -d /scratch/mu/ad/kj/e1.XXXX)
  cp "$W/examples/$m.ferx" "$r/m.ferx"
  cd "$r" || exit 1
  for mode in off on; do
    unset FERX_KJIT
    [ $mode = on ] && export FERX_KJIT=1
    timeout 600 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$W/data/warfarin.csv" > out.txt 2>&1
    echo "$m $mode exit $? : $(grep -m1 -iE 'error' out.txt | cut -c1-200)"
  done
  cd /
  rm -rf "$r"
done
