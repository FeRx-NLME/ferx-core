#!/bin/bash
J=8 bash /scratch/mu/ad/kj/build_inc.sh || exit 1
for m in zero_order_absorption warfarin_ode_time warfarin_ode_lagtime; do
  echo "######## $m"
  bash /scratch/mu/ad/kj/dbg_model.sh "$m.ferx" 2>&1 | head -25
done
