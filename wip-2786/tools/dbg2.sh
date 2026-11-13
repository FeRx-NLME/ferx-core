#!/bin/bash
D=/root/package/ferx-core/.worktrees/lanes/data
for m in zero_order_absorption warfarin_ode_time; do
  echo "######## $m"
  bash /scratch/mu/ad/kj/dbg_model.sh "$m.ferx" "$D/warfarin.csv" 2>&1 | head -20
done
