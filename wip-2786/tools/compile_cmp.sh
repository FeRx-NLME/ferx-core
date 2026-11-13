#!/bin/bash
# compile_cmp.sh: rebuild, then compile totals (µs/IR) on two models and their instruction counts.
J=8 bash /scratch/mu/ad/kj/build_inc.sh || exit 1
D=/root/package/ferx-core/.worktrees/lanes/data
bash /scratch/mu/ad/kj/compile_stats.sh warfarin.ferx $D/warfarin.csv | tail -1
bash /scratch/mu/ad/kj/compile_stats.sh three_cpt_iv_ode.ferx $D/three_cpt_iv.csv | tail -1
bash /scratch/mu/ad/kj/noval_scan.sh 2>&1 | grep -v exit
