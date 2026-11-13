#!/bin/bash
# Rebuild, then time the covariance step on four models.
J=8 bash /scratch/mu/ad/kj/build_inc.sh || exit 1
D=/root/package/ferx-core/.worktrees/lanes/data
K=/scratch/mu/ad/kj
bash $K/covtime.sh warfarin.ferx $D/warfarin.csv
bash $K/covtime.sh two_cpt_oral_cov.ferx $D/two_cpt_oral_cov.csv
bash $K/covtime.sh mm_oral.ferx $D/mm_oral.csv
bash $K/covtime.sh warfarin_ode.ferx $D/warfarin.csv
