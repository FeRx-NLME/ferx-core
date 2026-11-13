#!/bin/bash
# regress_ng.sh: non-Gaussian models, compiled path off vs on (estimates, counters).
J=8 bash /scratch/mu/ad/kj/build_inc.sh || exit 1
D=/root/package/ferx-core/.worktrees/lanes/data
export B=/scratch/mu/ad/ferx-kjit-inc
for m in tte_weibull tte_exponential tte_gompertz tte_competing_risks rtte_exponential rtte_weibull_reset binary_logistic ctmm_2state ctmm_pd_2state mctmm_3state pktte_joint; do
  echo "### $m"
  bash /scratch/mu/ad/kj/fitcmp.sh "$m.ferx" "$D/$m.csv" 2>&1 | grep -v '^== off'
done
