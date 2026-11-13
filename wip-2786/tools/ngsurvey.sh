#!/bin/bash
# ngprobe.sh: record every survival model's non-Gaussian term once per subject; outcomes.
# (built)
W=/root/package/ferx-core/.worktrees/lanes
for m in tte_weibull tte_exponential tte_gompertz tte_competing_risks rtte_exponential rtte_weibull_reset binary_logistic ctmm_2state ctmm_pd_2state mctmm_3state pktte_joint dtmm_2state; do
  r=$(mktemp -d /scratch/mu/ad/kj/ng.XXXX)
  cp "$W/examples/$m.ferx" "$r/m.ferx"
  cd "$r" || exit 1
  d=$W/data/$m.csv
  [ -f "$d" ] || d=$(ls $W/data/${m%%_*}*.csv 2>/dev/null | head -1)
  FERX_KJIT=1 FERX_KJIT_NGPROBE=1 FERX_KJIT_SURVEY=1 timeout 300 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$d" > out.txt 2> err.txt
  echo "== $m"; grep -h "kjit-survey" err.txt | tr "|" "\n" | sed "s/^ *//" | sort -u | head -40
  cd /
  rm -rf "$r"
done
