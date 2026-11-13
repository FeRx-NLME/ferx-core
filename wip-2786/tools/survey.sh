#!/bin/bash
# Survey the value-read sites and unsupported ops each model's recording hits.
B=${B:-/scratch/mu/ad/ferx-kjit-cf}
S=/scratch/speed/bench
E=/root/package/ferx-core/.worktrees/lanes/examples
D=/root/package/ferx-core/.worktrees/lanes/data
CASES=${CASES:-"$S/pk1.ferx@$S/pk1_n500.csv $S/pk2.ferx@$S/pk2_n300.csv $E/two_cpt_oral_cov.ferx@$D/two_cpt_oral_cov.csv $E/warfarin.ferx@$D/warfarin.csv $S/wiov.ferx@$S/wiov_n200.csv"}
for c in $CASES; do
  m=${c%@*}; d=${c#*@}
  r=$(mktemp -d /scratch/mu/ad/kj/sv.XXXX); cp "$m" "$r/m.ferx"
  out=$(cd "$r" && FERX_KJIT=1 FERX_KJIT_SURVEY=1 FERX_KJIT_DEBUG=1 timeout 600 "$B" m.ferx --data "$d" 2>&1 | grep -E 'kjit' | sort | uniq -c | sort -rn | head -6 | cut -c1-900)
  echo "######## $(basename "$m")"
  echo "$out"
  rm -rf "$r"
done
