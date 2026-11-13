#!/bin/bash
# Worst norm-wise jet difference (plan vs generic second-order recording) per example.
W=/root/package/ferx-core/.worktrees/lanes
for m in ${MODELS:-two_cpt_iv@two_cpt_iv two_cpt_oral_cov@two_cpt_oral_cov warfarin@warfarin pk1}; do
  name=${m%@*}; data=${m#*@}
  if [ "$name" = pk1 ]; then
    r=$(mktemp -d /scratch/mu/ad/kj/w.XXXX); cp /scratch/speed/bench/pk1.ferx "$r/m.ferx"
    w=$(cd "$r" && FERX_KJIT=1 FERX_KJIT_DEBUG=1 /scratch/mu/ad/ferx-kjit-cf m.ferx --data /scratch/speed/bench/pk1_n500.csv 2>&1 | grep -oE 'jets rel \S+' | awk '{print $3}' | sort -g | tail -1)
    rm -rf "$r"
  else
    w=$(N=5000 bash /scratch/mu/ad/kj/one.sh "$name.ferx" "$W/data/$data.csv" 'vs generic' | grep -oE 'jets rel \S+' | awk '{print $3}' | sort -g | tail -1)
  fi
  f=$(N=5000 bash /scratch/mu/ad/kj/one.sh "$name.ferx" "$W/data/$data.csv" 'failed validation' 2>/dev/null | wc -l)
  echo "$name worst jets rel (norm-wise) = ${w:-0}"
done
