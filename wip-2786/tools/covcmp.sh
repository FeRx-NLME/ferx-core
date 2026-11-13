#!/bin/bash
# covcmp.sh <example> <data>: covariance SEs with the compiled path off and on.
B=${B:-/scratch/mu/ad/ferx-kjit-inc}
W=/root/package/ferx-core/.worktrees/lanes
for mode in off on; do
  r=$(mktemp -d /scratch/mu/ad/kj/cc.XXXX)
  cp "$W/examples/$1" "$r/m.ferx"
  if [ "$mode" = on ]; then export FERX_KJIT=1 FERX_KJIT_DEBUG=1; else unset FERX_KJIT FERX_KJIT_DEBUG; fi
  s0=$SECONDS; (cd "$r" && timeout 1200 "$B" m.ferx --data "$2" > out.txt 2> err.txt)
  echo "== $mode $((SECONDS-s0)) s $(grep -m1 -oE "kjit: compiled .*" $r/err.txt | grep -oE "taped [0-9]+|cov [0-9]+" | tr "\n" " ")"
  sed -n '/^theta:/,/^warnings:/p' "$r/m-fit.yaml" > "/scratch/mu/ad/kj/cc-$mode.txt"
  rm -rf "$r"
done
python3 - <<'PY'
import re
a=open('/scratch/mu/ad/kj/cc-off.txt').read().splitlines(); b=open('/scratch/mu/ad/kj/cc-on.txt').read().splitlines()
num=re.compile(r'-?\d+\.\d+(?:e[-+]?\d+)?')
import math
w=0; d=0; sc=0
for x,y in zip(a,b):
    for u,v in zip(num.findall(x),num.findall(y)):
        u,v=float(u),float(v)
        assert math.isfinite(u) and math.isfinite(v)
        d=max(d,abs(u-v)); sc=max(sc,abs(u),abs(v))
print("lines",len(a),len(b),"max abs diff",d,"norm-wise",d/max(sc,1e-300))
PY
