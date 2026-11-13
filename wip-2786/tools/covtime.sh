#!/bin/bash
# covtime.sh <example> <data> [reps]: covariance_wall_time_secs and wall time, compiled path
# off/on interleaved; medians.
B=${B:-/scratch/mu/ad/ferx-kjit-inc}
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/ct.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
: > off.txt
: > on.txt
for i in $(seq "${3:-7}"); do
  for mode in off on; do
    if [ "$mode" = on ]; then export FERX_KJIT=1; else unset FERX_KJIT; fi
    taskset -c 3 "$B" m.ferx --data "$2" --threads 1 > /dev/null 2>&1
    v=$(grep -m1 covariance_wall_time_secs m-fit.yaml | awk '{print $2}')
    w=$(grep -m1 -E '^  wall_time_secs' m-fit.yaml | awk '{print $2}')
    echo "$v $w" >> "$mode.txt"
  done
done
python3 - "$1" <<'PY'
import sys, statistics as st
def m(f, c): return st.median(float(l.split()[c]) for l in open(f) if l.strip())
print(f"{sys.argv[1]}: cov off {m('off.txt',0):.4f} on {m('on.txt',0):.4f} | fit off {m('off.txt',1):.4f} on {m('on.txt',1):.4f}")
PY
cd /
rm -rf "$r"
