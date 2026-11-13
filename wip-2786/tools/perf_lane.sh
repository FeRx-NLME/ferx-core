#!/bin/bash
# perf_lane.sh <example> <data> [T1]: user-space instruction profile with the compiled path on
# and the promotion point at T1 (default: never compile — the lane tier alone).
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/pf.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_T1=${3:-100000000} perf record -q -e instructions:u -c 100000 -o perf.data taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
perf report -i perf.data --no-children --sort symbol --stdio 2> /dev/null | grep -E '^ +[0-9]' | head -${N:-25} > top.txt
rustfilt < top.txt | sed -E 's/::h[0-9a-f]{16}//' | cut -c1-150
cd /
rm -rf "$r"
