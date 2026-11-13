#!/bin/bash
# perf_on.sh <example> <data> [on|off]: perf profile of one fit; top symbols.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/pf.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
if [ "${3:-on}" = on ]; then export FERX_KJIT=1; fi
perf record -q -g -o perf.data taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
perf report -i perf.data --no-children --sort symbol --stdio 2>/dev/null | grep -E '^ +[0-9]' | head -${N:-30} | cut -c1-160
cd /scratch/mu/ad/kj && rm -rf "$r"
