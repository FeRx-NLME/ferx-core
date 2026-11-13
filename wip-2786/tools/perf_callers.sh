#!/bin/bash
# perf_callers.sh <example> <data> <symbol-substring>: who calls a hot symbol (DWARF call graph).
W=/root/package/ferx-core/.worktrees/lanes
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
r=$(mktemp -d /scratch/mu/ad/kj/pc.XXXX)
cp "$M" "$r/m.ferx"
cd "$r" || exit 1
export FERX_KJIT=1
perf record -q -F ${F:-400} --call-graph dwarf,16384 -o perf.data taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
perf report -i perf.data --no-children --sort symbol --symbol-filter="$3" -G --stdio 2>/dev/null | grep -E '[0-9]+\.[0-9]+%' | head -${N:-40} | rustfilt | sed -E 's/<[^<>]*>//g; s/::h[0-9a-f]{16}//g' | cut -c1-170
cd /scratch/mu/ad/kj && rm -rf "$r"
