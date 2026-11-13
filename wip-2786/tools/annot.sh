#!/bin/bash
# annot.sh <example> <data> <symbol-substring>: hottest instructions of one symbol (lane tier
# alone), with their source lines.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/an.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_T1=${T1:-100000000} perf record -q -e instructions:u -c 20000 -o perf.data taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
sym=$(perf report -i perf.data --no-children --sort symbol --stdio 2> /dev/null | grep -E '^ +[0-9]' | grep -m1 "$3" | awk '{print $3}')
perf annotate -i perf.data --stdio -l "$sym" 2> /dev/null > ann.txt
grep -E '^\s+[0-9]+\.[0-9]+' ann.txt | sort -rn | head -${N:-40}
cd /
rm -rf "$r"
