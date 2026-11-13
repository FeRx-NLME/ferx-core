#!/bin/bash
# stats_full.sh <example> <data>: one full fit with kjit debug; counters and compile lines.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/sf.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 taskset -c 2 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > out.txt 2> err.txt
grep -E '^kjit: compiled|^kjit: [0-9]+ distinct' err.txt
grep -E '^kjit-compile' err.txt | awk '{n++; s+=$(NF-1)} END {print n " compiles, " s " ms"}'
grep -E '^kjit-compile' err.txt | sort -k12 -n -r | head -5
grep -E 'kjit: (subject not recorded|validation failed|spot)' err.txt | sort | uniq -c | sort -rn | head -5
cd /scratch/mu/ad/kj && rm -rf "$r"
