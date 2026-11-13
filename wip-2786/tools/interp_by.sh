#!/bin/bash
# interp_by.sh <example> <data>: kjit counters and interpreted instructions per layout.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/w.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 2> err.txt > /dev/null
grep -E 'kjit: (compiled|interpreted)' err.txt
echo "compiles: $(grep -c kjit-compile err.txt)"
cd /
rm -rf "$r"
