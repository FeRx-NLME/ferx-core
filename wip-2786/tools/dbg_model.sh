#!/bin/bash
# dbg_model.sh <example> [data]: one fit with the compiled path on and its debug output
# (kjit lines only, deduplicated with counts).
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/dm.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
D=${2:-$(grep -oE 'data/[A-Za-z0-9_./-]+\.csv' m.ferx | head -1)}
case "$D" in /*) ;; *) D="$W/$D";; esac
FERX_KJIT=1 FERX_KJIT_DEBUG=1 timeout 600 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$D" > out.txt 2> err.txt
echo "exit $?  data $D"
grep -E '^kjit' err.txt | sed -E 's/[0-9]+\.[0-9]+e?-?[0-9]*/N/g' | sort | uniq -c | sort -rn | head -${N:-25}
grep -m3 -iE 'error' err.txt out.txt
cd /
rm -rf "$r"
