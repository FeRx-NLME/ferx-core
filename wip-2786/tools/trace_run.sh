#!/bin/bash
# trace_run.sh <example> <data>: debug run; compare traces at the first spot failure.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/tr.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 timeout 900 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2> err.txt
python3 /scratch/mu/ad/kj/trace_cmp.py err.txt
cp err.txt /scratch/mu/ad/kj/trace-err.txt
cd /scratch/mu/ad/kj && rm -rf "$r"
