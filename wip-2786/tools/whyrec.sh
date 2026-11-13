#!/bin/bash
# whyrec.sh <filter>: run a test group with plan debug on; distinct recording-failure reasons.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$\|\.o$\|\.txt$" | head -1)
FERX_KJIT_DEBUG=1 timeout 3000 "$BIN" "$1" --test-threads="${2:-1}" > /scratch/mu/ad/why.log 2>&1
grep -oE "kjit: (subject|term) not recorded: .{0,140}" /scratch/mu/ad/why.log | sort | uniq -c | sort -rn | head -30
