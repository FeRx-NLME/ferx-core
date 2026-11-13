#!/bin/bash
# onetest.sh <filter> [seconds]: one test, single-threaded, with output, under a time limit.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$\|\.o$\|\.txt$" | head -1)
s=$(date +%s)
timeout "${2:-300}" "$BIN" "$1" --test-threads=1 --nocapture > /scratch/mu/ad/one.log 2>&1
echo "rc=$? after $(( $(date +%s) - s )) s"
grep -E "^test result|panicked|kjit" /scratch/mu/ad/one.log | grep -v kjit-compile | head -20 | cut -c1-250
