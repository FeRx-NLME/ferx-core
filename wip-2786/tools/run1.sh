#!/bin/bash
# run1.sh <filter>: run matching lib tests from the latest test binary, single-threaded, with output.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v '\.d$' | head -1)
timeout 1800 "$BIN" "$1" --test-threads=1 --nocapture > /scratch/mu/ad/run1.log 2>&1
echo "rc=$?"
grep -vE '^\s*$' /scratch/mu/ad/run1.log | grep -vE '^kjit-compile' | tail -${N:-30} | cut -c1-240
