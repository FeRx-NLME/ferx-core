#!/bin/bash
# grp.sh <filter> [threads]: run a lib-test group from the latest binary; summary + failures.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$\|\.o$\|\.txt$" | head -1)
timeout 3000 "$BIN" "$1" --test-threads="${2:-3}" > /scratch/mu/ad/grp.log 2>&1
grep -E "^test result|FAILED|panicked" /scratch/mu/ad/grp.log | head -40 | cut -c1-260
