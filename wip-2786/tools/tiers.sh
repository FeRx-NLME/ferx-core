#!/bin/bash
# tiers.sh <filter>: run one test (with --nocapture) under each fixed tier; print its marker lines.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$" | head -1)
for t in 0 1 2 3; do
  FERX_TMP_TIER=$t timeout 1800 "$BIN" "$1" --test-threads=1 --nocapture > /scratch/mu/ad/tier$t.log 2>&1
  echo "tier $t: $(grep -E '^test result' /scratch/mu/ad/tier$t.log)"
  grep -E "^TIERBITS" /scratch/mu/ad/tier$t.log | head -5
done
