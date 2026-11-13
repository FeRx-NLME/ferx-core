#!/bin/bash
# runmany.sh <filter>...: each filter's test result line, from the latest lib test binary.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* 2>/dev/null | grep -v '\.d$' | head -1)
[ -z "$BIN" ] && BIN=$(ls -t /scratch/tgt-lanes-test/debug/deps/ferx_core-* | grep -v '\.d$' | head -1)
for t in "$@"; do
  timeout 1800 "$BIN" "$t" --test-threads=1 > /scratch/mu/ad/runmany-last.log 2>&1
  echo "$t: $(grep -E '^test result' /scratch/mu/ad/runmany-last.log)"
  grep -E "panicked" -A2 /scratch/mu/ad/runmany-last.log | head -6 | cut -c1-250
done
