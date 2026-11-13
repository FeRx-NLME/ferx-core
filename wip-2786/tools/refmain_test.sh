#!/bin/bash
# refmain_test.sh [filter]: build main's lib tests (reference worktree), then run the filter.
cd /scratch/mu/ref-main || exit 1
pkill -x rust-analyzer; pkill -f "message-format=json"; sleep 1
CARGO_PROFILE_DEV_DEBUG=0 cargo test --lib --features survival,markov,nn -j2 \
  --target-dir /scratch/tgt-refmain --no-run > /scratch/mu/ad/refmain-build.log 2>&1
echo "build exit $?"
if [ -n "$1" ]; then
  BIN=$(ls -t /scratch/tgt-refmain/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$\|\.o$\|\.txt$" | head -1)
  timeout 1800 "$BIN" "$1" --test-threads=1 --nocapture > /scratch/mu/ad/refmain-run.log 2>&1
  grep -E "^test result|BOBYQA|panicked" /scratch/mu/ad/refmain-run.log | head
fi
