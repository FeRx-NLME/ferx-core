#!/bin/bash
# groups.sh: build the lib tests, then run each named group alone; failures with their panics.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
pkill -x rust-analyzer; pkill -f "message-format=json"; sleep 1
CARGO_PROFILE_DEV_DEBUG=0 cargo test --lib --features survival,markov,nn -j2 \
  --target-dir /scratch/tgt-lanes-test --no-run > /scratch/mu/ad/libtest-build.log 2>&1 || { grep -E '^error' -A6 /scratch/mu/ad/libtest-build.log | head -30; exit 1; }
BIN=$(ls -t /scratch/tgt-lanes-test/debug/deps/ferx_core-* 2>/dev/null | grep -v '\.d$' | head -1)
[ -x "$BIN" ] || BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | head -1)
echo "bin $BIN"
for g in "$@"; do
  timeout 1800 "$BIN" "$g" --test-threads=3 > "/scratch/mu/ad/grp-${g//:/_}.log" 2>&1
  echo "## $g: $(grep 'test result' "/scratch/mu/ad/grp-${g//:/_}.log")"
  grep -E "^test .* FAILED$" "/scratch/mu/ad/grp-${g//:/_}.log" | head -8
  grep -A2 "panicked at" "/scratch/mu/ad/grp-${g//:/_}.log" | grep -v "^--" | head -12 | cut -c1-220
done
