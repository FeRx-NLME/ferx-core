#!/bin/bash
# Quick debugging builds: incremental, many codegen units, own target dir. Prints elapsed time.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
pkill -x rust-analyzer; pkill -f "message-format=json"; sleep 1
s=$(date +%s)
CARGO_PROFILE_CI_FAST_INCREMENTAL=true CARGO_PROFILE_CI_FAST_CODEGEN_UNITS=256 \
  cargo build --profile ci-fast -p ferx-cli --features ferx-core/survival,ferx-core/markov,ferx-core/nn -j"${J:-3}" --target-dir /scratch/tgt-lanes-inc > /scratch/mu/ad/build-inc.log 2>&1
rc=$?
echo "exit $rc after $(( $(date +%s) - s )) s"
grep -E '^error' -A8 /scratch/mu/ad/build-inc.log | head -30
[ $rc -eq 0 ] && cp /scratch/tgt-lanes-inc/ci-fast/ferx /scratch/mu/ad/ferx-kjit-inc
exit $rc
