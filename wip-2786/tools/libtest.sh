#!/bin/bash
# libtest.sh [filters...]: lib tests with every feature, low-memory profile, one job pair.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
# rust-analyzer restarts on its own and the two do not fit in memory together: keep it down
# for the whole run.
( while true; do pkill -x rust-analyzer; sleep 3; done ) &
RA_KILLER=$!
trap 'kill $RA_KILLER 2>/dev/null' EXIT
sleep 1
s=$(date +%s)
CARGO_PROFILE_DEV_DEBUG=0 cargo test --lib --features survival,markov,nn -j2 \
  --target-dir /scratch/tgt-lanes-test --no-run > /scratch/mu/ad/libtest-build.log 2>&1
rc=$?
echo "build exit $rc after $(( $(date +%s) - s )) s"
if [ $rc -ne 0 ]; then
  grep -E '^error' -A8 /scratch/mu/ad/libtest-build.log | head -40
  exit $rc
fi
s=$(date +%s)
CARGO_PROFILE_DEV_DEBUG=0 cargo test --lib --features survival,markov,nn -j2 \
  --target-dir /scratch/tgt-lanes-test -- --test-threads=3 "$@" > /scratch/mu/ad/libtest.log 2>&1
rc=$?
echo "test exit $rc after $(( $(date +%s) - s )) s"
grep -E '^test result|FAILED|panicked' /scratch/mu/ad/libtest.log | head -40
exit $rc
