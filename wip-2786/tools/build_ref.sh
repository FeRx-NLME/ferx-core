#!/bin/bash
# build_ref.sh: the merge-base reference binary with every feature (survival, markov, nn).
cd /scratch/mu/ref-main || exit 1
pkill -x rust-analyzer; pkill -f "message-format=json"; sleep 1
s=$(date +%s)
CARGO_PROFILE_CI_FAST_INCREMENTAL=true CARGO_PROFILE_CI_FAST_CODEGEN_UNITS=256 \
  cargo build --profile ci-fast -p ferx-cli --features ferx-core/survival,ferx-core/markov,ferx-core/nn -j"${J:-6}" --target-dir /scratch/tgt-refmain > /scratch/mu/ad/build-ref.log 2>&1
rc=$?
echo "ref exit $rc after $(( $(date +%s) - s )) s"
grep -E '^error' -A8 /scratch/mu/ad/build-ref.log | head -20
[ $rc -eq 0 ] && cp /scratch/tgt-refmain/ci-fast/ferx /scratch/mu/ad/ferx-main-feat
exit $rc
