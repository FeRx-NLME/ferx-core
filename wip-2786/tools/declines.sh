#!/bin/bash
# declines.sh <filter>...: run each filter group with the recorder's debug on; list distinct
# decline reasons (kernel/term/subject not recorded, fresh-recording declines) with counts,
# and the groups' pass/fail tallies.
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$\|\.o$\|\.txt$" | head -1)
: > /scratch/mu/ad/declines.log
for f in "$@"; do
  FERX_KJIT_DEBUG=1 timeout 900 "$BIN" "$f" --test-threads=2 >> /scratch/mu/ad/declines.log 2>&1
done
echo "== declines"
grep -o "kjit: [a-z ]*not recorded: [^0-9]*[0-9]*\|kjit: a fresh recording declines" /scratch/mu/ad/declines.log | sort | uniq -c | sort -rn | head -20
echo "== results"
grep -E "^test result" /scratch/mu/ad/declines.log
grep -E " FAILED$" /scratch/mu/ad/declines.log | head -40
