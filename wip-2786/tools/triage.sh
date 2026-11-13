#!/bin/bash
# triage.sh [full]: rerun the tests in /scratch/mu/ad/failing.txt one by one (`full`: first
# refill it from the last full-suite log's failures).
if [ "$1" = full ]; then
  grep -E "FAILED$" /scratch/mu/ad/libtest-full.log | awk '{print $2}' | sed 's/.*:://' > /scratch/mu/ad/failing.txt
fi
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$\|\.o$\|\.txt$" | head -1)
: > /scratch/mu/ad/triage.out
while read -r t; do
  timeout 600 "$BIN" "$t" --test-threads=1 > /scratch/mu/ad/triage-one.log 2>&1
  r=$(grep -E "^test result" /scratch/mu/ad/triage-one.log | grep -oE "[0-9]+ passed; [0-9]+ failed")
  echo "== $t: ${r:-TIMEOUT}" >> /scratch/mu/ad/triage.out
  grep -E "panicked" -A3 /scratch/mu/ad/triage-one.log | head -4 | cut -c1-250 >> /scratch/mu/ad/triage.out
done < /scratch/mu/ad/failing.txt
grep -c "1 passed" /scratch/mu/ad/triage.out
