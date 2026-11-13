#!/bin/bash
# whyeach.sh: for each test in /scratch/mu/ad/failing.txt, the recording-failure reasons (debug).
cd /root/package/ferx-core/.worktrees/lanes || exit 1
BIN=$(ls -t /scratch/tgt-lanes-test/debug/build/ferx-core/*/out/ferx_core-* | grep -v "\.d$\|\.o$\|\.txt$" | head -1)
: > /scratch/mu/ad/whyeach.out
while read -r t; do
  FERX_KJIT_DEBUG=1 timeout 300 "$BIN" "$t" --test-threads=1 > /scratch/mu/ad/whyeach-one.log 2>&1
  echo "== $t" >> /scratch/mu/ad/whyeach.out
  grep -oE "kjit: (subject|term) not recorded: .{0,150}|kjit: a fresh recording declines.{0,60}|kjit: loop run failed.{0,60}|kjit: guard failed #[0-9]+: [^ ]+" /scratch/mu/ad/whyeach-one.log | sort | uniq -c | sort -rn | head -6 >> /scratch/mu/ad/whyeach.out
done < /scratch/mu/ad/failing.txt
