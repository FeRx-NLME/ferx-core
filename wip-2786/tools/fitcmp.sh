#!/bin/bash
# fitcmp.sh <example> <data>: full fit with the compiled path off and on; OFV, estimates, counters.
B=${B:-/scratch/mu/ad/ferx-kjit-inc}
W=/root/package/ferx-core/.worktrees/lanes
for mode in off on; do
  r=$(mktemp -d /scratch/mu/ad/kj/fc.XXXX)
  cp "$W/examples/$1" "$r/m.ferx"
  if [ "$mode" = on ]; then export FERX_KJIT=1 FERX_KJIT_DEBUG=1; else unset FERX_KJIT FERX_KJIT_DEBUG; fi
  (cd "$r" && timeout 1200 "$B" m.ferx --data "$2" > out.txt 2> err.txt)
  echo "== $mode: $(grep -m1 -E '^OFV' "$r/out.txt")  $(grep -m1 -oE 'kjit: compiled [0-9].*' "$r/err.txt" | cut -c1-140)"
  grep -E '^ +[A-Z_]+[A-Z0-9_]* +[-0-9.e]+' "$r/out.txt" | head -12 | tr -s ' ' > "/scratch/mu/ad/kj/fc-$mode.txt"
  rm -rf "$r"
done
diff -q /scratch/mu/ad/kj/fc-off.txt /scratch/mu/ad/kj/fc-on.txt > /dev/null && echo "estimates identical" || diff /scratch/mu/ad/kj/fc-off.txt /scratch/mu/ad/kj/fc-on.txt | head -10
