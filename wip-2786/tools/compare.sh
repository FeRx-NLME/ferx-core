#!/bin/bash
# wiov fit with the compiled recordings off and on: OFV, estimates, counters, time.
B=${1:-/scratch/mu/ad/ferx-kjit-cf}
D=/scratch/speed/bench/wiov_n200.csv
for mode in off on; do
  r=$(mktemp -d /scratch/mu/ad/kj/run.XXXX); cp /scratch/mu/ad/kj/m.ferx "$r/m.ferx"
  cd "$r" || exit 1
  if [ "$mode" = on ]; then export FERX_KJIT=1 FERX_KJIT_DEBUG=1; else unset FERX_KJIT FERX_KJIT_DEBUG; fi
  s=$(date +%s.%N)
  taskset -c 2 "$B" m.ferx --data "$D" > out.txt 2> err.txt
  rc=$?
  e=$(date +%s.%N)
  echo "== $mode  exit $rc  wall $(python3 -c "print(round($e-$s,2))") s"
  grep -m1 -iE 'OFV' out.txt
  grep -E 'kjit: (compiled|validation|subject not)' err.txt | sort | uniq -c | head -5
  grep -E '^ *(TVCL|TVV|TVKA|ETA_|KAPPA|PROP)' out.txt | head -8 | tr -s ' ' > "/scratch/mu/ad/kj/est-$mode.txt"
  cd /scratch/mu/ad/kj && rm -rf "$r"
done
diff /scratch/mu/ad/kj/est-off.txt /scratch/mu/ad/kj/est-on.txt && echo "estimates identical"
