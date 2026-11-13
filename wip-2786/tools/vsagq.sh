#!/bin/bash
# vsagq.sh: Laplace / AGQ variants, merge-base reference vs this branch (no build).
W=/root/package/ferx-core/.worktrees/lanes
D=$W/data
X=/scratch/mu/ad/kj/agqex
M=/scratch/mu/ad/ferx-main-feat
B=/scratch/mu/ad/ferx-kjit-inc
one() {
  local tag=$1 bin=$2 ex=$3 data=$4
  r=$(mktemp -d /scratch/mu/ad/kj/va.XXXX)
  cp "$ex" "$r/m.ferx"
  if [ "$tag" = on ]; then export FERX_KJIT=1 FERX_KJIT_DEBUG=1; else unset FERX_KJIT FERX_KJIT_DEBUG; fi
  s=$(date +%s%N)
  (cd "$r" && timeout 1500 "$bin" m.ferx --data "$data" > out.txt 2> err.txt)
  rc=$?
  e=$(( ($(date +%s%N) - s) / 1000000 ))
  echo "  $tag rc=$rc t=${e}ms $(grep -m1 -E '^OFV' "$r/out.txt") $(grep -m1 -oE 'guard-fails [0-9]+' "$r/err.txt") $(grep -m1 -oE 'bad [0-9]+' "$r/err.txt") $(grep -m1 -oE 'taped [0-9]+' "$r/err.txt") $(grep -m1 -oE 'cov [0-9]+' "$r/err.txt")"
  [ $rc -ne 0 ] && grep -m2 -iE 'error|panic' "$r/err.txt" "$r/out.txt" | cut -c1-200
  grep -E "^ +[A-Za-z_0-9]+: -?[0-9][0-9.e+-]*$" "$r/m-fit.yaml" 2>/dev/null | grep -viE "elapsed|time|seconds|_ms|threads" > "/scratch/mu/ad/kj/va-$tag.txt"
  rm -rf "$r"
}
for pair in "warfarin_lap.ferx warfarin.csv" "warfarin_agq3.ferx warfarin.csv" \
  "warfarin_lap3.ferx warfarin.csv" "warfarin_ode_lap.ferx warfarin.csv" \
  "tte_weibull_agq3.ferx tte_weibull.csv"; do
  set -- $pair
  echo "### $1"
  one main $M "$X/$1" "$D/$2"
  one on $B "$X/$1" "$D/$2"
  if [ ! -s /scratch/mu/ad/kj/va-on.txt ]; then
    echo "  (no estimates captured)"
  elif diff -q /scratch/mu/ad/kj/va-main.txt /scratch/mu/ad/kj/va-on.txt > /dev/null; then
    echo "  estimates identical ($(wc -l < /scratch/mu/ad/kj/va-on.txt) values)"
  else
    diff /scratch/mu/ad/kj/va-main.txt /scratch/mu/ad/kj/va-on.txt | head -8 | sed 's/^/  /'
  fi
done
