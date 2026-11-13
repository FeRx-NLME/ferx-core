#!/bin/bash
# vsmain.sh: build, then full fits on main (merge base) vs this branch with the compiled path on.
# Prints OFV of both and the estimate diff per model.
J=6 bash /scratch/mu/ad/kj/build_inc.sh || exit 1
W=/root/package/ferx-core/.worktrees/lanes
D=$W/data
M=/scratch/mu/ad/ferx-main-feat
B=/scratch/mu/ad/ferx-kjit-inc
one() {
  local tag=$1 bin=$2 ex=$3 data=$4
  r=$(mktemp -d /scratch/mu/ad/kj/vm.XXXX)
  cp "$W/examples/$ex" "$r/m.ferx"
  if [ "$tag" = on ]; then export FERX_KJIT=1 FERX_KJIT_DEBUG=1; else unset FERX_KJIT FERX_KJIT_DEBUG; fi
  s=$(date +%s%N)
  (cd "$r" && timeout 1500 "$bin" m.ferx --data "$data" > out.txt 2> err.txt)
  rc=$?
  e=$(( ($(date +%s%N) - s) / 1000000 ))
  echo "  $tag rc=$rc t=${e}ms $(grep -m1 -E '^OFV' "$r/out.txt") $(grep -m1 -oE 'guard-fails [0-9]+' "$r/err.txt") $(grep -m1 -oE 'bad [0-9]+' "$r/err.txt")"
  grep -E "^ +[A-Za-z_0-9]+: -?[0-9][0-9.e+-]*$" "$r/m-fit.yaml" | grep -viE "elapsed|time|seconds|_ms|threads" > "/scratch/mu/ad/kj/vm-$tag.txt"
  rm -rf "$r"
}
for pair in "warfarin.ferx warfarin.csv" "two_cpt_oral_cov.ferx two_cpt_oral_cov.csv" \
  "warfarin_iov.ferx warfarin.csv" "one_cpt_transit.ferx transit_oral.csv" \
  "oral_with_lagtime.ferx one_cpt_oral_lagtime.csv" "warfarin_ode.ferx warfarin.csv" \
  "three_cpt_iv_ode.ferx three_cpt_iv.csv" "mm_oral.ferx mm_oral.csv" \
  "tte_weibull.ferx tte_weibull.csv" "rtte_weibull_reset.ferx rtte_weibull_reset.csv" \
  "binary_logistic.ferx binary_logistic.csv" "ctmm_2state.ferx ctmm_2state.csv" \
  "mctmm_3state.ferx mctmm_3state.csv" "pktte_joint.ferx pktte_joint.csv"; do
  set -- $pair
  echo "### $1"
  one main $M "$1" "$D/$2"
  one on $B "$1" "$D/$2"
  if diff -q /scratch/mu/ad/kj/vm-main.txt /scratch/mu/ad/kj/vm-on.txt > /dev/null; then
    echo "  estimates identical"
  else
    diff /scratch/mu/ad/kj/vm-main.txt /scratch/mu/ad/kj/vm-on.txt | head -6 | sed 's/^/  /'
  fi
done
