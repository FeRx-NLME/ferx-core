#!/bin/bash
# revlane_ab.sh [rounds]: this branch's binary, reverse kernels starting on the scalar function
# (A) vs on the lane tier (B, FERX_TMP_REVLANE), interleaved; best wall time and OFV bits.
W=/root/package/ferx-core/.worktrees/lanes
S=/scratch/speed/bench
BIN=/scratch/mu/ad/ferx-kjit-inc
R=${1:-3}
one() {
  local m=$1 d=$2 lane=$3
  local r; r=$(mktemp -d /scratch/mu/ad/kj/rl.XXXX)
  cp "$m" "$r/m.ferx"
  if [ "$lane" = 1 ]; then export FERX_TMP_REVLANE=1; else unset FERX_TMP_REVLANE; fi
  local s; s=$(date +%s%N)
  (cd "$r" && timeout 3600 "$BIN" m.ferx --data "$d" > out.txt 2> err.txt)
  local rc=$?
  local e=$(( ($(date +%s%N) - s) / 1000000 ))
  echo "$e $rc $(grep -m1 -E '^OFV' "$r/out.txt" | awk '{print $2}')"
  rm -rf "$r"
}
for c in "$W/examples/ctmm_2state.ferx@$W/data/ctmm_2state.csv" \
         "$W/examples/mctmm_3state.ferx@$W/data/mctmm_3state.csv" \
         "$W/examples/three_cpt_iv_ode.ferx@$W/data/three_cpt_iv.csv" \
         "$S/pk2.ferx@$S/pk2_n300.csv" "$S/wiov.ferx@$S/wiov_n200.csv"; do
  m=${c%%@*}; d=${c##*@}
  ba=999999999; bb=999999999
  for i in $(seq "$R"); do
    read -r ta ra oa < <(one "$m" "$d" 0)
    read -r tb rb ob < <(one "$m" "$d" 1)
    [ "$ra" = 0 ] && [ "$ta" -lt "$ba" ] && ba=$ta
    [ "$rb" = 0 ] && [ "$tb" -lt "$bb" ] && bb=$tb
  done
  echo "$(basename "$m"): func-first ${ba}ms  lane-first ${bb}ms  ofv $oa / $ob"
done
