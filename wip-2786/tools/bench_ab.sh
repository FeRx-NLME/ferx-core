#!/bin/bash
# bench_ab.sh [rounds]: the agreed large benchmark sets, merge-base reference (A) vs this branch
# with the compiled path on (B), interleaved A/B per round; best wall time each and OFVs.
S=/scratch/speed/bench
A=/scratch/mu/ad/ferx-main-feat
B=/scratch/mu/ad/ferx-kjit-inc
R=${1:-3}
one() {
  local bin=$1 m=$2 d=$3 on=$4
  local r; r=$(mktemp -d /scratch/mu/ad/kj/ab.XXXX)
  cp "$S/$m.ferx" "$r/m.ferx"
  if [ "$on" = 1 ]; then export FERX_KJIT=1; else unset FERX_KJIT; fi
  unset FERX_KJIT_DEBUG
  local s; s=$(date +%s%N)
  (cd "$r" && timeout 3600 "$bin" m.ferx --data "$S/$d.csv" --clean > out.txt 2> err.txt)
  local rc=$?
  local e=$(( ($(date +%s%N) - s) / 1000000 ))
  echo "$e $rc $(grep -m1 -E '^OFV' "$r/out.txt" | awk '{print $2}')"
  rm -rf "$r"
}
for pair in ${CASES:-pk2@pk2_n300 pk2_ode@pk2_n300 wiov@wiov_n200 mm@mm_n200}; do
  m=${pair%%@*}; d=${pair##*@}
  ba=999999999; bb=999999999; oa=; ob=
  for i in $(seq "$R"); do
    read -r ta ra oa < <(one $A "$m" "$d" 0)
    read -r tb rb ob < <(one $B "$m" "$d" 1)
    [ "$ra" = 0 ] && [ "$ta" -lt "$ba" ] && ba=$ta
    [ "$rb" = 0 ] && [ "$tb" -lt "$bb" ] && bb=$tb
    echo "  $m round $i: A ${ta}ms rc=$ra  B ${tb}ms rc=$rb"
  done
  echo "## $m: A ${ba}ms  B ${bb}ms  B/A $(python3 -c "print(f'{$bb/$ba:.2f}')")  ofv A $oa  B $ob"
done
