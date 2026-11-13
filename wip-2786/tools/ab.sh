#!/bin/bash
# Interleaved A/B/C, one core: A = main 0bda4d0a, B = prototype with FERX_KJIT=1, C = prototype off.
A=/scratch/mu/ad/ferx-main-0bda4d0a
P=/scratch/mu/ad/ferx-kjit-rel
S=/scratch/speed/bench
E=/root/package/ferx-core/.worktrees/lanes/examples
Dd=/root/package/ferx-core/.worktrees/lanes/data
CASES=${CASES:-"$S/wiov.ferx@$S/wiov_n200.csv $S/pk2.ferx@$S/pk2_n300.csv $S/mm.ferx@$S/mm_n200.csv $E/two_cpt_oral_cov.ferx@$Dd/two_cpt_oral_cov.csv"}
ROUNDS=${ROUNDS:-3}
one() { # label bin kjit model data
  local d; d=$(mktemp -d /scratch/mu/ad/kj/ab.XXXX); cp "$4" "$d/m.ferx"
  local out
  out=$(cd "$d" && if [ "$3" = 1 ]; then export FERX_KJIT=1; fi; taskset -c 2 perf stat -x, -e instructions,cycles "$2" m.ferx --data "$5" 2>&1)
  local rc=$?
  local ins cyc ofv
  ins=$(echo "$out" | awk -F, '/instructions/{print $1}')
  cyc=$(echo "$out" | awk -F, '/cycles/{print $1}')
  ofv=$(echo "$out" | grep -m1 -E '^OFV' | grep -oE '[-0-9]+\.[0-9]+')
  rm -rf "$d"
  printf "%-18s %-8s rc=%s %8.2fG instr %8.2fG cycles  OFV %s\n" "$(basename "$4" .ferx)" "$1" "$rc" "$(python3 -c "print($ins/1e9)")" "$(python3 -c "print($cyc/1e9)")" "$ofv"
}
for c in $CASES; do
  m=${c%@*}; d=${c#*@}
  for r in $(seq "$ROUNDS"); do
    one main "$A" 0 "$m" "$d"
    one kjit-on "$P" 1 "$m" "$d"
    one kjit-off "$P" 0 "$m" "$d"
  done
done
