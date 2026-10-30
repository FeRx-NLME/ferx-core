#!/bin/bash
# Usage (from the ferx-core repo root): bash run.sh <ferx-binary> <label> <case>... [-- extra ferx args]
# Case names: pk1_500 pk1_2000 pk1_sparse pk2 mm
# Env vars set by the caller are passed through (FERX_XP_* prototype switches).
B="$(cd "$(dirname "$0")" && pwd)"
BIN=$1; LABEL=$2; shift 2
CASES=(); EXTRA=()
while [ $# -gt 0 ]; do
  if [ "$1" == "--" ]; then shift; EXTRA=("$@"); break; fi
  CASES+=("$1"); shift
done
OUT=$B/out/$LABEL; mkdir -p "$OUT"
for c in "${CASES[@]}"; do
  case $c in
    pk1_500)    m=pk1; d=pk1_n500 ;;
    pk1_2000)   m=pk1; d=pk1_n2000 ;;
    pk1_sparse) m=pk1; d=pk1_sparse_n300 ;;
    pk2)        m=pk2; d=pk2_n300 ;;
    mm)         m=mm;  d=mm_n200 ;;
    *@*)        m=${c%%@*}; d=${c##*@} ;;
    *) echo "unknown case $c"; continue ;;
  esac
  log=$OUT/${c//@/__}.log
  t0=$(python3 -c 'import time;print(time.time())')
  "$BIN" $B/$m.ferx --data $B/$d.csv --clean "${EXTRA[@]}" > "$log" 2>&1
  rc=$?
  t1=$(python3 -c 'import time;print(time.time())')
  wall=$(python3 -c "print(f'{$t1-$t0:.2f}')")
  ofv=$(grep -m1 -E '^OFV:' "$log" | awk '{print $2}')
  fofv=$(grep -m1 -E '^Final OFV' "$log" | awk '{print $4}')
  evals=$(grep -E '^Eval +[0-9]+' "$log" | tail -1 | awk '{print $2}' | tr -d ':')
  iters=$(grep -m1 -E '^Iterations:' "$log" | awk '{print $2}')
  conv=$(grep -m1 -E '^Converged:' "$log" | awk '{print $2}')
  xp=$(grep -m1 '^\[xp\] obj_calls' "$log" | sed 's/\[xp\] //')
  prof=$(grep -m1 '^\[profile\] inner:' "$log" | sed 's/\[profile\] inner: //' | cut -c1-60)
  printf "%-10s %-14s wall=%7ss evals=%5s iters=%5s OFV=%14s conv=%s rc=%s\n" "$LABEL" "$c" "$wall" "$evals" "$iters" "$ofv" "$conv" "$rc"
  [ -n "$xp" ] && echo "      $prof | $xp"
done
