#!/bin/bash
# Stationarity audit: fit, then refit starting from the fitted estimates; report ΔOFV.
# Usage (from worktree root): bash restart_audit.sh <bin> <label> model@data ...
# model may be "ex:NAME" for examples/NAME.ferx with data/DATA.csv, or a bench model name.
B="$(cd "$(dirname "$0")" && pwd)"
BIN=$1; LABEL=$2; shift 2
OUT=$B/out/$LABEL; mkdir -p "$OUT"
for c in "$@"; do
  m=${c%%@*}; d=${c##*@}
  if [[ $m == ex:* ]]; then
    mf=examples/${m#ex:}.ferx; df=data/$d.csv; stem=${m#ex:}
  else
    mf=$B/$m.ferx; df=$B/$d.csv; stem=$m
  fi
  log1=$OUT/${stem}_fit.log
  "$BIN" "$mf" --data "$df" --clean > "$log1" 2>&1
  ofv1=$(grep -m1 -E '^OFV:' "$log1" | awk '{print $2}')
  python3 $B/evalvariant.py "$mf" "${stem}-fit.yaml" "$OUT/${stem}_restart.ferx" > /dev/null
  log2=$OUT/${stem}_restart.log
  "$BIN" "$OUT/${stem}_restart.ferx" --data "$df" --clean > "$log2" 2>&1
  ofv2=$(grep -m1 -E '^OFV:' "$log2" | awk '{print $2}')
  ev2=$(grep -E '^Eval +[0-9]+' "$log2" | tail -1 | awk '{print $2}' | tr -d ':')
  d=$(python3 -c "print(f'{float(\"$ofv2\")-float(\"$ofv1\"):+.4f}')" 2>/dev/null)
  printf "%-26s fit=%14s restart=%14s dOFV=%10s restart_evals=%s\n" "$stem" "$ofv1" "$ofv2" "$d" "$ev2"
done
