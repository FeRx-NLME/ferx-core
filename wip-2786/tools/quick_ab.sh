#!/bin/bash
# Quick off/on instructions for a few models with one binary (directional, not the final A/B).
B=${B:-/scratch/mu/ad/ferx-kjit-cf}
S=/scratch/speed/bench
for m in ${MODELS:-pk1@pk1_n500 pk2@pk2_n300 wiov@wiov_n200}; do
  name=${m%@*}; data=${m#*@}
  for mode in off on; do
    r=$(mktemp -d /scratch/mu/ad/kj/q.XXXX); cp "$S/$name.ferx" "$r/m.ferx"
    out=$(cd "$r" && if [ "$mode" = on ]; then export FERX_KJIT=1; fi; taskset -c 2 perf stat -x, -e instructions "$B" m.ferx --data "$S/$data.csv" 2>&1)
    ins=$(echo "$out" | awk -F, '/instructions/{print $1}')
    ofv=$(echo "$out" | grep -m1 -E '^OFV' | grep -oE '[-0-9]+\.[0-9]+')
    printf "%-6s %-4s %8.2fG  OFV %s\n" "$name" "$mode" "$(python3 -c "print($ins/1e9)")" "$ofv"
    rm -rf "$r"
  done
done
