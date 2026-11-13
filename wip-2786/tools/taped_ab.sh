#!/bin/bash
# Compiled path off / inner only / inner + taped: instructions, OFV, counters.
B=${B:-/scratch/mu/ad/ferx-kjit-cf}
S=/scratch/speed/bench
export FERX_KJIT_GLOBAL_CSE=1
for m in ${MODELS:-pk1@pk1_n500 pk2@pk2_het wiov@wiov_het}; do
  name=${m%@*}; data=${m#*@}
  for mode in off inner taped; do
    r=$(mktemp -d /scratch/mu/ad/kj/t.XXXX); cp "$S/$name.ferx" "$r/m.ferx"
    out=$(cd "$r" && case $mode in
        inner) export FERX_KJIT=1 FERX_KJIT_DEBUG=1 FERX_KJIT_TAPED=0 ;;
        taped) export FERX_KJIT=1 FERX_KJIT_DEBUG=1 ;;
      esac; taskset -c 2 perf stat -x, -e instructions "$B" m.ferx --data "$S/$data.csv" 2>&1)
    ins=$(echo "$out" | awk -F, '/instructions/{print $1}')
    ofv=$(echo "$out" | grep -m1 -E '^OFV' | grep -oE '[-0-9]+\.[0-9]+')
    st=$(echo "$out" | grep -oE 'runs [0-9]+ taped [0-9]+ guard-fails [0-9]+' | head -1)
    val=$(echo "$out" | grep -c 'failed validation')
    printf "%-6s %-9s %-6s %8.2fG  OFV %-12s %s  valfail=%s\n" "$name" "$data" "$mode" "$(python3 -c "print($ins/1e9)")" "$ofv" "$st" "$val"
    rm -rf "$r"
  done
done
