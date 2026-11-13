#!/bin/bash
# perf_bench.sh <model> <data>: perf profile of one benchmark-set fit, compiled path on;
# prints the top symbols (self time).
r=/scratch/mu/ad/kj/perf_run
rm -rf "$r"; mkdir -p "$r"
cp "/scratch/speed/bench/$1.ferx" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 perf record -F 999 -g -o perf.data -- /scratch/mu/ad/ferx-kjit-inc m.ferx --data "/scratch/speed/bench/$2.csv" --clean > out.txt 2> err.txt
echo "rc=$? $(grep -m1 OFV out.txt) $(grep -m1 Elapsed out.txt)"
perf report -i perf.data --no-children --sort symbol --stdio 2>/dev/null | grep -E '^ +[0-9]' | head -40 | cut -c1-150
echo "--- inclusive"
perf report -i perf.data --children --sort symbol --stdio 2>/dev/null | grep -E '^ +[0-9]' | grep -iE 'record|plan|kjit|kernel_jit|compile|cranelift|lower|build_chunk|fold|Recording|format|hash' | head -25 | cut -c1-150
