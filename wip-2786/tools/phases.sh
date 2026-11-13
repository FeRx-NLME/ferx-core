#!/bin/bash
# phases.sh <example> <data>: inclusive user-space instruction share of the fit's phases,
# compiled path off vs on (DWARF call graphs).
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/ph.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
PAT='reverse_seeds|run_loop_rev|call_seeds|LaneProg|Decoded|interpret|forward_logged|run_loop|lower_chunk_opt|promote|Tiered|kernel_jit_loop|LoopRun|Hook|Lanes'
for mode in off on; do
  unset FERX_KJIT
  [ "$mode" = on ] && export FERX_KJIT=1
  perf record -q -e instructions:u -F 2000 --call-graph dwarf,32768 -o p.data taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
  perf report -i p.data --children --sort symbol --stdio -g none 2> /dev/null | grep -E '^ +[0-9]' > rep.txt
  echo "=== $mode"
  rustfilt < rep.txt | sed -E 's/::h[0-9a-f]{16}//; s/<[^<>]*>//g; s/<[^<>]*>//g' | grep -E "$PAT" | cut -c1-160 | head -40
done
cd /
rm -rf "$r"
