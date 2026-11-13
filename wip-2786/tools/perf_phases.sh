#!/bin/bash
# perf_phases.sh <example> <data> [on|off]: inclusive (children) cost of key phase functions.
W=/root/package/ferx-core/.worktrees/lanes
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
r=$(mktemp -d /scratch/mu/ad/kj/pp.XXXX)
cp "$M" "$r/m.ferx"
cd "$r" || exit 1
if [ "${3:-on}" = on ]; then export FERX_KJIT=1; fi
perf record -q -g -o perf.data taskset -c 2 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
perf report -i perf.data --children --sort symbol --stdio 2>/dev/null > rep.txt
grep -E '^ +[0-9]' rep.txt | grep -iE "${PAT:-covariance|run_covariance|outer|inner_|find_ebe|kjit|canonical_jets|record|linear_segment|spectral|rtape|ode_predictions|compute_predictions|sdtab|postfit|cwres|npde|lower_chunk|Tiered|interpret|cranelift}" | head -${N:-40} | awk '{printf "%7s %7s %s\n", $1, $2, substr($0, index($0,$5), 140)}'
cd /scratch/mu/ad/kj && rm -rf "$r"
