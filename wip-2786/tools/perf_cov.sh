#!/bin/bash
# perf_cov.sh <example> <data>: perf profile of one fit with the compiled path on; children view
# of the covariance step's pieces and the top self symbols.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/pf.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 perf record -q -g -o perf.data taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
echo "--- children"
perf report -i perf.data --children --sort symbol --stdio 2>/dev/null | grep -E '^ +[0-9]' \
  | grep -E 'cov_hessian|hessian_with|PlannedFg|reverse_layout|forward_layout|Tiered.*(interpret|promote|call)|lower_chunk|run_loop_rev|rsweep|record_body|fit_inner|run_covariance|compute_covariance|8kernel_jit' \
  | head -30 | sed -E 's/_R[A-Za-z0-9_]*(kernel_jit|hessian|rtape|covariance|fit)/…\1/' | cut -c1-150
echo "--- self"
perf report -i perf.data --no-children --sort dso,symbol --stdio 2>/dev/null | grep -E '^ +[0-9]' | head -25 | cut -c1-150
cd /
rm -rf "$r"
