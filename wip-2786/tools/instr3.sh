#!/bin/bash
# instr3.sh <example> <data>: instructions and cycles for: compiled path off; on without the
# planned covariance step; on. One core, the covariance step's own share from the yaml.
W=/root/package/ferx-core/.worktrees/lanes
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
r=$(mktemp -d /scratch/mu/ad/kj/ic.XXXX)
cp "$M" "$r/m.ferx"
cd "$r" || exit 1
for mode in off nocov on; do
  unset FERX_KJIT FERX_KJIT_NOCOV
  case $mode in nocov) export FERX_KJIT=1 FERX_KJIT_NOCOV=1;; on) export FERX_KJIT=1;; esac
  perf stat -x, -e instructions:u,cycles:u -o stat.txt taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > out.txt 2>&1
  printf "%-6s %s cov %ss\n" "$mode" "$(awk -F, '/instructions|cycles/ {printf "%s=%.4fG ", $3, $1/1e9}' stat.txt)" "$(grep -m1 covariance_wall_time_secs m-fit.yaml | awk '{print $2}')"
done
cd / && rm -rf "$r"
