#!/bin/bash
# bench_kernels.sh <example> <data>: per-kernel compile/run benchmark (FERX_KJIT_BENCH), short fit.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/bk.XXXX)
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
python3 - "$M" "$r/m.ferx" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
t = re.sub(r"^(\s*maxiter\s*=\s*)\d+", r"\g<1>3", t, flags=re.M)
open(sys.argv[2], "w").write(t)
EOF
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_BENCH=1 taskset -c 2 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2> err.txt
grep kjit-bench err.txt | sort -k3 -n -r | head -${N:-12}
grep 'kjit-bench TOTAL' err.txt
cd /scratch/mu/ad/kj && rm -rf "$r"
