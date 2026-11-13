#!/bin/bash
# perf_tree.sh <example> <data> [on|off]: DWARF call-graph profile; inclusive cost of ferx frames.
W=/root/package/ferx-core/.worktrees/lanes
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
r=$(mktemp -d /scratch/mu/ad/kj/pt.XXXX)
cp "$M" "$r/m.ferx"
cd "$r" || exit 1
if [ "${3:-on}" = on ]; then export FERX_KJIT=1; fi
perf record -q -F ${F:-300} --call-graph dwarf,32768 -o perf.data taskset -c 2 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
perf report -i perf.data --children --sort symbol --stdio --no-demangle 2>/dev/null > rep.txt
grep -E '^ +[0-9]+\.[0-9]+%' rep.txt | grep ferx_core | head -${N:-60} | awk '{printf "%7s %7s  ", $1, $2; s=$0; sub(/.*\] /,"",s); print substr(s,1,150)}' > top.txt
python3 - top.txt <<'EOF'
import re, subprocess, sys
for line in open(sys.argv[1]):
    m = re.match(r'(\s*\S+\s+\S+\s+)(\S+)', line)
    if not m:
        continue
    sym = m.group(2)
    try:
        d = subprocess.run(['rustfilt'], input=sym, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        d = sym
    d = re.sub(r'<[^<>]*>', '', d)
    d = re.sub(r'::h[0-9a-f]{16}$', '', d)
    print(m.group(1) + d[:140])
EOF
cd /scratch/mu/ad/kj && rm -rf "$r"
