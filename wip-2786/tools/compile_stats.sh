#!/bin/bash
# compile_stats.sh <example> <data>: every Cranelift compile of a fit: count, IR, ms, µs/IR,
# by tier and direction.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/cs.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 2> err.txt > /dev/null
grep kjit-compile err.txt | python3 -c '
import sys, re, collections
agg = collections.defaultdict(lambda: [0, 0, 0.0])
for line in sys.stdin:
    m = re.search(r"ir (\d+) lanes (\d+) m (\d+) rev (\w+) quick (\w+) (\w+) ([0-9.]+) ms", line)
    if not m: continue
    ir, lanes, mm, rev, quick, opt, ms = m.groups()
    k = (f"lanes={lanes}", f"rev={rev}", f"quick={quick}", opt)
    a = agg[k]; a[0] += 1; a[1] += int(ir); a[2] += float(ms)
tot = [0, 0, 0.0]
for k, (n, ir, ms) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
    print(" ".join(k), f"n={n} ir={ir} ms={ms:.1f} us/ir={1000*ms/max(ir,1):.2f}")
    tot[0] += n; tot[1] += ir; tot[2] += ms
print(f"total n={tot[0]} ir={tot[1]} ms={tot[2]:.1f} us/ir={1000*tot[2]/max(tot[1],1):.2f}")
'
cd /
rm -rf "$r"
