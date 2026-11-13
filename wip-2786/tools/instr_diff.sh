#!/bin/bash
# instr_diff.sh <example> <data>: per-symbol instruction samples, off vs on, largest differences.
W=/root/package/ferx-core/.worktrees/lanes
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
r=$(mktemp -d /scratch/mu/ad/kj/id.XXXX)
cp "$M" "$r/m.ferx"
cd "$r" || exit 1
for mode in off on; do
  if [ "$mode" = on ]; then export FERX_KJIT=1; else unset FERX_KJIT; fi
  perf record -q -e instructions:u -c 200000 -o p-$mode.data taskset -c 3 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" --threads 1 > /dev/null 2>&1
  perf report -i p-$mode.data --no-children --sort symbol --stdio -F overhead,sample,symbol 2>/dev/null | grep -E '^ +[0-9]' > $mode.txt
done
python3 - off.txt on.txt <<'EOF'
import re, sys, subprocess
def load(p):
    d = {}
    for line in open(p):
        m = re.match(r'\s*[\d.]+%\s+(\d+)\s+\[.\]\s+(.*)', line)
        if not m:
            continue
        sym = m.group(2).strip()
        sym = re.sub(r'::h[0-9a-f]{16}$', '', sym)
        d[sym] = d.get(sym, 0) + int(m.group(1))
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = set(a) | set(b)
diff = sorted(keys, key=lambda k: -(b.get(k, 0) - a.get(k, 0)))
tot_a, tot_b = sum(a.values()), sum(b.values())
print(f"total samples off {tot_a} on {tot_b}  (x0.2M instr)")
def dm(s):
    try:
        return subprocess.run(['rustfilt'], input=s, capture_output=True, text=True).stdout.strip()
    except Exception:
        return s
print("--- more on")
for k in diff[:22]:
    print(f"{b.get(k,0)-a.get(k,0):+7d}  {dm(k)[:130]}")
print("--- less on")
for k in diff[-12:]:
    print(f"{b.get(k,0)-a.get(k,0):+7d}  {dm(k)[:130]}")
EOF
cd /scratch/mu/ad/kj && rm -rf "$r"
