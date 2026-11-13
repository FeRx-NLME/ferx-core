#!/bin/bash
# dump_hist.sh <example> <data>: op histogram of the first dumped chunk program.
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/dh.XXXX)
cp "$W/examples/$1" "$r/m.ferx"
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DUMP=1 timeout 300 /scratch/mu/ad/ferx-kjit-inc m.ferx --data "$2" 2> dump.txt > /dev/null
python3 - dump.txt <<'EOF'
import re, sys, collections
progs = []
cur = None
for line in open(sys.argv[1]):
    if line.startswith('kjit-chunk spec='):
        cur = []
        progs.append(cur)
    elif line.startswith('kjit-chunk   ') and cur is not None:
        cur.append(line.split(None, 2)[2].strip())
big = max(progs, key=len)
print('programs', len(progs), 'sizes', sorted(len(p) for p in progs)[-6:])
h = collections.Counter()
for ins in big:
    m = re.match(r'(\w+)(?: \{.*?op: (\w+(?:\(\d+\))?))?', ins)
    h[m.group(1) + (':' + m.group(2) if m.group(2) else '')] += 1
for k, v in h.most_common(25):
    print(f'{v:6} {k}')
EOF
cd /scratch/mu/ad/kj && rm -rf "$r"
