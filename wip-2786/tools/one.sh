#!/bin/bash
# one.sh <example name> <data csv> [grep pattern]: short run of one example with kjit debug.
B=${B:-/scratch/mu/ad/ferx-kjit-cf}
W=/root/package/ferx-core/.worktrees/lanes
r=$(mktemp -d /scratch/mu/ad/kj/one.XXXX)
python3 - "$W/examples/$1" "$r/m.ferx" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
if re.search(r"^\s*maxiter\s*=", t, re.M):
    t = re.sub(r"^(\s*maxiter\s*=\s*)\d+", r"\g<1>3", t, flags=re.M)
open(sys.argv[2], "w").write(t)
EOF
cd "$r" || exit 1
FERX_KJIT=1 FERX_KJIT_DEBUG=1 timeout 300 "$B" m.ferx --data "$2" 2>&1 | grep -E "${3:-kjit}" | head -${N:-12}
cd /scratch/mu/ad/kj && rm -rf "$r"
