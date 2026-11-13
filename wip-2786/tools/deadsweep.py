"""deadsweep.py: repeatedly `cargo check --lib --tests` (every feature) and delete every
item rustc reports as never used/constructed (functions, methods, structs, consts, enums,
traits, statics, type aliases), by its reported line, until none is left. Prints a log."""
import re
import subprocess
import sys
sys.path.insert(0, '/scratch/mu/ad')
from rmitem import item_end

ROOT = '/root/package/ferx-core/.worktrees/lanes/'
CMD = ('CARGO_PROFILE_DEV_DEBUG=0 cargo check --lib --profile test --features survival,markov,nn -j2 '
       '--target-dir /scratch/tgt-lanes-check --message-format=short 2>&1')
PAT = re.compile(r'^(src/[^:]+):(\d+):(\d+): warning: (?:function|method|associated function|struct|constant|enum|trait|static|type alias|associated constant|macro definition) `([^`]+)` is never (?:used|constructed)')
# Several names in one warning ("methods `a` and `b` are never used") are reported at the first.
PAT2 = re.compile(r'^(src/[^:]+):(\d+):(\d+): warning: (?:functions|methods|associated functions|associated items|constants|structs)( `[^`]+`(?:,| and))+')


def remove_at(path, line):
    s = open(ROOT + path).read()
    lines = s.split('\n')
    li = line - 1
    top = li
    while top > 0 and (lines[top - 1].lstrip().startswith('///') or lines[top - 1].lstrip().startswith('#[')):
        top -= 1
    off_top = sum(len(l) + 1 for l in lines[:top])
    off_sig = sum(len(l) + 1 for l in lines[:li])
    end = item_end(s, off_sig)
    if s[end:end + 1] == '\n':
        end += 1
    if s[end:end + 1] == '\n' and s[off_top - 2:off_top] == '\n\n':
        end += 1
    open(ROOT + path, 'w').write(s[:off_top] + s[end:])
    return s[off_sig:s.index('\n', off_sig)].strip()


subprocess.run("pkill -x rust-analyzer", shell=True)
for rnd in range(30):
    out = subprocess.run(CMD, shell=True, cwd=ROOT, capture_output=True, text=True).stdout
    errs = [l for l in out.split('\n') if ': error' in l]
    if errs:
        print('ERRORS:', *errs[:10], sep='\n')
        sys.exit(1)
    hits = []
    for l in out.split('\n'):
        m = PAT.match(l)
        if m:
            hits.append((m.group(1), int(m.group(2)), m.group(4)))
    if not hits:
        print('clean after', rnd, 'rounds')
        multi = [l for l in out.split('\n') if PAT2.match(l)]
        for l in multi:
            print('MULTI:', l)
        break
    # Bottom-up per file so line numbers stay valid.
    seen = set()
    for path, line, name in sorted(hits, key=lambda h: (h[0], -h[1])):
        if (path, line) in seen:
            continue
        seen.add((path, line))
        sig = remove_at(path, line)
        print(f'round {rnd}: {path}:{line} {name}: {sig[:90]}')
