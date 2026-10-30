#!/usr/bin/env python3
"""Summarise a macOS `sample` output file.

1. prints the 'Sort by top of stack' self-time table (top N)
2. computes INCLUSIVE sample counts for every function name (counted once per
   stack path) from the call tree, and prints the top N by inclusive count,
   optionally filtered by a substring.
Usage: sample_summary.py FILE [N] [filter]
"""
import re, sys, collections

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
flt = sys.argv[3] if len(sys.argv) > 3 else None
lines = open(path, errors="replace").read().splitlines()

# --- self time table
start = None
for i, l in enumerate(lines):
    if l.startswith("Sort by top of stack"):
        start = i
        break
if start is not None:
    print("== top of stack (self) ==")
    k = 0
    for l in lines[start + 1:]:
        if not l.strip():
            if k > 0:
                break
            continue
        print("  " + l.strip()[:170])
        k += 1
        if k >= N:
            break

# --- inclusive from call tree
# lines look like: "    +   ! : | 123 funcname  (in binary) + 45  [0x...]"
pat = re.compile(r"^([\s+!:|]*)(\d+)\s+(.*?)\s+\(in ([^)]*)\)")
tree_start = None
for i, l in enumerate(lines):
    if l.startswith("Call graph:"):
        tree_start = i
        break
incl = collections.Counter()
stack = []  # (depth, name)
total_by_thread = 0
if tree_start is not None:
    for l in lines[tree_start + 1:]:
        if l.startswith("Total number in stack") or l.startswith("Sort by top"):
            break
        m = pat.match(l)
        if not m:
            continue
        depth = len(m.group(1))
        cnt = int(m.group(2))
        name = m.group(3)
        name = re.sub(r"::h[0-9a-f]{16}", "", name)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        # count inclusive only if the name is not already on the stack (recursion)
        if all(n != name for _, n in stack):
            incl[name] += cnt
        stack.append((depth, name))
    print("\n== inclusive (top %d%s) ==" % (N, (" filter=" + flt) if flt else ""))
    items = [(n, c) for n, c in incl.items() if (flt is None or flt in n)]
    items.sort(key=lambda x: -x[1])
    for n, c in items[:N]:
        print("  %7d  %s" % (c, n[:160]))
