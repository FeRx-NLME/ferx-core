#!/usr/bin/env python3
"""Per-subject EBE_OFV difference between two runs' sdtabs: cmp_subjects.py A.csv B.csv"""
import csv
import sys


def per_subject(path):
    out = {}
    for row in csv.DictReader(open(path)):
        out[row["ID"]] = float(row["EBE_OFV"])
    return out


a, b = per_subject(sys.argv[1]), per_subject(sys.argv[2])
diffs = sorted(((a[k] - b[k], k) for k in a if k in b), key=lambda t: -abs(t[0]))
print(f"total diff (A-B) = {sum(d for d, _ in diffs):.3f} over {len(diffs)} subjects")
for d, k in diffs[:10]:
    print(f"  ID {k}: A={a[k]:.3f} B={b[k]:.3f} diff={d:+.3f}")
