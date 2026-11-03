#!/usr/bin/env python3
"""Median wall time per model/arm/K from a serial (jobs = 1) run directory:
timing.py RUNS_DIR"""
import json
import os
import statistics as st
import sys

groups = {}
for dp, _, fs in os.walk(sys.argv[1]):
    if "result.json" in fs:
        r = json.load(open(os.path.join(dp, "result.json")))
        if r["rc"] == 0:
            w = r["fit"].get("wall") or r["wall_cli"]
            groups.setdefault((r["model"], r["arm"], r["k"]), []).append(w)
print("| model | arm | K | n | wall s (median) |")
print("|---|---|---|---|---|")
for key in sorted(groups):
    print(f"| {key[0]} | {key[1]} | {key[2] or '—'} | {len(groups[key])} | {st.median(groups[key]):.2f} |")
