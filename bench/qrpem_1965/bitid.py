#!/usr/bin/env python3
"""Compare estimates of the same model/arm/K/seed between two run directories,
exactly (bit-for-bit on the YAML's printed digits): bitid.py OLD_RUNS NEW_RUNS"""
import json
import os
import sys

old_root, new_root = sys.argv[1], sys.argv[2]
for dp, _, fs in sorted(os.walk(new_root)):
    if "result.json" not in fs:
        continue
    rel = os.path.relpath(dp, new_root)
    old_p = os.path.join(old_root, rel, "result.json")
    if not os.path.exists(old_p):
        print(f"{rel}: no old run")
        continue
    a, b = json.load(open(old_p))["fit"], json.load(open(os.path.join(dp, "result.json")))["fit"]
    same = all(a[k] == b[k] for k in ("theta", "omega", "sigma", "ofv"))
    print(f"{rel}: {'IDENTICAL' if same else 'DIFFERENT'}  ofv {a['ofv']} -> {b['ofv']}")
