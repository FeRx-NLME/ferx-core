#!/usr/bin/env python3
"""Print per-seed eval -2LL and estimates for one model/arm: perseed.py MODEL ARM_DIR"""
import glob
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
model, arm = sys.argv[1], sys.argv[2]
for p in sorted(glob.glob(os.path.join(here, "runs", model, arm, "s*", "result.json"))):
    r = json.load(open(p))
    if r.get("rc") != 0:
        print(r["seed"], "FAIL")
        continue
    f = r["fit"]
    ev = r.get("eval", {}).get("minus2_log_likelihood")
    th = " ".join(f"{k}={v:.4g}" for k, v in f["theta"].items())
    om = " ".join(f"{k}={v:.4g}" for k, v in f["omega"].items())
    sg = " ".join(f"{k}={v:.4g}" for k, v in f["sigma"].items())
    print(f"s{r['seed']:02d} eval={ev:.3f} | {th} | {om} | {sg}")
