"""Sparse TDM dataset for tests/dw_sequential_occasions.rs (#2829).

24 subjects, six 100 mg oral doses 24 h apart, one trough 23 h after each dose:
288 rows, 144 doses and 144 observations. `OCC` numbers the dosing intervals 1-6,
so every occasion holds exactly one observation.

The residuals are proportional, 20%, and AR(1) with rho = 0.8 within a subject, so
a one-compartment fit leaves positively autocorrelated IWRES: the Durbin-Watson
warning must fire. Split at the occasions there is no lag pair at all.

Run from this directory: `python3 gen.py` rewrites sparse_tdm.csv (seed 2829).
"""
import math
import random

random.seed(2829)
rows = ["ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,OCC"]
for i in range(1, 25):
    cl = 0.2 * math.exp(random.gauss(0, 0.3))
    v = 10.0
    ka = 1.5
    k = cl / v
    e = random.gauss(0, 1)
    for d in range(6):
        td = 24.0 * d
        rows.append(f"{i},{td:g},.,1,100.0,1,0,1,{d + 1}")
        t = td + 23.0
        c = sum(
            100 * ka / (v * (ka - k)) * (math.exp(-k * (t - 24 * j)) - math.exp(-ka * (t - 24 * j)))
            for j in range(d + 1)
        )
        e = 0.8 * e + math.sqrt(1 - 0.64) * random.gauss(0, 1)
        rows.append(f"{i},{t:g},{c * (1 + 0.2 * e):.5f},0,.,1,0,0,{d + 1}")
open("sparse_tdm.csv", "w").write("\n".join(rows) + "\n")
