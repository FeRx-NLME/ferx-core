"""Write the #2441 anchor datasets: variance_floor_prop_k1.csv and variance_floor_prop_k1e6.csv.

One-compartment IV bolus, CL = 1 * exp(eta_CL), V = 10 * exp(eta_V), omega^2 0.09 / 0.04,
proportional error sd 0.1, dose 0.001 at t = 0, 12 subjects with 9 samples each out to 48 h.
The concentration falls from about 1e-4 to about 1e-7, so under proportional error most
samples have residual variances (f * 0.1)^2 below the old absolute floor of 1e-12.

The k1e6 file is the same data in units a million times larger: AMT and DV multiplied by 1e6.
That is a change of units only, so every -2LL shifts by exactly 2 * n * ln(1e6). There the
smallest concentration is still above 1e-5, so no variance is below 1e-12: the old floor
applied at k = 1 and never at k = 1e6.

Usage: python3 make_variance_floor_prop.py   (writes next to this script)
"""
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TIMES = [0.5, 1, 2, 4, 8, 12, 24, 36, 48]
CL, V, OM_CL, OM_V, SIGMA, DOSE = 1.0, 10.0, 0.09, 0.04, 0.1, 0.001


def rows(seed=2441, n=12):
    rng = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        cl = CL * math.exp(rng.gauss(0.0, math.sqrt(OM_CL)))
        v = V * math.exp(rng.gauss(0.0, math.sqrt(OM_V)))
        out.append((i, 0.0, 0.0, 1, DOSE, 1, 1))
        for t in TIMES:
            f = DOSE / v * math.exp(-cl / v * t)
            dv = f * (1.0 + rng.gauss(0.0, SIGMA))
            out.append((i, float(t), dv, 0, 0.0, 1, 0))
    return out


def write(name, k):
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        fh.write("ID,TIME,DV,EVID,AMT,CMT,MDV\n")
        for i, t, dv, evid, amt, cmt, mdv in rows():
            fh.write(f"{i},{t!r},{dv * k!r},{evid},{amt * k!r},{cmt},{mdv}\n")


write("variance_floor_prop_k1.csv", 1.0)
write("variance_floor_prop_k1e6.csv", 1e6)
