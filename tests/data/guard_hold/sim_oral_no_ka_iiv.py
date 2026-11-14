"""Generate sim_oral_no_ka_iiv.csv, the #2797 guard-hold fixture.

40 subjects, one 100-unit oral dose, one-compartment PK with between-subject
variability on CL (30% log-normal) and V (20%) and **none** on KA (fixed at
1.0), 10% proportional residual error. A model that estimates an ETA on KA (or
an IOV kappa on KA) has nothing to estimate there, so the variance collapses
onto ferx's internal lower guard (omega = exp(-12) = 6.1442123533282098e-6),
which is what tests/guard_hold.rs and the NONMEM anchor in
nonmem_anchor/guard_hold_* need.

OCC splits each subject's record at 24 h (1 before, 2 from 24 h on) for the
IOV variant; the BSV-only models ignore it.

Run from this directory: python3 sim_oral_no_ka_iiv.py
"""

import math
import random

random.seed(2797)
times = [0.5, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120]
rows = ["ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,OCC"]
for i in range(1, 41):
    cl = 0.13 * math.exp(random.gauss(0, 0.3))
    v = 8 * math.exp(random.gauss(0, 0.2))
    ka = 1.0
    rows.append(f"{i},0,.,1,100,1,0,1,1")
    k = cl / v
    for t in times:
        c = 100 * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
        occ = 1 if t < 24 else 2
        rows.append(f"{i},{t},{c * (1 + random.gauss(0, 0.1)):.5f},0,.,1,0,0,{occ}")
with open("sim_oral_no_ka_iiv.csv", "w") as f:
    f.write("\n".join(rows) + "\n")
