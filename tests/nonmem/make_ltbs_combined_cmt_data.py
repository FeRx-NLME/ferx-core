"""Seeded two-endpoint fixture for the per-CMT LTBS combined-error anchor (#2729).

The issue's motivating shape: plasma concentration (CMT=2) and cumulative urine
amount (CMT=3), each with its own PROP and ADD. One-compartment oral, a fraction
FE of the clearance renal. Each endpoint is simulated under
log(y) = log(F) + W*eps, W = sqrt(PROP^2 + ADD^2/F^2). Both endpoints are
sampled at t = 0, where F = 0 and the prediction sits on the LTBS floor; those
rows get a fixed log(0.01).

Writes ltbs_combined_cmt.csv (DV on the natural scale).
"""
import math
import os
import random

random.seed(27292)
CL, V, KA, FE = 2.0, 10.0, 1.5, 0.4
SIG = {2: (0.1, 0.05), 3: (0.2, 0.5)}
OM_CL, OM_V = 0.09, 0.04
DOSE = 100.0
TIMES = [0.0, 0.5, 1, 2, 4, 8, 12, 24]
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for i in range(1, 17):
    cl = CL * math.exp(random.gauss(0, math.sqrt(OM_CL)))
    v = V * math.exp(random.gauss(0, math.sqrt(OM_V)))
    k = cl / v
    c = DOSE * KA / (KA - k)
    rows.append((i, 0.0, None, 1, DOSE, 1))
    for t in TIMES:
        conc = c / v * (math.exp(-k * t) - math.exp(-KA * t))
        auc_amt = c * ((1 - math.exp(-k * t)) / k - (1 - math.exp(-KA * t)) / KA)
        urine = FE * k * auc_amt
        for cmt, f in ((2, conc), (3, urine)):
            prop, add = SIG[cmt]
            if f <= 0.0:
                dv = 0.01
            else:
                w = math.sqrt(prop**2 + add**2 / f**2)
                dv = math.exp(math.log(f) + w * random.gauss(0, 1))
            rows.append((i, t, dv, 0, None, cmt))

with open(os.path.join(here, "ltbs_combined_cmt.csv"), "w") as fh:
    fh.write("ID,TIME,DV,LNDV,EVID,AMT,CMT,MDV\n")
    for i, t, dv, evid, amt, cmt in rows:
        # LNDV is NONMEM's DV: the log of the printed natural value, so both
        # engines read the same number.
        d = "." if dv is None else "%.10g" % dv
        ld = "." if dv is None else "%.17g" % math.log(float(d))
        a = "." if amt is None else "%g" % amt
        fh.write("%d,%g,%s,%s,%d,%s,%d,%d\n" % (i, t, d, ld, evid, a, cmt, 1 if evid else 0))
