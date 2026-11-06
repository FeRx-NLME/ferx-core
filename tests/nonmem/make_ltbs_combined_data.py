"""Seeded fixture for the LTBS combined-error NONMEM anchor (#2729).

One-compartment oral model, 100 mg at t = 0, simulated under
log(y) = log(F) + W*eps, W = sqrt(PROP^2 + ADD^2/F^2). The sampling runs from
~9 mg/L down to ~0.06 mg/L, so ADD/F runs from ~0.005 to ~0.8 against
PROP = 0.1 and both terms are live. Each subject also has an observation at
t = 0, where F = 0 and the prediction sits on the LTBS floor (1e-12): its DV is
a fixed log(0.01), since a draw with W ~ 5e10 has no meaning.

Writes ltbs_combined.csv (DV on the log scale; NONMEM and `DV ~ log_combined`)
and ltbs_combined_natural.csv (DV = exp of that; `log(DV) ~ combined`).
"""
import math
import os
import random

random.seed(2729)
CL, V, KA, PROP, ADD = 2.0, 10.0, 1.5, 0.1, 0.05
OM_CL, OM_V = 0.09, 0.04
TIMES = [0.0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 20, 24]
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for i in range(1, 17):
    cl = CL * math.exp(random.gauss(0, math.sqrt(OM_CL)))
    v = V * math.exp(random.gauss(0, math.sqrt(OM_V)))
    k = cl / v
    rows.append((i, 0.0, None, 1, 100.0, 1))
    for t in TIMES:
        f = 100.0 * KA / (v * (KA - k)) * (math.exp(-k * t) - math.exp(-KA * t))
        if f <= 0.0:
            ldv = math.log(0.01)
        else:
            w = math.sqrt(PROP**2 + ADD**2 / f**2)
            ldv = math.log(f) + w * random.gauss(0, 1)
        rows.append((i, t, ldv, 0, None, 1))


def write(name, tf, fmt):
    with open(os.path.join(here, name), "w") as fh:
        fh.write("ID,TIME,DV,EVID,AMT,CMT,MDV\n")
        for i, t, dv, evid, amt, cmt in rows:
            d = "." if dv is None else fmt % tf(dv)
            a = "." if amt is None else "%g" % amt
            fh.write("%d,%g,%s,%d,%s,%d,%d\n" % (i, t, d, evid, a, cmt, 1 if evid else 0))


# The natural file is the log file's printed value exponentiated, at full precision,
# so `log(DV)` recovers what NONMEM reads to the last bit or two.
write("ltbs_combined.csv", lambda x: x, "%.10g")
write("ltbs_combined_natural.csv", lambda x: math.exp(float("%.10g" % x)), "%.17g")
