# Generates pktte_ebe.csv (#1885). Run from nonmem_anchor/: python3 pktte_ebe_gen.py
#
# PK: 1-cpt oral, CL = 1*exp(eta), V = 10, KA = 1; 100 into the depot at 0/12/24, so the
# later doses land on residual drug. Hazard h = H0*exp(BETA*C), H0 = 0.01, BETA = 0.3.
# PK observations are simulated at a per-subject eta with 10% proportional error (fixed
# seed). Event rows are chosen by hand, exact and right-censored mixed, several after a
# later dose. EVID=2 rows on 2..48 step 2 carry the predict grid ($TABLE CHZ/HAZ there).
import math
import random

random.seed(1885)
V, KA = 10.0, 1.0
DOSES = [0.0, 12.0, 24.0]


def conc(t, cl):
    k = cl / V
    c = 0.0
    for td in DOSES:
        if t >= td:
            dt = t - td
            c += 100 * KA / (V * (KA - k)) * (math.exp(-k * dt) - math.exp(-KA * dt))
    return c


ETA_TRUE = [0.35, -0.30, 0.10, -0.45, 0.25, 0.0, -0.15, 0.50]
EVENTS = [(5.2, 1), (9.7, 0), (14.3, 1), (18.0, 1), (25.5, 1), (30.0, 0), (36.0, 1), (40.0, 0)]
PK_TIMES = [1, 3, 6, 11, 14, 20, 26, 30]
GRID = [float(t) for t in range(2, 49, 2)]

rows = ["ID,TIME,DV,EVID,AMT,CMT,RATE,MDV"]
for i, (eta, (te, dv)) in enumerate(zip(ETA_TRUE, EVENTS), start=1):
    cl = math.exp(eta)
    recs = [(td, 1, f"{i},{td:g},.,1,100,1,0,1") for td in DOSES]
    for t in PK_TIMES:
        y = conc(t, cl) * (1 + 0.1 * random.gauss(0, 1))
        recs.append((t, 2, f"{i},{t:g},{y:.4f},0,.,2,0,0"))
    recs.append((te, 3, f"{i},{te:g},{dv},0,0,3,0,0"))
    for t in GRID:
        recs.append((t, 4, f"{i},{t:g},.,2,.,.,0,1"))
    recs.sort(key=lambda r: (r[0], r[1]))
    rows += [r[2] for r in recs]

with open("pktte_ebe.csv", "w") as f:
    f.write("\n".join(rows) + "\n")
