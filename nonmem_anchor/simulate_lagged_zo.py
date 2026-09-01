"""Simulate a 1-cpt model with a single LAGGED ZERO-ORDER absorption route.

ferx:   d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central
NONMEM: ADVAN13 $DES elimination only, with D1=DUR and ALAG1=LAG (RATE=-2 dosing).

Two doses so the second zero-order window OPENS while drug from the first is still
present -- the non-degeneracy requirement (a single dose cannot test a window's
incoming side).
"""
import math, random

CL_T, V_T, DUR_T, LAG_T = 5.0, 50.0, 2.0, 1.5
OM_CL, OM_V, PROP_SD = 0.09, 0.04, 0.15
DOSES = [(0.0, 100.0), (12.0, 100.0)]
# No obs before LAG: IPRED there is exactly 0 and a proportional error on 0 is
# degenerate (Y=0, no information, NONMEM F=0). t=12.4 IS kept -- it is inside the
# SECOND dose's pre-arrival window but with drug from the first still present, which
# is exactly the incoming-side case a single-dose fixture cannot reach.
OBS = [2.3, 3.1, 4.2, 8.0, 11.5, 12.4, 13.8, 15.0, 18.0, 24.0]
N_SUBJ = 12

def amount(t, cl, v):
    k = cl / v
    tot = 0.0
    for (td, amt) in DOSES:
        r = amt / DUR_T
        s = t - td - LAG_T          # time since this dose's window opened
        if s <= 0:
            continue
        if s <= DUR_T:
            tot += (r / k) * (1.0 - math.exp(-k * s))
        else:
            tot += (r / k) * (1.0 - math.exp(-k * DUR_T)) * math.exp(-k * (s - DUR_T))
    return tot

random.seed(20260901)
rows_nm, rows_ferx = [], []
for i in range(1, N_SUBJ + 1):
    cl = CL_T * math.exp(random.gauss(0.0, math.sqrt(OM_CL)))
    v = V_T * math.exp(random.gauss(0.0, math.sqrt(OM_V)))
    for (td, amt) in DOSES:
        # NONMEM: RATE=-2 => duration modeled by D1. ferx: plain AMT, zero_order() reads it.
        rows_nm.append((i, td, 0.0, amt, -2, 1, 1, 1))
        rows_ferx.append((i, td, 0.0, amt, 1, 1, 1))
    for t in OBS:
        ipred = amount(t, cl, v) / v
        dv = ipred * (1.0 + random.gauss(0.0, PROP_SD))
        rows_nm.append((i, t, dv, 0.0, 0, 0, 1, 0))
        rows_ferx.append((i, t, dv, 0.0, 0, 1, 0))

# NONMEM requires time-ordered records within a subject, doses first at a tie.
rows_nm.sort(key=lambda r: (r[0], r[1], 0 if r[5] == 1 else 1))
rows_ferx.sort(key=lambda r: (r[0], r[1], 0 if r[4] == 1 else 1))

with open("lagged_zo_nm.csv", "w") as f:
    f.write("ID,TIME,DV,AMT,RATE,EVID,CMT,MDV\n")
    for r in rows_nm:
        f.write("%d,%.4f,%.6f,%.1f,%d,%d,%d,%d\n" % r)

with open("lagged_zo.csv", "w") as f:
    f.write("ID,TIME,DV,AMT,EVID,CMT,MDV\n")
    for r in rows_ferx:
        f.write("%d,%.4f,%.6f,%.1f,%d,%d,%d\n" % r)

print("wrote lagged_zo_nm.csv and lagged_zo.csv")
print("check: subject-1-like curve at truth, obs 2.3 / 13.8 =",
      round(amount(2.3, CL_T, V_T) / V_T, 6), round(amount(13.8, CL_T, V_T) / V_T, 6))
