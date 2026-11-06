#!/usr/bin/env python3
"""Dataset of the `iov_kappa_two_mode` anchor (#2730). Standard library only.

One-compartment oral, 100 mg at the start of each of two occasions, samples at 8, 12 and 24 h
after each dose. Those are all past 1/k, where C(t; ka) rises and then falls in ka, so one
concentration has a solution on each side of the peak: the objective in an occasion's kappa on
KA has two modes. Subject 1's second occasion absorbs at KA * exp(-2.3). Its deep mode is
there, and a descent from kappa = 0 falls to a shallow one at kappa > 0.

Writes two files. `iov_kappa_two_mode.csv` opens occasion 2 with EVID=4 at TIME=72 (a reset
subject, on which the multi-start scans every coordinate). `iov_kappa_two_mode_noreset.csv`
is the same records with an ordinary dose (EVID=1).
"""
import math
import random

CL, V, KA = 1.0, 10.0, 1.0
OM_CL, OM_V, OM_K, SD = 0.09, 0.09, 1.0, 0.3
DOSE, SAMPLES, OCC2 = 100.0, (8.0, 12.0, 24.0), 72.0
N_SUBJECTS = 6


def conc(t, cl, v, ka):
    k = cl / v
    return DOSE * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))


def rows(evid2):
    rng = random.Random(2730)
    out = []
    for sid in range(1, N_SUBJECTS + 1):
        cl = CL * math.exp(rng.gauss(0, math.sqrt(OM_CL)))
        v = V * math.exp(rng.gauss(0, math.sqrt(OM_V)))
        kappas = [rng.gauss(0, 0.3), rng.gauss(0, 0.3)]
        if sid == 1:
            cl, v, kappas = CL, V, [0.1, -2.3]
        for occ, start, evid in ((1, 0.0, 1), (2, OCC2, evid2)):
            out.append((sid, start, ".", evid, DOSE, 1, occ))
            ka = KA * math.exp(kappas[occ - 1])
            for t in SAMPLES:
                dv = conc(t, cl, v, ka) + rng.gauss(0, SD if sid != 1 else 0.02)
                out.append((sid, start + t, f"{dv:.4f}", 0, ".", 2, occ))
    return out


for name, evid2 in (("iov_kappa_two_mode.csv", 4), ("iov_kappa_two_mode_noreset.csv", 1)):
    with open(name, "w") as f:
        f.write("ID,TIME,DV,EVID,AMT,CMT,OCC\n")
        for r in rows(evid2):
            f.write(",".join(str(c) for c in r) + "\n")
