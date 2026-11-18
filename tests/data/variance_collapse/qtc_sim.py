"""Generate qtc_sim.csv, the #3005 collapsed-variance fixture.

A synthetic C-QTc design shaped like the verapamil + placebo arms of the PhysioNet
ECGRDVQ study used in issue #3005 (none of that data is used here): 44 subjects, 16
time points, odd IDs active with a one-compartment oral concentration profile, even
IDs placebo (CONC = 0), and a per-subject baseline covariate BLC. DV is simulated
from the linear mixed model

    DV = INT + TEFF[TIME] + ACT*ACTIVE + (SLOPE + ETA_SLOPE)*CONC + BL*BLC + ETA_INT + EPS

at the issue's NONMEM estimates (omega^2_INT = 35, omega^2_SLOPE = 652, sigma = 5.85).
For this draw (seed 2) the ML optimum is omega^2_SLOPE = 87.41 at OFV 3307.9166
(lme4 `REML = FALSE`: -2 log L = 4601.782, i.e. 3307.916 + 704 ln 2 pi). From the
start `omega ETA_SLOPE ~ 600` the default FOCEI L-BFGS collapses it to ~3e-5 at
3308.6881 and reports converged.

    python3 qtc_sim.py > qtc_sim.csv
"""

import math
import random

rng = random.Random(2)
times = [-0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 12, 14, 24]
teff = [0, -7.67, -3.71, -5.59, -1.19, -2.14, -4.58, -7.86, -6.10, -8.04, -6.40,
        -7.54, -8.32, -11.47, -10.36, -6.00]
INT, ACT, SLOPE, BL = -1.10, 2.20, 13.46, -0.222
w_int, w_s, sd = 35.0, 652.0, 5.85
print("ID,TIME,TIDX,DV,MDV,EVID,CONC,ACTIVE,BLC")
for i in range(1, 45):
    active = 1 if i % 2 == 1 else 0
    blc = rng.gauss(0, 6)
    ka = rng.uniform(0.8, 3.0)
    ke = rng.uniform(0.12, 0.25)
    scale = rng.uniform(0.1, 0.25)
    e_int = rng.gauss(0, math.sqrt(w_int))
    e_s = rng.gauss(0, math.sqrt(w_s))
    for k, t in enumerate(times):
        c = 0.0
        if active and t > 0:
            c = scale * ka / (ka - ke) * (math.exp(-ke * t) - math.exp(-ka * t)) * 1.6
        c = round(c, 5)
        dv = (INT + teff[k] + ACT * active + (SLOPE + e_s) * c + BL * blc + e_int
              + rng.gauss(0, sd))
        print(f"{i},{t},{k + 1},{round(dv, 4)},0,0,{c},{active},{round(blc, 4)}")
