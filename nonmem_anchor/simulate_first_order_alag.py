#!/usr/bin/env python3
"""Deterministic simulator for a 1-cpt first-order absorption model with an
estimated lag time (ALAG1) -- validates PR2 of the #486 analytic-gradient-
completion plan (built-in `first_order(ka)` input-rate forcing + lagtime on
ferx's event-driven ODE sensitivity walk).

ferx model: `d/dt(central) = first_order(ka=KA) - CL/V*central`, ALAG1 = TVLAG
(a single `central` state fed directly by R_in(tad) = dose*ka*exp(-ka*tad),
tad = t - (dose.time + lag)). This is mathematically identical to the classic
NONMEM depot->central first-order absorption (ADVAN2 TRANS2) with ALAG1: the
"input into central" from an exponentially-decaying depot IS exactly
R_in(tad) = dose*ka*exp(-ka*tad) -- same trajectory, same likelihood.

Emits two CSVs from the SAME underlying simulation:
  - first_order_alag_ferx.csv  -- CMT=1 for both dose and obs (ferx's single
    `central` state; the model routes the dose into first_order() directly).
  - first_order_alag_nm.csv    -- CMT=1 (depot) for the dose, CMT=2 (central)
    for observations (NONMEM ADVAN2 convention).

Pure stdlib, deterministic (seed=486), reproducible by re-running.
"""
import csv
import math
import random

SEED = 486
N_SUBJ = 30
DOSE_AMT = 100.0
OBS_TIMES = [0.6, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0]  # all > TVLAG=0.5

TVCL, TVV, TVKA, TVLAG = 5.0, 50.0, 1.0, 0.5
OMEGA_CL, OMEGA_KA = 0.09, 0.04
SIGMA_PROP = 0.10  # proportional residual SD


def conc(t, lag, cl, v, ka):
    tad = t - lag
    if tad <= 0.0:
        return 0.0
    k = cl / v
    if abs(ka - k) < 1e-9:
        # degenerate ka==k limit (not hit at these truths, kept for safety)
        return DOSE_AMT * ka * tad * math.exp(-ka * tad) / v
    a2 = DOSE_AMT * ka / (ka - k) * (math.exp(-k * tad) - math.exp(-ka * tad))
    return a2 / v


def main():
    rng = random.Random(SEED)
    ferx_rows = [("ID", "TIME", "DV", "AMT", "EVID", "CMT", "MDV")]
    nm_rows = [("ID", "TIME", "DV", "AMT", "EVID", "CMT", "MDV")]
    for sid in range(1, N_SUBJ + 1):
        eta_cl = rng.gauss(0.0, math.sqrt(OMEGA_CL))
        eta_ka = rng.gauss(0.0, math.sqrt(OMEGA_KA))
        cl = TVCL * math.exp(eta_cl)
        ka = TVKA * math.exp(eta_ka)
        v = TVV
        lag = TVLAG

        ferx_rows.append((sid, 0, ".", DOSE_AMT, 1, 1, 1))
        nm_rows.append((sid, 0, ".", DOSE_AMT, 1, 1, 1))
        for t in OBS_TIMES:
            c = conc(t, lag, cl, v, ka)
            eps = rng.gauss(0.0, SIGMA_PROP)
            dv = max(c * (1.0 + eps), 1e-6)
            ferx_rows.append((sid, t, round(dv, 4), ".", 0, 1, 0))
            nm_rows.append((sid, t, round(dv, 4), ".", 0, 2, 0))

    with open("first_order_alag_ferx.csv", "w", newline="") as f:
        csv.writer(f).writerows(ferx_rows)
    with open("first_order_alag_nm.csv", "w", newline="") as f:
        csv.writer(f).writerows(nm_rows)


if __name__ == "__main__":
    main()
