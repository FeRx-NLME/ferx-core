#!/usr/bin/env python3
"""Simulate IV-bolus data for the #1070 TAD x estimated-lagtime anchor.

Truth model (the convention BOTH engines must agree on):
    ALAG1 = TVLAG * exp(eta_lag)            dose arrives at t_dose + ALAG1
    TAD(t) = t - (most recent arrival <= t)
    dA/dt = -(CL/V) * A * (1 + THETA_TAD * TAD)
Pure-stdlib RK4, fixed seed. No ferx code involved, so the dataset does not
inherit either engine's convention by construction.

Anchor-design constraint (measured, #1070 probe B): ferx NaN-poisons the whole
trajectory if any record precedes the first lagged arrival. TVLAG=0.5 with
omega_lag=0.02 keeps lag < 0.8 at 4 SD, so the first observation at t=2 is safe
for every simulated subject.
"""
import random
import math

TVCL, TVV, THETA_TAD, TVLAG = 1.0, 20.0, 0.3, 0.5
OM_CL, OM_LAG, SIG_PROP = 0.09, 0.02, 0.10
N_SUB = 12
OBS_A = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 24.0]
OBS_B = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 11.5, 14.0, 15.0, 16.0, 18.0, 20.0, 24.0]
DOSE_T_A = [0.0]
DOSE_T_B = [0.0, 12.0]


def simulate(dose_times, obs_times, cl, v, lag):
    arrivals = sorted(t + lag for t in dose_times)
    events = sorted(set(arrivals) | set(obs_times))
    t = arrivals[0]
    a_amt = 100.0  # first dose lands
    out = {}
    remaining = list(arrivals[1:])

    def tad(tt):
        past = [a for a in arrivals if a <= tt + 1e-12]
        return tt - max(past)

    def deriv(tt, a):
        return -(cl / v) * a * (1.0 + THETA_TAD * tad(tt))

    for tgt in events:
        if tgt <= t + 1e-12:
            if tgt in obs_times:
                out[tgt] = a_amt
            continue
        while t < tgt - 1e-12:
            h = min(0.005, tgt - t)
            k1 = deriv(t, a_amt)
            k2 = deriv(t + h / 2, a_amt + h * k1 / 2)
            k3 = deriv(t + h / 2, a_amt + h * k2 / 2)
            k4 = deriv(t + h, a_amt + h * k3)
            a_amt += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += h
        if remaining and abs(tgt - remaining[0]) < 1e-9:
            a_amt += 100.0
            remaining.pop(0)
        if tgt in obs_times:
            out[tgt] = a_amt
    return out


def build(path, dose_times, obs_times, seed):
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,MDV,EVID,AMT,CMT,WT"]
    for i in range(1, N_SUB + 1):
        cl = TVCL * math.exp(rng.gauss(0, math.sqrt(OM_CL)))
        lag = TVLAG * math.exp(rng.gauss(0, math.sqrt(OM_LAG)))
        conc = simulate(dose_times, obs_times, cl, TVV, lag)
        recs = []
        for dt in dose_times:
            recs.append((dt, 0, ".", 1, 100.0, 1))
        for ot in obs_times:
            ipred = conc[ot] / TVV
            dv = ipred * (1.0 + rng.gauss(0, SIG_PROP))
            recs.append((ot, 1, f"{dv:.6f}", 0, 0.0, 1))
        recs.sort(key=lambda r: (r[0], r[1]))  # dose (0) sorts before obs (1)
        for (n, (tt, isobs, dv, _mdv, amt, cmt)) in enumerate(recs):
            wt = 70.0 + 2.0 * n  # varies within subject -> reader marks WT time-varying
            if isobs:
                rows.append(f"{i},{tt},{dv},0,0,0,{cmt},{wt}")
            else:
                rows.append(f"{i},{tt},.,1,1,{amt},{cmt},{wt}")
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {path}: {len(rows) - 1} rows")


build("runA/tad_lag_A.csv", DOSE_T_A, OBS_A, 20260828)
build("runB/tad_lag_B.csv", DOSE_T_B, OBS_B, 20260829)
