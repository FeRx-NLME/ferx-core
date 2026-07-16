#!/usr/bin/env python3
"""
Generate a single-dose oral PK dataset with **per-route absorption lag** — two
first-order pathways where each carries its OWN lag (`first_order(ka=KA, lag=L)`),
for anchoring ferx's per-route lag against NONMEM.

Structural model — an immediate-release (IR) route with no lag plus a
delayed-release (DR) route with a gastric-delay lag LAG2:

    central' = R_in(tad) - (CL/V)*central
    R_in(tad) = FR1 * KA1*exp(-KA1*tad)                      (IR, tad > 0)
              + FR2 * KA2*exp(-KA2*(tad-LAG2)) * [tad > LAG2] (DR, delayed onset)
    FR2 = 1 - FR1 ,  dose = F*amt  (F = 1)

The two pathways split the dose by FR1 / (1-FR1); the DR route only begins
absorbing after LAG2 — the per-route lag ferx adds via the `lag=` argument, and the
NONMEM F1=0 + PODO trick with a shifted TAD in $DES (per_route_lag.ctl).

Pure standard library (deterministic seed), so re-running reproduces the CSV
byte-for-byte.

Data-generating truths — identical to examples/per_route_lag_absorption.ferx
structure (IR + DR): TVCL=5, TVV=50, TVFR1=0.5, TVKA1=1.2, TVKA2=0.8, TVLAG2=3.0;
IIV on CL and V (omega = 0.09 each, ~30% CV); proportional residual SD 0.15. (The IR
route delivers from t=0, so total concentration is > 0 everywhere and a proportional
error model is well-defined — no dead-time zero-prediction.)

Output (NONMEM-format): ID,TIME,DV,AMT,EVID,CMT,MDV  (dose CMT=1, obs CMT=2),
matching nonmem_anchor/per_route_lag.ctl (DEPOT inert dose carrier with F1=0). Pass
`--ferx` to emit the ferx-keyed copy (obs CMT 1) for data/per_route_lag_oral.csv.
"""
import math
import random
import sys

# ---- design / truths -------------------------------------------------------
SEED = 7
N_SUB = 20
DOSE = 100.0  # mg, single oral dose into CMT=1 at t=0
OBS_TIMES = [0.25, 0.5, 1.0, 1.5, 2.0, 2.9, 3.1, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0]
TVCL, TVV = 5.0, 50.0
TVFR1 = 0.5                      # IR-pathway fraction (DR = 1 - FR1)
TVKA1 = 1.2                      # IR pathway absorption rate (1/h)
TVKA2 = 0.8                      # DR pathway absorption rate (1/h)
TVLAG2 = 3.0                     # DR pathway onset lag (h)
OM_CL, OM_V = 0.09, 0.09         # IIV variances on CL, V
SIG_PROP = 0.15                  # proportional residual SD
DT = 0.005                       # RK4 step (h); every OBS_TIME is an integer multiple


def first_order(tad, ka, dose):
    """First-order (Bateman) absorption input rate scaled by the dose (mg/h)."""
    if tad <= 0.0:
        return 0.0
    return dose * ka * math.exp(-ka * tad)


def r_in(tad, fr1, dose):
    """Per-route-lag input: IR route (no lag) + DR route (lag TVLAG2)."""
    return fr1 * first_order(tad, TVKA1, dose) + (1.0 - fr1) * first_order(
        tad - TVLAG2, TVKA2, dose
    )


def simulate_subject(cl, v, fr1, dose):
    """RK4-integrate central' = R_in - (CL/V)*central; return {obs_time: conc}."""
    k = cl / v
    nsteps = int(round(OBS_TIMES[-1] / DT))
    obs_steps = {int(round(ot / DT)): ot for ot in OBS_TIMES}

    def deriv(t, central):
        return r_in(t, fr1, dose) - k * central

    central = 0.0
    recorded = {}
    for step in range(nsteps + 1):
        t = step * DT
        if step in obs_steps:
            recorded[obs_steps[step]] = central / v  # concentration = A/V
        if step == nsteps:
            break
        k1 = deriv(t, central)
        k2 = deriv(t + DT / 2, central + DT / 2 * k1)
        k3 = deriv(t + DT / 2, central + DT / 2 * k2)
        k4 = deriv(t + DT, central + DT * k3)
        central += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return recorded


def fmt(x):
    return f"{x:g}"


def main():
    obs_cmt = 1 if "--ferx" in sys.argv else 2
    rng = random.Random(SEED)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV"]
    for sid in range(1, N_SUB + 1):
        cl = TVCL * math.exp(rng.gauss(0.0, math.sqrt(OM_CL)))
        v = TVV * math.exp(rng.gauss(0.0, math.sqrt(OM_V)))
        conc = simulate_subject(cl, v, TVFR1, DOSE)
        rows.append(f"{sid},0,.,{fmt(DOSE)},1,1,1")
        for ot in OBS_TIMES:
            dv = conc[ot] * (1.0 + rng.gauss(0.0, SIG_PROP))  # proportional error
            rows.append(f"{sid},{fmt(ot)},{fmt(dv)},.,0,{obs_cmt},0")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
