#!/usr/bin/env python3
"""
Generate a single-dose oral PK dataset with Savic (2007) transit-compartment
absorption, for anchoring ferx's built-in transit() / igd() absorption against
NONMEM.

Pure standard library (math, random) — no numpy/scipy required. Deterministic
(seed below), so re-running reproduces transit_oral.csv byte-for-byte.

Structural model — identical to examples/transit_savic.ferx (1-cpt, central
tracked as CONCENTRATION, mg/L):

    depot'   = R_in(tad) - KA*depot                       # depot   (mg)
    central' = KA*depot/V - (CL/V)*central                # central (mg/L)
    R_in(tad) = dose * KTR*(KTR*tad)^N * exp(-KTR*tad) / Gamma(N+1)
    KTR = (N+1)/MTT ,  dose = F*amt  (F = 1)

Data-generating truths: TVCL=5, TVV=50, TVKA=1, TVMTT=1, TVN=3;
IIV on CL and V only (omega = 0.09 each, ~30% CV); proportional residual SD 0.15.

The oral dose feeds R_in over time (the depot bolus is NOT added) — exactly the
ferx transit() convention and the NONMEM F1=0 + PODO trick. So DV = central.

Output (NONMEM-format): ID,TIME,DV,AMT,EVID,CMT,MDV   (dose CMT=1, obs CMT=2).
"""
import math
import random

# ---- design / truths -------------------------------------------------------
SEED = 7
N_SUB = 20
DOSE = 100.0  # mg, single oral dose into CMT=1 at t=0
OBS_TIMES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
TVCL, TVV, TVKA, TVMTT, TVN = 5.0, 50.0, 1.0, 1.0, 3.0
OM_CL, OM_V = 0.09, 0.09  # IIV variances on CL, V
SIG_PROP = 0.15           # proportional residual SD
DT = 0.005                # RK4 step (h); every OBS_TIME is an integer multiple


def r_in(tad, ktr, n, lng, dose):
    """Savic transit input rate into the depot (log-domain for stability)."""
    if tad <= 0.0:
        return 0.0
    x = ktr * tad
    return math.exp(math.log(dose) + math.log(ktr) + n * math.log(x) - x - lng)


def simulate_subject(cl, v, ka, mtt, n, dose):
    """RK4-integrate the 2-state ODE; return {obs_time: central concentration}."""
    ktr = (n + 1.0) / mtt
    lng = math.lgamma(n + 1.0)  # ln Gamma(n+1)
    k = cl / v
    nsteps = int(round(OBS_TIMES[-1] / DT))
    obs_steps = {int(round(ot / DT)): ot for ot in OBS_TIMES}

    def deriv(t, depot, central):
        ri = r_in(t, ktr, n, lng, dose)
        return (ri - ka * depot, ka * depot / v - k * central)

    depot, central = 0.0, 0.0
    recorded = {}
    for step in range(nsteps + 1):
        t = step * DT
        if step in obs_steps:
            recorded[obs_steps[step]] = central
        if step == nsteps:
            break
        k1d, k1c = deriv(t, depot, central)
        k2d, k2c = deriv(t + DT / 2, depot + DT / 2 * k1d, central + DT / 2 * k1c)
        k3d, k3c = deriv(t + DT / 2, depot + DT / 2 * k2d, central + DT / 2 * k2c)
        k4d, k4c = deriv(t + DT, depot + DT * k3d, central + DT * k3c)
        depot += DT / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
        central += DT / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)
    return recorded


def fmt(x):
    return f"{x:g}"


def main():
    random.seed(SEED)
    lines = ["ID,TIME,DV,AMT,EVID,CMT,MDV"]
    for sid in range(1, N_SUB + 1):
        cl = TVCL * math.exp(random.gauss(0.0, math.sqrt(OM_CL)))
        v = TVV * math.exp(random.gauss(0.0, math.sqrt(OM_V)))
        profile = simulate_subject(cl, v, TVKA, TVMTT, TVN, DOSE)
        # dose record
        lines.append(f"{sid},0,.,{fmt(DOSE)},1,1,1")
        # observation records (CMT=2 = central)
        for ot in OBS_TIMES:
            ipred = profile[ot]
            dv = ipred * (1.0 + random.gauss(0.0, SIG_PROP))
            if dv <= 0.0:
                dv = ipred * 0.01
            lines.append(f"{sid},{fmt(ot)},{dv:.4f},.,0,2,0")
    with open("transit_oral.csv", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    n_obs = N_SUB * len(OBS_TIMES)
    print(f"wrote transit_oral.csv: {N_SUB} subjects, {n_obs} observations")


if __name__ == "__main__":
    main()
