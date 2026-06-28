#!/usr/bin/env python3
"""
Generate a single-dose oral PK dataset with **mixed** (zero-order + first-order)
absorption, for anchoring ferx's mixed-pathway model
(FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR), issue FeRx-NLME/ferx-core#505)
against NONMEM.

Pure standard library (math, random) -- no numpy/scipy. Deterministic (seed
below), so re-running reproduces mixed_oral.csv byte-for-byte.

Structural model -- identical to examples/mixed_absorption.ferx
(1-cpt, central tracked as AMOUNT mg; concentration = central/V):

    central' = R_in(tad) - (CL/V)*central
    R_in(tad) = FZO1 * KA*exp(-KA*tad) * dose   (first-order pathway)
              + FZO  * dose/DUR  for 0 < tad <= DUR, else 0   (zero-order pathway)
    FZO1 = 1 - FZO ,  dose = F*amt  (F = 1)

The two pathways split the dose by FZO / (1-FZO): a zero-order (constant-rate)
input over DUR, and a first-order input at rate KA. The whole dose is delivered
through R_in over time (the bolus is NOT added) -- exactly the ferx
first_order()/zero_order() convention and the NONMEM F1=0 + PODO trick.

Data-generating truths -- identical to examples/mixed_absorption.ferx:
TVCL=5, TVV=50, TVFZO=0.4, TVKA=1.0, TVDUR=3.0; IIV on CL and V only
(omega = 0.09 each, ~30% CV); proportional residual SD 0.15.

Output (NONMEM-format): ID,TIME,DV,AMT,EVID,CMT,MDV   (dose CMT=1, obs CMT=2),
matching nonmem_anchor/mixed_zero_first.ctl (DEPOT inert dose carrier with F1=0;
zero+first sum feeds CENTRAL = CMT 2). Pass `--ferx` to emit the ferx-keyed copy
(obs CMT 1, the single-state ferx model) for data/mixed_oral.csv.
"""
import math
import random
import sys

# ---- design / truths -------------------------------------------------------
SEED = 7
N_SUB = 20
DOSE = 100.0  # mg, single oral dose into CMT=1 at t=0
OBS_TIMES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
TVCL, TVV = 5.0, 50.0
TVFZO = 0.4                      # zero-order-pathway fraction (first-order = 1 - FZO)
TVKA = 1.0                       # first-order absorption rate (1/h)
TVDUR = 3.0                      # zero-order input duration (h)
OM_CL, OM_V = 0.09, 0.09         # IIV variances on CL, V
SIG_PROP = 0.15                  # proportional residual SD
DT = 0.005                       # RK4 step (h); every OBS_TIME and DUR is a multiple


def first_order(tad, ka, dose):
    """First-order (Bateman) absorption input rate scaled by the dose (mg/h)."""
    if tad <= 0.0:
        return 0.0
    return dose * ka * math.exp(-ka * tad)


def zero_order(tad, dur, dose):
    """Zero-order (constant-rate) absorption input rate over (0, dur] (mg/h)."""
    if tad <= 0.0 or tad > dur:
        return 0.0
    return dose / dur


def r_in(tad, fzo, dose):
    """Mixed input rate: FZO zero-order pathway + (1-FZO) first-order pathway."""
    return fzo * zero_order(tad, TVDUR, dose) + (1.0 - fzo) * first_order(
        tad, TVKA, dose
    )


def simulate_subject(cl, v, fzo, dose):
    """RK4-integrate central' = R_in - (CL/V)*central; return {obs_time: conc}."""
    k = cl / v
    nsteps = int(round(OBS_TIMES[-1] / DT))
    obs_steps = {int(round(ot / DT)): ot for ot in OBS_TIMES}

    def deriv(t, central):
        return r_in(t, fzo, dose) - k * central

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
        # FZO fixed at the typical value (no IIV on the fraction in this DGP).
        conc = simulate_subject(cl, v, TVFZO, DOSE)
        # Dose record: CMT=1 (inert depot carrier), EVID=1, MDV=1, DV missing.
        rows.append(f"{sid},0,.,{fmt(DOSE)},1,1,1")
        for ot in OBS_TIMES:
            dv = conc[ot] * (1.0 + rng.gauss(0.0, SIG_PROP))  # proportional error
            rows.append(f"{sid},{fmt(ot)},{fmt(dv)},.,0,{obs_cmt},0")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
