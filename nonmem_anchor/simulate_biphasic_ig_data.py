#!/usr/bin/env python3
"""
Generate a single-dose oral PK dataset with **biphasic** inverse-Gaussian
(Freijer & Post) absorption, for anchoring ferx's pathway-fraction mechanism
(FR1*igd(...) + FR2*igd(...), issue FeRx-NLME/ferx-core#388) against NONMEM.

Pure standard library (math, random) -- no numpy/scipy. Deterministic (seed
below), so re-running reproduces biphasic_ig_oral.csv byte-for-byte.

Structural model -- identical to examples/biphasic_igd_absorption.ferx
(1-cpt, central tracked as AMOUNT mg; concentration = central/V):

    central' = R_in(tad) - (CL/V)*central
    R_in(tad) = FR1 * IG(tad; MAT1, CV2_1) + FR2 * IG(tad; MAT2, CV2_2)
    IG(tad; mat, cv2) = dose * sqrt(mat/(2*pi*cv2*tad^3))
                              * exp(-(tad-mat)^2 / (2*cv2*mat*tad))
    FR2 = 1 - FR1 ,  dose = F*amt  (F = 1)

The two IG pathways split the dose by FR1 / (1-FR1): a fast pathway (short MAT1)
and a slow pathway (long MAT2). The whole dose is delivered through R_in over
time (the bolus is NOT added) -- exactly the ferx igd() convention and the
NONMEM F1=0 + PODO trick.

Data-generating truths -- identical to examples/biphasic_igd_absorption.ferx:
TVCL=5, TVV=50, TVFR1=0.6, TVMAT1=0.5, TVMAT2=4.0, TVCV2_1=0.2, TVCV2_2=0.5;
IIV on CL and V only (omega = 0.09 each, ~30% CV); proportional residual SD 0.15.

Output (NONMEM-format): ID,TIME,DV,AMT,EVID,CMT,MDV   (dose CMT=1, obs CMT=2),
matching nonmem_anchor/freijer_biphasic_ig.ctl (DEPOT inert dose carrier with
F1=0; IG sum feeds CENTRAL = CMT 2).
"""
import math
import random

# ---- design / truths -------------------------------------------------------
SEED = 7
N_SUB = 20
DOSE = 100.0  # mg, single oral dose into CMT=1 at t=0
OBS_TIMES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
TVCL, TVV = 5.0, 50.0
TVFR1 = 0.6                      # fast-pathway fraction (slow = 1 - FR1)
TVMAT1, TVCV2_1 = 0.5, 0.2       # fast pathway: short mean absorption time
TVMAT2, TVCV2_2 = 4.0, 0.5       # slow pathway: long mean absorption time
OM_CL, OM_V = 0.09, 0.09         # IIV variances on CL, V
SIG_PROP = 0.15                  # proportional residual SD
DT = 0.005                       # RK4 step (h); every OBS_TIME is an integer multiple
TWO_PI = 2.0 * math.pi


def ig(tad, mat, cv2, dose):
    """Inverse-Gaussian absorption-time density scaled by the dose (mg/h)."""
    if tad <= 0.0:
        return 0.0
    arg = -((tad - mat) ** 2) / (2.0 * cv2 * mat * tad)
    return dose * math.sqrt(mat / (TWO_PI * cv2 * tad ** 3)) * math.exp(arg)


def r_in(tad, fr1, dose):
    """Biphasic input rate: FR1 fast pathway + (1-FR1) slow pathway."""
    return fr1 * ig(tad, TVMAT1, TVCV2_1, dose) + (1.0 - fr1) * ig(
        tad, TVMAT2, TVCV2_2, dose
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
    rng = random.Random(SEED)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV"]
    for sid in range(1, N_SUB + 1):
        cl = TVCL * math.exp(rng.gauss(0.0, math.sqrt(OM_CL)))
        v = TVV * math.exp(rng.gauss(0.0, math.sqrt(OM_V)))
        # FR1 fixed at the typical value (no IIV on the fraction in this DGP).
        conc = simulate_subject(cl, v, TVFR1, DOSE)
        # Dose record: CMT=1 (inert depot carrier), EVID=1, MDV=1, DV missing.
        rows.append(f"{sid},0,.,{fmt(DOSE)},1,1,1")
        for ot in OBS_TIMES:
            dv = conc[ot] * (1.0 + rng.gauss(0.0, SIG_PROP))  # proportional error
            rows.append(f"{sid},{fmt(ot)},{fmt(dv)},.,0,2,0")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
