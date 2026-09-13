#!/usr/bin/env python3
"""
Generate a single-dose oral PK dataset with **parallel dual first-order
absorption** whose pathway fraction carries **logit-normal IIV**, for anchoring
ferx's logit-normal mu-referencing (FeRx-NLME/ferx-core#918) against NONMEM
SAEM with explicit MU_ referencing.

Pure standard library (math, random) -- no numpy/scipy. Deterministic (seed
below), so re-running reproduces logit_fraction_oral.csv byte-for-byte.

Structural model (1-cpt, two first-order absorption pathways splitting the dose
by FR1 / (1-FR1); the whole dose is delivered through the two inputs, no bolus)::

    central' = FR1*D*KA1*exp(-KA1*t) + FR2*D*KA2*exp(-KA2*t) - (CL/V)*central

which has the exact Bateman superposition used below::

    conc(t) = (D/V) * sum_i FRi * KAi/(KAi - k) * (exp(-k*t) - exp(-KAi*t)),  k = CL/V

Data-generating truths::

    CL = 5 L/h, V = 50 L, KA1 = 2.0 /h (fast), KA2 = 0.2 /h (slow)
    FR1_i = inv_logit(LOGIT_FR1 + eta_FR1,i),  LOGIT_FR1 = logit(0.6) = 0.405465
    IIV: omega^2(CL) = omega^2(V) = 0.09, omega^2(eta_FR1) = 0.25 (logit scale)
    proportional residual SD = 0.08

The fraction is *identifiable from the shape* of the curve (fast vs slow
pathway), not from the exposure magnitude -- unlike a bioavailability F, which
is confounded with CL/V in oral-only data. That is what makes this a clean test
of the estimator rather than of the design.

Output (NONMEM-format): ID,TIME,DV,AMT,EVID,CMT,MDV   (dose CMT=1, obs CMT=2),
matching nonmem_anchor/logit_fraction_saem.ctl (DEPOT inert dose carrier with
F1=0; the first-order sum feeds CENTRAL = CMT 2). Re-key obs to CMT=1 for
ferx's single-state model (data/logit_fraction_oral.csv).
"""
import math
import random

# ---- design / truths -------------------------------------------------------
SEED = 918
N_SUB = 60
DOSE = 100.0  # mg, single oral dose into CMT=1 at t=0
# Dense early sampling: the fast pathway is what the fraction acts on, so the
# absorption phase is where the per-subject FR1 signal lives.
OBS_TIMES = [
    0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0,
    4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0,
]
TVCL, TVV = 5.0, 50.0
TVKA1, TVKA2 = 2.0, 0.2  # fast / slow pathway absorption rates (1/h)
TVFR1 = 0.6  # typical fast-pathway fraction (probability scale)
LOGIT_FR1 = math.log(TVFR1 / (1.0 - TVFR1))  # 0.405465 -- the mu-referenced theta
OM_CL, OM_V = 0.09, 0.09  # IIV variances on CL, V
OM_FR1 = 0.25  # IIV variance of eta_FR1, on the logit scale
SIG_PROP = 0.08  # proportional residual SD


def conc(t, cl, v, fr1):
    """Exact Bateman superposition of the two first-order pathways (mg/L)."""
    k = cl / v
    total = 0.0
    for frac, ka in ((fr1, TVKA1), (1.0 - fr1, TVKA2)):
        total += frac * ka / (ka - k) * (math.exp(-k * t) - math.exp(-ka * t))
    return DOSE / v * total


def inv_logit(x):
    return 1.0 / (1.0 + math.exp(-x))


def fmt(x):
    return f"{x:g}"


def main():
    rng = random.Random(SEED)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV"]
    for sid in range(1, N_SUB + 1):
        cl = TVCL * math.exp(rng.gauss(0.0, math.sqrt(OM_CL)))
        v = TVV * math.exp(rng.gauss(0.0, math.sqrt(OM_V)))
        # Logit-normal individual fraction: eta enters on the logit scale.
        fr1 = inv_logit(LOGIT_FR1 + rng.gauss(0.0, math.sqrt(OM_FR1)))
        # Dose record: CMT=1 (inert depot carrier), EVID=1, MDV=1, DV missing.
        rows.append(f"{sid},0,.,{fmt(DOSE)},1,1,1")
        for t in OBS_TIMES:
            dv = conc(t, cl, v, fr1) * (1.0 + rng.gauss(0.0, SIG_PROP))
            rows.append(f"{sid},{fmt(t)},{fmt(dv)},.,0,2,0")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
