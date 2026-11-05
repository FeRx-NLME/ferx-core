#!/usr/bin/env python3
"""Simulate a one-compartment oral population in which some subjects are flip-flop.

For #2096 (the inner screen's descent budget). Fitted by examples/flip_flop_pop.ferx (the
closed form) and examples/flip_flop_pop_ode.ferx (its ODE twin).

Each subject gets one oral dose into the depot. Its concentration is the Bateman function

    C(t) = DOSE·KA / (V·(KA − k)) · (exp(−k·t) − exp(−KA·t)),   k = CL / V,

which is unchanged when KA and k swap and V becomes V·k/KA. A subject with KA < k is
flip-flop: its curve is fitted equally well by the swapped pair, so its individual objective
has a second mode that is off every random-effect axis (η_V and η_KA move together by
ln(k/KA)). With ω_KA = 1 around TVKA = 0.2 and a typical k of 0.1, about a quarter of the
subjects are flip-flop. The closed-form fit has the twin seed (`inner_twin_seed`, #2095); the
ODE twin takes no seed, so only the Sobol screen (`inner_screen`, #2013) reaches that mode.

The generator is plain Python and never runs ferx, so the data do not depend on the
engine they test. It is deterministic given SEED, so the committed CSV is reproducible.

Writes data/flip_flop_pop.csv (columns ID,TIME,DV,EVID,AMT,CMT,MDV): one dose row per
subject (EVID=1, CMT=1) plus its 12 observations (EVID=0, CMT=1, the readout of the closed
form; the ODE twin reads `central`). Prints the number of flip-flop subjects to stderr.
"""
import math
import os
import random
import sys

SEED = 2096
N_SUBJECTS = 80
DOSE = 100.0
TVCL, TVV, TVKA = 1.0, 10.0, 0.2
OMEGA_CL, OMEGA_V, OMEGA_KA = 0.09, 0.09, 1.0  # variances of ETA_CL, ETA_V, ETA_KA
SIGMA_PROP = 0.1  # proportional residual SD
OBS_TIMES = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0, 96.0]
# Every sample is kept at its simulated value, however small: a 10 % proportional error
# keeps it positive. Clamping the late, tiny ones to a floor (1e-3) puts points on a curve
# the model does not have, and gave the correctly specified fit a Durbin-Watson of 1.16;
# dropping those below an LLOQ of 0.05 instead left 9 subjects with two individual modes
# and a fit that stagnates, a harder fixture than this one is meant to be.


def bateman(t, cl, v, ka):
    k = cl / v
    if abs(ka - k) < 1e-12 * k:
        # The KA = k limit: C(t) = DOSE·k·t·exp(−k·t) / V.
        return DOSE * k * t * math.exp(-k * t) / v
    return DOSE * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))


def main():
    rng = random.Random(SEED)
    rows = ["ID,TIME,DV,EVID,AMT,CMT,MDV"]
    n_flip_flop = 0
    for sid in range(1, N_SUBJECTS + 1):
        cl = TVCL * math.exp(rng.gauss(0.0, math.sqrt(OMEGA_CL)))
        v = TVV * math.exp(rng.gauss(0.0, math.sqrt(OMEGA_V)))
        ka = TVKA * math.exp(rng.gauss(0.0, math.sqrt(OMEGA_KA)))
        n_flip_flop += ka < cl / v
        rows.append(f"{sid},0,.,1,{DOSE:g},1,1")
        for t in OBS_TIMES:
            ipred = bateman(t, cl, v, ka)
            dv = ipred * (1.0 + SIGMA_PROP * rng.gauss(0.0, 1.0))
            assert dv > 0.0, f"subject {sid}, t = {t}: non-positive DV {dv}"
            rows.append(f"{sid},{t:g},{dv:.6g},0,.,1,0")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "flip_flop_pop.csv")
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {os.path.normpath(out)}: {N_SUBJECTS} subjects, {n_flip_flop} flip-flop (KA < CL/V)", file=sys.stderr)


if __name__ == "__main__":
    main()
