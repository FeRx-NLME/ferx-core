#!/usr/bin/env python3
"""Simulate oral data for the #1124 TAD-in-`[odes]` anchor.

Truth model (the convention both engines must agree on):

    A1' = -KA*A1                                     depot
    A2' =  KA*A1 - (CL/V)*A2*(1 + THETA_TAD*TAD)     central
    TAD(t) = t - (most recent dose time <= t)

Pure-stdlib RK4, fixed seed. No ferx code involved, so the dataset does not
inherit either engine's convention by construction.

Anchor-design notes
-------------------
* **Two doses.** With a single dose `TAD == TAFD` and the anchor could not tell a
  `TAD` defect from a `TAFD` one — the term under test would be pinned against the
  wrong quantity. The second dose at t = 12 re-anchors `TAD`, and observations
  straddle it.
* **The second dose lands on residual drug.** At the typical-value parameters the
  central amount just before it (t = 11.5) is 22.90, a quarter of the eventual
  peak of 91.21 at t = 14 — so the post-second-dose trajectory depends on both the
  new input *and* the decaying old one. A dose arriving into an empty compartment
  would leave the incoming side of the event degenerate: `g(x⁻) = 0` cancels any
  error there instead of exposing it.
* **No lagtime.** #1124 is independent of #1070: the closed form dropped the `TAD`
  term with no lag anywhere. Leaving the lag out keeps this anchor free of the
  `MTIME`/`MTDIFF` machinery `tad_lag_B` needs, and free of #1070's separate
  gradient question. `TDOS` switches exactly at a dose record, which NONMEM
  already breaks the integration on.
* **No `WT` column.** `tad_lag_{A,B}` carry a `WT` covariate with a FIXED-zero
  exponent purely to force ferx onto its event-driven predictor, because the dense
  one returned `NaN` for a `TAD`-reading RHS under a lagtime (#1110). #1124 routes
  every model-time-reading RHS to the event-driven path directly, so that
  workaround is obsolete and is deliberately not repeated here.
* **The ferx twin is a *one-state* model** (`d/dt(central) = first_order(ka=KA) …`)
  while the NONMEM twin uses an explicit DEPOT compartment. These are the same
  system: ferx's `first_order` forcing delivers `KA*F*D*exp(-KA*(t-t_dose))` into
  central, which is exactly the outflow of a linear depot holding `F*D*exp(-KA*t)`.
  The depot is unaffected by central, so the central trajectory is identical.
  Both read the same one-record-per-dose dataset.
"""
import math
import random

TVCL, TVV, TVKA, THETA_TAD = 1.0, 20.0, 0.8, 0.3
OM_CL, SIG_PROP = 0.09, 0.10
N_SUB = 12
DOSE_T = [0.0, 12.0]
DOSE_AMT = 100.0
# Observations straddle the second dose so the `TAD` re-anchoring is exercised on
# both sides, and run out to 24 h where the dropped-term error is largest.
OBS_T = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.5, 12.5, 14.0, 16.0, 20.0, 24.0]


def tad(t):
    return t - max(d for d in DOSE_T if d <= t + 1e-12)


def simulate(cl, v, ka):
    """RK4 over [0, 24] with a hard break at each dose; returns {t: A2}."""
    events = sorted(set(DOSE_T) | set(OBS_T))
    a1 = a2 = 0.0
    t = 0.0
    out = {}

    def deriv(tt, x1, x2):
        return -ka * x1, ka * x1 - (cl / v) * x2 * (1.0 + THETA_TAD * tad(tt))

    for tgt in events:
        while t < tgt - 1e-12:
            h = min(0.001, tgt - t)
            k11, k12 = deriv(t, a1, a2)
            k21, k22 = deriv(t + h / 2, a1 + h / 2 * k11, a2 + h / 2 * k12)
            k31, k32 = deriv(t + h / 2, a1 + h / 2 * k21, a2 + h / 2 * k22)
            k41, k42 = deriv(t + h, a1 + h * k31, a2 + h * k32)
            a1 += h / 6 * (k11 + 2 * k21 + 2 * k31 + k41)
            a2 += h / 6 * (k12 + 2 * k22 + 2 * k32 + k42)
            t += h
        if tgt in DOSE_T:
            a1 += DOSE_AMT  # bolus into the depot, at the record time (no lag)
        if tgt in OBS_T:
            out[tgt] = a2
    return out


def main():
    random.seed(20260829)
    rows = ["ID,TIME,DV,MDV,EVID,AMT,CMT"]
    for sid in range(1, N_SUB + 1):
        cl = TVCL * math.exp(random.gauss(0.0, math.sqrt(OM_CL)))
        v = TVV
        amounts = simulate(cl, v, TVKA)
        recs = []
        for d in DOSE_T:
            recs.append((d, 0, f"{sid},{d:g},.,1,1,{DOSE_AMT:g},1"))
        for t in OBS_T:
            ipred = amounts[t] / v
            dv = ipred * (1.0 + random.gauss(0.0, SIG_PROP))
            recs.append((t, 1, f"{sid},{t:g},{dv:.6f},0,0,.,1"))
        # Dose records sort before observations at the same time.
        recs.sort(key=lambda r: (r[0], r[1]))
        rows.extend(r[2] for r in recs)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
