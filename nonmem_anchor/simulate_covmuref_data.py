#!/usr/bin/env python3
"""Simulate the two #619 covariate mu-referencing anchor datasets.

Both are 1-compartment IV bolus (100 mg at t = 0), proportional residual error,
60 subjects x 7 samples, written in NONMEM layout to `data/`:

* `covmuref_additive.csv` — CL = (TVCL + (CRCL - 90) * TH_CRCL) * exp(ETA_CL),
  V = TVV * exp(ETA_V). Truth: TVCL 5, TH_CRCL 0.05, TVV 50,
  omega^2_CL 0.09, omega^2_V 0.04, sigma 0.1 (sd, proportional).
  The additive renal gradient is the #619 form: NONMEM needs a *nonlinear*
  MU (`MU_1 = LOG(THETA(1) + (CRCL-90)*THETA(2))`).
* `covmuref_power.csv` — CL = TVCL * (WT/70)^TH_WT * exp(ETA_CL), same V and
  error. Truth: TVCL 5, TH_WT 0.75. Linear-in-theta MU
  (`MU_1 = LOG(THETA(1)) + THETA(2)*LOG(WT/70)`), NONMEM's efficient case.

CRCL ~ U(40, 140) and WT ~ U(45, 110) are per-subject constants (the exact
engine's regime); the fluconazole time-varying case is exercised separately.
Deterministic: stdlib `random.Random(619)`, no third-party dependency.
"""
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N, TIMES = 60, [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
DOSE = 100.0
rng = random.Random(619)


def write(path: Path, cov_name: str, cov, cl_fn):
    rows = ["ID,TIME,DV,AMT,EVID,CMT,%s" % cov_name]
    eta_cl = [rng.gauss(0.0, math.sqrt(0.09)) for _ in range(N)]
    eta_v = [rng.gauss(0.0, math.sqrt(0.04)) for _ in range(N)]
    for i in range(N):
        cl = cl_fn(cov[i]) * math.exp(eta_cl[i])
        v = 50.0 * math.exp(eta_v[i])
        rows.append(f"{i + 1},0,0,{DOSE:g},1,1,{cov[i]:.1f}")
        for t in TIMES:
            c = DOSE / v * math.exp(-cl / v * t)
            dv = c * (1.0 + rng.gauss(0.0, 0.1))
            rows.append(f"{i + 1},{t:g},{dv:.6g},0,0,1,{cov[i]:.1f}")
    path.write_text("\n".join(rows) + "\n")
    print("wrote", path, "subjects", N, "CL range", cl_fn(min(cov)), cl_fn(max(cov)))


crcl = [rng.uniform(40.0, 140.0) for _ in range(N)]
write(ROOT / "data/covmuref_additive.csv", "CRCL", crcl, lambda c: 5.0 + (c - 90.0) * 0.05)
wt = [rng.uniform(45.0, 110.0) for _ in range(N)]
write(ROOT / "data/covmuref_power.csv", "WT", wt, lambda w: 5.0 * (w / 70.0) ** 0.75)
