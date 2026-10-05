#!/usr/bin/env python3
"""Dosing/sampling template for the #1661 wide-ODE anchor (pure stdlib, seed 1661).

Writes `wide_ode_myelo_template.csv`: the event skeleton (doses, observation times and
the per-subject PK columns) with DV = 0. NONMEM's own `$SIMULATION`
(`wide_ode_myelo_sim.ctl`) fills in DV, and `wide_ode_myelo_from_sim.py` turns its
table into the fitted dataset `wide_ode_myelo.csv`.

Design, per subject:
  - individual 2-cpt PK parameters as data columns (V1I, K10I, K12I, K21I), log-normal
    around 10 L, 0.1/h, 0.2/h, 0.1/h — a sequential PK/PD fit;
  - AMT=1 into each cell compartment (CMT 3..7) at t=0, scaled to baseline by
    F3..F7 = BAS;
  - three 150 mg IV boluses into CMT 1, two weeks apart (t = 0, 336, 672 h), so the
    later doses land on residual drug and a depressed cell count;
  - 12 circulating-cell observations (CMT 7), none on a dose time.
"""
import random

N_SUBJECTS = 24
DOSE_TIMES = [0.0, 336.0, 672.0]
OBS_TIMES = [24.0, 96.0, 168.0, 264.0, 360.0, 432.0, 504.0, 600.0, 696.0, 768.0, 840.0, 1008.0]

rng = random.Random(1661)
rows = []
for sid in range(1, N_SUBJECTS + 1):
    pk = {
        "V1I": 10.0 * rng.lognormvariate(0.0, 0.2),
        "K10I": 0.1 * rng.lognormvariate(0.0, 0.2),
        "K12I": 0.2 * rng.lognormvariate(0.0, 0.2),
        "K21I": 0.1 * rng.lognormvariate(0.0, 0.2),
    }
    pkcols = [f"{pk[k]:.6f}" for k in ("V1I", "K10I", "K12I", "K21I")]
    events = []
    for cmt in range(3, 8):
        events.append((0.0, 0, 1.0, cmt, 1, 1))  # baseline seed, sorted before the drug dose
    for t in DOSE_TIMES:
        events.append((t, 1, 150.0, 1, 1, 1))
    for t in OBS_TIMES:
        events.append((t, 2, 0.0, 7, 0, 0))
    events.sort(key=lambda e: (e[0], e[1]))
    for t, _, amt, cmt, evid, mdv in events:
        rows.append([str(sid), f"{t:g}", f"{amt:g}", str(cmt), str(evid), str(mdv), "0"] + pkcols)

with open("wide_ode_myelo_template.csv", "w") as f:
    f.write("ID,TIME,AMT,CMT,EVID,MDV,DV,V1I,K10I,K12I,K21I\n")
    for r in rows:
        f.write(",".join(r) + "\n")
