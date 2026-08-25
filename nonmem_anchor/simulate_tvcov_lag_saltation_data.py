#!/usr/bin/env python3
"""Datasets for the #1060 time-varying-covariate x IIV-lagtime NONMEM anchors.

Oral 1-cpt (`depot -> central`), `ALAG1 = TVP*exp(ETA3)`, `WT` allometric on
**both** `CL` and `KA`. The `KA` term is load-bearing: with a plain `KA` the depot
compartment's own velocity carries no covariate, the post-arrival field does not
jump, and neither #1060 nor the multi-dose divergence reproduces.

Four datasets, each isolating one question:

  A `tvcov_lag_saltation.csv`
      One dose at t = 0 on a `WT = 70` row; first record at t = 1 carries
      `WT = 140`. Every arrival lands in (0.52, 0.95), i.e. strictly inside the
      crossed interval, and every dose is a FIRST dose, so `g(x-) = 0` and only
      the post-arrival snapshot is under test. This is #1060 in isolation.

  B `tvcov_lag_saltation_multidose.csv`
      A adds a second dose at t = 6 whose arrival (~6.7) lands with residual drug
      present, so the PRE side is live too. Dose row `WT = 150`, next record
      (t = 7) `WT = 75`.

  D `tvcov_lag_saltation_md_control.csv`
      B with the t = 6 dose row's `WT` changed to 75, so the dose row and the next
      record agree. One number different from B; isolates the interval
      `[dose record, lagged arrival]` and nothing else.

  C `tvcov_lag_saltation_md_shrink.csv`
      B plus an extra record at t = 6.5 carrying `WT = 75` -- i.e. BETWEEN the dose
      row and the arrival. Pins which record's parameters govern that interval.

Etas are hand-picked (no RNG); only the residual noise is seeded, so the files are
byte-reproducible.
"""

import math
import random

TVCL, TVV, TVKA, TVP = 10.0, 50.0, 1.0, 0.7
WCL, WKA = 0.75, 0.75
AMT, PROP_SD = 100.0, 0.05

ETAS = [
    (0.20, -0.10, 0.28), (-0.15, 0.12, -0.24), (0.05, 0.20, 0.14),
    (-0.25, -0.18, -0.30), (0.30, 0.05, 0.19), (-0.05, -0.22, -0.12),
    (0.12, 0.16, 0.30), (-0.18, 0.08, -0.18),
]

SINGLE = [(0.0, 70.0, "dose"), (1.0, 140.0, "obs"), (1.5, 140.0, "obs"),
          (2.0, 140.0, "obs"), (3.0, 140.0, "obs"), (4.0, 150.0, "obs"),
          (6.0, 150.0, "obs"), (8.0, 160.0, "obs"), (12.0, 160.0, "obs")]

MULTI = [(0.0, 70.0, "dose"), (1.0, 140.0, "obs"), (1.5, 140.0, "obs"),
         (2.0, 140.0, "obs"), (3.0, 140.0, "obs"), (4.0, 150.0, "obs"),
         (5.0, 150.0, "obs"), (6.0, 150.0, "dose"), (7.0, 75.0, "obs"),
         (7.5, 75.0, "obs"), (8.0, 75.0, "obs"), (10.0, 75.0, "obs"),
         (12.0, 80.0, "obs"), (16.0, 80.0, "obs"), (24.0, 80.0, "obs")]


def flow(ad, ac, dt, ka, k):
    """Analytic advance of `depot -> central` over `dt` at constant (ka, k)."""
    ad2 = ad * math.exp(-ka * dt)
    ac2 = ac * math.exp(-k * dt) + ad * ka / (k - ka) * (
        math.exp(-ka * dt) - math.exp(-k * dt))
    return ad2, ac2


def build(grid, seed):
    """Simulate `grid` under the end-of-interval convention (the segment ENDING at
    an event runs on that event's record; a lagged arrival runs on its dose row's).
    """
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV,WT"]
    for i, (e1, e2, e3) in enumerate(ETAS, start=1):
        v, lag = TVV * math.exp(e2), TVP * math.exp(e3)
        ka_of = lambda wt: TVKA * (wt / 70.0) ** WKA
        k_of = lambda wt: (TVCL * (wt / 70.0) ** WCL * math.exp(e1)) / v
        events = [(t + lag, wt, "arrival") if kind == "dose" else (t, wt, "obs")
                  for (t, wt, kind) in grid]
        events.sort(key=lambda e: (e[0], 0 if e[2] == "arrival" else 1))
        ad = ac = 0.0
        cur = grid[0][0]
        out = {}
        for (t, wt, kind) in events:
            if t > cur:
                ad, ac = flow(ad, ac, t - cur, ka_of(wt), k_of(wt))
                cur = t
            if kind == "arrival":
                ad += AMT
            else:
                out[t] = ac / v
        for (n, (t, wt, kind)) in enumerate(grid):
            # `TIME` formatting matches the files the reference runs consumed: the
            # subject's first record is written `0`, every later one as a Python
            # float repr (`1.0`, `6.0`, `24.0`). Keep it byte-stable.
            ts = "0" if n == 0 else f"{t}"
            if kind == "dose":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f}")
            else:
                dv = out[t] * (1.0 + rng.gauss(0.0, PROP_SD))
                rows.append(f"{i},{ts},{dv:.6f},0,0,2,0,{wt:.0f}")
    return rows


def write(path, rows):
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"{path}: {len(rows) - 1} rows")


if __name__ == "__main__":
    write("tvcov_lag_saltation.csv", build(SINGLE, 1060))

    multi = build(MULTI, 10601)
    write("tvcov_lag_saltation_multidose.csv", multi)

    # D: only the t = 6 dose row's WT changes (150 -> 75).
    control = []
    for line in multi:
        p = line.split(",")
        if len(p) == 8 and p[1] == "6.0" and p[4] == "1":
            p[7] = "75"
        control.append(",".join(p))
    write("tvcov_lag_saltation_md_control.csv", control)

    # C: B plus a WT = 75 record at t = 6.5, between the dose row and the arrival.
    # Its DV is a placeholder -- this file exists for the PRED comparison, not a fit.
    shrink = [multi[0]]
    for line in multi[1:]:
        shrink.append(line)
        p = line.split(",")
        if p[1] == "6.0" and p[4] == "1":
            shrink.append(f"{p[0]},6.5,0.5,0,0,2,0,75")
    write("tvcov_lag_saltation_md_shrink.csv", shrink)
