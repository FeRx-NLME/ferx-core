#!/usr/bin/env python3
"""Datasets for the two steady-state-lagtime EDGE anchors (#1121 follow-up).

Reproducible generator (pure stdlib, seed 11) for four committed CSVs.  Run from
this directory:

    python3 simulate_ss_lag_edge_data.py

`ss_lag_infusion{,_flatwt}.csv` - an SS **infusion** whose PREVIOUS cycle is
    still running at the dose record (`II - ALAG < T_inf`), so the pre-arrival
    window carries a residual rate that belongs to no dose event.
      ID 1  ALAG 0.2   II - ALAG = 11.8 > T_inf = 6   control, cycle finished
      ID 2  ALAG 8.0   II - ALAG =  4.0 < 6           residual, ends at t = 2
      ID 3  ALAG 14    >= II, so the phase clamps to 0 and the window is [0, 6]

`ss_lag_ge_ii{,_flatwt}.csv` - an SS **bolus** with `ALAG >= II`, where the seed
    phase `II - ALAG` is not a phase at all.
      ID 1  ALAG  3.0  <  II = 12   control
      ID 2  ALAG 15.0  >= II        phase clamps to 0: the record holds the peak

Each pair is the same geometry twice: the `_flatwt` variant holds `WT` constant
so ferx routes the subject to its STATIC predictors, while the varying-`WT`
variant routes it to the two event-driven walks.  Four routes, one oracle.

DV is simulated from the periodic-train closed form with 5 % proportional noise.
It only sets the OFV level; the oracle here is NONMEM's `PRED` column, which is
eta = 0 and therefore independent of the DV values.
"""
import math
import random

TVCL, TVV, WTEXP = 10.0, 50.0, 0.75


def cl_of(wt):
    return TVCL * (wt / 70.0) ** WTEXP


def wt_at(t, breaks):
    """Piecewise-constant covariate: `breaks` is [(t_from, value), ...] sorted."""
    v = breaks[0][1]
    for tb, vb in breaks:
        if t >= tb:
            v = vb
    return v


def conc(t, doses, breaks, v=TVV):
    """1-cpt IV concentration at `t` from a list of (t_start, amt, rate) pulses,
    integrated through a piecewise-constant CL.  rate = 0 means a bolus."""
    # Explicit forward integration on a fine grid: exact enough for DV, and it
    # makes no assumption about the code under test.
    edges = sorted({0.0}
                   | {tb for tb, _ in breaks}
                   | {d[0] for d in doses}
                   | {d[0] + (d[1] / d[2] if d[2] > 0 else 0.0) for d in doses}
                   | {t})
    edges = [e for e in edges if e <= t + 1e-12]
    amt = 0.0
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        for ts, dose_amt, rate in doses:
            if rate == 0.0 and abs(ts - a) < 1e-12:
                amt += dose_amt
        k = cl_of(wt_at(b, breaks)) / v
        rin = sum(r for ts, da, r in doses
                  if r > 0 and ts <= a + 1e-12 and ts + da / r >= b - 1e-12)
        dt = b - a
        if k > 0:
            amt = amt * math.exp(-k * dt) + (rin / k) * (1.0 - math.exp(-k * dt))
        else:
            amt = amt + rin * dt
    for ts, dose_amt, rate in doses:
        if rate == 0.0 and abs(ts - t) < 1e-12:
            amt += dose_amt
    return amt / v


def train(lag, amt, rate, ii, horizon=None, n_back=60):
    """The pulse train an SS dose stands for, under the rule MEASURED from
    NONMEM 7.6.0 (runs e6/e7):

      the pulse preceding the dose record sits at `-p` where
      `p = max(ii - lag, 0)` — i.e. the record sees cycle phase `ii - lag`,
      CLAMPED at 0 rather than wrapped, so a lag of a full interval or more puts
      a pulse on the record itself;

      earlier pulses follow every `ii` back into the past, and the one real
      arrival is at `lag`.  There are no successors — an SS dose delivers once.
    """
    del horizon
    p = max(ii - lag, 0.0)
    return ([(-p - m * ii, amt, rate) for m in range(n_back)]
            + [(lag, amt, rate)])


def emit(path, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


COLS = ["ID", "TIME", "DV", "AMT", "RATE", "EVID", "CMT", "MDV", "WT", "SS", "II", "LG"]


def build(spec, rng):
    rows = []
    for sid, lag, breaks, obs, amt, rate in spec:
        doses = train(lag, amt, rate, 12.0, max(obs) + 1.0)
        wt0 = breaks[0][1]
        rows.append(dict(ID=sid, TIME=0, DV=".", AMT=amt,
                         RATE=(rate if rate > 0 else "."), EVID=1, CMT=1, MDV=1,
                         WT=wt0, SS=1, II=12, LG=lag))
        # covariate-change records that are not observation times
        for tb, vb in breaks[1:]:
            if tb not in obs:
                rows.append(dict(ID=sid, TIME=tb, DV=".", AMT=".", RATE=".",
                                 EVID=2, CMT=1, MDV=1, WT=vb, SS=".", II=".",
                                 LG=lag))
        for t in obs:
            c = conc(t, doses, breaks)
            dv = c * (1.0 + rng.gauss(0.0, 0.05))
            rows.append(dict(ID=sid, TIME=t, DV=f"{dv:.6f}", AMT=".", RATE=".",
                             EVID=0, CMT=1, MDV=0, WT=wt_at(t, breaks), SS=".",
                             II=".", LG=lag))
        rows.sort(key=lambda r: (r["ID"], float(r["TIME"]), -r["EVID"]))
    return rows


rng = random.Random(11)

# --- SS infusion, previous cycle still running at the record ---------------
# T_inf = 600/100 = 6 h, II = 12.
#   ID 1  LG = 0.2  ->  II - LG = 11.8 > 6   previous cycle long finished (control)
#   ID 2  LG = 8.0  ->  II - LG =  4.0 < 6   previous cycle STILL RUNNING at t = 0,
#                                            ending at 8 - 12 + 6 = 2.
# WT steps at t = 1 (inside that residual window) and t = 9 (inside the real
# infusion [8, 14]).
INF_BREAKS = [(0.0, 70.0), (1.0, 140.0), (9.0, 70.0)]
INF_OBS = [0.5, 1.5, 2.0, 3.0, 6.0, 7.5, 8.5, 10.0, 11.0, 14.0, 18.0, 23.0]
emit("ss_lag_infusion.csv", build([(1, 0.2, INF_BREAKS, INF_OBS, 600, 100),
                      (2, 8.0, INF_BREAKS, INF_OBS, 600, 100),
                      (3, 14.0, INF_BREAKS, INF_OBS, 600, 100)], rng), COLS)

# --- SS bolus, ALAG >= II --------------------------------------------------
# SS bolus, II = 12.
#   ID 1  LG =  3.0  <  II    (control)
#   ID 2  LG = 15.0  >= II    pre-arrival window [0, 15) contains a VIRTUAL
#                             pulse at 15 - 12 = 3.
# WT steps at t = 5, between that virtual pulse and the real arrival.
GEII_BREAKS = [(0.0, 70.0), (5.0, 140.0)]
GEII_OBS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 14.5, 16.0, 20.0, 26.0]
emit("ss_lag_ge_ii.csv", build([(1, 3.0, GEII_BREAKS, GEII_OBS, 600, 0),
                      (2, 15.0, GEII_BREAKS, GEII_OBS, 600, 0)], rng), COLS)

FLAT = [(0.0, 70.0)]
emit("ss_lag_infusion_flatwt.csv", build([(1, 0.2, FLAT, INF_OBS, 600, 100),
                       (2, 8.0, FLAT, INF_OBS, 600, 100),
                       (3, 14.0, FLAT, INF_OBS, 600, 100)], rng), COLS)
emit("ss_lag_ge_ii_flatwt.csv", build([(1, 3.0, FLAT, GEII_OBS, 600, 0),
                       (2, 15.0, FLAT, GEII_OBS, 600, 0)], rng), COLS)
