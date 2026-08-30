#!/usr/bin/env python3
"""Datasets for the #1073 dosing-form x lagged-arrival NONMEM anchors.

`tvcov_lag_saltation_multidose` (#1060/#1073 anchor B) puts ONE dosing form — an
oral bolus — in the discriminating cell: a lagged arrival landing inside a record
interval whose covariate differs from the dose row's. The convention it pins is
not specific to that form, so each dataset here moves one other dosing feature
into the same cell and changes nothing else:

  E2 `dose_form_lag_infusion.csv`
      Rate-defined INFUSION with a lagtime. The second infusion's window END
      (`t + ALAG + dur`) falls strictly between two records, so this also pins the
      non-record rate-off boundary, which is a second event kind governed by the
      same rule and which the closed-form and ODE engines used to answer
      differently.

  E3 `dose_form_lag_ss.csv`
      STEADY STATE (`SS=1`, `II=12`) with a lagtime. The SS dose's own
      equilibration is a dose *attribute* and must keep reading the dose row's
      snapshot even though the interval it opens no longer does.

  E4 `dose_form_lag_reset.csv`
      EVID=4 RESET + dose with a lagtime, so the reset, the dose record and the
      arrival are three distinct instants and their ordering is observable.

  E5 `dose_form_lag_two_cmt.csv`
      TWO dose compartments carrying DIFFERENT lagtimes (`ALAG1`, `ALAG2`), the
      NONMEM-expressible analogue of parallel absorption with per-route delays.
      Both arrivals land inside covariate-crossing intervals, and they land at
      different times, so one dose's arrival sits inside the other's
      `[record, arrival]` window.

Every dataset keeps the two properties that make the cell discriminating: the
dose row's covariate DIFFERS from the record that terminates the interval its
arrival lands in, and every dose after the first lands with residual drug present
so both sides of the boundary are live.

Etas are hand-picked (no RNG); only the residual noise is seeded, so the files are
byte-reproducible.
"""

import math
import random

TVCL, TVV, TVKA = 10.0, 50.0, 1.0
WCL, WKA = 0.75, 0.75
AMT, PROP_SD = 100.0, 0.05

# (ETA_CL, ETA_V, ETA_LAG) — the same eight subjects as the #1060 family, so a
# reader comparing the two sets is looking at one variable.
ETAS = [
    (0.20, -0.10, 0.28), (-0.15, 0.12, -0.24), (0.05, 0.20, 0.14),
    (-0.25, -0.18, -0.30), (0.30, 0.05, 0.19), (-0.05, -0.22, -0.12),
    (0.12, 0.16, 0.30), (-0.18, 0.08, -0.18),
]


def k_of(wt, e_cl, v):
    return (TVCL * (wt / 70.0) ** WCL * math.exp(e_cl)) / v


def ka_of(wt):
    return TVKA * (wt / 70.0) ** WKA


def iv_flow(ac, dt, k, rate):
    """1-cpt central advance over `dt` at constant `k` with zero-order input `rate`."""
    if k <= 0.0:
        return ac + rate * dt
    return ac * math.exp(-k * dt) + rate / k * (1.0 - math.exp(-k * dt))


def oral_flow(ad, ac, dt, ka, k):
    ad2 = ad * math.exp(-ka * dt)
    ac2 = ac * math.exp(-k * dt) + ad * ka / (k - ka) * (
        math.exp(-ka * dt) - math.exp(-k * dt))
    return ad2, ac2


def write(path, rows):
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"{path}: {len(rows) - 1} rows")


# --------------------------------------------------------------- E2 infusion

# (time, WT, kind, amt, rate). Infusion 2 starts at 6 (+lag) and runs 0.5 h, so
# its window end lands strictly inside (6, 7] under every subject's lagtime.
E2_GRID = [
    (0.0, 70.0, "dose", 100.0, 100.0),
    (1.0, 140.0, "obs", 0.0, 0.0),
    (2.0, 140.0, "obs", 0.0, 0.0),
    (3.0, 140.0, "obs", 0.0, 0.0),
    (4.0, 150.0, "obs", 0.0, 0.0),
    (5.0, 150.0, "obs", 0.0, 0.0),
    (6.0, 150.0, "dose", 50.0, 100.0),
    (7.0, 75.0, "obs", 0.0, 0.0),
    (8.0, 75.0, "obs", 0.0, 0.0),
    (10.0, 75.0, "obs", 0.0, 0.0),
    (12.0, 80.0, "obs", 0.0, 0.0),
    (16.0, 80.0, "obs", 0.0, 0.0),
    (24.0, 80.0, "obs", 0.0, 0.0),
]


def sim_e2(e_cl, e_v, lag):
    """IV infusions with a lagtime, under the record convention: every segment —
    including the piece cut off by the window end — runs on the record that
    terminates the interval containing it."""
    v = TVV * math.exp(e_v)
    recs = E2_GRID
    events = [(t, "rec", i) for i, (t, _, _, _, _) in enumerate(recs)]
    for (t, _, kind, amt, rate) in recs:
        if kind == "dose" and rate > 0.0:
            events.append((t + lag, "on", -1))
            events.append((t + lag + amt / rate, "off", -1))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "rec" else 1))

    def wt_at(t):
        for (rt, wt, _, _, _) in recs:
            if rt >= t - 1e-12:
                return wt
        return recs[-1][1]

    ac, cur, rate_on, out = 0.0, recs[0][0], 0.0, {}
    for (t, etype, idx) in events:
        if t > cur:
            ac = iv_flow(ac, t - cur, k_of(wt_at(t), e_cl, v), rate_on)
            cur = t
        if etype == "on":
            rate_on += 100.0
        elif etype == "off":
            rate_on -= 100.0
        elif recs[idx][2] == "obs":
            out[t] = ac / v
    return out


def emit_e2(path, seed):
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,AMT,RATE,EVID,CMT,MDV,WT"]
    for (i, (e1, e2, e3)) in enumerate(ETAS, start=1):
        ip = sim_e2(e1, e2, 0.7 * math.exp(e3))
        for (n, (t, wt, kind, amt, rate)) in enumerate(E2_GRID):
            ts = "0" if n == 0 else f"{t}"
            if kind == "dose":
                rows.append(f"{i},{ts},.,{amt},{rate},1,1,1,{wt:.0f}")
            else:
                dv = ip[t] * (1.0 + rng.gauss(0.0, PROP_SD))
                rows.append(f"{i},{ts},{dv:.6f},0,0,0,1,0,{wt:.0f}")
    write(path, rows)


# ------------------------------------------------------------------- E3 SS

# An SS=1 oral dose at t=0 (II=12) establishes the periodic profile, then a
# second SS-cycle dose at t=12 whose arrival crosses a covariate change.
E3_GRID = [
    (0.0, 70.0, "ss"),
    (1.0, 140.0, "obs"),
    (2.0, 140.0, "obs"),
    (4.0, 150.0, "obs"),
    (8.0, 150.0, "obs"),
    (11.0, 150.0, "obs"),
    (12.0, 150.0, "dose"),
    (13.0, 75.0, "obs"),
    (14.0, 75.0, "obs"),
    (16.0, 75.0, "obs"),
    (20.0, 80.0, "obs"),
    (24.0, 80.0, "obs"),
]
E3_II = 12.0


def sim_e3(e_cl, e_v, lag, grid=None):
    """Oral depot->central with an SS=1 first dose. The steady state is reached by
    running enough cycles that the trough has converged, which is what NONMEM's
    SS solves for exactly; 200 cycles is far past convergence for these rates.

    `grid` defaults to `E3_GRID`; `emit_e3p` passes the variant with an extra
    pre-arrival sample. The steady-state pre-load runs at the dose row's WT = 70
    either way — the equilibration is a property of that record — and only the
    forward walk sees the record grid."""
    grid = E3_GRID if grid is None else grid
    v = TVV * math.exp(e_v)
    ad = ac = 0.0
    ka0, k0 = ka_of(70.0), k_of(70.0, e_cl, v)
    for _ in range(200):
        ad += AMT
        ad, ac = oral_flow(ad, ac, E3_II, ka0, k0)

    def wt_at(t):
        for (rt, wt, _) in grid:
            if rt >= t - 1e-12:
                return wt
        return grid[-1][1]

    events = [(t, "rec", i) for i, (t, _, _) in enumerate(grid)]
    for (t, _, kind) in grid:
        if kind in ("ss", "dose"):
            events.append((t + lag, "arr", -1))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "rec" else 1))

    cur, out = grid[0][0], {}
    for (t, etype, idx) in events:
        if t > cur:
            w = wt_at(t)
            ad, ac = oral_flow(ad, ac, t - cur, ka_of(w), k_of(w, e_cl, v))
            cur = t
        if etype == "arr":
            ad += AMT
        elif grid[idx][2] == "obs":
            out[t] = ac / v
    return out


def emit_e3(path, seed):
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV,WT,SS,II"]
    for (i, (e1, e2, e3)) in enumerate(ETAS, start=1):
        ip = sim_e3(e1, e2, 0.7 * math.exp(e3))
        for (n, (t, wt, kind)) in enumerate(E3_GRID):
            ts = "0" if n == 0 else f"{t}"
            if kind == "ss":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f},1,{E3_II}")
            elif kind == "dose":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f},0,0")
            else:
                dv = ip[t] * (1.0 + rng.gauss(0.0, PROP_SD))
                rows.append(f"{i},{ts},{dv:.6f},0,0,2,0,{wt:.0f},0,0")
    write(path, rows)


# ---------------------------------------------------------------- E4 EVID=4

# EVID=4 at t=6: reset AND dose on one row, with a lagtime, so the reset (t=6),
# the dose record (t=6) and the arrival (~6.7) are three distinct events.
E4_GRID = [
    (0.0, 70.0, "dose"),
    (1.0, 140.0, "obs"),
    (2.0, 140.0, "obs"),
    (3.0, 140.0, "obs"),
    (4.0, 150.0, "obs"),
    (5.0, 150.0, "obs"),
    (6.0, 150.0, "reset_dose"),
    (7.0, 75.0, "obs"),
    (7.5, 75.0, "obs"),
    (8.0, 75.0, "obs"),
    (10.0, 75.0, "obs"),
    (12.0, 80.0, "obs"),
    (16.0, 80.0, "obs"),
    (24.0, 80.0, "obs"),
]


def sim_e4(e_cl, e_v, lag):
    v = TVV * math.exp(e_v)

    def wt_at(t):
        for (rt, wt, _) in E4_GRID:
            if rt >= t - 1e-12:
                return wt
        return E4_GRID[-1][1]

    events = [(t, "rec", i) for i, (t, _, _) in enumerate(E4_GRID)]
    for (t, _, kind) in E4_GRID:
        if kind in ("dose", "reset_dose"):
            events.append((t + lag, "arr", -1))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "rec" else 1))

    ad = ac = 0.0
    cur, out = E4_GRID[0][0], {}
    for (t, etype, idx) in events:
        if t > cur:
            w = wt_at(t)
            ad, ac = oral_flow(ad, ac, t - cur, ka_of(w), k_of(w, e_cl, v))
            cur = t
        if etype == "arr":
            ad += AMT
        elif E4_GRID[idx][2] == "reset_dose":
            # EVID=4 zeroes every compartment at the record time; the dose it
            # carries lands later, at the lagged arrival.
            ad = ac = 0.0
        elif E4_GRID[idx][2] == "obs":
            out[t] = ac / v
    return out


def emit_e4(path, seed):
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV,WT"]
    for (i, (e1, e2, e3)) in enumerate(ETAS, start=1):
        ip = sim_e4(e1, e2, 0.7 * math.exp(e3))
        for (n, (t, wt, kind)) in enumerate(E4_GRID):
            ts = "0" if n == 0 else f"{t}"
            if kind == "dose":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f}")
            elif kind == "reset_dose":
                rows.append(f"{i},{ts},.,{AMT},4,1,1,{wt:.0f}")
            else:
                dv = ip[t] * (1.0 + rng.gauss(0.0, PROP_SD))
                rows.append(f"{i},{ts},{dv:.6f},0,0,2,0,{wt:.0f}")
    write(path, rows)


# ------------------------------------------------- E3P pre-arrival SS sample

# E3 with ONE extra observation, at t = 0.5. Every subject's lagtime lies in
# [0.536, 0.951], so that sample falls strictly inside the pre-arrival window
# `[dose row, arrival)` for all eight — which the E3 grid never does, its first
# observation being at t = 1.
#
# That gap matters more than it looks. E3 pins the pre-arrival *propagation*,
# because the seeded state only reaches an observation after flowing through the
# arrival and the dose pulse. Nothing in it reads the pre-arrival state directly,
# so a seed that were right up to an additive constant, or evaluated at the wrong
# phase but corrected by the pulse, would still pass. This grid reads it.
#
# It doubles as an invariance check of the #1073 kind: the new record carries
# WT = 140, the same value the t = 1 record already contributes to the interval
# it terminates, so splitting `(0, arrival]` in two must not move any *other*
# prediction. Adding a sample cannot change the pharmacology.
E3P_GRID = [E3_GRID[0], (0.5, 140.0, "obs")] + E3_GRID[1:]


def emit_e3p(path, seed):
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV,WT,SS,II"]
    for (i, (e1, e2, e3)) in enumerate(ETAS, start=1):
        lag = 0.7 * math.exp(e3)
        assert 0.5 < lag < E3_II, (
            f"subject {i}: t=0.5 must be strictly inside [0, lag) and lag inside "
            f"one interval, got lag={lag:.4f}"
        )
        ip = sim_e3(e1, e2, lag, grid=E3P_GRID)
        for (n, (t, wt, kind)) in enumerate(E3P_GRID):
            ts = "0" if n == 0 else f"{t}"
            if kind == "ss":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f},1,{E3_II}")
            elif kind == "dose":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f},0,0")
            else:
                dv = ip[t] * (1.0 + rng.gauss(0.0, PROP_SD))
                rows.append(f"{i},{ts},{dv:.6f},0,0,2,0,{wt:.0f},0,0")
    write(path, rows)


# ------------------------------------------------- E3B non-first SS record

# E3 and E3P both put the `SS=1` row first, where the compartments are empty
# before it. That makes one half of the steady-state load invisible: an SS record
# does not merely *fill* the compartments, it **replaces** them, and where that
# replacement happens is exactly what #1121 moved — from the lagged arrival back to
# the dose record.
#
# Here a plain dose runs first and its drug is still present at t = 12, where an
# `SS=1` row lands with a lagtime. The sample at t = 12.3 sits inside that record's
# pre-arrival window under every subject's lagtime, so it reads whichever state the
# engine believes is there: the previous dose's residual (if the replacement is
# deferred to the arrival) or the steady-state tail (if it happens at the record).
# Those differ by a large factor, not a few percent — this is the geometry where
# the two conventions separate loudly rather than subtly.
E3B_GRID = [
    (0.0, 70.0, "dose"),
    (1.0, 140.0, "obs"),
    (2.0, 140.0, "obs"),
    (4.0, 150.0, "obs"),
    (8.0, 150.0, "obs"),
    (11.0, 150.0, "obs"),
    (12.0, 150.0, "ss"),
    (12.3, 75.0, "obs"),
    (13.0, 75.0, "obs"),
    (14.0, 75.0, "obs"),
    (16.0, 75.0, "obs"),
    (20.0, 80.0, "obs"),
    (24.0, 80.0, "obs"),
]


def sim_e3b(e_cl, e_v, lag):
    """Plain oral dose at t=0, then an `SS=1` record at t=12 carrying a lagtime.

    The steady state is equilibrated under the SS *record's* own WT (150) — it is a
    property of that row — and loaded at the record, at phase `II - lag`, replacing
    whatever the first dose left behind. The walk then advances to the arrival under
    the records that terminate the intervening intervals (WT = 75)."""
    v = TVV * math.exp(e_v)

    def wt_at(t):
        for (rt, wt, _) in E3B_GRID:
            if rt >= t - 1e-12:
                return wt
        return E3B_GRID[-1][1]

    events = [(t, "rec", i) for i, (t, _, _) in enumerate(E3B_GRID)]
    for (t, _, kind) in E3B_GRID:
        if kind in ("ss", "dose"):
            events.append((t + lag, "arr", -1))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "rec" else 1))

    ad = ac = 0.0
    cur, out = E3B_GRID[0][0], {}
    for (t, etype, idx) in events:
        if t > cur:
            w = wt_at(t)
            ad, ac = oral_flow(ad, ac, t - cur, ka_of(w), k_of(w, e_cl, v))
            cur = t
        if etype == "arr":
            ad += AMT
        elif E3B_GRID[idx][2] == "ss":
            # Replace the state with the periodic solution at phase `II - lag`,
            # equilibrated under this record's WT. `oral_flow` from the converged
            # trough, through the pulse, for `II - lag` hours.
            wt_ss = E3B_GRID[idx][1]
            kas, ks = ka_of(wt_ss), k_of(wt_ss, e_cl, v)
            ad = ac = 0.0
            for _ in range(200):
                ad += AMT
                ad, ac = oral_flow(ad, ac, E3_II, kas, ks)
            ad += AMT
            ad, ac = oral_flow(ad, ac, E3_II - lag, kas, ks)
        elif E3B_GRID[idx][2] == "obs":
            out[t] = ac / v
    return out


def emit_e3b(path, seed):
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,AMT,EVID,CMT,MDV,WT,SS,II"]
    for (i, (e1, e2, e3)) in enumerate(ETAS, start=1):
        lag = 0.7 * math.exp(e3)
        assert 0.3 < lag < E3_II, (
            f"subject {i}: t=12.3 must fall strictly inside [12, 12+lag), got "
            f"lag={lag:.4f}"
        )
        ip = sim_e3b(e1, e2, lag)
        for (n, (t, wt, kind)) in enumerate(E3B_GRID):
            ts = "0" if n == 0 else f"{t}"
            if kind == "ss":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f},1,{E3_II}")
            elif kind == "dose":
                rows.append(f"{i},{ts},.,{AMT},1,1,1,{wt:.0f},0,0")
            else:
                dv = ip[t] * (1.0 + rng.gauss(0.0, PROP_SD))
                rows.append(f"{i},{ts},{dv:.6f},0,0,2,0,{wt:.0f},0,0")
    write(path, rows)


# ------------------------------------------------ E5 SS INFUSION + lagtime

# E3's family is all oral boluses, so the seed they exercise is the plain
# `trough -> pulse -> decay` one. A steady-state *infusion* takes a different
# branch: the phase `II - lag` the record is seeded at lies past the end of the
# previous cycle's infusion window, so reconstructing it means crossing the
# rate-off boundary — an active leg then a quiet leg — rather than decaying
# monotonically. Under flat covariates that branch is already covered by unit
# tests; under time-varying ones nothing anchored it, which is a gap in exactly
# the code #1121 changed.
#
# 1-cpt IV, `RATE`-defined (AMT 100 / RATE 100 -> a 1 h window), `II = 12`, so
# `phase = II - lag` is about 11.3 and comfortably past `dur`. The sample at
# t = 0.5 is inside every subject's pre-arrival window, and a second, plain
# infusion at t = 12 keeps the later cycle live.
E5_GRID = [
    (0.0, 70.0, "ss", 100.0, 100.0),
    (0.5, 140.0, "obs", 0.0, 0.0),
    (1.0, 140.0, "obs", 0.0, 0.0),
    (2.0, 140.0, "obs", 0.0, 0.0),
    (4.0, 150.0, "obs", 0.0, 0.0),
    (8.0, 150.0, "obs", 0.0, 0.0),
    (11.0, 150.0, "obs", 0.0, 0.0),
    (12.0, 150.0, "dose", 100.0, 100.0),
    (13.0, 75.0, "obs", 0.0, 0.0),
    (14.0, 75.0, "obs", 0.0, 0.0),
    (16.0, 75.0, "obs", 0.0, 0.0),
    (20.0, 80.0, "obs", 0.0, 0.0),
    (24.0, 80.0, "obs", 0.0, 0.0),
]
E5_II = 12.0


def sim_e5(e_cl, e_v, lag):
    """1-cpt IV with an `SS=1` rate-defined infusion carrying a lagtime.

    The steady state is equilibrated under the dose row's own WT = 70 as a
    repeating [infuse `dur`, decay `II - dur`] cycle, then advanced to phase
    `II - lag` — across the rate-off boundary — and loaded at the record. From
    there the walk runs under each terminating record's WT, and each dose's own
    infusion window opens at `t + lag`."""
    v = TVV * math.exp(e_v)
    dur = E5_GRID[0][3] / E5_GRID[0][4]
    k70 = k_of(70.0, e_cl, v)
    rate0 = E5_GRID[0][4]

    ac = 0.0
    for _ in range(400):
        ac = iv_flow(ac, dur, k70, rate0)
        ac = iv_flow(ac, E5_II - dur, k70, 0.0)
    # Phase `II - lag`, reached by replaying one cycle's active then quiet leg.
    phase = E5_II - lag
    assert phase >= dur, "the seed must land past the infusion window's end"
    ac = iv_flow(ac, dur, k70, rate0)
    ac = iv_flow(ac, phase - dur, k70, 0.0)

    def wt_at(t):
        for (rt, wt, _, _, _) in E5_GRID:
            if rt >= t - 1e-12:
                return wt
        return E5_GRID[-1][1]

    # Infusion windows, opened at each dose's lagged arrival.
    windows = [
        (t + lag, t + lag + amt / rate, rate)
        for (t, _, kind, amt, rate) in E5_GRID
        if kind in ("ss", "dose")
    ]

    def rate_at(t):
        # Half-open [start, end): a boundary instant belongs to the segment after
        # it, matching how the walk's break points are laid out.
        return sum(r for (s, e, r) in windows if s - 1e-12 <= t < e - 1e-12)

    breaks = sorted(
        {t for (t, _, _, _, _) in E5_GRID}
        | {s for (s, _, _) in windows}
        | {e for (_, e, _) in windows}
    )
    obs_at = {t: kind for (t, _, kind, _, _) in E5_GRID}

    out, cur = {}, breaks[0]
    for t in breaks[1:]:
        w = wt_at(t)
        ac = iv_flow(ac, t - cur, k_of(w, e_cl, v), rate_at(cur))
        cur = t
        if obs_at.get(t) == "obs":
            out[t] = ac / v
    return out


def emit_e5(path, seed):
    rng = random.Random(seed)
    rows = ["ID,TIME,DV,AMT,RATE,EVID,CMT,MDV,WT,SS,II"]
    for (i, (e1, e2, e3)) in enumerate(ETAS, start=1):
        lag = 0.7 * math.exp(e3)
        assert 0.5 < lag < E5_II, f"subject {i}: lag={lag:.4f} out of range"
        ip = sim_e5(e1, e2, lag)
        for (n, (t, wt, kind, amt, rate)) in enumerate(E5_GRID):
            ts = "0" if n == 0 else f"{t}"
            if kind == "ss":
                rows.append(
                    f"{i},{ts},.,{amt},{rate},1,1,1,{wt:.0f},1,{E5_II}")
            elif kind == "dose":
                rows.append(f"{i},{ts},.,{amt},{rate},1,1,1,{wt:.0f},0,0")
            else:
                dv = ip[t] * (1.0 + rng.gauss(0.0, PROP_SD))
                rows.append(f"{i},{ts},{dv:.6f},0,0,0,1,0,{wt:.0f},0,0")
    write(path, rows)


if __name__ == "__main__":
    emit_e2("dose_form_lag_infusion.csv", 10732)
    emit_e3("dose_form_lag_ss.csv", 10733)
    emit_e4("dose_form_lag_reset.csv", 10734)
    emit_e3p("dose_form_lag_ss_prearrival.csv", 10735)
    emit_e3b("dose_form_lag_ss_midstream.csv", 10736)
    emit_e5("dose_form_lag_ss_infusion.csv", 10737)
