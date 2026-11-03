#!/usr/bin/env python3
"""Generate the #1978 reset-clock fixtures: what TIME reads after an EVID=3/4 restart.

Three endpoints, each on two-period data where period 2 restarts TIME at 0 on a reset row:

  reset_clock_ode  IV bolus 100 at 0/24/48 per period, CL x (1 + SLP*T) (in $DES / [odes])
  reset_clock_cat  binary endpoint, logit = TH0 + THT*TIME + eta (in $ERROR / [binary_model])
  reset_clock_tte  joint PK-TTE, hazard H0*exp(GAM*T)*exp(BETA*C), one TTE record in period 2

For each, two CSVs:

  <name>.csv          TIME as written: period 2 restarts at 0. NONMEM's and ferx's input.
  <name>_shifted.csv  period 2 moved past period 1 exactly as ferx's reader moves it
                      (`max_eff_time + RESET_SEGMENT_GAP - time`, gap 1.0, src/io/datareader.rs).
                      NONMEM run on this file is what "the clock keeps running" would score.

The data are simulated with TIME as written (the clock restarts). That choice only shapes the
data; the measurement is what each engine scores on it, and the `_shifted` arm is scored on the
same rows. Stdlib only, seeded, so the files regenerate byte for byte:

    python3 make_reset_clock_fixtures.py      # from nonmem_anchor/
"""

import math
import random

GAP = 1.0  # RESET_SEGMENT_GAP in src/io/datareader.rs


def fmt(x):
    return "." if x is None else ("%d" % x if float(x).is_integer() else "%.6g" % x)


def write(name, header, rows):
    with open(name, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def shifted(rows, period_of, time_col=1):
    """Shift every period-2 row past its subject's period 1, the way ferx's reader does."""
    out = []
    last = {}
    for r, p in zip(rows, period_of):
        sid = r[0]
        if p == 1:
            last[sid] = max(last.get(sid, 0.0), r[time_col])
            out.append(list(r))
        else:
            q = list(r)
            q[time_col] = r[time_col] + last[sid] + GAP
            out.append(q)
    return out


# --- reset_clock_ode -----------------------------------------------------------------------
# ID TIME DV AMT EVID CMT MDV. Doses at 0/24/48 per period; observations never co-timed with
# a dose, so #1969's one-ULP move (a trough listed before its dose) is not in play here.
def make_ode():
    rng = random.Random(1978)
    cl, v, slp = 2.0, 20.0, 0.02
    om_cl, om_v, prop = 0.09, 0.04, 0.10
    obs_t = [1, 6, 12, 23, 25, 30, 36, 47, 49, 60, 72]
    dose_t = [0, 24, 48]
    rows, per = [], []
    for sid in range(1, 21):
        cli = cl * math.exp(rng.gauss(0, math.sqrt(om_cl)))
        vi = v * math.exp(rng.gauss(0, math.sqrt(om_v)))
        for period in (1, 2):
            # amount at t: superpose doses; elimination rate k1*(1 + slp*t), t as written
            k1 = cli / vi

            def amount(t):
                a = 0.0
                for d in dose_t:
                    if d <= t:
                        integ = k1 * ((t - d) + slp * (t * t - d * d) / 2)
                        a += 100.0 * math.exp(-integ)
                return a

            events = [(t, "dose") for t in dose_t] + [(t, "obs") for t in obs_t]
            events.sort(key=lambda e: (e[0], e[1] == "obs"))
            for t, kind in events:
                if kind == "dose":
                    evid = 4 if (period == 2 and t == 0) else 1
                    rows.append([sid, t, None, 100, evid, 1, 1])
                else:
                    c = amount(t) / vi
                    dv = c * (1 + rng.gauss(0, prop))
                    rows.append([sid, t, round(dv, 6), None, 0, 1, 0])
                per.append(period)
    hdr = ["ID", "TIME", "DV", "AMT", "EVID", "CMT", "MDV"]
    write("reset_clock_ode.csv", hdr, rows)
    write("reset_clock_ode_shifted.csv", hdr, shifted(rows, per))


# --- reset_clock_cat -----------------------------------------------------------------------
# ID TIME DV AMT EVID CMT MDV. No doses: an EVID=3 row opens period 2. The binary endpoint
# sits on CMT 1 so NONMEM's ADVAN1 accepts the observation records unchanged.
def make_cat():
    rng = random.Random(19782)
    th0, tht, om = -1.0, 0.03, 0.5
    times = [2, 10, 20, 30, 45, 60]
    rows, per = [], []
    for sid in range(1, 51):
        eta = rng.gauss(0, math.sqrt(om))
        for period in (1, 2):
            if period == 2:
                rows.append([sid, 0, None, None, 3, 1, 1])
                per.append(2)
            for t in times:
                p = 1 / (1 + math.exp(-(th0 + tht * t + eta)))
                rows.append([sid, t, 1 if rng.random() < p else 0, None, 0, 1, 0])
                per.append(period)
    hdr = ["ID", "TIME", "DV", "AMT", "EVID", "CMT", "MDV"]
    write("reset_clock_cat.csv", hdr, rows)
    write("reset_clock_cat_shifted.csv", hdr, shifted(rows, per))


# --- reset_clock_tte -----------------------------------------------------------------------
# ID TIME DV AMT EVID CMT MDV. Period 1: PK only, doses 0/24. Period 2: EVID=4 dose at 0,
# PK observations, and one TTE record on CMT 2 (exact event, or censored at 22). EVID=4
# zeroes the cumulative hazard with the rest of the state, so period 2's survival runs from
# its own restart.
def make_tte():
    rng = random.Random(19783)
    cl, v, h0, beta, gam = 1.0, 10.0, 0.02, 0.15, 0.05
    om_cl, prop = 0.09, 0.10
    p1_obs, p1_dose = [1, 6, 12, 23, 25, 30, 36, 47], [0, 24]
    p2_obs, cens = [1, 6, 12, 23], 22.0
    rows, per = [], []
    for sid in range(1, 41):
        k = cl * math.exp(rng.gauss(0, math.sqrt(om_cl))) / v

        def conc(t, doses):
            return sum(100.0 / v * math.exp(-k * (t - d)) for d in doses if d <= t)

        p1 = [(t, "dose") for t in p1_dose] + [(t, "obs") for t in p1_obs]
        p1.sort(key=lambda e: (e[0], e[1] == "obs"))
        for t, kind in p1:
            if kind == "dose":
                rows.append([sid, t, None, 100, 1, 1, 1])
            else:
                c = conc(t, p1_dose)
                rows.append([sid, t, round(c * (1 + rng.gauss(0, prop)), 6), None, 0, 1, 0])
            per.append(1)

        # period 2: draw the event time by inverting the cumulative hazard on a fine grid
        target = -math.log(rng.random())
        chz, t, dt, te = 0.0, 0.0, 1e-3, None
        while t < cens:
            hm = h0 * math.exp(gam * (t + dt / 2)) * math.exp(beta * conc(t + dt / 2, [0]))
            if chz + hm * dt >= target:
                te = t + (target - chz) / hm
                break
            chz += hm * dt
            t += dt
        rec = [(0, "dose")] + [(x, "obs") for x in p2_obs]
        rec.append((round(te, 4), "event") if te is not None else (cens, "cens"))
        rec.sort(key=lambda e: (e[0], e[1] != "dose"))
        for t, kind in rec:
            if kind == "dose":
                rows.append([sid, t, None, 100, 4, 1, 1])
            elif kind == "obs":
                c = conc(t, [0])
                rows.append([sid, t, round(c * (1 + rng.gauss(0, prop)), 6), None, 0, 1, 0])
            else:
                rows.append([sid, t, 1 if kind == "event" else 0, None, 0, 2, 0])
            per.append(2)
    hdr = ["ID", "TIME", "DV", "AMT", "EVID", "CMT", "MDV"]
    write("reset_clock_tte.csv", hdr, rows)
    write("reset_clock_tte_shifted.csv", hdr, shifted(rows, per))


if __name__ == "__main__":
    make_ode()
    make_cat()
    make_tte()
