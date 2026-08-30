#!/usr/bin/env python3
"""Independent deterministic reference for runB (OMEGA 0 FIX).

CL = 1.0, V = 20, lag = 0.5, THETA_TAD = 0.3, doses 100 at t=0 and t=12.
TAD is anchored at the most recent ARRIVAL (t_dose + lag), which is the
convention both engines claim to implement. Prints IPRED = A/V at each
observation time, plus the TAD each engine should be seeing.

Fine fixed-step RK4 -- no adaptive solver, no ferx code, no NONMEM code.
"""
CL, V, LAG, KT = 1.0, 20.0, 0.5, 0.3
DOSES = [0.0, 12.0]
OBS = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 11.5, 14.0, 15.0, 16.0, 18.0, 20.0, 24.0]
ARR = sorted(d + LAG for d in DOSES)


def tad(t):
    past = [a for a in ARR if a <= t + 1e-12]
    return t - max(past) if past else None


def deriv(t, a):
    td = tad(t)
    td = 0.0 if td is None else td
    return -(CL / V) * a * (1.0 + KT * td)


def run():
    h = 1e-5
    t, a = ARR[0], 100.0
    pending = list(ARR[1:])
    stops = sorted(set(OBS) | set(ARR[1:]))
    out = {}
    for tgt in stops:
        while t < tgt - 1e-12:
            step = min(h, tgt - t)
            k1 = deriv(t, a)
            k2 = deriv(t + step / 2, a + step * k1 / 2)
            k3 = deriv(t + step / 2, a + step * k2 / 2)
            k4 = deriv(t + step, a + step * k3)
            a += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += step
        if pending and abs(tgt - pending[0]) < 1e-9:
            a += 100.0
            pending.pop(0)
        if tgt in OBS:
            out[tgt] = a
    return out


res = run()
print(f"arrivals: {ARR}")
print(f"{'TIME':>6} {'TAD':>8} {'IPRED':>18}")
for t in OBS:
    print(f"{t:>6} {tad(t):>8.4f} {res[t] / V:>18.10f}")
