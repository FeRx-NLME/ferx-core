"""Closed form for nonmem_anchor/ss_arrival_flow.csv (#1275), outside both engines.

1-cpt oral, CL=2, V=20, KA=0.15; SS=1 depot dose AMT 100 at t=10, II=12, ALAG1=L on the
depot only. The pre-record periodic train is placed under the #1121 clamp: its last pulse
lands at 10 - max(II - L, 0). ID 4's depot dose at 0 is wiped by the SS=1 reset.
ID 5's SS dose is a depot INFUSION, RATE 50 (T_inf = 2 h).

Run from the repo root: python3 nonmem_anchor/ss_arrival_flow_closed_form.py
Measured: NONMEM ADVAN13 TOL=12 matches this to <= 7.1e-12 on every cell except ID 5 at
lag 13, where NONMEM infuses the record-time cycle for T_inf + (lag - II) = 3 h (matches
that reading to 4.0e-12) -- the documented lag > II infusion divergence, not #1275.
"""
import math, csv, sys

CL, V, KA, II = 2.0, 20.0, 0.15, 12.0
k = CL / V


def oral(t, a):
    return 0.0 if t < 0 else a * KA / (V * (KA - k)) * (math.exp(-k * t) - math.exp(-KA * t))


def oral_ramp(t, r):
    """Central concentration from a constant depot input r starting at 0 and never ending."""
    if t <= 0:
        return 0.0
    return r * KA / (V * (KA - k)) * ((1 - math.exp(-k * t)) / k - (1 - math.exp(-KA * t)) / KA)


def oral_inf(t, rate, tinf):
    return oral_ramp(t, rate) - oral_ramp(t - tinf, rate)


def iv(t, a):
    return 0.0 if t < 0 else a / V * math.exp(-k * t)


def conc(i, t, L):
    rec = 10.0
    p = max(II - L, 0.0)
    pulse = (lambda s: oral_inf(s, 50.0, 2.0)) if i == 5 else (lambda s: oral(s, 100.0))
    c = 0.0
    for n in range(0, 4000):
        c += pulse(t - (rec - p - n * II))
    c += pulse(t - (rec + L))
    if i in (1, 4):
        c += iv(t - 11, 100)
    if i == 3:
        c += iv(t - 15, 50)
    if i == 5:
        c += iv(t - 14, 100)
    return c


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "nonmem_anchor"
    for L in (11, 12, 13):
        worst = {}
        path = f"{root}/results/ss_arrival_flow_lag{L}.sdtab"
        for r in csv.reader(open(path), delimiter=" ", skipinitialspace=True):
            if not r or r[0] == "ID" or r[0].startswith("TABLE"):
                continue
            i, t, ev, pred = int(float(r[0])), float(r[1]), int(float(r[2])), float(r[3])
            if ev != 0:
                continue
            o = conc(i, t, L)
            rel = abs(pred - o) / o
            worst[i] = max(worst.get(i, 0.0), rel)
            if rel > 1e-6:
                print(f"MISMATCH lag {L} ID {i} t={t}: NONMEM {pred} vs closed form {o} (rel {rel:.3e})")
        print(f"lag {L}: worst NONMEM vs closed form per ID", {i: f"{w:.2e}" for i, w in worst.items()})
