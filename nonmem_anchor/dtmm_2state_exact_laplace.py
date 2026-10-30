#!/usr/bin/env python3
"""Exact Laplace objective of nonmem_anchor/dtmm_2state at NONMEM's final estimates (#2473).

A third computation, outside both engines. The 2-state DTMM is separable: ETA(1) enters
only P(0->1) and ETA(2) only P(1->0), so each subject's empirical Bayes estimate is two
1-D Newton solves on closed-form derivatives of a Bernoulli count likelihood, and the
Laplace Hessian is exact (no finite differences):

    -2 log L_i = 2 f(eta_hat) + log|Omega| + log|H|,   f = NLL + eta' Omega^-1 eta / 2

Measured: total 1049.4079889254; ferx 1049.4077942 (2.0e-4); NONMEM 1049.4031789 (4.8e-3,
concentrated on subjects 48 and 11 — two awake steps or fewer under omega_AS = 0.063, where
NONMEM's numerical second derivatives are least accurate). Stdlib only:

    python3 dtmm_2state_exact_laplace.py .      # from nonmem_anchor/
"""

import math
import sys

d = sys.argv[1]
A, B, C = -1.0065933892463419, 0.88239637849927277, -0.64393697409745099
W1, W2 = 0.062830339797752571, 0.31079088071690125
subj = {}
for i, line in enumerate(open(d + "/dtmm_2state.csv")):
    if i == 0: continue
    ID, T, DV, MDV, CMT, DOSE, PDV = line.strip().split(",")
    s = subj.setdefault(ID, {"dose": float(DOSE), "chain": []})
    if MDV == "0": s["chain"].append(int(DV))
il = lambda x: 1 / (1 + math.exp(-x))
def solve(n_move, n_stay, base, w):
    e = 0.0
    for _ in range(100):
        p = il(base + e); n = n_move + n_stay
        g = n * p - n_move + e / w
        h = n * p * (1 - p) + 1 / w
        e -= g / h
    p = il(base + e)
    f = -(n_move * math.log(p) if n_move else 0) - (n_stay * math.log(1 - p) if n_stay else 0) + e * e / (2 * w)
    return e, 2 * f + math.log(w) + math.log(h)
nm = {}
for i, line in enumerate(open(d + "/results/dtmm_2state.phi")):
    if i < 2: continue
    t = line.split(); nm[t[1]] = (float(t[2]), float(t[3]), float(t[-1]))
tot = 0; worst = 0
for ID, s in subj.items():
    c = s["chain"]; n = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for a, b in zip(c, c[1:]): n[(a, b)] += 1
    e1, o1 = solve(n[(0, 1)], n[(0, 0)], A + B * s["dose"], W1)
    e2, o2 = solve(n[(1, 0)], n[(1, 1)], C, W2)
    o = o1 + o2; tot += o
    worst = max(worst, abs(o - nm[ID][2]))
    if ID in ("1", "2"): print(ID, e1, nm[ID][0], e2, nm[ID][1], o, nm[ID][2])
print("total", tot, "worst per-subject vs NONMEM", worst)
print("sum NONMEM phi OBJ", sum(v[2] for v in nm.values()))
rows = []
for ID, s in subj.items():
    c = s["chain"]; n = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for a, b in zip(c, c[1:]): n[(a, b)] += 1
    o = solve(n[(0,1)], n[(0,0)], A + B*s["dose"], W1)[1] + solve(n[(1,0)], n[(1,1)], C, W2)[1]
    rows.append((abs(o - nm[ID][2]), ID, n, o, nm[ID][2]))
for r in sorted(rows, reverse=True)[:4]: print(r)
