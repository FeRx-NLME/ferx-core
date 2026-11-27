"""The minimum of each subject's individual objective for tests/inner_kink_nonsmooth.rs's ORACLE.
Plain Python 3 (no dependencies; written against 3.12): python3 oracle.py clipped.csv [occ1|iiv]
obj = sum (DV - max(0, BASE*exp(-K*tad) - TH_occ))^2/sigma^2 + eta' Omega^-1 eta + sum kappa^2/omega_kappa,
BASE = TVBASE*exp(eta_1), TH_occ = TVTH*exp(eta_2 + kappa_occ), at clipped.ferx's initial estimates,
minimised by Nelder-Mead from 0 with three simplex sizes, polished three times. occ1: occasion 1
only, without kappa (the no-IOV variant of without_iov_the_kicks_reach_a_second_mode). iiv: both
occasions, without kappa (the no-IOV variant of the stall-escape default test, #3210)."""
import csv
import math
import sys
from collections import defaultdict

TVBASE, K, TVTH, OM, OMK, S2 = 10.0, 0.15, 4.0, (0.04, 0.09), 0.04, 0.15 ** 2


def obj(rows, x):
    th = [TVTH * math.exp(x[1] + sum(x[1 + o:2 + o])) for o in (0, 1, 2)]
    s = sum((dv - max(0.0, TVBASE * math.exp(x[0] - K * (t - t0)) - th[o])) ** 2 for t0, t, o, dv in rows) / S2
    return s + x[0] ** 2 / OM[0] + x[1] ** 2 / OM[1] + sum(v * v for v in x[2:]) / OMK


def nm(f, x0, step, it=4000):
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (step if j == i else 0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(it):
        o = sorted(range(n + 1), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        c = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        r = [2 * c[j] - pts[-1][j] for j in range(n)]
        fr = f(r)
        if fr < vals[0]:
            e = [3 * c[j] - 2 * pts[-1][j] for j in range(n)]
            fe = f(e)
            pts[-1], vals[-1] = (e, fe) if fe < fr else (r, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = r, fr
        else:
            k = [(c[j] + pts[-1][j]) / 2 for j in range(n)]
            fk = f(k)
            if fk < vals[-1]:
                pts[-1], vals[-1] = k, fk
            else:
                pts = [pts[0]] + [[(pts[0][j] + p[j]) / 2 for j in range(n)] for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
        if max(vals) - min(vals) < 1e-13:
            break
    i = min(range(n + 1), key=lambda i: vals[i])
    return vals[i], pts[i]


occ1 = sys.argv[2:] == ['occ1']
no_iov = occ1 or sys.argv[2:] == ['iiv']
data, t0s = defaultdict(list), {}
for r in csv.DictReader(open(sys.argv[1])):
    if occ1 and r['OCC'] != '1':
        continue
    key = (r['ID'], int(r['OCC']))
    if r['EVID'] == '4':
        t0s[key] = float(r['TIME'])
    else:
        data[r['ID']].append((t0s[key], float(r['TIME']), int(r['OCC']), float(r['DV'])))
for i, rows in data.items():
    def f(y, rows=rows):
        return obj(rows, y)
    best = min((nm(f, [0.0] * (2 if no_iov else 4), s) for s in (0.05, 0.2, 0.3)), key=lambda b: b[0])
    for _ in range(3):
        best = min(best, nm(f, best[1], 0.02), key=lambda b: b[0])
    print('    ("%s", [%s]),' % (i, ', '.join('%.6f' % v for v in best[1])))
