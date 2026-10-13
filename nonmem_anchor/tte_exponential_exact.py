#!/usr/bin/env python3
"""Exact marginal likelihood of examples/tte_exponential.ferx, outside ferx and NONMEM (#1781).

Brute-force quadrature over eta, Newton to the minimum, then R^-1, S^-1 and R^-1 S R^-1 standard
errors there. Usage: python3 tte_exponential_exact.py data/tte_exponential.csv

With `--at LAMBDA OMEGA2` (repeatable) it also evaluates R^-1, S^-1 and R^-1 S R^-1 at those
fixed points, without moving them — the same-point anchor
src/estimation/covariance.rs::the_sampler_points_match_the_exact_marginal_covariance reads.
At a point that is not the minimum, R is the Hessian of the OFV in (log lambda, log omega2),
which is a linear map of ferx's packed (log lambda, log omega), so the two agree there too.
"""
import math, csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))]
points = []
_a = sys.argv[2:]
while _a:
    assert _a[0] == "--at", _a
    points.append((float(_a[1]), float(_a[2])))
    _a = _a[3:]
data = [(float(r["TIME"]), int(float(r["DV"]))) for r in rows if r["EVID"] == "0"]
K = 4000
zs = [-12 + 24 * k / K for k in range(K + 1)]
w = [(1 if k in (0, K) else (4 if k % 2 else 2)) * (24 / K) / 3 for k in range(K + 1)]
phi = [math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) for z in zs]
def subj_L(lam, om2, t, d):
    om = math.sqrt(om2); s = 0.0
    for z, wk, p in zip(zs, w, phi):
        li = lam * math.exp(om * z)
        s += wk * p * (li ** d) * math.exp(-li * t)
    return s
def ofv(lam, om2):
    return -2 * sum(math.log(subj_L(lam, om2, t, d)) for t, d in data)
def ofv_x(x):  # x = (log λ, log ω²)
    return ofv(math.exp(x[0]), math.exp(x[1]))
# Newton with FD derivatives in x
x = [math.log(0.0337), math.log(0.08)]
def grad_hess(f, x, h=1e-4):
    n = len(x); f0 = f(x); g = [0]*n; H = [[0]*n for _ in range(n)]
    for i in range(n):
        xp = x[:]; xp[i] += h; xm = x[:]; xm[i] -= h
        fp, fm = f(xp), f(xm); g[i] = (fp - fm) / (2*h); H[i][i] = (fp - 2*f0 + fm) / h**2
    for i in range(n):
        for j in range(i+1, n):
            pp = x[:]; pp[i]+=h; pp[j]+=h; pm = x[:]; pm[i]+=h; pm[j]-=h
            mp = x[:]; mp[i]-=h; mp[j]+=h; mm = x[:]; mm[i]-=h; mm[j]-=h
            H[i][j] = H[j][i] = (f(pp)-f(pm)-f(mp)+f(mm))/(4*h*h)
    return f0, g, H
for it in range(30):
    f0, g, H = grad_hess(ofv_x, x)
    det = H[0][0]*H[1][1]-H[0][1]**2
    dx = [-( H[1][1]*g[0]-H[0][1]*g[1])/det, -(-H[1][0]*g[0]+H[0][0]*g[1])/det]
    x = [x[0]+dx[0], x[1]+dx[1]]
    if max(abs(v) for v in dx) < 1e-9: break
lam, om2 = math.exp(x[0]), math.exp(x[1])
f0, g, H = grad_hess(ofv_x, x)
print(f"MLE: lambda={lam:.6f} omega2={om2:.6f} OFV={f0:.4f} grad={g}")
# Covariance of x = 2 H^-1 (H is the OFV Hessian = 2·(−logL) Hessian); delta to natural.
det = H[0][0]*H[1][1]-H[0][1]**2
Ci = [[2*H[1][1]/det, -2*H[0][1]/det], [-2*H[1][0]/det, 2*H[0][0]/det]]
print(f"R-inv SE: SE(lambda)={lam*math.sqrt(Ci[0][0]):.6f} SE(omega2)={om2*math.sqrt(Ci[1][1]):.6f}")
for (l, o, tag) in [(0.038756, 0.054346, "ferx SAEM"), (0.033848, 0.081064, "ferx IMP"), (0.033591, 0.082876, "ferx IMPMAP"), (0.0337025, 0.0904614, "NONMEM IMP"), (0.033183, 0.291718, "FOCEI min")]:
    print(f"OFV_exact at {tag} ({l}, {o}) = {ofv(l, o):.4f}")
# Per-subject scores of −logL_i in x = (log λ, log ω²), central FD.
def nll_i(x, t, d):
    return -math.log(subj_L(math.exp(x[0]), math.exp(x[1]), t, d))
def inv2(M):
    dt = M[0][0]*M[1][1]-M[0][1]*M[1][0]
    return [[M[1][1]/dt, -M[0][1]/dt], [-M[1][0]/dt, M[0][0]/dt]]
def mul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def report(x, label):
    lam, om2 = math.exp(x[0]), math.exp(x[1])
    _, _, H = grad_hess(ofv_x, x)
    S = [[0.0, 0.0], [0.0, 0.0]]
    hs = 1e-5
    for t, d in data:
        g = []
        for k in range(2):
            xp = x[:]; xp[k] += hs; xm = x[:]; xm[k] -= hs
            g.append((nll_i(xp, t, d) - nll_i(xm, t, d)) / (2 * hs))
        for a in range(2):
            for b in range(2):
                S[a][b] += g[a] * g[b]
    Rm = [[H[i][j] / 2 for j in range(2)] for i in range(2)]   # R = Hessian of −logL
    Ri = inv2(Rm); Si = inv2(S); RSR = mul(mul(Ri, S), Ri)
    print(f"{label} (lambda={lam}, omega2={om2}):")
    for tag, C in [("R", Ri), ("S", Si), ("RSR", RSR)]:
        print(f"  {tag:>3}: SE(lambda)={lam*math.sqrt(C[0][0]):.7f} SE(omega2)={om2*math.sqrt(C[1][1]):.7f}")
report(x, "MLE")
for l, o in points:
    report([math.log(l), math.log(o)], "point")
