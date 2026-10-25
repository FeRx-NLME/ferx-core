"""Why the fit of nested_live_rho.csv lands at ρ = 0.648 when the data are drawn at 0.9.

Outside ferx, on the simulated residuals alone (`make_data.simulate()`'s `EPS`, no
structural model): the pooled lag-1 correlation within arms, and the AR(1) maximum
likelihood ρ with, per study, a free intercept (what PLACEBO[STUDYID] can absorb) and
then also a free per-arm amplitude on the time curve TIME / (TIME + ET50) (an
unpenalised K_ARM). ferx's K_ARM has a prior, so its fit falls between the two.

Stdlib only: AR(1) whitening is closed form (Prais–Winsten), so each study is a small
least-squares problem. Prints, as quoted in RHO_LIVE's comment:

    pooled lag-1 0.891
    per-study intercept      rho_hat 0.747
    + per-arm free amplitude rho_hat 0.205
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under `python -I`
from make_data import ET50, simulate  # noqa: E402


def whiten(x, rho):
    """Prais–Winsten: the AR(1) unit-variance correlation's inverse Cholesky applied."""
    return [math.sqrt(1.0 - rho * rho) * x[0]] + [x[t] - rho * x[t - 1] for t in range(1, len(x))]


def lstsq_ss(cols, y):
    """Residual sum of squares of y on cols, by the normal equations (≤ 3 columns)."""
    k = len(cols)
    a = [[sum(ci * cj for ci, cj in zip(cols[i], cols[j])) for j in range(k)] for i in range(k)]
    b = [sum(ci * yi for ci, yi in zip(cols[i], y)) for i in range(k)]
    for p in range(k):  # Gauss–Jordan with partial pivoting
        q = max(range(p, k), key=lambda r: abs(a[r][p]))
        a[p], a[q], b[p], b[q] = a[q], a[p], b[q], b[p]
        for r in range(k):
            if r != p:
                f = a[r][p] / a[p][p]
                a[r] = [u - f * v for u, v in zip(a[r], a[p])]
                b[r] -= f * b[p]
    beta = [b[i] / a[i][i] for i in range(k)]
    return sum((yi - sum(beta[i] * cols[i][t] for i in range(k))) ** 2 for t, yi in enumerate(y))


def neg2ll(rho, studies, amplitude):
    ss, n, logdet = 0.0, 0, 0.0
    for arms in studies.values():
        y, cols = [], [[] for _ in range(1 + (len(arms) if amplitude else 0))]
        for k, rs in enumerate(arms):
            m = len(rs)
            y += whiten([r["EPS"] for r in rs], rho)
            cols[0] += whiten([1.0] * m, rho)
            if amplitude:
                g = whiten([float(r["TIME"]) / (float(r["TIME"]) + ET50) for r in rs], rho)
                for j in range(len(arms)):
                    cols[1 + j] += g if j == k else [0.0] * m
            logdet += (m - 1) * math.log(1.0 - rho * rho)
        ss += lstsq_ss(cols, y)
        n += len(y)
    # The whitened rows have variance (1 − ρ²)σ², so the quadratic form is ss / (1 − ρ²).
    return n * math.log(ss / (n * (1.0 - rho * rho))) + logdet


rows = simulate()
arms, studies = {}, {}
for r in rows:
    arms.setdefault(r["ARMID"], []).append(r)
    studies.setdefault(r["STUDYID"], [])
    if arms[r["ARMID"]] not in studies[r["STUDYID"]]:
        studies[r["STUDYID"]].append(arms[r["ARMID"]])
eps = [r["EPS"] for r in rows]
mean = sum(eps) / len(eps)
var = sum((e - mean) ** 2 for e in eps) / len(eps)
lag = [a[t]["EPS"] * a[t + 1]["EPS"] for a in arms.values() for t in range(len(a) - 1)]
print(f"pooled lag-1 {sum(lag) / len(lag) / var:.3f}")
grid = [-0.5 + 1.47 * i / 296 for i in range(297)]
for amplitude, name in [(False, "per-study intercept     "), (True, "+ per-arm free amplitude")]:
    v = [neg2ll(rho, studies, amplitude) for rho in grid]
    print(f"{name} rho_hat {grid[v.index(min(v))]:.3f}")
