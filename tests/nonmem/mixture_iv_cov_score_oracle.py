"""Third-way score oracle for the mixture S / RSR anchor (#1956).

Computes the FOCEI per-subject objective of `mixture_iv_cov_{s,rsr,r0}.ctl` in closed
form (1-cpt IV bolus, proportional error, Omega 0.09 / Sigma 0.04 FIX), outside both
ferx and NONMEM, and forms S = sum_i g_i g_i' from central-difference per-subject
gradients under three score conventions:

  reopt        eta-hat re-estimated at every stencil point (the total derivative);
  eta_fixed    eta-hat held at its value at x-hat, prediction Jacobian recomputed;
  eta_g_fixed  eta-hat and the prediction Jacobian both held (ferx's fixed-mode score).

`--single` runs the one-population twin (CL = TVCL*exp(ETA), V = TVV) on the same data
at its MLE, to show the convention is not mixture-specific.

    uv run --with numpy python tests/nonmem/mixture_iv_cov_score_oracle.py
    uv run --with numpy python tests/nonmem/mixture_iv_cov_score_oracle.py --single

The FOCEI conditional Hessian carries the interaction variance term
2*sum (df/deta / f)^2 of a proportional error; with it the OFV at NONMEM's point is
301.582018 (NONMEM 301.58201838), without it 299.49.
"""

import csv
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("mixture_iv.csv")
OMEGA, SIGMA, DOSE = 0.09, 0.04, 100.0
# mixture_iv_cov.ext finals, as printed: TVCL1, TVCL2, V, P1.
X_MIX = np.array([0.980833, 2.84198, 9.97859, 0.471970])
# One-population MLE on the same data (this script, Nelder-Mead), 6 sig. fig.
X_SINGLE = np.array([1.76783, 9.77524])

SUBJECTS = {}
with open(DATA) as fh:
    for row in csv.DictReader(fh):
        if row["EVID"] == "0":
            SUBJECTS.setdefault(row["ID"], []).append((float(row["TIME"]), float(row["DV"])))
SUBJECTS = [(np.array([t for t, _ in v]), np.array([y for _, y in v])) for v in SUBJECTS.values()]


def pred(tvcl, v, eta, t):
    cl = tvcl * np.exp(eta)
    f = DOSE / v * np.exp(-cl / v * t)
    return f, f * (-cl / v * t)


def conditional(eta, tvcl, v, t, y):
    f, _ = pred(tvcl, v, eta, t)
    var = f * f * SIGMA
    return np.sum(np.log(var) + (y - f) ** 2 / var) + eta * eta / OMEGA


def ebe(tvcl, v, t, y):
    grid = np.linspace(-2.0, 2.0, 401)
    e = grid[np.argmin([conditional(g, tvcl, v, t, y) for g in grid])]
    for _ in range(100):
        h = 1e-5
        lo, mid, hi = (conditional(e + d, tvcl, v, t, y) for d in (-h, 0.0, h))
        d1, d2 = (hi - lo) / (2 * h), (hi - 2 * mid + lo) / h**2
        step = d1 / d2 if d2 > 0 else 0.01 * np.sign(d1)
        e -= step
        if abs(step) < 1e-13:
            break
    return e


def m2ll(tvcl, v, t, y, eta, g_fixed=None):
    """FOCEI -2LL_ik (no 2*pi): conditional at eta + log omega + log Htilde."""
    f, g = pred(tvcl, v, eta, t)
    if g_fixed is not None:
        g = g_fixed
    var = f * f * SIGMA
    htil = 1.0 / OMEGA + np.sum(g * g / var) + 2.0 * np.sum((g / f) ** 2)
    return np.sum(np.log(var) + (y - f) ** 2 / var) + eta * eta / OMEGA + np.log(OMEGA) + np.log(htil)


def grad(fun, x, h_rel=1e-5):
    g = np.zeros(len(x))
    for j in range(len(x)):
        h = h_rel * max(abs(x[j]), 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        g[j] = (fun(xp) - fun(xm)) / (2 * h)
    return g


def report(x0, subj_nll, conventions):
    total = lambda x: sum(subj_nll(x, i, "reopt") for i in range(len(SUBJECTS)))
    print("OFV (no 2*pi):", 2 * total(x0))
    n = len(x0)
    r = np.zeros((n, n))
    for j in range(n):
        h = 1e-4 * abs(x0[j])
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        r[:, j] = (grad(total, xp) - grad(total, xm)) / (2 * h)
    r_inv = np.linalg.inv(0.5 * (r + r.T))
    print(f"{'R':>16} SE:", np.sqrt(np.diag(r_inv)))
    for conv in conventions:
        s = sum(np.outer(gi, gi) for gi in (grad(lambda x: subj_nll(x, i, conv), x0) for i in range(len(SUBJECTS))))
        print(f"{'S ' + conv:>16} SE:", np.sqrt(np.diag(np.linalg.inv(s))))
        print(f"{'RSR ' + conv:>16} SE:", np.sqrt(np.diag(r_inv @ s @ r_inv)))


def mixture():
    cls = lambda x, k: (x[k], x[2])
    modes0 = [[ebe(*cls(X_MIX, k), t, y) for k in range(2)] for t, y in SUBJECTS]
    g0 = [[pred(*cls(X_MIX, k), modes0[i][k], t)[1] for k in range(2)] for i, (t, _) in enumerate(SUBJECTS)]

    def subj_nll(x, i, conv):
        t, y = SUBJECTS[i]
        terms = []
        for k in range(2):
            tvcl, v = cls(x, k)
            if conv == "reopt":
                val = m2ll(tvcl, v, t, y, ebe(tvcl, v, t, y))
            elif conv == "eta_fixed":
                val = m2ll(tvcl, v, t, y, modes0[i][k])
            else:
                val = m2ll(tvcl, v, t, y, modes0[i][k], g_fixed=g0[i][k])
            terms.append(np.log([x[3], 1.0 - x[3]][k]) - 0.5 * val)
        a = np.array(terms)
        return -(a.max() + np.log(np.sum(np.exp(a - a.max()))))

    print("mixture at the mixture_iv_cov.ext finals (TVCL1, TVCL2, V, P1)")
    report(X_MIX, subj_nll, ["reopt", "eta_fixed", "eta_g_fixed"])


def single():
    modes0 = [ebe(*X_SINGLE, t, y) for t, y in SUBJECTS]
    g0 = [pred(*X_SINGLE, modes0[i], t)[1] for i, (t, _) in enumerate(SUBJECTS)]

    def subj_nll(x, i, conv):
        t, y = SUBJECTS[i]
        if conv == "reopt":
            return 0.5 * m2ll(x[0], x[1], t, y, ebe(x[0], x[1], t, y))
        return 0.5 * m2ll(x[0], x[1], t, y, modes0[i], g_fixed=g0[i])

    print("one population at its MLE (TVCL, V)")
    report(X_SINGLE, subj_nll, ["reopt", "eta_g_fixed"])


if __name__ == "__main__":
    single() if "--single" in sys.argv else mixture()
