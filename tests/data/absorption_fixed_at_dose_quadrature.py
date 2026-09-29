#!/usr/bin/env python3
"""Independent reference for #1569: 2-cpt transit and inverse-Gaussian absorption under IOV
and a time-varying covariate, where doses are still absorbing when their parameters change.

Pure standard library (math) -- no numpy/scipy, no ferx code, no NONMEM code.
Deterministic: re-running reproduces the CSV byte-for-byte.

    python3 tests/data/absorption_fixed_at_dose_quadrature.py \
        > tests/data/absorption_fixed_at_dose_quadrature.csv

Model (the ODE twin of `pk two_cpt_transit(cl, v1, q, v2, n, mtt, f, lagtime)` or
`pk two_cpt_ig(cl, v1, q, v2, mat, cv2, f, lagtime)`):

    d/dt(central) = sum_d F_d * D_d * R_d(t - a_d) - (CL/V1 + Q/V1) * central + (Q/V2) * periph
    d/dt(periph)  = (Q/V1) * central - (Q/V2) * periph
    a_d           = t_d + ALAG_d                          (arrival of dose d)
    IPRED         = central / V1
    transit: R(x) = KTR (KTR x)^N exp(-KTR x) / Gamma(N + 1),  KTR = (N + 1)/MTT
    igd:     R(x) = sqrt(MAT / (2 pi CV2 x^3)) exp(-(x - MAT)^2 / (2 CV2 MAT x))

Individual parameters at a record (occasion `o`, covariate `WT`):

    CL = TVCL * (WT/70)^0.75 * exp(KCL[o])     V1 = TVV1 * exp(ETA_V)    Q = TVQ    V2 = TVV2
    kernel (transit):  N   = TVN   * exp(KA[o]),   MTT = TVMTT * exp(KB[o])
    kernel (igd):      MAT = TVMAT * exp(KA[o]),   CV2 = TVCV2 * exp(KB[o])
    F  = TVF * exp(KF[o])                      ALAG = TVLAG

Record conventions -- the ones ferx's event-driven ODE engine implements (#1073):
  * a segment is governed by the record that TERMINATES it: itself for a record, the next
    record ahead for a lagged dose arrival (not a record);
  * F, ALAG and the arrival are the dose record's own.

Two rules for an in-flight dose's absorption kernel, both tabulated:
  * fixed_at_dose    -- each dose keeps the kernel of its own dose record (#1569);
  * current_interval -- every open dose uses the governing record's kernel (ferx before
    #1569, and NONMEM's $DES when it reads the current MTT/NN).
The disposition (CL, V1, Q, V2) is the governing record's under both.

Method: on each segment [s0, s1] the disposition matrix K is constant, so
    x(s1) = expm(K*D) x(s0) + sum_d F_d D_d * integral_{max(s0,a_d)}^{s1} expm(K*(s1-tau)) e_c R_d(tau - a_d) dtau
with expm from the 2x2 spectral form and each scalar integral by tanh-sinh quadrature
(which absorbs the endpoint behaviour at an arrival).

Self-checks (the script exits non-zero if either fails):
  * an independent fixed-step RK4 of the same ODE, with a third state accumulating the
    eliminated mass, agrees with the spectral solution at every observation;
  * the RK4 mass ledger central + periph + eliminated == sum(F_d * D_d) at t = 400
    under fixed_at_dose (it is not under current_interval: that rule creates/destroys drug).

Scenarios, per kernel:
  * full    -- IOV on CL, both kernel parameters and F; WT on CL changing mid-occasion;
               the two rules differ.
  * cl_only -- IOV on CL only (the kernel and F carry no IOV); the two rules coincide.
"""
import math
import sys

# ---- population parameters ------------------------------------------------------------
TVCL, TVV1, TVQ, TVV2 = 4.0, 30.0, 3.0, 60.0
TVF, TVLAG = 0.8, 0.7
ETA_V = 0.1
KERNELS = {
    # name: (TVA, TVB) -- transit (N, MTT), igd (MAT, CV2)
    "transit": (3.0, 2.5),
    "ig": (2.5, 0.5),
}

FULL = {
    "KCL": [0.10, -0.15, 0.20],
    "KA": [0.00, 0.25, -0.20],
    "KB": [0.20, -0.30, 0.35],
    "KF": [0.00, -0.10, 0.10],
}
CL_ONLY = {
    "KCL": FULL["KCL"],
    "KA": [0.0, 0.0, 0.0],
    "KB": [0.0, 0.0, 0.0],
    "KF": [0.0, 0.0, 0.0],
}

# ---- design ---------------------------------------------------------------------------
# Doses: (time, amt, occasion, WT). Occasion 2 begins AT dose 3's record (t = 12);
# occasion 3 begins at the observation record t = 16.5, while dose 3 is still absorbing.
DOSES = [
    (0.0, 100.0, 1, 70.0),
    (6.0, 100.0, 1, 72.0),
    (12.0, 100.0, 2, 74.0),
    (18.0, 100.0, 3, 80.0),
]
# Observations: (time, occasion, WT). WT moves mid-occasion.
OBS = [
    (1.0, 1, 70.0),
    (2.5, 1, 70.0),
    (4.0, 1, 71.0),
    (5.5, 1, 72.0),
    (7.5, 1, 72.0),
    (9.0, 1, 73.0),
    (11.0, 1, 73.0),
    (13.5, 2, 74.0),
    (15.0, 2, 76.0),
    (16.5, 3, 77.0),
    (19.5, 3, 80.0),
    (22.0, 3, 82.0),
    (26.0, 3, 84.0),
    (36.0, 3, 86.0),
    (48.0, 3, 88.0),
]
T_LEDGER = 400.0


def params(kernel, kappa, occ, wt):
    o = occ - 1
    tva, tvb = KERNELS[kernel]
    return {
        "CL": TVCL * (wt / 70.0) ** 0.75 * math.exp(kappa["KCL"][o]),
        "V1": TVV1 * math.exp(ETA_V),
        "Q": TVQ,
        "V2": TVV2,
        "A": tva * math.exp(kappa["KA"][o]),
        "B": tvb * math.exp(kappa["KB"][o]),
        "F": TVF * math.exp(kappa["KF"][o]),
        "ALAG": TVLAG,
    }


def rate(kernel, a, b, x):
    """Input density R(x) of one dose's kernel (a, b); 0 for x <= 0."""
    if x <= 0.0:
        return 0.0
    if kernel == "transit":
        n, mtt = a, b
        ktr = (n + 1.0) / mtt
        return math.exp(math.log(ktr) + n * math.log(ktr * x) - ktr * x - math.lgamma(n + 1.0))
    mat, cv2 = a, b
    return math.sqrt(mat / (2.0 * math.pi * cv2 * x**3)) * math.exp(
        -((x - mat) ** 2) / (2.0 * cv2 * mat * x)
    )


# ---- tanh-sinh quadrature -------------------------------------------------------------
def tanh_sinh(f, a, b, tol=1e-15):
    """Integral of f over [a, b]; robust to integrable endpoint singularities.

    Nodes c +/- d*tanh(u), u = (pi/2) sinh(k h); the distance to each endpoint,
    d * (1 - tanh u) = d * e^{-u} / cosh u, is formed directly so nodes crowding an
    endpoint keep full precision. Truncated at u = 20 (weight ~1e-16 of the total).
    """
    if b <= a:
        return 0.0
    c, d = 0.5 * (a + b), 0.5 * (b - a)
    h = 0.5
    prev = None
    for _level in range(10):
        total = 0.0
        k = 0
        while True:
            t = k * h
            u = 0.5 * math.pi * math.sinh(t)
            if u > 20.0:
                break
            cu = math.cosh(u)
            w = 0.5 * math.pi * math.cosh(t) / (cu * cu)
            dist = d / (math.exp(u) * cu)
            if k == 0:
                total += w * f(c)
            else:
                total += w * (f(b - dist) + f(a + dist))
            k += 1
        est = total * h * d
        if prev is not None and abs(est - prev) <= tol * max(1.0, abs(est)):
            return est
        prev = est
        h *= 0.5
    return prev


# ---- 2-cpt spectral propagator ----------------------------------------------------------
def disposition(p):
    k10, k12, k21 = p["CL"] / p["V1"], p["Q"] / p["V1"], p["Q"] / p["V2"]
    K = [[-(k10 + k12), k21], [k12, -k21]]
    tr = K[0][0] + K[1][1]
    det = K[0][0] * K[1][1] - K[0][1] * K[1][0]
    disc = math.sqrt(tr * tr / 4.0 - det)
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc  # l1 = -beta (slow), l2 = -alpha (fast)
    return K, l1, l2


def expm_apply(K, l1, l2, t, x):
    """expm(K*t) @ x for a 2x2 K with distinct real eigenvalues l1, l2."""
    e1, e2 = math.exp(l1 * t), math.exp(l2 * t)
    # expm = [e1 (K - l2 I) - e2 (K - l1 I)] / (l1 - l2)
    a = [[K[0][0] - l2, K[0][1]], [K[1][0], K[1][1] - l2]]
    b = [[K[0][0] - l1, K[0][1]], [K[1][0], K[1][1] - l1]]
    out = []
    for i in range(2):
        s = 0.0
        for j in range(2):
            s += (e1 * a[i][j] - e2 * b[i][j]) * x[j]
        out.append(s / (l1 - l2))
    return out


# ---- timeline with ferx's record conventions ---------------------------------------------
DOSE_RECORD, ARRIVAL, OBSERVATION = 1, 2, 4  # ferx's kind_order ranks


def timeline(kernel, kappa):
    """(time, rank, kind, index, snapshot) sorted as ferx sorts its event timeline, and the
    governing record of the segment ending at each entry."""
    ev = []
    for k, (t, _amt, occ, wt) in enumerate(DOSES):
        p = params(kernel, kappa, occ, wt)
        ev.append((t, DOSE_RECORD, "dose", k, p))
        ev.append((t + p["ALAG"], ARRIVAL, "arrival", k, None))
    for j, (t, occ, wt) in enumerate(OBS):
        ev.append((t, OBSERVATION, "obs", j, params(kernel, kappa, occ, wt)))
    ev.sort(key=lambda e: (e[0], e[1]))
    # Governing record of the segment ending at entry i: itself if a record, else the
    # next record ahead (#1073).
    gov = [None] * len(ev)
    nxt = None
    for i in range(len(ev) - 1, -1, -1):
        if ev[i][4] is not None:
            nxt = ev[i][4]
        gov[i] = nxt
    return ev, gov


def dose_state(kernel, kappa):
    """Per dose: (arrival, F*amt, kernel (a, b) at its own record)."""
    out = []
    for (t, amt, occ, wt) in DOSES:
        p = params(kernel, kappa, occ, wt)
        out.append((t + p["ALAG"], p["F"] * amt, (p["A"], p["B"])))
    return out


def spectral(kernel, kappa, rule):
    ev, gov = timeline(kernel, kappa)
    doses = dose_state(kernel, kappa)
    x = [0.0, 0.0]
    cur = ev[0][0]
    arrived = [False] * len(DOSES)
    ipred = [None] * len(OBS)
    for i, (t, _rank, kind, idx, _snap) in enumerate(ev):
        if t > cur:
            p = gov[i]
            K, l1, l2 = disposition(p)
            x = expm_apply(K, l1, l2, t - cur, x)
            for k, (a, mass, own) in enumerate(doses):
                if not arrived[k]:
                    continue
                ka, kb = own if rule == "fixed_at_dose" else (p["A"], p["B"])
                lo = max(cur, a)
                i1 = tanh_sinh(
                    lambda tau: math.exp(l1 * (t - tau)) * rate(kernel, ka, kb, tau - a), lo, t
                )
                i2 = tanh_sinh(
                    lambda tau: math.exp(l2 * (t - tau)) * rate(kernel, ka, kb, tau - a), lo, t
                )
                # expm(K*s) e_c = [e^{l1 s} (K - l2 I) e_c - e^{l2 s} (K - l1 I) e_c] / (l1 - l2)
                for r in range(2):
                    c1 = K[r][0] - (l2 if r == 0 else 0.0)
                    c2 = K[r][0] - (l1 if r == 0 else 0.0)
                    x[r] += mass * (c1 * i1 - c2 * i2) / (l1 - l2)
            cur = t
        if kind == "arrival":
            arrived[idx] = True
        elif kind == "obs":
            ipred[idx] = x[0] / gov[i]["V1"]
    return ipred


def rk4(kernel, kappa, rule, t_end, h=2e-3):
    """Independent check: fixed-step RK4 of (central, periph, eliminated)."""
    ev, gov = timeline(kernel, kappa)
    ev = ev + [(t_end, 99, "end", 0, None)]
    gov = gov + [gov[-1]]
    doses = dose_state(kernel, kappa)
    y = [0.0, 0.0, 0.0]
    cur = ev[0][0]
    arrived = [False] * len(DOSES)
    ipred = [None] * len(OBS)
    for i, (t, _rank, kind, idx, _snap) in enumerate(ev):
        if t > cur:
            p = gov[i]
            k10, k12, k21 = p["CL"] / p["V1"], p["Q"] / p["V1"], p["Q"] / p["V2"]
            open_doses = [
                (a, mass, own if rule == "fixed_at_dose" else (p["A"], p["B"]))
                for (k, (a, mass, own)) in enumerate(doses)
                if arrived[k]
            ]

            def f(tt, yy):
                inp = sum(m * rate(kernel, ka, kb, tt - a) for (a, m, (ka, kb)) in open_doses)
                return [
                    inp - (k10 + k12) * yy[0] + k21 * yy[1],
                    k12 * yy[0] - k21 * yy[1],
                    k10 * yy[0],
                ]

            steps = max(1, math.ceil((t - cur) / h))
            dt = (t - cur) / steps
            for s in range(steps):
                t0 = cur + s * dt
                k1 = f(t0, y)
                k2 = f(t0 + dt / 2, [y[j] + dt / 2 * k1[j] for j in range(3)])
                k3 = f(t0 + dt / 2, [y[j] + dt / 2 * k2[j] for j in range(3)])
                k4 = f(t0 + dt, [y[j] + dt * k3[j] for j in range(3)])
                y = [y[j] + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(3)]
            cur = t
        if kind == "arrival":
            arrived[idx] = True
        elif kind == "obs":
            ipred[idx] = y[0] / gov[i]["V1"]
    return ipred, y


def main():
    out = ["scenario,obs,time,fixed_at_dose,current_interval"]
    ok = True
    for kernel in KERNELS:
        for name, kappa in (("full", FULL), ("cl_only", CL_ONLY)):
            scenario = f"{kernel}_{name}"
            fixed = spectral(kernel, kappa, "fixed_at_dose")
            current = spectral(kernel, kappa, "current_interval")
            check, y = rk4(kernel, kappa, "fixed_at_dose", T_LEDGER)
            worst = max(abs(a - b) / abs(a) for a, b in zip(fixed, check))
            delivered = sum(m for (_a, m, _k) in dose_state(kernel, kappa))
            ledger = sum(y)
            split = max(abs(a - b) / abs(a) for a, b in zip(fixed, current))
            print(
                f"{scenario}: spectral vs RK4 max rel {worst:.3e}; mass ledger {ledger:.12f} / "
                f"{delivered:.12f}; fixed vs current max rel {split:.3e}",
                file=sys.stderr,
            )
            if not worst < 1e-9 or not abs(ledger - delivered) < 1e-8 * delivered:
                ok = False
            for j, (t, _occ, _wt) in enumerate(OBS):
                out.append(f"{scenario},{j},{t!r},{fixed[j]!r},{current[j]!r}")
    print("\n".join(out))
    if not ok:
        sys.exit("self-check failed")


if __name__ == "__main__":
    main()
