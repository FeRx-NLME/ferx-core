#!/usr/bin/env python3
"""Independent reference for #1560: 1-cpt transit and inverse-Gaussian absorption under IOV
and a time-varying covariate, with a later occasion beginning while an earlier dose is still
absorbing AND still in central. The 1-cpt companion of
`absorption_fixed_at_dose_quadrature.py` (#1569, 2-cpt), built the same way.

Pure standard library (math) -- no numpy/scipy, no ferx code, no NONMEM code.
Deterministic: re-running reproduces the CSV byte-for-byte.

    python3 tests/data/absorption_walk_1cpt_quadrature.py \
        > tests/data/absorption_walk_1cpt_quadrature.csv

Model (the ODE twin of `pk one_cpt_transit(cl, v, n, mtt, f, lagtime)` or
`pk one_cpt_ig(cl, v, mat, cv2, f, lagtime)`):

    d/dt(central) = sum_d F_d * D_d * R_d(t - a_d) - (CL/V) * central
    a_d           = t_d + ALAG_d                          (arrival of dose d)
    IPRED         = central / V
    transit: R(x) = KTR (KTR x)^N exp(-KTR x) / Gamma(N + 1),  KTR = (N + 1)/MTT
    igd:     R(x) = sqrt(MAT / (2 pi CV2 x^3)) exp(-(x - MAT)^2 / (2 CV2 MAT x))

Individual parameters at a record (occasion `o`, covariate `WT`):

    CL = TVCL * (WT/70)^0.75 * exp(KCL[o])     V = TVV * exp(ETA_V)
    kernel (transit):  N   = TVN   * exp(KA[o]),   MTT = TVMTT * exp(KB[o])
    kernel (igd):      MAT = TVMAT * exp(KA[o]),   CV2 = TVCV2 * exp(KB[o])
    F  = TVF * exp(KF[o])                      ALAG = TVLAG

Record conventions -- the ones every ferx engine implements (#1073):
  * a segment is governed by the record that TERMINATES it: itself for a record, the next
    record ahead for a lagged dose arrival (not a record);
  * F, ALAG and the arrival are the dose record's own.

Two rules for an in-flight dose's kernel, both tabulated:
  * fixed_at_dose    -- each dose keeps the kernel of its own dose record (#1569, #1560);
  * current_interval -- every open dose uses the governing record's kernel.

Design: occasion 2 begins on an OBSERVATION record (t = 5), while dose 1 (t = 0, arrival
0.5) still has ~half its mass in the kernel; occasion 3 begins on a DOSE record (t = 12).

Method: on each segment [s0, s1], x(s1) = e^{-ke D} x(s0) + sum_d F_d D_d *
integral_{max(s0,a_d)}^{s1} e^{-ke (s1 - tau)} R_d(tau - a_d) dtau, each integral by
tanh-sinh quadrature.

Self-checks (the script exits non-zero if either fails): an independent fixed-step RK4 of
the same ODE agrees at every observation, and its mass ledger (central + eliminated) equals
sum(F_d * D_d) at t = 400 under fixed_at_dose.

Scenarios: transit_full / ig_full (IOV on CL, both kernel parameters and F; WT on CL);
transit_f_only (IOV on F alone, so the two kernel rules coincide and F is what moves).
"""
import math
import sys

TVCL, TVV = 3.0, 25.0
TVF, TVLAG = 0.8, 0.5
ETA_V = 0.1
KERNELS = {"transit": (2.0, 4.0), "ig": (4.0, 0.3)}

FULL = {
    "KCL": [0.10, -0.15, 0.20],
    "KA": [0.00, 0.25, -0.20],
    "KB": [0.20, -0.30, 0.35],
    "KF": [0.00, -0.10, 0.10],
}
F_ONLY = {
    "KCL": [0.0, 0.0, 0.0],
    "KA": [0.0, 0.0, 0.0],
    "KB": [0.0, 0.0, 0.0],
    "KF": [0.25, -0.30, 0.20],
}
SCENARIOS = [("transit", "full", FULL), ("transit", "f_only", F_ONLY), ("ig", "full", FULL)]

# Doses: (time, amt, occasion, WT).
DOSES = [
    (0.0, 100.0, 1, 70.0),
    (6.0, 100.0, 2, 72.0),
    (12.0, 100.0, 3, 74.0),
]
# Observations: (time, occasion, WT).
OBS = [
    (1.0, 1, 70.0),
    (2.5, 1, 70.0),
    (4.0, 1, 71.0),
    (5.0, 2, 72.0),
    (5.5, 2, 72.0),
    (7.0, 2, 73.0),
    (9.0, 2, 73.0),
    (11.0, 2, 74.0),
    (13.0, 3, 74.0),
    (15.0, 3, 76.0),
    (18.0, 3, 78.0),
    (24.0, 3, 80.0),
    (36.0, 3, 82.0),
    (48.0, 3, 84.0),
]
T_LEDGER = 400.0


def params(kernel, kappa, occ, wt):
    o = occ - 1
    tva, tvb = KERNELS[kernel]
    return {
        "CL": TVCL * (wt / 70.0) ** 0.75 * math.exp(kappa["KCL"][o]),
        "V": TVV * math.exp(ETA_V),
        "A": tva * math.exp(kappa["KA"][o]),
        "B": tvb * math.exp(kappa["KB"][o]),
        "F": TVF * math.exp(kappa["KF"][o]),
        "ALAG": TVLAG,
    }


def rate(kernel, a, b, x):
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


def tanh_sinh(f, a, b, tol=1e-15):
    """Integral of f over [a, b]; robust to integrable endpoint singularities."""
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


DOSE_RECORD, ARRIVAL, OBSERVATION = 1, 2, 4  # ferx's kind_order ranks


def timeline(kernel, kappa):
    ev = []
    for k, (t, _amt, occ, wt) in enumerate(DOSES):
        p = params(kernel, kappa, occ, wt)
        ev.append((t, DOSE_RECORD, "dose", k, p))
        ev.append((t + p["ALAG"], ARRIVAL, "arrival", k, None))
    for j, (t, occ, wt) in enumerate(OBS):
        ev.append((t, OBSERVATION, "obs", j, params(kernel, kappa, occ, wt)))
    ev.sort(key=lambda e: (e[0], e[1]))
    gov = [None] * len(ev)
    nxt = None
    for i in range(len(ev) - 1, -1, -1):
        if ev[i][4] is not None:
            nxt = ev[i][4]
        gov[i] = nxt
    return ev, gov


def dose_state(kernel, kappa):
    out = []
    for (t, amt, occ, wt) in DOSES:
        p = params(kernel, kappa, occ, wt)
        out.append((t + p["ALAG"], p["F"] * amt, (p["A"], p["B"])))
    return out


def spectral(kernel, kappa, rule):
    ev, gov = timeline(kernel, kappa)
    doses = dose_state(kernel, kappa)
    x = 0.0
    cur = ev[0][0]
    arrived = [False] * len(DOSES)
    ipred = [None] * len(OBS)
    for i, (t, _rank, kind, idx, _snap) in enumerate(ev):
        if t > cur:
            p = gov[i]
            ke = p["CL"] / p["V"]
            x *= math.exp(-ke * (t - cur))
            for k, (a, mass, own) in enumerate(doses):
                if not arrived[k]:
                    continue
                ka, kb = own if rule == "fixed_at_dose" else (p["A"], p["B"])
                lo = max(cur, a)
                x += mass * tanh_sinh(
                    lambda tau: math.exp(-ke * (t - tau)) * rate(kernel, ka, kb, tau - a), lo, t
                )
            cur = t
        if kind == "arrival":
            arrived[idx] = True
        elif kind == "obs":
            ipred[idx] = x / gov[i]["V"]
    return ipred


def rk4(kernel, kappa, rule, t_end, h=2e-3):
    ev, gov = timeline(kernel, kappa)
    ev = ev + [(t_end, 99, "end", 0, None)]
    gov = gov + [gov[-1]]
    doses = dose_state(kernel, kappa)
    y = [0.0, 0.0]  # central, eliminated
    cur = ev[0][0]
    arrived = [False] * len(DOSES)
    ipred = [None] * len(OBS)
    for i, (t, _rank, kind, idx, _snap) in enumerate(ev):
        if t > cur:
            p = gov[i]
            ke = p["CL"] / p["V"]
            open_doses = [
                (a, mass, own if rule == "fixed_at_dose" else (p["A"], p["B"]))
                for (k, (a, mass, own)) in enumerate(doses)
                if arrived[k]
            ]

            def f(tt, yy):
                inp = sum(m * rate(kernel, ka, kb, tt - a) for (a, m, (ka, kb)) in open_doses)
                return [inp - ke * yy[0], ke * yy[0]]

            steps = max(1, math.ceil((t - cur) / h))
            dt = (t - cur) / steps
            for s in range(steps):
                t0 = cur + s * dt
                k1 = f(t0, y)
                k2 = f(t0 + dt / 2, [y[j] + dt / 2 * k1[j] for j in range(2)])
                k3 = f(t0 + dt / 2, [y[j] + dt / 2 * k2[j] for j in range(2)])
                k4 = f(t0 + dt, [y[j] + dt * k3[j] for j in range(2)])
                y = [y[j] + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(2)]
            cur = t
        if kind == "arrival":
            arrived[idx] = True
        elif kind == "obs":
            ipred[idx] = y[0] / gov[i]["V"]
    return ipred, y


def main():
    out = ["scenario,obs,time,fixed_at_dose,current_interval"]
    ok = True
    for kernel, name, kappa in SCENARIOS:
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
