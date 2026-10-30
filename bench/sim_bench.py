#!/usr/bin/env python3
"""Simulate benchmark datasets for optimizer evaluation (pure Python, no numpy).

Writes NONMEM-format CSVs next to this file. Models (.ferx) are written by hand
with PERTURBED initial estimates so the optimizer has real work to do.
"""
import math, random, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def mvn2(rng, v11, v12, v22):
    # 2x2 Cholesky draw
    l11 = math.sqrt(v11)
    l21 = v12 / l11
    l22 = math.sqrt(v22 - l21 * l21)
    z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
    return l11 * z1, l21 * z1 + l22 * z2


def c1_oral(t, dose, cl, v, ka):
    k = cl / v
    if t <= 0:
        return 0.0
    return dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))


def c2_oral(t, dose, cl, v1, q, v2, ka):
    if t <= 0:
        return 0.0
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    beta = 0.5 * (s - math.sqrt(s * s - 4 * k21 * k10))
    alpha = k21 * k10 / beta
    a = (k21 - alpha) / ((ka - alpha) * (beta - alpha))
    b = (k21 - beta) / ((ka - beta) * (alpha - beta))
    c = (k21 - ka) / ((alpha - ka) * (beta - ka))
    return dose * ka / v1 * (a * math.exp(-alpha * t) + b * math.exp(-beta * t) + c * math.exp(-ka * t))


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def fmt(x):
    return f"{x:.6g}"


def sim_pk1(n, seed, path, nobs=6):
    """1-cpt oral, single dose 100, prop error sd 0.1; etas CL .09 V .04 KA .30."""
    rng = random.Random(seed)
    windows = [(0.5, 1.0), (1.0, 3.0), (3.0, 6.0), (8.0, 12.0), (24.0, 36.0), (48.0, 72.0)][:nobs]
    rows = []
    for i in range(1, n + 1):
        wt = max(40.0, min(120.0, rng.gauss(70, 12)))
        cl = 0.15 * (wt / 70) ** 0.75 * math.exp(rng.gauss(0, 0.3))
        v = 8.0 * math.exp(rng.gauss(0, 0.2))
        ka = 1.0 * math.exp(rng.gauss(0, math.sqrt(0.3)))
        rows.append([str(i), "0", ".", "1", "100", "1", "1", fmt(wt)])
        for lo, hi in windows:
            t = round(rng.uniform(lo, hi), 2)
            c = c1_oral(t, 100, cl, v, ka)
            dv = c * (1 + 0.1 * rng.gauss(0, 1))
            rows.append([str(i), fmt(t), fmt(max(dv, 1e-3)), "0", ".", "1", "0", fmt(wt)])
    write_csv(path, ["ID", "TIME", "DV", "EVID", "AMT", "CMT", "MDV", "WT"], rows)


def sim_pk2(n, seed, path):
    """2-cpt oral, covariates WT/CRCL, block(CL,V1), combined error."""
    rng = random.Random(seed)
    times = [0.5, 1, 2, 4, 8, 12, 24, 48]
    rows = []
    for i in range(1, n + 1):
        wt = max(40.0, min(130.0, rng.gauss(75, 15)))
        crcl = max(20.0, min(150.0, rng.gauss(90, 25)))
        e_cl, e_v1 = mvn2(rng, 0.10, 0.05, 0.10)
        e_v2 = rng.gauss(0, math.sqrt(0.10))
        e_ka = rng.gauss(0, math.sqrt(0.30))
        cl = 4.0 * (wt / 70) ** 0.75 * (crcl / 100) ** 0.4 * math.exp(e_cl)
        v1 = 40.0 * (wt / 70) * math.exp(e_v1)
        q = 8.0
        v2 = 80.0 * math.exp(e_v2)
        ka = 1.2 * math.exp(e_ka)
        rows.append([str(i), "0", ".", "1", "250", "1", "1", fmt(wt), fmt(crcl)])
        for t0 in times:
            t = round(t0 * math.exp(rng.gauss(0, 0.1)), 3)
            c = c2_oral(t, 250, cl, v1, q, v2, ka)
            sd = math.sqrt((0.10 * c) ** 2 + 0.05 ** 2)
            dv = c + sd * rng.gauss(0, 1)
            rows.append([str(i), fmt(t), fmt(max(dv, 1e-3)), "0", ".", "1", "0", fmt(wt), fmt(crcl)])
    write_csv(path, ["ID", "TIME", "DV", "EVID", "AMT", "CMT", "MDV", "WT", "CRCL"], rows)


def rk4_mm(state, dt, ka, v, vmax, km):
    def f(s):
        d, c = s
        return (-ka * d, ka * d / v - vmax * c / (km + c))
    k1 = f(state)
    k2 = f((state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1]))
    k3 = f((state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1]))
    k4 = f((state[0] + dt * k3[0], state[1] + dt * k3[1]))
    return (state[0] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            state[1] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def sim_mm(n, seed, path):
    """1-cpt oral MM elimination (ODE, concentration state), 3 doses q12h."""
    rng = random.Random(seed)
    doses = [0.0, 12.0, 24.0]
    obs = [1.0, 4.0, 8.0, 11.5, 25.0, 28.0, 36.0, 42.0]
    rows = []
    for i in range(1, n + 1):
        vmax = 1.2 * math.exp(rng.gauss(0, math.sqrt(0.10)))
        km = 4.0
        v = 10.0 * math.exp(rng.gauss(0, math.sqrt(0.10)))
        ka = 1.0 * math.exp(rng.gauss(0, math.sqrt(0.20)))
        ev = [(t, "dose") for t in doses] + [(round(t * math.exp(rng.gauss(0, 0.05)), 3), "obs") for t in obs]
        ev.sort(key=lambda x: (x[0], 0 if x[1] == "dose" else 1))
        s = (0.0, 0.0)
        tcur = 0.0
        dt = 0.005
        for t, kind in ev:
            while tcur < t - 1e-12:
                h = min(dt, t - tcur)
                s = rk4_mm(s, h, ka, v, vmax, km)
                tcur += h
            if kind == "dose":
                s = (s[0] + 100.0, s[1])
                rows.append([str(i), fmt(t), ".", "1", "100", "1", "1"])
            else:
                c = s[1]
                dv = c * (1 + 0.1 * rng.gauss(0, 1))
                rows.append([str(i), fmt(t), fmt(max(dv, 1e-3)), "0", ".", "1", "0"])
    write_csv(path, ["ID", "TIME", "DV", "EVID", "AMT", "CMT", "MDV"], rows)


if __name__ == "__main__":
    sim_pk1(500, 11, os.path.join(HERE, "pk1_n500.csv"))
    sim_pk1(2000, 12, os.path.join(HERE, "pk1_n2000.csv"))
    sim_pk1(300, 13, os.path.join(HERE, "pk1_sparse_n300.csv"), nobs=2)
    sim_pk2(300, 21, os.path.join(HERE, "pk2_n300.csv"))
    sim_mm(200, 31, os.path.join(HERE, "mm_n200.csv"))
    print("ok")
