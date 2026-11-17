#!/usr/bin/env python3
"""Simulate a parent + metabolite model with urine collected over K intervals, and write
the matching gated-accumulator ferx model.

The data: N_SUBJECTS subjects, each dosed by a 1-h IV infusion at the start of each of K
24-h collection intervals. Plasma parent is sampled at SAMPLE_OFFSETS after each dose; at
the end of each interval the amounts of parent and of metabolite excreted into urine over
that interval are observed, and an EVID=2 row advances the interval index IDX.

The truth is the reduced 4-state linear system
    d/dt parent = R(t) - (K12 + K15) * parent
    d/dt m1     = K12 * parent - KM * m1
    d/dt up     = K15 * parent          (cumulative parent in urine)
    d/dt um     = KM * m1               (cumulative metabolite in urine)
and an interval's urine observation is the difference of the cumulative amounts at its
ends. It is propagated between breakpoints exactly (matrix exponential of the augmented
generator, scaling and squaring) and `--check` compares it against a fine-step RK4.

ferx has no per-compartment reset, so the model collects each interval in its own
accumulator: 2 + 2K states, gate G_k = if (IDX == k) 1 else 0. Only one accumulator per
urine output is live in any segment; the rest are dormant. That is the shape this fixture
exists to exercise: a large linear system of which a few states move.

Writes urine_intervals_k<K>.csv and urine_intervals_k<K>.ferx next to this script (or into
--out). Usage: sim_urine_intervals.py [--k K] [--out DIR] [--check]. Deterministic given SEED.
"""
import argparse
import math
import os
import random

SEED = 5
N_SUBJECTS = 24
DOSE = 100.0  # mg per interval
T_INF = 1.0  # h
INTERVAL = 24.0  # h
SAMPLE_OFFSETS = [1.0, 2.0, 6.0, 12.0]  # h after each dose
# Truth (typical values), random effects (variances) and residual errors (variances).
TV = {"V": 20.0, "CLR": 1.0, "CLM": 2.0, "KM": 0.3}
OMEGA = {"V": 0.04, "CLR": 0.09, "CLM": 0.09}
SIGMA = {"PL": 0.01, "UP": 0.02, "UM": 0.02}

HERE = os.path.dirname(os.path.abspath(__file__))


def matmul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def expm(a):
    """exp(a) by scaling and squaring a Taylor series (valid for repeated eigenvalues)."""
    n = len(a)
    norm = max(sum(abs(x) for x in row) for row in a)
    s = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0 else 0
    sc = [[x / 2.0**s for x in row] for row in a]
    out = [[float(i == j) for j in range(n)] for i in range(n)]
    term = [row[:] for row in out]
    for k in range(1, 30):
        term = [[x / k for x in row] for row in matmul(term, sc)]
        out = [[out[i][j] + term[i][j] for j in range(n)] for i in range(n)]
    for _ in range(s):
        out = matmul(out, out)
    return out


def generator(p, rate):
    """Augmented 5x5 generator [[A, b], [0, 0]] for the state (parent, m1, up, um, 1)."""
    k12, k15, km = p["CLM"] / p["V"], p["CLR"] / p["V"], p["KM"]
    return [
        [-(k12 + k15), 0.0, 0.0, 0.0, rate],
        [k12, -km, 0.0, 0.0, 0.0],
        [k15, 0.0, 0.0, 0.0, 0.0],
        [0.0, km, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def step_exact(x, p, rate, h):
    e = expm([[v * h for v in row] for row in generator(p, rate)])
    return [sum(e[i][j] * (x + [1.0])[j] for j in range(5)) for i in range(4)]


def step_rk4(x, p, rate, h, n=4000):
    g = generator(p, rate)
    f = lambda y: [sum(g[i][j] * (y + [1.0])[j] for j in range(5)) for i in range(4)]
    d = h / n
    for _ in range(n):
        k1 = f(x)
        k2 = f([x[i] + d / 2 * k1[i] for i in range(4)])
        k3 = f([x[i] + d / 2 * k2[i] for i in range(4)])
        k4 = f([x[i] + d * k3[i] for i in range(4)])
        x = [x[i] + d / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]
    return x


def predictions(p, k, step):
    """Noise-free observations: (time, output key, value), in time order."""
    events = sorted(
        {0.0}
        | {i * INTERVAL + o for i in range(k) for o in [T_INF] + SAMPLE_OFFSETS}
        | {(i + 1) * INTERVAL for i in range(k)}
    )
    x, t, out, start = [0.0] * 4, 0.0, [], [0.0, 0.0]
    for t_next in events[1:]:
        rate = DOSE / T_INF if (t % INTERVAL) < T_INF - 1e-9 else 0.0
        x = step(x, p, rate, t_next - t)
        t = t_next
        off = round(t % INTERVAL, 9)
        if off in SAMPLE_OFFSETS:
            out.append((t, 1, x[0] / p["V"]))
        if off == 0.0:
            out.append((t, 2, x[2] - start[0]))
            out.append((t, 3, x[3] - start[1]))
            start = [x[2], x[3]]
    return out


def individual(rng):
    p = dict(TV)
    for name, var in OMEGA.items():
        p[name] = TV[name] * math.exp(rng.gauss(0.0, math.sqrt(var)))
    return p


def write_data(k, path):
    rng = random.Random(SEED)
    rows = ["ID,TIME,DV,AMT,RATE,EVID,CMT,MDV,IDX"]
    for sid in range(1, N_SUBJECTS + 1):
        p = individual(rng)
        obs = {(t, c): v for t, c, v in predictions(p, k, step_exact)}
        sd = {1: SIGMA["PL"], 2: SIGMA["UP"], 3: SIGMA["UM"]}
        for i in range(k):
            t0 = i * INTERVAL
            rows.append(f"{sid},{t0:g},0,{DOSE:g},{DOSE / T_INF:g},1,1,1,{i + 1}")
            for o in SAMPLE_OFFSETS:
                dv = obs[(t0 + o, 1)] * (1.0 + rng.gauss(0.0, math.sqrt(sd[1])))
                rows.append(f"{sid},{t0 + o:g},{dv:.6g},0,0,0,1,0,{i + 1}")
            t1 = t0 + INTERVAL
            for c in (2, 3):
                dv = obs[(t1, c)] * (1.0 + rng.gauss(0.0, math.sqrt(sd[c])))
                rows.append(f"{sid},{t1:g},{dv:.6g},0,0,0,{c},0,{i + 1}")
            if i + 1 < k:
                rows.append(f"{sid},{t1:g},0,0,0,2,1,1,{i + 2}")
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")


def write_model(k, path, data_rel):
    up = [f"up{i}" for i in range(1, k + 1)]
    um = [f"um{i}" for i in range(1, k + 1)]
    gates = "\n".join(f"  G{i} = if (IDX == {i}) 1 else 0" for i in range(1, k + 1))
    odes = "\n".join(
        [f"  d/dt(up{i}) = G{i} * K15 * parent" for i in range(1, k + 1)]
        + [f"  d/dt(um{i}) = G{i} * KM * m1" for i in range(1, k + 1)]
    )
    y_up = " + ".join(f"G{i} * up{i}" for i in range(1, k + 1))
    y_um = " + ".join(f"G{i} * um{i}" for i in range(1, k + 1))
    text = f"""# Parent + metabolite with urine collected over {k} intervals (bench/dormant_states/sim_urine_intervals.py).
# ferx has no per-compartment reset, so each interval's urine is its own accumulator,
# switched on by the time-varying interval index IDX: {2 + 2 * k} states, of which at most
# four move in any segment.
# Data: {data_rel}

[parameters]
  theta TVV(16, 0.1, 1000)
  theta TVCLR(1.3, 0.01, 100)
  theta TVCLM(1.5, 0.01, 100)
  theta TVKM(0.4, 0.001, 10)
  omega ETA_V ~ 0.1
  omega ETA_CLR ~ 0.1
  omega ETA_CLM ~ 0.1
  sigma PROP_PL ~ 0.02
  sigma PROP_UP ~ 0.02
  sigma PROP_UM ~ 0.02

[individual_parameters]
  V = TVV * exp(ETA_V)
  CLR = TVCLR * exp(ETA_CLR)
  CLM = TVCLM * exp(ETA_CLM)
  KM = TVKM
  K12 = CLM / V
  K15 = CLR / V
{gates}

[structural_model]
  ode(states=[parent, m1, {", ".join(up + um)}])

[odes]
  d/dt(parent) = -(K12 + K15) * parent
  d/dt(m1) = K12 * parent - KM * m1
{odes}

[scaling]
  y[CMT=1] = parent / V
  y[CMT=2] = {y_up}
  y[CMT=3] = {y_um}

[error_model]
  CMT=1: DV ~ proportional(PROP_PL)
  CMT=2: DV ~ proportional(PROP_UP)
  CMT=3: DV ~ proportional(PROP_UM)

[fit_options]
  method = focei
  covariance = true
"""
    with open(path, "w") as fh:
        fh.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="collection intervals per subject")
    ap.add_argument("--out", help="output directory (default: next to this script)")
    ap.add_argument("--check", action="store_true", help="compare exact vs RK4 and exit")
    args = ap.parse_args()
    if args.check:
        rng = random.Random(SEED)
        worst = 0.0
        for _ in range(3):
            p = individual(rng)
            a = predictions(p, args.k, step_exact)
            b = predictions(p, args.k, step_rk4)
            for (_, _, va), (_, _, vb) in zip(a, b):
                assert math.isfinite(va) and math.isfinite(vb)
                worst = max(worst, abs(va - vb) / max(abs(vb), 1e-300))
        print(f"exact vs RK4, 3 subjects, K={args.k}: worst relative difference {worst:.3e}")
        return
    out = args.out or HERE
    stem = f"urine_intervals_k{args.k}"
    write_data(args.k, os.path.join(out, f"{stem}.csv"))
    write_model(args.k, os.path.join(out, f"{stem}.ferx"), f"{stem}.csv")
    print(f"wrote {stem}.csv and {stem}.ferx in {out}")


if __name__ == "__main__":
    main()
