"""Dataset for #3010's shape: a modeled-duration infusion into a depot whose end moves with eta.

156 subjects, 28 doses q24h (`RATE=-2` into CMT 1, `D1 = TVD1*exp(eta)`, `F1 = exp(eta)`),
18 observations of the central concentration per subject, a constant `WT` covariate.
The 2-compartment disposition with first-order absorption from the depot is integrated with
RK4 at dt <= 0.01 h, cut at every infusion end, dose and sample, outside ferx (standard
library only; seeded, so the data are deterministic).

  --design fixed   samples at fixed times after doses 1, 2, 14, 27 and 28
  --design random  (default) 18 samples on random doses, 70 % of them 0.25-6 h after the dose,
                   so infusion ends fall on both sides of many samples

Run: python3 sim_modeled_infusion.py [--design fixed|random] > data.csv
"""
import argparse
import math
import random

N, AMT, II, NDOSE = 156, 100.0, 24.0, 28
CL, V1, Q, V2, KA, D1 = 4.0, 30.0, 2.0, 50.0, 1.0, 2.0
OM_CL, OM_V1, OM_D1, OM_F1 = 0.09, 0.0625, 0.16, 0.04
PROP_SD = 0.1
DT = 0.01
FIXED = [(0, [1, 2, 3, 6]), (1, [1.5, 2.5, 23]), (13, [1, 2, 4, 23]), (26, [2, 3, 23]),
         (27, [1, 2.5, 6, 12])]


def simulate(cl, v1, d1, f1, times):
    """Central concentration at `times` (sorted), RK4 cut at every infusion end and dose."""
    out = {}
    a = [0.0, 0.0, 0.0]
    t = 0.0

    def rate(t):
        j = int(t // II)
        return f1 * AMT / d1 if j < NDOSE and t - j * II < d1 else 0.0

    def f(a, r):
        return [r - KA * a[0],
                KA * a[0] - (cl + Q) / v1 * a[1] + Q / V2 * a[2],
                Q / v1 * a[1] - Q / V2 * a[2]]

    for tobs in times:
        while t < tobs - 1e-12:
            j = int(t // II)
            stop = min(e for e in (j * II + d1, (j + 1) * II, tobs) if e > t + 1e-12)
            while t < stop - 1e-12:
                h = min(DT, stop - t)
                r = rate(t + h / 2)
                k1 = f(a, r)
                k2 = f([x + h / 2 * k for x, k in zip(a, k1)], r)
                k3 = f([x + h / 2 * k for x, k in zip(a, k2)], r)
                k4 = f([x + h * k for x, k in zip(a, k3)], r)
                a = [x + h / 6 * (p + 2 * q + 2 * s + u)
                     for x, p, q, s, u in zip(a, k1, k2, k3, k4)]
                t += h
        out[tobs] = a[1] / v1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["fixed", "random"], default="random")
    args = ap.parse_args()
    random.seed(3010)
    print("ID,TIME,AMT,RATE,DV,MDV,EVID,CMT,WT")
    for i in range(1, N + 1):
        wt = random.uniform(50, 100)
        cl = CL * (wt / 70) ** 0.75 * math.exp(random.gauss(0, math.sqrt(OM_CL)))
        v1 = V1 * (wt / 70) * math.exp(random.gauss(0, math.sqrt(OM_V1)))
        d1 = D1 * math.exp(random.gauss(0, math.sqrt(OM_D1)))
        f1 = math.exp(random.gauss(0, math.sqrt(OM_F1)))
        if args.design == "fixed":
            obs = [j * II + o for j, offs in FIXED for o in offs]
        else:
            obs = []
            for _ in range(18):
                j = random.randrange(NDOSE)
                early = random.random() < 0.7
                obs.append(j * II + round(random.uniform(0.25, 6.0) if early
                                          else random.uniform(6.0, 23.9), 2))
        obs = sorted(set(obs))
        conc = simulate(cl, v1, d1, f1, obs)
        rows = [(j * II, 0, f"{i},{j * II:g},{AMT:g},-2,0,1,1,1,{wt:.1f}") for j in range(NDOSE)]
        for t in obs:
            dv = conc[t] * (1 + random.gauss(0, PROP_SD))
            rows.append((t, 1, f"{i},{t:g},0,0,{dv:.6f},0,0,2,{wt:.1f}"))
        for _, _, line in sorted(rows):
            print(line)


if __name__ == "__main__":
    main()
