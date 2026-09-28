#!/usr/bin/env python3
"""Simulate the two #1569 carry-over anchor datasets: a 1-cpt model with transit
(`transit_iov_mtt.csv`) or inverse-Gaussian (`ig_iov_mat.csv`) absorption, IOV on CL and
on the absorption parameter (MTT / MAT), and doses that are still absorbing when their
occasion's successor begins.

Pure standard library (math, random) -- no numpy/scipy. Deterministic (seeds below), so
re-running reproduces both CSVs byte-for-byte.

Design, per subject (24 subjects):
  doses 100 into CMT 1 at t = 0, 6, 12, 18 (q6h, overlapping absorption);
  occasion 1: t < 8.5 -- doses 0 and 6, observations 1, 3, 5, 7;
  occasion 2: begins at the OBSERVATION record t = 8.5 while dose 2 (t = 6) is still
              absorbing -- observations 8.5, 10, 11.5, dose 12, observations 13, 15, 17;
  occasion 3: begins at the DOSE record t = 18 while dose 3 (t = 12) is still absorbing --
              observations 19, 21, 24, 30, 36.

Model (the rule under test, #1569 -- each dose keeps the absorption kernel of its own
dose record; the disposition is the governing record's, NONMEM end-of-interval):

    d/dt(central) = sum_d AMT_d * R_d(t - t_d) - CL/V * central,   IPRED = central / V
    transit: R(x) = KTR (KTR x)^N exp(-KTR x) / Gamma(N + 1),       KTR = (N + 1)/MTT
    igd:     R(x) = sqrt(MAT / (2 pi CV2 x^3)) exp(-(x - MAT)^2 / (2 CV2 MAT x))
    CL  = TVCL * exp(ETA_CL + KAPPA_CL[occ]),  V = TVV * exp(ETA_V)
    MTT = TVMTT * exp(KAPPA_ABS[occ])   (transit)   /   MAT = TVMAT * exp(KAPPA_ABS[occ])  (igd)
    DV  = IPRED * (1 + EPS),  EPS ~ N(0, SIGMA^2)

Integrated with fixed-step RK4 (h = 0.005, steps aligned to every record). The data only
has to be plausible, not exact: both engines are anchored on the same rows.
"""
import math
import random

N_SUBJECTS = 24
DOSE_TIMES = [0.0, 6.0, 12.0, 18.0]
OBS_TIMES = [1.0, 3.0, 5.0, 7.0, 8.5, 10.0, 11.5, 13.0, 15.0, 17.0, 19.0, 21.0, 24.0, 30.0, 36.0]
AMT = 100.0


def occasion(t, is_dose):
    if t < 8.5:
        return 1
    if t < 18.0 or (t == 18.0 and not is_dose):
        return 2
    return 3


MODELS = {
    "transit_iov_mtt": {
        "seed": 1569,
        "TVCL": 5.0,
        "TVV": 30.0,
        "TVABS": 3.0,  # MTT
        "SHAPE": 3.0,  # N
        "OM_CL": 0.09,
        "OM_V": 0.04,
        "OM_IOV_CL": 0.04,
        "OM_IOV_ABS": 0.09,
        "SIGMA": 0.1,
    },
    "ig_iov_mat": {
        "seed": 1570,
        "TVCL": 5.0,
        "TVV": 30.0,
        "TVABS": 3.0,  # MAT
        "SHAPE": 0.5,  # CV2
        "OM_CL": 0.09,
        "OM_V": 0.04,
        "OM_IOV_CL": 0.04,
        "OM_IOV_ABS": 0.09,
        "SIGMA": 0.1,
    },
}


def kernel(name, abs_param, shape):
    """R(x) for one dose's kernel, fixed at its dose record."""
    if name == "transit_iov_mtt":
        n, mtt = shape, abs_param
        ktr = (n + 1.0) / mtt
        lg = math.lgamma(n + 1.0)

        def r(x):
            if x <= 0.0:
                return 0.0
            return math.exp(math.log(ktr) + n * math.log(ktr * x) - ktr * x - lg)

        return r
    mat, cv2 = abs_param, shape

    def r(x):
        if x <= 0.0:
            return 0.0
        return math.sqrt(mat / (2.0 * math.pi * cv2 * x**3)) * math.exp(
            -((x - mat) ** 2) / (2.0 * cv2 * mat * x)
        )

    return r


def simulate(name, cfg):
    rng = random.Random(cfg["seed"])
    rows = []
    for sid in range(1, N_SUBJECTS + 1):
        eta_cl = rng.gauss(0.0, math.sqrt(cfg["OM_CL"]))
        eta_v = rng.gauss(0.0, math.sqrt(cfg["OM_V"]))
        k_cl = [rng.gauss(0.0, math.sqrt(cfg["OM_IOV_CL"])) for _ in range(3)]
        k_abs = [rng.gauss(0.0, math.sqrt(cfg["OM_IOV_ABS"])) for _ in range(3)]
        v = cfg["TVV"] * math.exp(eta_v)

        def cl_at(occ):
            return cfg["TVCL"] * math.exp(eta_cl + k_cl[occ - 1])

        # Each dose's kernel, from its own record's occasion.
        doses = []
        for t in DOSE_TIMES:
            occ = occasion(t, True)
            abs_param = cfg["TVABS"] * math.exp(k_abs[occ - 1])
            doses.append((t, kernel(name, abs_param, cfg["SHAPE"])))

        # Records in NONMEM order: a dose record before an observation at the same time.
        records = [(t, 0, occasion(t, True)) for t in DOSE_TIMES]
        records += [(t, 1, occasion(t, False)) for t in OBS_TIMES]
        records.sort()
        a = 0.0
        cur = records[0][0]
        for (t, kind, occ) in records:
            if t > cur:
                ke = cl_at(occ) / v  # governed by the record that terminates the interval
                live = [(td, r) for (td, r) in doses if td < t]

                def f(tt, y):
                    return AMT * sum(r(tt - td) for (td, r) in live) - ke * y

                steps = max(1, math.ceil((t - cur) / 0.005))
                h = (t - cur) / steps
                for s in range(steps):
                    t0 = cur + s * h
                    k1 = f(t0, a)
                    k2 = f(t0 + h / 2, a + h / 2 * k1)
                    k3 = f(t0 + h / 2, a + h / 2 * k2)
                    k4 = f(t0 + h, a + h * k3)
                    a += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                cur = t
            if kind == 0:
                rows.append(f"{sid},{t:g},.,1,{AMT:g},1,1,{occ}")
            else:
                ipred = a / v
                dv = ipred * (1.0 + rng.gauss(0.0, cfg["SIGMA"]))
                rows.append(f"{sid},{t:g},{dv:.6f},0,.,1,0,{occ}")
    with open(f"{name}.csv", "w") as fh:
        fh.write("ID,TIME,DV,EVID,AMT,CMT,MDV,OCC\n")
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    for name, cfg in MODELS.items():
        simulate(name, cfg)
