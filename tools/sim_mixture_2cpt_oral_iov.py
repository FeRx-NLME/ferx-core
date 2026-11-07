#!/usr/bin/env python3
"""Simulate a two-compartment oral population with a lag, IOV on CL and a two-class mixture.

The fixture for the mixture outer-optimizer measurements (`optimizer = auto` picks BOBYQA for
every mixture model). Fitted by examples/mixture_2cpt_oral_iov.ferx.

Model, per subject i in latent class k (class 1 with probability P1), occasion o:
    CL = TVCL_k · exp(ETA_CL + KAPPA_CL[o]),  V1 = TVV1 · exp(ETA_V1),
    Q  = TVQ,  V2 = TVV2,  KA = TVKA · exp(ETA_KA),  lag = TVLAG,
    d(depot)/dt = −KA·depot,
    d(central)/dt = KA·depot − (CL + Q)/V1·central + Q/V2·periph,
    d(periph)/dt = Q/V1·central − Q/V2·periph,
    C = central / V1,  DV = C · (1 + SIGMA_PROP·ε).
Each subject takes DOSE every TAU hours for N_DOSES doses (one record per dose, so the dose
history is long), over N_OCC occasions of equal length; each occasion is sampled after its
last dose at SAMPLE_TIMES.

The system is linear with piecewise-constant coefficients, so each interval is propagated
exactly by exp(M·t), from Sylvester's formula on M's eigenvalues: −KA and the two roots of
λ² + (k10 + k12 + k21)·λ + k10·k21 = 0. No ODE solver and no ferx involved, so the data do not
depend on the engine they test. Occasions follow ferx's (and NONMEM's) end-of-interval rule: the
interval ending at a record runs under that record's occasion. Only CL carries IOV, so a dose's
absorption (KA, lag) never changes with the occasion. Deterministic given SEED.

Writes data/mixture_2cpt_oral_iov.csv (ID, TIME, AMT, DV, EVID, MDV, CMT, OCC). Prints the
class sizes to stderr.
"""
import math
import os
import random
import sys

SEED = 977
N_SUBJECTS = 120
P1 = 0.6  # probability of class 1
TVCL = (2.0, 6.0)  # class 1, class 2 (L/h)
TVV1, TVQ, TVV2, TVKA, TVLAG = 30.0, 5.0, 60.0, 1.0, 0.5
OMEGA_CL, OMEGA_V1, OMEGA_KA = 0.09, 0.09, 0.16  # variances
OMEGA_IOV_CL = 0.04  # KAPPA_CL variance
SIGMA_PROP = 0.1
DOSE, TAU, N_DOSES, N_OCC = 100.0, 12.0, 200, 4
SAMPLE_TIMES = (1.0, 3.0, 6.0, 11.5)  # hours after an occasion's last dose


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def expm_t(cl, v1, q, v2, ka, t):
    """exp(M·t) for the 3-state system, by Sylvester's formula (distinct eigenvalues)."""
    k10, k12, k21 = cl / v1, q / v1, q / v2
    m = [[-ka, 0.0, 0.0], [ka, -(k10 + k12), k21], [0.0, k12, -k21]]
    s = k10 + k12 + k21
    disc = math.sqrt(s * s - 4.0 * k10 * k21)
    lams = [-ka, (-s + disc) / 2.0, (-s - disc) / 2.0]
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(lams[i] - lams[j]) < 1e-9 * max(abs(lams[i]), 1.0):
                raise ValueError(f"repeated eigenvalue {lams[i]}; redraw")
    eye = [[float(i == j) for j in range(3)] for i in range(3)]
    out = [[0.0] * 3 for _ in range(3)]
    for i, li in enumerate(lams):
        p = eye
        for j, lj in enumerate(lams):
            if j != i:
                f = [[(m[r][c] - (lj if r == c else 0.0)) / (li - lj) for c in range(3)]
                     for r in range(3)]
                p = matmul(p, f)
        e = math.exp(li * t)
        out = [[out[r][c] + e * p[r][c] for c in range(3)] for r in range(3)]
    return out


def propagate(state, cl, v1, q, v2, ka, t):
    if t <= 0.0:
        return state
    e = expm_t(cl, v1, q, v2, ka, t)
    return [sum(e[r][c] * state[c] for c in range(3)) for r in range(3)]


def main():
    rng = random.Random(SEED)
    occ_len = N_DOSES // N_OCC  # doses per occasion
    rows = ["ID,TIME,AMT,DV,EVID,MDV,CMT,OCC"]
    n_class = [0, 0]
    for sid in range(1, N_SUBJECTS + 1):
        cls = 0 if rng.random() < P1 else 1
        n_class[cls] += 1
        eta_cl = rng.gauss(0.0, math.sqrt(OMEGA_CL))
        v1 = TVV1 * math.exp(rng.gauss(0.0, math.sqrt(OMEGA_V1)))
        ka = TVKA * math.exp(rng.gauss(0.0, math.sqrt(OMEGA_KA)))
        kappa = [rng.gauss(0.0, math.sqrt(OMEGA_IOV_CL)) for _ in range(N_OCC)]
        cl_occ = [TVCL[cls] * math.exp(eta_cl + kappa[o]) for o in range(N_OCC)]
        # Records: (time, kind, occasion), kind 0 = dose, 1 = observation.
        records = []
        for d in range(N_DOSES):
            records.append((d * TAU, 0, d // occ_len))
        for o in range(N_OCC):
            last_dose = ((o + 1) * occ_len - 1) * TAU
            for s in SAMPLE_TIMES:
                records.append((last_dose + s, 1, o))
        records.sort(key=lambda r: (r[0], r[1]))
        arrivals = sorted(r[0] + TVLAG for r in records if r[1] == 0)
        state = [0.0, 0.0, 0.0]
        t_prev, a_next = 0.0, 0
        for t, kind, occ in records:
            # The interval (t_prev, t] runs under this record's occasion; split it at the
            # lagged dose arrivals that fall inside.
            cl = cl_occ[occ]
            while a_next < len(arrivals) and arrivals[a_next] <= t:
                ta = arrivals[a_next]
                state = propagate(state, cl, v1, TVQ, TVV2, ka, ta - t_prev)
                state[0] += DOSE
                t_prev, a_next = ta, a_next + 1
            state = propagate(state, cl, v1, TVQ, TVV2, ka, t - t_prev)
            t_prev = t
            if kind == 0:
                rows.append(f"{sid},{t:g},{DOSE:g},.,1,1,1,{occ + 1}")
            else:
                conc = state[1] / v1
                dv = conc * (1.0 + SIGMA_PROP * rng.gauss(0.0, 1.0))
                assert dv > 0.0, f"subject {sid}, t = {t}: non-positive DV {dv}"
                # CMT 1 on observations, as in data/two_cpt_oral_cov.csv: the closed form
                # reads the central concentration.
                rows.append(f"{sid},{t:g},.,{dv:.6g},0,0,1,{occ + 1}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                       "mixture_2cpt_oral_iov.csv")
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {os.path.normpath(out)}: {N_SUBJECTS} subjects, class sizes {n_class}, "
          f"{N_DOSES} doses and {N_OCC * len(SAMPLE_TIMES)} samples each", file=sys.stderr)


if __name__ == "__main__":
    main()
