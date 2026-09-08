"""Simulate a one-compartment oral dataset with three dosing occasions and
inter-occasion variability on CL only (kappa^2 = 0.04), so `iovsearch` has a
structure to find. IIV on CL, V and KA; proportional residual error.
Engine-independent (numpy only); NONMEM-format columns plus OCC."""
import csv
import os

import numpy as np

rng = np.random.default_rng(1184)
n = 40
occasions = [0.0, 48.0, 96.0]
post = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
dose = 100.0
tvcl, tvv, tvka = 0.5, 10.0, 1.2
om = [0.09, 0.04, 0.30]
kappa_cl = 0.04
sigma = 0.15

rows = [("ID", "TIME", "DV", "EVID", "AMT", "CMT", "RATE", "MDV", "OCC")]
for i in range(1, n + 1):
    eta = rng.normal(0.0, np.sqrt(om))
    v, ka = tvv * np.exp(eta[1]), tvka * np.exp(eta[2])
    cls = [tvcl * np.exp(eta[0] + rng.normal(0.0, np.sqrt(kappa_cl))) for _ in occasions]
    records = []
    for occ, t0 in enumerate(occasions, 1):
        records.append((t0, None, occ))
        for p in post:
            records.append((t0 + p, True, occ))
    # Exact piecewise solution: within an occasion the clearance is constant,
    # so the one-compartment oral closed form applies from that occasion's
    # initial state (the amounts carried over from the earlier occasions).
    def conc_at(t):
        a_gut, a_c, t_prev = 0.0, 0.0, 0.0
        for j, t0 in enumerate(occasions):
            if t0 > t + 1e-9:
                break
            # advance the carried-over state from t_prev to t0 under occasion j-1
            if j > 0:
                k = cls[j - 1] / v
                dtj = t0 - t_prev
                a_c = a_c * np.exp(-k * dtj) + ka * a_gut / (ka - k) * (
                    np.exp(-k * dtj) - np.exp(-ka * dtj)
                )
                a_gut = a_gut * np.exp(-ka * dtj)
            a_gut += dose
            t_prev = t0
        j = max(jj for jj, t0 in enumerate(occasions) if t0 <= t + 1e-9)
        k = cls[j] / v
        dtj = t - t_prev
        a_c = a_c * np.exp(-k * dtj) + ka * a_gut / (ka - k) * (
            np.exp(-k * dtj) - np.exp(-ka * dtj)
        )
        return a_c / v

    for t, is_obs, occ in records:
        if is_obs is None:
            rows.append((i, t, ".", 1, dose, 1, 0, 1, occ))
        else:
            f = conc_at(t)
            y = max(f * (1.0 + sigma * rng.normal()), 0.001)
            rows.append((i, t, f"{y:.5g}", 0, ".", 1, 0, 0, occ))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iov_sim.csv")
with open(out, "w", newline="") as fh:
    csv.writer(fh).writerows(rows)
print("wrote", out, len(rows) - 1, "rows")
