"""Simulate a one-compartment oral dataset whose residual error is a power
model (SD = 0.25 * f^0.5), so `ruvsearch` has a feature to find. Engine-
independent (numpy only), NONMEM-format columns matching data/warfarin.csv."""
import csv
import os

import numpy as np

rng = np.random.default_rng(1182)
n = 40
times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0]
dose = 100.0
tvcl, tvv, tvka = 0.13, 8.0, 1.0
om = [0.09, 0.04, 0.30]
sigma, power = 0.25, 0.5

rows = [("ID", "TIME", "DV", "EVID", "AMT", "CMT", "RATE", "MDV")]
for i in range(1, n + 1):
    eta = rng.normal(0.0, np.sqrt(om))
    cl, v, ka = tvcl * np.exp(eta[0]), tvv * np.exp(eta[1]), tvka * np.exp(eta[2])
    k = cl / v
    rows.append((i, 0, ".", 1, dose, 1, 0, 1))
    for t in times:
        f = dose * ka / (v * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))
        y = f + sigma * f**power * rng.normal()
        y = max(y, 0.001)
        rows.append((i, t, f"{y:.5g}", 0, ".", 1, 0, 0))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_sim.csv")
with open(out, "w", newline="") as fh:
    csv.writer(fh).writerows(rows)
print("wrote", out, len(rows) - 1, "rows")
