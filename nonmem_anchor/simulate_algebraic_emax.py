"""Deterministic Emax time-course dataset for the #811 $PRED anchor.

y_ij = E0_i - EMAX * t_j / (ET50 + t_j) + eps_ij
E0_i = 10 * exp(eta_i),  eta ~ N(0, 0.09),  EMAX = 6, ET50 = 2, eps ~ N(0, 0.5^2)

Fixed seed, standard-library `random` only, so the committed CSV is reproducible
from this script alone.
"""
import random

random.seed(811)

TVE0, EMAX, ET50 = 10.0, 6.0, 2.0
OMEGA_SD = 0.3          # sqrt(0.09)
SIGMA_SD = 0.5
TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
N_SUBJ = 30

rows = ["ID,TIME,DV,MDV"]
for i in range(1, N_SUBJ + 1):
    e0 = TVE0 * (2.718281828459045 ** (random.gauss(0.0, OMEGA_SD)))
    for t in TIMES:
        y = e0 - EMAX * t / (ET50 + t) + random.gauss(0.0, SIGMA_SD)
        rows.append(f"{i},{t:g},{y:.4f},0")

out = "\n".join(rows) + "\n"
path = "nonmem_anchor/algebraic_emax.csv"
with open(path, "w") as fh:
    fh.write(out)
print(f"wrote {path}: {len(rows) - 1} rows, {N_SUBJ} subjects")
