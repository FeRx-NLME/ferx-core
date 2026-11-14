"""Deterministic circadian-baseline dataset for the #2720 `sin`/`cos` $PRED anchor.

y_ij = BASE_i * (1 + CIRC_ij) * (1 + eps_ij)
CIRC = AMP1_i * cos(2*pi*(t + SHIFT_i)/24 - PHS1) + AMP2 * sin(4*pi*(t + SHIFT_i)/24 - PHS2)
BASE_i = 120 * exp(eta_B),  AMP1_i = 0.12 * exp(eta_A),  SHIFT_i = eta_P (hours)
eta_B ~ N(0, 0.02), eta_A ~ N(0, 0.10), eta_P ~ N(0, 4.0), eps ~ N(0, 0.03^2)
PHS1 = 1.0 rad, AMP2 = 0.04, PHS2 = 0.5 rad

The phase random effect sits inside both trigonometric arguments, so the
derivative of the prediction in eta_P runs through cos and sin. Sampling times
are staggered per subject over 1.5 days, so every subject sees the full cycle at
different phases. Fixed seed, standard-library `random` only.
"""
import math
import random

random.seed(2720)

TVBASE, TVAMP1, PHS1, TVAMP2, PHS2 = 120.0, 0.12, 1.0, 0.04, 0.5
SD_B, SD_A, SD_P = math.sqrt(0.02), math.sqrt(0.10), 2.0
SIGMA_SD = 0.03
N_SUBJ = 30

rows = ["ID,TIME,DV,MDV"]
for i in range(1, N_SUBJ + 1):
    base = TVBASE * math.exp(random.gauss(0.0, SD_B))
    amp1 = TVAMP1 * math.exp(random.gauss(0.0, SD_A))
    shift = random.gauss(0.0, SD_P)
    start = (i % 4) * 1.5
    for k in range(10):
        t = start + 3.5 * k
        circ = amp1 * math.cos(2 * math.pi * (t + shift) / 24 - PHS1) + TVAMP2 * math.sin(
            4 * math.pi * (t + shift) / 24 - PHS2
        )
        y = base * (1 + circ) * (1 + random.gauss(0.0, SIGMA_SD))
        rows.append(f"{i},{t:g},{y:.4f},0")

out = "\n".join(rows) + "\n"
path = "nonmem_anchor/circadian_cos.csv"
with open(path, "w") as fh:
    fh.write(out)
print(f"wrote {path}: {len(rows) - 1} rows, {N_SUBJ} subjects")
