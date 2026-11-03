"""Simulate level_two_block.csv (#1963): the MBMA arms design of the binder's
Tier-1 fixtures, six studies of one subject each, three arms (OCC) at every time.

y = TVE0 * (1 + 0.01 * TIME^2) + B2[OCC] + PLACEBO[STUDY] + ETA_E0
    + TVEMAX * TIME / (TIME + TVET50) + eps

B2 and PLACEBO sum to zero over their levels. No intercept: TVE0 is read through
a factor of TIME, so the study random effect keeps the levels' mean.
Seeded, stdlib only. Run: python3 -I simulate_level_two_block.py > level_two_block.csv
"""
import random

rng = random.Random(1963)
TVE0, TVEMAX, TVET50 = 1.5, 3.0, 1.5
PLACEBO = [0.4, -0.3, 0.2, 0.1, -0.5]
PLACEBO.append(-sum(PLACEBO))
B2 = [0.3, -0.2]
B2.append(-sum(B2))
OMEGA, SIGMA_SD = 0.1, 0.5
TIMES = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]

print("ID,STUDY,OCC,TIME,DV,MDV")
for study in range(1, 7):
    eta = rng.gauss(0.0, OMEGA ** 0.5)
    for t in TIMES:
        for occ in (1, 2, 3):
            y = (TVE0 * (1 + 0.01 * t * t) + B2[occ - 1] + PLACEBO[study - 1] + eta
                 + TVEMAX * t / (t + TVET50) + rng.gauss(0.0, SIGMA_SD))
            print(f"{study},{study},{occ},{t:g},{y:.6f},0")
