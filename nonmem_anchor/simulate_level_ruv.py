"""Simulate level_ruv.csv for the #1879 NONMEM anchor (seed 1879).

Four studies of one subject each, three occasions (OCC) per subject, three
records per occasion. y = TVA * exp(-K * TIME), K = TVK * exp(ETA_K); the
proportional residual SD is PROP * exp(ERRSCALE[STUDY, OCC]) * exp(ETA_RUV).
ERRSCALE is sum-to-zero within each study. Run: python3 simulate_level_ruv.py
"""
import math
import os
import random

random.seed(1879)
TVA, TVK = 10.0, 0.15
OMEGA_K, OMEGA_RUV, PROP = 0.09, 0.16, 0.1
# Per study, the three occasions' levels; each row sums to zero.
LEVELS = [
    (0.30, -0.10, -0.20),
    (-0.25, 0.40, -0.15),
    (0.20, 0.10, -0.30),
    (0.50, -0.05, -0.45),
]
TIMES = {1: (0.5, 1.0, 2.0), 2: (4.0, 6.0, 8.0), 3: (12.0, 16.0, 24.0)}

rows = ["ID,STUDY,OCC,TIME,DV,MDV"]
for s, levels in enumerate(LEVELS, start=1):
    k = TVK * math.exp(random.gauss(0.0, math.sqrt(OMEGA_K)))
    eta_ruv = random.gauss(0.0, math.sqrt(OMEGA_RUV))
    for occ, times in TIMES.items():
        for t in times:
            f = TVA * math.exp(-k * t)
            sd = PROP * math.exp(levels[occ - 1]) * math.exp(eta_ruv)
            dv = f * (1.0 + sd * random.gauss(0.0, 1.0))
            rows.append(f"{s},{s},{occ},{t:g},{dv:.6f},0")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "level_ruv.csv"), "w") as fh:
    fh.write("\n".join(rows) + "\n")
