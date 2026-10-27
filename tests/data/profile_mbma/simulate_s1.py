"""Synthetic MBMA for the #1065 speed / dimension measurement S1 (the plan's `gen.py`, M6).

S studies x 2 arms (TRT 0 placebo, 1 drug) x T visits; ID = study. Arm means:

  y = TVE0 + B + PLACEBO[study, visit] - TRT * TVEMAX * exp(ETA) * t / (t + ET50) + eps

with a study baseline B (both arms) and an unstructured placebo per study and visit. B is
what lets the `[STUDY, TIME]` block absorb a study-level η and split per study (#2326):
without one, the drug-arm-only ETA is not absorbed (#2131) and the 800 levels are one
coupled group. Fixed seed, standard-library `random`
only. Run from the repo root:

  python3 tests/data/profile_mbma/simulate_s1.py 100 8 > tests/data/profile_mbma/s100t8.csv
  python3 tests/data/profile_mbma/simulate_s1.py 10 8 > tests/data/profile_mbma/s10t8.csv
"""
import math
import random
import sys

S, T = int(sys.argv[1]), int(sys.argv[2])
random.seed(1065)
TVE0, TVEMAX, ET50, OM, OM_B, SD = 5.0, 1.5, 2.0, 0.3, 0.1, 0.2
print("ID,STUDY,TRT,TIME,DV,MDV")
for s in range(1, S + 1):
    pl = [random.gauss(0, 0.4) for _ in range(T)]
    eta = random.gauss(0, math.sqrt(OM))
    b = random.gauss(0, math.sqrt(OM_B))
    for trt in (0, 1):
        for t in range(1, T + 1):
            f = TVE0 + b + pl[t - 1] - trt * TVEMAX * math.exp(eta) * t / (t + ET50)
            print(f"{s},{s},{trt},{t},{f + random.gauss(0, SD):.6f},0")
