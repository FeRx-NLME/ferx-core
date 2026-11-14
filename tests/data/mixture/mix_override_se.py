"""#2833 fixture P: a 2-class 1-cpt IV population whose [mixture] per-class overrides
are both interior, so their standard errors exist and can be anchored against NONMEM.

80 subjects x 7 observations, seed 2833. Class 1 (subjects 1-40): CL 1.2, omega_CL 0.25
(SD), proportional sigma 0.1. Class 2 (41-80): CL 2.5, omega_CL 0.55, sigma 0.2. V 10,
one 100-unit bolus at t = 0.

    python3 tests/data/mixture/mix_override_se.py > tests/data/mixture/mix_override_se.csv

The output is copied unchanged to nonmem_anchor/mix_override_se.csv.
"""
import math
import random

random.seed(2833)
times = [0.25, 0.5, 1, 2, 4, 8, 12]
N = 80
V = 10.0
amt = 100.0
print("ID,TIME,DV,AMT,EVID,CMT,WT")
for i in range(1, N + 1):
    c1 = i <= N // 2
    cl = 1.2 * math.exp(random.gauss(0, 0.25)) if c1 else 2.5 * math.exp(random.gauss(0, 0.55))
    sd = 0.1 if c1 else 0.2
    print(f"{i},0,0,{amt:g},1,1,70")
    for t in times:
        c = amt / V * math.exp(-cl / V * t)
        print(f"{i},{t:g},{c*(1+random.gauss(0,sd)):.6g},0,0,1,70")
