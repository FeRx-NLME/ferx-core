"""Generate mix_override_collapse.csv, the #2809 guard-hold fixture.

60 subjects, one 100-unit IV bolus, one-compartment PK, two latent classes of
30 subjects each. Class 1 has between-subject variability on CL (30%
log-normal around 1.2); class 2 has **none** (CL fixed at 2.5). V = 10, 10%
proportional residual error. A `[mixture]` model that estimates a class-2
override `omega(2) ETA_CL` collapses it onto ferx's internal lower guard.

    python3 mix_override_collapse.py > mix_override_collapse.csv
"""

import math
import random

random.seed(2809)
times = [0.25, 0.5, 1, 2, 4, 8, 12]
N = 60
V = 10.0
amt = 100.0
sd = 0.1
print("ID,TIME,DV,AMT,EVID,CMT,WT")
for i in range(1, N + 1):
    cls = 1 if i <= N // 2 else 2
    cl = 1.2 * math.exp(random.gauss(0, 0.3)) if cls == 1 else 2.5
    print(f"{i},0,0,{amt:g},1,1,70")
    for t in times:
        c = amt / V * math.exp(-cl / V * t)
        print(f"{i},{t:g},{c*(1+random.gauss(0,sd)):.6g},0,0,1,70")
