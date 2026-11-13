"""Generate small_real_iiv.csv, the #2832 rail-completion control fixture.

A real but small between-subject variance that a fit estimates **inside**
rail completion's probe span and **off** the guard, so the probe runs and must
find the guard uphill. 80 subjects, one 100-unit IV bolus, one-compartment PK,
16 samples each. CL is log-normal with 30% IIV around 1.0; V is log-normal
with a small but real IIV: omega^2 = 4e-5 (SD 0.0063; packed ln-SD -5.06,
0.94 above the -6 guard). 1% proportional residual error, so each subject's V
is pinned to ~0.25% and the small variance is identified.

    python3 small_real_iiv.py > small_real_iiv.csv
"""

import math
import random

random.seed(2832)
times = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
N = 80
amt = 100.0
sd = 0.01
print("ID,TIME,DV,AMT,EVID,CMT")
for i in range(1, N + 1):
    cl = 1.0 * math.exp(random.gauss(0, 0.3))
    v = 10.0 * math.exp(random.gauss(0, math.sqrt(4e-5)))
    print(f"{i},0,0,{amt:g},1,1")
    for t in times:
        c = amt / v * math.exp(-cl / v * t)
        print(f"{i},{t:g},{c*(1+random.gauss(0,sd)):.8g},0,0,1")
