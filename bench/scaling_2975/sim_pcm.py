# Synthetic 1-cpt IV bolus, 28 subjects, rich sampling, WT/SEX on CL. No real data.
# Usage: python3 sim_pcm.py OUT.csv SEED
import sys

import numpy as np

rng = np.random.default_rng(int(sys.argv[2]))
rows = []
for i in range(1, 29):
    sex = int(rng.random() < 0.32)
    wt = float(np.clip(rng.normal(134, 35), 60, 230))
    cl = (
        0.32
        * (1 + 0.005 * (wt - 134.1))
        * (1 + 0.39 * (sex - 0.3214))
        * np.exp(rng.normal(0, 0.044**0.5))
    )
    v = 68 * np.exp(rng.normal(0, 0.083**0.5))
    rows.append(f"{i},0,2000,0,.,1,1,{sex},{wt:.1f}")
    for t in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 24]:
        c = 2000 / v * np.exp(-cl / v * t) * (1 + rng.normal(0, 0.036**0.5))
        rows.append(f"{i},{t},0,0,{c:.4f},0,1,{sex},{wt:.1f}")
with open(sys.argv[1], "w") as f:
    f.write("ID,TIME,AMT,RATE,DV,MDV,CMT,SEX,WT\n" + "\n".join(rows) + "\n")
