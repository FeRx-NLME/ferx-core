# Synthetic 2-cpt infusion data with IRREGULAR, subject-specific dosing/sampling (no real data).
# Usage: python3 sim_irreg.py OUT.csv N SEED
import sys

import numpy as np
from scipy.linalg import expm

out, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(seed)
TVCL, TVV1, TQ, TCW, TBW, TGA = 0.09, 1.04, 0.52, 0.91, 0.65, 1.77
rows = []
for i in range(1, N + 1):
    GA = rng.uniform(24, 42)
    BW = float(np.clip(rng.normal(400 + 90 * (GA - 24), 300), 450, 4800))
    CW = BW * rng.uniform(1.0, 1.15)
    CL = TVCL * np.exp(
        np.log(BW / 2180) * TBW + np.log(GA / 34) * TGA + rng.normal(0, 0.15**0.5)
    )
    V1 = TVV1 * (CW / 2120) ** TCW
    Q = TQ * CL
    V2 = V1
    K = np.array([[-(CL + Q) / V1, Q / V2], [Q / V1, -Q / V2]])
    nd = int(rng.integers(1, 6))
    ii = 24.0 if GA < 30 else 12.0
    doses = [round(k * ii + rng.uniform(-1.5, 1.5) * (k > 0), 2) for k in range(nd)]
    durs = [float(rng.choice([0.5, 1.0, 0.33])) for _ in doses]
    amt = round(15 * BW / 1000, 2)
    nobs = int(rng.integers(1, 4))
    last = doses[-1]
    obs = sorted({round(rng.uniform(0.2, last + ii), 2) for _ in range(nobs)})
    events = sorted(set(doses + [d + u for d, u in zip(doses, durs)] + obs))
    A = np.zeros(2)
    t = 0.0
    recs = []
    for te in events:
        rate = sum(amt / u for d, u in zip(doses, durs) if d <= t < d + u - 1e-12)
        M = np.zeros((3, 3))
        M[:2, :2] = K
        M[0, 2] = rate
        A = (expm(M * (te - t)) @ np.r_[A, 1.0])[:2]
        t = te
        if te in obs:
            f = A[0] / V1
            dv = max(f * (1 + rng.normal(0, 0.05**0.5)) + rng.normal(0, 1.4**0.5), 0.1)
            recs.append((te, 0, 0, round(dv, 3), 0))
    for d, u in zip(doses, durs):
        recs.append((d, amt, round(amt / u, 3), 0, 1))
    for tt, a, r, dv, mdv in sorted(recs, key=lambda x: (x[0], -x[4])):
        rows.append(f"{i},{tt},{a},{r},{dv},{mdv},{BW:.0f},{CW:.0f},{GA:.1f}")
with open(out, "w") as fh:
    fh.write("ID,TIME,AMT,RATE,DV,MDV,BW,CW,GA\n" + "\n".join(rows) + "\n")
print("wrote", len(rows), "rows")
