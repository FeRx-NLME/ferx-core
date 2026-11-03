"""Dataset for the #979 `block_sigma` x IOV anchor.

Writes `block_sigma_iov.csv` next to this script: one-compartment IV bolus,
40 subjects, three occasions (`OCC` 1-3), a dose of 100 at the start of each
(t = 0, 24, 48) so occasions 2 and 3 open on residual drug, and a total
(FREE = 0) / unbound (FREE = 1) pair at 1, 4, 12 and 20 h after each dose.
The pair shares an `L2` id. True values: CL = 1.0, V = 10, omega_CL = 0.09,
an IOV kappa on CL with variance 0.04, and two proportional sigmas
(0.10, 0.20) correlated rho = 0.6 across the pair.

The DV values only need to be plausible: the anchor compares two engines
scoring the same data, so the generator is stdlib (no numpy) and need not
reproduce either engine's random stream.
"""

import csv
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N_SUBJ = 40
CL, V, OMEGA_CL, OMEGA_IOV = 1.0, 10.0, 0.09, 0.04
S_TOT, S_UNB, RHO = 0.10, 0.20, 0.6
DOSES = [0.0, 24.0, 48.0]
SAMPLES = [1.0, 4.0, 12.0, 20.0]


def conc(t, cl_of_occ):
    """Superposed bolus decay with occasion-wise CL: the drug of each earlier
    dose decays under the CL of whichever occasion each stretch falls in."""
    total = 0.0
    for td in DOSES:
        if t < td:
            continue
        log_c = math.log(100.0 / V)
        for k, start in enumerate(DOSES):
            end = DOSES[k + 1] if k + 1 < len(DOSES) else math.inf
            lo, hi = max(td, start), min(t, end)
            if hi > lo:
                log_c -= cl_of_occ[k] / V * (hi - lo)
        total += math.exp(log_c)
    return total


def main():
    rng = random.Random(979)
    rows = []
    for i in range(1, N_SUBJ + 1):
        eta = rng.gauss(0, math.sqrt(OMEGA_CL))
        cls = [CL * math.exp(eta + rng.gauss(0, math.sqrt(OMEGA_IOV))) for _ in DOSES]
        l2 = 0
        for occ, td in enumerate(DOSES, start=1):
            rows.append([i, td, 0, 1, 100, 1, 1, occ, 0, 0])
            for dt in SAMPLES:
                t = td + dt
                f = conc(t, cls)
                z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
                e_tot = S_TOT * z1
                e_unb = S_UNB * (RHO * z1 + math.sqrt(1 - RHO * RHO) * z2)
                l2 += 1
                rows.append([i, t, round(f * (1 + e_tot), 6), 0, 0, 1, 0, occ, 0, l2])
                rows.append([i, t, round(f * (1 + e_unb), 6), 0, 0, 1, 0, occ, 1, l2])
    with open(os.path.join(HERE, "block_sigma_iov.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "TIME", "DV", "EVID", "AMT", "CMT", "MDV", "OCC", "FREE", "L2"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
