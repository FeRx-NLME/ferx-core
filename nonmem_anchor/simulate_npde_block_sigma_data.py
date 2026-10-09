"""Datasets for the #1733 NPDE `block_sigma` anchor.

Writes two NONMEM-format CSVs next to this script:

* `npde_block_sigma_a.csv` (arm A): one-compartment IV bolus, doses of 100 at
  t = 0 and t = 12 (the second lands on residual drug), seven observations per
  subject, `combined(PROP, ADD)` residual with sigma = (0.2, 1.0) and a
  within-observation correlation rho = -0.9.
* `npde_block_sigma_b.csv` (arm B): same PK, a total (FREE = 0) / unbound
  (FREE = 1) pair at four times, the two proportional sigmas (0.05, 0.30)
  correlated rho = -0.8 across the pair; the pair shares an `L2` id.

The DV values only need to be plausible: the anchor compares the reference
distributions both tools build *from the same DV*, so the generator is stdlib
(no numpy) and need not reproduce either engine's stream.
"""

import csv
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N_SUBJ = 30
CL, V, OMEGA_CL = 1.0, 10.0, 0.09
DOSES = [0.0, 12.0]


def conc(t, cl):
    k = cl / V
    return sum(100.0 / V * math.exp(-k * (t - td)) for td in DOSES if t >= td)


def corr_pair(rng, s1, s2, rho):
    z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
    return s1 * z1, s2 * (rho * z1 + math.sqrt(1 - rho * rho) * z2)


def dose_rows(sid, extra):
    return [[sid, td, 0, 1, 100, 1, 1] + extra for td in DOSES]


def arm_a(rng):
    rows = []
    for i in range(1, N_SUBJ + 1):
        cl = CL * math.exp(rng.gauss(0, math.sqrt(OMEGA_CL)))
        recs = [(t, None) for t in [1.0, 2.0, 4.0, 8.0, 13.0, 16.0, 24.0]]
        recs += [(td, "dose") for td in DOSES]
        for t, kind in sorted(recs, key=lambda r: (r[0], r[1] is None)):
            if kind == "dose":
                rows.append([i, t, 0, 1, 100, 1, 1])
                continue
            f = conc(t, cl)
            e_prop, e_add = corr_pair(rng, 0.2, 1.0, -0.9)
            rows.append([i, t, round(f + f * e_prop + e_add, 6), 0, 0, 1, 0])
    return ["ID", "TIME", "DV", "EVID", "AMT", "CMT", "MDV"], rows


def arm_b(rng):
    rows = []
    for i in range(1, N_SUBJ + 1):
        cl = CL * math.exp(rng.gauss(0, math.sqrt(OMEGA_CL)))
        rows.append([i, 0.0, 0, 1, 100, 1, 1, 0, 0])
        for l2, t in enumerate([1.0, 4.0, 13.0, 24.0], start=1):
            if t > 12.0 and rows[-1][1] < 12.0:
                rows.append([i, 12.0, 0, 1, 100, 1, 1, 0, 0])
            f = conc(t, cl)
            e_tot, e_unb = corr_pair(rng, 0.05, 0.30, -0.8)
            rows.append([i, t, round(f * (1 + e_tot), 6), 0, 0, 1, 0, 0, l2])
            rows.append([i, t, round(f * (1 + e_unb), 6), 0, 0, 1, 0, 1, l2])
    return ["ID", "TIME", "DV", "EVID", "AMT", "CMT", "MDV", "FREE", "L2"], rows


def write(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    write("npde_block_sigma_a.csv", *arm_a(random.Random(1733)))
    write("npde_block_sigma_b.csv", *arm_b(random.Random(17332)))
