"""Arm-level data with time-ordered occasions, for tests/dw_advice_sequential_occasions.rs (#2858).

24 subjects on one running clock, a straight line in time with a subject-level
intercept. `OCC` numbers the occasions in time order and `UNIT` repeats `ID`, so
a `correlation ... by` line has a column that is constant within a subject. `PER`
groups the occasions in pairs, for a nested `PER > OCC` model.

- sparse.csv: 8 occasions, one row each (192 rows).
- rich.csv:   4 occasions, three rows each (288 rows).

The residuals are additive, SD 0.5, and AR(1) with rho = 0.7 across the whole
subject, so the dependence runs **across** occasions.

Run from this directory: `python3 gen.py` rewrites both files (seed 2858).
"""
import math
import random

random.seed(2858)


def write(name, n_occ, per_occ):
    rows = ["ID,TIME,DV,MDV,OCC,UNIT,PER"]
    for i in range(1, 25):
        e0 = 10.0 + random.gauss(0, 1.0)
        e = random.gauss(0, 1)
        for o in range(n_occ):
            for k in range(per_occ):
                t = o * per_occ + k + 1
                e = 0.7 * e + math.sqrt(1 - 0.49) * random.gauss(0, 1)
                rows.append(f"{i},{t},{e0 + 0.5 * t + 0.5 * e:.5f},0,{o + 1},{i},{o // 2 + 1}")
    open(name, "w").write("\n".join(rows) + "\n")


write("sparse.csv", 8, 1)
write("rich.csv", 4, 3)
