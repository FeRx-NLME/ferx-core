"""Build nested_levels_3_ar1.csv for #1082 PR5 (anchor V7) from ../nested_levels_3.csv.

Adds, per row: VIS, the row's visit index inside its arm (1..5, in TIME order), and
T1..T5, the arm's visit times (car1's Cholesky factor reads them). Every other column is
copied unchanged. NONMEM reads ARMID as L2 (one shared EPS(1..5) vector per arm), which
needs each arm's records contiguous: nested_levels_3_ar1.csv keeps the file's arm order.

The ferx twin reads nested_levels_3_ar1_ferx.csv: the same rows sorted by TIME within ID,
so the four arms of a sponsor interleave and a row's AR(1) lag inside its arm is not its
distance in the file."""
import csv
import os

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "nested_levels_3.csv")) as f:
    rows = list(csv.DictReader(f))
arms = {}
for r in rows:
    arms.setdefault(r["ARMID"], []).append(float(r["TIME"]))
out = []
for r in rows:
    times = sorted(arms[r["ARMID"]])
    assert len(times) == 5 and len(set(times)) == 5, r["ARMID"]
    r = dict(r)
    r["VIS"] = str(times.index(float(r["TIME"])) + 1)
    for k in range(5):
        r[f"T{k + 1}"] = repr(times[k]).rstrip("0").rstrip(".") if times[k] % 1 else str(int(times[k]))
    out.append(r)
cols = list(rows[0].keys()) + ["VIS", "T1", "T2", "T3", "T4", "T5"]
with open(os.path.join(here, "nested_levels_3_ar1.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(out)

by_time = sorted(out, key=lambda r: (int(r["ID"]), float(r["TIME"])))  # stable: arm order at ties
with open(os.path.join(here, "nested_levels_3_ar1_ferx.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(by_time)
