#!/usr/bin/env python3
"""Synthetic data for a one-state turnover model with a weekly time-varying driver X.

For #3153 (the opening step's objective scale). Fitted by turnover.ferx in this directory.

    dh/dt = KIN * (X + OFFSET) - KOUT * h,   h(0) = H0,   DV = h + N(0, SD^2).

40 subjects, a weekly EVID=2 row carrying X (a slow random walk), 1-4 observations each.
The generator is plain Python and never runs ferx, so the data do not depend on the engine
they test. It is deterministic given SEED, so the committed CSV is reproducible.

Writes turnover.csv next to this script (columns ID,TIME,DV,EVID,MDV,CMT,X,H0).
`python3 sim_turnover_collapse.py OUT.csv SEED N` writes another draw (the issue's own is
`OUT.csv 14 539`).
"""
import csv, math, os, random, sys

KIN, KOUT, OFFSET, SD = 0.0195, 0.0514, 9.77, 0.497
SEED, N_SUBJECTS = 4, 40


def main(out, seed, n):
    rnd = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        h0 = min(10.1, max(5.1, rnd.gauss(7.6, 0.93)))
        x = max(2.8, rnd.gauss(8.0, 1.6))
        k = rnd.choice([1, 2, 3, 4, 4, 4, 4])
        obs_t = sorted(rnd.sample([0, 2, 4, 8, 12, 16, 20, 26], k))
        sched = [("E", float(w)) for w in range(27)] + [("O", float(t)) for t in obs_t]
        sched.sort(key=lambda s: (s[1], 0 if s[0] == "O" else 1))
        h, prev_t = h0, 0.0
        for kind, t in sched:
            ss = KIN * (x + OFFSET) / KOUT  # exact solution, X constant since the last row
            h = ss + (h - ss) * math.exp(-KOUT * (t - prev_t))
            prev_t = t
            row = dict(ID=i, TIME=t, DV=".", EVID=2, MDV=1, CMT=1, X=None, H0=round(h0, 2))
            if kind == "O":
                row.update(DV=round(h + rnd.gauss(0, SD), 4), EVID=0, MDV=0, X=round(x, 3))
            else:
                x = max(2.8, min(21.6, x + rnd.gauss(-0.03, 0.6)))
                row["X"] = round(x, 3)
            rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        main(os.path.join(os.path.dirname(os.path.abspath(__file__)), "turnover.csv"), SEED, N_SUBJECTS)
