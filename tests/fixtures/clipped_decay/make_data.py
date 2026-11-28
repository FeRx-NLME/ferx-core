"""Generates clipped.csv: python3 make_data.py clipped.csv 41 150 0.15 (plain Python 3, written
against 3.12). It draws the 40 subjects of #3159 and writes IDs 1, 2, 14, 17, 20 and 27, or the IDs
given after the noise SD: stall_iiv.csv is python3 make_data.py stall_iiv.csv 45 150 0.15 40 (#3210)."""
import csv
import math
import random
import sys

out, seed, nobs, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
ids = [int(a) for a in sys.argv[5:]] or [1, 2, 14, 17, 20, 27]
rnd = random.Random(seed)
rows = []
for i in range(1, 41):
    eb, et = rnd.gauss(0, 0.2), rnd.gauss(0, 0.3)
    for occ in (1, 2):
        k = rnd.gauss(0, 0.35)
        t0 = (occ - 1) * 100.0
        base, th = 10 * math.exp(eb), 4 * math.exp(et + k)
        rows.append(dict(ID=i, TIME=t0, DV='.', EVID=4, AMT=1, CMT=1, MDV=1, OCC=occ))
        for j in range(1, nobs + 1):
            dt = j * 24.5 / nobs
            y = max(0.0, base * math.exp(-0.15 * dt) - th)
            rows.append(dict(ID=i, TIME=round(t0 + dt, 4), DV='%.4f' % (y + rnd.gauss(0, sd)), EVID=0, AMT='.', CMT=1, MDV=0, OCC=occ))
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(r for r in rows if r['ID'] in ids)
