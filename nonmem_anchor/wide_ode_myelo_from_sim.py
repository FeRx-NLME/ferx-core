#!/usr/bin/env python3
"""Turn NONMEM's `$SIMULATION` table into the #1661 anchor dataset (pure stdlib).

Reads `wide_ode_myelo_sim.tab` (written by `wide_ode_myelo_sim.ctl`, no header, the
template's columns in order) and writes `wide_ode_myelo.csv`, which both NONMEM
(`wide_ode_myelo.ctl`) and ferx (`wide_ode_myelo_fit.ferx`) fit.
"""
COLS = ["ID", "TIME", "AMT", "CMT", "EVID", "MDV", "DV", "V1I", "K10I", "K12I", "K21I"]
INTS = {"ID", "CMT", "EVID", "MDV"}

with open("wide_ode_myelo_sim.tab") as f, open("wide_ode_myelo.csv", "w") as out:
    out.write(",".join(COLS) + "\n")
    for line in f:
        vals = [float(x) for x in line.split()]
        if len(vals) != len(COLS):
            continue
        cells = []
        for name, v in zip(COLS, vals):
            if name in INTS:
                cells.append(str(int(round(v))))
            elif name in ("TIME", "AMT"):
                cells.append(f"{v:g}")
            else:
                cells.append(f"{v:.6f}")
        out.write(",".join(cells) + "\n")
