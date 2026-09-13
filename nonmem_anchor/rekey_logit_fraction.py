#!/usr/bin/env python3
"""Re-key nonmem_anchor/logit_fraction_oral.csv observations from CMT=2 (NONMEM's
CENTRAL, with an inert CMT=1 depot carrier) to CMT=1, which is what ferx's
single-state `ode(states=[central])` model expects. Writes to stdout.

    python3 nonmem_anchor/rekey_logit_fraction.py > data/logit_fraction_oral.csv
"""
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logit_fraction_oral.csv")

with open(SRC) as fh:
    for line in fh:
        parts = line.rstrip("\n").split(",")
        if parts[0] != "ID" and parts[4] == "0" and parts[5] == "2":
            parts[5] = "1"
        print(",".join(parts))
