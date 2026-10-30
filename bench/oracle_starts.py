#!/usr/bin/env python3
"""Write start-point oracle variants of pk1/pk2 models: theta-at-truth, var-at-truth, all-at-truth."""
import os, re
B = os.path.dirname(os.path.abspath(__file__))

truth = {
    "pk1": {
        "theta": {"TVCL": 0.15, "TVV": 8.0, "TVKA": 1.0, "THETA_WT": 0.75},
        "omega": {"ETA_CL": 0.09, "ETA_V": 0.04, "ETA_KA": 0.30},
        "block": {},
        "sigma": {"PROP_ERR": 0.1},
    },
    "pk2": {
        "theta": {"TVCL": 4.0, "TVV1": 40.0, "TVQ": 8.0, "TVV2": 80.0, "TVKA": 1.2, "THETA_WT": 0.75, "THETA_CRCL": 0.4},
        "omega": {"ETA_V2": 0.10, "ETA_KA": 0.30},
        "block": {("ETA_CL", "ETA_V1"): (0.10, 0.05, 0.10)},
        "sigma": {"PROP_ERR": 0.10, "ADD_ERR": 0.05},
    },
}


def rewrite(src, dst, t, do_theta, do_var):
    out = []
    for l in open(src).read().splitlines():
        m = re.match(r"^(\s*)theta\s+([A-Za-z0-9_]+)\(\s*([^,]+),(.*)$", l)
        if m and do_theta and m.group(2) in t["theta"]:
            l = f"{m.group(1)}theta {m.group(2)}({t['theta'][m.group(2)]},{m.group(4)}"
        m = re.match(r"^(\s*)omega\s+([A-Za-z0-9_]+)\s*~\s*([-0-9.eE+]+)(.*)$", l)
        if m and do_var and m.group(2) in t["omega"]:
            l = f"{m.group(1)}omega {m.group(2)} ~ {t['omega'][m.group(2)]}{m.group(4)}"
        m = re.match(r"^(\s*)block_omega\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*=\s*\[.*\](.*)$", l)
        if m and do_var and (m.group(2), m.group(3)) in t["block"]:
            v = t["block"][(m.group(2), m.group(3))]
            l = f"{m.group(1)}block_omega ({m.group(2)}, {m.group(3)}) = [{v[0]}, {v[1]}, {v[2]}]{m.group(4)}"
        m = re.match(r"^(\s*)sigma\s+([A-Za-z0-9_]+)\s*~\s*([-0-9.eE+]+)(.*)$", l)
        if m and do_var and m.group(2) in t["sigma"]:
            l = f"{m.group(1)}sigma {m.group(2)} ~ {t['sigma'][m.group(2)]}{m.group(4)}"
        out.append(l)
    open(dst, "w").write("\n".join(out) + "\n")


for base in ["pk1", "pk2", "pk1_ode", "pk2_ode"]:
    key = "pk1" if base.startswith("pk1") else "pk2"
    t = truth[key]
    src = os.path.join(B, base + ".ferx")
    rewrite(src, os.path.join(B, base + "_thT.ferx"), t, True, False)
    rewrite(src, os.path.join(B, base + "_varT.ferx"), t, False, True)
    rewrite(src, os.path.join(B, base + "_allT.ferx"), t, True, True)
print("ok")
