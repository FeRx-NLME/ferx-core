#!/usr/bin/env python3
"""evalvariant.py MODEL FITYAML OUT [key=value ...]
Write OUT = MODEL with [parameters] initial values replaced by the estimates in
FITYAML (theta / diagonal omega / sigma), and [fit_options] overridden by the
key=value pairs (e.g. maxiter=0 ode_reltol=1e-10). Only diagonal omegas and
block_omega of size 2 (reads omega covariance lines if present) are handled.
"""
import re, sys

model, fy, out = sys.argv[1], sys.argv[2], sys.argv[3]
kv = [a.split("=", 1) for a in sys.argv[4:]]

# --- tiny YAML reader for the sections we need
est = {"theta": {}, "omega": {}, "sigma": {}, "omega_cov": {}}
sec = None
cur = None
for line in open(fy):
    if re.match(r"^[a-z_]+:", line):
        sec = line.split(":")[0]
        cur = None
        continue
    m = re.match(r"^  ([A-Za-z0-9_]+):\s*$", line)
    if m and sec in ("theta", "omega", "sigma"):
        cur = m.group(1)
        continue
    m = re.match(r"^    (estimate|variance|covariance|sd):\s*([-0-9.eE+]+)", line)
    if m and cur and sec in ("theta", "omega", "sigma"):
        key, val = m.group(1), float(m.group(2))
        if sec == "theta" and key == "estimate":
            est["theta"][cur] = val
        elif sec == "omega" and key == "variance":
            est["omega"][cur] = val
        elif sec == "omega" and key == "covariance":
            est["omega_cov"][cur] = val
        elif sec == "sigma" and key == "estimate":
            est["sigma"][cur] = val

lines = open(model).read().splitlines()
res = []
in_fo = False
keys = {k for k, _ in kv}
for l in lines:
    s = l.strip()
    m = re.match(r"^(\s*)theta\s+([A-Za-z0-9_]+)\(\s*([^,]+),(.*)$", l)
    if m and m.group(2) in est["theta"]:
        l = f"{m.group(1)}theta {m.group(2)}({est['theta'][m.group(2)]!r},{m.group(4)}"
    m = re.match(r"^(\s*)omega\s+([A-Za-z0-9_]+)\s*~\s*([-0-9.eE+]+)(.*)$", l)
    if m and m.group(2) in est["omega"]:
        l = f"{m.group(1)}omega {m.group(2)} ~ {est['omega'][m.group(2)]!r}{m.group(4)}"
    m = re.match(r"^(\s*)block_omega\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*=\s*\[.*\](.*)$", l)
    if m:
        a, b = m.group(2), m.group(3)
        va = est["omega"].get(a)
        vb = est["omega"].get(b)
        cov = est["omega_cov"].get(f"{b}__{a}", est["omega_cov"].get(f"{a}__{b}"))
        if va is not None and vb is not None and cov is not None:
            l = f"{m.group(1)}block_omega ({a}, {b}) = [{va!r}, {cov!r}, {vb!r}]{m.group(4)}"
    m = re.match(r"^(\s*)sigma\s+([A-Za-z0-9_]+)\s*~\s*([-0-9.eE+]+)(.*)$", l)
    if m and m.group(2) in est["sigma"]:
        l = f"{m.group(1)}sigma {m.group(2)} ~ {est['sigma'][m.group(2)]!r}{m.group(4)}"
    if s.startswith("["):
        if in_fo:
            for k, v in kv:
                res.append(f"  {k} = {v}")
            in_fo = False
        if s == "[fit_options]":
            in_fo = True
        res.append(l)
        continue
    if in_fo and "=" in s and s.split("=")[0].strip() in keys:
        continue
    res.append(l)
if in_fo:
    for k, v in kv:
        res.append(f"  {k} = {v}")
open(out, "w").write("\n".join(res) + "\n")
print("theta", est["theta"], "omega", est["omega"], "sigma", est["sigma"])
