"""Re-fit Pharmpy's simultaneous-stepwise `iivsearch_run5` ([CL,V]+[KA]) with
NONMEM from ferx's estimates for the same model, to tell a real optimum from
a ferx artifact: NONMEM's own run terminated with "sig. digits unreportable"
at OFV 655.47 from Pharmpy's inits (IIV_KA = 0.09), ferx reached 647.73 with
a small omega on KA.
"""
import json
import os

from pharmpy.modeling import read_model, set_initial_estimates
from pharmpy.tools import fit

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = read_model("iivsearch_sim/models/iivsearch_run5/model.ctl")
print("parameters:", model.parameters.names, flush=True)
ferx = {
    "TVCL": 0.128410,
    "TVV": 7.624160,
    "TVKA": 0.999402,
    "ETA_CL": 0.128270,
    "IIV_CL_IIV_V": 0.076327,
    "IIV_V": 0.076420,
    "ETA_V": 0.076420,
    "IIV_KA": 0.007872,
    "ETA_KA": 0.007872,
    "sigma": 0.023151,
}
inits = {name: ferx[name] for name in model.parameters.names if name in ferx}
print("setting:", inits, flush=True)
model = set_initial_estimates(model, inits)
res = fit(model, esttool="nonmem", name="verify_run5")
out = {
    "ofv": float(res.ofv),
    "minimization_successful": bool(res.minimization_successful),
    "estimates": {k: float(v) for k, v in res.parameter_estimates.items()},
}
print(json.dumps(out, indent=1), flush=True)
with open("verify_run5.json", "w") as f:
    json.dump(out, f, indent=1)
