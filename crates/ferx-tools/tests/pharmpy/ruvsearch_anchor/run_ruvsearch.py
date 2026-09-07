"""Pharmpy 2.2.0 ruvsearch on the warfarin proportional base, with NONMEM 7.5.1.

Writes the trajectory (cwres_models, summary_tool per iteration, final model)
so it can be compared with `ferx ruvsearch` on the same model and data.
"""
import json
import os
import sys

from pharmpy.modeling import read_model
from pharmpy.tools import fit, run_ruvsearch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = read_model("base.ctl")
print("fitting the base with NONMEM ...", flush=True)
res = fit(model, esttool="nonmem", name="base_fit")
print("base OFV", res.ofv, flush=True)
print("base estimates", dict(res.parameter_estimates), flush=True)

out = {}
for variant, skip in [("all", []), ("no_iiv", ["IIV_on_RUV"])]:
    print(f"running ruvsearch ({variant}) ...", flush=True)
    rs = run_ruvsearch(
        model=model,
        results=res,
        groups=4,
        p_value=0.001,
        skip=skip,
        max_iter=3,
        esttool="nonmem",
        name=f"ruvsearch_{variant}",
    )
    cw = rs.cwres_models.reset_index()
    cw["parameters"] = cw["parameters"].apply(lambda d: {k: float(v) for k, v in d.items()})
    st = rs.summary_tool.reset_index() if rs.summary_tool is not None else None
    out[variant] = {
        "base_ofv": float(res.ofv),
        "cwres_models": json.loads(cw.to_json(orient="records")),
        "summary_tool": json.loads(st.to_json(orient="records")) if st is not None else None,
        "final_model_code": rs.final_model.code,
        "final_ofv": float(rs.final_results.ofv) if rs.final_results is not None else None,
        "final_estimates": {k: float(v) for k, v in rs.final_results.parameter_estimates.items()}
        if rs.final_results is not None
        else None,
    }
    print(json.dumps(out[variant], indent=1)[:4000], flush=True)

with open("pharmpy_ruvsearch.json", "w") as f:
    json.dump(out, f, indent=1)
print("done", flush=True)
