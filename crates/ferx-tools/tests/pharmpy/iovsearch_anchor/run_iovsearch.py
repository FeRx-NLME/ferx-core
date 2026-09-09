"""Pharmpy 2.2.0 iovsearch on the IIV-only base, with NONMEM 7.5.1.

Writes the trajectory (per-step ranking tables, every model's OFV and
description, the final model) so it can be compared with `ferx iovsearch`
on the same model and data.
"""
import json
import os

from pharmpy.modeling import read_model
from pharmpy.tools import fit, run_iovsearch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = read_model("base.ctl")
print("fitting the base with NONMEM ...", flush=True)
res = fit(model, esttool="nonmem", name="base_fit")
print("base OFV", res.ofv, flush=True)
print("base estimates", dict(res.parameter_estimates), flush=True)

VARIANTS = [
    ("disjoint", dict(distribution="disjoint")),
    ("same_as_iiv", dict(distribution="same-as-iiv")),
]

out = {
    "base_ofv": float(res.ofv),
    "base_estimates": {k: float(v) for k, v in res.parameter_estimates.items()},
}
for variant, kwargs in VARIANTS:
    print(f"running iovsearch ({variant}) ...", flush=True)
    rs = run_iovsearch(
        model=model,
        results=res,
        column="OCC",
        rank_type="bic",
        esttool="nonmem",
        name=f"iovsearch_{variant}",
        **kwargs,
    )
    st = rs.summary_tool.reset_index()
    sm = rs.summary_models.reset_index()
    out[variant] = {
        "kwargs": kwargs,
        "summary_tool": json.loads(st.to_json(orient="records")),
        "summary_models": json.loads(sm.to_json(orient="records")),
        "final_model_code": rs.final_model.code,
        "final_description": rs.final_model.description,
        "final_ofv": float(rs.final_results.ofv) if rs.final_results is not None else None,
        "final_estimates": {k: float(v) for k, v in rs.final_results.parameter_estimates.items()}
        if rs.final_results is not None
        else None,
    }
    print(json.dumps(out[variant], indent=1)[:6000], flush=True)
    with open("pharmpy_iovsearch.json", "w") as f:
        json.dump(out, f, indent=1)

print("done", flush=True)
