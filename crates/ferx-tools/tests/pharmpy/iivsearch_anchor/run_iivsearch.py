"""Pharmpy 2.2.0 iivsearch on the diagonal-IIV base, with NONMEM 7.5.1.

Writes the trajectory (per-step ranking tables, every model's OFV and
description, the final model) so it can be compared with `ferx iivsearch`
on the same model and data.
"""
import json
import os

from pharmpy.modeling import read_model
from pharmpy.tools import fit, run_iivsearch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = read_model("base.ctl")
print("fitting the base with NONMEM ...", flush=True)
res = fit(model, esttool="nonmem", name="base_fit")
print("base OFV", res.ofv, flush=True)
print("base estimates", dict(res.parameter_estimates), flush=True)

VARIANTS = [
    ("td", dict(algorithm="top_down_exhaustive")),
    (
        "bu",
        dict(
            algorithm="bottom_up_stepwise",
            search_space="IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,@IIV)",
        ),
    ),
    (
        "sim",
        dict(
            algorithm="simultaneous_stepwise",
            search_space="IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,@IIV)",
        ),
    ),
]

out = {
    "base_ofv": float(res.ofv),
    "base_estimates": {k: float(v) for k, v in res.parameter_estimates.items()},
}
for variant, kwargs in VARIANTS:
    print(f"running iivsearch ({variant}) ...", flush=True)
    rs = run_iivsearch(
        model=model,
        results=res,
        rank_type="bic",
        esttool="nonmem",
        name=f"iivsearch_{variant}",
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
    with open("pharmpy_iivsearch.json", "w") as f:
        json.dump(out, f, indent=1)

print("done", flush=True)
