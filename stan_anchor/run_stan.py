"""Stan reference posteriors for tests/eta_density_stan_anchor.rs (#1716).

Runs stan_anchor/warfarin_gamma_cl.stan on data/warfarin.csv twice: the density
model (`gaussian_cl = 0`, a Gamma clearance) and its Gaussian twin
(`gaussian_cl = 1`, `CL = TVCL·exp(η)`, whose posterior differs). Writes the
posterior mean, SD and Monte Carlo SE of each population parameter to
stan_anchor/results/warfarin_gamma_cl.json.

    python stan_anchor/run_stan.py   # from the repo root; needs cmdstanpy + CmdStan
"""
import csv
import json
import math
import pathlib
import sys

import cmdstanpy

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARAMS = ["TVCL", "TVV", "TVKA", "OMEGA_CL", "OMEGA_V", "OMEGA_KA", "PROP_ERR"]


def data():
    rows = list(csv.DictReader(open(ROOT / "data" / "warfarin.csv")))
    ids = sorted({int(r["ID"]) for r in rows})
    index = {s: k + 1 for k, s in enumerate(ids)}
    dose = [0.0] * len(ids)
    obs = []
    for r in rows:
        i = index[int(r["ID"])]
        if r["EVID"] == "1":
            dose[i - 1] += float(r["AMT"])
        elif r["MDV"] == "0":
            obs.append((i, float(r["TIME"]), float(r["DV"])))
    return {
        "N": len(ids),
        "M": len(obs),
        "id": [o[0] for o in obs],
        "time": [o[1] for o in obs],
        "dv": [o[2] for o in obs],
        "dose": dose,
        # The ferx model's [parameters] (tests/eta_density_stan_anchor.rs). TVV ≥ 1 and
        # TVKA ≥ 0.1 exclude the one-compartment oral flip-flop mode (KA ↔ CL/V, V
        # rescaled; here near KA ≈ 0.016), where two of four chains settled at the wider
        # bounds (R̂ 1.8); the posterior both samplers target is this truncated one.
        "theta0": [0.2, 10.0, 1.5],
        "theta_lo": [0.001, 1.0, 0.1],
        "theta_hi": [10.0, 500.0, 50.0],
        "omega0": [0.09, 0.04, 0.30],
        "sigma0": 0.2,
    }


def run(gaussian_cl, seed):
    model = cmdstanpy.CmdStanModel(stan_file=ROOT / "stan_anchor" / "warfarin_gamma_cl.stan")
    d = dict(data(), gaussian_cl=gaussian_cl)
    fit = model.sample(
        data=d, chains=4, parallel_chains=1, iter_warmup=2000, iter_sampling=6000,
        seed=seed, adapt_delta=0.95, show_progress=False,
        # Every chain starts at the declared initial values, as ferx's do.
        inits={"log_theta": [math.log(x) for x in d["theta0"]],
               "log_sigma": math.log(d["sigma0"]),
               "omega": d["omega0"]},
    )
    s = fit.summary()
    out = {}
    for p in PARAMS:
        row = s.loc[p]
        out[p] = {
            "mean": float(row["Mean"]),
            "sd": float(row["StdDev"]),
            "mcse": float(row["MCSE"]),
            "ess_bulk": float(row.get("ESS_bulk", row.get("N_Eff", float("nan")))),
            "rhat": float(row.get("R_hat", float("nan"))),
        }
    div = int(fit.divergences.sum()) if hasattr(fit, "divergences") else -1
    return out, div


def main():
    density, div_d = run(0, 1716)
    twin, div_t = run(1, 1716)
    res = {
        "provenance": f"CmdStan {cmdstanpy.cmdstan_version()}, cmdstanpy {cmdstanpy.__version__}; "
        "4 chains x 6000 draws after 2000 warmup, adapt_delta 0.95, seed 1716",
        "divergences": {"density": div_d, "gaussian_twin": div_t},
        "density": density,
        "gaussian_twin": twin,
    }
    path = ROOT / "stan_anchor" / "results" / "warfarin_gamma_cl.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=2) + "\n")
    json.dump(res, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
