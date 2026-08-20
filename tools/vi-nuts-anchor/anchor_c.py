#!/usr/bin/env python
"""Anchor C — per-subject NUTS reference for the variational posterior.

VI_VALIDATION.md Anchor C. This is the only anchor that measures VI's *approximation* rather
than its optimizer, and the reason is what it compares against. Every other reference in the
document is Gaussian: AGQ's Laplace H^-1 is a Gaussian at the mode, so comparing q to it asks
"did VI find the right Gaussian?". NUTS samples the true posterior shape, so it asks the
different question "is a Gaussian right at all?" -- which is what docs/estimation/vi.qmd
publishes a number for ("understate posterior variance ... on the order of 20-25%").

Procedure: freeze (theta, Omega, sigma) at the AGQ estimate, sample p(eta_i | y_i, theta) with
NUTS for every subject, and compare that posterior's mean and covariance against the variational
(mu_i, S_i) ferx reports at the *same* parameters.

Deliberately not ferx's own `method = bayes`: it shares the likelihood code, which makes it an
internal consistency check rather than an anchor.

Usage:  tools/vi-nuts-anchor/run.sh
"""
import json
import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)

# The AGQ (n_agq = 9) estimate. It reproduces both ferx's and nlmixr2's FOCEI sigma to six
# decimals, which is what qualifies it as the common parameter vector: the comparison below is
# only about q, so both sides must sit at the same theta/Omega/sigma.
THETA = np.array([0.132687, 7.737464, 0.810901])  # TVCL, TVV, TVKA
OMEGA = np.array([0.028592, 0.009592, 0.336036])  # diag, in variance units
SIGMA = 0.010565  # proportional, SD scale
ETA_NAMES = ["eta_CL", "eta_V", "eta_KA"]


def read_warfarin(path):
    """(times, dv, mask, dose) padded to the widest subject, plus the subject ids."""
    rows = []
    with open(path) as fh:
        hdr = fh.readline().strip().split(",")
        ix = {k: i for i, k in enumerate(hdr)}
        for line in fh:
            f = line.strip().split(",")
            rows.append(f)
    ids = sorted({r[ix["ID"]] for r in rows}, key=int)
    obs, dose = {}, {}
    for r in rows:
        sid = r[ix["ID"]]
        evid = r[ix["EVID"]]
        if evid == "1":
            dose[sid] = dose.get(sid, 0.0) + float(r[ix["AMT"]])
        elif evid == "0" and r[ix["DV"]] != ".":
            obs.setdefault(sid, []).append((float(r[ix["TIME"]]), float(r[ix["DV"]])))
    n_max = max(len(v) for v in obs.values())
    t = np.zeros((len(ids), n_max))
    y = np.zeros((len(ids), n_max))
    m = np.zeros((len(ids), n_max))
    for i, sid in enumerate(ids):
        v = obs[sid]
        t[i, : len(v)] = [p[0] for p in v]
        y[i, : len(v)] = [p[1] for p in v]
        m[i, : len(v)] = 1.0
    d = np.array([dose[s] for s in ids])
    return ids, t, y, m, d


def predict(eta, t, dose):
    """1-cpt oral, single bolus into the depot, F = 1 -- the closed form ferx's one_cpt_oral
    and nlmixr2's linCmt() both evaluate. Written in jnp so NUTS can differentiate it."""
    cl = THETA[0] * jnp.exp(eta[..., 0:1])
    v = THETA[1] * jnp.exp(eta[..., 1:2])
    ka = THETA[2] * jnp.exp(eta[..., 2:3])
    ke = cl / v
    return dose[:, None] * ka / (v * (ka - ke)) * (jnp.exp(-ke * t) - jnp.exp(-ka * t))


def model(t, y, mask, dose):
    """Subjects are independent at fixed population parameters, so one joint sample over the
    stacked eta is equivalent to per-subject runs and lets NUTS see a single 30-dim geometry.

    `to_event(2)` rather than a `plate`: a plate at `dim=-2` combined with `to_event(1)` gives
    eta the shape `(n, 1, 3)`, whose spurious middle axis broadcasts each subject's eta against
    *every* subject's observations -- which shows up as all 10 subjects reporting an identical
    posterior. Declaring the whole `(n, 3)` block as one event keeps the prior exactly the same
    (Normal(0, sqrt(Omega)) independently per subject and per coordinate) with the right shape.
    """
    eta = numpyro.sample(
        "eta",
        dist.Normal(jnp.zeros((t.shape[0], 3)), jnp.sqrt(OMEGA)).to_event(2),
    )
    f = predict(eta, t, dose)
    sd = SIGMA * f
    # `mask` drops the padding rows without changing the geometry per subject.
    with numpyro.handlers.mask(mask=mask.astype(bool)):
        numpyro.sample("y", dist.Normal(f, sd), obs=y)


def ferx_posterior(path):
    """Parse the `eta_posterior` block: per-subject variational mean and full covariance."""
    txt = open(path, errors="replace").read()
    block = txt[txt.index("eta_posterior:") :]
    subs, cur, cov = [], None, None
    for ln in block.splitlines()[1:]:
        if ln and not ln.startswith(" "):
            break
        if (mm := re.match(r"\s*- id:\s*(\S+)", ln)) is not None:
            if cur:
                cur["cov"] = cov
                subs.append(cur)
            cur, cov = {"id": mm.group(1)}, []
        elif (mm := re.match(r"\s*mean:\s*\[(.*)\]", ln)) is not None:
            cur["mean"] = [float(x) for x in mm.group(1).split(",")]
        elif (mm := re.match(r"\s*-\s*\[(.*)\]", ln)) is not None and cov is not None:
            cov.append([float(x) for x in mm.group(1).split(",")])
    if cur:
        cur["cov"] = cov
        subs.append(cur)
    return subs


def laplace_cov(t, y, mask, dose, start):
    """H^-1 at the per-subject mode -- the *Gaussian* reference, kept alongside NUTS so the
    comparison can separate "q missed the right Gaussian" from "a Gaussian is the wrong shape".

    Started from the NUTS posterior mean and damped. An undamped Newton from eta = 0 diverges
    here: the depot term makes the log posterior very sharp in eta_KA, so a full step from a
    cold start overshoots into a region where the prediction underflows and the Hessian stops
    being positive definite (it returned NaN).
    """

    def nlp(eta, i):
        f = predict(eta[None, :], t[i : i + 1], dose[i : i + 1])[0]
        sd = SIGMA * f
        ll = -0.5 * jnp.sum(mask[i] * (jnp.log(sd**2) + (y[i] - f) ** 2 / sd**2))
        return -(ll - 0.5 * jnp.sum(eta**2 / OMEGA))

    out = []
    for i in range(t.shape[0]):
        fn = lambda e, i=i: nlp(e, i)
        e = jnp.array(start[i])
        for _ in range(50):
            g = jax.grad(fn)(e)
            h = jax.hessian(fn)(e)
            step = jnp.linalg.solve(h, g)
            # Backtrack until the objective actually decreases, so a bad curvature estimate
            # cannot throw the iterate out of the basin.
            a = 1.0
            for _ in range(30):
                cand = e - a * step
                if jnp.isfinite(fn(cand)) and fn(cand) <= fn(e):
                    break
                a *= 0.5
            e = e - a * step
            if jnp.linalg.norm(a * step) < 1e-12:
                break
        cov = jnp.linalg.inv(jax.hessian(fn)(e))
        out.append(np.array(cov))
    return np.stack(out)


def main():
    repo = os.environ.get("FERX_REPO", ".")
    results = os.path.expanduser(
        os.environ.get("FERX_VI_OUT", "~/.local/share/ferx-vi-validation/results")
    )
    q_file = os.environ.get("FERX_Q_FILE", os.path.join(results, "anchorc_q-fit.yaml"))
    if not os.path.exists(q_file):
        sys.exit(f"missing {q_file}: run the ferx side first (see run.sh)")

    # FERX_DATA / FERX_Q_FILE let the same comparison run on a thinned dataset. Warfarin's own
    # posterior is Gaussian to 0.1% (see the Laplace row below), so it cannot test the
    # understatement claim -- there is nothing for a Gaussian q to get wrong. A sparse variant
    # can.
    data = os.environ.get("FERX_DATA", os.path.join(repo, "data", "warfarin.csv"))
    ids, t, y, mask, dose = read_warfarin(data)
    print(f"data file: {data}")
    print(f"data: {len(ids)} subjects, {int(mask.sum())} observations")

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=20000, num_chains=4, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), jnp.array(t), jnp.array(y), jnp.array(mask), jnp.array(dose))
    mcmc.print_summary(exclude_deterministic=True)
    draws = np.array(mcmc.get_samples()["eta"])  # [draws, n_subj, 3]

    nuts_mean = draws.mean(axis=0)
    nuts_cov = np.stack([np.cov(draws[:, i, :].T) for i in range(len(ids))])

    ferx = ferx_posterior(q_file)
    vi_mean = np.array([s["mean"] for s in ferx])
    vi_cov = np.stack([np.array(s["cov"]) for s in ferx])
    lap_cov = laplace_cov(jnp.array(t), jnp.array(y), jnp.array(mask), jnp.array(dose), nuts_mean)

    out = {
        "ids": ids,
        "nuts_mean": nuts_mean.tolist(),
        "nuts_cov": nuts_cov.tolist(),
        "vi_mean": vi_mean.tolist(),
        "vi_cov": vi_cov.tolist(),
        "laplace_cov": lap_cov.tolist(),
    }
    dest = os.path.join(results, os.environ.get("FERX_ANCHOR_C_OUT", "anchor-c.json"))
    with open(dest, "w") as fh:
        json.dump(out, fh)

    print("\n=== Anchor C: variational posterior vs the NUTS reference ===")
    print(f"{'':<26}{'eta_CL':>12}{'eta_V':>12}{'eta_KA':>12}")
    md = np.median(np.abs(vi_mean - nuts_mean), axis=0)
    print(f"{'median |mean diff|':<26}" + "".join(f"{v:>12.2e}" for v in md))
    for label, ratio in (
        ("VI var / NUTS var", vi_cov / nuts_cov),
        ("Laplace var / NUTS var", lap_cov / nuts_cov),
        ("VI var / Laplace var", vi_cov / lap_cov),
    ):
        d = np.stack([np.diag(r) for r in ratio])
        print(f"{label + ' (median)':<26}" + "".join(f"{v:>12.3f}" for v in np.median(d, axis=0)))
        print(f"{'  range':<26}" + "".join(f"{lo:>6.2f}-{hi:<6.2f}" for lo, hi in zip(d.min(0), d.max(0))))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
