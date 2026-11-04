"""Reference NPDE for the #2656 anchor, computed outside both engines (numpy).

NONMEM's `$TABLE NPDE ESAMPLE=` is no reference for this model: it disagrees with the
exact reference distribution far beyond Monte-Carlo noise, see
`tests/iiv_ruv_dense_nonmem_anchor.rs`. This draws the model directly, 400,000 replicates
per subject (eta on CL and V, the residual factor exp(eta_ruv), and each block's correlated
EPS), decorrelates by the Cholesky factor of the simulated covariance, and writes
`results/npde_<probe>[_noruv].csv`, one row per observation in data order. `_noruv` omits
the residual factor from the draw.

Run: python3 npde_oracle.py
"""

import csv
import math
import os
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 400_000
INV = NormalDist().inv_cdf


def pair_eps(rng, block_rows, free):
    (a, b) = block_rows
    z1, z2 = rng.normal(size=K), rng.normal(size=K)
    s1, s2, rho = 0.12, 0.20, 0.6
    return {a: s1 * z1, b: s2 * (rho * z1 + math.sqrt(1 - rho * rho) * z2)}


def triple_eps(rng, block_rows, free):
    rho, sd = 0.5, 0.16
    c = np.full((3, 3), rho)
    np.fill_diagonal(c, 1.0)
    z = rng.normal(size=(K, 3)) @ np.linalg.cholesky(c).T
    return {row: sd * z[:, i] for i, row in enumerate(block_rows)}


def subject_npde(rows, eps_of, rng, scaled):
    t = np.array([float(r["TIME"]) for r in rows])
    y = np.array([float(r["DV"]) for r in rows])
    blocks = [int(r["BLK"]) for r in rows]
    cl = 1.05 * np.exp(rng.normal(0, math.sqrt(0.08), K))
    v = 9.5 * np.exp(rng.normal(0, math.sqrt(0.05), K))
    s = np.exp(rng.normal(0, math.sqrt(0.25), K)) if scaled else np.ones(K)
    f = np.zeros((K, len(t)))
    for td in (0.0, 24.0):
        m = t >= td
        f[:, m] += 100 / v[:, None] * np.exp(-(cl / v)[:, None] * (t[m] - td))
    e = np.zeros((K, len(t)))
    for b in sorted(set(blocks)):
        members = [i for i, x in enumerate(blocks) if x == b]
        for row, draw in eps_of(rng, members, None).items():
            e[:, row] = draw
    sim = f * (1 + s[:, None] * e)
    mean = sim.mean(0)
    chol = np.linalg.cholesky(np.cov(sim.T))
    y_dec = np.linalg.solve(chol, y - mean)
    sim_dec = np.linalg.solve(chol, (sim - mean).T).T
    pde = np.clip((sim_dec < y_dec).mean(0), 1 / K, 1 - 1 / K)
    return [INV(float(p)) for p in pde]


def main():
    for probe, data, eps_of in (
        ("bs_p06", "pair.csv", pair_eps),
        ("cs_p05", "triple.csv", triple_eps),
    ):
        with open(os.path.join(HERE, data)) as f:
            obs = [r for r in csv.DictReader(f) if r["MDV"] == "0"]
        ids = sorted({int(r["ID"]) for r in obs})
        for scaled, tag in ((True, ""), (False, "_noruv")):
            rng = np.random.default_rng(2656)
            path = os.path.join(HERE, "results", f"npde_{probe}{tag}.csv")
            with open(path, "w") as out:
                out.write("ID,NPDE\n")
                for i in ids:
                    rows = [r for r in obs if int(r["ID"]) == i]
                    for z in subject_npde(rows, eps_of, rng, scaled):
                        out.write(f"{i},{z:.6f}\n")


if __name__ == "__main__":
    main()
