"""Data and summary for the #2163 anchor (mixture_sim_class.ctl). Standard library only.

    python3 make_mixture_sim_class.py data              # writes mixture_sim_class.csv
    python3 make_mixture_sim_class.py summary <tab>     # writes results/mixture_sim_class_summary.csv

The dataset is a formula, not a draw, so tests/mixture_simulate_nonmem_anchor.rs rebuilds
it in memory, and the 20,000-subject CSV is not committed (.gitignore): subject i (1-based) has
WT = 50 + 0.003 * (i - 0.5), a dose of 100 into CMT 1 at t = 0 and observations at TIMES.

The summary, per WT stratum (three equal blocks of subjects, by WT), per observation time
and per quantity (IPRED, DV), holds the empirical quantiles at PROBS:
`sorted(values)[floor(p * n)]`. It also holds, per stratum, NONMEM's simulated class-1
share (from MIXN) next to the mean P(1) of the stratum, which certifies that NONMEM drew
classes at the mixing probabilities before its quantiles are used as the reference.
"""
import csv
import math
import sys

N = 20000
TIMES = [0.5, 1.0, 2.0, 4.0, 8.0]
PROBS = [0.1, 0.25, 0.5, 0.75, 0.9]
MIXL, BWT = -0.85, 0.05


def wt(i):
    return 50 + 0.003 * (i - 0.5)


def stratum(i):
    return 3 * (i - 1) // N


def p1(w):
    return 1 / (1 + math.exp(-(MIXL + BWT * (w - 70))))


def data():
    with open("mixture_sim_class.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["ID", "TIME", "DV", "AMT", "EVID", "CMT", "WT"])
        for i in range(1, N + 1):
            w = f"{wt(i):.4f}"
            out.writerow([i, 0, 0, 100, 1, 1, w])
            for t in TIMES:
                out.writerow([i, t, 0, 0, 0, 1, w])


def summary(tab):
    with open(tab) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    header = lines[1] if lines[0][0] == "TABLE" else lines[0]
    rows = [dict(zip(header, map(float, r))) for r in lines[lines.index(header) + 1:]]
    obs = [r for r in rows if r["EVID"] == 0]
    assert len(obs) == N * len(TIMES), len(obs)
    out = [["kind", "stratum", "time", "quantity", "p", "value", "n"]]
    for s in range(3):
        mine = [r for r in obs if stratum(int(r["ID"])) == s]
        first = [r for r in mine if r["TIME"] == TIMES[0]]
        share = sum(r["MIXN"] == 1 for r in first) / len(first)
        mean_p1 = sum(p1(wt(int(r["ID"]))) for r in first) / len(first)
        out.append(["class1_share", s, "", "", "", repr(share), len(first)])
        out.append(["mean_p1", s, "", "", "", repr(mean_p1), len(first)])
        for t in TIMES:
            at = [r for r in mine if r["TIME"] == t]
            for q in ["IPRED", "DV"]:
                v = sorted(r[q] for r in at)
                for p in PROBS:
                    out.append(["quantile", s, t, q, p, repr(v[math.floor(p * len(v))]), len(v)])
    with open("results/mixture_sim_class_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)


if __name__ == "__main__":
    {"data": lambda: data(), "summary": lambda: summary(sys.argv[2])}[sys.argv[1]]()
