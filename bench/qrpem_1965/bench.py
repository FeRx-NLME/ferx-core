#!/usr/bin/env python3
"""Phase 0 benchmark for epic #1965: does quasi-random (Sobol) importance
sampling in IMPMAP reduce Monte-Carlo variability versus pseudo-random IMPMAP
and SAEM?

Every arm x model x seed is fitted with the release `ferx` CLI, then its final
estimates are re-scored by one common evaluator — `method = imp`,
`imp_eval_only = true`, a large fixed K and a fixed seed — so all arms are
compared on the same importance-sampled -2 log L, not on each method's own
(noisy, differently defined) objective.

Usage:
    bench.py run  [--models a,b] [--arms x,y] [--seeds N] [--jobs J]
    bench.py eval [--jobs J]
    bench.py report
Concurrency: J jobs x THREADS threads; keep J*THREADS <= 8.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FERX = os.path.join(ROOT, "target", "release", "ferx")
RUNS = os.environ.get("BENCH_RUNS", os.path.join(HERE, "runs"))
THREADS = 2

MODELS = {
    # name: (model template, dataset)
    "warfarin": ("examples/warfarin_impmap.ferx", "data/warfarin.csv"),
    "two_cpt_oral_cov": ("examples/two_cpt_oral_cov.ferx", "data/two_cpt_oral_cov.csv"),
    "three_cpt_iv": ("examples/three_cpt_iv.ferx", "data/three_cpt_iv.csv"),
    "sparse_combined": ("bench/qrpem_1965/models/sparse_combined.ferx", "data/saem_sparse_combined.csv"),
}

IMPMAP_COMMON = [
    "method = impmap",
    "impmap_auto = false",
    "impmap_iterations = 200",
    "impmap_averaging = 50",
]


def arm_options(arm, seed, k):
    """[fit_options] lines for one arm. `k` = IS samples per subject."""
    if arm == "impmap_prng":
        return IMPMAP_COMMON + [f"impmap_samples = {k}", "impmap_proposal_df = normal",
                                "impmap_sobol = false", f"impmap_seed = {seed}"]
    if arm == "impmap_sobol":
        return IMPMAP_COMMON + [f"impmap_samples = {k}", "impmap_proposal_df = normal",
                                "impmap_sobol = true", f"impmap_seed = {seed}"]
    if arm == "impmap_t4":
        return IMPMAP_COMMON + [f"impmap_samples = {k}", "impmap_proposal_df = 4",
                                f"impmap_seed = {seed}"]
    # Phase 1b (needs a binary with quasi-random t draws): Sobol on the
    # default Student-t proposal.
    if arm == "impmap_t4_sobol":
        return IMPMAP_COMMON + [f"impmap_samples = {k}", "impmap_proposal_df = 4",
                                "impmap_sobol = true", f"impmap_seed = {seed}"]
    # Diagnostic arms for the two_cpt_oral_cov plateau (not part of the
    # quasi-random comparison): is the cross-seed spread a MAP-mode / proposal
    # coverage problem, or plain slow EM?
    if arm == "impmap_prng_mceta3":
        return IMPMAP_COMMON + [f"impmap_samples = {k}", "impmap_proposal_df = normal",
                                "impmap_mceta = 3", f"impmap_seed = {seed}"]
    if arm == "impmap_prng_defensive":
        return IMPMAP_COMMON + [f"impmap_samples = {k}", "impmap_proposal_df = normal",
                                "imp_defensive_alpha = 0.1", f"impmap_seed = {seed}"]
    if arm == "impmap_prng_iter600":
        return [o for o in IMPMAP_COMMON if not o.startswith("impmap_iterations")] + [
            "impmap_iterations = 600", f"impmap_samples = {k}", "impmap_proposal_df = normal",
            f"impmap_seed = {seed}"]
    if arm == "saem":
        return ["method = saem", f"seed = {seed}"]
    if arm == "focei":
        return ["method = focei", "maxiter = 500"]
    raise ValueError(arm)


ARMS = ["impmap_prng", "impmap_sobol", "impmap_t4", "saem", "focei"]
DETERMINISTIC = {"focei"}

EVAL_K = 20000
EVAL_SEED = 777


def strip_fit_options(text):
    """Drop the template's [fit_options] block (and [data]) — the harness owns them."""
    out, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            skip = s in ("[fit_options]", "[data]")
        if not skip:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def write_model(path, template_text, opts):
    body = strip_fit_options(template_text)
    body += "\n[fit_options]\n" + "".join(f"  {o}\n" for o in opts + [f"threads = {THREADS}",
                                                                    "covariance = false"])
    with open(path, "w") as f:
        f.write(body)


def run_ferx(model_path, data_path, cwd):
    t0 = time.time()
    p = subprocess.run([FERX, os.path.basename(model_path), "--data", data_path],
                       cwd=cwd, capture_output=True, text=True)
    wall = time.time() - t0
    with open(os.path.join(cwd, "stdout.txt"), "w") as f:
        f.write(p.stdout)
    with open(os.path.join(cwd, "stderr.txt"), "w") as f:
        f.write(p.stderr)
    return p.returncode, wall, p.stderr


def parse_fit(yaml_path):
    with open(yaml_path) as f:
        y = yaml.safe_load(f)
    res = {
        "theta": {k: v["estimate"] for k, v in (y.get("theta") or {}).items()},
        "omega": {},
        "sigma": {k: v["estimate"] for k, v in (y.get("sigma") or {}).items()},
        "ofv": y["objective_function"]["ofv"],
        "wall": y.get("estimation", {}).get("wall_time_secs"),
        "converged": y["model"].get("converged"),
    }
    for k, v in (y.get("omega") or {}).items():
        res["omega"][k] = v["variance"] if "variance" in v else v["covariance"]
    if y.get("importance_sampling"):
        imp = y["importance_sampling"]
        res["is"] = {k: imp[k] for k in ("minus2_log_likelihood", "mc_standard_error",
                                          "n_samples", "ess_min", "ess_median")}
    return res


def iter_trace(stderr):
    """IMPMAP verbose prints `iter k: -2logL(IS) = v` at k<=5, every 10th and the last."""
    return [(int(a), float(b)) for a, b in
            re.findall(r"iter\s+(\d+): -2logL\(IS\) = (-?[\d.]+)", stderr)]


# ---------------------------------------------------------------- eval model

THETA_RE = re.compile(r"^(\s*theta\s+)(\w+)\(([^,]+),(.*)$")
OMEGA_RE = re.compile(r"^(\s*omega\s+)(\w+)(\s*~\s*)([-\d.eE+]+)(.*)$")
BLOCK_RE = re.compile(r"^(\s*block_omega\s*\(([^)]*)\)\s*=\s*)\[[^\]]*\](.*)$")
SIGMA_RE = re.compile(r"^(\s*sigma\s+)(\w+)(\s*~\s*)([-\d.eE+]+)(\s*\(sd\))?(.*)$")


def fill_estimates(template_text, fit):
    """Template with every theta/omega/sigma init replaced by `fit`'s estimates."""
    out = []
    for line in strip_fit_options(template_text).splitlines():
        m = THETA_RE.match(line)
        if m and m.group(2) in fit["theta"]:
            out.append(f"{m.group(1)}{m.group(2)}({fit['theta'][m.group(2)]!r},{m.group(4)}")
            continue
        m = BLOCK_RE.match(line)
        if m:
            names = [n.strip() for n in m.group(2).split(",")]
            vals = []
            for i, ni in enumerate(names):
                for j in range(i + 1):
                    key = ni if i == j else f"{ni}__{names[j]}"
                    vals.append(fit["omega"][key])
            out.append(f"{m.group(1)}[{', '.join(repr(v) for v in vals)}]{m.group(3)}")
            continue
        m = OMEGA_RE.match(line)
        if m and m.group(2) in fit["omega"]:
            out.append(f"{m.group(1)}{m.group(2)}{m.group(3)}{fit['omega'][m.group(2)]!r}{m.group(5)}")
            continue
        m = SIGMA_RE.match(line)
        if m and m.group(2) in fit["sigma"]:
            # the fit YAML's sigma `estimate` is on the SD scale
            rest = m.group(6)
            rest = re.sub(r"#.*", "", rest)
            out.append(f"{m.group(1)}{m.group(2)}{m.group(3)}{fit['sigma'][m.group(2)]!r} (sd){rest}")
            continue
        out.append(line)
    return "\n".join(out) + "\n"


EVAL_OPTS = ["method = imp", "imp_eval_only = true", f"imp_samples = {EVAL_K}",
             "imp_auto = false", f"imp_seed = {EVAL_SEED}"]


# ---------------------------------------------------------------- jobs

def job_dir(model, arm, k, seed):
    return os.path.join(RUNS, model, f"{arm}_K{k}", f"s{seed:02d}")


def do_fit(model, arm, k, seed):
    d = job_dir(model, arm, k, seed)
    out = os.path.join(d, "result.json")
    if os.path.exists(out):
        return out
    os.makedirs(d, exist_ok=True)
    tmpl, data = MODELS[model]
    text = open(os.path.join(ROOT, tmpl)).read()
    write_model(os.path.join(d, "m.ferx"), text, arm_options(arm, seed, k))
    rc, wall, err = run_ferx(os.path.join(d, "m.ferx"), os.path.join(ROOT, data), d)
    res = {"model": model, "arm": arm, "k": k, "seed": seed, "rc": rc, "wall_cli": wall}
    if rc == 0:
        res["fit"] = parse_fit(os.path.join(d, "m-fit.yaml"))
        res["trace"] = iter_trace(err)
    else:
        res["error"] = err[-2000:]
    json.dump(res, open(out, "w"), indent=1)
    return out


def do_eval(result_path):
    res = json.load(open(result_path))
    if res.get("rc") != 0 or "eval" in res:
        return result_path
    d = os.path.join(os.path.dirname(result_path), "eval")
    os.makedirs(d, exist_ok=True)
    tmpl, data = MODELS[res["model"]]
    text = fill_estimates(open(os.path.join(ROOT, tmpl)).read(), res["fit"])
    body = text + "\n[fit_options]\n" + "".join(
        f"  {o}\n" for o in EVAL_OPTS + [f"threads = {THREADS}", "covariance = false"])
    open(os.path.join(d, "e.ferx"), "w").write(body)
    rc, wall, err = run_ferx(os.path.join(d, "e.ferx"), os.path.join(ROOT, data), d)
    if rc == 0:
        res["eval"] = parse_fit(os.path.join(d, "e-fit.yaml")).get("is")
        res["eval"]["wall_cli"] = wall
    else:
        res["eval_error"] = err[-2000:]
    json.dump(res, open(result_path, "w"), indent=1)
    return result_path


def cmd_run(a):
    models = a.models.split(",") if a.models else list(MODELS)
    arms = a.arms.split(",") if a.arms else ARMS
    ks = [int(x) for x in a.k.split(",")]
    jobs = []
    for m in models:
        for arm in arms:
            for k in (ks if arm.startswith("impmap") else [0]):
                seeds = [1] if arm in DETERMINISTIC else range(1, a.seeds + 1)
                jobs += [(m, arm, k, s) for s in seeds]
    print(f"{len(jobs)} fit jobs, {a.jobs} concurrent x {THREADS} threads", flush=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(do_fit, *j): j for j in jobs}
        for i, f in enumerate(cf.as_completed(futs), 1):
            j = futs[f]
            r = json.load(open(f.result()))
            tag = "ok" if r["rc"] == 0 else "FAIL"
            print(f"[{i}/{len(jobs)}] {tag} {j} {r['wall_cli']:.1f}s", flush=True)
            if a.eval and r["rc"] == 0:
                do_eval(f.result())


def all_results():
    out = []
    for dp, _, fs in os.walk(RUNS):
        if "result.json" in fs:
            out.append(os.path.join(dp, "result.json"))
    return sorted(out)


def cmd_eval(a):
    paths = all_results()
    print(f"{len(paths)} results to evaluate", flush=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for i, p in enumerate(ex.map(do_eval, paths), 1):
            r = json.load(open(p))
            v = r.get("eval", {}).get("minus2_log_likelihood")
            print(f"[{i}/{len(paths)}] {p[len(RUNS)+1:]} -> {v}", flush=True)


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def cmd_report(a):
    rows = [json.load(open(p)) for p in all_results()]
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["arm"], r["k"]), []).append(r)
    lines = []
    lines.append("| model | arm | K | n ok/n | eval −2LL mean | SD across seeds | "
                 "median param SD (rel.) | max param SD (rel.) | tail jitter SD | "
                 "wall s (median) | eval MC SE |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        g = groups[key]
        ok = [r for r in g if r["rc"] == 0 and "eval" in r]
        if not ok:
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | 0/{len(g)} | | | | | | | |")
            continue
        ev = [r["eval"]["minus2_log_likelihood"] for r in ok]
        # relative SD of each parameter across seeds (SD / |mean|)
        rel = []
        for block in ("theta", "omega", "sigma"):
            for name in ok[0]["fit"][block]:
                xs = [r["fit"][block][name] for r in ok]
                mu = st.mean(xs)
                if len(xs) > 1 and abs(mu) > 1e-12:
                    rel.append(sd(xs) / abs(mu))
        jit = []
        for r in ok:
            tail = [v for (k, v) in r.get("trace", []) if k >= 150]
            if len(tail) > 1:
                jit.append(sd(tail))
        walls = [r["fit"]["wall"] or r["wall_cli"] for r in ok]
        mcse = st.mean(r["eval"]["mc_standard_error"] for r in ok)
        f = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4g}"
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2] or '—'} | {len(ok)}/{len(g)} | {st.mean(ev):.3f} | "
            f"{f(sd(ev))} | {f(st.median(rel) if rel else None)} | {f(max(rel) if rel else None)} | "
            f"{f(st.mean(jit) if jit else None)} | {st.median(walls):.2f} | {mcse:.3f} |")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(HERE, "summary.md"), "w").write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--models")
    r.add_argument("--arms")
    r.add_argument("--k", default="300")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--eval", action="store_true", help="evaluate each fit right after it finishes")
    e = sp.add_parser("eval")
    e.add_argument("--jobs", type=int, default=4)
    sp.add_parser("report")
    a = ap.parse_args()
    {"run": cmd_run, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
