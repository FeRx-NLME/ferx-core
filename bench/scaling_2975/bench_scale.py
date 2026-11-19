# #2975 benchmark driver: iterations, EBE solves and OFV per (model, configuration), no
# covariance step. A configuration is a `+`-joined list of `[fit_options]` pairs, e.g.
# `optimizer=trust_region` or `optimizer=nlopt_lbfgs+parameter_scaling=abs`; `default` runs
# the model's own options.
# Usage (from the repo root, after `cargo build --profile ci-fast -p ferx-cli`):
#   python3 bench/scaling_2975/bench_scale.py OUTDIR bench/scaling_2975/models.json \
#       CONFIG[,CONFIG...] > OUTDIR/run.log
#   python3 bench/scaling_2975/summ_scale.py OUTDIR/run.log
import json
import os
import re
import subprocess
import sys
import time

FERX = os.environ.get("FERX_BIN", "target/ci-fast/ferx")
THREADS = os.environ.get("FERX_THREADS", "4")
OUT = sys.argv[1]
models = json.load(open(sys.argv[2]))
configs = sys.argv[3].split(",")


def settings(config):
    """The `[fit_options]` lines a configuration name stands for."""
    if config == "default":
        return []
    return [f"  {k.strip()} = {v.strip()}" for k, v in (kv.split("=") for kv in config.split("+"))]


def prep(src, name, config):
    """The model file with the configuration's options and absolute data paths."""
    s = open(src, encoding="utf-8").read()
    s = re.sub(r"(?m)^(\s*)(threads|covariance|covariance_method)\s*=.*\n", "", s)
    keys = [line.split("=")[0].strip() for line in settings(config)]
    for key in keys:
        s = re.sub(rf"(?m)^\s*{key}\s*=.*\n", "", s)
    base = os.path.abspath(os.path.dirname(src))

    def absolute(m):
        path = m.group(2)
        if not os.path.isabs(path):
            path = os.path.join(base, path).replace("\\", "/")
        return m.group(1) + path

    s = re.sub(r"(?m)^(\s*path\s*=\s*)(.+?)\s*$", absolute, s)
    if "[fit_options]" not in s:
        s += "\n[fit_options]\n"
    lines = [f"  threads = {THREADS}", "  covariance = false"] + settings(config)
    s = s.replace("[fit_options]", "[fit_options]\n" + "\n".join(lines), 1)
    p = os.path.join(OUT, name + ".ferx")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(s)
    return p


rows = []
for name, (src, data) in models.items():
    for config in configs:
        path = prep(src, name, config)
        env = {**os.environ, "FERX_PROFILE": "1"}
        cmd = [os.path.abspath(FERX), os.path.abspath(path), "--output-format", "json"]
        if data:
            cmd += ["--data", os.path.abspath(data)]
        t0 = time.perf_counter()
        p = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=OUT
        )
        wall = time.perf_counter() - t0
        r = dict(model=name, scheme=config, exit=p.returncode, wall=round(wall, 2))
        fj = os.path.join(OUT, name + "-fit.json")
        if p.returncode == 0 and os.path.exists(fj):
            d = json.load(open(fj, encoding="utf-8"))
            m = re.search(r"inner: (\d+) EBE solves; (\d+) analytic-grad", p.stdout + p.stderr)
            r.update(
                iters=d.get("n_iterations"),
                ofv=d.get("ofv"),
                conv=d.get("converged"),
                optimizer=d.get("optimizer"),
                ebe=int(m.group(1)) if m else -1,
                steps=int(m.group(2)) if m else -1,
            )
            os.remove(fj)
        rows.append(r)
        print(json.dumps(r), flush=True)
        for f in os.listdir(OUT):
            stale = f.endswith("-sdtab.csv") or f.endswith("-covtab.csv") or f.endswith(".yaml")
            if f.startswith(name) and stale:
                os.remove(os.path.join(OUT, f))
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(rows, fh, indent=0)
