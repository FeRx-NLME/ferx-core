# #2975 benchmark driver: iterations, EBE solves and OFV per (model, scheme), 4 threads, no covariance.
# Usage (from the repo root, after `cargo build --profile ci-fast -p ferx-cli`):
#   python3 bench/scaling_2975/bench_scale.py OUTDIR bench/scaling_2975/models.json SCHEME[,SCHEME...] > OUTDIR/run.log
#   python3 bench/scaling_2975/summ_scale.py OUTDIR/run.log
# A scheme name other than `band` is passed as FERX_SCALE_SCHEME; `band` runs the build's default.
import sys, os, re, json, subprocess, time, csv
FERX = os.environ.get("FERX_BIN", "target/ci-fast/ferx")
W = os.environ.get("FERX_REPO", ".")
OUT = sys.argv[1]; models = json.load(open(sys.argv[2])); schemes = sys.argv[3].split(",")
def prep(src, name):
    s = open(src, encoding="utf-8").read()
    s = re.sub(r"(?m)^(\s*)(threads|covariance|covariance_method)\s*=.*\n", "", s)
    base = os.path.abspath(os.path.dirname(src))
    s = re.sub(r"(?m)^(\s*path\s*=\s*)(.+?)\s*$",
               lambda m: m.group(1) + (m.group(2) if os.path.isabs(m.group(2)) else os.path.join(base, m.group(2)).replace("\\", "/")), s)
    if "[fit_options]" not in s: s += "\n[fit_options]\n"
    s = s.replace("[fit_options]", "[fit_options]\n  threads = 4\n  covariance = false", 1)
    p = os.path.join(OUT, name + ".ferx"); open(p, "w", encoding="utf-8").write(s); return p
rows = []
for name, (src, data) in models.items():
    path = prep(src, name)
    for sc in schemes:
        env = {**os.environ, "FERX_PROFILE": "1", "FERX_DEBUG_IDSCALE": "1"}
        env.pop("FERX_SCALE_SCHEME", None)
        if sc != "band": env["FERX_SCALE_SCHEME"] = sc
        cmd = [os.path.abspath(FERX), os.path.abspath(path), "--output-format", "json"] + (["--data", os.path.abspath(data)] if data else [])
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=OUT)
        wall = time.perf_counter() - t0
        r = dict(model=name, scheme=sc, exit=p.returncode, wall=round(wall, 2))
        fj = os.path.join(OUT, name + "-fit.json")
        if p.returncode == 0 and os.path.exists(fj):
            d = json.load(open(fj, encoding="utf-8"))
            m = re.search(r"inner: (\d+) EBE solves; (\d+) analytic-grad", p.stdout + p.stderr)
            r.update(iters=d.get("n_iterations"), ofv=d.get("ofv"), conv=d.get("converged"),
                     ebe=int(m.group(1)) if m else -1, steps=int(m.group(2)) if m else -1,
                     derivs=("axis derivatives available: true" in p.stderr) if sc not in ("band","legacy") else None)
            os.remove(fj)
        rows.append(r); print(json.dumps(r), flush=True)
        for f in os.listdir(OUT):
            if f.startswith(name) and (f.endswith("-sdtab.csv") or f.endswith("-covtab.csv") or f.endswith(".yaml")): os.remove(os.path.join(OUT, f))
json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=0)
