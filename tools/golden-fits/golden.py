#!/usr/bin/env python3
"""Golden fit set: fit the reference models, diff against the last release (#2635).

Four subcommands, all stdlib-only so the workflow needs no `pip install`:

  run     fit every model in models.tsv with one `ferx` binary and write results.json
  report  diff a head results.json against a base (the last release) and, optionally,
          the previous nightly; write the Markdown table for the job summary
  bands   measure the per-model, per-quantity spread from a set of nightly results.json
          files and write bands.json, the input of the fail-on-band mode
  accept  pin one model's head state in bands.json as an intentional change

The workflow (.github/workflows/golden-fits.yml) runs `run` twice — once with the
release tag's binary on the release tag's tree, once with HEAD's — then `report`.

Report-only until FAIL_ON_BAND_FROM. From that date `report` exits non-zero when
bands.json is missing (the measurement period is over and nobody committed the
bands), or when a model moved outside its measured band. That date is the whole
point: a report-only table with no end date is read for a week and then rots.

`tests/golden_fit_set.rs` pins the report's verdicts against synthetic results, so a
perturbation that this script cannot see is a red test, not a quiet nightly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The day the report stops being report-only. 30 nightlies after the workflow's
# first scheduled run (2026-11-05), plus a weekend's slack to commit the bands.
# Moving it is a reviewed edit to this line and to docs/development/golden-fits.qmd
# (tests/golden_fit_set.rs requires the two to agree).
FAIL_ON_BAND_FROM = "2026-12-07"

# Changes at or below these are printed as "=": they are below what `ferx`'s
# own JSON round-trip and a re-run on the same runner can distinguish. They only
# decide the glyph in the report; the fail-on-band mode uses the measured bands.
SAME_ABS_OFV = 1e-9
SAME_REL = 1e-12

# `bands` defaults. A quantity whose measured spread is exactly zero (a fully
# deterministic fit on an unchanged tree) gets the floor rather than a band of 0,
# which would fail on the last bit. The floors are a choice, not a measurement,
# and bands.json records them so the choice is visible.
DEFAULT_HEADROOM = 2.0
DEFAULT_MIN_NIGHTLIES = 30
FLOOR_REL = 1e-6
FLOOR_OFV = 1e-3
FLOOR_ITER = 2

RESULTS_SCHEMA = 1


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def read_manifest(path: Path) -> list[dict]:
    rows = []
    header = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if header is None:
            header = cols
            if header != ["id", "model", "data", "family"]:
                raise SystemExit(f"{path}: header must be id/model/data/family, got {header}")
            continue
        if len(cols) != len(header):
            raise SystemExit(f"{path}: expected {len(header)} tab-separated columns: {raw!r}")
        rows.append(dict(zip(header, cols)))
    return rows


# ---------------------------------------------------------------------------------
# Extraction from a `ferx --output-format json` fit file
# ---------------------------------------------------------------------------------


def num(x) -> float:
    """The JSON writer spells a non-finite float as `null`; read it back as NaN."""
    return float("nan") if x is None else float(x)


def matrix(m) -> list[list[float]]:
    rows, cols, data = m["rows"], m["cols"], m["data"]
    return [[num(data[r * cols + c]) for c in range(cols)] for r in range(rows)]


def lower_tri_se(se: list | None, n: int, i: int, j: int) -> float | None:
    """SE of omega[i][j] (i >= j) from `se_omega`, as `types::omega_se_at` reads it:
    diagonal-only when len == n, else the column-major packed lower triangle."""
    if se is None:
        return None
    n_lt = n * (n + 1) // 2
    if len(se) == n_lt and n_lt != n:
        idx = sum(n - k for k in range(j)) + (i - j)
        return num(se[idx]) if idx < len(se) else None
    if i == j and j < len(se):
        return num(se[i])
    return None


def extract(fit: dict) -> dict:
    """The quantities the report tracks, keyed so that base and head line up by name."""
    params: dict[str, dict] = {}

    def put(key, est, se):
        params[key] = {"est": est, "se": se}

    se_theta = fit.get("se_theta")
    for k, name in enumerate(fit.get("theta_names", [])):
        put(f"theta:{name}", num(fit["theta"][k]), num(se_theta[k]) if se_theta else None)

    def omega_block(prefix, mat, names, se):
        if mat is None:
            return
        om = matrix(mat)
        n = len(om)
        for i in range(n):
            for j in range(i + 1):
                if i != j and om[i][j] == 0.0:
                    continue  # a structural zero of a diagonal omega is not an estimate
                ni = names[i] if i < len(names) else str(i)
                nj = names[j] if j < len(names) else str(j)
                key = f"{prefix}:{ni}" if i == j else f"{prefix}:{ni},{nj}"
                put(key, om[i][j], lower_tri_se(se, n, i, j))

    omega_block("omega", fit.get("omega"), fit.get("eta_names", []), fit.get("se_omega"))
    omega_block("kappa", fit.get("omega_iov"), fit.get("kappa_names", []), fit.get("se_kappa"))

    se_sigma = fit.get("se_sigma")
    for k, name in enumerate(fit.get("sigma_names", [])):
        put(f"sigma:{name}", num(fit["sigma"][k]),
            num(se_sigma[k]) if se_sigma and k < len(se_sigma) else None)

    cov = fit.get("covariance_status")
    return {
        "ofv": num(fit.get("ofv")),
        "converged": bool(fit.get("converged")),
        "n_iterations": fit.get("n_iterations"),
        "method": fit.get("method"),
        "covariance_status": cov if isinstance(cov, str) else json.dumps(cov, sort_keys=True),
        "n_warnings": len(fit.get("warnings", [])),
        "params": params,
    }


# ---------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------


def fit_one(ferx: str, model: Path, data: Path, timeout: int, rec: dict) -> None:
    """Fit one model in a scratch directory and fill `rec` with its status and
    extracted quantities. A failed or timed-out fit is a row, never an abort."""
    with tempfile.TemporaryDirectory(prefix="golden-") as work:
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [ferx, str(model), "--data", str(data), "--output-format", "json"],
                cwd=work, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            rec.update(wall_s=round(time.monotonic() - t0, 2), status="timeout",
                       error=f"no result within {timeout} s")
            return
        rec["wall_s"] = round(time.monotonic() - t0, 2)
        fits = list(Path(work).glob("*-fit.json"))
        if proc.returncode != 0 or len(fits) != 1:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            rec.update(status="failed",
                       error=f"exit {proc.returncode}: " + " | ".join(tail))
            return
        rec.update(extract(json.loads(fits[0].read_text(encoding="utf-8"))))
        rec["status"] = "ok"


def cmd_run(a) -> int:
    only = set(a.only.split(",")) if a.only else None
    root = Path(a.root).resolve()
    ferx = str(Path(a.ferx).resolve())
    models = {}
    for row in read_manifest(Path(a.manifest)):
        if only and row["id"] not in only:
            continue
        model, data = root / row["model"], root / row["data"]
        rec = {
            "family": row["family"],
            "model_path": row["model"],
            "data_path": row["data"],
            "model_sha256": sha256(model),
            "data_sha256": sha256(data),
        }
        if rec["model_sha256"] is None or rec["data_sha256"] is None:
            rec.update(status="missing", error="model or data file not in this tree")
        else:
            fit_one(ferx, model, data, a.timeout, rec)
        models[row["id"]] = rec
        print(f"[golden] {row['id']}: {rec['status']} in {rec.get('wall_s', 0)} s"
              + (f" OFV {rec['ofv']:.6f}" if rec["status"] == "ok" else ""), flush=True)
    out = {
        "schema": RESULTS_SCHEMA,
        "label": a.label,
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": f"{os.uname().sysname}/{os.uname().machine}",
        "models": models,
    }
    text = json.dumps(out, indent=1, sort_keys=True, allow_nan=True) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------


def load(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def rel(h: float, b: float) -> float:
    """|h − b| / |b|: a 1% move reads 1.00e-02 and a 10× move reads 9. A base of
    exactly 0 falls back to the head's magnitude. NaN on EITHER side — including
    both — is NaN: never folded away, so a fit returning NaN is never a pass, even
    when the release returned NaN too."""
    if math.isnan(h) or math.isnan(b):
        return float("nan")
    if h == b:
        return 0.0
    return abs(h - b) / (abs(b) if b != 0.0 else abs(h))


def absd(h: float, b: float) -> float:
    """|h − b|; NaN when either side is NaN, as in `rel`."""
    if math.isnan(h) or math.isnan(b):
        return float("nan")
    return abs(h - b)


def worse(cur, new):
    """The larger of two deviations, where NaN outranks every number (cf. AGENTS.md:
    `f64::max` / Python's `max` would drop it and report the finite rows)."""
    if cur is None:
        return new
    if math.isnan(new[0]):
        return new if not math.isnan(cur[0]) else cur
    if math.isnan(cur[0]):
        return cur
    return new if new[0] > cur[0] else cur


def compare(head: dict, base: dict) -> dict:
    """Every deviation of one model's head record from its base record."""
    d = {"ofv": absd(head["ofv"], base["ofv"]),
         "iter": (head["n_iterations"] or 0) - (base["n_iterations"] or 0),
         "est": {}, "se": {}, "structure": []}
    hp, bp = head["params"], base["params"]
    for key in sorted(set(hp) | set(bp)):
        if key not in hp or key not in bp:
            d["structure"].append(("+" if key in hp else "-") + key)
            continue
        d["est"][key] = rel(num(hp[key]["est"]), num(bp[key]["est"]))
        hs, bs = hp[key]["se"], bp[key]["se"]
        if hs is None and bs is None:
            continue
        if hs is None or bs is None:
            d["structure"].append(f"{'+' if hs is not None else '-'}se({key})")
            continue
        d["se"][key] = rel(num(hs), num(bs))
    for f in ("converged", "covariance_status"):
        if head.get(f) != base.get(f):
            d["structure"].append(f"{f}: {base.get(f)} → {head.get(f)}")
    return d


def worst(devs: dict):
    w = None
    for k, v in devs.items():
        w = worse(w, (v, k))
    return w


def fmt_rel(w) -> str:
    if w is None:
        return "–"
    v, k = w
    if math.isnan(v):
        return f"**NaN** `{k}`"
    if v <= SAME_REL:
        return "="
    return f"{v:.2e} `{k}`"


def signed_ofv(h: float, b: float) -> str:
    if math.isnan(h) or math.isnan(b):
        return "**NaN**"
    if abs(h - b) <= SAME_ABS_OFV:
        return "="
    return f"{h - b:+.6g}"


def changed(d: dict) -> bool:
    def big(v, tol):
        return math.isnan(v) or v > tol
    return (big(d["ofv"], SAME_ABS_OFV) or d["iter"] != 0 or bool(d["structure"])
            or any(big(v, SAME_REL) for v in d["est"].values())
            or any(big(v, SAME_REL) for v in d["se"].values()))


def out_of_band(model_id: str, d: dict, bands: dict) -> list[str]:
    """Each tracked quantity of `model_id` whose deviation exceeds its measured band.
    A quantity with no measured band (new parameter, new model) is out of band:
    the band file has to be re-measured to cover it, not silently skipped."""
    b = bands.get("models", {}).get(model_id)
    if b is None:
        return ["no measured band for this model"]
    bad = []

    def check(name, v, tol):
        # `bands` writes NaN for a quantity that went NaN in its window, and
        # `v > NaN` is false for every v — so a non-finite band is no band.
        if tol is None or not math.isfinite(float(tol)):
            bad.append(f"{name}: no measured band" if tol is None
                       else f"{name}: band is not finite ({tol}) — re-measure")
        elif math.isnan(v) or v > tol:
            bad.append(f"{name}: {v:.3g} > band {tol:.3g}")

    check("OFV", d["ofv"], b.get("ofv"))
    check("iterations", abs(d["iter"]), b.get("n_iterations"))
    for kind in ("est", "se"):
        for k, v in d[kind].items():
            check(f"{kind}({k})", v, b.get(kind, {}).get(k))
    bad.extend(f"structure: {s}" for s in d["structure"])
    return bad


def accepted_drift(mid: str, h: dict, entry: dict, bands: dict) -> list[str]:
    """An accepted model is banded against the state it was ACCEPTED at, not exempted:
    a later, unrelated drift on the same model must still fail. An entry that pins no
    state is itself a failure — there is nothing to hold the model to."""
    state = entry.get("state") if isinstance(entry, dict) else None
    if not isinstance(state, dict) or "params" not in state or "ofv" not in state:
        return ["accepted entry pins no state — record it with `golden.py accept`"]
    pinned = dict(state, ofv=num(state["ofv"]))
    return out_of_band(mid, compare(h, pinned), bands)


def report_header(head, base, prev, bands, today, fail_mode) -> list[str]:
    lines = [f"## Golden fit set — `{head.get('label')}` ({head.get('platform')})", ""]
    if base:
        lines.append(f"Diffed against the last release, **`{base.get('label')}`** "
                     f"({base.get('platform')}).")
    else:
        lines.append("**No release baseline** — every row below is head-only. "
                     "See the job log for why the baseline run is missing.")
    if prev:
        lines.append(f" Δ prev = against the previous nightly, `{prev.get('label')}`.")
    if fail_mode:
        lines.append(f"\n**Fail-on-band** since {FAIL_ON_BAND_FROM}"
                     + (f": bands measured over {bands.get('n_nightlies')} nightlies."
                        if bands else "."))
    else:
        days = (dt.date.fromisoformat(FAIL_ON_BAND_FROM) - dt.date.fromisoformat(today)).days
        lines.append(f"\n**Report-only** until {FAIL_ON_BAND_FROM} ({days} days): this run "
                     "is measuring the natural spread. From that date a change outside "
                     "the measured band fails the job.")
    return lines + [
        "", "Δ est / Δ SE: the largest relative change over θ, ω, κ, σ, and the "
        "parameter it is on. `=` is unchanged to the last digits printed.", "",
        "| | model | family | OFV | Δ OFV | Δ est | Δ SE | iter (Δ) | Δ OFV prev | wall s |",
        "|---|---|---|---:|---:|---|---|---:|---:|---:|"]


LEGEND = ("Legend: `=` unchanged · `≠` changed (report-only) · `🔺` outside the measured "
          "band · `✔︎` outside the band, at the state accepted in bands.json · "
          "`🆕` no release value · `❌` fit failed or OFV not finite.")


def ok(rec) -> bool:
    return rec is not None and rec.get("status") == "ok"


class Report:
    """The rows, details and failures of one `report` run. Each row kind is its own
    method, so the verdict for each kind can be read (and changed) in one place."""

    def __init__(self, bands, fail_mode):
        self.bands = bands
        self.fail_mode = fail_mode
        self.enforce = fail_mode and bands is not None
        self.accepted = (bands or {}).get("accepted", {})
        self.rows: list[str] = []
        self.details: list[str] = []
        self.failures: list[str] = []

    def add(self, mid, h, b, p):
        if h["status"] != "ok":
            self.failed_row(mid, h, b)
        elif not ok(b):
            self.unbaselined_row(mid, h, b, p)
        else:
            self.diff_row(mid, h, b, p)

    def failed_row(self, mid, h, b):
        self.rows.append(f"| ❌ | `{mid}` | {h.get('family', '')} | {h['status']} "
                         f"| | | | | | {h.get('wall_s', '')} |")
        self.details.append(f"- `{mid}` {h['status']}: {h.get('error', '')}")
        if ok(b):
            self.failures.append(f"{mid}: fit {h['status']} at head, ok at the release")

    def unbaselined_row(self, mid, h, b, p):
        why = "new since release" if b is None else f"release: {b['status']}"
        prev_s = signed_ofv(h["ofv"], p["ofv"]) if ok(p) else ""
        self.rows.append(f"| 🆕 | `{mid}` | {h.get('family', '')} | {fmt_ofv(h['ofv'])} | "
                         f"{why} | | | {h.get('n_iterations')} | {prev_s} | "
                         f"{h.get('wall_s', '')} |")
        if not self.enforce:
            return
        if mid not in self.accepted:
            self.failures.append(f"{mid}: no release baseline to band against ({why})")
        elif drift := accepted_drift(mid, h, self.accepted[mid], self.bands):
            self.failures.append(f"{mid}: drifted from its accepted state: " + "; ".join(drift))

    def verdict(self, mid, h, d) -> tuple[str, list[str]]:
        """The row's glyph and its out-of-band list (vs the release)."""
        glyph = "❌" if math.isnan(h["ofv"]) else "≠" if changed(d) else "="
        if not self.enforce:
            return glyph, []
        oob = out_of_band(mid, d, self.bands)
        if mid in self.accepted:
            if drift := accepted_drift(mid, h, self.accepted[mid], self.bands):
                self.failures.append(f"{mid}: drifted from its accepted state: "
                                     + "; ".join(drift))
                return "🔺", oob
            # Outside the band vs the release, but at its accepted state.
            return ("✔︎" if oob else glyph), oob
        if oob:
            self.failures.append(f"{mid}: " + "; ".join(oob))
            return "🔺", oob
        return glyph, oob

    def diff_row(self, mid, h, b, p):
        d = compare(h, b)
        glyph, oob = self.verdict(mid, h, d)
        fam = h.get("family", "")
        if (h.get("model_sha256"), h.get("data_sha256")) != \
                (b.get("model_sha256"), b.get("data_sha256")):
            fam += " *(input files differ from the release)*"
        iters = h.get("n_iterations")
        it = f"{iters} ({d['iter']:+d})" if d["iter"] else f"{iters}"
        prev_s = signed_ofv(h["ofv"], p["ofv"]) if ok(p) else ""
        self.rows.append(f"| {glyph} | `{mid}` | {fam} | {fmt_ofv(h['ofv'])} | "
                         f"{signed_ofv(h['ofv'], b['ofv'])} | {fmt_rel(worst(d['est']))} | "
                         f"{fmt_rel(worst(d['se']))} | {it} | {prev_s} | "
                         f"{h.get('wall_s', '')} |")
        if changed(d) or oob:
            self.details.append(detail_block(mid, h, b, d, oob, self.accepted.get(mid)))


def fmt_ofv(v: float) -> str:
    return "**NaN**" if math.isnan(v) else f"{v:.4f}"


def cmd_report(a) -> int:
    head = load(a.head)
    if head is None:
        raise SystemExit(f"no head results at {a.head}")
    base, prev, bands = load(a.base), load(a.prev), load(a.bands)
    today = a.today or dt.date.today().isoformat()
    fail_mode = today >= FAIL_ON_BAND_FROM

    rep = Report(bands, fail_mode)
    for mid, h in head["models"].items():
        rep.add(mid, h, (base or {}).get("models", {}).get(mid),
                (prev or {}).get("models", {}).get(mid))
    if fail_mode and bands is None:
        rep.failures.insert(0, f"the report-only period ended on {FAIL_ON_BAND_FROM} and "
                            "tools/golden-fits/bands.json is not committed — measure it "
                            "with `golden.py bands` over the stored nightlies "
                            "(docs/development/golden-fits.qmd)")

    lines = report_header(head, base, prev, bands, today, fail_mode) + rep.rows
    if rep.details:
        lines += ["", "### Changed models", ""] + rep.details
    if rep.failures:
        lines += ["", "### Failing", ""] + [f"- {f}" for f in rep.failures]
    lines += ["", LEGEND]

    text = "\n".join(lines) + "\n"
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    if rep.failures and fail_mode:
        for f in rep.failures:
            print(f"::error title=golden fit set::{f}", file=sys.stderr)
        return 1
    return 0


def detail_block(mid, h, b, d, oob, accepted) -> str:
    rows = [f"<details><summary><code>{mid}</code></summary>", "",
            "| quantity | release | head | Δ |", "|---|---:|---:|---:|",
            f"| OFV | {b['ofv']:.6f} | {h['ofv']:.6f} | {signed_ofv(h['ofv'], b['ofv'])} |",
            f"| iterations | {b.get('n_iterations')} | {h.get('n_iterations')} | {d['iter']:+d} |"]
    for key in sorted(set(d["est"]) | set(d["se"])):
        hp, bp = h["params"][key], b["params"][key]
        if key in d["est"] and (math.isnan(d["est"][key]) or d["est"][key] > SAME_REL):
            rows.append(f"| {key} | {num(bp['est']):.6g} | {num(hp['est']):.6g} | "
                        f"{d['est'][key]:.2e} |")
        if key in d["se"] and (math.isnan(d["se"][key]) or d["se"][key] > SAME_REL):
            rows.append(f"| se({key}) | {num(bp['se']):.6g} | {num(hp['se']):.6g} | "
                        f"{d['se'][key]:.2e} |")
    for s in d["structure"]:
        rows.append(f"| {s} | | | |")
    if oob:
        rows += ["", "Outside the band: " + "; ".join(oob)]
    if accepted:
        rows += ["", f"Accepted in bands.json: {accepted.get('reason', '')}"]
    rows += ["", "</details>", ""]
    return "\n".join(rows)


# ---------------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------------


def spread(values: list[float], relative: bool) -> float:
    """Largest deviation from the median, relative to the median when `relative`.
    NaN when any value is NaN — a quantity that went NaN in the measurement
    window has no band, and `report` treats a NaN band as out of band."""
    if any(math.isnan(v) for v in values):
        return float("nan")
    med = statistics.median(values)
    dev = max(abs(v - med) for v in values)
    return dev / max(abs(med), 1e-300) if relative else dev


def cmd_bands(a) -> int:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.history]
    if len(runs) < a.min_nightlies:
        raise SystemExit(f"{len(runs)} nightlies given, {a.min_nightlies} required: the band "
                         "is a measurement of the natural spread and needs the full window")
    ids = sorted({m for r in runs for m in r["models"]})
    out_models = {}
    for mid in ids:
        recs = [r["models"][mid] for r in runs
                if mid in r["models"] and r["models"][mid].get("status") == "ok"]
        if len(recs) < a.min_nightlies:
            print(f"[bands] {mid}: only {len(recs)} ok nightlies — no band", file=sys.stderr)
            continue

        def band(vals, relative, floor):
            s = spread(vals, relative)
            return s if math.isnan(s) else max(a.headroom * s, floor)

        m = {"ofv": band([num(r["ofv"]) for r in recs], False, FLOOR_OFV),
             "n_iterations": max(math.ceil(a.headroom * spread(
                 [float(r["n_iterations"] or 0) for r in recs], False)), FLOOR_ITER),
             "est": {}, "se": {}}
        keys = set.intersection(*(set(r["params"]) for r in recs))
        for k in sorted(keys):
            m["est"][k] = band([num(r["params"][k]["est"]) for r in recs], True, FLOOR_REL)
            ses = [r["params"][k]["se"] for r in recs]
            if all(s is not None for s in ses):
                m["se"][k] = band([num(s) for s in ses], True, FLOOR_REL)
        out_models[mid] = m
    out = {"n_nightlies": len(runs),
           "measured_over": [r.get("label") for r in runs],
           "headroom": a.headroom,
           "floors": {"rel": FLOOR_REL, "ofv": FLOOR_OFV, "iterations": FLOOR_ITER},
           "accepted": {},
           "models": out_models}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[bands] {len(out_models)} models banded over {len(runs)} nightlies → {a.out}")
    return 0


def cmd_accept(a) -> int:
    """Record `id`'s head state as an intentional change in bands.json."""
    head = load(a.head)
    rec = (head or {}).get("models", {}).get(a.id)
    if rec is None or rec.get("status") != "ok":
        raise SystemExit(f"{a.id}: no ok record in {a.head} to accept")
    bands = load(a.bands)
    if bands is None:
        raise SystemExit(f"no bands file at {a.bands}: accept pins a state against bands")
    state = {k: rec[k] for k in ("ofv", "n_iterations", "converged",
                                 "covariance_status", "params")}
    bands.setdefault("accepted", {})[a.id] = {
        "reason": a.reason, "label": head.get("label"), "state": state}
    Path(a.bands).write_text(json.dumps(bands, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[accept] {a.id} pinned at {head.get('label')} OFV {num(rec['ofv']):.6f}")
    return 0


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="fit the golden set with one ferx binary")
    r.add_argument("--ferx", required=True, help="path to the ferx binary")
    r.add_argument("--root", default=".", help="tree the model/data paths resolve against")
    r.add_argument("--manifest", default=str(here / "models.tsv"))
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="")
    r.add_argument("--only", default="", help="comma-separated ids (default: all)")
    r.add_argument("--timeout", type=int, default=1200, help="per-model seconds")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="diff head against the release (and prev nightly)")
    p.add_argument("--head", required=True)
    p.add_argument("--base", help="release results.json (missing file = no baseline)")
    p.add_argument("--prev", help="previous nightly results.json (missing file = none)")
    p.add_argument("--bands", default=str(here / "bands.json"))
    p.add_argument("--today", help="YYYY-MM-DD; default the current date (tests pin it)")
    p.add_argument("--summary", help="append the Markdown here (default stdout)")
    p.set_defaults(fn=cmd_report)

    b = sub.add_parser("bands", help="measure bands.json from nightly results")
    b.add_argument("history", nargs="+", help="nightly head results.json files")
    b.add_argument("--out", default=str(here / "bands.json"))
    b.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    b.add_argument("--min-nightlies", type=int, default=DEFAULT_MIN_NIGHTLIES)
    b.set_defaults(fn=cmd_bands)

    c = sub.add_parser("accept", help="pin a model's head state as an intentional change")
    c.add_argument("id")
    c.add_argument("--head", required=True, help="results.json holding the accepted state")
    c.add_argument("--reason", required=True, help="issue/PR and what changed")
    c.add_argument("--bands", default=str(here / "bands.json"))
    c.set_defaults(fn=cmd_accept)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
