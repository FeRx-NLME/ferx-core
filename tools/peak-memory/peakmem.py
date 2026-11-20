#!/usr/bin/env python3
"""Peak-memory probe: measure the shipped `ferx` binary's peak RSS on a synthetic stiff
ODE model and hold it against a committed baseline (#3081).

The #2939 compiled-plan regression took one evaluation of a 9-state stiff PK/PD model
from 86 MB to 3-4 GB, and a full fit to 11 GB, and nothing noticed: it was found by
hand, after the fact. This script is the nightly that notices.

Three subcommands, stdlib-only so the workflow needs no `pip install`:

  run     run every case below on stiff_pkpd.ferx / stiff_pkpd.csv under
          `/usr/bin/time` and write results.json (peak RSS, wall time, fit facts)
  report  compare a results.json against baseline.json, write the Markdown table for
          the job summary, and exit non-zero on a regression
  accept  write a results.json's peaks into baseline.json — the reviewed way to move
          the baseline, in either direction

The ratchet is `docs/.lint-baseline`'s: the baseline may only move in a commit that
says why. A peak more than TOLERANCE above its baseline fails the job; a peak more than
TOLERANCE BELOW it is reported as an improvement to lock in, so a win is not quietly
spent by the next regression. `report` never edits baseline.json — `accept` does, and
`accept` is run by a person, on a results.json downloaded from the nightly, and
committed with a reason.

Linux is the reference platform (AGENTS.md, #1688) and the baseline records the
platform it was measured on. A run on another platform is reported but not judged:
glibc vs macOS libsystem and x86_64 vs arm64 move peak RSS by more than the tolerance.

`tests/peak_memory_probe.rs` pins the verdicts against synthetic results and pins the
model's `[fit_options]` lines this script rewrites, so a probe that cannot see a
regression is a red test, not a quiet nightly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE / "stiff_pkpd.ferx"
DATA = HERE / "stiff_pkpd.csv"
BASELINE = HERE / "baseline.json"

SCHEMA = 1

# A peak this far above its baseline fails the job; this far below is an improvement
# to ratchet in. The regression this exists for was 10-40x, so the band only has to
# sit clear of run-to-run noise, which docs/development/peak-memory.qmd records.
TOLERANCE = 0.25

# Fixed, not the runner's core count: per-thread pools are part of what is measured
# (#3079), so the baseline is only comparable at the thread count it was taken at.
# ubuntu-latest has 4 vCPUs. `run` refuses a fit whose `n_threads_used` disagrees.
FERX_THREADS = 4

# Each case is the model file with its two `[fit_options]` lines rewritten. The model
# file itself carries the `eval` spelling; `fit` is a short estimation with the
# covariance step, which is where e7070's post-fit spike lived.
#   eval  one evaluation at the initial estimates (NONMEM MAXEVAL=0)
#   fit   one outer iteration (NLopt budget n+1 evaluations), then the covariance step
CASES: dict[str, dict[str, str]] = {
    "eval": {"maxiter": "0", "covariance": "false"},
    "fit": {"maxiter": "1", "covariance": "true"},
}

# The `covariance_status` each case must report. The covariance step is where e7070's
# post-fit spike lived, so a `fit` case whose step was skipped, or failed before it
# assembled anything, measures a lower peak for a reason that is not a memory win —
# probe drift, not an improvement. `eval` must not run it, or it is a second `fit`.
EXPECTED_COVARIANCE: dict[str, str] = {
    "eval": "NotRequested",
    "fit": "Computed",
}

# The lines `case_model` rewrites, exactly as stiff_pkpd.ferx spells them.
FIT_OPTION_LINE = re.compile(r"^(  (maxiter|covariance) +=) (\S+)$", re.MULTILINE)

# What makes the probe a probe. If a change routes the model off the compiled plans or
# stops `auto` escalating, the peak can fall for a reason that has nothing to do with
# memory — `report` flags it rather than calling it an improvement.
ESCALATION_CODE = "W_ODE_SOLVER_ESCALATION_NOTE"

# Where a run was measured. The nightly's GitHub-hosted runner is the reference; anything
# else (a laptop, a container, emulation, a self-hosted runner) reports the same
# `platform` string while its allocator, kernel and CPU can move peak RSS by more than
# TOLERANCE. A baseline taken anywhere else is PROVISIONAL: the first reference run
# against it fails until its artifact is accepted, so a stand-in number cannot stay
# the guard by default.
REFERENCE_RUNNER = "github-hosted"


def runner() -> str:
    """`github-hosted` / `self-hosted` inside GitHub Actions, `local` elsewhere."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return "local"
    return os.environ.get("RUNNER_ENVIRONMENT") or "unknown"


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def platform() -> str:
    u = os.uname()
    return f"{u.sysname}/{u.machine}"


def case_model(text: str, case: str) -> str:
    """The model text with `maxiter` / `covariance` set for `case`. Exactly one line of
    each must match, or the rewrite would silently measure the wrong case."""
    want = CASES[case]
    seen: dict[str, int] = {}

    def sub(m: re.Match) -> str:
        key = m.group(2)
        seen[key] = seen.get(key, 0) + 1
        return f"{m.group(1)} {want[key]}"

    out = FIT_OPTION_LINE.sub(sub, text)
    if seen != {"maxiter": 1, "covariance": 1}:
        raise SystemExit(
            f"{MODEL.name}: expected one `  maxiter = ...` and one `  covariance = ...` "
            f"line in [fit_options], found {seen}")
    return out


# ---------------------------------------------------------------------------------
# /usr/bin/time
# ---------------------------------------------------------------------------------


def covariance_label(status) -> str:
    """`covariance_status` as one string: the variant name of a unit status
    (`"Computed"`), the sorted JSON of a status that carries data."""
    return status if isinstance(status, str) else json.dumps(status, sort_keys=True)


def probe_drift(case: str, rec: dict) -> list[str]:
    """What, in one ok case record, says the probe no longer measures the shape it
    exists for. Empty when the case is a probe."""
    drift = []
    if not rec.get("escalated"):
        drift.append("`auto` no longer escalates")
    # Fail closed: a missing, null or renamed field is no evidence of the plan path.
    if rec.get("gradient_method_outer") != "analytic":
        drift.append(f"outer gradient is `{rec.get('gradient_method_outer')}`, not analytic")
    want = EXPECTED_COVARIANCE.get(case)
    got = rec.get("covariance_status")
    if want is None:
        drift.append(f"no expected covariance status for case `{case}`")
    elif got != want:
        drift.append(f"covariance status is `{got}`, not `{want}`")
    return drift


def time_cmd() -> list[str]:
    """GNU time's verbose report on Linux; BSD time's on macOS (local runs only)."""
    return ["/usr/bin/time", "-l"] if sys.platform == "darwin" else ["/usr/bin/time", "-v"]


def parse_peak_rss(stderr: str) -> int | None:
    """Peak resident set size in BYTES from either time's report, or None.

    GNU:  `Maximum resident set size (kbytes): 113672`
    BSD:  `   113672192  maximum resident set size` (bytes on macOS)
    """
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if m:
        return int(m.group(1)) * 1024
    m = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", stderr, re.MULTILINE)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------


def run_case(ferx: str, case: str, timeout: int, rec: dict) -> None:
    """Run one case in a scratch directory and fill `rec`. A failed or timed-out run is
    a record with a status, never an abort: `report` turns it into a red row."""
    with tempfile.TemporaryDirectory(prefix="peakmem-") as work:
        model = Path(work) / MODEL.name
        model.write_text(case_model(MODEL.read_text(encoding="utf-8"), case), encoding="utf-8")
        cmd = time_cmd() + [ferx, str(model), "--data", str(DATA), "--output-format", "json",
                            "--threads", str(FERX_THREADS), "--clean"]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            rec.update(status="timeout", wall_s=round(time.monotonic() - t0, 1),
                       error=f"no result within {timeout} s")
            return
        rec["wall_s"] = round(time.monotonic() - t0, 1)
        peak = parse_peak_rss(proc.stderr)
        fits = list(Path(work).glob("*-fit.json"))
        if proc.returncode != 0 or peak is None or len(fits) != 1:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            rec.update(status="failed", error=f"exit {proc.returncode}: " + " | ".join(tail))
            return
        fit = json.loads(fits[0].read_text(encoding="utf-8"))
        used = fit.get("n_threads_used")
        if used != FERX_THREADS:
            rec.update(status="failed", error=f"fit ran on {used} threads, not "
                       f"{FERX_THREADS}: its peak is not comparable to the baseline")
            return
        warnings = fit.get("warnings", [])
        rec.update(
            status="ok",
            peak_rss_bytes=peak,
            ofv=fit.get("ofv"),
            n_iterations=fit.get("n_iterations"),
            covariance_status=covariance_label(fit.get("covariance_status")),
            gradient_method_outer=fit.get("gradient_method_outer"),
            escalated=any(ESCALATION_CODE in w for w in warnings),
        )


def cmd_run(a) -> int:
    ferx = str(Path(a.ferx).resolve())
    cases = a.cases.split(",") if a.cases else list(CASES)
    for c in cases:
        if c not in CASES:
            raise SystemExit(f"unknown case {c!r}; one of {', '.join(CASES)}")
    out = {
        "schema": SCHEMA,
        "label": a.label,
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform(),
        "runner": runner(),
        "threads": FERX_THREADS,
        "model_sha256": sha256(MODEL),
        "data_sha256": sha256(DATA),
        "cases": {},
    }
    for c in cases:
        rec: dict = {}
        run_case(ferx, c, a.timeout, rec)
        out["cases"][c] = rec
        peak = rec.get("peak_rss_bytes")
        print(f"[peakmem] {c}: {rec['status']} in {rec.get('wall_s', 0)} s"
              + (f", peak RSS {mb(peak)}" if peak else f" ({rec.get('error', '')})"),
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------


def mb(b: float | None) -> str:
    return "—" if b is None else f"{b / 2**20:.0f} MB"


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verdict(case: str, head: dict, base: dict | None, judged: bool) -> tuple[str, str, bool]:
    """(glyph, note, failed) for one case. `judged` is False on a platform or input
    mismatch: the numbers are shown, never failed on."""
    if head.get("status") != "ok":
        return "✗", f"{head.get('status', 'missing')}: {head.get('error', '')}", True
    peak = head.get("peak_rss_bytes")
    if not isinstance(peak, int) or peak <= 0:
        return "✗", "no peak RSS recorded", True
    if base is None or not isinstance(base.get("peak_rss_bytes"), int) \
            or base["peak_rss_bytes"] <= 0:
        return "?", "no baseline for this case — `accept` one", judged
    ratio = peak / base["peak_rss_bytes"]
    if not math.isfinite(ratio):
        return "✗", "non-finite ratio", True
    probe = probe_drift(case, head)
    probe_note = ("; probe drifted: " + ", ".join(probe)) if probe else ""
    if not judged:
        return "·", "not judged (see above)" + probe_note, False
    if ratio > 1 + TOLERANCE:
        return "▲", f"regression: more than {TOLERANCE:.0%} above baseline" + probe_note, True
    if probe:
        # A model that left the stiff / plan path, or a case whose covariance step did
        # not run as expected, measures something else; a lower peak from that is not
        # a win, and a higher one is caught above regardless.
        return "!", probe_note[2:], True
    if ratio < 1 - TOLERANCE:
        return "▼", (f"improvement: more than {TOLERANCE:.0%} below baseline — "
                     "lock it in with `peakmem.py accept`"), False
    return "=", "within tolerance", False


def cmd_report(a) -> int:
    head = load(a.results)
    base = load(a.baseline) if Path(a.baseline).is_file() else None
    lines = ["## Peak memory (#3081)", ""]
    notes = []
    judged = True
    if base is None:
        notes.append(f"No baseline at `{a.baseline}`: nothing to compare against.")
        judged = False
    else:
        if base.get("schema") != SCHEMA:
            notes.append(f"Baseline schema {base.get('schema')} ≠ {SCHEMA}.")
            judged = False
        if base.get("platform") != head.get("platform"):
            notes.append(f"Baseline measured on `{base.get('platform')}`, this run on "
                         f"`{head.get('platform')}`: shown, not judged. Re-baseline on "
                         "the reference platform with `peakmem.py accept`.")
            judged = False
        if base.get("threads") != head.get("threads"):
            notes.append(f"Baseline at {base.get('threads')} threads, this run at "
                         f"{head.get('threads')}: shown, not judged.")
            judged = False
        for k in ("model_sha256", "data_sha256"):
            if base.get(k) != head.get(k):
                notes.append(f"`{k}` differs from the baseline's: the probe's inputs "
                             "changed, so the baseline measured a different thing. "
                             "Re-measure and `accept` in the same PR.")
                judged = False
    provisional = False
    if judged and head.get("runner") == REFERENCE_RUNNER \
            and base.get("runner") != REFERENCE_RUNNER:
        notes.append(f"The baseline is provisional: measured on runner "
                     f"`{base.get('runner', 'local')}`, not `{REFERENCE_RUNNER}`. This run "
                     "is a reference measurement, so the job fails until it is accepted: "
                     "download this run's `peak-memory-results` and `peakmem.py accept` it.")
        provisional = True
    lines += [f"Run `{head.get('label')}` on `{head.get('platform')}` "
              f"(runner `{head.get('runner', 'local')}`), "
              f"{head.get('threads')} threads; tolerance ±{TOLERANCE:.0%}.", ""]
    lines += [f"> {n}" for n in notes] + ([""] if notes else [])
    lines += ["| case | baseline | head | Δ | wall | | note |",
              "|---|--:|--:|--:|--:|:-:|---|"]
    failed = []
    bcases = (base or {}).get("cases", {})
    for case in sorted(set(head.get("cases", {})) | set(bcases)):
        h = head.get("cases", {}).get(case, {"status": "missing", "error": "not run"})
        b = bcases.get(case)
        glyph, note, bad = verdict(case, h, b, judged)
        hp, bp = h.get("peak_rss_bytes"), (b or {}).get("peak_rss_bytes")
        delta = f"{hp / bp - 1:+.1%}" if isinstance(hp, int) and isinstance(bp, int) and bp else "—"
        wall = f"{h['wall_s']:.0f} s" if isinstance(h.get("wall_s"), (int, float)) else "—"
        lines.append(f"| {case} | {mb(bp)} | {mb(hp)} | {delta} | {wall} | {glyph} | {note} |")
        if bad:
            failed.append(case)
    if provisional:
        failed.append("provisional baseline")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as f:
            f.write(text)
    if failed:
        print(f"peak memory: {', '.join(failed)} failed (see table)", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------------
# accept
# ---------------------------------------------------------------------------------


def cmd_accept(a) -> int:
    head = load(a.results)
    if not a.reason.strip():
        raise SystemExit("--reason is required: the baseline moves only with a stated why")
    bad = [c for c, r in head.get("cases", {}).items() if r.get("status") != "ok"]
    if bad or not head.get("cases"):
        raise SystemExit(f"refusing to accept a run with failed or no cases: {bad}")
    drift = {c: d for c, r in head["cases"].items() if (d := probe_drift(c, r))}
    if drift:
        raise SystemExit(f"refusing to accept: the probe drifted, so it is no longer "
                         f"measuring the shape it exists for: {drift}")
    out = {
        "schema": SCHEMA,
        "platform": head["platform"],
        "runner": head.get("runner", "local"),
        "threads": head["threads"],
        "model_sha256": head["model_sha256"],
        "data_sha256": head["data_sha256"],
        "measured": head.get("label"),
        "created": head.get("created"),
        "reason": a.reason.strip(),
        "cases": {c: {"peak_rss_bytes": r["peak_rss_bytes"], "wall_s": r.get("wall_s")}
                  for c, r in sorted(head["cases"].items())},
    }
    Path(a.baseline).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
    for c, r in out["cases"].items():
        print(f"[peakmem] baseline {c}: {mb(r['peak_rss_bytes'])}")
    if out["runner"] != REFERENCE_RUNNER:
        print(f"[peakmem] runner `{out['runner']}` is not `{REFERENCE_RUNNER}`: this baseline "
              "is provisional, and the next nightly fails until its artifact is accepted")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="measure every case, write results.json")
    r.add_argument("--ferx", required=True, help="the `ferx` binary to measure")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default="local")
    r.add_argument("--cases", default="", help=f"comma-separated subset of {list(CASES)}")
    r.add_argument("--timeout", type=int, default=5400, help="seconds per case")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="compare results.json against the baseline")
    rp.add_argument("--results", required=True)
    rp.add_argument("--baseline", default=str(BASELINE))
    rp.add_argument("--summary", default="", help="append the table here ($GITHUB_STEP_SUMMARY)")
    rp.set_defaults(func=cmd_report)

    ac = sub.add_parser("accept", help="write results.json's peaks into the baseline")
    ac.add_argument("--results", required=True)
    ac.add_argument("--baseline", default=str(BASELINE))
    ac.add_argument("--reason", required=True)
    ac.set_defaults(func=cmd_accept)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
