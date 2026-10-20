#!/usr/bin/env python3
"""Split a CI test run into balanced shards (#2167, #2182).

    tools/ci-test-shard.py run SUITE K N [--lib-parts P] [--output PATH]
                           [--manifest DIR] [--list-all] -- BUILD_ARGS... [-- HARNESS...]
    tools/ci-test-shard.py invocations SUITE K N [--lib-parts P]
    tools/ci-test-shard.py plan SUITE N [--lib-parts P]
    tools/ci-test-shard.py has-target SUITE K N [--lib-parts P] TARGET
    tools/ci-test-shard.py check SUITE N DIR [--result R] [--budget-min M]
    tools/ci-test-shard.py check-nn CORE_DIR NN_DIR [--list PATH]
    tools/ci-test-shard.py weights SUITE LOG [--output PATH]

SUITE is `core` (`Tests + coverage (core)`: the lib and bin unit tests plus every
integration binary in the workspace) or `endpoints` (`Tests + coverage (TTE/CTMM
endpoints)`: the lib unit tests plus the feature-gated binaries listed in
`tools/ci-endpoint-tests.txt`). Each job is a matrix whose shard K of N runs
`run SUITE K N ... -- BUILD_ARGS`, and an upload job sends the N lcov files to
Codecov as one upload.

The `slow` and `slow-nn` suites are `slow-tests.yml` (#2182), and their `run` is a
plain `cargo test`, not cargo-llvm-cov. `slow` is core's target set minus the
`--exclude`d packages, plus the doc tests a bare `cargo test` also ran; `slow-nn`
is the lib unit tests plus `tools/slow-nn-tests.txt`, the binaries that gain a
test under `nn`. Their `run` writes a manifest (see `run_tests`), and the summary
job's `check` turns the shards' manifests into one verdict: a sharded run is green
only if every test target ran on exactly one shard, every binary ran every test
its own `--list` names (none filtered), every shard reported, and no test failed.
`check-nn` diffs the nn leg's `--list` of every binary against the core shards',
so a binary that gains a test under `nn` cannot be missing from the nn list.

Two units of work are scheduled:

* **Integration binaries**, whole. Splitting by binary, not by test (`cargo
  nextest --partition`), keeps the libtest harness: a `OnceLock` fixture shared by
  the tests of one binary (`fit_inputs_tests.rs`, `run_covariance_inner_scope.rs`,
  ...) is computed once per binary, not once per test.
* **The unit-test bundle** (`--lib --bins` for core, `--lib` for endpoints), split
  into P *lib parts* with libtest name filters. It is one binary per package, and
  at 900-1200 s it was the floor of whichever shard ran it. Parts 1..P-1 each run
  `-- FILTER...` (a test runs if its name CONTAINS any filter); part P, the *rest*,
  runs `-- --skip FILTER...` over every filter of the other parts. A test either
  contains one of those filters or it does not, so the parts together run every
  test at least once, whatever the weights say — a module the weights file has
  never heard of lands in the rest. Filters are file-level module paths
  (`estimation::profile::`), so the tests of one source file, and any fixture they
  share, stay in one part. Part p runs on shard p.

The integration target list comes from `cargo metadata` (core) or the endpoint
list (checked against `cargo metadata`), never from the weights file, so a new
test file is always run: with no weight it is scheduled at DEFAULT_WEIGHT. Stale
weights only unbalance the shards; they can never drop a test.

`run` executes every cargo-llvm-cov invocation of the shard with `--no-report`,
then writes the lcov file with `--no-run`. Both halves of that matter, and both
were measured (cargo-llvm-cov 0.8): an invocation WITHOUT `--no-report` deletes
the profiles of the invocations before it, and the `report` subcommand takes no
`--workspace`, so it silently drops every `crates/*` source from the report.
`--no-run --workspace` reads all accumulated profiles across the workspace.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT_TESTS = ROOT / "tools" / "ci-endpoint-tests.txt"
# Seconds assumed for a binary the weights file does not name. Most binaries
# take a few seconds; a new heavy one is caught at the next `weights` refresh.
DEFAULT_WEIGHT = 3.0
# Extra compile a shard pays for linking the lib unit-test harness, on top of the
# rlib and integration binaries every shard builds. Measured on #2168's run
# 39331279992: core 1/4 (lib harness + small binaries) compiled in 9m56s, the
# integration-only shards in 3m34s-5m13s.
LIB_HARNESS_COMPILE = 300.0
UNIT = "<unit>"  # the whole unit-test bundle's wall time
LIB = "lib:"  # prefix of a per-module unit-test weight (summed test seconds)
SLOW_NN_TESTS = ROOT / "tools" / "slow-nn-tests.txt"
SUITES = {
    "core": {"unit": ["--lib", "--bins"]},
    "endpoints": {"unit": ["--lib"]},
    # `doc`: a bare `cargo test` also runs the doc tests, and `--doc` cannot share
    # an invocation with `--lib`. `exclude`: the packages the workflow's build args
    # `--exclude`; `run` refuses build args that disagree, so the target set the
    # planner schedules is the one cargo builds.
    "slow": {"unit": ["--lib", "--bins"], "doc": True, "exclude": ["docs-lint"]},
    "slow-nn": {"unit": ["--lib"], "exclude": ["docs-lint"]},
}
TEST_SUITES = ("slow", "slow-nn")  # `run` is plain `cargo test` + a manifest
# The cargo that `run` invokes; overridable so the guard test can record the calls.
CARGO = os.environ.get("CI_TEST_SHARD_CARGO", "cargo")


def die(msg):
    sys.exit(f"ci-test-shard: {msg}")


def weights_path(suite):
    return ROOT / "tools" / f"ci-test-weights-{suite}.txt"


_META = []


def workspace_packages():
    if not _META:
        _META.append(json.loads(
            subprocess.run(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout
        )["packages"])
    return _META[0]


def workspace_test_targets(exclude=()):
    """Integration-test target names across the workspace (unique, checked)."""
    names = [
        t["name"]
        for p in workspace_packages()
        if p["name"] not in exclude
        for t in p["targets"]
        if "test" in t["kind"] and t.get("test", True)
    ]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        # `--workspace --test NAME` would select the target in every package.
        die(f"test target names are not unique: {dups}")
    return sorted(names)


def read_list(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def test_targets(suite):
    every = workspace_test_targets(SUITES[suite].get("exclude", ()))
    if suite in ("core", "slow"):
        return every
    path = SLOW_NN_TESTS if suite == "slow-nn" else ENDPOINT_TESTS
    listed = read_list(path)
    unknown = sorted(set(listed) - set(every))
    if unknown:
        die(f"{path.name} names test targets that do not exist: {unknown}")
    if len(set(listed)) != len(listed):
        die(f"{path.name} lists a target twice")
    return sorted(listed)


def binary_key(name):
    """A target name as the manifest spells its binary (`-` -> `_`, as rustc does)."""
    return name.replace("-", "_")


def expected_binaries(suite):
    """Every binary a full run of SUITE must report, keyed as `run` keys them:
    `<name>[unit]` for a lib or bin unit-test bundle, `doctest:<lib>` for the doc
    tests, the bare name for an integration binary. From `cargo metadata`, not
    from the planner, so `check` does not agree with the planner by construction."""
    spec = SUITES[suite]
    unit = spec["unit"]
    out = set()
    for p in workspace_packages():
        if p["name"] in spec.get("exclude", ()):
            continue
        for t in p["targets"]:
            kinds = t["kind"]
            lib = any(k in ("lib", "rlib", "dylib", "proc-macro") for k in kinds)
            if lib and t.get("test", True):
                out.add(f"{binary_key(t['name'])}[unit]")
            if lib and spec.get("doc") and t.get("doctest", True):
                out.add(f"doctest:{binary_key(t['name'])}")
            if "bin" in kinds and "--bins" in unit and t.get("test", True):
                out.add(f"{binary_key(t['name'])}[unit]")
    out.update(binary_key(t) for t in test_targets(suite))
    return sorted(out)


def load_weights(suite):
    w = {}
    path = weights_path(suite)
    if path.exists():
        for line in read_list(path):
            secs, name = line.split()
            w[name] = float(secs)
    return w


def lpt(items, bins, load):
    """Longest-processing-time-first; deterministic for fixed inputs."""
    out = [[] for _ in range(bins)]
    for name, secs in sorted(items, key=lambda kv: (-kv[1], kv[0])):
        k = min(range(bins), key=lambda i: (load[i], i))
        out[k].append(name)
        load[k] += secs
    return out


ROOT_TESTS = "-"  # `lib:-`: unit tests at a crate root, never named by a filter


def lib_groups(w):
    """`(filter, seconds)` per file-level module in the weights, nested and
    overlapping modules folded into the outermost one.

    libtest's filter is a SUBSTRING match, so `api::` also selects
    `api::levels::tests::x`, and `estimation::run_covariance::` every module
    below it. Two such modules in different parts would run the inner one twice,
    so a module whose `path::` contains another's is weighed with that one."""
    mods = {k[len(LIB):]: v for k, v in w.items() if k.startswith(LIB)}
    mods.pop(ROOT_TESTS, None)
    groups = {}
    for m, secs in mods.items():
        owners = [a for a in mods if f"{a}::" in f"{m}::"]
        root = min(owners, key=lambda a: (len(a), a))
        groups[root] = groups.get(root, 0.0) + secs
    return [(f"{g}::", v) for g, v in groups.items()]


def lib_parts(w, p):
    """P lists of module filters; the LAST is the rest and is always empty here —
    it runs `--skip` over the union of the others."""
    if p == 1:
        return [[]]
    groups = lib_groups(w)
    if len(groups) < p - 1:
        die(f"--lib-parts {p} needs per-module unit-test weights (`lib:` lines) "
            f"for at least {p - 1} modules; refresh the weights file")
    # The rest takes its share as part P; the modules LPT puts there are simply
    # not named, so they (and every module the weights do not know) run as rest.
    parts = lpt(groups, p, [0.0] * p)
    named = [sorted(x) for x in parts[: p - 1]]
    if any(not x for x in named):
        die(f"--lib-parts {p}: a filtered part came out empty, which would run "
            f"EVERY unit test; use fewer parts")
    return named + [[]]


def part_seconds(w, filters_by_part):
    """Predicted wall seconds of each lib part: the unit bundle's measured wall
    time, shared in proportion to the summed test seconds of each part."""
    total = sum(v for k, v in w.items() if k.startswith(LIB))
    unit = w.get(UNIT, DEFAULT_WEIGHT)
    if not total:
        return [unit] + [0.0] * (len(filters_by_part) - 1)
    groups = dict(lib_groups(w))
    named = [sum(groups[f] for f in fs) for fs in filters_by_part[:-1]]
    shares = named + [total - sum(named)]
    return [unit * s / total for s in shares]


def plan(suite, n, p):
    if not 1 <= p <= n:
        die(f"--lib-parts {p} is outside 1..{n}")
    w = load_weights(suite)
    filters = lib_parts(w, p)
    secs = part_seconds(w, filters)
    load = [0.0] * n
    for i in range(p):
        load[i] = secs[i] + LIB_HARNESS_COMPILE
    targets = [(t, w.get(t, DEFAULT_WEIGHT)) for t in test_targets(suite)]
    binaries = lpt(targets, n, load)
    return filters, binaries, load


def invocations(suite, k, n, p):
    """The cargo target/filter arguments of each invocation shard K runs."""
    if not 1 <= k <= n:
        die(f"shard {k} is outside 1..{n}")
    filters, binaries, _ = plan(suite, n, p)
    out = []
    if k <= p:
        unit = list(SUITES[suite]["unit"])
        if p == 1:
            out.append(unit)
        elif k < p:
            out.append(unit + ["--"] + filters[k - 1])
        else:
            rest = sorted(f for fs in filters[:-1] for f in fs)
            out.append(unit + ["--"] + [a for f in rest for a in ("--skip", f)])
    if k == 1 and SUITES[suite].get("doc"):
        out.append(["--doc"])
    tests = [a for t in binaries[k - 1] for a in ("--test", t)]
    if tests:
        out.append(tests)
    if not out:
        die(f"shard {k}/{n} of {suite} has nothing to run; use fewer shards")
    return out


def split_build(rest):
    """`--`-separated cargo build args from `run`'s own options."""
    if "--" not in rest:
        die("run: pass the cargo build arguments after `--`")
    i = rest.index("--")
    return rest[:i], rest[i + 1:]


def run(suite, k, n, p, output, build):
    """Every invocation with `--no-report` (one without it deletes the profiles
    of the ones before), then one `--no-run --workspace` report. Keeps going
    after a failing invocation, like `--no-fail-fast` within one, and exits
    non-zero if any failed."""
    failed = []
    for inv in invocations(suite, k, n, p):
        if "--" in inv:
            i = inv.index("--")
            targets, harness = inv[:i], inv[i:]
        else:
            targets, harness = inv, []
        cmd = [CARGO, "llvm-cov", "--no-report", "--no-fail-fast", *build, *targets, *harness]
        print("+ " + " ".join(cmd), flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(" ".join(targets))
    cmd = [CARGO, "llvm-cov", "--no-run", *build, "--lcov", "--output-path", output]
    print("+ " + " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        failed.append("report")
    if failed:
        die(f"shard {k}/{n} of {suite}: failed: {failed}")


ANSI = re.compile(r"(?:\x1b|\^\[)\[[0-9;]*[A-Za-z]")
RUNNING = re.compile(r"^\s*Running (unittests )?\S+ \((.+)\)\s*$")
DOC_TESTS = re.compile(r"^\s*Doc-tests (\S+)\s*$")
LISTED = re.compile(r"^(.+): test$")
OUTCOME = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored)\b")
RESULT = re.compile(
    r"^test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored; \d+ measured; "
    r"(\d+) filtered out; finished in ([\d.]+)s"
)


def binary_of(line):
    """The manifest key of the binary a cargo `Running` / `Doc-tests` line starts,
    or None. `target/.../ferx_core-<16 hex>` -> `ferx_core[unit]` for a unit-test
    bundle, `cli_ferx-<hash>` -> `cli_ferx`, `Doc-tests ferx_core` -> `doctest:ferx_core`."""
    m = RUNNING.match(line)
    if m:
        stem = re.sub(r"-[0-9a-f]{16}$", "", Path(m.group(2)).name)
        return binary_key(stem) + ("[unit]" if m.group(1) else "")
    m = DOC_TESTS.match(line)
    if m:
        return f"doctest:{binary_key(m.group(1))}"
    return None


class Manifest:
    """`manifest.tsv`, one tab-separated record per line, written as it goes so a
    shard killed by its timeout still leaves what it saw. Records:

        M key value               shard facts (suite, k, n, status, elapsed, done)
        E binary                  a binary a full run of the suite must report
        B binary                  `--list` of this shard's invocations saw it
        L binary test             ... and it lists this test
        I binary test             ... and `--list --ignored` lists this test
        XB / XL / XI              the same for `--list-all` (every binary of `slow`)
        S binary passed failed ignored filtered seconds   its `test result:` line
        R binary test outcome     one `test ... ok|FAILED|ignored` line
    """

    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.f = open(Path(directory) / "manifest.tsv", "w")

    def row(self, *cols):
        self.f.write("\t".join(str(c) for c in cols) + "\n")
        self.f.flush()


def read_manifest(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line:
            rows.append(line.split("\t"))
    return rows


def stream(cmd, on_line):
    """Run `cmd` with stderr merged into stdout — one pipe, so cargo's `Running`
    line always precedes the output of the binary it starts — echoing every line
    to the log. Returns the exit status."""
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
    )
    for raw in proc.stdout:
        sys.stdout.write(raw)
        sys.stdout.flush()
        on_line(ANSI.sub("", raw.rstrip("\n")))
    return proc.wait()


def build_excludes(build):
    out = set()
    for i, a in enumerate(build):
        if a == "--exclude" and i + 1 < len(build):
            out.add(build[i + 1])
        elif a.startswith("--exclude="):
            out.add(a.split("=", 1)[1])
    return out


def list_phase(man, prefix, invs, build):
    """`--list` and `--list --ignored` of each invocation: what the binaries
    themselves say they hold, the oracle `check` holds the run against."""
    failed = []
    for inv in invs:
        for tag, harness in (("L", ["--list"]), ("I", ["--list", "--ignored"])):
            cur = [None]

            def on_line(line, tag=tag, cur=cur):
                b = binary_of(line)
                if b is not None:
                    cur[0] = b
                    if tag == "L":
                        man.row(prefix + "B", b)
                    return
                m = LISTED.match(line)
                if m and cur[0] is not None:
                    man.row(prefix + tag, cur[0], m.group(1))

            if stream([CARGO, "test", *build, *inv, "--", *harness], on_line) != 0:
                failed.append(f"--list {' '.join(inv)}")
    return failed


def run_tests(suite, k, n, p, directory, list_all, build):
    """Plain `cargo test` for the `slow*` suites, writing `DIR/manifest.tsv`.

    Every invocation is first listed (`--list`, `--list --ignored`), then run with
    `--no-fail-fast`; a failing invocation does not stop the later ones, and the
    shard exits non-zero if any failed. `--list-all` also lists every binary of the
    `slow` suite, for `check-nn`. Lib parts are libtest filters, which `--list`
    honours, so a filtered part could never be held against its own `--list`:
    these suites run `--lib-parts 1` only."""
    if p != 1:
        die(f"{suite}: run with --lib-parts 1 — a filtered lib part cannot be "
            f"checked against the binary's own --list")
    if "--" in build:
        i = build.index("--")
        build, extra = build[:i], build[i + 1:]
    else:
        extra = []
    if "--workspace" not in build:
        die(f"{suite}: the build args must include --workspace")
    want = set(SUITES[suite]["exclude"])
    if build_excludes(build) != want:
        die(f"{suite}: the build args --exclude {sorted(build_excludes(build))}, but the "
            f"suite's target set excludes {sorted(want)}; change both together")
    if "--no-fail-fast" not in build:
        build = ["--no-fail-fast", *build]
    invs = invocations(suite, k, n, p)
    man = Manifest(directory)
    t0 = time.monotonic()
    for key, val in (("suite", suite), ("k", k), ("n", n)):
        man.row("M", key, val)
    for b in expected_binaries(suite):
        man.row("E", b)
    failed = []
    status = "interrupted"
    try:
        failed += list_phase(man, "", invs, build)
        if list_all:
            every = invocations("slow", 1, 1, 1)
            failed += list_phase(man, "X", every, build)
        man.row("M", "list_secs", f"{time.monotonic() - t0:.0f}")
        t1 = time.monotonic()
        for inv in invs:
            cur = [None]

            def on_line(line, cur=cur):
                b = binary_of(line)
                if b is not None:
                    cur[0] = b
                    return
                if cur[0] is None:
                    return
                m = OUTCOME.match(line)
                if m:
                    man.row("R", cur[0], m.group(1), m.group(2))
                    return
                m = RESULT.match(line)
                if m:
                    man.row("S", cur[0], *m.groups())
                    cur[0] = None

            if "--" in inv:
                i = inv.index("--")
                targets, harness = inv[:i], inv[i + 1:]
            else:
                targets, harness = inv, []
            tail = ["--", *harness, *extra] if harness or extra else []
            if stream([CARGO, "test", *build, *targets, *tail], on_line) != 0:
                failed.append(" ".join(targets))
        man.row("M", "test_secs", f"{time.monotonic() - t1:.0f}")
        status = "failed" if failed else "ok"
    finally:
        man.row("M", "elapsed", f"{time.monotonic() - t0:.0f}")
        man.row("M", "status", status)
        man.row("M", "done", "1")
    if failed:
        die(f"shard {k}/{n} of {suite}: failed: {failed}")


class Shard:
    def __init__(self, rows, where):
        self.where = where
        self.meta = {}
        self.expected = set()
        self.lists = {}  # prefix -> {binary: (listed set, ignored set)}
        self.summary = {}  # binary -> list of (passed, failed, ignored, filtered, secs)
        self.failed = {}  # binary -> [test]
        for r in rows:
            t = r[0]
            if t == "M":
                self.meta[r[1]] = r[2]
            elif t == "E":
                self.expected.add(r[1])
            elif t in ("B", "XB"):
                self.lists.setdefault(t[:-1], {}).setdefault(r[1], (set(), set()))
            elif t in ("L", "I", "XL", "XI"):
                pre = t[:-1]
                lst, ign = self.lists.setdefault(pre, {}).setdefault(r[1], (set(), set()))
                (lst if t[-1] == "L" else ign).add(r[2])
            elif t == "S":
                self.summary.setdefault(r[1], []).append(
                    tuple(int(x) for x in r[2:6]) + (float(r[6]),))
            elif t == "R" and r[3] == "FAILED":
                self.failed.setdefault(r[1], []).append(r[2])


def load_shards(directory):
    found = sorted(Path(directory).rglob("manifest.tsv"))
    return [Shard(read_manifest(f), f) for f in found]


def gh(level, msg):
    print(f"::{level}::{msg}", flush=True)


def check(suite, n, directory, result=None, budget_min=None):
    """The summary job's verdict over the shards' manifests. Each failed condition
    is its own `::error::`, naming the condition; exits 1 if any fired."""
    errors = []
    if result is not None and result != "success":
        errors.append(f"the {suite} shard jobs ended '{result}' (a job killed by "
                      f"timeout-minutes reports as cancelled) — see the shard logs")
    shards = {}
    for s in load_shards(directory):
        if s.meta.get("suite") != suite or s.meta.get("n") != str(n):
            errors.append(f"{s.where}: a manifest of suite {s.meta.get('suite')} "
                          f"N={s.meta.get('n')}, not {suite} N={n}")
            continue
        k = int(s.meta["k"])
        if k in shards:
            errors.append(f"two manifests for shard {k}/{n}")
        shards[k] = s
    missing = sorted(set(range(1, n + 1)) - set(shards))
    if missing:
        errors.append(f"no manifest from shard(s) {missing} of {n} — the job never "
                      f"ran its tests, or never uploaded what it saw")
    for k, s in sorted(shards.items()):
        if s.meta.get("done") != "1":
            errors.append(f"shard {k}/{n} never finished writing its manifest "
                          f"(killed at the job's timeout?)")
        elif s.meta.get("status") != "ok":
            errors.append(f"shard {k}/{n} ended '{s.meta.get('status')}'")
    expected = set().union(*(s.expected for s in shards.values())) if shards else set()
    where = {}
    for k, s in sorted(shards.items()):
        for b in s.lists.get("", {}):
            where.setdefault(b, []).append(k)
    for b in sorted(expected - set(where)):
        errors.append(f"{b}: no shard ran this test target")
    for b, ks in sorted(where.items()):
        if len(ks) > 1:
            errors.append(f"{b}: ran on several shards {ks}")
        if b not in expected:
            errors.append(f"{b}: a shard ran a binary outside the {suite} suite")
    for k, s in sorted(shards.items()):
        for b, (listed, ignored) in sorted(s.lists.get("", {}).items()):
            runs = s.summary.get(b, [])
            ran = sum(x[0] + x[1] + x[2] for x in runs)
            filtered = sum(x[3] for x in runs)
            if listed and ran == 0:
                errors.append(f"{b}: its --list names {len(listed)} tests but none "
                              f"ran (shard {k})")
            elif ran != len(listed) or filtered:
                errors.append(f"{b}: ran {ran} of the {len(listed)} tests its --list "
                              f"names, {filtered} filtered out (shard {k})")
            fails = sum(x[1] for x in runs)
            if fails:
                names = " ".join(sorted(s.failed.get(b, []))) or "see the log"
                errors.append(f"{b}: {fails} test(s) FAILED (shard {k}): {names}")
    if budget_min is not None:
        w = load_weights("slow")
        for k, s in sorted(shards.items()):
            secs = float(s.meta.get("elapsed", 0))
            if secs > 0.7 * budget_min * 60:
                gh("warning", f"shard {k}/{n} took {secs / 60:.1f} min, over 70% of its "
                              f"{budget_min} min budget — re-balance or add a shard")
            unit = 0.0
            for b, runs in s.summary.items():
                t = sum(x[4] for x in runs)
                if b.endswith("[unit]"):
                    unit += t
                    continue
                ref = w.get(b)
                if ref and t > 2 * ref and t > 60:
                    gh("warning", f"{b} took {t:.0f} s, over twice its {ref:.0f} s weight "
                                  f"— refresh tools/ci-test-weights-slow.txt")
            ref = w.get(UNIT)
            if ref and unit > 2 * ref and unit > 60:
                gh("warning", f"the unit tests took {unit:.0f} s, over twice their "
                              f"{ref:.0f} s weight")
    lines = [f"| shard | elapsed (min) | binaries | status |", "|---|---|---|---|"]
    for k, s in sorted(shards.items()):
        lines.append(f"| {k}/{n} | {float(s.meta.get('elapsed', 0)) / 60:.1f} | "
                     f"{len(s.lists.get('', {}))} | {s.meta.get('status', '?')} |")
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"### {suite}\n\n{report}\n")
    for e in errors:
        gh("error", e)
    if errors:
        sys.exit(1)
    print(f"{suite}: {len(where)} binaries over {n} shard(s), every listed test ran once")


def check_nn(core_dir, nn_dir, list_path=SLOW_NN_TESTS):
    """The binaries the nn leg must run are exactly those that gain a runnable
    test under `nn`: a test `--list` names that `--list --ignored` does not. The
    nn leg's `--list-all` is held against the core shards' lists, binary by
    binary; the lib unit tests always run there. Read from the compiler, not
    from the source text, so a `cfg` spelled in a way a grep misses still counts."""
    errors = []
    core = {}
    for s in load_shards(core_dir):
        for b, v in s.lists.get("", {}).items():
            core[b] = v
    nn = [s for s in load_shards(nn_dir)]
    if not core:
        errors.append(f"no core-shard lists under {core_dir}")
    if len(nn) != 1 or "X" not in nn[0].lists:
        errors.append(f"no nn --list-all manifest under {nn_dir}")
    if errors:
        for e in errors:
            gh("error", e)
        sys.exit(1)
    nn = nn[0]
    every = nn.lists["X"]
    ran = set(nn.summary)
    for b in sorted(set(core) ^ set(every)):
        side = "core" if b in core else "nn"
        errors.append(f"{b}: only the {side} build has this binary")
    gains = {}
    for b in sorted(set(core) & set(every)):
        runnable_core = core[b][0] - core[b][1]
        runnable_nn = every[b][0] - every[b][1]
        if runnable_nn - runnable_core:
            gains[b] = len(runnable_nn - runnable_core)
    listed = set(binary_key(t) for t in read_list(Path(list_path)))
    for b, g in sorted(gains.items()):
        if b not in ran:
            errors.append(f"{b}: gains {g} test(s) under nn but the nn leg does not run "
                          f"it — add it to tools/slow-nn-tests.txt")
    for b in sorted(listed - set(gains)):
        errors.append(f"{b}: listed in tools/slow-nn-tests.txt but gains no test under "
                      f"nn — drop it")
    for e in errors:
        gh("error", e)
    if errors:
        sys.exit(1)
    print(f"nn: {len(gains)} binaries gain tests under nn, all run: "
          + " ".join(f"{b}(+{g})" for b, g in sorted(gains.items())))


def module_of(test):
    """File-level module of a unit test: the path before its first `tests` /
    `*_tests` segment (`estimation::profile::tests::x` -> `estimation::profile`),
    `lib_tests::x` -> `lib_tests`; ROOT_TESTS for a test at the crate root."""
    segs = test.split("::")
    if len(segs) == 1:
        return ROOT_TESTS
    for i, s in enumerate(segs):
        if s == "tests" or s.endswith("_tests"):
            return "::".join(segs[: max(i, 1)])
    return "::".join(segs[:-1])


def unit_test_seconds(lines, t0, threads):
    """Per-test seconds of one libtest run from the log's completion times.

    libtest starts tests in name order on `threads` workers and starts the next
    one the moment one finishes, so the i-th started test (i >= threads) started
    at the (i - threads)-th completion; test i < threads started at `t0`. A
    `--report-time` suffix (`<1.234s>`, nightly or RUSTC_BOOTSTRAP) is used
    directly when present."""
    done = []  # (completion time, name, reported secs or None)
    for ts, name, rep in lines:
        done.append((ts, name, rep))
    order = sorted(name for _, name, _ in done)
    starts = {}
    comp = sorted(ts for ts, _, _ in done)
    for i, name in enumerate(order):
        starts[name] = t0 if i < threads else comp[i - threads]
    return {
        name: (rep if rep is not None else max(ts - starts[name], 0.0))
        for ts, name, rep in done
    }


def stamp(line):
    # `search`, not `match`: a log fetched with `gh run view --log` prefixes every
    # line with `job<TAB>step<TAB>` before the timestamp (#2182).
    m = re.search(r"(\d{4}-\d\d-\d\dT(\d\d):(\d\d):(\d\d(?:\.\d+)?))Z ", line)
    if not m:
        return None
    return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4))


def weights_from_log(suite, log, threads=4, output=None):
    """`finished in Xs` per `Running` binary, plus per-module unit-test seconds."""
    w = {}
    mods = {}
    cur = None
    unit_lines = None
    t0 = None
    for line in Path(log).read_text(errors="replace").splitlines():
        # Both spellings of an SGR colour code: the raw ESC of a downloaded log, and
        # the caret form `^[[32m` that `gh run view --log` writes (#2182).
        line = ANSI.sub("", line)
        m = re.search(r"Running (unittests )?(\S+)", line)
        if m:
            cur = UNIT if m.group(1) else Path(m.group(2)).stem
            unit_lines = [] if cur == UNIT else None
            t0 = stamp(line)
            continue
        if unit_lines is not None:
            m = re.search(r" test (\S+) \.\.\. (?:ok|FAILED|ignored(?:, .*)?)(?: <([\d.]+)s>)?$", line)
            if m and stamp(line) is not None:
                rep = float(m.group(2)) if m.group(2) else None
                unit_lines.append((stamp(line), m.group(1), rep))
        m = re.search(r"test result: \w+\..*finished in ([\d.]+)s", line)
        if m and cur:
            w[cur] = w.get(cur, 0.0) + float(m.group(1))
            if cur == UNIT and unit_lines and t0 is not None:
                for name, secs in unit_test_seconds(unit_lines, t0, threads).items():
                    key = module_of(name)
                    mods[key] = mods.get(key, 0.0) + secs
            cur = None
            unit_lines = None
    if UNIT not in w:
        die(f"no unit-test timings in {log}; wrong log?")
    rows = sorted(w.items(), key=lambda kv: (-kv[1], kv[0]))
    head = (
        f"# Per-binary test seconds for tools/ci-test-shard.py, suite `{suite}` (#2167).\n"
        f"# Generated by `tools/ci-test-shard.py weights {suite} <log>` from a CI log of\n"
        "# that suite; binaries under DEFAULT_WEIGHT are omitted. `<unit>` is the unit-\n"
        "# test bundle's wall time and `lib:<module>` the summed seconds of that\n"
        "# module's unit tests, which place the lib-part filters. Only balances the\n"
        "# shards — a stale or missing entry never drops a test.\n"
    )
    body = "".join(f"{s:.0f} {n}\n" for n, s in rows if s >= DEFAULT_WEIGHT)
    body += "".join(
        f"{s:.0f} {LIB}{n}\n"
        for n, s in sorted(mods.items(), key=lambda kv: (-kv[1], kv[0]))
        if s >= 1.0
    )
    Path(output or weights_path(suite)).write_text(head + body)


def parse_opts(argv):
    """Positional args plus the options; `opts` holds the ones only some
    subcommands take (`--manifest`, `--list-all`, `--result`, `--budget-min`)."""
    p, output, i = 1, None, 0
    pos, opts = [], {}
    while i < len(argv):
        a = argv[i]
        if a == "--lib-parts":
            p = int(argv[i + 1])
            i += 2
        elif a == "--output":
            output = argv[i + 1]
            i += 2
        elif a in ("--manifest", "--result", "--budget-min", "--list"):
            opts[a[2:]] = argv[i + 1]
            i += 2
        elif a == "--list-all":
            opts["list-all"] = True
            i += 1
        else:
            pos.append(a)
            i += 1
    return pos, p, output, opts


def suite_of(name):
    if name not in SUITES:
        die(f"unknown suite `{name}`; one of {sorted(SUITES)}")
    return name


def main(argv):
    if len(argv) < 2:
        die(__doc__)
    cmd, rest = argv[1], argv[2:]
    if cmd == "run":
        own, build = split_build(rest)
        pos, p, output, opts = parse_opts(own)
        if len(pos) != 3:
            die(__doc__)
        suite, k, n = suite_of(pos[0]), int(pos[1]), int(pos[2])
        if suite in TEST_SUITES:
            run_tests(suite, k, n, p, opts.get("manifest", "manifest"),
                      opts.get("list-all", False), build)
        else:
            run(suite, k, n, p, output or "lcov.info", build)
        return
    pos, p, output, opts = parse_opts(rest)
    if cmd == "invocations" and len(pos) == 3:
        for inv in invocations(suite_of(pos[0]), int(pos[1]), int(pos[2]), p):
            print(" ".join(inv))
    elif cmd == "plan" and len(pos) == 2:
        suite, n = suite_of(pos[0]), int(pos[1])
        filters, binaries, load = plan(suite, n, p)
        for i in range(n):
            lib = f"lib part {i + 1}/{p}, " if i < p else ""
            print(f"shard {i + 1}: {load[i]:6.0f} s, {lib}{len(binaries[i]):3d} binaries: "
                  f"{' '.join(binaries[i][:6])} ...")
        for i, fs in enumerate(filters[:-1], 1):
            print(f"lib part {i}: {' '.join(fs)}")
    elif cmd == "has-target" and len(pos) == 4:
        # One `present=true|false` line for `$GITHUB_OUTPUT`: the planner is the
        # step's command, so an error here fails the step instead of reading false.
        suite, k, n, name = suite_of(pos[0]), int(pos[1]), int(pos[2]), pos[3]
        if name not in test_targets(suite):
            die(f"has-target: `{name}` is not a test target of {suite}")
        hit = any(inv[i + 1] == name for inv in invocations(suite, k, n, p)
                  for i, a in enumerate(inv[:-1]) if a == "--test")
        print(f"present={'true' if hit else 'false'}")
    elif cmd == "check" and len(pos) == 3:
        budget = opts.get("budget-min")
        check(suite_of(pos[0]), int(pos[1]), pos[2], opts.get("result"),
              float(budget) if budget else None)
    elif cmd == "check-nn" and len(pos) == 2:
        check_nn(pos[0], pos[1], opts.get("list", SLOW_NN_TESTS))
    elif cmd == "weights" and len(pos) == 2:
        weights_from_log(suite_of(pos[0]), pos[1], output=output)
    else:
        die(__doc__)


if __name__ == "__main__":
    main(sys.argv)
