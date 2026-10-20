#!/usr/bin/env python3
"""Split a per-PR coverage test run into balanced shards (#2167).

    tools/ci-test-shard.py run SUITE K N [--lib-parts P] [--output PATH] -- BUILD_ARGS...
    tools/ci-test-shard.py invocations SUITE K N [--lib-parts P]
    tools/ci-test-shard.py plan SUITE N [--lib-parts P]
    tools/ci-test-shard.py weights SUITE LOG

SUITE is `core` (`Tests + coverage (core)`: the lib and bin unit tests plus every
integration binary in the workspace) or `endpoints` (`Tests + coverage (TTE/CTMM
endpoints)`: the lib unit tests plus the feature-gated binaries listed in
`tools/ci-endpoint-tests.txt`). Each job is a matrix whose shard K of N runs
`run SUITE K N ... -- BUILD_ARGS`, and an upload job sends the N lcov files to
Codecov as one upload.

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
SUITES = {
    "core": {"unit": ["--lib", "--bins"]},
    "endpoints": {"unit": ["--lib"]},
}
# The cargo that `run` invokes; overridable so the guard test can record the calls.
CARGO = os.environ.get("CI_TEST_SHARD_CARGO", "cargo")


def die(msg):
    sys.exit(f"ci-test-shard: {msg}")


def weights_path(suite):
    return ROOT / "tools" / f"ci-test-weights-{suite}.txt"


def workspace_test_targets():
    """Integration-test target names across the workspace (unique, checked)."""
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
    )
    names = [
        t["name"]
        for p in meta["packages"]
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
    every = workspace_test_targets()
    if suite == "core":
        return every
    listed = read_list(ENDPOINT_TESTS)
    unknown = sorted(set(listed) - set(every))
    if unknown:
        die(f"{ENDPOINT_TESTS.name} names test targets that do not exist: {unknown}")
    if len(set(listed)) != len(listed):
        die(f"{ENDPOINT_TESTS.name} lists a target twice")
    return sorted(listed)


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
    m = re.match(r"(\d{4}-\d\d-\d\dT(\d\d):(\d\d):(\d\d(?:\.\d+)?))Z ", line)
    if not m:
        return None
    return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4))


def weights_from_log(suite, log, threads=4):
    """`finished in Xs` per `Running` binary, plus per-module unit-test seconds."""
    w = {}
    mods = {}
    cur = None
    unit_lines = None
    t0 = None
    for line in Path(log).read_text(errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
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
    weights_path(suite).write_text(head + body)


def parse_opts(argv):
    p, output, i = 1, "lcov.info", 0
    pos = []
    while i < len(argv):
        a = argv[i]
        if a == "--lib-parts":
            p = int(argv[i + 1])
            i += 2
        elif a == "--output":
            output = argv[i + 1]
            i += 2
        else:
            pos.append(a)
            i += 1
    return pos, p, output


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
        pos, p, output = parse_opts(own)
        if len(pos) != 3:
            die(__doc__)
        run(suite_of(pos[0]), int(pos[1]), int(pos[2]), p, output, build)
        return
    pos, p, _ = parse_opts(rest)
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
    elif cmd == "weights" and len(pos) == 2:
        weights_from_log(suite_of(pos[0]), pos[1])
    else:
        die(__doc__)


if __name__ == "__main__":
    main(sys.argv)
