//! Guard: the sharded coverage jobs run every test exactly as the unsharded ones
//! did (#2167).
//!
//! `Tests + coverage (core)` used to be one `cargo llvm-cov --workspace --tests`,
//! and `Tests + coverage (TTE/CTMM endpoints)` one `--lib --test …` invocation.
//! Both are now matrices whose shard `k` runs `tools/ci-test-shard.py run SUITE k
//! N --lib-parts P`, so the old selection holds only if the union of the shards'
//! invocations is that selection. A planner bug that dropped a binary, or a
//! module of unit tests, would leave every shard green — the dropped tests simply
//! never run, and their lines read as uncovered against the patch gate with
//! nothing pointing at the cause. A binary scheduled twice wastes a shard's time.
//!
//! The oracle for the integration binaries is the filesystem (core) and the
//! endpoint list (endpoints), not `cargo metadata`, because `cargo metadata` is
//! what the script itself reads: agreeing with it would be agreeing with itself.
//! Cargo auto-discovers `<member>/tests/*.rs` and `<member>/tests/<dir>/main.rs`
//! (no member declares `[[test]]` or `autotests = false`), so that is the set.
//!
//! The unit tests cannot be listed from here (that needs the lib test harness
//! built), so their completeness is pinned structurally: the lib parts are libtest
//! substring filters, and the last part `--skip`s exactly the union of the others'
//! filters. Every test either contains one of those filters or does not, so that
//! shape runs every unit test, whatever the filters are.
//!
//! Not feature-gated: it must run in the base `--features ci` job it guards.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Integration-test target names Cargo auto-discovers under one member's `tests/`.
fn discovered_tests(tests_dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(tests_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        if path.is_file() && path.extension().is_some_and(|e| e == "rs") {
            out.push(stem);
        } else if path.is_dir() && path.join("main.rs").is_file() {
            out.push(stem);
        }
    }
}

fn every_test_target() -> Vec<String> {
    let root = repo_root();
    let mut names = Vec::new();
    discovered_tests(&root.join("tests"), &mut names);
    for member in std::fs::read_dir(root.join("crates")).unwrap().flatten() {
        discovered_tests(&member.path().join("tests"), &mut names);
    }
    names.sort();
    names
}

fn endpoint_list() -> Vec<String> {
    std::fs::read_to_string(repo_root().join("tools/ci-endpoint-tests.txt"))
        .unwrap()
        .lines()
        .map(|l| l.split('#').next().unwrap().trim())
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

fn shard_tool(args: &[String]) -> std::process::Output {
    Command::new("python3")
        .arg(repo_root().join("tools/ci-test-shard.py"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("python3 runs tools/ci-test-shard.py")
}

/// The argument list of each cargo invocation shard `k` of `n` runs.
fn invocations(suite: &str, k: usize, n: usize, p: usize) -> Vec<Vec<String>> {
    let args: Vec<String> = ["invocations", suite, &k.to_string(), &n.to_string()]
        .into_iter()
        .map(str::to_owned)
        .chain(["--lib-parts".to_owned(), p.to_string()])
        .collect();
    let out = shard_tool(&args);
    assert!(
        out.status.success(),
        "ci-test-shard.py {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().map(str::to_owned).collect())
        .collect()
}

/// One lib part: the shard that runs it, and its filters (positive or `--skip`).
struct LibPart {
    shard: usize,
    unit: Vec<String>,
    filters: Vec<String>,
    skips: Vec<String>,
}

/// Checks one suite at one (N, P): every expected binary on exactly one shard, and
/// P lib parts on shards 1..=P whose filters cover every unit test.
fn check_plan(suite: &str, unit: &[&str], expected: &[String], n: usize, p: usize) {
    let mut seen: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut parts = Vec::new();
    let mut docs = Vec::new();
    for k in 1..=n {
        for inv in invocations(suite, k, n, p) {
            if inv == ["--doc"] {
                docs.push(k);
                continue;
            }
            if inv.first().is_some_and(|a| a == "--test") {
                let mut it = inv.iter();
                while let Some(flag) = it.next() {
                    assert_eq!(
                        flag, "--test",
                        "{suite} {k}/{n}: unexpected `{flag}` in {inv:?}"
                    );
                    seen.entry(it.next().unwrap().clone()).or_default().push(k);
                }
                continue;
            }
            let (targets, harness) = match inv.iter().position(|a| a == "--") {
                Some(i) => (inv[..i].to_vec(), inv[i + 1..].to_vec()),
                None => (inv.clone(), Vec::new()),
            };
            let mut part = LibPart {
                shard: k,
                unit: targets,
                filters: vec![],
                skips: vec![],
            };
            let mut it = harness.into_iter();
            while let Some(a) = it.next() {
                if a == "--skip" {
                    part.skips.push(it.next().expect("`--skip` with no filter"));
                } else {
                    assert!(
                        !a.starts_with('-'),
                        "{suite} {k}/{n}: unexpected harness flag `{a}`"
                    );
                    part.filters.push(a);
                }
            }
            parts.push(part);
        }
    }
    let tag = format!("{suite} N={n} P={p}");
    // A bare `cargo test` — what slow-tests.yml ran before #2182 — also runs the
    // doc tests; the coverage suites never did (`--tests`).
    assert_eq!(
        docs,
        if suite == "slow" { vec![1] } else { vec![] },
        "{tag}: the doc tests must run exactly once in `slow`, and never elsewhere"
    );

    let missing: Vec<_> = expected.iter().filter(|t| !seen.contains_key(*t)).collect();
    assert!(
        missing.is_empty(),
        "{tag}: test binaries no shard runs: {missing:?}"
    );
    let twice: Vec<_> = seen.iter().filter(|(_, ks)| ks.len() > 1).collect();
    assert!(
        twice.is_empty(),
        "{tag}: test binaries run on several shards: {twice:?}"
    );
    let unknown: Vec<_> = seen.keys().filter(|t| !expected.contains(t)).collect();
    assert!(
        unknown.is_empty(),
        "{tag}: shards name tests outside the suite: {unknown:?}"
    );

    assert_eq!(
        parts.iter().map(|x| x.shard).collect::<Vec<_>>(),
        (1..=p).collect::<Vec<_>>(),
        "{tag}: lib part i must run once, on shard i"
    );
    for x in &parts {
        assert_eq!(
            x.unit, unit,
            "{tag}: lib part on shard {} runs other targets",
            x.shard
        );
    }
    if p == 1 {
        let x = &parts[0];
        assert!(
            x.filters.is_empty() && x.skips.is_empty(),
            "{tag}: a single lib part must run every unit test unfiltered"
        );
        return;
    }
    let (rest, named) = parts.split_last().unwrap();
    assert!(
        rest.filters.is_empty(),
        "{tag}: the last lib part must be `--skip`-only"
    );
    for x in named {
        assert!(
            !x.filters.is_empty(),
            "{tag}: lib part on shard {} has no filter, so it runs EVERY unit test",
            x.shard
        );
        assert!(x.skips.is_empty(), "{tag}: a filtered lib part also skips");
    }
    let union: BTreeSet<_> = named
        .iter()
        .flat_map(|x| x.filters.iter().cloned())
        .collect();
    let skipped: BTreeSet<_> = rest.skips.iter().cloned().collect();
    assert_eq!(
        skipped, union,
        "{tag}: the rest part must skip exactly the other parts' filters — a skip with \
         no matching filter drops those unit tests, a filter with no skip runs them twice"
    );
}

/// Every shard count and lib-part count, not just the ones `ci.yml` uses today: the
/// matrix size is a one-line edit, and the planner must be complete for whatever it
/// becomes.
#[test]
fn the_core_shards_run_every_test_exactly_once() {
    let all = every_test_target();
    assert!(
        all.len() > 100,
        "found only {} test targets — is the discovery looking in the right place?",
        all.len()
    );
    for n in 1..=6 {
        for p in 1..=n.min(3) {
            check_plan("core", &["--lib", "--bins"], &all, n, p);
        }
    }
}

#[test]
fn the_endpoint_shards_run_every_listed_test_exactly_once() {
    let listed = endpoint_list();
    assert!(
        listed.len() > 10,
        "tools/ci-endpoint-tests.txt lists only {listed:?}"
    );
    for n in 1..=4 {
        for p in 1..=n {
            check_plan("endpoints", &["--lib"], &listed, n, p);
        }
    }
}

/// `run` is what makes N invocations one coverage report, and both halves of its
/// recipe were measured, not assumed (cargo-llvm-cov 0.8): an invocation without
/// `--no-report` DELETES the profiles of the ones before it, and the `report`
/// subcommand has no `--workspace`, so it silently drops `crates/*` from the lcov.
/// A fake cargo records the calls. Its first `llvm-cov` call fails, so this also
/// pins that a red invocation neither stops the later ones (`--no-fail-fast`
/// across invocations) nor lets the shard exit green.
#[test]
fn run_reports_once_over_every_invocation_and_fails_if_any_did() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ci_test_shards_fake_cargo");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("calls.txt");
    let fake = dir.join("cargo");
    std::fs::write(
        &fake,
        format!(
            "#!/bin/sh\necho \"$*\" >> '{log}'\n[ \"$(wc -l < '{log}')\" -eq 1 ] && exit 3\nexit 0\n",
            log = log.display()
        ),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (k, n, p) = (2, 3, 2);
    let out = Command::new("python3")
        .arg(repo_root().join("tools/ci-test-shard.py"))
        .args([
            "run",
            "core",
            &k.to_string(),
            &n.to_string(),
            "--lib-parts",
            &p.to_string(),
        ])
        .args([
            "--output",
            "out.info",
            "--",
            "--workspace",
            "--profile",
            "ci-cov",
        ])
        .env("CI_TEST_SHARD_CARGO", &fake)
        .current_dir(repo_root())
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "a failing invocation left the shard green: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let calls: Vec<String> = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let planned = invocations("core", k, n, p);
    assert_eq!(
        calls.len(),
        planned.len() + 1,
        "expected every planned invocation plus one report, despite the first failing: {calls:#?}"
    );
    let (report, runs) = calls.split_last().unwrap();
    for (call, inv) in runs.iter().zip(&planned) {
        assert!(
            call.starts_with("llvm-cov --no-report --no-fail-fast --workspace --profile ci-cov "),
            "an invocation without `--no-report` deletes the earlier profiles: `{call}`"
        );
        assert!(
            call.ends_with(&inv.join(" ")),
            "`{call}` is not the planned {inv:?}"
        );
    }
    assert_eq!(
        report, "llvm-cov --no-run --workspace --profile ci-cov --lcov --output-path out.info",
        "the report must be one `--no-run` over the same build args"
    );
}

/// The body of job `name` in `ci.yml`, comment lines dropped (prose about a key must
/// not satisfy a check for the key itself).
fn job_body(name: &str) -> String {
    job_body_in(".github/workflows/ci.yml", name)
}

/// [`job_body`] for any workflow file.
fn job_body_in(workflow: &str, name: &str) -> String {
    let yml = std::fs::read_to_string(repo_root().join(workflow)).unwrap();
    let header = format!("\n  {name}:\n");
    let start = yml
        .find(&header)
        .unwrap_or_else(|| panic!("no job `{name}`"))
        + header.len();
    yml[start..]
        .lines()
        .take_while(|l| l.is_empty() || l.starts_with("   ") || l.trim_start().starts_with('#'))
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Both `Tests + coverage (core)` and `Tests + coverage (TTE/CTMM endpoints)` are
/// REQUIRED checks in the `main-protection` ruleset, and since the split each is an
/// upload job, not the shards. GitHub counts a skipped job as passing a required
/// check, and a bare `needs:` skips the job when a shard fails — so without both
/// pieces below a PR with red tests shows every required check green. Each
/// assertion dies on its own deletion, for each pair.
#[test]
fn each_required_coverage_check_goes_red_with_any_shard() {
    for (upload, shards, check) in [
        (
            "coverage-pr-upload",
            "coverage-pr",
            "Tests + coverage (core)",
        ),
        (
            "endpoints-upload",
            "endpoints",
            "Tests + coverage (TTE/CTMM endpoints)",
        ),
    ] {
        let body = job_body(upload);
        assert!(
            body.contains(&format!("name: {check}\n")),
            "`{upload}` no longer carries the required `{check}` name"
        );
        assert!(
            body.contains(&format!("needs: {shards}\n")),
            "`{upload}` does not wait for the `{shards}` shards"
        );
        let cond = body
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with("if:"))
            .unwrap_or_else(|| panic!("`{upload}` has no job-level `if:`"));
        // `always()`, not `!cancelled()`: the latter skips this job when a shard is
        // cancelled, and a cancel mid-run would then read green.
        assert!(
            cond.contains("always()"),
            "`{upload}`: `{cond}`: without `always()` a failed or cancelled shard SKIPS \
             this job, and a skipped required check counts as passing"
        );
        let guard = format!("if: needs.{shards}.result != 'success'");
        let at = body.find(&guard).unwrap_or_else(|| {
            panic!(
                "`{upload}` no longer fails on a non-successful shard — with `always()` \
                 it would then upload partial coverage and report green"
            )
        });
        // The guard step must actually fail: its body runs up to the next step.
        let step = &body[at..];
        let step = &step[..step[guard.len()..]
            .find("\n      - ")
            .map_or(step.len(), |i| i + guard.len())];
        assert!(
            step.lines().any(|l| l.trim() == "exit 1"),
            "`{upload}`'s shard-result step no longer exits non-zero, so it reports and \
             passes:\n{step}"
        );
        assert!(
            !job_body(shards).contains(&format!("name: {check}\n")),
            "the `{shards}` shards carry the bare required name `{check}` — the matrix \
             name must keep its `k/N` suffix so only `{upload}` reports it"
        );
    }
}

/// Each shard's planner must run where `bash -e` sees its exit status (#2168). Inlined
/// as `cargo llvm-cov … $(python3 tools/ci-test-shard.py …)`, a planner error is
/// swallowed (a simple command's status is the command's, not its substitutions') and
/// cargo runs with no target arguments — its default targets, green, the wrong suite.
/// Since #2167's `run`, the planner IS the step's command: pinned for both jobs.
#[test]
fn a_planner_failure_fails_the_shard() {
    for (job, suite) in [("coverage-pr", "core"), ("endpoints", "endpoints")] {
        let body = job_body(job);
        let run = body
            .lines()
            .map(str::trim)
            .find(|l| l.contains("ci-test-shard.py"))
            .unwrap_or_else(|| panic!("`{job}` no longer runs tools/ci-test-shard.py"));
        assert!(
            run.starts_with(&format!("run: python3 tools/ci-test-shard.py run {suite} ")),
            "`{job}`: `{run}` — the planner must be the step's own command, so its \
             failure is the step's failure"
        );
        assert!(
            !body.contains("$(") && !body.contains("cargo llvm-cov"),
            "`{job}` substitutes a command inline or calls cargo-llvm-cov around the \
             planner; a planner error there would be swallowed"
        );
    }
}

/// The keys of the two-space-indented mapping under the top-level `section:` line
/// (`on:` → the triggers, `jobs:` → the job keys).
fn top_level_keys(yml: &str, section: &str) -> Vec<String> {
    yml.lines()
        .skip_while(|l| l.trim_end() != format!("{section}:"))
        .skip(1)
        .take_while(|l| l.is_empty() || l.starts_with(' ') || l.starts_with('#'))
        .filter_map(|l| {
            let key = l.strip_prefix("  ")?.strip_suffix(':')?;
            (!key.starts_with([' ', '#', '-'])
                && key
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .then(|| key.to_owned())
        })
        .collect()
}

/// Whether a job-level `if:` admits `event`. Deliberately small: it reads only the
/// `github.event_name == / != '<event>'` comparisons ci.yml uses, and treats a
/// condition naming no `== '…'` event as event-independent (`always()` etc.).
fn runs_on(cond: Option<&str>, event: &str) -> bool {
    let Some(c) = cond else { return true };
    if c.contains(&format!("github.event_name != '{event}'")) {
        return false;
    }
    if c.contains("github.event_name == '") {
        return c.contains(&format!("github.event_name == '{event}'"));
    }
    true
}

/// `codecov.yml` holds Codecov's status until `after_n_builds` reports arrive, which is
/// right only while every ci.yml trigger that uploads at all uploads exactly that many.
/// Built as a truth table — every trigger in `on:` × every job holding an upload step,
/// through the job's `if:` — so a trigger dropped from one job's condition (dispatch
/// uploading only `endpoints`, pending forever) or a new upload job fails here. Zero is
/// allowed: an event that uploads nothing (a merge queue, #2043) expects no status.
#[test]
fn codecov_waits_for_every_report_of_an_event() {
    let root = repo_root();
    let codecov = std::fs::read_to_string(root.join("codecov.yml")).unwrap();
    let want: usize = codecov
        .lines()
        .find_map(|l| l.trim().strip_prefix("after_n_builds:"))
        .expect("codecov.yml no longer sets `after_n_builds`")
        .trim()
        .parse()
        .unwrap();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).unwrap();
    let events = top_level_keys(&ci, "on");
    assert!(
        events.iter().any(|e| e == "pull_request"),
        "trigger parse found {events:?}"
    );
    let uploaders: Vec<(String, Option<String>)> = top_level_keys(&ci, "jobs")
        .into_iter()
        .flat_map(|job| {
            let body = job_body(&job);
            let cond = body
                .lines()
                .find(|l| l.starts_with("    if:"))
                .map(|l| l.trim().to_owned());
            // One row per upload STEP, so a second upload in one job counts twice.
            let rows: Vec<_> = body
                .match_indices("codecov/codecov-action@")
                .map(|(at, _)| {
                    // The evaluator reads job-level conditions only, so an upload step
                    // must carry none of its own.
                    let step_start = body[..at].rfind("\n      - ").unwrap_or(0);
                    let step_end = body[at..].find("\n      - ").map_or(body.len(), |i| at + i);
                    assert!(
                        !body[step_start..step_end].contains("\n        if:"),
                        "job `{job}` has a Codecov step with its own `if:`, which this \
                         table ignores"
                    );
                    (job.clone(), cond.clone())
                })
                .collect();
            rows
        })
        .collect();
    assert!(
        uploaders.len() >= 2,
        "found only {uploaders:?} uploading to Codecov"
    );
    for event in &events {
        let n = uploaders
            .iter()
            .filter(|(_, c)| runs_on(c.as_deref(), event))
            .count();
        assert!(
            n == 0 || n == want,
            "on `{event}` ci.yml uploads {n} Codecov report(s) but codecov.yml waits for \
             {want} (`after_n_builds`); uploaders: {uploaders:?}"
        );
    }
}

/// A shard number outside `1..=N` is a workflow typo, not an empty shard: running
/// `cargo llvm-cov` with no target arguments would fall back to the package's
/// default targets and silently test something else. Likewise more lib parts than
/// shards, or an unknown suite.
#[test]
fn an_out_of_range_shard_is_refused() {
    for args in [
        ["invocations", "core", "0", "4", "--lib-parts", "1"],
        ["invocations", "core", "5", "4", "--lib-parts", "1"],
        ["invocations", "endpoints", "4", "3", "--lib-parts", "1"],
        ["invocations", "core", "1", "2", "--lib-parts", "3"],
        ["invocations", "nope", "1", "2", "--lib-parts", "1"],
    ] {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let out = shard_tool(&args);
        assert!(!out.status.success(), "{args:?} was accepted");
        // Refused by a check that names the problem, not by a crash further in.
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.starts_with("ci-test-shard: ") && !err.contains("Traceback"),
            "{args:?} was refused without saying why:\n{err}"
        );
    }
}

// ---------------------------------------------------------------------------
// `slow-tests.yml` (#2182): the `slow` / `slow-nn` suites, their manifests, and the
// summary jobs that turn four shards into one verdict.
// ---------------------------------------------------------------------------

const SLOW_YML: &str = ".github/workflows/slow-tests.yml";

/// The packages `slow-tests.yml` builds with `--exclude`; their tests are not the
/// slow suite's. Kept literal here: `slow_test_path_filter.rs` reads the yml's own
/// list, and the planner refuses build args that disagree with its suite.
const SLOW_EXCLUDED_MEMBERS: &[&str] = &["docs-lint"];

/// [`every_test_target`] without the members `slow-tests.yml` excludes.
fn slow_test_targets() -> Vec<String> {
    let root = repo_root();
    let mut names = Vec::new();
    discovered_tests(&root.join("tests"), &mut names);
    for member in std::fs::read_dir(root.join("crates")).unwrap().flatten() {
        let dir = member.file_name().to_string_lossy().into_owned();
        if !SLOW_EXCLUDED_MEMBERS.contains(&dir.as_str()) {
            discovered_tests(&member.path().join("tests"), &mut names);
        }
    }
    names.sort();
    names
}

fn slow_nn_list() -> Vec<String> {
    std::fs::read_to_string(repo_root().join("tools/slow-nn-tests.txt"))
        .unwrap()
        .lines()
        .map(|l| l.split('#').next().unwrap().trim())
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

/// T1. Same oracle as the core suite, minus the excluded members, at every N: the
/// workflow's 4 is a one-line edit. Plus the doc tests, once (see `check_plan`).
#[test]
fn the_slow_shards_run_every_test_exactly_once() {
    let all = slow_test_targets();
    assert!(
        all.len() > 100 && all.len() < every_test_target().len(),
        "slow discovery found {} targets of {} — is the docs-lint exclusion applied?",
        all.len(),
        every_test_target().len()
    );
    for n in 1..=6 {
        check_plan("slow", &["--lib", "--bins"], &all, n, 1);
    }
    check_plan("slow-nn", &["--lib"], &slow_nn_list(), 1, 1);
}

/// `run OWN... -- BUILD...` for a call that must be refused before any cargo call: a
/// fake cargo that fails, and a scratch manifest directory, so a refusal that
/// regresses into a real run neither builds the workspace nor writes into the tree.
fn refused_run(own: &[&str], build: &[&str]) -> std::process::Output {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ci_test_shards_refused");
    std::fs::create_dir_all(&dir).unwrap();
    let fake = dir.join("cargo");
    std::fs::write(&fake, "#!/bin/sh\nexit 99\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    Command::new("python3")
        .arg(repo_root().join("tools/ci-test-shard.py"))
        .arg("run")
        .args(own)
        .arg("--manifest")
        .arg(dir.join("manifest"))
        .arg("--")
        .args(build)
        .env("CI_TEST_SHARD_CARGO", &fake)
        .current_dir(repo_root())
        .output()
        .unwrap()
}

/// T2. The slow suites run the unit tests as ONE unfiltered invocation: a lib part
/// is a libtest filter, which `--list` honours too, so a filtered part could never
/// be held against the binary's own `--list` — `run` refuses `--lib-parts` > 1.
#[test]
fn the_slow_lib_runs_unfiltered() {
    let first = &invocations("slow", 1, 4, 1)[0];
    assert_eq!(
        first,
        &["--lib", "--bins"],
        "shard 1 of slow must open with the unfiltered unit tests"
    );
    let out = refused_run(
        &["slow", "1", "4", "--lib-parts", "2"],
        &["--workspace", "--exclude", "docs-lint"],
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() && err.contains("run with --lib-parts 1"),
        "`run slow --lib-parts 2` was not refused by name: {err}"
    );
}

/// T3. `tools/slow-nn-tests.txt` is the set of test files with a POSITIVE
/// `cfg(feature = "nn")` — a file-level `#![cfg]` or a `#[cfg]` on a test — and not
/// the `cfg_attr(any(not(slow-tests), nn), ignore)` gates, which only take tests
/// away under `nn`. This is the per-PR half; `check-nn` is the nightly half that
/// reads the compiler instead of the text.
#[test]
fn slow_nn_list_is_the_nn_enabled_files() {
    fn positive_nn_cfg(src: &str) -> bool {
        src.lines().map(str::trim).any(|l| {
            (l.starts_with("#[cfg(") || l.starts_with("#![cfg("))
                && l.contains("feature = \"nn\"")
                && !l.contains("not(feature = \"nn\")")
        })
    }
    fn sources(path: &Path, out: &mut String) {
        if path.is_file() {
            out.push_str(&std::fs::read_to_string(path).unwrap());
        } else if let Ok(entries) = std::fs::read_dir(path) {
            for e in entries.flatten() {
                sources(&e.path(), out);
            }
        }
    }
    let root = repo_root();
    let mut dirs = vec![root.join("tests")];
    for member in std::fs::read_dir(root.join("crates")).unwrap().flatten() {
        let dir = member.file_name().to_string_lossy().into_owned();
        if !SLOW_EXCLUDED_MEMBERS.contains(&dir.as_str()) {
            dirs.push(member.path().join("tests"));
        }
    }
    let targets: BTreeSet<String> = slow_test_targets().into_iter().collect();
    let mut found = BTreeSet::new();
    for dir in dirs {
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = e.path();
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            if !targets.contains(&stem) {
                continue; // `tests/common/` and fixture directories are not targets
            }
            let mut src = String::new();
            sources(&path, &mut src);
            if positive_nn_cfg(&src) {
                found.insert(stem);
            }
        }
    }
    assert!(
        found.len() >= 5,
        "found only {found:?} — is the cfg scan still matching the spelling in use?"
    );
    let listed: BTreeSet<String> = slow_nn_list().into_iter().collect();
    assert_eq!(
        listed, found,
        "tools/slow-nn-tests.txt must list exactly the test files with a positive \
         `cfg(feature = \"nn\")`: the nn job runs only those (plus the lib), so a file \
         missing from it runs its nn tests nowhere but the weekly full-nn leg"
    );
}

/// A fake `cargo` for `run slow`: records each call, answers `--list` (two tests,
/// one ignored) and a test run (one ok or FAILED, one ignored) for every binary the
/// arguments select, and fails the FIRST test run (not a `--list`).
fn fake_test_cargo(dir: &Path) -> PathBuf {
    let log = dir.join("calls.txt");
    let once = dir.join("failed_once");
    let fake = dir.join("cargo");
    let script = r#"#!/bin/sh
echo "$*" >> 'LOG'
list=0; ign=0; doc=0; lib=0; bins=""; prev=""
for a in "$@"; do
  case "$a" in --list) list=1;; --ignored) ign=1;; --doc) doc=1;; --lib) lib=1;; esac
  [ "$prev" = "--test" ] && bins="$bins $a"
  prev="$a"
done
out=ok; code=0
if [ $list = 0 ] && [ ! -e 'ONCE' ]; then touch 'ONCE'; out=FAILED; code=3; fi
emit() {
  echo "$1" >&2
  if [ $list = 1 ]; then
    [ $ign = 0 ] && echo "fast_one: test"
    echo "slow_one: test"
    echo ""
    echo "2 tests, 0 benchmarks"
  else
    echo "running 2 tests"
    echo "test fast_one ... $out"
    echo "test slow_one ... ignored, slow: opt in"
    if [ $out = ok ]; then p=1; f=0; else p=0; f=1; fi
    echo ""
    echo "test result: ok. $p passed; $f failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s"
  fi
}
[ $lib = 1 ] && emit "     Running unittests src/lib.rs (target/ci-test/deps/ferx_core-0123456789abcdef)"
[ $doc = 1 ] && emit "   Doc-tests ferx_core"
for b in $bins; do emit "     Running tests/$b.rs (target/ci-test/deps/$b-0123456789abcdef)"; done
exit $code
"#
    .replace("LOG", &log.display().to_string())
    .replace("ONCE", &once.display().to_string());
    std::fs::write(&fake, script).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    fake
}

fn manifest_rows(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| l.split('\t').map(str::to_owned).collect())
        .collect()
}

/// T4. `run slow` lists, then runs, every planned invocation; a failing one stops
/// neither the later ones nor the manifest, and fails the shard. The manifest names
/// every binary the shard ran, with its `--list` and its `test result:` counts.
#[test]
fn run_writes_a_manifest_and_fails_if_any_did() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ci_test_shards_slow_run");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fake = fake_test_cargo(&dir);
    let man_dir = dir.join("manifest");
    let out = Command::new("python3")
        .arg(repo_root().join("tools/ci-test-shard.py"))
        .args(["run", "slow", "1", "4", "--lib-parts", "1", "--manifest"])
        .arg(&man_dir)
        .args([
            "--",
            "--workspace",
            "--exclude",
            "docs-lint",
            "--profile",
            "ci-test",
            "--no-fail-fast",
        ])
        .env("CI_TEST_SHARD_CARGO", &fake)
        .current_dir(repo_root())
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "a failing invocation left the shard green: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let planned = invocations("slow", 1, 4, 1);
    let calls: Vec<String> = std::fs::read_to_string(dir.join("calls.txt"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        calls.len(),
        3 * planned.len(),
        "expected `--list`, `--list --ignored` and a run per invocation: {calls:#?}"
    );
    for call in &calls {
        assert!(
            call.starts_with(
                "test --workspace --exclude docs-lint --profile ci-test --no-fail-fast "
            ),
            "`{call}` is not a `cargo test` over the build args"
        );
    }
    let rows = manifest_rows(&man_dir.join("manifest.tsv"));
    let of = |t: &str| -> BTreeSet<String> {
        rows.iter()
            .filter(|r| r[0] == t)
            .map(|r| r[1].clone())
            .collect()
    };
    let mut want: BTreeSet<String> = ["ferx_core[unit]", "doctest:ferx_core"]
        .map(str::to_owned)
        .into();
    for inv in &planned {
        for w in inv.windows(2) {
            if w[0] == "--test" {
                want.insert(w[1].replace('-', "_"));
            }
        }
    }
    assert_eq!(
        of("B"),
        want,
        "the manifest's listed binaries are not the shard's"
    );
    assert_eq!(
        of("S"),
        want,
        "the manifest's `test result:` binaries are not the shard's"
    );
    let count = |t: &str| rows.iter().filter(|r| r[0] == t).count();
    assert_eq!(count("L"), 2 * want.len(), "two `--list` tests per binary");
    assert_eq!(
        count("I"),
        want.len(),
        "one `--list --ignored` test per binary"
    );
    assert!(
        rows.iter().any(|r| r[0] == "R"
            && r[1] == "ferx_core[unit]"
            && r[2] == "fast_one"
            && r[3] == "FAILED"),
        "the failing test is not in the manifest"
    );
    let meta = |k: &str| {
        rows.iter()
            .find(|r| r[0] == "M" && r[1] == k)
            .map(|r| r[2].clone())
    };
    assert_eq!(meta("status").as_deref(), Some("failed"));
    assert_eq!(meta("done").as_deref(), Some("1"));
    assert!(
        rows.iter().filter(|r| r[0] == "E").count() > want.len(),
        "the manifest must carry the suite's whole expected set, not just this shard's"
    );
}

/// `run slow` refuses build args whose `--exclude` disagrees with the suite: the
/// planner would schedule a target cargo does not build, or skip one it does.
#[test]
fn run_slow_refuses_a_mismatched_exclude() {
    for build in [
        vec!["--workspace"],
        vec![
            "--workspace",
            "--exclude",
            "docs-lint",
            "--exclude",
            "ferx-cli",
        ],
        vec!["--exclude", "docs-lint"],
    ] {
        let out = refused_run(&["slow", "1", "4", "--lib-parts", "1"], &build);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success() && err.starts_with("ci-test-shard: slow: "),
            "{build:?} was not refused by name: {err}"
        );
    }
}

/// One binary of a fixture manifest.
#[derive(Clone)]
struct Bin {
    name: &'static str,
    listed: usize,
    ignored: usize,
    passed: usize,
    failed: usize,
    filtered: usize,
}

impl Bin {
    /// A binary that ran every test it lists.
    fn ok(name: &'static str, listed: usize, ignored: usize) -> Self {
        Bin {
            name,
            listed,
            ignored,
            passed: listed - ignored,
            failed: 0,
            filtered: 0,
        }
    }
}

const EXPECTED: &[&str] = &["a", "b", "u[unit]", "z"];

fn write_manifest(dir: &Path, k: usize, n: usize, bins: &[Bin], done: bool) {
    let d = dir.join(format!("slow-manifest-core-{k}"));
    std::fs::create_dir_all(&d).unwrap();
    let mut s = format!("M\tsuite\tslow\nM\tk\t{k}\nM\tn\t{n}\n");
    for e in EXPECTED {
        s += &format!("E\t{e}\n");
    }
    for b in bins {
        s += &format!("B\t{}\n", b.name);
        for t in 0..b.listed {
            s += &format!("L\t{}\tt{t}\n", b.name);
        }
        for t in 0..b.ignored {
            s += &format!("I\t{}\tt{t}\n", b.name);
        }
        for t in 0..b.failed {
            s += &format!("R\t{}\tt{t}\tFAILED\n", b.name);
        }
        s += &format!(
            "S\t{}\t{}\t{}\t{}\t{}\t1.0\n",
            b.name, b.passed, b.failed, b.ignored, b.filtered
        );
    }
    s += "M\telapsed\t60\n";
    if done {
        let status = if bins.iter().any(|b| b.failed > 0) {
            "failed"
        } else {
            "ok"
        };
        s += &format!("M\tstatus\t{status}\nM\tdone\t1\n");
    }
    std::fs::write(d.join("manifest.tsv"), s).unwrap();
}

/// The valid two-shard fixture: `z` is a binary with no tests at all (the `src/bin`
/// bundles, an `nn_*` file without `nn`), which is legitimate and must stay green.
fn valid_shards() -> Vec<Vec<Bin>> {
    vec![
        vec![Bin::ok("a", 3, 1), Bin::ok("u[unit]", 2, 0)],
        vec![Bin::ok("b", 1, 0), Bin::ok("z", 0, 0)],
    ]
}

fn run_check(
    case: &str,
    shards: &[Vec<Bin>],
    present: &[usize],
    done: &[bool],
    result: &str,
) -> (bool, String) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("ci_test_shards_check")
        .join(case);
    let _ = std::fs::remove_dir_all(&dir);
    for (i, bins) in shards.iter().enumerate() {
        if present.contains(&(i + 1)) {
            write_manifest(&dir, i + 1, shards.len(), bins, done[i]);
        }
    }
    std::fs::create_dir_all(&dir).unwrap();
    let out = shard_tool(
        &[
            "check",
            "slow",
            &shards.len().to_string(),
            &dir.display().to_string(),
            "--result",
            result,
        ]
        .map(|s| s.to_string()),
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    (out.status.success(), text)
}

/// T5–T10. `check` over fixture manifests: the valid set is green, and each
/// corruption is red with its OWN `::error::` naming the condition — asserted on
/// the message, not only on the exit code, so a corruption caught by the wrong
/// rule (or by a crash) does not pass.
#[test]
fn check_names_each_way_a_sharded_run_can_be_incomplete() {
    let all = [1, 2];
    let both = [true, true];
    let (ok, text) = run_check("valid", &valid_shards(), &all, &both, "success");
    assert!(
        ok && !text.contains("::error::"),
        "the valid fixture is red:\n{text}"
    );

    let mut cases: Vec<(&str, Vec<Vec<Bin>>, Vec<usize>, [bool; 2], &str, &str)> = Vec::new();
    // (5) a target in no manifest
    let mut s = valid_shards();
    s[1].retain(|b| b.name != "b");
    cases.push((
        "t5",
        s,
        all.to_vec(),
        both,
        "success",
        "::error::b: no shard ran this test target",
    ));
    // (6) a target in two manifests
    let mut s = valid_shards();
    s[0].push(Bin::ok("b", 1, 0));
    cases.push((
        "t6",
        s,
        all.to_vec(),
        both,
        "success",
        "::error::b: ran on several shards [1, 2]",
    ));
    // (7) a binary whose --list is non-empty but which ran nothing
    let mut s = valid_shards();
    s[0][0] = Bin {
        passed: 0,
        ignored: 0,
        ..Bin::ok("a", 3, 1)
    };
    cases.push((
        "t7",
        s,
        all.to_vec(),
        both,
        "success",
        "::error::a: its --list names 3 tests but none ran (shard 1)",
    ));
    // (8) ran + ignored != --list (a filter that matched part of it)
    let mut s = valid_shards();
    s[0][0] = Bin {
        passed: 1,
        filtered: 1,
        ..Bin::ok("a", 3, 1)
    };
    cases.push((
        "t8",
        s,
        all.to_vec(),
        both,
        "success",
        "::error::a: ran 2 of the 3 tests its --list names, 1 filtered out (shard 1)",
    ));
    // (9) only 1 of 2 manifests
    cases.push((
        "t9",
        valid_shards(),
        vec![1],
        both,
        "success",
        "::error::no manifest from shard(s) [2] of 2",
    ));
    // (10) the shard jobs ended cancelled (a timeout-minutes kill) with complete manifests
    cases.push((
        "t10",
        valid_shards(),
        all.to_vec(),
        both,
        "cancelled",
        "::error::the slow shard jobs ended 'cancelled'",
    ));
    // a shard killed while writing
    cases.push((
        "unfinished",
        valid_shards(),
        all.to_vec(),
        [true, false],
        "success",
        "::error::shard 2/2 never finished",
    ));
    // a failing test
    let mut s = valid_shards();
    s[1][0] = Bin {
        passed: 0,
        failed: 1,
        ..Bin::ok("b", 1, 0)
    };
    cases.push((
        "failed",
        s,
        all.to_vec(),
        both,
        "success",
        "::error::b: 1 test(s) FAILED (shard 2): t0",
    ));

    let markers: Vec<&str> = cases.iter().map(|c| c.5).collect();
    for (case, shards, present, done, result, marker) in &cases {
        let (ok, text) = run_check(case, shards, present, done, result);
        assert!(!ok, "{case}: check passed a corrupted fixture:\n{text}");
        assert!(text.contains(marker), "{case}: no `{marker}` in:\n{text}");
        assert!(
            !text.contains("Traceback"),
            "{case}: check crashed:\n{text}"
        );
        // Distinct: no OTHER case's condition fires on this one, except (9)'s absent
        // shard, whose own targets necessarily also read as run nowhere.
        for other in &markers {
            if other != marker && !(*case == "t9" && other.contains("no shard ran")) {
                assert!(
                    !text.contains(other),
                    "{case} also reports `{other}`:\n{text}"
                );
            }
        }
    }
}

/// T14's offline half: `check-nn` on fixture manifests. The binaries the nn leg must
/// run are those whose nn `--list` holds a runnable test the core one does not;
/// a binary that only LOSES tests under nn (the #2138/#2144 ignore gates) needs
/// nothing. The nightly run is the online half: it reads the real compiler output.
#[test]
fn check_nn_requires_exactly_the_binaries_nn_adds_a_test_to() {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ci_test_shards_check_nn");
    // (binary, core listed, core ignored, nn listed, nn ignored)
    let table: &[(&str, &[&str], &[&str], &[&str], &[&str])] = &[
        ("ferx_core[unit]", &["x"], &[], &["x", "x_nn"], &[]),
        ("gains", &["y"], &[], &["y", "y_nn"], &[]),
        ("loses", &["k"], &[], &["k"], &["k"]),
        ("same", &["s"], &["s"], &["s"], &["s"]),
    ];
    let run = |case: &str, list: &[&str], nn_runs: &[&str]| -> (bool, String) {
        let dir = base.join(case);
        let _ = std::fs::remove_dir_all(&dir);
        for d in ["core/m1", "nn/m"] {
            std::fs::create_dir_all(dir.join(d)).unwrap();
        }
        let (mut core, mut nn) = (String::new(), String::from("M\tsuite\tslow-nn\n"));
        for (b, cl, ci, nl, ni) in table {
            core += &format!("B\t{b}\n");
            core += &cl
                .iter()
                .map(|t| format!("L\t{b}\t{t}\n"))
                .collect::<String>();
            core += &ci
                .iter()
                .map(|t| format!("I\t{b}\t{t}\n"))
                .collect::<String>();
            nn += &format!("XB\t{b}\n");
            nn += &nl
                .iter()
                .map(|t| format!("XL\t{b}\t{t}\n"))
                .collect::<String>();
            nn += &ni
                .iter()
                .map(|t| format!("XI\t{b}\t{t}\n"))
                .collect::<String>();
        }
        for b in nn_runs {
            nn += &format!("S\t{b}\t1\t0\t0\t0\t1.0\n");
        }
        std::fs::write(dir.join("core/m1/manifest.tsv"), core).unwrap();
        std::fs::write(dir.join("nn/m/manifest.tsv"), nn).unwrap();
        std::fs::write(dir.join("list.txt"), list.join("\n")).unwrap();
        let out = shard_tool(
            &[
                "check-nn",
                &dir.join("core").display().to_string(),
                &dir.join("nn").display().to_string(),
                "--list",
                &dir.join("list.txt").display().to_string(),
            ]
            .map(|s| s.to_string()),
        );
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    };
    let (ok, text) = run("valid", &["gains"], &["ferx_core[unit]", "gains"]);
    assert!(ok, "the valid fixture is red:\n{text}");
    assert!(
        !text.contains("loses"),
        "a binary that only loses tests was required:\n{text}"
    );

    let (ok, text) = run("missing", &[], &["ferx_core[unit]"]);
    assert!(
        !ok && text
            .contains("::error::gains: gains 1 test(s) under nn but the nn leg does not run it"),
        "a binary gaining a test under nn, absent from the list, passed:\n{text}"
    );
    let (ok, text) = run(
        "extra",
        &["gains", "same"],
        &["ferx_core[unit]", "gains", "same"],
    );
    assert!(
        !ok && text.contains("::error::same: listed in tools/slow-nn-tests.txt but gains no test"),
        "a listed binary that gains nothing under nn passed:\n{text}"
    );
}

/// T11. Each summary job reports even when a shard fails or is cancelled (a job
/// killed by `timeout-minutes` is `cancelled`), and hands the shards' aggregate
/// result to its check, so a non-success shard is red there. One gate, not two:
/// the result reaches the verdict only through `--result`, so deleting it is red
/// here (AGENTS.md: two redundant gates cover for each other).
#[test]
fn each_slow_summary_goes_red_with_any_shard() {
    let shards = job_body_in(SLOW_YML, "shards");
    let n = shards
        .lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("shard: ["))
        .expect("`shards` has no `shard: [...]` matrix")
        .trim_end_matches(']')
        .split(',')
        .count();
    assert!(
        shards.contains("name: Slow regression tests (${{ matrix.leg }} ${{ matrix.shard }}/"),
        "the shard jobs must carry a `k/N` suffix, so only `summary` reports the leg's name"
    );
    for (job, needs, check) in [
        (
            "summary",
            "needs: shards\n",
            format!("check slow {n} manifests --result ${{{{ needs.shards.result }}}}"),
        ),
        (
            "nn-summary",
            "needs: [shards, nn]\n",
            "check slow-nn 1 nn-manifest --result ${{ needs.nn.result }}".to_owned(),
        ),
    ] {
        let body = job_body_in(SLOW_YML, job);
        assert!(
            body.contains(needs),
            "`{job}` does not wait for its shards ({needs:?})"
        );
        let cond = body
            .lines()
            .find(|l| l.starts_with("    if:"))
            .unwrap_or_else(|| panic!("`{job}` has no job-level `if:`"));
        assert!(
            cond.contains("always()") && !cond.contains("cancelled()"),
            "`{job}`: `{cond}`: without `always()` a failed or cancelled shard SKIPS the \
             verdict, and a timed-out night reports nothing"
        );
        assert!(
            body.contains(&check),
            "`{job}` no longer runs `{check}` — the shard count must match the matrix \
             ({n}) and the shards' result must reach the check"
        );
    }
    let nn = job_body_in(SLOW_YML, "nn-summary");
    let at = nn
        .find("check-nn core-manifests nn-manifest")
        .expect("`nn-summary` runs no check-nn");
    let step = &nn[nn[..at].rfind("\n      - ").unwrap()..at];
    assert!(
        step.contains("if: success() || failure()"),
        "check-nn must run even when the nn job's own check is red:\n{step}"
    );
}

/// T12. The nn job runs the lib unit tests and exactly the listed binaries, with
/// `nn` on, and lists every binary for `check-nn`.
#[test]
fn the_nn_leg_runs_exactly_the_listed_binaries_and_the_lib() {
    let mut want = vec![vec!["--lib".to_owned()]];
    let mut listed = slow_nn_list();
    listed.sort();
    want.push(
        listed
            .iter()
            .flat_map(|t| ["--test".to_owned(), t.clone()])
            .collect(),
    );
    assert_eq!(
        invocations("slow-nn", 1, 1, 1),
        want,
        "the nn job's invocations are not --lib + the list"
    );
    let body = job_body_in(SLOW_YML, "nn");
    let run = body
        .lines()
        .map(str::trim)
        .find(|l| l.contains("ci-test-shard.py run"))
        .expect("the nn job does not run the planner");
    assert!(
        run.starts_with("python3 tools/ci-test-shard.py run slow-nn 1 1 --lib-parts 1 --list-all "),
        "the nn job must run `run slow-nn 1 1 --list-all`: `{run}`"
    );
    assert!(
        body.contains("ferx-core/nn"),
        "the nn job does not turn `nn` on"
    );
}

/// The canakinumab fetch keys on the planner: the anchor runs on exactly one shard
/// at every N, and an unknown target is an error, not a silent `false`.
#[test]
fn has_target_finds_the_one_shard_running_a_target() {
    for n in 1..=6 {
        let hits: Vec<usize> = (1..=n)
            .filter(|k| {
                let out = shard_tool(
                    &[
                        "has-target",
                        "slow",
                        &k.to_string(),
                        &n.to_string(),
                        "--lib-parts",
                        "1",
                        "mbma_canakinumab_case_study",
                    ]
                    .map(|s| s.to_string()),
                );
                assert!(out.status.success());
                match String::from_utf8_lossy(&out.stdout).trim() {
                    "present=true" => true,
                    "present=false" => false,
                    other => panic!("has-target printed `{other}`"),
                }
            })
            .collect();
        assert_eq!(hits.len(), 1, "N={n}: the anchor is on shards {hits:?}");
    }
    let out = shard_tool(&["has-target", "slow", "1", "4", "no_such_target"].map(str::to_owned));
    assert!(
        !out.status.success(),
        "an unknown target read as a plain `false`"
    );
}

/// T13. `weights` reads a log fetched with `gh run view --log`, which writes colour
/// codes in caret form (`^[[32m`) and prefixes every line with `job<TAB>step<TAB>`
/// before the timestamp. The fixture is 38 lines of the `3b70b24b` core log. Each
/// half of the fix dies here: strip only `\x1b[` and the `Running` lines never
/// match (no `<unit>`); `re.match` the stamp and no unit test is timed (no `lib:`).
#[test]
fn weights_parse_a_gh_log() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ci_test_shards_weights");
    std::fs::create_dir_all(&dir).unwrap();
    let out_path = dir.join("weights.txt");
    let out = shard_tool(
        &[
            "weights",
            "slow",
            "tests/fixtures/ci_test_shard/gh_run_view_log_excerpt.txt",
            "--output",
            &out_path.display().to_string(),
        ]
        .map(|s| s.to_string()),
    );
    assert!(
        out.status.success(),
        "weights failed on a gh log: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&out_path).unwrap();
    let body: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
    assert!(
        body.contains(&"693 <unit>"),
        "no unit-bundle weight:\n{text}"
    );
    assert!(
        body.contains(&"32 agq"),
        "no integration-binary weight:\n{text}"
    );
    assert!(
        body.iter().any(|l| l.ends_with(" lib:estimation::agq")),
        "no per-module unit-test weight — the timestamps were not read:\n{text}"
    );
}
