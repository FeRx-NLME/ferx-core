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
    for k in 1..=n {
        for inv in invocations(suite, k, n, p) {
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
    let yml = std::fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).unwrap();
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
