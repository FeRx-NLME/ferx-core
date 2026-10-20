//! Guard: the sharded `Tests + coverage (core)` job runs every test binary exactly
//! once (#2167).
//!
//! The job used to be one `cargo llvm-cov --workspace --tests`. It is now a matrix
//! whose shard `k` runs `tools/ci-test-shard.py args k N`, so `--tests` holds only
//! if the union of those argument lists is every test target. A planner bug that
//! dropped a binary would leave every shard green — the dropped tests simply never
//! run, and their lines read as uncovered against the patch gate with nothing
//! pointing at the cause. A binary scheduled twice wastes a shard's time.
//!
//! The oracle is the filesystem, not `cargo metadata`, because `cargo metadata` is
//! what the script itself reads: agreeing with it would be agreeing with itself.
//! Cargo auto-discovers `<member>/tests/*.rs` and `<member>/tests/<dir>/main.rs`
//! (no member declares `[[test]]` or `autotests = false`), so that is the set.
//!
//! Not feature-gated: it must run in the base `--features ci` job it guards.

use std::collections::BTreeMap;
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

fn shard_args(k: usize, n: usize) -> Vec<String> {
    let out = Command::new("python3")
        .arg(repo_root().join("tools/ci-test-shard.py"))
        .args(["args", &k.to_string(), &n.to_string()])
        .current_dir(repo_root())
        .output()
        .expect("python3 runs tools/ci-test-shard.py");
    assert!(
        out.status.success(),
        "ci-test-shard.py args {k} {n} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// Every shard count, not just the 4 `ci.yml` uses today: the matrix size is a
/// one-line edit, and the planner must be complete for whatever it becomes.
#[test]
fn the_shards_run_every_test_binary_exactly_once() {
    let all = every_test_target();
    assert!(
        all.len() > 100,
        "found only {} test targets — is the discovery looking in the right place?",
        all.len()
    );
    for n in 1..=6 {
        let mut seen: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let (mut lib_shards, mut bins_shards) = (Vec::new(), Vec::new());
        for k in 1..=n {
            let args = shard_args(k, n);
            let mut it = args.iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--test" => seen.entry(it.next().unwrap().clone()).or_default().push(k),
                    "--lib" => lib_shards.push(k),
                    // Tracked separately: `ferx-cli`'s unit tests live in its bin target,
                    // so dropping `--bins` alone silently drops them.
                    "--bins" => bins_shards.push(k),
                    other => panic!("shard {k}/{n} emitted an unexpected argument `{other}`"),
                }
            }
        }
        assert_eq!(
            lib_shards,
            vec![1],
            "N={n}: the `--lib` unit tests must run once, on shard 1"
        );
        assert_eq!(
            bins_shards,
            vec![1],
            "N={n}: the `--bins` unit tests must run once, on shard 1"
        );
        let missing: Vec<_> = all.iter().filter(|t| !seen.contains_key(*t)).collect();
        assert!(
            missing.is_empty(),
            "N={n}: test binaries no shard runs: {missing:?}"
        );
        let twice: Vec<_> = seen.iter().filter(|(_, ks)| ks.len() > 1).collect();
        assert!(
            twice.is_empty(),
            "N={n}: test binaries run on several shards: {twice:?}"
        );
        let unknown: Vec<_> = seen.keys().filter(|t| !all.contains(t)).collect();
        assert!(
            unknown.is_empty(),
            "N={n}: shards name non-existent tests: {unknown:?}"
        );
    }
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

/// `Tests + coverage (core)` is a REQUIRED check in the `main-protection` ruleset,
/// and since the split it is the upload job, not the shards. GitHub counts a skipped
/// job as passing a required check, and a bare `needs:` skips the job when a shard
/// fails — so without both pieces below a PR with red tests shows every required
/// check green. Each assertion dies on its own deletion.
#[test]
fn the_required_core_check_goes_red_with_any_shard() {
    let body = job_body("coverage-pr-upload");
    assert!(
        body.contains("name: Tests + coverage (core)\n"),
        "the upload job no longer carries the required `Tests + coverage (core)` name"
    );
    assert!(
        body.contains("needs: coverage-pr"),
        "the upload job does not wait for the shards"
    );
    let cond = body
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("if:"))
        .expect("the upload job has no job-level `if:`");
    // `always()`, not `!cancelled()`: the latter skips this job when a shard is
    // cancelled, and a cancel mid-run would then read green.
    assert!(
        cond.contains("always()"),
        "`{cond}`: without `always()` a failed or cancelled shard SKIPS this job, and a \
         skipped required check counts as passing"
    );
    let guard = "if: needs.coverage-pr.result != 'success'";
    let at = body.find(guard).unwrap_or_else(|| {
        panic!(
            "the upload job no longer fails on a non-successful shard — with `always()` it \
             would then upload partial coverage and report green"
        )
    });
    // The guard step must actually fail: its body runs up to the next step.
    let step = &body[at..];
    let step = &step[..step[guard.len()..]
        .find("\n      - ")
        .map_or(step.len(), |i| i + guard.len())];
    assert!(
        step.lines().any(|l| l.trim() == "exit 1"),
        "the shard-result step no longer exits non-zero, so it reports and passes:\n{step}"
    );
}

/// The shard's planner must run where `bash -e` sees its exit status. Inlined as
/// `cargo llvm-cov … $(python3 tools/ci-test-shard.py …)`, a planner error is swallowed
/// (a simple command's status is the command's, not its substitutions') and cargo runs
/// with no target arguments — its default targets, green, the wrong suite.
#[test]
fn a_planner_failure_fails_the_shard() {
    let body = job_body("coverage-pr");
    let lines: Vec<&str> = body.lines().map(str::trim).collect();
    let cargo = lines
        .iter()
        .find(|l| l.contains("cargo llvm-cov"))
        .expect("coverage-pr runs no `cargo llvm-cov`");
    assert!(
        !cargo.contains("$("),
        "`{cargo}` substitutes a command inline; its failure would be swallowed"
    );
    assert!(
        cargo.contains("$targets"),
        "`{cargo}` no longer takes the planner's targets"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("targets=$(python3 tools/ci-test-shard.py args ")),
        "coverage-pr no longer assigns `targets` from the planner on a line of its own"
    );
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
        .filter_map(|job| {
            let body = job_body(&job);
            let at = body.find("codecov/codecov-action@")?;
            // The evaluator reads job-level conditions only, so an upload step must
            // carry none of its own.
            let step_start = body[..at].rfind("\n      - ").unwrap_or(0);
            let step_end = body[at..].find("\n      - ").map_or(body.len(), |i| at + i);
            assert!(
                !body[step_start..step_end].contains("\n        if:"),
                "job `{job}`'s Codecov step has its own `if:`, which this table ignores"
            );
            let cond = body
                .lines()
                .find(|l| l.starts_with("    if:"))
                .map(|l| l.trim().to_owned());
            Some((job, cond))
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
/// default targets and silently test something else.
#[test]
fn an_out_of_range_shard_is_refused() {
    for (k, n) in [(0, 4), (5, 4)] {
        let out = Command::new("python3")
            .arg(repo_root().join("tools/ci-test-shard.py"))
            .args(["args", &k.to_string(), &n.to_string()])
            .output()
            .unwrap();
        assert!(!out.status.success(), "shard {k}/{n} was accepted");
    }
}
