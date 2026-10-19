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
        let mut unit_shards = Vec::new();
        for k in 1..=n {
            let args = shard_args(k, n);
            let mut it = args.iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--test" => seen.entry(it.next().unwrap().clone()).or_default().push(k),
                    "--lib" => unit_shards.push(k),
                    "--bins" => {}
                    other => panic!("shard {k}/{n} emitted an unexpected argument `{other}`"),
                }
            }
        }
        assert_eq!(
            unit_shards,
            vec![1],
            "N={n}: the `--lib --bins` unit tests must run once, on shard 1"
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
    assert!(
        cond.contains("!cancelled()") || cond.contains("always()"),
        "`{cond}`: without `!cancelled()` a failed shard SKIPS this job, and a skipped \
         required check counts as passing"
    );
    assert!(
        body.contains("if: needs.coverage-pr.result != 'success'"),
        "the upload job no longer fails on a non-successful shard — with `!cancelled()` \
         it would then upload partial coverage and report green"
    );
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
