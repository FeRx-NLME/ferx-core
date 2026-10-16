//! Guard: `.github/workflows/ci.yml` works under the `main` merge queue (#2042).
//!
//! The merge queue builds a PR on top of the current `main` and the PRs ahead of it and
//! merges only when every *required* check reports success on that commit. Three ways a
//! workflow edit breaks that without anyone noticing on a PR:
//!
//! - the `merge_group` trigger is dropped, so no required check ever reports in the
//!   queue and every PR waits there until it times out;
//! - a test job loses its `if:` and runs the whole suite in the queue, which is the
//!   slowdown the queue was set up to avoid;
//! - the job that posts `codecov/patch` on the merge-group commit goes, and the queue
//!   blocks on a status Codecov never sends there (no coverage upload runs in the queue).
//!
//! Plain text matching on the YAML, like `ci_workflow_endpoint_coverage.rs`: the file is
//! ours and small, and a YAML parser would be a new dev-dependency for three facts.

use std::path::PathBuf;

fn ci_yml() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yml");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The lines of one top-level job, from its `  <key>:` line to the next job key.
fn job_body(yml: &str, key: &str) -> String {
    let header = format!("  {key}:");
    let mut out = Vec::new();
    let mut inside = false;
    for line in yml.lines() {
        let is_job_key = line.starts_with("  ")
            && !line.starts_with("   ")
            && line.trim_end().ends_with(':')
            && !line.trim_start().starts_with('#');
        if is_job_key {
            inside = line.trim_end() == header;
            if inside {
                out.push(line);
            }
            continue;
        }
        if inside {
            out.push(line);
        }
    }
    assert!(
        !out.is_empty(),
        "ci.yml has no job `{key}` — did the job key change?"
    );
    out.join("\n")
}

/// `if:` lines of a job, comments excluded.
fn job_conditions(body: &str) -> Vec<String> {
    body.lines()
        .map(|l| l.trim_start())
        .filter(|l| l.starts_with("if:"))
        .map(str::to_string)
        .collect()
}

#[test]
fn ci_runs_in_the_merge_queue() {
    let yml = ci_yml();
    let on: Vec<&str> = yml
        .lines()
        .skip_while(|l| l.trim_end() != "on:")
        .take_while(|l| !l.starts_with("env:") && !l.starts_with("jobs:"))
        .collect();
    assert!(
        on.iter().any(|l| l.trim() == "merge_group:"),
        "ci.yml no longer triggers on `merge_group`, so no required check reports in the \
         merge queue and every queued PR waits until it times out (#2042). `on:` is:\n{}",
        on.join("\n")
    );
}

#[test]
fn the_test_jobs_are_skipped_in_the_merge_queue() {
    let yml = ci_yml();
    for job in ["endpoints", "release-semantics", "coverage-pr"] {
        let conds = job_conditions(&job_body(&yml, job));
        // `coverage-pr` names the events it runs on; the other two name the one they skip.
        let skips_queue = conds.iter().any(|c| {
            c.contains("github.event_name != 'merge_group'")
                || (c.contains("github.event_name ==") && !c.contains("merge_group"))
        });
        assert!(
            skips_queue,
            "job `{job}` runs in the merge queue: its `if:` is {conds:?}. The test and \
             coverage jobs already ran on the PR; in the queue they only make every merge \
             wait for the whole suite (#2042)."
        );
    }
}

#[test]
fn the_fast_gates_run_in_the_merge_queue() {
    let yml = ci_yml();
    // The required checks that catch a compile-level conflict between two PRs. A
    // condition on any of them would let the queue merge a combination that does not
    // build, which is what the queue exists to stop.
    for job in [
        "check",
        "clippy",
        "rustdoc",
        "fmt",
        "public-api",
        "docs-lint",
    ] {
        let conds = job_conditions(&job_body(&yml, job));
        assert!(
            conds.is_empty(),
            "fast gate `{job}` has a condition {conds:?}; it must run in the merge queue \
             (#2042)"
        );
    }
}

#[test]
fn codecov_patch_is_reported_on_the_merge_group_commit() {
    let yml = ci_yml();
    let body = job_body(&yml, "merge-queue-coverage-status");
    let conds = job_conditions(&body);
    assert!(
        conds
            .iter()
            .any(|c| c.contains("github.event_name == 'merge_group'")),
        "`merge-queue-coverage-status` must run only in the merge queue; its `if:` is \
         {conds:?}"
    );
    assert!(
        body.contains("context=codecov/patch") && body.contains("state=success"),
        "`merge-queue-coverage-status` no longer posts a successful `codecov/patch` status. \
         It is a required check that Codecov only sends after a coverage upload, and no \
         upload runs in the queue, so without this the queue blocks (#2042)."
    );
    assert!(
        body.contains("statuses: write"),
        "`merge-queue-coverage-status` needs `permissions: statuses: write` to post the \
         status"
    );
}
