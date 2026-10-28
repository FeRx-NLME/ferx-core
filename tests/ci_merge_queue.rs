//! Guard: `.github/workflows/ci.yml` works under the merge queue on `main` (#2042).
//!
//! `main` went red twice from two PRs that were each green and had no textual conflict
//! (#1768 × #1746, #1957 × #1925), because nothing built the combination before it
//! landed. The queue builds each PR on top of current `main` and the PRs ahead of it and
//! merges only if the `main-protection` ruleset's required checks pass there. That only
//! works if every required check *reports* on a `merge_group` event — a check that never
//! reports blocks the queue until it times out — and if what reports there is honest.
//!
//! So, evaluating each job's `if:` per event:
//!
//!  1. `ci.yml` triggers on `merge_group`.
//!  2. In the queue, the fast gates **run** (they are what catches a compile-level
//!     semantic conflict) and every required test/coverage job is **skipped** (a skipped
//!     job satisfies a required check; the tests already ran on the PR). A required job
//!     that *ran* there and failed on a skipped dependency — `endpoints-upload`'s
//!     `always()` did exactly that before this change — would block every PR.
//!  3. On a PR and on a push to `main` every required job still runs: skipping in the
//!     queue must not become skipping everywhere.
//!  4. `codecov/patch`, which Codecov posts only after an upload, is posted by one job
//!     that runs on `merge_group` **only** — on a PR it would forge the patch gate.
//!
//! The ruleset itself is repository settings, not a file, so [`REQUIRED_CHECKS`] mirrors
//! it by hand. Refresh it with
//!
//! ```text
//! gh api repos/FeRx-NLME/ferx-core/rulesets/16547557 \
//!   --jq '.rules[] | select(.type=="required_status_checks") | .parameters.required_status_checks[].context'
//! ```

use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// The `main-protection` ruleset's required status checks (see the module docs).
const REQUIRED_CHECKS: [&str; 10] = [
    "Check",
    "Clippy",
    "Docs lint",
    "Format",
    "Public API baseline",
    "Rustdoc",
    "Tests (release semantics)",
    "Tests + coverage (TTE/CTMM endpoints)",
    "Tests + coverage (core)",
    "codecov/patch",
];

/// Jobs that must run in the queue, by `name:`. The required fast gates plus
/// `Changelog`, which is not required but is a fast gate the issue puts in the queue.
const QUEUE_GATES: [&str; 7] = [
    "Check",
    "Clippy",
    "Docs lint",
    "Format",
    "Public API baseline",
    "Rustdoc",
    "Changelog",
];

/// The job that posts `codecov/patch` in the queue.
const PATCH_JOB: &str = "codecov-patch-merge-queue";

fn ci_yml() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_saphyr::from_str(&text).expect("ci.yml parses as YAML")
}

fn jobs(yml: &Value) -> &serde_json::Map<String, Value> {
    yml["jobs"]
        .as_object()
        .expect("ci.yml has a `jobs:` mapping")
}

// ---------------------------------------------------------------------------------------
// A deliberately tiny evaluator for job-level `if:` expressions.
//
// It knows `github.event_name`, string literals, `==`/`!=`, `!`/`&&`/`||`, parentheses
// and the status functions, and PANICS on anything else — so a job-level `if:` reading
// some other context is a loud test failure asking for a decision, never a silent guess.
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Str(String),
    Ident(String),
    LParen,
    RParen,
    Not,
    And,
    Or,
    Eq,
    Ne,
}

fn lex(src: &str) -> Vec<Tok> {
    let s = src.trim();
    let s = s
        .strip_prefix("${{")
        .and_then(|r| r.strip_suffix("}}"))
        .unwrap_or(s);
    let b: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let two: String = b[i..(i + 2).min(b.len())].iter().collect();
        match c {
            _ if c.is_whitespace() => i += 1,
            '(' => {
                out.push(Tok::LParen);
                i += 1
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1
            }
            '\'' => {
                let end = b[i + 1..]
                    .iter()
                    .position(|&x| x == '\'')
                    .unwrap_or_else(|| panic!("unterminated string in `{src}`"));
                out.push(Tok::Str(b[i + 1..i + 1 + end].iter().collect()));
                i += end + 2;
            }
            _ if two == "&&" => {
                out.push(Tok::And);
                i += 2
            }
            _ if two == "||" => {
                out.push(Tok::Or);
                i += 2
            }
            _ if two == "==" => {
                out.push(Tok::Eq);
                i += 2
            }
            _ if two == "!=" => {
                out.push(Tok::Ne);
                i += 2
            }
            '!' => {
                out.push(Tok::Not);
                i += 1
            }
            _ if c.is_ascii_alphabetic() => {
                let end = b[i..]
                    .iter()
                    .position(|&x| !(x.is_ascii_alphanumeric() || x == '_' || x == '.'))
                    .map_or(b.len(), |p| i + p);
                out.push(Tok::Ident(b[i..end].iter().collect()));
                i = end;
            }
            _ => panic!("unsupported character `{c}` in job-level `if: {src}`"),
        }
    }
    out
}

/// What a job's `if:` sees: the event, and whether every job it `needs:` succeeded.
struct Ctx<'a> {
    event: &'a str,
    needs_ok: bool,
}

#[derive(Debug, PartialEq)]
enum Val {
    Str(String),
    Bool(bool),
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    ctx: &'a Ctx<'a>,
    src: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Tok {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t.unwrap_or_else(|| panic!("unexpected end of `if: {}`", self.src))
    }
    fn bool(&self, v: Val) -> bool {
        match v {
            Val::Bool(b) => b,
            Val::Str(s) => panic!("string `{s}` used as a condition in `if: {}`", self.src),
        }
    }
    fn or(&mut self) -> Val {
        let mut l = self.and();
        while self.peek() == Some(&Tok::Or) {
            self.pos += 1;
            let r = self.and();
            l = Val::Bool(self.bool(l) | self.bool(r));
        }
        l
    }
    fn and(&mut self) -> Val {
        let mut l = self.cmp();
        while self.peek() == Some(&Tok::And) {
            self.pos += 1;
            let r = self.cmp();
            l = Val::Bool(self.bool(l) & self.bool(r));
        }
        l
    }
    fn cmp(&mut self) -> Val {
        let l = self.unary();
        match self.peek() {
            Some(Tok::Eq) | Some(Tok::Ne) => {
                let eq = self.next() == Tok::Eq;
                let r = self.unary();
                Val::Bool((l == r) == eq)
            }
            _ => l,
        }
    }
    fn unary(&mut self) -> Val {
        match self.next() {
            Tok::Not => {
                let v = self.unary();
                Val::Bool(!self.bool(v))
            }
            Tok::LParen => {
                let v = self.or();
                assert_eq!(
                    self.next(),
                    Tok::RParen,
                    "unbalanced `(` in `if: {}`",
                    self.src
                );
                v
            }
            Tok::Str(s) => Val::Str(s),
            Tok::Ident(id) => {
                let call = self.peek() == Some(&Tok::LParen);
                if call {
                    self.pos += 1;
                    assert_eq!(
                        self.next(),
                        Tok::RParen,
                        "status functions take no arguments"
                    );
                }
                match (id.as_str(), call) {
                    ("github.event_name", false) => Val::Str(self.ctx.event.to_string()),
                    ("always", true) => Val::Bool(true),
                    ("success", true) => Val::Bool(self.ctx.needs_ok),
                    // Nothing in this model fails or is cancelled.
                    ("failure", true) | ("cancelled", true) => Val::Bool(false),
                    _ => panic!(
                        "`{id}` in job-level `if: {}` is outside what this guard can \
                         evaluate — extend `tests/ci_merge_queue.rs` deliberately, deciding \
                         what the job does on a `merge_group` event",
                        self.src
                    ),
                }
            }
            t => panic!("unexpected `{t:?}` in `if: {}`", self.src),
        }
    }
}

/// Does a job with this `if:` run? Mirrors GitHub: no status function means an implicit
/// `success() &&`, i.e. a skipped dependency skips the job.
fn eval_if(expr: Option<&str>, ctx: &Ctx) -> bool {
    let Some(src) = expr else {
        return ctx.needs_ok;
    };
    let toks = lex(src);
    let has_status = toks.iter().any(|t| {
        matches!(t, Tok::Ident(id) if ["always", "success", "failure", "cancelled"].contains(&id.as_str()))
    });
    let mut p = Parser {
        toks,
        pos: 0,
        ctx,
        src,
    };
    let v = p.or();
    assert!(p.pos == p.toks.len(), "trailing tokens in `if: {src}`");
    let b = p.bool(v);
    if has_status {
        b
    } else {
        ctx.needs_ok && b
    }
}

fn needs(job: &Value) -> Vec<String> {
    match &job["needs"] {
        Value::Null => vec![],
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().map(|v| v.as_str().unwrap().to_string()).collect(),
        other => panic!("unexpected `needs:` {other}"),
    }
}

/// The ids of every job that runs on `event`, resolving `needs:` transitively.
fn running_jobs(yml: &Value, event: &str) -> BTreeSet<String> {
    let jobs = jobs(yml);
    let mut memo = std::collections::BTreeMap::<String, bool>::new();
    fn runs(
        id: &str,
        jobs: &serde_json::Map<String, Value>,
        event: &str,
        memo: &mut std::collections::BTreeMap<String, bool>,
    ) -> bool {
        if let Some(&r) = memo.get(id) {
            return r;
        }
        let job = &jobs[id];
        let needs_ok = needs(job).iter().all(|n| runs(n, jobs, event, memo));
        let r = eval_if(job["if"].as_str(), &Ctx { event, needs_ok });
        memo.insert(id.to_string(), r);
        r
    }
    jobs.keys()
        .filter(|id| runs(id, jobs, event, &mut memo))
        .cloned()
        .collect()
}

fn job_name(yml: &Value, id: &str) -> String {
    jobs(yml)[id]["name"].as_str().unwrap_or(id).to_string()
}

/// The job id whose `name:` is `name`; exactly one must exist.
fn job_named(yml: &Value, name: &str) -> String {
    let hits: Vec<&String> = jobs(yml)
        .iter()
        .filter(|(_, j)| j["name"].as_str() == Some(name))
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "required check `{name}` must be the `name:` of exactly one ci.yml job, found {hits:?}"
    );
    hits[0].clone()
}

#[test]
fn ci_runs_on_the_merge_queue() {
    let yml = ci_yml();
    let on = &yml["on"];
    assert!(
        on.get("merge_group").is_some(),
        "ci.yml has no `merge_group:` trigger, so no required check reports in the merge \
         queue and every queued PR waits until the queue times it out (#2042). Triggers: {on}"
    );
    let types = &on["merge_group"]["types"];
    assert!(
        types.is_null()
            || types
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t == "checks_requested"),
        "`merge_group:` must include `checks_requested`, the only merge-group activity: {types}"
    );
}

#[test]
fn in_the_queue_the_fast_gates_run_and_the_test_jobs_are_skipped() {
    let yml = ci_yml();
    let running = running_jobs(&yml, "merge_group");
    let running_names: BTreeSet<String> = running.iter().map(|id| job_name(&yml, id)).collect();

    for gate in QUEUE_GATES {
        let id = job_named(&yml, gate);
        assert!(
            running.contains(&id),
            "`{gate}` does not run in the merge queue, so a compile-level conflict between \
             two queued PRs lands on `main` unbuilt — the failure #2042 exists to stop"
        );
    }

    let expected: BTreeSet<String> = QUEUE_GATES
        .iter()
        .map(|s| s.to_string())
        .chain([job_name(&yml, PATCH_JOB)])
        .collect();
    let extra: Vec<_> = running_names.difference(&expected).collect();
    assert!(
        extra.is_empty(),
        "these jobs run in the merge queue: {extra:?}. The test and coverage jobs already \
         ran on the PR and must be skipped there (`if: github.event_name != 'merge_group'`). \
         A required job that RUNS in the queue on a skipped dependency fails on \
         `result == 'skipped'` and blocks every queued PR. If the job is a new fast gate, \
         add it to QUEUE_GATES."
    );

    // Every required check reports something in the queue: its job runs or is skipped
    // (both satisfy it), or — for `codecov/patch` — the patch job posts it.
    for check in REQUIRED_CHECKS {
        if check == "codecov/patch" {
            assert!(
                running.contains(PATCH_JOB),
                "nothing posts `codecov/patch` in the queue"
            );
        } else {
            job_named(&yml, check);
        }
    }
}

/// The jobs the queue relies on not to repeat must still run where they are enforced.
/// A `merge_group` exclusion written as a positive event list that dropped
/// `pull_request` would skip the PR's own test run and pass every required check green.
#[test]
fn on_a_pr_and_on_main_every_required_job_runs() {
    let yml = ci_yml();
    for event in ["pull_request", "push"] {
        let running = running_jobs(&yml, event);
        for check in REQUIRED_CHECKS.iter().filter(|c| **c != "codecov/patch") {
            let id = job_named(&yml, check);
            assert!(
                running.contains(&id),
                "required check `{check}` (job `{id}`) does not run on `{event}`"
            );
        }
    }
}

#[test]
fn codecov_patch_is_posted_on_the_merge_group_commit_and_nowhere_else() {
    let yml = ci_yml();
    for event in ["pull_request", "push", "schedule", "workflow_dispatch"] {
        assert!(
            !running_jobs(&yml, event).contains(PATCH_JOB),
            "`{PATCH_JOB}` runs on `{event}`: it would post a green `codecov/patch` on a \
             commit Codecov never graded, forging the patch gate"
        );
    }
    assert!(running_jobs(&yml, "merge_group").contains(PATCH_JOB));

    let job = &jobs(&yml)[PATCH_JOB];
    assert_eq!(
        job["permissions"]["statuses"], "write",
        "`{PATCH_JOB}` needs `statuses: write` to post a commit status"
    );
    let steps = job["steps"].as_array().expect("steps");
    let run: String = steps
        .iter()
        .filter_map(|s| s["run"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let env: String = steps
        .iter()
        .filter_map(|s| s.get("env"))
        .map(|e| e.to_string())
        .collect();
    // Whole shell words, not substrings: `context=codecov/patchx` contains the needle
    // and would post a status no ruleset reads.
    let words: Vec<&str> = run.split_whitespace().collect();
    for needle in [
        "\"repos/$REPO/statuses/$SHA\"",
        "state=success",
        "context=codecov/patch",
    ] {
        assert!(
            words.contains(&needle),
            "`{PATCH_JOB}` no longer posts `{needle}`:\n{run}"
        );
    }
    assert!(
        env.contains("github.event.merge_group.head_sha"),
        "`{PATCH_JOB}` must post on the merge-group commit (`merge_group.head_sha`), the \
         commit the queue reads required checks from: {env}"
    );
}

/// `github.event.pull_request` is empty on a `merge_group` event; a job that runs in the
/// queue and reads it would act on blanks.
#[test]
fn no_job_in_the_queue_reads_pull_request_context() {
    let yml = ci_yml();
    for id in running_jobs(&yml, "merge_group") {
        let body = jobs(&yml)[&id].to_string();
        assert!(
            !body.contains("github.event.pull_request"),
            "job `{id}` runs in the merge queue but reads `github.event.pull_request`, which \
             is empty on a `merge_group` event"
        );
    }
}

// --- the evaluator itself --------------------------------------------------------------

fn ev(expr: &str, event: &str, needs_ok: bool) -> bool {
    eval_if(Some(expr), &Ctx { event, needs_ok })
}

#[test]
fn the_evaluator_reads_the_spellings_ci_yml_uses() {
    // Each pair straddles: same expression, opposite verdicts.
    let neq = "github.event_name != 'merge_group'";
    assert!(ev(neq, "pull_request", true) && !ev(neq, "merge_group", true));

    let any = "github.event_name == 'pull_request' || github.event_name == 'push'";
    assert!(ev(any, "push", true) && !ev(any, "merge_group", true));

    // `always()` survives a skipped dependency; without it the job is skipped.
    let upload = "${{ always() && github.event_name != 'merge_group' }}";
    assert!(ev(upload, "pull_request", false) && !ev(upload, "merge_group", false));
    assert!(ev("${{ always() }}", "merge_group", false));
    assert!(!ev("github.event_name == 'push'", "push", false));
    assert!(!eval_if(
        None,
        &Ctx {
            event: "push",
            needs_ok: false
        }
    ));

    assert!(ev("!(github.event_name == 'push')", "schedule", true));
}

#[test]
#[should_panic(expected = "outside what this guard can evaluate")]
fn the_evaluator_refuses_a_context_it_does_not_model() {
    ev(
        "github.event.pull_request.head.repo.fork == false",
        "pull_request",
        true,
    );
}
