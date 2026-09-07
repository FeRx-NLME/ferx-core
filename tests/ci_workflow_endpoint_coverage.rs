//! Guard: the `Tests + coverage (TTE/CTMM endpoints)` job in `.github/workflows/ci.yml`
//! selects its integration-test binaries with an explicit `--test <name>` list rather
//! than `--tests`, because re-running the whole 122-binary base suite under a second
//! feature set bought no coverage (Codecov merges flags by summing hits) and cost ~15
//! minutes of instrumented compile per PR.
//!
//! The cost of that speed-up is a list that can drift: add `tests/foo.rs` using
//! `#[cfg(feature = "survival")]`, forget the workflow, and its lines silently read as
//! uncovered in the merged report — quietly failing the ≥90% patch gate for the very PR
//! that added them, with nothing pointing at the cause.
//!
//! So the invariant is pinned here as a plain set equality:
//!
//! > the `--test` names in that job == every `tests/*.rs` mentioning a `survival`,
//! > `markov` or `nn` feature cfg.
//!
//! `nn` is in that set because the job builds `--features ci,markov,nn`: the covariate-NN
//! surface is compiled out of the base `ci` build exactly like the endpoint families, and
//! one nn-gated test (`tests/tte_smoke.rs`, #442) needs `survival` *and* `nn` at once — so
//! there is no separate nn job to own it, and no exemption to carve out here.
//!
//! Tier-3 (`slow-tests`-gated) files are deliberately *inside* the set: without the
//! `slow-tests` feature their tests stay `#[ignore]`d, so listing them costs one cheap
//! compile each and keeps this a set equality with no exemption rule to get wrong.
//!
//! Not feature-gated itself — it must run in the base `--features ci` job, which is the
//! one job guaranteed to build every PR.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The default-off features this job's build turns on, and whose gated test files it
/// therefore has to list. Kept in one place so `--features ci,markov,nn` in `ci.yml`
/// and the detector below cannot drift apart. (`survival` is named explicitly even
/// though `markov = ["survival"]` implies it, because test files gate on either name.)
const GATED_FEATURES: [&str; 3] = ["survival", "markov", "nn"];

/// Every `tests/*.rs` whose source mentions one of [`GATED_FEATURES`] as a feature cfg.
///
/// Matching on the raw `feature = "…"` string is what makes the set self-maintaining:
/// any file that compiles feature-gated code *must* gate it (it would not build in the
/// base `ci` job otherwise), and gating means writing that string.
fn endpoint_test_files(tests_dir: &Path) -> BTreeSet<String> {
    // Skip self. The matcher below builds its needles with `format!`, so this file does
    // not currently quote any of them literally — but it used to, and a future edit that
    // inlines a cfg string here (in a doc comment, say) would make the file match itself
    // and demand a `--test` entry for a test that compiles no gated code. `file!()`
    // rather than a hard-coded name so a rename cannot silently re-introduce that.
    let self_stem = Path::new(file!())
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("file!() has a UTF-8 stem");

    let mut found = BTreeSet::new();
    for entry in std::fs::read_dir(tests_dir).expect("tests/ is readable") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if path.file_stem().and_then(|s| s.to_str()) == Some(self_stem) {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("test file is valid UTF-8");
        if GATED_FEATURES
            .iter()
            .any(|f| src.contains(&format!(r#"feature = "{f}""#)))
        {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("test file has a UTF-8 stem");
            found.insert(stem.to_string());
        }
    }
    found
}

/// The `--test <name>` arguments of the endpoint coverage step in `ci.yml`.
///
/// Scoped to that one step by scanning from its `- name:` line to the next step, so an
/// unrelated `--test` elsewhere in the workflow (e.g. a future targeted job) cannot
/// satisfy this assertion by accident.
fn workflow_test_selection(workflow: &str) -> BTreeSet<String> {
    const STEP: &str = "- name: Generate TTE/CTMM endpoint coverage report";
    let step_start = workflow
        .find(STEP)
        .expect("ci.yml still has the endpoint coverage step");
    let rest = &workflow[step_start + STEP.len()..];
    // The step's `run:` block ends at the next step in the same job.
    let step_body = match rest.find("\n      - name:") {
        Some(end) => &rest[..end],
        None => rest,
    };

    let mut selected = BTreeSet::new();
    for (idx, _) in step_body.match_indices("--test ") {
        let after = &step_body[idx + "--test ".len()..];
        let name: String = after
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '\\')
            .collect();
        assert!(!name.is_empty(), "`--test` with no target name in ci.yml");
        selected.insert(name);
    }
    selected
}

#[test]
fn endpoint_coverage_job_lists_every_feature_gated_test_file() {
    let root = repo_root();
    let workflow = std::fs::read_to_string(root.join(".github/workflows/ci.yml"))
        .expect("ci.yml is readable from the repo root");

    let selected = workflow_test_selection(&workflow);
    let expected = endpoint_test_files(&root.join("tests"));

    assert!(
        !expected.is_empty(),
        "no tests/*.rs mention a survival/markov/nn feature cfg — the detector must be \
         broken, since the endpoint job would then measure nothing"
    );

    let missing: Vec<_> = expected.difference(&selected).cloned().collect();
    let extra: Vec<_> = selected.difference(&expected).cloned().collect();

    assert!(
        missing.is_empty(),
        "these tests/*.rs use a `survival`/`markov`/`nn` feature cfg but are NOT built by the \
         `Tests + coverage (TTE/CTMM endpoints)` job in .github/workflows/ci.yml: {missing:?}. \
         Their feature-gated lines would read as uncovered in the merged Codecov report. \
         Add `--test <name>` for each to that job's `cargo llvm-cov` invocation."
    );
    assert!(
        extra.is_empty(),
        "the `Tests + coverage (TTE/CTMM endpoints)` job builds these test binaries, but they \
         no longer contain a `survival`/`markov`/`nn` feature cfg: {extra:?}. They are already run \
         and measured by `Tests + coverage (core)`, so drop their `--test` flags rather than \
         pay a second instrumented compile."
    );
}
