//! The corpus check: the workspace's test code must pass every rule, net of the
//! shrink-only baseline in `tools/lint-rules/baseline.txt`.
//!
//! This is what `tools/preflight.sh patterns` (CI job `Patterns lint`) runs, so a
//! PR that adds an unguarded `max`/`min` fold to a test fails before review, with
//! a message naming the rule and both escape hatches.

use patterns_lint::baseline::{self, Baseline};
use patterns_lint::{default_baseline_path, lint_tree, render_report, repo_root};

#[test]
fn test_code_passes_every_rule_net_of_the_baseline() {
    let root = repo_root();
    assert!(
        root.join("tests").is_dir() && root.join("src").is_dir(),
        "repo_root() resolved to {}, which is not the workspace root",
        root.display()
    );
    let findings = lint_tree(&root).expect("scan the tree");
    let base = Baseline::load(&default_baseline_path(&root)).expect("parse the baseline");
    let applied = baseline::apply(&base, findings);
    let report = render_report(&applied);
    assert!(
        report.is_empty(),
        "patterns-lint: {} failing finding(s), {} stale baseline entr(y/ies):\n\n{report}\n\
         Run `cargo run -p patterns-lint` for the same report. Do not add a line to the \
         baseline to make this pass — it is shrink-only.",
        applied.failing.len(),
        applied.stale.len()
    );
}

/// The scan reaches the places the rule exists for. If scope discovery broke and
/// found no test code at all, the check above would pass vacuously.
#[test]
fn the_scan_actually_reaches_test_code() {
    let root = repo_root();
    let files = patterns_lint::scan(&root).expect("scan the tree").files;
    let whole_test = files.iter().filter(|f| f.whole_test).count();
    assert!(
        whole_test > 100,
        "only {whole_test} whole-file test sources found — scope discovery is broken"
    );
    for must in [
        "tests/preflight_owns_the_fast_gates.rs",
        "crates/ferx-cli/tests/cli_ferx.rs",
    ] {
        assert!(
            files.iter().any(|f| f.rel == must && f.whole_test),
            "`{must}` was not scanned as test code"
        );
    }
    // A sibling `#[path]` test file, by the repo's `*_tests.rs` convention.
    assert!(
        files
            .iter()
            .any(|f| f.rel.starts_with("src/") && f.rel.ends_with("_tests.rs")),
        "no src/**/*_tests.rs sibling test file was scanned"
    );
}
