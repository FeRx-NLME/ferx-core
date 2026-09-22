//! Guard: the anchors that evaluate once run on every PR (#1132).
//!
//! A `slow-tests` gate marks a test that runs a fit to convergence, not a test that
//! looks like one. An anchor that evaluates once — a ferx run compared against a frozen
//! mrgsolve table, or a NONMEM objective at `maxiter = 0` — meets the Tier-2 contract in
//! `CLAUDE.md` and is not gated, however slow its oracle was to produce. Before #1132
//! the adaptive mrgsolve anchors and three NONMEM anchors carried
//! `ignore = "slow: …"` on tests that take a second, so a PR that broke them turned the
//! nightly red instead of itself.
//!
//! The gate is a run-time `#[ignore]` only: `Tests + coverage (core)` already builds
//! every `tests/*.rs` binary with `--tests`, so un-gating costs test time, not compile.
//! What can quietly undo it is someone re-adding the attribute out of habit, which is
//! what the three tests below pin:
//!
//! - **adaptive family, by rule** — every `tests/adaptive_*_anchor.rs` on disk (a glob,
//!   so a ninth anchor is covered the day it lands) compiles no `slow-tests` cfg, and
//!   each has a `*_matches_mrgsolve` test with no `ignore` of any spelling on it — the
//!   second half is what stops the first from being defeated by swapping the feature
//!   gate for a plain `#[ignore]`;
//! - **NONMEM family, by hand list** — the rule there is *runtime*, which a static test
//!   cannot measure, so the list is sealed by the measured seconds next to each name
//!   and grows only with a new measurement.

use std::path::{Path, PathBuf};

fn tests_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p
}

/// The NONMEM anchors that evaluate at fixed parameters and run on every PR. The wall
/// time is the whole binary under `--no-default-features --features ci --profile
/// ci-cov -- --include-ignored`, idle macOS, 2026-09-22 at `2e04abba`:
///
/// | binary | time | why it is one evaluation |
/// |---|---|---|
/// | `tad_lag_nonmem_anchor` | 1.0 s | `outer_maxiter: 0` in the test's `FitOptions` |
/// | `ss_lagtime_edge_nonmem_anchor` | 5.2 s | `maxiter = 0` in `nonmem_anchor/ss_lag_iv_fit.ferx` |
/// | `tvcov_lag_saltation_nonmem_anchor` | 10.4 s | `maxiter = 0` in `nonmem_anchor/tvcov_lag_saltation_fit.ferx` |
///
/// Add a name here only with its measured time under that same profile. A NONMEM
/// anchor that runs real fits (`dose_form_lag_nonmem_anchor`, ~15 min) stays gated.
const PER_PR_NONMEM_ANCHORS: [&str; 3] = [
    "tad_lag_nonmem_anchor",
    "ss_lagtime_edge_nonmem_anchor",
    "tvcov_lag_saltation_nonmem_anchor",
];

/// Source with `//` lines removed: a file that *describes* a cfg does not compile one.
/// Same shape as `uses_gated_feature_cfg` in `tests/ci_workflow_endpoint_coverage.rs`.
fn code_lines(src: &str) -> String {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Does this source compile a `slow-tests` cfg anywhere?
fn uses_slow_tests_cfg(src: &str) -> bool {
    code_lines(src).contains(r#"feature = "slow-tests""#)
}

/// The attribute block directly above each `fn <name>` whose name ends in `suffix`:
/// the contiguous run of `#[…]` lines, their indented continuations and `)]` closers,
/// walking up from the `fn` line. Doc and `//` lines are skipped, not collected.
fn attribute_blocks_of(src: &str, suffix: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("fn ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.ends_with(suffix) {
            continue;
        }
        let mut block = Vec::new();
        for above in lines[..i].iter().rev() {
            let t = above.trim_start();
            if t.starts_with("//") {
                continue;
            }
            if t.starts_with("#[") || t.starts_with(")]") || above.starts_with("    ") {
                block.push(t.to_string());
            } else {
                break;
            }
        }
        block.reverse();
        out.push((name, block.join("\n")));
    }
    out
}

fn adaptive_anchor_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(tests_dir())
        .expect("tests/ is readable")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            name.starts_with("adaptive_") && name.ends_with("_anchor.rs")
        })
        .collect();
    files.sort();
    files
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).expect("test file is valid UTF-8")
}

#[test]
fn adaptive_anchors_compile_no_slow_tests_gate() {
    let files = adaptive_anchor_files();
    // Eight at #1132. A floor, not an equality: a new anchor is covered by the glob, and
    // an empty glob (a renamed directory, a changed naming scheme) must not pass vacuously.
    assert!(
        files.len() >= 8,
        "expected at least 8 tests/adaptive_*_anchor.rs, found {}: {files:?}",
        files.len()
    );
    let gated: Vec<_> = files
        .iter()
        .filter(|p| uses_slow_tests_cfg(&read(p)))
        .collect();
    assert!(
        gated.is_empty(),
        "adaptive mrgsolve anchors are one deterministic run against a frozen table and \
         run on every PR (#1132); these carry a `slow-tests` gate: {gated:?}"
    );
}

#[test]
fn adaptive_anchors_have_an_unignored_mrgsolve_comparison() {
    for path in adaptive_anchor_files() {
        let blocks = attribute_blocks_of(&read(&path), "_matches_mrgsolve");
        assert!(
            !blocks.is_empty(),
            "{path:?} has no `fn *_matches_mrgsolve`: an adaptive anchor must compare \
             against its mrgsolve table under that name, so this guard can see it"
        );
        for (name, attrs) in blocks {
            assert!(
                attrs.contains("#[test]"),
                "{path:?}: `{name}` is not a `#[test]` (attributes: {attrs:?})"
            );
            assert!(
                !attrs.contains("ignore"),
                "{path:?}: `{name}` is ignored — the mrgsolve comparison runs on every PR \
                 (#1132) (attributes: {attrs:?})"
            );
        }
    }
}

#[test]
fn cheap_nonmem_anchors_compile_no_slow_tests_gate() {
    for stem in PER_PR_NONMEM_ANCHORS {
        let path = tests_dir().join(format!("{stem}.rs"));
        assert!(
            !uses_slow_tests_cfg(&read(&path)),
            "{path:?} evaluates at fixed parameters and runs on every PR (#1132; measured \
             time in PER_PR_NONMEM_ANCHORS); it carries a `slow-tests` gate"
        );
    }
}

#[test]
fn detectors_see_the_gate_shapes_they_exist_to_catch() {
    let gated = "#[test]\n#[cfg_attr(\n    not(feature = \"slow-tests\"),\n    ignore = \"slow\"\n)]\nfn x_matches_mrgsolve() {}\n";
    assert!(uses_slow_tests_cfg(gated));
    assert!(!uses_slow_tests_cfg(
        "// #[cfg_attr(not(feature = \"slow-tests\"), ignore)]\nfn f() {}\n"
    ));

    let blocks = attribute_blocks_of(gated, "_matches_mrgsolve");
    assert_eq!(blocks.len(), 1);
    assert!(blocks[0].1.contains("#[test]") && blocks[0].1.contains("ignore"));

    let plain = "/// doc\n#[test]\n#[ignore]\nfn y_matches_mrgsolve() {}\n";
    let blocks = attribute_blocks_of(plain, "_matches_mrgsolve");
    assert_eq!(blocks[0].0, "y_matches_mrgsolve");
    assert!(blocks[0].1.contains("#[ignore]"));

    let clean = "}\n\n/// doc\n#[test]\nfn z_matches_mrgsolve() {}\n";
    let blocks = attribute_blocks_of(clean, "_matches_mrgsolve");
    assert_eq!(blocks[0].1, "#[test]");
}
