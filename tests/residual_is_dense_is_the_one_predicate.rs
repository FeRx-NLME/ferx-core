//! Guard for #1081 phase 0: engine code asks "can `R` couple two observation rows?"
//! through one predicate, and builds a dense `R` through one structure.
//!
//! Before #1081 about forty sites answered that question by testing
//! `residual_correlations.is_empty()`. A second source of off-diagonal residual
//! terms (the serial CS / AR(1) / CAR(1) correlation of #1081) that does not go
//! through every one of them is silently ignored by the one it misses: that site
//! scores the model with independent residuals and still returns a finite,
//! plausible objective. So the question now has one answer —
//! `ResidualCov::is_dense`, reached directly or through
//! `ModelParameters::residual_is_dense` / `CompiledModel::residual_is_dense` — and
//! the dense builders take a `ResidualCov` rather than the bare `block_sigma` list.
//!
//! Two checks keep it that way:
//!
//! 1. no production code tests `<…>correlations.is_empty()` outside the allow-list
//!    below, every entry of which asks a question about the `block_sigma`
//!    *declaration* rather than about `R`'s shape;
//! 2. no production code calls the `pub` slice-taking builders (kept for the public
//!    API), which describe a `block_sigma`-only structure — in-crate callers use
//!    their `_cov` twins.
//!
//! Scope: production code under `src/` — `*_tests.rs` siblings, `tests/` folders,
//! `test_helpers` files and the bodies of inline `#[cfg(test)] mod … { }` modules
//! are skipped. The remaining gap is an alias: `let c = &p.residual_correlations;
//! if c.is_empty()` is not seen. None exists today. A rustfmt-wrapped chain
//! (`p\n    .residual_correlations\n    .is_empty()`) *is* seen.
//!
//! Not feature-gated: runs in the base `--features ci` job.

use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable directory") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) != Some("tests") {
                rust_sources(&path, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.ends_with("_tests.rs") && !name.contains("test_helpers") {
                out.push(path);
            }
        }
    }
}

/// The production part of a source file: every line outside an inline
/// `#[cfg(test)] mod name { … }` body, which is blanked (so line numbers still
/// match the file). Same scanner as `tests/bloq_in_force_is_the_one_reader.rs`.
fn production_part(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut prev_is_cfg_test = false;
    let mut depth: i64 = 0;
    for line in src.split_inclusive('\n') {
        let t = line.trim();
        let opens = prev_is_cfg_test && depth == 0 && t.contains("mod ") && t.ends_with('{');
        if depth > 0 || opens {
            depth += line.matches('{').count() as i64 - line.matches('}').count() as i64;
            out.push('\n');
            prev_is_cfg_test = false;
            continue;
        }
        if !t.starts_with("#[path") {
            prev_is_cfg_test = t == "#[cfg(test)]";
        }
        out.push_str(line);
    }
    out
}

/// Every `<ident ending in "correlations">.is_empty()` in `src` as `(line, trimmed line)`,
/// including the rustfmt-wrapped chain form (`…correlations` ending one line and
/// `.is_empty()` starting the next), reported at the `.is_empty()` line with the two
/// trimmed lines joined.
fn emptiness_tests(src: &str) -> Vec<(usize, String)> {
    const NEEDLE: &str = "correlations.is_empty()";
    let mut out = Vec::new();
    let mut prev = "";
    for (i, line) in src.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("//") {
            continue;
        }
        if line.contains(NEEDLE) {
            out.push((i + 1, t.to_string()));
        } else if t.starts_with(".is_empty()") && prev.ends_with("correlations") {
            out.push((i + 1, format!("{prev}{t}")));
        }
        prev = t;
    }
    out
}

/// The `pub` builders whose `&[ResidualCorrelation]` argument can only describe a
/// `block_sigma` structure. Each has a `pub(crate)` `<name>_cov` twin.
const SLICE_WRAPPERS: &[&str] = &[
    "individual_nll_into_with_schedule",
    "foce_subject_nll",
    "foce_subject_nll_standard",
    "foce_subject_nll_interaction_dense",
    "foce_population_nll",
    "compute_cwres",
    "compute_r_matrix_with_correlations",
    "compute_r_matrix_with_correlations_scaled",
    "compute_dr_df_matrices",
    "compute_d2r_df2_matrices",
];

/// Every call of a [`SLICE_WRAPPERS`] function in `src` as `(line, trimmed line)` —
/// a name followed by `(` with no identifier character before it and not preceded
/// by `fn ` (the definition). `foo_cov(` does not match `foo(`.
fn wrapper_calls(src: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        for name in SLICE_WRAPPERS {
            let needle = format!("{name}(");
            let mut from = 0;
            while let Some(at) = line[from..].find(&needle) {
                let at = from + at;
                let before = line[..at].chars().next_back();
                let is_ident_tail = before.is_some_and(|c| c.is_alphanumeric() || c == '_');
                if !is_ident_tail && !line[..at].trim_end().ends_with("fn") {
                    out.push((i + 1, line.trim().to_string()));
                }
                from = at + needle.len();
            }
        }
    }
    out
}

/// `(file under src/, trimmed line)` of the emptiness tests that ask about the
/// `block_sigma` declaration itself, not about whether `R` is dense.
const ALLOWED: &[(&str, &str, &str)] = &[
    // Reporting: the `block_sigma` ρ is reported as its own block (a serial ρ is a θ).
    (
        "src/io/output.rs",
        "if !result.residual_correlations.is_empty() {",
        "reports the block_sigma rows",
    ),
    (
        "src/io/fitrx.rs",
        "let residual_correlation_fixed_resolved = if w.sigma.residual_correlations.is_empty() {",
        "round-trips the block_sigma FIX flags",
    ),
    (
        "src/estimation/uncertainty_samples.rs",
        "residual_correlations: if fit_result.residual_correlations.is_empty() {",
        "carries the block_sigma ρ into an uncertainty draw",
    ),
    (
        "src/api/postfit.rs",
        "if template.residual_correlations.is_empty() {",
        "SE of the packed block_sigma ρ coordinates; None (not Some([])) without any",
    ),
    // The declaration's own validation.
    (
        "src/parser/model_parser.rs",
        "if correlations.is_empty() {",
        "validate_residual_correlations: checks the declared block_sigma pairs",
    ),
    (
        "src/api/validation.rs",
        "|| (rho_fixed.is_empty() && !model.residual_correlations.is_empty());",
        "is a block_sigma ρ free to estimate (E_BLOCK_SIGMA_CHAIN_UNSUPPORTED)",
    ),
    (
        "src/api/validation.rs",
        "if model.residual_correlations.is_empty()",
        "W_L2_UNUSED: an L2 column pairs block_sigma rows only",
    ),
    (
        "src/api/validation.rs",
        "if !model.residual_correlations.is_empty() {",
        "row-order-dependent (time, occasion) pairing of block_sigma rows",
    ),
    // Per-observation diagonal: the within-observation `combined(...)` term is a
    // property of the block_sigma pairs, never of R's off-diagonal shape.
    (
        "src/stats/residual_error.rs",
        "if correlations.is_empty() {",
        "variance_at_with_correlations / compute_r_diag_with_correlations / \
         compute_iwres_with_correlations: diagonal only",
    ),
];

/// How many times each allowed `(file, line)` may occur.
fn allowed_count(file: &str, line: &str) -> usize {
    match (file, line) {
        ("src/stats/residual_error.rs", "if correlations.is_empty() {") => 3,
        _ => 1,
    }
}

fn scan<F>(find: F) -> Vec<(String, usize, String)>
where
    F: Fn(&str) -> Vec<(usize, String)>,
{
    // Keys are repo-relative (`src/api/…`), never `src`-relative: a string literal
    // `api/…` in `tests/` reads as a fixture under the tracked top-level `api/`
    // directory to `tests/slow_test_path_filter.rs`.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    rust_sources(&root.join("src"), &mut sources);
    assert!(
        sources.len() > 50,
        "found only {} sources under src/",
        sources.len()
    );
    let mut hits = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("source file is valid UTF-8");
        let rel = path
            .strip_prefix(&root)
            .expect("under the manifest dir")
            .to_string_lossy()
            .replace('\\', "/");
        for (n, line) in find(&production_part(&src)) {
            hits.push((rel.clone(), n, line));
        }
    }
    hits
}

#[test]
fn the_scanners_see_gates_and_calls_and_skip_test_modules() {
    // The scanners' own fixture, so a scanner that stopped matching anything cannot
    // pass the guards below by finding nothing.
    let src = "fn a(p: &P) -> bool { p.residual_correlations.is_empty() }\n\
               fn b(p: &P) -> bool { p.residual_is_dense() }\n\
               // if p.residual_correlations.is_empty() {\n\
               fn c(c: &[R]) -> bool { !draw_correlations.is_empty() }\n\
               fn d() { foce_subject_nll(a, b); foce_subject_nll_cov(a, b); }\n\
               pub fn foce_subject_nll(\n\
               fn e() { x.my_compute_cwres(1); crate::stats::likelihood::compute_cwres(1) }\n\
               #[cfg(test)]\nmod tests {\n  fn f(p: &P) -> bool { p.residual_correlations.is_empty() }\n  \
               fn g() { foce_subject_nll(a) }\n}\n\
               fn h(p: &P) -> bool { p.residual_correlations.is_empty() }\n\
               fn k(p: &P) -> bool {\n    p.residual_correlations\n        .is_empty()\n}\n\
               fn m(p: &P) -> bool {\n    p.omega_corrs\n        .is_empty()\n}\n";
    let prod = production_part(src);
    let gates: Vec<usize> = emptiness_tests(&prod).into_iter().map(|(n, _)| n).collect();
    // 16: a rustfmt-wrapped chain; 20 is the same shape on a non-correlation field.
    assert_eq!(
        gates,
        vec![1, 4, 13, 16],
        "gates on lines 1, 4, 13 and 16 only"
    );
    let calls: Vec<usize> = wrapper_calls(&prod).into_iter().map(|(n, _)| n).collect();
    assert_eq!(calls, vec![5, 7], "wrapper calls on lines 5 and 7 only");
}

#[test]
fn no_production_code_tests_residual_correlations_for_emptiness() {
    let mut offenders = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    for (file, n, line) in scan(emptiness_tests) {
        if ALLOWED.iter().any(|(f, l, _)| *f == file && *l == line) {
            seen.push((file, line));
        } else {
            offenders.push(format!("{file}:{n}: {line}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "production code decides whether the residual covariance R is dense by testing \
         `…correlations.is_empty()`. A correlation source that is not a `block_sigma` pair \
         (#1081) is invisible to that test, so the site would score the model with \
         independent residuals. Ask `ResidualCov::is_dense` (or `ModelParameters::\
         residual_is_dense` / `CompiledModel::residual_is_dense`) instead — or, if the \
         site asks about the `block_sigma` declaration itself, add it to `ALLOWED` with \
         the reason:\n{}",
        offenders.join("\n")
    );
    // Every allow-list entry must still exist, as often as it says: a stale entry
    // would let a new gate with the same text through unnoticed.
    for (f, l, why) in ALLOWED {
        let got = seen.iter().filter(|(sf, sl)| sf == f && sl == l).count();
        assert_eq!(
            got,
            allowed_count(f, l),
            "allow-list entry {f}: `{l}` ({why}) occurs {got} time(s)"
        );
    }
}

#[test]
fn no_production_code_calls_a_block_sigma_only_dense_builder() {
    let offenders: Vec<String> = scan(wrapper_calls)
        .into_iter()
        .map(|(f, n, l)| format!("{f}:{n}: {l}"))
        .collect();
    assert!(
        offenders.is_empty(),
        "production code calls a `pub` dense-R builder that takes `&[ResidualCorrelation]`. \
         That slice can only describe `block_sigma` pairs, so a second correlation source \
         (#1081) would be dropped on this path. Call its `_cov` twin with \
         `params.residual_cov()` / `model.residual_cov()` instead:\n{}",
        offenders.join("\n")
    );
}
