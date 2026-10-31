//! Guard for #1727 (follow-up to #1710): the per-method interaction rule lives in
//! `crate::types::interaction_for`, and production code reads the raw
//! `FitOptions::interaction` field only where it is already resolved.
//!
//! The top-level field defaults to `true` and is only what a stage whose method does not
//! fix the flag (Gauss-Newton, Laplace, …) inherits; `method = foce` turns it off at the
//! stage loop, not on the caller's struct. So a post-hoc entry taking `(fit, options)`
//! that reads `options.interaction` scores a FOCE fit under the FOCEI objective — the
//! #1710 leak, 151 OFV units and an SIR ESS of 4.9 / 1000 on warfarin_iov. Such a read
//! compiles, runs and passes every fixture that happens to be FOCEI, which is why it is
//! pinned here instead of left to review.
//!
//! The guard is a ratchet over **files**: every file that reads a `.interaction` field
//! is listed in [`ALLOWED`] with its exact count and the reason its reads see resolved
//! options. A new read anywhere fails — in an unlisted file, or as one more in a listed
//! one — and so does a read that disappeared, so the list cannot go stale. A new read
//! of *resolved* options (stage options, `post_opts`, `fitted_marginal_options`'s
//! result, a stored `FitResult::interaction`) is fixed by bumping the count; a read of a
//! caller's raw options is fixed by routing it through `interaction_for` (or
//! `fit_inputs::fitted_marginal_options` for a post-hoc step on a fit).
//!
//! Scope: production code under `src/` — `*_tests.rs` siblings, `tests/` folders,
//! `test_helpers` files and the bodies of inline `#[cfg(test)] mod … { }` modules are
//! skipped (the same scope as `bloq_in_force_is_the_one_reader.rs`). A read is any
//! `<receiver>.interaction` — the receiver may sit on an earlier line, as rustfmt
//! splits a long chain — not followed by an identifier character or an assignment
//! `=` (`==` and a match guard's `=>` are reads), whatever the receiver: the field is also on `FitResult`, and those reads count
//! too, because the scanner cannot tell the two types apart — a per-file reason says
//! which they are. Remaining gaps: a granularity of one file, so a raw read can trade
//! places with a resolved one that was deleted in the same change; and a destructuring
//! `let FitOptions { interaction, .. } = …`, which has no `.interaction` to find.
//!
//! Not feature-gated: runs in the base `--features ci` job.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// `(repo-relative path, read count, why these reads see a resolved flag)`. Written from
/// the repo root, not `src/`: `slow_test_path_filter.rs` reads every path literal under
/// `tests/` as a repo-relative input, so an `api/…` entry would name a top-level `api/`.
const ALLOWED: &[(&str, usize, &str)] = &[
    (
        "src/api/fit.rs",
        9,
        "the stage loop resolves `stage_opts.interaction` through `interaction_for`; the \
         `post_opts` shadow resolves it for the post-fit block, so every later \
         `options.interaction` there is the resolved flag; the optimizer label reads it \
         through `analytic_outer_gradient_for_interaction` on a stage's options",
    ),
    (
        "src/api/postfit.rs",
        3,
        "`ContributionSource::of` takes the post-fit options and compares them to \
         `interaction_for(method, ..)` itself",
    ),
    (
        "src/estimation/cov_diagnostics.rs",
        1,
        "`cov_objective_method` names the covariance step's objective, on the same options \
         as `covariance.rs`",
    ),
    (
        "src/estimation/covariance.rs",
        3,
        "the covariance step's score and objective, called with `post_opts` from `fit` and \
         with `fitted_marginal_options`'s result from `run_covariance`; the third names the \
         FOCE / FOCEI marginal a SIR pass weights by in `W_COV_NON_STATIONARY_ESTIMATE` \
         (#2536), on the same resolved options",
    ),
    (
        "src/estimation/fit_inputs.rs",
        1,
        "`fitted_marginal_options` reads the stored `fit.interaction` and resolves it \
         against `fit.method` in `scoring_options`",
    ),
    (
        "src/estimation/gauss_newton.rs",
        13,
        "the GN optimizer and its gradient kernels receive the stage options (the exact \
         fixed-EBE score and EBE-response correction pick their objective from them, #2482)",
    ),
    (
        "src/estimation/impmap.rs",
        1,
        "the IMP-MAP inner pass, run on a stage's options",
    ),
    (
        "src/estimation/inner_optimizer.rs",
        1,
        "`InnerMode::for_options` is documented as taking a stage's resolved options",
    ),
    (
        "src/estimation/mixture.rs",
        2,
        "the mixture outer loop, run on a stage's options",
    ),
    (
        "src/estimation/outer_optimizer.rs",
        15,
        "the outer optimizer and objective, run on a stage's options; \
         `warm_ebe_start_check` (#2097) runs on the Gauss–Newton stage's options",
    ),
    (
        "src/estimation/profile.rs",
        5,
        "the profiled outer loop is a stage of the stage loop, on its stage options; the \
         profile solve's adjoint level gradient (#2109) differentiates the same stage objective",
    ),
    (
        "src/estimation/saem.rs",
        1,
        "the SAEM stage's final objective, on its stage options",
    ),
    (
        "src/estimation/vi/run.rs",
        1,
        "the VI stage's objective, on its stage options",
    ),
    (
        "src/io/fitrx.rs",
        2,
        "`FitResult::interaction` round-tripped through the `.fitrx` record, not a `FitOptions`",
    ),
    (
        "src/parser/model_parser.rs",
        2,
        "both `method =` arms resolve the parsed flag through `interaction_for`",
    ),
    (
        "src/types.rs",
        1,
        "`FitOptions::non_interaction_stage` applies the same inherit rule to the GN stages \
         of the chain, whose flag `interaction_for` passes through",
    ),
];

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
/// `#[cfg(test)] mod name { … }` body, which is blanked (so line numbers still match the
/// file). Same rules, and the same per-character brace-counting gap, as the copy in
/// `bloq_in_force_is_the_one_reader.rs`.
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

/// Every `.interaction` read in `src` as `(line number, line)`.
///
/// Scans the whole text, not line by line: rustfmt breaks a long chain as
/// `options\n    .interaction`, where the receiver is on the line above (review r1). So
/// the receiver is the last non-whitespace character before the `.`, across newlines,
/// and the assignment test looks past newlines too. Whole-line `//` comments are blanked
/// first; a trailing comment is still scanned, which can only over-count.
fn field_reads(src: &str) -> Vec<(usize, String)> {
    const NEEDLE: &str = ".interaction";
    let code: String = src
        .split_inclusive('\n')
        .map(|line| {
            if line.trim_start().starts_with("//") {
                "\n"
            } else {
                line
            }
        })
        .collect();
    let lines: Vec<&str> = code.lines().collect();
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = code[from..].find(NEEDLE) {
        let at = from + rel;
        let before = code[..at].trim_end().chars().next_back();
        let after = &code[at + NEEDLE.len()..];
        let next = after.chars().next();
        let has_receiver =
            before.is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | ')' | ']' | '?'));
        let is_ident_tail = next.is_some_and(|c| c.is_alphanumeric() || c == '_');
        let a = after.trim_start();
        let is_assignment = a.starts_with('=') && !a.starts_with("==") && !a.starts_with("=>");
        if has_receiver && !is_ident_tail && !is_assignment {
            let n = code[..at].matches('\n').count();
            out.push((n + 1, lines[n].trim().to_string()));
        }
        from = at + NEEDLE.len();
    }
    out
}

#[test]
fn the_classifier_sees_reads_and_skips_assignments_and_test_modules() {
    // The scanner's own fixture, so a scanner that stopped matching anything cannot pass
    // the guard below by matching the allow-list's counts by accident.
    let src = "fn a(o: &O) -> bool { o.interaction }\n\
               fn b(o: &mut O) { o.interaction = interaction_for(m, o.interaction); }\n\
               fn c(o: &O) -> bool { self.options.interaction == true }\n\
               fn d(o: &O) -> bool { o.interactions.is_empty() || o.interaction_for() }\n\
               // a comment naming options.interaction\n\
               #[cfg(test)]\n#[path = \"x_tests.rs\"]\nmod sib;\n\
               fn f(o: &O) -> bool { get().interaction }\n\
               #[cfg(test)]\nmod tests {\n  fn g(o: &O) -> bool { o.interaction }\n  \
               mod inner {\n    fn h(o: &O) -> bool { o.interaction }\n  }\n}\n\
               fn k(o: &O) -> bool { !o.interaction }\n\
               fn l(o: &O) -> u8 { match 0 { _ if o.interaction => 1, _ => 0 } }\n\
               fn m(o: &O) -> bool {\n    o\n        // a comment between\n        .interaction\n}\n\
               fn n(o: &mut O) {\n    o\n        .interaction = true;\n}\n\
               fn p(o: &[O]) -> bool { o[0].interaction && get()?.interaction }\n";
    let lines: Vec<usize> = field_reads(&production_part(src))
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    // 1; 2 (the right-hand side only, not the assignment); 3; 9 (a call receiver);
    // 17: production code *after* an inline test module (with a nested module inside
    // it), which must still be scanned; 18: a match guard, whose `=>` is not an `=`;
    // 22: a rustfmt-split chain whose receiver is two lines up, past a comment (review
    // r1) — while the split *assignment* on line 26 is not a read; 28 twice: an index
    // and a `?` receiver.
    assert_eq!(
        lines,
        vec![1, 2, 3, 9, 17, 18, 22, 28, 28],
        "reads on lines 1, 2, 3, 9, 17, 18, 22 and 28 (twice) only"
    );
}

#[test]
fn the_allow_list_names_each_file_once() {
    let mut seen = std::collections::BTreeSet::new();
    for (path, count, why) in ALLOWED {
        assert!(seen.insert(*path), "{path} is listed twice");
        assert!(
            *count > 0,
            "{path} is listed with no reads; delete the entry"
        );
        assert!(!why.trim().is_empty(), "{path} is listed without a reason");
    }
}

#[test]
fn raw_interaction_reads_stay_on_the_allow_list() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = root.join("src");
    let mut sources = Vec::new();
    rust_sources(&src_dir, &mut sources);
    assert!(
        sources.len() > 50,
        "found only {} sources under src/",
        sources.len()
    );

    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("source file is valid UTF-8");
        let rel = path
            .strip_prefix(&root)
            .expect("source under the repo root")
            .to_string_lossy()
            .replace('\\', "/");
        for (n, line) in field_reads(&production_part(&src)) {
            found
                .entry(rel.clone())
                .or_default()
                .push(format!("{rel}:{n}: {line}"));
        }
    }

    let allowed: BTreeMap<&str, usize> = ALLOWED.iter().map(|(p, n, _)| (*p, *n)).collect();
    let mut problems = Vec::new();
    for (rel, reads) in &found {
        match allowed.get(rel.as_str()) {
            None => problems.push(format!(
                "{rel} is not on the allow-list but reads `.interaction` {} time(s):\n  {}",
                reads.len(),
                reads.join("\n  ")
            )),
            Some(&n) if n != reads.len() => problems.push(format!(
                "{rel} reads `.interaction` {} time(s), the allow-list says {n}:\n  {}",
                reads.len(),
                reads.join("\n  ")
            )),
            Some(_) => {}
        }
    }
    for rel in allowed.keys() {
        if !found.contains_key(*rel) {
            problems.push(format!(
                "{rel} is on the allow-list but no longer reads `.interaction`; delete its entry"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "the raw `FitOptions::interaction` field defaults to `true` and is only what a \
         stage whose method does not fix the flag inherits; reading a caller's options \
         directly scores a `method = foce` fit under the FOCEI objective (#1710). Resolve \
         it through `crate::types::interaction_for(method, options.interaction)` — or, for \
         a post-hoc step on a fit, through `fit_inputs::fitted_marginal_options` — and if \
         the read is of options that are already resolved, update its count in `ALLOWED` \
         with the reason (#1727):\n{}",
        problems.join("\n")
    );
}
