//! **An IMP test that tunes `impmap_*`, or an IMPMAP test that tunes `imp_*`,
//! tunes nothing** (#1476).
//!
//! `run_imp` reads `imp_iterations` / `imp_samples` / `imp_seed`; `run_impmap`
//! reads the `impmap_*` family. The two families are separate `FitOptions`
//! fields with identical suffixes, so setting the wrong one compiles, runs and
//! passes — on the *defaults*, 200 iterations × 1000 samples, with the seed the
//! author wrote never applied. Three unit tests in `src/estimation/impmap.rs`
//! had that shape. Measured serially under `cargo llvm-cov --profile ci-cov`
//! they cost 202 s where the intended 2 × 40 costs 0.3 s, and because a short
//! `fit()` queues behind a long one on the shared fit pool they were the
//! five-minute stall in the lib run of both per-PR coverage jobs.
//!
//! Nothing failed, which is the point: a test that is merely slow is invisible
//! until someone reads a CI log with timestamps. So the shape is pinned here.
//!
//! The rule is per *function* (per file for a `.ferx` model): a body that names
//! exactly one of the two methods and assigns only the *other* method's knobs
//! is a mismatch. A body naming both — a `focei → impmap → imp` chain — may set
//! either family, and a helper that names neither is not judged.

use regex::Regex;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Tracked files only, via `git ls-files` — a filesystem walk from the main
/// checkout would read every sibling branch parked under the git-ignored
/// `.claude/worktrees/` (see `public_api_boundary.rs::scannable_files`).
fn tracked_files(root: &Path, extension: &str) -> Vec<PathBuf> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .expect("`git ls-files` runs (the guard scans committed files only)");
    assert!(
        out.status.success(),
        "`git ls-files` failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("tracked paths are UTF-8")
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(|s| root.join(s))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(extension))
        .collect()
}

/// Which way a body is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mismatch {
    /// Names IMP only, assigns `impmap_*` only.
    ImpWithImpmapKnobs,
    /// Names IMPMAP only, assigns `imp_*` only.
    ImpmapWithImpKnobs,
}

struct Patterns {
    names_imp: Regex,
    names_impmap: Regex,
    sets_imp_knob: Regex,
    sets_impmap_knob: Regex,
    fn_start: Regex,
}

impl Patterns {
    fn new() -> Self {
        // `\b` after `Imp` is what keeps `EstimationMethod::Impmap` out of the
        // IMP pattern: `p`→`m` is not a word boundary. The DSL arm takes the
        // whole right-hand side of a `method =` / `methods =` line, so a chain
        // spelled on one line names every method in it.
        //
        // A knob is *assigned* by `name =` (not `==`) or, in a struct literal,
        // `name:`. A path separator needs no excusing — `FitOptions::imp_seed`
        // has its `::` *before* the name, and nothing follows a field name with
        // one. The leading class stands in for a lookbehind,
        // which the `regex` crate does not have: it stops `imp_` matching
        // inside a longer identifier.
        let knob = |family: &str| {
            Regex::new(&format!(
                r"(?m)(?:^|[^A-Za-z0-9_]){family}_[a-z_]+\s*(?:=(?:[^=]|$)|:)"
            ))
            .expect("knob pattern compiles")
        };
        Patterns {
            names_imp: Regex::new(r"EstimationMethod::Imp\b|(?m)^\s*methods?\s*=[^\n]*\bimp\b")
                .expect("compiles"),
            names_impmap: Regex::new(
                r"EstimationMethod::Impmap\b|(?m)^\s*methods?\s*=[^\n]*\bimpmap\b",
            )
            .expect("compiles"),
            sets_imp_knob: knob("imp"),
            sets_impmap_knob: knob("impmap"),
            fn_start: Regex::new(r"(?m)^\s*(?:pub(?:\([a-z]+\))?\s+)?fn\s+(\w+)")
                .expect("compiles"),
        }
    }

    /// Judge one body of text.
    fn classify(&self, body: &str) -> Option<Mismatch> {
        let (imp, impmap) = (
            self.names_imp.is_match(body),
            self.names_impmap.is_match(body),
        );
        let (imp_knob, impmap_knob) = (
            self.sets_imp_knob.is_match(body),
            self.sets_impmap_knob.is_match(body),
        );
        match (imp, impmap) {
            (true, false) if impmap_knob && !imp_knob => Some(Mismatch::ImpWithImpmapKnobs),
            (false, true) if imp_knob && !impmap_knob => Some(Mismatch::ImpmapWithImpKnobs),
            _ => None,
        }
    }

    /// Judge a Rust source function by function. A body runs from its `fn`
    /// line to the next one, so a nested closure stays with its function.
    fn mismatches_in_rust(&self, src: &str) -> Vec<(String, Mismatch)> {
        let starts: Vec<(usize, String)> = self
            .fn_start
            .captures_iter(src)
            .map(|c| (c.get(0).unwrap().start(), c[1].to_string()))
            .collect();
        let mut out = Vec::new();
        for (i, (start, name)) in starts.iter().enumerate() {
            let end = starts.get(i + 1).map_or(src.len(), |(s, _)| *s);
            if let Some(m) = self.classify(&src[*start..end]) {
                out.push((name.clone(), m));
            }
        }
        out
    }
}

/// The shape #1476 found, verbatim but for the model text.
const OLD_SHAPE: &str = "
    #[test]
    fn imp_shared_anchor() {
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
    }
";

#[test]
fn the_classifier_flags_the_shape_that_was_found() {
    let p = Patterns::new();
    assert_eq!(
        p.mismatches_in_rust(OLD_SHAPE),
        vec![(
            "imp_shared_anchor".to_string(),
            Mismatch::ImpWithImpmapKnobs
        )]
    );
    // …and the mirror image, in struct-literal spelling.
    let mirror = "
    fn impmap_case() {
        let opts = FitOptions {
            method: EstimationMethod::Impmap,
            imp_iterations: 2,
            ..FitOptions::default()
        };
    }
";
    assert_eq!(
        p.mismatches_in_rust(mirror),
        vec![("impmap_case".to_string(), Mismatch::ImpmapWithImpKnobs)]
    );
}

/// The judgement is per function. `src/estimation/impmap.rs` — where the three
/// offenders lived — names both methods many times over, so a rule applied to
/// the file as a whole excuses everything in it.
#[test]
fn a_correct_neighbour_does_not_excuse_a_mismatch() {
    let p = Patterns::new();
    let correct_impmap = OLD_SHAPE
        .replace("imp_shared_anchor", "impmap_neighbour")
        .replace("EstimationMethod::Imp;", "EstimationMethod::Impmap;");
    let file = format!("{correct_impmap}{OLD_SHAPE}{correct_impmap}");
    assert_eq!(
        p.classify(&file),
        None,
        "premise: as one body it is excused"
    );
    assert_eq!(
        p.mismatches_in_rust(&file),
        vec![(
            "imp_shared_anchor".to_string(),
            Mismatch::ImpWithImpmapKnobs
        )]
    );
}

#[test]
fn the_classifier_flags_a_model_file_too() {
    let p = Patterns::new();
    let model = "[fit_options]\n  method = imp\n  impmap_iterations = 5\n";
    assert_eq!(p.classify(model), Some(Mismatch::ImpWithImpmapKnobs));
    let mirror = "[fit_options]\n  method = impmap\n  imp_samples = 50\n";
    assert_eq!(p.classify(mirror), Some(Mismatch::ImpmapWithImpKnobs));
}

/// The excusals are where a rule like this goes wrong, so each is pinned: a
/// scan over the tree only proves today's tree is clean.
#[test]
fn the_classifier_excuses_what_is_not_a_mismatch() {
    let p = Patterns::new();
    let excused = [
        // The corrected shape, both families.
        OLD_SHAPE.replace("opts.impmap_", "opts.imp_"),
        OLD_SHAPE.replace("EstimationMethod::Imp;", "EstimationMethod::Impmap;"),
        // A chain names both methods and may tune either.
        "fn chain() {\n  opts.methods = vec![EstimationMethod::Impmap, EstimationMethod::Imp];\n  \
         opts.impmap_iterations = 2;\n}\n"
            .to_string(),
        "[fit_options]\n  method = focei, impmap, imp\n  impmap_samples = 40\n".to_string(),
        // *Reading* the other family's knob is not setting it.
        "fn reads() {\n  opts.method = EstimationMethod::Imp;\n  \
         assert_eq!(opts.impmap_iterations, 200);\n  \
         if opts.impmap_samples == 300 {}\n  let _ = FitOptions::impmap_seed;\n}\n"
            .to_string(),
        // A helper that names neither method is not judged.
        "fn quick(opts: &mut FitOptions) {\n  opts.impmap_iterations = 2;\n}\n".to_string(),
        // Setting both families under one method: the right one is set.
        "fn both() {\n  opts.method = EstimationMethod::Imp;\n  opts.imp_iterations = 2;\n  \
         opts.impmap_iterations = 2;\n}\n"
            .to_string(),
    ];
    for src in &excused {
        assert_eq!(p.mismatches_in_rust(src), vec![], "wrongly flagged:\n{src}");
        assert_eq!(p.classify(src), None, "wrongly flagged as one body:\n{src}");
    }
}

/// `Impmap` must not read as `Imp`. Without the `\b` every IMPMAP test names
/// "both" methods and the rule excuses all of them — including a real mirror
/// mismatch — so this is asserted on its own rather than left to the scan.
#[test]
fn impmap_does_not_name_imp() {
    let p = Patterns::new();
    assert!(!p.names_imp.is_match("EstimationMethod::Impmap"));
    assert!(!p.names_imp.is_match("  method = impmap\n"));
    assert!(p.names_imp.is_match("EstimationMethod::Imp;"));
    assert!(p.names_imp.is_match("  method   = imp\n"));
    assert!(!p.sets_imp_knob.is_match("opts.impmap_iterations = 2;"));
    assert!(p.sets_impmap_knob.is_match("opts.impmap_iterations = 2;"));
}

#[test]
fn no_tracked_source_tunes_the_other_methods_knobs() {
    let root = repo_root();
    let p = Patterns::new();
    let this_file = Path::new(file!())
        .file_name()
        .expect("this file has a name");

    let mut offenders = Vec::new();
    let mut judged_rust = 0usize;
    let mut examined_rust = 0usize;
    for path in tracked_files(&root, "rs") {
        // This file carries the offending shapes as fixtures.
        if path.file_name() == Some(this_file) {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("tracked source is readable UTF-8");
        judged_rust += 1;
        // Every knob of either family contains `imp_` or `impmap_`; a file
        // with neither cannot hold a mismatch, and skipping the regexes over
        // it is most of this test's runtime in an unoptimised build.
        if !src.contains("imp_") && !src.contains("impmap_") {
            continue;
        }
        examined_rust += 1;
        for (name, m) in p.mismatches_in_rust(&src) {
            offenders.push(format!("{}: fn {name}: {m:?}", path.display()));
        }
    }
    let mut judged_models = 0usize;
    for path in tracked_files(&root, "ferx") {
        let src = std::fs::read_to_string(&path).expect("tracked model is readable UTF-8");
        judged_models += 1;
        if let Some(m) = p.classify(&src) {
            offenders.push(format!("{}: {m:?}", path.display()));
        }
    }

    // A scan that found nothing because it read nothing is not a pass.
    assert!(
        judged_rust > 100,
        "only {judged_rust} .rs files were scanned"
    );
    // 47 files mention a knob of either family at the time of writing; the
    // pre-filter above must not be what makes the scan quiet.
    assert!(
        examined_rust > 20,
        "only {examined_rust} .rs files got past the knob pre-filter"
    );
    assert!(
        judged_models > 10,
        "only {judged_models} .ferx files were scanned"
    );
    assert!(
        offenders.is_empty(),
        "these set the knobs of the method they do not run, so they run on the defaults \
         (200 iterations) with their seed unapplied — rename `impmap_*` ↔ `imp_*` (#1476):\n  {}",
        offenders.join("\n  ")
    );
}
