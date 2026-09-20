//! **An IMP test that tunes `impmap_*`, or an IMPMAP test that tunes `imp_*`,
//! tunes nothing** (#1476).
//!
//! `run_imp` reads `imp_iterations` / `imp_samples` / `imp_seed`; `run_impmap`
//! reads the `impmap_*` family. The two families are separate `FitOptions`
//! fields with identical suffixes, so setting the wrong one compiles, runs and
//! passes — on the *defaults*, 200 iterations × 1000 samples, with the seed the
//! author wrote never applied. Three unit tests in `src/estimation/impmap.rs`
//! had that shape. Measured serially under `cargo llvm-cov --profile ci-cov`
//! they cost 202 s where the intended 2 × 40 costs 0.12 s, and because a short
//! `fit()` queues behind a long one on the shared fit pool they were the
//! five-minute stall in the lib run of both per-PR coverage jobs.
//!
//! Nothing failed, which is the point: a test that is merely slow is invisible
//! until someone reads a CI log with timestamps. So the shape is pinned here.
//!
//! Two arms, because the two spellings have different owners:
//!
//! * **A `.ferx` model is judged by the engine.** A key written in
//!   `[fit_options]` that no stage of the method chain reads already gets
//!   *"fit option `impmap_iterations` is not used by method `IMP`"* from
//!   [`FitOptions::unsupported_keys_warnings`]. That path knows the parser's
//!   method aliases and the key table, so this file parses each tracked model
//!   and asks it, rather than re-deriving either.
//! * **Rust source is judged here**, function by function, because a field
//!   assignment leaves `user_set_keys` empty and the engine cannot see it. A
//!   body that names exactly one of the two methods and assigns **any** knob
//!   only the other one reads is a mismatch. A body naming both — a
//!   `focei → impmap → imp` chain — may set either family, and a helper that
//!   names neither is not judged.
//!
//! Which knob belongs to which method is read from [`method_specific_keys`],
//! not listed here: `imp_defensive_alpha` is one field that *both* methods
//! read, and a hand-written family would flag a correct IMPMAP test for
//! setting it.

use ferx_core::parser::model_parser::{parse_full_model, parse_full_model_file};
use ferx_core::{method_specific_keys, EstimationMethod};
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

/// The `[fit_options]` spellings of each method, as the parser accepts them.
/// A copy of `parse_method_token`'s arms, which is private — so
/// `the_method_aliases_are_the_parsers` pins the copy against the parser.
const IMP_TOKENS: [&str; 3] = ["imp", "importance_sampling", "importance-sampling"];
const IMPMAP_TOKENS: [&str; 3] = [
    "impmap",
    "importance_sampling_map",
    "importance-sampling-map",
];

/// Knobs that `own` reads and `other` does not, within the `imp_` / `impmap_`
/// families. A knob both methods read belongs to neither side of the rule.
fn exclusive_knobs(own: EstimationMethod, other: EstimationMethod) -> Vec<&'static str> {
    let others = method_specific_keys(other);
    method_specific_keys(own)
        .iter()
        .copied()
        .filter(|k| k.starts_with("imp_") || k.starts_with("impmap_"))
        .filter(|k| !others.contains(k))
        .collect()
}

/// Which way a body is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mismatch {
    /// Names IMP only, assigns a knob only IMPMAP reads.
    ImpWithImpmapKnobs,
    /// Names IMPMAP only, assigns a knob only IMP reads.
    ImpmapWithImpKnobs,
}

struct Patterns {
    names_imp: Regex,
    names_impmap: Regex,
    method_line: Regex,
    sets_imp_only_knob: Regex,
    sets_impmap_only_knob: Regex,
    fn_start: Regex,
}

impl Patterns {
    fn new() -> Self {
        // A knob is *assigned* by `name =` (not `==`) or, in a struct literal,
        // `name:`. A path separator needs no excusing — `FitOptions::imp_seed`
        // has its `::` *before* the name, and nothing follows a field name with
        // one. The leading class stands in for a lookbehind, which the `regex`
        // crate does not have: it stops `imp_` matching inside a longer
        // identifier. The trailing `\s*[=:]` does the same at the other end,
        // since the alternation is whole key names.
        let assigns = |knobs: &[&str]| {
            assert!(!knobs.is_empty(), "the production key table lost a family");
            Regex::new(&format!(
                r"(?m)(?:^|[^A-Za-z0-9_])(?:{})\s*(?:=(?:[^=]|$)|:)",
                knobs.join("|")
            ))
            .expect("knob pattern compiles")
        };
        Patterns {
            // `\b` after `Imp` is what keeps `EstimationMethod::Impmap` out of
            // the IMP pattern: `p`→`m` is not a word boundary. The same holds
            // for `run_imp(` against `run_impmap(`. A test that calls the
            // estimator directly never writes `EstimationMethod::` at all.
            names_imp: Regex::new(r"EstimationMethod::Imp\b|\brun_imp\s*\(").expect("compiles"),
            names_impmap: Regex::new(r"EstimationMethod::Impmap\b|\brun_impmap\s*\(")
                .expect("compiles"),
            // A model string inside a Rust test: the right-hand side of a
            // `method =` / `methods =` line, split into tokens below.
            method_line: Regex::new(r"(?m)^\s*methods?\s*=([^\n#]*)").expect("compiles"),
            sets_imp_only_knob: assigns(&exclusive_knobs(
                EstimationMethod::Imp,
                EstimationMethod::Impmap,
            )),
            sets_impmap_only_knob: assigns(&exclusive_knobs(
                EstimationMethod::Impmap,
                EstimationMethod::Imp,
            )),
            fn_start: Regex::new(r"(?m)^\s*(?:pub(?:\([a-z]+\))?\s+)?fn\s+(\w+)")
                .expect("compiles"),
        }
    }

    /// Whether the body names IMP, and whether it names IMPMAP.
    fn names(&self, body: &str) -> (bool, bool) {
        let mut imp = self.names_imp.is_match(body);
        let mut impmap = self.names_impmap.is_match(body);
        // A model written as a one-line Rust string has its line breaks as the
        // two characters `\n`; unescape them so `method =` starts a line.
        let unescaped = body.replace("\\n", "\n");
        for line in self.method_line.captures_iter(&unescaped) {
            // Tokens are compared whole: `importance-sampling` is a prefix of
            // `importance-sampling-map`, and `-` is a word boundary.
            for token in line[1].split(|c: char| c == ',' || c == '[' || c == ']' || c == '"') {
                let token = token.trim().to_ascii_lowercase();
                imp |= IMP_TOKENS.contains(&token.as_str());
                impmap |= IMPMAP_TOKENS.contains(&token.as_str());
            }
        }
        (imp, impmap)
    }

    /// Judge one body of text.
    fn classify(&self, body: &str) -> Option<Mismatch> {
        match self.names(body) {
            (true, false) if self.sets_impmap_only_knob.is_match(body) => {
                Some(Mismatch::ImpWithImpmapKnobs)
            }
            (false, true) if self.sets_imp_only_knob.is_match(body) => {
                Some(Mismatch::ImpmapWithImpKnobs)
            }
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

    /// How many function bodies in `src` name IMP or IMPMAP at all — the class
    /// the rule can say anything about.
    fn judged_bodies(&self, src: &str) -> usize {
        let starts: Vec<usize> = self.fn_start.find_iter(src).map(|m| m.start()).collect();
        starts
            .iter()
            .enumerate()
            .filter(|(i, start)| {
                let end = starts.get(i + 1).copied().unwrap_or(src.len());
                let (imp, impmap) = self.names(&src[**start..end]);
                imp || impmap
            })
            .count()
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

fn flagged(name: &str, m: Mismatch) -> Vec<(String, Mismatch)> {
    vec![(name.to_string(), m)]
}

#[test]
fn the_classifier_flags_the_shape_that_was_found() {
    let p = Patterns::new();
    assert_eq!(
        p.mismatches_in_rust(OLD_SHAPE),
        flagged("imp_shared_anchor", Mismatch::ImpWithImpmapKnobs)
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
        flagged("impmap_case", Mismatch::ImpmapWithImpKnobs)
    );
}

/// The smallest edit that undoes #1476 is **one line**: `imp_iterations` back
/// to `impmap_iterations`, leaving `imp_samples` and `imp_seed` correct. That
/// alone restores 200 iterations. A rule that excuses a body for setting *some*
/// knob of its own family passes it (review of #1478, finding 1), so the rule
/// is "any knob of the other method", and this is the fixture that says so.
#[test]
fn one_wrong_knob_among_right_ones_is_still_a_mismatch() {
    let p = Patterns::new();
    let one_line_revert = OLD_SHAPE
        .replace("opts.impmap_samples", "opts.imp_samples")
        .replace("opts.impmap_seed", "opts.imp_seed");
    assert!(one_line_revert.contains("opts.impmap_iterations = 2;"));
    assert_eq!(
        p.mismatches_in_rust(&one_line_revert),
        flagged("imp_shared_anchor", Mismatch::ImpWithImpmapKnobs)
    );
}

/// A test that calls `run_imp` / `run_impmap` directly never writes
/// `EstimationMethod::` — five tests in `src/estimation/impmap.rs` are that
/// shape — so the call itself has to name the method (finding 2).
#[test]
fn a_direct_estimator_call_names_its_method() {
    let p = Patterns::new();
    let direct = "
    fn imp_mixture_short_run() {
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 2;
        let r = run_imp(&model, &pop, &init, None, &opts);
    }
";
    assert_eq!(
        p.mismatches_in_rust(direct),
        flagged("imp_mixture_short_run", Mismatch::ImpWithImpmapKnobs)
    );
    let mirror = direct
        .replace("impmap_iterations", "imp_iterations")
        .replace("run_imp(", "run_impmap (");
    assert_eq!(
        p.mismatches_in_rust(&mirror),
        flagged("imp_mixture_short_run", Mismatch::ImpmapWithImpKnobs)
    );
    // `run_impmap(` must not read as `run_imp(`, or every direct IMPMAP caller
    // names both methods and is excused.
    assert_eq!(p.names("run_impmap(&m, &p, &i, None, &o)"), (false, true));
    assert_eq!(
        p.names("crate::estimation::impmap::run_imp(&m)"),
        (true, false)
    );
}

/// `imp_defensive_alpha` is one field, read by both estimators and listed
/// under both methods in the production key table. A correct IMPMAP test that
/// sets it must not be told to rename it to a field that does not exist
/// (finding 3).
#[test]
fn a_knob_both_methods_read_is_nobodys_mismatch() {
    assert!(method_specific_keys(EstimationMethod::Imp).contains(&"imp_defensive_alpha"));
    assert!(method_specific_keys(EstimationMethod::Impmap).contains(&"imp_defensive_alpha"));

    let p = Patterns::new();
    let correct_impmap = "
    fn impmap_defensive() {
        opts.method = EstimationMethod::Impmap;
        opts.imp_defensive_alpha = 0.1;
    }
";
    assert_eq!(p.mismatches_in_rust(correct_impmap), vec![]);
    // The same body with a knob only IMP reads is still caught.
    let wrong = correct_impmap.replace("imp_defensive_alpha = 0.1", "imp_seed = Some(1)");
    assert_eq!(
        p.mismatches_in_rust(&wrong),
        flagged("impmap_defensive", Mismatch::ImpmapWithImpKnobs)
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
        flagged("imp_shared_anchor", Mismatch::ImpWithImpmapKnobs)
    );
}

/// A model string inside a Rust test is judged by its `method =` line, under
/// every spelling the parser takes.
#[test]
fn the_classifier_flags_a_model_string_under_every_alias() {
    let p = Patterns::new();
    for token in IMP_TOKENS {
        let model = format!("[fit_options]\n  method = {token}\n  impmap_iterations = 5\n");
        assert_eq!(
            p.classify(&model),
            Some(Mismatch::ImpWithImpmapKnobs),
            "{token}"
        );
    }
    for token in IMPMAP_TOKENS {
        let model = format!("[fit_options]\n  method = {token}\n  imp_samples = 50\n");
        assert_eq!(
            p.classify(&model),
            Some(Mismatch::ImpmapWithImpKnobs),
            "{token}"
        );
    }
    // As it sits in Rust *source*: one line, with `\n` as two characters.
    let as_source = r#"fn t() { let src = "[fit_options]\n  method = imp\n  impmap_seed = 1\n"; }"#;
    assert!(!as_source.contains('\n'), "premise: no real line break");
    assert_eq!(
        p.mismatches_in_rust(as_source),
        flagged("t", Mismatch::ImpWithImpmapKnobs)
    );
}

/// The token lists above are a copy of a private parser function. Each entry
/// must parse to the method it is filed under, so a renamed or added alias
/// fails here instead of going unjudged.
#[test]
fn the_method_aliases_are_the_parsers() {
    let model_with = |token: &str| {
        format!(
            "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
             omega ETA_CL ~ 0.09\n  sigma EPS ~ 0.04\n\n[individual_parameters]\n  \
             CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\n[structural_model]\n  \
             pk one_cpt_iv(cl=CL, v=V)\n\n[error_model]\n  DV ~ proportional(EPS)\n\n\
             [fit_options]\n  method = {token}\n"
        )
    };
    for (tokens, want) in [
        (IMP_TOKENS, EstimationMethod::Imp),
        (IMPMAP_TOKENS, EstimationMethod::Impmap),
    ] {
        for token in tokens {
            let parsed = parse_full_model(&model_with(token))
                .unwrap_or_else(|e| panic!("`method = {token}` must parse: {e}"));
            assert_eq!(
                parsed.fit_options.method_chain(),
                vec![want],
                "`{token}` is filed under {want:?}"
            );
        }
    }
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
        "[fit_options]\n  method = [importance-sampling-map, importance-sampling]\n  \
         imp_samples = 40\n"
            .to_string(),
        // *Reading* the other family's knob is not setting it.
        "fn reads() {\n  opts.method = EstimationMethod::Imp;\n  \
         assert_eq!(opts.impmap_iterations, 200);\n  \
         if opts.impmap_samples == 300 {}\n  let _ = FitOptions::impmap_seed;\n}\n"
            .to_string(),
        // A helper that names neither method is not judged.
        "fn quick(opts: &mut FitOptions) {\n  opts.impmap_iterations = 2;\n}\n".to_string(),
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
    assert_eq!(p.names("EstimationMethod::Impmap"), (false, true));
    assert_eq!(p.names("  method = impmap\n"), (false, true));
    assert_eq!(
        p.names("  method = importance-sampling-map\n"),
        (false, true)
    );
    assert_eq!(p.names("EstimationMethod::Imp;"), (true, false));
    assert_eq!(p.names("  method   = imp\n"), (true, false));
    assert_eq!(
        p.names("  method = IMP  # comment naming impmap\n"),
        (true, false)
    );
    assert!(!p.sets_imp_only_knob.is_match("opts.impmap_iterations = 2;"));
    assert!(p
        .sets_impmap_only_knob
        .is_match("opts.impmap_iterations = 2;"));
}

/// A knob is a whole identifier. Read as a suffix, `simp_seed = …` counts as an
/// `imp_seed` assignment and flags an IMPMAP body that sets no IMP knob.
#[test]
fn a_longer_identifier_is_not_a_knob() {
    let p = Patterns::new();
    assert!(!p.sets_imp_only_knob.is_match("opts.simp_seed = 1;"));
    assert!(!p.sets_imp_only_knob.is_match("opts.imp_seed_base = 1;"));
    assert!(!p.sets_impmap_only_knob.is_match("let preimpmap_seed = 1;"));
    assert!(p.sets_imp_only_knob.is_match("imp_samples = 40"));

    let lookalike =
        "fn f() {\n  opts.method = EstimationMethod::Impmap;\n  opts.simp_seed = 1;\n}\n";
    assert_eq!(p.mismatches_in_rust(lookalike), vec![]);
}

#[test]
fn no_tracked_rust_source_tunes_the_other_methods_knobs() {
    let root = repo_root();
    let p = Patterns::new();
    let this_file = Path::new(file!())
        .file_name()
        .expect("this file has a name");

    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    let mut judged = 0usize;
    for path in tracked_files(&root, "rs") {
        // This file carries the offending shapes as fixtures.
        if path.file_name() == Some(this_file) {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("tracked source is readable UTF-8");
        scanned += 1;
        // Every knob of either family contains `imp_` or `impmap_`; a file
        // with neither cannot hold a mismatch, and skipping the regexes over
        // it is most of this test's runtime in an unoptimised build.
        if !src.contains("imp_") && !src.contains("impmap_") {
            continue;
        }
        judged += p.judged_bodies(&src);
        for (name, m) in p.mismatches_in_rust(&src) {
            offenders.push(format!("{}: fn {name}: {m:?}", path.display()));
        }
    }

    // A scan that found nothing because it judged nothing is not a pass: count
    // the bodies in the class the rule speaks about, not the files read.
    eprintln!("rust arm: {scanned} files scanned, {judged} bodies name IMP or IMPMAP");
    assert!(scanned > 100, "only {scanned} .rs files were scanned");
    // Measured when written: 546 files, 110 judged bodies. The floor is about
    // half of that — it exists to catch a scan that went quiet, not to count.
    assert!(
        judged > 50,
        "only {judged} function bodies name IMP or IMPMAP — the pre-filter or the name \
         patterns have gone quiet"
    );
    assert!(
        offenders.is_empty(),
        "these name one of IMP / IMPMAP and set a knob only the other reads, so that knob does \
         nothing — the run uses the default (200 iterations × 1000 samples, default seed). \
         Rename `impmap_*` ↔ `imp_*` (#1476):\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn no_tracked_model_writes_a_key_its_method_does_not_read() {
    let root = repo_root();
    let mut offenders = Vec::new();
    let mut parsed = 0usize;
    let mut judged = 0usize;
    for path in tracked_files(&root, "ferx") {
        // Fixtures that are *meant* not to parse exist; they carry no method.
        let Ok(model) = parse_full_model_file(&path) else {
            continue;
        };
        parsed += 1;
        let chain = model.fit_options.method_chain();
        if !chain.contains(&EstimationMethod::Imp) && !chain.contains(&EstimationMethod::Impmap) {
            continue;
        }
        judged += 1;
        for w in model.fit_options.unsupported_keys_warnings() {
            if w.contains("`imp_") || w.contains("`impmap_") {
                offenders.push(format!("{}: {w}", path.display()));
            }
        }
    }
    eprintln!("model arm: {parsed} models parsed, {judged} run IMP or IMPMAP");
    // Measured when written: 164 of 187 tracked models parse stand-alone, one
    // of which (`examples/warfarin_impmap.ferx`, under the alias
    // `importance_sampling_map`) runs either method.
    assert!(parsed > 100, "only {parsed} .ferx models parsed");
    assert!(
        judged >= 1,
        "no tracked model runs IMP or IMPMAP, so this arm judged nothing \
         (`examples/warfarin_impmap.ferx` is expected to)"
    );
    assert!(
        offenders.is_empty(),
        "the engine reports these `[fit_options]` keys as unread by the model's method:\n  {}",
        offenders.join("\n  ")
    );
}
