//! Guard for #1824: engine code reads the LOQ-censoring method through
//! `CompiledModel::bloq_in_force`, never the `bloq_method` field.
//!
//! `fit()` honours a per-call `FitOptions::bloq_method` by arming it on the thread
//! (and the fit pool) and having every reader ask `bloq_in_force`. A new reader of
//! the field compiles, runs and passes every fixture that stamps the field — and
//! silently scores a direct `fit()` caller's override under the model's method,
//! which is the defect #1824 fixed at ~45 sites. This makes that a red test.
//!
//! Scope: production code under `src/` — `*_tests.rs` siblings, `tests/` folders,
//! `test_helpers` files and the bodies of inline `#[cfg(test)] mod … { }` modules
//! are skipped. A read is `model.bloq_method` (any receiver
//! ending in `model`, so `self.model.` and `twin_model.` too) not followed by an
//! assignment `=`. `FitOptions::bloq_method` and `FitResult::bloq_method` share the
//! field name but not the receiver, and the accessor itself reads `self.bloq_method`;
//! a field read through a receiver not named `…model` is the remaining gap.
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
/// match the file). The body ends where its braces balance, so production code
/// *after* a test module is still scanned (review r1 #3: six files have some). A
/// `#[path = …] mod` sibling declaration has no `{` and blanks nothing. Braces are
/// counted per character, so one inside a string or char literal in a test module
/// could end it early or late — the remaining gap; no such module exists today.
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

/// Every `model.bloq_method` read in `src` as `(line number, line)`.
fn field_reads(src: &str) -> Vec<(usize, String)> {
    const NEEDLE: &str = "model.bloq_method";
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        let mut rest = line;
        while let Some(at) = rest.find(NEEDLE) {
            let after = &rest[at + NEEDLE.len()..];
            let next = after.chars().next();
            let is_ident_tail = next.is_some_and(|c| c.is_alphanumeric() || c == '_');
            let a = after.trim_start();
            let is_assignment = a.starts_with('=') && !a.starts_with("==");
            if !is_ident_tail && !is_assignment {
                out.push((i + 1, line.trim().to_string()));
            }
            rest = after;
        }
    }
    out
}

#[test]
fn the_classifier_sees_reads_and_skips_assignments_and_test_modules() {
    // The scanner's own fixture, so a scanner that stopped matching anything cannot
    // pass the guard below by finding nothing.
    let src = "fn a(model: &M) -> bool { matches!(model.bloq_method, B::M3) }\n\
               fn b(model: &mut M) { model.bloq_method = B::M3; }\n\
               fn c(model: &M) -> bool { model.bloq_method == B::M3 }\n\
               fn d(model: &M) -> B { model.bloq_in_force() }\n\
               fn e(o: &O) -> Option<B> { o.bloq_method }\n\
               #[cfg(test)]\n#[path = \"x_tests.rs\"]\nmod sib;\n\
               fn f(model: &M) -> B { model.bloq_method }\n\
               #[cfg(test)]\nmod tests {\n  fn g(model: &M) -> B { model.bloq_method }\n  \
               mod inner {\n    fn h(model: &M) -> B { model.bloq_method }\n  }\n}\n\
               fn k(model: &M) -> B { model.bloq_method }\n";
    let lines: Vec<usize> = field_reads(&production_part(src))
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    // 1, 3, 9: before any test module. 17: production code *after* an inline test
    // module (with a nested module inside it), which must still be scanned.
    assert_eq!(
        lines,
        vec![1, 3, 9, 17],
        "reads on lines 1, 3, 9 and 17 only"
    );
}

#[test]
fn no_production_code_reads_the_bloq_method_field() {
    let mut sources = Vec::new();
    rust_sources(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut sources,
    );
    assert!(
        sources.len() > 50,
        "found only {} sources under src/",
        sources.len()
    );

    let mut offenders = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("source file is valid UTF-8");
        for (n, line) in field_reads(&production_part(&src)) {
            offenders.push(format!("{}:{n}: {line}", path.display()));
        }
    }
    assert!(
        offenders.is_empty(),
        "production code reads `CompiledModel::bloq_method` directly, which ignores a \
         `FitOptions::bloq_method` override on a direct `fit()` (#1824). Read \
         `model.bloq_in_force()` instead:\n{}",
        offenders.join("\n")
    );
}
