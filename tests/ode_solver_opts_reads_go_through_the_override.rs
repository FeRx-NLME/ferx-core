//! Structural guard for #1212: production code must not read `OdeSpec::solver_opts` directly.
//!
//! The field carries the *parse-time* solver options. A fit's own `ode_reltol` / `ode_method` /
//! … reach the integrator through a fit-scoped override that `OdeSpec::effective_solver_opts`
//! merges over that field, because `fit` takes `&CompiledModel` and cannot restamp the spec. So
//! an integration site that reads the field directly silently runs at the parse-time value —
//! which is the entire bug #1212 fixed, and it fails *silently*: the fit completes, reports
//! success, and returns a number computed at an accuracy nobody asked for.
//!
//! Nineteen call sites were converted at once, and nothing in the type system stops the
//! twentieth from being written the old way. This test is what stops it. If it fires on a line
//! you just wrote, the fix is `spec.effective_solver_opts()`, not an entry in the allowlist.

use std::path::{Path, PathBuf};

/// Reads that are legitimately of the *baked* field rather than the effective options.
///
/// Kept as whole trimmed lines rather than file names, so allowing one line cannot quietly
/// permit a second read in the same file.
const ALLOWED: &[&str] = [
    // The single place the merge itself happens — by definition it reads the baked value.
    "crate::ode::solver::effective_solver_options(self.solver_opts)",
]
.as_slice();

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("src/ must be readable") {
        let path = entry.expect("a readable directory entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_production_code_reads_the_baked_solver_opts_field() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&root, &mut files);
    assert!(files.len() > 100, "the source walk found only {} files, which is not this crate — the guard would pass vacuously", files.len());

    let mut offenders: Vec<String> = Vec::new();
    for path in &files {
        // A sibling test file (`<file>_tests.rs`) is test code in its entirety.
        let is_test_file = path
            .file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.ends_with("_tests") || s == "types_test_helpers");
        let src = std::fs::read_to_string(path).expect("a readable source file");
        // Inline test modules sit at the end of a file by convention here, so everything from
        // the first `#[cfg(test)]` onward is test code — free to assert on the baked field,
        // which is exactly what a test pinning the parser's stamping should do.
        let production = match src.find("#[cfg(test)]") {
            Some(i) if !is_test_file => &src[..i],
            Some(_) | None if is_test_file => "",
            None => &src[..],
            _ => "",
        };
        for (n, line) in production.lines().enumerate() {
            let t = line.trim();
            if !t.contains(".solver_opts") {
                continue;
            }
            // Doc comments and ordinary comments name the field constantly; they run nothing.
            if t.starts_with("//") {
                continue;
            }
            // `ode.solver_opts.reltol = opts.ode_reltol;` — the parse-time stamping, which is
            // the writer this whole mechanism merges on top of.
            if t.contains(" = ")
                && t.split(" = ")
                    .next()
                    .is_some_and(|l| l.contains(".solver_opts"))
            {
                continue;
            }
            if ALLOWED.contains(&t) {
                continue;
            }
            offenders.push(format!(
                "{}:{}: {t}",
                path.strip_prefix(&root).unwrap_or(path).display(),
                n + 1
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "#1212: these read `OdeSpec::solver_opts` directly, so they integrate at the \
         parse-time options and ignore the ones the fit was given. Use \
         `spec.effective_solver_opts()`:\n  {}",
        offenders.join("\n  ")
    );
}
