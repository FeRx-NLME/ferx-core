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
    for entry in std::fs::read_dir(dir).expect("a source directory must be readable") {
        let path = entry.expect("a readable directory entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Every `src/` tree in the workspace: the engine, plus the members that consume it as an
/// ordinary crate and can reach `OdeSpec::solver_opts` through the public API just as well.
fn source_roots() -> Vec<PathBuf> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut roots = vec![repo.join("src")];
    for entry in std::fs::read_dir(repo.join("crates")).expect("crates/ must be readable") {
        let member = entry
            .expect("a readable directory entry")
            .path()
            .join("src");
        if member.is_dir() {
            roots.push(member);
        }
    }
    roots
}

/// The lines of `src` that are production code, as `(zero-based line number, line)`.
///
/// Inline test modules sit at the end of a file by convention here, so everything from the
/// first `#[cfg(test)]` **attribute** onward is test code — free to assert on the baked field,
/// which is exactly what a test pinning the parser's stamping should do. The attribute has to
/// be matched as a whole line and not with `str::find`: `src/ode/predictions.rs` names
/// `#[cfg(test)]` inside an ordinary comment on line 17, and a substring search cut the scan
/// off there — leaving ~6,000 lines holding half the converted call sites unguarded.
fn production_lines(src: &str, is_test_file: bool) -> Vec<(usize, &str)> {
    if is_test_file {
        // A sibling test file (`<file>_tests.rs`) is test code in its entirety.
        return Vec::new();
    }
    src.lines()
        .enumerate()
        .take_while(|(_, l)| l.trim() != "#[cfg(test)]")
        .collect()
}

#[test]
fn no_production_code_reads_the_baked_solver_opts_field() {
    let roots = source_roots();
    let mut files = Vec::new();
    for root in &roots {
        rust_sources(root, &mut files);
    }
    assert!(files.len() > 100, "the source walk found only {} files, which is not this workspace — the guard would pass vacuously", files.len());

    let mut offenders: Vec<String> = Vec::new();
    let mut scanned_lines = 0usize;
    for path in &files {
        let is_test_file = path
            .file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.ends_with("_tests") || s == "types_test_helpers");
        let src = std::fs::read_to_string(path).expect("a readable source file");
        let production = production_lines(&src, is_test_file);
        scanned_lines += production.len();
        for (n, line) in production {
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
            offenders.push(format!("{}:{}: {t}", path.display(), n + 1));
        }
    }

    assert!(
        scanned_lines > 100_000,
        "only {scanned_lines} lines were scanned as production code, which is a fraction of \
         this workspace — the boundary detection has stopped early again"
    );

    assert!(
        offenders.is_empty(),
        "#1212: these read `OdeSpec::solver_opts` directly, so they integrate at the \
         parse-time options and ignore the ones the fit was given. Use \
         `spec.effective_solver_opts()`:\n  {}",
        offenders.join("\n  ")
    );
}

/// The guard's own failure mode, and the one it shipped with: a file that *mentions* the
/// attribute in prose before its test module starts had everything after that comment treated
/// as test code and never scanned. `src/ode/predictions.rs` is that file, and it holds 10 of
/// the 19 converted call sites.
#[test]
fn the_test_module_boundary_ignores_a_comment_that_names_the_attribute() {
    let src = "//! `MonitorSpec` is named only by the `#[cfg(test)]` wrapper below.\n\
               fn integrate() { let o = spec.solver_opts; }\n\
               #[cfg(test)]\n\
               mod tests {\n\
               let baked = spec.solver_opts;\n\
               }\n";
    let production = production_lines(src, false);
    assert_eq!(
        production.len(),
        2,
        "the prose mention of the attribute cut the scan short: {production:?}"
    );
    assert!(
        production[1].1.contains("let o = spec.solver_opts;"),
        "the line after the prose mention must still be scanned"
    );

    // …while a real attribute still ends the production region, so a test module keeps its
    // licence to read the baked field.
    let production = production_lines(
        "fn f() {}\n#[cfg(test)]\nmod t { spec.solver_opts }\n",
        false,
    );
    assert_eq!(production.len(), 1);

    // And a sibling `_tests.rs` file is test code start to finish.
    assert!(production_lines("spec.solver_opts\n", true).is_empty());
}
