//! #1775: every place a failure leaves the engine **without** a `ferx check` code is
//! spelled out and listed here. `EngineError` has no `From<String>`, so a `?` on a
//! `String` helper inside an `EngineError` function no longer compiles; the tests below
//! keep it that way.

use super::EngineError;
use std::collections::BTreeMap;
use std::path::Path;

/// Every site in `src/` (outside test code) that turns a failure into an error with no
/// code: `(file, enclosing fn, number of sites in it, why it has no code)`.
///
/// Two literal spellings are counted, so the scan has no false positives:
/// `EngineError::uncoded` and `.map_err(|d| d.message)` (a `Diagnostic` dropped to its
/// text). A new site fails `every_uncoded_site_is_listed` until it is listed here; a
/// listed site that is gone fails it too, so the list cannot go stale.
///
/// Before adding a row, check `ferx check`'s codes: a refusal some helper already codes
/// goes through that helper's `Diagnostic`, not through `uncoded`.
const UNCODED_SITES: &[(&str, &str, usize, &str)] = &[
    (
        "src/api/adaptive.rs",
        "run_adaptive_population",
        3,
        "runtime: a subject's adaptive run, or its snapshot / frozen-replay verification, failed",
    ),
    (
        "src/api/simulate.rs",
        "simulate_with_options_diag",
        1,
        "runtime: a subject's posthoc EBE is non-finite",
    ),
    (
        "src/api/simulate.rs",
        "simulate_with_uncertainty_diag",
        1,
        "draw_parameter_samples: sampler failure or covariance shape; its logit-θ bound refusal has no code yet, #2044",
    ),
    (
        "src/estimation/run_covariance.rs",
        "run_covariance",
        1,
        "with_fit_scope: invalid ode_reltol / ode_abstol / ode_max_steps on the caller's FitOptions; no code yet, #2044",
    ),
    (
        "src/estimation/run_sir.rs",
        "run_sir",
        1,
        "with_fit_scope: invalid ODE solver options on the caller's FitOptions; no code yet, #2044",
    ),
    (
        "src/estimation/sir.rs",
        "run_sir_core_coded",
        1,
        "with_fit_scope: invalid ODE solver options on the caller's FitOptions; no code yet, #2044",
    ),
    (
        "src/estimation/sir.rs",
        "run_sir_in_box",
        6,
        "[mixture] vs per-class Ω/Σ mismatch on hand-built parameters and the logit-θ bound refusal (no code yet, #2044); \
         runtime: no positive proposal direction, every weight invalid, sampler build",
    ),
    (
        "src/io/fitrx.rs",
        "read_bundled_population",
        1,
        "the bundled-data replay only warns: this text feeds a warning, never an EngineError",
    ),
    (
        "src/api/run.rs",
        "read_population_with",
        1,
        "the String twin of read_population_with_diag, kept for String callers (#1772)",
    ),
];

/// The spellings counted as an uncoded site, built with `concat!` so this file does not
/// match itself.
const PATTERNS: &[&str] = &[
    concat!("EngineError::", "uncoded"),
    concat!(".map_err(|d| d", ".message)"),
];

/// The name of the nearest `fn` at or above line `i`.
fn enclosing_fn(lines: &[&str], i: usize) -> String {
    for line in lines[..=i].iter().rev() {
        let Some(pos) = line.find("fn ") else {
            continue;
        };
        let before = &line[..pos];
        if !(before.is_empty() || before.ends_with(' ') || before.ends_with('(')) {
            continue;
        }
        let name: String = line[pos + 3..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            return name;
        }
    }
    String::from("<top level>")
}

/// `(file, fn) → count` of the uncoded sites in the non-test code under `root/src`.
fn scan(root: &Path) -> BTreeMap<(String, String), usize> {
    let mut found = BTreeMap::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read a src dir") {
            let path = entry.expect("a dir entry").path();
            if path.is_dir() {
                // `src/api/tests/` holds api's sibling test modules.
                if path.file_name().is_some_and(|n| n != "tests") {
                    stack.push(path);
                }
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if !name.ends_with(".rs")
                || name.ends_with("_tests.rs")
                || name == "diagnostics_uncoded_sites.rs"
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read a source file");
            // An inline test module sits at the end of its file.
            let body = text.split("\n#[cfg(test)]\nmod ").next().unwrap();
            let lines: Vec<&str> = body.lines().collect();
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            for (i, line) in lines.iter().enumerate() {
                let n: usize = PATTERNS.iter().map(|p| line.matches(p).count()).sum();
                if n > 0 {
                    *found
                        .entry((rel.clone(), enclosing_fn(&lines, i)))
                        .or_insert(0) += n;
                }
            }
        }
    }
    found
}

/// Mutations — add an unlisted `EngineError::uncoded` in `src/` → red ("not listed");
/// delete a listed site, or add one to a listed fn → red ("listed … but found …").
#[test]
fn every_uncoded_site_is_listed() {
    let found = scan(Path::new(env!("CARGO_MANIFEST_DIR")));
    let listed: BTreeMap<(String, String), usize> = UNCODED_SITES
        .iter()
        .map(|(f, func, n, _)| ((f.to_string(), func.to_string()), *n))
        .collect();
    assert_eq!(listed.len(), UNCODED_SITES.len(), "a row is listed twice");
    let mut problems = Vec::new();
    for ((file, func), n) in &found {
        match listed.get(&(file.clone(), func.clone())) {
            None => problems.push(format!(
                "{file}::{func}: {n} uncoded site(s) not listed in UNCODED_SITES — \
                 give the failure a code, or list it with a reason"
            )),
            Some(m) if m != n => {
                problems.push(format!("{file}::{func}: listed {m} site(s) but found {n}"))
            }
            Some(_) => {}
        }
    }
    for ((file, func), m) in &listed {
        if !found.contains_key(&(file.clone(), func.clone())) {
            problems.push(format!(
                "{file}::{func}: listed {m} site(s) but found none — remove the stale row"
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The scan sees a planted site and skips test code, so a green
/// `every_uncoded_site_is_listed` is not a scan that sees nothing.
#[test]
fn the_scan_sees_a_planted_site_and_skips_test_code() {
    let dir = std::env::temp_dir().join(format!("ferx-uncoded-scan-{}", std::process::id()));
    let src = dir.join("src");
    std::fs::create_dir_all(src.join("tests")).unwrap();
    let site = PATTERNS[0];
    let drop = PATTERNS[1];
    std::fs::write(
        src.join("a.rs"),
        format!(
            "pub fn leaks() -> R {{\n    Err({site}\"x\"))\n}}\n\
             pub(crate) fn drops() -> R {{\n    f(){drop}?;\n    g(){drop}\n}}\n\
             \n#[cfg(test)]\nmod tests {{\n    fn t() {{ {site}\"y\"); }}\n}}\n"
        ),
    )
    .unwrap();
    std::fs::write(
        src.join("a_tests.rs"),
        format!("fn t() {{ {site}\"z\"); }}\n"),
    )
    .unwrap();
    std::fs::write(
        src.join("tests").join("b.rs"),
        format!("fn t() {{ {site}\"z\"); }}\n"),
    )
    .unwrap();
    let found = scan(&dir);
    std::fs::remove_dir_all(&dir).ok();
    let want: BTreeMap<(String, String), usize> = [
        (("src/a.rs".to_string(), "leaks".to_string()), 1),
        (("src/a.rs".to_string(), "drops".to_string()), 2),
    ]
    .into();
    assert_eq!(found, want);
}

/// Autoref specialization: `Yes` applies to `Probe<T, S>` only when `T: From<S>`;
/// otherwise method resolution falls back to `No` on `&Probe<T, S>`.
struct Probe<T, S>(std::marker::PhantomData<(T, S)>);

impl<T, S> Probe<T, S> {
    fn new() -> Self {
        Probe(std::marker::PhantomData)
    }
}

trait Yes {
    fn converts(&self) -> bool {
        true
    }
}
impl<T: From<S>, S> Yes for Probe<T, S> {}

trait No {
    fn converts(&self) -> bool {
        false
    }
}
impl<T, S> No for &Probe<T, S> {}

/// The planted leak. Mutation — re-add `impl From<String> for EngineError` (or
/// `From<&str>`) → red, on every PR (a `compile_fail` doctest would run only nightly).
#[test]
fn engine_error_has_no_from_string() {
    // The probe can say `true`, so its `false` below is a measurement.
    assert!((&Probe::<String, &str>::new()).converts());
    assert!(
        !(&Probe::<EngineError, String>::new()).converts(),
        "EngineError: From<String> is back — a `?` on a String helper would drop a \
         ferx check code silently (#1775); use EngineError::uncoded or the helper's Diagnostic"
    );
    assert!(
        !(&Probe::<EngineError, &str>::new()).converts(),
        "EngineError: From<&str> is back (#1775); use EngineError::uncoded"
    );
}
