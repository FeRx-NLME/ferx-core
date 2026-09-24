//! Tier-1 tests for #1382: every standard error, eigenvalue and condition number
//! a fit publishes says which estimator produced it.
//!
//! **The gap.** `FitResult` carried `cov_condition_number` and `cov_eigenvalues`
//! and printed standard errors, but recorded nothing about which of `R⁻¹`, `S⁻¹`
//! or the `R⁻¹SR⁻¹` sandwich they came out of; `covariance_method` lived on
//! `FitOptions`, which a fit object, a `{model}-fit.yaml` or a `.fitrx` bundle
//! does not carry. The reporter measured the same fit at condition number
//! **1.42e8** under the sandwich and **3.68e5** under `covariance_method = s` —
//! both correct for their estimator, and indistinguishable in the output. That
//! matters the moment the figure is compared against NONMEM, whose `$COVARIANCE`
//! default is `RSR` and not `R`.
//!
//! **Why the label is carried, not re-derived.** `FitOptions::covariance_method`
//! is what was *asked for*. Above `COV_HESSIAN_MAX_DIM` free parameters a
//! defaulted `r` is routed onto the cross-product (#1064), and `fit_inner` sees a
//! per-stage `FitOptions` clone rather than the one the covariance step ran under.
//! `estimation::covariance::published_covariance_method` is the single owner, and
//! its two unit tests (`the_published_label_names_the_routed_estimator_not_the_requested_one`,
//! `a_step_that_produced_no_matrix_publishes_no_estimator`) pin the rule itself.
//!
//! **Tier boundary** (#1382 review). Nothing here calls `fit()`: this file holds
//! the structural scan over the publishing sites and the warning-payload check,
//! both of which run against source text or a typed struct. The end-to-end cases
//! that do call `fit()` / `run_covariance` live in `tests/covariance_method_label.rs`,
//! which is Tier 2 by AGENTS.md's rule — public API, returning after a single
//! outer iteration.
use crate::types::CovarianceMethod;

/// **G5 — the pairing, structurally: no covariance matrix is written anywhere
/// without its estimator.**
///
/// The behavioural tests in `tests/covariance_method_label.rs` cover the paths a
/// fixture can reach. This covers the ones it cannot: `OuterResult` is **public
/// API with public fields**, and
/// `ferx-tools` calls `optimize_population` / `run_foce_gn` / `run_imp` /
/// `run_impmap` / `run_bayes` directly without going near `fit()`, so a new
/// estimator exit that sets `covariance_matrix` and forgets `covariance_method`
/// would reproduce the reported gap on a surface no test here fits through.
///
/// A **site** is one of:
///
/// * an `OuterResult { … }` / `FitResult { … }` / `FitWire { … }` literal (not the
///   `struct` definition) carrying a `covariance_matrix` field at its own brace
///   depth, or
/// * a `….covariance_matrix = …` field assignment, which is how `run_covariance`
///   publishes onto a cloned result.
///
/// Every site must carry the matching `covariance_method`, and the per-file count
/// is pinned exactly — a removed site fails as loudly as an unlabelled new one.
/// `FitOptions::covariance_method` is the *request* and is not a site: no
/// `covariance_matrix` accompanies it. Test modules are excluded (`_tests.rs`
/// siblings, `tests/` directories and the inline `mod …tests {` block) since they
/// build results as scaffolding.
///
/// `CovStepOutcome` is deliberately **not** a scanned literal: its fields are
/// `matrix` and `method`, so covering it would mean a second spelling for one
/// struct, and the invariant there is already owned by
/// `published_covariance_method` and asserted directly by its two unit tests.
///
/// What is written is pinned by G1 / G3 in `tests/covariance_method_label.rs`;
/// this pins only that *something* is written at every site — the half a fixture
/// cannot reach.
///
/// Mutation (run): delete `covariance_method` from any one estimator exit → that
/// site is unlabelled and this fires naming the file. Delete a whole exit's
/// covariance block → the count drops and this fires.
#[test]
fn every_published_covariance_matrix_is_written_with_its_estimator() {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("readable directory") {
            let path = entry.expect("readable dir entry").path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    /// What this line writes to `field`, if anything: `Some(value)` for a
    /// struct-literal field (`name: value,`), a shorthand (`name,` — the value is
    /// the binding of the same name) or an assignment (`x.name = value;`).
    ///
    /// The value matters, not just the presence: a site that writes
    /// `covariance_method: None` beside a real matrix is exactly the defect, and a
    /// presence-only check passes on it. (Measured: it did, until this returned
    /// the value.)
    fn writes<'a>(code: &'a str, field: &str) -> Option<&'a str> {
        let t = code.trim();
        if let Some(rest) = t.strip_prefix(field) {
            if let Some(v) = rest.strip_prefix(':') {
                return Some(v.trim().trim_end_matches(','));
            }
            if rest.starts_with(',') {
                return Some(field_shorthand());
            }
        }
        if let Some(pos) = t.find(&format!(".{field} =")) {
            let v = &t[pos + field.len() + 3..];
            return Some(v.trim().trim_end_matches(';'));
        }
        None
    }

    /// Sentinel for a shorthand field: a binding, never the literal `None`.
    fn field_shorthand() -> &'static str {
        "<shorthand>"
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources(&root.join("src"), &mut files);
    files.sort();
    assert!(
        files.len() > 100,
        "the scan found only {} files under src/ — it is measuring the walk, not the code",
        files.len()
    );

    // file -> (sites, of which labelled)
    let mut found: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for file in &files {
        let rel = file
            .strip_prefix(&root)
            .expect("under the manifest dir")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.ends_with("_tests.rs") || rel.contains("/tests/") || rel.contains("test_helpers") {
            continue;
        }
        let text = std::fs::read_to_string(file).expect("readable source");
        let lines: Vec<&str> = text.lines().collect();
        let end = lines
            .iter()
            .position(|l| l.starts_with("mod ") && l.contains("test") && l.ends_with('{'))
            .unwrap_or(lines.len());
        let code: Vec<&str> = lines[..end]
            .iter()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect();

        /// A site is correctly paired when the two values agree about whether a
        /// matrix exists: `None` beside `None` (an exit that runs no covariance
        /// step), or a real expression beside a real expression. `covariance_method:
        /// None` written beside a real matrix is the defect, not a pairing.
        fn paired(matrix: &str, method: Option<&str>) -> bool {
            match (matrix == "None", method) {
                (true, Some(m)) => m == "None",
                (false, Some(m)) => m != "None",
                (_, None) => false,
            }
        }

        let mut sites = 0usize;
        let mut labelled = 0usize;
        for (i, line) in code.iter().enumerate() {
            // (a) assignment form — `out.covariance_matrix = …`
            if let Some(v) = writes(line, "covariance_matrix") {
                if line.contains(".covariance_matrix =") {
                    sites += 1;
                    // The matching assignment lives in the same block; a generous
                    // ±30-line window covers every real ordering without spanning
                    // two different results.
                    let lo = i.saturating_sub(30);
                    let hi = (i + 30).min(code.len());
                    let method = code[lo..hi]
                        .iter()
                        .find_map(|l| writes(l, "covariance_method"));
                    if paired(v, method) {
                        labelled += 1;
                    }
                    continue;
                }
            }
            // (b) struct-literal form
            let is_literal = ["OuterResult {", "FitResult {", "FitWire {"]
                .iter()
                .any(|s| line.contains(s));
            if !is_literal || line.contains("struct ") {
                continue;
            }
            // Walk the literal's own brace depth, collecting only its top-level
            // fields so a nested literal's fields are not attributed to it.
            let mut depth = 0i32;
            let mut own_fields: Vec<&str> = Vec::new();
            for l in &code[i..] {
                let before = depth;
                for c in l.chars() {
                    match c {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
                }
                if before == 1 {
                    own_fields.push(l);
                }
                if depth <= 0 && before > 0 {
                    break;
                }
            }
            let Some(matrix) = own_fields
                .iter()
                .find_map(|l| writes(l, "covariance_matrix"))
            else {
                continue;
            };
            sites += 1;
            let method = own_fields
                .iter()
                .find_map(|l| writes(l, "covariance_method"));
            if paired(matrix, method) {
                labelled += 1;
            }
        }
        if sites > 0 {
            found.insert(rel, (sites, labelled));
        }
    }

    let expected: BTreeMap<String, (usize, usize)> = [
        // Every estimator exit publishes its own `OuterResult`.
        ("src/estimation/bayes.rs", (1, 1)),
        ("src/estimation/gauss_newton.rs", (2, 2)),
        ("src/estimation/impmap.rs", (2, 2)),
        // The NLopt exit, the built-in-BFGS exit, the `outer_maxiter = 0`
        // evaluation exit — plus `CovStepOutcome`'s two constructions live in
        // `covariance.rs`, not here.
        ("src/estimation/outer_optimizer.rs", (3, 3)),
        ("src/estimation/saem.rs", (1, 1)),
        ("src/estimation/trust_region.rs", (1, 1)),
        ("src/estimation/vi/run.rs", (1, 1)),
        // The standalone re-run, which assigns onto a cloned `FitResult`.
        ("src/estimation/run_covariance.rs", (1, 1)),
        // Two synthetic evaluation-only `OuterResult`s plus the `FitResult`
        // `fit_inner` assembles.
        ("src/api/fit.rs", (3, 3)),
        // The `.fitrx` wire: the `FitWire` the save side builds, and the
        // `FitResult` the load side rebuilds.
        ("src/io/fitrx.rs", (2, 2)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    assert_eq!(
        found, expected,
        "a covariance matrix is published somewhere that does not publish the estimator \
         that produced it, or a site was removed.\n#1382: `covariance_matrix`, the \
         standard errors, `cov_eigenvalues` and `cov_condition_number` are not comparable \
         across estimators — the same fit measured 1.42e8 under `rsr` and 3.68e5 under \
         `s`, and both are correct. `OuterResult` is public API that `ferx-tools` reads \
         without going through `fit()`, which is why this is enforced at every publishing \
         site rather than once in `fit_inner`. A new site must set `covariance_method` \
         from the covariance step (never from `FitOptions`) and be listed here."
    );
}

/// **G4 — the warning payload carries the estimator too.**
///
/// `docs/warnings.qmd` listed the `condition_number` warning's `details` as the
/// bare number. An agent or the R wrapper branching on that payload is in exactly
/// the position the reporter was: 1.42e8 and 3.68e5 are the same fit under two
/// estimators, and only one of the two is comparable to a NONMEM `MATRIX=R` run.
///
/// Driven through `diagnostic_details` rather than a whole ill-conditioned fit,
/// since the payload is assembled from typed `FitResult` fields and not from the
/// message prose.
#[test]
fn the_condition_number_warning_payload_names_the_estimator() {
    use crate::api::postfit::{diagnostic_details, DiagStats};
    use crate::types::WarningCode;

    let stats = DiagStats {
        cov_condition_number: Some(1.42e8),
        covariance_method: Some(CovarianceMethod::Sandwich),
        ..Default::default()
    };
    let details = diagnostic_details(&WarningCode::ConditionNumber, &stats)
        .expect("a finite condition number must produce a payload");
    assert_eq!(details["condition_number"], serde_json::json!(1.42e8));
    assert_eq!(
        details["covariance_method"],
        serde_json::json!("rsr"),
        "the figure alone is not actionable — it means different things under r, s \
         and rsr: {details}"
    );

    // The mirror image: no matrix, so no estimator, and the key is *absent*
    // rather than `null`. A `null` reads as "unknown estimator" for a number that
    // was computed; absent reads as "there was no estimator", which is the truth.
    let unlabelled = DiagStats {
        cov_condition_number: Some(1.42e8),
        covariance_method: None,
        ..Default::default()
    };
    let details = diagnostic_details(&WarningCode::ConditionNumber, &unlabelled)
        .expect("payload still carries the number");
    assert!(
        details.get("covariance_method").is_none(),
        "an unlabelled payload must omit the key, not emit null: {details}"
    );
}
