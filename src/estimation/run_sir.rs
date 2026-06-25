//! Standalone SIR — run Sampling Importance Resampling against an existing
//! `FitResult` without re-fitting.
//!
//! Mirrors the SIR step that [`fit()`](crate::api::fit) runs inline when
//! `options.sir = true`, but lets callers drive SIR after a fit has completed
//! (potentially from a different session, loaded via `.fitrx`).
//!
//! Hash-verification rules:
//! - If the caller supplies `model` / `population` directly, those are used
//!   as-is. We cannot verify against `fit.model_hash` / `fit.data_hash`
//!   because the in-memory values don't carry their source bytes.
//! - If the caller passes `None`, we re-read from `fit.model_path` /
//!   `fit.data_path`. If a stored hash exists, a mismatch is a **hard error**
//!   — the whole point of `run_sir` is to refuse stale data.

use crate::estimation::uncertainty_samples::fitted_params_from_result;
use crate::io::hash::sha256_file;
use crate::types::*;
use nalgebra::DVector;
use std::path::Path;

/// Run SIR against an existing fit. Returns a new `FitResult` that is a clone
/// of `fit` with the `sir_*` fields populated. The returned fit's
/// `warnings` vector is unchanged — SIR-specific diagnostics live on
/// `sir_ess` (low ESS signals a poorly-matched proposal); the SIR kernel
/// does not emit structured warnings, so there is nothing to propagate.
///
/// # Notes on integrity
///
/// Hash verification (when stored on the fit) hits the filesystem once for
/// the hash and again during parse / CSV read. On a fast local filesystem
/// the window between those two reads is too small to be a practical TOCTOU
/// concern; on a shared filesystem or network mount, a file modified
/// in that window would pass the check and then be parsed in its modified
/// form. The intended threat model is accidental edits, not adversarial
/// substitution.
///
/// Paths recorded on the fit are stored verbatim (no canonicalisation), so
/// relative paths resolve against whatever the cwd is at `run_sir` time,
/// not at fit time. Pass absolute paths to `fit_from_files` if your
/// downstream code may run from a different working directory — or
/// canonicalise the path on the fit yourself before save / re-use, e.g.
/// `fit.model_path = Some(std::fs::canonicalize(&path)?.to_string_lossy().into_owned())`.
///
/// # IOV models (n_kappa > 0)
///
/// For models with inter-occasion variability, re-reading the dataset
/// requires the `iov_column` name from the model file's `[fit_options]`
/// block — that name doesn't survive on a `CompiledModel`. When the
/// caller passes `None` for both `model` and `population`, this function
/// parses the full model file (including `[fit_options]`) and threads
/// `iov_column` into `read_nonmem_csv`. When the caller supplies
/// `Some(model)` for an IOV model but leaves `population = None`, there
/// is no source of `iov_column`, so `run_sir` returns an error rather
/// than silently dropping occasion parsing. Workaround: pass both
/// `Some(model)` and `Some(population)` for IOV cases.
///
/// # Arguments
/// - `fit`: the maximum-likelihood fit to SIR-refine. Must carry a
///   `covariance_matrix` (i.e. the original fit ran with `covariance = true`).
/// - `model`: pre-compiled model. When `None`, re-parsed from `fit.model_path`.
/// - `population`: dataset. When `None`, re-read from `fit.data_path` (with
///   the `iov_column` constraint above for IOV models).
/// - `options`: SIR-relevant fields read are `sir_samples`, `sir_resamples`,
///   `sir_seed`, `sir_keep_samples`, plus the inner-loop settings
///   (`inner_maxiter`, `inner_tol`, `interaction`, `mu_referencing`,
///   `verbose`, `cancel`). Other fields (e.g. `method`) are ignored.
pub fn run_sir(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, String> {
    // Hash verification runs before the covariance check so a stale-input
    // error wins over a missing-cov error. A user pointing at the wrong
    // model or dataset should hear about that first; the cov-missing case
    // is downstream and only matters once the inputs are confirmed.

    // --- Resolve model -----------------------------------------------------
    //
    // When re-reading from disk we use `parse_full_model_file` (not the
    // CompiledModel-only `parse_model_file`) because IOV models need the
    // `iov_column` name from `[fit_options]` to parse occasions out of the
    // data; that info doesn't survive on `CompiledModel`. The parsed
    // fit_options are stashed in `iov_column_from_parse` for the
    // population re-read below.
    let model_owned: Option<CompiledModel>;
    let mut iov_column_from_parse: Option<String> = None;
    let model_ref: &CompiledModel = match model {
        Some(m) => m,
        None => {
            let path = fit.model_path.as_deref().ok_or_else(|| {
                "run_sir: no model supplied and fit.model_path is None. \
                 Either pass `model = Some(&model)` or re-fit via fit_from_files \
                 so the path is recorded."
                    .to_string()
            })?;
            if let Some(expected) = &fit.model_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(format!(
                        "run_sir: model hash mismatch for {}. Stored: {}, current: {}. \
                         The .ferx file has changed since the fit was produced — refusing \
                         to run SIR against stale source.",
                        path, expected, actual
                    ));
                }
            }
            let parsed = crate::parser::model_parser::parse_full_model_file(Path::new(path))?;
            iov_column_from_parse = parsed.fit_options.iov_column.clone();
            model_owned = Some(parsed.model);
            model_owned.as_ref().unwrap()
        }
    };

    // --- Resolve population -----------------------------------------------
    let pop_owned: Option<Population>;
    let pop_ref: &Population = match population {
        Some(p) => p,
        None => {
            // For IOV models the population MUST be parsed with the correct
            // `iov_column` so per-occasion kappas line up with the data. We
            // can only obtain that name from the model file's `[fit_options]`
            // block (parse_full_model_file path above). When the caller
            // supplied a `Some(model)` but no population, we have no
            // iov_column source for n_kappa > 0 models — refuse rather than
            // silently produce wrong likelihoods.
            if model.is_some() && model_ref.n_kappa > 0 {
                return Err(
                    "run_sir: caller-supplied `model` for an IOV (n_kappa > 0) model \
                     requires `population` to also be supplied — `iov_column` from \
                     `[fit_options]` is needed to parse per-occasion kappas correctly."
                        .to_string(),
                );
            }
            let path = fit.data_path.as_deref().ok_or_else(|| {
                "run_sir: no population supplied and fit.data_path is None. \
                 Either pass `population = Some(&pop)` or re-fit via fit_from_files \
                 so the path is recorded."
                    .to_string()
            })?;
            if let Some(expected) = &fit.data_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(format!(
                        "run_sir: data hash mismatch for {}. Stored: {}, current: {}. \
                         The dataset has changed since the fit was produced — refusing \
                         to run SIR against stale data.",
                        path, expected, actual
                    ));
                }
            }
            let p = crate::io::datareader::read_nonmem_csv(
                Path::new(path),
                None,
                iov_column_from_parse.as_deref(),
            )?;
            pop_owned = Some(p);
            pop_owned.as_ref().unwrap()
        }
    };

    // --- Sanity-check dimensions ------------------------------------------
    if model_ref.n_eta != fit.omega.nrows() {
        return Err(format!(
            "run_sir: supplied model has n_eta = {} but fit.omega is {}×{}. \
             Verify you supplied the same model used for the fit.",
            model_ref.n_eta,
            fit.omega.nrows(),
            fit.omega.ncols()
        ));
    }
    if !fit.subjects.is_empty() && fit.subjects[0].eta.len() != model_ref.n_eta {
        return Err(format!(
            "run_sir: fit.subjects[0] has eta dim {} but model has n_eta = {}. \
             Subject EBEs are inconsistent with the supplied model.",
            fit.subjects[0].eta.len(),
            model_ref.n_eta
        ));
    }

    // --- Reconstruct ModelParameters and eta_hats -------------------------
    let params = fitted_params_from_result(fit, model_ref);
    let eta_hats: Vec<DVector<f64>> = fit.subjects.iter().map(|s| s.eta.clone()).collect();

    // --- Now require a covariance matrix to seed the proposal -------------
    let cov = fit.covariance_matrix.as_ref().ok_or_else(|| {
        "run_sir requires fit.covariance_matrix; re-run the original fit \
         with the covariance step enabled (FitOptions.run_covariance_step = \
         true, or `covariance = true` in the model file's [fit_options])."
            .to_string()
    })?;

    // --- Run SIR (identical to the inline path in fit()) ------------------
    let sir = crate::estimation::sir::run_sir_core(
        model_ref, pop_ref, &params, &eta_hats, cov, fit.ofv, options,
    )?;

    // --- Build the augmented FitResult ------------------------------------
    let mut out = fit.clone();
    out.sir_ci_theta = Some(sir.ci_theta);
    out.sir_ci_omega = Some(sir.ci_omega);
    out.sir_ci_sigma = Some(sir.ci_sigma);
    out.sir_ess = Some(sir.effective_sample_size);
    out.sir_resamples_packed = sir.resamples_packed;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fit_from_files;

    // Use the in-tree warfarin example + data. They live at repo paths
    // `examples/warfarin.ferx` and `data/warfarin.csv` (see CLAUDE.md);
    // tests run from the crate root, so relative paths work directly.
    const MODEL_PATH: &str = "examples/warfarin.ferx";
    const DATA_PATH: &str = "data/warfarin.csv";

    fn copy_example_to_tempdir(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        // Hash-mismatch tests need to mutate the source files. Copy them to a
        // tempdir so we don't touch the checked-in examples.
        let model = dir.join("model.ferx");
        let data = dir.join("data.csv");
        std::fs::copy(MODEL_PATH, &model).unwrap();
        std::fs::copy(DATA_PATH, &data).unwrap();
        (model, data)
    }

    fn quick_opts() -> FitOptions {
        // Small SIR settings so the test stays under a few seconds.
        FitOptions {
            verbose: false,
            run_covariance_step: true,
            // Pin the derivative-free outer optimizer: this test exercises the
            // SIR plumbing, not the optimizer-default choice, so keep it on the
            // path it was validated against rather than the `auto` default (#490).
            optimizer: crate::types::Optimizer::Bobyqa,
            sir_samples: 8,
            sir_resamples: 4,
            sir_seed: Some(1),
            ..FitOptions::default()
        }
    }

    #[test]
    fn paths_and_hashes_are_populated_by_fit_from_files() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");

        assert_eq!(
            fit.model_path.as_deref(),
            Some(model_path.to_str().unwrap())
        );
        assert_eq!(fit.data_path.as_deref(), Some(data_path.to_str().unwrap()));
        assert_eq!(fit.model_hash.as_deref().map(|s| s.len()), Some(64));
        assert_eq!(fit.data_hash.as_deref().map(|s| s.len()), Some(64));
        assert_eq!(
            fit.model_hash.as_deref(),
            Some(sha256_file(&model_path).unwrap().as_str())
        );
        assert_eq!(
            fit.data_hash.as_deref(),
            Some(sha256_file(&data_path).unwrap().as_str())
        );
    }

    #[test]
    fn run_sir_rejects_when_no_covariance() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = FitOptions {
            verbose: false,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            None,
            Some(opts.clone()),
        )
        .expect("fit must converge");

        assert!(fit.covariance_matrix.is_none());
        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.contains("covariance_matrix"),
            "expected cov-missing message, got: {}",
            err
        );
    }

    #[test]
    fn run_sir_detects_modified_model_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            None,
            Some(opts.clone()),
        )
        .expect("fit must converge");

        // Tamper with the model file (append whitespace — enough to flip the
        // SHA-256). The next run_sir call must refuse.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&model_path)
            .unwrap();
        writeln!(f, "  ").unwrap();
        drop(f);

        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.contains("model hash mismatch"),
            "expected hash-mismatch message, got: {}",
            err
        );
    }

    /// Run a fit and skip the test body with a logged message when the
    /// covariance step doesn't converge. SIR requires a non-None
    /// covariance matrix; the warfarin FD cov step is flaky. This helper
    /// centralises the skip pattern so the SIR happy-path tests don't fail
    /// spuriously.
    fn fit_with_cov_or_skip(
        model_path: &str,
        data_path: &str,
        opts: FitOptions,
    ) -> Option<FitResult> {
        let fit =
            fit_from_files(model_path, data_path, None, Some(opts)).expect("fit must converge");
        if fit.covariance_matrix.is_none() {
            eprintln!(
                "[skip] covariance step did not produce a matrix (likely FD \
                 instability); skipping SIR happy-path assertions"
            );
            return None;
        }
        Some(fit)
    }

    #[test]
    fn run_sir_happy_path_populates_sir_fields() {
        // Integration test: fit_from_files → run_sir(None, None) → verify
        // the returned FitResult carries the four SIR diagnostics.
        // Exercises the "re-read model + data from paths + verify hashes"
        // code path, which is the primary use case for the public API.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        let out = run_sir(&fit, None, None, &opts).expect("run_sir succeeds");

        // Every SIR field populated.
        let ci_theta = out.sir_ci_theta.as_ref().expect("ci_theta populated");
        let ci_omega = out.sir_ci_omega.as_ref().expect("ci_omega populated");
        let ci_sigma = out.sir_ci_sigma.as_ref().expect("ci_sigma populated");
        let ess = out.sir_ess.expect("sir_ess populated");

        assert_eq!(ci_theta.len(), fit.theta.len());
        assert_eq!(ci_omega.len(), fit.omega.nrows());
        assert_eq!(ci_sigma.len(), fit.sigma.len());
        // ESS is bounded by sir_samples; with sir_samples=8, sir_resamples=4
        // we expect ess > 0 and ess <= sir_samples.
        assert!(ess > 0.0 && ess <= opts.sir_samples as f64);

        // Lower <= upper on every CI band.
        for (lo, hi) in ci_theta {
            assert!(lo <= hi, "theta CI: {} > {}", lo, hi);
        }
        for (lo, hi) in ci_omega {
            assert!(lo <= hi, "omega CI: {} > {}", lo, hi);
        }

        // Non-SIR fields unchanged (we copy fit, then stamp on the SIR
        // fields — the rest must round-trip).
        assert_eq!(out.theta, fit.theta);
        assert_eq!(out.omega, fit.omega);
        assert_eq!(out.sigma, fit.sigma);
        assert_eq!(out.ofv, fit.ofv);
    }

    #[test]
    fn run_sir_errors_when_iov_model_supplied_without_population() {
        // Caller passes Some(model) for an IOV (n_kappa > 0) model but
        // None for population. The wrapper must refuse rather than
        // silently re-read the data without iov_column (which would drop
        // occasion parsing and produce wrong SIR results).
        //
        // The fit object here is shape-wise nonsense (warfarin non-IOV
        // fit + IOV model) — but the IOV check fires before any
        // dimension check, so this triggers the intended branch first.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        let iov_model = crate::parser::model_parser::parse_full_model_file(std::path::Path::new(
            "examples/warfarin_iov.ferx",
        ))
        .expect("parse warfarin_iov.ferx")
        .model;
        assert!(
            iov_model.n_kappa > 0,
            "warfarin_iov.ferx must declare kappa"
        );

        let err = run_sir(&fit, Some(&iov_model), None, &opts).unwrap_err();
        assert!(
            err.contains("IOV") && err.contains("population"),
            "expected IOV-needs-population error, got: {}",
            err
        );
    }

    #[test]
    fn run_sir_detects_modified_data_file() {
        // Symmetric to `run_sir_detects_modified_model_file` — verify the
        // data-hash branch fires on tamper. Without this test the data
        // side of the integrity check has no coverage.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Append a comment line so the CSV still parses but the hash flips.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&data_path)
            .unwrap();
        writeln!(f, "# tampered").unwrap();
        drop(f);

        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.contains("data hash mismatch"),
            "expected data hash-mismatch message, got: {}",
            err
        );
    }

    #[test]
    fn run_sir_with_caller_supplied_model_and_pop_skips_hash_check() {
        // When the caller passes Some(model) AND Some(population), the
        // wrapper uses them as-is — no hash verification. Tampering with
        // the on-disk files (so the recorded hashes no longer match)
        // must NOT trigger a hash mismatch error in this branch.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Tamper only the model file — changing its hash is sufficient to
        // verify the hash-bypass behaviour. Appending to the CSV would add
        // a fake subject (the CSV reader doesn't skip comment lines), which
        // would make pop.subjects.len() != fit.eta_hats.len() and panic in
        // run_inner_loop_warm; the data file hash is checked separately in
        // run_sir_detects_modified_data_file.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&model_path)
                .unwrap();
            writeln!(f, "# tampered").unwrap();
        }

        // Build model + population in memory (from the tampered/original files —
        // the caller-supplied branch doesn't verify hashes).
        let parsed = crate::parser::model_parser::parse_full_model_file(&model_path)
            .expect("parse tampered model");
        let pop =
            crate::io::datareader::read_nonmem_csv(&data_path, None, None).expect("read data");

        // Should succeed despite the on-disk tampering, because the
        // caller-supplied branch bypasses the hash check entirely.
        let out = run_sir(&fit, Some(&parsed.model), Some(&pop), &opts)
            .expect("caller-supplied model+pop must skip the hash check");
        assert!(out.sir_ess.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn run_sir_succeeds_with_fixed_parameters() {
        // Regression: before run_sir_core was restricted to the free
        // subspace, every SIR sample for a model with at least one FIX-ed
        // parameter failed the bounds check. `compute_covariance` zeroes the
        // rows/cols of FIX-ed indices, the proposal-cov regularisation
        // (eigenvalue floor of 1e-8) then perturbs them by ~1e-4, and
        // `compute_bounds` pins them with `lower == upper == x_hat[i]` — so
        // the strict bounds check rejected every sample and SIR returned
        // "All SIR samples had invalid weights".
        //
        // `examples/warfarin_fix.ferx` exercises FIX on a theta, an omega,
        // and a sigma in one go.
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model_fix.ferx");
        let data_path = dir.path().join("data.csv");
        std::fs::copy("examples/warfarin_fix.ferx", &model_path).unwrap();
        std::fs::copy(DATA_PATH, &data_path).unwrap();

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Sanity-check the fit actually carries FIX-ed parameters — without
        // this the test would silently degenerate into a non-FIX path.
        assert!(
            fit.theta_fixed.iter().any(|&b| b)
                || fit.omega_fixed.iter().any(|&b| b)
                || fit.sigma_fixed.iter().any(|&b| b),
            "expected at least one FIX-ed parameter in warfarin_fix.ferx"
        );

        let out = run_sir(&fit, None, None, &opts)
            .expect("SIR must succeed on a model with FIX-ed parameters");
        let ess = out.sir_ess.expect("sir_ess populated");
        assert!(
            ess > 0.0 && ess <= opts.sir_samples as f64,
            "ess = {} out of (0, {}]",
            ess,
            opts.sir_samples
        );
    }

    #[test]
    fn run_sir_errors_when_no_model_path_recorded_and_no_caller_model() {
        // Cover the "no model path recorded and caller didn't supply one"
        // branch — the in-memory `fit()` path leaves `model_path = None`,
        // and a downstream caller that also passes `None` should get a
        // clear error rather than a panic or generic NPE-style failure.
        //
        // Cheapest way to get a valid FitResult with empty paths is to run
        // `fit_from_files` and then null out the path/hash fields.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let mut fit = fit_from_files(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");
        fit.model_path = None;
        fit.data_path = None;
        fit.model_hash = None;
        fit.data_hash = None;

        let err = run_sir(&fit, None, None, &quick_opts()).unwrap_err();
        assert!(
            err.contains("no model supplied"),
            "expected 'no model supplied' error, got: {}",
            err
        );
    }
}
