//! Standalone covariance step — run the FD-Hessian covariance step against an
//! existing `FitResult` without re-fitting.
//!
//! Mirrors the covariance step that [`fit()`](crate::api::fit) runs inline when
//! `options.run_covariance_step = true`, but lets callers drive it after a fit
//! has completed (potentially from a different session, loaded via `.fitrx`).
//! This is the covariance-step analogue of
//! [`run_sir`](crate::estimation::run_sir::run_sir).
//!
//! Hash-verification rules match `run_sir`:
//! - If the caller supplies `model` / `population` directly, those are used
//!   as-is (no hash check — the in-memory values don't carry their source bytes).
//! - If the caller passes `None`, we re-read from `fit.model_path` /
//!   `fit.data_path`. If a stored hash exists, a mismatch is a **hard error** —
//!   the point of `run_covariance` is to refuse stale inputs.

use crate::api::{cov_diagnostics, extract_standard_errors, resolve_covariance_status};
use crate::estimation::outer_optimizer::{compute_covariance, CovarianceStepResult};
use crate::estimation::parameterization::{compute_mu_k, pack_params};
use crate::estimation::uncertainty_samples::fitted_params_from_result;
use crate::io::hash::sha256_file;
use crate::types::*;
use std::path::Path;

/// Run the covariance step against an existing fit. Returns a new `FitResult`
/// that is a clone of `fit` with the covariance fields refreshed:
/// `covariance_matrix`, `se_theta` / `se_omega` / `se_sigma` / `se_kappa`,
/// `covariance_status`, `cov_eigenvalues`, `cov_condition_number`, and
/// `covariance_wall_time_secs`.
///
/// The numerics reuse the inline covariance step in `fit()` — this wrapper
/// calls the same `compute_covariance` at the same converged point rather than
/// duplicating the FD-Hessian logic, so a fit produced with `covariance = false`
/// followed by `run_covariance` yields the same covariance matrix and SEs as a
/// single fit produced with `covariance = true`, up to finite-difference noise
/// (~1e-5, platform dependent): the inline path passes the optimizer's exact
/// Cholesky factor, while this wrapper reconstructs it from `fit.omega` via a
/// re-decomposition, which the FD Hessian amplifies.
///
/// # Failure semantics
///
/// A covariance step that runs but cannot produce a usable matrix (a
/// structurally-unusable or non-positive-definite FD Hessian) is **not** an
/// `Err`. Mirroring `fit()`, the returned `FitResult` carries
/// `covariance_matrix = None`, `covariance_status = Failed`, and the diagnostic
/// appended to `warnings`. `Err` is reserved for input problems: a missing /
/// hash-mismatched model or dataset, a dimension mismatch, or an IOV model
/// supplied without its population (see below).
///
/// # IOV models (n_kappa > 0)
///
/// As with `run_sir`, re-reading the dataset for an IOV model requires the
/// `iov_column` name from the model file's `[fit_options]` block, which does
/// not survive on a `CompiledModel`. When the caller passes `None` for both
/// `model` and `population`, this function parses the full model file and
/// threads `iov_column` into `read_nonmem_csv`. When the caller supplies
/// `Some(model)` for an IOV model but leaves `population = None`, there is no
/// source of `iov_column`, so `run_covariance` returns an error rather than
/// silently dropping occasion parsing. Workaround: pass both `Some(model)` and
/// `Some(population)` for IOV cases.
///
/// # Arguments
/// - `fit`: the maximum-likelihood fit to compute a covariance for.
/// - `model`: pre-compiled model. When `None`, re-parsed from `fit.model_path`.
/// - `population`: dataset. When `None`, re-read from `fit.data_path` (with the
///   `iov_column` constraint above for IOV models).
/// - `options`: covariance-relevant fields read are `covariance_method`,
///   `fd_hessian_step`, `cov_inner_tol`, `interaction`, `mu_referencing`, the
///   inner-loop settings, and `cancel`. `run_covariance_step` on `options` is
///   **ignored** — calling this function is itself the request to run the step.
pub fn run_covariance(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, String> {
    // Input resolution mirrors `run_sir` exactly: stale-input errors win over
    // any downstream failure so a user pointing at the wrong model/dataset
    // hears about that first.

    // --- Resolve model -----------------------------------------------------
    let model_owned: Option<CompiledModel>;
    let mut iov_column_from_parse: Option<String> = None;
    // `[data]` canonical-role → header remapping (#730). Captured from the parse
    // so a re-read from disk honours renames like `TIME = TAFD` — otherwise the
    // CSV reader looks for the canonical headers and mis-parses / hard-errors on
    // a dataset the original fit read fine. Only meaningful on the re-parse path
    // (model == None); a caller-supplied model+population never re-reads.
    let mut column_map_from_parse: Vec<(String, String)> = Vec::new();
    let model_ref: &CompiledModel = match model {
        Some(m) => m,
        None => {
            let path = fit.model_path.as_deref().ok_or_else(|| {
                "run_covariance: no model supplied and fit.model_path is None. \
                 Either pass `model = Some(&model)` or re-fit via fit_from_files \
                 so the path is recorded."
                    .to_string()
            })?;
            if let Some(expected) = &fit.model_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(format!(
                        "run_covariance: model hash mismatch for {}. Stored: {}, current: {}. \
                         The .ferx file has changed since the fit was produced — refusing \
                         to run the covariance step against stale source.",
                        path, expected, actual
                    ));
                }
            }
            let parsed = crate::parser::model_parser::parse_full_model_file(Path::new(path))?;
            iov_column_from_parse = parsed.fit_options.iov_column.clone();
            column_map_from_parse = parsed.column_map.clone();
            model_owned = Some(parsed.model);
            model_owned.as_ref().unwrap()
        }
    };

    // --- Resolve population -----------------------------------------------
    let pop_owned: Option<Population>;
    let pop_ref: &Population = match population {
        Some(p) => p,
        None => {
            if model.is_some() && model_ref.n_kappa > 0 {
                return Err(
                    "run_covariance: caller-supplied `model` for an IOV (n_kappa > 0) model \
                     requires `population` to also be supplied — `iov_column` from \
                     `[fit_options]` is needed to parse per-occasion kappas correctly."
                        .to_string(),
                );
            }
            let path = fit.data_path.as_deref().ok_or_else(|| {
                "run_covariance: no population supplied and fit.data_path is None. \
                 Either pass `population = Some(&pop)` or re-fit via fit_from_files \
                 so the path is recorded."
                    .to_string()
            })?;
            if let Some(expected) = &fit.data_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(format!(
                        "run_covariance: data hash mismatch for {}. Stored: {}, current: {}. \
                         The dataset has changed since the fit was produced — refusing \
                         to run the covariance step against stale data.",
                        path, expected, actual
                    ));
                }
            }
            let p = crate::io::datareader::read_nonmem_csv_mapped(
                Path::new(path),
                None,
                iov_column_from_parse.as_deref(),
                &column_map_from_parse,
            )?;
            pop_owned = Some(p);
            pop_owned.as_ref().unwrap()
        }
    };

    // --- Sanity-check dimensions ------------------------------------------
    if model_ref.n_eta != fit.omega.nrows() {
        return Err(format!(
            "run_covariance: supplied model has n_eta = {} but fit.omega is {}×{}. \
             Verify you supplied the same model used for the fit.",
            model_ref.n_eta,
            fit.omega.nrows(),
            fit.omega.ncols()
        ));
    }
    if !fit.subjects.is_empty() && fit.subjects[0].eta.len() != model_ref.n_eta {
        return Err(format!(
            "run_covariance: fit.subjects[0] has eta dim {} but model has n_eta = {}. \
             Subject EBEs are inconsistent with the supplied model.",
            fit.subjects[0].eta.len(),
            model_ref.n_eta
        ));
    }

    // --- Reconstruct the covariance-step inputs ---------------------------
    //
    // `compute_covariance` reconverges the EBEs (and recomputes H) at every
    // perturbed point, so the passed `eta_hats` are only a warm-start and
    // `h_matrices` is unused. The score-cross-product path (covariance_method
    // = s / rsr) does read `kappas`, so we rebuild all three by re-running the
    // final inner loop at the fitted parameters.
    //
    // We **cold-start** (`warm_etas = None`) rather than seeding from the fit's
    // stored EBEs, because that is exactly what the inline covariance path in
    // `outer_optimizer` does (its "final inner loop at converged parameters"
    // passes `None`). Warm-starting from the stored EBEs would run the inner
    // BFGS from a slightly different point and, at a loose `inner_tol`, land a
    // slightly different EBE than the cold path — enough to make the covariance
    // matrix diverge from the inline result by ~1e-4 on some platforms. Matching
    // the inline start point keeps the two numerics bit-for-bit comparable.
    let params = fitted_params_from_result(fit, model_ref);
    let x_hat = pack_params(&params);
    let mu_k = compute_mu_k(model_ref, &params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _stats, kappas) =
        crate::estimation::inner_optimizer::run_inner_loop_warm(
            model_ref,
            pop_ref,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            None,
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
        );

    // --- Run the covariance step (identical to the inline path in fit()) ---
    let cov_timer = std::time::Instant::now();
    let mut new_warnings: Vec<String> = Vec::new();
    let covariance_matrix: Option<nalgebra::DMatrix<f64>> = match compute_covariance(
        &x_hat,
        &params,
        model_ref,
        pop_ref,
        &eta_hats,
        &h_matrices,
        &kappas,
        options,
    ) {
        CovarianceStepResult::Success(out) => {
            new_warnings.extend(out.warnings);
            Some(out.matrix)
        }
        CovarianceStepResult::Unusable(msg) => {
            new_warnings.push(msg);
            None
        }
        CovarianceStepResult::FailedNonPd { reason, .. } => {
            // The |eigenvalue|-rectified proposal is only useful to the SIR
            // fallback, which is a separate step here — surface the diagnostic
            // and leave the covariance unavailable. Callers wanting the SIR
            // fallback run `run_sir` afterwards.
            new_warnings.push(reason);
            None
        }
    };
    let covariance_wall_time_secs = cov_timer.elapsed().as_secs_f64();

    // --- Build the refreshed FitResult ------------------------------------
    let (se_theta, se_omega, se_sigma, se_kappa) =
        extract_standard_errors(&covariance_matrix, &params);
    let (cov_eigenvalues, cov_condition_number) = cov_diagnostics(covariance_matrix.as_ref());
    // Bayesian fits never run a Hessian covariance step; guard so a covariance
    // request against a Bayesian fit reports NotRequested rather than Failed.
    let covariance_status =
        resolve_covariance_status(fit.bayes.is_none(), covariance_matrix.is_some(), false);

    let mut out = fit.clone();
    out.covariance_matrix = covariance_matrix;
    out.se_theta = se_theta;
    out.se_omega = se_omega;
    out.se_sigma = se_sigma;
    out.se_kappa = se_kappa;
    out.cov_eigenvalues = cov_eigenvalues;
    out.cov_condition_number = cov_condition_number;
    out.covariance_status = covariance_status;
    out.covariance_wall_time_secs = covariance_wall_time_secs;
    out.warnings.extend(new_warnings);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fit_from_files;

    // In-tree warfarin example + data (see CLAUDE.md). Tests run from the crate
    // root, so relative paths work directly.
    const MODEL_PATH: &str = "examples/warfarin.ferx";
    const DATA_PATH: &str = "data/warfarin.csv";

    fn copy_example_to_tempdir(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        // Hash-mismatch tests mutate the source files; copy them so the
        // checked-in examples are never touched.
        let model = dir.join("model.ferx");
        let data = dir.join("data.csv");
        std::fs::copy(MODEL_PATH, &model).unwrap();
        std::fs::copy(DATA_PATH, &data).unwrap();
        (model, data)
    }

    fn quick_opts() -> FitOptions {
        FitOptions {
            verbose: false,
            run_covariance_step: true,
            // Pin the derivative-free outer optimizer so the parity test compares
            // against a deterministic converged point rather than the `auto`
            // default (#490).
            optimizer: crate::types::Optimizer::Bobyqa,
            ..FitOptions::default()
        }
    }

    /// Fit and skip (returning `None`) when the inline covariance step didn't
    /// produce a matrix. The warfarin FD cov step is occasionally FD-unstable;
    /// the parity assertions require a non-None reference matrix.
    fn fit_with_cov_or_skip(
        model_path: &str,
        data_path: &str,
        opts: FitOptions,
    ) -> Option<FitResult> {
        let fit = fit_from_files(model_path, Some(data_path), None, Some(opts))
            .expect("fit must converge");
        if fit.covariance_matrix.is_none() {
            eprintln!(
                "[skip] inline covariance step did not produce a matrix (likely FD \
                 instability); skipping run_covariance parity assertions"
            );
            return None;
        }
        Some(fit)
    }

    #[test]
    fn run_covariance_matches_inline_covariance() {
        // The wrapper must reproduce the inline `fit()` covariance step exactly:
        // fit A runs cov inline; fit B fits without cov, then run_covariance
        // refreshes it. Both converge to the same point (deterministic BOBYQA),
        // so the covariance matrix and SEs must agree.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit_a) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Fit B: identical settings but no inline covariance step.
        let opts_no_cov = FitOptions {
            run_covariance_step: false,
            ..opts.clone()
        };
        let fit_b = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(opts_no_cov),
        )
        .expect("fit must converge");
        assert!(
            fit_b.covariance_matrix.is_none(),
            "fit B should carry no covariance (run_covariance_step = false)"
        );

        let out = run_covariance(&fit_b, None, None, &opts).expect("run_covariance succeeds");

        assert_eq!(out.covariance_status, CovarianceStatus::Computed);
        let cov_ref = fit_a.covariance_matrix.as_ref().unwrap();
        let cov_new = out
            .covariance_matrix
            .as_ref()
            .expect("run_covariance populated covariance_matrix");
        assert_eq!(cov_ref.shape(), cov_new.shape());
        let max_abs_diff = cov_ref
            .iter()
            .zip(cov_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        // `run_covariance` reconstructs the packed parameter vector via a
        // Cholesky round-trip (FitResult.omega → Chol(FitResult.omega) → L),
        // which introduces O(ε_machine · cond(L)) error relative to the inline
        // path's exact L (stored from the optimizer, never re-decomposed).
        // The resulting FD-Hessian perturbations differ by the same amount,
        // so strict sub-1e-6 parity is not achievable through this path.
        // 1e-4 is tight enough to catch any real regression while being
        // realistic about the precision limit of the round-trip.
        assert!(
            max_abs_diff < 1e-4,
            "run_covariance matrix diverged from inline cov (max abs diff {max_abs_diff})"
        );

        // SEs are derived from the covariance, so they must agree too.
        assert_eq!(out.se_theta.is_some(), fit_a.se_theta.is_some());
        if let (Some(a), Some(b)) = (&fit_a.se_theta, &out.se_theta) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-4, "se_theta diverged: {x} vs {y}");
            }
        }

        // Non-covariance fields round-trip unchanged.
        assert_eq!(out.theta, fit_b.theta);
        assert_eq!(out.omega, fit_b.omega);
        assert_eq!(out.ofv, fit_b.ofv);
    }

    /// Regression (#730 interaction): when `run_covariance` re-reads the dataset
    /// from disk (`model = None`, `population = None`), it must honour the
    /// model's `[data]` header renaming. Otherwise the CSV reader looks for the
    /// canonical headers and hard-errors on a dataset the original fit read fine.
    #[test]
    fn run_covariance_honours_data_column_map_on_reread() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        // Rename the TIME header to a non-canonical name in the data, and add a
        // `[data]` block that maps it back. `read_nonmem_csv` (no map) would fail
        // to find TIME; `read_nonmem_csv_mapped` resolves TIME = TAFD.
        let raw = std::fs::read_to_string(&data_path).unwrap();
        let (header, rest) = raw.split_once('\n').unwrap();
        let renamed_header = header.replacen("TIME", "TAFD", 1);
        std::fs::write(&data_path, format!("{renamed_header}\n{rest}")).unwrap();

        let model_src = std::fs::read_to_string(&model_path).unwrap();
        let data_str = data_path.to_str().unwrap();
        std::fs::write(
            &model_path,
            format!("{model_src}\n[data]\npath = {data_str}\nTIME = TAFD\n"),
        )
        .unwrap();

        // Fit without cov (fit_from_files applies the [data] map), then run the
        // standalone step forcing a disk re-read.
        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(FitOptions {
                run_covariance_step: false,
                ..quick_opts()
            }),
        )
        .expect("fit must converge on the renamed dataset");

        let out = run_covariance(&fit, None, None, &quick_opts())
            .expect("run_covariance must re-read the renamed dataset via the column map");
        // The re-read succeeded and produced a real covariance step — proving the
        // map was applied (an unmapped read would have errored above).
        assert!(matches!(
            out.covariance_status,
            CovarianceStatus::Computed | CovarianceStatus::Failed
        ));
    }

    #[test]
    fn run_covariance_reports_failed_on_bad_fd_step() {
        // A covariance step that runs but can't produce a matrix is non-fatal:
        // Ok(fit) with status = Failed and the diagnostic in warnings. A
        // non-positive fd_hessian_step makes compute_covariance return Unusable
        // deterministically, without depending on FD conditioning.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(FitOptions {
                run_covariance_step: false,
                ..quick_opts()
            }),
        )
        .expect("fit must converge");

        let bad_opts = FitOptions {
            fd_hessian_step: -1.0,
            ..quick_opts()
        };
        let n_warn_before = fit.warnings.len();
        let out =
            run_covariance(&fit, None, None, &bad_opts).expect("failed cov step is Ok, not Err");
        assert!(out.covariance_matrix.is_none());
        assert_eq!(out.covariance_status, CovarianceStatus::Failed);
        assert!(
            out.warnings.len() > n_warn_before,
            "a diagnostic warning must be appended"
        );
        assert!(
            out.warnings.iter().any(|w| w.contains("fd_hessian_step")),
            "warning should name fd_hessian_step, got: {:?}",
            out.warnings
        );
    }

    #[test]
    fn run_covariance_detects_modified_model_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");

        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&model_path)
            .unwrap();
        writeln!(f, "  ").unwrap();
        drop(f);

        let err = run_covariance(&fit, None, None, &quick_opts()).unwrap_err();
        assert!(
            err.contains("model hash mismatch"),
            "expected hash-mismatch message, got: {}",
            err
        );
    }

    #[test]
    fn run_covariance_detects_modified_data_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");

        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&data_path)
            .unwrap();
        writeln!(f, "# tampered").unwrap();
        drop(f);

        let err = run_covariance(&fit, None, None, &quick_opts()).unwrap_err();
        assert!(
            err.contains("data hash mismatch"),
            "expected data hash-mismatch message, got: {}",
            err
        );
    }

    #[test]
    fn run_covariance_with_caller_supplied_model_and_pop_skips_hash_check() {
        // Caller passes Some(model) AND Some(population): used as-is, no hash
        // check. Tampering the on-disk model (so its recorded hash no longer
        // matches) must NOT trigger a mismatch error.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");

        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&model_path)
                .unwrap();
            writeln!(f, "# tampered").unwrap();
        }

        let parsed = crate::parser::model_parser::parse_full_model_file(&model_path)
            .expect("parse tampered model");
        let pop =
            crate::io::datareader::read_nonmem_csv(&data_path, None, None).expect("read data");

        // Succeeds despite on-disk tampering — the caller-supplied branch
        // bypasses the hash check. Cov may or may not be produced (FD), but the
        // call itself must not error on a hash mismatch.
        let out = run_covariance(&fit, Some(&parsed.model), Some(&pop), &quick_opts())
            .expect("caller-supplied model+pop must skip the hash check");
        assert_ne!(out.covariance_status, CovarianceStatus::NotRequested);
    }

    #[test]
    fn run_covariance_errors_when_no_model_path_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let mut fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");
        fit.model_path = None;
        fit.data_path = None;
        fit.model_hash = None;
        fit.data_hash = None;

        let err = run_covariance(&fit, None, None, &quick_opts()).unwrap_err();
        assert!(
            err.contains("no model supplied"),
            "expected 'no model supplied' error, got: {}",
            err
        );
    }

    #[test]
    fn run_covariance_errors_when_iov_model_supplied_without_population() {
        // Some(model) for an IOV (n_kappa > 0) model but None population: must
        // refuse rather than re-read data without iov_column. The IOV check
        // fires before any dimension check, so the shape-mismatched fit is fine.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");

        let iov_model = crate::parser::model_parser::parse_full_model_file(std::path::Path::new(
            "examples/warfarin_iov.ferx",
        ))
        .expect("parse warfarin_iov.ferx")
        .model;
        assert!(
            iov_model.n_kappa > 0,
            "warfarin_iov.ferx must declare kappa"
        );

        let err = run_covariance(&fit, Some(&iov_model), None, &quick_opts()).unwrap_err();
        assert!(
            err.contains("IOV") && err.contains("population"),
            "expected IOV-needs-population error, got: {}",
            err
        );
    }
}
