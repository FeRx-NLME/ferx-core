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
use crate::estimation::covariance::{run_covariance_step_inner, CovStepOutcome};
use crate::estimation::parameterization::{compute_mu_k, pack_params, packed_len, unpack_params};
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
/// single fit produced with `covariance = true`.
///
/// When the fit carries the optimizer's exact packed vector
/// ([`FitResult::packed_estimate`] — the in-memory FOCE/FOCEI path), the match is
/// **bit-for-bit**: the parameters are rebuilt by *unpacking* that vector, so the
/// `OmegaMatrix`'s `Ω⁻¹` / `log|Ω|` come from the same Cholesky factor `L` the
/// inline step used. Reconstructing them from `fit.omega` instead re-decomposes
/// `chol(L·Lᵀ) ≠ L` to machine-ε; that tiny `Ω⁻¹` difference feeds the inner NLL
/// penalty, shifts the reconverged EBEs, and the FD Hessian amplifies it — up to
/// ~1e-1 on an ill-conditioned ω direction (the divergence #816's review surfaced).
/// A fit reloaded from `.fitrx` (no packed vector), or from SAEM /
/// importance-sampling / Bayes (whose inline step re-packs `omega` the same way
/// and so already agrees), takes that re-decomposition fallback. Every in-memory
/// packed-Cholesky-space fit — NLopt, BFGS, trust region, Gauss-Newton — carries
/// the vector and reproduces bit-for-bit.
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
    // #1212, same last hop as `fit()`: this call's `ode_reltol` / `ode_method` / … have to
    // reach the integrator, and the spec they would otherwise be read off carries the
    // parse-time values. It matters more here than almost anywhere else — the covariance step
    // is a second difference of the reconverged OFV, so running it at a *different* accuracy
    // than the fit that produced the estimates is exactly how a plausible-looking standard
    // error comes out wrong. The scope covers the whole call, including the re-parse and
    // re-read paths, and puts the per-subject fan-out on a pool whose workers carry the same
    // settings — arming alone would reach this thread and leave the workers on the model
    // file's.
    crate::api::with_fit_ode_scope(options, || {
        run_covariance_scoped(fit, model, population, options)
    })?
}

fn run_covariance_scoped(
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

    // This entry point re-runs the inner loop (EBEs → the prediction walk), so
    // it needs the same dose-compartment precondition `fit()` enforces (#375) —
    // otherwise a caller-supplied population with an unroutable dose aborts the
    // process from inside the walk, from a `Result`-returning API. Matches the
    // `Result` form used at the adaptive chokepoint rather than the
    // `predict()`/`simulate()` panic.
    crate::diagnostics::first_error(&crate::api::check_dose_compartments(model_ref, pop_ref))?;

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
    // Reconstruct the model parameters for the covariance step. The `omega` a fit
    // reports is `L·Lᵀ`; rebuilding an `OmegaMatrix` from that matrix re-decomposes
    // it (`chol(omega)`), and the resulting `Ω⁻¹` / `log|Ω|` differ from the ones the
    // inline covariance step used (built directly from the optimizer's exact Cholesky
    // factor `L`) by ~machine-epsilon. That difference feeds the inner NLL penalty
    // `½ηᵀΩ⁻¹η`, shifts the reconverged EBEs at each FD point, and the FD Hessian
    // amplifies it — badly on ill-conditioned ω directions (the #816-review
    // divergence: ~0.1 on a warfarin ω²(KA) with ~115% RSE).
    //
    // So when the fit carries the optimizer's exact packed vector (`packed_estimate`,
    // the in-memory packed-Cholesky-space path), **unpack it** to rebuild the
    // parameters — the `OmegaMatrix` is then built from that same `L`
    // (`from_chol_factor`), bit-for-bit identical to the inline path, and the
    // covariance reproduces exactly. The fallback re-decomposes from `omega`
    // (reloaded `.fitrx` fits, or SAEM/importance-sampling/Bayes, whose inline step
    // re-packs `omega` the same way and so already agrees). The length guard falls
    // back on any model/dimension mismatch.
    //
    // Note: on the reuse arm `packed_estimate` is the source of truth for the
    // numeric center — `base_params` supplies only the structural template
    // (names/masks/bounds/diagonal), and `compute_covariance` reads all θ/Ω/σ
    // values from `x_hat`. A `FitResult` whose `theta`/`omega`/`sigma` were mutated
    // in-process *after* the fit is therefore evaluated at the original packed
    // point; recompute the fit rather than editing its estimates in place.
    let base_params = fitted_params_from_result(fit, model_ref);
    let (params, x_hat) = match &fit.packed_estimate {
        // Alloc-free length guard (`packed_len`, not `pack_params(..).len()`);
        // `pack_params` is only needed on the fallback arm, as the actual `x_hat`.
        Some(v) if v.len() == packed_len(&base_params) => {
            (unpack_params(v, &base_params), v.clone())
        }
        _ => {
            let repacked = pack_params(&base_params);
            (base_params, repacked)
        }
    };
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
            // Cold reconvergence: match the fit's inner multi-start so the EBEs
            // land in the same basin (else SEs would differ from the inline path).
            options.inner_restarts,
        );

    // --- Run the covariance step (UNGATED: calling `run_covariance` IS the
    // request to run it, so it deliberately ignores `options.run_covariance_step`;
    // hence `run_covariance_step_inner`, not the gated `run_covariance_step`). The
    // `FailedNonPd` proposal is only useful to the SIR fallback, a separate step
    // here — callers wanting it run `run_sir` afterwards — so it is discarded.
    let CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: new_warnings,
        sir_fallback_proposal: _,
    } = run_covariance_step_inner(
        &x_hat,
        &params,
        model_ref,
        pop_ref,
        &eta_hats,
        &h_matrices,
        &kappas,
        options,
        None,
    );

    // --- Build the refreshed FitResult ------------------------------------
    let (se_theta, se_omega, se_sigma, se_kappa) =
        extract_standard_errors(&covariance_matrix, &params);
    let se_residual_correlations =
        crate::api::extract_residual_correlation_se(&covariance_matrix, &params);
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
    out.se_residual_correlations = se_residual_correlations;
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

    /// IOV analogue of `run_covariance_matches_inline_covariance` (#823). The
    /// reuse path packs/unpacks the `omega_iov` Cholesky block too, and the
    /// **diagonal-IOV branch of `unpack_params` reconstructs it through
    /// `OmegaMatrix::from_diagonal` (square-then-re-decompose)** rather than the
    /// `from_chol_factor` route the BSV `omega` takes — a distinct construction
    /// with no prior parity coverage.
    ///
    /// It is nonetheless **bit-for-bit**: both the inline covariance step
    /// (`compute_covariance(&x0, …)`) and this standalone step run the entire
    /// numeric path — the base OFV, every FD-Hessian perturbation, and
    /// `se_kappa`'s `iov.matrix[(i,i)]` factor — through the *same*
    /// `unpack_params(x0, template)` on the *same* packed vector
    /// `fit.packed_estimate`. So the `from_diagonal` construction is applied
    /// identically on both sides and cannot introduce a divergence; the
    /// asymmetry the issue flags lives inside `unpack_params`, not between the
    /// two callers. Observed `max_abs_diff == 0.0`, with `se_kappa` matching to
    /// the last bit.
    ///
    /// `fit_from_files` can't thread `iov_column` from `[fit_options]` (it
    /// passes `None`), so — like every other IOV test — this drives the direct
    /// `fit()` API with `read_nonmem_csv(.., Some("OCC"))` and hands both
    /// `Some(model)` and `Some(pop)` to `run_covariance` (the documented IOV
    /// workaround; a bare `Some(model)` for an IOV model is an error).
    #[test]
    fn run_covariance_matches_inline_covariance_iov() {
        use crate::api::fit;

        let model = crate::parser::model_parser::parse_full_model_file(std::path::Path::new(
            "examples/warfarin_iov.ferx",
        ))
        .expect("parse warfarin_iov.ferx")
        .model;
        assert!(model.n_kappa > 0, "warfarin_iov.ferx must declare kappa");
        let pop = crate::io::datareader::read_nonmem_csv(
            std::path::Path::new("data/warfarin_iov.csv"),
            None,
            Some("OCC"),
        )
        .expect("read warfarin_iov.csv");

        // FOCEI (the IOV case the issue calls for) on the deterministic BOBYQA
        // outer optimizer, so fit A and fit B converge to the same packed point.
        let opts = FitOptions {
            method: crate::types::EstimationMethod::FoceI,
            interaction: true,
            ..quick_opts()
        };

        // Fit A: inline covariance step. Skip (like the non-IOV sibling) if the
        // FD cov step didn't produce a matrix — the parity assertions need a
        // non-None reference, and FD conditioning is out of scope here.
        let fit_a = fit(&model, &pop, &model.default_params, &opts).expect("iov fit A converges");
        if fit_a.covariance_matrix.is_none() {
            eprintln!(
                "[skip] inline IOV covariance step produced no matrix (FD instability); \
                 skipping run_covariance IOV parity assertions"
            );
            return;
        }
        // The whole point of the IOV variant: se_kappa must actually be exercised.
        assert!(
            fit_a.se_kappa.is_some(),
            "inline IOV cov step must populate se_kappa"
        );

        // Fit B: identical settings, no inline covariance step.
        let fit_b = fit(
            &model,
            &pop,
            &model.default_params,
            &FitOptions {
                run_covariance_step: false,
                ..opts.clone()
            },
        )
        .expect("iov fit B converges");
        assert!(
            fit_b.covariance_matrix.is_none(),
            "fit B should carry no covariance (run_covariance_step = false)"
        );
        // The bit-exact reuse relies on the FOCEI fit carrying the packed vector;
        // guard it so a regression that stops populating it can't silently drop
        // run_covariance onto the divergent re-decomposition fallback.
        assert!(
            fit_b.packed_estimate.is_some(),
            "an IOV FOCEI fit must carry packed_estimate for run_covariance to reuse"
        );

        let out = run_covariance(&fit_b, Some(&model), Some(&pop), &opts)
            .expect("run_covariance succeeds for the IOV model");

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
        assert!(
            max_abs_diff < 1e-12,
            "IOV run_covariance matrix diverged from inline cov (max abs diff {max_abs_diff})"
        );

        // Every SE vector must agree bit-for-bit — se_kappa is the one the IOV
        // reuse path (`from_diagonal` round-trip) newly covers.
        let se_eq = |label: &str, a: &Option<Vec<f64>>, b: &Option<Vec<f64>>| {
            assert_eq!(a.is_some(), b.is_some(), "{label}: presence differs");
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(a.len(), b.len(), "{label}: length differs");
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() < 1e-12, "{label} diverged: {x} vs {y}");
                }
            }
        };
        se_eq("se_theta", &fit_a.se_theta, &out.se_theta);
        se_eq("se_omega", &fit_a.se_omega, &out.se_omega);
        se_eq("se_sigma", &fit_a.se_sigma, &out.se_sigma);
        se_eq("se_kappa", &fit_a.se_kappa, &out.se_kappa);

        // Non-covariance fields round-trip unchanged — including omega_iov.
        assert_eq!(out.theta, fit_b.theta);
        assert_eq!(out.omega, fit_b.omega);
        assert_eq!(out.omega_iov, fit_b.omega_iov);
        assert_eq!(out.ofv, fit_b.ofv);
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
        // `run_covariance` reuses the optimizer's exact packed vector
        // (`fit.packed_estimate`) to rebuild the parameters, so its `OmegaMatrix`
        // `Ω⁻¹` / `log|Ω|` are built from the same Cholesky factor `L` the inline
        // step used — the covariance is now reproduced **bit-for-bit** (observed
        // `max_abs_diff == 0.0`). Before this fix the two diverged by re-decomposing
        // `omega` (`chol(L·Lᵀ) ≠ L` to machine-ε), amplified by the FD Hessian to
        // ~0.1 on the warfarin ω²(KA) direction (~115% RSE). The bound is far below
        // any real regression while tolerating theoretical last-ULP parallelism noise.
        assert!(
            max_abs_diff < 1e-12,
            "run_covariance matrix diverged from inline cov (max abs diff {max_abs_diff})"
        );

        // Guard the propagation: the bit-exact reproduction relies on the FOCEI fit
        // carrying the optimizer's packed vector. If that stops being populated, the
        // covariance silently falls back to the ~1e-1-divergent re-decomposition path
        // — so assert it is present rather than let the fix rot.
        assert!(
            fit_b.packed_estimate.is_some(),
            "a FOCEI fit must carry packed_estimate for run_covariance to reuse"
        );

        // SEs are derived from the covariance, so they must agree bit-for-bit too.
        assert_eq!(out.se_theta.is_some(), fit_a.se_theta.is_some());
        if let (Some(a), Some(b)) = (&fit_a.se_theta, &out.se_theta) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-12, "se_theta diverged: {x} vs {y}");
            }
        }

        // Non-covariance fields round-trip unchanged.
        assert_eq!(out.theta, fit_b.theta);
        assert_eq!(out.omega, fit_b.omega);
        assert_eq!(out.ofv, fit_b.ofv);
    }

    /// Cover the re-decomposition **fallback** arm (`fit.packed_estimate == None`) —
    /// the path taken by `.fitrx`-reloaded fits and by SAEM / importance-sampling /
    /// Bayes. The bit-exact reuse test above only exercises the `Some` arm (every
    /// in-memory packed-space fit now carries the vector), so null it here to force
    /// the fallback and keep it from silently rotting. This path is *not* bit-exact:
    /// it re-decomposes `chol(fit.omega)` instead of reusing the exact `L`, so it
    /// matches the inline step only up to the FD-amplified re-decomposition
    /// divergence this PR documents (~0.1 on the ill-conditioned warfarin ω²(KA)
    /// direction). The assertions therefore check a *valid* covariance and guard
    /// against gross regressions, not exactness.
    #[test]
    fn run_covariance_fallback_without_packed_estimate() {
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

        let mut fit_b = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(FitOptions {
                run_covariance_step: false,
                ..opts.clone()
            }),
        )
        .expect("fit must converge");

        // Force the fallback: drop the packed vector so run_covariance rebuilds the
        // parameters by re-decomposing `fit.omega` (exactly the reloaded-fit path).
        fit_b.packed_estimate = None;
        let out = run_covariance(&fit_b, None, None, &opts).expect("run_covariance succeeds");

        assert_eq!(out.covariance_status, CovarianceStatus::Computed);
        let cov_ref = fit_a.covariance_matrix.as_ref().unwrap();
        let cov_new = out
            .covariance_matrix
            .as_ref()
            .expect("fallback run_covariance populated covariance_matrix");
        assert_eq!(cov_ref.shape(), cov_new.shape());

        // Valid variances → real SEs: catches a fallback that panics, returns the
        // wrong converged point, or yields a garbage/indefinite matrix.
        for i in 0..cov_new.nrows() {
            let var = cov_new[(i, i)];
            assert!(
                var.is_finite() && var > 0.0,
                "fallback covariance diagonal {i} is not a valid variance: {var}"
            );
        }

        // Loose agreement with the inline step: the re-decomposition divergence is
        // ~0.1 here, so this only guards against gross regressions, not the
        // bit-for-bit exactness the reuse arm provides.
        let max_abs_diff = cov_ref
            .iter()
            .zip(cov_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_diff < 0.5,
            "fallback run_covariance grossly diverged from inline cov (max abs diff {max_abs_diff})"
        );
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
