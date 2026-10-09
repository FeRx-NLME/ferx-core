//! Standalone covariance step — run the FD-Hessian covariance step against an
//! existing `FitResult` without re-fitting.
//!
//! Mirrors the covariance step that [`fit()`](crate::api::fit) runs inline when
//! `options.run_covariance_step = true`, but lets callers drive it after a fit
//! has completed (potentially from a different session, loaded via `.fitrx`).
//! This is the covariance-step analogue of
//! [`run_sir`](crate::estimation::run_sir::run_sir).
//!
//! Input rules (shared with `run_sir` through
//! `estimation::fit_inputs::resolve_fit_inputs`, #1622):
//! - `None` re-reads from `fit.model_path` / `fit.data_path`, reads the data the
//!   way the fit did (`[data_selection]` included) and binds the model from
//!   `fit.data_bindings`. If a stored hash exists, a mismatch is a **hard error**.
//! - A supplied `model` / `population` is not hash-checked (the in-memory values
//!   don't carry their source bytes), but must be the fitted one: the fit's
//!   bindings and θ count, the fit's subjects in the fit's order.
//! - A fit that records its reader settings (#1685) re-reads `fit.data_path` with
//!   them, so `Some(model)` with `population = None` does not read the model file.
//!   An older fit takes the model file's settings, which with `Some(model)` means
//!   reading the hash-verified model file; pass both to avoid it.
//! - A fit that carries a population fingerprint (every fit since #1685) refuses
//!   any population, supplied or re-read, that is not the one it was given.

use crate::api::{cov_diagnostics, extract_standard_errors, resolve_covariance_status};
use crate::estimation::covariance::{run_covariance_step_inner, CovStepOutcome};
use crate::estimation::parameterization::{compute_mu_k, pack_params, unpack_params};
use crate::estimation::uncertainty_samples::{
    fit_packed_estimate, fitted_params_from_result, PackedEstimate,
};
use crate::types::*;

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
/// ([`FitResult::packed_estimate`] — a FOCE / FOCEI / Laplace fit, in memory or
/// reloaded from a `.fitrx` bundle [`save_fit`](crate::io::fitrx::save_fit) wrote,
/// #1815), the match is **bit-for-bit**: the parameters are rebuilt by *unpacking*
/// that vector, so the `OmegaMatrix`'s `Ω⁻¹` / `log|Ω|` come from the same Cholesky
/// factor `L` the inline step used. Reconstructing them from `fit.omega` instead
/// re-decomposes `chol(L·Lᵀ) ≠ L` to machine-ε; that tiny `Ω⁻¹` difference feeds the
/// inner NLL penalty, shifts the reconverged EBEs, and the FD Hessian amplifies it —
/// up to ~1e-1 on an ill-conditioned ω direction (the divergence #816's review
/// surfaced). The vector is reused only when it unpacks bit-for-bit to the fit's
/// reported θ / Ω / σ / Ω_IOV / ρ under `model`; a fit whose estimates were edited
/// after the fit, or a model of the same packed length with a different layout, is
/// evaluated at the reported estimates instead. A fit with no vector — SAEM /
/// importance-sampling / Bayes, a fit built in R, a bundle saved before #1815 or
/// written by ferx-r — takes that re-decomposition fallback too. Pure Gauss-Newton,
/// SAEM, importance sampling and VI are not bit-for-bit with their inline step even
/// in memory (#1847).
///
/// # Failure semantics
///
/// A covariance step that runs but cannot produce a usable matrix (a
/// structurally-unusable or non-positive-definite FD Hessian) is **not** an
/// `Err`. Mirroring `fit()`, the returned `FitResult` carries
/// `covariance_matrix = None`, `covariance_status = Failed`, and the diagnostic
/// appended to `warnings`. `Err` is reserved for input problems: a missing /
/// hash-mismatched model or dataset, a dimension mismatch, a population that is
/// not the one the fit was given (#1685), or, on an older fit, an IOV model
/// supplied without its population (see below).
///
/// # IOV models (n_kappa > 0)
///
/// As with `run_sir`, re-reading the dataset for an IOV model requires the
/// `iov_column` the fit read with, which does not survive on a `CompiledModel`. A
/// fit that records its reader settings (#1685) carries it, so every cell runs.
/// On an older fit, `None` for both `model` and `population` parses the full model
/// file and threads its `iov_column` into the reader, while `Some(model)` for an
/// IOV model with `population = None` is an error rather than read occasions with
/// an `iov_column` the supplied model may not share: the model carries none, and
/// the model file's need not be the one it was built with. Workaround: pass both
/// `Some(model)` and `Some(population)`.
///
/// # Arguments
/// - `fit`: the maximum-likelihood fit to compute a covariance for.
/// - `model`: pre-compiled model. When `None`, re-parsed from `fit.model_path`.
/// - `population`: dataset. When `None`, re-read from `fit.data_path` (with the
///   `iov_column` constraint above for IOV models), routed by the model so a
///   joint model's event rows come back as event records (#1199). A supplied
///   population read without that routing is rejected (`E_ENDPOINT_UNROUTED`).
/// - `options`: the step's own settings — `covariance_method`, `fd_hessian_step`,
///   `analytic_cov_hessian`, `cov_inner_tol` — and `cancel` are the
///   caller's, as given. The scoring settings in [`ScoringSettings`](crate::ScoringSettings)
///   (`inner_maxiter`, `inner_tol`, `inner_restarts`, `mu_referencing`, `n_agq`,
///   `inner_optimizer`, `ebe_warm_start` and the six `ode_*` overrides) are **taken from
///   `fit.scoring_settings` wherever the caller leaves them at their
///   [`FitOptions::default`] value** (#426), so `run_covariance` with default options
///   reconverges the EBEs and scores the objective as the fit's inline step did. The test
///   is by value, as in [`run_sir`](crate::run_sir): an explicit default
///   (`inner_optimizer = Auto` on a fit recorded `lbfgs`) cannot be told from an unset one
///   and yields to the record; to override with a default value, clear or edit
///   `fit.scoring_settings` first. A fit without that record (a `.fitrx` written before
///   #426) falls back to `fit.sir_settings.scoring`, then to `options` as given. The step
///   settings are not recorded: on a fit that ran a non-default `covariance_method`,
///   default options compute the default step, not the fit's. `run_covariance_step` on
///   `options` is **ignored** — calling this function is itself the request to run the
///   step. `method` and `interaction` are not read from `options` either: they come from
///   the fit (`fit.method`, then `fit.interaction`), so the Hessian is taken of the
///   objective the estimates minimise (#1710, #1755). A `[mixture]` fit's per-class
///   overrides are rebuilt from the fit, with the same error as `run_sir` when they are
///   not available (#1704). `iov_occasion` is not read either: the population is
///   prepared as `fit()` prepared it — occasions derived under the fit's recorded rule
///   (`fit.iov_occasion`), DV log-transformed for a `log(DV) ~ …` model (#1783).
pub fn run_covariance(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, crate::diagnostics::EngineError> {
    // #426: the scoring settings the fit's objective ran under, wherever the caller left the
    // default. Resolved first, so the scope below carries the resolved inner and ODE settings
    // to every reconvergence, and the step differentiates the objective the fit minimised.
    let options = &crate::estimation::fit_inputs::resolve_scoring_options(fit, options);
    // #1710: differentiate the fit's own marginal, whatever `interaction` the caller carries.
    let options = &crate::estimation::fit_inputs::fitted_marginal_options(fit, options);
    // #1212, same last hop as `fit()`: this call's `ode_reltol` / `ode_method` / … have to
    // reach the integrator, and the spec they would otherwise be read off carries the
    // parse-time values. It matters more here than almost anywhere else — the covariance step
    // is a second difference of the reconverged OFV, so running it at a *different* accuracy
    // than the fit that produced the estimates is exactly how a plausible-looking standard
    // error comes out wrong. The scope covers the whole call, including the re-parse and
    // re-read paths, and puts the per-subject fan-out on a pool whose workers carry the same
    // settings — arming alone would reach this thread and leave the workers on the model
    // file's. #426: the scope carries this call's `inner_optimizer` / `ebe_warm_start` the
    // same way, so the reconverged EBEs use the resolved inner solver — the same source as
    // `inner_maxiter` / `inner_tol` — and not whatever another fit in the process last set.
    crate::api::with_fit_scope(options, || {
        run_covariance_scoped(fit, model, population, options)
    })?
}

fn run_covariance_scoped(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, crate::diagnostics::EngineError> {
    // Input resolution mirrors `run_sir` exactly: stale-input errors win over
    // any downstream failure so a user pointing at the wrong model/dataset
    // hears about that first.

    // --- Resolve model and population (#1622) ------------------------------
    //
    // Shared with `run_sir`: re-parsed and re-read the way the fit read them —
    // `[data]` renames (#730), `iov_column`, `[data_selection]` — and bound from
    // `fit.data_bindings`.
    let inputs = crate::estimation::fit_inputs::resolve_fit_inputs(
        fit,
        model,
        population,
        "run_covariance",
    )?;
    let model_ref = inputs.model();
    let pop_ref = inputs.population();

    // This entry point re-runs the inner loop (EBEs → the prediction walk), so
    // it needs the same dose-compartment precondition `fit()` enforces (#375) —
    // otherwise a caller-supplied population with an unroutable dose aborts the
    // process from inside the walk, from a `Result`-returning API. Matches the
    // `Result` form every other entry point uses (#898).
    crate::diagnostics::first_error(&crate::api::check_dose_compartments(model_ref, pop_ref))?;
    // …and the endpoint-routing precondition (#1199), as `fit()` enforces it: a
    // population read model-blind carries a joint model's event rows as Gaussian
    // observations, and the covariance step would be taken on the Gaussian half of
    // the likelihood. The re-read above is routed; this covers a supplied population.
    crate::diagnostics::first_error(&crate::api::check_endpoint_routing(
        model_ref, pop_ref, true,
    ))?;

    // --- The shape gate ----------------------------------------------------
    // Ω, σ and Ω_IOV against the model (#1833), before the EBE check below: a fit of a
    // different model has EBEs as wide as its Ω, and must get `E_PARAM_SHAPE`, not the
    // uncoded EBE message. `base_params` is used further down (see the comment there).
    let base_params =
        fitted_params_from_result(fit, model_ref).map_err(|e| e.in_context("run_covariance"))?;

    // --- Sanity-check dimensions ------------------------------------------
    if !fit.subjects.is_empty() && fit.subjects[0].eta.len() != model_ref.n_eta {
        return Err(crate::diagnostics::EngineError::from(format!(
            "fit.subjects[0] has eta dim {} but model has n_eta = {}. \
             Subject EBEs are inconsistent with the supplied model.",
            fit.subjects[0].eta.len(),
            model_ref.n_eta
        ))
        .in_context("run_covariance"));
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
    // So when the fit carries the optimizer's exact packed vector (`packed_estimate`:
    // a packed-Cholesky-space fit, in memory or reloaded from a `.fitrx` bundle
    // `save_fit` wrote, #1815), **unpack it** to rebuild the parameters — the
    // `OmegaMatrix` is then built from that same `L` (`from_chol_factor`),
    // bit-for-bit identical to the inline path, and the covariance reproduces
    // exactly. The vector is reused only when `fit_packed_estimate` finds it
    // `Usable`: this model's length, and an unpack bit-equal to the fit's reported
    // θ/Ω/σ/Ω_IOV/ρ. Anything else — no vector (SAEM/importance-sampling/Bayes, an
    // older or R-written bundle), a different layout, or estimates edited after the
    // fit — re-decomposes from the reported estimates, at the point they say.
    // (`base_params` is built above, ahead of the EBE check, as the shape gate.)
    let (params, x_hat) = match fit_packed_estimate(fit, &base_params) {
        PackedEstimate::Usable(v) => (unpack_params(v, &base_params), v.to_vec()),
        PackedEstimate::Absent | PackedEstimate::WrongLength(_) | PackedEstimate::Stale => {
            let repacked = pack_params(&base_params);
            (base_params, repacked)
        }
    };
    let mu_k = compute_mu_k(model_ref, &params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _stats, kappas) =
        crate::estimation::inner_optimizer::run_inner_loop_warm_seeded(
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
            // …and the fit's BFGS seed (#1389), for the same reason: the inline final
            // inner loop runs the stage's seed, and a different metric lands a
            // different η̂ at a loose `inner_tol` (measured 1.5e-11 on the warfarin
            // covariance, against the 1e-12 bit-parity bound).
            crate::estimation::inner_optimizer::InnerHessianSeed::for_options(options),
            options,
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
        method: covariance_method,
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
    // #1382: overwrite, never merge. `out` is a clone of the incoming fit, so a
    // label left over from that fit's own covariance step would outlive the
    // matrix it described — and this step may well have routed differently (a
    // caller re-running with `covariance_method = s`, say).
    out.covariance_method = covariance_method;
    out.covariance_status = covariance_status;
    out.covariance_wall_time_secs = covariance_wall_time_secs;
    // The incoming fit's covariance-step warnings describe a covariance step this
    // call has just replaced, so they are dropped before the new ones are added
    // (#1382 review). Measured on the 12-subject fixture: re-running an `r` fit
    // under `s` otherwise returns `covariance_method = s` beside a retained
    // "eigenvalue floor applied to FD Hessian" entry whose payload still reads
    // `{"covariance_method": "r", "condition_number": 4.09e9}` — a message about a
    // step that no longer exists (that floor warning cannot even fire under the
    // cross-product), carrying the stale condition number next to a `cov_condition_number`
    // field this function recomputed. Re-enriching it instead of dropping it would
    // be worse: it would stamp the `r` step's message with the `s` step's numbers.
    //
    // Scoped to the four covariance-step codes — exactly what `new_warnings`
    // replaces. The SE-derived post-fit diagnostics (`inflated_rse`,
    // `high_correlation`) are also stale after a re-run, but `run_covariance` does
    // not re-derive them at all; refreshing those is `fit_inner`'s postfit pass and
    // a separate change.
    out.warnings.retain(|w| !is_covariance_step_warning(w));
    out.warnings.extend(new_warnings);
    inputs.note_warnings(&mut out.warnings);
    // Rebuild so the machine-readable payloads agree with the fields above: the
    // entries are keyed by message, so the surviving native ones are preserved and
    // the new covariance warnings get `details` sourced from the *refreshed*
    // `cov_condition_number` / `cov_eigenvalues` / `covariance_method`. Without
    // this the new warnings reach `warnings` with no structured entry at all.
    crate::api::rebuild_warnings_structured(&mut out);
    Ok(out)
}

/// Does this warning describe the covariance step itself — the step
/// [`run_covariance`] recomputes and whose warnings it therefore replaces?
///
/// Classified rather than matched on prose, so the set tracks
/// [`crate::types::classify_warning`] instead of drifting from it. `Sir` is
/// deliberately absent: `run_covariance` discards the SIR fallback proposal and
/// never runs the sampler, so a SIR warning is not this function's to replace.
fn is_covariance_step_warning(msg: &str) -> bool {
    matches!(
        crate::types::classify_warning(msg).category,
        WarningCode::CovarianceStep
            | WarningCode::CovarianceFailed
            | WarningCode::CovarianceRegularized
            | WarningCode::ConditionNumber
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fit_from_files;

    // In-tree warfarin example + data (see AGENTS.md). Tests run from the crate
    // root, so relative paths work directly.
    const MODEL_PATH: &str = "examples/warfarin.ferx";
    const DATA_PATH: &str = "data/warfarin.csv";

    /// #1512: the model-selection strictness gate fails any fit carrying the eigenvalue-floor
    /// warning, so a re-run that replaces the step (an `r` fit re-run under `s`, whose `S⁻¹`
    /// never floors) must drop the old one — or the gate would exclude the fit on a floor that
    /// no longer describes its matrix. Pinned on the real message, chain prefix included.
    #[test]
    fn the_eigenvalue_floor_warning_is_superseded_by_a_rerun() {
        use crate::estimation::cov_diagnostics::{
            format_regularized_warning, CovHessianSource, CovRegularizationFacts,
        };
        let msg = format_regularized_warning(&CovRegularizationFacts {
            source: CovHessianSource::FdStencil,
            n_clipped: 1,
            n_free: 9,
            min_eigenvalue: 2.028e-7,
            max_eigenvalue: 2.285e3,
            floor: 2.285e-7,
            variance_inflation: 1.190e3,
            declines: &[],
            ode: None,
        });
        assert!(is_covariance_step_warning(&msg), "{msg}");
        assert!(is_covariance_step_warning(&format!("[FOCEI] {msg}")));
    }

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
            err.to_string().contains("model hash mismatch"),
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
            err.to_string().contains("data hash mismatch"),
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
            err.to_string().contains("no model supplied"),
            "expected 'no model supplied' error, got: {}",
            err
        );
    }

    #[test]
    fn run_covariance_errors_when_iov_model_supplied_without_population() {
        // Some(model) for an IOV (n_kappa > 0) model but None population: must
        // refuse rather than re-read data without iov_column. The IOV check
        // fires before any dimension check, so the shape-mismatched fit is fine.
        // #1685 T8: on a **legacy** fit — no recorded reader settings or
        // fingerprint, as an older `.fitrx` — the refusal stands; a fit that
        // records its `iov_column` runs instead (T9).
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let mut fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");
        fit.reader_settings = None;
        fit.population_fingerprint = None;

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
            err.to_string().contains("IOV") && err.to_string().contains("population"),
            "expected IOV-needs-population error, got: {}",
            err
        );
    }
}

/// #1622: `run_covariance` runs on the model **as fitted**, whichever inputs the
/// caller supplies. Analytic two-compartment oral fixture (`fit_inputs::test_fixtures`),
/// FOCEI with the analytic Dual2 inner gradient; the covariance is the FD-of-OFV
/// Hessian. The oracle is the call with `Some(model), Some(population)` from
/// `prepare_run`, the model bound on the fit's own data.
#[cfg(test)]
mod from_fit_bindings {
    use super::*;
    use crate::estimation::fit_inputs::test_fixtures::{case, design, unbound, Case, Kind};

    fn bits(m: &nalgebra::DMatrix<f64>) -> Vec<u64> {
        m.iter().map(|x| x.to_bits()).collect()
    }

    fn oracle(c: &Case) -> FitResult {
        run_covariance(
            &c.fit,
            Some(&c.prep.parsed.model),
            Some(&c.prep.population),
            &c.opts,
        )
        .expect("the oracle runs")
    }

    fn assert_same_covariance(got: &FitResult, want: &FitResult, what: &str) {
        let want_cov = want
            .covariance_matrix
            .as_ref()
            .unwrap_or_else(|| panic!("{what}: the oracle has a covariance"));
        let got_cov = got
            .covariance_matrix
            .as_ref()
            .unwrap_or_else(|| panic!("{what}: no covariance matrix; warnings {:?}", got.warnings));
        assert_eq!(got_cov.shape(), want_cov.shape(), "{what}");
        assert_eq!(bits(got_cov), bits(want_cov), "{what}: covariance bits");
        assert_eq!(got.covariance_status, CovarianceStatus::Computed, "{what}");
        let se = |f: &FitResult| {
            f.se_theta
                .as_ref()
                .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
        };
        assert_eq!(se(got), se(want), "{what}: se_theta bits");
    }

    /// T6's oracle on one kind: `None, None` equals the call with `prepare_run`'s
    /// bound model and population to the bit. The fixture is checked to be live first:
    /// the median relation is unresolved on the bare parse, the level block moves
    /// `n_theta`, and the selection drops a subject, so a re-read that skipped any of
    /// them would have something to differ on.
    fn assert_re_read_matches(kind: Kind) {
        {
            let c = case(kind);
            let bare = unbound(&c);
            let symbolic = bare
                .covariate_model
                .as_ref()
                .is_some_and(|s| !s.unresolved().is_empty());
            assert_eq!(
                symbolic,
                matches!(kind, Kind::Median | Kind::LevelMedian),
                "{kind:?}: the bare parse leaves the median relation unresolved"
            );
            let level = matches!(kind, Kind::Level | Kind::LevelMedian);
            assert_eq!(c.fit.theta.len(), if level { 8 } else { 6 }, "{kind:?}");
            assert_eq!(
                bare.n_theta != c.fit.theta.len(),
                level,
                "{kind:?}: an unbound level block has fewer θ than the fit"
            );
            assert_eq!(
                c.fit.subjects.len(),
                if kind == Kind::Select { 29 } else { 30 },
                "{kind:?}: [data_selection] dropped subject 3"
            );

            let want = oracle(&c);
            let got = run_covariance(&c.fit, None, None, &c.opts)
                .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            assert_same_covariance(&got, &want, &format!("{kind:?} None/None"));
            // #1685 T10: `(Some, None)` re-reads with the recorded settings now, not
            // the model file's; the result is the same to the bit.
            let got = run_covariance(&c.fit, Some(&c.prep.parsed.model), None, &c.opts)
                .unwrap_or_else(|e| panic!("{kind:?} Some/None: {e}"));
            assert_same_covariance(&got, &want, &format!("{kind:?} Some/None"));
        }
    }

    /// T6, per PR: the two kinds each of the resolver's two halves is visible on.
    ///
    /// Mutations — skip the bind in `resolve_fit_inputs`: `Median` returns `Ok` with
    /// no covariance; read with `read_population_routed_by` again: `Select` is
    /// refused on the subject count (and, without that guard, measured worst
    /// relative difference 20.0).
    #[test]
    fn the_re_read_binds_and_filters_as_the_fit_did() {
        assert_re_read_matches(Kind::Median);
        assert_re_read_matches(Kind::Select);
    }

    /// T6, every kind (Tier 3: five fits and ten covariance steps, which ran past
    /// 60 s under the coverage build of `Tests + coverage (core)`). Adds the plain
    /// control and the level kinds, where skipping the bind is refused on `n_theta`.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn the_re_read_equals_the_model_bound_on_the_fit_data() {
        for kind in [
            Kind::Plain,
            Kind::Median,
            Kind::Level,
            Kind::LevelMedian,
            Kind::Select,
            Kind::DoseFilter,
        ] {
            assert_re_read_matches(kind);
        }
    }

    /// T8. A fit that carries no bindings (an older `.fitrx`, here a live fit with
    /// `data_bindings` cleared): a model that needs bindings is refused with the
    /// binder's text behind the entry prefix — not re-bound on the data, and not run
    /// unbound. (A plain model records no bindings at all, so T6's `Plain` kind is
    /// the unaffected control.)
    ///
    /// Mutation — drop the empty-bindings refusal in `bind_from_fit`: `Level` panics
    /// or is refused on `n_theta`, `Median` is refused on its missing statistic; each
    /// assertion on the text dies.
    #[test]
    fn a_fit_without_bindings_is_refused_when_the_model_needs_them() {
        for (kind, has_level, has_stat) in [
            (Kind::Level, true, false),
            (Kind::Median, false, true),
            (Kind::LevelMedian, true, true),
        ] {
            let mut c = case(kind);
            assert!(!c.fit.data_bindings.is_empty(), "{kind:?}");
            c.fit.data_bindings = Default::default();
            let err = run_covariance(&c.fit, None, None, &c.opts)
                .map(|_| ())
                .expect_err("refused");
            assert!(
                err.to_string().starts_with(
                    "run_covariance: this fit carries no data-derived bindings, so the model \
                     cannot be rebuilt the way it was fitted: "
                ),
                "{kind:?}: {err}"
            );
            assert_eq!(
                err.to_string()
                    .contains("its theta level block(s) `SHIFT[STUDY]`"),
                has_level,
                "{kind:?}: {err}"
            );
            assert_eq!(
                err.to_string().contains("a statistic of `WT` symbolically"),
                has_stat,
                "{kind:?}: {err}"
            );
            assert!(
                err.to_string().ends_with(
                    "The fit is an older `.fitrx` bundle, or was made before ferx recorded \
                     these bindings with a fit. Refit the model to record them."
                ),
                "{kind:?}: {err}"
            );
        }
    }

    /// T9. The `Some(model)` cells. Each was a panic or an `Ok` without covariance
    /// before #1622:
    ///
    /// - an unbound level model is refused for not carrying the fit's bindings, and
    ///   — when the fit carries none to compare — for its θ count (was an index panic);
    /// - a bound level model without a population is bit-identical to the oracle
    ///   (was `Ok` with no covariance: the re-read had no level index column);
    /// - an unbound median model is refused (was `Ok` with no covariance).
    ///
    /// Mutations — delete the bindings comparison, the `n_theta` check, the index
    /// write, or the `check_covariate_model_bound` call: one cell each panics, or
    /// returns `Ok`, and dies.
    #[test]
    fn a_supplied_model_must_be_the_fitted_one() {
        let level = case(Kind::Level);
        let bare = unbound(&level);
        let err = run_covariance(
            &level.fit,
            Some(&bare),
            Some(&level.prep.population),
            &level.opts,
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "run_covariance: the supplied model is not bound with this fit's bindings: its \
             data-derived bindings (level layout, covariate statistics) differ from the fit's \
             `data_bindings`. Pass `model = None` to rebuild it from the fit, or bind it with \
             `ferx_core::api::bind_from_fit` and the fit's `data_bindings`."
        );

        let mut no_bindings = level.fit.clone();
        no_bindings.data_bindings = Default::default();
        let err = run_covariance(
            &no_bindings,
            Some(&bare),
            Some(&level.prep.population),
            &level.opts,
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "run_covariance: the model has n_theta = 6 but the fit has 8 θ. Verify you supplied \
             the same model the fit used, bound the way the fit was (`bind_from_fit` with the \
             fit's `data_bindings`)."
        );

        let got = run_covariance(
            &level.fit,
            Some(&level.prep.parsed.model),
            None,
            &level.opts,
        )
        .expect("a bound model re-reads its own population");
        assert_same_covariance(&got, &oracle(&level), "Level Some/None");

        // A population that is not the fit's subjects is refused: the EBEs are matched
        // by position, and the inner loop indexed past them. Both causes, one message
        // each (#1680 review r1, finding 5). First, one subject dropped (the shape a
        // `FitOptions` row filter the file does not state leaves)…
        const WHY: &str = "The fit's EBEs are matched to subjects by position, so the \
                           population must be the one the fit saw.";
        let mut short = level.prep.population.clone();
        short.subjects.pop();
        let err = run_covariance(
            &level.fit,
            Some(&level.prep.parsed.model),
            Some(&short),
            &level.opts,
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("run_covariance: the population has 29 subjects but the fit has 30. {WHY}")
        );
        // #1746: a population that is not the fit's has no `ferx check` code.
        assert_eq!(err.code(), None, "{err}");
        assert_eq!(err.context(), Some("run_covariance"), "{err}");
        // …then the right count in the wrong order, naming the first position.
        let mut swapped = level.prep.population.clone();
        swapped.subjects.swap(1, 2);
        let err = run_covariance(
            &level.fit,
            Some(&level.prep.parsed.model),
            Some(&swapped),
            &level.opts,
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "run_covariance: subject 2 of the population is `3`, but the fit's is `2`. {WHY}"
            )
        );
        // Finding 7: checked before binding, so a population with a level the fit
        // never saw hears that it is not the fit's, not the design advice of the
        // unseen-level refusal.
        let mut extra = level.prep.population.clone();
        let mut newcomer = extra.subjects[0].clone();
        newcomer.id = "99".to_string();
        newcomer.covariates.insert("STUDY".to_string(), 4.0);
        extra.subjects.push(newcomer);
        let err = run_covariance(&level.fit, None, Some(&extra), &level.opts)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("run_covariance: the population has 31 subjects but the fit has 30. {WHY}")
        );

        let median = case(Kind::Median);
        let bare = unbound(&median);
        let err = run_covariance(
            &median.fit,
            Some(&bare),
            Some(&median.prep.population),
            &median.opts,
        )
        .map(|_| ())
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("is not bound with this fit's bindings"),
            "{err}"
        );
        let mut no_bindings = median.fit.clone();
        no_bindings.data_bindings = Default::default();
        let err = run_covariance(
            &no_bindings,
            Some(&bare),
            Some(&median.prep.population),
            &median.opts,
        )
        .map(|_| ())
        .unwrap_err();
        assert!(
            err.to_string().starts_with(
                "run_covariance: [covariate_model] relations still need data-derived statistics:"
            ),
            "{err}"
        );
        // #1746 (review r1 #3/#4): an input refusal that `ferx check` codes keeps its
        // code and is attributed to the entry point. Mutation: assert the model bound
        // through `assert_covariate_model_bound(m)?` (String) again → `code()` dies.
        assert_eq!(err.code(), Some("E_COVSTAT_UNRESOLVED"), "{err}");
        assert_eq!(err.context(), Some("run_covariance"), "{err}");
    }

    /// The #1729 refusal, spelled out in full so deleting any sentence of it in
    /// `check_lent_stats` (or swapping a source clause) fails the equality. `levels`:
    /// the model has a theta level block, so only `prepare_run` is advised.
    fn stats_refusal(source: Source, model_median: f64, data_median: f64, levels: bool) -> String {
        let fix = if levels {
            "`prepare_run` on the fit's model and data files. The model has a theta level \
             block, which `bind_covariate_stats` does not bind."
        } else {
            "`prepare_run` on the fit's model and data files, or `bind_covariate_stats` on \
             a freshly parsed model with the fit's population."
        };
        let (against, there) = match source {
            Source::Supplied => ("the supplied population", "the supplied population"),
            _ => ("the fit's data, re-read from `fit.data_path`", "the data"),
        };
        let routed = if source == Source::Routed {
            " The data was re-read without the model file (the fit records no \
             `model_path`), so a `[data_selection]` in it was not applied: if the model \
             has one, pass `population = Some(&pop)` with the fit's population."
        } else {
            ""
        };
        format!(
            "run_covariance: this fit records no data-derived bindings (an older `.fitrx`), \
             so the supplied model's covariate statistics were checked against {against}, \
             and they differ: the median of `WT` is {model_median} in the model but \
             {data_median} in {there}. The model was bound on other data, and scoring the \
             fit's θ with it would centre the relations on that data, not on the data the θ \
             was estimated from. Re-parse the model and bind it on the fit's data: \
             {fix}{routed}"
        )
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Source {
        Supplied,
        /// Re-read with the model file's reader settings.
        ReRead,
        /// Re-read routed by the model (the fit records no `model_path`).
        Routed,
    }

    fn wt_median(m: &CompiledModel) -> f64 {
        m.data_bindings().covariate_stats["WT"].median
    }

    /// #1729 T1, the differential pair. A fit that records no bindings (an older
    /// `.fitrx`; here a live fit with `data_bindings` cleared), lent the `Median`
    /// model two ways: bound on the fit's own data, it runs bit-identical to the
    /// as-fitted `None, None` call, with the population supplied and re-read; bound on
    /// a design (WT × 1.3), it is refused in both cells. Before #1729 the design cells
    /// returned `Ok` with SE(THETA_CL_WT) −64% (measured on the plan).
    ///
    /// Mutations — delete the check: the design cells return `Ok`; make it always
    /// refuse: the fitted cells fail; compare the model's statistics with themselves:
    /// the design cells return `Ok`.
    #[test]
    fn an_empty_bindings_fit_scores_a_lent_model_only_on_its_own_statistics() {
        let c = case(Kind::Median);
        let want = run_covariance(&c.fit, None, None, &c.opts).expect("the as-fitted run");
        let mut fit = c.fit.clone();
        fit.data_bindings = Default::default();

        let fitted = &c.prep.parsed.model;
        for (pop, what) in [(Some(&c.prep.population), "Some"), (None, "None")] {
            let got = run_covariance(&fit, Some(fitted), pop, &c.opts)
                .unwrap_or_else(|e| panic!("fitted/{what}: {e}"));
            assert_same_covariance(&got, &want, &format!("fitted/empty/{what}"));
        }

        let design = design(&c);
        let lent = &design.parsed.model;
        // Live: the design's WT median is not the fit's.
        let (mm, dm) = (wt_median(lent), wt_median(fitted));
        assert_ne!(mm, dm);
        for (pop, source) in [
            (Some(&c.prep.population), Source::Supplied),
            (None, Source::ReRead),
        ] {
            let err = run_covariance(&fit, Some(lent), pop, &c.opts)
                .map(|_| ())
                .expect_err("a design-bound model is refused");
            assert_eq!(
                err.to_string(),
                stats_refusal(source, mm, dm, false),
                "{source:?}"
            );
        }
    }

    /// #1729 T2. `LevelMedian` on a design with a study the fit never saw: with the
    /// population supplied (the fit's index columns), the new check refuses it — it
    /// returned `Ok` with SE(TVCL) +90% before. Re-read, the unseen-level refusal
    /// still comes first: the check sits after the level index write.
    ///
    /// Mutations — delete the check: the `Some` cell returns `Ok`; move the check
    /// before the level index write: the `None` cell's text changes.
    #[test]
    fn a_design_bound_level_model_is_refused_on_its_statistics_after_its_levels() {
        let c = case(Kind::LevelMedian);
        let mut fit = c.fit.clone();
        fit.data_bindings = Default::default();
        let design = design(&c);
        let lent = &design.parsed.model;
        let (mm, dm) = (wt_median(lent), wt_median(&c.prep.parsed.model));

        let err = run_covariance(&fit, Some(lent), Some(&c.prep.population), &c.opts)
            .map(|_| ())
            .expect_err("refused");
        assert_eq!(
            err.to_string(),
            stats_refusal(Source::Supplied, mm, dm, true)
        );

        let err = run_covariance(&fit, Some(lent), None, &c.opts)
            .map(|_| ())
            .expect_err("refused");
        assert!(
            err.to_string().starts_with(
                "run_covariance: theta SHIFT[STUDY]: the design has 1 level(s) the fit \
                 estimated no theta for: `STUDY=3`."
            ),
            "{err}"
        );
    }

    /// #1729 T3, the refusal's input space in one test: the supplied, re-read and
    /// routed wordings (each must name its own source; the routed one adds the
    /// `[data_selection]` caveat), the routed re-read passing the fitted model (the
    /// legitimate path with no model file), a fit with recorded bindings left to them,
    /// and a supplied population that lacks the covariate, refused with the
    /// summariser's text behind the prefix.
    ///
    /// Mutations — swap the source clauses, drop either, or drop the routed caveat:
    /// one equality dies; delete any sentence of the message: every equality dies;
    /// drop the `data_bindings.is_empty()` gate: the recorded-bindings cell is refused.
    #[test]
    fn the_stats_refusal_names_where_the_population_came_from() {
        let c = case(Kind::Median);
        let mut fit = c.fit.clone();
        fit.data_bindings = Default::default();
        let design = design(&c);
        let lent = &design.parsed.model;
        let fitted = &c.prep.parsed.model;
        let (mm, dm) = (wt_median(lent), wt_median(fitted));

        // The routed re-read is the legacy path (#1685): a fit with recorded reader
        // settings re-reads with them, `[data_selection]` included.
        let mut routed = fit.clone();
        routed.model_path = None;
        routed.reader_settings = None;
        routed.population_fingerprint = None;
        let cells = [
            (&fit, Some(&c.prep.population), Source::Supplied),
            (&fit, None, Source::ReRead),
            (&routed, None, Source::Routed),
        ];
        for (f, pop, source) in cells {
            let err = run_covariance(f, Some(lent), pop, &c.opts)
                .map(|_| ())
                .expect_err("refused");
            assert_eq!(
                err.to_string(),
                stats_refusal(source, mm, dm, false),
                "{source:?}"
            );
        }
        // #1685, the other side of the routed caveat's gate: no `model_path`, but
        // recorded settings, so the re-read did apply the selection and says nothing
        // about it.
        let mut recorded = fit.clone();
        recorded.model_path = None;
        let err = run_covariance(&recorded, Some(lent), None, &c.opts)
            .map(|_| ())
            .expect_err("refused");
        assert_eq!(
            err.to_string(),
            stats_refusal(Source::ReRead, mm, dm, false)
        );
        // The supplied and re-read texts differ, so neither equality is the other's.
        assert_ne!(
            stats_refusal(Source::Supplied, mm, dm, false),
            stats_refusal(Source::ReRead, mm, dm, false)
        );

        let got = run_covariance(&routed, Some(fitted), None, &c.opts)
            .expect("the fitted model passes on the routed re-read");
        assert!(got.covariance_matrix.is_some(), "{:?}", got.warnings);

        // A fit that records its bindings decides by them, not by the population:
        // the fitted model on a supplied population with other WT values is not
        // re-summarised. Which population a caller supplies is #1685's question, so
        // this holds on a fit without a population fingerprint (an older `.fitrx`)…
        let mut unprinted = c.fit.clone();
        unprinted.reader_settings = None;
        unprinted.population_fingerprint = None;
        let got = run_covariance(&unprinted, Some(fitted), Some(&design.population), &c.opts)
            .expect("recorded bindings are not re-checked against the population");
        assert!(got.covariance_matrix.is_some(), "{:?}", got.warnings);
        // …and a fit with one refuses the design's population on its covariate values
        // (#1685), not on the statistics.
        let err = run_covariance(&c.fit, Some(fitted), Some(&design.population), &c.opts)
            .map(|_| ())
            .expect_err("not the fit's population");
        assert!(
            err.to_string().starts_with(
                "run_covariance: this population is not the one the fit was given: the \
                 covariate values of subject `1` differ"
            ),
            "{err}"
        );

        let mut absent = c.prep.population.clone();
        for s in &mut absent.subjects {
            s.covariates.remove("WT");
            for snap in s
                .obs_covariates
                .iter_mut()
                .chain(&mut s.dose_covariates)
                .chain(&mut s.pk_only_covariates)
                .chain(&mut s.reset_covariates)
            {
                snap.remove("WT");
            }
        }
        let err = run_covariance(&fit, Some(fitted), Some(&absent), &c.opts)
            .map(|_| ())
            .expect_err("refused");
        assert!(
            err.to_string().starts_with(
                "run_covariance: [covariate_model] needs summary statistics for covariate \
                 `WT`, but the dataset carries no non-missing value for it"
            ),
            "{err}"
        );
    }

    /// #1729 T4, the escape the refusal advises, on both sides of its level-block
    /// gate (#1734 review r1, finding 1). Per kind: the design-bound model's refusal
    /// advises the routes for that kind, and each advised route runs bit-identical to
    /// the as-fitted call. `prepare_run` on the fit's files is advised for both; a
    /// freshly parsed model bound with `bind_covariate_stats` only for `Median`. On
    /// `LevelMedian` that route is refused on its θ count, because the level block
    /// stays unbound, which is why the message does not offer it there.
    /// (`bind_covariate_stats` on the design's already-bound parse is a no-op, which
    /// is why the message says "freshly parsed".)
    ///
    /// Mutations — advise `bind_covariate_stats` for a level model too, or for no
    /// model: one kind's text equality dies.
    #[test]
    fn the_advised_route_runs() {
        for (kind, levels) in [(Kind::Median, false), (Kind::LevelMedian, true)] {
            let c = case(kind);
            let want = run_covariance(&c.fit, None, None, &c.opts).expect("the as-fitted run");
            let mut fit = c.fit.clone();
            fit.data_bindings = Default::default();
            let pop = Some(&c.prep.population);

            let design = design(&c);
            let lent = &design.parsed.model;
            let err = run_covariance(&fit, Some(lent), pop, &c.opts)
                .map(|_| ())
                .expect_err("refused");
            let (mm, dm) = (wt_median(lent), wt_median(&c.prep.parsed.model));
            assert_eq!(
                err.to_string(),
                stats_refusal(Source::Supplied, mm, dm, levels),
                "{kind:?}"
            );

            // `prepare_run` on the fit's files: advised for both kinds.
            let got = run_covariance(&fit, Some(&c.prep.parsed.model), pop, &c.opts)
                .unwrap_or_else(|e| panic!("{kind:?} prepare_run: {e}"));
            assert_same_covariance(&got, &want, &format!("{kind:?} prepare_run"));

            // A fresh parse bound with `bind_covariate_stats`: advised only without levels.
            let text = std::fs::read_to_string(&c.model_path).unwrap();
            let mut parsed =
                crate::parser::model_parser::parse_full_model_file(&c.model_path).unwrap();
            crate::api::bind_covariate_stats(&mut parsed, &text, &c.prep.population).unwrap();
            let fresh = run_covariance(&fit, Some(&parsed.model), pop, &c.opts);
            if levels {
                let err = fresh
                    .map(|_| ())
                    .expect_err("the level block stays unbound");
                assert!(
                    err.to_string()
                        .starts_with("run_covariance: the model has n_theta = 6 but the fit has 8"),
                    "{err}"
                );
            } else {
                let got = fresh.unwrap_or_else(|e| panic!("{kind:?} fresh parse: {e}"));
                assert_same_covariance(&got, &want, &format!("{kind:?} fresh parse"));
            }
        }
    }

    /// #1680 review r1, finding 2: with a supplied model and no population, the model
    /// file is still read — for its reader settings — so a changed or missing file is
    /// refused. Each refusal carries the entry prefix, says why the file is needed and
    /// how not to need it, and following that advice runs.
    ///
    /// Mutations — drop the `because` text from either failure, or the prefix from
    /// the read error: the matching assertion dies.
    ///
    /// #1685 T8: that is the **legacy** contract — a fit with no recorded reader
    /// settings or fingerprint (an older `.fitrx`). A fit that records its settings
    /// re-reads with them, so the model file is not read at all: with the file gone,
    /// `(Some, None)` runs bit-identical to `(Some, Some)`.
    ///
    /// Mutation — read the model file for its settings even when the fit records
    /// them: the last cell is refused on the missing file.
    #[test]
    fn a_supplied_model_reads_the_model_file_only_for_its_reader() {
        const BECAUSE: &str = " The population is re-read with this file's `[data]` \
                               renames and `[data_selection]`, which decide the rows the \
                               fit saw. Pass `population = Some(&pop)` as well, and the \
                               model file is not read.";
        let c = case(Kind::Median);
        let model = &c.prep.parsed.model;
        let mut legacy = c.fit.clone();
        legacy.reader_settings = None;
        legacy.population_fingerprint = None;

        let text = std::fs::read_to_string(&c.model_path).unwrap();
        std::fs::write(&c.model_path, format!("{text}\n# edited after the fit\n")).unwrap();
        let err = run_covariance(&legacy, Some(model), None, &c.opts)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("run_covariance: model hash mismatch for "),
            "{err}"
        );
        // #1746: an input `ferx check` has no code for carries none — "refused, no
        // code" stays distinct from a coded precondition — and is still attributed
        // to the entry point. Mutation: give non-diagnostic errors a generic code →
        // the `None` assert dies.
        assert_eq!(err.code(), None, "{err}");
        assert_eq!(err.context(), Some("run_covariance"), "{err}");
        assert!(
            err.message().starts_with("model hash mismatch for "),
            "{err}"
        );
        assert!(
            err.to_string()
                .ends_with(&format!("refusing to run against stale source.{BECAUSE}")),
            "{err}"
        );

        std::fs::remove_file(&c.model_path).unwrap();
        let err = run_covariance(&legacy, Some(model), None, &c.opts)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("run_covariance: cannot read the model file "),
            "{err}"
        );
        assert!(err.to_string().ends_with(BECAUSE), "{err}");

        // The advice: with the population supplied too, the file is not read.
        let want = run_covariance(&legacy, Some(model), Some(&c.prep.population), &c.opts)
            .expect("no model file needed");
        assert!(want.covariance_matrix.is_some());

        // Recorded settings: the missing file is never opened.
        let got = run_covariance(&c.fit, Some(model), None, &c.opts)
            .expect("a fit with recorded reader settings needs no model file");
        assert_same_covariance(&got, &want, "recorded settings, Some/None");
    }
}

/// #1815 T5: every packed-space engine's own vector passes the bit-equality guard,
/// so none of them is silently dropped onto the re-decomposition fallback. Warfarin
/// at `outer_maxiter = 3` (the guard is a property of pack/unpack at whatever point
/// the engine stopped, not of convergence). The plan's §0d sweep: 337/337 hold over
/// 31 examples × 11 engines; VI fails every one (its stored Ω is 1 ULP off its own
/// unpack — warfarin `0.3360326090905553` vs `…554`), pinned here as `Stale` so a
/// VI fix flips the row and prompts the docs (`foce.qmd`, #1847).
///
/// Mutations — compare the wrong field (e.g. `p.sigma` against `fit.theta`), or
/// return `Stale` unconditionally: every engine row dies; drop the Ω comparison:
/// the VI row dies.
#[cfg(test)]
mod packed_estimate_guard_per_engine {
    use super::*;
    use crate::estimation::uncertainty_samples::{fit_packed_estimate, PackedEstimate};
    use crate::types::{EstimationMethod as M, Optimizer as O};

    #[test]
    fn every_packed_space_engine_is_usable_and_vi_is_stale() {
        let model = crate::parser::model_parser::parse_model_file(std::path::Path::new(
            "examples/warfarin.ferx",
        ))
        .expect("model");
        let cases: [(&str, M, O, bool); 11] = [
            ("auto", M::FoceI, O::Auto, true),
            ("bfgs", M::FoceI, O::Bfgs, true),
            ("slsqp", M::FoceI, O::Slsqp, true),
            ("mma", M::FoceI, O::Mma, true),
            ("bobyqa", M::FoceI, O::Bobyqa, true),
            ("trust_region", M::FoceI, O::TrustRegion, true),
            ("foce", M::Foce, O::Auto, true),
            ("laplace", M::Laplace, O::Auto, true),
            ("gn", M::FoceGn, O::Auto, true),
            ("gn_hybrid", M::FoceGnHybrid, O::Auto, true),
            ("vi", M::Vi, O::Auto, false),
        ];
        let mut wrong = Vec::new();
        for (name, method, optimizer, usable) in cases {
            let opts = FitOptions {
                method,
                optimizer,
                interaction: method != M::Foce,
                outer_maxiter: 3,
                // VI ignores `outer_maxiter`; its default run is ~25 s in debug.
                vi_iters: 20,
                run_covariance_step: false,
                verbose: false,
                ..FitOptions::default()
            };
            let fit = crate::api::fit_from_files(
                "examples/warfarin.ferx",
                Some("data/warfarin.csv"),
                None,
                Some(opts),
            )
            .unwrap_or_else(|e| panic!("{name}: fit failed: {e}"));
            assert!(
                fit.packed_estimate.is_some(),
                "{name}: a packed-space engine must carry packed_estimate"
            );
            let base = fitted_params_from_result(&fit, &model).expect("base");
            let got = fit_packed_estimate(&fit, &base);
            let ok = if usable {
                matches!(got, PackedEstimate::Usable(_))
            } else {
                got == PackedEstimate::Stale
            };
            if !ok {
                wrong.push(format!("{name}: {got:?}"));
            }
        }
        assert!(wrong.is_empty(), "guard misclassified: {wrong:?}");
    }
}
