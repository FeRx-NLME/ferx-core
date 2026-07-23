#![allow(unused_imports)]
//! Extracted verbatim from `api/mod.rs` (production peel). See the module-
//! doc / Key Modules table for the split rationale.
use super::*;
use crate::diagnostics::{first_error, CheckReport, Diagnostic};
use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::parameterization::{
    chol_lt_idx, lower_tri_iter, omega_packed_len, theta_packs_log,
};
use crate::estimation::saem;
use crate::io::datareader::{
    read_nonmem_csv_filtered_mapped, read_nonmem_csv_filtered_tte, read_nonmem_csv_mapped,
    read_nonmem_csv_with_covariates_filtered_mapped, read_nonmem_csv_with_covariates_mapped,
    read_nonmem_csv_with_covariates_tte, SelectionFilter, ERR_COV_MISSING_COLUMNS,
    ERR_COV_NON_NUMERIC,
};
use crate::pk;
use crate::propensity_match::MatchMethod;
use crate::sim::adaptive::{
    AdaptiveRun, AdaptiveSubjectMetrics, ControllerCtx, DecisionLogEntry, DoseAction,
    DoseLedgerEntry, MonitorSpec,
};
use crate::stats::likelihood::{
    build_frem_r_override, compute_cwres, foce_subject_nll, foce_subject_nll_iov,
};
use crate::stats::residual_error::{
    compute_iwres_with_correlations, compute_r_matrix_with_correlations,
    compute_r_matrix_with_correlations_scaled, iwres_autocorrelation,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Rebuild `warnings_structured` from the current `warnings` vec.
///
/// Called after all late-injected warnings (LTBS splice, multi-start metadata)
/// have been appended so the structured field is always in sync with the flat list.
///
/// Each message is classified into a typed [`crate::types::WarningCode`] +
/// severity and — for the numeric diagnostics whose values are already stored
/// as typed fields on the `FitResult` — enriched with a machine-readable
/// `details` payload sourced from those fields (issue #781, increment 1). The
/// numbers thus come from typed state, not from re-parsing the message prose;
/// converting the remaining ~80 string push-sites to full at-source emission is
/// a later increment.
pub(crate) fn rebuild_warnings_structured(result: &mut FitResult) {
    // Entries emitted typed at source (`native_warnings`, already carrying a
    // `details` payload) are keyed by message and take precedence; every other
    // message is string-classified and then field-enriched. Keying by message
    // keeps this idempotent — a second rebuild re-captures the native entries.
    // Key by the *original* warning string: `classify_warning` strips a
    // `[METHOD]` chain prefix into `source_method`, so a native entry produced
    // in a multi-stage chain must reconstruct that prefix to match the raw
    // `warnings` entry (an unprefixed entry keys by its message unchanged).
    let native: std::collections::HashMap<String, crate::types::WarningEntry> =
        std::mem::take(&mut result.warnings_structured)
            .into_iter()
            .map(|e| {
                let key = match &e.source_method {
                    Some(m) => format!("[{m}] {}", e.message),
                    None => e.message.clone(),
                };
                (key, e)
            })
            .collect();
    let stats = DiagStats {
        dw_statistic: result.dw_statistic,
        iwres_lag1_r: result.iwres_lag1_r,
        shrinkage_eps: result.shrinkage_eps,
        cov_condition_number: result.cov_condition_number,
        cov_eigenvalues: result.cov_eigenvalues.as_deref(),
        shrinkage_eta: &result.shrinkage_eta,
        eta_names: &result.eta_names,
    };
    let entries: Vec<crate::types::WarningEntry> = result
        .warnings
        .iter()
        .map(|w| {
            if let Some(entry) = native.get(w) {
                entry.clone()
            } else {
                let mut e = crate::types::classify_warning(w);
                e.details = diagnostic_details(&e.category, &stats);
                e
            }
        })
        .collect();
    result.warnings_structured = entries;
}

/// Typed inputs for [`diagnostic_details`], sourced from `FitResult`'s fields.
/// `Default` gives finite-zero scalars, `None` options, and empty slices, so a
/// test can set only the fields a case needs via struct-update syntax.
#[derive(Default)]
pub(crate) struct DiagStats<'a> {
    pub(crate) dw_statistic: f64,
    pub(crate) iwres_lag1_r: f64,
    pub(crate) shrinkage_eps: f64,
    pub(crate) cov_condition_number: Option<f64>,
    pub(crate) cov_eigenvalues: Option<&'a [f64]>,
    pub(crate) shrinkage_eta: &'a [f64],
    pub(crate) eta_names: &'a [String],
}

/// Machine-readable `details` payload for the numeric diagnostic warning codes,
/// sourced from the typed `FitResult` fields rather than parsed from the message
/// text. A non-finite (`NaN`/`±Inf`) statistic is never emitted as a JSON `null`
/// (`serde_json` maps non-finite floats to `null`) — instead:
///
/// - the single/paired-stat codes (`DwAutocorrelation`, `EpsShrinkage`,
///   `ConditionNumber`) return `None` (whole `details` key omitted) when a
///   required statistic is missing or non-finite;
/// - the covariance codes (`CovarianceFailed`, `CovarianceRegularized`) build a
///   partial object, **skipping** individual non-finite/absent fields and
///   emitting whatever is available; they return `None` only when nothing is.
///
/// `cov_condition_number` in particular is documented as `+Inf` for a
/// near-singular parameter space, so it is dropped from the payload there.
pub(crate) fn diagnostic_details(
    code: &crate::types::WarningCode,
    s: &DiagStats,
) -> Option<serde_json::Value> {
    use crate::types::WarningCode;
    match code {
        WarningCode::DwAutocorrelation
            if s.dw_statistic.is_finite() && s.iwres_lag1_r.is_finite() =>
        {
            Some(serde_json::json!({
                "durbin_watson": s.dw_statistic,
                "iwres_lag1_autocorr": s.iwres_lag1_r,
            }))
        }
        WarningCode::EpsShrinkage if s.shrinkage_eps.is_finite() => Some(serde_json::json!({
            "eps_shrinkage": s.shrinkage_eps,
            "eps_shrinkage_pct": 100.0 * s.shrinkage_eps,
        })),
        WarningCode::EtaShrinkage => {
            // The high-shrinkage ETAs behind the message, with each shrinkage
            // as a percent — sourced from the typed `shrinkage_eta` field.
            let high: Vec<serde_json::Value> = high_shrinkage_eta_indices(s.shrinkage_eta)
                .into_iter()
                .map(|i| {
                    serde_json::json!({
                        "eta": eta_label(s.eta_names, i),
                        "shrinkage_pct": 100.0 * s.shrinkage_eta[i],
                    })
                })
                .collect();
            if high.is_empty() {
                None
            } else {
                Some(serde_json::json!({
                    "threshold_pct": 100.0 * ETA_SHRINKAGE_WARN_THRESHOLD,
                    "high_shrinkage_etas": high,
                }))
            }
        }
        WarningCode::ConditionNumber => match s.cov_condition_number {
            Some(c) if c.is_finite() => Some(serde_json::json!({ "condition_number": c })),
            _ => None,
        },
        WarningCode::CovarianceFailed | WarningCode::CovarianceRegularized => {
            covariance_details(s.cov_condition_number, s.cov_eigenvalues)
        }
        _ => None,
    }
}

/// `details` for the covariance-step warning codes: the condition number and,
/// when the covariance matrix was produced (so eigenvalues exist — typically
/// the regularized case, not a hard failure), the smallest eigenvalue and the
/// count of negative ones (which diagnose a non-PD Hessian). Non-finite values
/// are skipped; returns `None` if nothing usable is available.
fn covariance_details(
    cov_condition_number: Option<f64>,
    cov_eigenvalues: Option<&[f64]>,
) -> Option<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    if let Some(c) = cov_condition_number {
        if c.is_finite() {
            obj.insert("condition_number".to_string(), serde_json::json!(c));
        }
    }
    if let Some(eigs) = cov_eigenvalues {
        let finite: Vec<f64> = eigs.iter().copied().filter(|v| v.is_finite()).collect();
        if !finite.is_empty() {
            let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let n_neg = finite.iter().filter(|&&v| v < 0.0).count();
            obj.insert("min_eigenvalue".to_string(), serde_json::json!(min));
            obj.insert(
                "n_negative_eigenvalues".to_string(),
                serde_json::json!(n_neg),
            );
        }
    }
    if obj.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(obj))
    }
}

/// Probe whether NLopt CRS2-LM (used for global_search) is available.
pub(crate) fn probe_nlopt_algorithms() -> Vec<String> {
    fn dummy_obj(_x: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()) -> f64 {
        0.0
    }
    let available = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Crs2Lm,
            1,
            dummy_obj,
            nlopt::Target::Minimize,
            (),
        );
    }));
    if available.is_err() {
        vec![
            "NLopt CRS2-LM not available in this build — global_search = true will fail. \
             Install a full NLopt build: brew install nlopt / apt install libnlopt-dev"
                .to_string(),
        ]
    } else {
        vec![]
    }
}

/// Eigenvalues and condition number of the correlation matrix of free
/// (non-fixed) parameters.  Fixed parameters have zero diagonal in the
/// covariance matrix and are excluded so that the correlation scaling does not
/// divide by zero and the condition number reflects only the identifiable
/// parameter space.
///
/// Returns `(None, None)` when `cov` is `None` or fewer than two free
/// parameters exist (after excluding parameters whose diagonal entry is
/// `<= 0`).  Parameters with non-positive diagonals are treated as fixed and
/// silently excluded; the remaining free subblock is used for the computation.
/// Threshold below which an off-diagonal omega/kappa entry is treated as
/// structurally zero for correlation reporting.  Matches the threshold used
/// in `io/output.rs` when emitting the `correlation:` field.
const OFFDIAG_EPS: f64 = 1e-15;

/// Compute a parameter-level correlation matrix from an omega/kappa matrix.
///
/// For lognormal pairs uses `(exp(ω_ij)−1)/√((exp(ω_ii)−1)(exp(ω_jj)−1))`.
/// For additive pairs uses `ω_ij/√(ω_ii·ω_jj)` (eta-level).
/// Mixed pairs fall back to eta-level and append a warning.
/// Returns `None` when the matrix is diagonal (no off-diagonals above
/// `OFFDIAG_EPS`).
pub(crate) fn compute_param_corr(
    omega: &DMatrix<f64>,
    log_transformed: &[bool],
    names: &[String],
    warn_prefix: &str,
    warnings: &mut Vec<String>,
) -> Option<DMatrix<f64>> {
    let n = omega.nrows();
    debug_assert_eq!(
        log_transformed.len(),
        n,
        "log_transformed must be parallel to omega diagonal (got {} for n={})",
        log_transformed.len(),
        n,
    );
    debug_assert_eq!(
        names.len(),
        n,
        "names must be parallel to omega diagonal (got {} for n={})",
        names.len(),
        n,
    );
    let has_offdiag = (0..n).any(|i| (0..i).any(|j| omega[(i, j)].abs() > OFFDIAG_EPS));
    if !has_offdiag {
        return None;
    }
    let mut corr = DMatrix::identity(n, n);
    for i in 0..n {
        for j in 0..i {
            let cov = omega[(i, j)];
            if cov.abs() <= OFFDIAG_EPS {
                continue;
            }
            let w_ii = omega[(i, i)];
            let w_jj = omega[(j, j)];
            let lt_i = *log_transformed.get(i).unwrap_or(&false);
            let lt_j = *log_transformed.get(j).unwrap_or(&false);
            let c = if lt_i && lt_j {
                let num = cov.exp() - 1.0;
                let den = ((w_ii.exp() - 1.0) * (w_jj.exp() - 1.0)).sqrt();
                if den > 0.0 {
                    num / den
                } else {
                    0.0
                }
            } else if !lt_i && !lt_j {
                let den = (w_ii * w_jj).sqrt();
                if den > 0.0 {
                    cov / den
                } else {
                    0.0
                }
            } else {
                let name_i = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                let name_j = names.get(j).map(|s| s.as_str()).unwrap_or("?");
                warnings.push(format!(
                    "{}: {} × {} have mixed lognormal/additive parameterizations; \
                     falling back to eta-level correlation",
                    warn_prefix, name_i, name_j
                ));
                let den = (w_ii * w_jj).sqrt();
                if den > 0.0 {
                    cov / den
                } else {
                    0.0
                }
            };
            corr[(i, j)] = c;
            corr[(j, i)] = c;
        }
    }
    Some(corr)
}

/// Whether `stage_idx` is the last *estimating* stage of `chain` — the single
/// stage that owns the covariance / SIR step. Running it on every stage is
/// wasteful and not expected (#615); only the final estimate needs SEs.
///
/// An evaluation-only IMP (`imp_eval_only`, NONMEM `IMP EONLY=1`) is a likelihood
/// evaluation, not an estimator, so trailing eval-only IMP stages cede the
/// covariance step to the preceding estimator. A default (estimating) IMP is a
/// real estimator that owns the step itself, so it must not be skipped. Split out
/// of `fit()` so the per-stage gating is unit-testable.
pub(crate) fn is_last_estimating_stage(
    chain: &[EstimationMethod],
    stage_idx: usize,
    imp_eval_only: bool,
) -> bool {
    let is_last = stage_idx + 1 == chain.len();
    let trailing_eval_only_imp = imp_eval_only
        && chain[stage_idx + 1..]
            .iter()
            .all(|&m| m == EstimationMethod::Imp);
    is_last || trailing_eval_only_imp
}

/// Resolve the reported [`CovarianceStatus`] from the three signals that
/// determine it: whether the covariance step was requested, whether it produced
/// a covariance matrix, and whether the SIR fallback (`covariance_fallback =
/// sir`) produced a result. Pulled out of `fit()` so the precedence — a real
/// covariance always wins over a fallback, which wins over a plain failure — is
/// unit-testable without driving a full fit to a non-PD Hessian.
pub(crate) fn resolve_covariance_status(
    run_covariance_step: bool,
    has_covariance_matrix: bool,
    has_sir_fallback: bool,
) -> CovarianceStatus {
    if !run_covariance_step {
        CovarianceStatus::NotRequested
    } else if has_covariance_matrix {
        CovarianceStatus::Computed
    } else if has_sir_fallback {
        CovarianceStatus::SirFallback
    } else {
        CovarianceStatus::Failed
    }
}

/// Pure gate for the non-PD-Hessian SIR fallback: should it run? It fires only
/// when the user opted in (`covariance_fallback = sir`), the FD-Hessian
/// covariance did **not** succeed (`!has_covariance_matrix`), a normal
/// `sir = true` run did **not** already produce intervals (`!normal_sir_ran`),
/// and `compute_covariance` actually handed back a fallback proposal
/// (`has_fallback_proposal`). Split out of [`resolve_sir_fallback`] so the
/// decision is unit-testable without driving a fit to a non-PD Hessian (#264).
pub(crate) fn should_run_sir_fallback(
    fallback_is_sir: bool,
    has_covariance_matrix: bool,
    normal_sir_ran: bool,
    has_fallback_proposal: bool,
) -> bool {
    fallback_is_sir && !has_covariance_matrix && !normal_sir_ran && has_fallback_proposal
}

/// Run the non-PD-Hessian SIR fallback when [`should_run_sir_fallback`] permits.
///
/// Returns `Some(SirResult)` when the fallback fired and SIR succeeded; `None`
/// when the gate declined, the run was cancelled, or SIR itself failed (the
/// failure case pushes a `"SIR fallback failed: …"` warning). Extracted from
/// `fit_inner` so the gate → `run_sir_core` → warning wiring is exercised by a
/// unit test with a controlled (tame) proposal, rather than relying on a real
/// non-PD fit — which the optimizer's fixed warmup budget cannot reach and a
/// degenerate fixture cannot reliably survive in SIR (#264).
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_sir_fallback(
    options: &FitOptions,
    has_covariance_matrix: bool,
    normal_sir_ran: bool,
    fallback_proposal: Option<&DMatrix<f64>>,
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    ofv: f64,
    warnings: &mut Vec<String>,
) -> Option<crate::estimation::sir::SirResult> {
    if crate::cancel::is_cancelled(&options.cancel) {
        return None;
    }
    if !should_run_sir_fallback(
        options.covariance_fallback == CovarianceFallback::Sir,
        has_covariance_matrix,
        normal_sir_ran,
        fallback_proposal.is_some(),
    ) {
        return None;
    }
    let proposal =
        fallback_proposal.expect("should_run_sir_fallback guarantees a proposal is present");
    if options.verbose {
        eprintln!("\nRunning SIR fallback (non-PD Hessian)...");
    }
    match crate::estimation::sir::run_sir_core(
        model, population, params, eta_hats, proposal, ofv, options,
    ) {
        Ok(sir) => Some(sir),
        Err(e) => {
            warnings.push(format!("SIR fallback failed: {}", e));
            None
        }
    }
}

pub(crate) fn cov_diagnostics(cov: Option<&DMatrix<f64>>) -> (Option<Vec<f64>>, Option<f64>) {
    let cov = match cov {
        Some(m) => m,
        None => return (None, None),
    };
    let n = cov.nrows();
    let free: Vec<usize> = (0..n).filter(|&i| cov[(i, i)] > 0.0).collect();
    if free.len() < 2 {
        return (None, None);
    }
    let sub = DMatrix::from_fn(free.len(), free.len(), |a, b| cov[(free[a], free[b])]);
    let std_devs: Vec<f64> = (0..free.len()).map(|a| sub[(a, a)].sqrt()).collect();
    let cor = DMatrix::from_fn(free.len(), free.len(), |a, b| {
        sub[(a, b)] / (std_devs[a] * std_devs[b])
    });
    let eig = cor.symmetric_eigen();
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let min_ev = eigenvalues.last().copied().unwrap_or(0.0);
    let max_ev = eigenvalues.first().copied().unwrap_or(0.0);
    let condition_number = if min_ev > 1e-10 {
        max_ev / min_ev
    } else {
        f64::INFINITY
    };
    (Some(eigenvalues), Some(condition_number))
}

/// Compute per-subject diagnostics (IPRED, PRED, IWRES, CWRES)
pub(crate) fn compute_subject_results(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas_per_subject: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<SubjectResult> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let eta = &eta_hats[i];
            let h = &h_matrices[i];
            let kappas: &[DVector<f64>] = if i < kappas_per_subject.len() {
                kappas_per_subject[i].as_slice()
            } else {
                &[]
            };

            // Individual predictions: f(eta_hat), with occasion-specific kappas for IOV.
            // Uses the continuous per-occasion-aware prediction (issue #104) for IOV
            // and the TV-aware dispatcher for everyone else — so the sdtab IPRED/IWRES
            // match the IPRED that drove the FOCEI marginal at fit time.
            //
            // Previously this branch called `model_preds` with a single per-subject
            // `pk_params_ind` from `subject.covariates`, which on TV-covariate data
            // silently took the **non-TV** dose-superposition path while the OFV
            // was being computed on the event-driven path that honours per-event
            // covariate snapshots. Result: sdtab IPRED collapsed to ~0 in the
            // terminal phase for subjects with even mild TV covariates, IWRES
            // exploded, and the EPS-shrinkage warning fired even when the actual
            // fit (and the inner-loop EBE) were fine. Caught on the jasmine peds
            // vancomycin testdata — see `[[focei-laplace-not-sheiner-beal]]`.
            // For IOV subjects: ipred via predict_iov; compartment states are not
            // yet supported on the IOV path (tracked as follow-up), so they stay empty.
            // For all other subjects: compute_predictions_with_states returns both ipred
            // and the per-obs compartment state vector in one pass.
            let (ipred, compartment_states) = if !kappas.is_empty() {
                let kappa_slices: Vec<Vec<f64>> =
                    kappas.iter().map(|k| k.as_slice().to_vec()).collect();
                let iov_ipred = crate::pk::predict_iov(
                    model,
                    subject,
                    &params.theta,
                    eta.as_slice(),
                    &kappa_slices,
                );
                (iov_ipred, vec![])
            } else {
                crate::pk::compute_predictions_with_states(
                    model,
                    subject,
                    &params.theta,
                    eta.as_slice(),
                )
            };

            // Population predictions: f(eta = 0, kappa = 0). Routes through the
            // TV-cov-aware predictor (the same path `predict()` uses), so the sdtab
            // PRED column honours time-varying covariates, EVID=3/4 resets, and the
            // FREM prediction override — and stays consistent with the public
            // `predict()` output (#456).
            let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
            let pred =
                crate::pk::compute_predictions_with_tv(model, subject, &params.theta, &zero_eta);

            // Per-observation custom residual magnitude (#484): η-independent
            // (θ/covariate/TIME only), so build it once and feed both the IWRES
            // and CWRES diagnostics so they match the magnitude-aware OFV.
            let ruv_mult = model.ruv_obs_mult(subject, &params.theta);

            // IWRES (NaN on censored rows — see compute_cwres for CWRES handling).
            let mut iwres = compute_iwres_with_correlations(
                &subject.observations,
                &ipred,
                model.error_spec.obs_keys(subject).as_ref(),
                &model.error_spec,
                &params.sigma.values,
                &model.residual_correlations,
                ruv_mult.as_deref(),
            );
            // IIV on residual error (#409): the individual residual SD is scaled
            // by exp(η̂_ruv), so IWRES = (y−f)/(SD·exp(η̂_ruv)) = base / exp(η̂_ruv).
            // FREM covariate rows have no PK residual; their IWRES is left as-is.
            let ruv_sd = model.residual_var_scale(eta.as_slice()).sqrt();
            if ruv_sd != 1.0 {
                for (j, w) in iwres.iter_mut().enumerate() {
                    if subject.fremtype.get(j).copied().unwrap_or(0) == 0 {
                        *w /= ruv_sd;
                    }
                }
            }
            for (j, c) in subject.cens.iter().enumerate() {
                if *c != 0 {
                    iwres[j] = f64::NAN;
                }
            }

            // CWRES
            let frem_r_override = build_frem_r_override(
                model.frem_config.as_ref(),
                &subject.fremtype,
                &params.sigma.values,
            );
            let cwres = compute_cwres(
                subject,
                &ipred,
                eta,
                h,
                &params.omega,
                &params.sigma.values,
                &model.error_spec,
                &model.residual_correlations,
                frem_r_override.as_deref(),
                model.residual_error_eta,
                ruv_mult.as_deref(),
            );

            // OFV contribution
            let ofv_i = if !kappas.is_empty() {
                let omega_iov = params
                    .omega_iov
                    .as_ref()
                    .expect("omega_iov present when kappas non-empty");
                foce_subject_nll_iov(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                    kappas,
                    omega_iov,
                )
            } else {
                foce_subject_nll(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                )
            };

            SubjectResult {
                id: subject.id.clone(),
                eta: eta.clone(),
                ipred,
                pred,
                iwres,
                cwres,
                // Filled post-fit (only when npde_nsim > 0); see compute_npde_npd.
                npde: vec![],
                npd: vec![],
                ofv_contribution: 2.0 * ofv_i,
                cens: subject.cens.clone(),
                n_obs: subject.observations.len(),
                extra_columns: vec![],
                per_obs_tad: vec![],
                compartment_states,
                // Discrete-endpoint (binary) per-record diagnostics at this subject's
                // EBE η (§8.8.5). Empty for every model without a binary endpoint, so
                // existing output is untouched.
                #[cfg(feature = "survival")]
                discrete_rows: crate::categorical::binary_diagnostics(
                    model,
                    subject,
                    &params.theta,
                    eta.as_slice(),
                ),
            }
        })
        .collect()
}

/// Kappa shrinkage pooled across all subject-occasion pairs.
///
/// `1 - sqrt(mean(κ̂²)) / sqrt(omega_iov_kk)` for each kappa k, where the mean
/// runs over every (subject, occasion) pair.  Returns NaN for a given kappa when
/// the corresponding diagonal of `omega_iov` is non-positive or when fewer than
/// two (subject, occasion) observations are available.
pub(crate) fn compute_kappa_shrinkage(
    kappas_per_subject: &[Vec<DVector<f64>>],
    omega_iov: &DMatrix<f64>,
) -> Vec<f64> {
    let n_kappa = omega_iov.nrows();
    if n_kappa == 0 {
        return vec![];
    }
    // Flatten all per-subject per-occasion kappa vectors into one iterator.
    let all_kappas: Vec<&DVector<f64>> = kappas_per_subject
        .iter()
        .flat_map(|occ_kappas| occ_kappas.iter())
        .collect();
    let n = all_kappas.len();
    if n < 2 {
        return vec![f64::NAN; n_kappa];
    }
    (0..n_kappa)
        .map(|k| {
            let var = omega_iov[(k, k)];
            if var <= 0.0 {
                return f64::NAN;
            }
            let ms = all_kappas.iter().map(|kv| kv[k].powi(2)).sum::<f64>() / n as f64;
            1.0 - ms.sqrt() / var.sqrt()
        })
        .collect()
}

/// Kappa shrinkage broken out by occasion index.
///
/// Returns `shrinkage_by_occ[occ_idx][kappa_idx]` where `occ_idx` is the
/// **0-based position within each subject's own occasion list** — i.e. the
/// order in which distinct OCC values were first encountered in that subject's
/// rows (matching `split_obs_by_occasion`).
///
/// **Important limitation for unbalanced designs:** `occ_idx` is a position
/// index, *not* the raw OCC column value.  When subjects have different OCC
/// sequences (e.g., a late-entry subject whose data begins at OCC 2), their
/// position 0 maps to OCC 2 while other subjects' position 0 maps to OCC 1.
/// Pooling across position 0 then mixes kappas from different occasions.
/// For unbalanced designs use the pooled `shrinkage_kappa` instead, and
/// interpret per-occasion values only when the OCC column is aligned across
/// all subjects.
///
/// Returns an empty outer vec when fewer than two distinct occasions are present
/// or no kappa parameters exist.
pub(crate) fn compute_kappa_shrinkage_by_occ(
    kappas_per_subject: &[Vec<DVector<f64>>],
    omega_iov: &DMatrix<f64>,
) -> Vec<Vec<f64>> {
    let n_kappa = omega_iov.nrows();
    if n_kappa == 0 {
        return vec![];
    }
    // Determine max number of occasions across subjects.
    let n_occ = kappas_per_subject
        .iter()
        .map(|v| v.len())
        .max()
        .unwrap_or(0);
    if n_occ < 2 {
        return vec![];
    }
    (0..n_occ)
        .map(|occ_idx| {
            let occ_kappas: Vec<&DVector<f64>> = kappas_per_subject
                .iter()
                .filter_map(|occ_vecs| occ_vecs.get(occ_idx))
                .collect();
            let n = occ_kappas.len();
            (0..n_kappa)
                .map(|k| {
                    let var = omega_iov[(k, k)];
                    if var <= 0.0 || n < 2 {
                        return f64::NAN;
                    }
                    let ms = occ_kappas.iter().map(|kv| kv[k].powi(2)).sum::<f64>() / n as f64;
                    1.0 - ms.sqrt() / var.sqrt()
                })
                .collect()
        })
        .collect()
}

/// ETA shrinkage: `1 - sqrt(mean(eta_hat_k^2)) / sqrt(omega_kk)` for each random effect k.
///
/// Uses the uncentered second moment with `n` divisor (NONMEM / PsN / Monolix
/// convention), reflecting the population assumption that `E[eta_k] = 0`. This
/// differs from the centered, unbiased sample variance (n-1 divisor) — for small
/// `n` the unbiased form inflates SD by sqrt(n/(n-1)) and routinely produces
/// spurious negative shrinkage even on well-fit models.
pub(crate) fn compute_eta_shrinkage(subjects: &[SubjectResult], omega: &DMatrix<f64>) -> Vec<f64> {
    let n_eta = omega.nrows();
    let n_subj = subjects.len();
    if n_subj < 2 || n_eta == 0 {
        return vec![f64::NAN; n_eta];
    }
    (0..n_eta)
        .map(|k| {
            let omega_var = omega[(k, k)];
            if omega_var <= 0.0 {
                return f64::NAN;
            }
            let omega_sd = omega_var.sqrt();
            let ms = subjects.iter().map(|s| s.eta[k].powi(2)).sum::<f64>() / n_subj as f64;
            1.0 - ms.sqrt() / omega_sd
        })
        .collect()
}

/// EPS shrinkage: `1 - sqrt(mean(IWRES^2))` across all valid (non-NaN) residuals.
///
/// IWRES has model-imposed mean 0 and variance 1, so the uncentered second
/// moment with `n` divisor is the natural estimator (matches NONMEM).
pub(crate) fn compute_eps_shrinkage(subjects: &[SubjectResult]) -> f64 {
    let vals: Vec<f64> = subjects
        .iter()
        .flat_map(|s| s.iwres.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    let n = vals.len();
    if n < 2 {
        return f64::NAN;
    }
    let ms = vals.iter().map(|v| v.powi(2)).sum::<f64>() / n as f64;
    1.0 - ms.sqrt()
}

/// Threshold below which negative `shrinkage_eps` triggers a warning.
///
/// Small negative values are normal sampling noise around 0 on well-fit models
/// (the NONMEM uncentered estimator has a small downward bias when the sample
/// mean of IWRES is non-zero). Past this threshold the residual error model
/// genuinely fails to absorb the residuals at the EBE etas and the user should
/// see it.
const EPS_SHRINKAGE_WARN_THRESHOLD: f64 = -0.05;

/// Build the user-facing warning for notably-negative EPS shrinkage, or
/// `None` if the value is finite and above the threshold (or NaN).
pub(crate) fn eps_shrinkage_warning(shrinkage_eps: f64) -> Option<String> {
    if !shrinkage_eps.is_finite() || shrinkage_eps >= EPS_SHRINKAGE_WARN_THRESHOLD {
        return None;
    }
    Some(format!(
        "EPS shrinkage is notably negative ({:.1}%): mean(IWRES^2) > 1, \
         which means the residual error model does not absorb the residuals \
         at the final EBE etas. Common causes: SAEM converged to a local \
         optimum with under-fit sigma (try `method = [saem, focei]` to polish \
         with FOCEI, or different starts); model misspecification on a subset \
         of subjects; sigma at a bound. Inspect the IWRES distribution in the \
         sdtab.",
        100.0 * shrinkage_eps
    ))
}

/// ETA-shrinkage warning threshold (fraction). 0.30 is the classic
/// Savic & Karlsson (2009) 30% rule of thumb: above it, an EBE-based
/// diagnostic — and the individual estimates of that random effect — are
/// poorly informed by the data (η shrinks toward 0).
const ETA_SHRINKAGE_WARN_THRESHOLD: f64 = 0.30;

/// Indices of the ETAs whose shrinkage meets/exceeds the warning threshold.
/// `shrinkage` is per-ETA shrinkage as a fraction (0..1); non-finite entries
/// are ignored.
pub(crate) fn high_shrinkage_eta_indices(shrinkage: &[f64]) -> Vec<usize> {
    shrinkage
        .iter()
        .enumerate()
        .filter(|(_, &s)| s.is_finite() && s >= ETA_SHRINKAGE_WARN_THRESHOLD)
        .map(|(i, _)| i)
        .collect()
}

/// Label for the `i`-th ETA, falling back to a non-empty `eta_{i}` placeholder
/// when the name is missing — so both the message and the `details` payload
/// stay unambiguous.
fn eta_label(eta_names: &[String], i: usize) -> String {
    eta_names
        .get(i)
        .cloned()
        .unwrap_or_else(|| format!("eta_{i}"))
}

/// Build the user-facing warning for high ETA shrinkage, or `None` when no ETA
/// exceeds the threshold. `shrinkage` is per-ETA shrinkage as a fraction;
/// `eta_names` labels them in the same order.
pub(crate) fn eta_shrinkage_warning(shrinkage: &[f64], eta_names: &[String]) -> Option<String> {
    let high = high_shrinkage_eta_indices(shrinkage);
    if high.is_empty() {
        return None;
    }
    let list = high
        .iter()
        .map(|&i| format!("{} ({:.0}%)", eta_label(eta_names, i), 100.0 * shrinkage[i]))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "High ETA shrinkage (\u{2265} {:.0}%): {}. EBE-based diagnostics for these \
         random effects are unreliable and the data poorly inform their individual \
         estimates; consider removing the IIV on the affected parameter(s) or \
         collecting more informative data.",
        100.0 * ETA_SHRINKAGE_WARN_THRESHOLD,
        list
    ))
}

/// Relative tolerance (a fraction of the packed bound range) within which a
/// theta estimate counts as sitting on its lower/upper optimizer bound.
const BOUNDARY_REL_TOL: f64 = 1e-3;

/// Which bound a theta estimate is pinned to and the **effective** natural-space
/// bound value the optimizer actually constrains to, or `None` when the estimate
/// is interior. Evaluated in the optimizer's *packed* space (log when the lower
/// bound is `>= 0`, else identity) so the proximity test is scale-appropriate —
/// a log-scaled parameter is "at" its bound within a constant factor, not a
/// constant absolute gap.
///
/// The floors/caps mirror `compute_bounds` / `pack_params`: a log-packed theta
/// floors its estimate and lower bound at `1e-10` and caps the upper at `1e9`,
/// so the reported bound matches the constraint that was actually active (and
/// agrees with the estimate). Degenerate/non-finite bounds yield `None`.
pub(crate) fn theta_boundary_side(est: f64, lower: f64, upper: f64) -> Option<(&'static str, f64)> {
    use crate::estimation::parameterization::theta_packs_log;
    let (lo_eff, hi_eff, pe, pl, pu) = if theta_packs_log(lower) {
        let lo = lower.max(1e-10);
        let hi = upper.min(1e9);
        (lo, hi, est.max(1e-10).ln(), lo.ln(), hi.ln())
    } else {
        (lower, upper, est, lower, upper)
    };
    let range = pu - pl;
    if !range.is_finite() || range <= 0.0 || !pe.is_finite() {
        return None;
    }
    let frac = (pe - pl) / range;
    if frac <= BOUNDARY_REL_TOL {
        Some(("lower", lo_eff))
    } else if frac >= 1.0 - BOUNDARY_REL_TOL {
        Some(("upper", hi_eff))
    } else {
        None
    }
}

/// Free (non-fixed) theta estimates pinned to an optimizer bound, as
/// `(name, estimate, effective_bound, side)` per hit.
fn boundary_estimates(params: &ModelParameters) -> Vec<(String, f64, f64, &'static str)> {
    let mut hits = Vec::new();
    for i in 0..params.theta.len() {
        if params.theta_fixed.get(i).copied().unwrap_or(false) {
            continue;
        }
        let est = params.theta[i];
        if let Some((side, bound)) =
            theta_boundary_side(est, params.theta_lower[i], params.theta_upper[i])
        {
            let name = params
                .theta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("THETA{}", i + 1));
            hits.push((name, est, bound, side));
        }
    }
    hits
}

/// Construct a fit-end [`WarningEntry`] with the invariant fields every native
/// emitter shares (`severity: Warning`, `source_method: None`), varying only
/// `category`, `message`, and `details`.
fn warning_entry(
    category: WarningCode,
    message: String,
    details: Option<serde_json::Value>,
) -> WarningEntry {
    WarningEntry {
        severity: WarningSeverity::Warning,
        category,
        message,
        source_method: None,
        details,
    }
}

/// Shared scaffold for the list-style fit-end warnings: given a set of `hits`,
/// return `None` when empty, otherwise join a human `list` string (`line` per
/// hit), format the message (`msg` over the joined list), build the structured
/// `details`, and wrap it in a [`WarningEntry`] as `Some((message, entry))`.
fn list_warning<H>(
    hits: Vec<H>,
    category: WarningCode,
    line: impl Fn(&H) -> String,
    msg: impl Fn(&str) -> String,
    details: impl Fn(&[H]) -> serde_json::Value,
) -> Option<(String, WarningEntry)> {
    if hits.is_empty() {
        return None;
    }
    let list = hits.iter().map(|h| line(h)).collect::<Vec<_>>().join(", ");
    let msg = msg(&list);
    let entry = warning_entry(category, msg.clone(), Some(details(&hits)));
    Some((msg, entry))
}

/// Build the human message + native structured entry (with `details`) for
/// theta estimates pinned to an optimizer bound, or `None` when none are.
pub(crate) fn boundary_estimate_warning(
    params: &ModelParameters,
) -> Option<(String, WarningEntry)> {
    list_warning(
        boundary_estimates(params),
        WarningCode::BoundaryEstimate,
        |(name, est, _bound, side)| format!("{name} ({est:.4} at {side} bound)"),
        |list| {
            format!(
                "Parameter estimate(s) pinned to an optimizer bound: {list}. This often \
                 indicates non-identifiability or a too-tight bound; inspect the affected \
                 parameter(s) and consider relaxing the bound or simplifying the model."
            )
        },
        |hits| {
            let params_json: Vec<serde_json::Value> = hits
                .iter()
                .map(|(name, est, bound, side)| {
                    serde_json::json!({
                        "parameter": name,
                        "estimate": est,
                        "bound": bound,
                        "side": side,
                    })
                })
                .collect();
            serde_json::json!({ "parameters": params_json })
        },
    )
}

/// Relative-standard-error threshold (percent) above which a free THETA is
/// flagged as imprecisely estimated. 50% is a common rule of thumb for fixed
/// effects (well-estimated parameters typically sit well below it).
const RSE_WARN_THRESHOLD_PCT: f64 = 50.0;

/// Free (non-fixed) THETA estimates with a relative standard error above the
/// threshold, as `(name, estimate, se, rse_pct)`. Empty when the covariance
/// step produced no SEs, or none are imprecise. `RSE = 100 * se / |estimate|`.
fn inflated_rse_thetas(
    se_theta: &Option<Vec<f64>>,
    params: &ModelParameters,
) -> Vec<(String, f64, f64, f64)> {
    let Some(se) = se_theta else {
        return Vec::new();
    };
    let mut hits = Vec::new();
    for i in 0..params.theta.len() {
        if params.theta_fixed.get(i).copied().unwrap_or(false) {
            continue;
        }
        let (est, se_i) = (params.theta[i], se.get(i).copied().unwrap_or(f64::NAN));
        if est.abs() <= 1e-12 || !se_i.is_finite() {
            continue;
        }
        let rse = 100.0 * se_i / est.abs();
        if rse.is_finite() && rse > RSE_WARN_THRESHOLD_PCT {
            let name = params
                .theta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("THETA{}", i + 1));
            hits.push((name, est, se_i, rse));
        }
    }
    hits
}

/// Build the human message + native structured entry (with `details`) for
/// free THETAs whose relative standard error exceeds the threshold, or `None`.
pub(crate) fn inflated_rse_warning(
    se_theta: &Option<Vec<f64>>,
    params: &ModelParameters,
) -> Option<(String, WarningEntry)> {
    list_warning(
        inflated_rse_thetas(se_theta, params),
        WarningCode::InflatedRse,
        // One decimal so a borderline value (e.g. 50.4%) is not displayed as
        // "50%" — which would read as below the threshold it just tripped.
        |(name, _est, _se, rse)| format!("{name} ({rse:.1}%)"),
        |list| {
            format!(
                "High relative standard error (RSE > {:.0}%): {list}. These parameter(s) \
                 are imprecisely estimated — often a sign of over-parameterization or \
                 data that do not inform them; consider simplifying the model.",
                RSE_WARN_THRESHOLD_PCT
            )
        },
        |hits| {
            let params_json: Vec<serde_json::Value> = hits
                .iter()
                .map(|(name, est, se, rse)| {
                    serde_json::json!({
                        "parameter": name,
                        "estimate": est,
                        "se": se,
                        "rse_pct": rse,
                    })
                })
                .collect();
            serde_json::json!({
                "threshold_pct": RSE_WARN_THRESHOLD_PCT,
                "parameters": params_json,
            })
        },
    )
}

/// Absolute correlation above which a parameter pair is flagged as
/// collinear/non-identified. `0.95` is a common rule of thumb.
const CORRELATION_WARN_THRESHOLD: f64 = 0.95;

/// THETA (fixed-effect) pairs whose estimate correlation exceeds the threshold,
/// as `(name_a, name_b, correlation)`. Restricted to the THETA block — it
/// occupies the first `theta_names.len()` packed entries with unambiguous names,
/// and fixed-effect collinearity is the most actionable/interpretable case
/// (omega/sigma live in Cholesky space, where both the correlation and the
/// packed labels are murkier). Fixed / zero-variance thetas (diagonal `<= 0`)
/// are skipped. Empty when there is no covariance matrix.
pub(crate) fn high_correlation_pairs(
    cov: Option<&DMatrix<f64>>,
    theta_names: &[String],
) -> Vec<(String, String, f64)> {
    let Some(cov) = cov else {
        return Vec::new();
    };
    let n_theta = theta_names.len().min(cov.nrows());
    let mut pairs = Vec::new();
    for a in 0..n_theta {
        if cov[(a, a)] <= 0.0 {
            continue;
        }
        for b in (a + 1)..n_theta {
            if cov[(b, b)] <= 0.0 {
                continue;
            }
            // sqrt(var_a) * sqrt(var_b) avoids the overflow/underflow that
            // `(var_a * var_b).sqrt()` can hit for extreme variances.
            let denom = cov[(a, a)].sqrt() * cov[(b, b)].sqrt();
            if denom <= 0.0 || !denom.is_finite() {
                continue;
            }
            let r = cov[(a, b)] / denom;
            if r.is_finite() && r.abs() >= CORRELATION_WARN_THRESHOLD {
                pairs.push((theta_names[a].clone(), theta_names[b].clone(), r));
            }
        }
    }
    pairs
}

/// Build the human message + native structured entry (with `details`) for
/// highly correlated THETA pairs in the fit's covariance matrix, or `None`.
pub(crate) fn high_correlation_warning(result: &FitResult) -> Option<(String, WarningEntry)> {
    list_warning(
        high_correlation_pairs(result.covariance_matrix.as_ref(), &result.theta_names),
        WarningCode::HighCorrelation,
        |(a, b, r)| format!("{a} ~ {b} ({r:.2})"),
        |list| {
            format!(
                "Highly correlated parameter pair(s) (|r| >= {CORRELATION_WARN_THRESHOLD:.2}): \
                 {list}. Highly correlated estimates indicate over-parameterization or \
                 non-identifiability; consider fixing or removing one of each pair."
            )
        },
        |pairs| {
            let pairs_json: Vec<serde_json::Value> = pairs
                .iter()
                .map(|(a, b, r)| {
                    serde_json::json!({ "parameter_a": a, "parameter_b": b, "correlation": r })
                })
                .collect();
            serde_json::json!({
                "threshold": CORRELATION_WARN_THRESHOLD,
                "pairs": pairs_json,
            })
        },
    )
}

/// Build the human message + native structured entry for a **twin-less** transit /
/// IG absorption closed form whose *fitted* per-subject EBE crosses into the
/// flip-flop regime, or `None`.
///
/// The up-front [`check_absorption_flip_flop_no_twin`] reject only samples η = 0
/// typical values. The flip-flop boundary is parameter-dependent, so a subject
/// whose empirical-Bayes CL/V (or MTT/N, MAT/CV²) drives `ke` past the tilting
/// abscissa passes that check but still hits the closed form's clamp — an
/// identically-zero profile that silently degenerates *that subject's* likelihood
/// contribution (#785). A twin-*carrying* model reroutes such subjects to the ODE
/// twin per-eval (correct), so this fires only when there is no twin (a `lagtime` /
/// `f` / user-`[odes]` form declined the desugar).
///
/// Runs at fit end, once the EBEs are known, and *warns* rather than erroring — the
/// fit already completed — pointing at the ODE `transit()`/`igd()` forcing form,
/// which reroutes per-eval at the actual η.
pub(crate) fn absorption_flip_flop_ebe_warning(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<(String, WarningEntry)> {
    // Only a twin-less transit/IG closed form can degenerate this way: a twin-carrying
    // model reroutes per-eval, a non-absorption model has no flip-flop regime.
    if !matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
    ) || model.absorption_ode_equivalent.is_some()
    {
        return None;
    }
    // Twin-less transit/IG closed forms reject IOV up front, so `eta_hats` carry exactly
    // the `n_eta` BSV etas (`n_kappa == 0`). Guard the length so an unexpected shape
    // skips the subject rather than panicking inside `pk_params_at_time`.
    let expected = model.n_eta + model.n_kappa;
    let crossers: Vec<String> = population
        .subjects
        .iter()
        .zip(eta_hats)
        .filter(|(subject, eta)| {
            eta.len() == expected
                && crate::pk::absorption_flip_flop_at(model, subject, theta, eta.as_slice())
        })
        .map(|(subject, _)| subject.id.clone())
        .collect();
    if crossers.is_empty() {
        return None;
    }
    let is_transit = matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit
    );
    let (abscissa_label, ode_fn, params) = if is_transit {
        ("transit rate KTR = (n+1)/mtt", "transit()", "MTT / CL")
    } else {
        ("IG abscissa 1/(2·MAT·CV²)", "igd()", "MAT / CV² / CL")
    };
    let id_list = crossers.join(", ");
    let msg = format!(
        "{} subject(s) [{}] enter the analytic absorption closed form's clamp region — the \
         flip-flop regime (disposition rate ≥ {}), or (for a 2-cpt model) coincident disposition \
         eigenvalues — at their fitted empirical-Bayes estimates, where the closed form returns \
         an identically-zero concentration profile, silently degenerating those subjects' \
         likelihood contributions. The typical-value (η = 0) parameters are in-domain, so this \
         is not caught at fit start, and this twin-less model (a user `[odes]` / `[scaling]` / \
         `[initial_conditions]` form, or a parameter whose name collides with a reserved \
         `f`/`lagtime` slot) has no ODE twin to reroute to. \
         Rewrite it as an explicit ODE `{}` model (which reroutes per-eval at the actual η), or \
         check the {} starting estimates.",
        model.pk_model.canonical_name(),
        id_list,
        abscissa_label,
        ode_fn,
        params
    );
    let entry = warning_entry(
        WarningCode::FlipFlop,
        msg.clone(),
        Some(serde_json::json!({
            "phase": "ebe",
            "model": model.pk_model.canonical_name(),
            "subjects": crossers,
        })),
    );
    Some((msg, entry))
}

/// Extract standard errors from covariance matrix on the packed parameter scale,
/// then transform back to the original scale via delta method.
pub(crate) fn extract_standard_errors(
    cov: &Option<DMatrix<f64>>,
    template: &ModelParameters,
) -> (
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
) {
    let cov = match cov {
        Some(c) => c,
        None => return (None, None, None, None),
    };

    let n = cov.nrows();
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();

    // SE on packed scale
    let se_packed: Vec<f64> = (0..n)
        .map(|i| {
            let v = cov[(i, i)];
            if v > 0.0 {
                v.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Theta: SE on original scale. Log-packed thetas (lower bound >= 0) are
    // optimized as x = log(theta), so SE(theta) = theta * SE(x) (delta method).
    // Identity-packed thetas (negative lower bound — e.g. covariate exponents,
    // exposure–hazard slopes) are optimized on the natural scale already, so
    // SE(theta) = SE(x): multiplying by the estimate would mis-scale it (and
    // flip the sign for a negative estimate). See `theta_packs_log`.
    let se_theta: Vec<f64> = (0..n_theta)
        .map(|i| {
            // Guard a truncated `cov` (fewer rows than `template.theta`) the same
            // way the omega/sigma/kappa branches below do — report 0.0 rather than
            // panicking away an otherwise-converged fit.
            if i >= n {
                0.0
            } else if theta_packs_log(template.theta_lower[i]) {
                template.theta[i] * se_packed[i]
            } else {
                se_packed[i]
            }
        })
        .collect();

    // Omega: SE via multivariate delta method on Cholesky parameterization.
    //
    // Ω = L L^T, so omega_ij = Σ_{k≤min(i,j)} L_ik * L_jk.
    // Packed params: x = log(L_ii) for diagonals, x = L_ij for off-diags.
    // SE²(omega_ij) = g^T * C_omega * g, where g = ∂omega_ij/∂x.
    //
    // For diagonal omega the off-diagonal L elements are zero, so the formula
    // simplifies to the original: SE(omega_ii) = 2 * omega_ii * SE(log L_ii).
    // For block omega we compute the full lower triangle.
    let omega_start = n_theta;
    let se_omega: Vec<f64> = if template.omega.diagonal {
        (0..n_eta)
            .map(|i| {
                let idx = omega_start + i;
                if idx < n {
                    2.0 * template.omega.matrix[(i, i)] * se_packed[idx]
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        let n_lt = omega_packed_len(n_eta, false);
        let l = &template.omega.chol;

        // Extract omega sub-block of the full covariance matrix.
        let cov_omega = cov.view((omega_start, omega_start), (n_lt, n_lt));

        let mut se_vec = Vec::with_capacity(n_lt);
        // Column-major lower-triangle order — single source: `lower_tri_iter`.
        for (i, j) in lower_tri_iter(n_eta, false) {
            // Build gradient of omega_{ij} w.r.t. packed omega params.
            // omega_{ij} = Σ_{k=0}^{j} L_{ik} * L_{jk}
            let mut grad = vec![0.0f64; n_lt];
            for k in 0..=j {
                let idx_ik = chol_lt_idx(i, k, n_eta);
                let idx_jk = chol_lt_idx(j, k, n_eta);
                // Chain rule: ∂L_{ab}/∂x_{ab} = L_{ab} if a==b (log), else 1.
                let chain_ik = if i == k { l[(i, k)] } else { 1.0 };
                let chain_jk = if j == k { l[(j, k)] } else { 1.0 };
                grad[idx_ik] += l[(j, k)] * chain_ik;
                if i != j {
                    grad[idx_jk] += l[(i, k)] * chain_jk;
                } else {
                    // i == j: both terms contribute to the same index
                    grad[idx_ik] += l[(i, k)] * chain_ik;
                }
            }
            // SE²(omega_{ij}) = g^T * C_omega * g
            let mut var = 0.0;
            for a in 0..n_lt {
                if grad[a] == 0.0 {
                    continue;
                }
                for b in 0..n_lt {
                    if grad[b] == 0.0 {
                        continue;
                    }
                    var += grad[a] * cov_omega[(a, b)] * grad[b];
                }
            }
            se_vec.push(if var > 0.0 { var.sqrt() } else { 0.0 });
        }
        se_vec
    };

    // Sigma: SE via delta method (log-transformed)
    let sigma_start = omega_start + omega_packed_len(n_eta, template.omega.diagonal);
    let se_sigma: Vec<f64> = (0..n_sigma)
        .map(|i| {
            let idx = sigma_start + i;
            if idx < n {
                template.sigma.values[i] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    // IOV (kappa): SE for diagonal variances of omega_iov.
    //
    // The packed Cholesky layout is column-major (see `pack_params`); the flat
    // index of `L[i,i]` within the IOV block is `chol_lt_idx(i, i, n_kappa)` (the
    // single source of that offset — do not re-spell the formula here).
    // Same delta-method approximation as `se_omega`: SE(var_i) ≈ 2 * var_i * SE(log L_ii),
    // which is exact for diagonal IOV and a first-order approximation for block_kappa.
    // Off-diagonal covariance SEs are not currently reported (matches BSV omega).
    let kappa_start = sigma_start + n_sigma;
    let se_kappa: Option<Vec<f64>> = template.omega_iov.as_ref().map(|iov| {
        let n_kappa = iov.dim();
        (0..n_kappa)
            .map(|i| {
                let idx = if iov.diagonal {
                    kappa_start + i
                } else {
                    kappa_start + chol_lt_idx(i, i, n_kappa)
                };
                if idx < n {
                    2.0 * iov.matrix[(i, i)] * se_packed[idx]
                } else {
                    0.0
                }
            })
            .collect()
    });

    (Some(se_theta), Some(se_omega), Some(se_sigma), se_kappa)
}
