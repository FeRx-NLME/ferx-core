//! Model-selection primitives for automated model development (#1177, part of
//! the #1175 search epic): the Delattre-style BIC variants a search ranks on,
//! and the [`Strictness`] predicate that decides whether a candidate fit is
//! *eligible* to be ranked at all.
//!
//! Both operate on a finished [`FitResult`] only, so a search tool can rank a
//! `.fitrx` bundle without the model or data in hand, and ferx-r gets the same
//! answers from the same fields.
//!
//! # BIC variants
//!
//! `FitResult::bic` is the classical `OFV + p·ln(n_obs)`. Pharmpy's `iivsearch`
//! and `modelsearch` rank on the Delattre et al. (2014) *mixed* BIC instead,
//! which penalises population parameters of the random-effects class on
//! `ln(n_subjects)` and the rest on `ln(n_obs)` — ranking IIV structures on the
//! observation-count BIC systematically favours the wrong model. [`bic`] offers
//! all four of Pharmpy's variants, with the class tally coming from
//! `FitResult::bic_inputs` (filled by `fit()`).
//!
//! # Strictness
//!
//! `FitResult::converged` is a bool. It does not distinguish a genuine optimum
//! from an init stall (#751), a boundary estimate, an ill-conditioned
//! covariance step or a near-singular correlation matrix. Under automation all
//! of those become *model-selection errors*: a candidate that never left its
//! initial estimates is ranked on an OFV that says nothing about the model.
//! [`check_strictness`] evaluates the pyDarwin-style gates and returns the
//! *reasons*, so a search report can say why a candidate was excluded.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::estimation::parameterization::omega_packed_len;
use crate::types::{
    BicInputs, CompiledModel, CovarianceStatus, FitResult, ModelParameters, WarningCode,
};

/// Which BIC penalty to apply — the four variants of
/// `pharmpy.modeling.calculate_bic`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BicType {
    /// Delattre et al. (2014): random-class parameters on `ln(n_subjects)`,
    /// fixed-class on `ln(n_obs)`. Pharmpy's default and the one its structural
    /// and IIV searches rank on.
    #[default]
    Mixed,
    /// Free BSV Ω elements only, on `ln(n_subjects)` — Pharmpy's `iivsearch`
    /// criterion for comparing variability structures.
    Iiv,
    /// Every free parameter on `ln(n_subjects)`.
    Random,
    /// Every free parameter on `ln(n_obs)` — identical to `FitResult::bic`.
    Fixed,
}

impl BicInputs {
    /// Total free parameters in the tally; equals `FitResult::n_parameters`
    /// for a result produced by `fit()`.
    pub fn n_free(&self) -> usize {
        self.theta_random + self.theta_fixed + self.omega + self.kappa + self.sigma
    }

    /// Free parameters penalised on `ln(n_subjects)` by the mixed BIC.
    pub fn n_random_class(&self) -> usize {
        self.theta_random + self.omega + self.kappa + if self.sigma_random { self.sigma } else { 0 }
    }

    /// Free parameters penalised on `ln(n_obs)` by the mixed BIC.
    pub fn n_fixed_class(&self) -> usize {
        self.theta_fixed + if self.sigma_random { 0 } else { self.sigma }
    }
}

/// The BIC of a fit under one of the four penalty conventions.
///
/// `OFV + penalty`, where the penalty is:
///
/// | kind     | penalty |
/// |----------|---------|
/// | `Mixed`  | `n_random_class·ln(n_subjects) + n_fixed_class·ln(n_obs)` |
/// | `Iiv`    | `n_omega·ln(n_subjects)` |
/// | `Random` | `n_parameters·ln(n_subjects)` |
/// | `Fixed`  | `n_parameters·ln(n_obs)` |
///
/// with the counts from `FitResult::bic_inputs` and `n_obs` the same record
/// count `FitResult::bic` uses (Gaussian observations plus, under `survival`,
/// time-to-event records). `Fixed` therefore reproduces `FitResult::bic`
/// exactly.
///
/// Returns `NaN` when the tally cannot support the penalty: a bundle saved
/// before the tally existed (all-zero counts with `n_parameters > 0`), or a
/// zero subject / record count under a convention that takes its log.
pub fn bic(result: &FitResult, kind: BicType) -> f64 {
    let inp = &result.bic_inputs;
    if inp.n_free() != result.n_parameters {
        return f64::NAN;
    }
    let ln_subj = || {
        if result.n_subjects > 0 {
            Some((result.n_subjects as f64).ln())
        } else {
            None
        }
    };
    let ln_obs = || {
        if inp.n_obs > 0 {
            Some((inp.n_obs as f64).ln())
        } else {
            None
        }
    };
    let penalty = match kind {
        BicType::Fixed => ln_obs().map(|l| result.n_parameters as f64 * l),
        BicType::Random => ln_subj().map(|l| result.n_parameters as f64 * l),
        BicType::Iiv => ln_subj().map(|l| inp.omega as f64 * l),
        BicType::Mixed => {
            let random = inp.n_random_class();
            let fixed = inp.n_fixed_class();
            // A class with no members does not need its log to exist.
            let r = if random == 0 {
                Some(0.0)
            } else {
                ln_subj().map(|l| random as f64 * l)
            };
            let f = if fixed == 0 {
                Some(0.0)
            } else {
                ln_obs().map(|l| fixed as f64 * l)
            };
            match (r, f) {
                (Some(r), Some(f)) => Some(r + f),
                _ => None,
            }
        }
    };
    match penalty {
        Some(p) => result.ofv + p,
        None => f64::NAN,
    }
}

/// Tally the free packed parameters by Delattre class.
///
/// `held_mask` is `packed_held_mask(template)` (FIX or structural zero) — the
/// same mask `fit()` counts `n_parameters` from — and is walked segment by segment in
/// `pack_params` order: `θ`, `Ω` (Cholesky lower triangle), `σ`, `Ω_IOV`,
/// mixture Ω overrides, mixture σ overrides, `block_sigma` correlations. Each
/// segment's free entries land in one class, except `θ`, whose entries split on
/// `CompiledModel::theta_eta_linked`.
pub(crate) fn bic_inputs_for(
    model: &CompiledModel,
    template: &ModelParameters,
    held_mask: &[bool],
    n_obs: usize,
) -> BicInputs {
    let n_theta = template.theta.len();
    let n_omega = omega_packed_len(template.omega.dim(), template.omega.diagonal);
    let n_sigma = template.sigma.values.len();
    let n_iov = template
        .omega_iov
        .as_ref()
        .map_or(0, |m| omega_packed_len(m.dim(), m.diagonal));
    let (n_mix_omega, n_mix_sigma) = template.mixture.as_ref().map_or((0, 0), |mix| {
        (mix.omega_override_addr.len(), mix.sigma_override_addr.len())
    });
    let n_rho = template.residual_correlations.len();
    debug_assert_eq!(
        held_mask.len(),
        n_theta + n_omega + n_sigma + n_iov + n_mix_omega + n_mix_sigma + n_rho,
        "held mask must cover every packed segment"
    );

    let free = |range: std::ops::Range<usize>| -> usize {
        held_mask
            .get(range)
            .map_or(0, |seg| seg.iter().filter(|&&h| !h).count())
    };
    let mut out = BicInputs {
        n_obs,
        sigma_random: model.residual_error_eta.is_some(),
        ..BicInputs::default()
    };
    for (i, &held) in held_mask.iter().take(n_theta).enumerate() {
        if held {
            continue;
        }
        if model.theta_eta_linked.get(i).copied().unwrap_or(false) {
            out.theta_random += 1;
        } else {
            out.theta_fixed += 1;
        }
    }
    let mut cursor = n_theta;
    out.omega += free(cursor..cursor + n_omega);
    cursor += n_omega;
    out.sigma += free(cursor..cursor + n_sigma);
    cursor += n_sigma;
    out.kappa += free(cursor..cursor + n_iov);
    cursor += n_iov;
    out.omega += free(cursor..cursor + n_mix_omega);
    cursor += n_mix_omega;
    out.sigma += free(cursor..cursor + n_mix_sigma);
    cursor += n_mix_sigma;
    out.sigma += free(cursor..cursor + n_rho);
    out
}

/// The gates a candidate fit must pass before its criterion is trusted.
///
/// `Default` is the pyDarwin posture: require convergence, do not require a
/// covariance step, reject a condition number above 1000 or a parameter
/// correlation above 0.95 *when a covariance matrix is present*, reject
/// boundary estimates and init stalls. [`Strictness::none`] turns every gate
/// off for a "rank everything, report everything" run.
///
/// The two threshold gates are evaluated only when their input exists — a fit
/// without a covariance matrix has no condition number to test. Combine them
/// with `require_covariance` to make them mandatory; on their own they report
/// the untestable case in [`StrictnessVerdict::skipped`] rather than failing
/// or silently passing it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Strictness {
    /// Fail a fit with `converged == false`. This already covers an internal
    /// runaway-guard hit, which demotes `converged` (#1118).
    pub require_converged: bool,
    /// Fail unless the covariance step ran and produced uncertainty:
    /// `CovarianceStatus::Computed`, or `SirFallback` — the FD Hessian was not
    /// positive-definite but SIR delivered credible intervals, which is the
    /// uncertainty the user configured that fallback to produce. A SIR-fallback
    /// fit stores no covariance matrix, so the condition-number and correlation
    /// gates then report `skipped` rather than a verdict.
    pub require_covariance: bool,
    /// Fail when `FitResult::cov_condition_number` (largest over smallest
    /// eigenvalue of the free-parameter correlation matrix) exceeds this.
    /// pyDarwin's default is 1000.
    pub max_condition_number: Option<f64>,
    /// Fail when any off-diagonal of the covariance matrix's correlation form
    /// exceeds this in absolute value. pyDarwin's default is 0.95.
    pub max_correlation: Option<f64>,
    /// Fail a fit with a θ pinned to a declared bound — the predicate
    /// `bootstrap`'s `skip_estimate_near_boundary` applies
    /// ([`estimate_near_boundary`]).
    pub reject_on_boundary: bool,
    /// Fail a fit that never left its initial estimates ([`stalled_at_init`]).
    pub reject_init_stall: bool,
}

impl Default for Strictness {
    fn default() -> Self {
        Self {
            require_converged: true,
            require_covariance: false,
            max_condition_number: Some(1000.0),
            max_correlation: Some(0.95),
            reject_on_boundary: true,
            reject_init_stall: true,
        }
    }
}

impl Strictness {
    /// Every gate off: [`check_strictness`] passes any fit.
    pub fn none() -> Self {
        Self {
            require_converged: false,
            require_covariance: false,
            max_condition_number: None,
            max_correlation: None,
            reject_on_boundary: false,
            reject_init_stall: false,
        }
    }
}

/// The outcome of [`check_strictness`]: whether the fit passed, and the named
/// reason for every gate it failed or that could not be evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StrictnessVerdict {
    /// `failures.is_empty()`.
    pub passed: bool,
    /// One entry per failed gate, in [`Strictness`] field order, each naming the
    /// gate and the value that tripped it.
    pub failures: Vec<String>,
    /// Gates that were enabled but had no input to test on this fit (no
    /// covariance matrix, no initial estimates on an old bundle). Not failures,
    /// but a search report should show them.
    pub skipped: Vec<String>,
}

/// Evaluate every enabled gate of `s` against `result`.
pub fn check_strictness(result: &FitResult, s: &Strictness) -> StrictnessVerdict {
    let mut v = StrictnessVerdict::default();

    if s.require_converged && !result.converged {
        v.failures
            .push("did not converge (`converged = false`)".to_string());
    }

    if s.require_covariance {
        let reason = match result.covariance_status {
            CovarianceStatus::Computed if result.covariance_matrix.is_some() => None,
            CovarianceStatus::Computed => {
                Some("covariance step reported computed but stored no matrix")
            }
            CovarianceStatus::NotRequested => {
                Some("covariance step not requested (`covariance = false`)")
            }
            CovarianceStatus::Failed => Some("covariance step failed"),
            // Accepted: the user opted into SIR as the uncertainty source when
            // the Hessian is not PD, and it delivered.
            CovarianceStatus::SirFallback => None,
        };
        if let Some(r) = reason {
            v.failures.push(r.to_string());
        }
    }

    if let Some(max_cn) = s.max_condition_number {
        match result.cov_condition_number {
            Some(cn) if cn.is_nan() => {
                v.failures.push("condition number is NaN".to_string());
            }
            Some(cn) if cn > max_cn => {
                v.failures
                    .push(format!("condition number {cn:.4e} exceeds {max_cn:.4e}"));
            }
            Some(_) => {}
            None => v.skipped.push(
                "condition number: no covariance matrix (covariance step not run or failed)"
                    .to_string(),
            ),
        }
    }

    if let Some(max_r) = s.max_correlation {
        match max_abs_correlation_indexed(result.covariance_matrix.as_ref()) {
            Some((r, a, b)) if r > max_r => {
                let name = |k: usize| {
                    result
                        .theta_names
                        .get(k)
                        .cloned()
                        .unwrap_or_else(|| format!("packed coordinate {k}"))
                };
                v.failures.push(format!(
                    "parameter correlation |r| = {r:.4} exceeds {max_r:.4} ({} ~ {})",
                    name(a),
                    name(b)
                ));
            }
            Some(_) => {}
            None => v.skipped.push(
                "parameter correlation: no covariance matrix with two or more free parameters"
                    .to_string(),
            ),
        }
    }

    if s.reject_on_boundary {
        if let Some(entry) = result
            .warnings_structured
            .iter()
            .find(|w| w.category == WarningCode::BoundaryEstimate)
        {
            let names: Vec<String> = entry
                .details
                .as_ref()
                .and_then(|d| d.get("parameters"))
                .and_then(|p| p.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|p| p.get("parameter").and_then(|n| n.as_str()))
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let what = if names.is_empty() {
                entry.message.clone()
            } else {
                names.join(", ")
            };
            v.failures
                .push(format!("estimate pinned to a declared bound: {what}"));
        }
    }

    if s.reject_init_stall {
        match stalled_at_init(result) {
            Some(true) => v.failures.push(format!(
                "stalled at the initial estimates: no free parameter moved more than \
                 {:.0}% of its initial value (#751)",
                INIT_STALL_REL_TOL * 100.0
            )),
            Some(false) => {}
            None => v.skipped.push(
                "init stall: result carries no initial estimates to compare against".to_string(),
            ),
        }
    }

    v.passed = v.failures.is_empty();
    v
}

/// Whether any θ estimate is pinned to a declared optimizer bound — the
/// predicate behind `bootstrap`'s `skip_estimate_near_boundary` filter and
/// [`Strictness::reject_on_boundary`].
///
/// Reads the fit's own `BoundaryEstimate` warning, so it agrees with what the
/// user was told. A hit on an *internal* guard (the implicit θ caps, the Ω/σ
/// safety limits) is reported as `ParameterAtRunawayGuard` instead and demotes
/// `converged` (#1118); it is not a boundary estimate in this sense.
pub fn estimate_near_boundary(result: &FitResult) -> bool {
    result
        .warnings_structured
        .iter()
        .any(|w| w.category == WarningCode::BoundaryEstimate)
}

/// Relative displacement below which a free parameter counts as "still at its
/// initial value" for [`stalled_at_init`].
///
/// This is the natural-scale reading of the optimizer's `INIT_ESCAPE_STEP_S`
/// test (#751), which works in its scaled packed space where a coordinate is
/// normalised by its own magnitude — a ~1% move on any coordinate. On the
/// natural scale a 1% relative move is at least as easy to register, so a fit
/// the optimizer flagged as stalled is flagged here too, never the reverse.
pub const INIT_STALL_REL_TOL: f64 = 1e-2;

/// Whether the fit never left its initial estimates: no free θ, Ω or σ moved
/// by more than [`INIT_STALL_REL_TOL`] of its initial value (or by more than
/// that tolerance in absolute terms when the initial value is exactly zero,
/// the identity-packed case).
///
/// This is the #751 signature — a fit whose OFV is the OFV *of the initial
/// estimates*, which says nothing about the model — surfaced as a named
/// predicate on the result. Fixed parameters are ignored; a fit with no free
/// parameter at all cannot stall and reports `Some(false)`. IOV κ variances
/// are not compared (the result carries no κ initials).
///
/// `None` when the result carries no comparable initial estimates: a `.fitrx`
/// bundle saved before they were recorded, or shape-mismatched vectors.
pub fn stalled_at_init(result: &FitResult) -> Option<bool> {
    let shapes_ok = result.theta_init.len() == result.theta.len()
        && result.sigma_init.len() == result.sigma.len()
        && result.omega_init.shape() == result.omega.shape()
        && !(result.theta.is_empty() && result.sigma.is_empty() && result.omega.is_empty());
    if !shapes_ok {
        return None;
    }
    let moved = |x: f64, x0: f64| -> bool {
        let d = (x - x0).abs();
        if x0 == 0.0 {
            d > INIT_STALL_REL_TOL
        } else {
            d > INIT_STALL_REL_TOL * x0.abs()
        }
    };
    let is_fixed = |mask: &[bool], i: usize| mask.get(i).copied().unwrap_or(false);

    for (i, (&x, &x0)) in result.theta.iter().zip(&result.theta_init).enumerate() {
        if !is_fixed(&result.theta_fixed, i) && moved(x, x0) {
            return Some(false);
        }
    }
    for (i, (&x, &x0)) in result.sigma.iter().zip(&result.sigma_init).enumerate() {
        if !is_fixed(&result.sigma_fixed, i) && moved(x, x0) {
            return Some(false);
        }
    }
    let n = result.omega.nrows();
    for i in 0..n {
        for j in 0..=i {
            // Block-Ω FIX semantics: an element is held when either eta is.
            if is_fixed(&result.omega_fixed, i) || is_fixed(&result.omega_fixed, j) {
                continue;
            }
            if moved(result.omega[(i, j)], result.omega_init[(i, j)]) {
                return Some(false);
            }
        }
    }
    Some(true)
}

/// Largest absolute off-diagonal correlation implied by the fit's covariance
/// matrix, over every coordinate with a positive variance (θ, Ω, σ alike —
/// pyDarwin's and Pharmpy's `check_high_correlations` read the whole matrix,
/// unlike the θ-only `HighCorrelation` warning).
///
/// `None` without a covariance matrix or with fewer than two usable
/// coordinates.
pub fn max_abs_correlation(result: &FitResult) -> Option<f64> {
    max_abs_correlation_indexed(result.covariance_matrix.as_ref()).map(|(r, _, _)| r)
}

/// [`max_abs_correlation`] with the packed coordinate pair that attains it.
fn max_abs_correlation_indexed(cov: Option<&DMatrix<f64>>) -> Option<(f64, usize, usize)> {
    let cov = cov?;
    let n = cov.nrows().min(cov.ncols());
    let usable: Vec<usize> = (0..n)
        .filter(|&i| cov[(i, i)] > 0.0 && cov[(i, i)].is_finite())
        .collect();
    if usable.len() < 2 {
        return None;
    }
    let mut best: Option<(f64, usize, usize)> = None;
    for (ai, &a) in usable.iter().enumerate() {
        for &b in &usable[ai + 1..] {
            // sqrt·sqrt rather than sqrt(product): no overflow on extreme variances.
            let denom = cov[(a, a)].sqrt() * cov[(b, b)].sqrt();
            let r = (cov[(a, b)] / denom).abs();
            if !r.is_finite() {
                continue;
            }
            if best.is_none_or(|(br, _, _)| r > br) {
                best = Some((r, a, b));
            }
        }
    }
    best
}

#[cfg(test)]
#[path = "model_selection_tests.rs"]
mod tests;
