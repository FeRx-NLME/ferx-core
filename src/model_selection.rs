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
/// zero subject / record count under a convention that needs its log. A
/// count of zero never needs a log — every variant is `ofv` for a fit with no
/// free parameter, whatever `n_subjects` / `n_obs` say — so the four variants
/// agree on such a fit and `Fixed` stays equal to `FitResult::bic` on it.
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
    // `count · ln(n)`, where a class with no members does not need its log
    // to exist.
    let term = |count: usize, ln: &dyn Fn() -> Option<f64>| -> Option<f64> {
        if count == 0 {
            Some(0.0)
        } else {
            ln().map(|l| count as f64 * l)
        }
    };
    let penalty = match kind {
        BicType::Fixed => term(result.n_parameters, &ln_obs),
        BicType::Random => term(result.n_parameters, &ln_subj),
        BicType::Iiv => term(inp.omega, &ln_subj),
        BicType::Mixed => match (
            term(inp.n_random_class(), &ln_subj),
            term(inp.n_fixed_class(), &ln_obs),
        ) {
            (Some(r), Some(f)) => Some(r + f),
            _ => None,
        },
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
    /// uncertainty the user configured that fallback to produce. A fallback
    /// whose importance weights collapsed onto a single draw (Kish effective
    /// sample size below [`MIN_SIR_FALLBACK_ESS`]) delivered none — every
    /// interval has zero width — and fails like a failed step. A SIR-fallback
    /// fit stores no covariance matrix, so the condition-number and correlation
    /// gates then report `skipped` rather than a verdict.
    pub require_covariance: bool,
    /// Fail when `FitResult::cov_condition_number` (largest over smallest
    /// eigenvalue of the free-parameter correlation matrix) exceeds this.
    /// pyDarwin's default is 1000.
    pub max_condition_number: Option<f64>,
    /// Fail when any off-diagonal of the covariance matrix's correlation form
    /// exceeds this in absolute value ([`max_abs_correlation`]: the whole
    /// matrix, on the natural θ / Ω / σ scale). pyDarwin's default is 0.95.
    pub max_correlation: Option<f64>,
    /// Fail a fit with a θ pinned to a declared bound — the predicate
    /// `bootstrap`'s `skip_estimate_near_boundary` applies
    /// ([`estimate_near_boundary`]).
    pub reject_on_boundary: bool,
    /// Fail a fit that never left its initial estimates ([`stalled_at_init`]).
    ///
    /// This gate assumes a **cold start** from model-file initial estimates,
    /// where "did not move" means the OFV is the OFV *of the inits*. A
    /// candidate warm-started from its parent's final estimates (or a refit of
    /// a stored winner) legitimately converges within the tolerance of where it
    /// began, and the result carries nothing that distinguishes the two — turn
    /// this gate off for such a search and rely on `require_converged`.
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
                Some("covariance step reported computed but stored no matrix".to_string())
            }
            CovarianceStatus::NotRequested => {
                Some("covariance step not requested (`covariance = false`)".to_string())
            }
            CovarianceStatus::Failed => Some("covariance step failed".to_string()),
            // Accepted when the user opted into SIR as the uncertainty source
            // for a non-PD Hessian and it delivered — which a run whose
            // weights collapsed onto one draw did not, whatever `Ok` it
            // returned.
            CovarianceStatus::SirFallback => match result.sir_ess {
                Some(ess) if ess.is_finite() && ess >= MIN_SIR_FALLBACK_ESS => None,
                Some(ess) => Some(format!(
                    "SIR fallback collapsed: effective sample size {ess:.2} is below \
                     {MIN_SIR_FALLBACK_ESS} (every credible interval has zero width)"
                )),
                None => Some(
                    "SIR fallback recorded no effective sample size to judge it by".to_string(),
                ),
            },
        };
        if let Some(r) = reason {
            v.failures.push(r);
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
        match max_abs_correlation_indexed(result) {
            Some((r, a, b)) if r > max_r => {
                let n = result.covariance_matrix.as_ref().map_or(0, |c| c.nrows());
                let names = crate::io::output::packed_param_names(result, n);
                let name = |k: usize| {
                    names
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

/// Kish effective sample size below which a SIR *fallback* counts as having
/// delivered no uncertainty for [`Strictness::require_covariance`]: fewer than
/// two effective draws means the resample is one point repeated, and every
/// credible interval it reports has zero width. (SIR itself returns `Ok` with a
/// single finite weight and only prints the ESS under `verbose`.)
pub const MIN_SIR_FALLBACK_ESS: f64 = 2.0;

/// Relative displacement below which a free parameter counts as "still at its
/// initial value" for the natural-scale fallback of [`stalled_at_init`].
///
/// The optimizer's own `INIT_ESCAPE_STEP_S` test (#751) is a ~1 % move in its
/// *scaled packed* space, where a log-packed coordinate is normalised by
/// `|ln θ₀|`; on the natural scale that is roughly a `|ln θ₀|` % move, so the
/// two readings can disagree on a borderline fit (V₀ = 100 → 103 is a stall to
/// the optimizer and a move here). The optimizer's verdict is therefore
/// preferred whenever the result carries it (`FitResult::left_init`); this
/// tolerance only judges results that do not.
pub const INIT_STALL_REL_TOL: f64 = 1e-2;

/// Whether the fit never left its initial estimates — the #751 signature, a
/// fit whose OFV is the OFV *of the initial estimates* and says nothing about
/// the model — surfaced as a named predicate on the result.
///
/// Reads `FitResult::left_init`, the outer optimizer's own escape test, when
/// the fit recorded one. Otherwise (an older `.fitrx` bundle, an estimator
/// that does not run the test) it compares the estimates against the initial
/// values: stalled when no free θ, Ω or σ moved by more than
/// [`INIT_STALL_REL_TOL`] of its initial value (or by more than that tolerance
/// in absolute terms when the initial value is exactly zero, the
/// identity-packed case). Fixed parameters are ignored; a fit with no free
/// parameter at all cannot stall and reports `Some(false)`. IOV κ variances
/// are not compared (the result carries no κ initials).
///
/// Both readings are relative to where the fit *started*, which for a
/// warm-started candidate is not the model file — see
/// [`Strictness::reject_init_stall`].
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

    // Nothing free cannot stall — and the optimizer's own test reads "did not
    // move" on an empty vector, so this has to come first.
    let mut saw_free = false;
    let mut any_moved = false;
    for (i, (&x, &x0)) in result.theta.iter().zip(&result.theta_init).enumerate() {
        if !is_fixed(&result.theta_fixed, i) {
            saw_free = true;
            any_moved |= moved(x, x0);
        }
    }
    for (i, (&x, &x0)) in result.sigma.iter().zip(&result.sigma_init).enumerate() {
        if !is_fixed(&result.sigma_fixed, i) {
            saw_free = true;
            any_moved |= moved(x, x0);
        }
    }
    let n = result.omega.nrows();
    for i in 0..n {
        for j in 0..=i {
            // Block-Ω FIX semantics: an element is held when either eta is.
            if is_fixed(&result.omega_fixed, i) || is_fixed(&result.omega_fixed, j) {
                continue;
            }
            saw_free = true;
            any_moved |= moved(result.omega[(i, j)], result.omega_init[(i, j)]);
        }
    }
    if !saw_free {
        return Some(false);
    }
    if let Some(left) = result.left_init {
        return Some(!left);
    }
    Some(!any_moved)
}

/// Largest absolute off-diagonal correlation implied by the fit's covariance
/// matrix, over every coordinate with a positive variance (θ, Ω, σ alike —
/// pyDarwin's and Pharmpy's `check_high_correlations` read the whole matrix,
/// unlike the θ-only `HighCorrelation` warning).
///
/// Read on the **natural** scale, which is what those tools see in a NONMEM
/// `.cor`: the stored matrix is on the packed scale, and while a log-packed θ
/// or σ, or the diagonal of a diagonal Ω, is a monotone function of one packed
/// coordinate (correlations unchanged), a `block_omega` element mixes several
/// Cholesky entries, so its packed-scale correlations are not the ω
/// correlations. [`natural_scale_covariance`] applies that delta-method
/// transform before the correlations are read.
///
/// `None` without a covariance matrix or with fewer than two usable
/// coordinates.
pub fn max_abs_correlation(result: &FitResult) -> Option<f64> {
    max_abs_correlation_indexed(result).map(|(r, _, _)| r)
}

/// The fit's covariance matrix with every `block_omega` / block-κ segment
/// mapped from the packed Cholesky coordinates onto the natural `ω_ij` by the
/// delta method (`J C Jᵀ`, the same Jacobian the reported `se_omega` uses),
/// every other coordinate left as stored.
///
/// The untouched coordinates are each a monotone function of exactly one
/// packed coordinate (`θ = exp(x)` or identity, `σ = exp(x)`, `ω_ii = L_ii²`),
/// whose Jacobian is diagonal and cancels out of a correlation — so for a fit
/// with no block Ω this is the stored matrix and the correlations are already
/// natural-scale. `None` without a covariance matrix.
pub fn natural_scale_covariance(result: &FitResult) -> Option<DMatrix<f64>> {
    let cov = result.covariance_matrix.as_ref()?;
    let n = cov.nrows().min(cov.ncols());
    let (omega_diagonal, kappa_diagonal) = crate::io::output::packed_layout(result, n);
    let n_theta = result.theta_names.len();
    let n_eta = result.omega.nrows();
    let n_sigma = result.sigma_names.len();
    let n_kappa = result.kappa_names.len();
    let n_omega = omega_packed_len(n_eta, omega_diagonal);

    let mut j = DMatrix::<f64>::identity(n, n);
    let mut embed = |start: usize, matrix: &DMatrix<f64>, diagonal: bool| {
        let len = omega_packed_len(matrix.nrows(), diagonal);
        if diagonal || len == 0 || start + len > n {
            return;
        }
        let Some(l) = matrix.clone().cholesky() else {
            return;
        };
        let block = crate::estimation::parameterization::omega_cholesky_jacobian(&l.l());
        j.view_mut((start, start), (len, len)).copy_from(&block);
    };
    embed(n_theta, &result.omega, omega_diagonal);
    if n_kappa > 0 {
        if let Some(iov) = result.omega_iov.as_ref() {
            embed(n_theta + n_omega + n_sigma, iov, kappa_diagonal);
        }
    }
    let cov = cov.view((0, 0), (n, n));
    Some(&j * cov * j.transpose())
}

/// [`max_abs_correlation`] with the packed coordinate pair that attains it.
fn max_abs_correlation_indexed(result: &FitResult) -> Option<(f64, usize, usize)> {
    let cov = natural_scale_covariance(result)?;
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
