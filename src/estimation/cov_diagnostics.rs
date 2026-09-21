//! The user-facing text of the covariance step's regularization diagnostic (#520).
//!
//! Everything here is a **pure function over a small struct of facts**. The facts are
//! collected by [`crate::estimation::covariance::compute_covariance`] (which route served the
//! Hessian, what the eigen-floor did, which gate clause declined the analytic R-matrix, what
//! tolerance the ODEs integrate at); the prose is assembled here and nowhere else, so the
//! message can be unit-tested at every reachable combination of those facts without running a
//! fit.
//!
//! Why the split exists at all: before #520 the severity of `covariance_regularized` was graded
//! on the **count fraction** of clipped eigenvalues, so a single badly-negative eigenvalue —
//! 1 of 13, 7% — printed *"severity: minor. Standard errors are likely reliable."* directly
//! above an SE inflated 4400× and a `%RSE` of 24982. Count says how many directions the floor
//! touched; it says nothing about how far they were floored or how much of a parameter's
//! variance came out of the floor rather than out of the data. Both of those are magnitudes,
//! and both are graded here ([`grade_severity`]).
//!
//! The message's input space is enumerated in the sibling test file: route × severity ×
//! ODE-at-loose-tolerance × declined-clause. Every sentence must be true on every reachable
//! cell, which is why each conditional sentence is gated on the fact that makes it true rather
//! than emitted unconditionally.

use crate::ode::OdeSolverOptions;
use crate::types::CompiledModel;

/// Which Hessian the covariance step differentiated (#520).
///
/// The distinction is user-visible: the finite-difference stencil divides by `h²` and so
/// amplifies whatever noise the objective carries (integration error on an `[odes]` model, most
/// of all), while the exact analytic R-matrix from third-order sensitivities never
/// second-differences the objective at all. Advice that names `fd_hessian_step` or
/// `ode_reltol` is true on the first and false on the second, and before #520 the message said
/// "eigenvalue floor applied to FD Hessian" on both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CovHessianSource {
    /// Exact analytic R-matrix assembled from third-order sensitivities (#436, extended to
    /// `[odes]` models by #1291). No stencil, no step size.
    AnalyticRMatrix,
    /// Central second differences of the reconverged objective. The `1/h²` route.
    FdStencil,
}

impl CovHessianSource {
    /// How the message names this Hessian.
    pub(crate) fn label(self) -> &'static str {
        match self {
            CovHessianSource::AnalyticRMatrix => "the analytic R-matrix",
            CovHessianSource::FdStencil => "the FD Hessian",
        }
    }
}

/// Why the exact analytic covariance R-matrix was not used, named at the clause that declined
/// it (#520 C2).
///
/// The variants are in one-to-one correspondence with the clauses of the scope gate in
/// [`crate::sens::provider`] plus the three gates
/// [`crate::estimation::covariance::compute_covariance`] applies before it. There is exactly
/// one implementation: the gate itself returns this enum and tests `.is_some()`, so the clause
/// a user is told about cannot drift from the clause that actually fired.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CovScopeDecline {
    /// `analytic_cov_hessian = false` in `[fit_options]`.
    Disabled,
    /// Mixture model — the analytic assembly has no class-marginal term.
    Mixture,
    /// `method = laplace`, or an AGQ fit anchored on the exact `H`: the second derivative of
    /// `H` needs fourth-order sensitivities, which nothing computes.
    ExactHessianAnchor,
    /// The model itself is outside the analytic sensitivity scope (`ode_analytical_supported`
    /// / `analytical_supported` / their IOV twins declined).
    ModelOutOfScope,
    /// `gradient = fd` — the user's explicit opt-out from analytic sensitivities.
    GradientFd,
    /// A non-Gaussian data term (`[event_model]` TTE, `[binary_model]`, CTMM).
    NonGaussianEndpoint,
    /// FREM: covariate pseudo-observation rows are scored with an `EPSCOV²` override the
    /// Gaussian assembly does not apply.
    Frem,
    /// `ErrorSpec::Selected` — endpoints keyed by covariate branch rather than by CMT.
    SelectedErrorSpec,
    /// Log-transform-both-sides.
    LogTransform,
    /// The IOV shape the covariance assembly cannot serve (κ without the IOV arm, the IOV arm
    /// without κ, or IOV together with a lagtime).
    IovShape,
    /// `[scaling] obs_scale = ...`, i.e. `ScalingSpec::ExpressionScale`. The one clause with a
    /// one-line rewrite out of it.
    ExpressionScale,
    /// A Form-C `analytic_readout`.
    AnalyticReadout,
    /// `init(...)` on a closed-form model.
    AnalyticalInit,
    /// An η on the residual error.
    ResidualErrorEta,
    /// Declared residual correlations.
    ResidualCorrelations,
    /// A custom RUV magnitude expression.
    CustomRuvMagnitude,
    /// The subject routes to the event-driven walk on a closed-form model.
    EventWalkSubject,
    /// The individual-parameter program does not cover the required PK slots.
    IndivParamProgram,
    /// Every model-level clause passed but the per-subject assembly still bailed (a per-point
    /// provider decline, a non-finite block, an occasion/κ length mismatch, …).
    PerSubjectBail,
}

impl CovScopeDecline {
    /// The clause, as the message names it. Reads as the object of "declined because …".
    pub(crate) fn clause(self) -> &'static str {
        match self {
            CovScopeDecline::Disabled => "analytic_cov_hessian = false",
            CovScopeDecline::Mixture => "the model is a mixture model",
            CovScopeDecline::ExactHessianAnchor => {
                "the fit is anchored on the exact Hessian (method = laplace, or AGQ with that \
                 anchor), whose second derivative needs fourth-order sensitivities"
            }
            CovScopeDecline::ModelOutOfScope => {
                "the model is outside the analytic sensitivity scope"
            }
            CovScopeDecline::GradientFd => "gradient = fd",
            CovScopeDecline::NonGaussianEndpoint => {
                "the model has a non-Gaussian endpoint ([event_model] / [binary_model])"
            }
            CovScopeDecline::Frem => "the model is a FREM model",
            CovScopeDecline::SelectedErrorSpec => {
                "the error model selects endpoints by covariate branch"
            }
            CovScopeDecline::LogTransform => "the model is log-transform-both-sides",
            CovScopeDecline::IovShape => "the model's IOV structure is outside the assembly",
            CovScopeDecline::ExpressionScale => "[scaling] obs_scale = ... is in use",
            CovScopeDecline::AnalyticReadout => "the model declares an analytic readout",
            CovScopeDecline::AnalyticalInit => "the model declares init(...) compartment amounts",
            CovScopeDecline::ResidualErrorEta => "the residual error carries a random effect",
            CovScopeDecline::ResidualCorrelations => "the model declares residual correlations",
            CovScopeDecline::CustomRuvMagnitude => "the model declares a custom RUV magnitude",
            CovScopeDecline::EventWalkSubject => {
                "at least one subject routes to the event-driven walk"
            }
            CovScopeDecline::IndivParamProgram => {
                "the individual-parameter program does not cover the required PK slots"
            }
            CovScopeDecline::PerSubjectBail => {
                "at least one subject fell outside the analytic assembly"
            }
        }
    }

    /// The one-line way out, where there is one. `None` means the clause is a property of the
    /// model the user asked for, and the honest answer is that the FD stencil is the correct
    /// route for it — so the message says nothing rather than inventing advice.
    pub(crate) fn remedy(self) -> Option<&'static str> {
        match self {
            CovScopeDecline::ExpressionScale => Some(
                "writing the readout as an explicit expression ([scaling] y = central / V) \
                 instead of obs_scale moves the fit onto the analytic route",
            ),
            CovScopeDecline::GradientFd => {
                Some("dropping gradient = fd moves the fit onto the analytic route")
            }
            CovScopeDecline::Disabled => {
                Some("analytic_cov_hessian = true moves the fit onto the analytic route")
            }
            _ => None,
        }
    }
}

/// `ode_reltol` / `ode_abstol` as the covariance step's FD stencil actually saw them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct OdeToleranceFacts {
    pub reltol: f64,
    pub abstol: f64,
}

/// The measured plateau for the FD covariance stencil (#520, 2026-09-16): on the 3-cpt IV ODE
/// fixture the SEs move 2–7× between the default and `1e-6` / `1e-8`, and nothing beyond three
/// figures between `1e-6` and `1e-10`.
pub(crate) const COV_FD_PLATEAU_RELTOL: f64 = 1e-6;
/// Companion to [`COV_FD_PLATEAU_RELTOL`].
pub(crate) const COV_FD_PLATEAU_ABSTOL: f64 = 1e-8;

impl OdeToleranceFacts {
    /// Read the tolerances an `[odes]` model is integrated at, or `None` for a closed-form
    /// model. Reads `effective_solver_opts`, so a fit-scoped `FitOptions::ode_reltol` override
    /// is what gets reported (#1212).
    pub(crate) fn from_model(model: &CompiledModel) -> Option<Self> {
        model.ode_spec.as_ref().map(|spec| {
            let OdeSolverOptions { reltol, abstol, .. } = spec.effective_solver_opts();
            Self { reltol, abstol }
        })
    }

    /// Whether the tolerance is looser than the plateau the FD covariance stencil needs.
    ///
    /// The gate is "looser than the plateau", not "equal to the default", so that the sentence
    /// recommending a tighter tolerance is only emitted where it is true. A fit already at
    /// `1e-6` / `1e-8` or tighter is told nothing, because there is nothing left to tighten.
    pub(crate) fn looser_than_cov_plateau(self) -> bool {
        self.reltol > COV_FD_PLATEAU_RELTOL || self.abstol > COV_FD_PLATEAU_ABSTOL
    }
}

/// How badly the eigenvalue floor changed the answer (#520 C1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CovSeverity {
    Minor,
    Moderate,
    Severe,
}

impl CovSeverity {
    pub(crate) fn label(self) -> &'static str {
        match self {
            CovSeverity::Minor => "minor",
            CovSeverity::Moderate => "moderate",
            CovSeverity::Severe => "severe",
        }
    }

    /// What the severity means for the numbers the user is about to read.
    pub(crate) fn interpretation(self) -> &'static str {
        match self {
            CovSeverity::Minor => "Standard errors are likely reliable.",
            CovSeverity::Moderate => {
                "Standard errors should be interpreted with caution; consider SIR-based \
                 confidence intervals."
            }
            CovSeverity::Severe => {
                "Standard errors for the affected parameters come mostly from the floor rather \
                 than from the data and are not reliable; SIR-based confidence intervals are \
                 recommended."
            }
        }
    }
}

/// A negative eigenvalue this large relative to `λ_max` is curvature the data does not have,
/// not finite-difference rounding: the floor sits at `λ_max · 1e-10`, so this is four orders of
/// magnitude past it.
const SEVERE_NEG_RATIO: f64 = 1e-6;
/// Between this and [`SEVERE_NEG_RATIO`] the indefiniteness is above the floor's own scale but
/// small against the spectrum.
const MODERATE_NEG_RATIO: f64 = 1e-9;
/// A variance inflated 100× is a standard error inflated 10×.
const SEVERE_INFLATION: f64 = 100.0;
/// A variance inflated 4× is a standard error inflated 2×.
const MODERATE_INFLATION: f64 = 4.0;

/// Grade the regularization on **magnitude**, never on the clipped count (#520 C1).
///
/// Two independent magnitudes, either of which can carry the grade:
///
/// * `neg_ratio = |min λ| / λ_max` when `min λ < 0`, else 0 — how indefinite the Hessian was,
///   scale-free. The floor is at `λ_max · 1e-10`, so anything far above that is real.
/// * `variance_inflation` — the worst, over free coordinates, of the returned variance divided
///   by the variance the **unclipped** part of the spectrum alone supports. This is the direct
///   statement about the reported SEs: an inflation of 1 means the floor changed nothing a user
///   reads, and `∞` means some parameter's entire variance was manufactured by the floor.
///
/// A count fraction appears nowhere. `n_clipped = 1 of 13` was the input that produced
/// "likely reliable" next to a 4400×-inflated SE.
pub(crate) fn grade_severity(neg_ratio: f64, variance_inflation: f64) -> CovSeverity {
    // NaN is never severe by comparison (every `>` against NaN is false), and a NaN here would
    // mean the spectrum itself is degenerate — which the caller has already rejected. Compare
    // explicitly rather than folding, so a NaN grades `Minor` by falling through rather than by
    // silently winning a `max`.
    if neg_ratio > SEVERE_NEG_RATIO || variance_inflation > SEVERE_INFLATION {
        CovSeverity::Severe
    } else if neg_ratio > MODERATE_NEG_RATIO || variance_inflation > MODERATE_INFLATION {
        CovSeverity::Moderate
    } else {
        CovSeverity::Minor
    }
}

/// Everything the regularization message is built from. Collected at the one place that knows
/// all of it, formatted by [`format_regularized_warning`] and by nothing else.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct CovRegularizationFacts {
    /// Which Hessian was floored.
    pub source: CovHessianSource,
    pub n_clipped: usize,
    pub n_free: usize,
    pub min_eigenvalue: f64,
    pub max_eigenvalue: f64,
    pub floor: f64,
    /// Worst per-coordinate variance inflation caused by the floor; `1.0` when nothing was
    /// clipped, `f64::INFINITY` when a coordinate's variance is entirely floor-derived.
    pub variance_inflation: f64,
    /// The clause that declined the analytic R-matrix. `None` on the analytic route itself, and
    /// on an FD route whose reason the caller did not resolve.
    pub decline: Option<CovScopeDecline>,
    /// `Some` only when the model integrates ODEs. The tolerance sentence is additionally
    /// gated on the FD route and on the tolerance being looser than the plateau.
    pub ode: Option<OdeToleranceFacts>,
}

impl CovRegularizationFacts {
    /// `|min λ| / λ_max`, or 0 when the spectrum is non-negative (near-singular, not
    /// indefinite).
    pub(crate) fn neg_ratio(&self) -> f64 {
        if self.min_eigenvalue < 0.0 && self.max_eigenvalue > 0.0 {
            self.min_eigenvalue.abs() / self.max_eigenvalue
        } else {
            0.0
        }
    }

    pub(crate) fn severity(&self) -> CovSeverity {
        grade_severity(self.neg_ratio(), self.variance_inflation)
    }
}

/// Render an inflation factor. `∞` is a reachable, meaningful value (all of a coordinate's
/// variance came from the floored directions) and must not print as `inf`.
fn fmt_inflation(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.3e}×")
    } else {
        "unbounded".to_string()
    }
}

/// Assemble the `covariance_regularized` message.
///
/// The leading `Covariance step regularized:` token is load-bearing — `classify_warning`
/// (`types.rs`) keys `WarningCode::CovarianceRegularized` on it — and is pinned by a test.
pub(crate) fn format_regularized_warning(facts: &CovRegularizationFacts) -> String {
    let severity = facts.severity();
    let mut msg = format!(
        "Covariance step regularized: eigenvalue floor applied to {} ({} of {} free-block \
         eigenvalues clipped; min eig = {:.3e}, max eig = {:.3e}, |min eig|/max eig = {:.2e}, \
         floor = {:.3e}; worst variance inflation from the floor = {}; severity: {}). {}",
        facts.source.label(),
        facts.n_clipped,
        facts.n_free,
        facts.min_eigenvalue,
        facts.max_eigenvalue,
        facts.neg_ratio(),
        facts.floor,
        fmt_inflation(facts.variance_inflation),
        severity.label(),
        severity.interpretation(),
    );

    // Everything below is specific to the finite-difference stencil: it is the `1/h²` route,
    // the only one with a step size and the only one that can be moved by a tolerance. On the
    // analytic route the message ends above.
    if facts.source != CovHessianSource::FdStencil {
        return msg;
    }

    if let Some(decline) = facts.decline {
        msg.push_str(&format!(
            " The exact analytic covariance R-matrix was declined because {}",
            decline.clause()
        ));
        match decline.remedy() {
            Some(remedy) => msg.push_str(&format!("; {remedy}.")),
            None => msg.push('.'),
        }
    }

    if let Some(ode) = facts.ode {
        if ode.looser_than_cov_plateau() {
            msg.push_str(&format!(
                " This model integrates ODEs at ode_reltol = {:.0e} / ode_abstol = {:.0e}, and \
                 the FD covariance stencil amplifies integration noise by 1/h²; ode_reltol = \
                 {:.0e} / ode_abstol = {:.0e} is on the measured accuracy plateau for this \
                 stencil (#520).",
                ode.reltol, ode.abstol, COV_FD_PLATEAU_RELTOL, COV_FD_PLATEAU_ABSTOL,
            ));
        }
    }

    msg
}

/// The sentence appended to the "N of M subjects use finite-difference inner gradients" warning
/// on an `[odes]` model (#520 C2, addendum of 2026-09-16).
///
/// Measured on the TMDD QSS fixture from its true parameters: the FD inner (EBE) gradient reads
/// integration noise and stalls 49 OFV above the analytic route at the default tolerance,
/// recovering to 24 OFV at `1e-6` / `1e-8` and fully at `1e-9`. So the advice has two halves and
/// each is gated on being true:
///
/// * tighten — only when the current tolerance is looser than the plateau, and the `1e-9`
///   escalation is named because `1e-6` recovers only part of the stall on that model;
/// * move into the analytic scope — always true on an ODE model, and the better fix, since the
///   analytic route reaches the floor at the default tolerance and is 4–8× faster.
///
/// `None` for a closed-form model: the FD inner gradient there differences a closed form, which
/// carries no integration noise for a tolerance to remove.
pub(crate) fn fd_inner_gradient_tolerance_note(model: &CompiledModel) -> Option<String> {
    let ode = OdeToleranceFacts::from_model(model)?;
    if ode.looser_than_cov_plateau() {
        Some(format!(
            " This fit integrates ODEs at ode_reltol = {:.0e} / ode_abstol = {:.0e}, and a \
             finite-difference inner gradient reads that integration noise directly; \
             ode_reltol = {:.0e} / ode_abstol = {:.0e} (next step 1e-9 if the fit still stalls), \
             or moving the model into the analytic sensitivity scope, will help (#520).",
            ode.reltol, ode.abstol, COV_FD_PLATEAU_RELTOL, COV_FD_PLATEAU_ABSTOL,
        ))
    } else {
        Some(
            " This fit integrates ODEs, and a finite-difference inner gradient reads the \
             integrator's noise directly; moving the model into the analytic sensitivity scope \
             will help (#520)."
                .to_string(),
        )
    }
}

#[cfg(test)]
#[path = "cov_diagnostics_tests.rs"]
mod tests;
