//! The user-facing text of the covariance step's regularization diagnostic (#520).
//!
//! Everything here is a **pure function over a small struct of facts**. The facts are
//! collected by [`crate::estimation::covariance::compute_covariance`] (which route served the
//! Hessian, what the eigen-floor did to the *returned* covariance, which gate clauses declined
//! the analytic R-matrix, what tolerance the subjects' effective route integrates at); the
//! prose is assembled here and nowhere else, so the message can be unit-tested at every
//! reachable combination of those facts without running a fit.
//!
//! Why the split exists at all: before #520 the severity of `covariance_regularized` was graded
//! on the **count fraction** of clipped eigenvalues, so a single badly-negative eigenvalue —
//! 1 of 13, 7% — printed *"severity: minor. Standard errors are likely reliable."* directly
//! above an SE inflated 4400× and a `%RSE` of 24982. Count says how many directions the floor
//! touched; it says nothing about how far they were floored or how much of a parameter's
//! variance came out of the floor rather than out of the data. Both of those are magnitudes,
//! and both are graded here ([`grade`]).
//!
//! The message's input space is enumerated in the sibling test file: route × severity × which
//! magnitude carried the grade × ODE-at-loose-tolerance (including a closed-form model reaching
//! its ODE twin) × declined clauses (one, or several). Every sentence must be true on every
//! reachable cell, which is why each conditional sentence is gated on the fact that makes it
//! true rather than emitted unconditionally.

use crate::ode::OdeSolverOptions;
use crate::types::{CompiledModel, Population};

/// Which Hessian the covariance step differentiated (#520).
///
/// The distinction is user-visible: the finite-difference stencil divides by `h²` and so
/// amplifies whatever noise the objective carries (integration error on an `[odes]` model, most
/// of all), while the exact analytic R-matrix from third-order sensitivities never
/// second-differences the *objective*. Advice that names `fd_hessian_step` or `ode_reltol` is
/// true on the first and false on the second, and before #520 the message said "eigenvalue
/// floor applied to FD Hessian" on both.
///
/// The analytic route is not step-free everywhere — `third_order_fd_step` finite-differences
/// the `Dual2` jet to assemble the third-order blocks, which is a step across a *smooth*
/// sensitivity rather than across the objective, and #1505 records the one measured case where
/// that distinction bites (a lagged-dose arrival kink). What the label asserts is narrower and
/// stays true: this route never second-differences the objective, so neither `fd_hessian_step`
/// nor a looser `ode_reltol` reaches it through the `1/h²` mechanism.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CovHessianSource {
    /// Exact analytic R-matrix assembled from third-order sensitivities (#436, extended to
    /// `[odes]` models by #1291). No objective stencil, no `fd_hessian_step`.
    AnalyticRMatrix,
    /// Central second differences of the reconverged objective. The `1/h²` route.
    FdStencil,
    /// The per-subject salvage (#1514): `Σᵢ Rᵢ` with the in-scope subjects' terms assembled
    /// analytically and the declining subjects' terms second-differenced from *their own*
    /// marginal. Both mechanisms are present, so the label names both and the FD guidance
    /// below still applies — to the salvaged subjects' share of the matrix.
    HybridRMatrix,
}

impl CovHessianSource {
    /// How the message names this Hessian.
    pub(crate) fn label(self) -> &'static str {
        match self {
            CovHessianSource::AnalyticRMatrix => "the analytic R-matrix",
            CovHessianSource::FdStencil => "the FD Hessian",
            CovHessianSource::HybridRMatrix => "the hybrid analytic/FD R-matrix",
        }
    }

    /// Whether *any* part of this Hessian came through the `1/h²` objective stencil, and so
    /// whether the message's finite-difference tail (which clause declined the analytic route,
    /// what tolerance the ODEs integrate at) is true of it.
    ///
    /// Not `== FdStencil`. On the hybrid route the declined subjects' terms are genuinely
    /// second differences of the objective, so suppressing the tail there would withhold the
    /// one sentence that says *why* they declined — and the clause it names is exactly the
    /// thing a user can act on. Gating on the mechanism rather than on the route keeps the
    /// pure-analytic cell silent, which is what #520's label fix was about.
    pub(crate) fn emits_fd_guidance(self) -> bool {
        match self {
            CovHessianSource::AnalyticRMatrix => false,
            CovHessianSource::FdStencil | CovHessianSource::HybridRMatrix => true,
        }
    }
}

/// Why the exact analytic covariance R-matrix was not used, named at the clause that declined
/// it (#520 C2).
///
/// The variants are in one-to-one correspondence with the clauses of the scope gate in
/// [`crate::sens::provider`] plus the three gates
/// [`crate::estimation::covariance::compute_covariance`] applies before it. There is exactly
/// one implementation: the gate itself returns these values and the routing decision tests
/// whether the first one exists, so the clauses a user is told about cannot drift from the
/// clauses that actually fired.
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
    /// Every variant, in the order the gate evaluates them. Used by the tests that assert the
    /// message is non-empty and classified correctly on every clause; kept here, next to the
    /// variants, so a new clause is one edit rather than two. `#[cfg(test)]` because nothing
    /// in production enumerates the enum — without it the lib target reports it dead and
    /// `preflight.sh`'s `-Dunused` turns that into a hard error.
    #[cfg(test)]
    pub(crate) const ALL: [CovScopeDecline; 19] = [
        CovScopeDecline::Disabled,
        CovScopeDecline::Mixture,
        CovScopeDecline::ExactHessianAnchor,
        CovScopeDecline::ModelOutOfScope,
        CovScopeDecline::GradientFd,
        CovScopeDecline::NonGaussianEndpoint,
        CovScopeDecline::Frem,
        CovScopeDecline::SelectedErrorSpec,
        CovScopeDecline::LogTransform,
        CovScopeDecline::IovShape,
        CovScopeDecline::ExpressionScale,
        CovScopeDecline::AnalyticReadout,
        CovScopeDecline::AnalyticalInit,
        CovScopeDecline::ResidualErrorEta,
        CovScopeDecline::ResidualCorrelations,
        CovScopeDecline::CustomRuvMagnitude,
        CovScopeDecline::EventWalkSubject,
        CovScopeDecline::IndivParamProgram,
        CovScopeDecline::PerSubjectBail,
    ];

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

    /// The one-line **action** that clears this clause, where there is one. `None` means the
    /// clause is a property of the model the user asked for, and the honest answer is that the
    /// FD stencil is the correct route for it — so the message says nothing rather than
    /// inventing advice.
    ///
    /// Deliberately a bare action ("dropping gradient = fd") and **not** a promise ("… moves
    /// the fit onto the analytic route"). The promise depends on how many *other* clauses also
    /// declined, which this value cannot see: a non-Gaussian model with `gradient = fd` stays
    /// on the FD stencil after the flag comes off. [`format_regularized_warning`] owns the
    /// promise and gates it on the action clearing every listed clause (#1508 review §3).
    pub(crate) fn remedy(self) -> Option<&'static str> {
        match self {
            CovScopeDecline::ExpressionScale => Some(
                "writing the readout as an explicit expression ([scaling] y = central / V) \
                 instead of obs_scale",
            ),
            CovScopeDecline::GradientFd => Some("dropping gradient = fd"),
            CovScopeDecline::Disabled => Some("setting analytic_cov_hessian = true"),
            _ => None,
        }
    }
}

/// `ode_reltol` / `ode_abstol` as the covariance step's FD stencil actually saw them, on the
/// route the subjects actually took.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct OdeToleranceFacts {
    pub reltol: f64,
    pub abstol: f64,
    /// True when the model has no `[odes]` block of its own and reaches an integrator only
    /// through its closed-form absorption **ODE twin** (`absorption_ode_equivalent`): a
    /// transit / inverse-Gaussian model whose subjects carry IOV, time-varying covariates, a
    /// `TIME` switch, a steady-state record or an infusion (#719/#814). Those subjects
    /// integrate, so their FD covariance and FD inner gradient read integration noise exactly
    /// as an `[odes]` model's do — which is what #1508 review §4 found suppressed.
    pub via_twin: bool,
}

/// The measured plateau for the FD covariance stencil (#520, 2026-09-16): on the 3-cpt IV ODE
/// fixture the SEs move 2–7× between the default and `1e-6` / `1e-8`, and nothing beyond three
/// figures between `1e-6` and `1e-10`.
pub(crate) const COV_FD_PLATEAU_RELTOL: f64 = 1e-6;
/// Companion to [`COV_FD_PLATEAU_RELTOL`].
pub(crate) const COV_FD_PLATEAU_ABSTOL: f64 = 1e-8;

impl OdeToleranceFacts {
    /// Read the tolerances the population is **actually** integrated at, or `None` when no
    /// subject reaches an integrator at all.
    ///
    /// Not `model.ode_spec.is_some()`. A closed-form transit / inverse-Gaussian model carries
    /// an ODE twin, and [`CompiledModel::effective_for`] reroutes a subject onto it for IOV,
    /// time-varying covariates, a `TIME`-dependent structural parameter, a steady-state record
    /// or an infusion — with IOV, *every* subject. Keying on `ode_spec` alone suppressed the
    /// tolerance guidance on exactly those fits, whose FD covariance stencil reads the twin's
    /// integration noise like any other (#1508 review §4). `sync_ode_solver_opts` stamps the
    /// fit's tolerances onto the twin too, so the numbers reported are the ones in force.
    ///
    /// **Known gap.** The *flip-flop* reroute ([`crate::pk::effective_model_for_eval`]) is
    /// decided per `(θ, η)` at evaluation time, not per subject, so a model that only ever
    /// reaches its twin that way is still reported as closed-form here. That direction is
    /// silent-but-conservative: it withholds advice, it never states something false.
    pub(crate) fn from_route(model: &CompiledModel, population: &Population) -> Option<Self> {
        let opts = |spec: &crate::ode::predictions::OdeSpec, via_twin: bool| {
            let OdeSolverOptions { reltol, abstol, .. } = spec.effective_solver_opts();
            Self {
                reltol,
                abstol,
                via_twin,
            }
        };
        if let Some(spec) = model.ode_spec.as_ref() {
            return Some(opts(spec, false));
        }
        // No `[odes]` block: the only way to an integrator is the absorption twin, and only
        // for the subjects `effective_for` actually reroutes. Reading it per subject rather
        // than from `absorption_ode_equivalent.is_some()` keeps the sentence true on a
        // transit model whose population is plain — that one never integrates.
        population
            .subjects
            .iter()
            .find_map(|s| model.effective_for(s).ode_spec.as_ref())
            .map(|spec| opts(spec, true))
    }

    /// How the message names what integrates. The twin is named explicitly: a user looking at
    /// a `one_cpt_transit` model with no `[odes]` block has no reason to expect a sentence
    /// about integration tolerances, and "this model's absorption ODE twin" is the thing they
    /// can look up (`ode_reltol` applies to it — see the fit-options page).
    pub(crate) fn subject_label(self) -> &'static str {
        if self.via_twin {
            "this model's closed-form absorption ODE twin integrates at"
        } else {
            "this model integrates ODEs at"
        }
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
}

/// **Which** magnitude carried the grade (#1508 review §6).
///
/// [`grade`] is an `OR` over two independent magnitudes, and they mean different things to a
/// reader: an inflated reported variance is a statement about the standard errors on the page,
/// while an indefinite Hessian with *no* inflation says the floor rewrote curvature the
/// reported SEs happened not to load on. Before this, the severe cell printed "these SEs come
/// mostly from the floor" in both — false in the second, where the supplied inflation says the
/// floor contributed nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CovSeverityCause {
    /// Neither magnitude crossed a threshold — only reachable with [`CovSeverity::Minor`].
    Neither,
    /// The reported-variance inflation alone crossed the tier.
    Inflation,
    /// `|min λ| / λ_max` alone crossed the tier; the inflation did not.
    Indefiniteness,
    /// Both crossed it.
    Both,
}

impl CovSeverityCause {
    /// True when the reported variances are themselves inflated, i.e. when a sentence about
    /// where the *standard errors* came from is true.
    fn touches_reported_variance(self) -> bool {
        matches!(self, CovSeverityCause::Inflation | CovSeverityCause::Both)
    }
}

/// The grade and the magnitude that carried it. One value from one function, so the
/// interpretation cannot describe a different trigger than the one that fired.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct CovGrade {
    pub severity: CovSeverity,
    pub cause: CovSeverityCause,
}

impl CovGrade {
    /// What the grade means for the numbers the user is about to read.
    ///
    /// Six reachable cells (three tiers × "the reported variance moved" / "it did not"), and
    /// every sentence is true on its own cell only:
    ///
    /// * the two `Minor` cells collapse — `Neither` is the only cause a `Minor` can carry;
    /// * the inflation-bearing cells talk about the standard errors, because the metric they
    ///   were graded on **is** the ratio of reported variances (see
    ///   [`CovRegularizationFacts::variance_inflation`]);
    /// * the indefiniteness-only cells say what is true there and nothing more: the Hessian
    ///   was altered, and nothing measured says the printed SEs absorbed it.
    ///
    /// "mostly from the floor" in the severe/inflation cell is a claim about a number:
    /// `variance_inflation > SEVERE_INFLATION` and the sentence needs `> 2`, which
    /// `severe_inflation_threshold_supports_the_word_mostly` pins.
    pub(crate) fn interpretation(self) -> &'static str {
        match (self.severity, self.cause.touches_reported_variance()) {
            (CovSeverity::Minor, _) => "Standard errors are likely reliable.",
            (CovSeverity::Moderate, true) => {
                "Part of the reported standard errors for the affected parameters comes from \
                 the floor rather than from the data (the inflation factor above is the \
                 ratio); interpret them with caution and consider SIR-based confidence \
                 intervals."
            }
            (CovSeverity::Moderate, false) => {
                "The Hessian was indefinite beyond the floor's own scale and was altered to \
                 invert it; the reported standard errors do not load on the altered \
                 directions, but interpret them with caution and consider SIR-based \
                 confidence intervals."
            }
            (CovSeverity::Severe, true) => {
                "Standard errors for the affected parameters come mostly from the floor rather \
                 than from the data and are not reliable; SIR-based confidence intervals are \
                 recommended."
            }
            (CovSeverity::Severe, false) => {
                "The Hessian was materially altered by the floor and these standard errors are \
                 not reliable; SIR-based confidence intervals are recommended."
            }
        }
    }
}

// ── The thresholds, and the measurement they come from ──────────────────────────────────
//
// Measured on 2026-09-21 (#1508 review §5); the previous values were round numbers picked to
// be interpretable in SE terms, which let a variance inflated nearly 4× — an SE inflated nearly
// 2× — stay "minor … likely reliable … needs no action".
//
// **The experiment.** The free-block Hessian of a converged fit is dumped and its *smallest*
// eigenvalue is the one input varied (everything else — the eigenvectors and the rest of the
// spectrum — stays as measured), from well above the eigenvalue floor down through it. At each
// point the pipeline this module grades runs end to end, and the realised worst relative error
// of a reported standard error is measured against an **independent** truth: the exact inverse
// of the same Hessian while it is still positive definite, so the comparison isolates the
// floor's distortion rather than a difference of matrix. On the stiff fixture's FD-route
// Hessian (6×6, `[scaling] obs_scale = V`, 40 subjects):
//
// | `λ_min / floor` | clipped | `variance_inflation` | realised worst SE error | `√inflation − 1` |
// |---|---|---|---|---|
// | 1e4 | 1 | 1.5775e5 | 39786 % | 39618 % |
// | 1e3 | 1 | 1.5777e4 | 12466 % | 12460 % |
// | 1e2 | 1 | 1.5786e3 | 3873 % | 3873 % |
// | 10  | 1 | 1.5876e2 | 1160 % | 1160 % |
// | 3   | 1 | 4.8327e1 | 595.2 % | 595.2 % |
// | 1   | 1 | 1.6776e1 | 309.6 % | 309.6 % |
//
// **What the measurement establishes** is the last column: the printed inflation is the
// *square* of the reported-SE error factor, `realised = √inflation − 1`, confirmed against the
// independent truth to a worst relative residual of **4.2e-3** over four orders of magnitude
// (≤ 4.3e-4 below an inflation of 1.6e4). So a cut on the inflation is a cut on a stated
// standard-error error, and the two are set that way:
//
// * `MODERATE_INFLATION` — the **1 % realised SE error** line, `1.01² = 1.0201`. Headroom: the
//   ambient reproducibility of a reported SE on a clean fit is 1.9e-4 relative — `SE(TVMTT)`
//   moves 0.05271 → 0.05270 across eight orders of magnitude of `ode_reltol` on the
//   NONMEM-anchored transit fixture — so this cut sits ~50× above numerical noise and cannot
//   fire on it.
// * `SEVERE_INFLATION` — the **100 % realised SE error** line, `2² = 4.0`: the reported number
//   is wrong by more than itself. At 4.0 at least ¾ of the reported variance is floor-derived,
//   which is what makes the severe interpretation's word "mostly" a true statement about a
//   number (`severe_inflation_threshold_supports_the_word_mostly`).
//
// The three endpoint fits land far past both, measured on 2026-09-21: the stiff fixture's
// analytic route reports an inflation of 8.816e4 (implied realised SE error 29590 %), its FD
// route 1.366e10 (1.17e7 %), and the 3-cpt IV ODE fixture on the FD covariance route at the
// default tolerance (180 subjects, `analytic_cov_hessian = false`) 3.920e8 (1.98e6 %) — all
// three severe, all three next to `%RSE` figures in the thousands.
//
// **What this experiment does not calibrate**, said plainly: the `*_NEG_RATIO` cuts. That leg
// fires precisely when the reported variances did *not* move, so a realised-SE-error line
// cannot be drawn on it — which is why its interpretation makes no claim about the standard
// errors (#1508 review §6). Those two keep their mechanism: the floor sits at `λ_max · 1e-10`,
// `MODERATE_NEG_RATIO` is one decade above it (below that, the sweep's realised error stays in
// the 1e-8 noise) and `SEVERE_NEG_RATIO` four decades above.
//
// **And what the diagnostic still cannot see.** The same 3-cpt fixture, pinned at
// `ode_reltol = 1e-10`, converges to a Hessian with nothing clipped and so emits no warning at
// all — while #520 measured its SEs 2–7× too large at the *default* tolerance in runs where the
// floor never fired. That failure mode is the integration tolerance rather than the floor, this
// diagnostic is silent on it by construction, and it is what #520's PR 2 addresses.
//
// `severity_tiers_match_the_calibration_table` replays the rows above through [`grade`], so a
// constant that drifts away from the measurement reddens.

/// A negative eigenvalue this large relative to `λ_max` is curvature the data does not have,
/// not finite-difference rounding: the floor sits at `λ_max · 1e-10`, so this is four orders of
/// magnitude past it.
const SEVERE_NEG_RATIO: f64 = 1e-6;
/// Between this and [`SEVERE_NEG_RATIO`] the indefiniteness is above the floor's own scale but
/// small against the spectrum — one decade past the floor, where the sweep's realised error
/// first leaves its 1e-8 noise.
const MODERATE_NEG_RATIO: f64 = 1e-9;
/// The measured 100 %-realised-SE-error line: `realised = √inflation − 1`, so an inflation of
/// `2² = 4` is a reported standard error wrong by more than itself. Must stay `> 2.0` for the
/// severe interpretation's word "mostly" to be true; pinned by
/// `severe_inflation_threshold_supports_the_word_mostly`.
const SEVERE_INFLATION: f64 = 4.0;
/// The measured 1 %-realised-SE-error line, `1.01² = 1.0201`, ~50× above the 1.9e-4 ambient
/// reproducibility of a reported SE.
const MODERATE_INFLATION: f64 = 1.02;

/// Grade the regularization on **magnitude**, never on the clipped count (#520 C1), and say
/// which magnitude carried the grade (#1508 review §6).
///
/// Two independent magnitudes, either of which can carry the grade:
///
/// * `neg_ratio = |min λ| / λ_max` when `min λ < 0`, else 0 — how indefinite the Hessian was,
///   scale-free. The floor is at `λ_max · 1e-10`, so anything far above that is real.
/// * `variance_inflation` — the worst, over *reported* parameters, of the returned variance
///   divided by the variance the same estimator gives with the floored directions dropped
///   instead of floored. This is the direct statement about the SEs on the page: an inflation
///   of 1 means the floor changed nothing a user reads, and `∞` means some parameter's entire
///   variance was manufactured by the floor.
///
/// A count fraction appears nowhere. `n_clipped = 1 of 13` was the input that produced
/// "likely reliable" next to a 4400×-inflated SE.
pub(crate) fn grade(neg_ratio: f64, variance_inflation: f64) -> CovGrade {
    // NaN is never severe by comparison (every `>` against NaN is false), and a NaN here would
    // mean the spectrum itself is degenerate — which the caller has already rejected. Compare
    // explicitly rather than folding, so a NaN grades `Minor` by falling through rather than by
    // silently winning a `max`.
    let (severity, by_neg, by_inflation) =
        if neg_ratio > SEVERE_NEG_RATIO || variance_inflation > SEVERE_INFLATION {
            (
                CovSeverity::Severe,
                neg_ratio > SEVERE_NEG_RATIO,
                variance_inflation > SEVERE_INFLATION,
            )
        } else if neg_ratio > MODERATE_NEG_RATIO || variance_inflation > MODERATE_INFLATION {
            (
                CovSeverity::Moderate,
                neg_ratio > MODERATE_NEG_RATIO,
                variance_inflation > MODERATE_INFLATION,
            )
        } else {
            (CovSeverity::Minor, false, false)
        };
    let cause = match (by_neg, by_inflation) {
        (true, true) => CovSeverityCause::Both,
        (true, false) => CovSeverityCause::Indefiniteness,
        (false, true) => CovSeverityCause::Inflation,
        (false, false) => CovSeverityCause::Neither,
    };
    CovGrade { severity, cause }
}

/// Everything the regularization message is built from. Collected at the one place that knows
/// all of it, formatted by [`format_regularized_warning`] and by nothing else.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct CovRegularizationFacts<'a> {
    /// Which Hessian was floored.
    pub source: CovHessianSource,
    pub n_clipped: usize,
    pub n_free: usize,
    pub min_eigenvalue: f64,
    pub max_eigenvalue: f64,
    pub floor: f64,
    /// Worst inflation of a **reported** variance caused by the floor: the returned covariance
    /// run through the selected estimator (`R⁻¹`, or the sandwich `R⁻¹ S R⁻¹` under
    /// `covariance_method = rsr`) and through the reported-parameter delta transform, divided
    /// by the same two steps applied to the inverse with the floored directions dropped.
    ///
    /// Measuring it there rather than on packed `R⁻¹` is #1508 review §1: the same `R` used to
    /// print the same inflation for every `S`, though `S` can suppress or concentrate the
    /// floored direction, and the multivariate delta transform that produces a block-Ω SE
    /// mixes packed coordinates, so a packed-space diagonal is not the number the user reads.
    ///
    /// `1.0` when nothing was clipped, `f64::INFINITY` when a reported parameter's variance is
    /// entirely floor-derived.
    pub variance_inflation: f64,
    /// Every clause that declined the analytic R-matrix, in gate order. Empty on the analytic
    /// route itself, and on an FD route whose reasons the caller did not resolve.
    ///
    /// A slice, not a single value: the remedy sentence's promise ("moves the fit onto the
    /// analytic route") is only true when the named action clears *all* of them (#1508
    /// review §3).
    pub declines: &'a [CovScopeDecline],
    /// `Some` whenever some subject reaches an integrator — through `[odes]` or through a
    /// closed-form model's absorption ODE twin. The tolerance sentence is additionally gated
    /// on the FD route and on the tolerance being looser than the plateau.
    pub ode: Option<OdeToleranceFacts>,
}

impl CovRegularizationFacts<'_> {
    /// `|min λ| / λ_max`, or 0 when the spectrum is non-negative (near-singular, not
    /// indefinite).
    pub(crate) fn neg_ratio(&self) -> f64 {
        if self.min_eigenvalue < 0.0 && self.max_eigenvalue > 0.0 {
            self.min_eigenvalue.abs() / self.max_eigenvalue
        } else {
            0.0
        }
    }

    pub(crate) fn grade(&self) -> CovGrade {
        grade(self.neg_ratio(), self.variance_inflation)
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

/// Join clause / action fragments the way the sentence needs them: `a`, `a and b`,
/// `a, b and c`.
fn join_and(parts: &[&str]) -> String {
    match parts {
        [] => String::new(),
        [one] => one.to_string(),
        [head @ .., last] => format!("{} and {}", head.join(", "), last),
    }
}

/// The "why the analytic route was declined, and what to do about it" sentence(s).
///
/// Split out because its truth conditions are the fiddly part: the promise that an action
/// *moves the fit onto the analytic route* holds only when the action clears every clause that
/// declined. Before #1508's review the promise was unconditional, so a non-Gaussian model that
/// also carried `gradient = fd` was told that dropping the flag would move it — it would not.
/// Upper-case the first character. The action fragments are authored lower-case so they can
/// also sit mid-sentence; whichever one opens a sentence is capitalised here.
fn sentence_case(s: String) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => s,
    }
}

fn decline_sentences(declines: &[CovScopeDecline]) -> String {
    if declines.is_empty() {
        return String::new();
    }
    let clauses: Vec<&str> = declines.iter().map(|d| d.clause()).collect();
    let mut out = format!(
        " The exact analytic covariance R-matrix was declined because {}.",
        join_and(&clauses)
    );

    let actions: Vec<&str> = declines.iter().filter_map(|d| d.remedy()).collect();
    if actions.is_empty() {
        // No invented advice: every clause here is a property of the model the user asked
        // for, and the FD stencil is the correct route for it.
        return out;
    }
    let n_unremedied = declines.len() - actions.len();
    if n_unremedied == 0 {
        // Every clause has an action, so clearing all of them does move the fit — and the verb
        // has to say that no single one of them does it alone.
        let verb = if actions.len() == 1 {
            "moves"
        } else {
            "together move"
        };
        out.push_str(&format!(
            " {} {verb} the fit onto the analytic route.",
            sentence_case(join_and(&actions)),
        ));
    } else {
        // Two independent plurals here, and they agree with two different counts: the verb on
        // the *actions*, the residual noun phrase on the clauses that have none. Keying both
        // on one count is the agreement bug `two_remediable_clauses_next_to_an_unremediable_
        // one_name_the_plural_forms` was written to catch, and did.
        let cleared = if actions.len() == 1 {
            "clears that clause"
        } else {
            "clear those clauses"
        };
        let remaining = if n_unremedied == 1 {
            "the remaining clause has"
        } else {
            "the remaining clauses have"
        };
        out.push_str(&format!(
            " {} {cleared}, but {remaining} no one-line remedy, so the fit stays on the \
             finite-difference route until all of them are cleared.",
            sentence_case(join_and(&actions)),
        ));
    }
    out
}

/// Assemble the `covariance_regularized` message.
///
/// The leading `Covariance step regularized:` token is load-bearing — `classify_warning`
/// (`types.rs`) keys `WarningCode::CovarianceRegularized` on it — and is pinned by a test.
pub(crate) fn format_regularized_warning(facts: &CovRegularizationFacts) -> String {
    let grade = facts.grade();
    let mut msg = format!(
        "Covariance step regularized: eigenvalue floor applied to {} ({} of {} free-block \
         eigenvalues clipped; min eig = {:.3e}, max eig = {:.3e}, |min eig|/max eig = {:.2e}, \
         floor = {:.3e}; worst inflation of a reported variance = {}; severity: {}). {}",
        facts.source.label(),
        facts.n_clipped,
        facts.n_free,
        facts.min_eigenvalue,
        facts.max_eigenvalue,
        facts.neg_ratio(),
        facts.floor,
        fmt_inflation(facts.variance_inflation),
        grade.severity.label(),
        grade.interpretation(),
    );

    // Everything below is specific to the finite-difference stencil: it is the `1/h²` route,
    // the only one with an objective step size and the only one a tolerance moves through that
    // mechanism. On the pure-analytic route the message ends above; on the hybrid route it does
    // not, because a stencil did run — over the subjects the clauses below name.
    if !facts.source.emits_fd_guidance() {
        return msg;
    }

    msg.push_str(&decline_sentences(facts.declines));

    if let Some(ode) = facts.ode {
        if ode.looser_than_cov_plateau() {
            msg.push_str(&format!(
                " Note that {} ode_reltol = {:.0e} / ode_abstol = {:.0e}, and the FD covariance \
                 stencil amplifies integration noise by 1/h²; ode_reltol = {:.0e} / ode_abstol \
                 = {:.0e} is on the measured accuracy plateau for this stencil (#520).",
                ode.subject_label(),
                ode.reltol,
                ode.abstol,
                COV_FD_PLATEAU_RELTOL,
                COV_FD_PLATEAU_ABSTOL,
            ));
        }
    }

    msg
}

/// The note for cross-partial stencils that returned `NaN`/`Inf`, worded for the route that
/// produced the Hessian (#1514 review §1).
///
/// What survives in the matrix is route-dependent, and the difference is the whole content of
/// the sentence:
///
/// * **Whole-population stencil** — the cross-partial *is* the stencil, so a non-finite result
///   leaves the entry at its zero initialisation. Correlation between the named parameters is
///   wholly absent.
/// * **Hybrid** (#1514) — the entry already holds the in-scope subjects' cross-partial,
///   assembled analytically, and the stencil's contribution is added to it. Only the declined
///   subjects' share of that cross-partial is missing. Saying "set to 0" there overstates the
///   damage and points the reader at the wrong quantity.
///
/// Both wordings keep the `off-diagonal FD stencil` token that `classify_warning` keys
/// `WarningCode::CovarianceRegularized` on, and both keep the `fd_hessian_step` advice, which
/// is the actionable half and is true on either route — the stencil that failed is a stencil
/// either way.
///
/// [`CovHessianSource::AnalyticRMatrix`] shares the whole-population wording and is
/// unreachable: no stencil runs on that route, so there are no non-finite cross-partials to
/// name. It is written out rather than left to a catch-all so that a future fourth route has
/// to choose.
pub(crate) fn format_offdiag_nan_warning(names: &str, source: CovHessianSource) -> String {
    match source {
        CovHessianSource::HybridRMatrix => format!(
            "Covariance step: off-diagonal FD stencil(s) non-finite for {names}. \
             Those cross-partials keep only the analytically assembled subjects' \
             contribution — the finite-differenced subjects' share of them is missing — so \
             SE for these parameter(s) may be over-optimistic. Try tuning fd_hessian_step."
        ),
        CovHessianSource::FdStencil | CovHessianSource::AnalyticRMatrix => format!(
            "Covariance step: off-diagonal FD stencil(s) non-finite for {names}. \
             Cross-partial correlation set to 0; SE for these parameter(s) \
             may be over-optimistic. Try tuning fd_hessian_step."
        ),
    }
}

/// How many salvaged subject ids the message prints before it summarises the rest.
///
/// The list is a *pointer*, not a record: a user who wants all of them reads the model's scope
/// gate, and a 300-subject population that salvages 140 would otherwise put a paragraph of ids
/// into a warning that has one actionable sentence. Ten is enough to recognise a pattern (all
/// the sparse subjects, all of occasion 2) without the message becoming the id list.
const SALVAGE_ID_LIST_CAP: usize = 10;

/// The informational note naming the subjects whose covariance terms were finite-differenced
/// while the rest of the population used the exact analytic R-matrix (#1514).
///
/// `ids` are the salvaged subjects' ids in population order; duplicates are dropped (a dataset
/// may repeat an id across stacked occasions, and the same id printed twice reads as two
/// subjects). `n_total` is the whole population.
///
/// Returns `None` when nothing was salvaged — the pure-analytic and pure-FD routes both say
/// nothing here, so the note's presence *is* the statement that the hybrid route ran.
///
/// Every sentence is gated on being true of the cell it prints in: the counts and the
/// agreement come from `ids.len()` and `n_total`, and the "remaining" clause is only reachable
/// with at least one analytically-assembled subject, which the caller's short-circuit
/// guarantees (a full decline takes the population stencil instead).
pub(crate) fn format_salvage_note(ids: &[&str], n_total: usize) -> Option<String> {
    let mut seen: Vec<&str> = Vec::with_capacity(ids.len());
    for id in ids {
        if !seen.contains(id) {
            seen.push(id);
        }
    }
    let n = seen.len();
    if n == 0 || n_total == 0 {
        return None;
    }
    let listed: Vec<&str> = seen.iter().take(SALVAGE_ID_LIST_CAP).copied().collect();
    let id_list = if n > listed.len() {
        format!("{} and {} more", listed.join(", "), n - listed.len())
    } else {
        join_and(&listed)
    };
    // Singular / plural agreement on three different nouns (subject, term, marginal) plus the
    // verb, all keyed off the same count — written out rather than suffixed with "(s)", because
    // the one-subject cell is the common one and is what the measurement in #1514 was made on.
    let (id_label, verb, term, marginal) = if n == 1 {
        ("ID", "is", "its information term was", "its own marginal")
    } else {
        (
            "IDs",
            "are",
            "their information terms were",
            "their own marginals",
        )
    };
    let remaining = n_total.saturating_sub(n);
    let rest = if remaining == 1 {
        "the remaining subject was assembled analytically".to_string()
    } else {
        format!("the remaining {remaining} subjects were assembled analytically")
    };
    Some(format!(
        "W_COV_ANALYTIC_SALVAGE: {n} of {n_total} subjects ({id_label} {id_list}) {verb} outside \
         the exact analytic covariance R-matrix scope; {term} finite-differenced from \
         {marginal}, and {rest}. Each subject contributes its own term to the information \
         matrix, so only the named subjects' terms use a different estimator."
    ))
}

/// The sentence appended to the "N of M subjects use finite-difference inner gradients"
/// warning when the population reaches an integrator (#520 C2, addendum of 2026-09-16).
///
/// Measured on the TMDD QSS fixture from its true parameters: the FD inner (EBE) gradient reads
/// integration noise and stalls 49 OFV above the analytic route at the default tolerance,
/// recovering to 24 OFV at `1e-6` / `1e-8` and fully at `1e-9`. So the advice has two halves and
/// each is gated on being true:
///
/// * tighten — only when the current tolerance is looser than the plateau, and the `1e-9`
///   escalation is named because `1e-6` recovers only part of the stall on that model;
/// * move into the analytic scope — always true on a model that integrates, and the better
///   fix, since the analytic route reaches the floor at the default tolerance and is 4–8×
///   faster.
///
/// `None` only when *no* subject integrates. A closed-form transit / inverse-Gaussian model
/// whose subjects reroute to the absorption ODE twin does integrate, and is told so —
/// [`OdeToleranceFacts::from_route`] is what makes that cell reachable (#1508 review §4).
pub(crate) fn fd_inner_gradient_tolerance_note(
    model: &CompiledModel,
    population: &Population,
) -> Option<String> {
    let ode = OdeToleranceFacts::from_route(model, population)?;
    if ode.looser_than_cov_plateau() {
        Some(format!(
            " Note that {} ode_reltol = {:.0e} / ode_abstol = {:.0e}, and a finite-difference \
             inner gradient reads that integration noise directly; ode_reltol = {:.0e} / \
             ode_abstol = {:.0e} (next step 1e-9 if the fit still stalls), or moving the model \
             into the analytic sensitivity scope, will help (#520).",
            ode.subject_label(),
            ode.reltol,
            ode.abstol,
            COV_FD_PLATEAU_RELTOL,
            COV_FD_PLATEAU_ABSTOL,
        ))
    } else {
        Some(format!(
            " Note that {} a tolerance already at or inside the measured plateau, but a \
             finite-difference inner gradient still reads the integrator's noise directly; \
             moving the model into the analytic sensitivity scope will help (#520).",
            ode.subject_label(),
        ))
    }
}

#[cfg(test)]
#[path = "cov_diagnostics_tests.rs"]
mod tests;
