// TTE / survival non-Gaussian endpoint support: the TTE data term, its exact η-jets
// (`non_gaussian_eta_jets2`, #2786) and TTE simulation.
// See plans/tte-survival-markov.md §3.1, §2.3, §9.3, §8.8.2.

pub mod parametric;

use parametric::hazard_and_cum_hazard_g;
pub use parametric::{
    cum_hazard, hazard_and_cum_hazard, mean_survival, median_survival,
    sample_conditional_event_time, sample_event_time,
};

use nalgebra::DMatrix;
use rand::RngExt;
use std::collections::HashMap;

use crate::sens::num::PkNum;
use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
use crate::types::{CompiledModel, Subject};
use crate::types::{
    EndpointLikelihood, EventType, HazardFamily, HazardSpec, ObsRecord, RtteClock, SimOutcome,
    TteRecurrence,
};

// ─────────────────────────────────────────────────────────────────────────────
//  TTE data term
// ─────────────────────────────────────────────────────────────────────────────

/// Round-off tolerance for the cumulative-hazard monotonicity guard: an increment
/// `H(b) - H(a) < 0` counts as a violation only past `abstol + reltol * |H|`, mirroring
/// the ODE integrator's `mono_tol` (`src/ode/solver.rs`). ODE-accumulated callers pass
/// the configured solver tolerances ([`MonoTol::from_solver`]); the closed-form path
/// passes the tight [`MonoTol::analytic`] floor.
#[derive(Clone, Copy, Debug)]
pub struct MonoTol {
    pub reltol: f64,
    pub abstol: f64,
}

impl MonoTol {
    /// Tight floor for closed-form hazard families: `1e-9 + 1e-9*|H|`, ~6 orders above
    /// the f64 round-off (~1e-15 relative) yet scaling with `H`, so it catches any genuine
    /// negative increment without firing on a legitimate fit.
    pub fn analytic() -> Self {
        Self { reltol: 1e-9, abstol: 1e-9 }
    }

    /// Floor tied to the ODE solver's effective tolerances.
    pub fn from_solver(opts: &crate::ode::OdeSolverOptions) -> Self {
        Self { reltol: opts.reltol, abstol: opts.abstol }
    }
}

impl Default for MonoTol {
    /// The standard floor at the solver's default tolerances (reltol 1e-4, abstol 1e-6),
    /// delegating to [`crate::ode::OdeSolverOptions::default`] so the two stay in sync.
    fn default() -> Self {
        Self::from_solver(&crate::ode::OdeSolverOptions::default())
    }
}

/// Negative log-likelihood of a TTE endpoint for one subject (analytic hazards only):
///   RightCensored:    H(T) − H(entry)
///   Exact:            H(T) − H(entry) − log h(T)
///   IntervalCensored: −log [ exp(−(H(left)−H(entry))) − exp(−(H(right)−H(entry))) ]
/// Returns the 1e20 sentinel when the likelihood is numerically ill-defined.
#[cfg(test)]
pub fn tte_data_term(
    records: &[ObsRecord], hazard: &HazardSpec, recurrence: TteRecurrence, theta: &[f64],
    eta: &[f64], covariates: &HashMap<String, f64>,
) -> f64 {
    match hazard {
        HazardSpec::Analytic { family, param_fn } => {
            let params = param_fn(theta, eta, covariates);
            analytic_tte_nll_g(records, *family, recurrence, &params)
        }
        // ODE-accumulated hazards go through `tte_endpoint_nll` → `tte_ode_nll`; this
        // arm is kept explicit (not `unreachable!`) so the dispatch contract is testable.
        HazardSpec::OdeAccumulated { .. } => SUBJECT_SENTINEL_NLL,
    }
}

/// The analytic-family TTE data term over any [`PkNum`]; a dual instance seeded on the
/// parameters gives their exact derivatives (#2437).
pub fn analytic_tte_nll_g<T: PkNum>(
    records: &[ObsRecord], family: HazardFamily, recurrence: TteRecurrence, params: &[T],
) -> T {
    analytic_tte_nll_visit_g(records, family, recurrence, params, &mut |_| {})
}

/// [`analytic_tte_nll_g`] that also hands each scored record's [`RecordTerm`] to `visit`
/// (#901): the loop that scores the likelihood is the loop that reports its terms.
pub fn analytic_tte_nll_visit_g<T: PkNum>(
    records: &[ObsRecord], family: HazardFamily, recurrence: TteRecurrence, params: &[T],
    visit: &mut impl FnMut(RecordTerm<T>),
) -> T {
    let cumhaz_at = |t| hazard_and_cum_hazard_g(family, t, params).1;
    let hazard_at = |t| hazard_and_cum_hazard_g(family, t, params).0;
    // Analytic families carry only f64 round-off (no ODE solver error), so every
    // recurrence path uses the tight analytic monotone floor (#633).
    match recurrence {
        // Standard TTE / competing risks: per-record independent terms.
        TteRecurrence::Single => {
            tte_nll_from_curves_g(records, cumhaz_at, hazard_at, MonoTol::analytic(), visit)
        }
        // Clock-forward RTTE (Andersen–Gill): the cumulative hazard is integrated
        // once across the subject's whole risk window (§3.3), so the records couple.
        TteRecurrence::Repeated { clock: RtteClock::Forward } => rtte_forward_nll_from_curves_g(
            records,
            cumhaz_at,
            hazard_at,
            MonoTol::analytic(),
            visit,
        ),
        // Clock-reset RTTE: each inter-event gap is scored on its own duration `Δ_k` (§3.3).
        TteRecurrence::Repeated { clock: RtteClock::Reset } => {
            rtte_reset_nll_from_curves_g(records, cumhaz_at, hazard_at, MonoTol::analytic(), visit)
        }
    }
}

/// A subject's whole non-Gaussian term (TTE, binary and CTMM summed, θ fixed) with its
/// η-gradient and η-Hessian, `(q, ∇q, ∇²q)`, from its plan (#2786). `None` when the term
/// does not record, on the repel sentinel, or on a non-finite derivative.
pub fn non_gaussian_eta_jets2(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
) -> Option<(f64, Vec<f64>, DMatrix<f64>)> {
    let (q, g, h) = crate::sens::ode_provider::adjoint::terms::non_gaussian_jets2_planned(
        model, subject, theta, eta,
    )?;
    let n = eta.len();
    let h = DMatrix::from_fn(n, n, |k, l| h[k][l]);
    (q.is_finite() && q < REPEL_SENTINEL_FLOOR && h.iter().chain(&g).all(|v| v.is_finite()))
        .then_some((q, g, h))
}

/// `(q, ∇²q)` of one subject term from its plan (#2786), `None` on the declines of
/// [`non_gaussian_eta_jets2`]: the per-endpoint curvature the tests pin.
#[cfg(test)]
pub fn term_eta_hessian<S: crate::sens::ode_provider::adjoint::terms::TermEval>(
    model: &CompiledModel, subject: &Subject,
    kind: crate::sens::ode_provider::adjoint::terms::TermKind, term: &S, theta: &[f64],
    eta: &[f64],
) -> Option<(f64, DMatrix<f64>)> {
    let (q, g, h) = crate::sens::ode_provider::adjoint::terms::term_jets2_planned(
        model, subject, kind, term, theta, eta,
    )?;
    let n = eta.len();
    let h = DMatrix::from_fn(n, n, |k, l| h[k][l]);
    (q.is_finite() && q < REPEL_SENTINEL_FLOOR && h.iter().chain(&g).all(|v| v.is_finite()))
        .then_some((q, h))
}

/// The sorted-unique times an ODE-accumulated endpoint's likelihood reads `H` (and `h`) at:
/// each record's time (or interval ends), and its entry when left-truncated — as
/// `tte_ode_nll` collects them.
fn ode_hazard_times(records: &[ObsRecord]) -> Vec<f64> {
    let mut times: Vec<f64> = Vec::new();
    for r in records {
        let ObsRecord::Event { time, event_type, entry_time, .. } = r else {
            continue;
        };
        if *entry_time > 0.0 {
            times.push(*entry_time);
        }
        match event_type {
            EventType::IntervalCensored { left, right } => {
                times.push(*left);
                times.push(*right);
            }
            _ => times.push(*time),
        }
    }
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup();
    times
}

/// An ODE-accumulated (drug-driven) endpoint's negative log-likelihood on any VM number
/// type: the cumulative hazard and hazard from the dual walk
/// ([`crate::sens::ode_provider::adjoint::cumhaz_hazard_from_states`]) through the shared curve
/// scoring, at the model's own monotone tolerance. `None` outside that walk's scope.
#[cfg(test)]
pub fn ode_tte_nll_g<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel, subject: &Subject, chz_state: usize, records: &[ObsRecord], theta: &[T],
    eta: &[T],
) -> Option<T> {
    ode_tte_nll_visit_g(model, subject, chz_state, records, theta, eta, &mut |_| {})
}

/// [`ode_tte_nll_g`] handing each scored record's [`RecordTerm`] to `visit` (#901). It
/// returns `None` only before scoring (outside the walk's scope, or no solve), so a `None`
/// never leaves terms in the visitor.
pub fn ode_tte_nll_visit_g<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel, subject: &Subject, chz_state: usize, records: &[ObsRecord], theta: &[T],
    eta: &[T], visit: &mut impl FnMut(RecordTerm<T>),
) -> Option<T> {
    let times = ode_hazard_times(records);
    let solved =
        crate::sens::ode_provider::adjoint::ode_states_at_g(model, subject, theta, eta, &times)?;
    ode_tte_nll_from_states(model, subject, chz_state, records, &times, &solved, visit)
}

/// [`ode_tte_nll_g`] over a solve already saved at `times` (sorted, deduplicated, holding
/// every time the endpoint's records read): what one evaluation's endpoints share. Each
/// scored record's term goes to `visit`.
pub fn ode_tte_nll_from_states<T: crate::sens::num::vm::VmNum>(
    model: &CompiledModel, subject: &Subject, chz_state: usize, records: &[ObsRecord],
    times: &[f64], solved: &crate::sens::ode_provider::adjoint::DualStates<T>,
    visit: &mut impl FnMut(RecordTerm<T>),
) -> Option<T> {
    let (cum, haz) = crate::sens::ode_provider::adjoint::cumhaz_hazard_from_states(
        model, subject, chz_state, times, solved,
    )?;
    let tol = MonoTol::from_solver(&model.ode_spec.as_ref()?.effective_solver_opts());
    let at = |curve: &[T], t: f64| -> T {
        times.binary_search_by(|x| x.total_cmp(&t)).map_or(T::from_f64(f64::NAN), |i| curve[i])
    };
    Some(tte_nll_from_curves_g(records, |t| at(&cum, t), |t| at(&haz, t), tol, visit))
}

/// A subject's non-Gaussian data term as a function of `(θ, η)` on any VM number type —
/// what the plans record (#2786): differentiated in η alone (the inner gradient, the
/// Laplace η-Hessian) or in both (the outer gradient's θ block).
/// `None` when a record the subject carries has no exact channel.
pub trait SubjectTerm {
    fn eval<T: crate::sens::num::vm::VmNum>(&self, theta: &[T], eta: &[T]) -> Option<T>;
}

/// The subject's TTE data term summed over its endpoints (analytic hazards through their
/// programs, ODE-accumulated ones through the dual walk). `None` when a TTE endpoint the
/// subject has records on is not served exactly.
pub struct TteTerm<'a> {
    pub model: &'a CompiledModel,
    pub subject: &'a Subject,
    /// Each TTE endpoint the subject has records on, its records collected once.
    endpoints: Vec<TteEndpoint<'a>>,
    /// The times every ODE-accumulated endpoint reads (sorted, deduplicated): one dual solve
    /// saved there serves them all.
    ode_times: Vec<f64>,
}

struct TteEndpoint<'a> {
    hazard: &'a HazardSpec,
    /// An analytic hazard's dual-evaluable parameter program (#2437); `None` for an
    /// ODE-accumulated hazard.
    program: Option<&'a crate::parser::model_parser::EndpointExprProgram>,
    recurrence: TteRecurrence,
    /// Borrowed when every record of the subject is this endpoint's (one TTE endpoint, the
    /// common case): the term is built per evaluation, so a copy here was one per call.
    records: std::borrow::Cow<'a, [ObsRecord]>,
}

impl<'a> TteTerm<'a> {
    /// The subject's TTE endpoints and their records, gathered once for every evaluation.
    pub fn new(model: &'a CompiledModel, subject: &'a Subject) -> Self {
        use crate::types::EndpointLikelihood;
        let mut endpoints = Vec::new();
        let mut ode_times = Vec::new();
        for (cmt, endpoint) in &model.endpoints {
            let EndpointLikelihood::Tte { hazard, recurrence, hazard_program, .. } = endpoint
            else {
                continue;
            };
            let on_cmt = |r: &ObsRecord| matches!(r, ObsRecord::Event { cmt: c, .. } if c == cmt);
            let records: std::borrow::Cow<'a, [ObsRecord]> =
                if subject.obs_records.iter().all(on_cmt) {
                    std::borrow::Cow::Borrowed(&subject.obs_records[..])
                } else {
                    std::borrow::Cow::Owned(
                        subject.obs_records.iter().filter(|r| on_cmt(r)).cloned().collect(),
                    )
                };
            if records.is_empty() {
                continue;
            }
            if matches!(hazard, HazardSpec::OdeAccumulated { .. }) {
                ode_times.extend(ode_hazard_times(&records));
            }
            endpoints.push(TteEndpoint {
                hazard,
                program: hazard_program.as_ref(),
                recurrence: *recurrence,
                records,
            });
        }
        sort_dedup(&mut ode_times);
        TteTerm { model, subject, endpoints, ode_times }
    }

    /// The times the ODE-accumulated endpoints read; empty without one.
    pub fn ode_times(&self) -> &[f64] {
        &self.ode_times
    }

    /// The term over a caller's shared solve (`times`, sorted, a superset of
    /// [`Self::ode_times`]) or, without one, a solve of its own when an endpoint needs it.
    pub fn eval_shared<T: crate::sens::num::vm::VmNum>(
        &self, theta: &[T], eta: &[T],
        shared: Option<(&[f64], &crate::sens::ode_provider::adjoint::DualStates<T>)>,
    ) -> Option<T> {
        let own;
        let shared = match shared {
            Some(sh) => Some(sh),
            None if !self.ode_times.is_empty() => {
                own = crate::sens::ode_provider::adjoint::ode_states_at_g(
                    self.model,
                    self.subject,
                    theta,
                    eta,
                    &self.ode_times,
                )?;
                Some((self.ode_times.as_slice(), &own))
            }
            None => None,
        };
        let mut nll = T::from_f64(0.0);
        for ep in &self.endpoints {
            nll = nll
                + match ep.hazard {
                    HazardSpec::Analytic { family, .. } => {
                        let params = ep.program?.eval_g(theta, eta, &self.subject.covariates)?;
                        analytic_tte_nll_g(&ep.records, *family, ep.recurrence, &params)
                    }
                    HazardSpec::OdeAccumulated { chz_state } => {
                        let (times, solved) = shared?;
                        ode_tte_nll_from_states(
                            self.model,
                            self.subject,
                            *chz_state,
                            &ep.records,
                            times,
                            solved,
                            &mut |_| {},
                        )?
                    }
                };
        }
        Some(nll)
    }
}

/// Sort and deduplicate a time list (bit-equal times collapse).
pub fn sort_dedup(times: &mut Vec<f64>) {
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup_by(|a, b| a.to_bits() == b.to_bits());
}

impl SubjectTerm for TteTerm<'_> {
    fn eval<T: crate::sens::num::vm::VmNum>(&self, theta: &[T], eta: &[T]) -> Option<T> {
        self.eval_shared(theta, eta, None)
    }
}

/// Every non-Gaussian data term a subject carries — TTE, binary, CTMM — summed, at the
/// scale the likelihood scores them (`1×` here; the objective doubles it beside the
/// Gaussian `data_ll`). `None` when any record has no exact channel.
pub struct NonGaussianTerm<'a> {
    pub model: &'a CompiledModel,
    pub subject: &'a Subject,
    tte: Option<TteTerm<'a>>,
    /// Every time a state-driven endpoint reads — the ODE-accumulated hazards' and the
    /// drug-driven CTMM generators' sub-grids — sorted: one dual solve per evaluation, shared.
    ode_times: Vec<f64>,
}

impl<'a> NonGaussianTerm<'a> {
    pub fn new(model: &'a CompiledModel, subject: &'a Subject) -> Self {
        let tte = subject
            .obs_records
            .iter()
            .any(|r| matches!(r, ObsRecord::Event { .. }))
            .then(|| TteTerm::new(model, subject));
        #[cfg_attr(not(feature = "markov"), allow(unused_mut))]
        let mut ode_times: Vec<f64> =
            tte.as_ref().map_or_else(Vec::new, |t| t.ode_times().to_vec());
        #[cfg(feature = "markov")]
        if model.has_ctmm() {
            ode_times.extend(crate::markov::endpoint::ctmm_state_grid(model, subject));
        }
        sort_dedup(&mut ode_times);
        NonGaussianTerm { model, subject, tte, ode_times }
    }
}

impl SubjectTerm for NonGaussianTerm<'_> {
    fn eval<T: crate::sens::num::vm::VmNum>(&self, theta: &[T], eta: &[T]) -> Option<T> {
        let (model, subject) = (self.model, self.subject);
        let solved = if self.ode_times.is_empty() {
            None
        } else {
            Some(crate::sens::ode_provider::adjoint::ode_states_at_g(
                model,
                subject,
                theta,
                eta,
                &self.ode_times,
            )?)
        };
        let shared = solved.as_ref().map(|s| (self.ode_times.as_slice(), s));
        let mut q = T::from_f64(0.0);
        if let Some(tte) = self.tte.as_ref() {
            q = q + tte.eval_shared(theta, eta, shared)?;
        }
        if model.has_discrete() {
            q = q + crate::categorical::DiscreteTerm { model, subject }.eval(theta, eta)?;
        }
        #[cfg(feature = "markov")]
        if model.has_ctmm() {
            q = q + crate::markov::endpoint::CtmmTerm { model, subject }
                .eval_shared(theta, eta, shared)?;
        }
        Some(q)
    }
}

/// Exact `∂/∂η` of the subject's TTE data term at 1× scale: the inner EBE gradient's
/// TTE block. `None` (caller keeps FD) when an endpoint is not served exactly or the
/// value is the repel sentinel.
pub fn tte_subject_eta_grad(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    // The subject's compiled term (#2786).
    use crate::sens::ode_provider::adjoint::terms::{term_gradient_planned, TermKind};
    let term = TteTerm::new(model, subject);
    let (q, g) = term_gradient_planned(model, subject, TermKind::Tte, &term, theta, eta)?;
    (q.is_finite() && q < REPEL_SENTINEL_FLOOR && g.iter().all(|v| v.is_finite())).then_some((q, g))
}

/// Whether `subject`'s records on the TTE endpoint `cmt` have an exact η-channel: the
/// endpoint's ([`tte_endpoint_served`]), and for an ODE-accumulated hazard the subject
/// inside the dual walk's scope as well.
pub fn tte_record_served(model: &CompiledModel, subject: &Subject, cmt: usize) -> bool {
    tte_endpoint_served(model, cmt)
        && match model.endpoints.get(&cmt) {
            Some(crate::types::EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { .. },
                ..
            }) => crate::sens::ode_provider::adjoint::ode_hazard_walk_in_scope(model, subject),
            _ => true,
        }
}

/// Whether the TTE endpoint on `cmt` has an exact η-channel: an analytic hazard with a
/// dual-evaluable program, or an ODE-accumulated one with a compiled RHS (#2786).
pub fn tte_endpoint_served(model: &CompiledModel, cmt: usize) -> bool {
    match model.endpoints.get(&cmt) {
        Some(crate::types::EndpointLikelihood::Tte { hazard, hazard_program, .. }) => {
            match hazard {
                HazardSpec::Analytic { .. } => hazard_program.is_some(),
                HazardSpec::OdeAccumulated { .. } => {
                    model.ode_spec.as_ref().is_some_and(|o| o.rhs_program.is_some())
                }
            }
        }
        _ => false,
    }
}

/// `model` with every analytic hazard's program removed: its TTE terms then have no exact
/// η-channel, the shape of an ODE-accumulated hazard or a hand-built `param_fn`. For tests
/// that pin the FD route and its warnings.
#[cfg(test)]
pub fn strip_hazard_programs(mut model: CompiledModel) -> CompiledModel {
    for ep in model.endpoints.values_mut() {
        if let crate::types::EndpointLikelihood::Tte {
            hazard: HazardSpec::Analytic { .. },
            hazard_program,
            ..
        } = ep
        {
            *hazard_program = None;
        }
    }
    model
}

/// Objectives at or above this are the `1e20` repel sentinel, not a likelihood.
pub const REPEL_SENTINEL_FLOOR: f64 = 1e19;

/// Per-record TTE negative log-likelihood from `H(t)` (`cumhaz_at`) and `h(t)`
/// (`hazard_at`) curves, shared by the analytic and ODE-accumulated paths; formulas as
/// in `tte_data_term`, with left truncation at `entry_time`. Returns the 1e20 sentinel
/// when ill-defined, including a non-monotone `H` (`S = exp(−ΔH) > 1`).
#[cfg(test)]
pub fn tte_nll_from_curves(
    records: &[ObsRecord], cumhaz_at: impl Fn(f64) -> f64, hazard_at: impl Fn(f64) -> f64,
    tol: MonoTol,
) -> f64 {
    tte_nll_from_curves_g(records, cumhaz_at, hazard_at, tol, &mut |_| {})
}

/// `tte_nll_from_curves` over any [`PkNum`] curve values (#2437); every guard reads the
/// value only. Each scored record's [`RecordTerm`] goes to `visit` before it is added, so
/// the NLL is the in-order fold of the visited terms (#901). On a sentinel return the
/// visitor may already hold earlier terms; check the return value first.
pub fn tte_nll_from_curves_g<T: PkNum>(
    records: &[ObsRecord], cumhaz_at: impl Fn(f64) -> T, hazard_at: impl Fn(f64) -> T,
    tol: MonoTol, visit: &mut impl FnMut(RecordTerm<T>),
) -> T {
    let mut nll = T::from_f64(0.0);

    // A cumulative hazard is non-decreasing: `H(b) ≥ H(a)` for `b ≥ a`. A drug-driven
    // `[odes]` hazard is a free user expression with no `h ≥ 0` constraint, so a
    // sign-flipped / non-monotone hazard can make an increment negative — `S = exp(−ΔH)
    // > 1`, ill-posed. Treat an increment as a genuine violation only past the solver's
    // own round-off band `abstol + reltol·|H|` (see `MonoTol`), so a near-zero hazard's
    // quadrature noise on a flat `H` is tolerated while a larger negative step is not.
    // The simulation path hard-errors on the same non-monotone CHZ; here, with an
    // optimizer to steer, it folds into the shared 1e20 sentinel.
    // #618: the previous fixed `1e-3·|H|` term was 10× looser than the default `reltol`
    // and ignored a user-tightened `ode_reltol`, so a genuine negative step up to ~0.1%·H
    // slipped past as round-off; tying the floor to `tol` closes that gap.
    let monotone_violation = |hi: T, lo: T| cumhaz_monotone_violation_g(hi, lo, tol);

    for (index, record) in records.iter().enumerate() {
        let ObsRecord::Event { time, event_type, entry_time, .. } = record else {
            // Non-TTE records (discrete-state / count endpoints) don't contribute
            // to the survival likelihood; skip them.
            continue;
        };

        let h_entry = entry_lower_limit(*entry_time, &cumhaz_at);

        match event_type {
            EventType::RightCensored | EventType::Exact => {
                match score_event(event_type, *time, h_entry, &cumhaz_at, &hazard_at, tol) {
                    Some((dh, log_h, _)) => nll = nll + visit_point(visit, index, dh, log_h),
                    None => return T::from_f64(SUBJECT_SENTINEL_NLL),
                }
            }
            EventType::IntervalCensored { left, right } => {
                let h_l = cumhaz_at(*left);
                let h_r = cumhaz_at(*right);
                // H(left) ≥ H(entry) and H(right) > H(left): a non-monotone CHZ
                // (negative hazard) violates either and is ill-posed.
                if monotone_violation(h_l, h_entry) {
                    return T::from_f64(SUBJECT_SENTINEL_NLL);
                }
                let a = h_l - h_entry; // H(left) − H(entry) ≥ 0
                let delta = h_r - h_l; // H(right) − H(left) > 0 for a proper interval
                if !delta.br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0) {
                    // Degenerate: right ≤ left, or hazard non-monotone (bad params).
                    return T::from_f64(SUBJECT_SENTINEL_NLL);
                }
                // log P(left < T ≤ right | T > entry) = −a + log(1 − exp(−delta)), in the
                // log domain via expm1 to keep precision for a small Δ.
                let em1 = (-delta).jet_unary(crate::sens::num::JetFn::ExpM1);
                let log_prob = -a + em1.abs().ln();
                if !log_prob.br_abs_lt(f64::INFINITY) {
                    return T::from_f64(SUBJECT_SENTINEL_NLL);
                }
                visit(RecordTerm::Interval {
                    #[cfg(test)]
                    log_prob,
                });
                nll = nll - log_prob;
            }
        }
    }

    // `|nll| < ∞`: finite, through the branch API (#2786).
    if nll.br_abs_lt(f64::INFINITY) {
        nll
    } else {
        T::from_f64(SUBJECT_SENTINEL_NLL)
    }
}

/// Round-off-tolerant non-monotonicity test for a cumulative-hazard increment
/// `H(hi) − H(lo)`: a violation only past `abstol + reltol·|H|` ([`MonoTol`], #633, #618).
/// Shared by every survival loop; goes through the branch API so a recording type
/// records the decision (#2786).
fn cumhaz_monotone_violation_g<T: PkNum>(hi: T, lo: T, tol: MonoTol) -> bool {
    use crate::sens::num::BranchCmp::{Ge, Lt};
    let (a, b) = (hi.abs(), lo.abs());
    let m = if a.br_cmp(Ge, b) { a } else { b };
    let scale = T::from_f64(tol.abstol) + T::from_f64(tol.reltol) * m;
    (hi - lo).br_cmp(Lt, -scale)
}

/// Lower cumulative-hazard limit from left truncation: `H(entry)` when `entry > 0`, else
/// `0`. Shared by every survival path so `entry` always means "condition on survival
/// past entry", never "restart the clock" (#740).
#[inline]
fn entry_lower_limit<T: PkNum>(entry_time: f64, cumhaz_at: impl Fn(f64) -> T) -> T {
    if entry_time > 0.0 {
        cumhaz_at(entry_time)
    } else {
        T::from_f64(0.0)
    }
}

/// One survival record's scored likelihood term, as the three survival loops hand it to
/// their visitor (#901). A `Point`'s `index` is the record's position in the slice. The
/// loops' NLL is the in-order fold `dh − log_h` (`dh` when censored) / `−log_prob`, so a
/// residual reads the very `ΔH` the likelihood scored.
#[derive(Clone, Copy, Debug)]
pub enum RecordTerm<T> {
    /// A right-censored (`log_h = None`) or exact (`Some(log h)`) record whose
    /// cumulative-hazard increment over its own risk interval is `dh`.
    Point { index: usize, dh: T, log_h: Option<T> },
    /// An interval-censored record, scored as `−log P(left < T ≤ right | T > entry)`;
    /// `log_prob` is carried only for the tests' fold.
    Interval {
        #[cfg(test)]
        log_prob: T,
    },
}

/// Hand a `Point` record to `visit` and return its NLL term `dh − log_h` (`dh` when
/// censored) — the one place that term is formed, so the fold and the visitor agree.
#[inline]
fn visit_point<T: PkNum>(
    visit: &mut impl FnMut(RecordTerm<T>), index: usize, dh: T, log_h: Option<T>,
) -> T {
    visit(RecordTerm::Point { index, dh, log_h });
    match log_h {
        Some(l) => dh - l,
        None => dh,
    }
}

/// Deviance residual of one record (#901): `sign(M)·√(2·[ΔH − δ − δ·ln ΔH])` with
/// `M = δ − ΔH`, and `−√(2ΔH)` for a censor (no `0·ln 0`). `ΔH` is clamped at 0 (it may
/// sit a `MonoTol` band below); an event with `ΔH = 0` is `+∞`, as in R's `survival`.
/// The event radicand `x − 1 − ln x` is not clamped: no negative value in a ±2·10⁵-ulp
/// sweep around 1 or in 10⁶ draws in 1 ± 1e-4, and a `.max(0.0)` would hide a `NaN`.
pub fn deviance_residual(delta: f64, dh: f64) -> f64 {
    let dh = dh.max(0.0);
    if delta == 0.0 {
        return -(2.0 * dh).sqrt();
    }
    let radicand = dh - delta - delta * dh.ln();
    (delta - dh).signum() * (2.0 * radicand).sqrt()
}

/// One endpoint's [`TteObsDiagnostic`] rows (#901), one per `Event` record in order.
/// `terms = None` (the 1e20 sentinel) leaves residuals NaN; an interval-censored record
/// has NaN `dv` and residuals. `written_time` maps the engine clock back to data TIME.
pub fn tte_record_diagnostics(
    records: &[ObsRecord], terms: Option<&[RecordTerm<f64>]>, written_time: impl Fn(f64) -> f64,
) -> Vec<crate::types::TteObsDiagnostic> {
    let mut by_index: Vec<Option<(f64, bool)>> = vec![None; records.len()];
    for term in terms.unwrap_or_default() {
        if let RecordTerm::Point { index, dh, log_h } = *term {
            by_index[index] = Some((dh, log_h.is_some()));
        }
    }
    let mut rows = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let ObsRecord::Event { time, event_type, cmt, .. } = record else {
            continue;
        };
        let dv = match event_type {
            EventType::Exact => 1.0,
            EventType::RightCensored => 0.0,
            EventType::IntervalCensored { .. } => f64::NAN,
        };
        let (coxsnell, martingale, deviance, pit) = match by_index[index] {
            Some((dh, is_event)) => {
                let delta = if is_event { 1.0 } else { 0.0 };
                (dh, delta - dh, deviance_residual(delta, dh), (-dh).exp())
            }
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };
        rows.push(crate::types::TteObsDiagnostic {
            cmt: *cmt,
            time: written_time(*time),
            dv,
            coxsnell,
            martingale,
            deviance,
            pit,
            npde: f64::NAN,
        });
    }
    rows
}

/// Shared `RightCensored` / `Exact` scoring of one record at hazard-clock argument `arg`
/// with lower limit `h_lo`. Returns `(ΔH, Some(log h) for an exact event, H(arg))`, or
/// `None` when ill-defined (non-positive hazard at an event, non-monotone `H` past
/// [`MonoTol`]); callers fold `None` into the 1e20 sentinel. `IntervalCensored` is
/// handled by the callers, so that arm is unreachable here.
#[inline]
fn score_event<T: PkNum>(
    event_type: &EventType, arg: f64, h_lo: T, cumhaz_at: impl Fn(f64) -> T,
    hazard_at: impl Fn(f64) -> T, tol: MonoTol,
) -> Option<(T, Option<T>, T)> {
    match event_type {
        EventType::RightCensored => {
            let h_a = cumhaz_at(arg);
            if cumhaz_monotone_violation_g(h_a, h_lo, tol) {
                return None;
            }
            Some((h_a - h_lo, None, h_a))
        }
        EventType::Exact => {
            let h_val = hazard_at(arg);
            if h_val.br_cmp_c(crate::sens::num::BranchCmp::Le, 0.0) {
                return None;
            }
            let h_a = cumhaz_at(arg);
            if cumhaz_monotone_violation_g(h_a, h_lo, tol) {
                return None;
            }
            Some((h_a - h_lo, Some(h_val.ln()), h_a))
        }
        // Handled inline by the single-event path; rejected by the RTTE paths.
        EventType::IntervalCensored { .. } => None,
    }
}

/// Clock-forward (Andersen–Gill) recurrent-event (RTTE) negative log-likelihood:
///
/// ```text
///   −log L = H(T) − H(entry) − Σ_k log h(t_k)
/// ```
///
/// The cumulative hazard is integrated once over the risk window: each record's lower
/// limit is the previous record's time, so the increments telescope (summing independent
/// single-event terms would over-count by `Σ_k H(t_k)`; §3.3). Preconditions, enforced by
/// `api::check_rtte_records`: records in nondecreasing finite `time`; only the first
/// record's `entry_time` is used; `IntervalCensored` is unsupported (sentinel here).
/// Returns the 1e20 sentinel on a non-positive hazard at an event, a non-monotone `H`,
/// or a non-finite total.
pub fn rtte_forward_nll_from_curves(
    records: &[ObsRecord], cumhaz_at: impl Fn(f64) -> f64, hazard_at: impl Fn(f64) -> f64,
    tol: MonoTol,
) -> f64 {
    rtte_forward_nll_from_curves_g(records, cumhaz_at, hazard_at, tol, &mut |_| {})
}

/// [`rtte_forward_nll_from_curves`] over any [`PkNum`]; see [`tte_nll_from_curves_g`].
pub fn rtte_forward_nll_from_curves_g<T: PkNum>(
    records: &[ObsRecord], cumhaz_at: impl Fn(f64) -> T, hazard_at: impl Fn(f64) -> T,
    tol: MonoTol, visit: &mut impl FnMut(RecordTerm<T>),
) -> T {
    let mut nll = T::from_f64(0.0);
    // Running lower limit `H(lo)`: seeded from the entry, then each record's `H(time)`.
    let mut h_lo = T::from_f64(0.0);
    let mut seeded = false;

    for (index, record) in records.iter().enumerate() {
        let ObsRecord::Event { time, event_type, entry_time, .. } = record else {
            // Non-TTE records (discrete-state / count endpoints) don't contribute
            // to the survival likelihood; skip them.
            continue;
        };

        if !seeded {
            h_lo = entry_lower_limit(*entry_time, &cumhaz_at);
            seeded = true;
        }

        match event_type {
            EventType::RightCensored | EventType::Exact => {
                match score_event(event_type, *time, h_lo, &cumhaz_at, &hazard_at, tol) {
                    Some((dh, log_h, h_t)) => {
                        nll = nll + visit_point(visit, index, dh, log_h);
                        h_lo = h_t;
                    }
                    None => return T::from_f64(SUBJECT_SENTINEL_NLL),
                }
            }
            // Unsupported for RTTE (rejected at the fit boundary); defense in depth.
            EventType::IntervalCensored { .. } => return T::from_f64(SUBJECT_SENTINEL_NLL),
        }
    }

    // `|nll| < ∞`: finite, through the branch API (#2786).
    if nll.br_abs_lt(f64::INFINITY) {
        nll
    } else {
        T::from_f64(SUBJECT_SENTINEL_NLL)
    }
}

/// Clock-reset (gap-time / renewal) recurrent-event (RTTE) negative log-likelihood:
///
/// ```text
///   −log L = [ H(t₁) − H(entry) − log h(t₁) ]      first sojourn, absolute clock from 0
///          + Σ_{k≥2} [ H(Δ_k) − log h(Δ_k) ]       renewal gaps, Δ_k = t_k − t_{k−1}
///          + H(Δ_censor)
/// ```
///
/// The hazard clock resets at each event, so later gaps are scored at their duration.
/// For an exponential hazard this equals the clock-forward likelihood.
///
/// Left truncation follows convention B (#740): the first sojourn is in absolute time,
/// conditioned on survival to `entry` (the clock-forward first-record term), never a
/// renewal clock restarted at `entry`. With `entry == 0` it is the pure renewal form.
/// Only the first record's `entry_time` is used; later gaps are fresh renewals.
/// Preconditions and the 1e20 sentinel cases match [`rtte_forward_nll_from_curves`].
pub fn rtte_reset_nll_from_curves(
    records: &[ObsRecord], cumhaz_at: impl Fn(f64) -> f64, hazard_at: impl Fn(f64) -> f64,
    tol: MonoTol,
) -> f64 {
    rtte_reset_nll_from_curves_g(records, cumhaz_at, hazard_at, tol, &mut |_| {})
}

/// [`rtte_reset_nll_from_curves`] over any [`PkNum`]; see [`tte_nll_from_curves_g`].
pub fn rtte_reset_nll_from_curves_g<T: PkNum>(
    records: &[ObsRecord], cumhaz_at: impl Fn(f64) -> T, hazard_at: impl Fn(f64) -> T,
    tol: MonoTol, visit: &mut impl FnMut(RecordTerm<T>),
) -> T {
    let mut nll = T::from_f64(0.0);
    // End of the previous sojourn (a renewal point): `None` until the first record.
    let mut prev_time: Option<f64> = None;

    for (index, record) in records.iter().enumerate() {
        let ObsRecord::Event { time, event_type, entry_time, .. } = record else {
            // Non-TTE records (discrete-state / count endpoints) don't contribute
            // to the survival likelihood; skip them.
            continue;
        };

        // First record: absolute `t₁` with lower limit H(entry) (convention B); later
        // records: the gap `t − t_prev` with lower limit H(0) = 0.
        let (arg, h_lo) = match prev_time {
            None => {
                let h_entry = entry_lower_limit(*entry_time, &cumhaz_at);
                (*time, h_entry)
            }
            Some(prev) => (*time - prev, T::from_f64(0.0)),
        };
        // A negative gap means out-of-order records: fail closed. `entry > t₁` is
        // caught by the monotone check below instead.
        if arg < 0.0 {
            return T::from_f64(SUBJECT_SENTINEL_NLL);
        }

        match event_type {
            EventType::RightCensored | EventType::Exact => {
                match score_event(event_type, arg, h_lo, &cumhaz_at, &hazard_at, tol) {
                    Some((dh, log_h, _)) => nll = nll + visit_point(visit, index, dh, log_h),
                    None => return T::from_f64(SUBJECT_SENTINEL_NLL),
                }
            }
            // RTTE + interval censoring is unsupported (rejected at parse); sentinel.
            EventType::IntervalCensored { .. } => return T::from_f64(SUBJECT_SENTINEL_NLL),
        }
        prev_time = Some(*time);
    }

    // `|nll| < ∞`: finite, through the branch API (#2786).
    if nll.br_abs_lt(f64::INFINITY) {
        nll
    } else {
        T::from_f64(SUBJECT_SENTINEL_NLL)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FD Hessian: the tests' reference stencil
// ─────────────────────────────────────────────────────────────────────────────

/// 4-point central-stencil FD Hessian of `eval` at `eta_hat` with per-axis steps `eps`:
/// the tests' reference for the plans' exact Hessians (#2786), and the fallback for a
/// subject whose term no plan records. Entry (j,k) is
/// `(f(+sj,+sk) − f(+sj,−sk) − f(−sj,+sk) + f(−sj,−sk)) / (4·sj·sk)`.
pub fn data_term_hessian_fd(
    eval: impl Fn(&[f64]) -> f64, eta_hat: &[f64], eps: &[f64],
) -> DMatrix<f64> {
    let n = eta_hat.len();
    let mut h = DMatrix::zeros(n, n);

    let perturb = |j: usize, dj: f64, k: usize, dk: f64| -> f64 {
        let mut e = eta_hat.to_vec();
        e[j] += dj * eps[j];
        e[k] += dk * eps[k];
        eval(&e)
    };

    for j in 0..n {
        for k in 0..=j {
            let entry =
                (perturb(j, 1.0, k, 1.0) - perturb(j, 1.0, k, -1.0) - perturb(j, -1.0, k, 1.0)
                    + perturb(j, -1.0, k, -1.0))
                    / (4.0 * eps[j] * eps[k]);
            h[(j, k)] = entry;
            h[(k, j)] = entry;
        }
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Draw one uncensored latent event time, consuming exactly one uniform. `entry_time > 0`
/// draws conditional on survival past entry. Returns `f64::MAX` for degenerate / improper
/// cases (`λ = 0`; a Gompertz with `γ < 0` that never reaches the quantile).
fn draw_tte_latent<R: rand::Rng>(
    family: HazardFamily, params: &[f64], entry_time: f64, rng: &mut R,
) -> f64 {
    // Open01 samples from (0, 1) exclusive, avoiding the u=0 edge case that
    // would send -ln(u) to +∞.
    let u: f64 = rng.sample(rand::distr::Open01);
    if entry_time > 0.0 {
        sample_conditional_event_time(family, params, entry_time, u)
    } else {
        sample_event_time(family, params, u)
    }
}

/// Draw one TTE outcome within `window`: `(t_event, true)` for a draw strictly below the
/// window, else `(window, false)`. A `f64::MAX` degenerate draw is never an event, even
/// with an unbounded window (see [`observation_window`]).
fn draw_tte_outcome<R: rand::Rng>(
    family: HazardFamily, params: &[f64], entry_time: f64, window: f64, rng: &mut R,
) -> (f64, bool, bool) {
    let t_event = draw_tte_latent(family, params, entry_time, rng);
    // The `f64::MAX` degenerate sentinel (#763) is reported to the caller and never an
    // observed event, even with an unbounded (`+∞`) window.
    let degenerate = t_event >= f64::MAX;
    if t_event < window && !degenerate {
        (t_event, true, false)
    } else {
        (window, false, degenerate)
    }
}

/// The administrative right-censoring horizon for a TTE record, or `+∞` when it carries
/// none. Only a `RightCensored` record marks a horizon; an `Exact` / `IntervalCensored`
/// `time` is a realized event, and censoring a re-simulated draw there would bias a VPC.
fn observation_window(event_type: &EventType, time: f64) -> f64 {
    match event_type {
        EventType::RightCensored => time,
        EventType::Exact | EventType::IntervalCensored { .. } => f64::INFINITY,
    }
}

/// Resolve a competing-risks outcome: the earliest latent time wins (`total_cmp`, lowest
/// index on ties) and is observed only if before `window` and below the `f64::MAX`
/// sentinel. Returns `(winning cause, time, event_observed)`.
fn resolve_competing_risks(latents: &[f64], window: f64) -> (usize, f64, bool) {
    let (win_idx, &t_star) = latents
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .expect("at least one cause");
    let event = t_star < window && t_star < f64::MAX;
    (win_idx, if event { t_star } else { window }, event)
}

/// Destructure a TTE endpoint and evaluate its analytic hazard parameters at
/// `(theta, eta, covariates)`; `None` for a non-`Tte` endpoint or an ODE-accumulated
/// hazard. Shared by simulation and `predict_survival`.
pub fn tte_cause_params(
    endpoint: &EndpointLikelihood, theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>,
) -> Option<(HazardFamily, Vec<f64>)> {
    let EndpointLikelihood::Tte { hazard, .. } = endpoint else {
        return None;
    };
    // ODE-accumulated hazards have no closed-form cause params.
    let HazardSpec::Analytic { family, param_fn, .. } = hazard else {
        return None;
    };
    Some((*family, param_fn(theta, eta, covariates)))
}

/// Solve a subject's augmented ODE at `(theta, eta)` and read `H(t) = u[chz_state]` and
/// `h(t) = u̇[chz_state]` at each of `times` (joint PK-TTE, #564). Returns `(H, h)`
/// aligned to `times`, NaN where the solve did not reach. Shared by `tte_ode_nll` and
/// `api::predict_survival`.
#[cfg(feature = "survival")]
pub fn ode_cumhaz_hazard(
    model: &crate::types::CompiledModel, subject: &crate::types::Subject, chz_state: usize,
    theta: &[f64], eta: &[f64], times: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = times.len();
    let (mut cum, mut haz) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    let Some(ode) = model.ode_spec.as_ref() else {
        return (cum, haz);
    };
    // Single-pass integration: the disposition and `init()` read `TIME` at t=0 (#610);
    // every dose-record quantity (kernel, fractions, lags, `dur`, `F`, SS run-in) is read
    // at its own dose record, as the Gaussian predictions do (#1575).
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let mut pk_at_dose = Vec::new();
    crate::pk::compute_dose_pk_params_into(model, subject, theta, eta, &mut pk_at_dose);
    let dose_reads = crate::ode::predictions::DoseReads::PerDose(&pk_at_dose);
    let states = crate::ode::predictions::ode_dense_solve_states_reading(
        ode, &pk.values, dose_reads, subject, times,
    );
    if states.len() != n {
        return (cum, haz);
    }
    // `h(t)` is the hazard RHS at the integrated state; one buffer, only `chz_state` read.
    let mut du = vec![0.0; ode.n_states];
    let dose_lagtimes = crate::ode::predictions::dose_lagtimes_reading(subject, ode, dose_reads);
    let first_dose_time = crate::ode::predictions::earliest_dose_time(&subject.doses);
    for (i, &t) in times.iter().enumerate() {
        cum[i] = states[i][chz_state];
        let rhs_params = crate::ode::predictions::rhs_ext_params_at(
            &subject.doses,
            &dose_lagtimes,
            first_dose_time,
            &pk.values,
            t,
        );
        (ode.rhs)(&states[i], &rhs_params, t, &mut du);
        haz[i] = du[chz_state];
    }
    (cum, haz)
}

/// Draw one ODE-accumulated TTE latent event time: draw `u ~ U(0,1)` and root-find the
/// first time `H` reaches `−log u`. Returns `f64::MAX` when it does not fire by `horizon`
/// (which must be finite; left truncation is rejected up front). A failed solve returns
/// `Err` with the solver's reason; the caller censors the subject and warns
/// (`W_TTE_SOLVE_FAILED`, #2087). `u` is drawn before the solve, so RNG use is fixed.
#[cfg(feature = "survival")]
fn draw_ode_tte_latent<R: rand::Rng>(
    model: &crate::types::CompiledModel, subject: &crate::types::Subject, theta: &[f64],
    eta: &[f64], chz_state: usize, horizon: Option<f64>, rng: &mut R,
) -> Result<f64, String> {
    let horizon = horizon.expect(
        "ODE-accumulated TTE simulation requires a finite horizon; the simulate entry \
         points validate this before sampling",
    );
    let u: f64 = rng.sample(rand::distr::Open01);
    let threshold = -u.ln();
    // Integrated exactly as `ode_cumhaz_hazard` does (#610, #1575), so a simulated event
    // time is consistent with the fitted hazard.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let mut pk_at_dose = Vec::new();
    crate::pk::compute_dose_pk_params_into(model, subject, theta, eta, &mut pk_at_dose);
    let ode = model.ode_spec.as_ref().expect("ODE-accumulated hazard requires an [odes] block");
    match crate::ode::predictions::ode_solve_until_chz_threshold(
        ode,
        &pk.values,
        crate::ode::predictions::DoseReads::PerDose(&pk_at_dose),
        subject,
        chz_state,
        threshold,
        horizon,
    ) {
        crate::ode::ThresholdOutcome::Crossed(t) => Ok(t),
        crate::ode::ThresholdOutcome::CensoredAtHorizon => Ok(f64::MAX),
        crate::ode::ThresholdOutcome::SolveFailed(why) => Err(why),
    }
}

/// A subject's single RTTE cause: `(cmt, clock, family, params, entry)`, or `None` for a
/// non-RTTE subject. All rows at the RTTE CMT are one recurrent stream.
/// `api::validate_tte_simulatable` rejects a second RTTE CMT or a competing single-TTE
/// sibling; this panics if either (or a non-analytic hazard) slips through.
#[cfg(feature = "survival")]
fn rtte_cause(
    model: &crate::types::CompiledModel, subject: &crate::types::Subject, theta: &[f64],
    eta: &[f64],
) -> Option<(usize, RtteClock, HazardFamily, Vec<f64>, f64)> {
    let mut found: Option<(usize, RtteClock, HazardFamily, Vec<f64>, f64)> = None;
    for record in &subject.obs_records {
        let ObsRecord::Event { cmt, entry_time, .. } = record else {
            continue;
        };
        let Some(EndpointLikelihood::Tte {
            hazard,
            recurrence: TteRecurrence::Repeated { clock },
            ..
        }) = model.endpoints.get(cmt)
        else {
            continue;
        };
        // The parser rejects `type = rtte` with an ODE-accumulated hazard, so only
        // the analytic closed form reaches simulation.
        let HazardSpec::Analytic { family, param_fn, .. } = hazard else {
            panic!(
                "RTTE endpoint on CMT={cmt} carries a non-analytic hazard; the parser rejects \
                 `type = rtte` with an ODE-accumulated `hazard = ...`, so this is unreachable"
            );
        };
        match &found {
            None => {
                found = Some((
                    *cmt,
                    *clock,
                    *family,
                    param_fn(theta, eta, &subject.covariates),
                    *entry_time,
                ));
            }
            // Additional rows on the same CMT belong to the same recurrent stream;
            // one call to `simulate_rtte_stream` regenerates all of them.
            Some((c0, ..)) if *c0 == *cmt => {}
            Some((c0, ..)) => panic!(
                "subject '{}' has RTTE causes on multiple CMTs ({c0} and {cmt}); \
                 api::validate_tte_simulatable must reject multi-cause RTTE simulation",
                subject.id
            ),
        }
    }
    found
}

/// Generate one subject's RTTE stream to `horizon` and append the rows (analytic hazard
/// only).
///
/// - clock-forward: each event drawn conditional on survival past the previous one
///   ([`sample_conditional_event_time`]); the clock never resets.
/// - clock-reset: each gap a fresh draw from 0 ([`sample_event_time`]).
///
/// Left truncation (convention B, #740): forward seeds `t_prev = entry`; reset draws only
/// the first sojourn conditional on survival to `entry`. With `entry == 0` the RNG
/// sequence is byte-identical to the untruncated draw.
///
/// Emits one observed row per event strictly before `horizon`, then one censor row at
/// `horizon` (the reference RTTE layout). A pathological parameterization is handled per
/// subject (#762): the stream is skipped or truncated with a warning, never aborted.
#[cfg(feature = "survival")]
#[allow(clippy::too_many_arguments)]
fn simulate_rtte_stream<R: rand::Rng>(
    family: HazardFamily, params: &[f64], clock: RtteClock, horizon: f64, entry: f64, cmt: usize,
    id: &str, draw: usize, sim: usize, rng: &mut R,
    results: &mut Vec<crate::sim_result::SimulationResult>, warnings: &mut Vec<String>,
) {
    let push =
        |time: f64, observed: bool, results: &mut Vec<crate::sim_result::SimulationResult>| {
            results.push(crate::sim_result::SimulationResult {
                draw,
                sim,
                id: id.to_string(),
                time,
                cmt,
                ipred: f64::NAN,
                outcome: SimOutcome::Event { time, observed },
                mixnum: None, // stamped by the caller that holds the class (#2372)
            });
        };

    // Backstop against a degenerate hazard whose inter-event gaps vanish.
    const MAX_EVENTS: usize = 1_000_000;

    // Pre-flight degeneracy guard (#762): skip a stream whose expected event count is
    // huge, recording the censor plus a warning. Clock-specific (#772):
    //   * forward: `E[N] = H(horizon)` exactly;
    //   * reset: `E[N] ≈ horizon / median gap` (`H(horizon)` over-counts an increasing
    //     hazard, e.g. ~1e6 vs a few dozen). The median is computed at the fixed 0.5,
    //     so no RNG is drawn.
    // Left truncation (#740) counts only the window past `entry`.
    let expected = match clock {
        RtteClock::Forward => {
            let h_entry = if entry > 0.0 { cum_hazard(family, entry, params) } else { 0.0 };
            cum_hazard(family, horizon, params) - h_entry
        }
        RtteClock::Reset => (horizon - entry.max(0.0)) / sample_event_time(family, params, 0.5),
    };
    if !expected.is_finite() || expected > MAX_EVENTS as f64 {
        warnings.push(format!(
            "W_RTTE_DEGENERATE: subject '{id}' (CMT={cmt}) has a degenerate recurrent hazard \
             (~{expected:.3e} events expected before horizon {horizon}); its event stream was \
             skipped and the subject censored at the horizon (#762). Check the hazard \
             parameters / covariate values."
        ));
        push(horizon, false, results);
        return;
    }

    // Forward: left truncation seeds the clock at `entry`. Reset: only the first sojourn
    // is conditioned (`n_events == 0` in the loop). A non-positive `entry` clamps to 0.
    let mut t_prev = match clock {
        RtteClock::Forward => entry.max(0.0),
        RtteClock::Reset => 0.0_f64,
    };
    let mut n_events = 0usize;
    loop {
        let u: f64 = rng.sample(rand::distr::Open01);
        let t_next = match clock {
            // Condition the next event on survival past the previous one.
            RtteClock::Forward => sample_conditional_event_time(family, params, t_prev, u),
            // Renewal: a fresh gap from 0; the addition is guarded so a `f64::MAX` gap
            // stays `≥ horizon`. Under left truncation the first sojourn is drawn
            // conditional on survival to `entry` (#740).
            RtteClock::Reset => {
                if n_events == 0 && entry > 0.0 {
                    sample_conditional_event_time(family, params, entry, u)
                } else {
                    let gap = sample_event_time(family, params, u);
                    if gap >= f64::MAX - t_prev {
                        f64::MAX
                    } else {
                        t_prev + gap
                    }
                }
            }
        };
        // A draw at/after the horizon, the `f64::MAX` sentinel or NaN ends the stream.
        if t_next < horizon {
            // Backstop for an unlucky realisation (#762): truncate and warn, checked
            // before the push so no ~1e6 rows pile up.
            if n_events >= MAX_EVENTS {
                warnings.push(format!(
                    "W_RTTE_DEGENERATE: subject '{id}' (CMT={cmt}) produced over {MAX_EVENTS} \
                     events before horizon {horizon} (vanishing inter-event gaps); its stream was \
                     truncated and the subject censored at the horizon (#762)."
                ));
                break;
            }
            push(t_next, true, results);
            t_prev = t_next;
            n_events += 1;
        } else {
            break;
        }
    }
    // Trailing administrative right-censoring row at the horizon.
    push(horizon, false, results);
}

/// Draw TTE event/censoring outcomes for a subject and append them to `results`.
///
/// - Single endpoint: the draw is censored at the record's window.
/// - Competing risks (§3.6): one latent per cause CMT; the earliest is the observed
///   event, the others are censored at that time. The shared window is `+∞` if any
///   record is an event, else the common censoring window. One row per cause CMT.
///
/// `horizon` (`[simulation] horizon`, #522) overrides every per-record window, so a
/// competing-risks VPC censors at the planned study end. It never changes the number of
/// uniforms drawn. A degenerate analytic draw (#763) gives one `W_TTE_DEGENERATE_HAZARD`
/// line per cause; the `simulate*` entry points use `simulate_tte_reporting` (#1648).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub fn simulate_tte<R: rand::Rng>(
    model: &crate::types::CompiledModel, subject: &crate::types::Subject, theta: &[f64],
    eta: &[f64], draw: usize, sim: usize, horizon: Option<f64>, rng: &mut R,
    results: &mut Vec<crate::sim_result::SimulationResult>, warnings: &mut Vec<String>,
) {
    let mut degenerate = Vec::new();
    simulate_tte_reporting(
        model,
        subject,
        theta,
        eta,
        draw,
        sim,
        horizon,
        rng,
        results,
        warnings,
        &mut degenerate,
    );
    // A call never produces both kinds of warning: a failed ODE cause returns before the
    // degenerate check.
    warnings.extend(degenerate.into_iter().map(|cmt| degenerate_hazard_warning(&subject.id, cmt)));
}

/// The body of `simulate_tte`, reporting each degenerate analytic cause (#763) as its CMT
/// in `degenerate` instead of a warning line, so a many-replicate caller can report each
/// (subject, CMT) once (#1648).
#[allow(clippy::too_many_arguments)]
pub fn simulate_tte_reporting<R: rand::Rng>(
    model: &crate::types::CompiledModel, subject: &crate::types::Subject, theta: &[f64],
    eta: &[f64], draw: usize, sim: usize, horizon: Option<f64>, rng: &mut R,
    results: &mut Vec<crate::sim_result::SimulationResult>, warnings: &mut Vec<String>,
    degenerate: &mut Vec<usize>,
) {
    // RTTE subjects are one recurrent stream, routed first so the single / competing path
    // stays byte-identical. Invariants are enforced in `api::validate_tte_simulatable`.
    if let Some((cmt, clock, family, params, entry)) = rtte_cause(model, subject, theta, eta) {
        let horizon = horizon.expect(
            "RTTE simulation requires a finite horizon; the simulate entry points validate this \
             before sampling",
        );
        simulate_rtte_stream(
            family,
            &params,
            clock,
            horizon,
            entry,
            cmt,
            &subject.id,
            draw,
            sim,
            rng,
            results,
            warnings,
        );
        return;
    }

    // Each TTE cause carries its window and the kind of hazard that draws its latent;
    // non-TTE records are skipped.
    enum CauseKind {
        Analytic { family: HazardFamily, params: Vec<f64>, entry_time: f64 },
        Ode { chz_state: usize },
    }

    let mut causes: Vec<(usize, f64, CauseKind)> = Vec::new(); // (cmt, window, kind)
    for record in &subject.obs_records {
        let ObsRecord::Event { cmt, entry_time, time, event_type } = record else {
            // Non-TTE records don't contribute a competing-risk cause; skip them.
            continue;
        };
        let Some(EndpointLikelihood::Tte { hazard, .. }) = model.endpoints.get(cmt) else {
            continue;
        };
        // `horizon` overrides every window (#522); see `observation_window` otherwise.
        let window = horizon.unwrap_or_else(|| observation_window(event_type, *time));
        let kind = match hazard {
            HazardSpec::Analytic { family, param_fn, .. } => CauseKind::Analytic {
                family: *family,
                params: param_fn(theta, eta, &subject.covariates),
                entry_time: *entry_time,
            },
            HazardSpec::OdeAccumulated { chz_state } => CauseKind::Ode { chz_state: *chz_state },
        };
        causes.push((*cmt, window, kind));
    }

    let push = |cmt: usize, time: f64, observed: bool, results: &mut Vec<_>| {
        results.push(crate::sim_result::SimulationResult {
            draw,
            sim,
            id: subject.id.clone(),
            time,
            cmt,
            ipred: f64::NAN,
            outcome: SimOutcome::Event { time, observed },
            mixnum: None, // stamped by the caller that holds the class (#2372)
        });
    };

    match causes.as_slice() {
        [] => {}
        // Single endpoint: censor at its own window (analytic draw byte-identical to the
        // pre-ODE path; an ODE hazard root-finds its latent).
        [(cmt, window, kind)] => {
            let (t, observed) = match kind {
                CauseKind::Analytic { family, params, entry_time } => {
                    let (t, observed, is_degenerate) =
                        draw_tte_outcome(*family, params, *entry_time, *window, rng);
                    if is_degenerate {
                        degenerate.push(*cmt);
                    }
                    (t, observed)
                }
                CauseKind::Ode { chz_state } => {
                    match draw_ode_tte_latent(model, subject, theta, eta, *chz_state, horizon, rng)
                    {
                        Ok(latent) => {
                            let observed = latent < *window && latent < f64::MAX;
                            (if observed { latent } else { *window }, observed)
                        }
                        Err(why) => {
                            warnings.push(ode_solve_failed_warning(
                                &subject.id,
                                *cmt,
                                &why,
                                *window,
                                false,
                            ));
                            (*window, false)
                        }
                    }
                }
            };
            push(*cmt, t, observed, results);
        }
        // Competing risks: draw every cause in record order (even after a failed ODE
        // cause) so RNG consumption is fixed (#2087); the earliest wins.
        _ => {
            let drawn: Vec<Result<f64, String>> = causes
                .iter()
                .map(|(_, _, kind)| match kind {
                    CauseKind::Analytic { family, params, entry_time } => {
                        Ok(draw_tte_latent(*family, params, *entry_time, rng))
                    }
                    CauseKind::Ode { chz_state } => {
                        draw_ode_tte_latent(model, subject, theta, eta, *chz_state, horizon, rng)
                    }
                })
                .collect();
            // `max` over windows: any event-bearing cause (+∞) makes the subject
            // uncensored (the #494 anti-truncation rule); `min` would re-censor at an
            // observed event time.
            let window = causes.iter().map(|(_, w, _)| *w).fold(f64::NEG_INFINITY, f64::max);
            // A failed ODE cause leaves nothing to compete with: censor every cause at
            // the window and name each failure; skip the #763 check.
            if drawn.iter().any(Result::is_err) {
                for ((cmt, ..), d) in causes.iter().zip(&drawn) {
                    if let Err(why) = d {
                        warnings.push(ode_solve_failed_warning(
                            &subject.id,
                            *cmt,
                            why,
                            window,
                            true,
                        ));
                    }
                }
                for (cmt, ..) in &causes {
                    push(*cmt, window, false, results);
                }
                return;
            }
            let latents: Vec<f64> = drawn.into_iter().map(Result::unwrap).collect();
            let (win_idx, obs_time, event) = resolve_competing_risks(&latents, window);
            // Only an analytic `f64::MAX` latent is degenerate (#763); an ODE cause's is a
            // legitimate horizon censor.
            if !event {
                for (i, (cmt, _, kind)) in causes.iter().enumerate() {
                    if matches!(kind, CauseKind::Analytic { .. }) && latents[i] >= f64::MAX {
                        degenerate.push(*cmt);
                    }
                }
            }
            for (i, (cmt, ..)) in causes.iter().enumerate() {
                push(*cmt, obs_time, event && i == win_idx, results);
            }
        }
    }
}

/// Shared diagnostic for a subject whose analytic hazard draw degenerated (a
/// non-positive / non-finite effective rate, yielding the `f64::MAX` sentinel and so
/// no event) — collected into [`crate::api::SimulationOutput::warnings`] (#763).
#[cfg(feature = "survival")]
fn degenerate_hazard_warning(id: &str, cmt: usize) -> String {
    degenerate_hazard_text(id, cmt, "")
}

/// The one owner of the `W_TTE_DEGENERATE_HAZARD` text. `censored_in` lists the
/// replicates the subject was censored in (empty for a single replicate). The count is of
/// censorings, since a replicate where another cause fired is not reported (#1648).
#[cfg(feature = "survival")]
fn degenerate_hazard_text(id: &str, cmt: usize, censored_in: &str) -> String {
    format!(
        "W_TTE_DEGENERATE_HAZARD: subject '{id}' (CMT={cmt}) drew a non-positive / non-finite \
         effective hazard rate; no event was generated and the subject is censored at the \
         observation window{censored_in} (#763). Check the hazard parameters / covariate \
         values."
    )
}

/// How many indices a `W_TTE_DEGENERATE_HAZARD` line names before it counts the rest.
const DEGENERATE_INDEX_CAP: usize = 10;

/// Every degenerate analytic hazard draw (#763) of one simulation run that left its
/// subject censored, so each (subject, CMT) gets one `W_TTE_DEGENERATE_HAZARD` line with
/// its replicate count (#1648). Reads no random number and writes no row.
#[derive(Default)]
pub struct DegenerateHazardTally {
    /// Replicates simulated: every `sim` of every draw that ran.
    replicates: usize,
    /// Draws simulated. An uncertainty draw that was skipped never reaches the chokepoint
    /// and is not counted.
    draws: usize,
    /// Per (subject's position, CMT): id and the `(draw, sim)` of each affected replicate.
    /// Keyed on position since two subjects may share an id.
    hits: Vec<(usize, usize, String, Vec<(usize, usize)>)>,
}

impl DegenerateHazardTally {
    /// Count one draw of `n_sim` replicates, once the chokepoint is about to simulate it.
    pub fn count_draw(&mut self, n_sim: usize) {
        self.draws += 1;
        self.replicates += n_sim;
    }

    /// Record that the subject at `position` (named `id`) was censored in replicate
    /// `(draw, sim)` because its cause on `cmt` degenerated.
    pub fn record(&mut self, position: usize, id: &str, cmt: usize, draw: usize, sim: usize) {
        match self
            .hits
            .iter_mut()
            .find(|(at_position, c, _, _)| *at_position == position && *c == cmt)
        {
            Some((_, _, _, at)) => at.push((draw, sim)),
            None => self.hits.push((position, cmt, id.to_string(), vec![(draw, sim)])),
        }
    }

    /// Fold in the tally of a later batch whose `sim` 0 is replicate `sim_offset` of the
    /// whole (`compute_tte_npde` simulates in batches).
    pub fn absorb(&mut self, later: DegenerateHazardTally, sim_offset: usize) {
        self.replicates += later.replicates;
        self.draws = self.draws.max(later.draws);
        for (position, cmt, id, at) in later.hits {
            for (draw, sim) in at {
                self.record(position, &id, cmt, draw, sim + sim_offset);
            }
        }
    }

    /// One line per (subject, CMT) for a run of one parameter set, naming the affected
    /// replicates by their `sim` index.
    pub fn warnings_by_replicate(&self) -> Vec<String> {
        self.render(false)
    }

    /// One line per (subject, CMT) for a run over uncertainty draws, naming the affected
    /// **draws**: a `sim` index repeats in every draw, so it does not identify a replicate.
    /// A run of a single draw has no such ambiguity and reads as
    /// [`DegenerateHazardTally::warnings_by_replicate`].
    pub fn warnings_by_draw(&self) -> Vec<String> {
        self.render(self.draws > 1)
    }

    /// The lines in population order, then CMT order: the order a reader scans the data in,
    /// and the same for every seed.
    fn render(&self, by_draw: bool) -> Vec<String> {
        let n = self.replicates;
        let mut hits: Vec<&(usize, usize, String, Vec<(usize, usize)>)> =
            self.hits.iter().collect();
        hits.sort_by_key(|(position, cmt, _, _)| (*position, *cmt));
        hits.into_iter()
            .map(|(_, cmt, id, at)| {
                let k = at.len();
                // A single replicate: the per-call text, byte for byte.
                if n <= 1 {
                    return degenerate_hazard_warning(id, *cmt);
                }
                let censored_in = match (by_draw, k == n) {
                    (false, true) => format!(" in all {n} replicates"),
                    (false, false) => {
                        let sims: Vec<usize> = at.iter().map(|(_, sim)| *sim).collect();
                        format!(" in {k} of {n} replicates ({})", index_list("sim", &sims))
                    }
                    (true, true) => format!(
                        " in all {n} replicates of all {} simulated uncertainty draws",
                        self.draws
                    ),
                    (true, false) => {
                        let mut draws: Vec<usize> = at.iter().map(|(draw, _)| *draw).collect();
                        draws.dedup();
                        format!(
                            " in {k} of {n} replicates, in {} of {} simulated uncertainty draws \
                             ({})",
                            draws.len(),
                            self.draws,
                            index_list("draw", &draws)
                        )
                    }
                };
                degenerate_hazard_text(id, *cmt, &censored_in)
            })
            .collect()
    }
}

/// `"sim 3, 17, 41"`: the first [`DEGENERATE_INDEX_CAP`] indices under the row field's
/// name, then how many more there are.
fn index_list(label: &str, indices: &[usize]) -> String {
    let shown: Vec<String> =
        indices.iter().take(DEGENERATE_INDEX_CAP).map(usize::to_string).collect();
    let rest = indices.len().saturating_sub(DEGENERATE_INDEX_CAP);
    let more = if rest > 0 { format!(" and {rest} more") } else { String::new() };
    format!("{label} {}{more}", shown.join(", "))
}

/// Diagnostic for a subject whose ODE-accumulated event-time solve failed (#2087). The
/// solver's reason `why` selects the diagnosis and advice: a non-finite `[odes]` state →
/// `[odes]` diverged; a decreasing `H` → the `[event_model]` hazard went negative; a
/// non-finite `H` → both blocks; anything else → no diagnosis, both blocks. `competing`
/// selects the wording where every cause is censored.
#[cfg(feature = "survival")]
fn ode_solve_failed_warning(
    id: &str, cmt: usize, why: &str, window: f64, competing: bool,
) -> String {
    let odes_diverged = why.contains("[odes] state `");
    let hazard = why.contains("the cumulative hazard");
    let hazard_negative = hazard && why.contains("decreased");
    let mut diagnosis = String::new();
    if odes_diverged {
        diagnosis
            .push_str(" The [odes] right-hand side diverged for this subject's parameter draw.");
    }
    if hazard {
        diagnosis.push_str(if hazard_negative {
            " The [event_model] hazard went negative for this subject's parameter draw."
        } else {
            " The [event_model] hazard went non-finite for this subject's parameter draw."
        });
    }
    // A non-finite hazard may come from an `[odes]` state it reads, so it names both.
    let check = match (odes_diverged, hazard, hazard_negative) {
        (true, false, _) => "the [odes] right-hand side",
        (false, true, true) => "the [event_model] hazard expression",
        _ => "the [odes] right-hand side, the [event_model] hazard expression",
    };
    let censored = if competing {
        "every cause of the subject is censored at"
    } else {
        "the subject is censored at"
    };
    format!(
        "W_TTE_SOLVE_FAILED: subject '{id}' (CMT={cmt}): the event-time solve of the \
         ODE-accumulated hazard failed: {why}.{diagnosis} No event time could be simulated, so \
         {censored} the observation window (t={window}) and is not simulated past the \
         failure. The rest of the population still simulates. Check {check} and the parameter \
         values, including the drawn random effects."
    )
}

/// Cause-specific cumulative incidence functions from per-cause cumulative hazards on an
/// ascending grid (`cum_hazards[k][i] = H_k(t_i)`), by the discrete Aalen–Johansen
/// allocation: each step's all-cause survival drop `ΔS` is split by `ΔH_k / Σ_j ΔH_j`.
/// Guarantees `Σ_k F_k + S_all = 1` exactly, is `F = 1 − S` for one cause and exact for
/// constant hazards. Assumes `S_all(0) = 1`. Returns `(cif[k][i], s_all[i])`.
pub fn cif_curves(cum_hazards: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cause = cum_hazards.len();
    let n_grid = cum_hazards.first().map_or(0, |h| h.len());

    let s_all: Vec<f64> = (0..n_grid)
        .map(|i| {
            let h_tot: f64 = (0..n_cause).map(|k| cum_hazards[k][i]).sum();
            (-h_tot).exp()
        })
        .collect();

    let mut cif = vec![vec![0.0_f64; n_grid]; n_cause];
    let mut acc = vec![0.0_f64; n_cause];
    let mut cum_prev = vec![0.0_f64; n_cause]; // H_k(0) = 0
    let mut s_prev = 1.0_f64; // S_all(0)
    for i in 0..n_grid {
        // `drop > 0 ⟺ dh_tot > 0` (same cumulative hazards), so the guard only skips a
        // node carrying no mass.
        let drop = (s_prev - s_all[i]).max(0.0);
        let dh: Vec<f64> =
            (0..n_cause).map(|k| (cum_hazards[k][i] - cum_prev[k]).max(0.0)).collect();
        let dh_tot: f64 = dh.iter().sum();
        if drop > 0.0 && dh_tot > 0.0 && dh_tot.is_finite() {
            for k in 0..n_cause {
                acc[k] += drop * dh[k] / dh_tot;
            }
        }
        for k in 0..n_cause {
            cif[k][i] = acc[k];
            cum_prev[k] = cum_hazards[k][i];
        }
        s_prev = s_all[i];
    }

    (cif, s_all)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests for FD Hessian accuracy
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "residual_tests.rs"]
mod residual_tests;

#[cfg(any(test, feature = "test-support"))]
#[path = "endpoint_order_test_helpers.rs"]
pub mod endpoint_order_test_helpers;

#[cfg(test)]
#[path = "endpoint_order_tests.rs"]
mod endpoint_order_tests;

#[cfg(test)]
#[path = "degenerate_tally_tests.rs"]
mod degenerate_tally_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn fd_hessian_matches_analytic_quadratic() {
        // f(η) = a·η₀² + b·η₁² + c·η₀·η₁
        // Hessian = [[2a, c], [c, 2b]] — exact, no approximation error.
        let a = 3.0_f64;
        let b = 2.0_f64;
        let c = 1.5_f64;
        let eval = move |e: &[f64]| a * e[0] * e[0] + b * e[1] * e[1] + c * e[0] * e[1];
        let eta = &[0.1, -0.2];
        let eps = &[1e-4, 1e-4];
        let h = data_term_hessian_fd(eval, eta, eps);
        assert_abs_diff_eq!(h[(0, 0)], 2.0 * a, epsilon = 1e-6);
        assert_abs_diff_eq!(h[(1, 1)], 2.0 * b, epsilon = 1e-6);
        assert_abs_diff_eq!(h[(0, 1)], c, epsilon = 1e-6);
        assert_abs_diff_eq!(h[(1, 0)], c, epsilon = 1e-6);
    }

    #[test]
    fn fd_hessian_scalar_eta() {
        // f(η) = η² / 2 → f''(η) = 1.0
        let eval = |e: &[f64]| 0.5 * e[0] * e[0];
        let eta = &[0.5];
        let eps = &[1e-4];
        let h = data_term_hessian_fd(eval, eta, eps);
        assert_abs_diff_eq!(h[(0, 0)], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn tte_data_term_right_censored_exponential() {
        use crate::types::HazardFamily;
        // Simple: lambda=0.1, T=10, entry=0 → H(T) = 1.0
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let theta = &[0.1_f64];
        let eta = &[0.0_f64];
        let cov = HashMap::new();
        let nll = tte_data_term(&records, &hazard, TteRecurrence::Single, theta, eta, &cov);
        // -log L = H(T) = 0.1 * 10 = 1.0
        assert_abs_diff_eq!(nll, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn tte_cause_params_evaluates_tte_and_skips_non_tte() {
        use crate::types::{EndpointError, ErrorModel, HazardFamily, HazardParamFn};
        let theta = [0.1_f64];
        let eta = [0.0_f64];
        let cov = HashMap::new();

        // A `Tte` endpoint yields `Some((family, evaluated params))`.
        let param_fn: HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let tte = EndpointLikelihood::Tte {
            hazard: HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn },
            recurrence: TteRecurrence::Single,
            hazard_covariates: Vec::new(),
            hazard_program: None,
        };
        let (family, params) =
            tte_cause_params(&tte, &theta, &eta, &cov).expect("Tte endpoint must yield Some");
        assert_eq!(family, HazardFamily::Exponential);
        assert_eq!(params, vec![0.1]);

        // A non-`Tte` (Gaussian) endpoint takes the `None` branch, letting callers
        // `filter_map`/`continue` over a mixed endpoint map.
        let gaussian = EndpointLikelihood::Gaussian(EndpointError {
            error_model: ErrorModel::Additive,
            sigma_idx: vec![],
        });
        assert!(tte_cause_params(&gaussian, &theta, &eta, &cov).is_none());

        // An ODE-accumulated hazard has no closed-form cause params either (its H/h come
        // from the integrated CHZ state), so it also takes the `None` branch.
        let ode = EndpointLikelihood::Tte {
            hazard: HazardSpec::OdeAccumulated { chz_state: 2 },
            recurrence: TteRecurrence::Single,
            hazard_covariates: Vec::new(),
            hazard_program: None,
        };
        assert!(tte_cause_params(&ode, &theta, &eta, &cov).is_none());
    }

    #[test]
    fn tte_data_term_ode_accumulated_is_analytic_only_sentinel() {
        // The OdeAccumulated arm returns the sentinel (dispatched elsewhere).
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let hazard = HazardSpec::OdeAccumulated { chz_state: 2 };
        let cov = HashMap::new();
        let nll = tte_data_term(&records, &hazard, TteRecurrence::Single, &[0.1], &[0.0], &cov);
        assert_eq!(nll, SUBJECT_SENTINEL_NLL);
    }

    #[test]
    fn tte_nll_from_curves_degenerate_interval_returns_sentinel() {
        use crate::types::HazardFamily;
        // An interval with right ≤ left gives a non-positive Δ = H(right) − H(left); the
        // shared per-record path returns the 1e20 sentinel (covers the degenerate guard).
        let records = vec![ObsRecord::Event {
            time: 0.0,
            event_type: EventType::IntervalCensored { left: 10.0, right: 5.0 },
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let cov = HashMap::new();
        let nll = tte_data_term(&records, &hazard, TteRecurrence::Single, &[0.1], &[0.0], &cov);
        assert_eq!(nll, SUBJECT_SENTINEL_NLL);
    }

    /// A right-censored record with `H(T) < H(entry)` (`S > 1`) hits the 1e20 sentinel
    /// instead of a spurious negative NLL (#564).
    #[test]
    fn tte_nll_from_curves_rejects_non_monotone_censored() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 3,
        }];
        // CHZ is negative by t=5 (entry H=0): a clear monotonicity violation.
        let nll = tte_nll_from_curves(&records, |_t| -0.5, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, SUBJECT_SENTINEL_NLL,
            "non-monotone CHZ on a censor must be sentinel-guarded"
        );
    }

    /// The guard also covers an exact event: a positive `h(T)` with a decreasing `H`.
    #[test]
    fn tte_nll_from_curves_rejects_non_monotone_exact() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 3,
        }];
        let nll = tte_nll_from_curves(&records, |_t| -0.5, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, SUBJECT_SENTINEL_NLL,
            "non-monotone CHZ on an exact event must be sentinel-guarded"
        );
    }

    /// Non-TTE records in the slice are skipped: a mixed slice gives the Event-only NLL.
    #[test]
    fn tte_nll_from_curves_skips_non_event_records() {
        let event =
            ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 };
        let cumhaz = |t: f64| 0.1 * t;
        let hazard = |_t: f64| 0.1;
        let event_only = tte_nll_from_curves(&[event.clone()], cumhaz, hazard, MonoTol::default());
        assert!(
            event_only.is_finite() && event_only < SUBJECT_SENTINEL_NLL,
            "sanity: a well-posed exact event gives a finite NLL"
        );
        let mixed = tte_nll_from_curves(
            &[
                event,
                ObsRecord::DiscreteState { time: 1.0, raw_time: 1.0, state: 0, cmt: 3 },
                ObsRecord::Count { time: 2.0, raw_time: 2.0, count: 4, cmt: 4 },
            ],
            cumhaz,
            hazard,
            MonoTol::default(),
        );
        assert_eq!(
            event_only, mixed,
            "discrete-state / count records must be skipped by the TTE NLL"
        );
    }

    /// Same coexistence contract for the RTTE clock-forward NLL: interleaved
    /// non-TTE records are skipped, so the mixed slice equals the Event-only NLL.
    #[test]
    fn rtte_forward_nll_from_curves_skips_non_event_records() {
        let events = [
            ObsRecord::Event { time: 2.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 },
            ObsRecord::Event {
                time: 5.0,
                event_type: EventType::RightCensored,
                entry_time: 0.0,
                cmt: 2,
            },
        ];
        let cumhaz = |t: f64| 0.1 * t;
        let hazard = |_t: f64| 0.1;
        let event_only = rtte_forward_nll_from_curves(&events, cumhaz, hazard, MonoTol::default());
        let mut mixed = events.to_vec();
        mixed.insert(1, ObsRecord::DiscreteState { time: 3.0, raw_time: 3.0, state: 1, cmt: 3 });
        let mixed_nll = rtte_forward_nll_from_curves(&mixed, cumhaz, hazard, MonoTol::default());
        assert_eq!(event_only, mixed_nll, "non-TTE records must be skipped");
    }

    /// Same coexistence contract for the RTTE clock-reset (gap-time) NLL.
    #[test]
    fn rtte_reset_nll_from_curves_skips_non_event_records() {
        let events = [
            ObsRecord::Event { time: 2.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 },
            ObsRecord::Event {
                time: 5.0,
                event_type: EventType::RightCensored,
                entry_time: 0.0,
                cmt: 2,
            },
        ];
        let cumhaz = |t: f64| 0.1 * t;
        let hazard = |_t: f64| 0.1;
        let event_only = rtte_reset_nll_from_curves(&events, cumhaz, hazard, MonoTol::default());
        let mut mixed = events.to_vec();
        mixed.insert(1, ObsRecord::Count { time: 3.0, raw_time: 3.0, count: 2, cmt: 4 });
        let mixed_nll = rtte_reset_nll_from_curves(&mixed, cumhaz, hazard, MonoTol::default());
        assert_eq!(event_only, mixed_nll, "non-TTE records must be skipped");
    }

    /// A tiny negative dip within the round-off floor is tolerated (finite NLL).
    #[test]
    fn tte_nll_from_curves_tolerates_flat_hazard_roundoff() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 3,
        }];
        // H(5) = −1e−9, below the 1e−6 absolute floor ⇒ accepted, not sentinel.
        let nll = tte_nll_from_curves(&records, |_t| -1e-9, |_t| 0.0, MonoTol::default());
        assert!(nll.abs() < 1e-6, "round-off dip must stay finite, got {nll}");
    }

    /// Interval-censored with `H(left) < H(entry)`: the interval arm's monotone branch.
    #[test]
    fn tte_nll_from_curves_rejects_non_monotone_interval() {
        let records = vec![ObsRecord::Event {
            time: 0.0,
            event_type: EventType::IntervalCensored { left: 5.0, right: 10.0 },
            entry_time: 2.0,
            cmt: 3,
        }];
        // H(entry=2) = 1.0 but H(left=5) = 0.2 < H(entry): non-monotone ⇒ sentinel.
        let nll = tte_nll_from_curves(
            &records,
            |t| if t < 3.0 { 1.0 } else { 0.2 },
            |_t| 0.1,
            MonoTol::default(),
        );
        assert_eq!(
            nll, SUBJECT_SENTINEL_NLL,
            "non-monotone CHZ before an interval must be sentinel-guarded"
        );
    }

    /// #618: a genuine negative increment of 0.05% of a large `H` (between `reltol` and
    /// the old `1e-3` floor) hits the sentinel.
    #[test]
    fn tte_nll_from_curves_rejects_genuine_negative_increment_large_h() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 2.0,
            cmt: 3,
        }];
        // H(entry=2) = 100, H(5) = 99.95 ⇒ ΔH = −0.05 = −0.05% of H. Old floor
        // 1e-3·100 = 0.1 accepted it; new floor reltol·100 = 0.01 rejects it.
        let cumhaz = |t: f64| if t <= 2.0 { 100.0 } else { 99.95 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, SUBJECT_SENTINEL_NLL,
            "genuine negative increment on a large H must be sentinel-guarded"
        );
    }

    /// And the exact-event arm: a positive instantaneous `h(T)` does not excuse a
    /// genuinely negative accumulated increment on a large `H` (#618).
    #[test]
    fn tte_nll_from_curves_exact_rejects_genuine_negative_increment_large_h() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 2.0,
            cmt: 3,
        }];
        let cumhaz = |t: f64| if t <= 2.0 { 100.0 } else { 99.95 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, SUBJECT_SENTINEL_NLL,
            "exact-event genuine negative increment must be sentinel-guarded"
        );
    }

    /// No over-rejection: ΔH = −0.005 on H = 100 (below the reltol·100 = 0.01 floor) is finite.
    #[test]
    fn tte_nll_from_curves_tolerates_solver_band_dip_large_h() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 2.0,
            cmt: 3,
        }];
        let cumhaz = |t: f64| if t <= 2.0 { 100.0 } else { 99.995 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert!(
            nll.is_finite() && nll < SUBJECT_SENTINEL_NLL,
            "within-band round-off dip must stay finite, got {nll}"
        );
    }

    /// [`MonoTol::analytic`] rejects a small negative increment the default floor accepts.
    #[test]
    fn tte_nll_from_curves_analytic_floor_rejects_small_dip() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 2.0,
            cmt: 3,
        }];
        // ΔH = −1e−5 on H ≈ 1: above the analytic ~1e−9 floor (rejected) but below the
        // default ODE floor 1e−6 + reltol·1 ≈ 1e−4 (which tolerates it as round-off).
        let cumhaz = |t: f64| if t <= 2.0 { 1.0 } else { 1.0 - 1e-5 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::analytic());
        assert_eq!(
            nll, SUBJECT_SENTINEL_NLL,
            "analytic floor must reject a small genuine negative dip"
        );
        let nll_default = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert!(
            nll_default.is_finite() && nll_default < SUBJECT_SENTINEL_NLL,
            "same dip is within the ODE-sized default floor, got {nll_default}"
        );
    }

    #[test]
    fn tte_data_term_exact_event_exponential() {
        use crate::types::HazardFamily;
        // lambda=0.1, T=10, exact event → -log L = H(T) - log h(T) = 1.0 - log(0.1) = 1.0 + 2.303
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let theta = &[0.1_f64];
        let eta = &[0.0_f64];
        let cov = HashMap::new();
        let nll = tte_data_term(&records, &hazard, TteRecurrence::Single, theta, eta, &cov);
        let expected = 0.1 * 10.0 - (0.1_f64).ln(); // H - log h
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-10);
    }

    #[test]
    fn tte_data_term_left_truncation() {
        use crate::types::HazardFamily;
        // Exponential, entry=5, T=10, right-censored → H(T)-H(entry) = 0.1*(10-5) = 0.5
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::RightCensored,
            entry_time: 5.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.1],
            &[0.0],
            &HashMap::new(),
        );
        assert_abs_diff_eq!(nll, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn tte_data_term_interval_censored_exponential() {
        use crate::types::HazardFamily;
        // Exponential, lambda=0.2, interval (3, 5), entry=0.
        // H(3)=0.6, H(5)=1.0, delta=0.4
        // log_prob = −0.6 + log(1 − exp(−0.4)) = −0.6 + log(0.32968...) = −1.70881...
        // nll = 1.70881...
        let records = vec![ObsRecord::Event {
            time: 5.0, // right bound (time field = right for interval-censored)
            event_type: EventType::IntervalCensored { left: 3.0, right: 5.0 },
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.2],
            &[0.0],
            &HashMap::new(),
        );
        let expected = -(-(0.6_f64) + ((-0.4_f64).exp_m1().abs().ln()));
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-10);
        assert!(nll > 0.0 && nll.is_finite());
    }

    #[test]
    fn tte_data_term_interval_censored_tight_interval() {
        use crate::types::HazardFamily;
        // Tight interval (10, 10.0001), λ=0.1: the expm1 form stays finite and positive.
        let records = vec![ObsRecord::Event {
            time: 10.0001,
            event_type: EventType::IntervalCensored { left: 10.0, right: 10.0001 },
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.1],
            &[0.0],
            &HashMap::new(),
        );
        // NLL must be finite and positive; exact value verified via log-domain formula.
        assert!(nll.is_finite() && nll > 0.0, "nll = {nll}");
        let delta = 0.1_f64 * 0.0001; // H(right) - H(left)
        let a = 0.1_f64 * 10.0; // H(left) - H(entry)
        let expected = a - ((-delta).exp_m1().abs().ln());
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-8);
    }

    // ── RTTE clock-forward: telescoping cumulative hazard ──

    // Three records for one subject at constant hazard h = λ = 0.05: events at
    // t = 5 and t = 10, administrative right-censor at t = 30. Curves are closed
    // form: H(t) = λt, h(t) = λ.
    fn rtte_constant_hazard_records() -> Vec<ObsRecord> {
        vec![
            ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 },
            ObsRecord::Event { time: 10.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 },
            ObsRecord::Event {
                time: 30.0,
                event_type: EventType::RightCensored,
                entry_time: 0.0,
                cmt: 2,
            },
        ]
    }

    #[test]
    fn rtte_forward_telescopes_cumulative_hazard() {
        // Clock-forward telescopes: −log L = 30λ − 2·log λ, not the 45λ − 2·log λ of
        // summing independent single-event terms.
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;

        let rtte =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_rtte = 30.0 * lambda - 2.0 * lambda.ln(); // H(T) − Σ log h
        assert_abs_diff_eq!(rtte, expected_rtte, epsilon = 1e-12);

        // Contrast: the single-event per-record accumulator over-counts by Σ H(t_k).
        let independent = tte_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_independent = 45.0 * lambda - 2.0 * lambda.ln();
        assert_abs_diff_eq!(independent, expected_independent, epsilon = 1e-12);
        assert!(
            (independent - rtte - 15.0 * lambda).abs() < 1e-12,
            "the over-count must be exactly Σ_k H(t_k) = 15λ (rtte={rtte}, indep={independent})"
        );
    }

    #[test]
    fn rtte_forward_single_event_matches_standard_tte() {
        // With exactly one event record, the recurrent and single-event likelihoods
        // coincide (nothing to telescope) — the RTTE path must not diverge for K=1.
        let lambda = 0.05_f64;
        let records = vec![ObsRecord::Event {
            time: 7.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;
        let rtte =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let single = tte_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_abs_diff_eq!(rtte, single, epsilon = 1e-12);
    }

    #[test]
    fn rtte_forward_left_truncation_uses_first_entry() {
        // Delayed entry at t = 2 on the first record: the lower integration limit is
        // H(entry), so −log L = H(T) − H(entry) − Σ log h = (30−2)λ − 2·log λ.
        let lambda = 0.05_f64;
        // Same event/censor times as `rtte_constant_hazard_records`, but the subject
        // enters the risk set at t = 2 (left truncation carried on the first record).
        let records = vec![
            ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 2.0, cmt: 2 },
            ObsRecord::Event { time: 10.0, event_type: EventType::Exact, entry_time: 2.0, cmt: 2 },
            ObsRecord::Event {
                time: 30.0,
                event_type: EventType::RightCensored,
                entry_time: 2.0,
                cmt: 2,
            },
        ];
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;
        let rtte =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected = (30.0 - 2.0) * lambda - 2.0 * lambda.ln();
        assert_abs_diff_eq!(rtte, expected, epsilon = 1e-12);
    }

    #[test]
    fn rtte_forward_rejects_non_monotone_hazard() {
        // A decreasing `H` folds into the 1e20 sentinel, as on the single-event path.
        let records = rtte_constant_hazard_records();
        // Cumulative hazard that decreases after t = 6 (negative hazard region).
        let cumhaz_at = |t: f64| if t < 6.0 { t } else { 12.0 - t };
        let hazard_at = |_t: f64| 0.1;
        let nll = rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_eq!(nll, SUBJECT_SENTINEL_NLL, "non-monotone CHZ must return the sentinel");
    }

    #[test]
    fn tte_data_term_dispatches_rtte_forward() {
        use crate::types::{HazardFamily, RtteClock, TteRecurrence};
        // `tte_data_term` with `Repeated { Forward }` matches the curve helper.
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let via_dispatch = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Repeated { clock: RtteClock::Forward },
            &[lambda],
            &[0.0],
            &HashMap::new(),
        );
        let expected = 30.0 * lambda - 2.0 * lambda.ln();
        assert_abs_diff_eq!(via_dispatch, expected, epsilon = 1e-12);
    }

    // ── RTTE clock-reset (gap time): renewal likelihood ──

    #[test]
    fn rtte_reset_equals_forward_for_exponential() {
        // A memoryless hazard gives the same likelihood on both clocks.
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;
        let reset = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let forward =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_abs_diff_eq!(reset, forward, epsilon = 1e-12);
        // And both equal the closed form 30λ − 2·log λ.
        assert_abs_diff_eq!(reset, 30.0 * lambda - 2.0 * lambda.ln(), epsilon = 1e-12);
    }

    #[test]
    fn rtte_reset_weibull_gap_time_closed_form() {
        // Weibull scale=10, shape=2: H(t) = (t/10)^2, h(t) = 0.2·(t/10). Reset gaps 5, 5, 20:
        //   2·(0.25 − log 0.1) + 4.0, which differs from clock-forward.
        let records = rtte_constant_hazard_records();
        let cumhaz_at = |t: f64| (t / 10.0).powi(2);
        let hazard_at = |t: f64| 0.2 * (t / 10.0);

        let reset = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_reset = 2.0 * (0.25 - 0.1_f64.ln()) + 4.0;
        assert_abs_diff_eq!(reset, expected_reset, epsilon = 1e-12);

        let forward =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_forward = 9.0 - 0.1_f64.ln() - 0.2_f64.ln(); // H(30) − log h(5) − log h(10)
        assert_abs_diff_eq!(forward, expected_forward, epsilon = 1e-12);
        assert!(
            (reset - forward).abs() > 1.0,
            "reset ({reset}) and forward ({forward}) must differ for a time-varying hazard"
        );
    }

    #[test]
    fn rtte_reset_left_truncation_conditions_on_entry() {
        // Convention B (#740): first sojourn H(t₁) − H(entry) − log h(t₁) in absolute time,
        // later gaps fresh renewals. Weibull (time-varying) separates B from convention A.
        let cumhaz_at = |t: f64| (t / 10.0).powi(2);
        let hazard_at = |t: f64| 0.2 * (t / 10.0);
        let entry = 8.0;
        // Enter the risk set at 8; events at 10 and 13; administrative censor at 20.
        let records = vec![
            ObsRecord::Event {
                time: 10.0,
                event_type: EventType::Exact,
                entry_time: entry,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 13.0,
                event_type: EventType::Exact,
                entry_time: entry,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 20.0,
                event_type: EventType::RightCensored,
                entry_time: entry,
                cmt: 2,
            },
        ];
        let reset = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());

        // B: first sojourn conditioned on entry (absolute clock), then renewal gaps 3 and 7.
        let expected_b = (cumhaz_at(10.0) - cumhaz_at(entry) - hazard_at(10.0).ln()) // first sojourn
            + (cumhaz_at(3.0) - hazard_at(3.0).ln())                                 // gap 13−10
            + cumhaz_at(7.0); // censor gap 20−13
        assert_abs_diff_eq!(reset, expected_b, epsilon = 1e-12);

        // Convention A (renewal restarting at entry) differs materially here.
        let convention_a = (cumhaz_at(10.0 - entry) - hazard_at(10.0 - entry).ln())
            + (cumhaz_at(3.0) - hazard_at(3.0).ln())
            + cumhaz_at(7.0);
        assert!(
            (reset - convention_a).abs() > 1.0,
            "reset NLL ({reset}) must use convention B, not the renewal-from-entry A ({convention_a})"
        );
    }

    #[test]
    fn rtte_reset_left_truncation_equals_forward_for_exponential() {
        // Memoryless hazard: convention B ≡ pure renewal ≡ clock-forward under delayed
        // entry. B vs A is pinned by `rtte_reset_left_truncation_conditions_on_entry`.
        let lambda = 0.05_f64;
        let entry = 4.0;
        let records = vec![
            ObsRecord::Event { time: 6.0, event_type: EventType::Exact, entry_time: entry, cmt: 2 },
            ObsRecord::Event {
                time: 12.0,
                event_type: EventType::Exact,
                entry_time: entry,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 30.0,
                event_type: EventType::RightCensored,
                entry_time: entry,
                cmt: 2,
            },
        ];
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;
        let reset = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let forward =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_abs_diff_eq!(reset, forward, epsilon = 1e-12);
        // Closed form: λ(T − entry) − 2·log λ = 0.05·(30 − 4) − 2·log(0.05).
        assert_abs_diff_eq!(reset, lambda * (30.0 - entry) - 2.0 * lambda.ln(), epsilon = 1e-12);
    }

    #[test]
    fn rtte_reset_left_truncation_censored_first_record() {
        // A late entrant censored before any event: H(t) − H(entry), the RightCensored
        // first-record path with a nonzero lower limit.
        let cumhaz_at = |t: f64| (t / 10.0).powi(2);
        let hazard_at = |t: f64| 0.2 * (t / 10.0);
        let entry = 5.0;
        let records = vec![ObsRecord::Event {
            time: 12.0,
            event_type: EventType::RightCensored,
            entry_time: entry,
            cmt: 2,
        }];
        let reset = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_abs_diff_eq!(reset, cumhaz_at(12.0) - cumhaz_at(entry), epsilon = 1e-12);
        let forward =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_abs_diff_eq!(reset, forward, epsilon = 1e-12);
    }

    #[test]
    fn tte_data_term_dispatches_rtte_reset() {
        use crate::types::{HazardFamily, RtteClock, TteRecurrence};
        // Route a repeated-event subject through `tte_data_term` with `Repeated { Reset }`
        // and an exponential family; must match the reset curve helper (dispatch wired).
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic { family: HazardFamily::Exponential, param_fn };
        let via_dispatch = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Repeated { clock: RtteClock::Reset },
            &[lambda],
            &[0.0],
            &HashMap::new(),
        );
        // Exponential ⇒ reset == forward == 30λ − 2·log λ.
        assert_abs_diff_eq!(via_dispatch, 30.0 * lambda - 2.0 * lambda.ln(), epsilon = 1e-12);
    }

    #[test]
    fn rtte_reset_folds_unsupported_inputs_to_sentinel() {
        // Every ill-posed input to the reset likelihood folds into the 1e20 sentinel:
        // Δ<0, h≤0, non-monotone H on either arm, interval.
        let exact = |t: f64| ObsRecord::Event {
            time: t,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        };
        let censor = |t: f64| ObsRecord::Event {
            time: t,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 2,
        };
        let tol = MonoTol::analytic();
        let lin = |t: f64| 0.1 * t; // proper increasing H(Δ)
        let good_h = |_t: f64| 0.1_f64;
        // Decreasing after t = 3, so H(Δ) < 0 on a gap of 5 (non-monotone / negative hazard).
        let dec = |t: f64| if t < 3.0 { t } else { 1.0 - t };

        // Non-positive hazard at an Exact event → sentinel (Exact arm, h ≤ 0).
        assert_eq!(
            rtte_reset_nll_from_curves(&[exact(5.0)], lin, |_t| 0.0, tol),
            SUBJECT_SENTINEL_NLL
        );
        // Non-monotone cumulative hazard on an Exact gap → sentinel (Exact arm).
        assert_eq!(
            rtte_reset_nll_from_curves(&[exact(5.0)], dec, good_h, tol),
            SUBJECT_SENTINEL_NLL
        );
        // Non-monotone cumulative hazard on a censored gap → sentinel (RightCensored arm).
        assert_eq!(
            rtte_reset_nll_from_curves(&[censor(5.0)], dec, good_h, tol),
            SUBJECT_SENTINEL_NLL
        );
        // Out-of-order records → negative gap → sentinel (Δ < 0 guard).
        assert_eq!(
            rtte_reset_nll_from_curves(&[exact(10.0), exact(4.0)], lin, good_h, tol),
            SUBJECT_SENTINEL_NLL
        );
        // Interval-censored record is unsupported for RTTE → sentinel.
        let interval = ObsRecord::Event {
            time: 5.0,
            event_type: EventType::IntervalCensored { left: 3.0, right: 5.0 },
            entry_time: 0.0,
            cmt: 2,
        };
        assert_eq!(rtte_reset_nll_from_curves(&[interval], lin, good_h, tol), SUBJECT_SENTINEL_NLL);
    }

    // ── draw_tte_outcome: administrative censoring at the observation window ──

    #[test]
    fn draw_tte_censoring_fraction_matches_survival() {
        use rand::SeedableRng;
        // Exponential λ=0.1 over a window τ=10 ⇒ P(censored) = S(τ) = exp(−λτ) ≈ 0.3679.
        // 20 000 draws; proportion SE ≈ 0.0034 ⇒ a 0.02 tolerance is ~6σ.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let lambda = 0.1_f64;
        let window = 10.0_f64;
        let n = 20_000;
        let mut censored = 0usize;
        for _ in 0..n {
            let (t, observed, _) =
                draw_tte_outcome(HazardFamily::Exponential, &[lambda], 0.0, window, &mut rng);
            if observed {
                assert!(t > 0.0 && t < window, "event time must lie in (0, window): {t}");
            } else {
                assert_eq!(t, window, "censored time must equal the window");
                censored += 1;
            }
        }
        let frac = censored as f64 / n as f64;
        let expected = (-lambda * window).exp();
        assert!(
            (frac - expected).abs() < 0.02,
            "censoring fraction {frac} should track S(τ) = {expected}"
        );
    }

    #[test]
    fn draw_tte_infinite_window_never_censors() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..1000 {
            let (t, observed, _) =
                draw_tte_outcome(HazardFamily::Exponential, &[1.0], 0.0, f64::INFINITY, &mut rng);
            assert!(observed, "no censoring is possible with an infinite window");
            assert!(t.is_finite() && t > 0.0, "event time must be finite positive: {t}");
        }
    }

    #[test]
    fn draw_tte_degenerate_zero_hazard_censors_at_window() {
        use rand::SeedableRng;
        // λ=0 ⇒ sampler returns f64::MAX; with a finite window the draw must
        // censor at the window rather than emit a spurious event at f64::MAX.
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let window = 12.5_f64;
        for _ in 0..100 {
            let (t, observed, _) =
                draw_tte_outcome(HazardFamily::Exponential, &[0.0], 0.0, window, &mut rng);
            assert!(!observed, "zero hazard can never produce an event");
            assert_eq!(t, window);
        }
    }

    #[test]
    fn draw_tte_left_truncation_respects_entry_and_window() {
        use rand::SeedableRng;
        // entry=5, window=8, λ=0.2 (memoryless): P(event before window)
        // = 1 − exp(−0.2·3) ≈ 0.45 ⇒ both outcomes appear in 5 000 draws.
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let (entry, window) = (5.0_f64, 8.0_f64);
        let (mut saw_event, mut saw_censor) = (false, false);
        for _ in 0..5000 {
            let (t, observed, _) =
                draw_tte_outcome(HazardFamily::Exponential, &[0.2], entry, window, &mut rng);
            if observed {
                assert!(t > entry && t < window, "conditional event in (entry, window): {t}");
                saw_event = true;
            } else {
                assert_eq!(t, window);
                saw_censor = true;
            }
        }
        assert!(saw_event && saw_censor, "expected a mix of events and censored draws");
    }

    #[test]
    fn draw_tte_unbounded_window_degenerate_does_not_emit_spurious_event() {
        use rand::SeedableRng;
        // λ=0 ⇒ f64::MAX; with an unbounded window it must still be censored.
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        for _ in 0..100 {
            let (t, observed, _) =
                draw_tte_outcome(HazardFamily::Exponential, &[0.0], 0.0, f64::INFINITY, &mut rng);
            assert!(
                !observed,
                "a degenerate (no-event) draw must never be reported as an observed event"
            );
            assert_eq!(t, f64::INFINITY, "censored at the (unbounded) window");
        }
    }

    // ── simulate_rtte_stream: repeated-event streams to the horizon (Slice 3.3) ──

    /// Run one RTTE stream and return its event times, asserting it ends with one censor
    /// row at the horizon and every earlier row is an observed event, increasing in
    /// `(0, horizon)`, on `cmt` with NaN ipred.
    fn rtte_stream_events<R: rand::Rng>(
        family: HazardFamily, params: &[f64], clock: RtteClock, horizon: f64, rng: &mut R,
    ) -> Vec<f64> {
        let mut results = Vec::new();
        let mut warnings = Vec::new();
        simulate_rtte_stream(
            family,
            params,
            clock,
            horizon,
            0.0,
            2,
            "1",
            0,
            1,
            rng,
            &mut results,
            &mut warnings,
        );
        assert!(!results.is_empty(), "a stream always emits the censor row");
        let last = results.last().unwrap();
        match last.outcome {
            SimOutcome::Event { time, observed } => {
                assert!(!observed, "the final row is the administrative censor");
                assert_eq!(time, horizon, "the censor row sits at the horizon");
            }
            _ => panic!("RTTE stream must emit Event outcomes"),
        }
        let mut prev = 0.0_f64;
        let mut events = Vec::new();
        for r in &results[..results.len() - 1] {
            match r.outcome {
                SimOutcome::Event { time, observed } => {
                    assert!(observed, "non-final rows are observed events");
                    assert!(
                        time > prev && time < horizon,
                        "events strictly sorted inside (0, horizon): {time} after {prev}"
                    );
                    prev = time;
                    events.push(time);
                }
                _ => panic!("RTTE stream must emit Event outcomes"),
            }
        }
        assert!(
            results.iter().all(|r| r.cmt == 2 && r.id == "1" && r.ipred.is_nan()),
            "cmt / id / NaN-ipred echoed on every row"
        );
        events
    }

    #[test]
    fn rtte_stream_degenerate_high_rate_warns_and_skips() {
        use rand::SeedableRng;
        // An extreme hazard is skipped by the pre-flight guard (#762): only the censor row
        // plus a warning naming the subject, no panic.
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let horizon = 30.0_f64;
        for clock in [RtteClock::Forward, RtteClock::Reset] {
            let mut results = Vec::new();
            let mut warnings = Vec::new();
            simulate_rtte_stream(
                HazardFamily::Exponential,
                &[1.0e7], // λ·horizon = 3e8 ≫ MAX_EVENTS
                clock,
                horizon,
                0.0,
                2,
                "S1",
                0,
                1,
                &mut rng,
                &mut results,
                &mut warnings,
            );
            assert_eq!(
                results.len(),
                1,
                "{clock:?}: degenerate stream is skipped → only the censor row"
            );
            match results[0].outcome {
                SimOutcome::Event { time, observed } => {
                    assert!(!observed && time == horizon, "{clock:?}: censored at horizon");
                }
                _ => panic!("expected an Event outcome"),
            }
            assert_eq!(warnings.len(), 1, "{clock:?}: exactly one degenerate-stream warning");
            assert!(
                warnings[0].contains("W_RTTE_DEGENERATE")
                    && warnings[0].contains("S1")
                    && warnings[0].contains("#762"),
                "{clock:?}: warning must name the subject and issue: {}",
                warnings[0]
            );
        }
    }

    #[test]
    fn rtte_reset_ifr_weibull_not_skipped_despite_large_forward_cum_hazard() {
        use rand::SeedableRng;
        // #772: under the reset clock an IFR Weibull expects a few dozen events, though
        // `H(100) = (100/5.49)^5 ≈ 2e6 ≫ MAX_EVENTS`; the clock-specific pre-flight runs it.
        let params = [5.49_f64, 5.0]; // Weibull [scale, shape], strongly increasing hazard.
        let horizon = 100.0_f64;

        // Reset: a real multi-event renewal stream, and NO degenerate warning.
        let mut rng = rand::rngs::StdRng::seed_from_u64(20);
        let mut results = Vec::new();
        let mut warnings = Vec::new();
        simulate_rtte_stream(
            HazardFamily::Weibull,
            &params,
            RtteClock::Reset,
            horizon,
            0.0,
            2,
            "S1",
            0,
            1,
            &mut rng,
            &mut results,
            &mut warnings,
        );
        let observed = results
            .iter()
            .filter(|r| matches!(r.outcome, SimOutcome::Event { observed: true, .. }))
            .count();
        assert!(
            observed >= 5,
            "reset IFR Weibull renews ~horizon/mean_gap ≈ 20 events — not skipped (got {observed})"
        );
        assert!(
            warnings.is_empty(),
            "reset IFR Weibull is a legitimate subject, no degenerate warning: {warnings:?}"
        );

        // The same params under the forward clock (`E[N] ≈ 2e6`) are still skipped.
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(20);
        let mut fwd_results = Vec::new();
        let mut fwd_warnings = Vec::new();
        simulate_rtte_stream(
            HazardFamily::Weibull,
            &params,
            RtteClock::Forward,
            horizon,
            0.0,
            2,
            "S1",
            0,
            1,
            &mut rng2,
            &mut fwd_results,
            &mut fwd_warnings,
        );
        assert_eq!(
            fwd_results.len(),
            1,
            "forward H(horizon)≈2e6 is degenerate → skipped to the single censor row"
        );
        assert!(
            fwd_warnings.len() == 1 && fwd_warnings[0].contains("W_RTTE_DEGENERATE"),
            "forward clock still warns degenerate for the same params: {fwd_warnings:?}"
        );
    }

    #[test]
    fn draw_tte_outcome_flags_degenerate_rate() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        // λ = 0 ⇒ the sampler returns the f64::MAX sentinel ⇒ censored *and* flagged
        // degenerate (the third return element, #763).
        let (t, observed, degenerate) =
            draw_tte_outcome(HazardFamily::Exponential, &[0.0], 0.0, 10.0, &mut rng);
        assert!(!observed && degenerate, "a λ=0 draw is a degenerate censor");
        assert_eq!(t, 10.0, "censored at the window");
        // A positive-rate draw is never flagged degenerate — whether it fires an event
        // or is a genuine late censor.
        for _ in 0..200 {
            let (_, _observed, degenerate) =
                draw_tte_outcome(HazardFamily::Exponential, &[0.1], 0.0, 10.0, &mut rng);
            assert!(!degenerate, "a positive-rate draw is never degenerate");
        }
    }

    #[test]
    fn rtte_forward_exponential_count_is_poisson() {
        use rand::SeedableRng;
        // Homogeneous Poisson: λ=0.15, T=20 ⇒ mean = var = 3.0; N=4000 ⇒ SE ≈ 0.027, so
        // |mean − 3| < 0.15 is ~5σ, and var/mean ≈ 1.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0001);
        let (lambda, horizon, n) = (0.15_f64, 20.0_f64, 4000usize);
        let counts: Vec<f64> = (0..n)
            .map(|_| {
                rtte_stream_events(
                    HazardFamily::Exponential,
                    &[lambda],
                    RtteClock::Forward,
                    horizon,
                    &mut rng,
                )
                .len() as f64
            })
            .collect();
        let mean = counts.iter().sum::<f64>() / n as f64;
        let var = counts.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n as f64;
        let expected = lambda * horizon;
        assert!((mean - expected).abs() < 0.15, "mean events {mean} should track λT = {expected}");
        assert!(
            (var / mean - 1.0).abs() < 0.15,
            "count dispersion var/mean = {} should be ≈ 1 (Poisson)",
            var / mean
        );
    }

    #[test]
    fn rtte_forward_weibull_count_is_poisson() {
        use rand::SeedableRng;
        // Non-homogeneous Poisson: N(T) ~ Poisson((T/scale)^shape), scale=10, shape=1.5,
        // T=20 ⇒ H ≈ 2.828; pins the conditional sampling.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0002);
        let (scale, shape, horizon, n) = (10.0_f64, 1.5_f64, 20.0_f64, 5000usize);
        let mean = (0..n)
            .map(|_| {
                rtte_stream_events(
                    HazardFamily::Weibull,
                    &[scale, shape],
                    RtteClock::Forward,
                    horizon,
                    &mut rng,
                )
                .len() as f64
            })
            .sum::<f64>()
            / n as f64;
        let expected = (horizon / scale).powf(shape);
        assert!(
            (mean - expected).abs() < 0.15,
            "mean events {mean} should track H(T) = (T/scale)^shape = {expected}"
        );
    }

    #[test]
    fn rtte_forward_equals_reset_for_exponential() {
        use rand::SeedableRng;
        // Exponential is memoryless ⇒ both clocks draw a bit-identical stream.
        let (lambda, horizon) = (0.3_f64, 25.0_f64);
        let mut r_fwd = rand::rngs::StdRng::seed_from_u64(0xABCD_1234);
        let mut r_rst = rand::rngs::StdRng::seed_from_u64(0xABCD_1234);
        let fwd = rtte_stream_events(
            HazardFamily::Exponential,
            &[lambda],
            RtteClock::Forward,
            horizon,
            &mut r_fwd,
        );
        let rst = rtte_stream_events(
            HazardFamily::Exponential,
            &[lambda],
            RtteClock::Reset,
            horizon,
            &mut r_rst,
        );
        assert_eq!(fwd, rst, "exponential forward == reset (memoryless)");
        assert!(!fwd.is_empty(), "expected events for λT = {}", lambda * horizon);
    }

    #[test]
    fn rtte_forward_differs_from_reset_for_weibull() {
        use rand::SeedableRng;
        // Weibull is not memoryless ⇒ absolute-time (forward) and gap-time (reset)
        // streams share only their first event and then diverge from the same seed.
        let (scale, shape, horizon) = (4.0_f64, 2.0_f64, 40.0_f64);
        let mut r_fwd = rand::rngs::StdRng::seed_from_u64(0x0011_2233);
        let mut r_rst = rand::rngs::StdRng::seed_from_u64(0x0011_2233);
        let fwd = rtte_stream_events(
            HazardFamily::Weibull,
            &[scale, shape],
            RtteClock::Forward,
            horizon,
            &mut r_fwd,
        );
        let rst = rtte_stream_events(
            HazardFamily::Weibull,
            &[scale, shape],
            RtteClock::Reset,
            horizon,
            &mut r_rst,
        );
        assert!(
            fwd.len() >= 2 && rst.len() >= 2,
            "need multiple events to compare clocks (fwd={}, rst={})",
            fwd.len(),
            rst.len()
        );
        assert!(
            (fwd[0] - rst[0]).abs() < 1e-12,
            "the first event is a fresh draw from 0 either way"
        );
        assert_ne!(fwd, rst, "Weibull forward and reset must diverge after the first event");
    }

    #[test]
    fn rtte_reset_weibull_gaps_match_family_mean() {
        use rand::SeedableRng;
        // Reset: iid Weibull gaps with mean scale·Γ(1.5) = scale·√π/2; horizon ≫ mean gap.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFACE_0FF0);
        let (scale, shape, horizon) = (3.0_f64, 2.0_f64, 80.0_f64);
        let mut gaps = Vec::new();
        for _ in 0..3000 {
            let ev = rtte_stream_events(
                HazardFamily::Weibull,
                &[scale, shape],
                RtteClock::Reset,
                horizon,
                &mut rng,
            );
            let mut prev = 0.0_f64;
            for &t in &ev {
                gaps.push(t - prev);
                prev = t;
            }
        }
        let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let expected = scale * std::f64::consts::PI.sqrt() / 2.0; // scale·Γ(1.5)
        assert!(
            (mean_gap - expected).abs() < 0.08,
            "mean reset gap {mean_gap} should track scale·Γ(1+1/shape) = {expected} \
             (n = {} gaps)",
            gaps.len()
        );
    }

    #[test]
    fn rtte_stream_is_deterministic_for_seed() {
        use rand::SeedableRng;
        let run = || {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            rtte_stream_events(
                HazardFamily::Weibull,
                &[5.0, 1.3],
                RtteClock::Forward,
                50.0,
                &mut rng,
            )
        };
        assert_eq!(run(), run(), "same seed ⇒ identical stream");
    }

    #[test]
    fn rtte_stream_zero_hazard_emits_only_censor_row() {
        use rand::SeedableRng;
        // λ=0 ⇒ every draw is the degenerate f64::MAX sentinel ⇒ no events, just the
        // trailing censor at the horizon. Must not hang or panic, on either clock.
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let horizon = 15.0_f64;
        for clock in [RtteClock::Forward, RtteClock::Reset] {
            let mut results = Vec::new();
            let mut warnings = Vec::new();
            simulate_rtte_stream(
                HazardFamily::Exponential,
                &[0.0],
                clock,
                horizon,
                0.0,
                2,
                "s",
                0,
                1,
                &mut rng,
                &mut results,
                &mut warnings,
            );
            assert_eq!(results.len(), 1, "{clock:?}: only the censor row");
            match results[0].outcome {
                SimOutcome::Event { time, observed } => {
                    assert_eq!(time, horizon);
                    assert!(!observed);
                }
                _ => panic!("expected an Event outcome"),
            }
        }
    }

    // ── simulate_rtte_stream: left truncation (delayed entry, #740) ──

    /// Observed event times of one RTTE stream with a given left-truncation `entry`
    /// (excludes the trailing administrative-censor row).
    fn collect_rtte_events<R: rand::Rng>(
        family: HazardFamily, params: &[f64], clock: RtteClock, horizon: f64, entry: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        let mut results = Vec::new();
        let mut warnings = Vec::new();
        simulate_rtte_stream(
            family,
            params,
            clock,
            horizon,
            entry,
            2,
            "lt",
            0,
            1,
            rng,
            &mut results,
            &mut warnings,
        );
        results
            .iter()
            .filter_map(|r| match r.outcome {
                SimOutcome::Event { time, observed } if observed => Some(time),
                _ => None,
            })
            .collect()
    }

    /// Left truncation on an exponential hazard is an exact time shift by `entry` under
    /// the same seed (convention B, #740), on both clocks.
    #[test]
    fn rtte_left_truncation_exponential_is_entry_shift() {
        use rand::SeedableRng;
        let lambda = 0.03_f64;
        let entry = 12.0_f64;
        let horizon = 200.0_f64;
        for clock in [RtteClock::Forward, RtteClock::Reset] {
            let mut r_e = rand::rngs::StdRng::seed_from_u64(0x5117_7740);
            let mut r_0 = rand::rngs::StdRng::seed_from_u64(0x5117_7740);
            let events_e = collect_rtte_events(
                HazardFamily::Exponential,
                &[lambda],
                clock,
                horizon,
                entry,
                &mut r_e,
            );
            let events_0 = collect_rtte_events(
                HazardFamily::Exponential,
                &[lambda],
                clock,
                horizon,
                0.0,
                &mut r_0,
            );
            assert!(
                !events_e.is_empty(),
                "{clock:?}: a left-truncated stream should still produce events before the horizon"
            );
            // The non-truncated stream isn't shifted toward the horizon, so it has at least as
            // many events; each entry-conditioned event equals its counterpart plus `entry`.
            assert!(events_0.len() >= events_e.len());
            for (k, &t_e) in events_e.iter().enumerate() {
                assert_abs_diff_eq!(t_e, events_0[k] + entry, epsilon = 1e-9);
                assert!(t_e >= entry, "{clock:?}: no simulated event may precede entry");
            }
        }
    }

    /// Clock-reset left truncation on a Weibull hazard: the first sojourn lands past
    /// `entry`, the stream differs from the untruncated one, and its hand-recomputed
    /// convention-B likelihood matches `rtte_reset_nll_from_curves`.
    #[test]
    fn rtte_reset_left_truncation_weibull_conditions_on_entry() {
        use rand::SeedableRng;
        let params = [40.0_f64, 1.6_f64]; // Weibull scale=40, shape=1.6 (increasing hazard)
        let entry = 15.0_f64;
        let horizon = 300.0_f64;
        let mut r_e = rand::rngs::StdRng::seed_from_u64(0xB17E_5EED);
        let mut r_0 = rand::rngs::StdRng::seed_from_u64(0xB17E_5EED);
        let events_e = collect_rtte_events(
            HazardFamily::Weibull,
            &params,
            RtteClock::Reset,
            horizon,
            entry,
            &mut r_e,
        );
        let events_0 = collect_rtte_events(
            HazardFamily::Weibull,
            &params,
            RtteClock::Reset,
            horizon,
            0.0,
            &mut r_0,
        );
        assert!(!events_e.is_empty(), "expected at least one event before the horizon");
        assert!(
            events_e[0] >= entry,
            "first sojourn must be conditioned past entry: {} < {entry}",
            events_e[0]
        );
        // A time-varying hazard means conditioning on entry is not a shift, so the streams differ.
        assert!(
            events_e != events_0,
            "entry conditioning must change a time-varying-hazard stream"
        );
        // Round-trip: the realized stream is a valid convention-B fit input, and its fit NLL
        // equals the hand-recomposed convention-B objective on the same event times.
        let cumhaz_at = |t: f64| cum_hazard(HazardFamily::Weibull, t, &params);
        let hazard_at = |t: f64| hazard_and_cum_hazard(HazardFamily::Weibull, t, &params).0;
        let mut records: Vec<ObsRecord> = events_e
            .iter()
            .map(|&t| ObsRecord::Event {
                time: t,
                event_type: EventType::Exact,
                entry_time: entry,
                cmt: 2,
            })
            .collect();
        records.push(ObsRecord::Event {
            time: horizon,
            event_type: EventType::RightCensored,
            entry_time: entry,
            cmt: 2,
        });
        let nll = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let mut expected = cumhaz_at(events_e[0]) - cumhaz_at(entry) - hazard_at(events_e[0]).ln();
        for w in events_e.windows(2) {
            let gap = w[1] - w[0];
            expected += cumhaz_at(gap) - hazard_at(gap).ln();
        }
        expected += cumhaz_at(horizon - *events_e.last().unwrap());
        assert!(
            nll.is_finite() && nll < SUBJECT_SENTINEL_NLL,
            "left-truncated stream must fit, got {nll}"
        );
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-9);
    }

    /// On both clocks the first sojourn under delayed entry equals convention B's
    /// `sample_conditional_event_time(entry, u₁)`, not A's `entry + sample_event_time(u₁)`
    /// (Weibull, since A ≡ B for an exponential).
    #[test]
    fn rtte_left_truncation_first_sojourn_is_convention_b_not_a() {
        use rand::SeedableRng;
        let params = [40.0_f64, 1.6_f64]; // Weibull scale=40, shape=1.6 (time-varying)
        let entry = 15.0_f64;
        let horizon = 300.0_f64;
        let seed = 0xB17E_5EED_u64;
        for clock in [RtteClock::Forward, RtteClock::Reset] {
            // The exact first uniform the stream will consume (same seed + same Open01 draw).
            let mut probe = rand::rngs::StdRng::seed_from_u64(seed);
            let u1: f64 = probe.sample(rand::distr::Open01);
            let b = sample_conditional_event_time(HazardFamily::Weibull, &params, entry, u1);
            let a = entry + sample_event_time(HazardFamily::Weibull, &params, u1);
            assert!(b < horizon, "setup: first B event must land before the horizon");
            assert!(
                (a - b).abs() > 0.5,
                "setup: A ({a}) and B ({b}) must differ materially for a time-varying hazard"
            );
            let mut stream = rand::rngs::StdRng::seed_from_u64(seed);
            let events = collect_rtte_events(
                HazardFamily::Weibull,
                &params,
                clock,
                horizon,
                entry,
                &mut stream,
            );
            assert!(!events.is_empty(), "{clock:?}: expected a first event");
            assert_abs_diff_eq!(events[0], b, epsilon = 1e-12);
            assert!(
                (events[0] - a).abs() > 0.5,
                "{clock:?}: first sojourn must be convention B ({b}), not renewal-from-entry A ({a}); got {}",
                events[0]
            );
        }
    }

    /// A negative `entry` (a hand-built population) behaves exactly as `entry == 0` on both
    /// clocks: no pre-origin clock, no negative-time event.
    #[test]
    fn rtte_left_truncation_negative_entry_is_no_truncation() {
        use rand::SeedableRng;
        let params = [40.0_f64, 1.6_f64];
        let horizon = 300.0_f64;
        for clock in [RtteClock::Forward, RtteClock::Reset] {
            let mut r_neg = rand::rngs::StdRng::seed_from_u64(0x0000_DEAD);
            let mut r_zero = rand::rngs::StdRng::seed_from_u64(0x0000_DEAD);
            let neg = collect_rtte_events(
                HazardFamily::Weibull,
                &params,
                clock,
                horizon,
                -5.0,
                &mut r_neg,
            );
            let zero = collect_rtte_events(
                HazardFamily::Weibull,
                &params,
                clock,
                horizon,
                0.0,
                &mut r_zero,
            );
            assert_eq!(
                neg, zero,
                "{clock:?}: a negative entry must produce exactly the no-truncation stream"
            );
            assert!(
                neg.iter().all(|&t| t > 0.0),
                "{clock:?}: no simulated event may land at or before the origin"
            );
        }
    }

    // ── resolve_competing_risks: earliest cause wins, else administrative censor ──

    #[test]
    fn resolve_competing_risks_earliest_wins() {
        // Cause 1 (index 1) is earliest at t=3 < window=10 ⇒ observed event.
        let (idx, t, event) = resolve_competing_risks(&[5.0, 3.0, 8.0], 10.0);
        assert_eq!(idx, 1);
        assert_eq!(t, 3.0);
        assert!(event);
    }

    #[test]
    fn resolve_competing_risks_all_after_window_censors() {
        // No cause fires before the window ⇒ censored at the window, no event.
        let (idx, t, event) = resolve_competing_risks(&[12.0, 15.0], 10.0);
        assert_eq!(idx, 0, "winner is still the earliest latent");
        assert_eq!(t, 10.0);
        assert!(!event);
    }

    #[test]
    fn resolve_competing_risks_max_sentinel_never_wins() {
        // f64::MAX (degenerate cause) must not win over a real event.
        let (idx, t, event) = resolve_competing_risks(&[f64::MAX, 4.0], 10.0);
        assert_eq!(idx, 1);
        assert_eq!(t, 4.0);
        assert!(event);
    }

    #[test]
    fn resolve_competing_risks_tie_picks_lowest_index() {
        let (idx, t, event) = resolve_competing_risks(&[3.0, 3.0], 10.0);
        assert_eq!(idx, 0, "ties resolve to the lowest cause index");
        assert_eq!(t, 3.0);
        assert!(event);
    }

    // ── cif_curves: cause-specific cumulative incidence ──────────────────────

    #[test]
    fn cif_two_cause_exponential_closed_form() {
        // Constant hazards λ1=0.1, λ2=0.3 ⇒ closed form (grid-independent):
        //   F_k(t) = (λ_k/Σλ)·(1 − exp(−Σλ·t)),  S_all(t) = exp(−Σλ·t).
        let (l1, l2) = (0.1_f64, 0.3_f64);
        let grid = [0.0, 2.0, 5.0, 13.0, 40.0];
        let cum: Vec<Vec<f64>> =
            vec![grid.iter().map(|&t| l1 * t).collect(), grid.iter().map(|&t| l2 * t).collect()];

        let (cif, s_all) = cif_curves(&cum);
        let lsum = l1 + l2;
        for (i, &t) in grid.iter().enumerate() {
            let s_ref = (-lsum * t).exp();
            assert_abs_diff_eq!(s_all[i], s_ref, epsilon = 1e-12);
            assert_abs_diff_eq!(cif[0][i], (l1 / lsum) * (1.0 - s_ref), epsilon = 1e-12);
            assert_abs_diff_eq!(cif[1][i], (l2 / lsum) * (1.0 - s_ref), epsilon = 1e-12);
            // Invariant must hold exactly at every grid point.
            assert_abs_diff_eq!(cif[0][i] + cif[1][i] + s_all[i], 1.0, epsilon = 1e-12);
        }
        // t = 0 ⇒ no incidence, full survival.
        assert_eq!(cif[0][0], 0.0);
        assert_eq!(cif[1][0], 0.0);
        assert_eq!(s_all[0], 1.0);
    }

    #[test]
    fn cif_invariant_holds_for_mixed_families() {
        // On a coarse irregular grid Σ_k F_k + S_all = 1 must still be exact.
        let grid = [1.0_f64, 2.5, 4.0, 9.0, 20.0];
        // cause 0: Weibull scale=8, shape=1.7 ⇒ H=(t/8)^1.7
        let (sc, sh) = (8.0_f64, 1.7_f64);
        let cum0: Vec<f64> = grid.iter().map(|&t| (t / sc).powf(sh)).collect();
        // cause 1: constant λ=0.05 ⇒ H=0.05·t
        let cum1: Vec<f64> = grid.iter().map(|&t| 0.05 * t).collect();

        let (cif, s_all) = cif_curves(&[cum0, cum1]);
        for i in 0..grid.len() {
            assert_abs_diff_eq!(cif[0][i] + cif[1][i] + s_all[i], 1.0, epsilon = 1e-12);
            assert!(cif[0][i] >= 0.0 && cif[1][i] >= 0.0, "CIF non-negative");
            if i > 0 {
                assert!(cif[0][i] >= cif[0][i - 1] - 1e-15, "CIF non-decreasing");
                assert!(cif[1][i] >= cif[1][i - 1] - 1e-15, "CIF non-decreasing");
            }
        }
    }

    #[test]
    fn cif_time_varying_matches_independent_quadrature() {
        // Time-varying CIF (Weibull + Gompertz) against an independent composite-Simpson
        // integration of F_k(t) = ∫₀ᵗ h_k(u)·S_all(u) du, S_all(u) = exp(−Σ_j H_j(u)).
        let (sc, sh) = (10.0_f64, 1.6_f64); // cause 0: Weibull  H=(t/sc)^sh
        let (alpha, gamma) = (0.03_f64, 0.08_f64); // cause 1: Gompertz H=(α/γ)(e^{γt}−1)
        let h0 = |t: f64| (sh / sc) * (t / sc).powf(sh - 1.0);
        let cumh0 = |t: f64| (t / sc).powf(sh);
        let h1 = |t: f64| alpha * (gamma * t).exp();
        let cumh1 = |t: f64| (alpha / gamma) * ((gamma * t).exp() - 1.0);
        let s_all = |t: f64| (-(cumh0(t) + cumh1(t))).exp();

        // Independent reference: composite Simpson (N even) of ∫₀ᵗ h_k·S_all.
        let ref_cif = |hk: &dyn Fn(f64) -> f64, t: f64| -> f64 {
            let n = 60_000usize;
            let dt = t / n as f64;
            let mut acc = 0.0;
            for i in 0..=n {
                let u = i as f64 * dt;
                let w = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                acc += w * hk(u) * s_all(u);
            }
            acc * dt / 3.0
        };

        // `cif_curves` on a fine grid over [0, 15] (dt = 0.0025).
        let n_grid = 6000usize;
        let grid: Vec<f64> = (0..=n_grid).map(|i| i as f64 * 15.0 / n_grid as f64).collect();
        let chz = vec![
            grid.iter().map(|&t| cumh0(t)).collect::<Vec<_>>(),
            grid.iter().map(|&t| cumh1(t)).collect::<Vec<_>>(),
        ];
        let (cif, s) = cif_curves(&chz);

        // Check at five interior/boundary points (exact grid nodes).
        for &idx in &[1200usize, 2400, 3600, 4800, 6000] {
            let t = grid[idx];
            assert_abs_diff_eq!(s[idx], s_all(t), epsilon = 1e-12);
            assert_abs_diff_eq!(cif[0][idx], ref_cif(&h0, t), epsilon = 1e-3);
            assert_abs_diff_eq!(cif[1][idx], ref_cif(&h1, t), epsilon = 1e-3);
            // Sanity: still a valid partition.
            assert_abs_diff_eq!(cif[0][idx] + cif[1][idx] + s[idx], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn observation_window_only_right_censored_has_a_horizon() {
        // A right-censored record's `time` IS the administrative horizon.
        assert_eq!(observation_window(&EventType::RightCensored, 12.0), 12.0);
        // Event records carry no horizon: they draw uncensored (+∞).
        assert_eq!(observation_window(&EventType::Exact, 12.0), f64::INFINITY);
        assert_eq!(
            observation_window(&EventType::IntervalCensored { left: 1.0, right: 5.0 }, 5.0),
            f64::INFINITY
        );
    }

    #[test]
    fn cif_single_cause_is_one_minus_survival() {
        // One cause ⇒ CIF = 1 − S_all = 1 − S, invariant trivially holds.
        let grid = [0.0, 1.0, 4.0, 12.0];
        let lam = 0.2_f64;
        let cum = vec![grid.iter().map(|&t| lam * t).collect::<Vec<_>>()];
        let (cif, s_all) = cif_curves(&cum);
        for i in 0..grid.len() {
            assert_abs_diff_eq!(cif[0][i], 1.0 - s_all[i], epsilon = 1e-12);
        }
    }

    /// The solver's `SolveFailed` reasons, one per producer class, as the labelled CHZ
    /// driver words them (`ode/solver.rs` `until_threshold_with` / `with_nonfinite_state_note`
    /// with the labels of `ode/predictions.rs` `chz_threshold_state_labels`), each with the
    /// diagnosis sentence and the advice the warning must carry for it.
    #[cfg(feature = "survival")]
    const SOLVE_FAILED_REASONS: [(&str, Diagnosis, &str); 8] = [
        (
            "[odes] state `blowup` became non-finite before t=2.500017; step budget (10000) \
             exhausted before reaching t_end=30.000000 (reached t=2.500017)",
            Diagnosis::OdesDiverged,
            ODES,
        ),
        ("the cumulative hazard became non-finite at t=2.000000", Diagnosis::HazardNonFinite, BOTH),
        (
            "the cumulative hazard decreased (0.500000 → 0.400000) over [1.000000, 1.100000]: \
             non-monotone accumulator (negative rate / hazard)",
            Diagnosis::HazardNegative,
            HAZARD,
        ),
        (
            "[odes] state `blowup`, the cumulative hazard of CMT=3 became non-finite before \
             t=2.000000; step budget (10000) exhausted before reaching t_end=30.000000 \
             (reached t=2.000000)",
            Diagnosis::Both,
            BOTH,
        ),
        (
            "step budget (10000) exhausted before reaching t_end=30.000000 (reached t=4.000000)",
            Diagnosis::None,
            BOTH,
        ),
        (
            "aborted at t=1.000000 after 3 min-dt clamp(s) (ode_stiff_abort_after): the segment \
             is stability-limited",
            Diagnosis::None,
            BOTH,
        ),
        (
            "the step at t=1.000000 could not be formed even at min_dt (singular Jacobian system)",
            Diagnosis::None,
            BOTH,
        ),
        ("non-finite break time", Diagnosis::None, BOTH),
    ];

    /// Which diagnosis sentences a reason class must produce.
    #[cfg(feature = "survival")]
    #[derive(Clone, Copy, Debug)]
    enum Diagnosis {
        OdesDiverged,
        HazardNonFinite,
        HazardNegative,
        Both,
        None,
    }

    #[cfg(feature = "survival")]
    const ODES: &str = "Check the [odes] right-hand side and the parameter values, including \
                        the drawn random effects.";
    #[cfg(feature = "survival")]
    const HAZARD: &str = "Check the [event_model] hazard expression and the parameter values, \
                          including the drawn random effects.";
    #[cfg(feature = "survival")]
    const BOTH: &str = "Check the [odes] right-hand side, the [event_model] hazard expression \
                        and the parameter values, including the drawn random effects.";

    /// #2087: every producer class gets the code, subject, CMT, reason, censoring and
    /// population sentences, and exactly its own diagnosis sentences and whole advice
    /// sentence; each diagnosis is asserted present and absent elsewhere.
    #[cfg(feature = "survival")]
    #[test]
    fn ode_solve_failed_warning_covers_every_reason_class() {
        const ODES_SENTENCE: &str =
            "The [odes] right-hand side diverged for this subject's parameter draw.";
        const NONFINITE_SENTENCE: &str =
            "The [event_model] hazard went non-finite for this subject's parameter draw.";
        const NEGATIVE_SENTENCE: &str =
            "The [event_model] hazard went negative for this subject's parameter draw.";
        for (why, diagnosis, advice) in SOLVE_FAILED_REASONS {
            let w = ode_solve_failed_warning("7", 2, why, 30.0, false);
            assert!(w.starts_with("W_TTE_SOLVE_FAILED: subject '7' (CMT=2): "), "{w}");
            assert!(w.contains(&format!("failed: {why}.")), "{w}");
            let (odes, nonfinite, negative) = match diagnosis {
                Diagnosis::OdesDiverged => (true, false, false),
                Diagnosis::HazardNonFinite => (false, true, false),
                Diagnosis::HazardNegative => (false, false, true),
                Diagnosis::Both => (true, true, false),
                Diagnosis::None => (false, false, false),
            };
            assert_eq!(w.contains(ODES_SENTENCE), odes, "{diagnosis:?}: {w}");
            assert_eq!(w.contains(NONFINITE_SENTENCE), nonfinite, "{diagnosis:?}: {w}");
            assert_eq!(w.contains(NEGATIVE_SENTENCE), negative, "{diagnosis:?}: {w}");
            assert!(w.ends_with(advice), "{diagnosis:?} advice: {w}");
            assert!(
                w.contains(
                    "No event time could be simulated, so the subject is censored at the \
                     observation window (t=30) and is not simulated past the failure."
                ),
                "{w}"
            );
            assert!(w.contains("The rest of the population still simulates."), "{w}");
            assert!(!w.contains("degenerate"), "{w}");
            assert!(!w.contains("__chz"), "{w}");
            // No bare 0-based state index: every "state " is followed by a label, not a digit.
            for (i, _) in w.match_indices("state ") {
                let next = w[i + "state ".len()..].chars().next();
                assert!(!next.is_some_and(|c| c.is_ascii_digit()), "{w}");
            }
        }
    }

    /// #2087: only the competing-risks wording says every cause is censored.
    #[cfg(feature = "survival")]
    #[test]
    fn ode_solve_failed_warning_competing_wording_straddles_the_gate() {
        let (why, ..) = SOLVE_FAILED_REASONS[0];
        let cr = ode_solve_failed_warning("1", 3, why, 12.5, true);
        let single = ode_solve_failed_warning("1", 3, why, 12.5, false);
        assert!(
            cr.contains(
                "so every cause of the subject is censored at the observation window (t=12.5)"
            ),
            "{cr}"
        );
        assert!(!single.contains("every cause"), "{single}");
        assert!(
            single.contains("so the subject is censored at the observation window (t=12.5)"),
            "{single}"
        );
    }

    /// #2087: every reason class classifies as a `simulation` warning, not the category
    /// its interpolated solver text would otherwise match.
    #[cfg(feature = "survival")]
    #[test]
    fn ode_solve_failed_warning_classifies_as_simulation() {
        for competing in [false, true] {
            for (why, ..) in SOLVE_FAILED_REASONS {
                let w = ode_solve_failed_warning("7", 2, why, 30.0, competing);
                let e = crate::types::classify_warning(&w);
                assert_eq!(e.category.as_str(), "simulation", "{w}");
                assert_eq!(e.severity, crate::types::WarningSeverity::Warning, "{w}");
            }
        }
    }
}

#[cfg(test)]
mod exact_eta_hessian_tests {
    use super::*;
    use crate::types::{EndpointLikelihood, HazardSpec};

    /// The exponential family in closed form: with `λ = θ·e^η`, `−log L = Σ λ t_i − Σ_exact
    /// (log λ)` (no truncation in the fixture), so `∂²/∂η² = λ Σ t_i` — the exact Hessian
    /// must match it to rounding.
    #[test]
    fn the_exact_eta_hessian_is_the_closed_form_on_an_exponential_hazard() {
        let root = env!("CARGO_MANIFEST_DIR");
        let prep = crate::api::prepare_run(
            &format!("{root}/examples/tte_exponential.ferx"),
            Some(&format!("{root}/data/tte_exponential.csv")),
        )
        .unwrap();
        let model = &prep.parsed.model;
        let theta = &model.default_params.theta;
        let (cmt, ep) = model.endpoints.iter().next().unwrap();
        assert!(
            matches!(ep, EndpointLikelihood::Tte { hazard: HazardSpec::Analytic { .. }, .. }),
            "an analytic exponential endpoint"
        );
        let mut checked = 0;
        for s in prep.population.subjects.iter().take(8) {
            let recs: Vec<ObsRecord> = s
                .obs_records
                .iter()
                .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if c == cmt))
                .cloned()
                .collect();
            if recs.is_empty() {
                continue;
            }
            let eta = [0.13];
            let term = TteTerm::new(model, s);
            let kind = crate::sens::ode_provider::adjoint::terms::TermKind::Tte;
            let (_, h) = term_eta_hessian(model, s, kind, &term, theta, &eta).unwrap();
            let lambda = theta[0] * eta[0].exp();
            let t_sum: f64 = recs
                .iter()
                .map(|r| match r {
                    ObsRecord::Event { time, entry_time, .. } => {
                        assert_eq!(*entry_time, 0.0, "the closed form assumes no truncation");
                        *time
                    }
                    _ => 0.0,
                })
                .sum();
            let want = lambda * t_sum;
            assert!(((h[(0, 0)] - want) / want).abs() < 1e-14, "{} vs {want}", h[(0, 0)]);
            checked += 1;
        }
        assert!(checked > 0);
    }

    /// The inner gradient's TTE block in closed form on the exponential hazard:
    /// `∂/∂η [λ Σ t_i − Σ_exact log λ] = λ Σ t_i − n_exact`, matched to rounding, with the
    /// value equal to the production data term's.
    #[test]
    fn the_tte_eta_gradient_is_the_closed_form_on_an_exponential_hazard() {
        let root = env!("CARGO_MANIFEST_DIR");
        let prep = crate::api::prepare_run(
            &format!("{root}/examples/tte_exponential.ferx"),
            Some(&format!("{root}/data/tte_exponential.csv")),
        )
        .unwrap();
        let model = &prep.parsed.model;
        let theta = &model.default_params.theta;
        let mut checked = 0;
        for s in prep.population.subjects.iter().take(8) {
            if s.obs_records.is_empty() {
                continue;
            }
            let eta = [-0.21];
            let (v, g) = tte_subject_eta_grad(model, s, theta, &eta).expect("in scope");
            let lambda = theta[0] * eta[0].exp();
            let (mut t_sum, mut n_exact) = (0.0, 0.0);
            for r in &s.obs_records {
                if let ObsRecord::Event { time, event_type, .. } = r {
                    t_sum += time;
                    if matches!(event_type, EventType::Exact) {
                        n_exact += 1.0;
                    }
                }
            }
            let want = lambda * t_sum - n_exact;
            assert!((g[0] - want).abs() <= 1e-13 * (1.0 + want.abs()), "{} vs {want}", g[0]);
            let (cmt, ep) = model.endpoints.iter().next().unwrap();
            let crate::types::EndpointLikelihood::Tte { hazard, recurrence, .. } = ep else {
                unreachable!()
            };
            let recs: Vec<ObsRecord> = s
                .obs_records
                .iter()
                .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if c == cmt))
                .cloned()
                .collect();
            let f = tte_data_term(&recs, hazard, *recurrence, theta, &eta, &s.covariates);
            assert!((v - f).abs() <= 1e-12 * (1.0 + f.abs()), "value {v} vs {f}");
            checked += 1;
        }
        assert!(checked > 0);
    }
}

#[cfg(test)]
mod ode_hazard_exact_tests {
    use super::*;
    use crate::types::{EndpointLikelihood, HazardSpec};

    /// The joint PK-TTE (drug-driven, ODE-accumulated) hazard's η-Hessian is exact: the dual
    /// walk's curves through the shared scoring, against the stencil it replaces, to the
    /// stencil's accuracy; and its value against the production `tte_endpoint_nll`, to the
    /// two engines' solver agreement.
    #[test]
    fn the_ode_hazard_eta_hessian_is_exact() {
        let root = env!("CARGO_MANIFEST_DIR");
        let prep = crate::api::prepare_run(
            &format!("{root}/examples/pktte_joint.ferx"),
            Some(&format!("{root}/data/pktte_joint.csv")),
        )
        .unwrap();
        let model = &prep.parsed.model;
        let theta = &model.default_params.theta;
        let (cmt, chz) = model
            .endpoints
            .iter()
            .find_map(|(c, ep)| match ep {
                EndpointLikelihood::Tte {
                    hazard: HazardSpec::OdeAccumulated { chz_state },
                    ..
                } => Some((*c, *chz_state)),
                _ => None,
            })
            .expect("an ODE-accumulated endpoint");
        assert!(tte_endpoint_served(model, cmt));
        let (mut worst_h, mut worst_v, mut n) = (0.0f64, 0.0f64, 0);
        for s in prep.population.subjects.iter().take(5) {
            let recs: Vec<ObsRecord> = s
                .obs_records
                .iter()
                .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if *c == cmt))
                .cloned()
                .collect();
            if recs.is_empty() {
                continue;
            }
            let eta = [0.12];
            let term = TteTerm::new(model, s);
            let kind = crate::sens::ode_provider::adjoint::terms::TermKind::Tte;
            let (_, exact) =
                term_eta_hessian(model, s, kind, &term, theta, &eta).expect("in scope");
            let f = |e: &[f64]| {
                crate::stats::likelihood::tte_endpoint_nll_for_tests(model, s, cmt, &recs, theta, e)
            };
            // Richardson of the exact η-gradient (`O(h⁴)`), not a stencil of the value.
            let g = |e: f64| tte_subject_eta_grad(model, s, theta, &[e]).unwrap().1[0];
            let d = |h: f64| (g(eta[0] + h) - g(eta[0] - h)) / (2.0 * h);
            let fd = (4.0 * d(5e-4) - d(1e-3)) / 3.0;
            worst_h = worst_h.max((exact[(0, 0)] - fd).abs() / fd.abs().max(1e-12));
            let v = ode_tte_nll_g::<f64>(model, s, chz, &recs, theta, &eta).unwrap();
            worst_v = worst_v.max((v - f(&eta)).abs() / (1.0 + f(&eta).abs()));
            n += 1;
        }
        assert!(n > 2, "too few subjects ({n})");
        eprintln!(
            "ODE hazard: Hessian vs Richardson {worst_h:.2e}, value vs production {worst_v:.2e}"
        );
        // Measured: the value bit-identical to production's (0.0), the Hessian 3.65e-6 off
        // Richardson of the exact gradient — the frozen-step derivative against the adaptive
        // solve's slope at default tolerances. 2e-5 is 5× that; a dropped `h` term or CHZ
        // readout misses by O(1).
        assert!(worst_v < 1e-10, "value {worst_v:.2e}");
        assert!(worst_h < 2e-5, "Hessian {worst_h:.2e}");
    }

    /// A drug-driven hazard that also reads `TIME` (Gompertz-shaped, `H0·e^{GAM·TIME}·e^{BETA·C}`)
    /// is exact too: the dual walk integrates the clock, and `h(t)` reads the value path's
    /// anchors. The value matches production's non-Gaussian term; the η-gradient and the
    /// θ-score (`GAM` among them, the clock's own channel) match Richardson differences of it.
    #[test]
    fn a_time_dependent_ode_hazard_is_exact() {
        let root = env!("CARGO_MANIFEST_DIR");
        // The anchor's model with a random effect on CL (the anchor itself has none), so the
        // hazard's drug channel carries an η.
        let src = std::fs::read_to_string(format!("{root}/nonmem_anchor/pktte_tdep_fit.ferx"))
            .unwrap()
            .replace("  sigma PROP_ERR", "  omega ETA_CL ~ 0.09\n  sigma PROP_ERR")
            .replace("CL   = TVCL\n", "CL   = TVCL * exp(ETA_CL)\n");
        let mut model = crate::parser::model_parser::parse_model_string(&src).unwrap();
        assert_eq!(model.n_eta, 1);
        if let Some(ode) = model.ode_spec.as_mut() {
            ode.solver_opts.reltol = 1e-12;
            ode.solver_opts.abstol = 1e-14;
        }
        let (population, _) = crate::api::read_population_for(
            &model,
            &None,
            &format!("{root}/nonmem_anchor/pktte_tdep.csv"),
            None,
            None,
            None,
            &[],
        )
        .unwrap();
        let model = &model;
        let theta = model.default_params.theta.clone();
        let cmt = *model
            .endpoints
            .iter()
            .find(|(_, ep)| matches!(ep, EndpointLikelihood::Tte { .. }))
            .unwrap()
            .0;
        assert!(tte_endpoint_served(model, cmt), "a TIME-reading hazard is served");
        let rich = |f: &dyn Fn(f64) -> f64, x: f64| {
            let d = |h: f64| (f(x + h) - f(x - h)) / (2.0 * h);
            // Relative to `x` itself: a small parameter (a baseline hazard of 0.02) under an
            // absolute step is the oracle's own error, not the derivative's.
            let h = 1e-3 * x.abs().max(1e-2);
            (4.0 * d(h / 2.0) - d(h)) / 3.0
        };
        let nll = |s: &Subject, th: &[f64], e: &[f64]| {
            let mut acc = 0.0;
            crate::stats::likelihood::accumulate_non_gaussian_nll(
                model, s, th, e, None, 1.0, &mut acc,
            );
            acc
        };
        let (mut worst_v, mut worst_g, mut n) = (0.0f64, 0.0f64, 0);
        for s in population.subjects.iter().filter(|s| !s.obs_records.is_empty()).take(4) {
            assert!(tte_record_served(model, s, cmt), "subject {} is served", s.id);
            let eta: Vec<f64> = (0..model.n_eta).map(|k| 0.1 - 0.05 * k as f64).collect();
            let (v, g) = tte_subject_eta_grad(model, s, &theta, &eta).expect("exact");
            let v_ref = nll(s, &theta, &eta);
            worst_v = worst_v.max((v - v_ref).abs() / (1.0 + v_ref.abs()));
            for k in 0..eta.len() {
                let fd = rich(
                    &|x| {
                        let mut e = eta.clone();
                        e[k] = x;
                        nll(s, &theta, &e)
                    },
                    eta[k],
                );
                worst_g = worst_g.max((g[k] - fd).abs() / (1.0 + fd.abs()));
            }
            let (_, gt) = crate::sens::ode_provider::adjoint::non_gaussian_theta_score(
                model, s, &theta, &eta,
            )
            .expect("exact");
            for i in 0..theta.len() {
                let fd = rich(
                    &|x| {
                        let mut th = theta.clone();
                        th[i] = x;
                        nll(s, &th, &eta)
                    },
                    theta[i],
                );
                worst_g = worst_g.max((gt[i] - fd).abs() / (1.0 + fd.abs()));
            }
            n += 1;
        }
        assert!(n >= 3, "too few subjects ({n})");
        eprintln!("TIME-reading ODE hazard: value {worst_v:.2e}, gradients {worst_g:.2e}");
        assert!(worst_v < 1e-8, "value {worst_v:.2e}");
        assert!(worst_g < 1e-6, "gradient {worst_g:.2e}");
    }

    /// #1575's fixture (`tests/one_snapshot_dose_reads_survival.rs`) on the exact route: a
    /// `TIME`-reading `$PK` whose `TIME = 0` disposition snapshot differs from every dose
    /// record (`MTT = −9`, `ALAG1 = 10.5` there, `1` and `0.5` at the records), a transit
    /// kernel and a compartment lag read per dose, a `TAD`-reading hazard, doses at 1 and 13
    /// (the second on residual drug), η on `CL`, the lag and the kernel, and a `CL` the
    /// disposition reads at `TIME = 0` (1.5×) but no dose record does; and a steady-state
    /// twin (`SS=1`, the plain drug-driven hazard). Both take the event-driven dual walk with
    /// every segment on the disposition (`ode_states_at_g`), which the static walk declined.
    ///
    /// Value: the walk's `(H, h)` against production's dense single-pass solve
    /// (`ode_cumhaz_hazard`, a different engine) — dies if a dose quantity is read at the
    /// disposition (the kernel then delivers nothing, the lag moves every arrival), if a
    /// segment is integrated on a dose record's snapshot instead of the disposition (its `CL`),
    /// or if the `TAD` anchor drops the lag. Gradients: the exact η-gradient and θ-score
    /// against Richardson differences of the production objective.
    #[test]
    fn a_time_reading_pk_with_an_ode_hazard_is_exact() {
        const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, FIX)
  theta TVV(10.0, FIX)
  theta TVKA(1.0, FIX)
  theta TVMTT(1.0, FIX)
  theta TVN(3.0, FIX)
  theta TVLAG(0.5, FIX)
  theta TVH0(0.02, FIX)
  theta TVBETA(0.1, FIX)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.09
  omega ETA_MTT ~ 0.09
  sigma PROP_ERR ~ 0.01
[individual_parameters]
  CL    = TVCL * exp(ETA_CL) * (if (TIME < 0.5) 1.5 else 1.0)
  V     = TVV
  KA    = TVKA
  MTT   = TVMTT * exp(ETA_MTT) - 10.0 * (if (TIME < 0.5) 1.0 else 0.0)
  NTR   = TVN
  ALAG1 = TVLAG * exp(ETA_LAG) + 10.0 * (if (TIME < 0.5) 1.0 else 0.0)
  H0    = TVH0
  BETA  = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
[scaling]
  obs_scale = V
[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V)) + 0.002 * TAD
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
";
        use std::io::Write;
        // The steady-state twin: #1575's lag and transit kernel are rejected under `SS=1`
        // (`E_ABSORPTION_SS_LAG`), so it is the plain drug-driven hazard.
        let ss_model = MODEL
            .replace("  omega ETA_LAG ~ 0.09\n  omega ETA_MTT ~ 0.09\n", "")
            .replace("transit(n=NTR, mtt=MTT) - KA * depot", "-KA * depot")
            .replace("  MTT   = TVMTT * exp(ETA_MTT) - 10.0 * (if (TIME < 0.5) 1.0 else 0.0)\n", "")
            .replace("  ALAG1 = TVLAG * exp(ETA_LAG) + 10.0 * (if (TIME < 0.5) 1.0 else 0.0)\n", "")
            .replace(" + 0.002 * TAD", "")
            .replace(" * (if (TIME < 0.5) 1.5 else 1.0)", "");
        let (mut worst_v, mut worst_g) = (0.0f64, 0.0f64);
        for (src, ss) in [(MODEL, "0,0"), (ss_model.as_str(), "1,12")] {
            let model = crate::parser::model_parser::parse_model_string(src).unwrap();
            let (cmt, chz) = model
                .endpoints
                .iter()
                .find_map(|(c, ep)| match ep {
                    EndpointLikelihood::Tte {
                        hazard: HazardSpec::OdeAccumulated { chz_state },
                        ..
                    } => Some((*c, *chz_state)),
                    _ => None,
                })
                .unwrap();
            let theta = model.default_params.theta.clone();
            let eta = [0.1, -0.2, 0.15][..model.n_eta].to_vec();
            let rich = |f: &dyn Fn(f64) -> f64, x: f64| {
                let d = |h: f64| (f(x + h) - f(x - h)) / (2.0 * h);
                let h = 1e-3 * x.abs().max(1e-1);
                (4.0 * d(h / 2.0) - d(h)) / 3.0
            };
            let nll = |s: &Subject, th: &[f64], e: &[f64]| {
                let mut acc = 0.0;
                crate::stats::likelihood::accumulate_non_gaussian_nll(
                    &model, s, th, e, None, 1.0, &mut acc,
                );
                acc
            };
            {
                let data = format!(
                    "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n\
                 1,1,.,1,100,1,0,1,{ss}\n\
                 1,8,5,0,.,2,0,0,0,0\n\
                 1,13,.,1,100,1,0,1,0,0\n\
                 1,20,1,0,0,3,0,0,0,0\n"
                );
                let mut f = tempfile::NamedTempFile::new().unwrap();
                f.write_all(data.as_bytes()).unwrap();
                let (pop, _) = crate::api::read_population_for(
                    &model,
                    &None,
                    f.path().to_str().unwrap(),
                    None,
                    None,
                    None,
                    &[],
                )
                .unwrap();
                let s = &pop.subjects[0];
                assert!(tte_record_served(&model, s, cmt), "SS={ss}: served exactly");
                let times = [20.0];
                let walk = crate::sens::ode_provider::adjoint::ode_states_at_g::<f64>(
                    &model, s, &theta, &eta, &times,
                )
                .expect("the event walk serves it");
                let (cum, haz) = crate::sens::ode_provider::adjoint::cumhaz_hazard_from_states(
                    &model, s, chz, &times, &walk,
                )
                .unwrap();
                let (cum_d, haz_d) = ode_cumhaz_hazard(&model, s, chz, &theta, &eta, &times);
                assert!(
                    cum[0].is_finite()
                        && haz[0].is_finite()
                        && cum_d[0].is_finite()
                        && haz_d[0].is_finite()
                );
                worst_v = worst_v // nan-ok: all four values asserted finite above; the divisors are the dense cumulative hazard and hazard
                    .max((cum[0] - cum_d[0]).abs() / cum_d[0].abs())
                    .max((haz[0] - haz_d[0]).abs() / haz_d[0].abs());
                let (_, g) = tte_subject_eta_grad(&model, s, &theta, &eta).expect("exact");
                for k in 0..eta.len() {
                    let fd = rich(
                        &|x| {
                            let mut e = eta.clone();
                            e[k] = x;
                            nll(s, &theta, &e)
                        },
                        eta[k],
                    );
                    assert!(fd.is_finite() && g[k].is_finite());
                    // Every η is live: the lag and the kernel move the hazard at 20.
                    assert!(fd.abs() > 1e-4, "SS={ss}: η[{k}] is not live ({fd:.2e})");
                    worst_g = worst_g.max((g[k] - fd).abs() / (1.0 + fd.abs()));
                }
                let (_, gt) = crate::sens::ode_provider::adjoint::non_gaussian_theta_score(
                    &model, s, &theta, &eta,
                )
                .expect("exact");
                for i in 0..theta.len() {
                    let fd = rich(
                        &|x| {
                            let mut th = theta.clone();
                            th[i] = x;
                            nll(s, &th, &eta)
                        },
                        theta[i],
                    );
                    assert!(fd.is_finite() && gt[i].is_finite());
                    worst_g = worst_g.max((gt[i] - fd).abs() / (1.0 + fd.abs()));
                }
            }
        }
        eprintln!(
            "TIME-reading $PK ODE hazard: value vs dense {worst_v:.2e}, gradients {worst_g:.2e}"
        );
        // Measured on Linux: value 2.0e-15 against the dense engine (the same arithmetic on
        // another walk), gradients 2.2e-10 (Richardson's own error at these tolerances).
        assert!(worst_v < 1e-12, "value {worst_v:.2e}");
        assert!(worst_g < 5e-9, "gradient {worst_g:.2e}");
    }
}
