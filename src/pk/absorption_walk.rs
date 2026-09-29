//! Exact event-driven walk for the closed-form **transit** and **inverse-Gaussian**
//! absorption models under IOV and time-varying covariates (#1560).
//!
//! The static closed form (`one_cpt_transit_amt_g`, `convolve_1cpt`, …) superposes each
//! dose over its whole history at one parameter set, so it cannot carry drug across a
//! record where the disposition changes (#104). Until #1560 every such subject was rerouted
//! to the model's `transit()` / `igd()` ODE twin (#719). This walk serves them analytically
//! instead, by splitting the model at each record:
//!
//! * **Absorption** is fixed at the dose record (#1569): each dose keeps the `n`/`mtt`
//!   (`mat`/`cv2`), `F` and lagtime of its own record, exactly as the twin and the NONMEM
//!   anchors (`nonmem_anchor/transit_iov_mtt`, `ig_iov_mat`) do.
//! * **Disposition** (`ke`, or `α, β, k12, k21`) is the governing record's, constant on each
//!   interval — the same record the other engines use
//!   ([`crate::dosing::governing_record_indices`]).
//!
//! So on an interval `[s0, s1]`, with `W_d(r)` the windowed tilting term
//! ([`windowed_tilted_term`]) of dose `d` over `[s0 − a_d, s1 − a_d]` (`a_d` its arrival):
//!
//! ```text
//! 1-cpt:  A_c(s1) = A_c(s0)·e^{−ke·Δ} + Σ_d F_d·D_d·W_d(ke)
//! 2-cpt:  (A_c, A_p)(s1) = expm(K·Δ)·(A_c, A_p)(s0)
//!                        + Σ_d F_d·D_d·[ cα·W_d(α) + cβ·W_d(β),  k12/(α−β)·(W_d(β) − W_d(α)) ]
//! ```
//!
//! with `cα = (α−k21)/(α−β)`, `cβ = (k21−β)/(α−β)`. Every term is exact; the walk is one
//! function over [`PkNum`], so `T = f64` is the prediction and `T = Dual2`/`Dual1` its
//! sensitivities (`crate::sens::provider`), with no second copy of the formula.
//!
//! ## Domain — the one owner of "walk or twin"
//!
//! The tilting factorisation needs every disposition rate below the MGF abscissa of every
//! dose still being evaluated: under fixed-at-dose absorption the condition is per
//! **(interval, open dose) pair** (`ke < KTR_d`, or `α < KTR_d`; IG `< 1/(2·MAT_d·CV²_d)`),
//! plus distinct 2-cpt eigenvalues and `Q > 0`. [`walk_domain`] is the only place that
//! decides it: the walk calls it before propagating, and every router calls it on the same
//! `f64` values before choosing between the walk and the twin, so the prediction and the
//! gradient cannot take different routes for one subject.

use crate::pk::analytical_absorption::{
    windowed_tilted_term, IgAbsorption, TiltedAbsorption, TransitAbsorption,
};
use crate::pk::event_driven::{is_record, EventKind, EventSchedule};
use crate::sens::num::PkNum;
use crate::sens::propagate::{propagate_two_cpt_core_g, PkDual, TwoCptEigen};
use crate::types::{CompiledModel, PkModel, PkParams, Subject};

/// One dose as the walk sees it: every attribute is the dose record's own (#1569).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AbsDose<T: PkNum> {
    /// Arrival `t_dose + ALAG` — a dual when the lagtime is estimated.
    pub arrival: T,
    /// Bioavailable amount `F·AMT`.
    pub mass: T,
    /// Transit `n` / IG `MAT`.
    pub a: T,
    /// Transit `MTT` / IG `CV²`.
    pub b: T,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    Transit,
    Ig,
}

/// `(kernel, two_cpt)` for a closed-form absorption model, `None` for any other.
fn kernel_of(pk_model: PkModel) -> Option<(Kernel, bool)> {
    match pk_model {
        PkModel::OneCptTransit => Some((Kernel::Transit, false)),
        PkModel::TwoCptTransit => Some((Kernel::Transit, true)),
        PkModel::OneCptIg => Some((Kernel::Ig, false)),
        PkModel::TwoCptIg => Some((Kernel::Ig, true)),
        _ => None,
    }
}

/// True for the four closed-form absorption models this walk serves.
pub(crate) fn is_absorption_closed_form(pk_model: PkModel) -> bool {
    kernel_of(pk_model).is_some()
}

/// Which engine serves a subject the walk is eligible for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WalkDomain {
    /// Every (interval, open dose) pair is inside the tilting domain.
    Walk,
    /// Some pair is outside it — the flip-flop regime, confluent 2-cpt eigenvalues or
    /// `Q ≤ 0` — so the subject is served by the ODE twin, which is valid in both regimes.
    Twin,
    /// A non-physical parameter (`CL ≤ 0`, `V ≤ 0`, `n < 0`, `MTT ≤ 0`, …). The static
    /// closed form returns `0` there as a deliberate optimiser penalty, and so does the walk.
    Invalid,
}

/// The MGF abscissa of one dose's kernel: `KTR = (n+1)/MTT` (transit) or
/// `1/(2·MAT·CV²)` (IG). `None` for non-physical kernel parameters.
fn abscissa(kernel: Kernel, a: f64, b: f64) -> Option<f64> {
    match kernel {
        Kernel::Transit => (a >= 0.0 && b > 0.0).then(|| (a + 1.0) / b),
        Kernel::Ig => (a > 0.0 && b > 0.0).then(|| 1.0 / (2.0 * a * b)),
    }
}

/// The largest disposition rate the tilting factorisation evaluates on an interval —
/// `ke` (1-cpt) or `α` (2-cpt) — or the reason there is none.
fn fastest_rate(two_cpt: bool, p: &PkDual<f64>) -> Result<f64, WalkDomain> {
    if !(p.cl > 0.0 && p.v > 0.0) {
        return Err(WalkDomain::Invalid);
    }
    if !two_cpt {
        return Ok(p.cl / p.v);
    }
    if !(p.v2 > 0.0) || p.q < 0.0 {
        return Err(WalkDomain::Invalid);
    }
    if p.q == 0.0 {
        // A disconnected peripheral: valid, but outside the eigen-carry this walk uses.
        return Err(WalkDomain::Twin);
    }
    let (alpha, beta, _) = crate::sens::two_cpt::macro_rates_g::<f64>(p.cl, p.v, p.q, p.v2);
    if (alpha - beta).abs() < 1e-12 {
        return Err(WalkDomain::Twin);
    }
    Ok(alpha)
}

/// The parameters governing each event's interval: the record terminating it (#1073).
/// `None` only for a timeline with no record at all.
fn governing<'a, T: PkNum>(
    schedule: &EventSchedule,
    pk_at_dose: &'a [PkDual<T>],
    pk_at_obs: &'a [PkDual<T>],
    pk_at_pk_only: &'a [PkDual<T>],
) -> Vec<Option<&'a PkDual<T>>> {
    let rec = crate::dosing::governing_record_indices(schedule.events.len(), |i| {
        is_record(schedule.events[i].kind)
    });
    rec.iter()
        .map(|q| {
            q.map(|q| {
                let ev = schedule.events[q];
                match ev.kind {
                    EventKind::DoseRecord => &pk_at_dose[ev.orig_idx],
                    EventKind::PkOnly => &pk_at_pk_only[ev.orig_idx],
                    _ => &pk_at_obs[ev.orig_idx],
                }
            })
        })
        .collect()
}

/// **The** decision between the walk and the ODE twin (see the module docs). Reads only
/// values, so the `f64` router and a dual walk over the same parameters agree.
///
/// A pair is every interval of positive length paired with every dose that has arrived by
/// the interval's start — including doses long since absorbed, because the walk evaluates
/// their (vanishing) window too, and the factorisation diverges regardless of its size.
pub(crate) fn walk_domain<T: PkNum>(
    pk_model: PkModel,
    schedule: &EventSchedule,
    doses: &[AbsDose<T>],
    pk_at_dose: &[PkDual<T>],
    pk_at_obs: &[PkDual<T>],
    pk_at_pk_only: &[PkDual<T>],
) -> WalkDomain {
    let Some((kernel, two_cpt)) = kernel_of(pk_model) else {
        return WalkDomain::Twin;
    };
    let mut verdict = WalkDomain::Walk;
    for d in doses {
        if abscissa(kernel, d.a.val(), d.b.val()).is_none() {
            verdict = WalkDomain::Invalid;
        }
    }
    let gov = governing(schedule, pk_at_dose, pk_at_obs, pk_at_pk_only);
    let mut cur_t = match schedule.events.first() {
        Some(e) => e.time,
        None => return verdict,
    };
    for (i, ev) in schedule.events.iter().enumerate() {
        if ev.kind == EventKind::Reset {
            // Resets are rejected on these models at validation, and the walk has no
            // reset semantics of its own; the twin does.
            return WalkDomain::Twin;
        }
        if ev.time > cur_t {
            let Some(p) = gov[i] else {
                return WalkDomain::Twin;
            };
            let vals = PkDual {
                cl: p.cl.val(),
                v: p.v.val(),
                q: p.q.val(),
                v2: p.v2.val(),
                ka: 0.0,
                q3: 0.0,
                v3: 0.0,
                f: 0.0,
            };
            let rate = match fastest_rate(two_cpt, &vals) {
                Ok(r) => Some(r),
                Err(WalkDomain::Twin) => return WalkDomain::Twin,
                Err(_) => {
                    verdict = WalkDomain::Invalid;
                    None
                }
            };
            if let Some(r) = rate {
                for d in doses.iter().filter(|d| d.arrival.val() <= cur_t) {
                    if let Some(abs) = abscissa(kernel, d.a.val(), d.b.val()) {
                        if !(r < abs) {
                            return WalkDomain::Twin;
                        }
                    }
                }
            }
            cur_t = ev.time;
        }
    }
    verdict
}

/// Add one dose's delivery over `[s0, s1]` to the state, at the interval's disposition.
/// `t0`, `t1` are the window limits in the dose's own clock (`s − arrival`).
fn add_window<T: PkNum, A: TiltedAbsorption<T>>(
    abs: &A,
    mass: T,
    t0: T,
    t1: T,
    disp: &Disp<T>,
    state: &mut [T],
) {
    match *disp {
        Disp::One { ke } => {
            state[0] = state[0] + mass * windowed_tilted_term(abs, t0, t1, ke);
        }
        Disp::Two(e) => {
            let diff = e.alpha - e.beta;
            let wa = windowed_tilted_term(abs, t0, t1, e.alpha);
            let wb = windowed_tilted_term(abs, t0, t1, e.beta);
            let c_alpha = (e.alpha - e.k21) / diff;
            let c_beta = (e.k21 - e.beta) / diff;
            state[0] = state[0] + mass * (c_alpha * wa + c_beta * wb);
            state[1] = state[1] + mass * (e.k12 / diff) * (wb - wa);
        }
    }
}

/// The interval's disposition in the form the step reads.
#[derive(Clone, Copy)]
enum Disp<T: PkNum> {
    One { ke: T },
    Two(TwoCptEigen<T>),
}

/// Amount of `d` still in the absorption kernel at `t`: `F·D·(1 − CDF(t − a_d))`, the
/// untilted CDF being `G(·; 0)`. The analytical `depot` state (`one_cpt_transit_depot`).
fn undelivered<T: PkNum>(kernel: Kernel, d: &AbsDose<T>, t: f64) -> T {
    let tau = T::from_f64(t) - d.arrival;
    let zero = T::from_f64(0.0);
    let cdf = match kernel {
        Kernel::Transit => TransitAbsorption { n: d.a, mtt: d.b }.tilted_cdf(tau, zero),
        Kernel::Ig => IgAbsorption {
            mat: d.a,
            lambda: d.a / d.b,
        }
        .tilted_cdf(tau, zero),
    };
    d.mass * (T::from_f64(1.0) - cdf)
}

/// The walk. Returns `None` when [`walk_domain`] sends the subject to the twin; otherwise
/// the central concentration at every observation (clamped at `0` like the other walks, a
/// `NaN` kept), and — when `states` is `Some` — the per-observation states in the model's
/// analytical layout and convention (`[depot, central]` / `[depot, central, peripheral]`:
/// the kernel's undelivered amount, then concentrations), the one `[derived]` reads by the
/// model's `analytical_compartment_names`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn absorption_walk_g<T: PkNum>(
    pk_model: PkModel,
    schedule: &EventSchedule,
    doses: &[AbsDose<T>],
    pk_at_dose: &[PkDual<T>],
    pk_at_obs: &[PkDual<T>],
    pk_at_pk_only: &[PkDual<T>],
    n_obs: usize,
    mut states: Option<&mut Vec<Vec<T>>>,
) -> Option<Vec<T>> {
    let (kernel, two_cpt) = kernel_of(pk_model)?;
    let zero = T::from_f64(0.0);
    let mut preds = vec![zero; n_obs];
    if let Some(s) = states.as_deref_mut() {
        s.clear();
        s.resize(n_obs, Vec::new());
    }
    if n_obs == 0 || schedule.events.is_empty() {
        return Some(preds);
    }
    // A non-finite event time makes the subject non-finite (#1189), as on every engine.
    if schedule.non_finite_event_time {
        return Some(vec![T::from_f64(f64::NAN); n_obs]);
    }
    match walk_domain(
        pk_model,
        schedule,
        doses,
        pk_at_dose,
        pk_at_obs,
        pk_at_pk_only,
    ) {
        WalkDomain::Twin => return None,
        WalkDomain::Invalid => return Some(preds),
        WalkDomain::Walk => {}
    }
    let gov = governing(schedule, pk_at_dose, pk_at_obs, pk_at_pk_only);
    let mut state = [zero, zero];
    let mut arrived = vec![false; doses.len()];
    let mut cur_t = schedule.events[0].time;
    for (i, ev) in schedule.events.iter().enumerate() {
        let pk_now = gov[i];
        if ev.time > cur_t {
            let p = pk_now.expect("walk_domain declines a timeline with no record");
            let dt = T::from_f64(ev.time - cur_t);
            let disp = if two_cpt {
                let (alpha, beta, k21) = crate::sens::two_cpt::macro_rates_g(p.cl, p.v, p.q, p.v2);
                let e = TwoCptEigen {
                    alpha,
                    beta,
                    k10: p.cl / p.v,
                    k12: p.q / p.v,
                    k21,
                };
                propagate_two_cpt_core_g(&mut state, dt, &e, zero, zero);
                Disp::Two(e)
            } else {
                let ke = p.cl / p.v;
                state[0] = state[0] * (-(ke * dt)).exp();
                Disp::One { ke }
            };
            let (s0, s1) = (T::from_f64(cur_t), T::from_f64(ev.time));
            for (k, d) in doses.iter().enumerate() {
                if !arrived[k] {
                    continue;
                }
                let (t0, t1) = (s0 - d.arrival, s1 - d.arrival);
                match kernel {
                    Kernel::Transit => {
                        let abs = TransitAbsorption { n: d.a, mtt: d.b };
                        add_window(&abs, d.mass, t0, t1, &disp, &mut state);
                    }
                    Kernel::Ig => {
                        let abs = IgAbsorption {
                            mat: d.a,
                            lambda: d.a / d.b,
                        };
                        add_window(&abs, d.mass, t0, t1, &disp, &mut state);
                    }
                }
            }
            cur_t = ev.time;
        }
        match ev.kind {
            EventKind::Dose => arrived[ev.orig_idx] = true,
            EventKind::Obs => {
                let p = pk_now.expect("an observation is a record");
                let per_volume =
                    |amount: T, vol: T| if vol.val() > 0.0 { amount / vol } else { zero };
                let conc = per_volume(state[0], p.v);
                // `conc < 0` is false for `NaN`, which is kept rather than floored to 0.
                preds[ev.orig_idx] = if conc.val() < 0.0 { zero } else { conc };
                if let Some(s) = states.as_deref_mut() {
                    // The analytical state convention (`single_dose_states`): the kernel
                    // `depot` as an amount, central and peripheral as concentrations.
                    let depot = doses
                        .iter()
                        .zip(&arrived)
                        .filter(|(_, &a)| a)
                        .fold(zero, |acc, (d, _)| acc + undelivered(kernel, d, ev.time));
                    let row = if two_cpt {
                        vec![depot, conc, per_volume(state[1], p.v2)]
                    } else {
                        vec![depot, conc]
                    };
                    // Floored like `analytical_state_at_times` (a `NaN` is kept).
                    s[ev.orig_idx] = row
                        .into_iter()
                        .map(|x| if x.val() < 0.0 { zero } else { x })
                        .collect();
                }
            }
            EventKind::DoseRecord | EventKind::PkOnly => {}
            EventKind::Reset => unreachable!("walk_domain sends a reset timeline to the twin"),
        }
    }
    Some(preds)
}

/// Whether the walk may serve this subject at all — the structural half of the routing,
/// before any parameter is known. A closed-form transit/IG model whose subject the static
/// superposition cannot serve **only** because of IOV or time-varying covariates, and which
/// has an ODE twin to fall back to when [`walk_domain`] says so.
///
/// Everything else keeps its existing engine: a `TIME` read, a periodic SS dose, an infusion
/// or a reset still routes to the twin through [`CompiledModel::effective_for`] (#719), and a
/// static subject stays on the static superposition. `effective_for` itself is deliberately
/// unchanged — it has consumers (state/`[derived]` outputs, covariance diagnostics, the
/// inner-optimizer schedule gates) that know nothing of this walk, and for them the twin
/// remains the right, if slower, answer.
pub(crate) fn walk_eligible(model: &CompiledModel, subject: &Subject) -> bool {
    is_absorption_closed_form(model.pk_model)
        && model.ode_spec.is_none()
        && !model.is_algebraic()
        && model.absorption_ode_equivalent.is_some()
        && (subject.has_tv_covariates() || model.n_kappa > 0)
        && !crate::parser::model_parser::compiled_model_uses_time_builtin(model)
        && !subject.has_periodic_ss_dose()
        && !subject.doses.iter().any(|d| d.is_infusion())
        && !subject.has_resets()
}

/// The disposition slots of a `PkParams` as the walk's `f64` [`PkDual`].
pub(crate) fn disp_f64(p: &PkParams) -> PkDual<f64> {
    PkDual {
        cl: p.cl(),
        v: p.v(),
        q: p.q(),
        v2: p.v2(),
        ka: 0.0,
        q3: 0.0,
        v3: 0.0,
        f: p.f_bio(),
    }
}

/// The dose-record attributes of `subject.doses[k]` as the walk's `f64` [`AbsDose`].
pub(crate) fn dose_f64(
    pk_model: PkModel,
    subject: &Subject,
    k: usize,
    p: &PkParams,
) -> AbsDose<f64> {
    let (a, b) = match kernel_of(pk_model) {
        Some((Kernel::Ig, _)) => (p.mat(), p.cv2()),
        _ => (p.n_transit(), p.mtt()),
    };
    let d = &subject.doses[k];
    AbsDose {
        arrival: d.time + p.lagtime(),
        mass: p.bioavailable_amount(d.amt),
        a,
        b,
    }
}

/// The event schedule the walk runs on: the subject's timeline with each dose's arrival at
/// its own record's lagtime. Transit/IG doses are boluses (infusions are routed away), so
/// the schedule's infusion bounds are trivial.
pub(crate) fn walk_schedule(
    subject: &Subject,
    pk_model: PkModel,
    pk_at_dose: &[PkParams],
) -> EventSchedule {
    let lags: Vec<f64> = pk_at_dose.iter().map(|p| p.lagtime()).collect();
    EventSchedule::for_subject(subject, pk_model, &subject.doses, &lags)
}

/// The value path: predictions (and optionally states) for an eligible subject from its
/// per-event `f64` parameters, or `None` for the twin.
pub(crate) fn absorption_walk_predictions(
    pk_model: PkModel,
    subject: &Subject,
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
    states: Option<&mut Vec<Vec<f64>>>,
) -> Option<Vec<f64>> {
    let schedule = walk_schedule(subject, pk_model, pk_at_dose);
    let doses: Vec<AbsDose<f64>> = pk_at_dose
        .iter()
        .enumerate()
        .map(|(k, p)| dose_f64(pk_model, subject, k, p))
        .collect();
    let dose_disp: Vec<PkDual<f64>> = pk_at_dose.iter().map(disp_f64).collect();
    let obs_disp: Vec<PkDual<f64>> = pk_at_obs.iter().map(disp_f64).collect();
    let only_disp: Vec<PkDual<f64>> = pk_at_pk_only.iter().map(disp_f64).collect();
    absorption_walk_g(
        pk_model,
        &schedule,
        &doses,
        &dose_disp,
        &obs_disp,
        &only_disp,
        subject.obs_times.len(),
        states,
    )
}

#[cfg(test)]
#[path = "absorption_walk_tests.rs"]
mod tests;
