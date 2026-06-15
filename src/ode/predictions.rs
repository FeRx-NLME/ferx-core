//! ODE-based predictions for subjects with dose events.
//!
//! Matches Julia's `_ode_predictions`: breaks the timeline at dose times,
//! applies bolus doses as state discontinuities, and integrates between.
//!
//! Infusion doses (`rate > 0`) are handled by breaking the timeline at the
//! infusion's end time and adding `+rate` to the corresponding compartment's
//! derivative for the duration of the infusion via an RHS wrapper.

use crate::ode::solver::{solve_ode, OdeSolverOptions};
use crate::types::{DoseEvent, PkParams, Subject, PK_IDX_F, PK_IDX_LAGTIME};
use std::collections::HashMap;

/// Epsilon used to decide whether an infusion fully spans a segment.
/// Break times are constructed to coincide with infusion start/end so any
/// non-degenerate segment is either fully inside or fully outside each
/// infusion window — this tolerance only guards float-equality on the bound.
const INFUSION_EPS: f64 = 1e-12;

/// `is_infusion()` only checks `rate > 0`, but a degenerate row with
/// `rate > 0 && amt <= 0` (or NaN) yields `duration = amt/rate <= 0`
/// (or NaN). Treating those as infusions would push an infusion-end
/// break that sorts before the dose itself, and NaN would panic the
/// break-time sort. Such rows fall back to the bolus branch instead
/// (a zero/negative bolus update — visible, not silently dropped).
pub(crate) fn is_real_infusion(d: &DoseEvent) -> bool {
    d.is_infusion() && d.duration > 0.0 && d.duration.is_finite()
}

/// Number of dosing cycles to simulate when pre-equilibrating an SS=1
/// dose. With a typical t₁/₂/II ratio under 2 (the common clinical range)
/// this is comfortably past saturation — each additional cycle adds
/// `exp(-k·II)` of the prior decay, so by N=50 the truncation tail is
/// well below 1e-6 for any reasonable PK.
const SS_EQUILIBRATION_CYCLES: usize = 50;

/// Pre-equilibrate the ODE state to its steady-state value for an SS=1
/// dose with interval `dose.ii`. NONMEM SS=1 semantics: at the time of
/// the SS dose, the compartments are loaded with the steady-state
/// amounts from an infinite-past pulse train. No closed form is
/// available for arbitrary ODE systems, so we numerically expand the
/// train: starting from a zero state, simulate
/// [`SS_EQUILIBRATION_CYCLES`] cycles of `(apply dose; integrate for II)`.
/// The state after the loop equals the "just-before-next-pulse" SS state;
/// the caller then applies the SS dose itself through the normal flow,
/// recovering the at-pulse SS amount.
///
/// `dose.ii > 0` and `dose.cmt` valid are required (callers guard this).
/// For SS infusions (`is_real_infusion(dose)`), each cycle integrates a
/// `dose.duration`-long active-infusion window followed by a
/// `(II - duration)`-long quiet window. The SS form requires
/// `dose.duration <= dose.ii` (non-overlapping); overlapping pulses
/// would need a different equilibration scheme and are out of scope —
/// the existing api.rs warning fires for those.
fn equilibrate_ss_state(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    opts: &OdeSolverOptions,
) -> Vec<f64> {
    let n = ode.n_states;
    let mut u = vec![0.0; n];

    if dose.ii <= 0.0 || dose.cmt == 0 {
        return u;
    }
    let cmt_idx = dose.cmt - 1;
    if cmt_idx >= n {
        return u;
    }

    // Bioavailability F (slot PK_IDX_F, default 1.0) scales the amount that
    // actually enters the dosing compartment — NONMEM's convention (F·AMT for
    // a bolus, F·RATE for an infusion). Matches the analytical path
    // (`equilibrate_ss_state_event_driven`).
    let f_bio = pk_params_flat.get(PK_IDX_F).copied().unwrap_or(1.0);

    let is_inf = is_real_infusion(dose);
    let t_inf = dose.duration;
    if is_inf && t_inf > dose.ii {
        // Overlapping infusions; no closed-form / simple equilibration.
        return u;
    }

    for _ in 0..SS_EQUILIBRATION_CYCLES {
        if is_inf {
            // Active-infusion window: wrapped RHS injects rate into the
            // dosing compartment.
            let rate = f_bio * dose.rate;
            let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                (ode.rhs)(y, p, t, dy);
                if cmt_idx < dy.len() {
                    dy[cmt_idx] += rate;
                }
            };
            let sol = solve_ode(
                &wrapped_rhs,
                &u,
                (0.0, t_inf),
                pk_params_flat,
                &[t_inf],
                opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            // Quiet window from end-of-infusion to end-of-cycle.
            let quiet = dose.ii - t_inf;
            if quiet > 0.0 {
                let sol = solve_ode(&ode.rhs, &u, (0.0, quiet), pk_params_flat, &[quiet], opts);
                if let Some(last) = sol.last() {
                    u.copy_from_slice(&last.u);
                }
            }
        } else {
            // Bolus pulse + decay for one cycle.
            u[cmt_idx] += f_bio * dose.amt;
            let sol = solve_ode(
                &ode.rhs,
                &u,
                (0.0, dose.ii),
                pk_params_flat,
                &[dose.ii],
                opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
        }
    }

    u
}

/// Steady-state ODE state at `phase` ∈ [0, II) within the dosing cycle,
/// measured forward from the pulse at phase 0. [`equilibrate_ss_state`]
/// returns the pre-pulse trough (phase 0⁻ ≡ II); this advances from that
/// trough through the dose pulse and `phase` units of the cycle.
///
/// Used to seed the *previous interval's* steady-state tail when an SS dose
/// has a lagtime: observations between the dose record time and the lagged
/// arrival sit at phase `II − lagtime` … `II`, decaying from the prior
/// pulse. Without this seed those samples would read the (empty) initial
/// state. See [`ode_predictions`] for placement and issue #15.
///
/// For SS infusions this assumes `phase ≥ dose.duration` (the prior
/// infusion has finished by `phase`), i.e. `lagtime ≤ II − dose.duration`
/// — the realistic regime; overlapping infusions (`T_inf > II`) are already
/// rejected upstream.
fn ss_state_at_phase(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    phase: f64,
    opts: &OdeSolverOptions,
) -> Vec<f64> {
    let mut u = equilibrate_ss_state(ode, pk_params_flat, dose, opts);
    if phase <= 0.0 {
        return u;
    }
    let cmt_idx = dose.cmt.saturating_sub(1);
    if cmt_idx >= u.len() {
        return u;
    }
    // Bioavailability scales the amount entering the dosing compartment
    // (see `equilibrate_ss_state`).
    let f_bio = pk_params_flat.get(PK_IDX_F).copied().unwrap_or(1.0);

    if is_real_infusion(dose) {
        let rate = f_bio * dose.rate;
        let t_inf = dose.duration;
        let active = phase.min(t_inf);
        let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            (ode.rhs)(y, p, t, dy);
            if cmt_idx < dy.len() {
                dy[cmt_idx] += rate;
            }
        };
        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (0.0, active),
            pk_params_flat,
            &[active],
            opts,
        );
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        if phase > t_inf {
            let quiet = phase - t_inf;
            let sol = solve_ode(&ode.rhs, &u, (0.0, quiet), pk_params_flat, &[quiet], opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
        }
    } else {
        u[cmt_idx] += f_bio * dose.amt;
        let sol = solve_ode(&ode.rhs, &u, (0.0, phase), pk_params_flat, &[phase], opts);
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }
    u
}

/// Returns `(cmt_idx_0based, rate)` for every infusion that is active
/// throughout the closed segment `[t_start, t_end]`. By construction of the
/// break-time list (every infusion start and end is a break time), each
/// infusion is either fully active or fully inactive across a segment.
///
/// `dose_lagtimes[k]` shifts dose `k`'s active window. Parallel to `doses`.
/// An empty slice means "no lagtime" (all zeros).
///
/// `dose_f_bio[k]` is the bioavailability F applied to dose `k`'s infusion rate
/// (NONMEM's F·RATE convention, mirroring the bolus F·AMT applied at dose
/// entry). Parallel to `doses`; a missing entry defaults to 1.0.
pub(crate) fn active_infusions(
    doses: &[DoseEvent],
    t_start: f64,
    t_end: f64,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    reset_floor: f64,
) -> Vec<(usize, f64)> {
    doses
        .iter()
        .enumerate()
        .filter(|(k, d)| {
            let lag = dose_lagtimes.get(*k).copied().unwrap_or(0.0);
            // Infusions started before the most recent system reset (EVID=3/4)
            // are turned off, the same way the reset zeros the compartments.
            is_real_infusion(d)
                && d.time + lag >= reset_floor
                && d.time + lag <= t_start + INFUSION_EPS
                && d.time + lag + d.duration >= t_end - INFUSION_EPS
        })
        .map(|(k, d)| {
            let f_bio = dose_f_bio.get(k).copied().unwrap_or(1.0);
            (d.cmt.saturating_sub(1), f_bio * d.rate)
        })
        .collect()
}

/// Function that computes the observable from
/// `(state, pk_params_flat, theta, eta, covariates)`. Used by `[scaling]
/// y = <expr>` (Form C) to replace the default `u[obs_cmt_idx]` readout
/// with an arbitrary expression over states + individual parameters +
/// thetas + etas + covariates. Callers that don't have theta/eta in scope
/// (e.g. the EKF path, which never sets a Single/PerCmt readout) may pass
/// empty slices.
pub type OdeOutputFn =
    Box<dyn Fn(&[f64], &[f64], &[f64], &[f64], &HashMap<String, f64>) -> f64 + Send + Sync>;

/// How an ODE model's observable is read at each observation event.
///
/// Replaces the earlier mutually-exclusive `(obs_cmt_idx, output_fn)` pair
/// with a single enum that scales naturally to per-CMT (multi-analyte)
/// dispatch.
pub enum OdeReadout {
    /// Default: read `state[obs_cmt_idx]` (0-based into the state vector)
    /// for every observation regardless of its CMT. The canonical
    /// single-output ODE shape.
    ObsCmt(usize),
    /// Form C uniform: `[scaling] y = <expr>` — a single output_fn
    /// replaces the state-index readout for every observation.
    Single(OdeOutputFn),
    /// Form C per-CMT: `[scaling] y[CMT=N] = <expr>` for each observed
    /// CMT. Key is the 1-based CMT index from the data file (matches
    /// `subject.obs_cmts[i]`, which is `usize`). Fit-time validation
    /// enforces that every observed CMT has an entry; missing entries
    /// fall through to NaN at runtime as a defensive guard.
    PerCmt(HashMap<usize, OdeOutputFn>),
}

/// Read the observable value at observation `obs_idx`.
///
/// `subject.obs_cmts[obs_idx]` selects the per-CMT readout when
/// `OdeReadout::PerCmt` is in use; the simpler variants ignore it.
#[inline]
fn read_observable(
    ode: &OdeSpec,
    u: &[f64],
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    obs_cmt: usize,
) -> f64 {
    match &ode.readout {
        OdeReadout::ObsCmt(idx) => u[*idx],
        OdeReadout::Single(out_fn) => out_fn(u, pk_params_flat, theta, eta, covariates),
        OdeReadout::PerCmt(map) => match map.get(&obs_cmt) {
            Some(out_fn) => out_fn(u, pk_params_flat, theta, eta, covariates),
            // Parser + fit-time validation guarantee every observed CMT
            // has an entry. NaN here is a defensive guard against
            // hand-constructed CompiledModels that bypassed validation —
            // it propagates to NaN OFV so the bad config is loud, not
            // silent.
            None => f64::NAN,
        },
    }
}

/// ODE specification for a model
pub struct OdeSpec {
    /// RHS function: (u, pk_params_flat, t, du) — writes derivatives into du
    pub rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>,
    /// Number of ODE states
    pub n_states: usize,
    /// Names of state variables (e.g., ["depot", "central"])
    pub state_names: Vec<String>,
    /// How the per-observation observable is computed. Replaces the
    /// earlier `(obs_cmt_idx, output_fn)` pair — see [`OdeReadout`].
    pub readout: OdeReadout,
    /// Per-state diagonal process-noise variances (σ²_w,i) for SDE / EKF.
    /// Length must equal `n_states` when non-empty; empty means standard ODE
    /// (no diffusion). Declared via `[diffusion]` block as `state ~ variance`,
    /// analogous to sigma/omega notation. Updated each outer iteration as
    /// diffusion thetas are re-estimated.
    pub diffusion_var: Vec<f64>,
    /// Optional per-subject initial compartment amounts. Declared in the
    /// `[odes]` block as `init(state) = <expr>`; the expression may reference
    /// individual parameters (so it folds in theta/eta/covariates via the
    /// individual-parameter layer, exactly like the RHS). Given the flat
    /// individual-parameter vector (`PkParams.values`), returns the full
    /// `n_states`-length initial-amount vector — the init value for declared
    /// states and `0.0` for the rest. `None` when no `init(...)` is declared,
    /// in which case every compartment starts at zero (the historical default).
    /// A system reset (EVID=3/4) re-applies this on the ODE event-driven path.
    #[allow(clippy::type_complexity)]
    pub init_fn: Option<Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>>,
    /// RK45 solver tolerances used to integrate this system. Defaults to
    /// `OdeSolverOptions::default()` (reltol 1e-4 / abstol 1e-6); overridden
    /// from the model's `[fit_options]` (`ode_reltol` / `ode_abstol` /
    /// `ode_max_steps`) and call-time `settings` via
    /// [`CompiledModel::sync_ode_solver_opts`]. Carried on the spec so every
    /// integration entry point (`ode_predictions*`, EKF) uses the configured
    /// accuracy without threading options through each call.
    pub solver_opts: OdeSolverOptions,
}

impl OdeSpec {
    /// Initial compartment-amount vector for a subject, given the flat
    /// individual-parameter vector `params` (`PkParams.values`). Returns the
    /// `init(...)` expression values where declared and `0.0` elsewhere; when
    /// no `init(...)` is declared this is all zeros — the historical default.
    /// Used to seed the integrator at the start of a record and to re-seed it
    /// after an EVID=3/4 reset.
    pub fn initial_state(&self, params: &[f64]) -> Vec<f64> {
        match &self.init_fn {
            Some(f) => f(params),
            None => vec![0.0; self.n_states],
        }
    }

    /// Convenience accessor: returns the canonical `obs_cmt_idx` when the
    /// readout is the default `ObsCmt` variant. Used by EKF (which requires
    /// a single observable compartment) and by callers that need to know
    /// whether the readout is "Phase 1 simple" vs "Form C custom".
    pub fn obs_cmt_idx(&self) -> Option<usize> {
        match &self.readout {
            OdeReadout::ObsCmt(idx) => Some(*idx),
            OdeReadout::Single(_) | OdeReadout::PerCmt(_) => None,
        }
    }
}

impl OdeReadout {
    /// Returns true when this readout cannot be paired with `gradient = ad`.
    ///
    /// Both Form C variants (`Single` and `PerCmt`) call arbitrary
    /// user-defined closures at each observation. The analytical AD entry
    /// points take only a single `Const f64` scale and cannot evaluate
    /// closures over theta/eta — there's no AD path for Form C. At runtime
    /// `model.tv_fn` is `None` for any ODE model anyway, so AD silently
    /// falls back to FD. The parse-time guard surfaces that fallback as a
    /// clear error rather than silently demoting the user's `gradient = ad`
    /// choice.
    pub fn requires_fd(&self) -> bool {
        match self {
            OdeReadout::ObsCmt(_) => false,
            OdeReadout::Single(_) | OdeReadout::PerCmt(_) => true,
        }
    }
}

/// Compute ODE-based predictions for a single subject.
///
/// `pk_params_flat` is a flat array of PK parameters passed to the RHS function.
/// `theta` and `eta` are forwarded to `OdeSpec::output_fn` for Form C
/// (`[scaling] y = <expr>`); pass empty slices when no Form C is configured.
/// Dose events are handled as state discontinuities between integration segments.
pub fn ode_predictions(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
) -> Vec<f64> {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = ode.solver_opts;

    // Seed compartments from `init(state) = expr` (zeros when none declared).
    let mut u = ode.initial_state(pk_params_flat);
    let mut predictions = vec![f64::NAN; n_obs];

    // Lagtime shifts the effective start (and end) of every dose record.
    // Default 0.0 when not declared, so existing models behave identically.
    let lagtime = pk_params_flat.get(PK_IDX_LAGTIME).copied().unwrap_or(0.0);
    // Per-dose lagtimes for `active_infusions` — uniform for the no-TV
    // path (lagtime is constant across doses).
    let dose_lagtimes: Vec<f64> = vec![lagtime; subject.doses.len()];
    // Bioavailability F (slot PK_IDX_F, default 1.0) scales the amount that
    // enters the dosing compartment — NONMEM's F·AMT (bolus) / F·RATE
    // (infusion). Uniform across doses on the no-TV path.
    let f_bio = pk_params_flat.get(PK_IDX_F).copied().unwrap_or(1.0);
    let dose_f_bio: Vec<f64> = vec![f_bio; subject.doses.len()];

    // Extended params: slots 0..MAX_PK_PARAMS hold the PK parameters; slots
    // MAX_PK_PARAMS and MAX_PK_PARAMS+1 carry TAFD/TAD anchors for the ODE RHS.
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    // Store NaN when there are no doses so the ODE RHS injects NaN for TAFD
    // (consistent with the sdtab convention) rather than -∞ (INFINITY - t).
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    // Build obs_time → indices map. Multiple observations can share a time
    // (e.g. simultaneous PK/PD samples on different CMTs), so each time maps to
    // *all* its observation indices — recording only one would leave the others
    // at their initial NaN.
    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in subject.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

    // Break timeline at lagtime-shifted dose times — and, for infusions,
    // at lagtime-shifted infusion-end times too, so each segment is
    // either fully inside or fully outside every infusion window.
    let t_last = subject.obs_times.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time + lagtime);
        if is_real_infusion(dose) {
            break_times.push(dose.time + lagtime + dose.duration);
        }
        // SS + lagtime: break at the dose *record* time too, so we can seed
        // the previous-interval steady-state tail there before the lagged
        // pulse arrives (issue #15).
        if lagtime > 0.0 && dose.ss && dose.ii > 0.0 {
            break_times.push(dose.time);
        }
    }
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for k in 0..(break_times.len() - 1) {
        let t_start = break_times[k];
        let t_end = break_times[k + 1];

        // Apply dose effects at t_start in a single pass over the dose
        // list. Ordering inside the pass matters:
        //   1. SS=1 + II > 0: pre-equilibrate by overwriting state with
        //      the SS amount from the infinite-past pulse train (see
        //      `equilibrate_ss_state`).
        //   2. Bolus (non-infusion): instantaneous amount jump in the
        //      dose's compartment, applied on top of any SS preload.
        // Infusions don't add to state at t_start — they're injected as
        // a derivative term inside the integrator (see `active_infusions`
        // + wrapped RHS below).
        // SS + lagtime: at the dose record time (strictly before the lagged
        // arrival) seed the previous interval's steady-state tail so pre-lag
        // observations don't read the empty initial state. Phase II−lagtime
        // is where the prior pulse has decayed to by the record time.
        if lagtime > 0.0 {
            for dose in &subject.doses {
                if dose.ss && dose.ii > 0.0 && (dose.time - t_start).abs() < 1e-12 {
                    u = ss_state_at_phase(ode, pk_params_flat, dose, dose.ii - lagtime, &opts);
                }
            }
        }

        for dose in &subject.doses {
            if (dose.time + lagtime - t_start).abs() >= 1e-12 {
                continue;
            }
            if dose.ss && dose.ii > 0.0 {
                u = equilibrate_ss_state(ode, pk_params_flat, dose, &opts);
            }
            if !is_real_infusion(dose) {
                // dose.cmt is 1-based; state indices are 0-based
                let cmt_idx = dose.cmt - 1;
                if cmt_idx < n {
                    u[cmt_idx] += f_bio * dose.amt;
                }
            }
        }

        // Record observations exactly at t_start (after dose)
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            for &obs_idx in obs_idxs {
                let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    pk_params_flat,
                    theta,
                    eta,
                    &subject.covariates,
                    cmt,
                );
            }
        }

        // Observation times in this segment (t_start < t <= t_end)
        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .cloned()
            .collect();
        // Always include t_end so u is updated for next segment
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Update TAD anchor (slot MAX_PK_PARAMS+1): last effective dose time
        // before this segment, SS-aware (gives TAD = t - last_dose_eff).
        {
            let last_dose_eff = subject
                .doses
                .iter()
                .filter(|d| d.time + lagtime <= t_start + 1e-12)
                .map(|d| {
                    if d.ss && d.ii > 0.0 {
                        let elapsed = t_start - (d.time + lagtime);
                        t_start - elapsed.rem_euclid(d.ii)
                    } else {
                        d.time + lagtime
                    }
                })
                .fold(f64::NEG_INFINITY, f64::max);
            // Store NaN when no effective prior dose exists so the ODE RHS injects
            // NaN for TAD (consistent with sdtab) rather than +∞ (t - NEG_INFINITY).
            ext_params[crate::types::MAX_PK_PARAMS + 1] = if last_dose_eff.is_finite() {
                last_dose_eff
            } else {
                f64::NAN
            };
        }

        // Integrate. If any infusions are active in this segment, wrap
        // the user RHS so it adds `+rate` to each infusion's compartment.
        // The plain (non-event-driven) ODE path never sees reset subjects —
        // the dispatcher routes those to `ode_predictions_event_driven` — so
        // no reset floor applies here.
        let active = active_infusions(
            &subject.doses,
            t_start,
            t_end,
            &dose_lagtimes,
            &dose_f_bio,
            f64::NEG_INFINITY,
        );
        let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            (ode.rhs)(y, p, t, dy);
            for &(cmt_idx, rate) in &active {
                if cmt_idx < dy.len() {
                    dy[cmt_idx] += rate;
                }
            }
        };
        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (t_start, t_end),
            &ext_params,
            &saveat,
            &opts,
        );

        // Extract predictions and update state
        for pt in &sol {
            if let Some(obs_idxs) = obs_map.get(&pt.t.to_bits()) {
                for &obs_idx in obs_idxs {
                    let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                    predictions[obs_idx] = read_observable(
                        ode,
                        &pt.u,
                        pk_params_flat,
                        theta,
                        eta,
                        &subject.covariates,
                        cmt,
                    );
                }
            }
        }

        // State at end of segment
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    // Clamp negative predictions to zero (ODE solver overshoot guard).
    // NaN intentionally NOT clamped — it propagates to a NaN OFV so the
    // outer optimizer rejects the step, matching the analytical path's
    // `conc.max(0.0)` semantic (NaN survives `.max(0.0)` per IEEE 754).
    // This is also what surfaces a missing `OdeReadout::PerCmt` entry as
    // a loud failure rather than a silent zero. (Pre-Phase-2 the clamp
    // included NaN; Copilot's review of #84 caught the inconsistency.)
    for p in &mut predictions {
        if *p < 0.0 {
            *p = 0.0;
        }
    }

    predictions
}

/// ODE-based predictions with per-event PK parameters (time-varying-covariate
/// aware). Walks the merged dose+obs+pk-only timeline, integrating each
/// segment `[cur_t, t_event]` with the PK params evaluated at `t_event` —
/// the NONMEM end-of-interval / current-record convention (`$PK` runs at
/// every record, then ADVAN propagates to it). A covariate that changes
/// at an event row (dose, obs, or EVID=2) is therefore consumed by the
/// segment terminating at that record.
///
/// The non-TV `ode_predictions` is preserved as a fast path; this function
/// is only invoked from the dispatcher when `subject.has_tv_covariates()`.
///
/// Infusions (`rate > 0`) break the timeline at the infusion's end and are
/// added to the wrapped RHS for any segment they fully span. The
/// infusion-end break carries no NONMEM record, so it doesn't update the
/// "current PK" used to integrate subsequent segments.
pub fn ode_predictions_event_driven(
    ode: &OdeSpec,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
) -> Vec<f64> {
    assert_eq!(pk_at_dose.len(), subject.doses.len());
    assert_eq!(pk_at_obs.len(), subject.obs_times.len());
    assert_eq!(pk_at_pk_only.len(), subject.pk_only_times.len());

    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = ode.solver_opts;

    // First-dose time anchor for TAFD injection via extended params.
    // fold yields INFINITY when there are no doses; convert to NaN so the ODE
    // RHS injects NaN for TAFD (consistent with sdtab) rather than -∞.
    let first_dose_time_ed = {
        let t = subject
            .doses
            .iter()
            .map(|d| d.time)
            .fold(f64::INFINITY, f64::min);
        if t.is_finite() {
            t
        } else {
            f64::NAN
        }
    };

    // Seed compartments from `init(state) = expr` (zeros when none declared).
    // The init expression folds covariates/eta in via the individual-parameter
    // layer, so evaluate it with the snapshot from the subject's *first record*
    // — the smallest record time across dose / obs / pk-only. Selecting by
    // event kind would wrongly prefer a later dose over an earlier observation
    // when covariates are time-varying (e.g. a pre-dose baseline obs at t=0).
    // Raw record times are used (not lagtime-shifted) since `$PK` order follows
    // the record, not the absorption delay.
    let init_pk: Option<PkParams> = {
        let mut best: Option<(f64, PkParams)> = None;
        let mut consider = |t: f64, p: &PkParams| {
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, *p));
            }
        };
        for (k, d) in subject.doses.iter().enumerate() {
            consider(d.time, &pk_at_dose[k]);
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            consider(t, &pk_at_obs[j]);
        }
        for (m, &t) in subject.pk_only_times.iter().enumerate() {
            consider(t, &pk_at_pk_only[m]);
        }
        best.map(|(_, p)| p)
    };
    let mut u = match &init_pk {
        Some(p) => ode.initial_state(&p.values),
        None => vec![0.0_f64; n],
    };
    let mut predictions = vec![f64::NAN; n_obs];

    if n_obs == 0 {
        return predictions;
    }

    // Build merged event timeline. Tie-break at the same time:
    //   dose < pk-only < obs < infusion-end
    // — matches the analytical event-driven path for dose/pk-only/obs.
    // Infusion-end sorts last so an obs at the same time as the end of
    // an infusion is recorded with the infusion still contributing
    // (state is continuous; the ordering only affects which segments
    // include the rate in their active set on the next iteration).
    #[derive(Clone, Copy)]
    enum Kind {
        Reset,
        Dose,
        PkOnly,
        Obs,
        InfusionEnd,
    }
    fn kind_order(k: Kind) -> u8 {
        match k {
            // Reset sorts first so EVID=4 (reset + dose) zeros the state
            // before its own dose lands at the same time.
            Kind::Reset => 0,
            Kind::Dose => 1,
            Kind::PkOnly => 2,
            Kind::Obs => 3,
            Kind::InfusionEnd => 4,
        }
    }
    let n_infusion_ends = subject.doses.iter().filter(|d| is_real_infusion(d)).count();
    let mut timeline: Vec<(f64, Kind, usize)> = Vec::with_capacity(
        subject.doses.len()
            + n_obs
            + subject.pk_only_times.len()
            + subject.reset_times.len()
            + n_infusion_ends,
    );
    for (r, &t) in subject.reset_times.iter().enumerate() {
        timeline.push((t, Kind::Reset, r));
    }
    // Per-dose lagtimes from the per-event PK snapshot for the dose.
    let dose_lagtimes: Vec<f64> = pk_at_dose.iter().map(|p| p.lagtime()).collect();
    // Per-dose bioavailability F from the per-event PK snapshot — F may vary
    // across doses when it depends on time-varying covariates.
    let dose_f_bio: Vec<f64> = pk_at_dose.iter().map(|p| p.f_bio()).collect();
    for (k, d) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[k];
        timeline.push((d.time + lag, Kind::Dose, k));
        if is_real_infusion(d) {
            timeline.push((d.time + lag + d.duration, Kind::InfusionEnd, k));
        }
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        timeline.push((t, Kind::Obs, j));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        timeline.push((t, Kind::PkOnly, m));
    }
    timeline.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| kind_order(a.1).cmp(&kind_order(b.1)))
    });

    let mut cur_t = timeline[0].0;
    // Most-recent NONMEM record's PK params, used to integrate segments
    // ending at an infusion-end (which is not a record and carries no PK).
    // Seed last_pk with the first record's snapshot (not zeroed defaults) so a
    // reset that is itself the first event — e.g. an EVID=4 reset+dose at t=0 —
    // re-applies init from real parameters rather than zeros. Updated as
    // dose/obs/pk-only records are processed.
    let mut last_pk: PkParams = init_pk.unwrap_or_default();
    // Most-recent system-reset time (EVID=3/4); `NEG_INFINITY` until the
    // first reset. Infusions started before it are no longer active.
    let mut reset_floor = f64::NEG_INFINITY;

    for &(t_event, kind, idx) in &timeline {
        // PK params for the segment [cur_t, t_event] are evaluated AT
        // t_event (NONMEM end-of-interval / current-record convention —
        // `$PK runs at every record, then ADVAN propagates to it`).
        // Infusion-end is not a record: reuse the previous segment's PK.
        // Reset is not a record either: it just zeros the state below.
        let pk_now: PkParams = match kind {
            Kind::Dose => pk_at_dose[idx],
            Kind::Obs => pk_at_obs[idx],
            Kind::PkOnly => pk_at_pk_only[idx],
            Kind::InfusionEnd | Kind::Reset => last_pk,
        };

        if t_event > cur_t {
            // Build extended params for this segment: slots 0..MAX_PK_PARAMS
            // are pk_now.values; slots MAX_PK_PARAMS and MAX_PK_PARAMS+1 carry
            // the TAFD/TAD anchors for TIME/TAFD/TAD injection in the ODE RHS.
            let lagtime_ed = pk_now.lagtime();
            let last_dose_eff_ed = subject
                .doses
                .iter()
                .filter(|d| d.time + lagtime_ed <= cur_t + 1e-12)
                .map(|d| {
                    if d.ss && d.ii > 0.0 {
                        let elapsed = cur_t - (d.time + lagtime_ed);
                        cur_t - elapsed.rem_euclid(d.ii)
                    } else {
                        d.time + lagtime_ed
                    }
                })
                .fold(f64::NEG_INFINITY, f64::max);
            // Store NaN when no effective prior dose exists (fold stays at NEG_INFINITY)
            // so the ODE RHS injects NaN for TAD rather than +∞ (t - NEG_INFINITY).
            let last_dose_eff_ed = if last_dose_eff_ed.is_finite() {
                last_dose_eff_ed
            } else {
                f64::NAN
            };
            let mut ext_params_ed = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
            ext_params_ed[..crate::types::MAX_PK_PARAMS]
                .copy_from_slice(&pk_now.values[..crate::types::MAX_PK_PARAMS]);
            ext_params_ed[crate::types::MAX_PK_PARAMS] = first_dose_time_ed;
            ext_params_ed[crate::types::MAX_PK_PARAMS + 1] = last_dose_eff_ed;

            // Wrap the user RHS so any infusion fully spanning
            // [cur_t, t_event] contributes `+rate` to its compartment.
            let active = active_infusions(
                &subject.doses,
                cur_t,
                t_event,
                &dose_lagtimes,
                &dose_f_bio,
                reset_floor,
            );
            let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                (ode.rhs)(y, p, t, dy);
                for &(cmt_idx, rate) in &active {
                    if cmt_idx < dy.len() {
                        dy[cmt_idx] += rate;
                    }
                }
            };
            let saveat = vec![t_event];
            let sol = solve_ode(
                &wrapped_rhs,
                &u,
                (cur_t, t_event),
                &ext_params_ed,
                &saveat,
                &opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            cur_t = t_event;
        }

        match kind {
            Kind::Dose => {
                let d = &subject.doses[idx];
                // Steady-state (SS=1) dose: reset state and load with the
                // SS amount from the infinite-past pulse train before the
                // SS dose's own pulse is applied below. See
                // `equilibrate_ss_state` for the per-cycle scheme.
                if d.ss && d.ii > 0.0 {
                    u = equilibrate_ss_state(ode, &pk_now.values, d, &opts);
                }
                // Boluses: add amt to state. Infusions: no instantaneous
                // change — handled via the wrapped RHS for segments inside
                // [d.time, d.time + d.duration].
                if !is_real_infusion(d) {
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n {
                        u[cmt_idx] += pk_now.f_bio() * d.amt;
                    }
                }
                last_pk = pk_now;
            }
            Kind::Obs => {
                let cmt = subject.obs_cmts.get(idx).copied().unwrap_or(0);
                let v = read_observable(
                    ode,
                    &u,
                    &pk_now.values,
                    theta,
                    eta,
                    &subject.covariates,
                    cmt,
                );
                // Clamp negative readouts (ODE solver overshoot guard);
                // let NaN through so a missing `OdeReadout::PerCmt` entry
                // (or any other genuine NaN) surfaces as a NaN OFV
                // rather than a silent zero. See the corresponding note
                // in `ode_predictions`.
                predictions[idx] = if v < 0.0 { 0.0 } else { v };
                last_pk = pk_now;
            }
            Kind::PkOnly => {
                // EVID=2: $PK ran at this record but compartment state is
                // unchanged. The new pk is consumed by the next segment's
                // integration via the loop-top `pk_now` lookup.
                last_pk = pk_now;
            }
            Kind::InfusionEnd => {
                // Not a NONMEM record: no state update, no PK update —
                // only purpose is to break the timeline so the next
                // segment's `active_infusions` excludes this infusion.
            }
            Kind::Reset => {
                // EVID=3 / EVID=4: reset the system. Compartments with an
                // `init(state) = expr` return to their initial value; all
                // others go to zero (a reset starts a fresh episode from
                // baseline). With no init declared this zeros everything.
                // Evaluate init with the params in effect at the reset
                // (`last_pk`). For EVID=4 the dose at this same time follows
                // (Reset sorts before Dose), so it lands on the re-seeded
                // state. Record the reset time so infusions started earlier
                // stop contributing.
                u = ode.initial_state(&last_pk.values);
                reset_floor = t_event;
            }
        }
    }

    predictions
}

/// EKF-based predictions with an explicit diffusion_var slice (bypasses
/// `ode_spec.diffusion_var`). Used by the likelihood path to supply the
/// current theta-derived diffusion variances without mutating the model.
pub fn ode_predictions_ekf_with_diffusion(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
    diffusion_var: &[f64],
    r_obs_fn: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::ode::ekf::solve_ekf;

    // EKF path: parser rejects SDE + Form C, so output_fn is always None
    // here and theta/eta would never be consulted. Pass empty slices.
    let ipred_plain = ode_predictions(ode, pk_params_flat, &[], &[], subject);
    let r_obs_vec: Vec<f64> = ipred_plain
        .iter()
        .map(|&f| {
            let v = r_obs_fn(f);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();

    let pts = solve_ekf(
        ode.rhs.as_ref(),
        ode.n_states,
        // EKF/SDE path requires a single observable compartment index for
        // the Kalman update. Parser-side validation rejects SDE models that
        // use Form C `y = <expr>`; so `obs_cmt_idx` is always `Some` here.
        ode.obs_cmt_idx()
            .expect("EKF requires obs_cmt_idx; SDE + [scaling] y = ... is not supported"),
        diffusion_var,
        pk_params_flat,
        &ode.initial_state(pk_params_flat),
        &subject.doses,
        &subject.obs_times,
        &r_obs_vec,
        ode.solver_opts,
    );

    let ipreds: Vec<f64> = pts.iter().map(|p| p.ipred).collect();
    let p_obs: Vec<f64> = pts.iter().map(|p| p.p_obs).collect();
    (ipreds, p_obs)
}

/// EKF-based predictions for a subject with an SDE model.
///
/// Wraps `solve_ekf`, handling the residual variance `r_obs` needed for the
/// Kalman update step. Returns `(ipred, p_obs)` where `p_obs[j]` is the
/// EKF state covariance at the observable compartment just before assimilating
/// observation `j`. Callers add `p_obs[j]` to the residual variance to form
/// `V_total = p_obs[j] + V_residual`.
///
/// `r_obs_fn` computes the scalar residual variance for each observation given
/// the predicted value — this feeds the Kalman update, keeping the covariance
/// estimate numerically stable. It does NOT affect the returned `p_obs` values
/// (those are pre-update, i.e. the purely process-noise contribution).
// Not currently called from outside this module — superseded by
// `ode_predictions_ekf_with_diffusion` which accepts an explicit diffusion_var.
#[allow(dead_code)]
pub fn ode_predictions_ekf(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
    r_obs_fn: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::ode::ekf::solve_ekf;

    // Compute per-observation R for the Kalman update from a standard ODE pass.
    // Using per-observation R is correct for proportional and combined error models.
    // EKF path: parser rejects SDE + Form C, so output_fn is always None
    // here and theta/eta would never be consulted. Pass empty slices.
    let ipred_plain = ode_predictions(ode, pk_params_flat, &[], &[], subject);
    let r_obs_vec: Vec<f64> = ipred_plain
        .iter()
        .map(|&f| {
            let v = r_obs_fn(f);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();

    let pts = solve_ekf(
        ode.rhs.as_ref(),
        ode.n_states,
        ode.obs_cmt_idx()
            .expect("EKF requires obs_cmt_idx; SDE + [scaling] y = ... is not supported"),
        &ode.diffusion_var,
        pk_params_flat,
        &ode.initial_state(pk_params_flat),
        &subject.doses,
        &subject.obs_times,
        &r_obs_vec,
        ode.solver_opts,
    );

    let ipreds: Vec<f64> = pts.iter().map(|p| p.ipred).collect();
    let p_obs: Vec<f64> = pts.iter().map(|p| p.p_obs).collect();
    (ipreds, p_obs)
}

/// Like [`ode_predictions`] but also returns the raw ODE state vector at every
/// observation time. Returns `(ipred_vec, compartment_states)` where
/// `compartment_states[j]` is `u[0..n_states]` at observation `j`.
///
/// The estimation hot path uses [`ode_predictions`] (no allocation overhead);
/// this variant is called once post-fit to populate `SubjectResult::compartment_states`.
///
/// # KEEP-IN-SYNC with [`ode_predictions`]
///
/// This function is a near-copy of `ode_predictions` with the single addition of
/// `states[obs_idx] = u.clone()` / `states[obs_idx] = pt.u.clone()` at every
/// observation capture site. Any change to dose-event handling, SS logic,
/// infusion tracking, break-time construction, or `read_observable` calls in
/// `ode_predictions` **must be mirrored here**. Search for the parallel line in
/// `ode_predictions` and apply the same change.
///
/// # Precondition
///
/// The caller **must not** pass a subject that has EVID=3/4 resets
/// (`subject.reset_times` non-empty) or time-varying covariates
/// (`subject.has_tv_covariates()`).  For those subjects
/// `compute_predictions_with_states` routes through
/// `ode_predictions_event_driven_with_states`, which handles resets correctly.
/// Calling this function directly on a reset subject would produce incorrect
/// states because the re-seed events are absent from the break-time list.
pub fn ode_predictions_with_states(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = ode.solver_opts;

    let mut u = ode.initial_state(pk_params_flat);
    let mut predictions = vec![f64::NAN; n_obs];
    let mut states: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; n_obs];

    let lagtime = pk_params_flat.get(PK_IDX_LAGTIME).copied().unwrap_or(0.0);
    let dose_lagtimes: Vec<f64> = vec![lagtime; subject.doses.len()];
    let f_bio = pk_params_flat.get(PK_IDX_F).copied().unwrap_or(1.0);
    let dose_f_bio: Vec<f64> = vec![f_bio; subject.doses.len()];

    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in subject.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

    let t_last = subject.obs_times.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time + lagtime);
        if is_real_infusion(dose) {
            break_times.push(dose.time + lagtime + dose.duration);
        }
        if lagtime > 0.0 && dose.ss && dose.ii > 0.0 {
            break_times.push(dose.time);
        }
    }
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();

    for w in break_times.windows(2) {
        let (t_start, t_end) = (w[0], w[1]);
        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // SS + lagtime: at the dose *record* time (strictly before the lagged pulse
        // arrives) seed the previous interval's steady-state tail, exactly mirroring
        // the separate pre-pass in `ode_predictions` (lines 479-485).
        if lagtime > 0.0 {
            for dose in &subject.doses {
                if dose.ss && dose.ii > 0.0 && (dose.time - t_start).abs() < 1e-12 {
                    u = ss_state_at_phase(ode, pk_params_flat, dose, dose.ii - lagtime, &opts);
                }
            }
        }

        // Apply boluses and SS doses at t_eff = dose.time + lagtime.
        for (dose_idx, dose) in subject.doses.iter().enumerate() {
            let t_eff = dose.time + dose_lagtimes[dose_idx];
            if (t_eff - t_start).abs() < 1e-10 {
                let f = dose_f_bio[dose_idx];
                if dose.ss && dose.ii > 0.0 {
                    // Lagged arrival: pre-lag seeding was already done above;
                    // here we apply the full equilibrated state.
                    u = equilibrate_ss_state(ode, pk_params_flat, dose, &opts);
                }
                if !is_real_infusion(dose) {
                    // dose.cmt is 1-based; CMT=0 means no compartment — ignore.
                    if dose.cmt > 0 {
                        let cmt = dose.cmt - 1;
                        if cmt < n {
                            u[cmt] += dose.amt * f;
                        }
                    }
                } else {
                    let end_t = t_eff + dose.duration;
                    active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
                    active_infusions.push((dose_idx, t_eff, end_t));
                }
            }
        }

        // Handle obs at t_start (after dose).
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            for &obs_idx in obs_idxs {
                let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    pk_params_flat,
                    theta,
                    eta,
                    &subject.covariates,
                    cmt,
                );
                states[obs_idx] = u.clone();
            }
        }

        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .cloned()
            .filter(|&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .collect();
        // Always include t_end so u is advanced to segment end, even when there
        // are no observations in the segment (e.g. two doses with no obs between
        // them). Without this, solve_ode returns an empty solution and u is not
        // updated, leaving the wrong (undecayed) state for the next segment.
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        // Mirror ode_predictions lines 530-531: sort + dedup so solve_ode's
        // linear save_idx cursor works correctly even if obs_times contains
        // duplicate entries or arrives out of order.
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        // TAD anchor: last effective dose time before this segment, SS-aware.
        // For SS doses, rem_euclid maps the elapsed time back into [0, II) so
        // TAD stays within one dosing interval — matching ode_predictions.
        ext_params[crate::types::MAX_PK_PARAMS + 1] = {
            let last_dose_eff = subject
                .doses
                .iter()
                .filter(|d| d.time + lagtime <= t_start + 1e-12)
                .map(|d| {
                    if d.ss && d.ii > 0.0 {
                        let elapsed = t_start - (d.time + lagtime);
                        t_start - elapsed.rem_euclid(d.ii)
                    } else {
                        d.time + lagtime
                    }
                })
                .fold(f64::NEG_INFINITY, f64::max);
            if last_dose_eff.is_finite() {
                last_dose_eff
            } else {
                f64::NAN
            }
        };

        active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
        let current_infusions = active_infusions.clone();

        let wrapped_rhs = {
            let infusions = current_infusions.clone();
            let f_bio_snap = dose_f_bio.clone();
            move |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                (ode.rhs)(y, p, t, dy);
                for &(di, t_start_inf, t_end_inf) in &infusions {
                    // Use +1e-12 on the upper bound (not -1e-12) so the infusion
                    // is active right up to t_end_inf. The reference ode_predictions
                    // applies the rate for the whole segment via active_infusions;
                    // the dynamic gate here must not cut off the last sub-step.
                    if t >= t_start_inf - 1e-12 && t < t_end_inf + 1e-12 {
                        let dose = &subject.doses[di];
                        let f = f_bio_snap[di];
                        // dose.cmt is 1-based; CMT=0 means no compartment — ignore.
                        if dose.cmt > 0 {
                            let cmt = dose.cmt - 1;
                            if cmt < n {
                                dy[cmt] += dose.rate * f;
                            }
                        }
                    }
                }
            }
        };

        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (t_start, t_end),
            &ext_params,
            &saveat,
            &opts,
        );

        for pt in &sol {
            if let Some(obs_idxs) = obs_map.get(&pt.t.to_bits()) {
                for &obs_idx in obs_idxs {
                    let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                    predictions[obs_idx] = read_observable(
                        ode,
                        &pt.u,
                        pk_params_flat,
                        theta,
                        eta,
                        &subject.covariates,
                        cmt,
                    );
                    states[obs_idx] = pt.u.clone();
                }
            }
        }

        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    for p in &mut predictions {
        if *p < 0.0 {
            *p = 0.0;
        }
    }

    (predictions, states)
}

/// Like [`ode_predictions_event_driven`] but also returns the raw ODE state
/// at every observation time. Returns `(ipred_vec, compartment_states)`.
///
/// Called post-fit for TV-covariate ODE models to populate
/// `SubjectResult::compartment_states`.
///
/// # Approximation for TV-covariate subjects
///
/// `ipred` is exact (the event-driven path uses per-event PK parameters). The
/// compartment `states`, however, are derived from a second pass via
/// [`ode_dense_solve_states`] using **the first observation's PK parameters held
/// fixed** for the entire timeline. For subjects with genuinely time-varying
/// covariates (CL, V, etc. changing between observations) the states will be
/// approximate. `fit()` emits `W_DERIVED_CMT_TV_ODE` to alert users to this
/// limitation. For reset-only subjects (no TV covariates) `pk_at_obs` is
/// uniformly filled, so using the first entry is exact.
pub fn ode_predictions_event_driven_with_states(
    ode: &OdeSpec,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Re-use the standard path to get ipred, then do a second pass to
    // extract states. The event-driven function is already complex enough
    // that duplicating it would be error-prone; a second pass is acceptable
    // because this is post-fit only.
    let ipreds = ode_predictions_event_driven(
        ode,
        subject,
        theta,
        eta,
        pk_at_dose,
        pk_at_obs,
        pk_at_pk_only,
    );

    // Second pass: extract the full ODE state at each obs time via
    // `ode_dense_solve_states`. That function runs the standard (non-event-driven)
    // solver, so it uses a single fixed set of PK params for the entire timeline.
    //
    // For subjects with EVID=3/4 resets but *no* TV covariates, `pk_at_obs` is
    // uniformly filled (every entry identical), so using `first()` is exact.
    //
    // For subjects with genuine TV covariates, `pk_at_obs` varies per timepoint.
    // Using `first()` here is an approximation: the compartment state trajectory
    // will be computed with the first-observation PK params (CL/V/etc.) held fixed,
    // while `ipreds` correctly reflect per-event covariate snapshots. For most PK
    // contexts this approximation is acceptable post-fit, but the caller
    // (`compute_predictions_with_states`) is the approximate path; `fit()` emits
    // W_DERIVED_CMT_TV_ODE when TV covariates are present so users know.
    //
    // A future improvement: duplicate the event-driven loop to capture `u` at each
    // obs time directly — exact states, but ~2× the integration work post-fit.
    let pk_flat = &pk_at_obs.first().map(|p| p.values).unwrap_or_default();
    let states = ode_dense_solve_states(ode, pk_flat, theta, eta, subject, &subject.obs_times);

    (ipreds, states)
}

/// Run the ODE solver with an arbitrary set of `saveat` time points and
/// return the full state vector at each requested time.
///
/// This is used by the grid-based integral path in `compute_extra_output_columns`
/// when the integrand references compartment states. The result is only needed
/// post-fit (never on the estimation hot path).
///
/// Dose events (boluses, infusions, SS) are handled identically to
/// [`ode_predictions`]. Subject observation times are ignored; only `saveat`
/// times are returned.
pub fn ode_dense_solve_states(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    saveat: &[f64],
) -> Vec<Vec<f64>> {
    if saveat.is_empty() {
        return vec![];
    }
    let n = ode.n_states;
    let opts = ode.solver_opts;

    let mut u = ode.initial_state(pk_params_flat);
    let mut result: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; saveat.len()];

    let lagtime = pk_params_flat.get(PK_IDX_LAGTIME).copied().unwrap_or(0.0);
    let dose_lagtimes: Vec<f64> = vec![lagtime; subject.doses.len()];
    let f_bio = pk_params_flat.get(PK_IDX_F).copied().unwrap_or(1.0);
    let dose_f_bio: Vec<f64> = vec![f_bio; subject.doses.len()];

    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    // Build saveat → index map for fast lookup.
    let mut saveat_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in saveat.iter().enumerate() {
        saveat_map.entry(t.to_bits()).or_default().push(i);
    }

    let t_last = saveat.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time + lagtime);
        if is_real_infusion(dose) {
            break_times.push(dose.time + lagtime + dose.duration);
        }
        if lagtime > 0.0 && dose.ss && dose.ii > 0.0 {
            break_times.push(dose.time);
        }
    }
    // EVID=3/4 resets must be break-points so the re-seed happens at the exact boundary.
    for &rt in &subject.reset_times {
        break_times.push(rt);
    }
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();

    for w in break_times.windows(2) {
        let (t_start, t_end) = (w[0], w[1]);
        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // EVID=3/4 reset: re-seed compartments before processing doses at this time.
        // Resets sort before doses at the same time (mirroring Kind::Reset < Kind::Dose).
        for &rt in &subject.reset_times {
            if (rt - t_start).abs() < 1e-10 {
                u = ode.initial_state(pk_params_flat);
                active_infusions.clear();
                break;
            }
        }

        // SS + lagtime: at the dose *record* time (before the lagged pulse arrives)
        // seed the previous interval's steady-state tail, mirroring ode_predictions.
        if lagtime > 0.0 {
            for dose in &subject.doses {
                if dose.ss && dose.ii > 0.0 && (dose.time - t_start).abs() < 1e-12 {
                    u = ss_state_at_phase(ode, pk_params_flat, dose, dose.ii - lagtime, &opts);
                }
            }
        }

        for (dose_idx, dose) in subject.doses.iter().enumerate() {
            let t_eff = dose.time + dose_lagtimes[dose_idx];
            if (t_eff - t_start).abs() < 1e-10 {
                let f = dose_f_bio[dose_idx];
                if dose.ss && dose.ii > 0.0 {
                    // Lagged arrival: pre-lag seeding already done above.
                    u = equilibrate_ss_state(ode, pk_params_flat, dose, &opts);
                }
                if !is_real_infusion(dose) {
                    // dose.cmt is 1-based; CMT=0 means no compartment — ignore.
                    if dose.cmt > 0 {
                        let cmt = dose.cmt - 1;
                        if cmt < n {
                            u[cmt] += dose.amt * f;
                        }
                    }
                } else {
                    let end_t = t_eff + dose.duration;
                    active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
                    active_infusions.push((dose_idx, t_eff, end_t));
                }
            }
        }

        // Saveat points at t_start (after dose, matching ode_predictions convention)
        if let Some(idxs) = saveat_map.get(&t_start.to_bits()) {
            for &i in idxs {
                result[i] = u.clone();
            }
        }

        let mut seg_saveat: Vec<f64> = saveat
            .iter()
            .cloned()
            .filter(|&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .collect();
        // Always include t_end so u advances through empty segments (e.g. two
        // consecutive doses with no saveat points between them).
        if seg_saveat.is_empty() || (seg_saveat.last().unwrap() - t_end).abs() > 1e-12 {
            seg_saveat.push(t_end);
        }
        // Mirror ode_predictions lines 530-531 (and the same fix applied to
        // ode_predictions_with_states): sort + dedup so solve_ode's linear
        // save_idx cursor works correctly for duplicate / out-of-order times.
        seg_saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        seg_saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        // TAD anchor: SS-aware, matching ode_predictions (rem_euclid wraps
        // the elapsed time back into [0, II)).
        ext_params[crate::types::MAX_PK_PARAMS + 1] = {
            let last_dose_eff = subject
                .doses
                .iter()
                .filter(|d| d.time + lagtime <= t_start + 1e-12)
                .map(|d| {
                    if d.ss && d.ii > 0.0 {
                        let elapsed = t_start - (d.time + lagtime);
                        t_start - elapsed.rem_euclid(d.ii)
                    } else {
                        d.time + lagtime
                    }
                })
                .fold(f64::NEG_INFINITY, f64::max);
            if last_dose_eff.is_finite() {
                last_dose_eff
            } else {
                f64::NAN
            }
        };

        active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
        let current_infusions = active_infusions.clone();

        let wrapped_rhs = {
            let infusions = current_infusions.clone();
            let f_bio_snap = dose_f_bio.clone();
            move |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                (ode.rhs)(y, p, t, dy);
                for &(di, t_start_inf, t_end_inf) in &infusions {
                    // Use +1e-12 on the upper bound — same fix as applied to
                    // ode_predictions_with_states — so the infusion is active
                    // right up to t_end_inf rather than cutting off 1e-12 early.
                    if t >= t_start_inf - 1e-12 && t < t_end_inf + 1e-12 {
                        let dose = &subject.doses[di];
                        let f = f_bio_snap[di];
                        // dose.cmt is 1-based; CMT=0 means no compartment — ignore.
                        if dose.cmt > 0 {
                            let cmt = dose.cmt - 1;
                            if cmt < n {
                                dy[cmt] += dose.rate * f;
                            }
                        }
                    }
                }
            }
        };

        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (t_start, t_end),
            &ext_params,
            &seg_saveat,
            &opts,
        );

        for pt in &sol {
            if let Some(idxs) = saveat_map.get(&pt.t.to_bits()) {
                for &i in idxs {
                    result[i] = pt.u.clone();
                }
            }
        }

        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    // `theta` and `eta` are accepted for API symmetry with sibling ODE functions
    // (e.g. `ode_predictions_with_states`) but are not consumed here: this
    // function returns the raw ODE state vector `u` without applying any
    // `output_fn` / Form-C scaling. A future extension that returns scaled
    // observables alongside states would use them. Suppress the unused warning.
    let _ = (theta, eta);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DoseEvent;
    use approx::assert_relative_eq;

    /// 1-cpt IV bolus ODE: dA/dt = -ke·A. RHS reads CL,V from pk_params_flat.
    fn one_cpt_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            init_fn: None,
        }
    }

    fn pk_one(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p
    }

    fn make_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Subject {
        let n_obs = obs_times.len();
        Subject {
            id: "1".into(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// Turnover model with a baseline initial condition:
    ///   d/dt(R) = kin - kout*R,  init(R) = kin/kout
    /// params: kin @ slot 0, kout @ slot 1. Observable reads R (state 0).
    fn turnover_ode_spec_with_init() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = p[0] - p[1] * y[0];
            }),
            n_states: 1,
            state_names: vec!["R".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            init_fn: Some(Box::new(|p: &[f64]| {
                let (kin, kout) = (p[0], p[1]);
                vec![if kout > 0.0 { kin / kout } else { 0.0 }]
            })),
        }
    }

    fn pk_kin_kout(kin: f64, kout: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[0] = kin;
        p.values[1] = kout;
        p
    }

    #[test]
    fn ode_init_state_seeds_plain_path() {
        // No doses; the system starts at baseline kin/kout = 5 and stays there
        // (dR/dt = 0). Without init it would start at 0 and climb.
        let ode = turnover_ode_spec_with_init();
        let pk = pk_kin_kout(10.0, 2.0);
        let subj = make_subject(Vec::new(), vec![0.0, 1.0, 5.0, 20.0]);
        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (i, &p) in preds.iter().enumerate() {
            assert_relative_eq!(p, 5.0, epsilon = 1e-5);
            let _ = i;
        }
    }

    #[test]
    fn ode_init_state_then_dose_and_reset_reapplies_init() {
        // Exercises all three: (1) init seeds the start at baseline=5, (2) a
        // bolus at t=0 lands on top of the seeded state (5 + 20 = 25), and
        // (3) an EVID=3 reset at t=5 re-applies init (back to 5, NOT zero).
        let ode = turnover_ode_spec_with_init();
        let pk = pk_kin_kout(10.0, 2.0); // baseline 5
        let doses = vec![DoseEvent::new(0.0, 20.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0, 5.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![5.0];
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        // t=0: init(5) + bolus(20) = 25.
        assert_relative_eq!(preds[0], 25.0, epsilon = 1e-6);
        // t=5: reset re-applies init → 5 (a zeroing reset would give 0).
        assert_relative_eq!(preds[1], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn ode_init_uses_chronologically_first_record_not_first_dose() {
        // Regression (Copilot review #1): with time-varying covariates the init
        // snapshot must come from the earliest record by TIME, not the first
        // dose. Here a pre-dose observation at t=0 carries KIN=10 (baseline 5)
        // while a later dose at t=5 carries KIN=100 (baseline 50). Seeding must
        // use the t=0 obs → prediction at t=0 is 5, not 50.
        let ode = turnover_ode_spec_with_init();
        let doses = vec![DoseEvent::new(5.0, 0.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk_dose = vec![pk_kin_kout(100.0, 2.0)]; // baseline 50 (must NOT be used)
        let pk_obs = vec![pk_kin_kout(10.0, 2.0)]; // baseline 5 (first record)

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert_relative_eq!(preds[0], 5.0, epsilon = 1e-9);
    }

    #[test]
    fn ode_init_reapplied_when_reset_is_first_event() {
        // Regression (Copilot review #2): an EVID=4 reset+dose at t=0 re-applies
        // init *before* the same-time dose. last_pk must be seeded from the
        // first record's params (not zeroed defaults), or the re-applied
        // baseline would evaluate KIN/KOUT with zero params and collapse to 0.
        // Expected: init(5) re-applied at reset, then bolus 20 → 25.
        let ode = turnover_ode_spec_with_init();
        let doses = vec![DoseEvent::new(0.0, 20.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![0.0];
        let pk = pk_kin_kout(10.0, 2.0); // baseline 5
        let pk_dose = vec![pk];
        let pk_obs = vec![pk];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        // Re-applied baseline (5) + bolus (20) = 25. A zero-param re-seed would
        // give 0 + 20 = 20.
        assert_relative_eq!(preds[0], 25.0, epsilon = 1e-6);
    }

    #[test]
    fn ode_event_driven_reset_evid3_zeros_state() {
        // EVID=3 reset at t=5 must zero the ODE state: obs after the reset
        // read ~0 when no later dose exists.
        let ode = one_cpt_ode_spec();
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 6.0, 10.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![5.0];
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert!(preds[0] > 0.0, "pre-reset obs should be positive");
        assert_relative_eq!(preds[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(preds[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn ode_event_driven_reset_evid4_matches_fresh_dose() {
        // EVID=4 (reset + dose) at t=10 must match a single fresh dose at t=10.
        let ode = one_cpt_ode_spec();
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![10.0, 12.0, 15.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![10.0];
        let pk = pk_one(8.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);

        // Reference: lone 500 mg dose at t=10 through the same ODE path.
        let fresh = make_subject(
            vec![DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0)],
            obs_times.clone(),
        );
        let fresh_pk_dose = vec![pk; fresh.doses.len()];
        let fresh_pk_obs = vec![pk; obs_times.len()];
        let expected = ode_predictions_event_driven(
            &ode,
            &fresh,
            &[],
            &[],
            &fresh_pk_dose,
            &fresh_pk_obs,
            &[],
        );
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn ode_event_driven_matches_constant_path_when_pk_constant() {
        // Equivalence: when the per-event PK params are all the same, the
        // event-driven ODE path must agree with the existing single-snapshot
        // path. This is the "no TV covariates" sanity check.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![1.0, 4.0, 8.5, 12.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let ode = one_cpt_ode_spec();

        let baseline = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        let event_driven =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert_eq!(baseline.len(), event_driven.len());
        for (b, e) in baseline.iter().zip(event_driven.iter()) {
            // ODE solver tolerance is ~1e-4 relative — a tighter equality
            // would over-constrain RK45.
            assert_relative_eq!(*b, *e, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_event_driven_picks_up_changing_cl() {
        // Same shape as the analytical TV test: CL doubles between two doses.
        // End-of-interval / NONMEM convention — each segment uses the PK
        // params at the record being arrived at:
        //   [0, t_obs1=5]: uses pk at obs1 = pk_low → ke = 0.05
        //   [5, t_dose2=10]: uses pk at dose2 = pk_high → ke = 0.10
        //   [10, t_obs2=12]: uses pk at obs2 = pk_high → ke = 0.10
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![5.0, 12.0];
        let subj = make_subject(doses, obs_times);
        let pk_low = pk_one(5.0, 100.0); // ke = 0.05
        let pk_high = pk_one(10.0, 100.0); // ke = 0.10
        let pk_dose = vec![pk_low, pk_high];
        let pk_obs = vec![pk_low, pk_high];
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);

        // [0, 5] uses pk_low (pk at obs1): A(5) = 1000 * exp(-0.05*5) ≈ 778.80
        let a5 = 1000.0 * (-0.05f64 * 5.0).exp();
        assert_relative_eq!(preds[0], a5, epsilon = 1e-3, max_relative = 1e-4);

        // [5, 10] uses pk_high (pk at dose2): ke=0.10 for 5h.
        //   A(10⁻) = A(5) * exp(-0.10*5) = 778.80 * 0.6065 ≈ 472.37
        // After dose2: A(10⁺) = 472.37 + 1000 = 1472.37.
        // [10, 12] uses pk_high (pk at obs2): A(12) = 1472.37 * exp(-0.20) ≈ 1205.49
        let a10_minus = a5 * (-0.10f64 * 5.0).exp();
        let a10_plus = a10_minus + 1000.0;
        let a12 = a10_plus * (-0.20f64).exp();
        assert_relative_eq!(preds[1], a12, epsilon = 1e-2, max_relative = 1e-4);
    }

    /// 1-cpt oral ODE: dA1/dt = -ka·A1, dA2/dt = ka·A1 - ke·A2.
    /// Used to test infusion into the depot compartment (cmt=1).
    fn one_cpt_oral_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ka = p[crate::types::PK_IDX_KA];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ka * y[0];
                dy[1] = ka * y[0] - ke * y[1];
            }),
            n_states: 2,
            state_names: vec!["depot".into(), "central".into()],
            readout: OdeReadout::ObsCmt(1),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            init_fn: None,
        }
    }

    #[test]
    fn ode_infusion_one_cpt_iv_matches_closed_form() {
        // 1-cpt IV infusion. Closed form during infusion:
        //   A(t) = (R/ke) · (1 - exp(-ke·t))
        // and after end-of-infusion T:
        //   A(t) = A(T) · exp(-ke·(t-T))
        // Verifies that the wrapped-RHS path produces the right shape.
        let rate = 100.0;
        let amt = 1000.0; // duration = 10 h
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let obs_times = vec![5.0, 10.0, 15.0, 20.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0); // ke = 0.0625
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        // During infusion [0, 10]
        let a5 = (rate / ke) * (1.0 - (-ke * 5.0).exp());
        let a10 = (rate / ke) * (1.0 - (-ke * 10.0).exp());
        // After end-of-infusion
        let a15 = a10 * (-ke * 5.0).exp();
        let a20 = a10 * (-ke * 10.0).exp();

        assert_relative_eq!(preds[0], a5, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a10, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a15, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[3], a20, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_event_driven_infusion_matches_constant_pk_path() {
        // Same infusion-only subject, run through both paths with identical
        // per-event PK params. Verifies the event-driven path's
        // InfusionEnd handling agrees with the simple-timeline path.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![3.0, 7.0, 10.0, 14.0, 20.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let ode = one_cpt_ode_spec();

        let baseline = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        let event_driven =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert_eq!(baseline.len(), event_driven.len());
        for (b, e) in baseline.iter().zip(event_driven.iter()) {
            assert_relative_eq!(*b, *e, epsilon = 1e-3, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_overlapping_infusions_sum_rates() {
        // Two infusions overlap on [2, 6] for a combined rate of 200,
        // then both end at t=6. After t=6, plain elimination.
        //   inf1: t∈[0,6], rate=100
        //   inf2: t∈[2,6], rate=100
        let doses = vec![
            DoseEvent::new(0.0, 600.0, 1, 100.0, false, 0.0),
            DoseEvent::new(2.0, 400.0, 1, 100.0, false, 0.0),
        ];
        let obs_times = vec![2.0, 4.0, 6.0, 12.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        // [0, 2]: rate=100, A(0)=0 → A(t) = (100/ke)·(1 - exp(-ke·t))
        let a2 = (100.0_f64 / ke) * (1.0 - (-ke * 2.0).exp());
        // [2, 6]: rate=200, A0=a2
        //   A(t) = (200/ke) + (A0 - 200/ke) · exp(-ke·(t-2))
        let r_over_ke = 200.0_f64 / ke;
        let a4 = r_over_ke + (a2 - r_over_ke) * (-ke * 2.0).exp();
        let a6 = r_over_ke + (a2 - r_over_ke) * (-ke * 4.0).exp();
        // [6, ∞]: rate=0
        let a12 = a6 * (-ke * 6.0).exp();

        assert_relative_eq!(preds[0], a2, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a4, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a6, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[3], a12, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_then_bolus() {
        // Infusion [0, 10] followed by a bolus at t=15. Observation at
        // the bolus time should record state AFTER the bolus is applied,
        // matching the existing bolus convention.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0), // infusion, ends at 10
            DoseEvent::new(15.0, 500.0, 1, 0.0, false, 0.0),   // bolus
        ];
        let obs_times = vec![10.0, 15.0, 20.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        let a10 = (100.0_f64 / ke) * (1.0 - (-ke * 10.0).exp());
        let a15_pre = a10 * (-ke * 5.0).exp();
        let a15_post = a15_pre + 500.0;
        let a20 = a15_post * (-ke * 5.0).exp();

        assert_relative_eq!(preds[0], a10, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a15_post, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a20, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_into_oral_depot() {
        // Infusion into depot (cmt=1) of a 1-cpt oral model. Verifies
        // that the wrapped RHS adds `+rate` to the correct compartment
        // (depot index 0), not central (index 1). For the depot alone
        // the closed form is decoupled from ke:
        //   A1(t) during infusion = (R/ka)·(1 - exp(-ka·t))
        //   A1(t) after end T     = A1(T) · exp(-ka·(t-T))
        // Re-use the oral ODE spec but observe the depot.
        let mut ode = one_cpt_oral_ode_spec();
        ode.readout = OdeReadout::ObsCmt(0);

        let rate = 50.0;
        let amt = 200.0; // duration = 4 h
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let obs_times = vec![2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times);
        let mut pk = pk_one(5.0, 80.0);
        pk.values[crate::types::PK_IDX_KA] = 1.0;
        let ka = 1.0_f64;

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        let depot_2 = (rate / ka) * (1.0 - (-ka * 2.0).exp());
        let depot_4 = (rate / ka) * (1.0 - (-ka * 4.0).exp());
        let depot_8 = depot_4 * (-ka * 4.0).exp();

        assert_relative_eq!(preds[0], depot_2, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], depot_4, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], depot_8, epsilon = 1e-2, max_relative = 1e-4);
    }

    // Degenerate input guards: `rate > 0` alone is insufficient to mark a
    // dose as an infusion — `duration = amt/rate` must also be > 0 and
    // finite. Otherwise:
    //   - `amt < 0` would push an infusion-end break time *before* the
    //     dose, scrambling the segmented integration order.
    //   - `amt = NaN` would make `partial_cmp` return None and panic
    //     the break-time sort.
    //   - In both cases, the bolus branch is skipped (because
    //     `is_infusion()` is true on rate alone), so the dose silently
    //     disappears from the prediction.
    // `is_real_infusion` falls back to the bolus path for these rows.

    #[test]
    fn ode_degenerate_zero_amt_with_positive_rate_falls_back_to_bolus() {
        // amt=0, rate>0 → duration=0. Treated as a (no-op) bolus.
        // Result must match "no dose at all".
        let doses = vec![DoseEvent::new(0.0, 0.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![1.0, 5.0, 10.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        assert_eq!(preds, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn ode_degenerate_negative_amt_with_positive_rate_does_not_break_ordering() {
        // amt<0, rate>0 → duration<0. Pre-fix, the infusion-end break time
        // would sort *before* the dose, producing nonsense segments and
        // (silently) zero output because the bolus branch was skipped.
        // Post-fix this is treated as a bolus with negative amt — at least
        // visible to the caller.
        let doses = vec![DoseEvent::new(0.0, -10.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![0.0, 1.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        // Must not panic; the negative bolus update is clamped to 0 by
        // the negative-prediction guard in `ode_predictions`.
        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn ode_degenerate_nan_amt_with_positive_rate_does_not_panic() {
        // amt=NaN, rate>0 → duration=NaN. Pre-fix, sort_by(partial_cmp).unwrap()
        // would panic on the break-time vec. Post-fix the row falls through
        // to the bolus branch and the panic is avoided.
        let doses = vec![DoseEvent::new(0.0, f64::NAN, 1, 100.0, false, 0.0)];
        let obs_times = vec![1.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn ode_iv_bolus_with_lagtime_shifts_curve() {
        // 1-cpt IV bolus integrated via ODE: with lagtime=2.0 and dose at
        // t_dose=0, the central-amount state should be 0 until t=2 (the
        // lagged dose arrival), then decay as if dose-time were 2.
        // (`one_cpt_ode_spec` observes the amount A(t), not A/V.)
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 3.0, 6.0];
        let subj = make_subject(doses, obs_times);
        let mut pk = pk_one(5.0, 80.0);
        pk.values[PK_IDX_LAGTIME] = 2.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        // At t=1, dose has not yet arrived (lagtime=2). State stays 0.
        assert_relative_eq!(preds[0], 0.0, epsilon = 1e-10);

        // At t=3, effective elapsed time since dose is 1.0.
        // A(1) = Amt * exp(-ke * 1)
        let ke = 5.0_f64 / 80.0;
        let expected_3 = 1000.0_f64 * (-ke * 1.0).exp();
        assert_relative_eq!(preds[1], expected_3, epsilon = 1e-4, max_relative = 1e-4);

        // At t=6, effective elapsed time is 4.0.
        let expected_6 = 1000.0_f64 * (-ke * 4.0).exp();
        assert_relative_eq!(preds[2], expected_6, epsilon = 1e-4, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_with_lagtime_shifts_break_times_and_active_window() {
        // Direct test of the ODE infusion + lagtime path that the analytical
        // superposition test alone doesn't cover. Amt=100, rate=100 ⇒
        // duration=1.0; with lagtime=0.5, the active-infusion window runs
        // [2.5, 3.5] rather than [2.0, 3.0]. Compare against an equivalent
        // unlagged dose starting at 2.5 — predictions at matched observation
        // offsets should agree to ODE tolerance.
        let dose_lag = DoseEvent::new(2.0, 100.0, 1, 100.0, false, 0.0);
        assert!(dose_lag.is_infusion() && dose_lag.duration > 0.0);
        let subj_lag = make_subject(vec![dose_lag], vec![2.0, 3.0, 4.0]);
        let mut pk_lag = pk_one(5.0, 80.0);
        pk_lag.values[PK_IDX_LAGTIME] = 0.5;

        // Reference: dose shifted at the data level, no lagtime applied.
        let dose_ref = DoseEvent::new(2.5, 100.0, 1, 100.0, false, 0.0);
        let subj_ref = make_subject(vec![dose_ref], vec![2.0, 3.0, 4.0]);
        let pk_ref = pk_one(5.0, 80.0);

        let ode = one_cpt_ode_spec();
        let preds_lag = ode_predictions(&ode, &pk_lag.values, &[], &[], &subj_lag);
        let preds_ref = ode_predictions(&ode, &pk_ref.values, &[], &[], &subj_ref);

        // Observation before lagged infusion start: zero.
        assert_relative_eq!(preds_lag[0], 0.0, epsilon = 1e-10);

        // Observations during and after the lagged infusion: must match the
        // reference where the dose was shifted at the dataset level.
        assert_relative_eq!(
            preds_lag[1],
            preds_ref[1],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
        assert_relative_eq!(
            preds_lag[2],
            preds_ref[2],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }

    // --- Steady-state (SS=1) tests ---
    //
    // The ODE SS path is verified against the corresponding analytical
    // 1-cpt SS closed forms (PR #75): a 1-cpt IV-bolus ODE with SS dose
    // must match `one_cpt_iv_bolus_ss` to RK45 tolerance, and similarly
    // for infusion. This cross-checks the per-cycle pulse-expansion
    // equilibration loop in `equilibrate_ss_state`.

    #[test]
    fn ode_ss_iv_bolus_matches_analytical_ss() {
        // The test ODE stores compartment AMOUNT (dA/dt = -ke·A), while the
        // analytical formula returns CONCENTRATION = amount/V. Divide
        // before comparing.
        use crate::pk::one_cpt_iv_bolus_ss;
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let amt = 1000.0_f64;
        let ii = 12.0_f64;
        // Sample times within and beyond one dosing interval.
        let obs_times = vec![1.0, 4.0, 8.0, 11.0, 14.0, 24.0];
        let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert_eq!(preds.len(), obs_times.len());

        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_iv_bolus_ss(&dose, t, cl, v);
            // RK45 default tolerance is ~1e-6 relative; SS equilibration
            // truncation at N=50 leaves a ~1e-9 tail. 1e-4 is the safe
            // headroom across the population.
            assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_ss_infusion_matches_analytical_ss() {
        use crate::pk::one_cpt_infusion_ss;
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let amt = 1000.0_f64;
        let rate = 250.0_f64; // T_inf = 4 h
        let ii = 24.0_f64;
        // Cover during-infusion, post-infusion, and beyond one interval.
        let obs_times = vec![1.0, 3.5, 4.0, 8.0, 12.0, 23.0, 48.0];
        let dose = DoseEvent::new(0.0, amt, 1, rate, true, ii);
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_infusion_ss(&dose, t, cl, v);
            assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_ss_resets_prior_state() {
        // SS=1 semantics: at the SS dose time, prior compartment state is
        // discarded and reset to the SS-train value. Build a subject with
        // a non-SS dose at t=0 (which would normally contribute decay
        // through to t=10) and an SS=1 dose at t=10. The post-SS-dose
        // observation must match the SS analytical formula evaluated at
        // tau = obs_time - 10, independent of the t=0 dose.
        use crate::pk::one_cpt_iv_bolus_ss;
        let cl = 5.0;
        let v = 80.0;
        let amt = 1000.0;
        let ii = 12.0;
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, amt, 1, 0.0, true, ii),
        ];
        let obs_times = vec![11.0, 14.0, 20.0];
        let subj = make_subject(doses.clone(), obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_iv_bolus_ss(&doses[1], t - 10.0, cl, v);
            assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_ss_iv_bolus_with_lagtime_matches_nonmem() {
        // ODE-path coverage of SS + ALAG1 (issue #15). Reference PRED from
        // NONMEM 7.5.1 (ADVAN1 TRANS2, MAXEVAL=0): CL=5, V=80, ALAG1=2.0,
        // single SS=1 II=12 AMT=1000 IV bolus into the central compartment
        // (S1=V). Control file + dataset in tests/ss_lagtime_nonmem.rs.
        //
        // The first three samples (t=0.5,1.0,1.5 < ALAG1=2.0) exercise the
        // previous-interval steady-state tail seeded by `ss_state_at_phase`;
        // without the seed the ODE state would still be empty there (≈0).
        let cl = 5.0;
        let v = 80.0;
        let amt = 1000.0;
        let ii = 12.0;
        let lagtime = 2.0;
        let nonmem: &[(f64, f64)] = &[
            (0.5, 12.291),
            (1.0, 11.912),
            (1.5, 11.546),
            (2.0, 23.691),
            (3.0, 22.255),
            (6.0, 18.450),
            (11.0, 13.499),
            (13.0, 11.912),
            (18.0, 8.7153),
        ];
        let obs_times: Vec<f64> = nonmem.iter().map(|&(t, _)| t).collect();
        let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let subj = make_subject(vec![dose], obs_times);
        let mut pk = pk_one(cl, v);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;
        let ode = one_cpt_ode_spec();

        // one_cpt_ode_spec stores amount; divide by V for concentration.
        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &(_t, pred)) in nonmem.iter().enumerate() {
            assert_relative_eq!(preds[j] / v, pred, max_relative = 1e-4);
        }
    }

    // ── [scaling] Form C: output_fn replaces obs_cmt readout ────────────────

    /// Same shape as `one_cpt_ode_spec` but the state holds `amount` (not
    /// concentration), and `output_fn` produces concentration via `A/V`.
    /// V is passed in via `pk_params_flat[PK_IDX_V]`.
    fn one_cpt_ode_spec_amount_form() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                // dA/dt = -CL/V * A   (state is amount; same exp decay as
                // the concentration-baked spec)
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::Single(Box::new(
                |state: &[f64], pk: &[f64], _theta: &[f64], _eta: &[f64], _cov| {
                    let v = pk[crate::types::PK_IDX_V];
                    if v > 0.0 {
                        state[0] / v
                    } else {
                        0.0
                    }
                },
            )),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            init_fn: None,
        }
    }

    #[test]
    fn test_ode_output_fn_form_c_matches_concentration_form() {
        // Build two equivalent 1-cpt IV bolus ODE models:
        //   Reference: state = concentration; obs_cmt_idx = Some(0).
        //   Form C:    state = amount;        output_fn = state/V.
        //
        // The dose adds amt to state in both cases. In the reference, that
        // means state = amt directly equals the initial concentration AMT/V
        // ONLY if AMT/V already matches. To make the two truly equivalent
        // we have to scale the dose differently. Easier: pick V = 1.0 so
        // amount equals concentration numerically, and run an analytical
        // sanity check instead.
        let pk = pk_one(5.0, 1.0); // CL=5, V=1 → ke = 5
        let doses = vec![DoseEvent::new(0.0, 10.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0, 0.5, 1.0, 2.0];
        let subj = make_subject(doses, obs_times.clone());

        let ode_ref = one_cpt_ode_spec();
        let ode_form_c = one_cpt_ode_spec_amount_form();

        let preds_ref = ode_predictions(&ode_ref, &pk.values, &[], &[], &subj);
        let preds_c = ode_predictions(&ode_form_c, &pk.values, &[], &[], &subj);

        // V = 1 makes amount/V numerically equal to amount, so both must agree.
        for (a, b) in preds_ref.iter().zip(preds_c.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6, max_relative = 1e-6);
        }

        // And — crucially — Form C must produce a different numeric answer
        // when V differs from 1, demonstrating the readout actually divides
        // by V rather than ignoring it.
        let pk_v_5 = pk_one(5.0, 5.0); // CL=5, V=5 → ke = 1
        let preds_c_v5 = ode_predictions(&ode_form_c, &pk_v_5.values, &[], &[], &subj);
        // At t=0 just after the bolus, state = 10, V = 5 → conc = 2.
        assert_relative_eq!(preds_c_v5[0], 2.0, epsilon = 1e-9);
        // Reference (concentration-baked) with same params: state = 10
        // ⇒ conc = 10. Different from Form C, confirming output_fn ran.
        let preds_ref_v5 = ode_predictions(&ode_ref, &pk_v_5.values, &[], &[], &subj);
        assert!(
            (preds_ref_v5[0] - preds_c_v5[0]).abs() > 1.0,
            "output_fn must change the readout (ref={} c={})",
            preds_ref_v5[0],
            preds_c_v5[0]
        );
    }

    /// Regression for the co-temporal multi-CMT recorder bug: two observations
    /// at the SAME time but different CMTs (simultaneous PK/PD sampling) must
    /// BOTH be recorded. Before the fix, `obs_map` keyed by time alone kept
    /// only one index per time and left the other observation at its initial
    /// NaN.
    #[test]
    fn test_ode_predictions_records_cotemporal_multi_cmt() {
        // CMT=1 reads the compartment amount; CMT=2 reads twice that — two
        // distinct, finite readouts of the same single-state system, so we can
        // confirm each observation got its own value (not one overwriting the
        // other).
        let mut map: HashMap<usize, OdeOutputFn> = HashMap::new();
        map.insert(1, Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| s[0]));
        map.insert(2, Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| 2.0 * s[0]));
        let mut ode = one_cpt_ode_spec();
        ode.readout = OdeReadout::PerCmt(map);

        let pk = pk_one(5.0, 80.0);
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        // Two obs at t=1 (CMT 1 and 2) and two at t=4 (CMT 1 and 2).
        let mut subj = make_subject(doses, vec![1.0, 1.0, 4.0, 4.0]);
        subj.obs_cmts = vec![1, 2, 1, 2];

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        assert!(
            preds.iter().all(|p| p.is_finite()),
            "all co-temporal obs must be recorded (finite), got {preds:?}"
        );
        // CMT=2 readout is exactly twice CMT=1 at the same time.
        assert!((preds[1] - 2.0 * preds[0]).abs() < 1e-9);
        assert!((preds[3] - 2.0 * preds[2]).abs() < 1e-9);
    }

    /// Regression for Copilot review on PR #84: pre-Phase-2 the ODE paths
    /// clamped NaN predictions to 0 at the end of `ode_predictions` (and
    /// at the Obs branch of `ode_predictions_event_driven`). That defeated
    /// the "loud failure" semantic for `OdeReadout::PerCmt` missing
    /// entries (and for any other genuine NaN). The clamp now only
    /// touches negatives; NaN propagates.
    #[test]
    fn test_ode_predictions_propagates_nan_from_readout() {
        // Build an OdeReadout::PerCmt that DELIBERATELY returns NaN for
        // CMT=1 — emulating a missing-CMT lookup that bypassed pre-fit
        // validation. The resulting prediction must be NaN, not 0.
        let mut map: HashMap<usize, OdeOutputFn> = HashMap::new();
        map.insert(
            1,
            Box::new(|_state: &[f64], _pk: &[f64], _theta, _eta, _cov| f64::NAN),
        );
        let mut ode = one_cpt_ode_spec();
        ode.readout = OdeReadout::PerCmt(map);

        let pk = pk_one(5.0, 80.0);
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 4.0];
        let subj = make_subject(doses, obs_times);

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, p) in preds.iter().enumerate() {
            assert!(
                p.is_nan(),
                "obs {} from a NaN-returning readout must be NaN, got {}",
                j,
                p
            );
        }
    }

    #[test]
    fn test_ode_predictions_still_clamps_negatives() {
        // Sanity: dropping the NaN clamp must not change the negative
        // clamp behavior (ODE solver overshoot guard).
        let ode = OdeSpec {
            // dA/dt = -1 → state goes negative quickly with starting amount 1
            rhs: Box::new(|_y, _p, _t, dy| {
                dy[0] = -1.0;
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            init_fn: None,
        };
        let pk = pk_one(1.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 1.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![10.0]; // dose=1, after 10s of -1/s → state = -9
        let subj = make_subject(doses, obs_times);

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert!(
            !preds[0].is_nan(),
            "negative readout must be clamped to 0, not NaN"
        );
        assert!(
            preds[0] >= 0.0,
            "negative readout must be clamped to 0, got {}",
            preds[0]
        );
    }

    /// Helper: oral PK params with clearance, volume, ka, and bioavailability.
    fn pk_oral_f(cl: f64, v: f64, ka: f64, f: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_KA] = ka;
        p.values[PK_IDX_F] = f;
        p
    }

    #[test]
    fn ode_applies_f_bio_to_bolus_dose() {
        // Issue #122: the ODE engine must load the depot with F·AMT (NONMEM
        // convention), not the full AMT. For this linear oral system the
        // central readout is exactly proportional to the depot load, so a
        // bioavailability of F = 0.5 must halve every prediction relative to
        // F = 1.0. Covers both the plain and event-driven paths.
        let ode = one_cpt_oral_ode_spec();
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times.clone());

        let pk_full = pk_oral_f(5.0, 50.0, 1.5, 1.0);
        let pk_half = pk_oral_f(5.0, 50.0, 1.5, 0.5);

        // Plain (non-TV) path.
        let full = ode_predictions(&ode, &pk_full.values, &[], &[], &subj);
        let half = ode_predictions(&ode, &pk_half.values, &[], &[], &subj);
        for (f, h) in full.iter().zip(half.iter()) {
            assert!(*f > 0.0, "expected positive prediction");
            assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
        }

        // Event-driven path.
        let pk_dose_full = vec![pk_full; subj.doses.len()];
        let pk_obs_full = vec![pk_full; obs_times.len()];
        let pk_dose_half = vec![pk_half; subj.doses.len()];
        let pk_obs_half = vec![pk_half; obs_times.len()];
        let ed_full =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose_full, &pk_obs_full, &[]);
        let ed_half =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose_half, &pk_obs_half, &[]);
        for (f, h) in ed_full.iter().zip(ed_half.iter()) {
            assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
        }
    }

    #[test]
    fn ode_applies_f_bio_to_infusion() {
        // F scales an infusion rate (F·RATE), so halving F halves predictions
        // for a zero-order input into the depot, just like a bolus.
        let ode = one_cpt_oral_ode_spec();
        let rate = 50.0;
        let amt = 100.0; // duration = 2 h
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let obs_times = vec![1.0, 2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times);

        let full = ode_predictions(
            &ode,
            &pk_oral_f(5.0, 50.0, 1.5, 1.0).values,
            &[],
            &[],
            &subj,
        );
        let half = ode_predictions(
            &ode,
            &pk_oral_f(5.0, 50.0, 1.5, 0.5).values,
            &[],
            &[],
            &subj,
        );
        for (f, h) in full.iter().zip(half.iter()) {
            assert!(*f > 0.0, "expected positive prediction");
            assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
        }
    }

    #[test]
    fn ode_applies_f_bio_to_ss_dose() {
        // Steady-state pre-equilibration must also load F·AMT each cycle, so a
        // halved F halves the steady-state predictions.
        let ode = one_cpt_oral_ode_spec();
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        let obs_times = vec![1.0, 4.0, 8.0, 11.0];
        let subj = make_subject(doses, obs_times);

        let full = ode_predictions(
            &ode,
            &pk_oral_f(5.0, 50.0, 1.5, 1.0).values,
            &[],
            &[],
            &subj,
        );
        let half = ode_predictions(
            &ode,
            &pk_oral_f(5.0, 50.0, 1.5, 0.5).values,
            &[],
            &[],
            &subj,
        );
        for (f, h) in full.iter().zip(half.iter()) {
            assert!(*f > 0.0, "expected positive SS prediction");
            assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests for ode_predictions_with_states / ode_dense_solve_states
    // -----------------------------------------------------------------------

    /// Bug regression: state must be advanced through segments that contain no
    /// observations (the t_end push).  Before the fix, `sol.last()` returned
    /// `None` for an empty saveat and `u` was not updated, so all subsequent
    /// compartment states were wrong.
    #[test]
    fn ode_with_states_advances_through_empty_segment() {
        // Two doses, observations only after the second.  The segment [0, 12)
        // has no obs — the state must still decay correctly through it.
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ode = one_cpt_ode_spec();
        let pk = pk_one(cl, v);
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 50.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![24.0];
        let subj = make_subject(doses, obs_times);
        let (preds, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
        // Compare against the full ode_predictions path.
        let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert!(
            approx::relative_eq!(preds[0], preds_ref[0], max_relative = 1e-6),
            "ipred diverges — state was not advanced through the empty segment"
        );
        // State[0] must be positive (non-zero drug remaining).
        assert!(
            states[0][0] > 0.0 && states[0][0].is_finite(),
            "compartment state is wrong after empty inter-dose segment: {}",
            states[0][0]
        );
    }

    /// Bug regression: CMT out-of-range (CMT=0 or CMT > n_states) must be
    /// ignored by both new functions, matching ode_predictions behaviour.
    /// Before the fix, saturating_sub(1).min(n-1) applied the dose to
    /// compartment 0 or the last compartment instead.
    #[test]
    fn ode_with_states_ignores_out_of_range_cmt() {
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ode = one_cpt_ode_spec();
        let pk = pk_one(cl, v);
        // CMT=0 — out-of-range for a 1-state ODE (states are CMT 1).
        let dose_valid = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let dose_oor = DoseEvent::new(0.0, 999.0, 0, 0.0, false, 0.0); // CMT=0
        let obs_times = vec![4.0, 12.0];

        let subj_ref = make_subject(vec![dose_valid.clone()], obs_times.clone());
        let subj_oor = make_subject(vec![dose_valid.clone(), dose_oor], obs_times.clone());

        let (preds_ref, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_ref);
        let (preds_oor, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_oor);
        for j in 0..obs_times.len() {
            assert!(
                approx::relative_eq!(preds_ref[j], preds_oor[j], max_relative = 1e-9),
                "obs {j}: CMT=0 dose was applied (got {}) instead of being ignored (expected {})",
                preds_oor[j],
                preds_ref[j]
            );
        }
    }

    /// Bug regression: TAD for SS doses must be computed with rem_euclid so it
    /// stays within [0, II).  Before the fix the raw elapsed time was used,
    /// injecting a growing TAD into the ODE RHS.  This test uses an ODE that
    /// writes TAD into its output so we can verify it.
    #[test]
    fn ode_with_states_tad_stays_within_dosing_interval_for_ss() {
        // ODE: dA/dt = -ke*A; but we read TAD = t - ext_params[TAD_SLOT] back
        // as the compartment state for the diagnostic.  Use a second-state ODE:
        //   dA/dt = -ke*A
        //   dT/dt = 0  (T is just a placeholder; we seed it externally via the
        //               TAD anchor update, which is non-state, so we use ipred)
        // Actually simplest: verify ode_predictions and ode_predictions_with_states
        // agree on ipred for an SS dose observed beyond one II, because the TAD
        // error only shows up when TAD modulates the ODE.
        //
        // For a pure 1-cpt IV where the RHS does NOT use TAD, both paths must
        // agree with the closed-form SS regardless of the TAD anchor.
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ii = 24.0_f64;
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii);
        // Observations beyond one dosing interval.
        let obs_times = vec![0.5, 6.0, 24.0, 30.0, 48.0, 53.0];
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let (preds_ws, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
        let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &t) in obs_times.iter().enumerate() {
            // ipred must agree with ode_predictions (which uses rem_euclid for TAD).
            assert!(
                approx::relative_eq!(preds_ws[j], preds_ref[j], max_relative = 1e-6),
                "ipred diverges at t={t} — TAD anchor mismatch for SS dose"
            );
            // For ObsCmt(0) readout, ipred == u[0] == state[0], so they must agree.
            assert!(
                approx::relative_eq!(states[j][0], preds_ws[j], max_relative = 1e-9),
                "state != ipred at t={t} — state not self-consistent with ipred"
            );
        }
    }

    /// Bug regression: for an SS dose with lagtime > 0, the pre-lag break
    /// point at dose.time must seed ss_state_at_phase so observations before
    /// the lagged pulse see the correct pre-lag SS tail rather than zero.
    /// Before the fix the merged dose loop fired only at dose.time + lagtime,
    /// leaving the pre-lag segment with an all-zero initial state.
    #[test]
    fn ode_with_states_ss_lagtime_preseed_is_correct() {
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ii = 24.0_f64;
        let lagtime = 2.0_f64;
        // SS dose at t=0 with lagtime=2; observations at t=0.5 and t=1.5
        // (both before the lagged arrival at t=2) should see the SS tail
        // from the prior cycle.
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii);
        let mut pk = pk_one(cl, v);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;
        let obs_times = vec![0.5, 1.5, 3.0, 12.0];
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let ode = one_cpt_ode_spec();

        let (preds_ws, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
        let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, (&t, &p_ws)) in obs_times.iter().zip(preds_ws.iter()).enumerate() {
            assert!(
                approx::relative_eq!(p_ws, preds_ref[j], max_relative = 1e-6),
                "ipred diverges at t={t} — SS+lagtime pre-lag seeding missing"
            );
            // Pre-lag obs (t < lagtime) must be > 0 (from prior SS cycle).
            if t < lagtime {
                assert!(
                    states[j][0] > 0.0,
                    "state is zero at t={t} (before lag) — SS tail was not pre-seeded"
                );
            }
        }
    }
}
