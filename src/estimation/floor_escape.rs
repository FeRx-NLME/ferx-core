//! The outer optimizer's escape from ferx's internal θ log-packing floor (#2339).
//!
//! A θ declared `≥ 0` is packed as `ln θ`, floored at
//! [`LOG_PACK_FLOOR`](crate::estimation::parameterization::LOG_PACK_FLOOR) = `1e-10`. On that
//! floor the packed gradient `θ · ∂OFV/∂θ` is `1e-10` times the real one, so a gradient
//! optimizer reads a point where the OFV still falls steeply in θ as stationary, and every
//! verification leg from it gains nothing. The profiled MBMA anchor (A2,
//! `nonmem_anchor/profile_mbma_fit.ferx`) shows it: from TVE0 ≥ 52 the first L-BFGS step
//! puts TVE0 on the floor, the between-arm κ grows to 2071 to absorb the baseline, and the
//! fit stopped `converged: true` 171.9 OFV above NONMEM's optimum, profiled and unprofiled
//! alike. NONMEM reaches the optimum from the same starts, and agrees with ferx's objective
//! at the collapsed point (273.8479).
//!
//! So after the outer optimizer stops, [`escape_floor`] asks whether any free θ sits on
//! that floor (the same walk as the post-fit "collapse" warning,
//! `api::postfit::collapsed_coordinates`) while the OFV falls as it rises. One evaluation
//! answers it: the objective with each floored θ at [`FLOOR_PROBE_REL`] of its start. If
//! the probe is no lower, the floor is a genuine boundary optimum and the fit is returned
//! untouched. Otherwise the fit is restarted from its own estimates with the floored θ
//! (and any variance coordinate on its lower rail, which a collapsed θ drags with it) back
//! at their starts. Measured on A2, that restart lands on NONMEM's optimum (Δ ≈ 2.5e-8)
//! from every start that collapses. A restart that ends no lower, or on the floor again,
//! leaves a fit that is **not** reported as converged: a point on the floor with a downhill
//! probe is not a minimum. Nor is a restart that leaves a θ below 1/1000 of its start —
//! in practice just above the floor — with a downhill probe there ([`near_floor_thetas`],
//! #2629).

use crate::estimation::outer_optimizer::{
    above_start_warning, ofv_is_valid, runs_outer_optimizer, OuterResult, StartEval, StartRescore,
};
use crate::estimation::parameterization::{pack_params, unpack_params, PackedCoordKind};
use crate::types::{FitOptions, ModelParameters};

/// The probe's θ, as a fraction of the θ's start. Small enough to stay in the floor's
/// neighbourhood, large enough that the objective resolves the slope: on A2 the drop at
/// TVE0 = 0.055 (start 55) is ≈ 0.058 OFV, against a [`FLOOR_PROBE_GAIN`] of 1e-6.
pub(crate) const FLOOR_PROBE_REL: f64 = 1e-3;

/// The OFV drop at the probe that says the floor is not a minimum. The profiled
/// verification gain (`profile::PROFILED_VERIFY_GAIN`): a deterministic evaluation, so
/// anything above it is a real slope.
pub(crate) const FLOOR_PROBE_GAIN: f64 = 1e-6;

/// The free θ on ferx's internal log-packing floor whose start was above it: the ones a
/// restart can put back. `start` is the template the optimizer was started from.
pub(crate) fn floored_thetas(result: &ModelParameters, start: &ModelParameters) -> Vec<usize> {
    crate::api::postfit::collapsed_coordinates(result)
        .into_iter()
        .filter(|&(i, kind)| {
            kind == PackedCoordKind::Theta
                && start
                    .theta
                    .get(i)
                    .is_some_and(|&s| s > crate::estimation::parameterization::LOG_PACK_FLOOR)
        })
        .map(|(i, _)| i)
        .collect()
}

/// The free θ a floor restart left **just above** the internal log-packing floor: below its
/// own probe point ([`FLOOR_PROBE_REL`] of its start), on ferx's floor rather than a declared
/// bound, and not exactly on it (those are [`floored_thetas`]').
///
/// Read only after a restart (#2629). There the restart can stop with a θ a few ln-units off
/// the floor — on F2 (`nonmem_anchor/profile_mbma_fit.ferx`, unprofiled, two studies) EMAX at
/// 3.31e-10, 1.197 ln-units up, where the packed gradient `θ · ∂OFV/∂θ` is still ~1e-10 of
/// the real one — and the exact-rail test misses it, so the restart was reported converged
/// 20.6 OFV above the optimum. The first stop keeps [`floored_thetas`]: widening its probe
/// and reset to this set was measured to report a worse point (66.02 instead of 42.79).
///
/// The post-fit guard warning (`api::postfit::collapsed_coordinates`) keeps the exact rail
/// too: such a θ did not reach the guard, so "not on the floor" is true of it there.
pub(crate) fn near_floor_thetas(result: &ModelParameters, start: &ModelParameters) -> Vec<usize> {
    let on_rail = floored_thetas(result, start);
    (0..result.theta.len())
        .filter(|&i| {
            !result.theta_fixed.get(i).copied().unwrap_or(false)
                && crate::estimation::parameterization::theta_guard_is_internal(start, i, "lower")
                // Also "started above the floor": a start on it puts the probe point 1e3
                // below the floor, where no packed θ can be. One gate, not two (AGENTS.md).
                && result.theta[i] < FLOOR_PROBE_REL * start.theta[i]
                && !on_rail.contains(&i)
        })
        .collect()
}

/// `result` with each floored θ at [`FLOOR_PROBE_REL`] of its start.
pub(crate) fn probe_params(
    result: &ModelParameters,
    start: &ModelParameters,
    floored: &[usize],
) -> ModelParameters {
    let mut p = result.clone();
    for &i in floored {
        p.theta[i] = FLOOR_PROBE_REL * start.theta[i];
    }
    p
}

/// `result` with every coordinate on a collapse rail back at its start value — the point
/// the restart begins from. The floored θ are among them: [`floored_thetas`] is a subset of
/// the same walk. In packed space, so a Cholesky diagonal reset leaves a positive-definite
/// Ω.
///
/// The FIX mask is the user's (`start`'s), not the stop's: the outer optimizer's flat-θ
/// pre-flight marks a θ the objective does not read as FIX on its result, with a warning
/// that lives on that result only. Restarted with the stop's mask, the pre-flight would skip
/// the already-FIXed θ and an adopted restart would report the FIX with no reason (#2460
/// review, row 1). With the user's mask the restart's own pre-flight freezes it again, and
/// warns.
pub(crate) fn restart_params(result: &ModelParameters, start: &ModelParameters) -> ModelParameters {
    let mut x = pack_params(result);
    let x_start = pack_params(start);
    for (i, _) in crate::api::postfit::collapsed_coordinates(result) {
        if let (Some(xi), Some(&si)) = (x.get_mut(i), x_start.get(i)) {
            *xi = si;
        }
    }
    let mut p = unpack_params(&x, result);
    p.theta_fixed = start.theta_fixed.clone();
    p
}

/// The warning of an adopted floor restart. A function so the `classify_warning` table test
/// classifies the text this site emits; filed under `optimizer_health` on "resumed from the
/// best point seen".
pub(crate) fn floor_restart_warning(
    names: &[String],
    floor_ofv: f64,
    probe_ofv: f64,
    restarted_ofv: f64,
) -> String {
    format!(
        "Outer optimization stopped with {} on ferx's internal 1e-10 log-packing floor, where \
         the objective still falls as it rises (OFV {floor_ofv:.6} on the floor, \
         {probe_ofv:.6} just above it), and was resumed from the best point seen with \
         {} reset to the initial value{}; it reached OFV = {restarted_ofv:.6}, and the \
         reported estimates are from the resumed run.",
        names.join(", "),
        if names.len() == 1 { "it" } else { "them" },
        if names.len() == 1 { "" } else { "s" },
    )
}

/// A θ a floor restart left below its probe point ([`near_floor_thetas`]), as the warning
/// names it: its value there and its probe point.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NearFloorTheta {
    pub(crate) name: String,
    pub(crate) value: f64,
    pub(crate) probe: f64,
}

/// The restart's near-floor θ whose probe was downhill (#2629), with the OFV of the
/// restart's point and of the probe, both scored evaluation-only at `inner_maxiter`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NearFloorStop {
    pub(crate) thetas: Vec<NearFloorTheta>,
    pub(crate) inner_maxiter: usize,
    pub(crate) at_ofv: f64,
    pub(crate) probe_ofv: f64,
}

/// What `resolve_floor_escape`'s `near_floor` hook returns for a restart with near-floor θ:
/// them, and the restart's point and the probe, both scored evaluation-only at
/// `inner_maxiter`.
pub(crate) struct NearFloorProbe<T> {
    pub(crate) thetas: Vec<NearFloorTheta>,
    pub(crate) inner_maxiter: usize,
    pub(crate) at: T,
    pub(crate) probe: T,
}

/// The warning of an adopted restart that ended on the floor again, or with a θ below its
/// probe point and the objective still falling there (#2629): lower than the first stop, so
/// it is the one reported, but no more a minimum than the first. It names the θ the
/// *restart* left on the floor or below its probe point, which need not be the first
/// stop's, each kind in its own clause. The second is worded by what was tested, "below
/// 1/1000 of its start", not "just above the floor": the predicate is relative to the
/// start, so a θ started at 1e4 and left at 5 meets it 24 ln-units off the floor (#2632
/// review, row 2). Filed
/// under `convergence` on "not reported as converged", which `classify_warning` tests before
/// "resumed from the best point seen".
pub(crate) fn floor_refloored_warning(
    names: &[String],
    floor_ofv: f64,
    probe_ofv: f64,
    restarted_names: &[String],
    near: Option<&NearFloorStop>,
    restarted_ofv: f64,
) -> String {
    let mut clauses = Vec::new();
    if !restarted_names.is_empty() {
        clauses.push(format!("{} on the floor again", restarted_names.join(", ")));
    }
    if let Some(near) = near.filter(|n| !n.thetas.is_empty()) {
        let list = |f: fn(&NearFloorTheta) -> f64| {
            near.thetas
                .iter()
                .map(|t| format!("{} = {:.3e}", t.name, f(t)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let n = near.thetas.len();
        // The re-score runs at its own budget (`NearFloorStop::inner_maxiter`); when it does
        // not print as the run's OFV, the budget is named so the two figures for one point do
        // not read as a contradiction (#2632 review, row 1).
        let budget = if format!("{:.6}", near.at_ofv) == format!("{restarted_ofv:.6}") {
            String::new()
        } else {
            format!(", both scored at `inner_maxiter = {}`", near.inner_maxiter)
        };
        clauses.push(format!(
            "{} below 1/1000 of {} start{} ({}), where the objective still falls as {} rise{} \
             (OFV {:.6} with {} at {}, against {:.6} where it stopped{budget})",
            near.thetas
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            if n == 1 { "its" } else { "their" },
            if n == 1 { "" } else { "s" },
            list(|t| t.value),
            if n == 1 { "it" } else { "they" },
            if n == 1 { "s" } else { "" },
            near.probe_ofv,
            if n == 1 { "it" } else { "them" },
            list(|t| t.probe),
            near.at_ofv,
        ));
    }
    format!(
        "Outer optimization stopped with {} on ferx's internal 1e-10 log-packing floor, where \
         the objective still falls as it rises (OFV {floor_ofv:.6} on the floor, \
         {probe_ofv:.6} just above it), and was resumed from the best point seen with {} reset \
         to the initial value{}; the resumed run reached OFV = {restarted_ofv:.6} but stopped \
         with {}. The reported estimates are from the resumed run, and the fit is not reported \
         as converged: try a start closer to the expected estimate, or `optimizer = bobyqa`.",
        names.join(", "),
        if names.len() == 1 { "it" } else { "them" },
        if names.len() == 1 { "" } else { "s" },
        clauses.join(", and with "),
    )
}

/// The warning of a floor stop the restart did not improve on. Filed under `convergence` on
/// "not reported as converged".
pub(crate) fn floor_not_converged_warning(
    names: &[String],
    floor_ofv: f64,
    probe_ofv: f64,
) -> String {
    format!(
        "Outer optimization stopped with {} on ferx's internal 1e-10 log-packing floor, where \
         the objective still falls as it rises (OFV {floor_ofv:.6} on the floor, \
         {probe_ofv:.6} just above it), and a restart from the initial value{} found no lower \
         point off the floor. The fit is not reported as converged: try a start closer to the expected \
         estimate, or `optimizer = bobyqa`.",
        names.join(", "),
        if names.len() == 1 { "" } else { "s" },
    )
}

/// Run `run` from `start`, then escape the log-packing floor as described in the module
/// doc. `run(params, options)` is the outer optimizer — the profiled or the unprofiled one
/// — and is called at most six times: the fit, the evaluation-only probe
/// (`outer_maxiter = 0`, no covariance step), the restart, the re-score of the restart's
/// point and the probe of any θ it left below its probe point (#2629, two calls), and the
/// start's re-score below.
/// `penalty` is added to `OuterResult::ofv` when the two fits are ranked (priors, NN
/// regularisation), as `optimize_nlopt` ranks its attempts.
///
/// Last, whatever it reports is held against the **first** run's starting objective by
/// [`above_start_warning`] (#2461) — the one place that check runs. The restart starts from
/// the floored stop with θ reset, so its own first evaluation can sit far above the user's
/// start; measured against that, the hand-written transit ODE of
/// `transit_flip_flop_fit_matches_ode_twin` walked from −228.71 to the floor at −59.31, was
/// restarted, and was reported `converged` at −191.93. `n_subj` only sizes the message.
///
/// A start that scored `∞` is first re-scored by [`rescored_start`] (#2619).
pub(crate) fn escape_floor<T: StoppedFit>(
    start: &ModelParameters,
    options: &FitOptions,
    n_subj: usize,
    run: impl Fn(&ModelParameters, &FitOptions) -> T,
    penalty: impl Fn(&ModelParameters) -> f64,
) -> T {
    let first = run(start, options);
    let mut fit_start = *first.start_eval();
    let mut reported = escape_floor_from(first, start, options, &run, &penalty);
    let ranked = reported.ofv() + penalty(reported.params());
    if ranked.is_finite() {
        fit_start = rescored_start(fit_start, start, options, &run);
    }
    *reported.start_eval_mut() = fit_start;
    let was_converged = *reported.converged_mut();
    if let Some(w) = above_start_warning(
        reported.converged_mut(),
        fit_start,
        ranked,
        n_subj,
        options.max_unconverged_frac,
    ) {
        let warnings = reported.warnings_mut();
        warnings.push(w);
        if was_converged {
            warnings.push("Outer optimization did not converge".to_string());
        }
    }
    reported
}

/// The start [`escape_floor`] holds the report against: `fit_start`, unless it scored `∞`,
/// in which case it is re-scored once, evaluation-only, at the default inner budget (#2619).
///
/// A profiled fit records `∞` when its first profile solve fails outright, which a small
/// `inner_maxiter` does: on the profile tests' F2 (two studies, `max_unconverged_frac = 0`)
/// at `inner_maxiter = 1` the start scored `∞` and the run was reported `converged` at
/// 66.022863, while at `inner_maxiter = 2` the same start scored 26.465827, was rejected by
/// the guard, and the same 66.022863 was not converged. Re-scored at 3, 50 or 200 the start is
/// 26.465827 in every printed digit (macOS arm64 and Linux aarch64), so the two runs now get
/// one verdict. The unprofiled loop never records `∞` (a non-finite objective is a finite
/// divergence sentinel there), so in practice this re-scores profiled starts only.
///
/// Skipped, leaving the start unscored as before, when the default budget is no larger than
/// the run's own (the re-score would repeat the same deterministic failure) or the fit was
/// cancelled. A re-score that is itself `∞`, or that records no start (the unprofiled
/// evaluation-only path records none), leaves it unscored too. An evaluation-only fit never
/// gets here with an `∞` start: its report is that same failed evaluation, `∞`, which the
/// caller's finite-report gate declines (#2626 review, row 3).
fn rescored_start<T: StoppedFit>(
    fit_start: Option<StartEval>,
    start: &ModelParameters,
    options: &FitOptions,
    run: impl Fn(&ModelParameters, &FitOptions) -> T,
) -> Option<StartEval> {
    let budget = FitOptions::default().inner_maxiter;
    if fit_start.is_none_or(|s| s.ofv.is_finite())
        || budget <= options.inner_maxiter
        || crate::cancel::is_cancelled(&options.cancel)
    {
        return fit_start;
    }
    let eval_only = FitOptions {
        outer_maxiter: 0,
        inner_maxiter: budget,
        run_covariance_step: false,
        optimizer_trace: false,
        ..options.clone()
    };
    match *run(start, &eval_only).start_eval() {
        Some(s) if s.ofv.is_finite() => Some(StartEval {
            rescored: Some(StartRescore {
                run_inner_maxiter: options.inner_maxiter,
                rescore_inner_maxiter: budget,
            }),
            ..s
        }),
        _ => fit_start,
    }
}

/// [`escape_floor`] after its first run: the probe and the restart.
fn escape_floor_from<T: StoppedFit>(
    first: T,
    start: &ModelParameters,
    options: &FitOptions,
    run: impl Fn(&ModelParameters, &FitOptions) -> T,
    penalty: impl Fn(&ModelParameters) -> f64,
) -> T {
    if !runs_outer_optimizer(options)
        || crate::cancel::is_cancelled(&options.cancel)
        || !ofv_is_valid(first.ofv())
    {
        return first;
    }
    let floored = floored_thetas(first.params(), start);
    if floored.is_empty() {
        return first;
    }
    let eval_only = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..options.clone()
    };
    let probe = run(&probe_params(first.params(), start, &floored), &eval_only);
    let names_of = |idx: &[usize]| -> Vec<String> {
        idx.iter()
            .map(|&i| {
                start
                    .theta_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("THETA{}", i + 1))
            })
            .collect()
    };
    resolve_floor_escape(
        first,
        &names_of(&floored),
        &probe,
        |stopped| run(&restart_params(stopped.params(), start), options),
        |r| r.ofv() + penalty(r.params()),
        |r| names_of(&floored_thetas(r.params(), start)),
        |r| {
            let near = near_floor_thetas(r.params(), start);
            if near.is_empty() || crate::cancel::is_cancelled(&options.cancel) {
                return None;
            }
            let thetas = names_of(&near)
                .into_iter()
                .zip(&near)
                .map(|(name, &i)| NearFloorTheta {
                    name,
                    value: r.params().theta[i],
                    probe: FLOOR_PROBE_REL * start.theta[i],
                })
                .collect();
            // Both scored at one budget, at least the default: a probe that moves a θ the
            // EBEs read is a cold start for them, which a run's own small budget does not
            // re-solve. Measured on F2 at `inner_maxiter = 1` (#2629): the probe scored at the
            // run's budget is 42.790440, above the restart's 42.788001, so the gate declined;
            // at the default budget the point is the same 42.788001 and the probe 42.766729.
            // The same rule as `rescored_start` (#2619).
            let inner_maxiter = options
                .inner_maxiter
                .max(FitOptions::default().inner_maxiter);
            let scored = FitOptions {
                inner_maxiter,
                optimizer_trace: false,
                ..eval_only.clone()
            };
            Some(NearFloorProbe {
                thetas,
                inner_maxiter,
                at: run(r.params(), &scored),
                probe: run(&probe_params(r.params(), start, &near), &scored),
            })
        },
        options.verbose,
    )
}

/// What [`escape_floor`] and [`resolve_floor_escape`] read and write of a fit:
/// [`OuterResult`]'s fields, behind a trait so both are unit-tested on a stand-in.
pub(crate) trait StoppedFit {
    fn params(&self) -> &ModelParameters;
    fn ofv(&self) -> f64;
    fn converged_mut(&mut self) -> &mut bool;
    fn evals_mut(&mut self) -> &mut usize;
    fn warnings_mut(&mut self) -> &mut Vec<String>;
    /// The covariance step's wall time, which `fit_inner` subtracts from the stage's time
    /// to get the method's.
    fn covariance_secs_mut(&mut self) -> &mut f64;
    /// The fit's first evaluation ([`OuterResult::start_eval`]).
    fn start_eval(&self) -> &Option<StartEval>;
    fn start_eval_mut(&mut self) -> &mut Option<StartEval>;
}

impl StoppedFit for OuterResult {
    fn params(&self) -> &ModelParameters {
        &self.params
    }
    fn ofv(&self) -> f64 {
        self.ofv
    }
    fn converged_mut(&mut self) -> &mut bool {
        &mut self.converged
    }
    fn evals_mut(&mut self) -> &mut usize {
        &mut self.n_iterations
    }
    fn warnings_mut(&mut self) -> &mut Vec<String> {
        &mut self.warnings
    }
    fn covariance_secs_mut(&mut self) -> &mut f64 {
        &mut self.covariance_wall_time_secs
    }
    fn start_eval(&self) -> &Option<StartEval> {
        &self.start_eval
    }
    fn start_eval_mut(&mut self) -> &mut Option<StartEval> {
        &mut self.start_eval
    }
}

/// The decision behind [`escape_floor`], pure so every branch is unit-testable without a
/// fit: given the stopped fit, the names of its floored θ and the probe, return the fit to
/// report. `restart` runs only when the probe's `ranked` objective is lower than the
/// stop's by more than [`FLOOR_PROBE_GAIN`] — ranked, so the gate and the adoption below
/// judge the same objective when priors or an NN penalty are on (#2460 review, row 6). The
/// restart is adopted only on a strictly lower `ranked` objective, so the escape never
/// reports a higher one. `floored_in` names the θ a result left on the floor, so a restart
/// that collapses again is caught and named. The probe counts as one evaluation.
///
/// `near_floor` asks the same of an adopted restart's θ left below its probe point
/// ([`near_floor_thetas`], #2629): `None` when there are none (or the fit was cancelled),
/// else them and two
/// evaluation-only scores at one inner budget — the restart's own point, and the probe with
/// them at their probe points (two more evaluations). They count only when the probe's
/// `ranked` objective is lower than the point's by more than [`FLOOR_PROBE_GAIN`] — the gate
/// the first stop passes — so a θ resting at a genuine near-zero optimum keeps the restart
/// converged. The exact-rail θ of `floored_in` are not probed again, as before.
///
/// The run that is not reported still ran its covariance step, if one was asked for: each
/// optimizer runs the step itself, at its own final point and with its own final EBEs, so
/// the step cannot be deferred to the kept result without changing it on every fit that
/// never reaches the floor. Its seconds are added to the kept result's covariance time, so
/// `fit_inner` books them as covariance and not as estimation (#2460 review, row 7). The
/// probe is evaluation-only and runs no covariance step.
pub(crate) fn resolve_floor_escape<T: StoppedFit>(
    first: T,
    names: &[String],
    probe: &T,
    restart: impl FnOnce(&T) -> T,
    ranked: impl Fn(&T) -> f64,
    floored_in: impl Fn(&T) -> Vec<String>,
    near_floor: impl FnOnce(&T) -> Option<NearFloorProbe<T>>,
    verbose: bool,
) -> T {
    let probe_ofv = probe.ofv();
    if !(ofv_is_valid(probe_ofv) && ranked(&first) - ranked(probe) > FLOOR_PROBE_GAIN) {
        return first;
    }
    let floor_ofv = first.ofv();
    if verbose {
        eprintln!(
            "Outer optimizer stopped with {} on the log-packing floor (OFV {floor_ofv:.6}, \
             {probe_ofv:.6} just above it); restarting from the initial value.",
            names.join(", "),
        );
    }
    let mut first = first;
    let mut second = restart(&first);
    if !(ofv_is_valid(second.ofv()) && ranked(&second) < ranked(&first)) {
        *first.evals_mut() += *second.evals_mut() + 1;
        *first.covariance_secs_mut() += *second.covariance_secs_mut();
        *first.converged_mut() = false;
        first
            .warnings_mut()
            .push(floor_not_converged_warning(names, floor_ofv, probe_ofv));
        return first;
    }
    *second.evals_mut() += *first.evals_mut() + 1;
    *second.covariance_secs_mut() += *first.covariance_secs_mut();
    let again = floored_in(&second);
    let near = near_floor(&second).and_then(|p| {
        *second.evals_mut() += 2;
        (ofv_is_valid(p.at.ofv())
            && ofv_is_valid(p.probe.ofv())
            && ranked(&p.at) - ranked(&p.probe) > FLOOR_PROBE_GAIN)
            .then(|| NearFloorStop {
                inner_maxiter: p.inner_maxiter,
                at_ofv: p.at.ofv(),
                probe_ofv: p.probe.ofv(),
                thetas: p.thetas,
            })
    });
    let w = if again.is_empty() && near.is_none() {
        floor_restart_warning(names, floor_ofv, probe_ofv, second.ofv())
    } else {
        *second.converged_mut() = false;
        floor_refloored_warning(
            names,
            floor_ofv,
            probe_ofv,
            &again,
            near.as_ref(),
            second.ofv(),
        )
    };
    second.warnings_mut().push(w);
    second
}

#[cfg(test)]
#[path = "floor_escape_tests.rs"]
mod tests;
