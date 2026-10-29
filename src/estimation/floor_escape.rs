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
//! probe is not a minimum.

use crate::estimation::outer_optimizer::{ofv_is_valid, runs_outer_optimizer, OuterResult};
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

/// `result` with each floored θ, and every coordinate on a collapse rail, back at its
/// start value — the point the restart begins from. In packed space, so a Cholesky
/// diagonal reset leaves a positive-definite Ω.
pub(crate) fn restart_params(
    result: &ModelParameters,
    start: &ModelParameters,
    floored: &[usize],
) -> ModelParameters {
    let mut x = pack_params(result);
    let x_start = pack_params(start);
    let reset = crate::api::postfit::collapsed_coordinates(result)
        .into_iter()
        .map(|(i, _)| i)
        .chain(floored.iter().copied());
    for i in reset {
        if let (Some(xi), Some(&si)) = (x.get_mut(i), x_start.get(i)) {
            *xi = si;
        }
    }
    unpack_params(&x, result)
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

/// The warning of a floor stop the restart did not leave. Filed under `convergence` on
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
/// — and is called at most three times: the fit, the evaluation-only probe
/// (`outer_maxiter = 0`, no covariance step), and the restart. `penalty` is added to
/// `OuterResult::ofv` when the two fits are ranked (priors, NN regularisation), as
/// `optimize_nlopt` ranks its attempts.
pub(crate) fn escape_floor<T: StoppedFit>(
    start: &ModelParameters,
    options: &FitOptions,
    run: impl Fn(&ModelParameters, &FitOptions) -> T,
    penalty: impl Fn(&ModelParameters) -> f64,
) -> T {
    let first = run(start, options);
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
    let names: Vec<String> = floored
        .iter()
        .map(|&i| {
            start
                .theta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("THETA{}", i + 1))
        })
        .collect();
    resolve_floor_escape(
        first,
        &names,
        probe.ofv(),
        |stopped| run(&restart_params(stopped.params(), start, &floored), options),
        |r| r.ofv() + penalty(r.params()),
        |r| !floored_thetas(r.params(), start).is_empty(),
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
}

/// The decision behind [`escape_floor`], pure so every branch is unit-testable without a
/// fit: given the stopped fit, the names of its floored θ and the probe's OFV, return the
/// fit to report. `restart` runs only when the probe is lower than the floor by more than
/// [`FLOOR_PROBE_GAIN`]; its result is adopted only on a strictly lower `ranked`
/// objective, so the escape never reports a higher one. `floored_in` re-asks the floor
/// question of the restart, so a restart that collapses again is caught. The probe counts
/// as one evaluation.
pub(crate) fn resolve_floor_escape<T: StoppedFit>(
    first: T,
    names: &[String],
    probe_ofv: f64,
    restart: impl FnOnce(&T) -> T,
    ranked: impl Fn(&T) -> f64,
    floored_in: impl Fn(&T) -> bool,
    verbose: bool,
) -> T {
    if !(ofv_is_valid(probe_ofv) && first.ofv() - probe_ofv > FLOOR_PROBE_GAIN) {
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
    let adopt = ofv_is_valid(second.ofv()) && ranked(&second) < ranked(&first);
    let mut kept = if adopt {
        *second.evals_mut() += *first.evals_mut() + 1;
        second
    } else {
        *first.evals_mut() += *second.evals_mut() + 1;
        first
    };
    if adopt && !floored_in(&kept) {
        let w = floor_restart_warning(names, floor_ofv, probe_ofv, kept.ofv());
        kept.warnings_mut().push(w);
    } else {
        *kept.converged_mut() = false;
        kept.warnings_mut()
            .push(floor_not_converged_warning(names, floor_ofv, probe_ofv));
    }
    kept
}

#[cfg(test)]
#[path = "floor_escape_tests.rs"]
mod tests;
