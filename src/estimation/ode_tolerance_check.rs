//! Whether the objective at the final estimates is settled at the ODE tolerance the fit
//! integrated at (#2994).
//!
//! The default `ode_reltol = 1e-4` / `ode_abstol = 1e-6` is accurate enough for most models,
//! but not for every one. On a stiff feedback model (indisulam PK driving a transit neutrophil
//! chain with a `(BASE/neut)^GAM` term) it left ~4 OFV units of error against NONMEM, in a
//! direction that flipped with the data, and nothing in the output said so. A 4-unit error is
//! more than a 1-df likelihood-ratio threshold, so a model comparison at defaults can come out
//! the wrong way.
//!
//! The check is the experiment a user would run by hand: re-find the EBEs and re-score the
//! objective once at the final estimates with `reltol` and `abstol` one decade tighter, and
//! report the shift. Above [`OFV_SHIFT_WARN`] the fit carries `W_ODE_OFV_NOT_SETTLED`.
//!
//! A decade is an estimate of the remaining error, not a bound. It can read zero where the
//! error is not: when the step size is limited by something other than the tolerance — the
//! spacing of the output times, the initial step — both tolerances can accept the same steps.
//! On `examples/mm_iv.ferx` `reltol = 1e-1` and `1e-2` score within 1e-8 of each other and
//! 1.47 OFV units from the converged value, while from `1e-3` the decade moves it by 1.41. So a
//! silent check does not certify the tolerance; a warning does say it is too loose.
//!
//! The re-scored objective is the published one (`pop_nll_opts` under the options of the stage
//! that owns `FitResult::ofv` — the last estimating stage, or a trailing `agq_eval_only`
//! readout), so the shift is in the same units as the reported OFV; the fit-tolerance side is
//! re-scored here too rather than read off `FitResult::ofv`, so the penalties and priors that
//! the reported OFV may add cancel out of the difference instead of entering it.

use crate::estimation::outer_optimizer::pop_nll_opts;
use crate::estimation::parameterization::compute_mu_k;
use crate::types::{CompiledModel, EstimationMethod, FitOptions, ModelParameters, Population};
use nalgebra::{DMatrix, DVector};

/// The factor `reltol` and `abstol` are multiplied by for the re-solve: one decade, as the
/// issue that motivated the check asked for (#2994).
pub(crate) const TIGHTEN_FACTOR: f64 = 0.1;

/// How much the step budget grows for the re-solve. A decade tighter costs an order-`p`
/// stepper ~`10^(1/p)` more steps — 1.6× for rk45, 2.2× for rosenbrock23 — so 10 is a cap with
/// headroom; a segment that exhausted the fit's own budget is reported by
/// `W_ODE_SOLVER_DIAGNOSTICS` already. The same factor the covariance stencil uses (#520).
pub(crate) const MAX_STEPS_FACTOR: usize = 10;

/// The tightest `reltol` the re-solve runs at. A fit already at or inside it is not checked:
/// below it an explicit stepper is limited by round-off rather than by the tolerance, so a
/// further decade measures the arithmetic, not the integration.
pub(crate) const RELTOL_FLOOR: f64 = 1e-12;

/// `|ΔOFV|` above which the fit is warned. Well below the 3.84 of a 1-df likelihood-ratio
/// test, so a shift that could change a model comparison is reported, and far above the
/// re-solved EBEs' own noise. Measured on `examples/mm_iv.ferx` at its initial estimates
/// (macOS arm64, rk45, `reltol = abstol`): the decade shift is 3.2e-8 from 1e-10, 4.4e-4 from
/// 1e-6, 1.3e-3 from 1e-5, 0.052 from 1e-4 (whose error against 1e-10 is 0.050, so the decade
/// is a good estimate of the remaining error there), 0.41 from 3e-4 and 1.41 from 1e-3. So
/// that model is silent at the default and warned at `1e-3`.
pub(crate) const OFV_SHIFT_WARN: f64 = 0.1;

/// The token of [`not_settled_warning`]; `classify_warning` maps it to `ode_solver`.
pub(crate) const NOT_SETTLED_TOKEN: &str = "W_ODE_OFV_NOT_SETTLED";

/// The two objectives the check compared, and the tolerances each was integrated at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OdeToleranceShift {
    pub(crate) fit_reltol: f64,
    pub(crate) fit_abstol: f64,
    pub(crate) tight_reltol: f64,
    pub(crate) tight_abstol: f64,
    /// The objective (−2LL, without penalties) at the fit's tolerance.
    pub(crate) fit_ofv: f64,
    /// The same objective, EBEs re-found, at the tighter tolerance.
    pub(crate) tight_ofv: f64,
}

impl OdeToleranceShift {
    /// `tight_ofv − fit_ofv`; `NaN` when either side is not finite.
    pub(crate) fn shift(&self) -> f64 {
        self.tight_ofv - self.fit_ofv
    }

    /// Whether the fit should be warned: a shift past [`OFV_SHIFT_WARN`], or a side that is
    /// not finite (the comparison then measured nothing, which is not "settled" either).
    pub(crate) fn not_settled(&self) -> bool {
        let d = self.shift();
        !d.is_finite() || d.abs() > OFV_SHIFT_WARN
    }
}

/// Whether the last estimating stage's reported objective is the one `pop_nll_opts` scores:
/// FOCE, FOCEI (also with `n_agq > 1`), the Gauss–Newton variants and Laplace/AGQ. The
/// Monte-Carlo estimators report an objective this check would not reproduce, so they are out
/// of scope rather than compared against a different quantity.
pub(crate) fn method_in_scope(method: EstimationMethod) -> bool {
    matches!(
        method,
        EstimationMethod::Foce
            | EstimationMethod::FoceI
            | EstimationMethod::FoceGn
            | EstimationMethod::FoceGnHybrid
            | EstimationMethod::Laplace
    )
}

/// The ODE override the re-solve runs under, or `None` when there is nothing to check: a model
/// that integrates nothing, one solved exactly by matrix exponential (no tolerance enters), or
/// a fit already at [`RELTOL_FLOOR`]. Read on the thread the fit runs on, where its own
/// override is armed, so the tolerance tightened is the one the fit integrated at.
pub(crate) fn tightened_override(
    model: &CompiledModel,
    options: &FitOptions,
) -> Option<crate::ode::solver::OdeSolverOverride> {
    let eff = model.integrator_spec()?.effective_solver_opts();
    if eff.method == crate::ode::OdeMethod::Linear || eff.reltol <= RELTOL_FLOOR {
        return None;
    }
    Some(crate::ode::solver::OdeSolverOverride {
        reltol: Some((eff.reltol * TIGHTEN_FACTOR).max(RELTOL_FLOOR)),
        abstol: Some(eff.abstol * TIGHTEN_FACTOR),
        max_steps: Some(eff.max_steps.saturating_mul(MAX_STEPS_FACTOR)),
        ..options.ode_solver_override()
    })
}

/// Re-find every subject's EBEs warm from the fit's joint modes (η̂ and κ̂) and score the
/// objective, at whatever ODE tolerance is armed on the calling thread. The same seeded warm
/// solve an `agq_eval_only` stage runs, so on the fit-tolerance side it reproduces that
/// stage's objective.
fn rescored_ofv(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> f64 {
    use crate::estimation::inner_optimizer::{
        run_inner_loop_warm_seeded, InnerHessianSeed, PrevEtas,
    };
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let warm = crate::estimation::outer_optimizer::joint_warm_starts(eta_hats, kappas);
    let (etas, hs, _stats, kappas) = run_inner_loop_warm_seeded(
        model,
        population,
        params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&warm),
        Some(&mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
        InnerHessianSeed::for_options(options),
        options,
        PrevEtas::Solved,
    );
    2.0 * pop_nll_opts(model, population, params, &etas, &hs, &kappas, options)
}

/// Which modes the fit-tolerance side of the check scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FitSide {
    /// The fit's own η̂, `H` and κ̂, as they are: they are the modes the reported OFV was
    /// scored at (an estimating stage owns it).
    Stored,
    /// Re-found at the fit's tolerance from the stored modes, as the tight side is: the stored
    /// modes belong to a different objective than the one published — an `agq_eval_only`
    /// readout re-found its own and kept only the OFV.
    Resolved,
}

/// The check itself (#2994): `None` when it does not apply ([`tightened_override`]), `Err`
/// with the reason when the scope carrying the tighter tolerance could not be built.
///
/// `options` must be the options of the stage that owns the published OFV (its `method`,
/// `interaction`, `n_agq` and inner settings), so the objective re-scored is the one the fit
/// reported. The tight side always re-finds the modes from the stored ones; `fit_side` says
/// what the fit-tolerance side scores.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ofv_tolerance_shift(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
    fit_side: FitSide,
) -> Option<Result<OdeToleranceShift, String>> {
    let eff = model.integrator_spec()?.effective_solver_opts();
    let tight = tightened_override(model, options)?;
    let fit_ofv = match fit_side {
        FitSide::Stored => {
            2.0 * pop_nll_opts(
                model, population, params, eta_hats, h_matrices, kappas, options,
            )
        }
        FitSide::Resolved => rescored_ofv(model, population, params, eta_hats, kappas, options),
    };
    let tight_ofv = match crate::api::with_fit_scope_ode(options, tight, || {
        rescored_ofv(model, population, params, eta_hats, kappas, options)
    }) {
        Ok(v) => v,
        Err(e) => return Some(Err(e.to_string())),
    };
    Some(Ok(OdeToleranceShift {
        fit_reltol: eff.reltol,
        fit_abstol: eff.abstol,
        tight_reltol: tight.reltol.expect("set by tightened_override"),
        tight_abstol: tight.abstol.expect("set by tightened_override"),
        fit_ofv,
        tight_ofv,
    }))
}

/// The token of [`not_checked_warning`]; `classify_warning` maps it to `ode_solver`.
pub(crate) const NOT_CHECKED_TOKEN: &str = "W_ODE_OFV_SETTLEMENT_UNCHECKED";

/// The warning for a check that applied but could not run, because the scope carrying the
/// tighter tolerance could not be built (`reason`). Never silent: an inapplicable check and a
/// failed one must not read the same (#2994 review r1).
pub(crate) fn not_checked_warning(reason: &str) -> String {
    format!(
        "{NOT_CHECKED_TOKEN}: whether the OFV is settled at the fit's ODE tolerance could not \
         be checked: re-solving at a one decade tighter ode_reltol / ode_abstol could not be \
         set up ({reason}). The fit and its OFV are unaffected; refit once with tighter \
         ode_reltol / ode_abstol and compare the OFV before comparing models by it (#2994)."
    )
}

/// The `W_ODE_OFV_NOT_SETTLED` text for `s`, or `None` when the objective is settled.
pub(crate) fn not_settled_warning(s: &OdeToleranceShift) -> Option<String> {
    if !s.not_settled() {
        return None;
    }
    let OdeToleranceShift {
        fit_reltol,
        fit_abstol,
        tight_reltol,
        tight_abstol,
        fit_ofv,
        tight_ofv,
    } = *s;
    let tolerances = format!(
        "ode_reltol = {tight_reltol:.0e} / ode_abstol = {tight_abstol:.0e} instead of the \
         fit's {fit_reltol:.0e} / {fit_abstol:.0e}"
    );
    let finding = if s.shift().is_finite() {
        format!(
            "moves the OFV by {:+.4} ({fit_ofv:.4} -> {tight_ofv:.4}), more than \
             {OFV_SHIFT_WARN} units: the objective is not settled at the fit's ODE tolerance, \
             so OFV differences of that size between models may be integration error rather \
             than a difference in fit.",
            s.shift()
        )
    } else {
        format!(
            "gives a non-finite objective ({fit_ofv:.4} -> {tight_ofv:.4}), so whether the OFV \
             is settled at the fit's ODE tolerance could not be measured."
        )
    };
    Some(format!(
        "{NOT_SETTLED_TOKEN}: re-solving the EBEs and the objective at the final estimates \
         with {tolerances} {finding} Refit with ode_reltol / ode_abstol at most \
         {tight_reltol:.0e} / {tight_abstol:.0e} and compare again, tightening until the OFV \
         stops moving (#2994)."
    ))
}

#[cfg(test)]
#[path = "ode_tolerance_check_tests.rs"]
mod tests;
