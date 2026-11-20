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

/// The tightest `reltol` the re-solve runs at: below it an explicit stepper is limited by
/// round-off rather than by the tolerance, so a further decade measures the arithmetic, not
/// the integration. A fit already there keeps it, while its `abstol` is still tightened.
pub(crate) const RELTOL_FLOOR: f64 = 1e-12;

/// The tightest `abstol` the re-solve runs at, for the same reason. The check is skipped only
/// when neither tolerance can be tightened (#2994 review r2): a fit at `ode_reltol = 1e-12`
/// with `ode_abstol = 1e-6` can still be dominated by its absolute tolerance.
pub(crate) const ABSTOL_FLOOR: f64 = 1e-14;

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
    /// The objective (−2LL, without penalties) at the fit's tolerance, EBEs re-found from the
    /// fit's modes exactly as on the tight side.
    pub(crate) fit_ofv: f64,
    /// The same objective, EBEs re-found, at the tighter tolerance.
    pub(crate) tight_ofv: f64,
    /// The inner tolerance the EBEs were re-found at: the owning stage's.
    pub(crate) inner_tol: f64,
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
/// with no integrator, or a fit whose `reltol` and `abstol` are both already at their floors
/// ([`RELTOL_FLOOR`], [`ABSTOL_FLOOR`]).
///
/// A model routed to the exact matrix-exponential propagation is checked too (#2994 review
/// r6): its PK is exact, but the objective can still integrate at the tolerance — a
/// time-inhomogeneous CTMM likelihood solves its occupancy ODE at
/// `effective_solver_opts().for_integration()`, and a segment the linear propagator declines
/// falls back to an integrated one. When nothing in the objective integrates, both sides
/// score the same and the check stays silent, at the cost of one inner-loop pass. Each tolerance is tightened one decade, but not past its floor. Read on
/// the thread the fit runs on, where its own override is armed, so the tolerance tightened is
/// the one the fit integrated at.
pub(crate) fn tightened_override(
    model: &CompiledModel, options: &FitOptions,
) -> Option<crate::ode::solver::OdeSolverOverride> {
    let eff = model.integrator_spec()?.effective_solver_opts();
    let reltol = (eff.reltol * TIGHTEN_FACTOR).max(RELTOL_FLOOR).min(eff.reltol);
    let abstol = (eff.abstol * TIGHTEN_FACTOR).max(ABSTOL_FLOOR).min(eff.abstol);
    if reltol >= eff.reltol && abstol >= eff.abstol {
        return None;
    }
    Some(crate::ode::solver::OdeSolverOverride {
        reltol: Some(reltol),
        abstol: Some(abstol),
        max_steps: Some(eff.max_steps.saturating_mul(MAX_STEPS_FACTOR)),
        ..options.ode_solver_override()
    })
}

/// Whether the trajectories a re-scored pass **kept** were integrated, not padded (#2994
/// review r2): no segment left unfinished, no step force-accepted at `min_dt`, and no
/// prediction walk abandoned before integrating. Each returns a finite trajectory (or a
/// sentinel-clamped objective) that never met the tolerance, so a small shift from it would
/// read as a settled objective. A stiff-aborted or diverged segment is an unfinished one, so it
/// is caught by that counter; an abandoned walk (an unorderable timeline, #1234) never starts
/// a segment, so it has its own; an `auto` attempt the guard discarded and re-solved cleanly
/// does not count.
///
/// `with_fd_jacobian_integration` keeps its tight pass only when that pass has no kept clamp,
/// no kept unfinished segment and a finite `H`, and an abandoned walk's predictions are `NaN`,
/// so a kept finite-difference `H` is clean by this test by construction and needs no separate
/// accounting. `None` when clean, else the counters that were not, for the warning.
pub(crate) fn unclean_integration(st: &crate::ode::solver::OdeSolverStats) -> Option<String> {
    let parts: Vec<String> = [
        (st.kept_clamped_steps(), "steps clamped at the minimum step size"),
        (st.kept_unfinished_segments(), "unfinished segments"),
        (st.abandoned_non_finite_timeline, "abandoned non-finite timelines"),
    ]
    .into_iter()
    .filter(|(n, _)| *n > 0)
    .map(|(n, what)| format!("{n} {what}"))
    .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// Re-find every subject's EBEs warm from the fit's joint modes (η̂ and κ̂) and score the
/// objective, at whatever ODE tolerance is armed on the calling thread — `Err` with the reason
/// when the result cannot be trusted (#2994 review r2, r3):
///
/// * **an integration behind it was not clean** ([`unclean_integration`]). Every integration
///   that forms a subject's term — the EBE search, its `H`, every AGQ node — is counted,
///   because each subject is solved *and* scored in one task under one
///   [`crate::ode::solver::SolverStatsScope`]. The scope is per thread, and a subject's solve
///   and score fan out over nothing but that subject, so the task's thread runs all of it. A
///   finite-difference `H` whose tighter pass was kept is the one integration the scope does
///   not see, and it is clean by the same test ([`unclean_integration`]);
/// * **the EBE pass fails the production guard** ([`ebe_guard_rejects`]): a start the inner
///   loop hard-rejected (its η̂ / `H` are placeholders), or more unconverged subjects than
///   `max_unconverged_frac` allows. Such a pass is rejected by every objective evaluation
///   of the fit, and scoring it here would blame EBE error on the ODE tolerance.
///
/// The solve is the seeded warm one an `agq_eval_only` stage runs, per subject, on both sides
/// of the check; the score is `pop_nll_opts`' term
/// for that subject (the AGQ marginal when the stage has nodes, the FOCE/FOCEI one otherwise).
///
/// `None` when the fit was cancelled during the pass: its subjects stopped solving, so there
/// is nothing to judge, and the check is skipped rather than reported.
///
/// [`ebe_guard_rejects`]: crate::estimation::outer_optimizer::ebe_guard_rejects
fn rescore_checked(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    eta_hats: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], options: &FitOptions,
) -> Option<Result<f64, String>> {
    let (ofv, ebe_stats, solver) =
        rescore_counted(model, population, params, eta_hats, kappas, options);
    if crate::cancel::is_cancelled(&options.cancel) {
        return None;
    }
    // Every reason, not the first: a clamped trajectory usually also stalls the EBE search,
    // and the clamp is the cause the user can act on.
    let mut why = Vec::new();
    if crate::estimation::outer_optimizer::ebe_guard_rejects(
        &ebe_stats,
        population.subjects.len(),
        ofv,
        options.max_unconverged_frac,
    ) {
        why.push(format!(
            "the re-found EBEs fail the fit's own convergence guard ({} of {} subjects \
             hard-rejected at their start, {} unconverged, max_unconverged_frac = {})",
            ebe_stats.n_start_rejected,
            population.subjects.len(),
            ebe_stats.n_unconverged,
            options.max_unconverged_frac
        ));
    }
    if let Some(counters) = unclean_integration(&solver) {
        why.push(format!("the solve did not integrate cleanly: {counters}"));
    }
    Some(if why.is_empty() { Ok(ofv) } else { Err(why.join("; ")) })
}

/// [`rescore_checked`]'s objective, with the EBE pass's statistics and the solver counters of
/// every integration behind it, before either is judged.
fn rescore_counted(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    eta_hats: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], options: &FitOptions,
) -> (f64, crate::estimation::inner_optimizer::InnerLoopStats, crate::ode::solver::OdeSolverStats) {
    use crate::estimation::inner_optimizer::{
        cacheable_schedule, find_ebe_cached, InnerHessianSeed, InnerLoopStats, InnerSolvePolicy,
        PrevEtas,
    };
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let warm = crate::estimation::outer_optimizer::joint_warm_starts(eta_hats, kappas);
    let policy = InnerSolvePolicy {
        seed: InnerHessianSeed::for_options(options),
        ..InnerSolvePolicy::for_stage(options, model)
    }
    .with_prev_etas(PrevEtas::Solved);
    let min_obs = (options.min_obs_for_convergence_check as usize).max(1);
    let per_subject = crate::par::map(population.subjects.len(), |i| {
        // A cancelled fit stops solving: the remaining subjects return at once, and the
        // caller discards the pass (`ofv_tolerance_shift` reads the token after it).
        if crate::cancel::is_cancelled(&options.cancel) {
            return (f64::NAN, false, false, crate::ode::solver::OdeSolverStats::default());
        }
        let subject = &population.subjects[i];
        let scope = crate::ode::solver::SolverStatsScope::enter();
        let schedule = cacheable_schedule(model, subject);
        let ebe = find_ebe_cached(
            model,
            subject,
            params,
            options.inner_maxiter,
            options.inner_tol,
            Some(warm[i].as_slice()),
            Some(&mu_k),
            options.inner_restarts,
            schedule.as_ref(),
            policy,
        );
        let nll =
            score_subject(model, subject, params, &ebe.eta, &ebe.h_matrix, &ebe.kappas, options);
        let unconverged = !ebe.converged && subject.n_scored_rows() >= min_obs;
        (nll, unconverged, ebe.hard_reject, scope.collected())
    });
    let mut ofv = 0.0;
    let mut ebe_stats = InnerLoopStats::default();
    let mut solver = crate::ode::solver::OdeSolverStats::default();
    for (nll, unconverged, hard_reject, st) in &per_subject {
        ofv += 2.0 * nll;
        ebe_stats.n_unconverged += usize::from(*unconverged);
        ebe_stats.n_start_rejected += usize::from(*hard_reject);
        solver.merge(st);
    }
    (ofv, ebe_stats, solver)
}

/// One subject's term of `pop_nll_opts` at the given modes: the AGQ marginal when the stage
/// has nodes (which lays its own grid and anchor around `eta`, and ignores `h`), the
/// FOCE/FOCEI one at `h` otherwise. The one scoring rule both sides of the check share.
fn score_subject(
    model: &CompiledModel, subject: &crate::types::Subject, params: &ModelParameters,
    eta: &DVector<f64>, h: &DMatrix<f64>, kappas: &[DVector<f64>], options: &FitOptions,
) -> f64 {
    match options.agq_nodes() {
        Some(n_nodes) => crate::estimation::agq::agq_subject_term(
            model,
            subject,
            params,
            eta.as_slice(),
            kappas,
            n_nodes,
            options.hessian_anchor(),
        ),
        None => crate::estimation::outer_optimizer::subject_nll(
            model,
            subject,
            params,
            eta,
            h,
            kappas,
            options.interaction,
        ),
    }
}

/// The check itself (#2994): `None` when it does not apply ([`tightened_override`]), `Err`
/// with the reason when the tighter scope could not be built, or a re-scored side cannot be
/// trusted ([`rescore_checked`]).
///
/// `options` must be the options of the stage that owns the published OFV (its `method`,
/// `interaction`, `n_agq` and inner settings), so the objective re-scored is the one the fit
/// reported.
///
/// Both sides re-find the EBEs the same way — one warm solve per subject from the fit's
/// stored modes, under the same inner settings — and differ only in the ODE tolerance armed
/// (#2994 review r7). Scoring the stored modes on one side and re-optimized ones on the other
/// would put the extra EBE optimization into the shift, which could then warn on a model whose
/// objective integrates nothing, or cancel a real tolerance shift.
///
/// `None` as well when the fit is cancelled during the check: it is skipped, not reported.
pub(crate) fn ofv_tolerance_shift(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    eta_hats: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], options: &FitOptions,
) -> Option<Result<OdeToleranceShift, String>> {
    let eff = model.integrator_spec()?.effective_solver_opts();
    let tight = tightened_override(model, options)?;
    let fit_ofv = match rescore_checked(model, population, params, eta_hats, kappas, options)? {
        Ok(v) => v,
        Err(why) => return Some(Err(format!("at the fit's own tolerance, {why}"))),
    };
    let tight_ofv = match crate::pool::with_fit_scope_ode(options, tight, || {
        let out = rescore_checked(model, population, params, eta_hats, kappas, options);
        // Its plans, recorded under the tighter override, serve nothing after it; and the
        // tighter scope's pool outlives the call (idle pools are cached), so its workers drop
        // their buffers here, on that pool (#3077).
        crate::parser::model_parser::rhs_regcode::kernel_jit::release_fit_plans();
        out
    }) {
        Ok(None) => return None,
        Ok(Some(Ok(v))) => v,
        Ok(Some(Err(why))) => return Some(Err(format!("at the tighter tolerance, {why}"))),
        Err(e) => return Some(Err(format!("its scope could not be set up: {e}"))),
    };
    Some(Ok(OdeToleranceShift {
        fit_reltol: eff.reltol,
        fit_abstol: eff.abstol,
        tight_reltol: tight.reltol.expect("set by tightened_override"),
        tight_abstol: tight.abstol.expect("set by tightened_override"),
        fit_ofv,
        tight_ofv,
        inner_tol: options.inner_tol,
    }))
}

/// The token of [`not_checked_warning`]; `classify_warning` maps it to `ode_solver`.
pub(crate) const NOT_CHECKED_TOKEN: &str = "W_ODE_OFV_SETTLEMENT_UNCHECKED";

/// The warning for a check that applied but could not be completed (`reason`): the scope
/// carrying the tighter tolerance could not be built, or a re-scored side did not integrate
/// cleanly or failed the EBE convergence guard. Never silent: an inapplicable check, a failed one and a settled objective must not
/// read the same (#2994 review r1, r2).
pub(crate) fn not_checked_warning(reason: &str) -> String {
    format!(
        "{NOT_CHECKED_TOKEN}: whether the OFV is settled at the fit's ODE tolerance could not \
         be checked: re-solving at a tighter ode_reltol / ode_abstol could not be completed, \
         because {reason}. The fit and its OFV are unaffected; refit once with tighter \
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
        ..
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
