//! Why a gradient-driven outer optimizer stopped short (#2713): how far the outer gradient
//! the fit used at the reported estimates is from the exact one.
//!
//! A non-IOV subject the analytic outer gradient declines numerically takes the held-EBE
//! fallback, which omits the EBE-response term. When the optimizer then stops without
//! converging, the question is whether that fallback is to blame, or the objective itself
//! (EBEs that change mode, a kink in the model). The size of the difference between the two
//! gradients at the reported point tells the reader whether the fallback is in play there,
//! and the rerun with `reconverge_gradient_interval = 1` settles it, so the warning reports
//! the first and names the second. It does not attempt a verdict.

use super::*;

/// The fit's outer gradient at the reported estimates, set against the exact one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StopGradientGap {
    /// Subjects whose outer gradient at the estimates is the held-EBE fallback.
    pub(crate) n_held: usize,
    /// `‖g_fit − g_exact‖ / ‖g_exact‖` over the coordinates the bounds leave free.
    pub(crate) gap: f64,
}

/// Whether a fit that stopped this way gets the report: a gradient-driven optimizer that
/// did not converge, on a route whose declined subjects can take the held-EBE fallback.
/// Under IOV every decline is reconverged, under `reconverge_gradient_interval = 1` every
/// gradient is, and AGQ and mixtures assemble their gradient elsewhere; in all of these
/// the fit's gradient is already the exact one, or not the one measured here.
pub(crate) fn wants_stop_report(
    resolved: Optimizer,
    converged: bool,
    model: &CompiledModel,
    template: &ModelParameters,
    options: &FitOptions,
) -> bool {
    matches!(
        resolved,
        Optimizer::NloptLbfgs
            | Optimizer::Slsqp
            | Optimizer::Mma
            | Optimizer::Lbfgs
            | Optimizer::Bfgs
    ) && !converged
        && model.n_kappa == 0
        && options.reconverge_gradient_interval != 1
        && options.agq_nodes().is_none()
        && template.mixture.is_none()
        && !crate::cancel::is_cancelled(&options.cancel)
}

/// [`optimize_population`]'s hook: add the stop report to `result`'s warnings when
/// [`wants_stop_report`] says the stop is one to explain.
pub(crate) fn attach_stop_report(
    resolved: Optimizer,
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    options: &FitOptions,
    result: &mut OuterResult,
) {
    if !wants_stop_report(resolved, result.converged, model, template, options) {
        return;
    }
    if let Some(gap) = stop_gradient_gap(
        model,
        population,
        template,
        &result.params,
        &result.eta_hats,
        &result.h_matrices,
        options,
    ) {
        result
            .warnings
            .push(stop_report_warning(gap, population.subjects.len()));
    }
}

/// The fit's outer gradient at `params` (the reported EBEs `eta_hats` held), and its gap to
/// the exact one, in which every held-EBE subject is reconverged instead. The two differ
/// only in those subjects' terms, so the exact one is the fit's plus the difference of the
/// two assemblies over those subjects alone: one population gradient, plus one held-EBE and
/// one reconverged gradient per held subject. `None` when the inputs do not match the
/// population or the exact gradient has no free coordinate left to measure against.
pub(crate) fn stop_gradient_gap(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    options: &FitOptions,
) -> Option<StopGradientGap> {
    let n = population.subjects.len();
    if eta_hats.len() != n || h_matrices.len() != n {
        return None;
    }
    let PackedStart { bounds, .. } = pack_with_bounds(template);
    let x = pack_params(params);
    let assemble = |pop: &Population, idx: &[usize], log: &OuterFdDeclineLog| {
        let ehs: Vec<DVector<f64>> = idx.iter().map(|&i| eta_hats[i].clone()).collect();
        let hms: Vec<DMatrix<f64>> = idx.iter().map(|&i| h_matrices[i].clone()).collect();
        population_gradient_sens_mixed_dx(
            &x,
            template,
            model,
            pop,
            &ehs,
            &hms,
            &bounds,
            options,
            OuterTrial::unknown(),
            log,
            false,
        )
        .0
    };
    let all: Vec<usize> = (0..n).collect();
    let fit_log = OuterFdDeclineLog::new(n);
    let mut g_fit = assemble(population, &all, &fit_log);
    let priors = build_prior_set(model, template);
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    nn_reg.penalty_and_gradient(&params.theta, &mut g_fit);
    priors.penalty_and_gradient(&x, &mut g_fit);

    let held: Vec<usize> = fit_log
        .declined_indices()
        .into_iter()
        .filter(|&i| !fit_log.is_structural(i))
        .collect();
    let mut g_exact = g_fit.clone();
    if !held.is_empty() {
        let sub = Population {
            subjects: held
                .iter()
                .map(|&i| population.subjects[i].clone())
                .collect(),
            covariate_names: population.covariate_names.clone(),
            dv_column: population.dv_column.clone(),
            input_columns: population.input_columns.clone(),
            exclusions: population.exclusions.clone(),
            warnings: Vec::new(),
        };
        let reconverged = assemble(&sub, &held, &OuterFdDeclineLog::reconverging(held.len()));
        let as_held = assemble(&sub, &held, &OuterFdDeclineLog::new(held.len()));
        for ((g, r), h) in g_exact.iter_mut().zip(&reconverged).zip(&as_held) {
            *g += r - h;
        }
    }
    let exact = free_norm(&x, &bounds, &g_exact, &g_exact);
    let diff: Vec<f64> = g_fit.iter().zip(&g_exact).map(|(a, b)| a - b).collect();
    let gap = free_norm(&x, &bounds, &g_exact, &diff) / exact;
    (exact > 0.0 && gap.is_finite()).then_some(StopGradientGap {
        n_held: held.len(),
        gap,
    })
}

/// The norm of `v` over the coordinates of `x` free to move along `-g`: a coordinate at a
/// bound that `g` pushes against cannot move, so it is no part of the step either gradient
/// would take.
fn free_norm(x: &[f64], bounds: &PackedBounds, g: &[f64], v: &[f64]) -> f64 {
    (0..x.len())
        .filter(|&k| {
            !((x[k] <= bounds.lower[k] && g[k] > 0.0) || (x[k] >= bounds.upper[k] && g[k] < 0.0))
        })
        .map(|k| v[k] * v[k])
        .sum::<f64>()
        .sqrt()
}

/// The warning [`attach_stop_report`] adds. With no held-EBE subject at the estimates the
/// final gradient is exact, so the reader is pointed at the objective; otherwise it names
/// the count and the gap, and the rerun that decides whether the fallback caused the stop.
/// A spent evaluation budget is reported by its own warning and is not told apart here
/// (the stop status stays inside the optimizer), so the advice is conditioned on it.
pub(crate) fn stop_report_warning(gap: StopGradientGap, n_subjects: usize) -> String {
    const LEAD: &str = "The outer optimizer stopped without converging (#2713).";
    const OBJECTIVE: &str = "EBEs that change mode between nearby estimates, or a kink in \
         the model such as `if`, `abs` or `max`";
    if gap.n_held == 0 {
        format!(
            "{LEAD} At the reported estimates every subject's outer gradient is exact \
             (analytic, or reconverged), so the fixed-EBE fallback does not bias the final \
             gradient, though it can still have biased the gradient at a line-search trial \
             point. Unless the stop was the evaluation budget, look first at the objective: \
             {OBJECTIVE}."
        )
    } else {
        format!(
            "{LEAD} At the reported estimates {} of {n_subjects} subjects carry fixed-EBE \
             outer gradients, and the gradient the fit used there is {:.3}% off the exact \
             one. Unless the stop was the evaluation budget, test whether that caused it by \
             rerunning with `reconverge_gradient_interval = 1`: if the stop goes away, the \
             fallback was the cause; if it stays, look at the objective: {OBJECTIVE}.",
            gap.n_held,
            100.0 * gap.gap
        )
    }
}

#[cfg(test)]
#[path = "outer_optimizer_stop_tests.rs"]
mod tests;
