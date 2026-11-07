//! Why a gradient-driven outer optimizer stopped without converging (#2713).
//!
//! A line search ends in `Failure` when it cannot lower the objective along the direction
//! its gradient gives. #2713 named three candidate causes, each with its own remedy:
//!
//! 1. the gradient is not the objective's: declined subjects carry a fixed-EBE gradient,
//!    which drops the EBE-response term;
//! 2. the objective jumps: an EBE changes mode, so the objective is discontinuous and no
//!    gradient sees the jump;
//! 3. the objective is not differentiable there, at a kink (`if`, `abs`, `max`).
//!
//! The decline counts of [`super::outer_fd_fallback_warning`] cannot tell these apart. This
//! probe does, at the estimates the fit reports, with the objective itself as the judge:
//! the objective with every EBE re-solved ([`super::reconverged_probe`]), evaluated along
//! the gradient the fit uses and along the exact one (every declined subject reconverged),
//! at steps whose first-order prediction is a drop of 1 to 0.001 in the OFV. A direction
//! that cannot gain a tenth of its predicted drop at any of them does not descend.
//!
//! | the fit's gradient | the exact gradient | an EBE switches mode | verdict |
//! |---|---|---|---|
//! | does not descend | descends | — | the fixed-EBE gradient (cause 1) |
//! | does not descend | does not descend | yes | objective jumps (cause 2) |
//! | does not descend | does not descend | no | not differentiable (cause 3) |
//! | descends | — | — | no cause at this point |
//!
//! The cost is one reconverged gradient over the subjects declined here, plus up to nine
//! population inner solves, once, after the optimizer has stopped.

use super::*;

/// The OFV drops the probe's steps are sized to predict, largest first: `t·‖g‖` for a step
/// `t` along the unit direction.
const PREDICTED_DROPS: [f64; 4] = [1.0, 1e-1, 1e-2, 1e-3];

/// The share of its predicted drop a step must realise to count as descent (Armijo's `c`).
/// Generous, so that the inner solve's noise cannot pass for descent or hide it: at the
/// smallest step it asks for a drop of `1e-4`.
const DESCENT_SHARE: f64 = 0.1;

/// The longest step probed, in packed coordinates (log θ, Cholesky Ω, log σ).
const MAX_STEP: f64 = 0.1;

/// An EBE that moves by more than this many prior SDs on some axis over a probe step has
/// switched mode: a smooth mode moves by `O(step)`, and the steps here are short.
const MODE_SWITCH_SDS: f64 = 0.5;

/// What the objective did along one direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectionProbe {
    /// Some step realised [`DESCENT_SHARE`] of its predicted drop.
    pub(crate) descends: bool,
    /// When none did: the subjects whose EBE switched mode over the shortest step probed.
    /// `0` when the direction descends.
    pub(crate) mode_switches: usize,
}

/// The probe at the reported estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FailureProbe {
    /// Subjects whose analytic outer gradient declines at the reported estimates, and so
    /// carry a salvage in the fit's gradient.
    pub(crate) n_declined: usize,
    /// `‖g_fit − g_exact‖ / ‖g_exact‖` over the free coordinates; `0` with no declines.
    pub(crate) gap: f64,
    /// Along the fit's gradient.
    pub(crate) fit: DirectionProbe,
    /// Along the exact gradient: the fit's own when no subject declines.
    pub(crate) exact: DirectionProbe,
}

/// Which candidate cause the probe points at (the module table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailureVerdict {
    FixedEbeGradient,
    ModeJumps(usize),
    NotDifferentiable,
    NoCauseHere,
}

impl FailureProbe {
    pub(crate) fn verdict(&self) -> FailureVerdict {
        match (self.fit.descends, self.exact.descends) {
            // The line search had descent here, whatever the exact gradient does.
            (true, _) => FailureVerdict::NoCauseHere,
            (false, true) => FailureVerdict::FixedEbeGradient,
            (false, false) if self.exact.mode_switches > 0 => {
                FailureVerdict::ModeJumps(self.exact.mode_switches)
            }
            (false, false) => FailureVerdict::NotDifferentiable,
        }
    }
}

/// Whether [`optimize_population`]'s result is a stop this probe explains: a gradient-driven
/// optimizer that returned without converging, on its own (not cancelled, budget left).
pub(crate) fn wants_failure_probe(
    resolved: Optimizer,
    converged: bool,
    n_iterations: usize,
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
        && n_iterations < options.outer_maxiter
        && !crate::cancel::is_cancelled(&options.cancel)
}

/// [`optimize_population`]'s hook: probe `result` when [`wants_failure_probe`] says its stop
/// is one to explain, and add the verdict to its warnings.
pub(crate) fn attach_failure_probe(
    resolved: Optimizer,
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    options: &FitOptions,
    result: &mut OuterResult,
) {
    if !wants_failure_probe(resolved, result.converged, result.n_iterations, options) {
        return;
    }
    if let Some(probe) = diagnose_outer_failure(
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
            .push(outer_failure_warning(&probe, population.subjects.len()));
    }
}

/// Walk along `-g` from `x` at the [`PREDICTED_DROPS`] steps, longest first, projected onto
/// `bounds`, and stop at the first that descends. `eval` is the objective with the EBEs
/// re-solved, returning them too; `f0`/`eta0` are its value and EBEs at `x`, and `sd` the
/// prior SD per η axis. `None` when no step can be formed (a zero projected gradient, or
/// every step clamped away).
pub(crate) fn probe_direction(
    g: &[f64],
    x: &[f64],
    bounds: &PackedBounds,
    f0: f64,
    eta0: &[DVector<f64>],
    sd: &[f64],
    eval: impl Fn(&[f64]) -> (f64, Vec<DVector<f64>>),
) -> Option<DirectionProbe> {
    // The projected gradient: a coordinate on a bound the step would push past is dropped.
    let d: Vec<f64> = (0..x.len())
        .map(|i| {
            let di = -g[i];
            let out =
                (x[i] <= bounds.lower[i] && di < 0.0) || (x[i] >= bounds.upper[i] && di > 0.0);
            if out || !di.is_finite() {
                0.0
            } else {
                di
            }
        })
        .collect();
    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 || !f0.is_finite() || f0 >= 1e20 {
        return None;
    }
    let mut last_etas = None;
    for drop in PREDICTED_DROPS {
        let t = (drop / norm).min(MAX_STEP) / norm;
        let xt: Vec<f64> = (0..x.len())
            .map(|i| (x[i] + t * d[i]).clamp(bounds.lower[i], bounds.upper[i]))
            .collect();
        // The first-order drop of the step actually taken, clamping included.
        let predicted: f64 = (0..x.len()).map(|i| g[i] * (x[i] - xt[i])).sum();
        if !(predicted > 0.0) {
            continue;
        }
        let (f, etas) = eval(&xt);
        if f < 1e20 && f0 - f >= DESCENT_SHARE * predicted {
            return Some(DirectionProbe {
                descends: true,
                mode_switches: 0,
            });
        }
        last_etas = Some(etas);
    }
    let etas = last_etas?;
    let mode_switches = etas
        .iter()
        .zip(eta0)
        .filter(|(e, e0)| {
            e.len() == e0.len()
                && (0..e.len()).any(|k| {
                    let s = sd.get(k).copied().unwrap_or(1.0);
                    // A non-finite EBE is a switch too: it left the mode it had.
                    let moved = ((e[k] - e0[k]) / s).abs();
                    !moved.is_finite() || moved > MODE_SWITCH_SDS
                })
        })
        .count();
    Some(DirectionProbe {
        descends: false,
        mode_switches,
    })
}

/// Probe the reported estimates `params`, with their EBEs `eta_hats` and inner Hessians
/// `h_matrices` (module docs). `None` where the probe does not apply:
/// AGQ or a mixture (whose objective is not the per-subject sum differentiated here), IOV
/// (whose declined subjects take another salvage), EBEs that do not match the population, or
/// a point at which no step can be formed.
pub(crate) fn diagnose_outer_failure(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    options: &FitOptions,
) -> Option<FailureProbe> {
    let n = population.subjects.len();
    probe_with_logs(
        model,
        population,
        template,
        params,
        eta_hats,
        h_matrices,
        options,
        |reconverge| {
            if reconverge {
                OuterFdDeclineLog::reconverging(n)
            } else {
                OuterFdDeclineLog::new(n)
            }
        },
    )
}

/// [`diagnose_outer_failure`] with the decline logs its two gradients are assembled under:
/// `log(false)` for the fit's, `log(true)` for the exact one (every declined subject
/// reconverged). A test hands in logs that force a numerical decline.
#[allow(clippy::too_many_arguments)]
fn probe_with_logs(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    options: &FitOptions,
    log: impl Fn(bool) -> OuterFdDeclineLog,
) -> Option<FailureProbe> {
    let n = population.subjects.len();
    if options.agq_nodes().is_some()
        || template.mixture.is_some()
        || model.n_kappa > 0
        || eta_hats.len() != n
        || h_matrices.len() != n
    {
        return None;
    }
    let PackedStart { bounds, .. } = pack_with_bounds(template);
    let x = pack_params(params);
    let priors = build_prior_set(model, template);
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    // The objective the optimizer minimised, penalties included, at every EBE re-solved
    // warm from the reported ones.
    let objective = |xv: &[f64]| -> (f64, Vec<DVector<f64>>) {
        let (raw, etas) = reconverged_probe(xv, template, model, population, eta_hats, options);
        if raw >= 1e20 {
            return (raw, etas);
        }
        let theta = unpack_params(xv, template).theta;
        (
            raw + nn_reg.penalty_value(&theta) + priors.penalty(xv),
            etas,
        )
    };
    let gradient = |log: &OuterFdDeclineLog| -> Vec<f64> {
        let mut g = population_gradient_sens_mixed_dx(
            &x,
            template,
            model,
            population,
            eta_hats,
            h_matrices,
            &bounds,
            options,
            OuterTrial::unknown(),
            log,
            false,
        )
        .0;
        nn_reg.penalty_and_gradient(&params.theta, &mut g);
        priors.penalty_and_gradient(&x, &mut g);
        g
    };
    let fit_log = log(false);
    let g_fit = gradient(&fit_log);
    let n_declined = fit_log.declined_indices().len();
    let (f0, eta0) = objective(&x);
    let sd: Vec<f64> = (0..model.n_eta)
        .map(|k| params.omega.matrix[(k, k)].max(0.0).sqrt())
        .map(|s| if s > 0.0 { s } else { 1.0 })
        .collect();
    let probe = |g: &[f64]| probe_direction(g, &x, &bounds, f0, &eta0, &sd, &objective);
    let fit = probe(&g_fit)?;
    // Cost only: with nothing declined the exact gradient is the fit's.
    if n_declined == 0 {
        return Some(FailureProbe {
            n_declined,
            gap: 0.0,
            fit,
            exact: fit,
        });
    }
    let g_exact = gradient(&log(true));
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|a| a * a).sum::<f64>().sqrt();
    let gap = norm(&mut g_fit.iter().zip(&g_exact).map(|(a, b)| a - b))
        / norm(&mut g_exact.iter().copied());
    let exact = probe(&g_exact)?;
    Some(FailureProbe {
        n_declined,
        gap,
        fit,
        exact,
    })
}

/// The warning for a probed stop: what the objective did, and the cause it points at.
pub(crate) fn outer_failure_warning(probe: &FailureProbe, n_subjects: usize) -> String {
    let lead = "The outer optimizer stopped without converging. Probing the objective at the \
                reported estimates, with every EBE re-solved (#2713):";
    let carried = if probe.n_declined == 0 {
        " every subject has the exact analytic outer gradient there, so the fit's gradient \
         is the exact one."
            .to_string()
    } else {
        format!(
            " the fit's outer gradient carries a fallback for {} of {n_subjects} subjects \
             there and is {:.0}% off the exact one (those subjects reconverged).",
            probe.n_declined,
            100.0 * probe.gap
        )
    };
    let finding = match probe.verdict() {
        FailureVerdict::FixedEbeGradient => {
            " The objective falls along the exact gradient but not along the fit's: the \
             fixed-EBE outer gradients of the declined subjects, which omit the EBE-response \
             term, are what the line search could not descend along. Setting \
             `reconverge_gradient_interval = 1` gives them the reconverged gradient on every \
             evaluation."
                .to_string()
        }
        FailureVerdict::ModeJumps(k) => format!(
            " The objective does not fall along the exact gradient even over a step predicted \
             to lower the OFV by 0.001, and over that step the EBEs of {k} subjects switch \
             mode: the objective jumps where an EBE changes mode, which no gradient sees. \
             This is a property of the model and data (multimodal EBEs), not of the gradient; \
             SAEM, or a derivative-free optimizer (`optimizer = bobyqa`), does not rely on \
             one."
        ),
        FailureVerdict::NotDifferentiable => {
            " The objective does not fall along the exact gradient even over a step predicted \
             to lower the OFV by 0.001, and no EBE switches mode over it: the objective is not \
             differentiable at the reported estimates, for example at a kink an `if`, `abs`, \
             `max` or `min` puts in the model."
                .to_string()
        }
        FailureVerdict::NoCauseHere => {
            " The objective falls along the fit's own gradient, so the reported estimates are \
             not where the line search ran out of descent: the stop came from a trial point \
             (a fixed-EBE gradient or an EBE mode switch there) or from the line search's own \
             tolerances."
                .to_string()
        }
    };
    format!("{lead}{carried}{finding}")
}

#[cfg(test)]
#[path = "outer_optimizer_failure_tests.rs"]
mod tests;
