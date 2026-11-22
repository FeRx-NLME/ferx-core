use crate::estimation::inner_optimizer::{
    find_ebe_for_stage, run_inner_loop_warm_map, run_inner_loop_warm_seeded, stencil_inner_tol,
    GuardStop, InnerHessianSeed, InnerLoopStats, InnerSolvePolicy, PrevEtas,
};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::{
    foce_subject_nll_cov_at, foce_subject_nll_iov_cov, SUBJECT_SENTINEL_NLL,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
// `SymmetricEigen` is used only by this module's `#[cfg(test)]` code (the non-PD
// fallback tests); gate the import so a non-test build doesn't flag it unused.
#[cfg(test)]
use nalgebra::SymmetricEigen;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// The covariance/SE subsystem moved to `estimation::covariance` (refactor T4). The
// re-export exists only so this module's `#[cfg(test)] mod tests` can reach
// `compute_covariance` / `CovarianceStepResult` via `super::*`; no non-test code uses
// the historical `outer_optimizer::{..}` path, so gate it to avoid an unused-import
// warning in a non-test build.
#[cfg(test)]
pub(crate) use crate::estimation::covariance::{compute_covariance, CovarianceStepResult};

/// Result of outer optimization
/// Per-subject mixture posteriors lifted out of a converged `[mixture]` fit
/// (#977 Phase 5), threaded onto `SubjectResult.pmix` / `.mixest` in postfit.
pub struct MixturePosteriors {
    /// `PMIX_ik` — posterior class-membership probabilities per subject (each of
    /// length `K`), in subject order.
    pub pmix: Vec<Vec<f64>>,
    /// `MIXEST_i` — argmax-posterior class per subject, **0-based** (converted to
    /// the 1-based NONMEM convention when written onto `SubjectResult`).
    pub mixest: Vec<usize>,
    /// [`MixtureEval::n_fd_h_redone`](crate::estimation::mixture::MixtureEval) of the
    /// evaluation that scored the reported objective.
    pub(crate) n_fd_h_redone: usize,
    /// `MixtureEval::class_nll` of that evaluation, `[subject][class]`, for the merged-class check (#2907).
    pub(crate) class_nll: Vec<Vec<f64>>,
}

pub struct OuterResult {
    pub params: ModelParameters,
    pub ofv: f64,
    pub converged: bool,
    pub n_iterations: usize,
    pub eta_hats: Vec<DVector<f64>>,
    pub h_matrices: Vec<DMatrix<f64>>,
    /// Per-occasion kappa EBEs for each subject. Empty vecs when `n_kappa == 0`.
    pub kappas: Vec<Vec<DVector<f64>>>,
    pub covariance_matrix: Option<DMatrix<f64>>,
    /// Which estimator produced `covariance_matrix` — `R⁻¹`, `S⁻¹` or the
    /// `R⁻¹SR⁻¹` sandwich (#1382). Lifted onto
    /// [`crate::types::FitResult::covariance_method`]. Carried rather than re-derived
    /// because above [`crate::types::COV_HESSIAN_MAX_DIM`] a defaulted `r` is routed
    /// onto the cross-product (#1064). `Some` exactly when `covariance_matrix` is.
    pub covariance_method: Option<crate::types::CovarianceMethod>,
    /// Wall-clock time spent inside this stage's covariance-step block
    /// (`compute_covariance` / SIR-fallback construction), in seconds.
    /// `0.0` when `run_covariance_step` was false for this stage.
    pub covariance_wall_time_secs: f64,
    pub warnings: Vec<String>,
    /// Estimated OFV evaluations saved by the SAEM mu-ref gradient step M-step.
    /// Non-None only when method=saem and mu_referencing=true.
    pub saem_mu_ref_m_step_evals_saved: Option<u64>,
    /// NONMEM's `SAEMOBJ` of a SAEM run (`estimation::saem_objective`), accumulated over the
    /// convergence phase's draws. `None` for every other method, for a mixture or IOV model,
    /// and for a SAEM run with no convergence iteration.
    pub saem_objective_value: Option<f64>,
    /// Number of subjects that used HMC at least once during the SAEM E-step.
    /// `None` when `n_leapfrog = 0` (MH-only run) or for non-SAEM methods.
    pub saem_n_subjects_hmc: Option<usize>,
    /// Combined (primary + componentwise) MH acceptance over the trailing
    /// post-burn-in iterations the acceptance diagnostic reads. `None` for
    /// non-SAEM methods and when no post-burn-in proposal was made.
    pub saem_mh_accept_tail: Option<f64>,
    /// The numerical M-step optimizer a SAEM / IMP / IMPMAP fit resolved `mstep_optimizer` to
    /// (`newton`, `bobyqa` or `lbfgs`); `None` for other methods.
    pub mstep_optimizer_used: Option<String>,
    pub ebe_convergence_warnings: u32,
    pub max_unconverged_subjects: u32,
    pub total_ebe_fallbacks: u32,
    /// See `FitResult::total_ebe_nm_polished` (#2571).
    pub total_ebe_nm_polished: u32,
    /// Gradient at the best-OFV point in packed space. `Some` for NLopt gradient runs
    /// when a gradient iteration improved the OFV; for derivative-free NLopt it is the
    /// post-fit reporting gradient (#997, `final_gradient_source`); `None` for BFGS and SAEM.
    pub final_gradient: Option<Vec<f64>>,
    /// `"optimizer"` or `"finite_difference"` — where `final_gradient` came from;
    /// lifted onto [`crate::types::FitResult::final_gradient_source`], where the
    /// distinction is documented. `None` exactly when `final_gradient` is `None`.
    pub final_gradient_source: Option<String>,
    /// The covariance step's Newton decrement at the estimate (#2785), lifted onto
    /// [`crate::types::FitResult::newton_decrement`]. `None` when no step tested it.
    pub newton_decrement: Option<f64>,
    /// The outer optimizer's own stop state (#2785) — NLopt's status (`FtolReached`,
    /// `Failure`, …) on the NLopt path — lifted onto
    /// [`crate::types::FitResult::optimizer_stop`]. `None` where no optimizer reports one.
    pub optimizer_stop: Option<String>,
    /// Whether the outer optimizer spent its evaluation budget on the leg that produced
    /// these estimates (#2799), lifted onto [`crate::types::FitResult::eval_budget_spent`].
    /// The final state, after the latch and polish adjustments; `false` off the NLopt path.
    pub eval_budget_spent: bool,
    /// Fallback proposal covariance for the SIR sampler, set when the FD
    /// Hessian is non-PD. Built from the `|eigenvalue|`-rectified free-block
    /// Hessian, inflated 4×, and embedded into the full packed parameter space.
    /// `None` when the Hessian succeeded or the covariance step was skipped.
    pub sir_fallback_proposal: Option<DMatrix<f64>>,
    /// Per-iteration parameter trace from IMPMAP. `None` for all other methods.
    pub impmap_trace: Option<crate::types::ImpmapTrace>,
    /// Posterior summaries + diagnostics from a Bayesian (`method=bayes`) run.
    /// `Some` only for `EstimationMethod::Bayes`; `None` for all point
    /// estimators. Carried here so the chain dispatch can lift it onto
    /// `FitResult.bayes` through the generic OuterResult → FitResult path.
    pub bayes: Option<crate::types::BayesResult>,
    /// Per-subject conditional distribution of the random effects, estimated by
    /// the post-fit SAEM conditional-distribution pass. `Some` only when
    /// `method = saem` and `saem_conddist = true`; `None` for every other
    /// estimator and for SAEM runs that did not request the pass (#257).
    pub cond_dist: Option<CondDist>,
    /// The optimizer's **exact** final packed vector over the free parameters — the one
    /// the inline covariance step used. `Some` for every packed-Cholesky optimizer (NLopt,
    /// BFGS, trust region, Gauss-Newton), including `outer_maxiter = 0`; `None` for SAEM,
    /// IMP (they rebuild `omega` from the matrix, which re-decomposes identically) and Bayes.
    /// Lets `run_covariance` reproduce the inline FD Hessian bit-for-bit: `omega → chol` is
    /// not the exact inverse of the stored `L·Lᵀ`, and the FD Hessian amplifies the gap.
    pub packed_estimate: Option<Vec<f64>>,
    /// Whether the fit left its initial estimates by the optimizer's own
    /// `INIT_ESCAPE_STEP_S` test (#751), measured on the restored best point.
    /// `Some` for the NLopt outer loop (`optimize_nlopt`) and for a #2506 start-probe
    /// result on the other outer optimizers (`verify_unmoved_start`), `None` otherwise;
    /// lifted onto `FitResult::left_init`.
    pub left_init: Option<bool>,
    /// This is the result of the #2506 start probe, which verified the initial estimates to
    /// be an optimum — a restart from a perturbation of them converged back to their
    /// objective — and its estimates are still within the escape step of them
    /// (`left_init == Some(false)`). A warm start at an optimum, not a #751 stall. `false`
    /// otherwise; lifted onto `FitResult::stationary_start`.
    pub stationary_start: bool,
    /// Per-subject mixture posteriors from the final mixture eval. `Some` only for
    /// a converged `[mixture]` fit (#977); `None` for every non-mixture path.
    pub mixture_posteriors: Option<MixturePosteriors>,
    /// Variational-inference result from a `method = vi` run. `Some` only for
    /// `EstimationMethod::Vi`; carried here so the chain dispatch lifts it onto
    /// `FitResult.vi` through the same generic path `bayes` uses.
    pub vi: Option<crate::types::ViResult>,
    /// The fit's first evaluation, unguarded — the reference [`above_start_warning`] holds
    /// the reported objective against (#2461). `Some` for the NLopt and profiled loops
    /// (#2519). It is the **fit's** start: restart legs inherit it through [`RestartLeg`]
    /// and the floor escape (#2339) passes it on, so a restart cannot pass against its own.
    pub(crate) start_eval: Option<StartEval>,
    /// The lowest −2 log L an IMP / IMPMAP iteration reached; `None` for every other
    /// estimator, and when no iteration was finite. `fit()` compares the final marginal
    /// against it (`crate::api::postfit::mcem_marginal_warning`, #2652).
    pub(crate) mcem_lowest_objective: Option<f64>,
    /// The per-iteration objectives of an IMP / IMPMAP run; `None` for every other
    /// estimator, and for a run its wall-clock limit stopped. `fit()` tests them for settling
    /// (`crate::api::postfit::mcem_settling_warning`, #2686).
    pub(crate) mcem_objective_trail: Option<crate::estimation::impmap::McemObjectiveTrail>,
}

impl OuterResult {
    /// An unconverged result at `params` with every other field empty, zero or `None`:
    /// the base each estimator's literal fills in with struct-update syntax
    /// (`OuterResult { ofv, converged, .., ..OuterResult::from_params(params) }`), so a
    /// field one estimator does not report needs no line in all the others.
    pub(crate) fn from_params(params: ModelParameters) -> Self {
        OuterResult {
            params,
            ofv: 0.0,
            converged: false,
            n_iterations: 0,
            eta_hats: Vec::new(),
            h_matrices: Vec::new(),
            kappas: Vec::new(),
            covariance_matrix: None,
            covariance_method: None,
            covariance_wall_time_secs: 0.0,
            warnings: Vec::new(),
            saem_mu_ref_m_step_evals_saved: None,
            saem_objective_value: None,
            saem_n_subjects_hmc: None,
            saem_mh_accept_tail: None,
            mstep_optimizer_used: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            total_ebe_nm_polished: 0,
            final_gradient: None,
            final_gradient_source: None,
            newton_decrement: None,
            optimizer_stop: None,
            eval_budget_spent: false,
            sir_fallback_proposal: None,
            impmap_trace: None,
            bayes: None,
            cond_dist: None,
            packed_estimate: None,
            left_init: None,
            stationary_start: false,
            mixture_posteriors: None,
            vi: None,
            start_eval: None,
            mcem_lowest_objective: None,
            mcem_objective_trail: None,
        }
    }
}

/// Whether this fit runs an outer optimizer at all. `outer_maxiter == 0` is an
/// evaluation-only run (NONMEM `MAXEVAL=0`) where no optimizer is constructed (see
/// [`optimize_population`]); shared so diagnostics cannot claim one ran (#1381).
pub(crate) fn runs_outer_optimizer(options: &FitOptions) -> bool {
    options.outer_maxiter > 0
}

/// The outer optimizer a fit actually runs, and the warning owed when an explicit
/// choice could not be honoured. **Single source of truth for `auto`**:
/// `Optimizer::resolve_auto` alone is wrong on a mixture model (#1381).
/// `analytic_outer_gradient` is a parameter so the same rule answers the
/// counterfactual "what would `auto` pick had the gradient not been forced?".
pub(crate) fn resolve_outer_optimizer(
    requested: Optimizer, model: &CompiledModel, has_mixture: bool, analytic_outer_gradient: bool,
) -> (Optimizer, Option<String>) {
    // Mixture models (#977, #3025). An analytic posterior-weighted outer gradient exists
    // for the mixture objective, so `auto` resolves exactly as for a single population
    // (NLopt L-BFGS on the exact gradient where it is available, BOBYQA where it is
    // not), and an explicit NLopt optimizer -- gradient-based or BOBYQA -- is honoured
    // without a warning (#2852): those route through `optimize_nlopt`, whose objective
    // closure branches to `mixture_gradient`. The built-in BFGS / L-BFGS and
    // trust-region paths do not carry the mixture objective and fall back to what
    // `auto` resolves to (L-BFGS on the exact gradient, BOBYQA without one: never a
    // gradient-based optimizer on a gradient the loop does not form), with a warning.
    if has_mixture {
        // Exhaustive, no wildcard: a new variant must say here whether it carries
        // the mixture objective.
        let resolved = match requested {
            Optimizer::Bobyqa | Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma => {
                requested
            }
            Optimizer::Auto | Optimizer::Bfgs | Optimizer::Lbfgs | Optimizer::TrustRegion => {
                Optimizer::Auto.resolve_auto_given_analytic(model, analytic_outer_gradient)
            }
        };
        // The warning is owed exactly when an *explicit* choice did not run — keyed
        // on the outcome, not on a second list of variants, so "replaced" can never
        // be said of an optimizer that ran as requested (#2852). `Auto` asked for
        // no optimizer in particular and resolves silently by design.
        let replaced = requested != Optimizer::Auto && resolved != requested;
        let warning = replaced.then(|| {
            format!(
                "Mixture models are optimized with an NLopt method (L-BFGS / SLSQP / MMA / \
                 BOBYQA); the requested `{}` optimizer does not carry the mixture \
                 objective and was replaced by {}.",
                requested.label(),
                if resolved == Optimizer::NloptLbfgs {
                    "NLopt L-BFGS"
                } else {
                    "BOBYQA (no analytic outer gradient)"
                }
            )
        });
        return (resolved, warning);
    }
    (requested.resolve_auto_given_analytic(model, analytic_outer_gradient), None)
}

/// Run the outer optimizer for one stage from `init_params`. The returned `converged`
/// carries **no** above-start verdict (#2461): only [`crate::fit`] checks the result
/// against the first evaluation, so a direct caller may get `converged: true` above
/// its start objective.
pub fn optimize_population(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    // A population handed as read has no level cells under nested levels; give them here, as
    // `fit()` does, rather than have every subject refused (#3004).
    let cells = crate::stats::nested_levels::with_level_cells(model, population);
    let population = &*cells;
    // `outer_maxiter == 0` requests an evaluation-only run (NONMEM `MAXEVAL=0`):
    // report the objective at the *initial* parameters with no minimisation. This
    // must short-circuit before any optimizer is constructed, because NLopt's
    // `set_maxeval(0)` means "no limit", not "zero evaluations" — so the gradient
    // NLopt path (`NloptLbfgs`/`Slsqp`/`Mma`) would otherwise run a *full fit* on
    // a `maxiter = 0` request, making the reported OFV an optimizer- and
    // platform-dependent converged value rather than the deterministic θ₀ check
    // callers expect (see #562: ferx-r's `settings = list(maxiter = 0)` silently
    // ran to convergence, so `two_cpt_oral_cov_ode`'s "init" OFV diverged ~534
    // from its analytical sibling on x86 Linux). BOBYQA and the built-in BFGS loop
    // already honour 0, but routing every optimizer through one eval-only path
    // keeps the semantics uniform.
    if !runs_outer_optimizer(options) {
        return evaluate_at_initial_params(model, population, init_params, options);
    }
    // Resolve `auto` once here so the concrete optimizer flows through the outer loop
    // (`resolve_outer_optimizer` owns the mixture arm and the `auto` rule, #1381).
    let (resolved, downgrade) = resolve_outer_optimizer(
        options.optimizer,
        model,
        init_params.mixture.is_some(),
        crate::sens::provider::analytic_outer_gradient_available(model),
    );
    let optimizer_downgrade_warning: Vec<String> = downgrade.into_iter().collect();
    let owned_opts;
    let options = if resolved == options.optimizer {
        options
    } else {
        owned_opts = FitOptions { optimizer: resolved, ..options.clone() };
        &owned_opts
    };
    // Records which subjects the *optimizer's* outer-gradient evaluations salvaged off the
    // exact analytic gradient (#1154). Created after the `runs_outer_optimizer` short-circuit
    // above, so an evaluation-only run — which computes no outer gradient — cannot report
    // one. The flat-theta pre-flight below writes to a log of its own that is dropped: it
    // runs for derivative-free BOBYQA and trust-region fits too, and a decline recorded
    // there would make those fits warn about — and give advice for — a gradient their
    // optimizer never used (#1529 review). A gradient-driven optimizer re-evaluates the
    // same start point on its first iteration, so nothing it would report is lost.
    let declines = OuterFdDeclineLog::new(population.subjects.len());
    let preflight_declines = OuterFdDeclineLog::new(population.subjects.len());

    // Pre-flight flat-theta guard (#826): a free theta whose outer gradient is identically
    // ~0 at the start never reaches the objective, and gives gradient NLopt a zero direction
    // that fails on eval 1. Freeze such thetas (as FIX) and warn. Skipped for mixtures (the
    // probe is the single-population gradient and would mis-freeze mixing logits).
    // When nothing is frozen the first evaluation is warm-started from the pre-flight's
    // EBEs (#2076) instead of repeating the cold multi-start (#1941); when a theta is
    // frozen the template differs, so the first evaluation solves cold.
    let frozen_params;
    //
    // The pre-flight also probes near-zero identity-packed θ for the NLopt scale vector
    // (#2159, `identity_theta_scales`); a mixture is probed on its own objective. Only
    // the NLopt optimizers without an explicit `none` / `rescale2` read the result.
    let probe_identity_scales =
        matches!(
            resolved,
            Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma | Optimizer::Bobyqa
        ) && matches!(options.parameter_scaling, ParameterScaling::Auto | ParameterScaling::Abs);
    let (init_params, preflight_warnings, first_eval_etas, identity_scales) =
        if init_params.mixture.is_some() {
            let scales = if probe_identity_scales {
                mixture_identity_theta_scales(model, population, init_params, options)
            } else {
                Vec::new()
            };
            (init_params, Vec::new(), None, scales)
        } else {
            let preflight = freeze_flat_thetas(
                model,
                population,
                init_params,
                options,
                &preflight_declines,
                probe_identity_scales,
            );
            match preflight.frozen {
                Some((fp, w)) => {
                    frozen_params = fp;
                    (&frozen_params, w, None, preflight.identity_scales)
                }
                None => (init_params, Vec::new(), preflight.etas, preflight.identity_scales),
            }
        };
    let first_eval_etas = first_eval_etas.as_deref();

    let mut result = match resolved {
        // `Auto` is resolved away above; group it with the NLopt path defensively.
        Optimizer::Slsqp
        | Optimizer::NloptLbfgs
        | Optimizer::Mma
        | Optimizer::Bobyqa
        | Optimizer::Auto => optimize_nlopt(
            model,
            population,
            init_params,
            options,
            &declines,
            first_eval_etas,
            &identity_scales,
        ),
        // #2506: these publish no `left_init`, so a run that did not move is probed here,
        // on what is left of `outer_maxiter`.
        Optimizer::Bfgs | Optimizer::Lbfgs => {
            let result =
                optimize_bfgs(model, population, init_params, options, &declines, first_eval_etas);
            let budget = options.outer_maxiter.saturating_sub(result.n_iterations);
            verify_unmoved_start(
                model,
                population,
                init_params,
                options,
                result,
                budget,
                |x, b| bfgs_scale(x, b, init_params, options),
                |p, o| optimize_bfgs(model, population, p, o, &declines, None),
            )
        }
        Optimizer::TrustRegion => {
            let result = crate::estimation::trust_region::optimize_trust_region_seeded(
                model,
                population,
                init_params,
                options,
                first_eval_etas,
            );
            let budget = options.outer_maxiter.saturating_sub(result.n_iterations);
            verify_unmoved_start(
                model,
                population,
                init_params,
                options,
                result,
                budget,
                |x, _| vec![1.0; x.len()],
                |p, o| {
                    crate::estimation::trust_region::optimize_trust_region_seeded(
                        model, population, p, o, None,
                    )
                },
            )
        }
    };
    // Surface the optimizer-downgrade and freeze warnings ahead of the optimizer's
    // own (they explain the substituted optimizer / why a parameter was held fixed,
    // which the reader wants before any convergence notes).
    if !optimizer_downgrade_warning.is_empty() || !preflight_warnings.is_empty() {
        let mut w = optimizer_downgrade_warning;
        w.extend(preflight_warnings);
        w.append(&mut result.warnings);
        result.warnings = w;
    }
    // Per-subject outer FD fallbacks that actually happened (#1154). Emitted here, from
    // the runtime log, rather than from a probe in `fit_inner`: only this scope knows
    // which evaluations ran, and the log is empty for every route that never reaches the
    // analytic branch — so no route gate is needed and none can go stale.
    if let Some(w) = outer_fd_fallback_warning(population, &declines) {
        result.warnings.push(w);
    }
    result
}

/// What the flat-theta pre-flight ([`freeze_flat_thetas`]) hands back to
/// [`optimize_population`].
struct FlatThetaPreflight {
    /// The initial estimates with the flat thetas fixed, and one warning per
    /// frozen theta; `None` when nothing is flat.
    frozen: Option<(ModelParameters, Vec<String>)>,
    /// The EBEs of the pre-flight's cold inner solve at the packed, clamped start — where
    /// the first evaluation solves again when nothing is frozen (#2076). Converged modes,
    /// taken as warm starts ([`InnerSolvePolicy::init_is_mode`]); an IOV entry is `η̂` then
    /// each `κ̂` ([`joint_warm_starts`], #2217). `None` if nothing was solved or any
    /// subject did not converge.
    etas: Option<Vec<DVector<f64>>>,
    /// `(θ index, scale)` for each near-zero identity-packed θ the NLopt outer loop
    /// divides by its initial magnitude — see [`identity_theta_scales`] (#2159).
    /// Empty unless the caller asked for the probe.
    identity_scales: Vec<(usize, f64)>,
}

/// Pre-flight flat-theta detection (#826): flags free thetas whose start gradient is
/// negligible, then **confirms** each with a perturbation probe — only a theta that
/// leaves the reconverged objective exactly unchanged is frozen (a tiny start gradient
/// alone can be coincidental, and freezing it biases the fit).
/// `frozen` is `None` when nothing is flat, else a clone with the flat thetas marked
/// `theta_fixed` plus warnings. `etas`: see [`FlatThetaPreflight`].
fn freeze_flat_thetas(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, declines: &OuterFdDeclineLog, probe_identity_scales: bool,
) -> FlatThetaPreflight {
    let n_theta = init_params.theta.len();
    // Nothing to freeze if every theta is already fixed.
    if (0..n_theta).all(|i| init_params.theta_fixed.get(i).copied().unwrap_or(false)) {
        return FlatThetaPreflight { frozen: None, etas: None, identity_scales: Vec::new() };
    }

    let PackedStart { packed: mut x, bounds, .. } = pack_with_bounds(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let params = unpack_params(&x, init_params);
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    // One cold inner solve, then one outer-gradient eval — the same pair the first
    // optimizer iteration would run, so the pre-flight costs roughly one iteration.
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let cold_etas = vec![DVector::zeros(n_eta); n_subj];
    // `run_inner_loop_warm_seeded`'s solve, with each subject's convergence read back: the
    // EBEs are handed to the first evaluation only if every subject reached a mode.
    let (ehs, hms, base_stats, kappas, converged) = run_inner_loop_warm_map(
        model,
        population,
        &params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&cold_etas),
        Some(&mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
        InnerSolvePolicy {
            seed: InnerHessianSeed::for_options(options),
            ..InnerSolvePolicy::for_stage(options, model)
        },
        None,
        |_, ebe| ebe.converged && !ebe.hard_reject,
    );
    // The objective at the start, and at the start with one θ moved and the EBEs
    // re-solved: shared by the flat-θ probe below and `identity_theta_scales`. Both
    // are needed only when one of them has a candidate, so the base is evaluated
    // once, on first use.
    let base_cell = std::cell::OnceCell::new();
    let base_ofv = || {
        *base_cell.get_or_init(|| {
            2.0 * pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options)
        })
    };
    // With the inner-loop statistics, which the #2159 probe hands to the EBE guard.
    let reconverged = |theta_i: usize, value: f64| -> (f64, InnerLoopStats) {
        let mut p = params.clone();
        p.theta[theta_i] = value;
        let mu = compute_mu_k(model, &p.theta, options.mu_referencing);
        let (e, h, stats, k) = run_inner_loop_warm_seeded(
            model,
            population,
            &p,
            options.inner_maxiter,
            options.inner_tol,
            Some(&cold_etas),
            Some(&mu),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
            InnerHessianSeed::for_options(options),
            options,
            PrevEtas::Placeholder,
        );
        (2.0 * pop_nll_opts(model, population, &p, &e, &h, &k, options), stats)
    };
    let reconverged_ofv = |theta_i: usize, value: f64| reconverged(theta_i, value).0;
    // The clean −2LL gradient at the start: the flat-θ test below, and the slope of the
    // #2159 identity-scale rule.
    let mut grad_eval_idx = 0usize;
    let grad = population_gradient(
        &x,
        n_subj,
        init_params,
        model,
        population,
        &ehs,
        &hms,
        &kappas,
        &bounds,
        options,
        &mut grad_eval_idx,
        OuterTrial::unknown(),
        declines,
    );
    let frozen = 'freeze: {
        let is_free = |i: usize| !init_params.theta_fixed.get(i).copied().unwrap_or(false);
        let Some(candidates) = flat_theta_candidates(&grad, n_theta, is_free) else {
            break 'freeze None;
        };

        // Confirm flatness with a perturbation probe: an identifiable theta can have a
        // coincidentally tiny start gradient (e.g. a hazard baseline `H0` at `BETA = 0`; #826
        // froze one and biased a joint PK-TTE fit). Only a theta that leaves the reconverged
        // objective *exactly* unchanged is frozen. One inner solve per (rare) candidate.
        // True ⇒ moving this theta changes the objective ⇒ identifiable, not flat.
        let moves_objective = |i: usize| -> bool {
            let ti = params.theta[i];
            let (lo, hi) = (init_params.theta_lower[i], init_params.theta_upper[i]);
            // Could not build a distinct in-bounds probe — do not freeze (safe side).
            let Some(probe) = flat_theta_probe_value(ti, lo, hi) else {
                return true;
            };
            flat_theta_probe_moves(base_ofv(), reconverged_ofv(i, probe))
        };
        let flat: Vec<usize> = candidates.into_iter().filter(|&i| !moves_objective(i)).collect();
        if flat.is_empty() {
            break 'freeze None;
        }

        let mut frozen = init_params.clone();
        let mut warnings = Vec::new();
        for &i in &flat {
            // Pin the FIX at the *clamped* value the gradient was actually evaluated at
            // (`params.theta`, not the raw `init_params.theta`): an out-of-bounds initial
            // theta must not be frozen — nor reported — outside its declared bounds, and
            // the printed value must match the point the flatness was decided at.
            frozen.theta[i] = params.theta[i];
            frozen.theta_fixed[i] = true;
            let name = init_params.theta_names.get(i).map(|s| s.as_str()).unwrap_or("<theta>");
            warnings.push(flat_theta_warning(name, params.theta[i]));
        }
        Some((frozen, warnings))
    };
    // An unconverged subject is not a valid warm start (it would skip the cold multi-start
    // and change the first objective), so then all are dropped (#2076).
    // #2159: probed after the flat-θ freeze so a frozen θ is not probed again.
    let identity_scales = if probe_identity_scales {
        let held = |i: usize| {
            init_params.theta_fixed.get(i).copied().unwrap_or(false)
                || frozen.as_ref().is_some_and(|(fp, _)| fp.theta_fixed[i])
        };
        // The probe sees the objective the optimizer minimises — penalties included and
        // the EBE guard applied; the flat-θ probe above keeps the clean −2LL, since a θ
        // only a prior reaches is still unmapped.
        let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
        let priors = build_prior_set(model, init_params);
        let with_theta = |i: usize, value: f64| {
            let mut p = params.clone();
            p.theta[i] = value;
            p
        };
        let objective = |clean: f64, p: &ModelParameters, stats: &InnerLoopStats| {
            identity_probe_objective(
                clean + identity_probe_penalty(p, &nn_reg, &priors),
                stats,
                n_subj,
                options.max_unconverged_frac,
            )
        };
        // The slope and curvature of the penalized objective along each candidate, from the
        // gradient above and the exact R-matrix's candidate block; `None` (and the probes)
        // when a subject is outside the recorders' scope.
        let exact = |candidates: &[usize]| {
            identity_exact_derivatives(
                model,
                population,
                init_params,
                &params.theta,
                &x,
                &ehs,
                options,
                (&nn_reg, &priors),
                candidates,
            )
        };
        // Whether the optimizer could take a move of the θ's own size, without re-solving:
        // the penalized objective at the start's EBEs is finite and below the `1e20` wall.
        // The exact rule's stand-in for the probes' "an infeasible probe scales the θ" arm
        // (#2291 review).
        let feasible = |i: usize, value: f64| {
            let p = with_theta(i, value);
            let v = 2.0 * pop_nll_opts(model, population, &p, &ehs, &hms, &kappas, options)
                + identity_probe_penalty(&p, &nn_reg, &priors);
            ofv_is_valid(v)
        };
        identity_theta_scales(
            &params.theta,
            &init_params.theta_lower,
            &init_params.theta_upper,
            held,
            || objective(base_ofv(), &params, &base_stats),
            |candidates: &[usize]| {
                #[cfg(test)]
                if IDENTITY_EXACT_OFF.with(std::cell::Cell::get) {
                    return None;
                }
                exact(candidates)
            },
            feasible,
            |i, value| {
                #[cfg(test)]
                IDENTITY_PROBE_SOLVES.with(|c| c.set(c.get() + 1));
                let (clean, stats) = reconverged(i, value);
                objective(clean, &with_theta(i, value), &stats)
            },
        )
    } else {
        Vec::new()
    };
    FlatThetaPreflight {
        frozen,
        etas: converged.iter().all(|&c| c).then(|| joint_warm_starts(&ehs, &kappas)),
        identity_scales,
    }
}

/// The flat-θ candidates of the #826 pre-flight: each free θ whose outer gradient is
/// negligible, absolutely and against the largest free θ gradient; `None` when there is
/// none or the whole θ gradient is ~0. θ are the first `n_theta` packed coordinates, so `grad[i]`
/// is d(OFV)/d(coord_i) for θ `i`. Only the machine-precision-zero case is frozen,
/// never a weakly identified θ. Shared by [`freeze_flat_thetas`] and the profiled loop
/// (#2136) so the two agree.
pub(crate) fn flat_theta_candidates(
    grad: &[f64], n_theta: usize, is_free: impl Fn(usize) -> bool,
) -> Option<Vec<usize>> {
    const FLAT_ABS: f64 = 1e-8;
    const FLAT_REL: f64 = 1e-6;
    let g_at = |i: usize| grad.get(i).copied().unwrap_or(0.0).abs();
    // Reference scale for "negligible". If the whole theta gradient is ~0 the
    // objective is globally flat (no data reaches it) — a different pathology, so
    // bail rather than freeze every parameter.
    let g_max = (0..n_theta).filter(|&i| is_free(i)).map(g_at).fold(0.0_f64, f64::max);
    if g_max <= FLAT_ABS {
        return None;
    }
    let candidates: Vec<usize> = (0..n_theta)
        .filter(|&i| is_free(i) && g_at(i) <= FLAT_ABS && g_at(i) <= FLAT_REL * g_max)
        .collect();
    (!candidates.is_empty()).then_some(candidates)
}

/// The value the #826 perturbation probe moves a flat-θ candidate at `ti` to, inside
/// `(lo, hi)`; `None` when no distinct in-bounds probe exists (then the θ is not frozen).
pub(crate) fn flat_theta_probe_value(ti: f64, lo: f64, hi: f64) -> Option<f64> {
    let delta = (ti.abs() * 0.5).max((hi - lo).abs() * 0.1).max(1e-2);
    let mut probe = ti + delta;
    if !(probe < hi) {
        probe = ti - delta;
    }
    if !(probe > lo) {
        probe = 0.5 * (lo + hi);
    }
    ((probe - ti).abs() > 1e-12).then_some(probe)
}

/// Whether the probe moved the reconverged objective from `base` to `probed` — i.e. the
/// candidate is identifiable, not flat.
pub(crate) fn flat_theta_probe_moves(base: f64, probed: f64) -> bool {
    (probed - base).abs() > 1e-6 * (1.0 + base.abs())
}

/// The warning for a θ the #826 pre-flight froze at `val`.
pub(crate) fn flat_theta_warning(name: &str, val: f64) -> String {
    format!(
        "[parameters] `{name}` has no effect on the objective (gradient ≈ 0 at the \
         initial estimate) — it is likely computed but never used (unmapped, or dropped \
         from the structural / scaling model). Freezing it at its initial value ({val}) \
         so the remaining parameters can be estimated; map or remove `{name}` to silence \
         this."
    )
}

/// Slope and curvature `(∂f/∂x_i, ∂²f/∂x_i²)` of the penalized objective along each
/// candidate at the start (#2159): exact −2LL derivatives
/// ([`crate::estimation::covariance::exact_candidate_derivatives`]) plus the penalties'.
/// `None` when not exactly available (subject out of recorder scope or unconverged, exact
/// R-matrix off, or an active NN smoothness penalty — Gauss–Newton Hessian only).
#[allow(clippy::too_many_arguments)]
fn identity_exact_derivatives(
    model: &CompiledModel, population: &Population, template: &ModelParameters, theta: &[f64],
    x: &[f64], warm: &[DVector<f64>], options: &FitOptions,
    (nn_reg, priors): (
        &crate::estimation::nn_reg::NnRegularizer,
        &crate::estimation::priors::PriorSet,
    ),
    candidates: &[usize],
) -> Option<Vec<(f64, f64)>> {
    if !nn_reg.hessian_is_exact() {
        return None;
    }
    let clean = crate::estimation::covariance::exact_candidate_derivatives(
        model, population, template, x, warm, options, candidates,
    )?;
    let mut g = vec![0.0; x.len()];
    nn_reg.penalty_and_gradient(theta, &mut g);
    priors.penalty_and_gradient(x, &mut g);
    let mut pen = vec![0.0; candidates.len()];
    let mut add_diagonal = |a: usize, b: usize, v: f64| {
        if a == b {
            if let Some(c) = candidates.iter().position(|&i| i == a) {
                pen[c] += v;
            }
        }
    };
    nn_reg.add_packed_hessian(theta, &mut add_diagonal);
    priors.add_hessian(&mut add_diagonal);
    Some(
        candidates
            .iter()
            .zip(clean)
            .zip(pen)
            .map(|((&i, (gc, hc)), hp)| (gc + g.get(i).copied().unwrap_or(f64::NAN), hc + hp))
            .collect(),
    )
}

/// An identity-packed θ (negative lower bound) whose initial magnitude is below this
/// is a candidate for [`identity_theta_scales`]'s probe (#2159). Above it the NLopt
/// scale of 1 is within a factor of ten of the θ's own size.
const NEAR_ZERO_IDENTITY_THETA: f64 = 0.1;
/// The probe spacing, as a fraction of `|θ₀|`: the probes sit at `θ₀ ∓ h·|θ₀|`, or —
/// when the box has no room for that on one side — at `θ₀ + h·|θ₀|` and
/// `θ₀ + 2h·|θ₀|` on the side that has it, with `h` reduced as far as the box
/// requires (see `identity_probe_points`).
const IDENTITY_PROBE_FRACTION: f64 = 0.5;
/// Below this `h` the box leaves no room for probes, and the θ keeps the unit scale.
const IDENTITY_PROBE_MIN_FRACTION: f64 = 0.05;
/// A candidate is scaled when the minimizer the probes predict lies within this many
/// multiples of `|θ₀|` of the start.
///
/// Measured at the issue's two kinds of near-zero θ (macOS arm64, FOCEI objective
/// at the start, `h = ½`): warfarin's `TVV_S` (`V = TVV_S / 1e-6`, θ₀ = 1e-5) has a
/// second difference of 124 and predicts its minimizer 0.12·|θ₀| away (it is
/// 0.23·|θ₀| away, at 7.74e-6); in the two-class `mixture_iv` model the same volume
/// predicts 0.02·|θ₀|. `two_cpt_oral_cov`'s covariate exponents started at 0.001
/// have second differences of 6.3e-6 and 1.1e-5, under the 1.0e-3 noise floor at
/// OFV −1033, so they keep scale 1 before the bar is consulted; read anyway, their
/// predicted steps are 708 and 261 `|θ₀|` (the minimizers are 0.65 and ≈ 0.3). The
/// bar sits about two orders of magnitude from each side.
const IDENTITY_MAX_PREDICTED_STEP: f64 = 10.0;

/// The near-zero identity-packed θ the NLopt outer loop divides by their initial
/// magnitude, as `(θ index, |θ₀|)` (#2159).
///
/// An identity-packed θ gets scale 1 in the NLopt scale vector. Started at 1e-5,
/// that is a coordinate whose useful moves are five orders of magnitude below the
/// optimizer's unit step and its stopping tolerances: BOBYQA's first probes put
/// warfarin's `TVV_S` (`V = TVV_S / 1e-6`) at a negative volume and the fit stops at
/// OFV −205.97 against −286.00, SLSQP at −198.95, L-BFGS at −269.47, each reporting
/// convergence.
///
/// Dividing every such θ by |θ₀| is wrong the other way: a covariate exponent started
/// at 0.001 is usually an O(1) slope, and scaling it by 0.001 cost BOBYQA 14 OFV
/// units on `two_cpt_oral_cov`. The value alone cannot tell the two apart; the
/// objective can. Each candidate — free, identity-packed, `0 < |θ₀| <`
/// [`NEAR_ZERO_IDENTITY_THETA`] — is moved to two more points inside its declared
/// bounds (`identity_probe_points`: `θ₀ ∓ δ` with `δ = ½|θ₀|`, or one-sided
/// `θ₀ ± δ, θ₀ ± 2δ` on the side with room when `θ₀` sits near a bound) with the
/// EBEs re-solved (`probe_ofv`). The quadratic through the three equally spaced
/// points `x₀ < x₁ < x₂` (spacing δ, objectives `f₀, f₁, f₂`) has its minimizer at
/// `x* = x₁ − δ·(f₂ − f₀) / (2·(f₀ − 2f₁ + f₂))`, a Newton step from θ₀ of
/// `|x* − θ₀| / |θ₀|` in units of the θ's own size.
///
/// It is scaled when that step is at most [`IDENTITY_MAX_PREDICTED_STEP`]·|θ₀|: the
/// moves that matter are then of the θ's own size. The statistic is a **ratio of
/// objective differences**, so it does not change when every likelihood contribution
/// is repeated `k` times — more subjects, or more observations per subject — which
/// multiplies the objective by `k` and leaves its minimizer where it was (#2159
/// review: a cut on the size of a ΔOFV, total or per subject, does change).
///
/// The θ keeps the unit scale when the second difference shows no positive curvature
/// above `1e-6·(1 + |f(θ₀)|)` — the flat-θ probe's "objective unchanged" tolerance,
/// relative so that it scales with the objective too — since a step predicted from
/// noise says nothing. A probe whose objective is not valid ([`ofv_is_valid`]: `NaN`,
/// diverged, or rejected by the EBE guard, which the callers map to the optimizer's
/// `1e20` wall — `identity_probe_objective`) means a move of the θ's own size is one
/// the optimizer cannot take, and the θ is scaled. Two extra objective evaluations
/// per candidate.
///
/// `held(i)` is true for a θ the optimizer does not move (fixed, or frozen by the
/// flat-θ pre-flight). `base_ofv` is the objective at the start, asked for only when
/// there is a candidate; a start whose own objective is not valid says nothing about
/// scale, and nothing is probed. Both objectives must be the one the optimizer
/// minimises, penalties included (`penalized_identity_probe`).
fn identity_theta_scales(
    theta: &[f64], theta_lower: &[f64], theta_upper: &[f64], held: impl Fn(usize) -> bool,
    base_ofv: impl FnOnce() -> f64, exact: impl FnOnce(&[usize]) -> Option<Vec<(f64, f64)>>,
    mut feasible: impl FnMut(usize, f64) -> bool, mut probe_ofv: impl FnMut(usize, f64) -> f64,
) -> Vec<(usize, f64)> {
    let n = theta.len().min(theta_lower.len()).min(theta_upper.len());
    let candidates: Vec<usize> = (0..n)
        .filter(|&i| {
            let magnitude = theta[i].abs();
            !held(i)
                && !theta_packs_log(theta_lower[i])
                && magnitude > 0.0
                && magnitude < NEAR_ZERO_IDENTITY_THETA
        })
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    let base = base_ofv();
    if !ofv_is_valid(base) {
        return Vec::new();
    }
    // The same quadratic from exact derivatives at θ₀, when every subject has them: its
    // second difference over the probe spacing is `f″·δ²` and its minimizer is the Newton
    // step `−f′/f″` (#2159 follow-up).
    let exact = exact(&candidates).filter(|d| {
        d.len() == candidates.len() && d.iter().all(|(g, h)| g.is_finite() && h.is_finite())
    });
    let mut scales = Vec::new();
    for (c, i) in candidates.into_iter().enumerate() {
        let t = theta[i];
        let magnitude = t.abs();
        let Some(points) = identity_probe_points(t, theta_lower[i], theta_upper[i]) else {
            continue;
        };
        if let Some(d) = &exact {
            let (g, h) = d[c];
            let delta = points[1] - points[0];
            let scaled = h * delta * delta > 1e-6 * (1.0 + base.abs())
                && (g / h).abs() <= IDENTITY_MAX_PREDICTED_STEP * magnitude;
            // The probes' other arm: a move of the θ's own size the optimizer cannot take
            // (a non-finite or walled objective) scales it, whatever the local derivatives
            // say. Checked at the probe points with the EBEs held, only for a θ the
            // derivatives would leave on the unit scale.
            if scaled || points.iter().any(|&p| p != t && !feasible(i, p)) {
                scales.push((i, magnitude));
            }
            continue;
        }
        // Objectives at the three points in ascending order; θ₀ is one of them.
        let mut f = [0.0; 3];
        for (fk, &xk) in f.iter_mut().zip(&points) {
            *fk = if xk == t { base } else { probe_ofv(i, xk) };
        }
        if !f.iter().all(|&v| ofv_is_valid(v)) {
            scales.push((i, magnitude));
            continue;
        }
        let curvature = f[0] - 2.0 * f[1] + f[2];
        if !(curvature > 1e-6 * (1.0 + base.abs())) {
            continue;
        }
        let delta = points[1] - points[0];
        let minimizer = points[1] - delta * (f[2] - f[0]) / (2.0 * curvature);
        if (minimizer - t).abs() <= IDENTITY_MAX_PREDICTED_STEP * magnitude {
            scales.push((i, magnitude));
        }
    }
    scales
}

/// The three equally spaced probe points of [`identity_theta_scales`], ascending and
/// all inside `[lower, upper]`, one of them `t` itself; `None` when the box leaves
/// no room for a spacing of at least [`IDENTITY_PROBE_MIN_FRACTION`]`·|t|`.
///
/// Central (`t − δ, t, t + δ`) when both sides have room for the full
/// `δ = IDENTITY_PROBE_FRACTION·|t|`; otherwise whichever of the central and the
/// one-sided construction (`t, t + δ, t + 2δ` toward the side with more room)
/// affords the larger spacing, each `δ` kept to 90% of the room it uses. Neither
/// side is assumed: a negative lower bound does not put zero in the box when the
/// upper bound is negative too (`theta X(-0.001, -5, -0.0009)`), and a θ at its
/// outward bound still gets probes on the inward side (#2159 review).
fn identity_probe_points(t: f64, lower: f64, upper: f64) -> Option<[f64; 3]> {
    let full = IDENTITY_PROBE_FRACTION * t.abs();
    let (down, up) = (t - lower, upper - t);
    if !(down >= 0.0 && up >= 0.0) {
        return None;
    }
    let central = full.min(0.9 * down).min(0.9 * up);
    let one_sided = full.min(0.9 * down.max(up) / 2.0);
    let points = if central >= one_sided {
        [t - central, t, t + central]
    } else if up >= down {
        [t, t + one_sided, t + 2.0 * one_sided]
    } else {
        [t - 2.0 * one_sided, t - one_sided, t]
    };
    let spacing = points[1] - points[0];
    (spacing >= IDENTITY_PROBE_MIN_FRACTION * t.abs()).then_some(points)
}

/// The parameter-prior and covariate-NN penalties the NLopt objective closure adds
/// to the clean −2LL at `params`. The #2159 probe must see a θ whose small moves a
/// tight prior or the NN penalty makes matter (#2159 review).
fn identity_probe_penalty(
    params: &ModelParameters, nn_reg: &crate::estimation::nn_reg::NnRegularizer,
    priors: &crate::estimation::priors::PriorSet,
) -> f64 {
    nn_reg.penalty_value(&params.theta) + priors.penalty(&pack_params(params))
}

/// The objective the outer optimizer sees for penalized `raw`: `raw`, or the `1e20` wall
/// when the EBE guard ([`ebe_guard_rejects`]) or a non-finite value rejects the point, as
/// the NLopt closure decides it — so the #2159 probe reads no likelihood the optimizer
/// would refuse.
fn identity_probe_objective(
    raw: f64, stats: &InnerLoopStats, n_subj: usize, max_unconverged_frac: f64,
) -> f64 {
    if !raw.is_finite() || ebe_guard_rejects(stats, n_subj, raw, max_unconverged_frac) {
        1e20
    } else {
        raw
    }
}

/// [`identity_theta_scales`] on the K-fold mixture objective (#2159): the flat-θ
/// pre-flight is skipped for a mixture, so the probe evaluates what the optimizer
/// minimises — [`crate::estimation::mixture::mixture_ofv`] plus penalties and the EBE
/// guard — at the packed, clamped start and each probe.
fn mixture_identity_theta_scales(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions,
) -> Vec<(usize, f64)> {
    let PackedStart { packed: mut x, bounds, .. } = pack_with_bounds(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let params = unpack_params(&x, init_params);
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    let priors = build_prior_set(model, init_params);
    let ofv_at = |p: &ModelParameters| {
        let m = crate::estimation::mixture::mixture_ofv(model, population, p, options, None);
        identity_probe_objective(
            m.ofv + identity_probe_penalty(p, &nn_reg, &priors),
            &m.ebe_stats,
            population.subjects.len(),
            options.max_unconverged_frac,
        )
    };
    identity_theta_scales(
        &params.theta,
        &init_params.theta_lower,
        &init_params.theta_upper,
        |i| init_params.theta_fixed.get(i).copied().unwrap_or(false),
        || ofv_at(&params),
        // The R-matrix recorders do not differentiate the K-fold mixture objective.
        |_| None,
        // Only the exact rule asks; the probes judge feasibility themselves.
        |_, _| true,
        |i, value| {
            let mut p = params.clone();
            p.theta[i] = value;
            ofv_at(&p)
        },
    )
}

/// Evaluate the population objective at the initial parameters without running the outer
/// optimizer (`outer_maxiter == 0`, NONMEM `MAXEVAL=0`): one cold inner solve per subject,
/// OFV `2 · pop_nll`, and the covariance step if requested. `converged = false`,
/// `n_iterations = 0`, and no "did not converge" warning. Every optimizer routes here,
/// so `maxiter = 0` is deterministic.
fn evaluate_at_initial_params(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let PackedStart { packed: mut x, bounds, .. } = pack_with_bounds(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let params = unpack_params(&x, init_params);

    // Mixture (#977 Phase 3): the eval-only OFV is the K-fold log-sum-exp at the
    // initial parameters; EBEs reported are the MIXEST class per subject.
    let is_mixture = params.mixture.is_some();
    // #2013: the IDs of the subjects whose EBE the Sobol screen moved, as on the
    // estimating path's final re-solve. Feeds the warning only.
    let mut screen_moved_ids: Vec<String> = Vec::new();
    let (eta_hats, h_matrices, kappas, ofv, mixture_posteriors) = if is_mixture {
        let m = crate::estimation::mixture::mixture_ofv(model, population, &params, options, None);
        // Carry PMIX/MIXEST like the converged path so an eval-only mixture run
        // still emits the PMIX_*/MIXEST sdtab columns (#977), and the MIXEST
        // class's κ̂ as the converged path does since #985 (#1932). Empty for a
        // non-IOV mixture.
        let posteriors = MixturePosteriors {
            pmix: m.pmix,
            mixest: m.mixest,
            n_fd_h_redone: m.n_fd_h_redone,
            class_nll: m.class_nll,
        };
        (m.mixest_etas, m.mixest_h_mats, m.mixest_kappas, m.ofv, Some(posteriors))
    } else {
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        // Genuine cold start (`None` ≡ zeros, #2014), so the guarded inner multi-start runs as on
        // an estimating fit's first evaluation (#891). The final re-solve every outer path
        // shares (#2097), with the #2013 screen-moved IDs.
        let FinalInnerSolve {
            etas: eta_hats,
            h_mats: h_matrices,
            kappas,
            screen_moved_ids: moved_ids,
        } = final_inner_solve(model, population, &params, &mu_k, None, options);
        screen_moved_ids = moved_ids;
        let ofv = 2.0
            * pop_nll_opts(model, population, &params, &eta_hats, &h_matrices, &kappas, options);
        (eta_hats, h_matrices, kappas, ofv, None)
    };

    if options.verbose {
        eprintln!("Iter {:>4}: OFV = {:.6}", 0, ofv);
        eprintln!("outer_maxiter = 0: evaluation only, no optimization performed.");
    }

    let mut warnings: Vec<String> =
        ebe_screen_moved_warning(&screen_moved_ids).into_iter().collect();
    // Mixture (#983 Phase 6): the covariance step now builds its FD Hessian on the
    // K-fold mixture OFV (`compute_covariance` branches on `template.mixture`), so
    // it runs for mixtures exactly like the single-population path.
    let newton_decrement;
    let (covariance_matrix, covariance_wall_time_secs, sir_fallback_proposal, covariance_method) = {
        let out = crate::estimation::covariance::run_covariance_step(
            &x,
            init_params,
            model,
            population,
            &eta_hats,
            &h_matrices,
            &kappas,
            options,
            options.verbose.then_some("Computing covariance matrix..."),
        );
        let crate::estimation::covariance::CovStepOutcome {
            matrix,
            wall_time_secs,
            warnings: cov_warnings,
            sir_fallback_proposal,
            method: covariance_method,
            newton_decrement: decrement,
            non_stationary_optimum: _,
        } = out;
        warnings.extend(cov_warnings);
        newton_decrement = decrement;
        (matrix, wall_time_secs, sir_fallback_proposal, covariance_method)
    };

    OuterResult {
        // Eval-only: carry the packed `x` the inline covariance step centred on, so a later
        // `run_covariance` reproduces it bit-for-bit (#816 follow-up; `chol(L·Lᵀ) ≠ L`).
        packed_estimate: Some(x.clone()),
        mixture_posteriors,
        ofv,
        eta_hats,
        h_matrices,
        kappas,
        covariance_matrix,
        covariance_method,
        covariance_wall_time_secs,
        warnings,
        sir_fallback_proposal,
        newton_decrement,
        ..OuterResult::from_params(params)
    }
}

/// Warm-started variant: starts from given EBEs and H-matrices instead of zeros.
/// Used by the Gauss-Newton hybrid to polish from the GN result.
pub fn optimize_population_warm(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, warm_etas: &[DVector<f64>], warm_h_mats: &[DMatrix<f64>],
) -> OuterResult {
    // For now, delegate to the standard path — the inner loop warm-starts
    // from the provided EBEs automatically via the NloptState initialization.
    // TODO: pass warm_etas into the NLopt state directly for tighter coupling.
    let _ = (warm_etas, warm_h_mats);
    optimize_population(model, population, init_params, options)
}

// ═══════════════════════════════════════════════════════════════════════════
//  NLopt-based outer optimizer (matches Julia's NLopt path exactly)
// ═══════════════════════════════════════════════════════════════════════════

/// Identity-Hessian first-step overshoot guard for the scaled gradient.
///
/// SLSQP and L-BFGS start with `H = I`, so the first step is `−∇f` projected onto the box;
/// with |∇f|∞ ≈ 10²–10³ (analytic FOCE gradients) it pins every component to a corner and
/// the fit stays at init (#55 SLSQP, #960 L-BFGS). This rescales `g` in place by one
/// scalar so no component of the step exceeds `clamp(half_width, 0.1, 1.0)` in scaled
/// space (the clamp keeps it effective on fixed / very narrow and on very wide bounds).
/// Uniform, so the direction is unchanged. Returns whether the cap fired.
///
/// Applied on every SLSQP eval but only the first L-BFGS eval ([`should_cap_gradient`]):
/// capping later corrupts L-BFGS's `(s, y)` curvature (#960). MMA and BOBYQA are never capped.
pub(crate) fn cap_scaled_gradient(g: &mut [f64], lower_s: &[f64], upper_s: &[f64]) -> bool {
    debug_assert_eq!(g.len(), lower_s.len());
    debug_assert_eq!(g.len(), upper_s.len());
    let mut worst_ratio = 0.0_f64;
    for i in 0..g.len() {
        let budget = ((upper_s[i] - lower_s[i]).abs() * 0.5).clamp(0.1, 1.0);
        let ratio = g[i].abs() / budget;
        if ratio > worst_ratio {
            worst_ratio = ratio;
        }
    }
    if worst_ratio > 1.0 {
        for gi in g.iter_mut() {
            *gi /= worst_ratio;
        }
        true
    } else {
        false
    }
}

/// Whether the overshoot cap ([`cap_scaled_gradient`]) fires on this gradient eval.
/// `n_grad_evals` counts this eval too, so the first is `1`.
///
/// - **SLSQP** — every eval (its QP stores no curvature from gradients, #55).
/// - **L-BFGS** — the first eval, and later ones only on the stall-retry pass while
///   still at the initial estimates (`hold_cap_at_init`; see [`optimize_nlopt`]).
///   Holding it from the start corrupts the `(s, y)` curvature: ~11 OFV on
///   `scaling_convergence`, ~4 on the 2-cpt transit fit (#960).
/// - **MMA / BOBYQA** — never.
pub(crate) fn should_cap_gradient(
    algo: nlopt::Algorithm, n_grad_evals: usize, hold_cap_at_init: bool,
) -> bool {
    match algo {
        nlopt::Algorithm::Slsqp => true,
        nlopt::Algorithm::Lbfgs => n_grad_evals == 1 || hold_cap_at_init,
        _ => false,
    }
}

/// Scaled-space displacement from the initial estimates below which a point still counts
/// as "at init" (~1% on O(1) scaled coordinates). Gates: the [`optimize_nlopt`] stall
/// retry, [`should_cap_gradient`]'s held cap, and [`failure_is_converged_plateau`]'s
/// refusal to call a bare `Failure` converged at init (#751: a stalled fit moved 1.4e-4).
/// Well above the FOCE noise floor (~1e-4) and below a real first step (O(0.1)).
pub(crate) const INIT_ESCAPE_STEP_S: f64 = 1e-2;

/// L∞ distance between two scaled vectors, or `NaN` for mismatched lengths or any
/// non-finite coordinate. Callers compare against [`INIT_ESCAPE_STEP_S`], and `NaN`
/// fails both comparisons — the conservative reading for
/// [`failure_is_converged_plateau`] and [`should_cap_gradient`] alike (an `f64::max`
/// fold would drop `NaN` and report "at init"). Lengths never differ (`debug_assert`).
pub(crate) fn max_scaled_deviation(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() != b.len() {
        return f64::NAN;
    }
    let mut worst = 0.0_f64;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let deviation = (ai - bi).abs();
        if !deviation.is_finite() {
            return f64::NAN;
        }
        if deviation > worst {
            worst = deviation;
        }
    }
    worst
}

/// The population objective the outer loop minimises: [`pop_nll`] (FOCE/FOCEI) or the
/// AGQ marginal when the stage enables quadrature (`agq_nodes()`). Production sites
/// needing "the objective for this fit" use this or [`run_inner_loop_and_nll`], never
/// `pop_nll` directly. Non-AGQ fits forward to `pop_nll` unchanged.
pub(crate) fn pop_nll_opts(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    eta_hats: &[DVector<f64>], h_matrices: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> f64 {
    if let Some(n_nodes) = options.agq_nodes() {
        // AGQ integrates over whatever random effects the subject has: η alone, or the
        // stacked (η, κ₁..κ_K) under IOV — the joint marginal, not the η-only one. The modes
        // are the ones the shared inner loop already converged (`find_ebe_iov` returns the
        // joint mode); AGQ does not re-optimise them, it lays its grid around them.
        // AGQ builds its selected anchor itself; it does not use the FOCE Jacobians
        // in `h_matrices`. See `crate::estimation::agq` for the two anchor definitions.
        return crate::estimation::agq::agq_population_nll(
            model,
            population,
            params,
            eta_hats,
            kappas,
            n_nodes,
            options.hessian_anchor(),
        );
    }
    pop_nll(model, population, params, eta_hats, h_matrices, kappas, options.interaction)
}

/// Dispatch to the IOV-aware or standard population NLL based on model.n_kappa.
/// `kappas` is ignored (may be empty) when `model.n_kappa == 0`.
pub(crate) fn pop_nll(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    eta_hats: &[DVector<f64>], h_matrices: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>],
    interaction: bool,
) -> f64 {
    // Preserve the existing subject-index summation order, including at width 1.
    subject_nlls(model, population, params, eta_hats, h_matrices, kappas, interaction).iter().sum()
}

/// Each subject's term of [`pop_nll`], in subject order. Because the population objective
/// is exactly their sum, a subject's EBEs can be chosen without reference to anyone
/// else's, which is what [`lowest_per_subject`] relies on.
fn subject_nlls(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    eta_hats: &[DVector<f64>], h_matrices: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<f64> {
    let eval = |(i, subject): (usize, &Subject)| {
        subject_nll(
            model,
            subject,
            params,
            &eta_hats[i],
            &h_matrices[i],
            kappas.get(i).map_or(&[], Vec::as_slice),
            interaction,
        )
    };
    if crate::pool::parallelize_cheap_subject_pass(population.subjects.len()) {
        let subjects = &population.subjects;
        crate::par::map(subjects.len(), |i| eval((i, &subjects[i])))
    } else {
        population.subjects.iter().enumerate().map(eval).collect()
    }
}

/// Shared subject dispatch for separate and fused population evaluations, and the
/// per-subject term the profile solve (#1065) differentiates.
pub(crate) fn subject_nll(
    model: &CompiledModel, subject: &Subject, params: &ModelParameters, eta: &DVector<f64>,
    h_matrix: &DMatrix<f64>, kappas: &[DVector<f64>], interaction: bool,
) -> f64 {
    subject_nll_at(model, subject, params, eta, h_matrix, kappas, interaction, None)
}

/// [`subject_nll`] given the predictions at `eta` when the caller holds them
/// (`EbeResult::ipreds`); the IOV marginal predicts its own.
#[allow(clippy::too_many_arguments)]
fn subject_nll_at(
    model: &CompiledModel, subject: &Subject, params: &ModelParameters, eta: &DVector<f64>,
    h_matrix: &DMatrix<f64>, kappas: &[DVector<f64>], interaction: bool, ipreds: Option<&[f64]>,
) -> f64 {
    if model.n_kappa > 0 {
        if let Some(ref iov) = params.omega_iov {
            return foce_subject_nll_iov_cov(
                model,
                subject,
                &params.theta,
                eta,
                h_matrix,
                &params.omega,
                &params.sigma.values,
                // Live `block_sigma` off-diagonals (#847, #979), as the non-IOV call below.
                model.residual_cov_at(params),
                interaction,
                kappas,
                iov,
            );
        }
    }
    foce_subject_nll_cov_at(
        model,
        subject,
        &params.theta,
        eta,
        h_matrix,
        &params.omega,
        &params.sigma.values,
        // Live `block_sigma` off-diagonals (#847): this is the objective the
        // outer optimizer minimises, so it has to be scored at the ρ the
        // optimizer is currently proposing.
        model.residual_cov_at(params),
        interaction,
        ipreds,
    )
}

/// Continue from each EBE directly into its marginal contribution on the same
/// worker. Only the population sum needs a barrier. AGQ keeps its separate
/// quadrature evaluation; it must never receive the FOCE marginal instead.
#[allow(clippy::type_complexity)]
fn run_inner_loop_and_nll(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    prev_etas: Option<&[DVector<f64>]>,
    // What `prev_etas` are ([`PrevEtas`]).
    prev: PrevEtas,
    mu_k: Option<&[f64]>,
    schedules: Option<&[Option<crate::pk::event_driven::EventSchedule>]>,
    cancel: Option<&crate::cancel::CancelFlag>,
) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, InnerLoopStats, Vec<Vec<DVector<f64>>>, f64) {
    let (etas, h_matrices, stats, kappas, nll, _, _) = run_inner_loop_and_nll_prepared(
        model, population, params, options, prev_etas, None, mu_k, None, schedules, prev, cancel,
        None,
    );
    (etas, h_matrices, stats, kappas, nll)
}

fn skip_duplicate_laplace_terminal_capture() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FERX_NO_LAPLACE_GRADIENT_CAPTURE_SKIP").map(|v| v != "1").unwrap_or(true)
    })
}

pub(crate) fn agq_inner_solve_policy(
    model: &CompiledModel, options: &FitOptions, fused_gradient: bool,
) -> InnerSolvePolicy {
    InnerSolvePolicy {
        capture_terminal_hessian: (!skip_duplicate_laplace_terminal_capture() || !fused_gradient)
            && matches!(options.hessian_anchor(), HessianAnchor::Exact),
        ..InnerSolvePolicy::for_stage(options, model)
    }
}

/// Same dispatch as [`run_inner_loop_and_nll`], plus the AGQ gradient-fusion path and an
/// optional caller-hoisted schedule cache
/// ([`crate::estimation::inner_optimizer::build_schedule_cache`]; `None` rebuilds per eval).
/// The last element is the per-subject `nllᵢ` (in `population.subjects` order) the #1520
/// salvage guard compares on; empty on the AGQ branch, which disarms the guard.
#[allow(clippy::type_complexity)]
fn run_inner_loop_and_nll_prepared(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    options: &FitOptions,
    prev_etas: Option<&[DVector<f64>]>,
    // Where `prev_etas` were predicted from, when they are the warm-start predictor's
    // first-order starts (#2549): the modes, which a subject starts from where they score
    // lower or its solve fails from the prediction, and the incumbent's exact inner
    // Hessians, their seeds ([`run_inner_loop_predicted_map_cached`]).
    predicted_from: Option<crate::estimation::inner_optimizer::PredictedFromAll<'_>>,
    mu_k: Option<&[f64]>,
    // The fused AGQ gradient's inputs, and whether it forms the mode response (#2549).
    agq_gradient_inputs: Option<(
        &ModelParameters,
        &[f64],
        &PackedBounds,
        bool,
        Option<&OuterFdDeclineLog>,
    )>,
    schedules: Option<&[Option<crate::pk::event_driven::EventSchedule>]>,
    // What `prev_etas` are ([`PrevEtas`]): converged modes at `params` only on the first
    // evaluation seeded from the flat-theta pre-flight (#2076), an earlier pass's modes on
    // the evaluations after it.
    prev: PrevEtas,
    // Polled per subject (#574). Only a caller whose guard keeps a hard-rejected pass
    // out of every accepted state passes `Some`: the NLopt objective and the global
    // pre-search. The built-in BFGS loop passes `None` and polls between iterations.
    cancel: Option<&crate::cancel::CancelFlag>,
    // Stops the pass once the EBE guard is certain to reject it (#2709). Honoured on the
    // schedule-cached and predicted passes, which are the ones the NLopt objective runs;
    // the uncached arm solves every subject. A caller passing `Some` must read
    // `GuardStop::stopped` and reject a stopped evaluation: its sum is over placeholders.
    guard_stop: Option<&GuardStop>,
) -> (
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    InnerLoopStats,
    Vec<Vec<DVector<f64>>>,
    f64,
    Option<crate::estimation::agq::PopulationEvaluation>,
    Vec<f64>,
) {
    if options.agq_nodes().is_some() {
        // Capture each subject's terminal exact Hessian only where the objective below will
        // actually read it: the objective-only evaluation under the exact anchor (#1389).
        // The Gauss-Newton anchor never reuses it, and the gradient path recomputes the
        // whole second-order jet for its Laplace derivative sweep anyway, so capturing there
        // would be a full provider pass per subject thrown away.
        let policy = agq_inner_solve_policy(model, options, agq_gradient_inputs.is_some())
            .with_prev_etas(prev);
        let take_terminal_work =
            |_: &Subject, ebe: &crate::estimation::inner_optimizer::EbeResult| {
                (ebe.terminal_hessian.clone(), ebe.nll)
            };
        let (etas, h_matrices, stats, kappas, terminal_work) =
            match (schedules, prev_etas.zip(predicted_from)) {
                // Predicted starts, each with its mode to fall back to (#2549).
                (_, Some((start, from))) => {
                    crate::estimation::inner_optimizer::run_inner_loop_predicted_map_cached(
                        model,
                        population,
                        params,
                        options.inner_maxiter,
                        options.inner_tol,
                        start,
                        from,
                        mu_k,
                        options.min_obs_for_convergence_check as usize,
                        options.inner_restarts,
                        schedules,
                        policy,
                        cancel,
                        guard_stop,
                        take_terminal_work,
                    )
                }
                (Some(cache), None) => {
                    crate::estimation::inner_optimizer::run_inner_loop_warm_map_cached(
                        model,
                        population,
                        params,
                        options.inner_maxiter,
                        options.inner_tol,
                        prev_etas,
                        mu_k,
                        options.min_obs_for_convergence_check as usize,
                        options.inner_restarts,
                        cache,
                        policy,
                        cancel,
                        guard_stop,
                        take_terminal_work,
                    )
                }
                (None, None) => crate::estimation::inner_optimizer::run_inner_loop_warm_map(
                    model,
                    population,
                    params,
                    options.inner_maxiter,
                    options.inner_tol,
                    prev_etas,
                    mu_k,
                    options.min_obs_for_convergence_check as usize,
                    options.inner_restarts,
                    policy,
                    cancel,
                    take_terminal_work,
                ),
            };
        // A cancel observed during the pass, or a stop on a certain guard rejection
        // (#2709), left placeholders, not modes: do not run the quadrature on them (#574).
        if cancel.is_some_and(crate::cancel::CancelFlag::is_cancelled)
            || guard_stop.is_some_and(GuardStop::stopped)
        {
            return (etas, h_matrices, stats, kappas, f64::INFINITY, None, Vec::new());
        }
        let n_nodes = options.agq_nodes().expect("AGQ branch");
        let evaluation =
            agq_gradient_inputs.map(
                |(template, x, bounds, want_eta_dx, declines)| match schedules {
                    Some(cache) => crate::estimation::agq::agq_population_evaluate_with_schedules(
                        model,
                        population,
                        params,
                        template,
                        x,
                        &etas,
                        &kappas,
                        bounds,
                        options,
                        n_nodes,
                        options.hessian_anchor(),
                        cache,
                        &terminal_work,
                        want_eta_dx,
                        declines,
                    ),
                    None => crate::estimation::agq::agq_population_evaluate(
                        model,
                        population,
                        params,
                        template,
                        x,
                        &etas,
                        &kappas,
                        bounds,
                        options,
                        n_nodes,
                        options.hessian_anchor(),
                        want_eta_dx,
                        declines,
                    ),
                },
            );
        let nll = evaluation.as_ref().map_or_else(
            || {
                crate::estimation::agq::agq_population_nll_prepared(
                    model,
                    population,
                    params,
                    &etas,
                    &kappas,
                    n_nodes,
                    options.hessian_anchor(),
                    schedules,
                    Some(&terminal_work),
                )
            },
            |evaluation| evaluation.nll,
        );
        return (etas, h_matrices, stats, kappas, nll, evaluation, Vec::new());
    }
    let policy = InnerSolvePolicy {
        accelerate_exact_outer: true,
        ..InnerSolvePolicy::for_stage(options, model)
    }
    .with_prev_etas(prev);
    let finish_subject =
        |subject: &Subject, ebe: &crate::estimation::inner_optimizer::EbeResult| {
            subject_nll_at(
                model,
                subject,
                params,
                &ebe.eta,
                &ebe.h_matrix,
                &ebe.kappas,
                options.interaction,
                ebe.ipreds.as_deref(),
            )
        };
    let (etas, h_matrices, stats, kappas, contributions) =
        if let Some((start, from)) = prev_etas.zip(predicted_from) {
            crate::estimation::inner_optimizer::run_inner_loop_predicted_map_cached(
                model,
                population,
                params,
                options.inner_maxiter,
                options.inner_tol,
                start,
                from,
                mu_k,
                options.min_obs_for_convergence_check as usize,
                options.inner_restarts,
                schedules,
                policy,
                cancel,
                guard_stop,
                finish_subject,
            )
        } else if let Some(cache) = schedules {
            crate::estimation::inner_optimizer::run_inner_loop_warm_map_cached(
                model,
                population,
                params,
                options.inner_maxiter,
                options.inner_tol,
                prev_etas,
                mu_k,
                options.min_obs_for_convergence_check as usize,
                options.inner_restarts,
                cache,
                policy,
                cancel,
                guard_stop,
                finish_subject,
            )
        } else {
            run_inner_loop_warm_map(
                model,
                population,
                params,
                options.inner_maxiter,
                options.inner_tol,
                prev_etas,
                mu_k,
                options.min_obs_for_convergence_check as usize,
                options.inner_restarts,
                policy,
                cancel,
                finish_subject,
            )
        };
    // A cancelled pass sums placeholder contributions (#574); never report it as an OFV.
    // A pass stopped on a certain guard rejection (#2709) sums them too, but its caller
    // reads `GuardStop::stopped` and rejects the evaluation, so that sum is never an OFV.
    let nll = if cancel.is_some_and(crate::cancel::CancelFlag::is_cancelled) {
        f64::INFINITY
    } else {
        contributions.iter().sum()
    };
    (etas, h_matrices, stats, kappas, nll, None, contributions)
}

/// State passed through NLopt's user-data mechanism
struct NloptState {
    cached_etas: Vec<DVector<f64>>,
    /// What `cached_etas` are: `ModesHere` while the pre-flight's converged modes are
    /// unconsumed (taken as warm starts whatever their values, #2076); `Solved` once an
    /// adopted pass wrote them (so an exact-zero EBE is not re-multi-started); `Placeholder`
    /// for a cold start's zero fill.
    cached_etas_kind: PrevEtas,
    /// The incumbent's mode response `dη̂/dx` and the point it was taken at, for the
    /// Almquist Eq. 48 warm start `η̂ + (dη̂/dx)(x − x₀)`, taken whole (#2549,
    /// [`crate::estimation::sens_outer_gradient::predict_warm_etas`]).
    pred_jac: Option<Vec<Vec<DVector<f64>>>>,
    pred_anchor: Vec<f64>,
    /// The incumbent's exact inner Hessians at its modes, from the gradient pass that formed
    /// `pred_jac` (#2549): the next warm solves' BFGS seeds where the policy seeds with the
    /// exact inner Hessian (Laplace), in place of a fresh one at each start.
    seed_hessians: Option<Vec<Option<DMatrix<f64>>>>,
    cached_h_mats: Vec<DMatrix<f64>>,
    /// Mixture per-class EBE warm-start cache `[class][subject]` (#977 Phase 3).
    /// Empty for non-mixture models.
    cached_etas_by_class: Vec<Vec<DVector<f64>>>,
    best_ofv: f64,
    n_evals: usize,
    /// Count of gradient evaluations so far. Distinct from `n_evals` (which
    /// also counts objective-only line-search probes); drives the
    /// `reconverge_gradient_interval` schedule.
    n_grad_evals: usize,
    /// Previous parameter vector — used to compute step_norm for the trace.
    prev_x: Vec<f64>,
    last_improvement_eval: usize,
    best_at_last_improvement: f64,
    /// Scaled-space step norms `‖xs − prev_x‖` of the last
    /// [`EXPANSION_HISTORY`] evals, oldest first. Feeds [`step_is_expanding`] (#1530).
    recent_steps: std::collections::VecDeque<f64>,
    /// Sticky once latched — subsequent evals return `best_ofv` with zero
    /// gradient so SLSQP/L-BFGS xtol/ftol fires in microseconds instead
    /// of grinding through `maxeval` at full inner-loop cost.
    stagnation_stopped: bool,
    /// `best_ofv` when the #1934 decrement last vetoed a latch. A window that expires
    /// with the incumbent still at that objective latches without asking again: the
    /// same point gives the same decrement, so re-asking would re-arm forever.
    vetoed_at_best_ofv: Option<f64>,
    /// The latch overrode a decrement that still predicted a gain (the case above).
    latch_overrode_veto: bool,
    /// The #1520 salvage guard's bookkeeping: which point a trial is measured against is
    /// decided per optimizer by [`SalvageGuardPolicy`], and the state remembers the
    /// candidates (best evaluation, last gradient point, previous `xs`).
    salvage_guard: SalvageGuardState,
}

impl NloptState {
    /// A cold EBE seed for the next evaluation: every subject at zero, and no mode response
    /// to predict from. The response belongs to the modes it was taken at, so a start
    /// that drops those modes drops it too; `0 + (dη̂/dx)·Δx` is not a prediction of anything
    /// and, being non-zero, would be taken for a warm start (#2549 review r2 #1).
    fn reset_to_cold_start(&mut self) {
        for e in self.cached_etas.iter_mut() {
            e.fill(0.0);
        }
        self.cached_etas_kind = PrevEtas::Placeholder;
        self.pred_jac = None;
        self.pred_anchor.clear();
        self.seed_hessians = None;
    }

    /// [`Self::reset_to_cold_start`], then `modes` as the next evaluation's warm starts when
    /// there is one per subject: a cold incumbent's or a verification leg's held modes. They
    /// are a solve's output, so the cache is [`PrevEtas::Solved`] and an exactly-zero mode
    /// among them is warm like the rest (#2688). With `None`, or a count that does not match,
    /// the start stays cold.
    fn restart_from_modes(&mut self, modes: Option<Vec<DVector<f64>>>) {
        self.reset_to_cold_start();
        if let Some(modes) = modes.filter(|m| m.len() == self.cached_etas.len()) {
            self.cached_etas = modes;
            self.cached_etas_kind = PrevEtas::Solved;
        }
    }
}

/// Whether the EBE convergence guard rejects this outer trial:
///
/// 1. **Hard reject** — any subject rejected at its inner start; its `(η, H)` is a
///    placeholder, so reject regardless of `max_unconverged_frac` / `min_obs` (#603).
/// 2. **Too many unconverged** — the fraction of (long-enough-record) subjects whose inner
///    optimizer failed exceeds `max_unconverged_frac` and the OFV is finite.
///
/// A negative `max_unconverged_frac` disables only the fraction trigger. One predicate for
/// all evaluation sites (#603).
pub(crate) fn ebe_guard_rejects(
    stats: &InnerLoopStats, n_subj: usize, raw_ofv: f64, max_unconverged_frac: f64,
) -> bool {
    if stats.n_start_rejected > 0 {
        return true;
    }
    let frac = stats.n_unconverged as f64 / (n_subj as f64).max(1.0);
    raw_ofv.is_finite() && frac > max_unconverged_frac && max_unconverged_frac >= 0.0
}

/// Objective value for a guard-rejected outer step, **consistent** with the center-push
/// gradient `g[i] = 100·(xs[i] − c[i])` (`c` = scaled bound midpoint) returned alongside it.
///
/// NLopt's gradient line search (More-Thuente) reconciles `f` and `∇f`: it interpolates on
/// both, so the returned objective must integrate the returned gradient. The historical
/// pairing — flat `f = 1e20` with a non-zero center-push gradient — violates that
/// (`∇(const) = 0 ≠ center-push`), and on a stiff objective whose **first** optimizer step
/// overshoots straight into the EBE guard (ODE + `iiv_on_ruv`: a large step diverges the
/// inner EBEs and overflows the `exp(2·η_ruv)` marginal) the line search fails on iteration
/// one, before any curvature is built. Returning the quadratic bowl `BASE + 50·Σ(xs − c)²`
/// — whose gradient is exactly the `100·(xs − c)` center-push — lets the line search
/// backtrack to a feasible step. `BASE` is a wall far above any feasible OFV yet low enough
/// that the quadratic term stays f64-resolvable (the old `1e20` swamped it). Only the
/// gradient optimizers need this; derivative-free BOBYQA keeps the flat `1e20` wall.
/// Whether an outer run's **first** evaluation, rejected by the EBE guard, is solved again
/// warm from the EBEs it just found before it is judged (#2682).
///
/// The guard rejects a *step*: too many unconverged inner loops make the objective there
/// untrustworthy, so the optimizer backs away to its incumbent. At the start there is no
/// incumbent, and [`adopt_warm_start`] keeps the warm-start cache at zero until there is
/// one (#1290) — so a start whose cold inner solve leaves more than `max_unconverged_frac`
/// of the subjects unconverged is re-solved cold at every later trial too, each is
/// rejected the same way, and the optimizer walks the centre-push gradient toward the box
/// midpoint. Reported on an 8-state ODE with 5,112 subjects: eval 1 printed the guard
/// penalty `1000000050451.003662`, no later evaluation was feasible, and the run hit a 2 h
/// cap; at `max_unconverged_frac = 1.0` the same start scored −5732.82 and the fit moved
/// on at once. A second inner pass from the first one's EBEs gives the unconverged
/// subjects the budget a warm start would have, at the one point where no warm start
/// exists.
///
/// The profiled loop (`profile::optimize_profiled`) applies the same rule to its first
/// profile solve, re-solving from `ProfileWarm::from_solve`.
///
/// Only the fraction trigger qualifies. A hard start rejection holds a placeholder
/// `(η, H)` that is no warm start, and a non-finite objective is not something more inner
/// iterations repair.
pub(crate) fn wants_start_warm_rescore(
    stats: &InnerLoopStats, n_subj: usize, raw_ofv: f64, max_unconverged_frac: f64,
) -> bool {
    stats.n_start_rejected == 0 && ebe_guard_rejects(stats, n_subj, raw_ofv, max_unconverged_frac)
}

/// One evaluation's inner solve as the objective closure consumes it: EBEs, inner
/// Hessians, inner-loop statistics, κ̂, the objective, and AGQ's population evaluation.
type StartSolve = (
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    InnerLoopStats,
    Vec<Vec<DVector<f64>>>,
    f64,
    Option<crate::estimation::agq::PopulationEvaluation>,
);

/// Whether [`wants_start_warm_rescore`]'s re-solve replaces the cold solve: the guard
/// accepts it **and** its objective is finite. The guard alone answers `false` for a
/// non-finite objective, so without the second test a `NaN` re-solve would be published
/// as an unguarded start (#2682).
pub(crate) fn warm_rescore_accepted(
    stats: &InnerLoopStats, n_subj: usize, raw_ofv: f64, max_unconverged_frac: f64,
) -> bool {
    raw_ofv.is_finite() && !ebe_guard_rejects(stats, n_subj, raw_ofv, max_unconverged_frac)
}

/// The verbose line for an evaluation the optimizer was handed a penalty for instead of
/// its objective (#2682): which trigger rejected it, with the count behind it.
fn guarded_eval_line(
    stats: &InnerLoopStats, n_subj: usize, raw_ofv: f64, max_unconverged_frac: f64,
) -> String {
    if !raw_ofv.is_finite() {
        format!("rejected, the objective is not finite ({raw_ofv})")
    } else if stats.n_start_rejected > 0 {
        format!(
            "rejected by the EBE convergence guard, {} of {} subjects rejected at their inner \
             start",
            stats.n_start_rejected, n_subj,
        )
    } else {
        format!(
            "rejected by the EBE convergence guard, {} of {} subjects' inner loops did not \
             converge, against `max_unconverged_frac` = {max_unconverged_frac} (OFV there \
             {raw_ofv:.6})",
            stats.n_unconverged, n_subj,
        )
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch: `true` runs every pass in full, the behaviour before #2709, so a test can
    /// compare a fit with and without the stop. Read on the thread the objective runs on.
    pub(crate) static GUARD_STOP_DISABLED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test counter: evaluations whose inner pass the #2709 stop cut short.
    pub(crate) static GUARD_STOPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether the #2709 stop is switched off: only ever by a test ([`GUARD_STOP_DISABLED`]).
#[cfg(test)]
fn guard_stop_disabled() -> bool {
    GUARD_STOP_DISABLED.with(std::cell::Cell::get)
}

/// Whether the #2709 stop is switched off: never outside a test.
#[cfg(not(test))]
fn guard_stop_disabled() -> bool {
    false
}

/// The statistics an evaluation is judged and reported on, and how many subjects its pass
/// solved when it stopped on a certain guard rejection (#2709). A stopped pass reports the
/// counts of the subjects it solved ([`GuardStop::observed_stats`]): its end-of-pass `pass`
/// statistics also count every unsolved placeholder, as a hard rejection and as unconverged.
fn stopped_pass_stats(
    guard_stop: Option<&GuardStop>, pass: InnerLoopStats,
) -> (InnerLoopStats, Option<usize>) {
    match guard_stop.filter(|g| g.stopped()) {
        Some(g) => (g.observed_stats(), Some(g.solved())),
        None => (pass, None),
    }
}

/// The verbose line for an evaluation whose inner pass stopped once the EBE guard's
/// rejection was certain (#2709): the trigger, the counts it stopped on, and how many
/// subjects it left unsolved. The counts are of the `solved` subjects only.
fn stopped_eval_line(
    stats: &InnerLoopStats, solved: usize, n_subj: usize, max_unconverged_frac: f64,
) -> String {
    let trigger = if stats.n_start_rejected > 0 {
        format!("{} subjects rejected at their inner start", stats.n_start_rejected)
    } else {
        format!(
            "{} subjects' inner loops did not converge, against `max_unconverged_frac` = \
             {max_unconverged_frac}",
            stats.n_unconverged
        )
    };
    format!(
        "rejected by the EBE convergence guard after {solved} of {n_subj} subjects: {trigger}; \
         the other {} were not solved",
        n_subj.saturating_sub(solved),
    )
}

/// A first evaluation's cold inner solve, held while [`wants_start_warm_rescore`]'s warm
/// re-solve runs, with what that re-solve overwrites.
struct ColdStart {
    solved: StartSolve,
    cached_etas: Vec<DVector<f64>>,
    /// What `cached_etas` was read as, restored with it. The re-solve reads its warm start
    /// as [`PrevEtas::Solved`]: the cold solve's modes, an exactly-zero η̂ among them (#2688).
    cached_etas_kind: PrevEtas,
    contribs: Vec<f64>,
    mixeval: Option<crate::estimation::mixture::MixtureEval>,
    mixmodes: Option<crate::estimation::mixture::MixtureClassModes>,
}

/// The floor of [`guard_penalty_value`]: every value it produces is `>= GUARD_PENALTY_BASE`.
/// `api::fit`'s resume check reads it (#1617): a checkpoint objective at or above it came
/// from a guard-rejected point, violating [`crate::io::checkpoint::Checkpoint::ofv`]'s
/// contract. Five orders above real −2LL (~10⁶–10⁷) and below the `1e20` wall.
///
/// `guard_penalty_value` returns the bowl `BASE + 50·Σ(xs − c)²` (`c` the scaled bound
/// midpoint), which integrates the center-push gradient `100·(xs − c)` returned with it:
/// NLopt's line search interpolates on `f` and `∇f`, and a flat `1e20` with a non-zero
/// gradient failed it on iteration one on stiff fits (ODE + `iiv_on_ruv`). BOBYQA keeps
/// the flat `1e20` wall.
pub(crate) const GUARD_PENALTY_BASE: f64 = 1e12;

fn guard_penalty_value(xs: &[f64], lower_s: &[f64], upper_s: &[f64]) -> f64 {
    const BASE: f64 = GUARD_PENALTY_BASE;
    let pen: f64 = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let c = (lower_s[i] + upper_s[i]) / 2.0;
            let d = x - c;
            d * d
        })
        .sum();
    BASE + 50.0 * pen
}

/// Latches `stagnation_stopped` once recent evals show no OFV progress (SLSQP on
/// poorly-identified, e.g. γ-bearing, FOCEI problems can otherwise spend 30+ min at a
/// numerically-flat OFV before its xtol/ftol criteria fire).
///
/// `enabled = false` disables the guard entirely: never latches and never reports
/// stagnation, so the optimizer runs to its own termination criterion (or `outer_maxiter`).
fn detect_stagnation(
    state: &mut NloptState, n: usize, enabled: bool, short_window: bool,
    still_descending: impl FnOnce() -> bool,
) -> bool {
    if !enabled {
        return false;
    }
    if state.stagnation_stopped {
        return true;
    }
    // Absolute OFV improvement below this is treated as noise. Matches
    // typical FOCE EBE-loop precision (~1e-3 OFV units) — see
    // `inner_tol` default and Sheiner–Beal linearisation comment in
    // [types.rs:959].
    const STAGNATION_THRESHOLD: f64 = 1e-3;

    let improved = (state.best_at_last_improvement - state.best_ofv) > STAGNATION_THRESHOLD;
    if improved {
        state.last_improvement_eval = state.n_evals;
        state.best_at_last_improvement = state.best_ofv;
        false
    } else if state.n_evals.saturating_sub(state.last_improvement_eval)
        >= stagnation_window(n, short_window)
    {
        // #1934: a window without a 1e-3 improvement may be a slow creep down a valley. If a
        // Newton step from the incumbent still predicts gain, re-arm the window (keeping
        // `best_at_last_improvement`); if the incumbent has not moved since the last veto, latch.
        if state.vetoed_at_best_ofv == Some(state.best_ofv) {
            state.latch_overrode_veto = true;
        } else if still_descending() {
            state.last_improvement_eval = state.n_evals;
            state.vetoed_at_best_ofv = Some(state.best_ofv);
            return false;
        }
        state.stagnation_stopped = true;
        true
    } else {
        false
    }
}

/// The dimension [`stagnation_window`] is sized in: the packed coordinates the optimizer can
/// move, i.e. those `fixed` (the `PackedStart::fixed` mask) does not hold at `lower ==
/// upper`. A `FIX` parameter or a structural-zero Ω entry is no search direction, so a window
/// counted over every packed coordinate ran `n_held` evaluations longer than the same model
/// written without them (#2597).
fn stagnation_dimension(fixed: &[bool]) -> usize {
    fixed.iter().filter(|&&held| !held).count()
}

/// Evals without a significant OFV improvement after which the stagnation
/// guard latches.
///
/// **Long window, `max(3·(n+1), 50)`.** Sized for the FD-gradient era: three
/// attempted descent steps with their gradient probes, and long enough that a
/// line search at the start of a fit gets a real chance. It is what BOBYQA gets
/// (its interpolation-model rebuilds legitimately spend many evals flat, and it has
/// its own reachable `xtol`/`ftol` stops), and what a gradient fit gets before it
/// has made any progress (#751's init-stall retry keys off the fit's position, not
/// this guard, but there is no reason to cut such a fit short either).
///
/// **Short window, `max(n+1, 10)`, for a gradient fit that has already
/// descended (#1530).** A gradient optimizer's callback computes its gradient
/// inside the same call (analytic, or FD without extra NLopt evals), so every
/// callback is an iterate or a line-search probe, and `n+1` of them after real
/// descent with none buying 1e-3 is a run of line searches that have stopped
/// paying. The FOCE/FOCEI gradient stops are an unreachable `1e-12`, so the
/// gradient test is the only way out other than this guard — and it cannot fire
/// when the outer gradient does not vanish at the optimum of the objective being
/// evaluated. On `clofarabine_brooks` (FOCEI, `nlopt_lbfgs`) it plateaus at a norm
/// of ≈1.09 while the OFV is flat to the sixth decimal (likely the fixed-EBE
/// gradient's missing inner-response term, #1529). That fit reached its final OFV
/// within 0.0014 at eval 30 and then spent evals 31–61 (45% of estimation) until
/// the line search failed.
///
/// The short window applies when [`use_short_stagnation_window`] holds and
/// [`step_is_expanding`] does not.
fn stagnation_window(n: usize, short_window: bool) -> usize {
    if short_window {
        (n + 1).max(10)
    } else {
        (3 * (n + 1)).max(50)
    }
}

/// Does a gradient-based NLopt run get reachable `xtol`/`ftol` stops? Only the quadrature
/// objectives (Laplace, FOCEI `n_agq > 1`) — a cost/accuracy choice (measured at the stop
/// setting in `optimize_nlopt_attempt`). FOCE/FOCEI rely on the gradient norm, the
/// stagnation guard and the #2462 verification legs. Shared with
/// [`use_short_stagnation_window`].
fn gradient_run_has_reachable_stops(options: &FitOptions) -> bool {
    options.agq_nodes().is_some()
}

/// Whether [`stagnation_window`]'s short window applies: a gradient optimizer (not BOBYQA)
/// without reachable stops ([`gradient_run_has_reachable_stops`]) whose last significant
/// *feasible* improvement lies past the first feasible eval (so a guard penalty on eval 1
/// cannot pass for descent).
fn use_short_stagnation_window(
    algo: nlopt::Algorithm, reachable_stops: bool, last_sig_feasible_eval: usize,
) -> bool {
    !matches!(algo, nlopt::Algorithm::Bobyqa) && !reachable_stops && last_sig_feasible_eval > 1
}

/// Steps [`step_is_expanding`] looks at: the latest and the two before it.
const EXPANSION_HISTORY: usize = 3;

/// Step-to-step growth factor [`step_is_expanding`] counts as an expansion.
const EXPANSION_RATIO: f64 = 1.5;

/// Is the optimizer's step growing? True when each of the last two steps is at least
/// [`EXPANSION_RATIO`] times the one before — the veto on the short window (#1530).
/// Separates a saddle creep from a fit at its optimum: on `bioavailability` the OFV sits at
/// 648.8203 for a dozen evals while steps grow ×2–4.5, then drops 1.48 into a better basin.
/// Two ratios, since one larger probe in an erratic tail is no trend. Fewer than three
/// steps, zero-then-zero, or a `NaN` step is not an expansion.
fn step_is_expanding(steps: &std::collections::VecDeque<f64>) -> bool {
    if steps.len() < EXPANSION_HISTORY {
        return false;
    }
    let k = steps.len();
    let grows = |a: f64, b: f64| b >= EXPANSION_RATIO * a && b > 0.0;
    grows(steps[k - 3], steps[k - 2]) && grows(steps[k - 2], steps[k - 1])
}

/// The stagnation guard's per-eval step: record the scaled step norm, pick the window
/// ([`use_short_stagnation_window`] vetoed by [`step_is_expanding`]), then
/// [`detect_stagnation`]. Called once per non-latched eval after `best_ofv` and the
/// plateau tracker are updated (also by the trace-replay tests). `still_descending`
/// ([`predicted_remaining_gain`]) is asked when a window elapses; `true` re-arms (#1934).
fn stagnation_after_eval(
    state: &mut NloptState, n: usize, enabled: bool, algo: nlopt::Algorithm, reachable_stops: bool,
    last_sig_feasible_eval: usize, step: f64, still_descending: impl FnOnce() -> bool,
) -> bool {
    if state.recent_steps.len() == EXPANSION_HISTORY {
        state.recent_steps.pop_front();
    }
    state.recent_steps.push_back(step);
    let short_window = use_short_stagnation_window(algo, reachable_stops, last_sig_feasible_eval)
        && !step_is_expanding(&state.recent_steps);
    detect_stagnation(state, n, enabled, short_window, still_descending)
}

/// Does a run the stagnation guard stopped need a bare `Failure`'s plateau verdict? A
/// latch hands NLopt a zero gradient, so its next return (`Success`-class or
/// `MaxEvalReached`) is forced, not found, and gets the same plateau-length plus
/// cold-restart check (a `Failure` already does). Before #1530 latches were reported
/// converged; on fluconazole the latched 738.05 re-solves cold to 741.60.
fn latched_stop_needs_plateau_check(
    converged: bool, max_eval_reached: bool, latched: bool,
) -> bool {
    latched && (converged || max_eval_reached)
}

/// [`latched_stop_needs_plateau_check`] after the polish or verification legs. Under the
/// #2462 verification a spent budget is final: it ended the legs before one could clear
/// the latch, so the stop is unverified, and the plateau check is what the leg replaces.
/// The BOBYQA polish keeps the #1530 rule, where a latch on the last permitted evaluation
/// is still judged on the plateau.
fn stop_needs_plateau_check_after_legs(
    verify_on: bool, polish_end: PolishEnd, converged: bool, max_eval_reached: bool, latched: bool,
) -> bool {
    !(verify_on && polish_end == PolishEnd::BudgetSpent)
        && latched_stop_needs_plateau_check(converged, max_eval_reached, latched)
}

/// Should this evaluation's EBEs become the warm start for the next one? (#1290)
///
/// The EBE surface is multimodal (#864 / #891); adopting every eval's EBEs makes the outer
/// objective path-dependent (a bad probe leaves the cache in a bad basin, so L-BFGS sees
/// its own start move and fails on eval 1). Anchoring to the best point seen makes
/// re-evaluating the incumbent reproduce its objective. A guarded eval never contributes:
/// its `ofv` is the synthetic `guard_penalty_value`.
fn adopt_warm_start(guarded: bool, ofv: f64, best_ofv: f64) -> bool {
    !guarded && ofv < best_ofv
}

/// Replace the EBE store's `η̂` (with κ̂ and `H = ∂f/∂η`) by the mode a polished gradient
/// pass differentiated at (`ModeResponses::polished`, stacked `[η̂, κ̂]`), so the next solve
/// warm-starts from the mode rather than the off-mode `η̂`. Subjects served in place, or
/// whose polished width is not `η` plus the stored κ̂ cells (nested levels), are untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adopt_polished_modes(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters, x: &[f64],
    polished: &[Option<Vec<f64>>], ehs: &mut [DVector<f64>], hms: &mut [DMatrix<f64>],
    kappas: &mut [Vec<DVector<f64>>],
) {
    if polished.iter().all(Option::is_none) || model.has_nested_levels() {
        return;
    }
    let theta = unpack_params(x, init_params).theta;
    let n_eta = model.n_eta;
    for (i, stacked) in polished.iter().enumerate() {
        let Some(stacked) = stacked else { continue };
        let cells: usize = kappas.get(i).map_or(0, |k| k.iter().map(|c| c.len()).sum());
        if i >= ehs.len() || stacked.len() != n_eta + cells {
            continue;
        }
        let Some(jac) = crate::sens::ode_provider::adjoint::exact_eta_jacobian(
            model,
            &population.subjects[i],
            &theta,
            stacked,
        ) else {
            continue;
        };
        if jac.ncols() < n_eta || !jac.iter().all(|v| v.is_finite()) {
            continue;
        }
        hms[i] = jac.columns(0, n_eta).into_owned();
        ehs[i] = DVector::from_column_slice(&stacked[..n_eta]);
        let mut at = n_eta;
        for cell in kappas.get_mut(i).into_iter().flatten() {
            let w = cell.len();
            *cell = DVector::from_column_slice(&stacked[at..at + w]);
            at += w;
        }
    }
}

fn new_nlopt_state(
    n_subj: usize, n_eta: usize, x0: &[f64], salvage_guard: SalvageGuardPolicy,
) -> NloptState {
    NloptState {
        cached_etas: vec![DVector::zeros(n_eta); n_subj],
        cached_etas_kind: PrevEtas::Placeholder,
        pred_jac: None,
        pred_anchor: Vec::new(),
        seed_hessians: None,
        cached_h_mats: Vec::new(),
        cached_etas_by_class: Vec::new(),
        best_ofv: f64::INFINITY,
        n_evals: 0,
        n_grad_evals: 0,
        salvage_guard: SalvageGuardState::new(salvage_guard),
        prev_x: x0.to_vec(),
        last_improvement_eval: 0,
        best_at_last_improvement: f64::INFINITY,
        recent_steps: std::collections::VecDeque::new(),
        stagnation_stopped: false,
        vetoed_at_best_ofv: None,
        latch_overrode_veto: false,
    }
}

/// The options every global pre-search inner solve runs under: the fit's own with the inner
/// multi-start off. Each probe starts from η = 0 (cold, #2014) and would otherwise
/// multi-start: 10× the wall time on `examples/schnider.ferx`. The local optimizer's first
/// evaluation multi-starts at the point it picks.
fn presearch_probe_options(options: &FitOptions) -> FitOptions {
    FitOptions { inner_restarts: 0, ..options.clone() }
}

/// Run NLopt CRS2-LM as a gradient-free global pre-search before the local optimizer.
/// Returns the best point in the caller's scaled coordinates; `Err` when the NLopt build
/// lacks CRS2-LM. CRS2-LM keeps a pool of `population_size` points (default `10*(n+1)`),
/// needs explicit bounds, and is insensitive to the initial point.
fn run_global_presearch(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, scale: &[f64], lower_s: &[f64], upper_s: &[f64], x0: &[f64],
) -> Result<Vec<f64>, String> {
    let n = x0.len();
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    // Built once for the whole pre-search: every probe below re-solves the same
    // subjects' schedules, which depend only on `model` + `population` (never on the
    // trial `params`) — see `run_inner_loop_and_nll_prepared`.
    let schedule_cache =
        crate::estimation::inner_optimizer::build_schedule_cache(model, population);

    // Probe CRS2-LM availability — some NLopt builds (notably the
    // minimal one in the homebrew nlopt-rs crate) ship without it.
    // Catch the panic so we surface a useful warning instead of
    // crashing the fit.
    let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fn dummy(_x: &[f64], _g: Option<&mut [f64]>, _d: &mut ()) -> f64 {
            0.0
        }
        let _opt =
            nlopt::Nlopt::new(nlopt::Algorithm::Crs2Lm, n, dummy, nlopt::Target::Minimize, ());
    }));
    if probe.is_err() {
        return Err("NLopt CRS2-LM not available in this build — install a full \
             NLopt (brew install nlopt / apt install libnlopt-dev) and rebuild"
            .into());
    }

    let probe_options = presearch_probe_options(options);
    let options = &probe_options;

    let n_evals = Arc::new(AtomicUsize::new(0));
    let n_evals_cl = Arc::clone(&n_evals);
    let verbose = options.verbose;

    // Covariate-NN (DCM) regularizer, built once from the observed covariate
    // distribution + NN architecture. A strict no-op when both λ are 0, so the
    // pre-search objective stays byte-identical for unregularized fits.
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    let priors = build_prior_set(model, init_params);

    // Helper: evaluate the FOCE OFV at a single point in scaled space,
    // independent of any NLopt state. Used to compute the user's initial
    // OFV up-front (for the keep-best-of-(user, CRS2-LM) compare below).
    let eval_at_scaled = |xs: &[f64]| -> f64 {
        let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let params = unpack_params(&x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let cached_zero = vec![DVector::zeros(n_eta); n_subj];
        let (_, _, ebe_stats, _, nll) = run_inner_loop_and_nll(
            model,
            population,
            &params,
            options,
            Some(&cached_zero),
            PrevEtas::Placeholder,
            Some(&mu_k),
            Some(&schedule_cache),
            options.cancel.as_ref(),
        );
        // Penalized objective fed to the optimizer (unregularized fits unchanged).
        let raw = 2.0 * nll + nn_reg.penalty_value(&params.theta) + priors.penalty(&x);
        let guarded = ebe_guard_rejects(&ebe_stats, n_subj, raw, options.max_unconverged_frac);
        if !raw.is_finite() || guarded {
            1e20
        } else {
            raw
        }
    };

    let initial_ofv = eval_at_scaled(x0);
    if options.verbose {
        eprintln!(
            "Initial OFV at user-supplied parameters: {:.6} (used as fallback if global \
             pre-search doesn't beat it)",
            initial_ofv,
        );
    }

    // Derivative-free pre-search: no gradient is ever formed, so no salvage guard.
    let pre_state = new_nlopt_state(n_subj, n_eta, x0, SalvageGuardPolicy::Off);

    let pre_objective = |xs: &[f64], _grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
        if crate::cancel::is_cancelled(&options.cancel) {
            return 1e20;
        }
        let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let params = unpack_params(&x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);

        let (_, hms, ebe_stats, _, nll) = run_inner_loop_and_nll(
            model,
            population,
            &params,
            options,
            Some(&state.cached_etas),
            // Always the zero fill below: the pre-search keeps no modes between evals.
            PrevEtas::Placeholder,
            Some(&mu_k),
            Some(&schedule_cache),
            options.cancel.as_ref(),
        );
        // Penalized objective fed to the optimizer (unregularized fits unchanged).
        let raw_ofv = 2.0 * nll + nn_reg.penalty_value(&params.theta) + priors.penalty(&x);

        let ebe_guard =
            ebe_guard_rejects(&ebe_stats, n_subj, raw_ofv, options.max_unconverged_frac);
        let ofv = if ebe_guard {
            1e20
        } else if raw_ofv.is_finite() {
            raw_ofv
        } else {
            1e20
        };

        // CRS2-LM samples globally, so warm-starting EBEs with the
        // best-so-far cached etas is mostly noise-amplifying — keep
        // them at zeros for the next eval. The local optimizer that
        // follows starts from a sensible point and warm-starts cleanly.
        state.cached_etas = vec![DVector::zeros(n_eta); n_subj];
        state.cached_h_mats = hms;
        state.n_evals += 1;
        n_evals_cl.fetch_add(1, Ordering::Relaxed);

        if ofv < state.best_ofv {
            state.best_ofv = ofv;
            if verbose {
                eprintln!("Global pre-search eval {:>4}: OFV = {:.6}", state.n_evals, ofv);
            }
        }

        ofv
    };

    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Crs2Lm,
        n,
        pre_objective,
        nlopt::Target::Minimize,
        pre_state,
    );
    opt.set_lower_bounds(lower_s).map_err(|e| format!("CRS2-LM lower bounds: {:?}", e))?;
    opt.set_upper_bounds(upper_s).map_err(|e| format!("CRS2-LM upper bounds: {:?}", e))?;

    // Default budget: 30 * (n + 1) — modest budget that's enough to
    // probe a few candidate basins without dominating the wall time of
    // the subsequent local refine. Users with hard-to-find optima can
    // bump `global_maxeval` to e.g. 200*(n+1) for a thorough sweep.
    let max_eval = if options.global_maxeval > 0 {
        options.global_maxeval as u32
    } else {
        30 * (n as u32 + 1)
    };
    opt.set_maxeval(max_eval).map_err(|e| format!("CRS2-LM maxeval: {:?}", e))?;

    if options.verbose {
        eprintln!(
            "Starting NLopt CRS2-LM global pre-search ({} parameters, max {} evals)...",
            n, max_eval
        );
    }

    let mut x_pre = x0.to_vec();
    let pre_ofv = match opt.optimize(&mut x_pre) {
        Ok((status, ofv)) => {
            if options.verbose {
                eprintln!(
                    "Global pre-search finished: {:?}, best OFV = {:.6} after {} evals",
                    status,
                    ofv,
                    n_evals.load(Ordering::Relaxed),
                );
            }
            ofv
        }
        Err((fail, ofv)) => {
            if options.verbose {
                eprintln!(
                    "Global pre-search stopped: {:?}, best OFV = {:.6} after {} evals",
                    fail,
                    ofv,
                    n_evals.load(Ordering::Relaxed),
                );
            }
            ofv
        }
    };

    // Keep the better of the user's initials and CRS2-LM's best: CRS2-LM samples freely,
    // so for good inits its best point is often worse.
    if initial_ofv.is_finite() && initial_ofv <= pre_ofv {
        if options.verbose {
            eprintln!(
                "Global pre-search did not beat user-supplied initials \
                 ({:.4} vs {:.4}); keeping user initials for local optimisation.",
                pre_ofv, initial_ofv,
            );
        }
        Ok(x0.to_vec())
    } else {
        Ok(x_pre)
    }
}

/// nlmixr2-style `rescale2` preconditioner: scale each packed param by its bounds
/// half-range `(hi−lo)/2` (value/bounds-based, not curvature-based). Fixed and degenerate
/// ranges fall back to 1.0. Selected by `parameter_scaling = rescale2`
/// ([`ParameterScaling::Rescale2`]).
fn compute_rescale2_scale(bounds: &PackedBounds) -> Vec<f64> {
    (0..bounds.lower.len())
        .map(|k| {
            let hw = (bounds.upper[k] - bounds.lower[k]).abs() * 0.5;
            if hw.is_finite() && hw > 1e-6 {
                hw
            } else {
                1.0
            }
        })
        .collect()
}

/// Resolve [`ParameterScaling::Auto`]: `Abs` (normalise by |packed value|) for the gradient
/// optimizers (`Bfgs`, `Lbfgs`, `NloptLbfgs`, `Slsqp`), `None` otherwise (#335).
///
/// `Abs` presents O(1) coordinates, the right preconditioner for a quasi-Newton / SQP step.
/// The earlier `Rescale2` (bound half-width) was wrong for them: bound width is unrelated
/// to curvature, so it drove L-BFGS into a bound (warfarin FOCEI −286→−243; tvcov to a +166
/// local min) and froze SLSQP's first step (two_cpt_oral_cov −1026, no move). `Abs` recovers
/// warfarin −286, tvcov −188.6 and two_cpt_oral_cov −1165, and keeps SLSQP's warfarin_iov
/// cold-start win (OFV 307.8).
///
/// Derivative-free `Bobyqa` stays unscaled: per-coordinate scaling distorts its trust-region
/// quadratic model and regresses multi-cpt / PD fits (emax_pkpd −36.8→−13.5, three_cpt_iv
/// −730.6→−715.9). `Mma` / `TrustRegion` take the unscaled legacy branch. Non-`Auto` values
/// pass through.
fn resolve_scaling(ps: ParameterScaling, opt: Optimizer) -> ParameterScaling {
    match ps {
        ParameterScaling::Auto => match opt {
            // Gradient optimizers condition best with `Abs` (see the doc comment above).
            Optimizer::Bfgs | Optimizer::Lbfgs | Optimizer::NloptLbfgs | Optimizer::Slsqp => {
                ParameterScaling::Abs
            }
            _ => ParameterScaling::None,
        },
        other => other,
    }
}

/// An identity-packed θ whose initial magnitude exceeds this is divided by `|θ₀|` in the
/// NLopt scale vector under `auto` / `abs` (#2146). Without it warfarin with `TVV_S` = 1e6
/// stops at −265.42 unmoved (vs −286.00). At a bar of 1 `ctmm_pd_2state`'s `LQ01 = -1.2`
/// moves (712.65 → 712.92); at 10 no fixture changes (macOS arm64).
const LARGE_IDENTITY_THETA: f64 = 10.0;

/// The NLopt outer loop's per-coordinate scale vector: the optimizer sees `x[i] / scale[i]`.
///
/// [`resolve_scaling`] picks the base. Under magnitude scaling it is
/// [`compute_scale_packed`], **unless any θ is identity-packed**, when every coordinate
/// keeps 1.0: mixing raw and O(1) magnitudes hurts the quasi-Newton Hessian (SAD_SCEN1
/// 510+ evals vs ~90; re-measured on #2146 as mixed, SLSQP 104 → 178).
///
/// Then, under `auto` / `abs` only, on identity-packed θ (at the front, so `x0[i]` is θᵢ):
/// - |θ₀| > [`LARGE_IDENTITY_THETA`] is divided by |θ₀| (#2146), also for BOBYQA / MMA;
/// - the near-zero θ in `identity_scales` take their probed scale
///   ([`identity_theta_scales`], #2159), or `|x0[i]|` when larger (#2272).
fn nlopt_scale_vector(
    x0: &[f64], init_params: &ModelParameters, bounds: &PackedBounds, options: &FitOptions,
    auto_scale_iov: bool, identity_scales: &[(usize, f64)],
) -> Vec<f64> {
    let n = x0.len();
    let has_identity_theta = init_params.theta_lower.iter().any(|&lo| lo < 0.0);
    let mut scale = match resolve_scaling(options.parameter_scaling, options.optimizer) {
        ParameterScaling::Rescale2 => compute_rescale2_scale(bounds),
        ParameterScaling::Abs => {
            if has_identity_theta {
                vec![1.0; n]
            } else {
                compute_scale_packed(x0, init_params)
            }
        }
        // `Auto` is resolved away by `resolve_scaling`; group with `None`.
        ParameterScaling::None | ParameterScaling::Auto => {
            if (options.scale_params || auto_scale_iov) && !has_identity_theta {
                compute_scale_packed(x0, init_params)
            } else {
                vec![1.0; n]
            }
        }
    };
    if !matches!(options.parameter_scaling, ParameterScaling::Auto | ParameterScaling::Abs) {
        return scale;
    }
    for (i, &lo) in init_params.theta_lower.iter().enumerate() {
        if lo < 0.0 && x0[i].abs() > LARGE_IDENTITY_THETA {
            scale[i] = x0[i].abs();
        }
    }
    // The probe ran at the first leg's start; a restart leg's θ may have grown by orders of
    // magnitude, so the probed scale is a floor under the current magnitude, not a
    // replacement (#2272). On the first leg the two are equal.
    for &(i, s) in identity_scales {
        scale[i] = s.max(x0[i].abs());
    }
    scale
}

/// Resolve the derivative-free `bobyqa` outer optimizer's `ftol_rel` stop tolerance.
///
/// `override_ftol` (`[fit_options] outer_ftol`) wins when set. Otherwise auto-select:
/// `1e-8` for a **pure non-Gaussian** model (TTE or the Phase-4 categorical family,
/// #760) — its data objective is evaluated *exactly*, so the looser historical `1e-6`
/// stopped BOBYQA short of the optimum on the near-flat frailty/random-effect-ω² ridge
/// (#469: a Weibull shape-frailty read 0.204 vs the NONMEM/nlmixr2 0.175 consensus;
/// `1e-8` lands 0.176) — and `1e-6` for everything else. The floor is deliberate: on a
/// **noisy** objective (ODE solver error, or an FD-inner FOCE model such as LTBS) `1e-8`
/// is unreachable, so BOBYQA would grind toward its maxeval budget instead of converging
/// (≈3× the evaluations on an ODE fit). A non-Gaussian endpoint carried on an ODE
/// disposition (`is_ode` true) therefore keeps `1e-6`.
/// OFVs at or above this are treated as diverged/invalid.
///
/// The inner objective clamps a blown-up value to a `~1e20` sentinel, which is
/// *finite* — so `is_finite()` alone does not separate a diverged run from a real
/// one. This cutoff sits well below the sentinel and far above any legitimate
/// population OFV, so a real fit never trips it.
pub(crate) const DIVERGENCE_OFV: f64 = 1e14;

/// Whether an OFV is a real population objective rather than a clamped sentinel or `NaN`.
/// One rule for multi-start ranking (`api::fit::multistart_prefers`) and the convergence
/// verdict.
pub(crate) fn ofv_is_valid(ofv: f64) -> bool {
    ofv.is_finite() && ofv < DIVERGENCE_OFV
}

/// **The one exemption from [`gate_converged_on_objective`] (#1303).** `method = vi` with
/// `vi_final_ofv = none` reports `ofv = NaN` deliberately (the ELBO is not a −2 log L;
/// [`crate::ViFinalOfv::None`]), so the `NaN` declares no objective was published and must
/// not demote every default VI fit. `vi_final_ofv = laplace` is gated as usual — this
/// exempts a setting, not an estimator.
pub(crate) fn publishes_no_objective(method: EstimationMethod, options: &FitOptions) -> bool {
    method == EstimationMethod::Vi && options.vi_final_ofv == crate::types::ViFinalOfv::None
}

/// The `W_` token every non-finite-objective demotion carries, so
/// [`crate::types::classify_warning`] routes it on the token rather than on
/// prose that a later edit could change out from under it (#1303).
pub(crate) const NONFINITE_OBJECTIVE_TOKEN: &str = "W_NONFINITE_OBJECTIVE";

/// Which way the reported objective failed [`ofv_is_valid`], in the user's words.
///
/// The three classes are genuinely different outcomes and the remediation
/// differs, so the message names which one happened rather than saying
/// "non-finite" for all of them:
///
/// * `NaN` — a prediction, a variance, or a likelihood term went `NaN` and
///   poisoned the sum. Nothing was optimized; the parameters reported are
///   wherever the optimizer happened to stop.
/// * `±inf` — an overflow rather than an indeterminate form.
/// * the **sentinel** — finite, and the reason `is_finite()` alone is not the
///   gate. The inner objective clamps a blown-up individual contribution to a
///   [`SUBJECT_SENTINEL_NLL`] rail, the outer objective doubles it, and an
///   optimizer can report `Success` sitting on that plateau because every
///   direction looks flat. The fit was *repelled*, not solved. See
///   [`DIVERGENCE_OFV`].
/// * **too large to be a fit** — finite, at or above [`DIVERGENCE_OFV`] but below
///   the sentinel, so no subject sits on the `1e20` rail (#2038). Usually a
///   subject's own objective is that large on its own terms: measured, a
///   proportional error on a prediction of 0 bottoms out at the residual-SD floor
///   and scores `4.6e14`. Calling that the sentinel sent the user looking for a
///   clamp that never happened.
///
/// The line is drawn on the population total, so it is a reading, not a proof,
/// in both directions: an unclamped subject can reach `1e20` on its own (the same
/// zero prediction against a DV of 12000 scores `1.44e20`), and the IOV + FREM
/// refusal in `crate::stats::likelihood` returns `1e18`, so a subject refused
/// there reads as too large rather than as clamped. The estimators only have the
/// total; `fit()` has the subjects, and rereads the label from them where it can
/// (`api::postfit::name_nonfinite_subjects`, #2265).
pub(crate) fn nonfinite_objective_reason(ofv: f64) -> &'static str {
    if ofv.is_nan() {
        "NaN"
    } else if ofv.is_infinite() {
        "infinite"
    } else if ofv >= SUBJECT_SENTINEL_NLL {
        REASON_AT_SENTINEL
    } else {
        REASON_TOO_LARGE
    }
}

/// The finite-sentinel label of [`nonfinite_objective_reason`]. One spelling, since
/// `fit()`'s reread from the subjects (#2265) writes the same labels and compares them
/// against the gate's: two copies would read as a relabel on every reread.
pub(crate) const REASON_AT_SENTINEL: &str = "at the divergence sentinel";

/// The past-the-cutoff label of [`nonfinite_objective_reason`]; see
/// [`REASON_AT_SENTINEL`].
pub(crate) const REASON_TOO_LARGE: &str = "too large to be a fit";

/// **The gate (#1303).** Demote a convergence verdict the reported objective does not
/// support, and return the warning saying why (`None` when nothing to demote).
///
/// The optimizer's stop rule reports on its trace, while the reported objective is
/// recomputed at the restored best point, so `converged` could be `true` with `ofv = NaN`.
/// **`is_finite()` is not the test**: the clamped `~1e20` sentinel is finite;
/// [`ofv_is_valid`] covers both (pinned by
/// `ofv_is_valid_rejects_the_clamped_sentinel_not_just_non_finite`).
/// Called at every site publishing a `(converged, ofv)` pair (`run_foce_gn` / `run_imp` /
/// `run_bayes` are public; `fit()` adds the prior penalty after the optimizer), pinned by
/// `every_published_convergence_verdict_is_gated_on_its_objective`.
/// **The incoming verdict is overwritten, not consulted**: a run already demoted for another
/// reason must still report the objective failure. Callers that can see the message twice
/// deduplicate (`api::fit`).
pub(crate) fn gate_converged_on_objective(converged: &mut bool, ofv: f64) -> Option<String> {
    if ofv_is_valid(ofv) {
        return None;
    }
    *converged = false;
    Some(nonfinite_objective_message(ofv, nonfinite_objective_reason(ofv)))
}

/// The [`gate_converged_on_objective`] message for `ofv` under the label `reason`.
/// Separate from the gate so `fit()` can rebuild it word for word when the
/// subjects give a better reading of the label than the total does (#2265); the
/// estimators' copies are deduplicated by suffix, so the two must not drift.
pub(crate) fn nonfinite_objective_message(ofv: f64, reason: &str) -> String {
    format!(
        "{NONFINITE_OBJECTIVE_TOKEN}: the objective at the final estimates is {reason} ({ofv:?}), so \
         the run did not converge on a solution of the problem posed and is reported \
         converged: false whatever the optimizer's own stop rule said. The parameter \
         estimates are wherever the optimizer stopped, not a minimum, and every quantity \
         derived from the objective (OFV, AIC, BIC, standard errors, any likelihood-ratio \
         comparison) is meaningless. Look for the subject or record that poisons the \
         objective — a non-finite dose time, lagtime, bioavailability or infusion duration, \
         a covariate model that overflows at an observed covariate value, or a residual \
         variance driven to zero — and fix the data or the model rather than the optimizer \
         settings."
    )
}

/// The `W_` token a run that never scored the real objective carries (#1617), so
/// [`crate::types::classify_warning`] routes it on the token rather than on prose a
/// later edit could change out from under it.
pub(crate) const NO_FEASIBLE_EVAL_TOKEN: &str = "W_NO_FEASIBLE_EVAL";

/// What an outer run's evaluation trace says about whether it ever stood on
/// feasible ground — the input to [`no_feasible_eval_warning`].
pub(crate) struct RunFeasibility {
    /// Unguarded evaluations, read off the feasible-eval plateau tracker. `0` means
    /// the optimizer never once scored the real objective.
    pub(crate) feasible_evals: usize,
    /// Evaluations [`ebe_guard_rejects`] rejected.
    pub(crate) n_guard_rejected: usize,
    /// Evaluations the `guarded` branch absorbed because the objective was not finite, counted
    /// at the source (#1617): cancellation and the latched stagnation guard bump the callback
    /// counter without an objective, so "callbacks minus guard rejections" would be wrong.
    pub(crate) n_nonfinite: usize,
    /// Worst per-evaluation unconverged-subject count seen (the fraction trigger).
    pub(crate) max_unconverged: usize,
    /// Subjects in the population — the denominator the guard's fraction is taken over.
    pub(crate) n_subj: usize,
    /// `FitOptions::max_unconverged_frac`, quoted back in the remedy.
    pub(crate) max_unconverged_frac: f64,
    /// `FitOptions::inner_maxiter`, quoted back in the remedy.
    pub(crate) inner_maxiter: usize,
    /// The user cancelled. Cancelled evaluations short-circuit at the top of the
    /// objective, *before* the feasible counter advances, so `feasible_evals == 0`
    /// then says nothing about the model and must not be reported as a defect.
    pub(crate) cancelled: bool,
}

impl RunFeasibility {
    /// Evaluations that actually produced an objective for the optimizer to reject —
    /// the number the message calls the run's *trials*. Deliberately the sum of the two
    /// causes rather than the raw callback count, so the two cannot disagree: see
    /// [`RunFeasibility::n_nonfinite`].
    pub(crate) fn n_trials(&self) -> usize {
        self.n_guard_rejected + self.n_nonfinite
    }

    /// Whether the run never scored the real objective. Spelled once because the best-seen
    /// restore (which point is reported) and [`no_feasible_eval_warning`] (what is said about
    /// it) must agree.
    pub(crate) fn no_feasible(&self) -> bool {
        self.feasible_evals == 0 && !self.cancelled
    }
}

/// Lower-case the first character, for a sentence fragment spliced after "For those, ".
/// Spelled out rather than inlined so the shared remedy text exists exactly once — it is
/// the arm's whole advice in one place and the mixed clause's in the other.
fn lower_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
    }
}

/// Demote `converged` and name the reason when **no** outer evaluation was feasible
/// (#1617): every trial was guard-rejected or non-finite.
///
/// No other gate catches it: guard-rejected trials score [`guard_penalty_value`], a bowl
/// centred on the scaled box midpoint, so the optimizer walks there and NLopt returns
/// `Success`; the final re-score is a finite OFV that [`gate_converged_on_objective`] and
/// [`cold_solve_verdict`] accept. Measured: warfarin at `inner_maxiter = 5` reported
/// `converged = true` with θ at `sqrt(lower·upper)` to 3e-9.
///
/// The unconverged share is reported from the **worst** trial (what sizes `inner_maxiter`).
/// Two cause arms: the **guard** (remedy `inner_maxiter` / `max_unconverged_frac`) and
/// **non-finite** objectives (the data/model remedy). The guard's start-rejection trigger
/// cannot fire on an all-guarded run ([`adopt_warm_start`] never updates the cache, #1290;
/// pinned by `adopt_warm_start_takes_only_feasible_improvements`), so it has no arm, and
/// the guard sentence says "against" `max_unconverged_frac` to stay true regardless.
/// `cancelled` short-circuits: a cancelled run is infeasible by construction.
pub(crate) fn no_feasible_eval_warning(
    converged: &mut bool, run: &RunFeasibility,
) -> Option<String> {
    if !run.no_feasible() {
        return None;
    }
    *converged = false;

    let n_subj = run.n_subj.max(1);
    let poisoned_objective_remedy = "Look for the parameter value or record that poisons \
         the objective at and around the starting estimates — a non-finite dose time, \
         lagtime, bioavailability or infusion duration, a covariate model that overflows, \
         or a residual variance driven to zero.";
    let (cause, remedy) = if run.n_guard_rejected > 0 {
        (
            format!(
                "the EBE convergence guard rejected {} of them; at worst {} of {} subjects' \
                 inner loops failed to converge, against `max_unconverged_frac` = {}.",
                run.n_guard_rejected, run.max_unconverged, n_subj, run.max_unconverged_frac,
            ),
            format!(
                "Raise `inner_maxiter` (currently {}) so the inner loop can reach the EBEs \
                 within its budget, or raise `max_unconverged_frac` (`1.0` disables the guard) \
                 to let the fit proceed with the unconverged subjects.",
                run.inner_maxiter,
            ),
        )
    } else {
        (
            "every trial returned a non-finite objective, so the EBE convergence guard never \
             came into it."
                .to_string(),
            format!("{poisoned_objective_remedy} This is not an optimizer setting."),
        )
    };
    // A trace with both causes gets both remedies (#1617 review round 3 #1). The guard
    // remedy alone would be advice that cannot reach the trials whose objective was not a
    // number — `inner_maxiter` does not make a `NaN` finite — and the reader has no way to
    // know the sentence only covers part of what the sentence before it just reported.
    let mixed = if run.n_guard_rejected > 0 && run.n_nonfinite > 0 {
        (
            format!(" The remaining {} returned a non-finite objective.", run.n_nonfinite,),
            format!(" For those, {}", lower_first(poisoned_objective_remedy)),
        )
    } else {
        (String::new(), String::new())
    };

    Some(format!(
        "{NO_FEASIBLE_EVAL_TOKEN}: the outer optimizer finished without a single feasible \
         objective evaluation — all {} of its trials were rejected before they could be \
         ranked, so no OFV was ever minimised and the estimates reported are the ones the \
         fit started from. Specifically, {cause}{} {remedy}{}",
        run.n_trials(),
        mixed.0,
        mixed.1,
    ))
}

/// The token every [`above_start_warning`] message starts with (#2461).
pub(crate) const ABOVE_START_TOKEN: &str = "W_ABOVE_START";

/// The first evaluation of an outer run — the point it was started from — as the objective
/// saw it **before** the EBE convergence guard (#2461). [`above_start_warning`] is its only
/// reader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StartEval {
    /// The objective the optimizer ranks (priors and NN penalty included), unguarded.
    pub ofv: f64,
    /// Whether the EBE convergence guard rejected the evaluation.
    pub guard_rejected: bool,
    /// Subjects whose inner loop did not converge there.
    pub n_unconverged: usize,
    /// Subjects rejected at their inner start, whose `(η, H)` is a placeholder.
    pub n_start_rejected: usize,
    /// Set when the run's own start scored `∞` and this is that start re-scored at a larger
    /// inner budget (#2619, `floor_escape::escape_floor`); `None` for a start scored first time.
    pub rescored: Option<StartRescore>,
    /// Set when the guard rejected the start's cold inner solve and the warm re-solve from
    /// its own EBEs was accepted instead (#2682): the subjects unconverged cold. The other
    /// fields then describe the warm re-solve.
    pub warm_rescored_from: Option<usize>,
}

/// The inner budgets of a re-scored start (#2619): the run's own, at which the starting
/// evaluation failed, and the one it was re-scored at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StartRescore {
    pub run_inner_maxiter: usize,
    pub rescore_inner_maxiter: usize,
}

/// Demote `converged` and say why when an outer run reports an objective materially above
/// the one its own starting estimates scored (#2461).
///
/// The best-seen restore only ranks guard-accepted evaluations, so a rejected start is never
/// a candidate: on `transit_flip_flop_fit_matches_ode_twin`, started at the optimum, one of
/// four subjects fell back to Nelder–Mead, the guard rejected the start (−228.81) and the
/// run walked off to −59.31 (macOS arm64) / −215.687 (Linux aarch64).
///
/// `reported` is the published objective plus the penalties `start.ofv` carries.
/// "Materially" is [`PLATEAU_CONSISTENCY_REL_TOL`]·(1 + |start|). A start with a
/// hard-rejected subject is not compared (placeholder `(η, H)`), nor one that scored `∞`
/// (re-scored first where possible, #2619, `start.rescored`).
pub(crate) fn above_start_warning(
    converged: &mut bool, start: Option<StartEval>, reported: f64, n_subj: usize,
    max_unconverged_frac: f64,
) -> Option<String> {
    let start = start?;
    if start.n_start_rejected > 0
        || !start.ofv.is_finite()
        || !reported.is_finite()
        || cold_solve_is_consistent(Some(start.ofv), reported)
    {
        return None;
    }
    *converged = false;
    let rescored = match start.rescored {
        Some(r) => format!(
            " The starting evaluation failed outright at `inner_maxiter` = {}, so that figure is \
             the start re-scored once, evaluation-only, at `inner_maxiter` = {}.",
            r.run_inner_maxiter, r.rescore_inner_maxiter,
        ),
        None => String::new(),
    };
    let cause = if start.guard_rejected {
        format!(
            " The EBE convergence guard rejected that starting evaluation: {} of {} subjects' \
             inner loops did not converge there, against `max_unconverged_frac` = {}, so the \
             optimizer never ranked the starting point and was pushed away from it.",
            start.n_unconverged,
            n_subj.max(1),
            max_unconverged_frac,
        )
    } else {
        String::new()
    };
    Some(format!(
        "{ABOVE_START_TOKEN}: the outer optimizer reported OFV {reported:.6}, {:.4} above the \
         {:.6} its starting estimates scored, so the fit is not reported as converged.\
         {rescored}{cause}",
        reported - start.ofv,
        start.ofv,
    ))
}

/// Resolve a model's declared parameter priors (#254) against `template`'s packed layout,
/// for an optimizer that needs the penalty on its objective. **The gate for a bad prior is
/// [`crate::api::validation::check_parameter_priors`]** (run by `fit_inner` before any
/// optimizer; pinned by `fit_refuses_an_unresolvable_prior`), so an `Err` here means the
/// caller bypassed `fit()` (ferx-tools, unit tests) and the answer is an empty set, not a
/// panic.
pub(crate) fn build_prior_set(
    model: &CompiledModel, template: &ModelParameters,
) -> crate::estimation::priors::PriorSet {
    crate::estimation::priors::PriorSet::build(model, template).unwrap_or_default()
}

pub(crate) fn resolve_outer_ftol(
    is_non_gaussian: bool, is_ode: bool, override_ftol: Option<f64>,
) -> f64 {
    override_ftol.unwrap_or(if is_non_gaussian && !is_ode { 1e-8 } else { 1e-6 })
}

/// Absolute OFV improvement below which a step counts as "no significant
/// progress" for the plateau tracker. Matches the stagnation guard's
/// `STAGNATION_THRESHOLD` (both key off the ~1e-3 FOCE EBE-loop precision).
const PLATEAU_OFV_THRESHOLD: f64 = 1e-3;

/// Minimum consecutive flat tail evals (no improvement above `PLATEAU_OFV_THRESHOLD`) for a
/// bare NLopt `Failure`/`ForcedStop` to count as a converged plateau (#751). Below the
/// shortest observed good-fit tail (npde ≈ 8, schnider ≈ 19); a real stall has none.
const PLATEAU_MIN_FLAT_EVALS: usize = 5;

/// Relative tolerance for the best-seen ↔ cold final inner-loop OFV self-consistency guard.
/// A converged fit's EBE fixpoint reproduces cold; a large positive gap (SS-oral: best-seen
/// 83.3 vs cold 121.4) is a warm-start artifact, rejected even on a flat trace.
const PLATEAU_CONSISTENCY_REL_TOL: f64 = 1e-3;

/// Which of the final inner loop's EBE candidates the fit reports (#833): the cold re-solve,
/// a re-solve seeded from the incumbent EBEs, and those EBEs as left (when they belong to
/// this point, [`IncumbentSolve`]). The lowest finite objective wins.
///
/// - **Ties go to the earliest** (callers pass cold first), so unimodal fits are unchanged.
/// - **Non-finite never wins**: `NaN`/`±inf` are skipped, not compared (`warm < NaN` is
///   false, which would pin the fit to a blown-up cold solve).
/// - **All non-finite** returns index 0, and `gate_converged_on_objective` demotes (#1303).
fn reported_candidate(objectives: &[f64]) -> usize {
    objectives
        .iter()
        .enumerate()
        .filter(|(_, o)| o.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("both finite"))
        .map_or(0, |(i, _)| i)
}

/// The EBE state of the best evaluation seen, copied out of the objective closure so the
/// final inner loop can **score** it and seed a re-solve from it (#833); `etas`/`h_mats`/
/// `kappas` are the triple `pop_nll_opts` consumes. `xs` is carried because scoring is only
/// sound at the same point (`h_mats` is that point's curvature), and
/// [`BestPoint::observe`] also records guard-penalised evals whose EBEs are not adopted.
#[derive(Clone)]
struct IncumbentSolve {
    xs: Vec<f64>,
    etas: Vec<DVector<f64>>,
    h_mats: Vec<DMatrix<f64>>,
    kappas: Vec<Vec<DVector<f64>>>,
    /// A mixture's every class's modes at `xs`, `[class][subject]` (empty for a single
    /// population): the warm start of the remaining-gain solve.
    class_etas: Vec<Vec<DVector<f64>>>,
    /// Under the quadrature, each subject's packed score at `xs`, when this evaluation formed
    /// its gradient: the BHHH decrement a tolerance stop reads (#2549).
    scores: Option<Vec<Vec<f64>>>,
}

/// What a cold re-solve at the final estimates says about `reference_ofv` (#833).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColdSolveVerdict {
    /// The cold solve lands on `reference_ofv` (or better). Nothing to recover, nothing
    /// to report.
    Reproduces,
    /// It lands materially above it, by this much.
    Worse(f64),
    /// It did not produce a usable number at all (`NaN` / `±inf`). Its own arm because a
    /// gap test cannot see it: every comparison against `NaN` is false, so `Worse` would
    /// silently decline both the retry and the warning in the one case where the cold
    /// EBEs are certainly not the ones to report.
    NonFinite,
}

impl ColdSolveVerdict {
    /// Did the cold solve fail to reproduce the reference?
    fn missed(self) -> bool {
        !matches!(self, ColdSolveVerdict::Reproduces)
    }
}

/// Compare a cold re-solve against `reference_ofv` on the absolute [`BASIN_SHIFT_ABS_TOL`]
/// (the opposite direction's bound). The former relative `1e-3·(1 + |OFV|)` hid gaps of 2.3
/// and 0.37 on #2646's benchmarks. Against `best_seen_ofv` it triggers the incumbent
/// scoring and re-solve; against the reported OFV it is the warning.
fn cold_solve_verdict(cold_ofv: f64, reference_ofv: f64) -> ColdSolveVerdict {
    if !cold_ofv.is_finite() {
        return ColdSolveVerdict::NonFinite;
    }
    let gap = cold_ofv - reference_ofv;
    if gap > BASIN_SHIFT_ABS_TOL {
        ColdSolveVerdict::Worse(gap)
    } else {
        ColdSolveVerdict::Reproduces
    }
}

/// The user-facing message for a cold re-solve that did not reproduce the reported
/// objective (#833), or `None`. It does **not** diagnose multimodality: a start-dependent
/// EBE surface and a too-small `inner_maxiter` are indistinguishable here, so both are
/// named. `ids` ([`subjects_cold_lost`]) are largest gap first; empty when the gap is
/// diffuse or not split by subject (AGQ).
fn ebe_start_dependence_warning(
    cold_ofv: f64, reported_ofv: f64, ids: &[String],
) -> Option<String> {
    match cold_solve_verdict(cold_ofv, reported_ofv) {
        ColdSolveVerdict::Reproduces => None,
        ColdSolveVerdict::Worse(gap) => Some(format!(
            "W_EBE_START_DEPENDENT: empirical Bayes estimates at the final parameters \
             depend on the inner loop's starting point — re-solving them cold scores \
             {gap:.4} OFV units worse than the EBEs the optimizer minimised against \
             ({cold_ofv:.4} vs the reported {reported_ofv:.4}){named}. The reported fit \
             keeps the lowest-scoring candidate, subject by subject. Either the individual objective has more than one \
             mode at these estimates, or the inner loop cannot reach it from a cold start \
             within `inner_maxiter`; in both cases the EBEs — and the diagnostics built on \
             them (IPRED, IWRES, CWRES, shrinkage) and the covariance step — depend on \
             where the inner loop starts. Raising `inner_maxiter`, or `inner_restarts` for \
             a suspected second mode, tells the two apart.",
            named = if ids.is_empty() {
                String::new()
            } else {
                format!(", on {}", counted_subjects(ids))
            },
        )),
        ColdSolveVerdict::NonFinite => Some(format!(
            "W_EBE_START_DEPENDENT: re-solving the empirical Bayes estimates cold at the \
             final parameters returned a non-finite objective ({cold_ofv}), against the \
             reported {reported_ofv:.4} from the EBEs the optimizer minimised against. The \
             reported fit uses the latter, but an inner solve that diverges from a cold \
             start is the strongest form of start dependence: every EBE-derived diagnostic \
             (IPRED, IWRES, CWRES, shrinkage) and the covariance step depend on the start. \
             Check the subjects with the largest individual objectives, and refit with a \
             larger `inner_maxiter` or with `inner_restarts` before reading them."
        )),
    }
}

/// How many subject IDs a warning lists before it counts the rest.
const WARNING_ID_LIST_CAP: usize = 10;

/// `ids` for a warning: "1 subject (A)", or "12 subjects (A, B, … and 2 more)".
fn counted_subjects(ids: &[String]) -> String {
    let n = ids.len();
    let listed = ids[..n.min(WARNING_ID_LIST_CAP)].join(", ");
    let list = if n > WARNING_ID_LIST_CAP {
        format!("{listed} and {} more", n - WARNING_ID_LIST_CAP)
    } else {
        listed
    };
    if n == 1 {
        format!("1 subject ({list})")
    } else {
        format!("{n} subjects ({list})")
    }
}

/// The user-facing message for subjects whose EBE the cold re-solve's Sobol screen (#2013)
/// or flip-flop twin seed (#2095) moved, or `None`. Filed under `ebe_start_dependent` by its
/// `W_` token; each named EBE is one a start from η = 0 does not reach (a move is accepted
/// only at a lower objective more than 0.5 away in the Ω⁻¹ metric). Like #833's warning it
/// does not diagnose multimodality or name a candidate.
pub(crate) fn ebe_screen_moved_warning(ids: &[String]) -> Option<String> {
    if ids.is_empty() {
        return None;
    }
    let who = counted_subjects(ids);
    Some(format!(
        "W_EBE_START_DEPENDENT: at the final parameters, the inner loop's Sobol screen \
         (`inner_screen`) or flip-flop twin seed (`inner_twin_seed`) moved the empirical \
         Bayes estimates of {who} to a lower \
         individual objective than the descent from η = 0 reached. Either the individual \
         objective has more than one mode there, or that descent stopped short of it; in \
         both cases these EBEs, and the diagnostics built on them (IPRED, IWRES, CWRES, \
         shrinkage), depend on where the inner loop starts. `inner_screen = 0` turns the \
         screen off, and `inner_twin_seed = false` the twin seed."
    ))
}

/// How far, in absolute OFV, a cold re-solve may land *below* best-seen before the run
/// counts as optimized under trapped EBEs (#1565). Absolute because one subject's mode
/// switch is worth the same at OFV 790 as at 65,000. Untrapped subjects agree to ~1e-4
/// (#1565 MM fixture), so 0.05 is well above noise and far below the 88-unit gap.
const BASIN_SHIFT_ABS_TOL: f64 = 0.05;

/// Whether the mid-leg basin probe runs at this feasible-evaluation count: 32, 64, 128, …
/// — a logarithmic number of cold passes per leg, none before the trajectory has had a
/// chance to settle (#2093).
fn basin_probe_due(feasible_evals: usize) -> bool {
    feasible_evals >= BASIN_PROBE_FIRST && feasible_evals.is_power_of_two()
}

/// The first feasible evaluation at which [`basin_probe_due`] fires.
const BASIN_PROBE_FIRST: usize = 32;

/// What the mid-leg basin probe does with a cold re-solve that beat the best seen (#2600).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BasinProbeAction {
    /// End the leg for a basin-shift restart: a fresh optimizer from the incumbent, its
    /// first inner solve cold (#2093).
    Restart,
    /// Keep the leg running, with the cold re-solve's modes as the next evaluation's warm
    /// starts. The optimizer's curvature memory survives; the surface it minimises from
    /// here on is the one the cold modes define.
    Absorb,
}

/// The share of the population that has to sit in a lower cold mode before the mid-leg
/// probe restarts the leg rather than absorbing the modes into it ([`basin_probe_action`]).
const BASIN_PROBE_RESTART_SHARE: f64 = 0.01;

/// Restart or absorb, from how many subjects the probe's cold re-solve put in a lower
/// individual mode (`switched`, [`subjects_in_a_lower_cold_mode`]) out of `n_subjects`
/// (#2600).
///
/// The #2093 probe restarted on any gap above [`BASIN_SHIFT_ABS_TOL`]. On a 10-subject
/// fixture one subject switching is a tenth of the objective, and the curvature pairs the
/// optimizer collected on the trapped surface are worth little afterwards. On a
/// 5,937-subject FOCEI fit a handful of the subjects will almost always sit in a lower cold
/// mode, so every leg was cut at its 32nd evaluation — five legs, gains 33.5 / 14.3 / 0.99 /
/// 1.14 / 0.55 from 19 / 9 / 3 / 5 / 4 switched subjects — no leg reached its own stopping
/// rule, and the fit ended with |g|∞ = 16.3. Restarting throws the L-BFGS memory away,
/// and 19 of 5,937 terms changing leaves most of it describing the same surface; so below
/// [`BASIN_PROBE_RESTART_SHARE`] of the population the switch is absorbed into the running
/// leg instead. The end-of-leg check (#1565) still runs on every leg, so a gap that
/// survives to the leg's own stop still files its restart and its warning.
///
/// `switched = None` — the count is not available (AGQ, whose per-subject term is not the
/// one the count scores, or no held state at the incumbent) — restarts, as before #2600.
///
/// `cold_guard_rejected` — the cold solve fails the EBE guard an ordinary evaluation must
/// pass ([`ebe_guard_rejects`]: too many unconverged subjects, or a start rejected) — also
/// restarts, as before #2600. Absorbing makes the cold solve the incumbent in every tracker
/// ([`absorb_cold_incumbent`]), and an objective the optimizer would have rejected must not
/// become the bar every later valid evaluation has to beat, nor the EBE state the final
/// restore scores. The restart's own first inner solve is guarded as usual.
fn basin_probe_action(
    switched: Option<usize>, n_subjects: usize, cold_guard_rejected: bool,
) -> BasinProbeAction {
    match switched {
        Some(k)
            if !cold_guard_rejected
                && (k as f64) < basin_probe_restart_share() * n_subjects as f64 =>
        {
            BasinProbeAction::Absorb
        }
        _ => BasinProbeAction::Restart,
    }
}

/// [`BASIN_PROBE_RESTART_SHARE`], or this thread's `BASIN_PROBE_RESTART_SHARE_TEST`
/// override (tests only, for the A/B of the two arms on one fit).
fn basin_probe_restart_share() -> f64 {
    #[cfg(test)]
    if let Some(share) = BASIN_PROBE_RESTART_SHARE_TEST.with(|c| c.get()) {
        return share;
    }
    BASIN_PROBE_RESTART_SHARE
}

/// The objective the mid-leg basin probe compares against: the active leg's incumbent
/// (#2600). A #2462 verification leg solves at a tighter tolerance and ranks in `leg_best`,
/// so its probe uses that leg's options and `leg_best` rather than mixing surfaces.
fn basin_probe_reference(
    in_verify_leg: bool, leg_best: &BestPoint, best_seen: &BestPoint,
) -> Option<f64> {
    let best = if in_verify_leg { leg_best } else { best_seen };
    best.get().map(|b| b.ofv_clean)
}

/// Make the mid-leg probe's cold solve the leg's incumbent when it absorbs it (#2600):
/// `ofv = (penalized, clean)` at `solve.xs` with EBE state `solve`; warm starts are built
/// from `solve` itself ([`outer_cache_warm_starts`], #2674).
///
/// Every improvement tracker moves to the cold objective, or a later trial between it and
/// the warm one would count as an improvement: `state.best_ofv` ([`adopt_warm_start`]),
/// `best` (best-seen or the leg's incumbent), `best_solve`, the plateau tracker's baseline;
/// `last_gradient` is cleared (it belongs to the warm modes). The mode response and
/// inner-Hessian seeds go too ([`NloptState::reset_to_cold_start`]).
#[allow(clippy::too_many_arguments)]
fn absorb_cold_incumbent(
    state: &mut NloptState, best: &mut BestPoint, best_solve: &mut Option<IncumbentSolve>,
    last_gradient: &mut Option<Vec<f64>>, plateau: &mut (f64, usize, usize),
    (ofv, ofv_clean): (f64, f64), model: &CompiledModel, solve: IncumbentSolve,
) {
    state.restart_from_modes(Some(outer_cache_warm_starts(model, &solve.etas, &solve.kappas)));
    if ofv < state.best_ofv {
        state.best_ofv = ofv;
    }
    if best.observe(state.n_evals, &solve.xs, ofv, ofv_clean) {
        *best_solve = Some(solve);
        *last_gradient = None;
    }
    if plateau.0 - ofv > PLATEAU_OFV_THRESHOLD {
        plateau.0 = ofv;
        plateau.1 = plateau.2;
    }
}

/// By how much a cold re-solve at the restored point *beat* the objective the optimizer
/// minimised there, when by more than [`BASIN_SHIFT_ABS_TOL`] (#1565). The other half of
/// [`failure_is_converged_plateau`] / [`cold_solve_verdict`] (which ask "worse?"): some
/// warm EBEs sat in a secondary mode, so θ̂ optimises a surrogate. Measured on the #1565 MM
/// ODE: best-seen 872.70, cold 784.33, `converged = true`; a refit reached 775.39.
/// `None` when a side is missing (mixture; no feasible eval) or non-finite
/// ([`ColdSolveVerdict::NonFinite`] owns that).
fn cold_solve_beats_best_seen(cold_ofv: Option<f64>, best_seen_ofv: Option<f64>) -> Option<f64> {
    let (cold, best) = (cold_ofv?, best_seen_ofv?);
    if !cold.is_finite() || !best.is_finite() {
        return None;
    }
    let gap = best - cold;
    (gap > BASIN_SHIFT_ABS_TOL).then_some(gap)
}

/// How many subjects a cold re-solve put in a lower mode than the optimizer's EBEs (#1565):
/// those whose `−2LL` term drops by more than [`PLATEAU_OFV_THRESHOLD`]. Each side is
/// scored with its own `(η̂, H)` through `pop_nll`'s per-subject term, so the count sums to
/// the gap. Asked only when the held state belongs to this point and not under AGQ.
#[allow(clippy::too_many_arguments)]
fn subjects_in_a_lower_cold_mode(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    (held_etas, held_h_mats, held_kappas): (&[DVector<f64>], &[DMatrix<f64>], &[Vec<DVector<f64>>]),
    cold_etas: &[DVector<f64>], cold_h_mats: &[DMatrix<f64>], cold_kappas: &[Vec<DVector<f64>>],
    interaction: bool,
) -> usize {
    population
        .subjects
        .iter()
        .enumerate()
        .filter(|&(i, subject)| {
            let nll =
                |etas: &[DVector<f64>], hms: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>]| {
                    subject_nll(
                        model,
                        subject,
                        params,
                        &etas[i],
                        &hms[i],
                        kappas.get(i).map_or(&[], Vec::as_slice),
                        interaction,
                    )
                };
            let warm = nll(held_etas, held_h_mats, held_kappas);
            let cold = nll(cold_etas, cold_h_mats, cold_kappas);
            // `>` on a difference: a `NaN` side is never counted as a shift.
            2.0 * (warm - cold) > PLATEAU_OFV_THRESHOLD
        })
        .count()
}

/// The user-facing message for a run whose cold re-solve beat best-seen (#1565). Reaches
/// `FitResult` only when the gap survives (an adopted [`RestartCause::BasinShift`] replaces
/// the result). Filed under `ebe_start_dependent`, the other direction of #833. `report`
/// says which objective is published and whether a restart was possible (#2097).
fn ebe_basin_shift_warning(
    cold_ofv: f64, best_seen_ofv: f64, gap: f64, n_subjects: Option<usize>,
    report: BasinShiftReport,
) -> String {
    let who = match n_subjects {
        Some(1) => "1 subject's EBEs".to_string(),
        Some(n) => format!("{n} subjects' EBEs"),
        None => "some subjects' EBEs".to_string(),
    };
    let (reported, remedy) = match report {
        BasinShiftReport::ColdRestartable => (
            "not the objective now reported",
            " — or raise `maxiter` if the fit's evaluation budget ran out before it could \
             be restarted.",
        ),
        BasinShiftReport::Cold => ("not the objective now reported", "."),
        BasinShiftReport::Warm => (
            "and the reported OFV and EBEs are still that surrogate's: the cold ones score \
             lower at these same estimates",
            ".",
        ),
    };
    format!(
        "W_EBE_START_DEPENDENT: re-solving the empirical Bayes estimates cold at the final \
         parameters scores {gap:.4} OFV units *lower* than the objective the optimizer \
         minimised there ({cold_ofv:.4} vs {best_seen_ofv:.4}): {who} stayed in a higher \
         individual mode throughout the warm-started trajectory, so the parameters \
         minimise that higher surrogate, {reported}, and the fit is not reported as \
         converged. Refit starting from these estimates — the first inner solve of a new \
         fit is cold, so it starts in the lower modes{remedy}"
    )
}

/// Which objective a fit whose cold re-solve beat its own publishes, and whether the
/// leg could have been restarted — the two facts [`ebe_basin_shift_warning`] states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BasinShiftReport {
    /// The NLopt leg: it reports the cold solve, and `optimize_nlopt` restarts it while
    /// budget remains, so a surviving warning can mean the budget ran out.
    ColdRestartable,
    /// Reports the cold solve, with no restart (the trust region).
    Cold,
    /// Keeps reporting the optimizer's own warm EBEs and OFV (Gauss–Newton).
    Warm,
}

/// One candidate for the EBEs a fit reports: the triple `pop_nll_opts` consumes and the
/// −2LL it scores them at.
pub(crate) type EbeCandidate = (Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<DVector<f64>>>, f64);

/// A borrowed EBE triple: the η̂, `H` and κ̂ `pop_nll_opts` consumes.
type EbeTriple<'a> = (&'a [DVector<f64>], &'a [DMatrix<f64>], &'a [Vec<DVector<f64>>]);

/// Score an EBE triple on the fit's objective at `params` (#833).
pub(crate) fn score_final_ebes(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    (etas, h_mats, kappas): (Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<DVector<f64>>>),
    options: &FitOptions,
) -> EbeCandidate {
    let nll = pop_nll_opts(model, population, params, &etas, &h_mats, &kappas, options);
    (etas, h_mats, kappas, 2.0 * nll)
}

/// The `base` candidate, with each subject another candidate beats by more than
/// [`BASIN_SHIFT_ABS_TOL`] in `−2LL` moved onto its lowest, plus each subject's source.
/// `terms[c][i]` is subject `i`'s term under candidate `c`. The bound keeps this a choice
/// between modes (a plain minimum reported −1199.3302 below the optimum −1199.3264 on
/// `tests/run_covariance_reported_ebes.rs`). `(η̂, H, κ̂)` move together; lowest per
/// [`reported_candidate`], so non-finite never wins and a non-finite `base` term moves.
fn lowest_per_subject(
    candidates: &[EbeCandidate], terms: &[Vec<f64>], base: usize,
) -> ((Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<DVector<f64>>>), Vec<usize>) {
    let source: Vec<usize> = (0..terms[0].len())
        .map(|i| {
            let column: Vec<f64> = terms.iter().map(|t| t[i]).collect();
            let lowest = reported_candidate(&column);
            let moves = !column[base].is_finite()
                || 2.0 * (column[base] - column[lowest]) > BASIN_SHIFT_ABS_TOL;
            if moves {
                lowest
            } else {
                base
            }
        })
        .collect();
    let pick = |i: usize| &candidates[source[i]];
    let etas = (0..source.len()).map(|i| pick(i).0[i].clone()).collect();
    let h_mats = (0..source.len()).map(|i| pick(i).1[i].clone()).collect();
    // A model without IOV may carry no κ̂ at all; that shape is kept, since postfit branches
    // on `kappas.is_empty()`.
    let kappas = if candidates[0].2.is_empty() {
        Vec::new()
    } else {
        (0..source.len()).map(|i| pick(i).2.get(i).cloned().unwrap_or_default()).collect()
    };
    ((etas, h_mats, kappas), source)
}

/// The subjects `other` beats the cold solve on by more than [`BASIN_SHIFT_ABS_TOL`] in
/// `−2LL`, largest gap first: the ones a cold start cannot be said to reproduce (#2646).
/// Terms are the `nll`s of [`subject_nlls`]; the `>` means a `NaN` side never counts.
fn subjects_cold_lost(population: &Population, cold: &[f64], other: &[f64]) -> Vec<String> {
    let mut lost: Vec<(f64, &str)> = population
        .subjects
        .iter()
        .zip(cold.iter().zip(other))
        .map(|(subject, (c, o))| (2.0 * (c - o), subject.id.as_str()))
        .filter(|&(gap, _)| gap > BASIN_SHIFT_ABS_TOL)
        .collect();
    lost.sort_by(|a, b| b.0.total_cmp(&a.0));
    lost.into_iter().map(|(_, id)| id.to_string()).collect()
}

/// [`subjects_cold_lost`] between two EBE triples at `params`. Empty under AGQ, whose
/// per-subject term is not the one [`subject_nll`] computes. This is the per-subject half
/// of the trigger for [`best_final_candidate`]: population totals can agree while subjects
/// differ both ways, one held in a lower mode and another in a higher one.
fn subjects_cold_lost_to(
    model: &CompiledModel, population: &Population, params: &ModelParameters, cold: EbeTriple<'_>,
    other: EbeTriple<'_>, options: &FitOptions,
) -> Vec<String> {
    if options.agq_nodes().is_some() {
        return Vec::new();
    }
    let terms = |(etas, h_mats, kappas): EbeTriple<'_>| {
        subject_nlls(model, population, params, etas, h_mats, kappas, options.interaction)
    };
    subjects_cold_lost(population, &terms(cold), &terms(other))
}

/// What [`best_final_candidate`] decided.
pub(crate) struct FinalPick {
    pub(crate) reported: EbeCandidate,
    /// [`subjects_cold_lost`] against the reported EBEs. Empty under AGQ.
    cold_lost_ids: Vec<String>,
    /// Under AGQ, which has no per-subject term: did a whole other candidate win?
    pub(crate) cold_replaced: bool,
}

impl FinalPick {
    /// Drop the #2013 subjects whose reported EBEs are no longer the cold solve's, read off
    /// [`Self::cold_lost_ids`] rather than the source candidate (candidates at one mode differ
    /// by ~1e-6: 224 of 458 on #2646's CJ).
    fn retain_cold_reported(&self, ids: &mut Vec<String>) {
        if self.cold_replaced {
            ids.clear();
        }
        let lost: HashSet<&str> = self.cold_lost_ids.iter().map(String::as_str).collect();
        ids.retain(|id| !lost.contains(id.as_str()));
    }
}

/// #833's recovery once a cold re-solve missed the optimizer's objective: a re-solve seeded
/// from `seed`, scored beside the cold solve (first, so ties keep it) and, when it belongs
/// to this point, the optimizer's `held` state (the inner solve is not monotone from its
/// seed, #1365). Shared by the NLopt leg and the trust region (#2097). A fourth candidate
/// moves each subject to the best of the others ([`lowest_per_subject`], #2646), except
/// under AGQ whose per-subject term is not [`subject_nll`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn best_final_candidate(
    model: &CompiledModel, population: &Population, params: &ModelParameters, mu_k: &[f64],
    options: &FitOptions, cold: EbeCandidate, seed: &[DVector<f64>], held: Option<EbeCandidate>,
) -> FinalPick {
    let s = final_inner_solve(model, population, params, mu_k, Some(seed), options);
    let warm = score_final_ebes(model, population, params, (s.etas, s.h_mats, s.kappas), options);
    let mut candidates = vec![cold, warm];
    candidates.extend(held);
    let mut terms = None;
    if options.agq_nodes().is_none() {
        let per_candidate: Vec<Vec<f64>> = candidates
            .iter()
            .map(|(etas, h_mats, kappas, _)| {
                subject_nlls(model, population, params, etas, h_mats, kappas, options.interaction)
            })
            .collect();
        let whole: Vec<f64> = candidates.iter().map(|c| c.3).collect();
        let (parts, source) =
            lowest_per_subject(&candidates, &per_candidate, reported_candidate(&whole));
        candidates.push(score_final_ebes(model, population, params, parts, options));
        let lowest = source.iter().enumerate().map(|(i, &c)| per_candidate[c][i]).collect();
        terms = Some((per_candidate, lowest));
    }
    let objectives: Vec<f64> = candidates.iter().map(|c| c.3).collect();
    let pick = reported_candidate(&objectives);
    let cold_lost_ids = terms.as_ref().map_or_else(Vec::new, |(per_candidate, lowest)| {
        let reported = per_candidate.get(pick).unwrap_or(lowest);
        subjects_cold_lost(population, &per_candidate[0], reported)
    });
    FinalPick {
        reported: candidates.swap_remove(pick),
        cold_lost_ids,
        cold_replaced: terms.is_none() && pick != 0,
    }
}

/// Which EBEs a final cold solve hands on: the cold ones, unless they missed the optimizer's
/// own. `cold_missed` is the population verdict; the per-subject test beside it is here
/// because the totals can agree while subjects differ both ways (#2646). The one trigger
/// for the NLopt leg and the trust region. A fit whose cold solve misses on neither keeps
/// it, with no further solve, so such a fit is bit-identical to before.
#[allow(clippy::too_many_arguments)]
fn recover_final_ebes(
    model: &CompiledModel, population: &Population, params: &ModelParameters, mu_k: &[f64],
    options: &FitOptions, cold: EbeCandidate, seed: &[DVector<f64>], held: Option<EbeCandidate>,
    cold_missed: bool,
) -> FinalPick {
    let a_subject_missed = held.as_ref().is_some_and(|h| {
        !subjects_cold_lost_to(
            model,
            population,
            params,
            (&cold.0, &cold.1, &cold.2),
            (&h.0, &h.1, &h.2),
            options,
        )
        .is_empty()
    });
    if cold_missed || a_subject_missed {
        best_final_candidate(model, population, params, mu_k, options, cold, seed, held)
    } else {
        FinalPick { reported: cold, cold_lost_ids: Vec::new(), cold_replaced: false }
    }
}

/// A final inner solve at `params` and what its Sobol screen (#2013) and flip-flop twin
/// seed (#2095) did: the EBE triple `pop_nll_opts` consumes, and the IDs of the subjects
/// whose EBE either moved off the mode the descent from η = 0 found.
#[derive(Clone)]
pub(crate) struct FinalInnerSolve {
    pub(crate) etas: Vec<DVector<f64>>,
    pub(crate) h_mats: Vec<DMatrix<f64>>,
    pub(crate) kappas: Vec<Vec<DVector<f64>>>,
    pub(crate) screen_moved_ids: Vec<String>,
}

/// The final EBE re-solve at an outer path's estimates, cold when `seed` is `None` — shared
/// by every outer path that reports EBEs (#2097). `run_inner_loop_warm_seeded`'s solve plus
/// which subjects the screen or twin seed moved (only a cold solve runs either).
pub(crate) fn final_inner_solve(
    model: &CompiledModel, population: &Population, params: &ModelParameters, mu_k: &[f64],
    seed: Option<&[DVector<f64>]>, options: &FitOptions,
) -> FinalInnerSolve {
    inner_solve_polled(model, population, params, mu_k, seed, options, None)
}

/// The cold solve at a verified stop ([`verified_stop_owes_a_restart`]), which also seeds a
/// [`LegStart::ColdModes`] round, or `None` if the fit was
/// cancelled during it. Unlike a final readout it polls the cancel flag per subject (#574),
/// and a subject the cancel skipped holds a placeholder, which must not seed a leg.
fn cold_modes_solve(
    model: &CompiledModel, population: &Population, params: &ModelParameters, mu_k: &[f64],
    options: &FitOptions,
) -> Option<FinalInnerSolve> {
    let cold =
        inner_solve_polled(model, population, params, mu_k, None, options, options.cancel.as_ref());
    (!crate::cancel::is_cancelled(&options.cancel)).then_some(cold)
}

/// [`final_inner_solve`], polling `cancel` if given.
fn inner_solve_polled(
    model: &CompiledModel, population: &Population, params: &ModelParameters, mu_k: &[f64],
    seed: Option<&[DVector<f64>]>, options: &FitOptions,
    cancel: Option<&crate::cancel::CancelFlag>,
) -> FinalInnerSolve {
    let (etas, h_mats, _, kappas, moved) = run_inner_loop_warm_map(
        model,
        population,
        params,
        options.inner_maxiter,
        options.inner_tol,
        seed,
        Some(mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
        // A seed is a set of modes an earlier pass returned, so an exactly-zero one among
        // them is warm like the rest (#2688); without one the solve is cold.
        InnerSolvePolicy::for_stage(options, model).with_prev_etas(if seed.is_some() {
            PrevEtas::Solved
        } else {
            PrevEtas::Placeholder
        }),
        cancel,
        |_, ebe| ebe.screen_moved || ebe.twin_moved,
    );
    let screen_moved_ids = population
        .subjects
        .iter()
        .zip(&moved)
        .filter(|&(_, &m)| m)
        .map(|(subject, _)| subject.id.clone())
        .collect();
    FinalInnerSolve { etas, h_mats, kappas, screen_moved_ids }
}

/// The final EBE check for an outer path that holds its optimizer's EBEs but no best-seen
/// objective — Gauss–Newton and the trust region (#2097). A cold re-solve
/// ([`final_inner_solve`]) is compared against `incumbent` on the NLopt leg's rules:
///
/// - cold **materially better** — `converged` demoted, [`ebe_basin_shift_warning`] (#1565);
/// - cold **materially worse** or non-finite — [`ebe_start_dependence_warning`] (#833/#2646);
/// - the cold screen moved subjects — [`ebe_screen_moved_warning`] (#2013).
///
/// With `adopt` (trust region) the cold solve is reported unless it missed, then
/// [`best_final_candidate`] picks as on the NLopt leg; without it (Gauss–Newton) the
/// incumbent is always reported and the check is diagnostic. A cancelled run returns the
/// incumbent untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn final_ebe_check(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    incumbent: EbeCandidate, adopt: bool, options: &FitOptions, converged: &mut bool,
) -> (EbeCandidate, Vec<String>) {
    if crate::cancel::is_cancelled(&options.cancel) {
        return (incumbent, Vec::new());
    }
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let cold_solve = final_inner_solve(model, population, params, &mu_k, None, options);
    let mut screen_moved_ids = cold_solve.screen_moved_ids;
    let cold = score_final_ebes(
        model,
        population,
        params,
        (cold_solve.etas, cold_solve.h_mats, cold_solve.kappas),
        options,
    );
    let (cold_ofv, incumbent_ofv) = (cold.3, incumbent.3);
    if options.verbose {
        eprintln!("Cold EBE re-solve: OFV = {cold_ofv:.6} (optimizer's EBEs {incumbent_ofv:.6})");
    }
    let mut warnings = Vec::new();
    if let Some(gap) = cold_solve_beats_best_seen(Some(cold_ofv), Some(incumbent_ofv)) {
        *converged = false;
        // AGQ's per-subject term is not the one `subjects_in_a_lower_cold_mode` scores.
        let n_subjects = options.agq_nodes().is_none().then(|| {
            subjects_in_a_lower_cold_mode(
                model,
                population,
                params,
                (&incumbent.0, &incumbent.1, &incumbent.2),
                &cold.0,
                &cold.1,
                &cold.2,
                options.interaction,
            )
        });
        let report = if adopt { BasinShiftReport::Cold } else { BasinShiftReport::Warm };
        warnings.push(ebe_basin_shift_warning(cold_ofv, incumbent_ofv, gap, n_subjects, report));
    }
    let (reported, cold_lost_ids) = if adopt {
        let seed = joint_warm_starts(&incumbent.0, &incumbent.2);
        let cold_missed = cold_solve_verdict(cold_ofv, incumbent_ofv).missed();
        let pick = recover_final_ebes(
            model,
            population,
            params,
            &mu_k,
            options,
            cold,
            &seed,
            Some(incumbent),
            cold_missed,
        );
        pick.retain_cold_reported(&mut screen_moved_ids);
        (pick.reported, pick.cold_lost_ids)
    } else {
        let ids = subjects_cold_lost_to(
            model,
            population,
            params,
            (&cold.0, &cold.1, &cold.2),
            (&incumbent.0, &incumbent.1, &incumbent.2),
            options,
        );
        (incumbent, ids)
    };
    warnings.extend(ebe_start_dependence_warning(cold_ofv, reported.3, &cold_lost_ids));
    warnings.extend(ebe_screen_moved_warning(&screen_moved_ids));
    (reported, warnings)
}

/// Classify a bare NLopt `Failure`/`ForcedStop` as convergence-at-a-plateau (#751). Counts
/// are over *feasible* (unguarded) evals only, so guard thrashing cannot pad them. `true`
/// only when all hold:
///   - **progress**: the last significant improvement is after feasible eval 1
///     (`last_sig_feasible_eval >= 2`; 1-based over feasible evals, `0` = none seen). Eval 1
///     only sets the baseline, so a stall at the start (L-BFGS's overshoot on warfarin
///     FOCEI) fails, and so does a guard-rejected eval 1 followed by a feasible one, which
///     is a failed start and not descent;
///   - **left init**: the best point is at least [`INIT_ESCAPE_STEP_S`] from `x₀` (the
///     user-ODE warfarin twin improved 0.028 once and went flat, #751);
///   - **plateau**: the flat tail (`feasible_evals − last_sig_feasible_eval`) is at least
///     `PLATEAU_MIN_FLAT_EVALS`;
///   - **consistency**: the cold `final_ofv` is not materially *worse* than
///     `best_seen_ofv` (materially better is [`cold_solve_beats_best_seen`]'s, #1565).
fn failure_is_converged_plateau(
    feasible_evals: usize, last_sig_feasible_eval: usize, best_seen_ofv: Option<f64>,
    final_ofv: f64, left_init: bool,
) -> bool {
    plateau_trace_shape(feasible_evals, last_sig_feasible_eval)
        && cold_solve_is_consistent(best_seen_ofv, final_ofv)
        && left_init
}

/// Scaled-space size of the start probe's perturbation (#2506): each free coordinate of
/// the start is moved this far before the probe runs. Ten times [`INIT_ESCAPE_STEP_S`], so
/// a probe that converges back to the start has left *its own* start, and its plateau
/// verdict is the ordinary #751 one rather than another no-move case.
const START_PROBE_STEP_S: f64 = 10.0 * INIT_ESCAPE_STEP_S;

/// The start probe's objective tolerance (#2506): a probe ending more than this below the
/// start found a missed descent. Measured on the #2506 warm refits (Linux aarch64): probes
/// end between 1.7e-7 below and 1.08e-3 above the start (`method = gn`), all others within
/// 1e-5. 9× the largest, and ~10× what `ftol = 1e-6` allows near OFV 1000.
const START_PROBE_TOL: f64 = 1e-2;

/// The start probe's verdict (#2506) on a fit that never left its start:
///
/// - `Some(false)` — the probe ended more than [`START_PROBE_TOL`] below: a stall;
/// - `Some(true)` — the probe **converged** to within the tolerance: a warm start at an
///   optimum;
/// - `None` — anything else: no evidence, the fit stays a stall.
///
/// The question is asked of the optimizer, not of a finite-difference certificate: bounds,
/// flat directions and saddles are the optimizer's to handle, and it handles them on every fit.
fn start_probe_verdict(start_ofv: f64, probe_ofv: f64, probe_converged: bool) -> Option<bool> {
    if !ofv_is_valid(start_ofv) || !ofv_is_valid(probe_ofv) {
        None
    } else if probe_ofv < start_ofv - START_PROBE_TOL {
        Some(false)
    } else if probe_converged && probe_ofv <= start_ofv + START_PROBE_TOL {
        Some(true)
    } else {
        None
    }
}

/// The start probe's starting point (#2506): every free scaled coordinate moved
/// [`START_PROBE_STEP_S`], alternating in sign, turned around at a bound (or half the
/// larger room). `None` when no coordinate moved at least [`INIT_ESCAPE_STEP_S`].
fn perturbed_start(x: &[f64], fixed: &[bool], lower: &[f64], upper: &[f64]) -> Option<Vec<f64>> {
    let mut out = x.to_vec();
    let mut moved = false;
    let mut sign = 1.0;
    for k in (0..x.len()).filter(|&k| !fixed[k]) {
        let (up, down) = (upper[k] - x[k], x[k] - lower[k]);
        let room = |s: f64| if s > 0.0 { up } else { down };
        let s = if room(sign) >= START_PROBE_STEP_S { sign } else { -sign };
        if room(s) >= START_PROBE_STEP_S {
            out[k] += s * START_PROBE_STEP_S;
        } else if up >= down {
            out[k] += 0.5 * up.max(0.0);
        } else {
            out[k] -= 0.5 * down.max(0.0);
        }
        moved |= (out[k] - x[k]).abs() >= INIT_ESCAPE_STEP_S;
        sign = -sign;
    }
    moved.then_some(out)
}

/// The probe [`optimize_nlopt`] hands the first attempt (#2506): given the perturbed start,
/// the penalized objective at the unmoved start, and the evaluations already spent, it
/// runs the probe. What the probe found is acted on by [`settle_start_probe`], after the
/// attempt, not by the attempt itself.
type StartProbe<'a> = dyn Fn(&ModelParameters, f64, usize) + 'a;

/// Whose chain a restart leg is on (#2707): the fit's, which is handed the start probe and
/// announces its restart, or a start probe's own, which neither probes nor announces.
enum RestartLegOwner<'a> {
    Fit(&'a StartProbe<'a>),
    Probe { spent_before: usize },
}

/// When an NLopt attempt hands its unmoved start to the #2506 [`StartProbe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeWhen {
    /// Whenever the attempt did not leave its start: the first attempt, whose start is the
    /// user's, and may be a warm start at an optimum as well as a #751 stall.
    Unmoved,
    /// Only when not leaving its start is the one thing that keeps the attempt from being
    /// reported converged (#2707): a `Failure`-class or verified stop that its acceptance
    /// rule takes with displacement and rejects without it, with no basin shift to restart
    /// on. A #1277 / #1565 restart leg. It starts at the previous leg's estimates, so a
    /// basin-shift restart whose gain is the EBE modes a cold re-solve found, not a step in
    /// θ, does not leave *its* start, and without the probe was reported unconverged however
    /// sound its stop. A restart that is converged, or rejected on any other ground, costs
    /// no probe.
    DisplacementAloneRejects,
}

/// Does an attempt that did not leave its start hand it to the probe, under `when`?
/// `decided_here` is whether the attempt's `converged` is decided by its own acceptance rule
/// (a `Failure`-class, latched or verified stop), `accepted(left)` that rule as a function
/// of displacement, and `basin_shift` whether the cold re-solve beat the best seen (#1565),
/// which restarts the leg instead. Pure, so each condition of the restart gate (#2707) is
/// testable without a fit.
fn start_probe_due(
    when: ProbeWhen, decided_here: bool, accepted: impl Fn(bool) -> bool, basin_shift: bool,
) -> bool {
    match when {
        ProbeWhen::Unmoved => true,
        ProbeWhen::DisplacementAloneRejects => {
            decided_here && !accepted(false) && accepted(true) && !basin_shift
        }
    }
}

/// What [`optimize_nlopt`] carries out of its first attempt once the start probe (#2506)
/// answered: the attempt to report, its restart cause, and whether to skip the #751 stall
/// retry ([`resolve_stall_retry`]).
///
/// - `Some(false)` — the first attempt, a stall, with [`start_probe_stall_warning`]; retried.
/// - `Some(true)` — [`verified_start_result`]; no retry.
/// - `None` — the first attempt, retried exactly when it did not leave its start.
///
/// The probe's evaluations count in `n_iterations` either way.
type ProbedAttempt = (OuterResult, Option<RestartCause>, Option<bool>);
fn settle_start_probe(
    first: (OuterResult, Option<RestartCause>), first_left_init: bool,
    probed: Option<ProbedAttempt>, penalized: impl Fn(&OuterResult) -> f64,
) -> ((OuterResult, Option<RestartCause>), bool) {
    let (mut first, first_cause) = first;
    match probed {
        Some((found, _, Some(false))) => {
            first.warnings.push(start_probe_stall_warning(
                first.ofv,
                first.n_iterations,
                "evaluations",
                found.ofv,
            ));
            first.n_iterations += found.n_iterations;
            ((first, first_cause), first_left_init)
        }
        Some((found, found_cause, Some(true))) => {
            let start_wins = penalized(&first) <= penalized(&found);
            // A start that wins is the user's own, never left.
            let left = found.left_init != Some(false) && !start_wins;
            ((verified_start_result(found, left, first, start_wins), found_cause), true)
        }
        Some((spent, _, None)) => {
            first.n_iterations += spent.n_iterations;
            ((first, first_cause), first_left_init)
        }
        None => ((first, first_cause), first_left_init),
    }
}

/// What a restart leg reports once its start probe (#2707, [`ProbeWhen::DisplacementAloneRejects`])
/// has answered, with the restart cause to carry on and whether the reported point is the
/// probe's (for the restart's warning, which names where the estimates come from).
///
/// - `Some(true)` — the probe's result, through [`verified_start_result`] as for the first
///   attempt ([`settle_start_probe`]): the restart's point is an optimum an independent run
///   came back to, and the probe's own verdict, warnings and restart cause stand, at the
///   restart's point when that is at or below the probe's. One difference: the published
///   `left_init` is the reported point's own, measured from the user's start, and
///   `stationary_start` its negation, whichever point wins.
/// - `Some(false)` — the probe's result as it is, verdict, warnings and displacement its own:
///   the probe ended more than [`START_PROBE_TOL`] below the restart, so the restart was not
///   finished and the probe's is the better fit. Unlike the first attempt's (#2506), this is
///   no stall to keep flagged: the restart's start is the previous leg's estimates, not the
///   user's, and a fit does not publish a point worse than one it already reached.
/// - `None` — the restart, unconverged as its own rule left it. A probe the budget cut before
///   its verdict hands on the spent budget and its advice.
///
/// Every probe's evaluations count in `n_iterations`, whichever result is kept. No probe ran
/// (`probed` is `None`) unless the restart did not leave its start and displacement was all
/// its acceptance rule rejected it on. Returns the result, the restart cause, whether the
/// reported point is the probe's, and whether the probe verified the point (`Some(true)`),
/// which lets a mid-descent restart that ties its stalled leg be adopted.
fn settle_restart_probe(
    restarted: (OuterResult, Option<RestartCause>), probed: Option<ProbedAttempt>,
    penalized: impl Fn(&OuterResult) -> f64,
) -> (OuterResult, Option<RestartCause>, bool, bool) {
    let (mut restarted, cause) = restarted;
    match probed {
        Some((found, found_cause, Some(true))) => {
            let start_wins = penalized(&restarted) <= penalized(&found);
            // The displacement published is the reported point's own, from the user's start
            // (both legs measure it against `escape_from`): a restart's start is the previous
            // leg's estimates, so a restart whose point wins is not, as on the first attempt,
            // a start the fit never left.
            let left = if start_wins { &restarted } else { &found }.left_init != Some(false);
            (
                verified_start_result(found, left, restarted, start_wins),
                found_cause,
                !start_wins,
                true,
            )
        }
        Some((mut found, found_cause, Some(false))) => {
            found.start_eval = restarted.start_eval;
            found.n_iterations += restarted.n_iterations;
            (found, found_cause, true, false)
        }
        Some((spent, _, None)) => {
            restarted.n_iterations += spent.n_iterations;
            // A probe cut by the budget before its verdict spent the rest of this fit's: the
            // restart's own leg stopped short of it, so the spent budget and the advice the
            // probe was given are the fit's (#2821 review r1).
            if spent.eval_budget_spent && !restarted.eval_budget_spent {
                restarted.eval_budget_spent = true;
                restarted.warnings.extend(
                    spent.warnings.into_iter().filter(|w| w.starts_with(EVAL_BUDGET_WARNING_HEAD)),
                );
            }
            (restarted, cause, false, false)
        }
        None => (restarted, cause, false, false),
    }
}

/// The opening of [`eval_budget_warning`], which a result's warnings are matched on.
const EVAL_BUDGET_WARNING_HEAD: &str = "Outer optimization hit the evaluation budget";

/// The warning an NLopt run that spent its evaluation budget unconverged carries.
fn eval_budget_warning(outer_maxiter: usize) -> String {
    format!(
        "{EVAL_BUDGET_WARNING_HEAD} (maxiter = {outer_maxiter}) before converging; increase \
         maxiter for a tighter fit."
    )
}

/// The result reported for a start the probe verified (#2506): the probe's `converged`
/// verdict and warnings, as the probe produced them, at the **lower** of the two points.
/// `start_wins` says the unmoved run's penalized objective is at or below the probe's;
/// the probe stops on its own `ftol`, which on an OFV near 1000 lets it end ~1e-3 above
/// a start that was already the optimum (measured on `two_cpt_iv`, Linux), and a fit
/// must not publish a point worse than one it already evaluated. When the start wins,
/// everything tied to the point — estimates, objective, EBEs, covariance, final gradient
/// — is the unmoved run's. `left` is whether the **reported** estimates are off the user's
/// start by [`INIT_ESCAPE_STEP_S`], decided by the caller for the point that wins: on the
/// first attempt the start that wins is the user's own, so never; on a restart leg it is the
/// previous leg's estimates, so the restart's own displacement ([`settle_restart_probe`]).
/// Within it the result is published `left_init = Some(false)` with
/// `stationary_start = true`, which `model_selection::stalled_at_init` reads as a warm
/// start rather than a stall. The
/// unmoved run's evaluations count in `n_iterations`, and its `start_eval` — the fit's
/// first evaluation, at the user's start — is the one published (#2461).
fn verified_start_result(
    mut probe: OuterResult, left: bool, unmoved: OuterResult, start_wins: bool,
) -> OuterResult {
    probe.start_eval = unmoved.start_eval;
    probe.left_init = Some(left);
    probe.stationary_start = !left;
    probe.n_iterations += unmoved.n_iterations;
    // The covariance quartet goes with `packed_estimate` (#816). The covariance step's warnings
    // cannot be told apart from the probe's others, so off NLopt they stay the probe's (a point
    // within `START_PROBE_TOL`); on NLopt the step is deferred to the reported point. The
    // probe's restart cause stays (adopted only when lower).
    if start_wins {
        probe.params = unmoved.params;
        probe.ofv = unmoved.ofv;
        probe.eta_hats = unmoved.eta_hats;
        probe.h_matrices = unmoved.h_matrices;
        probe.kappas = unmoved.kappas;
        probe.covariance_matrix = unmoved.covariance_matrix;
        probe.covariance_method = unmoved.covariance_method;
        probe.covariance_wall_time_secs = unmoved.covariance_wall_time_secs;
        probe.sir_fallback_proposal = unmoved.sir_fallback_proposal;
        probe.final_gradient = unmoved.final_gradient;
        probe.final_gradient_source = unmoved.final_gradient_source;
        probe.cond_dist = unmoved.cond_dist;
        probe.packed_estimate = unmoved.packed_estimate;
        probe.mixture_posteriors = unmoved.mixture_posteriors;
    }
    probe
}

/// The `FitResult` warning a fit carries when it stalled on its initial estimates and the
/// start probe reached a lower objective (#2506). The reported estimates stay the stalled
/// ones, so the stall is still flagged `W_STALLED_AT_INIT`; this says how much lower an
/// objective was within reach. Both objectives are the clean −2LL.
fn start_probe_stall_warning(
    stalled_ofv: f64, stalled_count: usize, unit: &str, probe_ofv: f64,
) -> String {
    format!(
        "Outer optimization stalled on its initial estimates (OFV = {stalled_ofv:.6} after \
         {stalled_count} {unit}): a restart from a perturbation of them reached OFV = \
         {probe_ofv:.6}, so the reported estimates are not an optimum. Refit from \
         different initial estimates."
    )
}

/// The #2506 start probe for non-NLopt outer optimizers — built-in BFGS / L-BFGS,
/// `trust_region`, `gn` / `gn_hybrid`, and the profiled loop (#2591) — which publish no
/// `left_init` of their own.
///
/// Movement is measured on the free coordinates in the rerun optimizer's scaled space
/// (`scale_of`: [`bfgs_scale`], `gauss_newton::gn_scale`, unit for trust region). When the
/// run did not move by [`INIT_ESCAPE_STEP_S`], `rerun` is called once from
/// [`perturbed_start`] with `outer_maxiter = budget`, and [`start_probe_verdict`] decides:
/// `Some(true)` → [`verified_start_result`]; `Some(false)` → `result` with
/// `left_init = Some(false)` and [`start_probe_stall_warning`]; `None` → `result` unchanged.
/// The probe's iterations count either way. Nothing runs when the fit moved, was
/// cancelled, has no valid objective, or `budget` is 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_unmoved_start(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, result: OuterResult, budget: usize,
    scale_of: impl FnOnce(&[f64], &PackedBounds) -> Vec<f64>,
    rerun: impl FnOnce(&ModelParameters, &FitOptions) -> OuterResult,
) -> OuterResult {
    let PackedStart { packed: mut x0, bounds, fixed, .. } = pack_with_bounds(init_params);
    clamp_to_bounds(&mut x0, &bounds);
    let scale = scale_of(&x0, &bounds);
    let x0_s = apply_scale(&x0, &scale);
    // Only the free coordinates: a held one is not the optimizer's to move, and on a
    // profiled fit (#2591) the held level coordinates are re-solved at every evaluation,
    // so they leave the start whether or not the structural ones did.
    let free_of = |x: &[f64]| -> Vec<f64> {
        x.iter().zip(&fixed).filter(|&(_, &f)| !f).map(|(&v, _)| v).collect()
    };
    let x0_free = free_of(&x0_s);
    let off_start = |r: &OuterResult| {
        !(max_scaled_deviation(&free_of(&apply_scale(&pack_params(&r.params), &scale)), &x0_free)
            < INIT_ESCAPE_STEP_S)
    };
    // `!(… < …)` so a non-finite deviation counts as moved: nothing to probe.
    if off_start(&result)
        || budget == 0
        || crate::cancel::is_cancelled(&options.cancel)
        || !ofv_is_valid(result.ofv)
    {
        return result;
    }
    let (lower_s, upper_s) =
        (apply_scale(&bounds.lower, &scale), apply_scale(&bounds.upper, &scale));
    let Some(xp_s) = perturbed_start(&x0_s, &fixed, &lower_s, &upper_s) else {
        return result;
    };
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    let priors = build_prior_set(model, init_params);
    let penalized = |r: &OuterResult| {
        r.ofv + nn_reg.penalty_value(&r.params.theta) + priors.penalty(&pack_params(&r.params))
    };
    let probe_options = FitOptions { outer_maxiter: budget, ..options.clone() };
    let probe = rerun(&unpack_params(&remove_scale(&xp_s, &scale), init_params), &probe_options);
    let (start_ofv, probe_ofv) = (penalized(&result), penalized(&probe));
    let verdict = start_probe_verdict(start_ofv, probe_ofv, probe.converged);
    if options.verbose {
        eprintln!(
            "Start probe: the fit did not leave its start (OFV = {start_ofv:.6}); a restart \
             from a perturbation of it reached OFV = {probe_ofv:.6} (converged = {}) → \
             stationary start = {verdict:?}",
            probe.converged,
        );
    }
    match verdict {
        Some(false) => {
            let mut stalled = result;
            stalled.warnings.push(start_probe_stall_warning(
                stalled.ofv,
                stalled.n_iterations,
                "iterations",
                probe.ofv,
            ));
            stalled.left_init = Some(false);
            stalled.n_iterations += probe.n_iterations;
            stalled
        }
        Some(true) => {
            let start_wins = start_ofv <= probe_ofv;
            let left = off_start(&probe) && !start_wins;
            verified_start_result(probe, left, result, start_wins)
        }
        None => OuterResult { n_iterations: result.n_iterations + probe.n_iterations, ..result },
    }
}

/// The trace half of [`failure_is_converged_plateau`], **progress** and **plateau**: what
/// can be read before the final inner solve. The #2462 verification leg reads it to decide
/// whether a bare `Failure` is a stop the plateau rule would accept.
fn plateau_trace_shape(feasible_evals: usize, last_sig_feasible_eval: usize) -> bool {
    let made_progress = last_sig_feasible_eval >= 2;
    let flat_tail = feasible_evals.saturating_sub(last_sig_feasible_eval);
    made_progress && flat_tail >= PLATEAU_MIN_FLAT_EVALS
}

/// The **consistency** half of [`failure_is_converged_plateau`]: the cold re-solve is not
/// materially worse than the best-seen objective. Also the verdict on a stop the #2462
/// leg verified, whose plateau the leg's gain test replaces.
fn cold_solve_is_consistent(best_seen_ofv: Option<f64>, final_ofv: f64) -> bool {
    best_seen_ofv
        .is_none_or(|best| final_ofv <= best + PLATEAU_CONSISTENCY_REL_TOL * (1.0 + best.abs()))
}

/// Per-subject outer gradients `∂nllᵢ/∂x` (packed) at `x` of the objective the fit
/// minimises (FOCE, FOCEI, quadrature, or a mixture's log-sum-exp), from the optimizer's
/// exact entry points, for [`bhhh_newton_decrement`] (#1934). `None` when any subject has
/// no analytic gradient, so the decrement never mixes exact and salvaged rows. Production
/// reads [`per_subject_outer_gradients_checked`]; this unchecked form is the tests'.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn per_subject_outer_gradients(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters, x: &[f64],
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], options: &FitOptions,
) -> Option<Vec<Vec<f64>>> {
    per_subject_outer_gradients_checked(
        model,
        population,
        init_params,
        x,
        ehs,
        kappas,
        options,
        false,
        None,
    )
}

/// Did one subject's integrations solve the trajectory the decrement assumes (#2143 review)?
/// Not when a segment stopped short and kept its padded tail (a stop the driver discarded
/// and re-solved does not count, as in the post-fit `kept_unfinished_segments`), diverged, or
/// — on a model routed to the exact path — was declined by the matrix exponential and
/// integrated instead, at the fit's tolerance and outside the tight re-solve.
fn integration_backs_a_decrement(st: &crate::ode::solver::OdeSolverStats) -> bool {
    let kept_unfinished = st.unfinished_segments.saturating_sub(st.discarded_unfinished_segments);
    kept_unfinished == 0 && st.diverged_segments == 0 && st.linear_declined_segments == 0
}

/// What a mixture's per-subject gradients start from: the incumbent's class modes
/// (`[class][subject]`, solved at this very `x`) and the fit's schedule cache.
#[derive(Clone, Copy)]
struct MixtureStart<'a> {
    class_etas: &'a [Vec<DVector<f64>>],
    schedules: Option<&'a crate::estimation::mixture::ScheduleCache>,
}

/// [`per_subject_outer_gradients`], and with `require_complete` also `None` when any
/// subject's integration stopped short of its horizon (#2143): the driver pads an exhausted
/// segment with its last state (#603), giving a finite gradient of a trajectory never
/// integrated. Counted per worker ([`crate::ode::solver::SolverStatsScope`] is per thread).
#[allow(clippy::too_many_arguments)]
fn per_subject_outer_gradients_checked(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters, x: &[f64],
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], options: &FitOptions,
    require_complete: bool, mixture_start: Option<MixtureStart<'_>>,
) -> Option<Vec<Vec<f64>>> {
    if !crate::sens::provider::analytic_outer_gradient_available(model) {
        return None;
    }
    // One subject's row, with its integrations counted on its own worker.
    let checked = |row: &dyn Fn() -> Option<Vec<f64>>| {
        let scope = require_complete.then(crate::ode::solver::SolverStatsScope::enter);
        let gradient = row();
        if scope.is_some_and(|s| !integration_backs_a_decrement(&s.collected())) {
            return None;
        }
        gradient
    };
    if init_params.mixture.is_some() {
        // A mixture's rows take every class's mode, which `ehs` (the MIXEST class's) does
        // not carry: one evaluation at `x` solves them, warm from the incumbent's class modes
        // when it has them (`mixture_start`), else cold.
        let params = unpack_params(x, init_params);
        let (eval, modes) = match mixture_start {
            Some(start) => crate::estimation::mixture::mixture_ofv_cached(
                model,
                population,
                &params,
                &FitOptions { inner_restarts: 0, ..options.clone() },
                Some(start.class_etas),
                start.schedules,
            ),
            None => crate::estimation::mixture::mixture_ofv_with_modes(
                model, population, &params, options, None,
            ),
        };
        if eval.ebe_stats.n_unconverged > 0 || eval.ebe_stats.n_start_rejected > 0 {
            return None;
        }
        return crate::estimation::mixture::mixture_subject_gradients(
            model,
            population,
            &params,
            options,
            &eval,
            Some(&modes),
            None,
            checked,
        );
    }
    let objective = crate::estimation::covariance::cov_objective(options);
    // One unpack for the population: every subject is at the same packed point.
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(init_params, x);
    crate::par::map(population.subjects.len(), |i| {
        let subject = &population.subjects[i];
        checked(&|| {
            // The stacked `[η, κ₁ … κ_K]` (just `η` without IOV).
            let mut mode: Vec<f64> = ehs[i].iter().copied().collect();
            for kap in kappas.get(i).map(Vec::as_slice).unwrap_or(&[]) {
                mode.extend(kap.iter().copied());
            }
            let g = crate::estimation::sens_outer_gradient::subject_objective_gradient_at(
                model,
                subject,
                pt.as_ref()?,
                &mode,
                objective,
            )?;
            g.iter().all(|v| v.is_finite()).then_some(g)
        })
    })
    .into_iter()
    .collect()
}

/// The BHHH Newton decrement at a point (#1934, #1564): the OFV a Newton step would still
/// gain, from the per-subject gradients alone.
///
/// With `gᵢ = ∂nllᵢ/∂x`, `s = Σᵢ gᵢ`, `S = Σᵢ gᵢgᵢᵀ`: `∇²OFV ≈ 2S`, `∇OFV = 2s`, so the
/// quadratic decrease is `sᵀS⁻¹s` — in OFV units and invariant to linear rescaling of `x`,
/// unlike `‖g‖` (which cannot tell a flat optimum, #1530, from a slow creep, #1934).
///
/// `penalty_grad` / `penalty_hess` (OFV units, packed; priors and the NN regularizer, else
/// zeros) enter halved on the `nll` scale. Only movable coordinates are kept: not `fixed`,
/// and not on a bound with `−s` pointing through it (#1564's `pk1_sparse` read 0.30
/// otherwise). Eigenvalues below `1e-10·λ_max` are dropped. `None` when nothing is free or
/// a component is not finite.
pub(crate) fn bhhh_newton_decrement(
    per_subject: &[Vec<f64>], penalty_grad: &[f64], penalty_hess: &DMatrix<f64>, x: &[f64],
    bounds: &PackedBounds, fixed: &[bool],
) -> Option<f64> {
    bhhh_newton_step(per_subject, penalty_grad, penalty_hess, x, bounds, fixed).map(|(d, _)| d)
}

/// [`bhhh_newton_decrement`] with the step it predicts from: `−S⁻¹s` in packed coordinates,
/// on the eigenvalues the decrement keeps, and zero on every coordinate it does not free.
fn bhhh_newton_step(
    per_subject: &[Vec<f64>], penalty_grad: &[f64], penalty_hess: &DMatrix<f64>, x: &[f64],
    bounds: &PackedBounds, fixed: &[bool],
) -> Option<(f64, Vec<f64>)> {
    let n = x.len();
    let s_full: Vec<f64> = (0..n)
        .map(|k| per_subject.iter().map(|g| g[k]).sum::<f64>() + 0.5 * penalty_grad[k])
        .collect();
    // An infinite bound (an unbounded θ, packed as ±∞ — also `ln(∞)` under log packing)
    // binds nothing: without the `is_finite` the test reads `∞ <= ∞` and drops the
    // coordinate whenever its gradient points toward that side.
    let at = |b: f64, xk: f64| b.is_finite() && (xk - b).abs() <= 1e-8 * (1.0 + b.abs());
    let free: Vec<usize> = (0..n)
        .filter(|&k| {
            !fixed[k]
                && !(at(bounds.lower[k], x[k]) && s_full[k] > 0.0)
                && !(at(bounds.upper[k], x[k]) && s_full[k] < 0.0)
        })
        .collect();
    if free.is_empty() {
        return None;
    }
    let m = free.len();
    let s = DVector::from_iterator(m, free.iter().map(|&k| s_full[k]));
    let mut curvature = DMatrix::from_fn(m, m, |a, b| 0.5 * penalty_hess[(free[a], free[b])]);
    for g in per_subject {
        let gf = DVector::from_iterator(m, free.iter().map(|&k| g[k]));
        curvature.ger(1.0, &gf, &gf, 1.0);
    }
    if s.iter().chain(curvature.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    let eig = nalgebra::SymmetricEigen::new(curvature);
    let lambda_max = eig.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    if lambda_max <= 0.0 {
        return None;
    }
    let mut decrement = 0.0;
    let mut step_free = DVector::zeros(m);
    for (j, &lambda) in eig.eigenvalues.iter().enumerate() {
        if lambda > 1e-10 * lambda_max {
            let along = eig.eigenvectors.column(j).dot(&s);
            decrement += along * along / lambda;
            step_free -= eig.eigenvectors.column(j) * (along / lambda);
        }
    }
    let mut step = vec![0.0; n];
    for (a, &k) in free.iter().enumerate() {
        step[k] = step_free[a];
    }
    decrement.is_finite().then_some((decrement, step))
}

/// Largest [`bhhh_newton_decrement`] at which the stagnation guard may still latch
/// (#1934). Above it, a window without a 1e-3 improvement is read as a slow creep, not a
/// plateau, and the window is re-armed instead.
///
/// Measured at the would-be latch of every bundled closed-form example whose default fit
/// latches (33 of them, Linux x86_64 at `c25832ca`): the largest is 2.6e-4
/// (`warfarin_logit_f`), and all but three read below 1e-6. The #1934 creep reads 14.8
/// at its eval-60 latch, and its optimum 1.8e-9. `0.1` sits ~400× above the first and
/// ~150× below the second, and a fit predicted to gain less than a tenth of an OFV unit
/// has nothing left in it a user could act on.
pub(crate) const LATCH_MAX_PREDICTED_GAIN: f64 = 0.1;

/// The `ode_reltol` an integrated model's decrement is formed at (#2001); a fit already
/// tighter keeps its own. Not a threshold: the decrement is compared against
/// [`LATCH_MAX_PREDICTED_GAIN`] as on a closed form, and this only has to put the
/// integrator's contribution far below it. Measured on Linux x86_64 against the exact
/// (matrix-exponential) value of the same linear models: within 1.3e-5 (rk45) and 1.1e-4
/// (rosenbrock23) on `two_cpt_iv_ode`, whose default-tolerance decrement at its optimum is
/// 1.85 against an exact 0.008; and 5.7e-7 against the closed form's 5.8e-7 at NONMEM's
/// optimum of the #1934 model (the default tolerance reads up to 1.7e-4 there).
const DECREMENT_ODE_RELTOL: f64 = 1e-9;

/// How much the step budget grows for the tight re-solve. Tightening the default
/// `ode_reltol` 1e-5× costs an order-`p` stepper ~(1e5)^(1/p) more steps: ~10× for rk45,
/// ~46× for rosenbrock23, whose step control runs on an O(h³) local error. 100 covers both
/// with 2× headroom; it is a cap, paid only by a solve that needs it.
const DECREMENT_ODE_MAX_STEPS_FACTOR: usize = 100;

/// The warning for a stagnation latch that overrode the #1934 veto: a Newton step from
/// the best point still predicted `gain`, but a full re-armed window did not realise any
/// of it.
fn veto_override_warning(gain: Option<f64>) -> String {
    format!(
        "Stagnation guard stopped the outer optimizer although a Newton step from the best \
         point was still predicted to lower the objective by {:.4} (BHHH Newton decrement, \
         max {LATCH_MAX_PREDICTED_GAIN}); a full further window did not realise it. The \
         estimates may not be at a minimum — check the final gradient.",
        gain.unwrap_or(f64::NAN)
    )
}

/// Is a predicted remaining gain small enough for the guard to latch? `None` — no
/// decrement could be formed (see [`predicted_remaining_gain`]) — latches as before #1934.
fn gain_is_negligible(predicted_gain: Option<f64>) -> bool {
    predicted_gain.is_none_or(|gain| gain <= LATCH_MAX_PREDICTED_GAIN)
}

/// The ODE solver override [`predicted_remaining_gain`] re-solves an integrated model under
/// (#2001), or `None` when nothing integrates (closed form without an absorption twin, or an
/// exact matrix-exponential ODE, #852). `reltol` drops to [`DECREMENT_ODE_RELTOL`] with
/// `abstol` scaled by the same factor; the step budget grows by
/// [`DECREMENT_ODE_MAX_STEPS_FACTOR`]; other fields are the fit's own override (read on the
/// objective closure's thread, where it is armed).
fn decrement_ode_override(
    model: &CompiledModel, options: &FitOptions,
) -> Option<crate::ode::solver::OdeSolverOverride> {
    let eff = model.integrator_spec()?.effective_solver_opts();
    if eff.method == crate::ode::OdeMethod::Linear {
        return None;
    }
    let reltol = eff.reltol.min(DECREMENT_ODE_RELTOL);
    Some(crate::ode::solver::OdeSolverOverride {
        reltol: Some(reltol),
        abstol: Some(eff.abstol * (reltol / eff.reltol)),
        max_steps: Some(eff.max_steps.saturating_mul(DECREMENT_ODE_MAX_STEPS_FACTOR)),
        ..options.ode_solver_override()
    })
}

/// The incumbent's per-subject EBEs as inner-loop warm starts: `η̂`, then each occasion
/// cell's `κ̂` (the layout `EbeResult` reports and `find_ebe_iov` accepts whole), so the
/// tight re-solve starts at the incumbent's full conditional mode rather than at κ = 0
/// (#2001 review). A subject with no κ cells is its `η̂` alone.
pub(crate) fn joint_warm_starts(
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>],
) -> Vec<DVector<f64>> {
    ehs.iter()
        .enumerate()
        .map(|(i, eta)| joint_warm_start(eta, kappas.get(i).map(Vec::as_slice).unwrap_or(&[])))
        .collect()
}

/// The warm starts the FOCE/FOCEI outer caches hand the next evaluation:
/// [`joint_warm_starts`] (`η̂` then every `κ̂`, #2261), except on ODE + IOV, where they are
/// `η̂` alone and κ restarts at 0 (#2508): there the BFGS partial depends on its start on the
/// RK45 noise floor, so the objective became path-dependent (busulfan: OFV moved 2–4e-3 on
/// re-evaluation, 118 evaluations to the latch instead of 63).
pub(super) fn outer_cache_warm_starts(
    model: &CompiledModel, ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>],
) -> Vec<DVector<f64>> {
    if crate::estimation::inner_optimizer::is_ode_iov(model) {
        ehs.to_vec()
    } else {
        joint_warm_starts(ehs, kappas)
    }
}

/// One subject's [`joint_warm_starts`]: `eta`, then each of its κ `cells` in order.
pub(crate) fn joint_warm_start(eta: &DVector<f64>, cells: &[DVector<f64>]) -> DVector<f64> {
    DVector::from_iterator(
        eta.len() + cells.iter().map(|c| c.len()).sum::<usize>(),
        eta.iter().chain(cells.iter().flat_map(|c| c.iter())).copied(),
    )
}

/// [`bhhh_newton_decrement`] of the penalized objective at packed `x` with EBEs `ehs` /
/// `kappas`, or `None` where it cannot separate a creep from an optimum.
///
/// On an integrated model ([`decrement_ode_override`]) the default-tolerance gradient swamps
/// the signal (`two_cpt_iv_ode` 1.85 vs exact 0.008, #2001), so both EBEs and per-subject
/// gradients are re-solved at a tight tolerance, warm from `ehs` / `kappas`
/// ([`joint_warm_starts`]). Both, since a loose η̂ is not stationary for the tight objective
/// (2.14 vs 5.7e-7 on the #1934 model, rosenbrock23). A pass that cannot be formed is
/// `None`, which latches as before #1934.
#[allow(clippy::too_many_arguments)]
fn predicted_remaining_gain(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters, x: &[f64],
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds,
    options: &FitOptions, nn_reg: &crate::estimation::nn_reg::NnRegularizer,
    priors: &crate::estimation::priors::PriorSet,
) -> Option<f64> {
    predicted_newton_step(
        model,
        population,
        init_params,
        x,
        ehs,
        kappas,
        bounds,
        options,
        nn_reg,
        priors,
    )
    .map(|(gain, _)| gain)
}

/// [`predicted_remaining_gain`] with the step behind it ([`bhhh_newton_step`]).
#[allow(clippy::too_many_arguments)]
fn predicted_newton_step(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters, x: &[f64],
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds,
    options: &FitOptions, nn_reg: &crate::estimation::nn_reg::NnRegularizer,
    priors: &crate::estimation::priors::PriorSet,
) -> Option<(f64, Vec<f64>)> {
    if !crate::sens::provider::analytic_outer_gradient_available(model) {
        return None;
    }
    let per_subject = match decrement_ode_override(model, options) {
        None => {
            // A model routed to the exact path is checked too: a segment the matrix
            // exponential declines is integrated at the fit's tolerance (#2143 review).
            let checked = model.honors_ode_solver_opts();
            per_subject_outer_gradients_checked(
                model,
                population,
                init_params,
                x,
                ehs,
                kappas,
                options,
                checked,
                None,
            )?
        }
        Some(ode) => {
            #[cfg(test)]
            TIGHT_RESOLVES.with(|n| n.set(n.get() + 1));
            tight_per_subject_gradients(
                model,
                population,
                init_params,
                x,
                ehs,
                kappas,
                options,
                ode,
            )?
        }
    };
    let n = x.len();
    let theta = unpack_params(x, init_params).theta;
    let mut penalty_grad = vec![0.0; n];
    let mut penalty_hess = DMatrix::zeros(n, n);
    nn_reg.add_packed_gradient(&theta, &mut penalty_grad);
    nn_reg.add_packed_hessian(&theta, &mut |i, j, v| penalty_hess[(i, j)] += v);
    priors.add_gradient(x, &mut penalty_grad);
    priors.add_hessian(&mut |i, j, v| penalty_hess[(i, j)] += v);
    bhhh_newton_step(
        &per_subject,
        &penalty_grad,
        &penalty_hess,
        x,
        bounds,
        &packed_fixed_mask(init_params),
    )
}

/// [`predicted_remaining_gain`] for a quadrature objective (Laplace, AGQ; #2549): the BHHH
/// decrement from each subject's exact quadrature score (`SubjectScoreContext::score_routed`)
/// plus penalties; decides whether a quadrature tolerance stop takes the polish legs.
/// `scores`: the incumbent gradient's rows when available, else formed here.
#[allow(clippy::too_many_arguments)]
fn quadrature_predicted_remaining_gain(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters, x: &[f64],
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], scores: Option<&[Vec<f64>]>,
    class_etas: &[Vec<DVector<f64>>], schedules: &crate::estimation::mixture::ScheduleCache,
    bounds: &PackedBounds, options: &FitOptions, nn_reg: &crate::estimation::nn_reg::NnRegularizer,
    priors: &crate::estimation::priors::PriorSet,
) -> Option<f64> {
    let formed: Vec<Vec<f64>>;
    // A mixture's rows are its class log-sum-exp gradients at every class's own mode, not the
    // single-population score at the MIXEST class's `ehs` (which, read as one, predicts a
    // phantom gain at the mixture's own optimum, #3025).
    let mixture_rows;
    let per_subject: &[Vec<f64>] = if init_params.mixture.is_some() {
        mixture_rows = per_subject_outer_gradients_checked(
            model,
            population,
            init_params,
            x,
            ehs,
            kappas,
            options,
            false,
            (!class_etas.is_empty())
                .then_some(MixtureStart { class_etas, schedules: Some(schedules) }),
        )?;
        &mixture_rows
    } else {
        match scores.filter(|s| s.len() == population.subjects.len()) {
            Some(s) => s,
            None => {
                let context = crate::estimation::agq::SubjectScoreContext::new(
                    model,
                    init_params,
                    x,
                    options,
                    bounds,
                    false,
                );
                let subjects = &population.subjects;
                formed = crate::par::map(subjects.len(), |i| {
                    let k = kappas.get(i).map(Vec::as_slice).unwrap_or(&[]);
                    context
                        .score_routed(&subjects[i], ehs.get(i)?.as_slice(), k, false)
                        .map(|(g, _)| g)
                })
                .into_iter()
                .collect::<Option<_>>()?;
                &formed
            }
        }
    };
    let n = x.len();
    let theta = unpack_params(x, init_params).theta;
    let mut penalty_grad = vec![0.0; n];
    let mut penalty_hess = DMatrix::zeros(n, n);
    nn_reg.add_packed_gradient(&theta, &mut penalty_grad);
    nn_reg.add_packed_hessian(&theta, &mut |i, j, v| penalty_hess[(i, j)] += v);
    priors.add_gradient(x, &mut penalty_grad);
    priors.add_hessian(&mut |i, j, v| penalty_hess[(i, j)] += v);
    bhhh_newton_decrement(
        per_subject,
        &penalty_grad,
        &penalty_hess,
        x,
        bounds,
        &packed_fixed_mask(init_params),
    )
}

/// The per-subject outer gradients at `x` with the EBEs re-solved first, both under the
/// tight ODE override `ode` ([`predicted_remaining_gain`], #2001). `None` when the scope
/// cannot be built, the re-solve leaves a subject unconverged or start-rejected, or the
/// gradients cannot be formed.
#[allow(clippy::too_many_arguments)]
fn tight_per_subject_gradients(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters, x: &[f64],
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], options: &FitOptions,
    ode: crate::ode::solver::OdeSolverOverride,
) -> Option<Vec<Vec<f64>>> {
    crate::pool::with_fit_scope_ode(options, ode, || {
        if init_params.mixture.is_some() {
            // A mixture solves its class modes itself, under this scope.
            return per_subject_outer_gradients_checked(
                model,
                population,
                init_params,
                x,
                ehs,
                kappas,
                options,
                true,
                None,
            );
        }
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let warm = joint_warm_starts(ehs, kappas);
        let (ehs, _, stats, kappas) = run_inner_loop_warm_seeded(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&warm),
            Some(&mu_k),
            0,
            0,
            InnerHessianSeed::for_options(options),
            options,
            PrevEtas::Solved,
        );
        // The decrement assumes every η̂ is a mode; one that did not converge (or a
        // start the inner loop refused) still yields a finite gradient, formed as if
        // it were. No decrement then — the latch proceeds as before #1934.
        if stats.n_unconverged > 0 || stats.n_start_rejected > 0 {
            return None;
        }
        per_subject_outer_gradients_checked(
            model,
            population,
            init_params,
            x,
            &ehs,
            &kappas,
            options,
            true,
            None,
        )
    })
    .ok()?
}

#[cfg(test)]
thread_local! {
    /// The #2159 pre-flight's probe re-solves on this thread: one population inner solve
    /// each, the cost the exact rule exists to remove.
    pub(crate) static IDENTITY_PROBE_SOLVES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Test switch: decide the #2159 identity scales from the probes alone.
    pub(crate) static IDENTITY_EXACT_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// How many tight re-solves [`predicted_remaining_gain`] started on this thread, so a
    /// test can see that an out-of-scope objective skips one (#2143 review).
    static TIGHT_RESOLVES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Replaces [`outer_eval_budget`]'s result in `optimize_nlopt_attempt`, so a test can
    /// end the budget exactly where a BOBYQA leg stopped (#1782) — `outer_maxiter`
    /// moves it only in steps of `n + 1`.
    pub(crate) static EVAL_BUDGET_OVERRIDE: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
    /// The objective-evaluation count at which the first BOBYQA leg returned, last run
    /// on this thread (#1782).
    pub(crate) static FIRST_LEG_EVALS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    /// Replaces `POLISH_MIN_GAIN` in `optimize_nlopt_attempt`; `-inf` makes every polish
    /// leg count as still descending, so the run reaches the leg cap (#1782).
    pub(crate) static POLISH_MIN_GAIN_OVERRIDE: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
}

/// The best point an optimizer has seen — the single source of truth for "where the fit is",
/// for the final restore (#59: NLopt returns the *last* point) and the checkpoint (#1317: a
/// write during a line search recorded a probe at 2.76e6 on a fit plateaued at 51786).
///
/// Ranked by `ofv`, the objective minimised (penalized under NN regularization); `ofv_clean`
/// (unpenalized −2LL) is carried for the plateau check and the checkpoint. `x` is in its
/// owner's space (scaled for NLopt, packed for BFGS / Gauss-Newton), so
/// [`BestPoint::write_checkpoint`] takes the map to packed space.
#[derive(Debug, Clone, Default)]
pub(crate) struct BestPoint {
    inner: Option<BestPointInner>,
}

/// The incumbent held by a [`BestPoint`].
#[derive(Debug, Clone)]
pub(crate) struct BestPointInner {
    /// The point, in its owner's optimizer space (see [`BestPoint`]).
    pub(crate) x: Vec<f64>,
    /// Objective minimised by the optimizer at `x` (penalized under NN regularization).
    pub(crate) ofv: f64,
    /// Unpenalized −2LL at the same point.
    pub(crate) ofv_clean: f64,
    /// Iteration / eval index at which `x` was seen.
    pub(crate) iter: usize,
}

impl BestPoint {
    /// An empty tracker (nothing observed yet).
    pub(crate) const fn new() -> Self {
        Self { inner: None }
    }

    /// Record an eval, keeping it when it improves on the incumbent; returns whether adopted.
    /// The first observation is always kept (#59 needs something), a NaN `ofv` never displaces a
    /// finite incumbent, and a NaN incumbent is displaceable (`5.0 < NaN` is false).
    pub(crate) fn observe(&mut self, iter: usize, x: &[f64], ofv: f64, ofv_clean: f64) -> bool {
        if self.inner.as_ref().is_none_or(|b| ofv < b.ofv || (b.ofv.is_nan() && !ofv.is_nan())) {
            self.inner = Some(BestPointInner { x: x.to_vec(), ofv, ofv_clean, iter });
            true
        } else {
            false
        }
    }

    /// The incumbent, or `None` before the first observation.
    pub(crate) fn get(&self) -> Option<&BestPointInner> {
        self.inner.as_ref()
    }

    /// The incumbent's ranking objective, or `+∞` when nothing has been seen.
    pub(crate) fn ofv(&self) -> f64 {
        self.inner.as_ref().map(|b| b.ofv).unwrap_or(f64::INFINITY)
    }

    /// Write a checkpoint (#755) at the **best** point seen rather than at the
    /// caller's current one (#1317), mapping `x` into packed parameter space
    /// with `to_packed` (a plain copy where the owner already works there).
    /// Callers gate this on [`crate::io::checkpoint::is_due`], so the mapping
    /// allocates only when a write will actually happen.
    pub(crate) fn write_checkpoint(&self, to_packed: impl FnOnce(&[f64]) -> Vec<f64>) {
        if let Some(b) = self.get() {
            crate::io::checkpoint::maybe_write(b.iter, b.ofv_clean, &to_packed(&b.x));
        }
    }
}

/// What one NLopt attempt did, beyond the estimates it produced — the two facts
/// [`optimize_nlopt`] needs to decide whether to run another one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttemptOutcome {
    /// The restored best point is more than [`INIT_ESCAPE_STEP_S`] away from
    /// `x₀` in scaled space — the fit actually moved.
    pub(crate) left_init: bool,
    /// NLopt returned a bare `Failure`/`ForcedStop` *and*
    /// [`failure_is_converged_plateau`] rejected it: the run quit while its OFV
    /// trace was still descending, at a point that is not a minimum. See
    /// [`resolve_mid_descent_restart_adopting`].
    pub(crate) mid_descent_stall: bool,
    /// The cold re-solve at the restored point beat the objective the optimizer
    /// minimised there ([`cold_solve_beats_best_seen`]), and the run passes the same
    /// guards a mid-descent restart does (finite objective, budget left, a feasible
    /// evaluation). See [`RestartCause::BasinShift`].
    pub(crate) basin_shift: bool,
    /// The stagnation guard latched during this attempt (#1530).
    pub(crate) stagnation_latched: bool,
    /// `converged` was decided by the plateau check ([`failure_is_converged_plateau`])
    /// rather than read off NLopt's return code.
    pub(crate) plateau_checked: bool,
}

impl AttemptOutcome {
    /// Why this attempt is worth restarting from its own best point, if it is. A run
    /// that both stalled mid-descent and shows a basin shift is filed as the stall: it
    /// was not finished either way, and keeping the #1277 cause keeps that path — its
    /// strictly-lower adoption rule and its warning — exactly as it was.
    fn restart_cause(&self) -> Option<RestartCause> {
        if self.mid_descent_stall {
            Some(RestartCause::MidDescent)
        } else if self.basin_shift {
            Some(RestartCause::BasinShift)
        } else {
            None
        }
    }
}

/// Why [`resolve_mid_descent_restart_adopting`] restarts a run from its own best point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RestartCause {
    /// NLopt returned a bare `Failure` and the plateau check rejected it: the run quit
    /// while still descending (#1277).
    MidDescent,
    /// The run minimised against EBEs some subjects held in a higher individual mode
    /// than a cold solve finds (#1565). The restart's first inner solve is cold, so it
    /// optimizes the objective the fit reports rather than the trapped surrogate.
    BasinShift,
}

/// Run the NLopt outer optimizer, with two guarded second attempts for the two
/// ways a run can stop somewhere that is not a minimum.
///
/// The first attempt is the default configuration: the identity-Hessian
/// overshoot cap fires on L-BFGS's opening gradient eval only, because holding
/// it on corrupts the `(s, y)` curvature pairs of a fit that is descending
/// normally (#960 — measured at ~11 OFV units on `scaling_convergence`).
///
/// **Stall at the start** (#751). When that attempt ends with the estimates
/// still on top of the initial ones — the opening line search failed and the fit
/// never recovered (the user-ODE warfarin twin quit at eval 4, 0.03 OFV below
/// its start and 35 short of the optimum, and reported the initial estimates
/// plus their standard errors as the result) — it is re-run from the same start
/// with the cap **held on until the fit escapes** `INIT_ESCAPE_STEP_S`. A fit
/// that never moved has no curvature worth protecting, so the trade the default
/// declines is exactly the right one here. The retry is adopted only when it
/// *both* escaped the initial estimates and reached a lower OFV — a retry that
/// stalled too keeps the first attempt even if its OFV reads lower, since a
/// lower objective at a point the fit never actually reached is not an
/// improvement to report. That retry is L-BFGS-only: SLSQP is already capped on
/// every eval, and the derivative-free algorithms never take this step at all.
///
/// **Stall mid-descent** (#1277) — [`resolve_mid_descent_restart_adopting`]. A run that
/// *did* leave its start can still die with a bare `Failure` while its OFV is
/// dropping by hundreds per eval, and the existing retry, gated on `left_init`,
/// never sees it. Then the optimizer is simply restarted from the best point it
/// reached, which resets L-BFGS's `(s, y)` memory and the trust region with it.
pub(crate) fn optimize_nlopt(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, declines: &OuterFdDeclineLog, first_eval_etas: Option<&[DVector<f64>]>,
    identity_scales: &[(usize, f64)],
) -> OuterResult {
    // The attempts minimised the *penalized* objective (covariate-NN
    // regularization, parameter priors), so they are ranked on it too:
    // `result.ofv` is the clean −2LL, and comparing that alone would prefer the
    // less-regularized attempt — the opposite of what the penalty asks for. A
    // no-op (`+ 0.0`) when neither penalty is on.
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    let priors = build_prior_set(model, init_params);
    let penalized = |result: &OuterResult| {
        result.ofv
            + nn_reg.penalty_value(&result.params.theta)
            + priors.penalty(&pack_params(&result.params))
    };
    // `resolve_stall_retry` ranks on `(result, mid_descent_stall)` pairs so the
    // winning attempt carries its own mid-descent verdict into the restart
    // decision below — the first attempt's would be the wrong one to act on when
    // the held-cap retry is the result being reported.
    let attempt = |init: &ModelParameters, hold_cap_at_init: bool, probe: Option<&StartProbe>| {
        let (result, outcome) = optimize_nlopt_attempt_probed(
            model,
            population,
            init,
            options,
            hold_cap_at_init,
            declines,
            None,
            first_eval_etas,
            identity_scales,
            true,
            probe.map(|p| (p, ProbeWhen::Unmoved)),
        );
        ((result, outcome.restart_cause()), outcome.left_init)
    };
    // #2506: a first attempt that never leaves its start is probed — restarted from a
    // perturbation of that start, on what is left of the budget — to tell a warm start at
    // an optimum from a #751 stall. The probe's result is held here, so a probe that found
    // a lower objective can be reported in place of the stall. Each restart leg below hands
    // it the same way when displacement alone rejects the leg (#2707); every leg takes what
    // its own call left, so a result never reaches the wrong leg.
    let probed: std::cell::RefCell<Option<ProbedAttempt>> = std::cell::RefCell::new(None);
    // One #1277 / #1565 restart leg, from `stalled`'s estimates: run, settled against its
    // start probe when it was handed one (#2707), with the legs' evaluations summed and the
    // restart's warning pushed. `owner` says whose chain the leg is on. The fit's own leg is
    // handed the probe and announces its restart. A probe's own leg never probes, and does
    // not announce: a verified probe's warnings are reported as the fit's
    // (`verified_start_result`), and a notice that a run the user never asked for "was
    // resumed" — with that run's evaluation counts — describes nothing they fitted. Its
    // `spent_before` is what was spent before the probe began, which `stalled`'s own
    // `n_iterations` does not count; on the fit's chain every leg already counts the ones
    // before it.
    let restart_leg = |stalled: &OuterResult, cause: RestartCause, owner: RestartLegOwner| {
        let (spent_before, probe) = match owner {
            RestartLegOwner::Fit(probe) => (0, Some(probe)),
            RestartLegOwner::Probe { spent_before } => (spent_before, None),
        };
        let announce = probe.is_some();
        // `escape_from = init_params`, not the point this leg starts at. The
        // restart begins at the stalled estimates, so measuring the published
        // escape verdict against *its own* start would report a fit that
        // recovered thousands of OFV units as having never moved whenever the
        // restart's own step is small — and `model_selection::stalled_at_init`
        // prefers that flag over its `theta_init` comparison, so the recovered
        // fit would carry `W_STALLED_AT_INIT` and be thrown out by the default
        // `Strictness::reject_init_stall`. The question the published flag
        // answers is "are the reported estimates still on top of the values the
        // user supplied", and that reference does not move when the optimizer
        // is restarted (#1277 review).
        //
        // The leg's own stop rules keep measuring displacement from its own start, so
        // one that cannot leave the stalled point is still caught. Where that is all
        // that rejects it, the #2506 probe settles it instead (#2707): a basin-shift
        // restart's gain is mostly the EBE modes the cold re-solve found, not a θ step.
        let mut leg = RestartLeg::continuing(init_params, stalled);
        leg.spent_evals += spent_before;
        let (restarted, restarted_outcome) = optimize_nlopt_attempt_probed(
            model,
            population,
            &stalled.params,
            options,
            false,
            declines,
            Some(leg),
            None,
            identity_scales,
            true,
            probe.map(|p| (p, ProbeWhen::DisplacementAloneRejects)),
        );
        let (mut restarted, restarted_cause, from_probe, verified) = settle_restart_probe(
            (restarted, restarted_outcome.restart_cause()),
            probe.and_then(|_| probed.take()),
            &penalized,
        );
        // `n_iterations` reports the evaluations the fit spent, and the
        // restart is a continuation of the same fit rather than a fresh one,
        // so the two legs add up — and the second leg ran on what the first
        // left of the budget (`RestartLeg::spent_evals`), so the sum is
        // bounded by `outer_maxiter` like any other fit's.
        restarted.n_iterations += stalled.n_iterations;
        // Pushed on the restart's own result, so it reaches `FitResult` only
        // if the restart is adopted — a rejected restart is dropped whole.
        // `classify_warning` files both under `optimizer_health` on the phrase
        // "resumed from the best point seen".
        if announce {
            restarted.warnings.push(match cause {
                RestartCause::MidDescent => mid_descent_restart_warning(
                    options.optimizer,
                    stalled.ofv,
                    stalled.n_iterations,
                    from_probe,
                ),
                RestartCause::BasinShift => {
                    basin_shift_restart_warning(stalled.ofv, stalled.n_iterations, from_probe)
                }
            });
        }
        (restarted, restarted_cause, verified)
    };
    // The probe. With `chain`, its leg runs the fit's own restart chain
    // ([`resolve_restart_chain`]) before its verdict is read (#2707): from a perturbation of
    // a restart's point, on a multimodal inner problem, the probe's warm EBEs can be trapped
    // as the fit's were, and a probe stopped at a basin shift has not finished — it would
    // return no verdict on a start that is an optimum. The chain's last restart cause is not
    // carried on: the probe has had its restarts. The first attempt's probe keeps #2506's
    // single leg.
    let probe_for = |chain: bool| {
        let (restart_leg, probed, penalized) = (&restart_leg, &probed, &penalized);
        move |start: &ModelParameters, start_ofv: f64, spent_evals: usize| {
            // Its covariance step is deferred like every attempt's: `optimize_nlopt` runs it
            // once, on the attempt it reports. `start_eval` is the unmoved attempt's, set by
            // `verified_start_result` if the probe is reported.
            let (result, outcome) = optimize_nlopt_attempt(
                model,
                population,
                start,
                options,
                false,
                declines,
                Some(RestartLeg { escape_from: init_params, spent_evals, start_eval: None }),
                None,
                identity_scales,
                true,
            );
            let (result, cause) = if chain {
                let finished = resolve_restart_chain(
                    options.verbose,
                    crate::cancel::is_cancelled(&options.cancel),
                    (result, outcome.restart_cause()),
                    |stalled, cause| {
                        restart_leg(
                            stalled,
                            cause,
                            RestartLegOwner::Probe { spent_before: spent_evals },
                        )
                    },
                    penalized,
                );
                (finished, None)
            } else {
                (result, outcome.restart_cause())
            };
            let verdict = start_probe_verdict(start_ofv, penalized(&result), result.converged);
            if options.verbose {
                eprintln!(
                    "Start probe: the leg did not leave its start (OFV = {start_ofv:.6}); a \
                     restart from a perturbation of it reached OFV = {:.6} (converged = {}) → \
                     stationary start = {verdict:?}",
                    penalized(&result),
                    result.converged,
                );
            }
            *probed.borrow_mut() = Some((result, cause, verdict));
        }
    };
    let (first_probe, restart_probe) = (probe_for(false), probe_for(true));
    let ((first, first_cause), first_left_init) = attempt(init_params, false, Some(&first_probe));
    let first_attempt =
        settle_start_probe((first, first_cause), first_left_init, probed.take(), &penalized);
    let (first, restart_cause) = resolve_stall_retry(
        options.optimizer,
        options.verbose,
        first_attempt,
        || attempt(init_params, true, None),
        |(result, _)| penalized(result),
    );
    // #2093: a basin-shift restart may itself end in a basin shift (the restarted leg's
    // warm EBEs can be trapped again as θ moves on), so it is re-run while each restart is
    // adopted, budget remains, and at most `MAX_BASIN_RESTARTS` times. The legs share one
    // budget (`RestartLeg::spent_evals`), so the total stays bounded by `outer_maxiter`.
    // A mid-descent stall keeps its single restart: a second stall reports as a stall.
    let mut result = resolve_restart_chain(
        options.verbose,
        crate::cancel::is_cancelled(&options.cancel),
        (first, restart_cause),
        |stalled, cause| restart_leg(stalled, cause, RestartLegOwner::Fit(&restart_probe)),
        &penalized,
    );
    // The covariance step, once, on the attempt reported, at the exact packed vector it
    // ended on (`packed_estimate`, which every NLopt attempt sets).
    if let Some(x_hat) = result.packed_estimate.clone() {
        let crate::estimation::covariance::CovStepOutcome {
            matrix,
            wall_time_secs,
            warnings,
            sir_fallback_proposal,
            method,
            newton_decrement,
            non_stationary_optimum,
        } = crate::estimation::covariance::run_covariance_step(
            &x_hat,
            init_params,
            model,
            population,
            &result.eta_hats,
            &result.h_matrices,
            &result.kappas,
            options,
            options.verbose.then_some("Computing covariance matrix..."),
        );
        result.covariance_matrix = matrix;
        result.covariance_method = method;
        result.covariance_wall_time_secs = wall_time_secs;
        result.sir_fallback_proposal = sir_fallback_proposal;
        result.warnings.extend(warnings);
        result.newton_decrement = newton_decrement;
        // #2785: whatever NLopt and the plateau rule concluded, the point the fit reports
        // is not stationary — the stop the issue traced was a bare `Failure` the #751
        // plateau check accepted on an OFV trace that had stopped falling.
        crate::estimation::covariance::demote_non_stationary_optimum(
            non_stationary_optimum,
            &mut result.converged,
            &mut result.warnings,
        );
    }
    result
}

/// The restart chain behind [`optimize_nlopt`] (#2093): [`resolve_mid_descent_restart_adopting`], then,
/// while each restart is adopted and itself ends in a basin shift, another basin-shift restart,
/// at most [`MAX_BASIN_RESTARTS`] more. A restarted leg can trap again as θ moves on; the legs
/// share one evaluation budget (`RestartLeg::spent_evals`), and a restarted leg's own
/// `basin_shift` already requires budget left, so the total stays bounded by `outer_maxiter`.
/// A mid-descent stall keeps its single restart: a second stall reports as a stall. `restart`
/// returns the restarted result, the restart cause *its* leg ended with, and whether its start
/// probe verified it (a tie then adopts it). Pure, like
/// [`resolve_mid_descent_restart_adopting`], so every branch is unit-testable without NLopt.
fn resolve_restart_chain<T>(
    verbose: bool, cancelled: bool, first: (T, Option<RestartCause>),
    mut restart: impl FnMut(&T, RestartCause) -> (T, Option<RestartCause>, bool),
    ofv_of: impl Fn(&T) -> f64,
) -> T {
    let (mut current, mut cause) = first;
    let mut basin_restarts = 0usize;
    loop {
        let mut next_cause = None;
        let (next, adopted) = resolve_mid_descent_restart_adopting(
            verbose,
            cancelled,
            (current, cause),
            |stalled, c| {
                let (restarted, its_cause, verified) = restart(stalled, c);
                next_cause = its_cause;
                (restarted, verified)
            },
            &ofv_of,
        );
        current = next;
        match next_cause {
            Some(RestartCause::BasinShift) if adopted && basin_restarts < MAX_BASIN_RESTARTS => {
                basin_restarts += 1;
                cause = Some(RestartCause::BasinShift);
            }
            _ => return current,
        }
    }
}

/// Test hook (#2506): models, by name, whose first NLopt attempt stops **at its start** (a
/// one-evaluation leg; with a status, that NLopt status is reported and polish legs are
/// skipped) — a case no fixture produces on demand. Later legs run normally unless
/// `whole_fit` makes the whole budget one evaluation. Global (not thread-local) because
/// `fit()` runs on a pool worker; names are test-unique, as `covariance::forced_cov_decline`.
#[cfg(test)]
pub(crate) mod forced_stop_at_start {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    #[derive(Clone, Copy)]
    pub(crate) struct Stop {
        pub(crate) status: Option<nlopt::SuccessState>,
        pub(crate) whole_fit: bool,
    }

    static STOPS: Mutex<BTreeMap<String, Stop>> = Mutex::new(BTreeMap::new());

    pub(crate) fn set(model_name: &str, status: Option<nlopt::SuccessState>, whole_fit: bool) {
        STOPS.lock().unwrap().insert(model_name.to_string(), Stop { status, whole_fit });
    }

    /// The stop for a leg of `model_name`'s fit; `first_leg` is false on every leg but
    /// the first attempt.
    pub(crate) fn get(model_name: &str, first_leg: bool) -> Option<Stop> {
        let stop = STOPS.lock().unwrap().get(model_name).copied()?;
        (first_leg || stop.whole_fit).then_some(stop)
    }
}

/// How many basin-shift restarts [`optimize_nlopt`] runs at most after the first (#2093).
/// Each must be adopted and the legs share the fit's budget, so this only bounds the work
/// on a fit whose warm EBEs keep re-trapping.
const MAX_BASIN_RESTARTS: usize = 3;

/// Restart a run that stopped mid-descent from the best point it reached, and
/// report whichever attempt ended lower (#1277).
///
/// NLopt's `Failure` is ambiguous, and [`failure_is_converged_plateau`] already
/// splits it: a flat tail plus a self-consistent cold re-solve is a finished
/// fit, anything else is a genuine stall. This acts on the second half, which
/// until now was only *reported* ("Outer optimization did not converge") and
/// never acted on. Measured on the #1277 fixture — the two-cpt DCM in
/// `tests/fixtures/two_cpt_dcm_regularized.ferx` at `nn_l2 = 0`, started without
/// the `init` line it has carried since #1561 — L-BFGS quit at
/// eval 12 with the trace still falling ~2600 OFV per eval; restarting from that
/// point ran to convergence 3886 OFV units lower (3209.81 → −676.77; 7902 →
/// −540 by eval 44 on Linux, where the abort sits at a different OFV). The
/// mechanism is not NN-specific, and it is not one mechanism. Luksan's `plis`
/// line search is capped at ten step reductions and ten extrapolations, and a
/// cap hit on an iteration where the algorithm has just restarted — which the
/// first iteration always is — returns a bare `Failure` that discards every
/// point it accepted (#1411, `plis.c` instrumented). The eval-12 abort is the
/// *extrapolation* cap inside the first line search: the identity-Hessian
/// overshoot cap ([`should_cap_gradient`]) shrank the opening gradient 1.7e5×,
/// so every trial point showed a huge real decrease next to a tiny predicted
/// slope and the search extrapolated eleven times — before any `(s, y)` pair
/// exists. A fresh run from the best point works there because it re-arms that
/// cap around a gradient that is now O(1). The other shape is the *reduction*
/// cap later in the same fixture, where a subject's EBE mode switch degenerates
/// the curvature pair and the next direction is ~1e17 long; there the fresh
/// `(s, y)` memory is what helps.
///
/// Three things keep it from making any fit worse:
///
/// - it runs **only** on `mid_descent_stall`, so a converged fit, a plateaued
///   `Failure`, and a run that spent its `maxiter` budget (a `MaxEvalReached`
///   *success* state, not a `Failure`) are all untouched — and the restart runs
///   on what the stalled leg left of the budget ([`RestartLeg::spent_evals`]),
///   so it never hands a fit a second one (#1428; the first version did, and
///   said here that it could not). The two legs together are bounded by
///   `outer_maxiter` the way any single run is — `maxeval` is a soft bound for
///   L-BFGS, checked between line searches, so a few evals of overshoot are
///   possible on either leg;
/// - the restart is adopted only on a **strictly lower** penalized objective
///   (a tie only in the two cases below). Unlike the `left_init` retry above there is no second condition to check:
///   this attempt starts *at* the reported point, so anywhere it ends is
///   somewhere the fit genuinely reached; and
/// - `cancelled` short-circuits it, because a cancelled run stops through the
///   objective's 1e20 short-circuit rather than at a minimum and would look like
///   a stall to every test here.
///
/// There is exactly one restart for a stall — a second stall reports as a stall, which
/// is the honest signal and caps what this adds at one extra optimization. A basin-shift
/// restart may repeat ([`optimize_nlopt`], `MAX_BASIN_RESTARTS`, #2093), on the same
/// shared budget. (The
/// ceiling for the whole of [`optimize_nlopt`] is three, since the `left_init`
/// retry can fire first and hand its own result here.)
///
/// Deliberately **not** gated on the optimizer, unlike the `left_init` retry.
/// The line-search caps above are L-BFGS's, but the predicate here is not "did
/// Luksan abort" — it is "did NLopt return a bare `Failure` while the trace was
/// still descending, at a point a cold re-solve does not call a plateau", which
/// SLSQP (its own line search, `positive directional derivative`) and the
/// derivative-free methods can also produce; and the adoption rule (strictly
/// lower, or the stalled result stands) makes a pointless restart cost one
/// optimization and change nothing. An adopted restart is named in
/// `FitResult.warnings` ([`mid_descent_restart_warning`], `optimizer_health`).
///
/// **Basin shift** ([`RestartCause::BasinShift`], #1565) is the second reason to run
/// the same restart: the run finished, but the cold re-solve at its restored point beat
/// the objective it had minimised there, because some subjects' warm-started EBEs sat
/// in a higher individual mode for the whole trajectory. The restart's first inner
/// solve is cold, so it optimizes the objective the fit reports. Every guard above
/// applies unchanged, with one difference in the adoption rule: a tie goes to the
/// **restart**. The stalled run's own `converged` was demoted for the gap (see
/// [`ebe_basin_shift_warning`]), and a restart that starts at that point and cannot
/// improve on it is the evidence that the point is a minimum of the reported objective
/// after all — so its own verdict is the honest one to report. It still never reports a
/// higher objective, for the same reason as the stall case. A mid-descent restart its start
/// probe verified (#2707) takes a tie the same way: the probe came back to its point, which
/// is the same evidence, and dropping it would discard that verdict and the probe's
/// evaluations with it.
///
/// Pulled out as a pure fn — `restart` is handed the stalled result and returns
/// its successor, and whether its probe verified it — so every branch is unit-testable
/// without driving two NLopt fits.
///
/// Returns the reported result and whether it is the restart's (adopted).
fn resolve_mid_descent_restart_adopting<T>(
    verbose: bool, cancelled: bool, first: (T, Option<RestartCause>),
    restart: impl FnOnce(&T, RestartCause) -> (T, bool), ofv_of: impl Fn(&T) -> f64,
) -> (T, bool) {
    let (first, cause) = first;
    let Some(cause) = cause else {
        return (first, false);
    };
    if cancelled {
        return (first, false);
    }
    let (second, verified) = restart(&first, cause);
    // `ofv_is_valid` before the comparison, not `<` alone. A strict `<` rejects
    // `NaN` and `+inf` for free but **adopts `−inf`**, and a restart can return a
    // non-finite objective the same way any run can — `gate_converged_on_objective`
    // exists precisely because the final cold solve can hand back a `NaN` after a
    // trace that looked fine. Demoting that run's `converged` does not stop it
    // being ranked here, so without this guard a `−inf` successor would replace a
    // finite incumbent and take its usable estimates and diagnostics with it
    // (#1277 review).
    let second_ofv = ofv_of(&second);
    let first_ofv = ofv_of(&first);
    let ties_go_to_the_restart = verified || cause == RestartCause::BasinShift;
    let adopt = ofv_is_valid(second_ofv)
        && if ties_go_to_the_restart { second_ofv <= first_ofv } else { second_ofv < first_ofv };
    if !adopt {
        return (first, false);
    }
    if verbose {
        let why = match cause {
            RestartCause::MidDescent => "stopped mid-descent",
            RestartCause::BasinShift => "was minimised under EBEs a cold re-solve beats",
        };
        eprintln!(
            "Fit {why} (OFV = {first_ofv:.6}); restarted the optimizer from that point and \
             reached OFV = {second_ofv:.6}.",
        );
    }
    (second, true)
}

/// [`resolve_mid_descent_restart_adopting`] without the flags, for an unprobed restart: the
/// form the unit tests read.
#[cfg(test)]
fn resolve_mid_descent_restart<T>(
    verbose: bool, cancelled: bool, first: (T, Option<RestartCause>),
    restart: impl FnOnce(&T, RestartCause) -> T, ofv_of: impl Fn(&T) -> f64,
) -> T {
    resolve_mid_descent_restart_adopting(
        verbose,
        cancelled,
        first,
        |stalled, cause| (restart(stalled, cause), false),
        ofv_of,
    )
    .0
}

/// The stall-retry decision behind [`optimize_nlopt`], pure for unit tests. `first` is
/// `(result, left_init)`; `retry` runs **only** when it stalled. L-BFGS only; kept only when
/// it escaped init and reached a lower OFV, so it never makes the outcome worse.
fn resolve_stall_retry<T>(
    optimizer: Optimizer, verbose: bool, first: (T, bool), retry: impl FnOnce() -> (T, bool),
    ofv_of: impl Fn(&T) -> f64,
) -> T {
    let (first, left_init) = first;
    if left_init || !matches!(optimizer, Optimizer::NloptLbfgs) {
        return first;
    }
    let (retry, retry_left_init) = retry();
    // `ofv_is_valid` for the same reason as in [`resolve_mid_descent_restart_adopting`]:
    // `<` alone adopts a `−inf` retry over a finite first attempt. Latent here
    // rather than reported — this retry only runs on a fit that never left its
    // start — but the two resolvers share one adoption rule and having only one
    // of them screen the successor is how they drift (#1277 review).
    let retry_ofv = ofv_of(&retry);
    if !retry_left_init || !ofv_is_valid(retry_ofv) || !(retry_ofv < ofv_of(&first)) {
        return first;
    }
    if verbose {
        eprintln!(
            "Fit stalled on its initial estimates (OFV = {:.6}); retried with the \
             identity-Hessian cap held on and reached OFV = {:.6}.",
            ofv_of(&first),
            ofv_of(&retry),
        );
    }
    retry
}

/// What makes the #1277 restart leg different from a fresh attempt — the three
/// things it inherits from the leg it continues. See [`optimize_nlopt_attempt`].
pub(crate) struct RestartLeg<'a> {
    /// The point the *published* escape verdict is measured against: the user's
    /// initial estimates, not the stalled leg's estimates this leg starts at.
    pub(crate) escape_from: &'a ModelParameters,
    /// Objective evaluations the stalled leg already spent. The restart runs on
    /// what is left of the fit's `outer_maxiter` budget, not on a fresh one — a
    /// restart is a continuation of the same fit, and a user's `maxiter` is a
    /// bound on the fit (#1428).
    pub(crate) spent_evals: usize,
    /// The fit's first evaluation, published on the leg's result in place of the leg's own
    /// (see [`OuterResult::start_eval`]) — the counterpart of `escape_from` for #2461.
    pub(crate) start_eval: Option<StartEval>,
}

impl<'a> RestartLeg<'a> {
    /// The leg that continues `stalled`, a fit started at `init_params`.
    pub(crate) fn continuing(init_params: &'a ModelParameters, stalled: &OuterResult) -> Self {
        Self {
            escape_from: init_params,
            spent_evals: stalled.n_iterations,
            start_eval: stalled.start_eval,
        }
    }
}

/// The evaluation budget one NLopt run is given: `outer_maxiter × (n + 1)` for the
/// gradient methods, plus BOBYQA's `40 × (n + 1)` triangulation headroom for the
/// derivative-free one. Spelled once so the restart leg's remaining budget is
/// computed from the same number the first leg was given.
fn outer_eval_budget(algo: nlopt::Algorithm, n: usize, outer_maxiter: usize) -> u32 {
    let per_iter = n as u32 + 1;
    let base = (outer_maxiter as u32).saturating_mul(per_iter);
    if matches!(algo, nlopt::Algorithm::Bobyqa) {
        // BOBYQA is derivative-free: each eval is one objective call, not
        // n+1 (gradient methods FD the gradient inside one outer iter).
        // Give it enough headroom to triangulate a quadratic in n-D and
        // still make real trust-region progress: 40 evals/param baseline
        // plus the outer_maxiter budget. The setup phase alone costs
        // 2n+1 evals before any movement.
        base.saturating_add(40 * per_iter)
    } else {
        base
    }
}

/// `ftol_rel` / `xtol_rel` of the #1782 BOBYQA polish leg. These are the tolerances
/// that, set for the *whole* run, closed every gap the issue measured (the issue's
/// Linux numbers: `two_cpt_iv` to the L-BFGS optimum, `warfarin_iov` to within
/// 1.4e-5). Run as polish legs they end where the stagnation guard's 1e-3 window
/// latches, e.g. `warfarin_iov` 1.1e-3 above the optimum on Linux arm64. Each is a
/// ceiling: a user who asked for tighter keeps their own.
///
/// A profiled BOBYQA fit's polish legs (`optimize_profiled`, #2336) run at the same
/// ceilings.
pub(crate) const POLISH_FTOL_REL: f64 = 1e-10;
pub(crate) const POLISH_XTOL_REL: f64 = 1e-7;
/// A polish leg that lowered the objective by more than this is followed by
/// another: it was still descending. The 1e-3 the stagnation guard and the
/// plateau check already treat as the FOCE objective's noise floor.
const POLISH_MIN_GAIN: f64 = 1e-3;
/// Polish legs after the first BOBYQA stop, at most. Each costs at least the
/// `2n + 1` evaluations BOBYQA spends rebuilding its interpolation model.
///
/// Also the cap on a profiled gradient fit's verification legs (`optimize_profiled`,
/// #2285), which run through the same [`drive_polish_legs`]; one there costs about 5–20
/// evaluations. Changing it changes both.
pub(crate) const MAX_POLISH_LEGS: usize = 4;

/// Does this NLopt return need the #1782 polish leg before it may be called
/// converged? Only a BOBYQA *tolerance* stop, which it reports three ways:
/// `FtolReached`, `XtolReached` (its final Newton step), and a bare `Success` —
/// NLopt's `bobyqb_` falls through to its exit with `rc = NLOPT_SUCCESS` when the
/// trust radius reaches `rhoend = xtol_rel · rhobeg`, which is how the default
/// `examples/ctmm_pd_2state.ferx` fit stopped. All three are relative criteria the run
/// can meet mid-descent. Not a latched stagnation guard
/// (its stop is forced and goes to the plateau check instead), not a spent budget,
/// not a cancelled run, and not the gradient methods, whose default stops are the
/// unreachable 1e-12 ones. A profiled BOBYQA fit polishes the same stops (#2336), with
/// `latched = false`: its loop has no stagnation guard.
pub(crate) fn bobyqa_stop_needs_polish(
    algo: nlopt::Algorithm, result: &Result<(nlopt::SuccessState, f64), (nlopt::FailState, f64)>,
    latched: bool, cancelled: bool,
) -> bool {
    matches!(algo, nlopt::Algorithm::Bobyqa)
        && matches!(
            result,
            Ok((
                nlopt::SuccessState::Success
                    | nlopt::SuccessState::FtolReached
                    | nlopt::SuccessState::XtolReached,
                _
            ))
        )
        && !latched
        && !cancelled
}

/// [`bobyqa_stop_needs_polish`], or the same tolerance stop from a gradient optimizer run with
/// reachable stops (`gradient_run_has_reachable_stops`: the quadrature, #2549). Laplace/AGQ stop
/// on the relative `ftol_rel = 1e-6`, met mid-descent on a slow creep as BOBYQA's is: on
/// `pk2_big` Laplace it stopped 0.10 OFV above where the exact-gradient regime ends. So such a
/// stop takes the same #1782 polish legs from the best point at the `POLISH_*` tolerances,
/// with the last leg's stop deciding `converged`. So does a quadrature gradient run's bare
/// `Failure` or `RoundoffLimited` — a line search that found no decrease, which FOCE/FOCEI's
/// #2462 verification legs catch and the quadrature has none for: from one perturbed start
/// `pk2_big` Laplace ended on one at −3446.13 and the plateau check called it converged, 39 OFV
/// above the optimum. The caller polishes either only where a gain is predicted (the BHHH
/// decrement). A latched, spent, forced or cancelled stop is never polished; for BOBYQA only
/// its tolerance stops are, as before.
pub(crate) fn tolerance_stop_needs_polish(
    algo: nlopt::Algorithm, reachable_stops: bool,
    result: &Result<(nlopt::SuccessState, f64), (nlopt::FailState, f64)>, latched: bool,
    cancelled: bool,
) -> bool {
    let quadrature_stop = matches!(
        result,
        Ok((
            nlopt::SuccessState::Success
                | nlopt::SuccessState::FtolReached
                | nlopt::SuccessState::XtolReached,
            _
        )) | Err((nlopt::FailState::Failure | nlopt::FailState::RoundoffLimited, _))
    ) && !latched
        && !cancelled;
    bobyqa_stop_needs_polish(algo, result, latched, cancelled)
        || (quadrature_stop
            && reachable_stops
            && matches!(algo, nlopt::Algorithm::Lbfgs | nlopt::Algorithm::Slsqp))
}

/// BOBYQA's initial trust radius per coordinate, in scaled space.
///
/// NLopt's default rhobeg is 25% of the bound-width — huge in our log-space
/// packing (theta bounds can span 40+ log units), so the initial 2n+1
/// interpolation probes land in regions where the EBE inner loop fails and the
/// OFV gets clamped to 1e20, poisoning the quadratic model. 0.5 in scaled space
/// is a ~1.6× move on the natural parameter scale — small enough to stay
/// feasible at start, large enough to see real OFV signal.
pub(crate) fn bobyqa_initial_step(lower_s: &[f64], upper_s: &[f64]) -> Vec<f64> {
    lower_s
        .iter()
        .zip(upper_s)
        .map(|(lo, hi)| {
            let half_width = (hi - lo).abs() * 0.5;
            0.5_f64.min(half_width.max(1e-6))
        })
        .collect()
}

/// How a [`drive_polish_legs`] run ended: the #1782 polish of a BOBYQA tolerance stop,
/// or the #2285 verification legs of a profiled gradient fit (`optimize_profiled`).
/// `min_gain` below is `POLISH_MIN_GAIN` for the one; the other settles on
/// `profiled_leg_settles`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PolishEnd {
    /// The first stop needed no leg (`needs_polish` was false); nothing ran.
    NotNeeded,
    /// The last leg ran and settled: either it stopped on something that needs no leg
    /// (its own status decides), or it passed the settle test: it gained no more than
    /// `min_gain`, or for a profiled fit `profiled_leg_settles`. A budget that runs out
    /// during a leg ends here too, on that leg's `MaxEvalReached`.
    Settled,
    /// A stop wanted a leg but the evaluation budget was already spent before it —
    /// reachable, since BOBYQA checks `maxeval` *before* an evaluation and can return
    /// `FtolReached` after the last permitted one.
    BudgetSpent,
    /// `MAX_POLISH_LEGS` legs ran and the last one still failed the settle test: it
    /// gained more than `min_gain`, or a profiled leg was still moving when it stopped.
    LegCapReached { last_gain: f64 },
    /// A quadrature stop's last leg settled on a line-search `Failure` while the stop's BHHH
    /// Newton decrement still predicted `predicted` > `POLISH_MIN_GAIN` (#2585): the leg
    /// could not take a gain the local model says is there, so the stop is not verified. Set by the caller from the
    /// last `needs_polish` decision, not by [`drive_legs`].
    Stalled { predicted: f64 },
    /// A leg was wanted but could not start (no feasible point to polish from, #1617);
    /// the verdict is left to the paths that handle that run.
    Skipped,
}

/// The #1782 polish loop, with the NLopt run abstracted out so every exit is
/// unit-testable without driving BOBYQA. Shared by two callers: the BOBYQA polish
/// (`min_gain = POLISH_MIN_GAIN`, `needs_polish` = [`bobyqa_stop_needs_polish`]) and a
/// profiled fit's verification legs, which settle on their own test through
/// [`drive_legs`] (`profiled_leg_settles`, #2285, #2340). `first` is the first leg's
/// NLopt return; `min_gain` is the gain above which another leg runs; `needs_polish`
/// classifies a return; `budget_left`
/// reads what remains of the fit's evaluation budget; `leg` runs one polish leg on
/// that budget and returns its NLopt return and the objective gain it bought, or
/// `None` if it could not start. Returns the **last** leg's NLopt return — the one
/// whose status decides `converged` — and why the loop ended.
pub(crate) fn drive_polish_legs<R>(
    first: R, min_gain: f64, needs_polish: impl Fn(&R) -> bool, budget_left: impl Fn() -> u32,
    leg: impl FnMut(u32) -> Option<(R, f64)>,
) -> (R, PolishEnd) {
    // `!(gain > MIN)` rather than `gain <= MIN`, so a `NaN` gain settles instead of
    // counting as "still descending".
    drive_legs(first, |_, gain| !(gain > min_gain), needs_polish, budget_left, leg)
}

/// [`drive_polish_legs`] with the settle test as a predicate: `settles(r, gain)` says
/// whether a leg that returned `r` and gained `gain` ends the loop. When it does not,
/// another leg runs, up to `MAX_POLISH_LEGS`.
pub(crate) fn drive_legs<R>(
    first: R, settles: impl Fn(&R, f64) -> bool, needs_polish: impl Fn(&R) -> bool,
    budget_left: impl Fn() -> u32, mut leg: impl FnMut(u32) -> Option<(R, f64)>,
) -> (R, PolishEnd) {
    let mut result = first;
    let mut legs = 0usize;
    let mut last_gain = f64::INFINITY;
    loop {
        if !needs_polish(&result) {
            let end = if legs == 0 { PolishEnd::NotNeeded } else { PolishEnd::Settled };
            return (result, end);
        }
        if legs > 0 && settles(&result, last_gain) {
            return (result, PolishEnd::Settled);
        }
        if legs == MAX_POLISH_LEGS {
            return (result, PolishEnd::LegCapReached { last_gain });
        }
        let left = budget_left();
        if left == 0 {
            return (result, PolishEnd::BudgetSpent);
        }
        match leg(left) {
            None => return (result, PolishEnd::Skipped),
            Some((r, gain)) => {
                result = r;
                last_gain = gain;
                legs += 1;
            }
        }
    }
}

/// How a quadrature polish ended, given `stalled_gain` (the BHHH decrement behind the last
/// decision to polish, `None` when the last stop needed no polish) and `last`, the last
/// leg's return. A loop that settled on a leg's gain test with that decision standing, the
/// leg having ended on a line-search `Failure`, is [`PolishEnd::Stalled`].
///
/// The `Failure` and not the decrement's size is what tells a stall: near an optimum the
/// decrement overstates what is left (on `pk2_big`, 0.41 predicted 0.001–0.003 OFV from the
/// optimum, legs ending `FtolReached`), while every start that stalled 10–240 OFV above it
/// ended its leg on `Failure` with 2.8 to 74 predicted (#2585).
pub(crate) fn quadrature_polish_end(
    end: PolishEnd, stalled_gain: Option<f64>, last: &NloptReturn,
) -> PolishEnd {
    match (end, stalled_gain, last) {
        (PolishEnd::Settled, Some(predicted), Err((nlopt::FailState::Failure, _))) => {
            PolishEnd::Stalled { predicted }
        }
        (end, _, _) => end,
    }
}

/// The warning for [`PolishEnd::Stalled`].
pub(crate) fn polish_stalled_warning(predicted: f64) -> String {
    format!(
        "Outer optimization stopped where a Newton step is still predicted to lower the \
         objective by {predicted:.3e}, and a polish leg from there failed its line search \
         without gaining, so it is not reported as converged. The objective is not smooth at that point, typically \
         because some subjects' conditional modes switch between local minima; re-run from \
         other initial estimates and compare the objective."
    )
}

/// The warning for [`PolishEnd::LegCapReached`].
pub(crate) fn polish_leg_cap_warning(last_gain: f64) -> String {
    format!(
        "Outer optimization was still improving after {MAX_POLISH_LEGS} polish legs \
         from its tolerance stop (the last lowered the objective by {last_gain:.3e}), so it is \
         not reported as converged; re-run from these estimates or compare against a \
         gradient-based optimizer."
    )
}

type NloptReturn = Result<(nlopt::SuccessState, f64), (nlopt::FailState, f64)>;

/// The inner tolerance an unprofiled verification leg runs at (#2462), or the user's
/// `inner_tol` when tighter. At the default `1e-5` the EBE re-solve noise leaves L-BFGS no
/// step it can certify on an ill-conditioned objective: on A2's unprofiled twin 16 of 80
/// starts stopped 1.7e-4 to 5.3e-4 above NONMEM, all `converged`, and a fresh leg at the
/// fit's own settings gained exactly 0 from every one of them. At `1e-10` the whole grid
/// lands within 5.1e-8 (#2462's plan, §3).
pub(crate) const UNPROFILED_VERIFY_INNER_TOL: f64 = 1e-10;

/// An unprofiled verification leg settles when its own gain is at most this (#2462).
/// Absolute OFV, measured inside the leg at its tolerance. Over A2's 80 starts (Linux
/// aarch64), a first leg from a stop already within 1e-5 of the optimum gained at most
/// 5.9e-6 and one from a stuck stop at least 1.70e-4, so 3e-5 sits 5.1× above the one and
/// 5.7× below the other. On eight more fixtures with a known optimum (10 to 100 subjects,
/// |OFV| 62 to 3792) the genuine side reached at most 1.8e-6, with no trend in either, so
/// the threshold is not scaled.
///
/// It is not the only way a leg settles; see [`unprofiled_leg_settles`].
pub(crate) const UNPROFILED_VERIFY_GAIN: f64 = 3e-5;

/// Does an unprofiled fit get the #2462 verification leg at all? FOCE / FOCEI with a
/// gradient optimizer (L-BFGS, SLSQP), whose `1e-12` stops are unreachable, so that its
/// stops are bare `Failure`s the plateau rule accepts or tolerance stops met mid-descent.
/// Not a quadrature objective, which has reachable stops
/// (`gradient_run_has_reachable_stops`) and its own noise floor. A mixture takes the
/// incumbent-seeded legs (#2745), each class's inner solve at the leg tolerance, but not
/// the [`LegStart`] passes, which score the single-population objective.
pub(crate) fn unprofiled_verify_applies(algo: nlopt::Algorithm, reachable_stops: bool) -> bool {
    matches!(algo, nlopt::Algorithm::Lbfgs | nlopt::Algorithm::Slsqp) && !reachable_stops
}

/// Does a stop need a verification leg before it may be called converged (#2462)? Every
/// stop that the status or the plateau rule would otherwise report converged: a success
/// or tolerance status, and `Failure` / `ForcedStop` / `RoundoffLimited`. Not a spent
/// budget or time, not a cancelled run.
///
/// The failures only once the fit has `left_init`, the guard
/// `failure_is_converged_plateau` and the profiled `profiled_stop_needs_verify` keep: a
/// leg from a start L-BFGS could not leave gains nothing and would report the initial
/// estimates converged. `plateau_shape` is the rest of the plateau rule that can be read
/// before the final inner solve (progress, and a flat tail); it gates `Failure` /
/// `ForcedStop` at the first stop, which without it is not reported converged and goes to
/// the #1277 restart instead. A leg's own stop passes `true`: its gain test replaces the
/// flat tail.
pub(crate) fn unprofiled_stop_needs_verify(
    result: &NloptReturn, cancelled: bool, left_init: bool, plateau_shape: bool,
) -> bool {
    !cancelled
        && match result {
            Ok((status, _)) => !matches!(
                status,
                nlopt::SuccessState::MaxEvalReached | nlopt::SuccessState::MaxTimeReached
            ),
            Err((nlopt::FailState::RoundoffLimited, _)) => left_init,
            Err((nlopt::FailState::Failure | nlopt::FailState::ForcedStop, _)) => {
                left_init && plateau_shape
            }
            Err(_) => false,
        }
}

/// Does a verified stop also need `left_init` to be reported converged (#2462)? Only where
/// the verdict before the leg asked for it: a first stop of `Failure` / `ForcedStop`, or a
/// latched one, both of which went to the plateau rule. A success-class or
/// `RoundoffLimited` stop was converged on its status alone, so a refit started at its own
/// optimum, which never leaves its initial estimates, keeps that verdict once a leg has
/// verified it.
pub(crate) fn verified_stop_needs_left_init(first: &NloptReturn, latched: bool) -> bool {
    latched || matches!(first, Err((nlopt::FailState::Failure | nlopt::FailState::ForcedStop, _)))
}

/// Does an unprofiled verification leg end the loop (#2462)? Any of three tests settles it:
///
/// - its own `gain`, first feasible evaluation to its best, is at most
///   [`UNPROFILED_VERIFY_GAIN`];
/// - it is the first leg (`first_leg`) and its gain is at most [`UNPROFILED_FIRST_LEG_GAIN`]
///   (#2507);
/// - or, from the second leg on, it ended no further below the lowest leg before it than
///   the objective can resolve there (`net`, that leg's best minus this leg's, is at most
///   `resolution`). `resolution` is measured, not chosen: the leg's cold-seeded first score
///   of the incumbent against that incumbent's own score in the leg that found it — the
///   same point at the same tolerance, so their gap is what the objective itself disagrees
///   with itself by. All three are at the leg tolerance.
///
/// The net-change test is for an objective whose noise is larger than the threshold. On the
/// 200-subject Michaelis–Menten benchmark (`tests/ebe_basin_shift.rs`, a multimodal inner
/// problem) every leg at the optimum gains 9.3e-5 to 2.4e-3, from where its cold-seeded
/// first evaluation lands, while the incumbent only wanders: −5.9e-4 to +6.5e-4 from leg to
/// leg. On the gain alone every start reached the leg cap. A leg that ends above where the
/// last one ended is measuring that noise, not a descent — and so is one that ends lower by
/// less than the gap between two scores of its own starting point (#2482: on
/// `ctmm_pd_2state`, whose optimum sits on a bound, the second leg ended 7.5e-5 lower where
/// its start's two scores differed by 9.1e-5, and a third leg of ~60 evaluations ran).
///
/// The first-leg test is #2507's. `UNPROFILED_VERIFY_GAIN` tells a loose stop that was
/// already at the optimum (its first leg gains at most 5.9e-6 on A2) from one that was
/// stuck (at least 1.70e-4). A stuck stop then cost a second leg, from a point that is
/// itself a stop at the tight tolerance, and that leg never descended: on A2's 80-start
/// grid (macOS arm64) every one of the 16 second legs gained at most 8.1e-9, and on the
/// rifapentine benchmark's 116-subject ODE model it ended 2.2e-5 *above* the first,
/// after ~150 evaluations. A first leg runs from a cold EBE seed to its own stop at
/// `UNPROFILED_VERIFY_INNER_TOL`, which is the configuration that put the whole A2 grid
/// within 5.1e-8 of NONMEM when run as the fit itself; it needs a second leg only when it
/// gained more than the objective's own noise floor, where it may still be descending.
///
/// A `NaN` gain (a leg with no feasible evaluation) does not settle, and neither does a
/// missing `net`.
pub(crate) fn unprofiled_leg_settles(
    gain: f64, net: Option<f64>, resolution: Option<f64>, first_leg: bool,
) -> bool {
    gain <= UNPROFILED_VERIFY_GAIN
        || (first_leg && gain <= UNPROFILED_FIRST_LEG_GAIN)
        || net.is_some_and(|d| d <= resolution.unwrap_or(0.0).max(0.0))
}

/// The gain at or below which the first unprofiled verification leg settles (#2507; see
/// [`unprofiled_leg_settles`]). `POLISH_MIN_GAIN`, the 1e-3 the stagnation guard, the plateau
/// check and the BOBYQA polish already treat as the FOCE objective's noise floor: above it a
/// leg may still be descending.
pub(crate) const UNPROFILED_FIRST_LEG_GAIN: f64 = POLISH_MIN_GAIN;

/// One unprofiled verification leg as the loop judges it (#2462).
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnprofiledLeg {
    pub(crate) ret: NloptReturn,
    /// The leg's own gain: its first feasible objective minus its best, `NaN` with none.
    pub(crate) gain: f64,
    /// The leg's best objective at its tolerance, `None` with no feasible evaluation.
    pub(crate) best: Option<f64>,
    /// The leg's first feasible objective — its cold-seeded score of the incumbent it
    /// starts from — `None` with no feasible evaluation.
    pub(crate) first: Option<f64>,
}

/// The #2462 verification legs: [`drive_legs`] settling on [`unprofiled_leg_settles`].
/// `first_needs_verify` judges the first stop, which also needs the plateau shape
/// ([`unprofiled_stop_needs_verify`]); `needs_verify` judges each leg's own return. Returns
/// the last return, why the loop ended, and whether the stop was verified: the loop
/// settled on a leg whose own return still needed verifying, so the leg's test, not its
/// status, decided it. A leg that ran out of budget settles on `MaxEvalReached` and is not
/// verified. The fourth value is the last leg's net change against the lowest leg before it
/// (that leg's best minus this one's, so positive when it ended lower), `None` with no
/// earlier feasible leg or when this leg had no feasible evaluation.
pub(crate) fn unprofiled_verify_legs(
    first: NloptReturn, first_needs_verify: bool, needs_verify: impl Fn(&NloptReturn) -> bool,
    budget_left: impl Fn() -> u32, mut leg: impl FnMut(u32) -> Option<UnprofiledLeg>,
) -> (NloptReturn, PolishEnd, bool, Option<f64>) {
    // (return, is the first stop, net change against the lowest previous leg, the
    // resolution measured at this leg's start, is the first leg)
    type Step = (NloptReturn, bool, Option<f64>, Option<f64>, bool);
    let mut prev_best: Option<f64> = None;
    let mut legs_run = 0usize;
    let ((last, _, last_net, _, _), end) = drive_legs(
        (first, true, None, None, false),
        |s: &Step, gain| {
            #[cfg(test)]
            if UNPROFILED_VERIFY_NEVER_SETTLE.with(|c| c.get()) {
                return false;
            }
            unprofiled_leg_settles(gain, s.2, s.3, s.4)
        },
        |s: &Step| {
            if s.1 {
                first_needs_verify
            } else {
                needs_verify(&s.0)
            }
        },
        budget_left,
        |left| {
            let l = leg(left)?;
            legs_run += 1;
            let net = prev_best.zip(l.best).map(|(p, b)| p - b);
            // Only from the second leg: the first leg's incumbent was scored at the loose
            // tolerance, so the gap to it is not the objective's resolution at this one.
            // Against the lowest leg so far, which is where this leg started (#2507).
            let resolution = prev_best.zip(l.first).map(|(p, f)| (f - p).abs());
            // The lowest leg so far, the one the fit restores (#2507): a leg that ended above
            // it leaves it in place, so the next leg is measured against it too.
            prev_best = match (prev_best, l.best) {
                (Some(p), Some(b)) => Some(p.min(b)),
                (p, b) => b.or(p),
            };
            Some(((l.ret, false, net, resolution, legs_run == 1), l.gain))
        },
    );
    let verified = matches!(end, PolishEnd::Settled) && needs_verify(&last);
    (last, end, verified, last_net)
}

/// Is a finished #2462 verification leg's incumbent the one the fit keeps (#2507)? The first
/// leg with a feasible evaluation is; a later one only when it ended strictly lower than the
/// incumbent kept so far, since both are at the leg tolerance and the fit restores the lowest
/// point it evaluated (#59).
///
/// `live` is the state the run's final restore reads with that point and that every leg
/// overwrites from its first improving evaluation (each leg's own incumbent starts empty):
/// in `optimize_nlopt_once`, the optimizer gradient (`last_gradient`) and the incumbent's
/// whole EBE / `H` / κ solve (`best_solve`). `kept_best` / `kept_state` hold the kept leg's
/// objective and that state. A kept leg's is recorded; a leg that is not kept has `live`
/// replaced by the kept leg's, so `final_gradient` belongs to the point the fit reports and
/// the final inner loop scores the held EBE state *at* that point — a state from another
/// point fails its `xs` check and can only seed a fresh solve, which on a multimodal inner
/// problem need not find the same modes. Before #2507 every later leg was kept, so the
/// rifapentine benchmark reported its second leg's 1378.1047474 over the first's
/// 1378.1047252.
pub(crate) fn keep_verify_leg<G: Clone>(
    kept_best: &mut Option<f64>, kept_state: &mut Option<G>, leg_best: Option<f64>, live: &mut G,
) -> bool {
    match leg_best {
        Some(b) if kept_best.is_none_or(|k| b < k) => {
            *kept_best = Some(b);
            *kept_state = Some(live.clone());
            true
        }
        _ => {
            if let Some(g) = kept_state {
                *live = g.clone();
            }
            false
        }
    }
}

/// Where a leg starts. Every leg starts at the incumbent's own modes (#2549) unless it opens
/// one of the further passes a verified stop gets (#2645): a round of
/// [`unprofiled_verify_legs`] from somewhere the warm legs cannot reach, which replaces the
/// incumbent only by ending lower ([`keep_verify_leg`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LegStart {
    /// The incumbent, at its own modes.
    Incumbent,
    /// The incumbent, at a cold solve's modes. A warm leg verifies the stop on the modes the
    /// fit already holds, so it cannot see that another mode set has a lower optimum in θ,
    /// and a subject's other mode can score *worse* at the incumbent and still lead there: no
    /// per-subject comparison at that point finds it. On the propofol benchmark (24 subjects,
    /// 3-cpt closed form) a cold solve at the incumbent scores −3038.866 against −3038.980,
    /// on one subject, and the leg from those modes reaches −3039.288158, NONMEM's
    /// −3039.2882 (macOS arm64). On the cefepime benchmark (458 subjects) four subjects score
    /// 2.32 worse cold, and the leg ends 2.76 lower. It runs only when
    /// [`subjects_off_the_held_modes`] finds a subject to move.
    ColdModes,
    /// The BHHH Newton step from the incumbent, at its modes. A leg restarts L-BFGS with no
    /// curvature memory, so on a long valley its first line search can fail at a point that
    /// is no minimum. On the busulfan benchmark (1518 subjects, ODE + IOV; TVCL and TVDROP
    /// nearly confounded) the fit stopped `converged` at 126579.41, with |g|∞ 4.26 and
    /// a predicted gain of 7.9, and every leg gained under 1e-6; a leg from this step reaches
    /// 126565.48 with |g|∞ 0.08 (macOS arm64). It runs only when that gain exceeds
    /// [`LATCH_MAX_PREDICTED_GAIN`], the guard's own line between a plateau and a creep.
    NewtonStep,
}

/// Does a stop that ended `end` get the pass opening at `start`? After the warm legs, not
/// instead of them, because no start wins everywhere: the warm leg is what
/// `transit_flip_flop_fit_matches_ode_twin` needs (#2549), and the others can lose outright,
/// so the fit keeps whichever ended lower. Only from a stop the legs settled: a leg cap or a
/// spent budget is not a verified incumbent, a stop the #2093 basin probe ended goes to the
/// basin-shift restart, and a cancelled fit starts nothing. `Settled` does not rule the last
/// out: a cancel during a leg ends the loop there, as settled.
#[cfg_attr(not(test), allow(unused_variables))] // `start` is the tests' switch
pub(crate) fn leg_pass_applies(
    start: LegStart, end: PolishEnd, basin_probe_fired: bool, cancelled: bool,
) -> bool {
    #[cfg(test)]
    if LEG_PASSES_OFF.with(|o| o.get()).contains(&start) {
        return false;
    }
    matches!(end, PolishEnd::Settled) && !basin_probe_fired && !cancelled
}

/// Does a verified stop already belong to a restart, and so take no [`LegStart`] pass? True
/// when the end of the attempt, were it at this stop, would restart it: a cold solve that
/// fails the consistency check (#1277) or beats the objective (#1565). A pass could lower the
/// incumbent just enough to pass those checks, and the restart that would have gone further
/// would never run. On a simulated 80-subject dataset (`restart_wins.csv`) rounds took a stop at
/// 778.06, cold 781.31, down to 777.17, consistent and reported converged, where the restarts
/// from 778.06 reach 775.45. A stop that takes passes is one nothing was going to restart, so
/// passes never replace a restart. `best` and `cold` are at the attempt's final check's own
/// terms: `cold` at the fit's `inner_tol`, not the legs'.
pub(crate) fn verified_stop_owes_a_restart(best: f64, cold: f64) -> bool {
    !cold_solve_is_consistent(Some(best), cold)
        || cold_solve_beats_best_seen(Some(cold), Some(best)).is_some()
}

/// The fit's own state a pass can leave behind, restored when the pass is discarded, and
/// whether the fit takes passes at all (`mixture`).
pub(crate) struct PassState<'a> {
    pub(crate) basin_probe_fired: &'a std::cell::Cell<bool>,
    pub(crate) stagnation_latched: &'a std::sync::atomic::AtomicBool,
    /// The incumbent's objective at the leg tolerance: what a kept pass changes.
    pub(crate) kept_best: &'a std::cell::Cell<Option<f64>>,
    /// A mixture fit takes no pass (#2745): a pass's cold solve, Newton step and comparison of
    /// modes are the single-population objective's and do not score a mixture's classes.
    pub(crate) mixture: bool,
}

/// Runs each [`LegStart`] pass that applies after the verdict `warm`, and returns the verdict
/// that stands: a pass's only if it moved the incumbent ([`keep_verify_leg`]). A discarded pass
/// leaves nothing behind. Its latch and its #2093 probe describe the point it explored, not
/// the incumbent, so they go back: a cold round's probe would otherwise stop the Newton round
/// and send a verified stop to a restart on another point's evidence.
///
/// `stop_is_free` is asked once, before the first pass, whether the verified stop may take
/// passes at all ([`verified_stop_owes_a_restart`]).
pub(crate) fn drive_leg_passes<V>(
    warm: V, end_of: impl Fn(&V) -> PolishEnd, cancelled: impl Fn() -> bool, state: PassState<'_>,
    stop_is_free: impl FnOnce() -> bool, mut pass: impl FnMut(LegStart, &V) -> V,
) -> V {
    if state.mixture {
        return warm;
    }
    let mut verdict = warm;
    let mut stop_is_free = Some(stop_is_free);
    for start in [LegStart::ColdModes, LegStart::NewtonStep] {
        let probe_fired = state.basin_probe_fired.get();
        if !leg_pass_applies(start, end_of(&verdict), probe_fired, cancelled()) {
            continue;
        }
        #[cfg(test)]
        LEG_PASSES_APPLIED.with(|c| c.set(c.get() + 1));
        if stop_is_free.take().is_some_and(|free| !free()) {
            break;
        }
        let verified_best = state.kept_best.get();
        let verified_latched = state.stagnation_latched.load(Ordering::Relaxed);
        let round = pass(start, &verdict);
        if state.kept_best.get() == verified_best {
            state.stagnation_latched.store(verified_latched, Ordering::Relaxed);
            state.basin_probe_fired.set(probe_fired);
        } else {
            verdict = round;
        }
    }
    verdict
}

/// Does a Newton step's predicted `gain` warrant a [`LegStart::NewtonStep`] leg? Above
/// [`LATCH_MAX_PREDICTED_GAIN`], and only with more subjects than coordinates the step moves.
/// `S = Σ gᵢgᵢᵀ` has rank at most the subject count, and on `s`'s own span `sᵀS⁺s` reads as a
/// constant rather than a distance to the optimum: one subject reads exactly 1 wherever it
/// is. On the one-subject `fit_on_ss_absorption_converges` that sent a leg along a flat
/// direction, V from 20 to 6.66 for 0.002 OFV.
pub(crate) fn newton_gain_warrants_a_leg(gain: f64, n_subjects: usize, step: &[f64]) -> bool {
    gain > LATCH_MAX_PREDICTED_GAIN && n_subjects > step.iter().filter(|d| **d != 0.0).count()
}

/// The longest step, on any packed coordinate, a [`LegStart::NewtonStep`] leg starts from.
/// `S` is nearly singular along the very valley the step is for, so the full step overshoots:
/// on the busulfan benchmark it scored ~2.2e5 against 126579, and the leg ended at 182723.
/// Capped at 0.1 or 0.3 (10% to 35% on a log-packed θ), the same step reached 126565.48
/// in two legs and one; 0.3 is the cap that took one.
const NEWTON_START_MAX_STEP: f64 = 0.3;

/// `x + step`, the step scaled down so that no coordinate moves more than
/// [`NEWTON_START_MAX_STEP`], then each coordinate held inside its bounds.
pub(crate) fn newton_start(x: &[f64], step: &[f64], bounds: &PackedBounds) -> Vec<f64> {
    let longest = step.iter().fold(0.0_f64, |m, d| m.max(d.abs()));
    let shrink = (NEWTON_START_MAX_STEP / longest).min(1.0);
    x.iter()
        .zip(step)
        .enumerate()
        .map(|(k, (v, d))| (v + shrink * d).clamp(bounds.lower[k], bounds.upper[k]))
        .collect()
}

/// How many subjects' `−2LL` terms differ between the held and the cold modes by more than
/// [`PLATEAU_OFV_THRESHOLD`], in either direction: a [`LegStart::ColdModes`] pass has nothing
/// to explore when this is zero, which is every unimodal fit, so there it costs one inner pass
/// and no leg. `>` on a difference: a `NaN` side is never counted.
pub(crate) fn subjects_off_the_held_modes(held: &[f64], cold: &[f64]) -> usize {
    held.iter().zip(cold).filter(|(h, c)| 2.0 * (*h - *c).abs() > PLATEAU_OFV_THRESHOLD).count()
}

/// The internal switch for the #2462 verification leg. `false` restores the pre-#2462
/// verdict on every stop; kept so a change in a test can be attributed to the leg.
const UNPROFILED_VERIFY_LEG: bool = true;

#[cfg(test)]
thread_local! {
    /// `optimize_nlopt_attempt` calls on this thread, for the tests.
    pub(crate) static NLOPT_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Turns the #2462 leg off on this thread, for the tests' mutation checks.
    pub(crate) static UNPROFILED_VERIFY_OFF: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// The #2645 passes turned off on this thread, for the tests' straddles.
    pub(crate) static LEG_PASSES_OFF: std::cell::Cell<&'static [LegStart]> =
        const { std::cell::Cell::new(&[]) };
    /// [`LegStart`] passes `drive_leg_passes` reached on this thread, for the tests (#2745).
    pub(crate) static LEG_PASSES_APPLIED: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// The per-class modes each #2462 leg of a mixture started from on this thread, for the
    /// tests (#2837 r1 #5).
    pub(crate) static MIXTURE_LEG_START_MODES: std::cell::RefCell<Vec<Vec<Vec<DVector<f64>>>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Test switch: every #2462 leg after the first kept one is treated as not kept, so a
    /// dropped leg is followed by another whatever the last digits of the objective do.
    pub(crate) static UNPROFILED_VERIFY_DROP_LATER_LEGS: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// The #2462 legs `optimize_nlopt_attempt` ran on this thread, for the tests.
    pub(crate) static UNPROFILED_VERIFY_LEGS: std::cell::RefCell<Vec<UnprofiledLeg>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Each mid-leg basin probe that found a gap on this thread (#2600): the gap and the
    /// switched-subject count [`basin_probe_action`] decided on.
    pub(crate) static BASIN_PROBE_DECISIONS: std::cell::RefCell<Vec<(f64, Option<usize>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Overrides `BASIN_PROBE_RESTART_SHARE` on this thread (#2600).
    pub(crate) static BASIN_PROBE_RESTART_SHARE_TEST: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
    /// Makes no #2462 leg settle on this thread, so every stop runs `MAX_POLISH_LEGS` legs —
    /// for the #2507 test of the kept leg's state.
    pub(crate) static UNPROFILED_VERIFY_NEVER_SETTLE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Whether the held `best_solve` at the final restore belonged to the restored point
    /// (`None` with none held), for the #2507 tests.
    pub(crate) static HELD_SOLVE_AT_RESTORED: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    /// Scales the first subject's held `H` at the final restore, for the #2646 test of this
    /// leg: a natural gap between a cold and a warm EBE needs a fixture the next search-path
    /// change retires, as `tests/ebe_basin_shift.rs` records four times over.
    pub(crate) static HELD_FIRST_H_SCALE: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
}

fn unprofiled_verify_enabled() -> bool {
    #[cfg(test)]
    if UNPROFILED_VERIFY_OFF.with(|o| o.get()) {
        return false;
    }
    UNPROFILED_VERIFY_LEG
}

/// The warning for an unprofiled verification that reached the leg cap (#2462). It names
/// the two numbers the settle test read: `last_net`, how far the last leg ended below the
/// one before it, and `last_gain`, the last leg's own gain, which runs from its
/// cold-seeded first evaluation and so includes the cold-versus-warm EBE gap.
pub(crate) fn unprofiled_verify_cap_warning(last_gain: f64, last_net: Option<f64>) -> String {
    let net =
        last_net.map_or(String::new(), |d| format!(" ended {d:.3e} below the leg before it, and"));
    format!(
        "Outer optimization was still improving after {MAX_POLISH_LEGS} verification legs \
         at a tight inner tolerance: the last{net} lowered the objective by {last_gain:.3e} \
         from its own first evaluation, so it is not reported as converged; re-run from \
         these estimates."
    )
}

/// [`optimize_nlopt_attempt`] with its own covariance step.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn optimize_nlopt_once(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, hold_cap_at_init: bool, declines: &OuterFdDeclineLog,
    restart: Option<RestartLeg<'_>>, first_eval_etas: Option<&[DVector<f64>]>,
    identity_scales: &[(usize, f64)],
) -> (OuterResult, AttemptOutcome) {
    optimize_nlopt_attempt(
        model,
        population,
        init_params,
        options,
        hold_cap_at_init,
        declines,
        restart,
        first_eval_etas,
        identity_scales,
        false,
    )
}

/// One NLopt outer-optimizer run. Returns the result and the
/// [`AttemptOutcome`] — whether the fit left its initial estimates, and whether
/// it stopped mid-descent — which is what [`optimize_nlopt`] runs a second
/// attempt on. `hold_cap_at_init` is the retry mode — see
/// [`should_cap_gradient`].
///
/// `restart` is `Some` for the #1277 restart leg only, and carries the two
/// things that leg inherits from the stalled one ([`RestartLeg`]):
///
/// - `escape_from` separates **where this attempt starts** from **what its
///   published escape verdict is measured against**, which are the same thing
///   for every attempt but the restart. `None` means "this attempt's own
///   start", the ordinary case. The restart reports [`OuterResult::left_init`]
///   relative to `escape_from` instead — it begins at a stalled fit's estimates,
///   but the flag that reaches `model_selection::stalled_at_init` has to answer
///   "are the reported estimates still on top of the values the *user*
///   supplied". The two internal consumers of displacement keep this attempt's
///   own start either way: the L-BFGS overshoot cap ("has the fit moved yet?")
///   and [`failure_is_converged_plateau`] ("did it descend, or twitch and die?")
///   are both questions about this leg, not about the pair.
/// - `spent_evals` is subtracted from the run's evaluation budget
///   ([`outer_eval_budget`]), so the two legs together never exceed the
///   `outer_maxiter` the user set. The caller does not restart on an empty
///   budget: `set_maxeval(0)` means *unlimited* to NLopt, not "no evaluations".
///
/// With `defer_covariance` the covariance step is left to the caller: [`optimize_nlopt`] runs
/// it once, on the attempt it reports, rather than on every attempt of its retry and restart
/// chain (a restarted fit used to pay a full step for a result it then replaced).
#[allow(clippy::too_many_arguments)]
fn optimize_nlopt_attempt(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, hold_cap_at_init: bool, declines: &OuterFdDeclineLog,
    restart: Option<RestartLeg<'_>>, first_eval_etas: Option<&[DVector<f64>]>,
    identity_scales: &[(usize, f64)], defer_covariance: bool,
) -> (OuterResult, AttemptOutcome) {
    optimize_nlopt_attempt_probed(
        model,
        population,
        init_params,
        options,
        hold_cap_at_init,
        declines,
        restart,
        first_eval_etas,
        identity_scales,
        defer_covariance,
        None,
    )
}

/// [`optimize_nlopt_attempt`] with the #2506 start probe: when the attempt ends without
/// leaving its start, `start_probe` is run from a perturbation of that start
/// ([`perturbed_start`]) on what is left of the budget, and its verdict
/// ([`start_probe_verdict`]) decides whether the unmoved start is a warm start at an
/// optimum. [`optimize_nlopt`] passes one on the first attempt ([`ProbeWhen::Unmoved`]) and
/// on each restart leg ([`ProbeWhen::DisplacementAloneRejects`], #2707); the held-cap retry
/// and the probe itself pass `None`.
#[allow(clippy::too_many_arguments)]
fn optimize_nlopt_attempt_probed(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, hold_cap_at_init: bool, declines: &OuterFdDeclineLog,
    restart: Option<RestartLeg<'_>>, first_eval_etas: Option<&[DVector<f64>]>,
    identity_scales: &[(usize, f64)], defer_covariance: bool,
    start_probe: Option<(&StartProbe<'_>, ProbeWhen)>,
) -> (OuterResult, AttemptOutcome) {
    #[cfg(test)]
    NLOPT_ATTEMPTS.with(|c| c.set(c.get() + 1));
    let escape_from = restart.as_ref().map(|r| r.escape_from);
    let spent_evals = restart.as_ref().map_or(0, |r| r.spent_evals);
    let inherited_start = restart.as_ref().map(|r| r.start_eval);
    let PackedStart { packed: mut x0, bounds, fixed, .. } = pack_with_bounds(init_params);
    clamp_to_bounds(&mut x0, &bounds);
    let n = x0.len();
    // The coordinates the optimizer can move: a held one (`FIX`, or a structural-zero Ω entry)
    // sits at `lower == upper`. The stagnation window is sized in these (#2597): a held
    // coordinate is no search direction, so counting it padded the window — 35 evaluations
    // instead of 19 on a 6-η model with one 2 × 2 block, 16 of them spent flat.
    let n_free = stagnation_dimension(&fixed);
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let mut warnings = Vec::new();

    // Covariate-NN (DCM) regularizer (L2 + smoothness). No-op when both λ are 0,
    // so the penalized objective/gradient below stay byte-identical for
    // unregularized fits. The optimizer (and everything that ranks or gates on
    // what it minimised — `best_ofv`, stagnation, `best_seen`, the stall retry)
    // sees the penalized objective; everything user-facing (verbose `Eval`
    // lines, the trace, the checkpoint, the plateau self-consistency check, and
    // the reported OFV/AIC/BIC) sees the clean −2LL, which is what `final_ofv`
    // recomputes from a fresh `pop_nll_opts` at the end.
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    // Parameter priors (#254) ride alongside on exactly the same contract: the
    // optimizer minimises the penalized objective, everything user-facing keeps
    // the clean −2LL, and `FitResult` reports the two halves separately.
    let priors = build_prior_set(model, init_params);
    // Built once for the whole outer optimization: every eval below re-solves the same
    // subjects' schedules, which depend only on `model` + `population` (never on the
    // trial `params`) — see `run_inner_loop_and_nll_prepared`.
    let schedule_cache =
        crate::estimation::inner_optimizer::build_schedule_cache(model, population);

    // Per-element scale factors: present O(1) coordinates to NLopt — see
    // `nlopt_scale_vector` for which coordinates stay unscaled and why.
    //
    // IOV + SLSQP auto-enables per-coordinate scaling (#101): IOV models pack disparate
    // magnitudes (block-diagonal omega plus the kappa block), and SLSQP's uniform gradient cap
    // (`cap_scaled_gradient`, every SLSQP eval) would otherwise rescale the whole gradient by
    // the worst (theta) component and starve the omega / omega_iov step. The #99 regression that
    // made scaling default-off was on non-IOV models and other algorithms (notably MMA), so the
    // auto-enable is scoped to the combination that needs it. Since #155/#490 the default
    // optimizer is `Auto` (`NloptLbfgs` / BOBYQA, gradient-free, no starvation), so this fires
    // only for an explicit `optimizer = slsqp` on IOV models.
    let auto_scale_iov = model.n_kappa > 0 && matches!(options.optimizer, Optimizer::Slsqp);
    let scale =
        nlopt_scale_vector(&x0, init_params, &bounds, options, auto_scale_iov, identity_scales);
    // The EBE warm-start predictor (`NloptState::pred_jac`) where the adjoint forms the mode
    // response: FOCE/FOCEI, and Laplace (AGQ with one node), on a BSV-only, non-mixture
    // model the outer gradient serves; not AGQ with more than one node. An IOV
    // subject's κ̂ is not predicted, and its seeded warm solve already starts near the mode.
    //
    // Not on a `[covariate_nn]` model: there it costs time and saves none. Measured on the
    // DCM with normalised inputs (`center`/`scale`), ten ±20% perturbed starts each way
    // (Linux x86_64): the optima it reaches match those without it (median OFV −1452 vs
    // −1483, best −1590 vs −1588, better on 5 of 10), but the fits run to the iteration cap
    // either way, so the response through the network's ~150 weights only adds cost: 1.4× the
    // wall time (1340 s vs 928 s). The landscape of the weights, not EBE tracking, limits
    // those fits. On the unnormalised shipped example
    // it also changed which optimum the shipped start reached (OFV −164.5 vs −1173.6): on
    // saturating units the path decides the optimum, as for the inner seed (#1474).
    let warm_predictor = warm_predictor_serves(model, init_params, options);
    let lower_s: Vec<f64> = (0..n).map(|i| bounds.lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| bounds.upper[i] / scale[i]).collect();
    // Scale x0 into optimizer space: xs[i] = x[i] / scale[i].
    for i in 0..n {
        x0[i] /= scale[i];
    }
    // Snapshot of the scaled starting point. `x0` itself is handed to NLopt as
    // the mutable iterate (and later overwritten with the restored best point),
    // so the "how far has the fit moved from init?" tests below — the L-BFGS
    // overshoot cap and the plateau verdict — need their own copy.
    let x0_start_s: Vec<f64> = x0.clone();
    // The reference for the *published* escape verdict — see `escape_from`. Packed
    // and scaled through the same `bounds`/`scale` as `x0_start_s` so the two are
    // compared in one consistent space; `scale` is derived from this attempt's own
    // start, which is fine because `max_scaled_deviation` only needs both points in
    // the same metric, not a canonical one.
    let escape_start_s: Vec<f64> = match escape_from {
        None => x0_start_s.clone(),
        Some(origin) => {
            let mut v = pack_params(origin);
            clamp_to_bounds(&mut v, &bounds);
            for i in 0..n {
                v[i] /= scale[i];
            }
            v
        }
    };

    // Optional gradient-free global pre-search (NLopt CRS2-LM). Samples
    // within the parameter bounds and lets the local optimizer pick up
    // from the best point found — useful for poorly-identified models
    // where the local optimizer can land in a degenerate basin from a
    // far-from-truth start. The pre-search runs the same FOCE objective
    // as the main optimizer (no shortcuts), so each global eval is a
    // full inner-loop pass; budget is `global_maxeval` (0 → auto:
    // `30 * (n_params + 1)`, see `run_global_presearch`).
    if options.global_search {
        let pre_x = run_global_presearch(
            model,
            population,
            init_params,
            options,
            &scale,
            &lower_s,
            &upper_s,
            &x0,
        );
        match pre_x {
            Ok(best_x) => x0 = best_x,
            Err(e) => warnings.push(format!("global_search disabled: {}", e)),
        }
    }

    // The iterate NLopt is actually handed, snapshotted after the optional global
    // pre-search has had its say. Deliberately *not* `x0_start_s`: that snapshot
    // predates the pre-search, and `global_search = true` moves `x0` to a point the
    // pre-search already scored feasibly — which is the point a run that then scores
    // nothing feasible should fall back to (#1617), not the start the pre-search was
    // run to improve on. Identical to `x0_start_s` whenever the pre-search is off,
    // which is the default.
    let optimizer_start_s: Vec<f64> = x0.clone();

    let mut state =
        new_nlopt_state(n_subj, n_eta, &x0, SalvageGuardPolicy::for_optimizer(options.optimizer));
    // #2076: the flat-theta pre-flight's EBEs, solved at this attempt's start, seed the
    // first evaluation — but only on a leg that starts there: not a restart leg (it
    // starts at the stalled estimates), and not after a global pre-search that moved
    // the start.
    if let Some(etas) = first_eval_etas
        .filter(|e| restart.is_none() && optimizer_start_s == x0_start_s && e.len() == n_subj)
    {
        state.cached_etas = etas.to_vec();
        state.cached_etas_kind = PrevEtas::ModesHere;
    }

    // External counter mirrors state.n_evals — nlopt doesn't hand `state`
    // back after `opt.optimize()`, so we need an Arc to read the final
    // count for reporting. Keep both in sync inside the objective closure.
    let n_evals_outer = Arc::new(AtomicUsize::new(0));
    let n_evals_cl = Arc::clone(&n_evals_outer);
    // Set when the stagnation guard latches, so the verdict below can tell a
    // guard-forced `Success` from one NLopt reached on its own (#1530).
    let stagnation_latched = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stagnation_latched_cl = Arc::clone(&stagnation_latched);
    // #1934: the decrement of the last veto, and whether the latch overrode it, so a
    // guard stop against a still-predicted gain is reported rather than silent.
    let last_veto_gain: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
    let last_veto_gain_cl = Arc::clone(&last_veto_gain);
    let latch_overrode_veto = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let latch_overrode_veto_cl = Arc::clone(&latch_overrode_veto);

    // Best-seen accumulator (issue #59). NLopt returns the last evaluated
    // point, not the best one — when the stagnation guard short-circuits
    // by returning `best_ofv` with zero gradient, the optimizer can drift
    // a step or two off the true minimum before its xtol/ftol fires. We
    // track the best `(xs, ofv, ofv_clean)` externally and restore x0 to it
    // after optimize() returns, before the final inner loop and covariance
    // step. `ofv` is what the optimizer minimised (penalized under covariate-NN
    // regularization) and ranks the points; `ofv_clean` is the −2LL at the same
    // point, kept so the plateau self-consistency check compares it against the
    // equally clean `final_ofv` instead of against a penalized number.
    let best_seen: Arc<Mutex<BestPoint>> = Arc::new(Mutex::new(BestPoint::new()));
    let best_seen_cl = Arc::clone(&best_seen);

    // The full EBE state of the best feasible eval, published by the objective closure
    // (#833): NLopt never hands `state` back, so it has to be copied out as it happens.
    //
    // All three of `(η, H, κ)` and not just the etas, because the incumbent is scored as
    // a *candidate* and not only used as a seed. A seed alone would leave the headline
    // invariant unproven: `run_inner_loop_warm` is not monotone from its seed — the FREM
    // arm of `find_ebe` can adopt a Nelder–Mead restart over a better BFGS partial
    // (#1365) — so a re-solve from the incumbent EBEs can come back *above* the objective
    // those EBEs already had. Scoring them as they stand costs one `pop_nll_opts` pass
    // and closes that hole.
    //
    // A seed that belongs to a slightly different point (the tracker also observes
    // guard-penalised evals, whose EBEs are not adopted) can only cost an inner solve,
    // never a wrong number: every candidate is scored before one is chosen.
    let best_solve: Arc<Mutex<Option<IncumbentSolve>>> = Arc::new(Mutex::new(None));
    let best_solve_cl = Arc::clone(&best_solve);

    let last_gradient: Arc<Mutex<Option<Vec<f64>>>> = Arc::new(Mutex::new(None));
    let last_gradient_cl = Arc::clone(&last_gradient);

    // Externalised OFV-plateau tracker counting *feasible* (unguarded) evals
    // only: `(baseline_ofv, last_sig_feasible_eval, feasible_evals)`.
    // `feasible_evals` is a 1-based running count of unguarded evals;
    // `last_sig_feasible_eval` is the feasible-eval index at which the feasible
    // best OFV last improved by more than `PLATEAU_OFV_THRESHOLD` over
    // `baseline_ofv`. Guarded evals are excluded entirely so (a) a guard-penalty
    // value (which pollutes `state.best_ofv`) can never seed a fake "improvement"
    // — feasible eval 1 only establishes the baseline, real progress must land on
    // a later feasible eval — and (b) a tail of guard-rejected boundary probes
    // cannot pad the plateau length (both the index and the count are feasible-
    // only, so `flat_tail = feasible_evals − last_sig_feasible_eval` measures flat
    // *feasible* evals). Distinct from (and independent of) the stagnation guard's
    // own bookkeeping so it works even when that guard is disabled. Read after
    // `optimize()` to tell a plateaued optimum from a genuine early stall (#751).
    let plateau_tracker: Arc<Mutex<(f64, usize, usize)>> =
        Arc::new(Mutex::new((f64::INFINITY, 0, 0)));
    let plateau_tracker_cl = Arc::clone(&plateau_tracker);

    // EBE stats accumulator: tracks worst unconverged count and total fallbacks.
    #[derive(Default)]
    struct EbeAccum {
        max_unconverged: usize,
        total_fallback: usize,
        total_nm_polished: usize,
        n_convergence_warnings: usize,
        /// Evaluations rejected because the objective was not a finite number rather than
        /// by the EBE guard (#1617 review round 3 #2). Counted here, not inferred from the
        /// callback count, because the latched-stagnation fast path increments that counter
        /// and returns the incumbent objective without producing a trial at all.
        n_nonfinite: usize,
    }
    let ebe_accum: Arc<Mutex<EbeAccum>> = Arc::new(Mutex::new(EbeAccum::default()));
    let ebe_accum_cl = Arc::clone(&ebe_accum);

    // Select NLopt algorithm. `optimize_population` resolves `Auto` to a concrete
    // optimizer before dispatching here, so `Auto` should never reach this match;
    // map it to BOBYQA (auto's FD fallback) rather than the catch-all SLSQP so a
    // future bypass degrades to the safe derivative-free path, not a silent SLSQP.
    let algo = match options.optimizer {
        Optimizer::Slsqp => nlopt::Algorithm::Slsqp,
        Optimizer::NloptLbfgs => nlopt::Algorithm::Lbfgs,
        Optimizer::Mma => nlopt::Algorithm::Mma,
        Optimizer::Bobyqa | Optimizer::Auto => nlopt::Algorithm::Bobyqa,
        _ => nlopt::Algorithm::Slsqp,
    };

    let verbose = options.verbose;

    // #2462: the verification leg's state, shared with the objective. While `in_leg` is
    // set the inner loop runs at `leg_options`' tolerance and every feasible evaluation is
    // ranked in `leg_best`, the leg's own incumbent, instead of `best_seen`, which holds
    // objectives at the user's tolerance; `leg_first` is the leg's first feasible
    // objective, for its gain.
    let verify_on = unprofiled_verify_enabled()
        && unprofiled_verify_applies(algo, gradient_run_has_reachable_stops(options));
    let leg_options = FitOptions {
        inner_tol: options.inner_tol.min(UNPROFILED_VERIFY_INNER_TOL),
        ..options.clone()
    };
    let in_leg = std::cell::Cell::new(false);
    let leg_best = std::cell::RefCell::new(BestPoint::new());
    let leg_first: std::cell::Cell<Option<f64>> = std::cell::Cell::new(None);
    // Set when the #2093 basin probe ends a leg. Such a stop belongs to the basin-shift
    // restart, not to a verification leg: on `final_ebe_solve`'s starved warfarin fit the
    // probe stopped at 620.8 to restart from a cold re-solve's lower basin, and legs from
    // there walked to −270.92 and settled, while the restart reaches −286.0042.
    let basin_probe_fired = std::cell::Cell::new(false);
    // #2461: the run's first evaluation, unguarded, for `above_start_warning`.
    let start_eval: std::cell::Cell<Option<StartEval>> = std::cell::Cell::new(None);

    // NLopt objective: receives xs (scaled), unscales before running inner loop.
    // Gradient: d(OFV)/d(xs[i]) = d(OFV)/d(x[i]) * scale[i] (chain rule).
    let objective = |xs: &[f64], grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
        // Cooperative cancellation: short-circuit cheaply so NLopt burns through
        // its remaining iteration budget in microseconds instead of minutes.
        if crate::cancel::is_cancelled(&options.cancel) {
            if let Some(g) = grad {
                for gi in g.iter_mut() {
                    *gi = 0.0;
                }
            }
            return 1e20;
        }
        // Stagnation guard: once latched, every subsequent eval returns
        // `best_ofv` with zero gradient. SLSQP / L-BFGS see a stationary
        // point and terminate via xtol_rel within a couple of evals,
        // instead of grinding through the remaining maxeval budget at
        // full inner-loop cost. See `detect_stagnation` doc comment for
        // the trigger criterion.
        if state.stagnation_stopped {
            if let Some(g) = grad {
                for gi in g.iter_mut() {
                    *gi = 0.0;
                }
            }
            state.n_evals += 1;
            n_evals_cl.fetch_add(1, Ordering::Relaxed);
            return state.best_ofv;
        }
        // Unscale from optimizer space to real (log/Cholesky) space.
        let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let params = unpack_params(&x, init_params);
        // The options the inner solve and the gradient run at: the leg's while it runs.
        let in_verify_leg = in_leg.get();
        let eval_options = if in_verify_leg { &leg_options } else { options };

        // EBE warm-start cache: the update below is withheld unless this eval
        // improves on the best seen, so the warm start always comes from the
        // incumbent. See `adopt_warm_start` for why the alternative deadlocks the
        // line search (#1290). `cached_etas` needs no snapshot — the update is a
        // plain assignment we can skip — but the mixture branch writes
        // `cached_etas_by_class` before the objective is known, so that one has to
        // be saved to be restored. Only for a mixture: this is the per-eval hot
        // path, and an unconditional clone would cost an allocation per subject on
        // every fit.
        let warm_start_by_class =
            params.mixture.is_some().then(|| state.cached_etas_by_class.clone());

        // Mixture models (#977): K-fold log-sum-exp objective with a per-class
        // serial inner solve. The per-eval `kappas` slot stays empty: on the mixture
        // path the gradient reads `mixeval` (not these kappas), so the MIXEST-class κ
        // are only needed at the final inner loop below, where they *are* carried. The
        // MIXEST-class EBEs stand in for the warm-start / trace. The full `MixtureEval`
        // is kept in `mixeval` for the analytic gradient below.
        let mut mixeval: Option<crate::estimation::mixture::MixtureEval> = None;
        // Every class's κ̂ for an IOV mixture's exact gradient (the eval keeps only the
        // MIXEST class's).
        let mut mixmodes: Option<crate::estimation::mixture::MixtureClassModes> = None;
        let fuse_agq_gradient = grad.is_some()
            && options.agq_nodes().is_some()
            && !reconverge_this_eval(options, state.n_grad_evals)
            && crate::estimation::agq::analytic_gradient_available(model);
        // `contribs` is the per-subject `2·nllᵢ` behind `raw_ofv` (#1520 salvage guard);
        // left empty on the mixture branch, whose objective has no per-subject
        // decomposition, which disarms the guard there.
        let mut contribs: Vec<f64> = Vec::new();
        // #2682: a guard-rejected first evaluation is solved a second time, warm from its own
        // EBEs, before it is judged — see `wants_start_warm_rescore`. `cold_start` holds the
        // first solve (and what it overwrote) while the second runs.
        let mut cold_start: Option<ColdStart> = None;
        let mut warm_rescored_from: Option<usize> = None;
        // #2709: a pass the guard is certain to reject stops solving subjects. Only once
        // the fit has scored a feasible evaluation: so never on the first evaluation, whose
        // rejected EBEs #2682's warm re-solve starts from, and never on a run with no
        // feasible evaluation at all, whose `W_NO_FEASIBLE_EVAL` reports the worst count.
        let feasible_seen = plateau_tracker_cl.lock().unwrap().2 > 0;
        let guard_stop = (feasible_seen && !guard_stop_disabled()).then(|| {
            GuardStop::new(
                n_subj,
                eval_options.min_obs_for_convergence_check as usize,
                options.max_unconverged_frac,
            )
        });
        let (ehs, hms, ebe_stats, kappas, raw_ofv, agq_evaluation) = loop {
            let solved = if params.mixture.is_some() {
                let warm = (!state.cached_etas_by_class.is_empty())
                    .then_some(state.cached_etas_by_class.as_slice());
                let mut m =
                    if grad.is_some() && crate::estimation::mixture::needs_class_modes(&params) {
                        let (m, modes) = crate::estimation::mixture::mixture_ofv_cached(
                            model,
                            population,
                            &params,
                            eval_options,
                            warm,
                            Some(&schedule_cache),
                        );
                        mixmodes = Some(modes);
                        m
                    } else {
                        crate::estimation::mixture::mixture_ofv_cached(
                            model,
                            population,
                            &params,
                            eval_options,
                            warm,
                            Some(&schedule_cache),
                        )
                        .0
                    };
                let stats = InnerLoopStats {
                    n_unconverged: m.ebe_stats.n_unconverged,
                    n_fallback: m.ebe_stats.n_fallback,
                    n_nm_polished: m.ebe_stats.n_nm_polished,
                    n_start_rejected: m.ebe_stats.n_start_rejected,
                };
                let ofv = m.ofv;
                // A derivative-free eval (`grad` is `None` — e.g. BOBYQA)
                // never touches `mixeval` or the analytic gradient, so avoid the full
                // per-class EBE cache clone: move `etas_by_class` straight into the
                // warm-start cache and the MIXEST EBEs into the result. When a gradient
                // *is* requested the analytic path reads `m.etas_by_class`, so it must
                // stay intact and the cache takes a clone.
                if grad.is_some() {
                    state.cached_etas_by_class = m.etas_by_class.clone();
                    let out = (
                        m.mixest_etas.clone(),
                        m.mixest_h_mats.clone(),
                        stats,
                        Vec::new(),
                        ofv,
                        None,
                    );
                    mixeval = Some(m);
                    out
                } else {
                    state.cached_etas_by_class = std::mem::take(&mut m.etas_by_class);
                    (
                        std::mem::take(&mut m.mixest_etas),
                        std::mem::take(&mut m.mixest_h_mats),
                        stats,
                        Vec::new(),
                        ofv,
                        None,
                    )
                }
            } else {
                let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
                let predicted: Option<Vec<DVector<f64>>> = match &state.pred_jac {
                    Some(jac) if state.cached_etas_kind != PrevEtas::ModesHere => {
                        #[cfg(test)]
                        crate::estimation::inner_optimizer::record_prediction(
                            &state.cached_etas,
                            jac,
                            &state.pred_anchor,
                            &x,
                        );
                        Some(crate::estimation::sens_outer_gradient::predict_warm_etas(
                            &state.cached_etas,
                            jac,
                            &state.pred_anchor,
                            &x,
                        ))
                    }
                    _ => None,
                };
                // Read once: pre-flight modes are at this leg's start, the next evaluation is
                // elsewhere. They stay a solve's output whether or not this pass is adopted;
                // the zero placeholder stays one unless it is (the adoption below).
                let prev = state.cached_etas_kind;
                state.cached_etas_kind = prev.after_pass(false);
                let (ehs, hms, ebe_stats, kappas, nll, prepared, nll_i) =
                    run_inner_loop_and_nll_prepared(
                        model,
                        population,
                        &params,
                        eval_options,
                        Some(predicted.as_ref().unwrap_or(&state.cached_etas)),
                        predicted.is_some().then_some(
                            crate::estimation::inner_optimizer::PredictedFromAll {
                                modes: &state.cached_etas,
                                seeds: state.seed_hessians.as_deref(),
                            },
                        ),
                        Some(&mu_k),
                        fuse_agq_gradient.then_some((
                            init_params,
                            x.as_slice(),
                            &bounds,
                            warm_predictor,
                            Some(declines),
                        )),
                        Some(&schedule_cache),
                        prev,
                        options.cancel.as_ref(),
                        guard_stop.as_ref(),
                    );
                contribs = nll_i.iter().map(|v| 2.0 * v).collect();
                (ehs, hms, ebe_stats, kappas, 2.0 * nll, prepared)
            };
            if let Some(cold) = cold_start.take() {
                // The warm re-score is judged on its own; when the guard still rejects it the
                // evaluation is the cold one, as it was before #2682, and so is the warm-start
                // cache — no incumbent exists yet to have earned a non-zero one (#1290).
                state.cached_etas = cold.cached_etas;
                state.cached_etas_kind = cold.cached_etas_kind;
                let accepted = warm_rescore_accepted(
                    &solved.2,
                    n_subj,
                    solved.4,
                    options.max_unconverged_frac,
                );
                if verbose {
                    eprintln!(
                        "Eval {:>4}: start rejected by the EBE convergence guard ({} of {} \
                         subjects unconverged from the cold start); re-solved warm from its own \
                         EBEs: {} unconverged, {}",
                        state.n_evals + 1,
                        cold.solved.2.n_unconverged,
                        n_subj,
                        solved.2.n_unconverged,
                        if accepted { "accepted" } else { "still rejected" },
                    );
                }
                if accepted {
                    warm_rescored_from = Some(cold.solved.2.n_unconverged);
                    break solved;
                }
                contribs = cold.contribs;
                mixeval = cold.mixeval;
                mixmodes = cold.mixmodes;
                break cold.solved;
            }
            if start_eval.get().is_none()
                && !crate::cancel::is_cancelled(&options.cancel)
                && wants_start_warm_rescore(
                    &solved.2,
                    n_subj,
                    solved.4,
                    options.max_unconverged_frac,
                )
            {
                // A mixture's warm start is `cached_etas_by_class`, which its branch above has
                // already set to this solve's modes; only the single-class cache is set here.
                let warm = if params.mixture.is_some() {
                    state.cached_etas.clone()
                } else {
                    outer_cache_warm_starts(model, &solved.0, &solved.3)
                };
                cold_start = Some(ColdStart {
                    cached_etas: std::mem::replace(&mut state.cached_etas, warm),
                    cached_etas_kind: std::mem::replace(
                        &mut state.cached_etas_kind,
                        PrevEtas::Solved,
                    ),
                    solved,
                    contribs: std::mem::take(&mut contribs),
                    mixeval: mixeval.take(),
                    mixmodes: mixmodes.take(),
                });
                continue;
            }
            break solved;
        };
        // The pass above polls the flag per subject (#574), so a cancel that lands inside
        // it comes back here as placeholders. Leave the way the entry check does, before
        // the guard accounting, the state caches, the trace and any gradient pass see the
        // evaluation.
        if crate::cancel::is_cancelled(&options.cancel) {
            if let Some(g) = grad {
                g.fill(0.0);
            }
            return 1e20;
        }
        // Penalized objective fed to the optimizer (unregularized fits unchanged).
        // When a gradient is wanted the penalty gradient comes out of the same
        // pass (one network evaluation per curvature node) and is spliced into
        // `grad_raw` below; `clean_ofv` keeps the −2LL for the user-facing
        // streams.
        let clean_ofv = raw_ofv;
        let mut nn_grad: Vec<f64> =
            if grad.is_some() && nn_reg.is_active() { vec![0.0; n] } else { Vec::new() };
        let nn_penalty = if nn_grad.is_empty() {
            nn_reg.penalty_value(&params.theta)
        } else {
            nn_reg.penalty_and_gradient(&params.theta, &mut nn_grad)
        };
        // Parameter priors (#254). Unlike the NN penalty this is already a
        // function of the packed vector, so there is nothing to map: `x` *is*
        // the space the penalty is defined in, and `prior_grad` is spliced into
        // `grad_raw` below without a change of variables. Value and gradient come from one
        // call on purpose (`PriorSet::penalty_and_gradient`): taken separately, deleting only
        // the gradient half left the whole integration suite green.
        let mut prior_grad: Vec<f64> =
            if grad.is_some() && priors.is_active() { vec![0.0; n] } else { Vec::new() };
        let prior_penalty = if prior_grad.is_empty() {
            priors.penalty(&x)
        } else {
            priors.penalty_and_gradient(&x, &mut prior_grad)
        };
        let raw_ofv = raw_ofv + nn_penalty + prior_penalty;

        // #2709: a pass stopped on a certain rejection reports the counts of the subjects it
        // solved; its unsolved placeholders are not results. Rejected by construction: the
        // counts it stopped on already cross the threshold. `ebe_guard_rejects` alone does
        // not always see that. Its fraction arm needs a finite `raw_ofv`, and `raw_ofv` can
        // be non-finite here: on the AGQ path it is `INFINITY` (the quadrature skip), and a
        // solved subject's term can be non-finite on any path. A fraction-triggered stop
        // has `n_start_rejected = 0` in the observed stats, so the hard-rejection arm
        // returns false too. The `stopped_after` gate covers that case: a stopped pass is
        // always a guard rejection (see `GuardStop::record` for the counter this moves).
        let (ebe_stats, stopped_after) = stopped_pass_stats(guard_stop.as_ref(), ebe_stats);
        #[cfg(test)]
        if stopped_after.is_some() {
            GUARD_STOPS.with(|n| n.set(n.get() + 1));
        }
        // EBE convergence guard: reject step when too many subjects unconverged or any
        // subject was hard-rejected at its inner start.
        let ebe_guard_triggered = stopped_after.is_some()
            || ebe_guard_rejects(&ebe_stats, n_subj, raw_ofv, options.max_unconverged_frac);
        {
            let mut acc = ebe_accum_cl.lock().unwrap();
            if acc.max_unconverged < ebe_stats.n_unconverged {
                acc.max_unconverged = ebe_stats.n_unconverged;
            }
            acc.total_fallback += ebe_stats.n_fallback;
            acc.total_nm_polished += ebe_stats.n_nm_polished;
            if ebe_guard_triggered {
                acc.n_convergence_warnings += 1;
            } else if !raw_ofv.is_finite() {
                acc.n_nonfinite += 1;
            }
        }

        // Guard-rejected (EBE guard or non-finite OFV). For the gradient optimizers return
        // a quadratic penalty consistent with the center-push gradient below, so NLopt's
        // line search can backtrack instead of failing on a first-step overshoot (#486; see
        // `guard_penalty_value`). Derivative-free BOBYQA (`grad` is `None`) keeps the wall.
        let guarded = ebe_guard_triggered || !raw_ofv.is_finite();
        if start_eval.get().is_none() {
            start_eval.set(Some(StartEval {
                ofv: raw_ofv,
                guard_rejected: ebe_guard_triggered,
                n_unconverged: ebe_stats.n_unconverged,
                n_start_rejected: ebe_stats.n_start_rejected,
                rescored: None,
                warm_rescored_from,
            }));
        }
        let ofv = if guarded {
            if grad.is_some() {
                guard_penalty_value(xs, &lower_s, &upper_s)
            } else {
                1e20
            }
        } else {
            raw_ofv
        };
        // The −2LL twin of `ofv` for the user-facing streams (trace, checkpoint,
        // verbose lines, plateau check). Identical to `ofv` unless a covariate-NN
        // penalty is on; a guarded eval carries its sentinel in both.
        let ofv_clean = if guarded { ofv } else { clean_ofv };

        // #1520: whether a population gradient is formed at this point (recorded for the
        // salvage guard after the evaluation, below).
        let gradient_requested = grad.is_some() && !guarded;
        // Compute gradient if requested (central FD with fixed EBEs)
        let mut grad_norm_for_trace: Option<f64> = None;
        // Per-coordinate scaled gradient for the trace (#640); only populated
        // for a genuine OFV gradient (not the guard-penalty push), so it is
        // present exactly when `grad_norm_for_trace` is.
        let mut grad_vec_for_trace: Option<Vec<f64>> = None;
        // This evaluation's mode responses, when its gradient formed them (#2549).
        let mut eta_dx = crate::estimation::sens_outer_gradient::ModeResponses {
            want_dx: warm_predictor,
            ..Default::default()
        };
        if let Some(g) = grad {
            // A rejected or non-finite point has no useful population gradient: steepest
            // ascent toward bounds center nudges the optimizer back. Fall through (no early
            // return) so the eval is still traced and prev_x / stagnation stay correct,
            // just skipping the expensive population gradient (#603 review #5).
            if guarded {
                for i in 0..g.len() {
                    let center_s = (lower_s[i] + upper_s[i]) / 2.0;
                    g[i] = 100.0 * (xs[i] - center_s);
                }
            } else {
                // d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x); then scale for optimizer space.
                // Mixture (#977 Phase 4): analytic posterior-weighted gradient, a
                // declined (subject × class) term salvaged on its own and recorded for
                // the #1154 warning (#2906). No whole-objective difference (#3004).
                // #1520: per the optimizer's own acceptance test (`SalvageGuardPolicy`),
                // the incumbent the salvage guard may measure this gradient point against.
                let guard_reference = state.salvage_guard.reference(xs);
                let mut grad_raw = if let Some(mev) = &mixeval {
                    crate::estimation::mixture::mixture_outer_gradient(
                        model,
                        population,
                        init_params,
                        &x,
                        eval_options,
                        mev,
                        mixmodes.as_ref(),
                        declines,
                        Some(&schedule_cache),
                    )
                } else {
                    population_gradient_with_agq_evaluation(
                        &x,
                        n_subj,
                        init_params,
                        model,
                        population,
                        &ehs,
                        &hms,
                        &kappas,
                        &bounds,
                        eval_options,
                        &mut state.n_grad_evals,
                        agq_evaluation,
                        // #1520: the optimizer-facing objective at this point and the
                        // incumbent it will be judged against. `raw_ofv` is the
                        // penalized value — the one the optimizer ranks on — and the
                        // EBE-guard sentinel cannot reach here (a guarded evaluation
                        // takes the center-push arm above).
                        OuterTrial {
                            ofv: raw_ofv,
                            contribs: &contribs,
                            incumbent: guard_reference,
                        },
                        declines,
                        // The predictor reads the mode response only where this evaluation
                        // becomes the incumbent's warm start.
                        // So does a quadrature tolerance stop's decrement (`scores`).
                        // And, with it, the `η̂` a polished subject's gradient was taken at
                        // ([`adopt_polished_modes`]).
                        adopt_warm_start(guarded, ofv, state.best_ofv).then_some(&mut eta_dx),
                    )
                };
                // Splice in the NN penalty gradient (computed above, in the same
                // pass as its value). It lands in packed-x space: NN weights are
                // identity-packed, so a natural-space weight coordinate *is* a
                // packed coordinate. The `* scale[k]` below is the chain rule
                // `x = x_s · scale` and applies to this term exactly as it does
                // to the likelihood part — do not "simplify" it away for the
                // penalty on the strength of NN scales usually being 1.0;
                // `compute_scale` gives any |w| > 0.1 a non-unit scale.
                for (gk, nk) in grad_raw.iter_mut().zip(&nn_grad) {
                    *gk += nk;
                }
                // Parameter priors (#254): `2(x−m)/s²` at the priored
                // coordinates, computed above in the same pass as the value and
                // already in packed-x space, so the `* scale[k]` chain rule
                // below applies to it unchanged.
                for (gk, pk) in grad_raw.iter_mut().zip(&prior_grad) {
                    *gk += pk;
                }
                let mut sq = 0.0_f64;
                for k in 0..g.len() {
                    let gi = if grad_raw[k].is_finite() { grad_raw[k] * scale[k] } else { 0.0 };
                    g[k] = gi;
                    sq += gi * gi;
                }
                grad_norm_for_trace = Some(sq.sqrt());
                // Clone the scaled gradient for the trace only when a trace is
                // open — this runs on every gradient eval (the hot path), so an
                // unconditional `to_vec` would waste an allocation per eval on
                // the default trace-off path (#640 review). Snapshot before the
                // SLSQP cap below so it matches `grad_norm_for_trace`.
                if crate::estimation::trace::is_active() {
                    grad_vec_for_trace = Some(g.to_vec());
                }
                let hold_cap =
                    hold_cap_at_init && max_scaled_deviation(xs, &x0_start_s) < INIT_ESCAPE_STEP_S;
                if should_cap_gradient(algo, state.n_grad_evals, hold_cap) {
                    cap_scaled_gradient(g, &lower_s, &upper_s);
                }
                // Gate on the global best (same tracker as the `best_seen` update
                // below) so `last_gradient` always reflects the best point seen.
                {
                    // Inside a #2462 leg the incumbent is the leg's own.
                    let global_best = if in_verify_leg {
                        leg_best.borrow().ofv()
                    } else {
                        best_seen_cl.lock().unwrap().ofv()
                    };
                    if ofv < global_best {
                        *last_gradient_cl.lock().unwrap() = Some(grad_raw.clone());
                    }
                }
            }
        }

        // Update state. The EBE cache is adopted only when this eval improved on the
        // best objective seen; otherwise the incumbent's EBEs stay in place — kept
        // by skipping the `cached_etas` write, restored from the snapshot for the
        // mixture cache the branch above has already overwritten (#1290).
        let warm_start_improved = adopt_warm_start(guarded, ofv, state.best_ofv);
        let (mut ehs, mut hms, mut kappas) = (ehs, hms, kappas);
        if warm_start_improved {
            adopt_polished_modes(
                model,
                population,
                init_params,
                &x,
                &eta_dx.polished,
                &mut ehs,
                &mut hms,
                &mut kappas,
            );
            // The next evaluation starts every subject at this one's whole mode, κ̂ with
            // η̂, rather than restarting an IOV subject's κ at 0 (#2261) — except on an
            // ODE + IOV model, see `outer_cache_warm_starts` (#2508).
            state.cached_etas = outer_cache_warm_starts(model, &ehs, &kappas);
            state.cached_etas_kind = state.cached_etas_kind.after_pass(true);
            // The next evaluation predicts its warm start from this incumbent's mode
            // response; a gradient-free evaluation keeps the last response, re-anchored here.
            if warm_predictor {
                if let Some(j) = eta_dx.eta_dx.take() {
                    state.pred_jac = Some(j);
                    // The incumbent's exact inner Hessians go with its responses: the next
                    // warm solves' seeds where the policy seeds with the exact one.
                    state.seed_hessians = eta_dx.hessians.take();
                } else {
                    // The modes moved without a new response, and the carried seeds are
                    // the Hessians at the old ones: the next warm solves form their own
                    // (#2585 review #3). The response itself is kept, re-anchored.
                    state.seed_hessians = None;
                }
                state.pred_anchor.clone_from(&x);
            }
            // Publish the incumbent's whole EBE state for the final inner loop (#833).
            // The clone happens once per *improving* eval, not per eval, and `hms` is
            // moved into the (write-only) state cache afterwards rather than cloned
            // twice.
            *best_solve_cl.lock().unwrap() = Some(IncumbentSolve {
                xs: xs.to_vec(),
                etas: ehs,
                h_mats: hms.clone(),
                kappas: kappas.clone(),
                class_etas: if params.mixture.is_some() {
                    state.cached_etas_by_class.clone()
                } else {
                    Vec::new()
                },
                scores: eta_dx.scores.take(),
            });
            state.cached_h_mats = hms;
        } else {
            state.cached_h_mats = hms;
            if let Some(prev) = warm_start_by_class {
                state.cached_etas_by_class = prev;
            }
        }
        // #1520: record this evaluation for the salvage guard — the objective the
        // optimizer saw (`ofv`, the sentinel for a guarded evaluation, which the state
        // ignores because neither flag is set then), its per-subject decomposition, and
        // whether it was adopted as the fit's incumbent on the warm-start rule.
        state.salvage_guard.observe(xs, ofv, contribs, gradient_requested, warm_start_improved);
        state.n_evals += 1;
        n_evals_cl.fetch_add(1, Ordering::Relaxed);
        if ofv < state.best_ofv {
            state.best_ofv = ofv;
            if verbose {
                if guarded {
                    // A guarded evaluation's `ofv` is `guard_penalty_value` (or the `1e20`
                    // wall), not a likelihood: printed as an OFV it read as one (#2682).
                    let line = match stopped_after {
                        Some(solved) => stopped_eval_line(
                            &ebe_stats,
                            solved,
                            n_subj,
                            options.max_unconverged_frac,
                        ),
                        None => guarded_eval_line(
                            &ebe_stats,
                            n_subj,
                            raw_ofv,
                            options.max_unconverged_frac,
                        ),
                    };
                    eprintln!("Eval {:>4}: {line}", state.n_evals);
                } else if nn_reg.is_active() {
                    eprintln!(
                        "Eval {:>4}: OFV = {:.6} (penalized objective {:.6})",
                        state.n_evals, ofv_clean, ofv
                    );
                } else {
                    eprintln!("Eval {:>4}: OFV = {:.6}", state.n_evals, ofv);
                }
            }
        }
        // Record the *feasible*-eval index at which the feasible best OFV last
        // improved significantly (> `PLATEAU_OFV_THRESHOLD` below the last
        // recorded baseline). The gap between this and the feasible-eval count is
        // the flat-tail length — the plateau signal read after `optimize()`
        // (#751). Guarded evals are skipped entirely and do not advance the
        // feasible counter: their `guard_penalty_value` leaks into
        // `state.best_ofv`, so counting them would let a guard→feasible transition
        // masquerade as descent *and* let a tail of boundary probes pad the
        // plateau length. Feasible eval 1 sets the baseline; genuine progress must
        // land on a later feasible eval. Independent of the stagnation guard so it
        // is populated even when that guard is off. `ofv == raw_ofv` here.
        if !guarded {
            let mut pt = plateau_tracker_cl.lock().unwrap();
            pt.2 += 1; // one more feasible eval (1-based count / index)
            let feasible_idx = pt.2;
            if feasible_idx == 1 {
                // First feasible eval: establish the baseline objective. Not
                // "progress" — `last_sig_feasible_eval == 1` here.
                pt.0 = ofv;
                pt.1 = feasible_idx;
            } else if pt.0 - ofv > PLATEAU_OFV_THRESHOLD {
                pt.0 = ofv;
                pt.1 = feasible_idx;
            }
        }
        // `best_seen` tracks the global minimum across the whole run so the
        // final restore (issue #59) lands on the true best point, even when the
        // optimizer drifts away from it before terminating.
        //
        // #2462: a verification leg ranks its feasible evaluations in its own incumbent,
        // at its own tolerance; the leg copies it into `best_seen` when it ends. Mixing the
        // two tolerances in one tracker would keep a loose objective that is lower only by
        // its noise, and the next leg would restart from it.
        if in_verify_leg {
            if !guarded {
                leg_best.borrow_mut().observe(state.n_evals, xs, ofv, ofv_clean);
                if leg_first.get().is_none() {
                    leg_first.set(Some(ofv));
                }
            }
        } else {
            best_seen_cl.lock().unwrap().observe(state.n_evals, xs, ofv, ofv_clean);
        }
        // After updating best_ofv, check whether we've stalled. If yes,
        // `stagnation_stopped` is latched and the early-return at the
        // top of the closure trips on the next eval.
        let step = xs.iter().zip(&state.prev_x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let last_sig_feasible_eval = plateau_tracker_cl.lock().unwrap().1;
        // #1934: consulted only when a window has elapsed, so the per-subject gradient
        // pass it costs is paid once per would-be latch, not per evaluation. Measured at
        // the incumbent with the incumbent's own EBEs — the point a latch would report.
        let eval_no = state.n_evals;
        let still_descending = || {
            let incumbent = best_solve_cl.lock().unwrap();
            let Some(inc) = incumbent.as_ref() else {
                return false;
            };
            let x_inc: Vec<f64> = inc.xs.iter().zip(&scale).map(|(v, s)| v * s).collect();
            let gain = predicted_remaining_gain(
                model,
                population,
                init_params,
                &x_inc,
                &inc.etas,
                &inc.kappas,
                &bounds,
                options,
                &nn_reg,
                &priors,
            );
            let descending = !gain_is_negligible(gain);
            if descending {
                *last_veto_gain_cl.lock().unwrap() = gain;
            }
            if verbose && descending {
                eprintln!(
                    "Eval {:>4}: no OFV improvement above 1e-3 in the last window, but a \
                     Newton step from the best point is predicted to gain {:.4} more \
                     (BHHH Newton decrement, max {LATCH_MAX_PREDICTED_GAIN}) — not stopping.",
                    eval_no,
                    gain.unwrap_or(f64::NAN),
                );
            }
            descending
        };
        // A #2462 leg does not latch: its gains sit far below the guard's 1e-3 window, and
        // its own stop and the gain test are what end it.
        if !in_verify_leg
            && stagnation_after_eval(
                state,
                n_free,
                options.stagnation_guard,
                algo,
                gradient_run_has_reachable_stops(options),
                last_sig_feasible_eval,
                step,
                still_descending,
            )
        {
            stagnation_latched_cl.store(true, Ordering::Relaxed);
            latch_overrode_veto_cl.store(state.latch_overrode_veto, Ordering::Relaxed);
        }
        // #2093: a basin probe *during* the leg. The end-of-leg cold re-solve (#1565) is the
        // only place a warm-start trap was detected, so a leg minimising against EBEs some
        // subjects hold in a higher individual mode ran until NLopt itself gave up.
        // Measured on `examples/warfarin_dcm.ferx`: 302 evaluations crawling −311 → −314
        // while a cold solve at the incumbent scored −520, and the restart then spent the
        // rest of the budget. On a geometric schedule of feasible evaluations (32, 64,
        // 128, …: a logarithmic number of cold passes per leg) the same cold solve the
        // end-of-leg check runs is scored at the incumbent; when it beats the best seen by
        // that check's own margin ([`cold_solve_beats_best_seen`]), the leg stops through
        // the stagnation early-return and the end-of-leg check files the basin-shift
        // restart. A leg whose probe finds nothing continues on a bit-identical trajectory.
        // #2600: a gap carried by under `BASIN_PROBE_RESTART_SHARE` of the subjects does not
        // end the leg; their cold modes become the warm starts and the leg keeps its L-BFGS
        // memory ([`basin_probe_action`]).
        let mut basin_probe_stopped = false;
        if !state.stagnation_stopped && !guarded && init_params.mixture.is_none() {
            let feasible = plateau_tracker_cl.lock().unwrap().2;
            if basin_probe_due(feasible) {
                let incumbent = best_solve_cl.lock().unwrap().as_ref().map(|inc| inc.xs.clone());
                // The active leg's objective: inside a #2462 verification leg, its own
                // incumbent at its own tolerance, never `best_seen`'s loose one (#2600 review).
                let best = {
                    let leg = leg_best.borrow();
                    let seen = best_seen_cl.lock().unwrap();
                    basin_probe_reference(in_verify_leg, &leg, &seen)
                };
                let probe_options = eval_options;
                if let Some(xs_inc) = incumbent {
                    let x_inc: Vec<f64> = xs_inc.iter().zip(&scale).map(|(v, s)| v * s).collect();
                    let p = unpack_params(&x_inc, init_params);
                    let mu_k = compute_mu_k(model, &p.theta, probe_options.mu_referencing);
                    let (ehs, hms, cold_stats, kappas) = run_inner_loop_warm_seeded(
                        model,
                        population,
                        &p,
                        probe_options.inner_maxiter,
                        probe_options.inner_tol,
                        None,
                        Some(&mu_k),
                        probe_options.min_obs_for_convergence_check as usize,
                        probe_options.inner_restarts,
                        InnerHessianSeed::for_options(probe_options),
                        probe_options,
                        PrevEtas::Placeholder,
                    );
                    let cold = 2.0
                        * pop_nll_opts(model, population, &p, &ehs, &hms, &kappas, probe_options);
                    if let Some(gap) = cold_solve_beats_best_seen(Some(cold), best) {
                        // #2600: how many subjects the gap is, which decides between ending
                        // the leg and absorbing the cold modes into it.
                        let switched = if options.agq_nodes().is_none() {
                            best_solve_cl
                                .lock()
                                .unwrap()
                                .as_ref()
                                .filter(|inc| {
                                    inc.xs == xs_inc && inc.etas.len() == population.subjects.len()
                                })
                                .map(|inc| {
                                    subjects_in_a_lower_cold_mode(
                                        model,
                                        population,
                                        &p,
                                        (&inc.etas, &inc.h_mats, &inc.kappas),
                                        &ehs,
                                        &hms,
                                        &kappas,
                                        options.interaction,
                                    )
                                })
                        } else {
                            None
                        };
                        #[cfg(test)]
                        BASIN_PROBE_DECISIONS.with(|d| d.borrow_mut().push((gap, switched)));
                        // The same EBE guard an ordinary evaluation passes (#2600 review r3).
                        let cold_guard_rejected = ebe_guard_rejects(
                            &cold_stats,
                            population.subjects.len(),
                            cold,
                            options.max_unconverged_frac,
                        );
                        match basin_probe_action(
                            switched,
                            population.subjects.len(),
                            cold_guard_rejected,
                        ) {
                            BasinProbeAction::Restart => {
                                state.stagnation_stopped = true;
                                basin_probe_stopped = true;
                                basin_probe_fired.set(true);
                                if verbose {
                                    eprintln!(
                                        "Eval {:>4}: basin probe — a cold re-solve at the best \
                                         point scores {cold:.6}, {gap:.4} below the best seen \
                                         ({switched:?} subjects in a lower cold mode); ending \
                                         this leg for a basin-shift restart.",
                                        state.n_evals,
                                    );
                                }
                            }
                            BasinProbeAction::Absorb => {
                                // The cold solve is the incumbent now, in every tracker an
                                // evaluation's improvement is measured against.
                                let cold_penalized =
                                    cold + nn_reg.penalty_value(&p.theta) + priors.penalty(&x_inc);
                                let solve = IncumbentSolve {
                                    xs: xs_inc.clone(),
                                    etas: ehs,
                                    h_mats: hms,
                                    kappas,
                                    class_etas: Vec::new(),
                                    scores: None,
                                };
                                let mut leg_tracker = leg_best.borrow_mut();
                                let mut seen_tracker = best_seen_cl.lock().unwrap();
                                absorb_cold_incumbent(
                                    state,
                                    if in_verify_leg {
                                        &mut leg_tracker
                                    } else {
                                        &mut seen_tracker
                                    },
                                    &mut best_solve_cl.lock().unwrap(),
                                    &mut last_gradient_cl.lock().unwrap(),
                                    &mut plateau_tracker_cl.lock().unwrap(),
                                    (cold_penalized, cold),
                                    model,
                                    solve,
                                );
                                if verbose {
                                    eprintln!(
                                        "Eval {:>4}: basin probe — a cold re-solve at the best \
                                         point scores {cold:.6}, {gap:.4} below the best seen, \
                                         from {} of {} subjects; continuing this leg from \
                                         their cold modes.",
                                        state.n_evals,
                                        switched.unwrap_or(0),
                                        population.subjects.len(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        // The basin probe prints its own reason; this one would misstate it.
        if state.stagnation_stopped && verbose && !basin_probe_stopped {
            eprintln!(
                "Eval {:>4}: stopping early — no OFV improvement above 1e-3 in the \
                 last window. Whether this is convergence is decided after the \
                 final inner loop (plateau and cold-restart consistency check).",
                state.n_evals,
            );
        }

        // Optimizer trace (step_norm in scaled space)
        if crate::estimation::trace::is_active() {
            let step_norm = if step > 0.0 { Some(step) } else { None };
            let method_str = crate::estimation::trace::method_label(options);
            let optimizer_str = match algo {
                nlopt::Algorithm::Bobyqa => "bobyqa",
                nlopt::Algorithm::Mma => "mma",
                nlopt::Algorithm::Lbfgs => "nlopt_lbfgs",
                _ => "slsqp",
            };
            let values = crate::estimation::parameterization::coordinate_values(&params);
            crate::estimation::trace::write_foce(
                state.n_evals,
                method_str,
                ofv_clean,
                grad_norm_for_trace,
                step_norm,
                optimizer_str,
                Some(ebe_stats.n_unconverged),
                Some(ebe_stats.n_fallback),
                &values,
                grad_vec_for_trace.as_deref(),
            );
        }

        // Checkpoint (#755): unscale the *best-seen* point only when a write is
        // due. The objective runs once per eval, so gate on `is_due` to avoid a
        // per-eval allocation on the (default) no-checkpoint-due path. Writing
        // the best point rather than this eval's is what stops a line-search
        // probe that happens to land on the due window from being recorded as
        // where the fit is (#1317) — `best_seen` is updated just above, so it is
        // never empty here, and it is the same tracker the #59 restore reads.
        //
        // #1617 review round 7: and only once something feasible has been *seen*. Until
        // then `best_seen` ranks `guard_penalty_value`s, so its minimum is the scaled
        // midpoint of the declared bounds — and a checkpoint is a *starting point* for the
        // next run, which is the one place the exit-time fallback below cannot help: resume
        // from such a file and the midpoint becomes the resumed fit's own start, so the
        // fallback faithfully restores it and #1617's point defect survives the interrupt.
        // Writing nothing is the right answer rather than writing the optimizer start,
        // because an absent file means "resume from the model's initial estimates", which is
        // exactly where a run that has found nothing should begin. `is_due()` is tested
        // first: it is `false` whenever no sink is active, so the default path does not take
        // the plateau lock.
        if crate::io::checkpoint::is_due() && plateau_tracker_cl.lock().unwrap().2 > 0 {
            best_seen_cl
                .lock()
                .unwrap()
                .write_checkpoint(|best_xs| (0..n).map(|i| best_xs[i] * scale[i]).collect());
        }

        state.prev_x = xs.to_vec();

        ofv
    };

    // Create NLopt optimizer with state (operates in scaled xs space)
    // `&objective`, not `objective`: the #1782 polish leg below runs a second NLopt
    // instance on the same closure, and so on the same best-seen tracker and counters.
    let mut opt = nlopt::Nlopt::new(algo, n, &objective, nlopt::Target::Minimize, state);
    opt.set_lower_bounds(&lower_s).unwrap();
    opt.set_upper_bounds(&upper_s).unwrap();
    // The budget this leg may spend: the fit's, less what a stalled leg it
    // continues already used. `.max(1)`, never `0`: NLopt reads `maxeval = 0` as
    // unlimited, and `optimize_nlopt` does not restart on an empty budget anyway.
    let full_budget = outer_eval_budget(algo, n, options.outer_maxiter);
    #[cfg(test)]
    let forced_stop = forced_stop_at_start::get(&model.name, restart.is_none());
    #[cfg(test)]
    let full_budget = EVAL_BUDGET_OVERRIDE
        .with(|o| o.get())
        .or_else(|| forced_stop.filter(|s| s.whole_fit).map(|_| 1))
        .unwrap_or(full_budget);
    let max_eval = full_budget.saturating_sub(spent_evals as u32).max(1);
    // A forced stop starves this leg alone; the fit's budget is left for later legs.
    #[cfg(test)]
    let max_eval = if forced_stop.is_some() { 1 } else { max_eval };
    if matches!(algo, nlopt::Algorithm::Bobyqa) {
        opt.set_maxeval(max_eval).unwrap();
        // BOBYQA's xtol_rel controls rho_end / rho_start — i.e. how much
        // it must shrink the trust radius to declare success. 1e-12 is
        // unreachable in any realistic budget and forces MaxevalReached
        // at an arbitrary interim point; the default xtol 1e-4 in scaled
        // log-space is a ~0.01% move in the natural-scale parameter, which
        // is plenty tight for NLME work.
        opt.set_xtol_rel(options.outer_xtol).unwrap();
        // ftol_rel is the objective-change stop; see `resolve_outer_ftol` for the
        // `None` auto-selection (1e-8 pure non-Gaussian, 1e-6 otherwise) and #469 rationale.
        let ftol =
            resolve_outer_ftol(model.has_non_gaussian(), model.is_ode_based(), options.outer_ftol);
        opt.set_ftol_rel(ftol).unwrap();
        // Not NLopt's default rhobeg — see `bobyqa_initial_step`.
        opt.set_initial_step(&bobyqa_initial_step(&lower_s, &upper_s)).unwrap();
    } else {
        opt.set_maxeval(max_eval).unwrap();
        if gradient_run_has_reachable_stops(options) {
            // The quadrature stops where its objective settles — the reachable
            // objective-change / step-size criteria BOBYQA gets — rather than on FOCE/FOCEI's
            // unreachable 1e-12 stops, stagnation guard and verification legs. This began as
            // a noise floor (the grid response and anchor Hessian were central differences);
            // the gradient is exact now, and it stays as a measured cost/accuracy choice
            // (#2549): giving Laplace/AGQ the FOCEI regime cost 1.15–2.9× on the `pk1` fits
            // (ODE AGQ-3 222 G → 651 G instructions) for ≤1e-5 OFV, and 7× on `pk2_big` Laplace
            // for 0.10. What decides a hard quadrature fit is the EBE warm-start predictor
            // (`warm_predictor_serves`): with it `pk2_big` Laplace reaches −3485.11 under these
            // stops, against −3458.81 without.
            opt.set_xtol_rel(options.outer_xtol).unwrap();
            let ftol = resolve_outer_ftol(
                model.has_non_gaussian(),
                model.is_ode_based(),
                options.outer_ftol,
            );
            opt.set_ftol_rel(ftol).unwrap();
        } else {
            // FOCE objective is noisy from EBE re-estimation; let maxeval be the primary
            // stopping criterion and rely on the analytic gradient to drive |g| down.
            opt.set_xtol_rel(1e-12).unwrap();
            opt.set_ftol_rel(1e-12).unwrap();
        }
    }

    if options.verbose {
        eprintln!("Starting NLopt {:?} optimization ({} parameters)...", algo, n);
    }

    // Run optimization (the first leg is the `drive_polish_legs` argument below).

    // #1782: BOBYQA's default stop is *relative* (`ftol_rel = 1e-6`, `xtol_rel = 1e-4`),
    // and on its own it is not evidence of a minimum — at |OFV| ≈ 430 a 4e-4 step
    // satisfies it while the trace is still falling. Measured on `two_cpt_iv` (−432.0317
    // against the −432.1668 L-BFGS reaches, macOS) and `warfarin_iov` (+1.53 OFV on
    // Linux), each reporting `converged: YES`. So a tolerance stop is not the end of the
    // run: restart BOBYQA from the best point seen under tight tolerances, which rebuilds
    // its interpolation model at a fresh trust radius, and let the *last* leg's stop
    // decide `converged`. One leg is not always enough — on `warfarin_iov` (macOS) the
    // first polish ended 2.4e-3 above the optimum on `XtolReached` — so a leg that still
    // bought more than `POLISH_MIN_GAIN` is followed by another, up to `MAX_POLISH_LEGS`.
    // Every leg feeds the same best-seen tracker, so the restore below picks the lowest
    // point whichever leg found it, and the legs share the run's budget.
    let best_point = || best_seen.lock().unwrap().get().map(|b| (b.x.clone(), b.ofv));
    // The tolerances every polish leg runs at: the `POLISH_*` ceilings, or the user's
    // own when tighter. Bound once so the setters and the verbose line cannot disagree.
    let polish_xtol = options.outer_xtol.min(POLISH_XTOL_REL);
    let polish_ftol =
        resolve_outer_ftol(model.has_non_gaussian(), model.is_ode_based(), options.outer_ftol)
            .min(POLISH_FTOL_REL);
    let first_leg = opt.optimize(&mut x0);
    #[cfg(test)]
    FIRST_LEG_EVALS.with(|c| c.set(Some(n_evals_outer.load(Ordering::Relaxed))));
    let mut opt_slot = Some(opt);
    // `POLISH_MIN_GAIN`, which a test can lower to force the leg cap.
    #[cfg(not(test))]
    let polish_min_gain = POLISH_MIN_GAIN;
    #[cfg(test)]
    let polish_min_gain = POLISH_MIN_GAIN_OVERRIDE.with(|o| o.get()).unwrap_or(POLISH_MIN_GAIN);
    let budget_left = || {
        let used = spent_evals + n_evals_outer.load(Ordering::Relaxed);
        full_budget.saturating_sub(used as u32)
    };
    // #2462: an unprofiled FOCE/FOCEI gradient stop is not evidence of a minimum either.
    // At the default `inner_tol` the EBE re-solve noise leaves L-BFGS no step it can
    // certify on an ill-conditioned objective, so it stops — on a bare `Failure` the
    // plateau rule accepts, or on a tolerance status — 1e-4 above the optimum. A leg at
    // the fit's own settings gains exactly 0 there. So each stop that would be reported
    // converged gets a leg from the best point at a tight inner tolerance
    // (`UNPROFILED_VERIFY_INNER_TOL`), with its own incumbent at that tolerance, the latch
    // cleared, and a cold EBE seed for its first evaluation; it settles on
    // `unprofiled_leg_settles`. The cold seed matters: from its own modes, a subject's
    // inner BFGS at the incumbent can fail to certify a step and fall back to Nelder–Mead,
    // which scored A2's (55, 5, 3.2) start 1.6e-9 below every neighbouring point, more than
    // L-BFGS's first step could gain there, and that leg ended `Failure` with no gain.
    let latched_at_stop = stagnation_latched.load(Ordering::Relaxed);
    let verified_needs_left_init = verified_stop_needs_left_init(&first_leg, latched_at_stop);
    // The verified stop's cold solve and the point it was taken at, kept for the final cold
    // re-solve at the restored point: a deterministic function of its inputs, so at the same
    // point it is this one, not another descent from η = 0 per subject.
    let cold_kept = std::cell::RefCell::new(None::<(Vec<f64>, FinalInnerSolve)>);
    // The best leg incumbent copied into `best_seen` so far (#2507), at the leg tolerance.
    let mut kept_leg_best: Option<f64> = None;
    // `kept_leg_best` as the code around the legs reads it, while they hold the original.
    let kept_best_seen = std::cell::Cell::new(None::<f64>);
    // The optimizer gradient and the whole EBE state (`best_solve`) at that incumbent
    // ([`keep_verify_leg`]).
    let mut kept_leg_state: Option<(Option<Vec<f64>>, Option<IncumbentSolve>)> = None;
    let (result, polish_end, verified_stop, last_net) = if verify_on {
        let cancelled = crate::cancel::is_cancelled(&options.cancel);
        // Measured on the incumbent the leg starts from, as `left_init` below is.
        let left_init_at_stop = best_point()
            .filter(|(x, _)| x.len() == n)
            .is_some_and(|(x, _)| max_scaled_deviation(&x, &x0_start_s) >= INIT_ESCAPE_STEP_S);
        let plateau_shape = {
            let (_, last_sig, feasible) = *plateau_tracker.lock().unwrap();
            plateau_trace_shape(feasible, last_sig)
        };
        let first_needs_verify = !basin_probe_fired.get()
            && unprofiled_stop_needs_verify(
                &first_leg,
                cancelled,
                left_init_at_stop,
                plateau_shape,
            );
        // Read live: a cancel during a leg ends the legs there.
        let leg_needs_verify = |r: &NloptReturn| {
            !basin_probe_fired.get()
                && unprofiled_stop_needs_verify(
                    r,
                    crate::cancel::is_cancelled(&options.cancel),
                    left_init_at_stop,
                    true,
                )
        };
        // The cold solve at the verified stop: judges the stop, then seeds the cold round.
        let cold_at_stop = std::cell::RefCell::new(None::<FinalInnerSolve>);
        // Set for the first leg of a pass ([`leg_pass_applies`]).
        let next_start = std::cell::Cell::new(LegStart::Incumbent);
        let mut run_leg = |left: u32| {
            if plateau_tracker.lock().unwrap().2 == 0 {
                return None;
            }
            let (best_x, before) = best_point().filter(|(x, _)| x.len() == n)?;
            // The point and modes the leg starts at: the incumbent and its own modes when the
            // held state is this point's. A pass needs that held state to compare against.
            let mut start_x = best_x.clone();
            let held = match next_start.replace(LegStart::Incumbent) {
                LegStart::Incumbent => best_solve_cl
                    .lock()
                    .unwrap()
                    .as_ref()
                    .filter(|inc| inc.xs == best_x)
                    .map(|inc| outer_cache_warm_starts(model, &inc.etas, &inc.kappas)),
                LegStart::ColdModes => {
                    let x_inc: Vec<f64> = best_x.iter().zip(&scale).map(|(v, s)| v * s).collect();
                    let p = unpack_params(&x_inc, init_params);
                    let cold = cold_at_stop.take()?;
                    let incumbent = best_solve_cl.lock().unwrap();
                    let inc = incumbent.as_ref().filter(|inc| inc.xs == best_x)?;
                    let terms = |etas: &[DVector<f64>],
                                 h_mats: &[DMatrix<f64>],
                                 kappas: &[Vec<DVector<f64>>]| {
                        subject_nlls(
                            model,
                            population,
                            &p,
                            etas,
                            h_mats,
                            kappas,
                            options.interaction,
                        )
                    };
                    let moved = subjects_off_the_held_modes(
                        &terms(&inc.etas, &inc.h_mats, &inc.kappas),
                        &terms(&cold.etas, &cold.h_mats, &cold.kappas),
                    );
                    if verbose {
                        eprintln!(
                            "Cold solve at the verified incumbent: {moved} of {} subjects off \
                             its modes.",
                            population.subjects.len(),
                        );
                    }
                    if moved == 0 {
                        return None;
                    }
                    Some(outer_cache_warm_starts(model, &cold.etas, &cold.kappas))
                }
                LegStart::NewtonStep => {
                    let incumbent = best_solve_cl.lock().unwrap();
                    let inc = incumbent.as_ref().filter(|inc| inc.xs == best_x)?;
                    let x_inc: Vec<f64> = best_x.iter().zip(&scale).map(|(v, s)| v * s).collect();
                    let (gain, step) = predicted_newton_step(
                        model,
                        population,
                        init_params,
                        &x_inc,
                        &inc.etas,
                        &inc.kappas,
                        &bounds,
                        options,
                        &nn_reg,
                        &priors,
                    )?;
                    if verbose {
                        eprintln!(
                            "Newton step from the verified incumbent: predicted gain {gain:.4} \
                             (a leg runs above {LATCH_MAX_PREDICTED_GAIN})."
                        );
                    }
                    if !newton_gain_warrants_a_leg(gain, population.subjects.len(), &step) {
                        return None;
                    }
                    start_x = newton_start(&x_inc, &step, &bounds)
                        .iter()
                        .zip(&scale)
                        .map(|(v, s)| v / s)
                        .collect();
                    Some(outer_cache_warm_starts(model, &inc.etas, &inc.kappas))
                }
            };
            let mut state = opt_slot.take()?.recover_user_data();
            x0.copy_from_slice(&start_x);
            state.prev_x = start_x;
            // Unlatched: a latched state answers every evaluation with the incumbent
            // and a zero gradient, and the latch carries into the recovered state.
            state.stagnation_stopped = false;
            state.recent_steps.clear();
            stagnation_latched.store(false, Ordering::Relaxed);
            // The leg's own incumbent, so its first evaluation is adopted as the
            // warm start whatever the loose objective was.
            state.best_ofv = f64::INFINITY;
            state.last_improvement_eval = state.n_evals;
            state.best_at_last_improvement = f64::INFINITY;
            // The leg's first evaluation re-solves the incumbent's own modes at the leg's
            // tolerance: a verification leg tightens the inner solve at the point it
            // starts from, so its inner solves start at the modes of that point (#2549).
            // A cold start re-selected every mode at `UNPROFILED_VERIFY_INNER_TOL`: on the
            // ODE twin of `transit_flip_flop_fit_matches_ode_twin` it left one of four
            // subjects unconverged, the EBE guard rejected the leg's start, and the leg
            // walked to the TVNTR floor (−59.31 against the incumbent's −228.59). No mode
            // response carries over: it belongs to the modes of the evaluation it was
            // taken at, and the leg's first point needs none (Δx = 0). Cold only when the
            // held state is another point's. The cold-mode pass starts its first leg at a
            // cold solve's modes the same way, so that leg's first evaluation is a warm
            // re-solve of them and a subject the cold solve left unconverged is not what
            // the guard judges (#2645).
            state.restart_from_modes(held);
            // A mixture's classes start at the incumbent's own modes too, which
            // [`keep_verify_leg`] put back if the last leg was dropped (#2837 r1 #5); cold when
            // the held state is another point's.
            state.cached_etas_by_class = best_solve_cl
                .lock()
                .unwrap()
                .as_ref()
                .filter(|inc| inc.xs == best_x)
                .map(|inc| inc.class_etas.clone())
                .unwrap_or_default();
            #[cfg(test)]
            if init_params.mixture.is_some() {
                MIXTURE_LEG_START_MODES
                    .with(|m| m.borrow_mut().push(state.cached_etas_by_class.clone()));
            }
            *leg_best.borrow_mut() = BestPoint::new();
            leg_first.set(None);
            let mut leg = nlopt::Nlopt::new(algo, n, &objective, nlopt::Target::Minimize, state);
            leg.set_lower_bounds(&lower_s).unwrap();
            leg.set_upper_bounds(&upper_s).unwrap();
            leg.set_maxeval(left).unwrap();
            // The first leg's own stops (`verify_on` excludes the reachable ones).
            leg.set_xtol_rel(1e-12).unwrap();
            leg.set_ftol_rel(1e-12).unwrap();
            in_leg.set(true);
            let r = leg.optimize(&mut x0);
            in_leg.set(false);
            opt_slot = Some(leg);
            let best = leg_best.borrow().get().map(|b| b.ofv);
            let gain = leg_first.get().zip(best).map_or(f64::NAN, |(f, b)| f - b);
            // The first leg's incumbent replaces the loose one: the two are at different
            // tolerances, and the loose one can be lower by its noise alone. A later leg's
            // replaces it only when lower; otherwise the kept leg's gradient and EBE state
            // go back in, which this leg's improving evaluations overwrote
            // ([`keep_verify_leg`]).
            let mut live =
                (last_gradient.lock().unwrap().take(), best_solve.lock().unwrap().take());
            #[cfg(test)]
            let judged = best.filter(|_| {
                kept_leg_best.is_none() || !UNPROFILED_VERIFY_DROP_LATER_LEGS.with(|c| c.get())
            });
            #[cfg(not(test))]
            let judged = best;
            if keep_verify_leg(&mut kept_leg_best, &mut kept_leg_state, judged, &mut live) {
                *best_seen.lock().unwrap() = leg_best.borrow().clone();
            }
            kept_best_seen.set(kept_leg_best);
            *last_gradient.lock().unwrap() = live.0;
            *best_solve.lock().unwrap() = live.1;
            let outcome = UnprofiledLeg { ret: r, gain, best, first: leg_first.get() };
            #[cfg(test)]
            UNPROFILED_VERIFY_LEGS.with(|v| v.borrow_mut().push(outcome));
            if options.verbose {
                eprintln!(
                    "Verification leg from OFV {before:.9} at inner_tol = {:e}: {:?}, \
                         gain {gain:.3e}, best {best:?}",
                    leg_options.inner_tol, outcome.ret,
                );
            }
            Some(outcome)
        };
        let warm = unprofiled_verify_legs(
            first_leg,
            first_needs_verify,
            leg_needs_verify,
            &budget_left,
            &mut run_leg,
        );
        drive_leg_passes(
            warm,
            |v| v.1,
            || crate::cancel::is_cancelled(&options.cancel),
            PassState {
                basin_probe_fired: &basin_probe_fired,
                stagnation_latched: &stagnation_latched,
                kept_best: &kept_best_seen,
                mixture: init_params.mixture.is_some(),
            },
            || {
                // Clean, as the final check reads it: the cold objective carries no penalty.
                let incumbent = best_seen.lock().unwrap().get().map(|b| (b.x.clone(), b.ofv_clean));
                let Some((best_x, best)) = incumbent.filter(|(x, _)| x.len() == n) else {
                    return false;
                };
                let x_inc: Vec<f64> = best_x.iter().zip(&scale).map(|(v, s)| v * s).collect();
                let p = unpack_params(&x_inc, init_params);
                let mu_k = compute_mu_k(model, &p.theta, options.mu_referencing);
                let Some(cold) = cold_modes_solve(model, population, &p, &mu_k, options) else {
                    return false;
                };
                let cold_ofv = 2.0
                    * pop_nll_opts(
                        model,
                        population,
                        &p,
                        &cold.etas,
                        &cold.h_mats,
                        &cold.kappas,
                        options,
                    );
                let owed = verified_stop_owes_a_restart(best, cold_ofv);
                if verbose && owed {
                    eprintln!(
                        "Verified stop at OFV {best:.6}, cold {cold_ofv:.6}: left to its \
                         restart, no further rounds."
                    );
                }
                *cold_kept.borrow_mut() = Some((x_inc, cold.clone()));
                *cold_at_stop.borrow_mut() = Some(cold);
                !owed
            },
            |start, verified| {
                next_start.set(start);
                unprofiled_verify_legs(
                    verified.0,
                    true,
                    leg_needs_verify,
                    &budget_left,
                    &mut run_leg,
                )
            },
        )
    } else {
        // The predicted gain behind the last decision to polish a quadrature stop; `None`
        // once a stop needs no polish. A leg that then settles leaves it standing.
        let stalled_gain = std::cell::Cell::new(None::<f64>);
        let (r, end) = drive_polish_legs(
            first_leg,
            polish_min_gain,
            |r| {
                stalled_gain.set(None);
                let reachable = gradient_run_has_reachable_stops(options);
                if !tolerance_stop_needs_polish(
                    algo,
                    reachable,
                    r,
                    stagnation_latched.load(Ordering::Relaxed),
                    crate::cancel::is_cancelled(&options.cancel),
                ) {
                    return false;
                }
                if !reachable {
                    return true;
                }
                // A leg the #2093 basin probe ended belongs to the basin-shift restart, not
                // to a polish (as for the #2462 verification legs).
                if basin_probe_fired.get() {
                    return false;
                }
                // A quadrature stop is polished where a Newton step from the best point is
                // predicted to gain more than a polish leg needs to continue
                // (`POLISH_MIN_GAIN`): the BHHH decrement of its exact per-subject scores. On
                // an easy fit, already at its optimum to 1e-5, it reads far below and the
                // stop stands; a slow creep reads above it (#2549). Where the decrement cannot
                // be formed the stop is polished.
                let incumbent = best_solve.lock().unwrap();
                let Some(inc) = incumbent.as_ref() else {
                    return true;
                };
                let x_inc: Vec<f64> = inc.xs.iter().zip(&scale).map(|(v, s)| v * s).collect();
                let gain = quadrature_predicted_remaining_gain(
                    model,
                    population,
                    init_params,
                    &x_inc,
                    &inc.etas,
                    &inc.kappas,
                    inc.scores.as_deref(),
                    &inc.class_etas,
                    &schedule_cache,
                    &bounds,
                    options,
                    &nn_reg,
                    &priors,
                );
                if verbose {
                    eprintln!(
                        "Quadrature tolerance stop: a Newton step from the best point is \
                         predicted to gain {} (polished above {polish_min_gain:e}).",
                        gain.map_or("an amount that cannot be formed".to_string(), |g| format!(
                            "{g:.3e}"
                        ))
                    );
                }
                let polish = gain.is_none_or(|g| g > polish_min_gain);
                if polish {
                    stalled_gain.set(gain);
                }
                polish
            },
            budget_left,
            |left| {
                // Nothing feasible seen means the best point is the guard penalty's minimum
                // (#1617) — not a place to polish from; the verdict below reports that run.
                if plateau_tracker.lock().unwrap().2 == 0 {
                    return None;
                }
                let (best_x, before) = best_point().filter(|(x, _)| x.len() == n)?;
                let mut state = opt_slot.take()?.recover_user_data();
                x0.copy_from_slice(&best_x);
                state.prev_x = best_x;
                // A full stagnation window for the leg: the previous leg's flat tail is
                // what ended it, and must not latch this one on its first eval.
                state.last_improvement_eval = state.n_evals;
                state.best_at_last_improvement = state.best_ofv;
                let mut leg =
                    nlopt::Nlopt::new(algo, n, &objective, nlopt::Target::Minimize, state);
                leg.set_lower_bounds(&lower_s).unwrap();
                leg.set_upper_bounds(&upper_s).unwrap();
                leg.set_maxeval(left).unwrap();
                leg.set_xtol_rel(polish_xtol).unwrap();
                leg.set_ftol_rel(polish_ftol).unwrap();
                if matches!(algo, nlopt::Algorithm::Bobyqa) {
                    leg.set_initial_step(&bobyqa_initial_step(&lower_s, &upper_s)).unwrap();
                }
                if options.verbose {
                    eprintln!(
                        "{algo:?} stopped on its tolerance (best objective = {before:.6}); polishing \
                     from the best point with ftol_rel = {polish_ftol:e}, \
                     xtol_rel = {polish_xtol:e} ({left} evaluations left).",
                    );
                }
                let r = leg.optimize(&mut x0);
                opt_slot = Some(leg);
                Some((r, best_point().map_or(0.0, |(_, after)| before - after)))
            },
        );
        // A leg that gained nothing from a stop whose Newton decrement still predicts more
        // than a leg needs to continue has not verified it: the objective is not smooth
        // there (#2585).
        let end = quadrature_polish_end(end, stalled_gain.get(), &r);
        (r, end, false, None)
    };
    let opt = opt_slot;
    #[cfg(test)]
    let (result, polish_end, verified_stop) = match forced_stop.and_then(|stop| stop.status) {
        Some(status) => (Ok((status, f64::NAN)), PolishEnd::NotNeeded, false),
        None => (result, polish_end, verified_stop),
    };

    // `max_eval_reached` distinguishes a spent evaluation budget from other
    // non-convergence: it gets its own warning ("increase maxiter") rather than
    // the generic "did not converge" message.
    let mut max_eval_reached = false;
    // A bare NLopt `Failure`/`ForcedStop` is ambiguous: it is returned both by a
    // genuine mid-descent stall *and* by the analytic-gradient L-BFGS default
    // (#639) settling onto a plateaued optimum whose ∇ has dropped below the
    // floor its line search can beat. We defer that verdict — see
    // `stationarity_check_pending` — and resolve it below on the OFV trace: the
    // feasible-eval plateau length plus a cold-restart self-consistency check at
    // the restored best point. (The analytic gradient norm is deliberately *not*
    // used — it still reads O(1) at these genuine optima; the resolution block
    // explains why.)
    let mut stationarity_check_pending = false;
    // #2785: NLopt's own word, published beside the verdict below, which can overrule it in
    // either direction.
    let optimizer_stop = Some(nlopt_stop_label(&result));
    let mut converged = match &result {
        Ok((status, _)) => {
            if options.verbose {
                eprintln!("NLopt finished: {:?}", status);
            }
            max_eval_reached = matches!(status, nlopt::SuccessState::MaxEvalReached);
            matches!(
                status,
                nlopt::SuccessState::Success
                    | nlopt::SuccessState::FtolReached
                    | nlopt::SuccessState::XtolReached
                    | nlopt::SuccessState::StopValReached
            )
        }
        Err((fail, _)) => {
            if options.verbose {
                eprintln!("NLopt stopped: {:?}", fail);
            }
            match fail {
                nlopt::FailState::RoundoffLimited => true,
                nlopt::FailState::Failure | nlopt::FailState::ForcedStop => {
                    stationarity_check_pending = true;
                    false
                }
                _ => false,
            }
        }
    };
    // #1782: a polish that could not finish is not a converged fit, whatever the last
    // NLopt status says — the tolerance stop it was meant to verify still stands
    // unverified. A spent budget gets the ordinary "increase maxiter" advice below.
    //
    // #2462, the same for an unprofiled gradient stop: a leg cap or a spent budget is not
    // converged, and not a plateau for the check below to accept either. A stop a leg
    // verified is converged whatever the last leg's status, `Failure` included, once the
    // cold re-solve below reproduces it; that check is the plateau rule's own, so it is
    // deferred the same way.
    match polish_end {
        PolishEnd::BudgetSpent => {
            converged = false;
            max_eval_reached = true;
            stationarity_check_pending = false;
        }
        PolishEnd::LegCapReached { last_gain } => {
            converged = false;
            stationarity_check_pending = false;
            warnings.push(if verify_on {
                unprofiled_verify_cap_warning(last_gain, last_net)
            } else {
                polish_leg_cap_warning(last_gain)
            });
        }
        PolishEnd::Stalled { predicted } => {
            converged = false;
            stationarity_check_pending = false;
            warnings.push(polish_stalled_warning(predicted));
        }
        PolishEnd::NotNeeded | PolishEnd::Settled | PolishEnd::Skipped => {}
    }
    if verified_stop {
        converged = false;
        max_eval_reached = false;
        stationarity_check_pending = true;
    }
    let latched = stagnation_latched.load(Ordering::Relaxed);
    if stop_needs_plateau_check_after_legs(
        verify_on,
        polish_end,
        converged,
        max_eval_reached,
        latched,
    ) {
        // The guard, not the budget, ended the run — even when its latch landed on
        // the last permitted eval — so no "increase maxiter" warning either.
        converged = false;
        max_eval_reached = false;
        stationarity_check_pending = true;
    }

    drop(opt);

    // #1617: what this run's evaluation trace says about whether it ever stood on
    // feasible ground. Read here, as soon as `opt` is dropped and the counters are final,
    // because three separate decisions below key on it: whether the budget advice makes
    // any sense (immediately below), which point the best-seen restore reports, and the
    // verdict plus its warning. The cancellation field is not belt-and-braces — a
    // cancelled run returns `1e20` from the top of the objective *before* the feasible
    // counter advances, so its trace is infeasible by construction and says nothing about
    // the model (see [`RunFeasibility::no_feasible`]).
    let feasibility = {
        let feasible_evals = plateau_tracker.lock().unwrap().2;
        let ebe_now = ebe_accum.lock().unwrap();
        RunFeasibility {
            feasible_evals,
            n_guard_rejected: ebe_now.n_convergence_warnings,
            n_nonfinite: ebe_now.n_nonfinite,
            max_unconverged: ebe_now.max_unconverged,
            n_subj,
            max_unconverged_frac: options.max_unconverged_frac,
            inner_maxiter: options.inner_maxiter,
            cancelled: crate::cancel::is_cancelled(&options.cancel),
        }
    };

    // Whether "increase `maxiter`" is advice this run can act on (#1617 review). A run
    // with no feasible evaluation cannot: there is no fit to tighten, and a longer *outer*
    // budget buys only more rejected trials. `W_NO_FEASIBLE_EVAL` names the budget that
    // actually binds — `inner_maxiter`, or `max_unconverged_frac` — so offering the outer
    // knob beside it sends the reader at the wrong one. Only the *advice* is withheld:
    // `max_eval_reached` itself still reads true, so `latched_stop_needs_plateau_check`
    // above and the second arm's own conditions are unchanged.
    //
    // One binding read by **both** budget warnings rather than the same condition written
    // twice. The `MaxEvalReached` arm just below is reachable (pinned by
    // `a_run_with_no_feasible_evaluation_is_not_told_to_raise_maxiter`); the bare
    // `Failure`-with-spent-budget arm further down is not: it needs NLopt to return a bare
    // `Failure`, and the guard penalty is an exact quadratic whose value and gradient agree
    // (#486), so no line search fails on it. Measured with this gate removed (`slsqp` /
    // `nlopt_lbfgs` / `mma`, three outer budgets, all-rejected trace): every run ended in a
    // *success* state with its budget exactly spent. Written twice, the unreachable copy would
    // be an independently breakable gate no test can reach.
    let budget_advice_applies = !feasibility.no_feasible();

    // A spent evaluation budget gets a targeted "increase maxiter" warning; every
    // other non-convergence falls through to the generic "did not converge"
    // warning below. There is no automatic second optimization — a user who wants
    // SLSQP sets `optimizer = slsqp` as the primary (issue #657).
    if max_eval_reached && budget_advice_applies {
        warnings.push(eval_budget_warning(options.outer_maxiter));
        if options.verbose {
            eprintln!(
                "NLopt hit the evaluation budget (maxiter = {}) without converging — \
                 increase maxiter for a tighter fit.",
                options.outer_maxiter,
            );
        }
    }

    // Restore the best-seen point (issue #59). NLopt returns the last
    // evaluated `x0`, not the best-seen one — when the stagnation guard
    // short-circuits, the last few evals return `best_ofv` with zero
    // gradient and the optimizer can drift off the true minimum before
    // termination. Replacing `x0` with the best-seen xs guarantees the
    // final inner loop and covariance step run at the actual minimum.
    // `best_seen_ofv` is the *clean* −2LL at the restored point: it is compared
    // against the equally clean `final_ofv` in the plateau self-consistency
    // check below, and mixing a penalized best with a clean final would loosen
    // that check by exactly the penalty magnitude.
    //
    // #1617: unless there was nothing feasible to rank (`feasibility`, read above). Every
    // evaluation of such a run was scored by `guard_penalty_value`, whose minimum is the
    // scaled midpoint of the declared bounds, so the best-seen point is the box centre — a
    // place the optimizer reached because the *penalty* is stationary there, not the
    // objective. Restoring it dresses a failed run as a fit at a specific wrong estimate
    // (the issue's warfarin run matched `sqrt(lower·upper)` in every θ to 3e-9, with
    // `converged = true`). Fall back to the point this leg started from, and say so in
    // `warnings` below. Leaving `best_seen_ofv` at `None` matters too: the #833 candidate
    // check would otherwise compare the finite re-score against a `1e12` penalty and call
    // that a reproduction.
    let mut best_seen_ofv: Option<f64> = None;
    if feasibility.no_feasible() {
        x0.copy_from_slice(&optimizer_start_s);
        if options.verbose {
            eprintln!(
                "No feasible evaluation in {} outer trials — reporting the starting \
                 estimates rather than the guard penalty's minimum.",
                n_evals_outer.load(Ordering::Relaxed),
            );
        }
    } else if let Some(best) = best_seen.lock().unwrap().get() {
        if best.x.len() == n {
            x0.copy_from_slice(&best.x);
            best_seen_ofv = Some(best.ofv_clean);
            if options.verbose {
                eprintln!(
                    "Restored best-seen point (OFV = {:.6}) for final inner loop \
                     and covariance step.",
                    best.ofv_clean,
                );
            }
        }
    }

    // Did the fit leave its initial estimates? Measured on the restored best
    // point (still in scaled space here), which is what the reported estimates
    // and the covariance step are built from. Feeds both the plateau verdict
    // below and the stall retry in `optimize_nlopt`.
    let left_init = max_scaled_deviation(&x0, &x0_start_s) >= INIT_ESCAPE_STEP_S;
    // The verdict that leaves this function on `OuterResult`, measured against
    // `escape_from` when the caller supplied one. Identical to `left_init` on every
    // path but the #1277 restart, whose own start is a previous attempt's estimates
    // rather than the user's initial values.
    let published_left_init = match escape_from {
        None => left_init,
        Some(_) => max_scaled_deviation(&x0, &escape_start_s) >= INIT_ESCAPE_STEP_S,
    };

    // The restored point in the scaled space the objective closure worked in, kept for
    // the #833 candidate check below: the incumbent EBE state may be scored only if it
    // was computed *here* (see `IncumbentSolve`).
    let restored_xs: Vec<f64> = x0.to_vec();

    // Unscale x0 back from optimizer space to real (log/Cholesky) space.
    for i in 0..n {
        x0[i] *= scale[i];
    }

    let final_params = unpack_params(&x0, init_params);
    let final_is_mixture = final_params.mixture.is_some();

    // The −2LL of a *cold* re-solve at the restored point, kept separately from the
    // reported `final_ofv` because the plateau self-consistency check (#751) is a
    // statement about the cold restart specifically. `None` on the mixture path,
    // whose own solve has no cold/warm split.
    let mut cold_ofv: Option<f64> = None;
    // #1565: how many subjects the cold solve put in a lower individual mode than the
    // optimizer's own EBEs, when it beat the best-seen objective and the count is
    // measurable (see `subjects_in_a_lower_cold_mode`). Feeds the warning only.
    let mut basin_subjects: Option<usize> = None;
    // #2013: the IDs of the subjects whose EBE the cold solve's Sobol screen moved off the
    // mode the cold descent found. Feeds the warning only.
    let mut screen_moved_ids: Vec<String> = Vec::new();
    // #2646: the subjects whose reported EBEs score materially below their cold ones.
    // Feeds the warning only.
    let mut cold_lost_ids: Vec<String> = Vec::new();

    // Final inner loop at converged parameters. Mixture (#977 Phase 3): the OFV
    // is the K-fold log-sum-exp and the reported EBEs are the MIXEST class.
    let (final_ehs, final_hms, final_kappas, final_ofv, final_mixture_posteriors) =
        if final_is_mixture {
            let m = crate::estimation::mixture::mixture_ofv(
                model,
                population,
                &final_params,
                options,
                None,
            );
            // #985: carry the MIXEST class's per-occasion κ̂ into postfit. Empty for a
            // non-IOV mixture, so the downstream `kappas.is_empty()` branches (sdtab
            // IPRED/IWRES/CWRES, per-subject OFV, κ shrinkage, `.fitrx` `ebe_kappas`)
            // behave exactly as before for non-IOV models, and reflect the IOV the fit
            // actually used for an IOV mixture instead of κ = 0.
            (
                m.mixest_etas,
                m.mixest_h_mats,
                m.mixest_kappas,
                m.ofv,
                Some(MixturePosteriors {
                    pmix: m.pmix,
                    mixest: m.mixest,
                    n_fd_h_redone: m.n_fd_h_redone,
                    class_nll: m.class_nll,
                }),
            )
        } else {
            let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
            // The final re-solve every outer path shares (#2097), with the #2013
            // screen-moved IDs read back alongside it.
            //
            // The cold solve is kept for its own sake: `failure_is_converged_plateau`
            // reads it as the warm-start-artifact probe (#751), and that check is only
            // worth anything while the number it reads is genuinely cold.
            let cold_solve = match cold_kept
                .take()
                .filter(|(x, _)| x.iter().map(|v| v.to_bits()).eq(x0.iter().map(|v| v.to_bits())))
            {
                Some((_, solved)) => solved,
                None => {
                    final_inner_solve(model, population, &final_params, &final_mu_k, None, options)
                }
            };
            // Only the cold solve can screen: the seeded re-solve below is warm.
            screen_moved_ids = cold_solve.screen_moved_ids;
            let (cold_ehs, cold_hms, cold_kappas, cold) = score_final_ebes(
                model,
                population,
                &final_params,
                (cold_solve.etas, cold_solve.h_mats, cold_solve.kappas),
                options,
            );
            cold_ofv = Some(cold);

            // #833: the cold EBEs are not automatically the ones the optimizer's own
            // objective was measured at. On a weakly-identified or multimodal inner
            // problem (#864 / #891), or simply one the inner budget cannot re-converge
            // from η = 0, the cold restart settles at a different η̂ than the warm
            // trajectory did and the reported OFV comes out *above* the best-seen value
            // the point was restored for — measured at +3.5 on the fluconazole 2-cpt
            // binding model and +6.96 on the FREM warfarin fixture (+3513 for the same
            // fixture on glibc, #1349) — which also drags the covariance step off the
            // reported minimum.
            //
            // When that happens, two more candidates are scored on the same objective:
            // the incumbent EBEs exactly as the optimizer left them, and a re-solve
            // seeded from them. The incumbent is scored rather than merely used as a
            // seed because `run_inner_loop_warm` is *not* monotone from its seed — the
            // FREM arm of `find_ebe` can still adopt a Nelder–Mead restart over a better
            // BFGS partial (#1365) — so the seeded re-solve alone could come back above
            // the objective its own seed already had, and the invariant this fix exists
            // for would not hold on precisely the path that motivated it. Scoring the
            // incumbent costs one `pop_nll_opts` pass and no inner loop.
            //
            // The re-solve is *conditional* on the cold solve having failed to reproduce
            // `best_seen_ofv`, in total or on some subject (#2646). On a unimodal inner
            // problem — most fits — it reproduces it, there is nothing to recover, and
            // the fit stays bit-identical to the pre-#833 behaviour, at the cost of the
            // population passes that comparison takes and no inner solve. A
            // non-finite cold objective asks for the retry too; that is `NonFinite`'s
            // own arm rather than a gap test, because every comparison against `NaN` is
            // false.
            let cold_missed_the_incumbent = best_seen_ofv
                .map_or(!cold.is_finite(), |best| cold_solve_verdict(cold, best).missed());
            let held = best_solve
                .lock()
                .unwrap()
                .take()
                .filter(|inc| inc.etas.len() == population.subjects.len());
            #[cfg(test)]
            HELD_SOLVE_AT_RESTORED.with(|c| c.set(held.as_ref().map(|inc| inc.xs == restored_xs)));
            #[cfg(test)]
            let held = held.map(|mut inc| {
                if let Some(scale) = HELD_FIRST_H_SCALE.with(|c| c.get()) {
                    inc.h_mats[0] *= scale;
                }
                inc
            });
            // #1565: the opposite direction. Counting is a diagnostic for the warning,
            // and it reads the held state as *this point's*, so it needs the same
            // `xs` match as scoring it does below; AGQ's per-subject term is not the
            // one `subject_nll` computes.
            if cold_solve_beats_best_seen(Some(cold), best_seen_ofv).is_some()
                && options.agq_nodes().is_none()
            {
                basin_subjects = held.as_ref().filter(|inc| inc.xs == restored_xs).map(|inc| {
                    subjects_in_a_lower_cold_mode(
                        model,
                        population,
                        &final_params,
                        (&inc.etas, &inc.h_mats, &inc.kappas),
                        &cold_ehs,
                        &cold_hms,
                        &cold_kappas,
                        options.interaction,
                    )
                });
            }
            let cold_candidate = (cold_ehs, cold_hms, cold_kappas, cold);
            let (ehs, hms, kappas, ofv) = match held {
                Some(inc) => {
                    // Scoring the held state is only sound when it belongs to the point
                    // that was restored — its `h_mats` are that point's curvature. When
                    // it does not (the tracker can rank a guard-penalised eval best,
                    // whose EBEs are never adopted), it still seeds the re-solve.
                    let held_is_at_this_point = inc.xs == restored_xs;
                    let seed = joint_warm_starts(&inc.etas, &inc.kappas);
                    let held = held_is_at_this_point.then(|| {
                        score_final_ebes(
                            model,
                            population,
                            &final_params,
                            (inc.etas, inc.h_mats, inc.kappas),
                            options,
                        )
                    });
                    let pick = recover_final_ebes(
                        model,
                        population,
                        &final_params,
                        &final_mu_k,
                        options,
                        cold_candidate,
                        &seed,
                        held,
                        cold_missed_the_incumbent,
                    );
                    pick.retain_cold_reported(&mut screen_moved_ids);
                    cold_lost_ids = pick.cold_lost_ids;
                    pick.reported
                }
                None => cold_candidate,
            };
            (ehs, hms, kappas, ofv, None)
        };

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    // Resolve a deferred `Failure`/`ForcedStop` verdict (see
    // `stationarity_check_pending`). NLopt's analytic-gradient L-BFGS default
    // (#639) returns a bare `NLOPT_FAILURE` *at* a plateaued optimum — its line
    // search can no longer beat an OFV already flat to ~8 significant figures —
    // and the raw enum then libels a finished fit as `converged=false`. That
    // both fails the honest convergence tests and tags the point non-stationary
    // right before the FD-of-OFV covariance step, whose R-matrix is only
    // well-conditioned at a true minimum (issue #751).
    //
    // The analytic gradient norm is *not* a usable stationarity proxy here: at
    // these genuine optima it still reads O(1) (npde ≈ 1.8, schnider ≈ 0.05)
    // because the best-point EBEs the outer gradient reuses differ slightly from
    // the cold-restart `final_ehs`, and because weakly-identified directions
    // carry a large scaled ∂OFV/∂x at a flat OFV. Decide on the OFV trace
    // instead — the quantity that actually defines convergence for a noisy FOCE
    // objective:
    //   (a) plateau — the best OFV has not improved by more than
    //       `PLATEAU_OFV_THRESHOLD` for at least `PLATEAU_MIN_FLAT_EVALS` evals
    //       (a real stall, e.g. SS-oral quitting after ~5 evals still plunging,
    //       has no flat tail); and
    //   (b) self-consistency — re-running the inner loop cold at the restored
    //       best point reproduces the best-seen OFV (the SS-oral stall's
    //       best-seen 83.3 vs cold 121.4 exposes a warm-start artifact).
    // Both must hold; a genuine mid-descent stall fails at least one, so this
    // never papers over non-convergence.
    // The number this check reads is the *cold* re-solve, not the reported
    // `final_ofv` — since #833 those differ whenever the warm re-solve found a
    // better EBE mode, and handing it the warm one would make the probe compare the
    // best-seen objective against a re-run of itself (`consistent` would then be
    // true by construction and the check would stop rejecting warm-start artifacts).
    let consistency_ofv = cold_ofv.unwrap_or(final_ofv);
    // #1565, read here rather than with its warning below because a restart leg's probe
    // (#2707) is not run on a leg that will be restarted on a basin shift anyway.
    let basin_gap = cold_solve_beats_best_seen(cold_ofv, best_seen_ofv);
    let (_, last_sig_feasible_eval, feasible_evals) = *plateau_tracker.lock().unwrap();
    // The acceptance rule of a stop whose `converged` is decided here (a bare `Failure` /
    // `ForcedStop`, a latched stop, or a verified one — `stationarity_check_pending`), as a
    // function of whether the leg left its start: the #2462 verified-stop branch, else
    // [`failure_is_converged_plateau`]. Read twice on a restart leg, with and without the
    // displacement, to decide whether displacement alone rejects it.
    let stop_accepted = |left: bool| {
        if verified_stop {
            cold_solve_is_consistent(best_seen_ofv, consistency_ofv)
                && (left || !verified_needs_left_init)
        } else {
            failure_is_converged_plateau(
                feasible_evals,
                last_sig_feasible_eval,
                best_seen_ofv,
                consistency_ofv,
                left,
            )
        }
    };
    let probe_due = |when: ProbeWhen| {
        start_probe_due(when, stationarity_check_pending, &stop_accepted, basin_gap.is_some())
    };
    // #2506: a fit that never left its start may have *started* at an optimum — a
    // bootstrap sample or a refit warm-started from a previous fit's estimates.
    // Displacement cannot tell that from a #751 stall, so the optimizer is asked instead:
    // the probe restarts it from a perturbation of the start and reports whether it came
    // back ([`start_probe_verdict`]). Run only when this attempt did not move, scored the
    // real objective at least once (with no feasible evaluation the start's objective is no
    // reference to compare against, and #1617 reports the run as it is), was not cancelled,
    // and has budget left — the probe continues the same fit, on its budget. On a restart leg,
    // only when displacement is all that rejects it (`ProbeWhen::DisplacementAloneRejects`).
    if let Some((probe, _)) = start_probe.filter(|&(_, when)| {
        !left_init
            && probe_due(when)
            && !feasibility.no_feasible()
            && !crate::cancel::is_cancelled(&options.cancel)
            && spent_evals + n_evals_outer.load(Ordering::Relaxed) < full_budget as usize
    }) {
        if let Some(mut xp) =
            perturbed_start(&restored_xs, &packed_fixed_mask(init_params), &lower_s, &upper_s)
        {
            for i in 0..n {
                xp[i] *= scale[i];
            }
            let start_ofv =
                final_ofv + nn_reg.penalty_value(&final_params.theta) + priors.penalty(&x0);
            probe(
                &unpack_params(&xp, init_params),
                start_ofv,
                spent_evals + n_evals_outer.load(Ordering::Relaxed),
            );
        }
    }
    if stationarity_check_pending {
        if stop_accepted(left_init) {
            converged = true;
        }
        if options.verbose {
            let flat_tail = feasible_evals.saturating_sub(last_sig_feasible_eval);
            eprintln!(
                "Plateau check: flat_tail = {} feasible evals (min {}), feasible_evals = {}, \
                 best-seen {:?} vs cold {:.6} (reported {:.6}), left init = {} → converged = {}",
                flat_tail,
                PLATEAU_MIN_FLAT_EVALS,
                feasible_evals,
                best_seen_ofv,
                consistency_ofv,
                final_ofv,
                left_init,
                converged,
            );
        }
    }
    // Read *here*, before `gate_converged_on_objective` below can demote
    // `converged` for an entirely different reason — see
    // [`worth_restarting_mid_descent`], which is handed this and the objective.
    let stalled_mid_descent = stationarity_check_pending && !converged;
    // #1617: no feasible evaluation at all. Placed *after* `stalled_mid_descent` is read,
    // for the same reason `gate_converged_on_objective` is: that boolean must mean "NLopt
    // returned a bare `Failure` and the plateau check rejected it", not "`converged` is
    // false for any reason at all". The infeasibility is kept out of the #1277 restart by
    // [`worth_restarting_mid_descent`], which is handed it as a separate argument —
    // *not* by this statement's position (#1617 review round 1 #1: the first version
    // claimed the position did it, and an all-infeasible leg that ended in `Failure`
    // rather than `Success` was restarted anyway). Placed *before* the covariance step and
    // the generic "did not converge" line, so the specific reason reaches the user
    // alongside both (the same ordering #1303 uses).
    if let Some(w) = no_feasible_eval_warning(&mut converged, &feasibility) {
        warnings.push(w);
    }
    // Whether any of the fit's budget is left for a restart to run on (#1428).
    // `n_evals_outer` counts objective calls, the same thing NLopt's `maxeval`
    // counts — but `maxeval` is a *soft* bound for L-BFGS: `luksan/plis.c` checks
    // it only between line searches, so a line search that fails after crossing
    // it comes back as a bare `Failure`, not `MaxEvalReached`, and the count can
    // sit a few evals past the budget here.
    let budget_left = spent_evals + n_evals_outer.load(Ordering::Relaxed) < full_budget as usize;

    // #1565: a cold re-solve materially *below* the objective the optimizer minimised
    // here means θ was optimized under EBEs some subjects held in a higher mode, on
    // whichever path NLopt stopped by — `Success`/`FtolReached` as much as the plateau-
    // checked `Failure`. The reported OFV (the cold one) is right at these estimates;
    // the estimates are not the optimum of it, so the run is not converged. Demoted
    // *after* `stalled_mid_descent` is read, which must keep meaning "bare `Failure`
    // the plateau check rejected": folding this in would also fire the "stopped
    // mid-descent with its budget spent" advice on a run that did nothing of the kind.
    // `optimize_nlopt` restarts such a run ([`RestartCause::BasinShift`], repeated up to
    // `MAX_BASIN_RESTARTS` times since #2093); this
    // warning reaches the user only if that restart is not adopted or not possible.
    // (`basin_gap` is read above, before the start probe.)
    if let (Some(gap), Some(cold), Some(best)) = (basin_gap, cold_ofv, best_seen_ofv) {
        converged = false;
        warnings.push(ebe_basin_shift_warning(
            cold,
            best,
            gap,
            basin_subjects,
            BasinShiftReport::ColdRestartable,
        ));
        if options.verbose {
            eprintln!(
                "Basin check: cold {cold:.6} beats best-seen {best:.6} by {gap:.4} \
                 (> {BASIN_SHIFT_ABS_TOL}; {basin_subjects:?} subjects in a lower cold mode) \
                 → converged = false"
            );
        }
    }

    // A cold re-solve that does not reproduce the reported objective — whether it lands
    // materially above it or returns no usable number at all — says the EBEs at these
    // estimates depend on where the inner loop starts, and so does every diagnostic
    // built on them. Surface it rather than silently reporting the best candidate
    // (#833). [`ebe_start_dependence_warning`] owns both arms and the wording.
    if let Some(w) =
        cold_ofv.and_then(|cold| ebe_start_dependence_warning(cold, final_ofv, &cold_lost_ids))
    {
        warnings.push(w);
    }
    if let Some(w) = ebe_screen_moved_warning(&screen_moved_ids) {
        warnings.push(w);
    }
    if latch_overrode_veto.load(Ordering::Relaxed) {
        warnings.push(veto_override_warning(*last_veto_gain.lock().unwrap()));
    }

    // Why `optimize_nlopt` would restart this leg from its own best point.
    let mid_descent_stall = worth_restarting_mid_descent(
        stalled_mid_descent,
        final_ofv,
        budget_left,
        feasibility.no_feasible(),
    );
    // The same three guards as the stall: a non-finite objective, no budget to run on, or
    // nothing feasible to restart from all rule a restart out for the same reasons,
    // whatever prompted it.
    let basin_shift = worth_restarting_mid_descent(
        basin_gap.is_some(),
        final_ofv,
        budget_left,
        feasibility.no_feasible(),
    );
    // Covariance step (skip if user cancelled — it's expensive and the result
    // will be discarded by the top-level fit() anyway).
    // Mixture (#983 Phase 6): `compute_covariance` builds the FD Hessian on the
    // K-fold mixture OFV when `template.mixture` is set, so the step runs for
    // mixtures too. The `final_ehs`/`final_hms` handed in are the MIXEST-class
    // EBEs; the mixture branch reconverges per class internally and does not use
    // them as a warm start.
    let newton_decrement;
    let (covariance_matrix, covariance_wall_time_secs, sir_fallback_proposal, covariance_method) = {
        let out = if defer_covariance {
            crate::estimation::covariance::CovStepOutcome::skipped()
        } else {
            crate::estimation::covariance::run_covariance_step(
                &x0,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
                options.verbose.then_some("Computing covariance matrix..."),
            )
        };
        let crate::estimation::covariance::CovStepOutcome {
            matrix,
            wall_time_secs,
            warnings: cov_warnings,
            sir_fallback_proposal,
            method: covariance_method,
            newton_decrement: decrement,
            non_stationary_optimum,
        } = out;
        warnings.extend(cov_warnings);
        newton_decrement = decrement;
        // #2785: the "did not converge" line below covers this demotion too.
        if non_stationary_optimum {
            converged = false;
        }
        (matrix, wall_time_secs, sir_fallback_proposal, covariance_method)
    };

    // #1303. Placed *before* the plain "did not converge" line so a demoted run
    // carries both: the generic notice a consumer already greps for, and the
    // specific reason. `final_ofv` is what this `OuterResult` publishes, and it
    // is recomputed at the restored best point — NLopt can report `Success` on
    // its own trace and still hand back a `NaN` here (measured on a population
    // with one unorderable timeline: trace bottoms out at 1e12, `Final OFV =
    // NaN`), which is exactly the disagreement this closes.
    if let Some(w) = gate_converged_on_objective(&mut converged, final_ofv) {
        warnings.push(w);
    }
    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }
    // A stall with no budget left is not restarted (see
    // `worth_restarting_mid_descent`), and the reason the user can act on is the
    // budget, not the stall — a line search that fails after crossing `maxeval`
    // is a bare `Failure`, so the `max_eval_reached` branch above never saw it.
    // Same advice as that branch, not pushed twice.
    // `budget_advice_applies` for the same reason as the `max_eval_reached` arm above, and
    // from the same binding — see the comment there for why it is one binding and not two.
    if stalled_mid_descent && !budget_left && !max_eval_reached && budget_advice_applies {
        warnings.push(format!(
            "Outer optimization stopped mid-descent with its evaluation budget (maxiter = {}) \
             spent, so it was not restarted; increase maxiter for a tighter fit.",
            options.outer_maxiter,
        ));
    }

    // The gradient to report, and where it came from (#997 §1). A gradient-based
    // NLopt run already has one at the best point; a derivative-free one has
    // nothing, which is exactly the case where `converged` most needs checking —
    // so compute it here, once, at the same restored point the estimates and the
    // covariance step were built from. Skipped on cancellation (the result is
    // discarded anyway) and when the caller opted out of the `2·n_free` evals.
    let (final_gradient, final_gradient_source) = match last_gradient.lock().unwrap().clone() {
        Some(g) => (Some(g), Some("optimizer".to_string())),
        None if options.report_final_gradient && !crate::cancel::is_cancelled(&options.cancel) => {
            if options.verbose {
                eprintln!(
                    "Computing the gradient at the solution for reporting ({} free \
                     coordinates) — the optimizer supplied none; exact where the outer \
                     gradient is, else finite differences.",
                    packed_fixed_mask(init_params).iter().filter(|f| !**f).count(),
                );
            }
            let (g, source) = reporting_gradient(
                &x0,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                &bounds,
                options,
                &nn_reg,
                &priors,
            );
            (Some(g), Some(source.to_string()))
        }
        None => (None, None),
    };

    let ebe_final = ebe_accum.lock().unwrap();
    let result = OuterResult {
        ofv: final_ofv,
        converged,
        // NLopt doesn't expose an "iteration" count (BOBYQA/SLSQP don't have
        // iterations in the textbook sense), so report the number of
        // objective-function evaluations instead — the only monotone
        // progress counter NLopt exposes, and the quantity most users
        // actually care about ("how much work did the fit do").
        n_iterations: n_evals_outer.load(Ordering::Relaxed),
        eta_hats: final_ehs,
        h_matrices: final_hms,
        kappas: final_kappas,
        covariance_matrix,
        covariance_method,
        covariance_wall_time_secs,
        warnings,
        ebe_convergence_warnings: ebe_final.n_convergence_warnings as u32,
        max_unconverged_subjects: ebe_final.max_unconverged as u32,
        total_ebe_fallbacks: ebe_final.total_fallback as u32,
        total_ebe_nm_polished: ebe_final.total_nm_polished as u32,
        final_gradient,
        final_gradient_source,
        sir_fallback_proposal,
        // The exact packed vector this stage's inline covariance step used (#816
        // follow-up): reused by `run_covariance` to avoid re-decomposing omega.
        packed_estimate: Some(x0.clone()),
        left_init: Some(published_left_init),
        mixture_posteriors: final_mixture_posteriors,
        // A restart leg publishes the fit's start, not its own (see `OuterResult::start_eval`).
        start_eval: inherited_start.unwrap_or(start_eval.get()),
        newton_decrement,
        optimizer_stop,
        // #2799: the final budget state — after the polish (`PolishEnd::BudgetSpent`), a
        // verified stop and a stagnation latch on the last permitted eval have each had their
        // say above, so it is the same fact the "increase maxiter" warning is gated on.
        eval_budget_spent: max_eval_reached,
        // `stationary_start` stays false: set by `settle_start_probe` on the probe's result, never
        // on an attempt's own.
        ..OuterResult::from_params(final_params)
    };
    (
        result,
        AttemptOutcome {
            left_init,
            mid_descent_stall,
            basin_shift,
            // A #2462 leg clears the latch it starts from; the attempt still latched.
            stagnation_latched: latched || latched_at_stop,
            plateau_checked: stationarity_check_pending,
        },
    )
}

/// NLopt's stop state as the fit publishes it (#2785): the `SuccessState` / `FailState`
/// variant name (`FtolReached`, `Failure`, `RoundoffLimited`, …), the same word the verbose
/// log prints after "NLopt finished:" / "NLopt stopped:". `converged` is a verdict built on
/// this and on checks after it, and can overrule it either way — a bare `Failure` at a
/// plateau is converged (#751), an `FtolReached` at a non-stationary point is not (#2785) —
/// so the raw state is kept beside it for a reader asking why.
fn nlopt_stop_label<T>(result: &Result<(nlopt::SuccessState, T), (nlopt::FailState, T)>) -> String {
    match result {
        Ok((status, _)) => format!("{status:?}"),
        Err((fail, _)) => format!("{fail:?}"),
    }
}

/// Whether a run that stopped mid-descent is worth restarting from its own best
/// point (#1277).
///
/// `stalled_mid_descent` is the verdict read **before**
/// [`gate_converged_on_objective`] runs, because that gate demotes `converged`
/// for a different fact — the objective at the final estimates is not a number —
/// and a run demoted for *that* must not be restarted: the estimates are
/// wherever the objective went non-finite, and handing them back as a starting
/// point re-runs the same poisoned fit. That is why the objective is a second
/// argument here rather than folded into the boolean at the call site: written
/// as one expression the two facts are separated only by statement order, and
/// moving the read three lines down would silently turn every non-finite fit
/// into a restart.
///
/// `budget_left` is whether this run left any of the fit's evaluation budget
/// unspent (#1428). The restart continues the fit on that remainder, so a run
/// that died on its last permitted evaluation has nothing to continue with —
/// and `set_maxeval(0)` would hand NLopt an *unlimited* budget, not an empty
/// one. Such a run is reported as it stands, with an "increase maxiter" warning
/// of its own: the ordinary budget warning is keyed on `MaxEvalReached`, which a
/// line search that fails after crossing `maxeval` never returns. (#1617)
///
/// `no_feasible_eval` is [`RunFeasibility::no_feasible`]: not one evaluation scored the real
/// objective (#1617). Such a leg has nothing to descend from: after the restore it reports the
/// point it was handed, so the restart would begin exactly where this leg began (same cap
/// state, fresh `(s, y)` memory as this leg had). That is a deterministic replay of a prefix
/// of the leg that just failed, which cannot find a feasible evaluation the first leg missed,
/// and the strictly-lower adoption rule would discard it. A fourth argument, like the
/// objective, rather than a term folded into `stalled_mid_descent` at the call site (which
/// reads that verdict *before* it is demoted).
///
/// This is **not** the same trade as the #751 `left_init` retry, which *is* run on an
/// infeasible leg: that retry passes `hold_cap_at_init = true`, so its opening step is
/// genuinely different, and a smaller first step is one of the ways a starved inner loop
/// converges. One replay is waste; one different configuration is a chance.
fn worth_restarting_mid_descent(
    stalled_mid_descent: bool, final_ofv: f64, budget_left: bool, no_feasible_eval: bool,
) -> bool {
    stalled_mid_descent && ofv_is_valid(final_ofv) && budget_left && !no_feasible_eval
}

/// The `FitResult` warning a fit carries when it was restarted mid-descent and
/// the restart was adopted (#1428). The phrase "resumed from the best point
/// seen" is what `classify_warning` files under `optimizer_health`, and what the
/// PR-time regression test (`nn::regularizer_fit_tests::first_line_search_abort_is_avoided_or_resumed_not_reported`)
/// looks for; keep it if the wording changes.
fn mid_descent_restart_warning(
    optimizer: Optimizer, stalled_ofv: f64, stalled_evals: usize, from_probe: bool,
) -> String {
    format!(
        "Outer optimizer ({}) stopped mid-descent (bare Failure at OFV = {:.6} after {} \
         evaluations) and was resumed from the best point seen on the remaining maxiter \
         budget; {}",
        optimizer.label(),
        stalled_ofv,
        stalled_evals,
        restart_provenance(from_probe),
    )
}

/// The closing clause of a restart's warning: where the reported estimates come from. When
/// the resumed run did not leave its start and its #2707 start probe reported a lower point
/// ([`settle_restart_probe`]), they are the probe's, not the resumed run's.
fn restart_provenance(from_probe: bool) -> &'static str {
    if from_probe {
        "the resumed run did not leave its start, and the reported estimates are from a \
         restart from a perturbation of that point, which reached a lower objective."
    } else {
        "the reported estimates are from the resumed run."
    }
}

/// The `FitResult` warning a fit carries when it was restarted for a basin shift
/// (#1565) and the restart was adopted. Its phrase "resumed from the best point seen"
/// files it under `optimizer_health` with the #1277 notice: the gap itself was
/// resolved (had it survived, the restart's own `W_EBE_START_DEPENDENT` would say so),
/// so what remains to report is that the fit took a second leg, and why.
fn basin_shift_restart_warning(
    first_cold_ofv: f64, first_evals: usize, from_probe: bool,
) -> String {
    format!(
        "Outer optimization had minimised against empirical Bayes estimates that some \
         subjects held in a higher individual mode than a cold re-solve finds (cold OFV = \
         {first_cold_ofv:.6} after {first_evals} evaluations, below the objective the \
         optimizer saw there), and was resumed from the best point seen with cold EBEs on \
         the remaining maxiter budget; {}",
        restart_provenance(from_probe),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
//  Hand-rolled BFGS outer optimizer (legacy fallback)
// ═══════════════════════════════════════════════════════════════════════════

/// The scale vector the built-in BFGS / L-BFGS loop divides the clamped packed start `x`
/// by: the space its steps, bounds and stopping rules live in. Also the space the #2506
/// start probe measures their movement in ([`verify_unmoved_start`]).
pub(crate) fn bfgs_scale(
    x: &[f64], bounds: &PackedBounds, init_params: &ModelParameters, options: &FitOptions,
) -> Vec<f64> {
    match resolve_scaling(options.parameter_scaling, options.optimizer) {
        ParameterScaling::Rescale2 => compute_rescale2_scale(bounds),
        ParameterScaling::Abs => compute_scale_packed(x, init_params),
        ParameterScaling::None | ParameterScaling::Auto => {
            if options.scale_params {
                compute_scale_packed(x, init_params)
            } else {
                vec![1.0; x.len()]
            }
        }
    }
}

fn optimize_bfgs(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions, declines: &OuterFdDeclineLog, first_eval_etas: Option<&[DVector<f64>]>,
) -> OuterResult {
    let PackedStart { packed: mut x, bounds, .. } = pack_with_bounds(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let n = x.len();
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let mut warnings = Vec::new();
    // #2076: the first evaluation solves at `x`, where the flat-theta pre-flight
    // already solved; start it from the pre-flight's EBEs when it handed them over.
    let mut cached_etas: Vec<DVector<f64>> = match first_eval_etas {
        Some(etas) if etas.len() == n_subj => etas.to_vec(),
        _ => vec![DVector::zeros(n_eta); n_subj],
    };
    // What `cached_etas` are. `ModesHere` while they still hold those converged modes,
    // unconsumed: the first `fdfg` call takes them as warm starts whatever their values
    // ([`InnerSolvePolicy::init_is_mode`]). Every `fdfg` result becomes the cache, so
    // from then on they are `Solved` — an exactly-zero EBE among them included.
    let cached_etas_kind =
        std::cell::Cell::new(if first_eval_etas.is_some_and(|e| e.len() == n_subj) {
            PrevEtas::ModesHere
        } else {
            PrevEtas::Placeholder
        });

    // Covariate-NN (DCM) regularizer. No-op when both λ are 0. Penalty is added
    // to the optimizer-facing `f_only`/`fdfg` values (and `fdfg`'s gradient) but
    // NOT to `ofv_at_fixed`, which the final reported OFV reuses — so the
    // reported OFV/AIC/BIC stay the unpenalized −2LL. The trace, checkpoint and
    // verbose `Iter` lines report the clean value too (`clean_ofv_at` below).
    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    let priors = build_prior_set(model, init_params);
    // The −2LL behind a penalized objective value at scaled point `xs`: what the
    // user-facing streams print, so they agree with the reported `Final OFV`.
    let clean_ofv_at = |xs: &[f64], f_penalized: f64, scale: &[f64]| -> f64 {
        if !nn_reg.is_active() && !priors.is_active() {
            return f_penalized;
        }
        let x_real: Vec<f64> = xs.iter().zip(scale).map(|(v, s)| v * s).collect();
        f_penalized
            - nn_reg.penalty_value(&unpack_params(&x_real, init_params).theta)
            - priors.penalty(&x_real)
    };

    // Closures operating on unscaled real (log/Cholesky) space.
    let ofv_at_fixed = |x: &[f64],
                        eta_hats: &[DVector<f64>],
                        h_matrices: &[DMatrix<f64>],
                        kappas: &[Vec<DVector<f64>>]|
     -> f64 {
        let params = unpack_params(x, init_params);
        2.0 * pop_nll_opts(model, population, &params, eta_hats, h_matrices, kappas, options)
    };

    let f_only = |x: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (_, _, _, _, nll) = run_inner_loop_and_nll(
            model,
            population,
            &params,
            options,
            Some(prev_etas),
            // A line-search probe is never at the point the cache's modes were solved at.
            cached_etas_kind.get().after_pass(false),
            Some(&mu_k),
            None,
            None, // the built-in BFGS loop polls between iterations (#574)
        );
        // Penalized objective fed to the optimizer (unregularized fits unchanged).
        let ofv = 2.0 * nll + nn_reg.penalty_value(&params.theta) + priors.penalty(x);
        if ofv.is_finite() {
            ofv
        } else {
            1e20
        }
    };

    // `incumbent` is the #1520 salvage guard's reference: the optimizer-facing objective
    // and per-subject `2·nllᵢ` of the last accepted iterate. Returns this point's
    // `2·nllᵢ` as its last element so the loop can promote it.
    //
    // `predicted_from`: where `prev_etas` were predicted from, when they are predicted
    // starts (`run_inner_loop_predicted_map_cached`); `eta_dx`: where to put this point's mode
    // response, formed by the gradient's own sweeps.
    let fdfg = |x: &[f64],
                prev_etas: &[DVector<f64>],
                predicted_from: Option<
        crate::estimation::inner_optimizer::PredictedFromAll<'_>,
    >,
                grad_eval_idx: &mut usize,
                incumbent: Option<(f64, &[f64])>,
                eta_dx: Option<&mut crate::estimation::sens_outer_gradient::ModeResponses>|
     -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<f64>) {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (mut ehs, mut hms, _, mut kappas, nll, _, contribs) = run_inner_loop_and_nll_prepared(
            model,
            population,
            &params,
            options,
            Some(prev_etas),
            predicted_from,
            Some(&mu_k),
            None,
            None,
            cached_etas_kind.replace(PrevEtas::Solved),
            None, // the built-in BFGS loop polls between iterations (#574)
            None,
        );
        let contribs: Vec<f64> = contribs.iter().map(|v| 2.0 * v).collect();
        let ofv = 2.0 * nll;
        // The value the optimizer ranks this point on, for the guard: the penalties are
        // re-added below together with their gradients, in the original order, so the
        // returned `f` and `g` are unchanged by this read.
        let trial = OuterTrial {
            ofv: ofv + nn_reg.penalty_value(&params.theta) + priors.penalty(x),
            contribs: &contribs,
            incumbent,
        };
        // d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x), and with `eta_dx` the mode response from the
        // same sweeps.
        let mut own = crate::estimation::sens_outer_gradient::ModeResponses::default();
        let slot = eta_dx.unwrap_or(&mut own);
        let mut g = population_gradient_with_agq_evaluation(
            x,
            n_subj,
            init_params,
            model,
            population,
            &ehs,
            &hms,
            &kappas,
            &bounds,
            options,
            grad_eval_idx,
            None,
            trial,
            declines,
            Some(&mut *slot),
        );
        // A polished subject's `η̂` (and `H`) replace its off-mode ones in what comes back
        // as the next solve's warm start.
        adopt_polished_modes(
            model,
            population,
            init_params,
            x,
            &slot.polished,
            &mut ehs,
            &mut hms,
            &mut kappas,
        );
        // Penalized value + matching gradient fed to the optimizer (unregularized
        // fits unchanged), in one pass. `ofv_at_fixed` above stays clean for
        // final reporting.
        let ofv = ofv + nn_reg.penalty_and_gradient(&params.theta, &mut g);
        // Parameter priors (#254), value and gradient from one call, in the same
        // packed space `g` is already expressed in.
        let ofv = ofv + priors.penalty_and_gradient(x, &mut g);
        let f = if ofv.is_finite() { ofv } else { 1e20 };
        // The EBEs come back as the next solve's warm starts, κ̂ with η̂ (#2261), except
        // on an ODE + IOV model (#2508).
        (f, g, outer_cache_warm_starts(model, &ehs, &kappas), hms, contribs)
    };

    // Per-element scale factors for the BFGS outer loop.
    let scale = bfgs_scale(&x, &bounds, init_params, options);
    let lower_s: Vec<f64> = (0..n).map(|i| bounds.lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| bounds.upper[i] / scale[i]).collect();
    let bounds_s = PackedBounds { lower: lower_s, upper: upper_s };

    // Wrappers that operate in scaled space; unscale before calling base closures.
    let fdfg_s =
        |xs: &[f64],
         prev_etas: &[DVector<f64>],
         predicted_from: Option<crate::estimation::inner_optimizer::PredictedFromAll<'_>>,
         grad_eval_idx: &mut usize,
         incumbent: Option<(f64, &[f64])>,
         eta_dx: Option<&mut crate::estimation::sens_outer_gradient::ModeResponses>|
         -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<f64>) {
            let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
            let (f, g_r, ehs, hms, contribs) =
                fdfg(&x_r, prev_etas, predicted_from, grad_eval_idx, incumbent, eta_dx);
            let g_s: Vec<f64> = (0..n).map(|i| g_r[i] * scale[i]).collect();
            (f, g_s, ehs, hms, contribs)
        };

    let f_only_s = |xs: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        f_only(&x_r, prev_etas)
    };

    // Scale initial x into optimizer space.
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();

    // Gradient-evaluation counter driving the reconverge schedule; advanced
    // inside `population_gradient` so it counts actual gradient evals (not
    // outer iterations or objective-only line-search probes).
    let mut grad_eval_idx = 0usize;
    // EBE warm-start predictor (Almquist Eq. 48, #2549): each accepted point's inner solves
    // start at `η̂ + (dη̂/dx)·Δx`, with the mode response from the accepted point's own
    // gradient sweeps, as in the NLopt objective (`warm_predictor_serves`, which also reads
    // the `FERX_EBE_PREDICTOR=0` A/B switch). Every `fdfg_s` call below is at a point the
    // backtracking line search accepted, so each one forms the response the next step
    // predicts from.
    let use_predictor = warm_predictor_serves(model, init_params, options);
    let mut last = crate::estimation::sens_outer_gradient::ModeResponses {
        want_dx: true,
        ..Default::default()
    };
    let (mut f_val, mut g, ehs, _, contribs) = fdfg_s(
        &xs,
        &cached_etas,
        None,
        &mut grad_eval_idx,
        None,
        use_predictor.then_some(&mut last),
    );
    cached_etas = ehs;
    // #1520: the salvage guard's incumbent. Every `fdfg_s` call below is at a point the
    // Armijo backtracking (on `f_only_s`) has already accepted, so the incumbent is simply
    // the previous accepted iterate; the guard can fire here only if a subject blew up
    // at an accepted point, which the population gate excludes.
    let mut incumbent: (f64, Vec<f64>) = (f_val, contribs);
    let mut x_anchor_real: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    if options.verbose {
        eprintln!("Iter {:>4}: OFV = {:.6}", 0, clean_ofv_at(&xs, f_val, &scale));
    }

    // Two outer Hessian strategies share this loop: `Optimizer::Lbfgs` uses a
    // limited-memory L-BFGS two-loop recursion over the last `LBFGS_MEMORY`
    // curvature pairs (no dense matrix); `Optimizer::Bfgs` keeps the full inverse
    // Hessian `h_inv`. Both consume the same analytic gradient and Eq. 48 warm
    // EBEs below.
    let use_lbfgs = matches!(options.optimizer, Optimizer::Lbfgs);
    const LBFGS_MEMORY: usize = 10;
    let mut s_hist: Vec<DVector<f64>> = Vec::new();
    let mut y_hist: Vec<DVector<f64>> = Vec::new();
    let mut h_inv = DMatrix::<f64>::identity(n, n);
    let mut converged = false;
    let mut n_iterations = 0;
    let mut stall_count = 0;
    // Best accepted iterate, so an interrupted run's checkpoint holds it (#1317).
    let mut best = BestPoint::new();

    // #2799: as in pure Gauss–Newton, every `break` below sets this, so a loop that runs off
    // the end of `outer_maxiter` spent its budget.
    let mut exited_early = false;
    for iter in 1..=options.outer_maxiter {
        n_iterations = iter;

        if crate::cancel::is_cancelled(&options.cancel) {
            warnings.push("cancelled by user".to_string());
            exited_early = true;
            break;
        }

        let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Snapshot the scaled gradient that `g_norm` is taken from, before the
        // step overwrites `g` with `g_new`. The trace's `grad:*` columns log
        // this vector so `sqrt(Σ gᵢ²) == grad_norm` holds (#640). Like the
        // existing `grad_norm` column, it reflects the pre-step point.
        let g_for_trace: Vec<f64> = g.to_vec();
        if g_norm < options.outer_gtol {
            if options.verbose {
                eprintln!("Converged at iteration {} (|g| = {:.2e})", iter, g_norm);
            }
            converged = true;
            exited_early = true;
            break;
        }

        let mut d: Vec<f64> = if use_lbfgs {
            lbfgs_two_loop(&g, &s_hist, &y_hist)
        } else {
            let g_vec = DVector::from_column_slice(&g);
            (-&h_inv * &g_vec).iter().copied().collect()
        };

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 || !dg.is_finite() {
            // Non-descent direction: discard curvature memory and take steepest
            // descent (L-BFGS clears its history; dense BFGS resets `h_inv`).
            d = g.iter().map(|gi| -gi).collect();
            s_hist.clear();
            y_hist.clear();
            h_inv = DMatrix::identity(n, n);
        }

        let alpha =
            backtracking_line_search_warm(&xs, &d, &g, f_val, &bounds_s, &cached_etas, &f_only_s);

        if alpha < 1e-18 {
            stall_count += 1;
            if stall_count >= 10 {
                if options.verbose {
                    eprintln!("Stopping: line search stalled at iteration {}", iter);
                }
                exited_early = true;
                break;
            }
            s_hist.clear();
            y_hist.clear();
            h_inv = DMatrix::identity(n, n);
            continue;
        }
        stall_count = 0;

        let xs_old = xs.clone();
        for i in 0..n {
            xs[i] = (xs[i] + alpha * d[i]).clamp(bounds_s.lower[i], bounds_s.upper[i]);
        }

        // Eq. 48: predict the accepted point's EBEs from the anchor before the inner solve,
        // falling back per subject to its mode where the solve fails from the prediction;
        // the plain warm start when there is no response.
        let x_new_real: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let warm: Option<Vec<DVector<f64>>> = last.eta_dx.as_ref().map(|jac| {
            crate::estimation::sens_outer_gradient::predict_warm_etas(
                &cached_etas,
                jac,
                &x_anchor_real,
                &x_new_real,
            )
        });
        let mut new_last = crate::estimation::sens_outer_gradient::ModeResponses {
            want_dx: true,
            ..Default::default()
        };
        let (f_new, g_new, ehs, _, contribs) = fdfg_s(
            &xs,
            warm.as_deref().unwrap_or(&cached_etas),
            warm.is_some().then_some(crate::estimation::inner_optimizer::PredictedFromAll {
                modes: &cached_etas,
                seeds: last.hessians.as_deref(),
            }),
            &mut grad_eval_idx,
            Some((incumbent.0, incumbent.1.as_slice())),
            use_predictor.then_some(&mut new_last),
        );
        cached_etas = ehs;
        incumbent = (f_new, contribs);
        if use_predictor {
            last = new_last;
            x_anchor_real = x_new_real;
        }

        if use_lbfgs {
            // Push the new curvature pair (s = Δx, y = Δg) with the same `s·y > 0`
            // filter `bfgs_update` uses, capping the history at `LBFGS_MEMORY`.
            let s = DVector::from_iterator(n, (0..n).map(|i| xs[i] - xs_old[i]));
            let y = DVector::from_iterator(n, (0..n).map(|i| g_new[i] - g[i]));
            if s.dot(&y) > 1e-12 {
                s_hist.push(s);
                y_hist.push(y);
                if s_hist.len() > LBFGS_MEMORY {
                    s_hist.remove(0);
                    y_hist.remove(0);
                }
            }
        } else {
            bfgs_update(&mut h_inv, &xs, &xs_old, &g_new, &g, n);
        }

        let prev_ofv = f_val;
        f_val = f_new;
        g = g_new;

        if options.verbose && (iter % 10 == 0 || iter <= 5) {
            eprintln!(
                "Iter {:>4}: OFV = {:.6}  |g| = {:.2e}  alpha = {:.2e}",
                iter,
                clean_ofv_at(&xs, f_val, &scale),
                g_norm,
                alpha
            );
        }

        // Optimizer trace (step_norm in scaled space)
        if crate::estimation::trace::is_active() {
            let step_norm: f64 = (0..n).map(|i| (xs[i] - xs_old[i]).powi(2)).sum::<f64>().sqrt();
            let method_str = crate::estimation::trace::method_label(options);
            let optimizer_str = match options.optimizer {
                Optimizer::Lbfgs => "lbfgs",
                _ => "bfgs",
            };
            // Recompute the real (unscaled) point rather than reuse
            // `x_new_real`, which may already have been moved into the
            // predictor's anchor above.
            let x_real: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
            let values = crate::estimation::parameterization::coordinate_values(&unpack_params(
                &x_real,
                init_params,
            ));
            crate::estimation::trace::write_foce(
                iter,
                method_str,
                clean_ofv_at(&xs, f_val, &scale),
                Some(g_norm),
                Some(step_norm),
                optimizer_str,
                None,
                None,
                &values,
                Some(&g_for_trace),
            );
        }

        // Checkpoint (#755): unscale the best-seen point when a write is due (the
        // trace's `x_real` is scoped to the trace block above). Armijo makes the
        // accepted iterates of this loop monotone in `f_val`, so the incumbent is
        // normally this iteration's point; tracking it explicitly keeps the
        // written point correct anyway when the penalized objective the line
        // search ranks on and the clean OFV the checkpoint reports diverge under
        // NN regularization, and keeps every driver on one rule (#1317).
        best.observe(iter, &xs, f_val, clean_ofv_at(&xs, f_val, &scale));
        if crate::io::checkpoint::is_due() {
            best.write_checkpoint(|best_xs| (0..n).map(|i| best_xs[i] * scale[i]).collect());
        }

        let rel_change = (f_val - prev_ofv).abs() / (f_val.abs() + 1.0);
        if rel_change < 1e-8 && g_norm < 0.1 {
            if options.verbose {
                eprintln!(
                    "Converged at iteration {} (rel OFV change: {:.2e}, |g| = {:.2e})",
                    iter, rel_change, g_norm
                );
            }
            converged = true;
            exited_early = true;
            break;
        }
    }

    // Unscale xs back to real (log/Cholesky) space for unpacking and covariance.
    let x_final: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    let final_params = unpack_params(&x_final, init_params);
    let bfgs_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (final_ehs, final_hms, _, final_kappas) = run_inner_loop_warm_seeded(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&cached_etas),
        Some(&bfgs_final_mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
        InnerHessianSeed::for_options(options),
        options,
        cached_etas_kind.get(),
    );
    let final_ofv = ofv_at_fixed(&x_final, &final_ehs, &final_hms, &final_kappas);

    let out = crate::estimation::covariance::run_covariance_step(
        &x_final,
        init_params,
        model,
        population,
        &final_ehs,
        &final_hms,
        &final_kappas,
        options,
        options.verbose.then_some("Computing covariance matrix..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
        method: covariance_method,
        newton_decrement,
        non_stationary_optimum,
    } = out;
    warnings.extend(cov_warnings);
    // #2785: the "did not converge" line below covers this demotion too.
    if non_stationary_optimum {
        converged = false;
    }

    // #1303 — same gate as the NLopt path above, on this path's own `final_ofv`,
    // which is likewise recomputed by the final inner loop and can therefore
    // disagree with the descent's own stop rule.
    if let Some(w) = gate_converged_on_objective(&mut converged, final_ofv) {
        warnings.push(w);
    }
    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }

    OuterResult {
        // The exact packed vector this stage's inline covariance step used (#816
        // follow-up): reused by `run_covariance` to avoid re-decomposing omega.
        packed_estimate: Some(x_final.clone()),
        ofv: final_ofv,
        converged,
        n_iterations,
        eta_hats: final_ehs,
        h_matrices: final_hms,
        kappas: final_kappas,
        covariance_matrix,
        covariance_method,
        covariance_wall_time_secs,
        warnings,
        sir_fallback_proposal,
        newton_decrement,
        eval_budget_spent: crate::estimation::gauss_newton::gn_loop_budget_spent(
            options.outer_maxiter,
            exited_early,
        ),
        ..OuterResult::from_params(final_params)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Shared utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Central-FD `d(OFV)/d(x)` that **re-converges the EBEs at every perturbed
/// point** (warm-started from `warm_etas` with `warm_kappas`, [`joint_warm_starts`], so an
/// IOV probe starts at the mode's κ̂ rather than κ = 0, #2261), rather than holding them
/// fixed. Every probe starts at the base mode, so it re-solves at the stencil tolerance
/// ([`stencil_inner_tol`]): at a loose `inner_tol` the solve could stop where it began.
///
/// For IOV models the variance components — especially `omega_iov` — are
/// weakly identified, and the EBE response dominates their gradient: raising
/// `omega_iov` un-shrinks the per-occasion kappas and improves the fit, an
/// effect the fixed-EBE gradient ([`ad_population_gradient`]) misses entirely.
/// The result is that gradient optimizers leave `omega_iov` pinned at its
/// initial value while derivative-free methods (which re-solve the EBEs at
/// each trial point) move it freely. Re-converging the inner loop inside the
/// FD stencil restores the correct descent direction. See issue #101 rec #2.
///
/// This costs `2·n_free` inner-loop solves per gradient, so it is gated to IOV
/// models (`model.n_kappa > 0`) and to TTE / discrete models with η, whose fixed-EBE
/// score drops a log|H̃| term (`gauss_newton::fixed_ebe_score_drops_ebe_response`,
/// #1778); every other path keeps the cheap analytical fixed-EBE gradient, which
/// already converges OMEGA correctly (issue #99).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconverged_fd_gradient(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    warm_etas: &[DVector<f64>], warm_kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let fixed = packed_fixed_mask(init_params);
    let warm = joint_warm_starts(warm_etas, warm_kappas);
    let probe = |xv: &[f64], tol: f64| {
        reconverged_probe_ofv(xv, init_params, model, population, &warm, options, tol)
    };
    let tol = stencil_inner_tol(options.inner_tol);
    let eval = |xv: &[f64]| probe(xv, tol).0;
    let tight = |xv: &[f64]| {
        let (v, stats) = probe(xv, crate::estimation::fd_resolution::tight_tol(tol));
        crate::estimation::fd_resolution::inner_resolved(&stats).then_some(v)
    };
    // Same bounded central-difference policy as the per-subject reconverged-FD gradients —
    // shared so the `eps`/clamp/missing-probe convention can't drift (#466 review round 4 #8).
    let floor = fd_step_floors(init_params, model);
    central_diff_packed_gated(x, &fixed, &floor, bounds, eval, Some(&tight))
}

/// One probe of [`reconverged_fd_gradient`]: the OFV at packed `xv`, re-solving the inner
/// loop warm from `warm` at the stencil tolerance. Matches the objective closure's
/// definition, 2·pop_nll, and its guard: a probe the objective would reject (non-finite,
/// or excess EBE non-convergence) is the same `1e20` wall, which [`central_diff_packed_gated`]
/// treats as missing rather than as a value (#2378). `tol` is the inner tolerance — the
/// stencil's, or its tight twin's — and the inner loop's stats come back with the value.
fn reconverged_probe_ofv(
    xv: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    warm: &[DVector<f64>], options: &FitOptions, tol: f64,
) -> (f64, InnerLoopStats) {
    let params = unpack_params(xv, init_params);
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm_seeded(
        model,
        population,
        &params,
        options.inner_maxiter,
        tol,
        Some(warm),
        Some(&mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
        InnerHessianSeed::for_options(options),
        options,
        // The modes the gradient is taken at.
        PrevEtas::Solved,
    );
    let raw = 2.0 * pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
    let n_subj = population.subjects.len();
    let v = if !raw.is_finite()
        || ebe_guard_rejects(&ebe_stats, n_subj, raw, options.max_unconverged_frac)
    {
        1e20
    } else {
        raw
    };
    (v, ebe_stats)
}

/// The gradient of the objective the covariance step's `R` is the Hessian of, at `x` (#2541):
/// the OFV, plus the prior penalty, plus `nn_reg`'s covariate-NN penalty. `nn_reg` is the
/// regularizer whose curvature the step adds to `R`
/// ([`crate::estimation::covariance::cov_nn_regularizer`]; `None` when the estimator did not
/// minimise the penalty), so `g` and `H` take the penalty from one value and cannot disagree
/// about it (#2541 review r1 §4). Exact where [`exact_reporting_gradient`] is, otherwise its
/// central-FD twin [`reporting_fd_gradient`], through the same [`reporting_gradient`] dispatch
/// the post-fit reporting gradient takes. `options` is the covariance step's, so a #1781
/// quadrature objective is the one differentiated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cov_objective_gradient(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    ehs: &[DVector<f64>], hms: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds, options: &FitOptions,
    nn_reg: Option<&crate::estimation::nn_reg::NnRegularizer>,
) -> Vec<f64> {
    // No penalty: zero λ builds the inert regularizer (the λ fields are read by nothing else).
    let inert;
    let nn_reg = match nn_reg {
        Some(reg) => reg,
        None => {
            let zero_lambda =
                FitOptions { nn_l2_lambda: 0.0, nn_smooth_lambda: 0.0, ..options.clone() };
            inert =
                crate::estimation::nn_reg::NnRegularizer::build(model, population, &zero_lambda);
            &inert
        }
    };
    let priors = build_prior_set(model, init_params);
    reporting_gradient(
        x,
        init_params,
        model,
        population,
        ehs,
        hms,
        kappas,
        bounds,
        options,
        nn_reg,
        &priors,
    )
    .0
}

/// The penalized outer objective's gradient at `x`, and where it came from: exact
/// ([`exact_reporting_gradient`], `"analytic"`) where it is available, else central
/// differences ([`reporting_fd_gradient`], `"finite_difference"`). One dispatch for the
/// post-fit `final_gradient` and the covariance step's stationarity test (#2541 review r1 §5).
#[allow(clippy::too_many_arguments)]
fn reporting_gradient(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    ehs: &[DVector<f64>], hms: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds, options: &FitOptions, nn_reg: &crate::estimation::nn_reg::NnRegularizer,
    priors: &crate::estimation::priors::PriorSet,
) -> (Vec<f64>, &'static str) {
    match exact_reporting_gradient(
        x,
        init_params,
        model,
        population,
        ehs,
        hms,
        kappas,
        bounds,
        options,
        nn_reg,
        priors,
    ) {
        Some(g) => (g, "analytic"),
        None => (
            reporting_fd_gradient(
                x,
                init_params,
                model,
                population,
                ehs,
                kappas,
                bounds,
                options,
                nn_reg,
                priors,
            ),
            "finite_difference",
        ),
    }
}

/// The exact gradient of the **penalized** outer objective at the reported estimates, for a
/// run whose optimizer supplied none — what [`reporting_fd_gradient`] differences, taken
/// instead from the analytic outer gradient at the final modes (the EBE response included,
/// so it is the marginal's gradient, as the stencil's re-solved EBEs make that one), plus the
/// NN-penalty and prior gradients. `None` — the caller takes the stencil — outside the
/// analytic gradient's model scope, under `reconverge_gradient_interval`, when any subject
/// would take a finite-difference salvage (the label would then claim more than
/// it delivers; a subject served exactly at its re-solved EBE, [`declined_salvage`], is
/// exact), or on a non-finite component. A mixture differentiates its K-fold objective at every class's
/// modes, re-solved once at the estimates.
#[allow(clippy::too_many_arguments)]
fn exact_reporting_gradient(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    final_ehs: &[DVector<f64>], final_hms: &[DMatrix<f64>], final_kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds, options: &FitOptions, nn_reg: &crate::estimation::nn_reg::NnRegularizer,
    priors: &crate::estimation::priors::PriorSet,
) -> Option<Vec<f64>> {
    if !crate::sens::provider::analytic_outer_gradient_available(model)
        || reconverge_this_eval(options, 0)
    {
        return None;
    }
    let params = unpack_params(x, init_params);
    let mut g = if params.mixture.is_some() {
        let (eval, modes) = crate::estimation::mixture::mixture_ofv_with_modes(
            model, population, &params, options, None,
        );
        crate::estimation::mixture::mixture_gradient_with_modes(
            model,
            population,
            &params,
            options,
            &eval,
            Some(&modes),
        )?
    } else {
        if final_ehs.len() != population.subjects.len() {
            return None;
        }
        let declines = OuterFdDeclineLog::new(population.subjects.len());
        let mut idx = 0usize;
        let g = population_gradient_with_agq_evaluation(
            x,
            population.subjects.len(),
            init_params,
            model,
            population,
            final_ehs,
            final_hms,
            final_kappas,
            bounds,
            options,
            &mut idx,
            None,
            OuterTrial::unknown(),
            &declines,
            None,
        );
        if !declines.all_declines_exact() {
            return None;
        }
        g
    };
    nn_reg.add_packed_gradient(&params.theta, &mut g);
    priors.add_gradient(x, &mut g);
    for (gk, &fixed) in g.iter_mut().zip(&packed_fixed_mask(init_params)) {
        if fixed {
            *gk = 0.0;
        }
    }
    (g.len() == x.len() && g.iter().all(|v| v.is_finite())).then_some(g)
}

/// Central-FD gradient of the **penalized** outer objective at the reported
/// estimates, for a run whose optimizer supplied no gradient of its own (#997 §1).
///
/// A derivative-free fit (BOBYQA, the `optimizer = auto` choice whenever the
/// analytic FOCE/FOCEI gradient is unavailable) reports `converged` with
/// `final_gradient = None`, so there is nothing in the result to tell "the
/// objective stopped moving" from "the optimizer stopped moving" — the two arms
/// #997 measured 1.54 OFV apart, both reporting success. This computes the
/// missing quantity once, after the fit, purely so the claim is checkable.
///
/// It is **not** [`reconverged_fd_gradient`] with a different name, and the
/// difference is the reason for the second function rather than a flag on the
/// first: that one is a gradient *fed to the optimizer*, so it differentiates the
/// likelihood alone and the caller splices the NN-penalty and prior gradients in
/// afterwards (they are available analytically in the same pass). Here there is
/// no caller to splice anything: the quantity that has to come out is
/// `∇(OFV + penalty)` — the objective the fit actually converged on, matching
/// [`FitResult::final_gradient`](crate::types::FitResult::final_gradient)'s
/// documented contract — so the penalties go inside the stencil. The EBEs are
/// re-solved (warm-started from the final ones, κ̂ included) at every perturbed point, like
/// the objective closure itself, rather than held fixed.
///
/// Cost: `2·n_free` objective evaluations, one gradient's worth.
#[allow(clippy::too_many_arguments)]
fn reporting_fd_gradient(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    warm_etas: &[DVector<f64>], warm_kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds,
    options: &FitOptions, nn_reg: &crate::estimation::nn_reg::NnRegularizer,
    priors: &crate::estimation::priors::PriorSet,
) -> Vec<f64> {
    let n_subj = population.subjects.len();
    let fixed = packed_fixed_mask(init_params);
    let warm = joint_warm_starts(warm_etas, warm_kappas);
    // `tight` re-solves at the gate's τ/100 (#2406): the mixture branch reads its inner
    // tolerance from the options it is handed, so it takes an override clone.
    let tight_options = FitOptions {
        inner_tol: crate::estimation::fd_resolution::tight_tol(options.inner_tol),
        ..options.clone()
    };
    let stencil_tol = stencil_inner_tol(options.inner_tol);
    let probe = |xv: &[f64], tight: bool| -> (f64, InnerLoopStats) {
        let params = unpack_params(xv, init_params);
        // Mirror the objective closure's own two branches, so the mixture path
        // differentiates the K-fold log-sum-exp it minimised rather than a
        // single-class stand-in for it.
        let (raw, stats) = if params.mixture.is_some() {
            let mixture_options = if tight { &tight_options } else { options };
            let m = crate::estimation::mixture::mixture_ofv(
                model,
                population,
                &params,
                mixture_options,
                None,
            );
            let stats = InnerLoopStats {
                n_unconverged: m.ebe_stats.n_unconverged,
                n_fallback: m.ebe_stats.n_fallback,
                n_nm_polished: m.ebe_stats.n_nm_polished,
                n_start_rejected: m.ebe_stats.n_start_rejected,
            };
            (m.ofv, stats)
        } else {
            let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
            let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm_seeded(
                model,
                population,
                &params,
                options.inner_maxiter,
                if tight {
                    crate::estimation::fd_resolution::tight_tol(stencil_tol)
                } else {
                    stencil_tol
                },
                Some(&warm),
                Some(&mu_k),
                options.min_obs_for_convergence_check as usize,
                options.inner_restarts,
                InnerHessianSeed::for_options(options),
                options,
                // The final modes.
                PrevEtas::Solved,
            );
            let nll = pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
            (2.0 * nll, ebe_stats)
        };
        let raw = raw + nn_reg.penalty_value(&params.theta) + priors.penalty(xv);
        let v = if !raw.is_finite()
            || ebe_guard_rejects(&stats, n_subj, raw, options.max_unconverged_frac)
        {
            1e20
        } else {
            raw
        };
        (v, stats)
    };
    let eval = |xv: &[f64]| probe(xv, false).0;
    let tight = |xv: &[f64]| {
        let (v, stats) = probe(xv, true);
        crate::estimation::fd_resolution::inner_resolved(&stats).then_some(v)
    };
    let floor = fd_step_floors(init_params, model);
    central_diff_packed_gated(x, &fixed, &floor, bounds, eval, Some(&tight))
}

/// Bounded central-difference of a packed-space scalar `eval`, skipping fixed
/// coordinates. Shared by the population and per-subject reconverged-FD gradients so they
/// cannot drift (#466 review round 2).
///
/// A probe `eval` cannot score is missing, not a value: any non-finite value, or the `1e20`
/// wall and above — the population stencils' EBE guard and the per-subject sentinel alike
/// ([`is_usable_subject_nll`](crate::stats::likelihood::is_usable_subject_nll)).
/// The sentinel is finite, so differenced as a value it gave a component of ~`1e20/2h`
/// (#1537, #2378); the test lives here, once, so no caller can forget to mask. With one
/// side missing the component is the one-sided difference against `eval(x)`, evaluated
/// once and only when needed; with both missing, or the centre missing too, it is zero.
///
/// Coordinate `k` is stepped by [`fd_step`]`(1e-4, xₖ, floor[k])`, `floor` from
/// [`fd_step_floors`]: `1e-4·(1 + |xₖ|)` on every coordinate but a small identity-packed θ,
/// which is stepped relative to its declared magnitude rather than by an absolute `1e-4` in
/// its own units — 12 % of a θ of 8e-4, and straight through zero below 1e-4 (#2187). A
/// `[covariate_nn]` weight is identity-packed too but keeps the unit floor, since its θ₀ is
/// a draw rather than a declared magnitude (#2406).
///
/// That relative step is only as good as the probes resolve it, so a coordinate whose floor is
/// below 1 goes through the resolution gate of
/// [`resolve_floored_step`](crate::estimation::fd_resolution::resolve_floored_step) (#2406):
/// `tight` is `eval` re-solved at τ/100, `None` where a solve fell to Nelder–Mead or did not
/// converge, and the step grows ×10 until the two agree or growth runs into truncation error.
/// A coordinate whose `tight` was `None` at every floor keeps its first step's difference.
/// Without `tight` no coordinate is gated, and a floor-1 coordinate takes the ungated difference
/// either way. A floored one for which no floor gave a difference takes the ungated difference
/// at its own floor, with its one-sided fallback — the covariance gate's fallback too. Floor 1
/// was the worst step there: on a `V = TVV/k` volume it is the step through `V ≤ 0` that made
/// the probe go missing (#2406 review r1).
pub(crate) fn central_diff_packed_gated(
    x: &[f64], fixed: &[bool], floor: &[f64], bounds: &PackedBounds, eval: impl Fn(&[f64]) -> f64,
    tight: Option<&dyn Fn(&[f64]) -> Option<f64>>,
) -> Vec<f64> {
    let n = x.len();
    let eps = 1e-4;
    let mut grad = vec![0.0_f64; n];
    let mut xw = x.to_vec();
    let usable = crate::stats::likelihood::is_usable_subject_nll;
    let mut centre: Option<f64> = None;
    for k in 0..n {
        if fixed[k] {
            continue;
        }
        let floor_k = floor[k];
        if let Some(tight) = tight.filter(|_| floor_k < 1.0) {
            let resolved =
                crate::estimation::fd_resolution::resolve_floored_step(floor_k, |fl, at_tight| {
                    let h = fd_step(eps, x[k], fl);
                    let xp = (x[k] + h).min(bounds.upper[k]);
                    let xm = (x[k] - h).max(bounds.lower[k]);
                    if xp <= xm {
                        return None;
                    }
                    let mut probe = |xk: f64| {
                        xw[k] = xk;
                        let v = if at_tight { tight(&xw) } else { Some(eval(&xw)) };
                        xw[k] = x[k];
                        v.filter(|&v| usable(v))
                    };
                    let fp = probe(xp)?;
                    let fm = probe(xm)?;
                    Some(crate::estimation::fd_resolution::Difference {
                        value: (fp - fm) / (xp - xm),
                        numer: (fp - fm).abs(),
                        scale: fp.abs().max(fm.abs()),
                        span: xp - xm,
                    })
                });
            if let Some(r) = resolved {
                if r.value.is_finite() {
                    grad[k] = r.value;
                }
                continue;
            }
        }
        let h = fd_step(eps, x[k], floor_k);
        let xp = (x[k] + h).min(bounds.upper[k]);
        let xm = (x[k] - h).max(bounds.lower[k]);
        // Skip only probes that collapsed onto one point (a box pinned to `x`). The span is
        // in the coordinate's own units, so an absolute cutoff would zero the gradient of a
        // θ declared below ~1e-12, whose relative step is that small (#2187 review).
        if xp <= xm {
            continue;
        }
        let denom = xp - xm;
        xw[k] = xp;
        let fp = eval(&xw);
        xw[k] = xm;
        let fm = eval(&xw);
        xw[k] = x[k];
        let d = match (usable(fp), usable(fm)) {
            (true, true) => (fp - fm) / denom,
            (false, false) => continue,
            (plus, _) => {
                let f0 = *centre.get_or_insert_with(|| eval(x));
                if !usable(f0) {
                    continue;
                }
                if plus {
                    (fp - f0) / (xp - x[k])
                } else {
                    (f0 - fm) / (x[k] - xm)
                }
            }
        };
        if d.is_finite() {
            grad[k] = d;
        }
    }
    grad
}

/// [`central_diff_packed_gated`] without the gate: the difference every coordinate took before
/// #2406, which the tests pin the shared policy on.
#[cfg(test)]
fn central_diff_packed(
    x: &[f64], fixed: &[bool], floor: &[f64], bounds: &PackedBounds, eval: impl Fn(&[f64]) -> f64,
) -> Vec<f64> {
    central_diff_packed_gated(x, fixed, floor, bounds, eval, None)
}

/// Per-subject packed gradient `dᵢ = d(nllᵢ)/dx` at the **held** EBE `η̂ᵢ` and held
/// prediction Jacobian — the per-subject term of [`ad_population_gradient`] and of the
/// fixed-EBE scores the Gauss-Newton and trust-region paths difference. No declined subject
/// takes it any more ([`declined_salvage`], #3004).
///
/// It costs `2·n_free` objective evaluations of this one subject (or one closed-form
/// Laplace / Sheiner–Beal pass where `subject_nll_pop_grad` has one) and **no** inner
/// re-solve. For FOCEI the envelope theorem makes the held-η̂ data/prior gradient exact;
/// what it omits is the `½·∂log|H̃|/∂η · dη̂/dx` EBE response, the same term the
/// `reconverge_gradient_interval` schedule governs for every other fixed-EBE gradient.
///
/// Returns `(nllᵢ, d(nllᵢ)/dx)`, the gradient of length `x.len()`; the caller scales by
/// 2 and zeroes fixed coordinates, matching the analytic per-subject convention. `nllᵢ`
/// is the held-EBE objective at `x`, which tells a repelled subject from a real one
/// ([`structural_fill`]).
#[allow(clippy::too_many_arguments)]
fn subject_fixed_ebe_gradient(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    subj_idx: usize, eta_hat: &DVector<f64>, h_matrix: &DMatrix<f64>, kappas: &[DVector<f64>],
    bounds: &PackedBounds, options: &FitOptions,
) -> (f64, Vec<f64>) {
    // The held-EBE score as is: it omits the `log|H̃|` EBE-response term. A subject whose
    // response matters never takes this gradient: a declined subject is served at its
    // re-solved EBE or reconverged ([`declined_salvage`]).
    crate::estimation::gauss_newton::subject_nll_pop_grad(
        x,
        init_params,
        model,
        population,
        subj_idx,
        eta_hat,
        h_matrix,
        kappas,
        bounds,
        options,
    )
}

/// Is subject `subject`'s analytic outer-gradient decline **structural** — one it takes
/// at every parameter point of the fit, rather than the parameter-dependent decline
/// (non-PD inner Hessian at a blown-up trial) #1529 made cheap? (#1536)
///
/// For such a subject the exact route can never serve it, so it takes the reconverged
/// gradient at every evaluation (a fixed-EBE gradient's missing EBE-response term, once the
/// stand-in, stalled L-BFGS on `fluconazole_radboudumc` at `‖g‖ ≈ 655`, OFV 807.88–810.28
/// against 738.05; NONMEM 734.64, #1536).
/// The adjoint's kernel scores dense residual covariances itself, so the one structural
/// cause left is a subject no recorder serves: its decline is scope, not numerics, and
/// recurs at every evaluation exactly as a model-level exclusion does.
pub(crate) fn is_structural_decline(
    model: &CompiledModel, subject: &Subject, init_params: &ModelParameters, eta_hat: &[f64],
) -> bool {
    // A subject no recorder can serve declines at every evaluation, as a model-level
    // exclusion does; any other decline is numerical (a non-PD factor at a trial point).
    crate::sens::ode_provider::adjoint::exact_eta_jacobian(
        model,
        subject,
        &init_params.theta,
        eta_hat,
    )
    .is_none()
}

/// The salvage a declined subject takes in [`population_gradient_sens_mixed_dx`] and its IOV
/// twin: **never** the held-EBE gradient, which omits the `½·∂log|H̃|/∂η · dη̂/dx` EBE-response
/// term and so biases every step it enters (#1529, #1536, #3004).
///
/// A decline is either *structural* ([`is_structural_decline`]: no recorder serves the
/// subject, at any parameter point) or a runtime one (the mode system would not solve at the
/// trial point's `η̂`, or one recording failed there). A runtime decline is served exactly by
/// the adjoint at the subject's EBE re-solved at the trial point ([`exact_at_resolved_ebe`]),
/// which is one inner solve and one adjoint pass, where the reconverged difference costs
/// `2·n_free` solves. Only where that fails too (no recorder serves it) or for a structural
/// decline is the gradient the reconverged finite-difference score
/// ([`subject_reconverged_fd_gradient`]), which carries the EBE response by re-solving at
/// every probe. Each outcome is recorded on `declines`, so the warning and
/// [`exact_reporting_gradient`] say which subjects were served exactly.
///
/// The second element is the stacked `[η̂, κ̂…]` the exact route took its gradient at, handed
/// back exactly as a polished pass hands its mode ([`adopt_polished_modes`]): the store then
/// holds the mode the gradient differentiated at, so the next evaluation warm-starts there
/// instead of declining again from the same off-mode `η̂`. `None` for a reconverged salvage,
/// whose probes each re-solve their own mode.
///
/// The #1520 guard runs first ([`declined_subject_gradient`]): at a blown-up SLSQP/MMA trial
/// the subject contributes zero and this function is not called.
#[allow(clippy::too_many_arguments)]
fn declined_salvage(
    structural: bool, subj_idx: usize, x: &[f64], init_params: &ModelParameters,
    model: &CompiledModel, subject: &Subject, warm: &DVector<f64>, bounds: &PackedBounds,
    options: &FitOptions, declines: &OuterFdDeclineLog,
) -> (Vec<f64>, Option<Vec<f64>>) {
    if !structural {
        if let Some((g, mode)) =
            exact_at_resolved_ebe(x, init_params, model, subject, warm, options)
        {
            declines.record_exact(subj_idx);
            return (g, Some(mode));
        }
        declines.record_reconverged(subj_idx);
    }
    (subject_reconverged_fd_gradient(x, init_params, model, subject, warm, bounds, options), None)
}

/// The exact analytic outer gradient of one subject at its EBE re-solved at `x` (warm from
/// `warm`, the stacked `[η̂, κ̂…]` under IOV) at the FD stencil's tight tolerance: the
/// gradient a declined subject would have had, had the mode the optimizer handed the
/// adjoint been a stationary one. `None` when the re-solve is not resolved or the adjoint
/// declines the subject there too. Alongside the gradient, the stacked mode it was taken at.
fn exact_at_resolved_ebe(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, subject: &Subject,
    warm: &DVector<f64>, options: &FitOptions,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(init_params, x)?;
    let (_, ebe) = resolve_subject_ebe(
        x,
        init_params,
        model,
        subject,
        warm.as_slice(),
        crate::estimation::fd_resolution::tight_tol(options.inner_tol),
        options,
    );
    if !crate::estimation::fd_resolution::ebe_resolved(&ebe) {
        return None;
    }
    let stacked = joint_warm_start(&ebe.eta, &ebe.kappas);
    let (g, _, polished) = subject_analytic_outer_gradient_dx(
        model,
        subject,
        &pt,
        stacked.as_slice(),
        options.interaction,
        false,
    );
    // A re-solve left above the polish target is polished inside the pass: the gradient is
    // taken at that mode, so that is the mode the store must adopt, not the re-solve.
    Some((g.ok()?, polished.unwrap_or_else(|| stacked.as_slice().to_vec())))
}

/// A declined non-IOV subject's salvage ([`declined_salvage`]), unless it is **repelled** at
/// the trial point: its objective at the held `η̂`, `h_matrix` is NaN/∞ or the `1e20`
/// sentinel ([`structural_fill`]). It then contributes zero: the population objective there
/// carries the sentinel, so no line search accepts the point and its gradient is never used
/// to take a step, while a re-solve is exactly the #1529 cost at the very points subjects get
/// repelled (#1537 review).
#[allow(clippy::too_many_arguments)]
fn non_iov_declined_salvage(
    structural: bool, x: &[f64], init_params: &ModelParameters, model: &CompiledModel,
    population: &Population, subj_idx: usize, eta_hat: &DVector<f64>, h_matrix: &DMatrix<f64>,
    bounds: &PackedBounds, options: &FitOptions, declines: &OuterFdDeclineLog,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let subject = &population.subjects[subj_idx];
    let params = unpack_params(x, init_params);
    let nll = crate::stats::likelihood::foce_subject_nll_cov(
        model,
        subject,
        &params.theta,
        eta_hat,
        h_matrix,
        &params.omega,
        &params.sigma.values,
        model.residual_cov_at(&params),
        options.interaction,
    );
    match structural_fill(nll) {
        DeclinedFill::Zero => {
            declines.record_repelled(subj_idx);
            (vec![0.0; x.len()], None)
        }
        DeclinedFill::Reconverge => declined_salvage(
            structural,
            subj_idx,
            x,
            init_params,
            model,
            subject,
            eta_hat,
            bounds,
            options,
            declines,
        ),
    }
}

/// Which fill a declined subject's salvage uses, keyed on its objective at the held `η̂`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeclinedFill {
    Zero,
    Reconverge,
}

/// The fill for a declined subject ([`non_iov_declined_salvage`]), keyed on its held-EBE
/// objective at `x`: [`DeclinedFill::Zero`] when it is repelled, otherwise
/// [`DeclinedFill::Reconverge`] (the exact or reconverged salvage).
fn structural_fill(nll: f64) -> DeclinedFill {
    if crate::stats::likelihood::is_usable_subject_nll(nll) {
        DeclinedFill::Reconverge
    } else {
        DeclinedFill::Zero
    }
}

/// Central-FD per-subject packed gradient `dᵢ = d(nllᵢ)/dx` that **re-converges that
/// subject's EBE** (warm-started) at every perturbed point, so the Ω/σ EBE response is
/// included. `warm` is the subject's mode: `η̂`, or the stacked joint mode (`η̂` then each
/// cell's κ̂, [`joint_warm_start`]) under IOV, so every probe starts at κ̂ rather than κ = 0
/// (#2261); [`subject_mode_probe`] dispatches the IOV marginal on `n_kappa > 0`. IOV fits
/// reconverge unconditionally and ignore `reconverge_gradient_interval` (#1529). Since #1529 it serves two callers here: a structural decline
/// ([`non_iov_declined_salvage`], #1536), and the last resort of [`declined_salvage`] for
/// a subject the exact route cannot serve at its re-solved EBE (#3004). GN,
/// trust-region and the covariance step's `S` read it for a subject the adjoint declines
/// (`covariance::subject_total_score`, #2974).
#[allow(clippy::too_many_arguments)]
pub(crate) fn subject_reconverged_fd_gradient(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, subject: &Subject,
    warm: &DVector<f64>, bounds: &PackedBounds, options: &FitOptions,
) -> Vec<f64> {
    let probe = subject_mode_probe(init_params, model, subject, warm.as_slice(), options);
    subject_central_diff(x, init_params, model, bounds, options, probe)
}

/// The subject's EBE re-solved at the packed point `x` to `tol`, warm-started from `warm` (the
/// stacked `[η̂, κ̂…]` under IOV), with the parameters `x` unpacks to: the one re-solve the
/// reconverged difference ([`subject_mode_probe`]) and the exact salvage
/// ([`exact_at_resolved_ebe`]) both start from.
fn resolve_subject_ebe(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, subject: &Subject,
    warm: &[f64], tol: f64, options: &FitOptions,
) -> (ModelParameters, crate::estimation::inner_optimizer::EbeResult) {
    let params = unpack_params(x, init_params);
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let ebe = find_ebe_for_stage(
        model,
        subject,
        &params,
        options.inner_maxiter,
        tol,
        Some(warm),
        Some(&mu_k),
        0,
        options,
    );
    (params, ebe)
}

/// The subject's marginal NLL at a packed point with its mode re-solved there (warm-started
/// from `warm`, the stacked `[η̂, κ̂…]` under IOV), and whether that solve counts as resolved:
/// `probe(x, tol)`. Mirrors the objective's per-subject term (`foce_subject_nll`, summed by
/// `pop_nll`; the IOV marginal for `n_kappa > 0`). A non-finite or sentinel value is missing
/// to `central_diff_packed_gated`, which takes the one-sided difference (#2378).
pub(crate) fn subject_mode_probe<'a>(
    init_params: &'a ModelParameters, model: &'a CompiledModel, subject: &'a Subject,
    warm: &'a [f64], options: &'a FitOptions,
) -> impl Fn(&[f64], f64) -> (f64, bool) + 'a {
    move |xv: &[f64], tol: f64| -> (f64, bool) {
        let (params, ebe) =
            resolve_subject_ebe(xv, init_params, model, subject, warm, tol, options);
        let nll = if model.n_kappa > 0 {
            crate::stats::likelihood::foce_subject_nll_iov_cov(
                model,
                subject,
                &params.theta,
                &ebe.eta,
                &ebe.h_matrix,
                &params.omega,
                &params.sigma.values,
                model.residual_cov_at(&params),
                options.interaction,
                &ebe.kappas,
                params.omega_iov.as_ref().expect("IOV model (n_kappa > 0) has omega_iov"),
            )
        } else {
            crate::stats::likelihood::foce_subject_nll_cov(
                model,
                subject,
                &params.theta,
                &ebe.eta,
                &ebe.h_matrix,
                &params.omega,
                &params.sigma.values,
                model.residual_cov_at(&params),
                options.interaction,
            )
        };
        (nll, crate::estimation::fd_resolution::ebe_resolved(&ebe))
    }
}

/// The per-subject reconverged-FD gradients' shared difference: `probe(x, tol)` is the
/// subject's objective re-solved at `tol` and whether that solve counts as resolved, taken at
/// the stencil tolerance and, on a floored coordinate, at its tight twin (#2406).
fn subject_central_diff(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, bounds: &PackedBounds,
    options: &FitOptions, probe: impl Fn(&[f64], f64) -> (f64, bool),
) -> Vec<f64> {
    let fixed = packed_fixed_mask(init_params);
    let tol = stencil_inner_tol(options.inner_tol);
    let eval = |xv: &[f64]| probe(xv, tol).0;
    let tight = |xv: &[f64]| {
        let (v, resolved) = probe(xv, crate::estimation::fd_resolution::tight_tol(tol));
        resolved.then_some(v)
    };
    let floor = fd_step_floors(init_params, model);
    central_diff_packed_gated(x, &fixed, &floor, bounds, eval, Some(&tight))
}

/// The exact analytic per-subject packed **outer** gradient at the packed point `pt` the
/// population loop unpacked once, or `None` when this subject does not get one — the adjoint
/// declined the subject's data shape at runtime, or a component came back non-finite. Both
/// route the subject to its salvage, so they are one gate, not two. `eta_hat` is the
/// subject's EBE, stacked `[η_bsv, κ₁ … κ_K]` under IOV. (#1154)
#[cfg(test)]
fn subject_analytic_outer_gradient(
    model: &CompiledModel, subject: &Subject,
    pt: &crate::sens::ode_provider::adjoint::PackedPoint<'_>, eta_hat: &[f64], interaction: bool,
) -> Option<Vec<f64>> {
    let g = crate::estimation::sens_outer_gradient::subject_gradient_at(
        model,
        subject,
        pt,
        eta_hat,
        interaction,
    )?;
    g.iter().all(|v| v.is_finite()).then_some(g)
}

/// `subject_analytic_outer_gradient`, and with `want_eta_dx` the subject's mode response
/// `dη̂/dx` formed from the same recording; the response is `None` wherever the gradient is.
/// A declined gradient comes back as its [`DeclineCause`] (#2713). The third element is the
/// `η̂` the gradient was taken at when the pass polished an off-mode start to the mode
/// (stacked `[η, κ]` under IOV): the store's `η̂` and `H` are replaced by it
/// ([`adopt_polished_modes`]).
fn subject_analytic_outer_gradient_dx(
    model: &CompiledModel, subject: &Subject,
    pt: &crate::sens::ode_provider::adjoint::PackedPoint<'_>, eta_hat: &[f64], interaction: bool,
    want_eta_dx: bool,
) -> (Result<Vec<f64>, DeclineCause>, Option<Vec<DVector<f64>>>, Option<Vec<f64>>) {
    use crate::sens::ode_provider::adjoint::{subject_adjoint_gradient_and_mode_at, ModeOut};
    // No data at all contributes nothing (`sens_outer_gradient::subject_gradient_at`).
    if subject.observations.is_empty() && subject.obs_records.is_empty() {
        return (Ok(vec![0.0; pt.x.len()]), None, None);
    }
    let objective = if interaction {
        crate::sens::ode_provider::adjoint::Objective::Focei
    } else {
        crate::sens::ode_provider::adjoint::Objective::Foce
    };
    let mut out = ModeOut { want_dx: want_eta_dx, ..ModeOut::default() };
    let g = subject_adjoint_gradient_and_mode_at(model, subject, pt, eta_hat, objective, &mut out);
    match g {
        Some(g) if g.iter().all(|v| v.is_finite()) => (Ok(g), out.eta_dx, out.eta_polished),
        _ if out.mode_system_declined => (Err(DeclineCause::ModeSystem), None, None),
        _ => (Err(DeclineCause::Recording), None, None),
    }
}

/// The analytic gradient where `x` is not the template's packed length: the adjoint
/// declines, but a subject with no data at all keeps the zero gradient it gives one.
fn packed_length_decline(subject: &Subject, np: usize) -> Result<Vec<f64>, DeclineCause> {
    if subject.observations.is_empty() && subject.obs_records.is_empty() {
        Ok(vec![0.0; np])
    } else {
        Err(DeclineCause::Recording)
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch: the NLopt objective runs without the EBE warm-start predictor
    /// (`NloptState::pred_jac`), for a fixture pinning a trajectory the predictor shortens.
    static WARM_PREDICTOR_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the EBE warm-start predictor serves this fit, for every outer optimizer that
/// predicts (the NLopt objective and the built-in BFGS/L-BFGS loop): a model
/// with BSV random effects, no IOV (κ̂ is not predicted), no `[mixture]` and no
/// `[covariate_nn]` (see the NLopt call site for the measurement), whose outer gradient the
/// adjoint serves, so the mode response comes out of that gradient's own sweeps.
///
/// Laplace too: the quadrature is centred on the conditional mode, so its response is
/// FOCEI's `−(∇²_η l)⁻¹ ∂(∇_η l)/∂x` with the exact inner Hessian, formed on the quadrature
/// tape (`the_quadrature_mode_response_is_the_focei_one`). It decides a hard Laplace fit:
/// on `pk2_big` it reaches −3485.11 in 59 G instructions, against −3458.81 in 90 G without.
///
/// Not AGQ with more than one node. Each response row is one more sweep of a tape that grows
/// with the node grid, while the inner work it saves does not: on `pk2_big` at three nodes
/// per effect (243) the predictor cost 1.5× (exact anchor) and 1.2× (Gauss–Newton) for
/// 0.026 and 0.013 OFV, and on the `pk1` AGQ-3 fits 0.88–1.02× for none (#2549).
fn warm_predictor_serves(
    model: &CompiledModel, init_params: &ModelParameters, options: &FitOptions,
) -> bool {
    options.agq_nodes().is_none_or(|n| n == 1)
        && model.n_eta > 0
        && model.n_kappa == 0
        && init_params.mixture.is_none()
        && !model.has_covariate_nn()
        && crate::sens::ode_provider::adjoint::outer_model_in_scope(model)
        && !warm_predictor_off()
        // The A/B timing switch, for every optimizer that predicts.
        && std::env::var("FERX_EBE_PREDICTOR").map_or(true, |v| v != "0")
}

/// Whether [`WARM_PREDICTOR_OFF`] is set on this thread (never outside tests). The fit pool
/// carries it onto the thread that runs a fit (`pool`'s scope arming).
pub(crate) fn warm_predictor_off() -> bool {
    #[cfg(test)]
    return WARM_PREDICTOR_OFF.with(|c| c.get());
    #[cfg(not(test))]
    false
}

/// RAII for [`WARM_PREDICTOR_OFF`]: set while alive, the previous value restored on drop (so
/// a guard the fit pool re-arms on the caller's own thread leaves the caller's in force).
#[cfg(test)]
pub(crate) struct WarmPredictorOff(bool);

#[cfg(test)]
impl WarmPredictorOff {
    pub(crate) fn new() -> Self {
        Self(WARM_PREDICTOR_OFF.with(|c| c.replace(true)))
    }
}

#[cfg(test)]
impl Drop for WarmPredictorOff {
    fn drop(&mut self) {
        WARM_PREDICTOR_OFF.with(|c| c.set(self.0));
    }
}

/// What the outer-gradient assembly knows about the point it is asked to differentiate,
/// beyond `x` itself (#1520): the objective there, the per-subject contributions that
/// objective is the sum of, and the same pair at the **incumbent** — the best evaluation
/// the optimizer has accepted so far. [`skip_fd_salvage`] is its only reader.
/// [`OuterTrial::unknown`] disarms the guard: the first evaluation of a fit, the
/// `freeze_flat_thetas` pre-flight, the mixture branch (no per-subject decomposition) and
/// every test that wants the plain assembly pass it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OuterTrial<'a> {
    /// The objective the optimizer ranks this point on — penalties included, the EBE-guard
    /// sentinel excluded (a guard-rejected evaluation never asks for a gradient).
    pub ofv: f64,
    /// `2·nllᵢ` per subject at this point, in `population.subjects` order.
    pub contribs: &'a [f64],
    /// The incumbent's `(ofv, contribs)`, or `None` before one exists.
    pub incumbent: Option<(f64, &'a [f64])>,
}

impl OuterTrial<'static> {
    /// No incumbent: the guard cannot fire and every declined subject takes the salvage.
    pub(crate) fn unknown() -> Self {
        Self { ofv: f64::NAN, contribs: &[], incumbent: None }
    }
}

/// Which incumbent the #1520 salvage guard measures a trial point against, per NLopt
/// algorithm — because "this point cannot be accepted" is a property of each optimizer's
/// **acceptance test**, not of the objective values alone (PR #1525 review, P1).
///
/// The guard's population gate is only a rejected-trial guarantee when the reference it
/// compares against is the point the optimizer's own acceptance test compares against.
/// That point is not, in general, the best evaluation seen: a rejected trial can undercut
/// the current iterate without passing sufficient decrease, and SLSQP's inexact line
/// search accepts its eleventh trial **without** sufficient decrease (`slsqp.c`, `L200`:
/// `if (h1 <= h3 / ten || line > 10) goto L240`), after which `L240` requests the
/// gradient at that point as the new iterate. Measured against the best evaluation, a
/// guard could then drop a subject's term at an accepted point and feed the hole into
/// SLSQP's BFGS update. So each algorithm gets the reference its own test justifies, or
/// none:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SalvageGuardPolicy {
    /// Never fire. NLopt L-BFGS (Luksan `plis`): its line search requests a gradient at
    /// every trial and accepts relative to the *current iterate*, which ferx cannot
    /// observe — a trial that satisfies sufficient decrease but fails the curvature
    /// condition is rejected yet can be the best evaluation seen, and the eventually
    /// accepted point can sit any distance above it. No reference ferx can form gives a
    /// rejected-trial guarantee there, so the salvage is always bought. Also the
    /// derivative-free pre-search and BOBYQA, which never form a gradient.
    Off,
    /// NLopt SLSQP: fire only at a **fresh** point (one whose `xs` differs bitwise from
    /// the previous evaluation's) and measure it against the **last point at which a
    /// gradient was requested**. With bounds only (ferx adds no nonlinear constraints)
    /// SLSQP requests a gradient in exactly two situations: `mode = −2`, the *first*
    /// trial of a line search, before its sufficient-decrease test; and `mode = −1`, the
    /// accepted iterate — which was evaluated objective-only (`mode = 1`) at the same
    /// `xs` immediately before, unless it *was* the first trial, in which case its
    /// gradient is already in hand and no `−1` follows. Hence a gradient request at a
    /// fresh point is a first trial, the previous gradient point is the current iterate
    /// (either the accepted `−1` re-evaluation or an accepted first trial), and a first
    /// trial worse than the iterate fails `h1 <= h3/10` and is rejected — the guarantee.
    /// The `line > 10` acceptance is a `−1` re-evaluation at a non-fresh point, where the
    /// guard is off by construction.
    LastGradientPointIfFresh,
    /// NLopt MMA: measure against the **best evaluation** seen. MMA copies a candidate's
    /// gradient into its model only inside `if (fcur < *minf …)` (`mma.c`), and with
    /// bounds only every improving evaluation is accepted, so the best evaluation *is*
    /// the acceptance reference and a point worse than it is never accepted.
    BestEvaluation,
}

impl SalvageGuardPolicy {
    /// The policy for the outer optimizer a fit resolved to. `Auto` is resolved before
    /// dispatch; if it ever reached here it would map to `Off`, the safe default.
    pub(crate) fn for_optimizer(optimizer: Optimizer) -> Self {
        match optimizer {
            Optimizer::Slsqp => Self::LastGradientPointIfFresh,
            Optimizer::Mma => Self::BestEvaluation,
            _ => Self::Off,
        }
    }
}

/// The per-fit bookkeeping behind [`SalvageGuardPolicy`]: what the NLopt objective
/// closure has to remember between evaluations to hand the mixed assemblies an
/// [`OuterTrial`] whose incumbent carries a rejected-trial guarantee. Fed by
/// [`SalvageGuardState::observe`] after every evaluation, read by
/// [`SalvageGuardState::reference`] before the gradient is formed. Pure bookkeeping, so
/// the optimizer sequences it exists for — SLSQP's `line > 10` acceptance among them —
/// can be scripted in a unit test without NLopt.
#[derive(Debug)]
pub(crate) struct SalvageGuardState {
    policy: SalvageGuardPolicy,
    /// Optimizer-facing objective and per-subject `2·nllᵢ` of the best non-guarded
    /// evaluation so far (the same adoption rule as the EBE warm start).
    best: Option<(f64, Vec<f64>)>,
    /// The same pair at the last evaluation whose gradient was actually formed.
    last_gradient: Option<(f64, Vec<f64>)>,
    /// `xs` of the previous evaluation of any kind, for the freshness test.
    prev_xs: Option<Vec<f64>>,
}

impl SalvageGuardState {
    pub(crate) fn new(policy: SalvageGuardPolicy) -> Self {
        Self { policy, best: None, last_gradient: None, prev_xs: None }
    }

    /// The incumbent to measure the gradient evaluation at `xs` against, or `None` when
    /// the policy gives no rejected-trial guarantee for it. Asked only where a gradient is
    /// actually formed — the closure's non-guarded gradient arm — since without one there
    /// is nothing to guard.
    pub(crate) fn reference(&self, xs: &[f64]) -> Option<(f64, &[f64])> {
        let pair = match self.policy {
            SalvageGuardPolicy::Off => None,
            SalvageGuardPolicy::LastGradientPointIfFresh => {
                let fresh = self.prev_xs.as_ref().is_none_or(|p| {
                    p.len() != xs.len() || p.iter().zip(xs).any(|(a, b)| a.to_bits() != b.to_bits())
                });
                if fresh {
                    self.last_gradient.as_ref()
                } else {
                    None
                }
            }
            SalvageGuardPolicy::BestEvaluation => self.best.as_ref(),
        };
        pair.map(|(f, c)| (*f, c.as_slice()))
    }

    /// Record an evaluation: `ofv` and `contribs` are its optimizer-facing objective and
    /// per-subject `2·nllᵢ` (empty when the objective has no per-subject decomposition),
    /// `gradient_requested` whether a gradient was actually formed there (the optimizer
    /// asked for one and the EBE guard did not reject the point), and `improved` whether
    /// the evaluation was adopted as the fit's incumbent (`adopt_warm_start`). A guarded
    /// evaluation passes `false` for both and only advances the freshness record.
    pub(crate) fn observe(
        &mut self, xs: &[f64], ofv: f64, contribs: Vec<f64>, gradient_requested: bool,
        improved: bool,
    ) {
        if gradient_requested {
            self.last_gradient = Some((ofv, contribs.clone()));
        }
        if improved {
            self.best = Some((ofv, contribs));
        }
        self.prev_xs = Some(xs.to_vec());
    }
}

/// Excess of an objective over its incumbent value, **per observation**, above which the
/// point counts as blown up (#1520). Applied to a subject (`2·nllᵢ` against its own
/// incumbent contribution, over that subject's observations) and to the population (the
/// optimizer-facing objective against the incumbent's, over every observation), by
/// [`skip_fd_salvage`].
///
/// Measured, not chosen. Every per-subject FD salvage of every gradient-based fit of the
/// bundled examples (default L-BFGS and SLSQP, 118 decline events at points worse than the
/// incumbent, 13 example × optimizer pairs) was instrumented with both excesses:
///
/// | | ordinary rejected trials | blown-up trials |
/// |---|---|---|
/// | subject excess per observation, declining subjects | ≤ **6.0** (`mm_multistart`, SLSQP) | ≥ **21.8** (`warfarin_if`, L-BFGS) |
/// | population excess per observation | ≤ **4.5** (`transit_2cpt`, SLSQP) | ≥ **36.4** (`mm_multistart`, SLSQP) |
///
/// The widest gap in either sorted distribution is the one between those columns (3.6× and
/// 8×; the next-widest is 2.0×), and 12 sits at its geometric middle: 2.0× above the
/// largest ordinary subject excess and 1.8× below the smallest blown-up one, 2.7× and 3.0×
/// for the population. On the far side the excesses run to 1.7e19 per observation (a
/// subject at the `1e20` non-finite sentinel). In likelihood terms a point over the line is
/// one at which every observation is, on average, more than `e⁻⁶` times less likely than at
/// the incumbent. Excesses rather than raw objectives because a `−2LL` difference is
/// unit-free and a raw `−2LL` is not: the incumbent objective is *negative* on 9 of the 13
/// pairs measured, so the issue's "relative to the current best objective" cannot be a
/// ratio of objectives, and a subject contributing 1e3 is ordinary on one dataset and blown
/// up on another. Per observation because a rich subject legitimately moves by tens of units
/// between ordinary trials that a sparse one covers in one.
pub(crate) const BLOWN_UP_EXCESS_PER_OBS: f64 = 12.0;

/// Whether an excess of `excess` objective units, spread over `n_obs` observations, is over
/// the [`BLOWN_UP_EXCESS_PER_OBS`] line. `n_obs` is floored at one so a subject with no
/// observation rows (a pure TTE subject) is judged on its whole excess. Any `NaN` compares
/// false, so a non-finite contribution never fires the guard.
fn blown_up(excess: f64, n_obs: usize) -> bool {
    excess > BLOWN_UP_EXCESS_PER_OBS * n_obs.max(1) as f64
}

/// Whether subject `i`'s per-subject salvage (the exact re-solve or reconverged FD) is
/// skipped at this trial point, its contribution to the population gradient dropped
/// instead (#1520). Both gates must hold:
///
/// 1. **The population is blown up**: the optimizer-facing objective exceeds the
///    incumbent's by more than [`BLOWN_UP_EXCESS_PER_OBS`] per observation, over every
///    observation in the population. This is a rejected-trial guarantee only when
///    `incumbent` is the point the optimizer's own acceptance test compares against —
///    which is what [`SalvageGuardPolicy`] supplies, per optimizer, and why NLopt L-BFGS
///    gets no incumbent at all. With that reference, a point over this line fails the
///    optimizer's sufficient-decrease test and is never an accepted iterate; the guard
///    cannot touch a curvature pair or a search direction. The built-in BFGS passes its
///    last accepted iterate, which its Armijo backtracking makes exact too.
/// 2. **The subject is blown up**: its `2·nllᵢ` exceeds its incumbent contribution by more
///    than the same line per observation of its own. This keeps the salvage for a subject
///    that is merely along for the ride at a bad point, and it is what makes the guard
///    per-subject rather than per-evaluation.
///
/// The two gates reject different inputs — a blown-up subject at a mildly worse point, and
/// an ordinary subject at a blown-up point — and each has a test that dies when it alone is
/// removed (`outer_fd_fallback::skip_fd_salvage_straddles_both_gates` on the predicate,
/// `outer_fd_fallback::guard_drops_the_subject_only_at_a_blown_up_rejected_point` on the
/// assembly, in the sibling test file). Without an incumbent, or with a contributions
/// vector that does not match, the guard is off.
pub(crate) fn skip_fd_salvage(
    trial: &OuterTrial<'_>, i: usize, n_obs: usize, n_obs_total: usize,
) -> bool {
    let Some((best_ofv, best_contribs)) = trial.incumbent else {
        return false;
    };
    if best_contribs.len() != trial.contribs.len() {
        return false;
    }
    let (Some(&c), Some(&b)) = (trial.contribs.get(i), best_contribs.get(i)) else {
        return false;
    };
    blown_up(trial.ofv - best_ofv, n_obs_total) && blown_up(c - b, n_obs)
}

/// The per-subject gradient a declined subject gets (#1520): **zero** when
/// [`skip_fd_salvage`] fires for it — the subject's contribution to the population
/// gradient is dropped at that point — and otherwise the salvage `salvage` computes:
/// the exact gradient at the re-solved EBE ([`declined_salvage`], #3004), else the
/// reconverged-FD gradient. Shared by the non-IOV and IOV mixed
/// assemblies so the guard has one implementation; the assemblies differ only in which
/// salvage they pass. Records the skip on `declines`.
///
/// Zero rather than the fixed-EBE gradient the issue ranked first, on measurement. At the
/// instrumented blown-up events the reconverged-FD gradient of the blown-up subject has
/// norm 60 … 180 while the fixed-EBE one has norm 1e3 … 1e4 (relative error 10 … 188): the
/// profiled objective the outer loop minimises is far flatter there than the fixed-`η̂`
/// one, because the re-solved EBEs absorb most of the blow-up. Zero is therefore the closer
/// stand-in for what the salvage would have returned. Where the guard fires — SLSQP and
/// MMA, see [`SalvageGuardPolicy`] — the optimizer discards the gradient at such a point
/// anyway, so the choice is moot there and zero is also the cheapest; under Luksan L-BFGS,
/// whose line search would have read it (measured: 51 → 58 evaluations on `warfarin_if`
/// with the term dropped, 68 with the fixed-EBE stand-in), the guard is off.
fn declined_subject_gradient(
    np: usize, population: &Population, i: usize, trial: &OuterTrial<'_>, n_obs_total: usize,
    declines: &OuterFdDeclineLog, salvage: impl FnOnce() -> (Vec<f64>, Option<Vec<f64>>),
) -> (Vec<f64>, Option<Vec<f64>>) {
    let n_obs = population.subjects[i].observations.len();
    if skip_fd_salvage(trial, i, n_obs, n_obs_total) {
        declines.record_skipped_salvage();
        declines.record_guarded(i);
        return (vec![0.0; np], None);
    }
    salvage()
}

/// Which subjects actually took a per-subject salvage outer gradient during this fit —
/// exact at the re-solved EBE ([`declined_salvage`], #3004), else reconverged FD —
/// because `subject_analytic_outer_gradient` (or its IOV twin) declined them (#1154).
///
/// One `AtomicBool` per subject, set on the fallback arm of the mixed assemblies. Recorded
/// at the point the gradient is evaluated, so it says what *ran* rather than what a probe
/// at some other parameter point predicts would run. That distinction is the whole reason
/// this is a log and not a predicate:
///
/// - **The provider's own declines are parameter-dependent.** A subject can decline at
///   `η = 0` (a plan failure, an inner Hessian that is not positive definite) and be served
///   at its EBE — so a zero-η probe reports an FD fallback for a subject that never took
///   one (PR #1418 review, finding 1).
/// - **The outer *assembly* declines away from the mode for a second reason.**
///   The adjoint's `mode_solve` needs `η̂` stationary and the exact inner Hessian `H`
///   nonsingular (an indefinite `H` only within `MODE_STATIONARY_TOL` of stationarity),
///   which holds at the EBE and routinely fails elsewhere. Measured before #2093 on the
///   bundled `examples/warfarin.ferx` + `data/warfarin.csv` against the forward provider's
///   positive-definite gate, 4 of the 10 subjects (ids 2, 4, 7, 10) failed it at `η = 0`
///   while all 10 were analytic at their EBEs (the retired
///   `sens_outer_gradient::tests::warfarin_zero_eta_non_pd_subjects_decline_and_their_ebes_do_not`).
///   For this function's purposes the consequence is the same: a zero-η probe would announce
///   fallbacks that a fit serving these subjects at their EBEs never takes. The off-mode
///   decline is pinned by `mode_solve_declines_an_indefinite_hessian_off_the_mode`.
///
/// Recording instead of probing also removes every gate this diagnostic would otherwise
/// need, because a decline can only be recorded on an evaluation that actually happened:
/// a derivative-free BOBYQA fit (including a mixture's `Auto` resolved to BOBYQA off the
/// analytic gradient by [`resolve_outer_optimizer`]),
/// a `reconverge_gradient_interval = 1` fit that bypasses the analytic branch on every
/// eval, a GN / trust-region fit, and an `outer_maxiter = 0` evaluation-only run all reach
/// the end with an empty log and say nothing.
pub(crate) struct OuterFdDeclineLog {
    declined: Vec<std::sync::atomic::AtomicBool>,
    /// The declined subjects whose decline is structural — no recorder serves them
    /// ([`is_structural_decline`]) — and who therefore take the reconverged
    /// finite-difference salvage (#1536). A subset of `declined`, so the warning can say
    /// which subjects got which gradient.
    structural: Vec<std::sync::atomic::AtomicBool>,
    /// The declined subjects whose [`is_structural_decline`] probe answered "not structural"
    /// once, so [`Self::is_structural`] does not record them again. Only the negative answer
    /// is kept: it routes the subject to the exact re-solve, which falls back to the same
    /// reconverged gradient a structural decline takes, so a kept answer can never change which
    /// gradient a subject gets; a kept *positive* one could pin a point-specific recording
    /// failure into the whole fit.
    not_structural: Vec<std::sync::atomic::AtomicBool>,
    /// The declined subjects served exactly at their EBE re-solved at the trial point
    /// ([`declined_salvage`]).
    exact: Vec<std::sync::atomic::AtomicBool>,
    /// The declined, non-structural subjects for which that failed too, so they took the
    /// reconverged finite-difference salvage.
    reconverged: Vec<std::sync::atomic::AtomicBool>,
    /// The declined subjects that contributed **zero** at some trial point because they were
    /// repelled there (non-finite or sentinel objective at the held `η̂`).
    repelled: Vec<std::sync::atomic::AtomicBool>,
    /// The declined subjects the #1520 guard dropped at some trial point. Not
    /// [`Self::repelled`]: the objective there was finite, the point merely blown up past the
    /// incumbent's.
    guarded: Vec<std::sync::atomic::AtomicBool>,
    /// How many `(subject, evaluation)` salvages [`skip_fd_salvage`] dropped (#1520). A
    /// count, not a per-subject flag: the same subject declining at two blown-up trials is
    /// two skipped salvages.
    skipped_salvages: AtomicUsize,
    /// Per declined subject, the [`DeclineCause`]s it declined for over the fit, one bit
    /// each: a subject can decline at its mode system on some trial points and in its
    /// recording on others (#2713).
    causes: Vec<std::sync::atomic::AtomicU8>,
    /// Test switch: subjects whose analytic outer gradient is treated as declined, so a
    /// test can reach a *numerical* decline on a fixture the adjoint serves.
    #[cfg(test)]
    forced: Vec<std::sync::atomic::AtomicBool>,
    /// Test hook: how many times [`Self::is_structural`] ran its probe (a recording).
    #[cfg(test)]
    pub(crate) structural_probes: AtomicUsize,
}

/// Why the analytic outer gradient declined a subject (#2713). A large fallback share has
/// two readings with different remedies, which the warning counts apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeclineCause {
    /// The mode system `H w = ∇_η F` would not solve: `H` singular at `η̂`, or indefinite at
    /// an `η̂` that is not stationary, an EBE off a strict mode.
    ModeSystem,
    /// No recorder served the subject there: its data shape is outside the recorders'
    /// scope, or the recording failed or came back non-finite at that point. Which of the
    /// two is [`is_structural_decline`]'s question, asked for the salvage.
    Recording,
}

/// What a declined subject's gradient was, worst case over the fit ([`OuterFdDeclineLog`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SalvageKind {
    /// No recorder serves it: reconverged finite differences at every point.
    Structural,
    /// Reconverged finite differences (the exact route failed).
    Reconverged,
    /// The adjoint at its re-solved EBE: exact.
    Exact,
    /// Zero at some trial point: the subject was repelled there (a non-finite or sentinel
    /// objective at the held `η̂`), so it contributed no gradient.
    Repelled,
    /// Zero at some trial point because the #1520 guard dropped it there.
    Guarded,
}

impl DeclineCause {
    /// This cause's bit in [`OuterFdDeclineLog::causes`].
    fn bit(self) -> u8 {
        match self {
            DeclineCause::ModeSystem => 1,
            DeclineCause::Recording => 2,
        }
    }
}

impl OuterFdDeclineLog {
    pub(crate) fn new(n_subjects: usize) -> Self {
        Self {
            declined: (0..n_subjects).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(),
            structural: (0..n_subjects)
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            not_structural: (0..n_subjects)
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            exact: (0..n_subjects).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(),
            reconverged: (0..n_subjects)
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            repelled: (0..n_subjects).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(),
            guarded: (0..n_subjects).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(),
            skipped_salvages: AtomicUsize::new(0),
            causes: (0..n_subjects).map(|_| std::sync::atomic::AtomicU8::new(0)).collect(),
            #[cfg(test)]
            forced: (0..n_subjects).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(),
            #[cfg(test)]
            structural_probes: AtomicUsize::new(0),
        }
    }

    /// Whether declined subject `i`'s decline is structural, asking `probe`
    /// ([`is_structural_decline`], one adjoint recording) only until it has answered "no" once:
    /// a subject that declines at every evaluation under IOV would otherwise pay a recording
    /// per evaluation for an answer that does not depend on the trial point. A structural
    /// subject is asked every time (see [`Self::not_structural`] for why).
    pub(crate) fn is_structural(&self, i: usize, probe: impl FnOnce() -> bool) -> bool {
        if self.not_structural.get(i).is_some_and(|f| f.load(Ordering::Relaxed)) {
            return false;
        }
        #[cfg(test)]
        self.structural_probes.fetch_add(1, Ordering::Relaxed);
        let structural = probe();
        if !structural {
            if let Some(f) = self.not_structural.get(i) {
                f.store(true, Ordering::Relaxed);
            }
        }
        structural
    }

    /// Treat subject `i`'s analytic outer gradient as declined (tests only).
    #[cfg(test)]
    pub(crate) fn force_decline(&self, i: usize) {
        self.forced[i].store(true, Ordering::Relaxed);
    }

    /// The cause a forced decline reports: a numerical one, at the mode system.
    #[cfg(test)]
    fn forced_cause(&self, i: usize) -> Option<DeclineCause> {
        self.forced[i].load(Ordering::Relaxed).then_some(DeclineCause::ModeSystem)
    }

    /// Record why subject `i` declined on this evaluation (#2713). Rayon workers,
    /// `Relaxed`, as [`Self::record`].
    pub(crate) fn record_cause(&self, i: usize, cause: DeclineCause) {
        if let Some(bits) = self.causes.get(i) {
            bits.fetch_or(cause.bit(), Ordering::Relaxed);
        }
    }

    /// How many subjects declined for `cause` at least once.
    fn n_cause(&self, cause: DeclineCause) -> usize {
        self.causes.iter().filter(|b| b.load(Ordering::Relaxed) & cause.bit() != 0).count()
    }

    /// Mark subject `i` as having taken the FD outer gradient on this evaluation.
    /// Called from the rayon workers, hence `Relaxed` — the log is read once, after
    /// every gradient evaluation has been joined.
    pub(crate) fn record(&self, i: usize) {
        if let Some(flag) = self.declined.get(i) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Mark declined subject `i` as a structural decline (#1536). Rayon workers,
    /// `Relaxed`, as above.
    pub(crate) fn record_structural(&self, i: usize) {
        if let Some(flag) = self.structural.get(i) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Mark declined subject `i` as served exactly at its re-solved EBE
    /// ([`declined_salvage`]). Rayon workers, `Relaxed`, as above.
    pub(crate) fn record_exact(&self, i: usize) {
        if let Some(flag) = self.exact.get(i) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Mark declined subject `i` as having taken the reconverged finite-difference salvage
    /// without being structural (the mixture's per-term salvage, an exact route that failed).
    pub(crate) fn record_reconverged(&self, i: usize) {
        if let Some(flag) = self.reconverged.get(i) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Mark declined subject `i` as contributing zero at a trial point because it was repelled
    /// there. Rayon workers, `Relaxed`, as above.
    pub(crate) fn record_repelled(&self, i: usize) {
        if let Some(flag) = self.repelled.get(i) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Mark declined subject `i` as dropped by the #1520 guard at a trial point.
    pub(crate) fn record_guarded(&self, i: usize) {
        if let Some(flag) = self.guarded.get(i) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// The worst salvage subject `i` took over the fit.
    fn salvage_kind(&self, i: usize) -> SalvageKind {
        let on = |v: &[std::sync::atomic::AtomicBool]| {
            v.get(i).is_some_and(|f| f.load(Ordering::Relaxed))
        };
        if on(&self.structural) {
            SalvageKind::Structural
        } else if on(&self.reconverged) {
            SalvageKind::Reconverged
        } else if on(&self.repelled) {
            SalvageKind::Repelled
        } else if on(&self.guarded) {
            SalvageKind::Guarded
        } else if on(&self.exact) {
            SalvageKind::Exact
        } else {
            // Every route that declares a decline records its kind; a decline with none
            // recorded took no gradient, which is the repelled subject's outcome.
            SalvageKind::Repelled
        }
    }

    #[cfg(test)]
    fn n_structural_for_test(&self) -> usize {
        self.structural.iter().filter(|f| f.load(Ordering::Relaxed)).count()
    }

    /// Whether every declined subject was served exactly: no finite-difference salvage or
    /// zero fill ran, so a gradient assembled over them is the exact one.
    pub(crate) fn all_declines_exact(&self) -> bool {
        self.declined_indices().into_iter().all(|i| self.salvage_kind(i) == SalvageKind::Exact)
    }

    /// Count one salvage the guard skipped (#1520). Rayon workers, `Relaxed`, as above.
    fn record_skipped_salvage(&self) {
        self.skipped_salvages.fetch_add(1, Ordering::Relaxed);
    }

    /// How many salvages the guard skipped over the fit so far.
    pub(crate) fn skipped_salvages(&self) -> usize {
        self.skipped_salvages.load(Ordering::Relaxed)
    }

    pub(crate) fn declined_indices(&self) -> Vec<usize> {
        self.declined
            .iter()
            .enumerate()
            .filter(|(_, f)| f.load(Ordering::Relaxed))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Warning naming the subjects whose analytic outer gradient declined at the EBE the optimizer
/// held and did not end up exact, from the runtime [`OuterFdDeclineLog`]. `None` when every
/// subject was served exactly, at the held EBE or re-solved ([`declined_salvage`]) — there
/// is no fallback to report then — and, by construction, when no analytic outer gradient was
/// ever evaluated (see the log's docs).
///
/// The outer twin of `inner_optimizer::fd_fallback_warning` (#1154). Since #466 the
/// salvage is per subject, and was indistinguishable from a fit that is simply slow: no
/// `W_*` code, no count, and a `gradient_method_outer` that keeps reporting
/// `analytic` because it reads a **model**-level predicate. Any non-empty log is
/// therefore already a mismatch with that label: the analytic branch had to have been
/// selected for a decline to be recordable at all.
///
/// What a declined subject got is read from the log, one [`SalvageKind`] per subject (the
/// worst over the fit), so the consequence sentence names only what ran: reconverged
/// finite differences for a subject no recorder serves ([`is_structural_decline`]) or whose
/// exact route failed too (correct, slower); the exact gradient at the re-solved EBE
/// (reported, but not a fallback, so alone it raises no warning); zero at some trial points
/// where the subject was repelled (non-finite or sentinel objective) or where the #1520 guard
/// dropped it, one sentence per producer, because the reason the optimizer rejects the point
/// differs. The route is not passed in, so no caller can pair a log with the wrong sentence.
///
/// "Could not be given" rather than "fell outside the provider's scope": the log records
/// every failure of `subject_analytic_outer_gradient`, which includes a
/// non-positive-definite inner Hessian at a trial point and a non-finite analytic
/// component — not only a data shape the provider declines (#1529 review).
pub(crate) fn outer_fd_fallback_warning(
    population: &Population, log: &OuterFdDeclineLog,
) -> Option<String> {
    let declined = log.declined_indices();
    if declined.is_empty() || log.all_declines_exact() {
        return None;
    }
    let n_fd = declined.len();
    let n_total = population.subjects.len();
    let example = declined
        .first()
        .and_then(|&i| population.subjects.get(i))
        .map(|s| format!(" (e.g. subject {})", s.id))
        .unwrap_or_default();
    const REPELLED: &str = "contributed no gradient at some of the trial points where they \
         declined: their objective there was non-finite or at the failed-subject sentinel, so \
         the population objective carries it and the optimizer rejects those points.";
    const GUARDED: &str = "contributed no gradient at some of the trial points where they \
         declined: the incumbent guard dropped them where the objective had blown up past the \
         incumbent's, points the optimizer rejects.";
    const STRUCTURAL: &str = "used reconverged finite-difference outer gradients because \
         no analytic recorder serves them: recording their exact derivatives failed, as it \
         will at every parameter point. Those gradients include the EBE-response term but \
         are slower.";
    const RECONVERGED: &str = "used reconverged finite-difference outer gradients; their \
         results are correct but slower.";
    const EXACT: &str = "were re-solved to their EBEs at the trial point and given the exact \
         analytic gradient there.";
    // Each group that is non-empty is named with its count, never the held-EBE remedy for a
    // subject that was reconverged, nor "exact" for one that was not.
    let count = |kind| declined.iter().filter(|&&i| log.salvage_kind(i) == kind).count();
    let groups: Vec<(usize, &str)> = [
        (count(SalvageKind::Structural), STRUCTURAL),
        (count(SalvageKind::Reconverged), RECONVERGED),
        (count(SalvageKind::Exact), EXACT),
        (count(SalvageKind::Repelled), REPELLED),
        (count(SalvageKind::Guarded), GUARDED),
    ]
    .into_iter()
    .filter(|&(n, _)| n > 0)
    .collect();
    let consequence = match groups.as_slice() {
        [(_, one)] => one.to_string(),
        many => {
            let mut parts = many.iter().enumerate().map(|(k, (n, text))| {
                let lead = if k == 0 { "of these, " } else { " Another " };
                format!("{lead}{n} {text}")
            });
            parts.by_ref().collect::<String>()
        }
    };
    // #1520: say when some of those salvages were skipped. One sentence, appended only
    // when the count is non-zero, so a fit the guard never touched reads as before.
    let skipped = match log.skipped_salvages() {
        0 => String::new(),
        k => format!(
            " {k} of their salvage gradients were skipped at trial points worse than the \
             incumbent where the subject's objective had blown up; the subject contributed \
             nothing to the outer gradient there, at points the optimizer rejects anyway."
        ),
    };
    Some(format!(
        "{n_fd} of {n_total} subjects could not be given the exact analytic outer \
         gradient at the EBE the optimizer held during this fit{example} ({why}) and \
         {consequence} The reported outer gradient method is the model-level route, not \
         the per-subject one.{skipped}",
        why = decline_causes_clause(log),
    ))
}

/// The parenthesis of [`outer_fd_fallback_warning`] that says why the subjects declined
/// (#2713): one clause per [`DeclineCause`] some subject declined for, with its count. The
/// counts cover every trial point of the fit, rejected ones included, and a subject that
/// declined for both causes counts under each. A log no route stamped a cause on (the
/// quadrature's) keeps the generic sentence.
fn decline_causes_clause(log: &OuterFdDeclineLog) -> String {
    let parts: Vec<String> = [
        (
            DeclineCause::ModeSystem,
            "with an inner Hessian at the EBE that was singular, or indefinite at an EBE \
             that was not stationary (an EBE off a strict mode), at some trial point",
        ),
        (
            DeclineCause::Recording,
            "with a shape no analytic recorder serves, or a recording that failed or came \
             back non-finite at some trial point",
        ),
    ]
    .into_iter()
    .filter_map(|(cause, text)| match log.n_cause(cause) {
        0 => None,
        n => Some(format!("{n} {text}")),
    })
    .collect();
    if parts.is_empty() {
        "a shape no analytic recorder serves, or a trial point where it could not be formed"
            .to_string()
    } else {
        parts.join("; ")
    }
}

/// `population_gradient_sens_mixed_dx` without the mode responses.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn population_gradient_sens_mixed(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    ehs: &[DVector<f64>], hms: &[DMatrix<f64>], bounds: &PackedBounds, options: &FitOptions,
    trial: OuterTrial<'_>, declines: &OuterFdDeclineLog,
) -> Vec<f64> {
    population_gradient_sens_mixed_dx(
        x,
        init_params,
        model,
        population,
        ehs,
        hms,
        bounds,
        options,
        trial,
        declines,
        false,
    )
    .0
}

/// Non-IOV population gradient assembled **per subject**: the exact analytic
/// (Almquist) gradient — including the EBE response on every θ/Ω/σ block — for
/// every subject inside the provider's scope, and [`declined_salvage`] for each subject
/// outside it (or whose analytic gradient came back non-finite): the adjoint at the EBE
/// re-solved at the trial point, else the reconverged finite-difference score, never the
/// held-EBE gradient, whose missing `log|H̃|` EBE response biased every step it entered
/// (#1529, #3004). This replaced the all-or-nothing `population_gradient_sens` (since
/// retired): previously a single out-of-scope subject forced the whole population onto the
/// θ-only fixed-EBE gradient, whose biased Ω/σ block left the variance components pinned at
/// their start and stalled SLSQP/L-BFGS/MMA above the derivative-free optimum
/// (focei-slsqp-fixed-ebe-gradient-bias).
///
/// Decline rates are parameter-dependent (the assembly needs a positive-definite inner
/// Hessian), and at a blown-up line-search trial most of a population can decline at once: on
/// cyclophosphamide 46 of 55 subjects did (#1529). The #1520 guard ([`skip_fd_salvage`])
/// drops the salvage at such a trial, and the exact route costs one inner solve and one
/// adjoint pass where the reconverged difference costs `2·n_free` solves. A **structural**
/// decline ([`is_structural_decline`]) takes the reconverged gradient directly (#1536).
/// Returns the packed `2·Σᵢ dᵢ` with fixed coordinates zeroed.
///
/// `trial` serves the #1520 guard only: a declined subject at a blown-up trial point
/// ([`skip_fd_salvage`]) contributes nothing instead of taking the salvage.
///
/// With `want_eta_dx` it also returns each subject's mode response `dη̂ᵢ/dx` from its
/// gradient's own recording
/// (`adjoint::subject_adjoint_gradient_with_eta_dx_at`): the EBE warm-start predictor's
/// Jacobians, zero for a subject the response does not reach. The third element holds, per
/// subject, the `η̂` a polished pass took its gradient at ([`adopt_polished_modes`]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn population_gradient_sens_mixed_dx(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    ehs: &[DVector<f64>], hms: &[DMatrix<f64>], bounds: &PackedBounds, options: &FitOptions,
    trial: OuterTrial<'_>, declines: &OuterFdDeclineLog, want_eta_dx: bool,
) -> (Vec<f64>, Option<Vec<Vec<DVector<f64>>>>, Vec<Option<Vec<f64>>>) {
    let np = x.len();
    let n_obs_total = population_n_obs(population);
    // One unpack for the population: every subject is at the same packed point.
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(init_params, x);
    #[allow(clippy::type_complexity)]
    let filled: Vec<(Vec<f64>, Option<Vec<DVector<f64>>>, Option<Vec<f64>>)> =
        crate::par::map(population.subjects.len(), |i| {
            let subject = &population.subjects[i];
            // Complete the fallback on this worker as soon as its analytic
            // result is known; do not wait for a second population-wide pass.
            let (analytic, dx, polished) = match pt.as_ref() {
                Some(pt) => subject_analytic_outer_gradient_dx(
                    model,
                    subject,
                    pt,
                    ehs[i].as_slice(),
                    options.interaction,
                    want_eta_dx,
                ),
                None => (packed_length_decline(subject, x.len()), None, None),
            };
            #[cfg(test)]
            let analytic = declines.forced_cause(i).map_or(analytic, Err);
            let mut polished_by_salvage = None;
            let g = match analytic {
                // Keep the exact analytic gradient for in-scope, finite subjects.
                Ok(g) => g,
                // Out-of-scope (or non-finite analytic) → per-subject salvage (exact at the
                // re-solved EBE, or reconverged FD for a structural decline, #1536), unless
                // the trial point has blown up (#1520).
                Err(cause) => {
                    declines.record(i);
                    declines.record_cause(i, cause);
                    let structural = declines.is_structural(i, || {
                        is_structural_decline(model, subject, init_params, ehs[i].as_slice())
                    });
                    if structural {
                        declines.record_structural(i);
                    }
                    let (g, mode) = declined_subject_gradient(
                        np,
                        population,
                        i,
                        &trial,
                        n_obs_total,
                        declines,
                        || {
                            non_iov_declined_salvage(
                                structural,
                                x,
                                init_params,
                                model,
                                population,
                                i,
                                &ehs[i],
                                &hms[i],
                                bounds,
                                options,
                                declines,
                            )
                        },
                    );
                    polished_by_salvage = mode;
                    g
                }
            };
            (g, dx, polished_by_salvage.or(polished))
        });
    let mut grad = vec![0.0f64; np];
    for (gi, _, _) in &filled {
        for k in 0..np {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(init_params);
    for k in 0..np {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    // A subject the response does not reach (declined, no data, a singular inner Hessian)
    // is predicted not to move.
    let (dxs, polished): (Vec<_>, Vec<_>) = filled.into_iter().map(|(_, dx, p)| (dx, p)).unzip();
    let eta_dx = want_eta_dx.then(|| {
        dxs.into_iter()
            .zip(ehs)
            .map(|(dx, e)| {
                dx.filter(|c| c.len() == np && c.iter().all(|v| v.len() == e.len()))
                    .unwrap_or_else(|| vec![DVector::zeros(e.len()); np])
            })
            .collect()
    });
    (grad, eta_dx, polished)
}

/// One IOV subject's exact analytic packed outer gradient at the stacked mode
/// `stacked = [η̂, κ̂₁ … κ̂_K]`, or `None` when the adjoint declines it. `pt` is `None` when
/// `x` is not the template's packed length: the adjoint declines then, but a subject with no
/// data at all (no observations, no endpoint records) keeps the zero gradient it gives one.
///
/// The analytic half of an IOV subject's outer gradient (the reconverged-FD salvage
/// [`subject_reconverged_fd_gradient`] is the other), kept separate so
/// [`population_gradient_sens_iov_polished`] can put its #1520 guard between the two halves.
/// The second element is the stacked `η̂` a polished pass took its gradient at.
fn subject_iov_analytic_gradient(
    pt: Option<&crate::sens::ode_provider::adjoint::PackedPoint<'_>>, model: &CompiledModel,
    subject: &Subject, stacked: &DVector<f64>, np: usize, options: &FitOptions,
) -> (Result<Vec<f64>, DeclineCause>, Option<Vec<f64>>) {
    match pt {
        Some(pt) => {
            let (g, _, polished) = subject_analytic_outer_gradient_dx(
                model,
                subject,
                pt,
                stacked.as_slice(),
                options.interaction,
                false,
            );
            (g, polished)
        }
        None => (packed_length_decline(subject, np), None),
    }
}

/// **IOV** population gradient assembled **per subject** — the IOV analogue of
/// [`population_gradient_sens_mixed_dx`]: the exact analytic stacked-η / block-Ω gradient for
/// every in-scope subject, and a per-subject reconverged-FD gradient
/// ([`subject_reconverged_fd_gradient`]) for any out-of-scope (or non-finite) one.
/// Replaces the former all-or-nothing IOV outer gradient, which dropped the *whole*
/// population to FD on the first out-of-scope subject — so a single infusion / steady-state
/// / wide-axis subject no longer forces the entire fit onto FD (#466 review round 2).
/// Returns the packed `2·Σᵢ dᵢ` with fixed coordinates zeroed.
///
/// `trial` serves the #1520 guard, exactly as on the non-IOV twin.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn population_gradient_sens_iov_mixed(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds,
    options: &FitOptions, trial: OuterTrial<'_>, declines: &OuterFdDeclineLog,
) -> Vec<f64> {
    population_gradient_sens_iov_polished(
        x,
        init_params,
        model,
        population,
        ehs,
        kappas,
        bounds,
        options,
        trial,
        declines,
    )
    .0
}

/// [`population_gradient_sens_iov_mixed`] and, per subject, the stacked `[η̂, κ̂]` a polished
/// pass took its gradient at ([`adopt_polished_modes`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn population_gradient_sens_iov_polished(
    x: &[f64], init_params: &ModelParameters, model: &CompiledModel, population: &Population,
    ehs: &[DVector<f64>], kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds,
    options: &FitOptions, trial: OuterTrial<'_>, declines: &OuterFdDeclineLog,
) -> (Vec<f64>, Vec<Option<Vec<f64>>>) {
    let np = x.len();
    let n_obs_total = population_n_obs(population);
    // One unpack for the population: every subject is at the same packed point.
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(init_params, x);
    let filled: Vec<(Vec<f64>, Option<Vec<f64>>)> =
        crate::par::map(population.subjects.len(), |i| {
            let subject = &population.subjects[i];
            let stacked = joint_warm_start(&ehs[i], &kappas[i]);
            let (analytic, polished) =
                subject_iov_analytic_gradient(pt.as_ref(), model, subject, &stacked, np, options);
            let mut polished_by_salvage = None;
            let g = match analytic {
                Ok(g) => g,
                Err(cause) => {
                    declines.record(i);
                    declines.record_cause(i, cause);
                    let structural = declines.is_structural(i, || {
                        is_structural_decline(model, subject, init_params, stacked.as_slice())
                    });
                    if structural {
                        declines.record_structural(i);
                    }
                    let (g, mode) = declined_subject_gradient(
                        np,
                        population,
                        i,
                        &trial,
                        n_obs_total,
                        declines,
                        || {
                            declined_salvage(
                                structural,
                                i,
                                x,
                                init_params,
                                model,
                                subject,
                                &stacked,
                                bounds,
                                options,
                                declines,
                            )
                        },
                    );
                    polished_by_salvage = mode;
                    g
                }
            };
            (g, polished_by_salvage.or(polished))
        });
    let mut grad = vec![0.0f64; np];
    for (gi, _) in &filled {
        for k in 0..np {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(init_params);
    for k in 0..np {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    (grad, filled.into_iter().map(|(_, p)| p).collect())
}

/// Compute `d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x)` by summing per-subject
/// gradients in parallel.  ETAs are fixed at their current EBE values.
///
/// `kappas` must have length `n_subj`; each `kappas[i]` is the IOV kappa
/// vector for subject `i` (empty for non-IOV models).
fn ad_population_gradient(
    x: &[f64], n_subj: usize, init_params: &ModelParameters, model: &CompiledModel,
    population: &Population, ehs: &[DVector<f64>], hms: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds, options: &FitOptions,
) -> Vec<f64> {
    debug_assert_eq!(ehs.len(), n_subj);
    debug_assert_eq!(hms.len(), n_subj);
    debug_assert_eq!(kappas.len(), n_subj);
    let np = x.len();
    // The fixed-EBE score, which omits the `log|H̃|` EBE-response term; IOV routes through the
    // reconverged-FD gradient, not here.
    let per_subj: Vec<Vec<f64>> = crate::par::map(n_subj, |i| {
        subject_fixed_ebe_gradient(
            x,
            init_params,
            model,
            population,
            i,
            &ehs[i],
            &hms[i],
            kappas[i].as_slice(),
            bounds,
            options,
        )
        .1
    });
    assemble_population_gradient(&per_subj, np)
}

/// Total observation rows across the population — the denominator of the population
/// gate in [`skip_fd_salvage`].
fn population_n_obs(population: &Population) -> usize {
    population.subjects.iter().map(|s| s.observations.len()).sum()
}

/// Assemble the covariance-step population gradient `2·Σᵢ gᵢ` from per-subject
/// gradients, summing over subjects in index order. Both the parallel
/// [`ad_population_gradient`] and the serial per-point gradient inside
/// [`compute_covariance`] route their reduction through here, so there is a
/// single summation order — which is what keeps the flattened (#256) covariance
/// bit-identical to the pre-flatten serial stencil for FOCE. `np` is the packed
/// parameter count; each `gᵢ` has length `np`.
fn assemble_population_gradient(per_subj: &[Vec<f64>], np: usize) -> Vec<f64> {
    (0..np).map(|k| per_subj.iter().map(|gi| gi[k]).sum::<f64>() * 2.0).collect()
}

/// Whether gradient evaluation number `grad_idx` (0-based, per optimization
/// run) should use the expensive reconverged path on a **non-IOV** model.
///
/// Driven by `reconverge_gradient_interval`: `0` disables it entirely; `N`
/// fires on evals `0, N, 2N, …`. The `interval != 0` guard also short-circuits
/// the modulo, so a `0` interval can never divide by zero. IOV models
/// reconverge unconditionally and never consult this.
pub(super) fn reconverge_this_eval(options: &FitOptions, grad_idx: usize) -> bool {
    let interval = options.reconverge_gradient_interval;
    interval != 0 && grad_idx % interval == 0
}

/// `FERX_SENS_CHECK=1` enables the per-eval analytic-vs-reconverged-FD outer
/// gradient cross-check in [`population_gradient`] (off by default — it doubles
/// the gradient cost, so it is a CI/diagnostic backstop, not a production path).
fn sens_check_enabled() -> bool {
    std::env::var("FERX_SENS_CHECK").map(|v| v == "1").unwrap_or(false)
}

/// Population gradient dispatcher. IOV models (`n_kappa > 0`) and TTE / discrete
/// models with η (#1778) use the EBE-reconverging FD gradient — their weakly-identified
/// variance components / η-dependent data-term curvature need it — and everything else
/// uses the cheap analytical fixed-EBE gradient unless the `reconverge_gradient_interval`
/// schedule opts this evaluation into the reconverged path.
///
/// `grad_eval_idx` is the caller's count of gradient evaluations so far; this function reads
/// it to apply the schedule and then advances it. Owning the counter here keeps every
/// optimizer path (NLopt objective, NLopt fallback, BFGS) on one definition of "gradient
/// evaluation", so they cannot drift apart in how they count or pick the gradient.
#[allow(clippy::too_many_arguments)]
pub(super) fn population_gradient(
    x: &[f64], n_subj: usize, init_params: &ModelParameters, model: &CompiledModel,
    population: &Population, ehs: &[DVector<f64>], hms: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds, options: &FitOptions,
    grad_eval_idx: &mut usize, trial: OuterTrial<'_>, declines: &OuterFdDeclineLog,
) -> Vec<f64> {
    population_gradient_with_agq_evaluation(
        x,
        n_subj,
        init_params,
        model,
        population,
        ehs,
        hms,
        kappas,
        bounds,
        options,
        grad_eval_idx,
        None,
        trial,
        declines,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn population_gradient_with_agq_evaluation(
    x: &[f64], n_subj: usize, init_params: &ModelParameters, model: &CompiledModel,
    population: &Population, ehs: &[DVector<f64>], hms: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>], bounds: &PackedBounds, options: &FitOptions,
    grad_eval_idx: &mut usize,
    agq_evaluation: Option<crate::estimation::agq::PopulationEvaluation>, trial: OuterTrial<'_>,
    declines: &OuterFdDeclineLog,
    eta_dx: Option<&mut crate::estimation::sens_outer_gradient::ModeResponses>,
) -> Vec<f64> {
    let reconverge = reconverge_this_eval(options, *grad_eval_idx);
    *grad_eval_idx += 1;
    // AGQ minimises a *different* objective (the quadrature marginal, not the FOCE/Laplace
    // one), so every analytic and fixed-EBE gradient below — all of them closed forms of
    // the FOCE marginal — is simply the gradient of the wrong function. Feeding one to the
    // outer optimizer would not fail loudly; it would converge, smoothly, to the FOCE
    // optimum while reporting AGQ OFVs. AGQ has its own gradient.
    if options.agq_nodes().is_some() {
        // Preferred: AGQ's own exact gradient — the analytic posterior-weighted score over
        // the nodes (Fisher identity) plus the grid-response term — which needs no inner
        // re-solve, against the FD path's `2·n_free` *full population objective*
        // re-evaluations. Exact at every `n_agq`. See `estimation::agq`.
        // `reconverge_gradient_interval` is honoured here too: it is the documented escape
        // hatch onto the numeric path, so it must override the analytic gradient for AGQ
        // exactly as it does for FOCE/FOCEI below.
        // `population_gradient_mixed` supplies the gradient of the quadrature objective for
        // **either** anchor: the fixed-node score is anchor-independent, and the grid-response
        // term differences whichever Hessian scales the grid (exact for `laplace`,
        // Gauss-Newton for `focei` — `anchor` selects it, matching the objective). If an
        // analytic subject score fails, only that subject is reconverged numerically.
        if let Some(g) = crate::estimation::agq::population_gradient_mixed(
            model,
            population,
            init_params,
            x,
            ehs,
            kappas,
            bounds,
            options,
            reconverge,
            agq_evaluation,
            Some(declines),
            eta_dx,
        ) {
            return g;
        }
        // A subject's numerical score also failed. Only the optimizer may use its
        // guarded population objective here; covariance rejects an unavailable score
        // rather than differentiating this penalty and squaring it into S.
        return reconverged_fd_gradient(
            x,
            init_params,
            model,
            population,
            ehs,
            kappas,
            bounds,
            options,
        );
    }
    // M3-censored models now have an exact analytic censored gradient on both the
    // FOCEI (`subject_packed_gradient` + `prepare`'s M3 branch) and the FOCE
    // (`subject_packed_gradient_foce`, censored rows excluded from R̃ and added as
    // `−logΦ`) paths, so M3 takes the analytic path like any other fit.
    let force_reconverge = reconverge;
    // Analytic gradient (Almquist 2015 Eq. 23, via the adjoint
    // `subject_adjoint_packed_gradient`): the exact marginal FOCEI gradient including the Eq. 46
    // EBE response on every θ/Ω/σ block — no fixed-EBE bias, no FD noise, so it
    // supersedes both branches below where it applies. Gated on
    // `analytic_outer_gradient_available`; a subject the adjoint declines takes a
    // per-subject salvage inside the mixed assembly (#466, #1529). FOCEI differentiates the
    // Almquist Laplace marginal, plain FOCE the Sheiner–Beal one (R̃ = JΩJᵀ + R⁰), on the same
    // EBE/inner-Hessian core; IOV models route to the stacked-η / block-Ω assembly.
    //
    // `reconverge` (from `reconverge_gradient_interval`) overrides the analytic path: it is
    // the documented escape hatch (#381 review findings #6/#7). `reconverge_gradient_interval
    // = 1` forces the reconverged-FD gradient on every eval even for analytical models.
    let iov_analytic = model.n_kappa > 0;
    // `gradient = fd` forces the numeric path for the outer gradient too (the inner
    // EBE gradient honours it via `analytic_inner_grad_supported`), so the option
    // fully disables the analytic sensitivities rather than only the inner half.
    // `analytic_outer_gradient_available` is the shared predicate that
    // `Optimizer::resolve_auto` and `build_info::gradient_method_outer` also use,
    // so the `auto` optimizer cannot pick a gradient-based optimizer while this
    // gate falls through to FD (#490 review).
    if !force_reconverge && crate::sens::provider::analytic_outer_gradient_available(model) {
        let g = if iov_analytic {
            // Per-subject: exact analytic for in-scope subjects, per-subject reconverged-FD
            // for out-of-scope ones — always `Some`, mirroring the non-IOV mixed path. A
            // single out-of-scope subject no longer drops the whole population to FD (and
            // so the reported `gradient_method` stays accurate) (#466 review round 2).
            let (g, polished) = population_gradient_sens_iov_polished(
                x,
                init_params,
                model,
                population,
                ehs,
                kappas,
                bounds,
                options,
                trial,
                declines,
            );
            if let Some(slot) = eta_dx {
                slot.polished = polished;
            }
            Some(g)
        } else {
            // Non-IOV: assemble per subject — exact analytic for in-scope
            // subjects, the exact gradient at the re-solved EBE (else reconverged FD) for
            // out-of-scope ones (#1529, #3004). Always `Some`; the finiteness backstop
            // below still guards it. This is the fix for focei-slsqp-fixed-ebe-gradient-bias:
            // one out-of-scope subject no longer drops the whole population to the biased
            // θ-only fixed-EBE fallback (`ad_population_gradient`).
            let (g, dx, polished) = population_gradient_sens_mixed_dx(
                x,
                init_params,
                model,
                population,
                ehs,
                hms,
                bounds,
                options,
                trial,
                declines,
                eta_dx.as_ref().is_some_and(|s| s.want_dx),
            );
            if let Some(slot) = eta_dx {
                slot.polished = polished;
                if let Some(dx) = dx {
                    slot.eta_dx = Some(dx);
                }
            }
            Some(g)
        };
        if let Some(g) = g {
            // Always-on finiteness backstop: a non-finite analytic component (the
            // class PR #381 review finding #3 warns about — a degenerate acos /
            // singular eigenvalue producing NaN) would poison the optimizer. Rather
            // than return it, fall through to the numeric path. Cheap (a scan of a
            // length-`np` vector) and reliable, unlike a mid-run magnitude compare
            // to reconverged-FD: with loosely-converged EBEs the analytic and
            // reconverged-FD gradients legitimately differ away from the optimum
            // (they agree to ~1e-11 only at convergence — see the unit tests), so a
            // value-tolerance assert here cries wolf. With FERX_SENS_CHECK=1 the
            // divergence is additionally reported for diagnosis.
            if g.iter().all(|v| v.is_finite()) {
                if sens_check_enabled() {
                    let fd = reconverged_fd_gradient(
                        x,
                        init_params,
                        model,
                        population,
                        ehs,
                        kappas,
                        bounds,
                        options,
                    );
                    let max_abs = g.iter().chain(fd.iter()).fold(1e-8_f64, |m, v| m.max(v.abs()));
                    let max_diff =
                        g.iter().zip(fd.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                    eprintln!(
                        "[FERX_SENS_CHECK] analytic vs reconverged-FD outer gradient: \
                         max abs diff {max_diff:.3e}, rel {:.2e} (interaction={})",
                        max_diff / max_abs,
                        options.interaction
                    );
                }
                return g;
            } else if options.verbose {
                eprintln!(
                    "warning: non-finite analytic outer gradient — falling back to the numeric path"
                );
            }
        }
    }
    // IOV models always reconverge the inner EBE solution inside the gradient.
    // For non-IOV models the default is the fixed-EBE analytical/AD gradient,
    // which is far cheaper but omits the response of (η̂, H) to the population
    // parameters — an omission that stalls SLSQP well above the derivative-free
    // optimum on ill-conditioned fits. The `reconverge_gradient_interval`
    // schedule (via `reconverge_this_eval`) opts a non-IOV fit into the
    // reconverged path (see focei-slsqp-fixed-ebe-gradient-bias). A TTE / discrete
    // model with η reconverges on every eval: its fixed-EBE score is biased, not just
    // noisy, and stalls every gradient optimizer short of the optimum (#1778).
    if model.n_kappa > 0
        || force_reconverge
        || crate::estimation::gauss_newton::fixed_ebe_score_drops_ebe_response(model)
    {
        reconverged_fd_gradient(x, init_params, model, population, ehs, kappas, bounds, options)
    } else {
        ad_population_gradient(
            x,
            n_subj,
            init_params,
            model,
            population,
            ehs,
            hms,
            kappas,
            bounds,
            options,
        )
    }
}

fn bfgs_update(
    h_inv: &mut DMatrix<f64>, x_new: &[f64], x_old: &[f64], g_new: &[f64], g_old: &[f64], n: usize,
) {
    let s: Vec<f64> = (0..n).map(|i| x_new[i] - x_old[i]).collect();
    let y: Vec<f64> = (0..n).map(|i| g_new[i] - g_old[i]).collect();
    let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
    if sy > 1e-12 {
        let rho = 1.0 / sy;
        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let eye = DMatrix::<f64>::identity(n, n);
        let rs_yt = rho * &s_vec * y_vec.transpose();
        let ry_st = rho * &y_vec * s_vec.transpose();
        let rss = rho * &s_vec * s_vec.transpose();
        *h_inv = (&eye - &rs_yt) * &*h_inv * (&eye - &ry_st) + rss;
    } else {
        *h_inv = DMatrix::identity(n, n);
    }
}

/// Limited-memory L-BFGS search direction `d = −H∇f` via the two-loop recursion
/// (Nocedal & Wright Alg. 7.4), using the most recent `(s, y)` history pairs
/// (newest last). The implicit inverse-Hessian seed is `γ·I` with
/// `γ = (sₖ·yₖ)/(yₖ·yₖ)` from the newest pair (Barzilai–Borwein scaling) — the
/// standard choice that keeps the step well-scaled without ever forming the
/// dense `n×n` matrix `bfgs_update` maintains. With no history it returns plain
/// steepest descent `−∇f`. Curvature filtering (`s·y > 0`) is enforced by the
/// caller before a pair is pushed, so every stored `ρᵢ = 1/(yᵢ·sᵢ)` is finite.
fn lbfgs_two_loop(g: &[f64], s_hist: &[DVector<f64>], y_hist: &[DVector<f64>]) -> Vec<f64> {
    let m = s_hist.len();
    debug_assert_eq!(m, y_hist.len());
    let mut q = DVector::from_column_slice(g);
    if m == 0 {
        return (-q).iter().copied().collect();
    }
    let rho: Vec<f64> = (0..m).map(|i| 1.0 / y_hist[i].dot(&s_hist[i])).collect();
    let mut alpha = vec![0.0f64; m];
    // First loop: newest → oldest.
    for i in (0..m).rev() {
        alpha[i] = rho[i] * s_hist[i].dot(&q);
        q -= alpha[i] * &y_hist[i];
    }
    // Seed with γ·I from the newest pair.
    let last = m - 1;
    let gamma = s_hist[last].dot(&y_hist[last]) / y_hist[last].dot(&y_hist[last]);
    let mut r = gamma * q;
    // Second loop: oldest → newest.
    for i in 0..m {
        let beta = rho[i] * y_hist[i].dot(&r);
        r += (alpha[i] - beta) * &s_hist[i];
    }
    (-r).iter().copied().collect()
}

fn backtracking_line_search_warm(
    x: &[f64], d: &[f64], g: &[f64], f0: f64, bounds: &PackedBounds, prev_etas: &[DVector<f64>],
    f_only: &dyn Fn(&[f64], &[DVector<f64>]) -> f64,
) -> f64 {
    let c1 = 1e-4;
    let n = x.len();
    let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
    if dg >= 0.0 {
        return 0.0;
    }

    let mut alpha = 1.0;
    let mut x_new = vec![0.0; n];
    for _ in 0..30 {
        for i in 0..n {
            x_new[i] = (x[i] + alpha * d[i]).clamp(bounds.lower[i], bounds.upper[i]);
        }
        let f_new = f_only(&x_new, prev_etas);
        if f_new <= f0 + c1 * alpha * dg {
            return alpha;
        }
        alpha *= 0.5;
        if alpha < 1e-18 {
            return 0.0;
        }
    }
    0.0
}

/// Analytic covariance-step gradient with ETAs/H fixed: `2·pop_nll` with no
/// omega-prior add-back (both the SB and Laplace marginals already carry Ω —
/// #243/#249). The production stencil inlines a serial variant (plus the #274 Δ
/// correction); this thin wrapper over [`ad_population_gradient`] is retained for
/// the gradient-consistency tests that finite-difference the fixed-EBE objective.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn covariance_gradient(
    x: &[f64], template: &ModelParameters, model: &CompiledModel, population: &Population,
    eta_hats: &[DVector<f64>], h_matrices: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds, options: &FitOptions,
) -> Vec<f64> {
    let n_subj = population.subjects.len();
    ad_population_gradient(
        x, n_subj, template, model, population, eta_hats, h_matrices, kappas, bounds, options,
    )
}

#[cfg(test)]
#[path = "outer_optimizer_tests.rs"]
pub(crate) mod tests;
