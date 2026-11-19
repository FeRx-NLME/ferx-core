//! Integration tests for the BOBYQA and trust_region outer optimizers.
//!
//! These fit the warfarin example and assert sane convergence behaviour.
//! They don't reproduce reference estimates bit-for-bit — BOBYQA is
//! derivative-free and trust-region uses a CG subproblem, so both take
//! different paths than SLSQP and can settle on slightly different
//! points. The asserts are loose enough to catch "regression to garbage"
//! while tolerating normal optimizer jitter.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

fn data_and_model() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

fn base_options() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false; // skip FD Hessian — not what we're testing
    opts.outer_maxiter = 100;
    opts
}

#[test]
fn slsqp_reaches_a_sane_ofv() {
    // Baseline: SLSQP (default) should converge on warfarin to a moderate OFV.
    // Used as a reference point for the BOBYQA / trust-region assertions.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Slsqp;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("slsqp fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "SLSQP OFV should be finite, got {}",
        result.ofv
    );
}

#[test]
fn bobyqa_fit_converges_to_finite_ofv() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Bobyqa;
    // BOBYQA is derivative-free: one "iteration" is cheap but it needs many
    // to triangulate a quadratic. Give it a bit more headroom.
    opts.outer_maxiter = 200;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("bobyqa fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "BOBYQA OFV must be finite, got {}",
        result.ofv
    );
    // Theta should stay inside the declared bounds from the model file.
    for (i, &th) in result.theta.iter().enumerate() {
        let lo = model.default_params.theta_lower[i];
        let hi = model.default_params.theta_upper[i];
        assert!(th > lo && th < hi, "theta[{}] = {} escaped bounds", i, th);
    }
}

#[test]
fn trust_region_fit_converges_to_finite_ofv() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.steihaug_max_iters = Some(30);
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "trust_region OFV must be finite, got {}",
        result.ofv
    );
    for (i, &th) in result.theta.iter().enumerate() {
        let lo = model.default_params.theta_lower[i];
        let hi = model.default_params.theta_upper[i];
        assert!(th > lo && th < hi, "theta[{}] = {} escaped bounds", i, th);
    }
}

#[test]
fn bobyqa_ofv_no_worse_than_slsqp_on_warfarin() {
    // Sanity check: BOBYQA must find a fit that is at least as good as SLSQP
    // by OFV (allowing a small slack for the derivative-free optimizer's
    // coarser termination). The earlier theta-agreement test was misleading
    // — it passed only because the original BOBYQA configuration barely
    // moved from the initial values, which made it spuriously "agree" with
    // SLSQP's local minimum. Once BOBYQA can actually explore (rhobeg set,
    // xtol loosened) it routinely finds a better OFV on warfarin than SLSQP
    // does, so the right invariant is OFV-not-worse, not theta-agreement.
    let (model, population) = data_and_model();

    let mut opts_slsqp = base_options();
    opts_slsqp.optimizer = Optimizer::Slsqp;
    let r_slsqp = fit(&model, &population, &model.default_params, &opts_slsqp)
        .expect("slsqp fit must succeed");

    let mut opts_bobyqa = base_options();
    opts_bobyqa.optimizer = Optimizer::Bobyqa;
    opts_bobyqa.outer_maxiter = 300;
    let r_bobyqa = fit(&model, &population, &model.default_params, &opts_bobyqa)
        .expect("bobyqa fit must succeed");

    // BOBYQA's OFV should be ≤ SLSQP's + 5 units of slack. The bug we
    // care about catching is "BOBYQA is stuck near the initial point",
    // which on warfarin produces an OFV gap of ~150 vs converged SLSQP
    // (worst-case observed pre-fix). A 5-unit slack sits well inside
    // that gap while staying loose enough to absorb the derivative-free
    // optimizer's coarser termination.
    assert!(
        r_bobyqa.ofv <= r_slsqp.ofv + 5.0,
        "BOBYQA OFV {} should be no worse than SLSQP OFV {} + 5",
        r_bobyqa.ofv,
        r_slsqp.ofv,
    );
}

#[test]
fn slsqp_stops_via_stagnation_well_before_a_generous_maxeval() {
    // Regression: on γ-bearing FOCEI scenarios SLSQP used to grind through
    // hundreds of post-convergence evals without terminating. The
    // stagnation guard short-circuits once recent evals show no OFV
    // improvement so NLopt's xtol/ftol can fire.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Slsqp;
    opts.outer_maxiter = 1000;

    let result =
        fit(&model, &population, &model.default_params, &opts).expect("slsqp fit must succeed");
    assert!(result.ofv.is_finite());
    // FD-mode unbounded budget is outer_maxiter * (n+1). Asserting a small
    // fraction of that keeps the assertion meaningful as n grows.
    let n_params = model.default_params.theta.len()
        + model.default_params.omega.matrix.nrows()
        + model.default_params.sigma.values.len();
    let budget = opts.outer_maxiter * (n_params + 1);
    let ceiling = budget / 4;
    assert!(
        result.n_iterations < ceiling,
        "SLSQP burned {} evals (ceiling {}, n={}) — stagnation guard should fire earlier",
        result.n_iterations,
        ceiling,
        n_params,
    );
}

// `stagnation_guard_toggle_runs_to_natural_termination` was removed:
// SLSQP's own `XtolReached` now fires on warfarin at ~eval 100 — well
// before the guard window (~50 evals past the last improvement at ~eval
// 65, so eval ~115).  Both guard-on and guard-off therefore exit with
// identical OFV at exactly the same eval count, so the toggle no longer
// discriminates and the e2e test cannot tell whether the guard wired
// through.  The guard's mechanism is now verified directly by the
// `detect_stagnation_*` unit tests in `src/estimation/outer_optimizer.rs`,
// which is stricter coverage than the e2e ever provided (every branch of
// the `detect_stagnation` predicate is exercised).
//
// The companion test
// `slsqp_stops_via_stagnation_well_before_a_generous_maxeval` (above) is
// retained: it still passes and provides end-to-end coverage that the
// outer loop terminates well below the maxeval ceiling.

#[test]
fn final_ofv_no_worse_than_best_seen_during_trace() {
    // Regression for issue #59: when the stagnation guard short-circuits
    // by returning `best_ofv` with zero gradient, NLopt would still return
    // its *last evaluated* x rather than the best-seen one. The final OFV
    // could then be measurably worse than an intermediate value (and the
    // covariance step would silently fail because the Hessian was being
    // computed off-minimum). The fix tracks the best (xs, OFV) externally
    // and restores x0 to it before the final inner loop runs.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Slsqp;
    opts.outer_maxiter = 1000;
    opts.optimizer_trace = true;
    opts.run_covariance_step = false;
    // Pin to a single worker. The optimizer trace keeps its `TraceWriter` in a
    // thread-local (`estimation::trace`), so init/write/finish must all land on
    // the same thread. `fit_inner` runs inside a shared rayon pool, and after a
    // per-subject `par_iter` rayon may resume the outer-loop continuation on a
    // *different* worker — stranding the writer (empty trace, `trace_path` None).
    // That migration is near-certain when the whole test binary runs concurrent
    // fits on the shared pool. One worker => no migration => deterministic trace.
    // (The regression under test — #59 best-seen restoration — is thread-agnostic.)
    opts.threads = Some(1);

    let result =
        fit(&model, &population, &model.default_params, &opts).expect("slsqp fit must succeed");
    assert!(result.ofv.is_finite());

    let trace_path = result
        .trace_path
        .clone()
        .expect("optimizer_trace=true must produce a trace path");
    let csv = std::fs::read_to_string(&trace_path).expect("trace file must be readable");
    assert_eq!(
        result.trace_csv.as_deref(),
        Some(csv.as_str()),
        "the fit holds the trace file's text"
    );
    let mut lines = csv.lines();
    lines.next().expect("trace must have a header"); // skip header
    let min_trace_ofv = lines
        .filter_map(|line| {
            // Columns: iter,method,phase,ofv,...
            let ofv = line.split(',').nth(3)?;
            ofv.parse::<f64>().ok()
        })
        .filter(|x| x.is_finite())
        .fold(f64::INFINITY, f64::min);

    assert!(
        min_trace_ofv.is_finite(),
        "trace must record at least one finite OFV"
    );
    // Tolerance absorbs the slight OFV change from the final fresh-start
    // inner loop (vs the warm-started ones during the trace). Without the
    // best-seen restoration, the gap can blow up arbitrarily as the
    // optimizer drifts off-minimum during the short-circuit phase.
    assert!(
        result.ofv <= min_trace_ofv + 0.5,
        "final OFV {} is worse than best-seen OFV {} (Δ = {})",
        result.ofv,
        min_trace_ofv,
        result.ofv - min_trace_ofv,
    );

    let _ = std::fs::remove_file(&trace_path);
}

/// Verify that the GN trust-region loop (replacing LM + backtracking) produces
/// a finite, non-NaN result after a handful of outer iterations. This exercises
/// the full `solve_trust_region_subproblem` → `run_inner` → TR-ratio path
/// without waiting for convergence. Uses `outer_maxiter = 5` so it runs fast
/// on every PR.
#[test]
fn gn_tr_warmstart_returns_finite_ofv() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.method = EstimationMethod::FoceGn;
    opts.outer_maxiter = 5;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("GN trust-region must not panic");
    assert!(
        result.ofv.is_finite(),
        "GN-TR OFV must be finite after 5 iters, got {}",
        result.ofv
    );
    assert!(
        !result.theta.iter().any(|x| x.is_nan()),
        "GN-TR theta must not contain NaN"
    );
}

/// The exact-Hessian trust region (#2975) returns a finite, non-NaN fit after a handful
/// of iterations, through the exact gradient and Hessian and the final EBE check.
#[test]
fn trust_region_returns_a_finite_fit_after_a_few_iterations() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.outer_maxiter = 5;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("trust_region must return");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        !result.theta.iter().any(|x| x.is_nan()),
        "theta must not contain NaN"
    );
}

/// `steihaug_max_iters` still parses but no longer has an effect (#2975): the trust region
/// solves its subproblem exactly, so the same fit with and without it is bit-identical.
#[test]
fn steihaug_max_iters_has_no_effect_on_the_trust_region() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.outer_maxiter = 5;
    let plain = fit(&model, &population, &model.default_params, &opts).expect("fit");
    opts.steihaug_max_iters = Some(2);
    let pinned = fit(&model, &population, &model.default_params, &opts).expect("fit");
    assert_eq!(plain.ofv.to_bits(), pinned.ofv.to_bits());
}

/// IOV analytical gradient path: fit() on warfarin_iov with method=focei and
/// outer_maxiter=5 must complete without panic and return a finite OFV.
///
/// This confirms the new IOV analytical gradient is wired through the outer
/// optimizer without running to convergence (slow-test) — any panic or NaN
/// means the IOV analytical path broke.
#[test]
fn iov_analytical_gradient_path_returns_finite_ofv() {
    let model =
        ferx_core::parser::model_parser::parse_model_file(Path::new("examples/warfarin_iov.ferx"))
            .expect("warfarin_iov model must parse");
    let population =
        ferx_core::read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
            .expect("warfarin_iov data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 5;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("IOV fit must not error");
    assert!(
        result.ofv.is_finite(),
        "IOV fit OFV must be finite, got {}",
        result.ofv
    );
}

/// Fast coverage for the `Rescale2` scaling path and the non-mu-ref (covariate)
/// θ-gradient FD branch — neither is reachable from the warfarin tests (warfarin
/// is all mu-ref + the fast tests use BOBYQA). `two_cpt_oral_cov` has
/// non-mu-referenced covariate coefficients (`THETA_WT`/`THETA_CRCL`), so:
///   - built-in `Bfgs` under the `Auto` default resolves to `Rescale2`, running
///     `compute_rescale2_scale` and the `optimize_bfgs` scale block;
///   - the covariate θ columns exercise the analytic Laplace θ-gradient's FD
///     branch (the mu-ref shortcut handles only mu-referenced thetas);
///   - `covariance_method = rsr` runs the score cross-product (the `log|H̃|`
///     EBE-response) over those covariate columns.
/// A modest `outer_maxiter` keeps it fast — this guards that the paths execute
/// and a covariance status is decided, not convergence accuracy.
#[test]
fn bfgs_rescale2_and_covariate_covariance_paths_run() {
    use ferx_core::{CovarianceMethod, CovarianceStatus};
    let model = parse_model_file(Path::new("examples/two_cpt_oral_cov.ferx"))
        .expect("two_cpt_oral_cov model must parse");
    let population = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
        .expect("two_cpt_oral_cov data must load");
    let mut opts = FitOptions::default(); // parameter_scaling = Auto (→ Rescale2 for bfgs)
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::Bfgs; // built-in BFGS scale block
    opts.outer_maxiter = 8; // exercise paths; convergence not required
    opts.run_covariance_step = true;
    opts.covariance_method = CovarianceMethod::Sandwich; // rsr → score cross-product
    opts.verbose = false;
    let r = fit(&model, &population, &model.default_params, &opts)
        .expect("two_cpt_oral_cov FOCEI bfgs+rsr fit must run");
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);
    assert_ne!(
        r.covariance_status,
        CovarianceStatus::NotRequested,
        "covariance was requested; status must be decided"
    );
}

/// Pre-flight flat-theta guard (#826): a model with a theta that never reaches
/// the objective (TVFLAT, declared but unmapped) must degrade gracefully — the
/// gradient NLopt path used to return `Failure` on eval 1 and pin every parameter
/// at its initial value. The fit now returns `Ok`, warns, freezes the flat theta
/// at its initial, and the remaining thetas move off their initials.
#[test]
fn flat_theta_is_frozen_and_fit_proceeds() {
    let model = parse_model_file(Path::new("examples/flat_theta_warfarin.ferx"))
        .expect("flat_theta_warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let init = &model.default_params;
    let mut opts = base_options();
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::NloptLbfgs; // gradient path that used to die at eval 1

    let r = fit(&model, &population, init, &opts).expect("flat-theta fit must return Ok");
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);

    // The flat param is named TVFLAT; locate it and its active siblings by name.
    let idx = |name: &str| r.theta_names.iter().position(|n| n == name).unwrap();
    let (i_flat, i_cl, i_v, i_ka) = (idx("TVFLAT"), idx("TVCL"), idx("TVV"), idx("TVKA"));

    // Flat theta: frozen (FIX) and held exactly at its initial value.
    assert!(r.theta_fixed[i_flat], "TVFLAT must be frozen");
    assert_eq!(
        r.theta[i_flat], init.theta[i_flat],
        "frozen TVFLAT must stay at its initial value"
    );
    // A warning names the flat param and its lack of effect on the objective.
    assert!(
        r.warnings
            .iter()
            .any(|w| w.contains("TVFLAT") && w.contains("no effect")),
        "a flat-theta warning must be emitted: {:?}",
        r.warnings
    );
    // The active thetas actually moved — the whole fit did not freeze at init.
    let moved = |i: usize| (r.theta[i] - init.theta[i]).abs() > 1e-6;
    assert!(
        moved(i_cl) || moved(i_v) || moved(i_ka),
        "active thetas must move off their initials (TVCL/TVV/TVKA): {:?}",
        r.theta
    );
}

/// #1000, Tier 3: the converged direction, end to end. `trust_region` must not
/// only stop labelling budget exhaustion as convergence — it has to be able to
/// report `converged` at all, which before the fix came for free from any `Ok(_)`
/// and now depends on the no-progress stopping rule firing.
///
/// Since #2975 the trust region is a Newton method on the exact Hessian and stops on the
/// exact Newton decrement: warfarin under `focei` settles in ~10 iterations, so reaching
/// `maxiter` would itself be the regression. The OFV is checked against the optimizer `auto`
/// picks for the same options rather than a hardcoded constant, and the bound is 1e-3 — the
/// BHHH trust region it replaced settled 4.1 OFV units above `auto` on this objective (#864).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn trust_region_reports_converged_when_it_settles() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::TrustRegion;
    opts.outer_maxiter = 1000;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region fit must succeed");

    assert!(
        result.converged,
        "a settled trust-region fit must report converged (OFV {}, warnings {:?})",
        result.ofv, result.warnings
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("iteration budget")),
        "a settled fit must not carry the budget-exhaustion warning: {:?}",
        result.warnings
    );
    // Anchor: the same fit under the default optimizer. `trust_region` may take a
    // slightly different path, but a run that reports converged must land at the
    // same optimum, not somewhere short of it.
    //
    // Deliberately **one-sided**: the failure this guards against is settling
    // *above* the optimum. A `trust_region` run that reaches a lower OFV than
    // BOBYQA is an improvement, not a regression, and must not turn the nightly
    // job red — so only the shortfall is bounded.
    let mut ref_opts = base_options();
    ref_opts.method = EstimationMethod::FoceI;
    ref_opts.optimizer = Optimizer::NloptLbfgs; // the reference: `auto` is `trust_region` since #2976
    ref_opts.outer_maxiter = 1000;
    let reference = fit(&model, &population, &model.default_params, &ref_opts)
        .expect("reference fit must succeed");
    assert!(
        result.ofv < reference.ofv + 1e-3,
        "trust_region reported converged at OFV {} but L-BFGS reaches {}",
        result.ofv,
        reference.ofv
    );
}

/// #2975, Tier 3: `trust_region` on an IOV model. Its objective, exact Hessian and final EBE
/// check carry κ̂, so it must land at the default optimizer's optimum (one-sided, 1e-3). On
/// the #2975 benchmark it took 8 iterations on `warfarin_iov` against L-BFGS's ~47.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn trust_region_reaches_the_default_optimum_on_an_iov_model() {
    let model =
        ferx_core::parser::model_parser::parse_model_file(Path::new("examples/warfarin_iov.ferx"))
            .expect("warfarin_iov model must parse");
    let population =
        ferx_core::read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
            .expect("warfarin_iov data must load");
    let mut opts = base_options();
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::TrustRegion;
    opts.outer_maxiter = 300;
    let tr = fit(&model, &population, &model.default_params, &opts).expect("trust_region fit");
    assert!(tr.converged, "warnings {:?}", tr.warnings);
    opts.optimizer = Optimizer::NloptLbfgs; // the reference: `auto` is `trust_region` since #2976
    let reference = fit(&model, &population, &model.default_params, &opts).expect("reference");
    assert!(
        tr.ofv < reference.ofv + 1e-3,
        "trust_region {} vs L-BFGS {}",
        tr.ofv,
        reference.ofv
    );
}

/// #2975, Tier 3: the trust region's basin-shift restart. From every θ at half its start
/// value, `two_cpt_oral_cov` carries four subjects' EBEs in a higher individual mode along the
/// trust region's few, large steps; the leg converges (exact decrement ~5e-10) about 22 OFV
/// above the optimum, where a cold re-solve scores lower. The restart chain (the NLopt leg's
/// #1565 / #2093 rules) must resume with the cold EBEs and reach the default optimizer's optimum
/// from the same start, and say it resumed.
/// Regression it catches: without the chain the fit ends at −1176.8 against −1199.3, reported
/// not converged (`W_EBE_START_DEPENDENT`).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn trust_region_resumes_with_cold_ebes_after_a_basin_shift() {
    let model = parse_model_file(Path::new("examples/two_cpt_oral_cov.ferx"))
        .expect("two_cpt_oral_cov must parse");
    let population = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
        .expect("two_cpt_oral_cov data must load");
    let mut start = model.default_params.clone();
    for (i, t) in start.theta.iter_mut().enumerate() {
        *t = (*t * 0.5).clamp(start.theta_lower[i], start.theta_upper[i]);
    }
    let mut opts = base_options();
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::TrustRegion;
    opts.outer_maxiter = 300;
    let tr = fit(&model, &population, &start, &opts).expect("trust_region fit");
    opts.optimizer = Optimizer::NloptLbfgs; // the reference: `auto` is `trust_region` since #2976
    let reference = fit(&model, &population, &start, &opts).expect("reference");
    assert!(
        tr.ofv < reference.ofv + 1e-3,
        "trust_region {} vs default {} (warnings {:?})",
        tr.ofv,
        reference.ofv,
        tr.warnings
    );
    assert!(tr.converged, "warnings {:?}", tr.warnings);
    assert!(
        tr.warnings
            .iter()
            .any(|w| w.contains("resumed from the best point seen with cold EBEs")),
        "the fit must report the restart: {:?}",
        tr.warnings
    );
}

/// #2975, Tier 3: the trust region re-models on the exact Hessian when the set of coordinates
/// free of their bounds changes. `bioavailability` starts at OFV 7.6e6 and pins coordinates to
/// their bounds one after another on the way in. Without the refresh, the BFGS model carried
/// curvature learned in the larger subspace (model decrement 2.5 where the exact one was 121) and
/// 2 of 5 starts converged to the second minimum at OFV 1104.4 instead of 647.3. The default
/// start was one of them.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn trust_region_remodels_when_the_free_set_changes() {
    let model = parse_model_file(Path::new("examples/bioavailability.ferx"))
        .expect("bioavailability must parse");
    let population = read_nonmem_csv(Path::new("data/bioavailability.csv"), None, None)
        .expect("bioavailability data must load");
    let mut opts = base_options();
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::TrustRegion;
    opts.outer_maxiter = 500;
    let tr = fit(&model, &population, &model.default_params, &opts).expect("trust_region fit");
    assert!(tr.converged, "warnings {:?}", tr.warnings);
    // The two minima are 457 OFV apart, so any bound well inside that gap separates them.
    assert!(
        (tr.ofv - 647.343).abs() < 0.01,
        "trust_region must reach the global minimum 647.343, got {}",
        tr.ofv
    );
}

/// #2975, Tier 3: a stop the noise test cannot settle gets one verification leg at a tight
/// inner tolerance. From every θ × 0.99, `warfarin_logit_f`'s trust region collapses its radius
/// with an exact decrement above the measured evaluation noise: the gradient disagrees with the
/// objective at `inner_tol = 1e-5`. The leg at `1e-10` settles it (gain 1.0e-7 in the #2975
/// benchmark), so the fit must be converged at the default optimizer's optimum.
/// Regression it catches: without the leg the fit ends at the optimum but `converged = false`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn trust_region_verifies_an_unsettled_stop_at_a_tight_inner_tolerance() {
    let model = parse_model_file(Path::new("examples/warfarin_logit_f.ferx"))
        .expect("warfarin_logit_f must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_logit_f.csv"), None, None)
        .expect("warfarin_logit_f data must load");
    let mut start = model.default_params.clone();
    for (i, t) in start.theta.iter_mut().enumerate() {
        *t = (*t * 0.99).clamp(start.theta_lower[i], start.theta_upper[i]);
    }
    let mut opts = base_options();
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::TrustRegion;
    opts.outer_maxiter = 300;
    let tr = fit(&model, &population, &start, &opts).expect("trust_region fit");
    opts.optimizer = Optimizer::NloptLbfgs; // the reference: `auto` is `trust_region` since #2976
    let reference = fit(&model, &population, &start, &opts).expect("reference");
    assert!(tr.converged, "warnings {:?}", tr.warnings);
    assert!(
        tr.ofv < reference.ofv + 1e-3,
        "trust_region {} vs L-BFGS {}",
        tr.ofv,
        reference.ofv
    );
}

/// #2975, Tier 3: where the exact Hessian is not resolved at the very start (no model yet to
/// keep), the trust region models with the finite-difference Hessian of the same objective.
/// `per_route_lag_absorption` on its own simulated design: at the initial estimates the exact
/// Hessian is non-finite (#3004), while it is fine at the optimum.
/// Regression it catches: without the fallback the fit stops on its first iteration at the
/// start values (`W_STALLED_AT_INIT`, about 11 OFV above the optimum).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn trust_region_starts_on_the_fd_hessian_where_the_exact_one_is_unresolved() {
    let src = std::fs::read_to_string("examples/per_route_lag_absorption.ferx")
        .expect("per_route_lag_absorption must read");
    let dir = std::env::temp_dir().join(format!("ferx-tr-fd-start-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let fit_with = |optimizer: &str| {
        let text = src.replacen(
            "[fit_options]",
            &format!("[fit_options]\n  optimizer = {optimizer}\n  threads = 1"),
            1,
        );
        let path = dir.join(format!("m_{optimizer}.ferx"));
        std::fs::write(&path, text).expect("write model");
        ferx_core::run_model_simulate(path.to_str().unwrap())
            .expect("simulate and fit")
            .0
    };
    let tr = fit_with("trust_region");
    let reference = fit_with("nlopt_lbfgs");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        tr.ofv < reference.ofv + 1e-3,
        "trust_region {} vs nlopt_lbfgs {} (warnings {:?})",
        tr.ofv,
        reference.ofv,
        tr.warnings
    );
    assert!(tr.n_iterations > 1, "the trust region must leave its start");
}
