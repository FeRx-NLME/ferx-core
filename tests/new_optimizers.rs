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

    let result =
        fit(&model, &population, &model.default_params, &opts).expect("slsqp fit must succeed");
    assert!(result.ofv.is_finite());

    let trace_path = result
        .trace_path
        .clone()
        .expect("optimizer_trace=true must produce a trace path");
    let csv = std::fs::read_to_string(&trace_path).expect("trace file must be readable");
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

/// Verify that the pre-warm cache change in `cost()` is transparent end-to-end:
/// after the refactor, `gradient()` is called on the same x that `cost()` just
/// evaluated (partial hit path), and the result must still be a finite, non-NaN OFV.
/// Uses a very low `outer_maxiter` so this runs fast on every PR.
#[test]
fn trust_region_cost_prewarm_cache_is_transparent() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.steihaug_max_iters = Some(5);
    opts.outer_maxiter = 5;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region must not panic with cost() pre-warm cache");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite after cost() pre-warm, got {}",
        result.ofv
    );
    assert!(
        !result.theta.iter().any(|x| x.is_nan()),
        "theta must not contain NaN after cost() pre-warm"
    );
}

#[test]
fn steihaug_max_iters_is_respected_by_trust_region() {
    // A very small steihaug_max_iters degrades step quality but should still
    // return a valid FitResult without crashing. Catches regressions in the
    // wiring from FitOptions to the argmin Steihaug subsolver.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.steihaug_max_iters = Some(2); // intentionally aggressive
    opts.outer_maxiter = 30;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region with tight CG budget must still return");
    assert!(result.ofv.is_finite());
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
