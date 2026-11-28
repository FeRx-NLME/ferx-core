//! #3153 — the opening step's objective scale.
//!
//! SLSQP and L-BFGS start from `H = I`, so the opening step is `−∇f`; the cap that keeps it
//! inside the box (`outer_optimizer::cap_factor`) divides the first gradient by a scalar `c`
//! (5.1e5 on the issue's data). When only the gradient was divided, NLopt's first line
//! search and first secant pair compared a raw objective with a `c`-times-smaller slope. On a
//! one-state turnover model with no random effects, L-BFGS's first pair came out with
//! `sᵀy < 0`, Luksan's PLIS restarted on the *raw* gradient, and the step ran to the first
//! bound: `TH_KIN` on its 1e-10 guard, `TH_KOUT` ×1400, `TH_SD` ×18000, and the fit ended
//! 1887 OFV above its optimum reporting `Converged: YES`. SLSQP, capped on every evaluation,
//! ended at a degenerate point at 8192.7 reported plain `CONVERGED`.
//!
//! Both optimizers now minimise `f / c` with gradient `∇f / c` for the first leg, so the
//! first step is the same and every later pair is consistent.
//!
//! The fixture (`tests/fixtures/turnover_collapse/`) is the issue's generator at 40 subjects,
//! seed 4: on it `main` ended at 98.13 with L-BFGS (`TH_KIN` on the guard) and at 581.86 with
//! SLSQP, both reported converged, and `trust_region` and `bobyqa` reach −30.7341. The per-PR
//! tests stop after four outer iterations, where `main` is already off the optimum (OFV 58.5
//! with `TH_KIN` 0.0036 under L-BFGS; 581.9 with `TH_KIN` 0 under SLSQP) and the fix is not
//! (−28.4 / −27.4, `TH_KIN` 0.021 against 0.0195 true). The convergence checks are Tier 3.
//! Numbers measured on Linux x86_64.

use ferx_core::types::{EstimationMethod, FitOptions, Optimizer};

/// The optimum `trust_region`, `bobyqa` and both optimizers below reach.
const OPTIMUM: f64 = -30.7341;
/// The internal log-packing guard a collapsed θ sits on.
const GUARD: f64 = 1e-10;

fn fit_with(optimizer: Optimizer, outer_maxiter: usize) -> ferx_core::FitResult {
    let p = ferx_core::prepare_run(
        "tests/fixtures/turnover_collapse/turnover.ferx",
        Some("tests/fixtures/turnover_collapse/turnover.csv"),
    )
    .expect("the turnover fixture loads");
    let options = FitOptions {
        method: EstimationMethod::FoceI,
        optimizer,
        outer_maxiter,
        run_covariance_step: false,
        threads: Some(1),
        verbose: false,
        ..p.parsed.fit_options.clone()
    };
    ferx_core::fit(&p.parsed.model, &p.population, &p.init_params, &options)
        .expect("the turnover fit returns a result")
}

/// After four outer iterations the fit is descending towards the optimum, not collapsing:
/// `TH_KIN` (θ 0) is within a factor of two of its true 0.0195, and the objective is below
/// zero, which neither `main` fit reached at this budget.
fn assert_not_collapsing(optimizer: Optimizer) {
    let r = fit_with(optimizer, 4);
    assert!(
        r.theta[0] > 0.01 && r.theta[0] < 0.04 && r.ofv < 0.0,
        "{optimizer:?}: θ_KIN {} and OFV {} after four iterations (main: ≤ 0.0036 and ≥ 58); θ = {:?}",
        r.theta[0],
        r.ofv,
        r.theta
    );
}

fn assert_converges_to_the_optimum(optimizer: Optimizer) {
    let r = fit_with(optimizer, 20);
    // Realised |Δ| 1e-5 or below on Linux x86_64 under both; `main` ended at 98.13 and 581.86.
    assert!(
        (r.ofv - OPTIMUM).abs() < 1e-3,
        "{optimizer:?}: expected the optimum {OPTIMUM}, got {} (θ = {:?})",
        r.ofv,
        r.theta
    );
    // No θ on its guard, which is where the collapsed fits sat (KIN = KOUT = 1e-10).
    assert!(
        r.theta.iter().all(|&t| t > 1e3 * GUARD),
        "{optimizer:?}: a θ on the guard: {:?}",
        r.theta
    );
    assert!(r.converged, "{optimizer:?}: {:?}", r.warnings);
}

/// L-BFGS, the default path.
#[test]
fn the_default_lbfgs_descends_from_the_turnover_start_instead_of_collapsing_onto_the_guard() {
    assert_not_collapsing(Optimizer::NloptLbfgs);
}

/// SLSQP, which `main` capped on every evaluation.
#[test]
fn slsqp_descends_from_the_turnover_start_instead_of_collapsing() {
    assert_not_collapsing(Optimizer::Slsqp);
}

#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn the_default_lbfgs_reaches_the_turnover_optimum() {
    assert_converges_to_the_optimum(Optimizer::NloptLbfgs);
}

#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn slsqp_reaches_the_turnover_optimum() {
    assert_converges_to_the_optimum(Optimizer::Slsqp);
}
