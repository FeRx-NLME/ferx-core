//! #1758 T5, the process-global row: `run_sir` applies the fit's recorded
//! `inner_optimizer` instead of inheriting the last `fit()` in the process.
//!
//! Until #426, `fit()` stored `inner_optimizer` in a process global and nothing
//! reset it, so before #1758 a standalone SIR re-solved every draw's EBEs with
//! whichever solver the most recent fit used. An R session that ran any other fit
//! in between was enough. Measured on the plan (macOS arm64): ESS
//! 78.37555008899564 in-fit against 78.37555022918151 after an intervening `bfgs`
//! fit. Since #426 the setting is per call, so default options with no record
//! score with `Auto`; the intervening `bfgs` fit is kept as the scenario.
//!
//! Fixture: warfarin, FOCEI, covariance step, 200 / 100 draws, seed 7, analytic
//! (`Dual2`) inner gradients (measured: `gradient_method_inner`, #426).
//! Mutation: drop `inner_optimizer` from `resolve_sir_options` — the draws
//! re-solve with the default solver, and the bit-identity dies. The premise
//! (default options score differently) is asserted first, so the row cannot
//! become a tautology.

use ferx_core::{fit, prepare_run, run_sir, FitOptions, FitResult, InnerOptimizer};

fn sir_outputs_cleared(fit: &FitResult) -> FitResult {
    let mut bare = fit.clone();
    bare.sir_ci_theta = None;
    bare.sir_ci_omega = None;
    bare.sir_ci_sigma = None;
    bare.sir_ci_kappa = None;
    bare.sir_ess = None;
    bare.sir_resamples_packed = None;
    bare.sir_seed = None;
    bare
}

fn ci_bits(ci: &Option<Vec<(f64, f64)>>) -> Option<Vec<(u64, u64)>> {
    ci.as_ref()
        .map(|v| v.iter().map(|(a, b)| (a.to_bits(), b.to_bits())).collect())
}

#[test]
fn run_sir_uses_the_recorded_inner_optimizer_not_the_last_fits() {
    let prep = prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv")).expect("prepare");
    let model = &prep.parsed.model;
    let pop = &prep.population;
    let lbfgs = FitOptions {
        verbose: false,
        method: ferx_core::EstimationMethod::FoceI,
        run_covariance_step: true,
        sir: true,
        sir_samples: 200,
        sir_resamples: 100,
        sir_seed: Some(7),
        inner_optimizer: InnerOptimizer::Lbfgs,
        ..prep.parsed.fit_options.clone()
    };
    let fit_a = fit(model, pop, &prep.init_params, &lbfgs).expect("lbfgs fit");
    let ess_a = fit_a.sir_ess.expect("in-fit SIR ran");
    assert!(ess_a.is_finite(), "in-fit ESS {ess_a}");
    assert_eq!(
        fit_a
            .sir_settings
            .as_ref()
            .map(|s| s.scoring.inner_optimizer),
        Some(InnerOptimizer::Lbfgs)
    );

    // Another fit in the same process moves the global to `bfgs`.
    let bfgs = FitOptions {
        verbose: false,
        inner_optimizer: InnerOptimizer::Bfgs,
        run_covariance_step: false,
        sir: false,
        outer_maxiter: 2,
        ..prep.parsed.fit_options.clone()
    };
    fit(model, pop, &prep.init_params, &bfgs).expect("bfgs fit");

    // Premise: without either record, default options re-solve with a different
    // inner solver and the ESS moves. Both cleared: since #1806 a fit with no SIR
    // record takes the stage record's settings.
    let mut no_record = sir_outputs_cleared(&fit_a);
    no_record.sir_settings = None;
    no_record.scoring_settings = None;
    let draws_only = FitOptions {
        verbose: false,
        sir_samples: 200,
        sir_resamples: 100,
        sir_seed: Some(7),
        ..FitOptions::default()
    };
    let before = run_sir(&no_record, Some(model), Some(pop), &draws_only).expect("run_sir");
    assert_ne!(
        before.sir_ess.map(f64::to_bits),
        Some(ess_a.to_bits()),
        "premise: the inner solver must change the ESS on this fixture"
    );

    let quiet = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    let out =
        run_sir(&sir_outputs_cleared(&fit_a), Some(model), Some(pop), &quiet).expect("run_sir");
    assert_eq!(
        out.sir_ess.map(f64::to_bits),
        Some(ess_a.to_bits()),
        "ESS {:?} vs in-fit {ess_a}",
        out.sir_ess
    );
    assert_eq!(
        ci_bits(&out.sir_ci_theta),
        ci_bits(&fit_a.sir_ci_theta),
        "θ"
    );
    assert_eq!(out.sir_settings, fit_a.sir_settings);
}
