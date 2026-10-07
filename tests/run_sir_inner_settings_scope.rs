//! #1767 findings 1 and 2: SIR's inner solver is the one its `SirSettings` records, and
//! it does not leak into the next standalone call.
//!
//! `inner_optimizer` / `ebe_warm_start` reach the EBE re-solves through process globals
//! (#426). Before #1767, `run_sir_core` stamped them from `options` but never applied
//! them, so a direct caller's record could name a solver that did not run; and `run_sir`
//! applied them unscoped, before validation, so they outlived the call — even an `Err`.
//!
//! Its own test binary, with one test, because the globals are process-wide: a
//! concurrent test's `fit()` in a shared binary could move them mid-run.
//!
//! Fixture: warfarin, FD inner gradients. Mutations: drop `with_inner_settings` from
//! `run_sir_core` (part 1: the draws follow the global); drop its `Restore` guard (part 2:
//! the covariance step after SIR re-solves with lbfgs); set the globals in `run_sir`
//! before validation again (part 3: the `Err` call leaks lbfgs).

use ferx_core::estimation::inner_optimizer::set_inner_optimizer;
use ferx_core::estimation::parameterization::pack_params;
use ferx_core::estimation::sir::run_sir_core;
use ferx_core::{fit, prepare_run, run_covariance, run_sir, FitOptions, InnerOptimizer};
use nalgebra::{DMatrix, DVector};

fn bits(m: &Option<DMatrix<f64>>) -> Vec<u64> {
    m.as_ref()
        .expect("covariance step ran")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn sir_runs_with_its_recorded_inner_solver_and_restores_the_process_one() {
    let prep = prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv")).expect("prepare");
    let model = &prep.parsed.model;
    let pop = &prep.population;

    // ── 1. A direct `run_sir_core` call draws with the solver it records ──────────
    let params = prep.init_params.clone();
    let etas: Vec<DVector<f64>> = (0..pop.subjects.len())
        .map(|_| DVector::zeros(model.n_eta))
        .collect();
    let proposal = DMatrix::from_diagonal(&DVector::from_element(pack_params(&params).len(), 0.01));
    let sir_opts = |mode| FitOptions {
        verbose: false,
        sir_samples: 40,
        sir_resamples: 20,
        sir_seed: Some(1767),
        inner_optimizer: mode,
        ..prep.parsed.fit_options.clone()
    };
    let core = |mode| {
        run_sir_core(model, pop, &params, &etas, &proposal, 0.0, &sir_opts(mode))
            .expect("run_sir_core")
    };
    set_inner_optimizer(InnerOptimizer::Bfgs);
    let lbfgs_under_bfgs = core(InnerOptimizer::Lbfgs);
    let bfgs = core(InnerOptimizer::Bfgs);
    set_inner_optimizer(InnerOptimizer::Lbfgs);
    let lbfgs_under_lbfgs = core(InnerOptimizer::Lbfgs);
    assert_eq!(
        lbfgs_under_bfgs.settings.inner_optimizer,
        InnerOptimizer::Lbfgs
    );
    assert_ne!(
        bfgs.effective_sample_size.to_bits(),
        lbfgs_under_lbfgs.effective_sample_size.to_bits(),
        "premise: the inner solver must move the ESS on this fixture"
    );
    assert_eq!(
        lbfgs_under_bfgs.effective_sample_size.to_bits(),
        lbfgs_under_lbfgs.effective_sample_size.to_bits(),
        "the recorded lbfgs must be what ran, whatever the process global held: {} vs {}",
        lbfgs_under_bfgs.effective_sample_size,
        lbfgs_under_lbfgs.effective_sample_size
    );

    // ── 2. …and leaves the process's solver as it found it ────────────────────────
    let fit_opts = FitOptions {
        verbose: false,
        run_covariance_step: true,
        sir: false,
        ..prep.parsed.fit_options.clone()
    };
    let fitted = fit(model, pop, &prep.init_params, &fit_opts).expect("fit");
    let quiet = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    let cov_under = |mode| {
        set_inner_optimizer(mode);
        bits(
            &run_covariance(&fitted, Some(model), Some(pop), &quiet)
                .expect("run_covariance")
                .covariance_matrix,
        )
    };
    let cov_lbfgs = cov_under(InnerOptimizer::Lbfgs);
    let cov_bfgs = cov_under(InnerOptimizer::Bfgs);
    assert_ne!(
        cov_bfgs, cov_lbfgs,
        "premise: the covariance step must see the process's inner solver"
    );
    core(InnerOptimizer::Lbfgs); // the global holds Bfgs going in
    let after_core = bits(
        &run_covariance(&fitted, Some(model), Some(pop), &quiet)
            .expect("run_covariance")
            .covariance_matrix,
    );
    assert_eq!(
        after_core, cov_bfgs,
        "run_sir_core must restore the process's inner solver"
    );

    // ── 3. A failing `run_sir` leaves it untouched too ────────────────────────────
    let mut no_cov = fitted.clone();
    no_cov.covariance_matrix = None;
    no_cov.sir_settings = Some(ferx_core::estimation::sir::SirSettings {
        inner_optimizer: InnerOptimizer::Lbfgs,
        ..Default::default()
    });
    let err = run_sir(&no_cov, Some(model), Some(pop), &quiet).expect_err("no covariance");
    assert!(err.contains("covariance_matrix"), "{err}");
    let after_err = bits(
        &run_covariance(&fitted, Some(model), Some(pop), &quiet)
            .expect("run_covariance")
            .covariance_matrix,
    );
    assert_eq!(
        after_err, cov_bfgs,
        "a run_sir that returned Err must not have moved the process's inner solver"
    );
}
