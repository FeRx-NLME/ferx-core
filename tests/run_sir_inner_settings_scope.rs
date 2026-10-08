//! #1767 findings 1 and 2, restated for #426: SIR's inner solver is the one its
//! `SirSettings` records, and nothing it or a `fit()` sets reaches the next standalone call.
//!
//! `inner_optimizer` / `ebe_warm_start` used to reach the EBE re-solves through process
//! globals that every `fit()` wrote and none restored, so a post-hoc step scored with
//! whichever solver the most recent fit in the process used. Since #426 they are carried by
//! the call's own fit scope (a thread-local, and pools whose workers carry the same value), so
//! each call reads its own options.
//!
//! Fixture: warfarin, analytic (`Dual2`) inner gradients (measured). Mutations: drop the inner settings from the scope
//! `run_sir_core` opens (part 1: the lbfgs draws come out as the default solver's); write a
//! fit's inner settings to a global again (part 2: the default-options covariance step after
//! an lbfgs fit re-solves with lbfgs); set them in `run_sir` before validation again (part 3:
//! the `Err` call leaks lbfgs).

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
fn sir_runs_with_its_recorded_inner_solver_and_leaks_nothing() {
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
    let lbfgs = core(InnerOptimizer::Lbfgs);
    let auto = core(InnerOptimizer::Auto);
    assert_eq!(
        lbfgs.settings.scoring.inner_optimizer,
        InnerOptimizer::Lbfgs
    );
    assert_ne!(
        lbfgs.effective_sample_size.to_bits(),
        auto.effective_sample_size.to_bits(),
        "the recorded lbfgs must be what the draws ran, not the unarmed default: {} vs {}",
        lbfgs.effective_sample_size,
        auto.effective_sample_size
    );

    // ── 2. A default-options covariance step reads the default, whatever ran before ─
    let fit_opts = |mode| FitOptions {
        verbose: false,
        run_covariance_step: true,
        sir: false,
        inner_optimizer: mode,
        ..prep.parsed.fit_options.clone()
    };
    let fitted = fit(
        model,
        pop,
        &prep.init_params,
        &fit_opts(InnerOptimizer::Auto),
    )
    .expect("fit");
    let quiet = |mode| FitOptions {
        verbose: false,
        inner_optimizer: mode,
        ..FitOptions::default()
    };
    let cov = |mode| {
        bits(
            &run_covariance(&fitted, Some(model), Some(pop), &quiet(mode))
                .expect("run_covariance")
                .covariance_matrix,
        )
    };
    let cov_default = cov(InnerOptimizer::Auto);
    assert_ne!(
        cov(InnerOptimizer::Lbfgs),
        cov_default,
        "premise: the covariance step must see the caller's inner solver on this fixture"
    );
    core(InnerOptimizer::Lbfgs);
    assert_eq!(
        cov(InnerOptimizer::Auto),
        cov_default,
        "run_sir_core's lbfgs must not outlive the call"
    );
    let short = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        ..fit_opts(InnerOptimizer::Lbfgs)
    };
    fit(model, pop, &prep.init_params, &short).expect("lbfgs fit");
    assert_eq!(
        cov(InnerOptimizer::Auto),
        cov_default,
        "an lbfgs fit must not change the inner solver of a later default-options call"
    );

    // ── 3. A failing `run_sir` leaves nothing behind either ───────────────────────
    let mut no_cov = fitted.clone();
    no_cov.covariance_matrix = None;
    no_cov.sir_settings = Some(ferx_core::estimation::sir::SirSettings {
        scoring: ferx_core::ScoringSettings {
            inner_optimizer: InnerOptimizer::Lbfgs,
            ..Default::default()
        },
        ..Default::default()
    });
    let err = run_sir(
        &no_cov,
        Some(model),
        Some(pop),
        &quiet(InnerOptimizer::Auto),
    )
    .expect_err("no covariance");
    assert!(err.to_string().contains("covariance_matrix"), "{err}");
    assert_eq!(
        cov(InnerOptimizer::Auto),
        cov_default,
        "a run_sir that returned Err must not have moved the inner solver"
    );
}
