//! Tier-1 tests for [`PoolPlan`] — the nested-pool surface a many-fit tool
//! (bootstrap, SCM, model search) uses to split a thread budget between the
//! replicate level and `fit()`'s own per-subject level (#1115).
//!
//! The hazard the type exists for is the worker **stack**: `fit()`'s pools are
//! built with 32 MiB workers because wide ODE+IOV analytic gradients overflow a
//! stock 2 MiB Rayon stack, and an outer pool built from a bare
//! `ThreadPoolBuilder::new()` silently inherits the default — faulting on
//! exactly the models that need the bigger stack, and only under a tool, where
//! no existing test looks. Both halves are pinned below: the declared size, and
//! a runtime probe that actually touches more stack than the default provides.

use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

// ── budget arithmetic ────────────────────────────────────────────────────────

#[test]
fn from_budget_spends_the_budget_on_the_outer_level_first() {
    // (total_threads, n_units) -> (replicates, threads_per_fit)
    let cases = [
        // A bootstrap: far more replicates than threads, so every thread runs a
        // replicate and each fit is single-threaded.
        ((8, 200), (8, 1)),
        ((8, 8), (8, 1)),
        // Fewer units than threads: the leftover budget goes to the inner level.
        ((8, 2), (2, 4)),
        ((8, 1), (1, 8)),
        // The division floors rather than oversubscribing: 8 over 3 is 3 x 2,
        // two threads left idle.
        ((8, 3), (3, 2)),
        ((7, 2), (2, 3)),
        // Degenerate inputs still produce a runnable plan.
        ((1, 200), (1, 1)),
        ((4, 0), (1, 4)),
    ];
    for ((total, units), (reps, tpf)) in cases {
        let plan = PoolPlan::from_budget(total, units);
        assert_eq!(
            (plan.replicates(), plan.threads_per_fit()),
            (reps, tpf),
            "from_budget({total}, {units})"
        );
        assert!(
            plan.total_threads() <= total.max(1),
            "from_budget({total}, {units}) oversubscribed: {} > {total}",
            plan.total_threads()
        );
    }
}

#[test]
fn a_zero_budget_falls_back_to_the_engine_default() {
    // `0` means "the engine default", not "let Rayon decide" — whatever an
    // unpinned `fit()` would run on. That is cores-minus-one capped at 8, unless
    // a CLI explicitly sized the process-wide pool, so the plan has to resolve it
    // through the same helper `default_fit_pool` sizes itself with.
    let expected = crate::api::effective_default_threads();
    let plan = PoolPlan::from_budget(0, 200);
    assert_eq!(plan.replicates(), expected);
    assert_eq!(plan.threads_per_fit(), 1);
    assert_eq!(PoolPlan::default(), plan);
}

#[test]
fn a_plan_is_never_zero_wide_on_either_level() {
    // `0` workers would mean "Rayon picks", which is the ambiguity the type
    // removes — both levels clamp up to 1 instead.
    let plan = PoolPlan::new(0, 0);
    assert_eq!((plan.replicates(), plan.threads_per_fit()), (1, 1));
    assert_eq!(plan.total_threads(), 1);
}

// ── the stack-size hazard ────────────────────────────────────────────────────

#[test]
fn the_outer_pool_declares_the_ferx_worker_stack() {
    assert_eq!(FIT_RAYON_STACK_SIZE, 32 * 1024 * 1024);
    for plan in [
        PoolPlan::default(),
        PoolPlan::new(1, 1),
        PoolPlan::from_budget(8, 200),
    ] {
        assert_eq!(
            plan.worker_stack_size(),
            FIT_RAYON_STACK_SIZE,
            "a plan must never hand a caller the platform-default worker stack"
        );
    }
}

#[test]
fn the_outer_pool_really_has_more_than_the_default_stack() {
    // The declared size above is only a promise; this is the observation. A
    // Rayon worker built from `ThreadPoolBuilder::new()` gets the std default
    // (2 MiB on the platforms ferx runs on), so touching a 4 MiB frame inside
    // `install` overflows unless the plan applied `FIT_RAYON_STACK_SIZE`.
    // `install` runs `op` on a pool worker, so this is the worker's stack, not
    // the caller's.
    //
    // A regression here fails loudly but bluntly: a stack overflow aborts the
    // process rather than unwinding, so the whole test binary dies and its other
    // tests read as "not run". That is deliberate — the declared-size assertion
    // in the test above is the graceful half of the pair, and it is what names
    // the cause; this one is the observation that the declaration is real, and
    // there is no way to observe it without risking the overflow.
    const PROBE: usize = 4 * 1024 * 1024;
    let sum = PoolPlan::new(2, 1)
        .install(|| {
            let mut frame = [0u8; PROBE];
            // Write both ends so the pages are actually committed and the array
            // cannot be optimised away.
            frame[0] = 1;
            frame[PROBE - 1] = 2;
            std::hint::black_box(&frame);
            frame[0] as usize + frame[PROBE - 1] as usize
        })
        .expect("outer pool");
    assert_eq!(sum, 3);
}

// ── installing the outer pool ────────────────────────────────────────────────

#[test]
fn install_runs_the_closure_on_a_pool_of_replicates_width() {
    let plan = PoolPlan::new(3, 2);
    let width = plan
        .install(rayon::current_num_threads)
        .expect("outer pool");
    assert_eq!(
        width, 3,
        "the outer pool is sized by `replicates`, not by the budget"
    );
}

#[test]
fn apply_to_pins_the_inner_fit_thread_count() {
    let mut opts = FitOptions::default();
    PoolPlan::from_budget(8, 200).apply_to(&mut opts);
    assert_eq!(
        opts.threads,
        Some(1),
        "a bootstrap plan asks for single-threaded inner fits"
    );

    let mut opts = FitOptions::default();
    PoolPlan::from_budget(8, 2).apply_to(&mut opts);
    assert_eq!(opts.threads, Some(4));
}

// ── end-to-end: what the inner fit actually does with the plan ───────────────

fn one_cpt_model() -> CompiledModel {
    parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("parse")
}

fn subject(id: &str, scale: f64) -> Subject {
    let obs_times = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times.iter().map(|t| scale * 10.0 / (1.0 + t)).collect();
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1, 1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Enough subjects that a 4-worker pool genuinely splits the per-subject loop,
/// so a thread-count-dependent reduction would show up.
fn population() -> Population {
    Population {
        subjects: (0..6)
            .map(|i| subject(&format!("{}", i + 1), 0.8 + 0.1 * i as f64))
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn short_fit_opts() -> FitOptions {
    FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        ..Default::default()
    }
    .quiet()
}

#[test]
fn a_single_threaded_plan_does_not_spawn_subject_level_workers() {
    let model = one_cpt_model();
    let pop = population();
    let mut opts = short_fit_opts();
    PoolPlan::from_budget(8, 200).apply_to(&mut opts);

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit");
    assert_eq!(
        result.n_threads_used, 1,
        "the inner fit must stay on one worker so it does not compete with the replicate loop"
    );
}

#[test]
fn a_pinned_plan_also_caps_a_multi_start_fit() {
    // The hazard `PoolPlan` exists for, in its worst form: a tool that pins
    // `threads_per_fit = 1` and also asks for several starts. The start fan-out
    // must run on the pinned pool, not on the full-width shared one underneath
    // every replicate — otherwise an 8-wide outer pool times an 8-wide shared
    // pool is 64-way oversubscription on 8 cores (#1115).
    let model = one_cpt_model();
    let pop = population();
    let mut opts = short_fit_opts();
    opts.n_starts = 3;
    PoolPlan::from_budget(8, 200).apply_to(&mut opts);
    assert_eq!(opts.threads, Some(1));

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit");
    assert_eq!(
        result.n_threads_used, 1,
        "a multi-start fit must honor the pinned thread count too, or a plan's \
         budget is silently multiplied by `n_starts`"
    );
}

#[test]
fn a_fit_is_bit_identical_across_thread_counts() {
    // The reduction order guard from #703, re-pinned from the tool's angle: a
    // bootstrap that changes `threads_per_fit` must not change the estimates.
    let model = one_cpt_model();
    let pop = population();

    let run = |threads: usize| {
        let mut opts = short_fit_opts();
        PoolPlan::new(1, threads).apply_to(&mut opts);
        fit(&model, &pop, &model.default_params, &opts).expect("fit")
    };

    let one = run(1);
    let four = run(4);

    assert_eq!(one.n_threads_used, 1);
    assert_eq!(four.n_threads_used, 4);
    assert_eq!(
        one.ofv.to_bits(),
        four.ofv.to_bits(),
        "OFV differs: {} vs {}",
        one.ofv,
        four.ofv
    );
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&one.theta), bits(&four.theta), "theta");
    assert_eq!(bits(&one.sigma), bits(&four.sigma), "sigma");
    assert_eq!(
        bits(one.omega.as_slice()),
        bits(four.omega.as_slice()),
        "omega"
    );
    assert_eq!(one.n_iterations, four.n_iterations);
}
