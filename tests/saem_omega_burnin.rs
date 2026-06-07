//! Slow convergence test for the SAEM Ω burn-in (`saem_omega_burnin`).
//!
//! Run with:
//!
//!   cargo test --features slow-tests --test saem_omega_burnin
//!
//! ## What this guards
//!
//! On sparse data (few observations per subject) the SAEM E-step starts every
//! η at 0 and, with the exploration step size γ=1, the iteration-1 M-step would
//! install the spread of a cold-start chain that has only taken `n_mh_steps`
//! steps — far below the true Ω. Because the MH proposal scale is tied to Ω
//! (`δ·chol(Ω)`), that under-estimate shrinks the proposal and the chain can
//! never re-inflate Ω: between-subject variability collapses into the residual
//! error. The burn-in (default `omega_burnin = 20`) holds Ω at its starting
//! value while the chain warms up, which prevents the collapse.
//!
//! The dataset is fully synthetic (1 observation per subject — deliberately
//! under-identified per subject so per-subject recovery is slow and the failure
//! is visible within a bounded iteration budget). No proprietary data is used.
//!
//! The test runs the *same* fit twice — with the burn-in disabled
//! (`omega_burnin = 0`) and enabled — and asserts both recover the total
//! between-subject variance.
//!
//! ## Note on the burn-in vs the damped Ω SA step
//!
//! This test originally also asserted that *without* burn-in Ω collapses (the
//! no-burn-in run recovering far less variance than the burn-in run). That
//! contrast no longer holds: the damped Robbins-Monro step for the Ω sufficient
//! statistic (`OMEGA_SA_MAX_STEP` in `saem.rs`, added with the block-Ω rank-1
//! collapse fix) caps how fast a single cold/un-equilibrated draw can move Ω, so
//! Ω is no longer overwritten by the iteration-1 cold-start spread even with the
//! burn-in off. The damped step is a strict generalisation of the burn-in's
//! protection — it guards the same cold-start failure continuously rather than
//! only for the first `omega_burnin` iterations. The burn-in is retained as a
//! complementary guard; this test now checks that both configurations recover Ω.
//! The block-Ω collapse mechanism itself is guarded by the
//! `saem_block_omega_collapse` test.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, OmegaMatrix, Population, Subject};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};
use std::collections::HashMap;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  1.0, 500.0)
  omega ETA_CL ~ 0.10
  omega ETA_V  ~ 0.10
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// True total between-subject variance: trace(Ω) = ω²_CL + ω²_V.
const TRUE_TRACE: f64 = 0.09 + 0.09;

/// Build a sparse synthetic population: `n` subjects, one IV bolus, exactly one
/// observation per subject, with the observation time cycling over a grid so the
/// population still covers the concentration-time curve.
fn sparse_population(n: usize) -> Population {
    let times = [0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0];
    let subjects: Vec<Subject> = (1..=n)
        .map(|i| {
            let t = times[i % times.len()];
            Subject {
                id: format!("{i}"),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![t],
                observations: vec![0.0],
                obs_cmts: vec![1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
            }
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Simulate observations into the template population from the true parameters.
fn simulate_into(model: &ferx_core::types::CompiledModel, template: &Population) -> Population {
    // Truth: theta at the model's starting values, Ω diagonal 0.09 (the init is
    // 0.10 so this is a genuine, if small, estimation), 10% proportional error.
    let mut truth = model.default_params.clone();
    truth.theta = vec![5.0, 50.0];
    truth.omega = OmegaMatrix::from_diagonal(&[0.09, 0.09], vec!["ETA_CL".into(), "ETA_V".into()]);
    truth.sigma.values = vec![0.10];

    let sim = simulate_with_seed(model, template, &truth, 1, 20240527);

    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        let dv: Vec<f64> = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value().max(1e-6))
            .collect();
        subj.observations = dv;
    }
    pop
}

/// Run SAEM with the given burn-in and return the recovered trace(Ω).
///
/// `saem_n_mh_steps` is pinned at 3 (the pre-PR #148 default) so the chain is
/// deliberately under-mixed.  The current default (20) mixes the chain well
/// enough on its own to keep Ω from collapsing, which would mask the effect
/// being exercised here; pinning to 3 keeps the recovery driven by the burn-in
/// and the damped Ω SA step rather than by raw mixing.
fn fit_trace(
    model: &ferx_core::types::CompiledModel,
    population: &Population,
    omega_burnin: usize,
) -> f64 {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 120;
    opts.saem_n_convergence = 120;
    opts.saem_n_mh_steps = 3;
    opts.saem_omega_burnin = omega_burnin;
    opts.saem_seed = Some(7);
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result =
        fit(model, population, &model.default_params, &opts).expect("SAEM fit must succeed");

    // trace(Ω) = sum of the diagonal variances.
    result.omega.diagonal().iter().sum::<f64>()
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_burnin_recovers_omega_on_sparse_data() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let template = sparse_population(300);
    let population = simulate_into(&model, &template);

    let trace_burnin = fit_trace(&model, &population, 20);
    let trace_no_burnin = fit_trace(&model, &population, 0);

    eprintln!(
        "trace(Ω): truth={TRUE_TRACE:.3}  burnin={trace_burnin:.4}  no_burnin={trace_no_burnin:.4}"
    );

    // Both configurations recover the total BSV variance in the right
    // neighbourhood of the truth. The window is wide because the ω_CL/ω_V split
    // is only weakly identified by one observation per subject — the *total*
    // variance is the robust quantity. Both assertions fail on the pre-fix code
    // (no burn-in, no damped Ω SA step), where Ω collapses into residual error.
    for (label, trace) in [("burn-in", trace_burnin), ("no-burn-in", trace_no_burnin)] {
        assert!(
            trace > 0.5 * TRUE_TRACE && trace < 1.6 * TRUE_TRACE,
            "{label} run did not recover trace(Ω): got {trace:.4}, truth {TRUE_TRACE:.3}"
        );
    }
}
