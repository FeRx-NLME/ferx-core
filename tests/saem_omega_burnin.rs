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
//! The test runs the *same* fit twice — once with the burn-in disabled
//! (`omega_burnin = 0`, the pre-fix behaviour) and once with it enabled — and
//! asserts the burn-in recovers the total between-subject variance while the
//! no-burn-in run collapses well below it. The recovery assertion alone would
//! fail on the pre-fix code.

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
            .map(|s| s.dv_sim.max(1e-6))
            .collect();
        subj.observations = dv;
    }
    pop
}

/// Run SAEM with the given burn-in and return the recovered trace(Ω).
///
/// `saem_n_mh_steps` is pinned at 3 (the pre-PR #148 default) so the
/// cold-start collapse this test guards against is reproducible.  The new
/// default of 10 MH steps mixes the chain well enough on its own that the
/// no-burn-in run no longer collapses Ω substantially below the burn-in
/// run — so the new default masks the burn-in fix without removing the
/// underlying problem.  Anchoring to the pre-#148 default keeps the test
/// guarding the burn-in mechanism rather than the (orthogonal) MH-step
/// improvement.
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

    // 1. With the burn-in (default), the recovered total BSV variance is in the
    //    right neighbourhood of the truth. The window is wide because the
    //    ω_CL/ω_V split is only weakly identified by one observation per
    //    subject — the *total* variance is the robust quantity. This assertion
    //    fails on the pre-fix code, where Ω collapses.
    assert!(
        trace_burnin > 0.5 * TRUE_TRACE && trace_burnin < 1.6 * TRUE_TRACE,
        "burn-in run did not recover trace(Ω): got {trace_burnin:.4}, truth {TRUE_TRACE:.3}"
    );

    // 2. Disabling the burn-in collapses Ω: it recovers substantially less
    //    variance than the burn-in run. This is the contrast the burn-in fixes.
    assert!(
        trace_no_burnin < 0.6 * trace_burnin,
        "expected no-burn-in run to collapse Ω relative to burn-in run: \
         no_burnin={trace_no_burnin:.4}, burnin={trace_burnin:.4}"
    );
}
