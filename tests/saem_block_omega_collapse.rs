//! Slow convergence test guarding against the SAEM block-Ω rank-1 collapse.
//!
//! Run with:
//!
//!   cargo test --features slow-tests --test saem_block_omega_collapse
//!
//! ## What this guards
//!
//! For a `block_omega` (correlated) random-effects block, the pre-fix SAEM drove
//! every off-diagonal correlation toward ±1 and collapsed one variance toward
//! zero — a near rank-1 Ω. The mechanism: the eta MH proposal is preconditioned
//! by `chol(Ω)` and, during the γ=1 exploration phase, the single-draw Ω M-step
//! overwrote Ω with one warm-started, not-yet-equilibrated MCMC snapshot each
//! iteration. For a correlated block that snapshot is biased toward the chain's
//! current correlation; the bias feeds back through the proposal Cholesky and
//! runs away to a degenerate Ω (an absorbing state, since a degenerate `chol(Ω)`
//! can no longer propose moves that break the collinearity). FOCEI is immune
//! because it maximises the (linearised) marginal likelihood directly, with no
//! sampler in the loop.
//!
//! Two fixes work together and are both exercised by this test at their defaults:
//!   1. a componentwise (single-coordinate) MH kernel (Kuhn & Lavielle 2004),
//!   2. a damped Robbins-Monro step for the Ω sufficient statistic so a single
//!      draw cannot overwrite a correlated Ω during exploration.
//!
//! The model mirrors the real UVM 2-cpt case where the defect was found: a 3-eta
//! block on (CL, V1, V2) with V2 only weakly identified. Data are fully
//! synthetic; no proprietary data is used.
//!
//! On the pre-fix code this test fails on both assertions (max |corr| → ~0.99,
//! V2 variance → near zero). Verified manually by raising `OMEGA_SA_MAX_STEP`
//! and disabling the componentwise kernel.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, OmegaMatrix, Population, Subject};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};
use nalgebra::DMatrix;
use std::collections::HashMap;

const MODEL: &str = r#"
[parameters]
  theta TVCL(3.0,   0.1, 150.0)
  theta TVV1(50.0,  1.0, 2000.0)
  theta TVQ(2.0,    0.1, 50.0)
  theta TVV2(100.0, 1.0, 2500.0)
  # Start uncorrelated (off-diagonals 0), exactly as the UVM model does — the
  # collapse must not be seeded by the initial Ω.
  block_omega (ETA_CL, ETA_V1, ETA_V2) = [0.2, 0.0, 0.2, 0.0, 0.0, 0.2]
  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// True Ω: variances on the diagonal, with a strong CL~V1 correlation and a
// moderate V1~V2 correlation — both well inside the non-degenerate region, so a
// faithful estimator must NOT report near-±1 correlations.
const TRUE_VAR: [f64; 3] = [0.07, 0.09, 0.50];
const TRUE_CORR_CL_V1: f64 = 0.5;
const TRUE_CORR_CL_V2: f64 = 0.1;
const TRUE_CORR_V1_V2: f64 = 0.4;

fn true_omega() -> OmegaMatrix {
    let s = TRUE_VAR.map(f64::sqrt);
    let mut m = DMatrix::zeros(3, 3);
    for i in 0..3 {
        m[(i, i)] = TRUE_VAR[i];
    }
    let set = |m: &mut DMatrix<f64>, i: usize, j: usize, c: f64| {
        let v = c * s[i] * s[j];
        m[(i, j)] = v;
        m[(j, i)] = v;
    };
    set(&mut m, 0, 1, TRUE_CORR_CL_V1);
    set(&mut m, 0, 2, TRUE_CORR_CL_V2);
    set(&mut m, 1, 2, TRUE_CORR_V1_V2);
    OmegaMatrix::from_matrix(
        m,
        vec!["ETA_CL".into(), "ETA_V1".into(), "ETA_V2".into()],
        false,
    )
}

/// Build a synthetic population: `n` subjects, one 2 h IV infusion each, with a
/// rich-enough observation grid (cycled per subject) to identify the central
/// disposition while leaving V2 only weakly informed.
fn template_population(n: usize) -> Population {
    let grid = [0.5_f64, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0];
    let subjects: Vec<Subject> = (1..=n)
        .map(|i| {
            // 5 observations per subject, phase-shifted across the grid.
            let times: Vec<f64> = (0..5).map(|j| grid[(i + j) % grid.len()]).collect();
            let n_obs = times.len();
            Subject {
                id: format!("{i}"),
                // amt=1000, rate=500 → a 2 h infusion into the central compartment.
                doses: vec![DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0)],
                obs_times: times,
                observations: vec![0.0; n_obs],
                obs_cmts: vec![1; n_obs],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; n_obs],
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

fn simulate_into(model: &ferx_core::types::CompiledModel, template: &Population) -> Population {
    let mut truth = model.default_params.clone();
    truth.theta = vec![3.0, 50.0, 2.0, 100.0];
    truth.omega = true_omega();
    truth.sigma.values = vec![0.15];

    let sim = simulate_with_seed(model, template, &truth, 1, 20260607);

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

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_block_omega_does_not_collapse_to_rank1() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let template = template_population(120);
    let population = simulate_into(&model, &template);

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 400;
    opts.saem_n_convergence = 400;
    opts.saem_seed = Some(12345);
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &population, &model.default_params, &opts).expect("SAEM fit succeeds");

    let om = &result.omega;
    let var = [om[(0, 0)], om[(1, 1)], om[(2, 2)]];
    let corr = |i: usize, j: usize| om[(i, j)] / (om[(i, i)] * om[(j, j)]).sqrt();
    let c = [corr(0, 1), corr(0, 2), corr(1, 2)];

    eprintln!(
        "SAEM block Ω: var=[{:.4}, {:.4}, {:.4}]  corr=[CL,V1={:.3} CL,V2={:.3} V1,V2={:.3}]  (truth var={:?}, corr=[{:.2},{:.2},{:.2}])",
        var[0], var[1], var[2], c[0], c[1], c[2], TRUE_VAR, TRUE_CORR_CL_V1, TRUE_CORR_CL_V2, TRUE_CORR_V1_V2,
    );

    // 1. No rank-1 collapse: every off-diagonal correlation stays well away from
    //    ±1. The pre-fix code drives all three to ~0.99 here.
    let max_abs_corr = c.iter().cloned().fold(0.0_f64, |a, x| a.max(x.abs()));
    assert!(
        max_abs_corr < 0.9,
        "block Ω collapsed toward rank-1: |corr| max = {max_abs_corr:.3}, corr = {c:?}"
    );

    // 2. No variance collapse: each variance stays within a factor of the truth.
    //    The pre-fix code collapses the weakly-identified V2 toward zero
    //    (ratio < 0.05).
    for k in 0..3 {
        let ratio = var[k] / TRUE_VAR[k];
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "Ω variance {k} not recovered: got {:.4}, truth {:.4} (ratio {ratio:.3})",
            var[k],
            TRUE_VAR[k]
        );
    }

    // 3. The block structure is actually estimated, not zeroed out: at least one
    //    genuine positive off-diagonal correlation survives (the pre-fix failure
    //    mode is the opposite extreme — all three pinned near +1). Exact
    //    magnitude recovery is not asserted: at this subject count the
    //    individual ω_ij are only weakly identified, so the robust guarantee is
    //    "non-degenerate and structured", not "matches the simulation truth".
    let max_corr = c.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        max_corr > 0.2,
        "block Ω off-diagonals collapsed toward zero (no correlation structure recovered): corr = {c:?}"
    );
}
