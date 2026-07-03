//! Integration tests for covariate-selected residual error models (issue #658):
//! an `if/else` in `[error_model]` that picks the residual error by a covariate
//! flag (a free-vs-total assay), the residual analogue of the Form C `[scaling]
//! y = <expr>` readout selector (#650).
//!
//! Fast tests exercise the public parse / `predict()` / `fit()` boundary on an
//! **analytical** PK model (per-CMT error is ODE-only; covariate selection is
//! not). The per-observation dispatch (`ErrorSpec::{obs_keys, obs_key,
//! variance_at, ...}`) and the parser are unit-tested in `src/`. The slow test
//! is the numerical validation: simulate with a known `σ_total ≠ σ_unbound`
//! split and recover it — which fails if dispatch routes rows to the wrong
//! sigma. NONMEM equivalence (`$ERROR` with `IF (FREE.EQ.0)`) is documented in
//! `docs/model-file/error-model.qmd`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, ErrorSpec, Population};
use ferx_core::{
    fit, predict, simulate_with_options, simulate_with_seed, EstimationMethod, FitOptions,
    SimulateOptions,
};
use std::collections::HashMap;

mod common;

/// Analytical 1-cpt IV model whose residual error is selected by a per-row
/// `FREE` flag: total (`FREE == 0`) rows carry a small proportional error,
/// unbound (`FREE != 0`) rows a larger one — the *same* concentration readout
/// (same CMT), two different residual magnitudes.
const FREE_TOTAL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_TOTAL   ~ 0.05 (sd)
  sigma PROP_UNBOUND ~ 0.30 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous

[fit_options]
  method = focei
";

/// `n_subj` subjects, each dosed 100 into CMT 1 and observed at five times with
/// two rows per time — a total (`FREE=0`) and an unbound (`FREE=1`) measurement
/// of the *same* concentration (same CMT=1). `observations` is a placeholder;
/// `simulate_with_seed` fills real values in the slow recovery test.
fn free_total_pop(n_subj: usize) -> Population {
    let obs_times: Vec<f64> = vec![0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0];
    let free_flags = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let subjects = (0..n_subj)
        .map(|i| {
            let mut s = common::subject(
                &format!("{}", i + 1),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times.clone(),
                vec![1.0; obs_times.len()],
                vec![1; obs_times.len()],
            );
            s.obs_covariates = free_flags
                .iter()
                .map(|&f| HashMap::from([("FREE".to_string(), f)]))
                .collect();
            s
        })
        .collect();
    Population {
        covariate_names: vec!["FREE".to_string()],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

#[test]
fn parses_to_selected_error_spec() {
    let model = parse_model_string(FREE_TOTAL).expect("free/total model must parse");
    match &model.error_spec {
        ErrorSpec::Selected { endpoints, .. } => {
            assert_eq!(endpoints.len(), 2, "two selected endpoints");
        }
        other => panic!("expected ErrorSpec::Selected, got {other:?}"),
    }
    // The selector covariate is a required data column.
    assert!(model.referenced_covariates.iter().any(|c| c == "FREE"));
}

/// The selection is covariate-only, so predictions are identical regardless of
/// the flag — the two co-temporal rows share one closed-form concentration.
#[test]
fn predictions_finite_and_flag_independent() {
    let model = parse_model_string(FREE_TOTAL).expect("model parses");
    let pop = free_total_pop(1);
    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .iter()
        .map(|p| p.pred)
        .collect();
    assert!(
        preds.iter().all(|v| v.is_finite()),
        "predictions must be finite, got {preds:?}"
    );
    // Rows 0 (FREE=0) and 1 (FREE=1) are the same time → same prediction.
    assert!((preds[0] - preds[1]).abs() < 1e-9);
}

/// A condition covariate absent from the data is a hard error (`E_MISSING_COVARIATE`),
/// never silently read as 0.
#[test]
fn missing_selector_covariate_is_rejected() {
    let model = parse_model_string(FREE_TOTAL).expect("model parses");
    let mut pop = free_total_pop(1);
    // Drop FREE from the data entirely.
    pop.covariate_names.clear();
    for s in pop.subjects.iter_mut() {
        s.obs_covariates.clear();
        s.covariates.clear();
    }
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let err = fit(&model, &pop, &model.default_params, &opts)
        .expect_err("missing selector covariate must be rejected");
    assert!(
        err.contains("FREE"),
        "error should name the missing covariate FREE: {err}"
    );
}

/// `simulate` must enforce the same selector-covariate presence check as `fit()`
/// (#658 review): a missing selector covariate would otherwise silently read as 0.0
/// and route every row to branch 0, applying the wrong residual variance with no
/// diagnostic. `simulate_with_options` (the path the R wrapper uses) returns an
/// `Err` naming the missing covariate.
#[test]
fn simulate_missing_selector_covariate_is_rejected() {
    let model = parse_model_string(FREE_TOTAL).expect("model parses");
    let mut pop = free_total_pop(1);
    // Drop FREE from the data entirely.
    pop.covariate_names.clear();
    for s in pop.subjects.iter_mut() {
        s.obs_covariates.clear();
        s.covariates.clear();
    }
    let err = simulate_with_options(
        &model,
        &pop,
        &model.default_params,
        1,
        &SimulateOptions::default(),
    )
    .expect_err("simulate must reject a missing selector covariate");
    assert!(
        err.contains("FREE"),
        "error should name the missing covariate FREE: {err}"
    );
}

/// A covariate-selected error model cannot be combined with an SDE `[diffusion]`
/// block (#658 review): the EKF measurement-noise path binds a single
/// representative error model and cannot switch per observation, so the
/// combination is rejected at parse time rather than silently scoring every row
/// against branch 0 in a release build.
#[test]
fn selected_error_model_with_sde_is_rejected_at_parse() {
    let sde_selected = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_TOTAL   ~ 0.05 (sd)
  sigma PROP_UNBOUND ~ 0.30 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[diffusion]
  central ~ 0.5

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous
";
    let err = parse_model_string(sde_selected)
        .err()
        .expect("covariate-selected error on an SDE model must be rejected");
    assert!(
        err.contains("SDE") || err.to_lowercase().contains("diffusion"),
        "error should cite the SDE restriction: {err}"
    );
}

/// Same free/total selector as `FREE_TOTAL`, but the two branch sigmas are
/// declared in one `block_sigma` so their residuals are *correlated* — the
/// total and unbound assays of one sample co-vary (#669). A co-temporal
/// total (FREE=0) / unbound (FREE=1) pair now resolves to different branches
/// and picks up the cross-branch off-diagonal in the dense residual `R`.
const FREE_TOTAL_BLOCK: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_TOTAL, PROP_UNBOUND) = [0.0025, 0.0075, 0.09]

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous

[fit_options]
  method = focei
";

/// #669: a covariate-selected `[error_model]` combined with `block_sigma`
/// correlated residuals now parses and fits. The parser records one
/// cross-branch correlation, and a bounded FOCEI fit runs to a finite OFV with
/// the dense residual `R` path active (the co-temporal total/unbound pair
/// carries the off-diagonal covariance).
#[test]
fn selected_error_with_block_sigma_fits() {
    let model = parse_model_string(FREE_TOTAL_BLOCK).expect("block_sigma + selected must parse");
    assert!(
        matches!(model.error_spec, ErrorSpec::Selected { .. }),
        "expected a Selected error spec"
    );
    assert_eq!(
        model.residual_correlations.len(),
        1,
        "one cross-branch residual correlation from the block_sigma off-diagonal"
    );

    // Simulate DVs at truth, then run a bounded FOCEI fit (Tier-2: no
    // convergence loop, just a finite OFV out of the dense-R path).
    let design = free_total_pop(8);
    let truth = model.default_params.clone();
    let sims = simulate_with_seed(&model, &design, &truth, 1, 669669);
    let mut pop = design.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sims
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value())
            .collect();
    }

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.interaction = true;
    opts.run_covariance_step = false;
    opts.outer_maxiter = 2;
    opts.verbose = false;

    let r = fit(&model, &pop, &model.default_params, &opts)
        .expect("block_sigma + selected FOCEI fit must run");
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);
}

/// End-to-end numerical validation: simulate with a known split
/// (`σ_total = 0.05`, `σ_unbound = 0.30`) and recover it from a neutral start
/// where both sigmas begin at 0.15. Recovery of the *ordered, separated* split
/// proves each row's residual is scored against the correct endpoint — if
/// dispatch mixed the rows, the two sigmas would collapse toward a common value.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn recovers_free_total_sigma_split() {
    let model = parse_model_string(FREE_TOTAL).expect("model parses");
    let design = free_total_pop(40);

    // Simulate DVs at the true parameters (σ_total = 0.05, σ_unbound = 0.30).
    let truth = model.default_params.clone();
    let sims = simulate_with_seed(&model, &design, &truth, 1, 658658);
    let mut pop = design.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sims
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value())
            .collect();
    }

    // Refit from a neutral start where both residual sigmas begin equal (0.15).
    let mut start = model.default_params.clone();
    for s in start.sigma.values.iter_mut() {
        *s = 0.15;
    }

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.interaction = true;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let r = fit(&model, &pop, &start, &opts).expect("free/total FOCEI fit must run");
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);

    let sig = |name: &str| -> f64 {
        let i = r
            .sigma_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("sigma {name} present"));
        r.sigma[i]
    };
    let total = sig("PROP_TOTAL");
    let unbound = sig("PROP_UNBOUND");

    // Ordered and separated (the whole point of per-row selection), and each in
    // a loose band around its truth. Bands are wide enough for the modest sample
    // yet exclude the collapsed / swapped solutions a dispatch bug would produce.
    assert!(
        unbound > total,
        "unbound sigma must exceed total sigma (got total={total:.4}, unbound={unbound:.4})"
    );
    assert!(
        (0.02..0.12).contains(&total),
        "PROP_TOTAL should recover near 0.05, got {total:.4}"
    );
    assert!(
        (0.18..0.45).contains(&unbound),
        "PROP_UNBOUND should recover near 0.30, got {unbound:.4}"
    );
}
