//! `[covariate_model]` ≡ the classical expression it desugars to (#1111).
//!
//! The block is sugar: it emits nothing the hand-written path could not already
//! emit. That is also what makes it validatable without a new NONMEM run — the
//! classical covariate models are already NONMEM-anchored, so the anchor for
//! this feature is that the two forms are the *same model*, to the last bit.
//!
//! Both directions are checked on real data (`data/two_cpt_oral_cov.csv`, which
//! carries `WT` and `CRCL`):
//!
//! - `predict()` agrees **bit-for-bit** at a fixed parameter vector, and
//! - a short `fit()` reports a bit-identical OFV from the same start.
//!
//! Bit-for-bit rather than to-tolerance is deliberate: the desugar emits the
//! same expression tree in the same multiplication order, so any difference at
//! all would mean the generated text is not the model the author wrote.

use std::path::Path;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;
use ferx_core::{fit, predict, read_nonmem_csv};

const DATA: &str = "data/two_cpt_oral_cov.csv";

/// Everything the two models share. The θ that the block will generate are
/// declared here in both arms — with the same name, init and bounds — so the
/// two parameter vectors are elementwise identical and "the same start" is not
/// an approximation.
const HEAD: &str = r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)

  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15

  sigma PROP_ERR ~ 0.04 (sd)
";

const TAIL: &str = r"
[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";

/// The covariate model written by hand, the way it always has been.
fn classical() -> String {
    format!(
        "{HEAD}
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_CL_WT * (CRCL / 100)^THETA_CL_CRCL * exp(ETA_CL)
  V1 = TVV1 * (WT / 70)^THETA_V1_WT * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[covariates]
  WT   continuous
  CRCL continuous
{THETA_DECLS}{TAIL}"
    )
}

/// …and the same model stated declaratively.
fn declarative() -> String {
    format!(
        "{HEAD}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[covariates]
  WT   continuous
  CRCL continuous

[covariate_model]
  CL ~ WT   power(center = 70)  => THETA_CL_WT(0.6, 0.01, 5.0)
  CL ~ CRCL power(center = 100) => THETA_CL_CRCL(0.3, 0.01, 5.0)
  V1 ~ WT   power(center = 70)  => THETA_V1_WT(0.9, 0.01, 5.0)
{TAIL}"
    )
}

/// The generated θ, declared explicitly in the classical arm. Appended after
/// the shared block so both arms order θ identically — the desugar appends its
/// generated declarations to the end of `[parameters]`.
const THETA_DECLS: &str = r"
[parameters]
  theta THETA_CL_WT(0.6, 0.01, 5.0)
  theta THETA_CL_CRCL(0.3, 0.01, 5.0)
  theta THETA_V1_WT(0.9, 0.01, 5.0)
";

#[test]
fn the_block_is_the_classical_model_it_desugars_to() {
    let hand = parse_full_model(&classical()).expect("classical model must parse");
    let block = parse_full_model(&declarative()).expect("block-declared model must parse");

    // The two θ vectors must line up name for name, or "the same parameter
    // vector" below would be comparing different models.
    assert_eq!(hand.model.theta_names, block.model.theta_names);
    assert_eq!(
        hand.model.default_params.theta,
        block.model.default_params.theta
    );

    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("covariate dataset must load");

    // 1. Predictions at a fixed parameter vector, bit for bit.
    let hand_pred = predict(&hand.model, &pop, &hand.model.default_params);
    let block_pred = predict(&block.model, &pop, &block.model.default_params);
    assert_eq!(hand_pred.len(), block_pred.len());
    for (h, b) in hand_pred.iter().zip(&block_pred) {
        assert_eq!(h.id, b.id);
        assert_eq!(h.time, b.time);
        assert_eq!(h.pred, b.pred, "PRED differs for subject {}", h.id);
    }

    // 2. …and the objective function after a couple of outer iterations from
    //    that same start. `maxiter = 2` keeps this a Tier-2 test: it exercises
    //    the fit path without running a convergence loop.
    // Tier-2: a couple of outer iterations, never a convergence loop.
    let opts = FitOptions {
        outer_maxiter: 2,
        ..FitOptions::default()
    };
    let hand_fit = fit(&hand.model, &pop, &hand.model.default_params, &opts)
        .expect("short classical fit must not error");
    let block_fit = fit(&block.model, &pop, &block.model.default_params, &opts)
        .expect("short block-declared fit must not error");
    assert_eq!(
        hand_fit.ofv, block_fit.ofv,
        "the block must be the same objective function as the expression it generates"
    );
    assert_eq!(hand_fit.theta, block_fit.theta);
}

#[test]
fn the_relation_table_is_echoed_on_the_fit_result() {
    // The point of the block: a search reads the covariate model back off the
    // result instead of regexing the `.ferx` file.
    let block = parse_full_model(&declarative()).expect("block-declared model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("covariate dataset must load");
    let opts = FitOptions {
        outer_maxiter: 2,
        ..FitOptions::default()
    };
    let result = fit(&block.model, &pop, &block.model.default_params, &opts)
        .expect("short block-declared fit must not error");

    let relations = &result.covariate_relations;
    assert_eq!(relations.len(), 3);
    assert_eq!(relations[0].parameter, "CL");
    assert_eq!(relations[0].covariate, "WT");
    assert_eq!(relations[0].form, "power");
    assert_eq!(relations[0].center, Some(70.0));
    assert_eq!(relations[0].thetas.len(), 1);
    assert_eq!(relations[0].thetas[0].name, "THETA_CL_WT");

    // Each echoed estimate is the θ the fit actually reports, joined by name —
    // that join is the work a caller would otherwise have to redo.
    for rel in relations {
        for theta in &rel.thetas {
            let idx = result
                .theta_names
                .iter()
                .position(|n| *n == theta.name)
                .expect("every echoed θ is a real θ of the fit");
            assert_eq!(theta.estimate, result.theta[idx]);
        }
    }
}

/// The same pair, fit to convergence: identical estimates and identical
/// standard errors.
///
/// The short fit above pins the objective; this pins the whole run, including
/// the covariance step — the level at which a difference in the *gradient* of
/// the generated expression, rather than in its value, would show up.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_two_forms_converge_to_the_same_estimates_and_ses() {
    let hand = parse_full_model(&classical()).expect("classical model must parse");
    let block = parse_full_model(&declarative()).expect("block-declared model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("covariate dataset must load");

    let opts = FitOptions {
        run_covariance_step: true,
        ..FitOptions::default()
    };
    let hand_fit = fit(&hand.model, &pop, &hand.model.default_params, &opts)
        .expect("classical fit must not error");
    let block_fit = fit(&block.model, &pop, &block.model.default_params, &opts)
        .expect("block-declared fit must not error");

    // Non-degeneracy first: two fits that both stalled at their start would
    // agree trivially and prove nothing about the generated expression.
    assert!(hand_fit.converged, "classical arm must actually converge");
    assert!(block_fit.converged, "block arm must actually converge");
    for name in ["THETA_CL_WT", "THETA_CL_CRCL", "THETA_V1_WT"] {
        let i = block_fit
            .theta_names
            .iter()
            .position(|n| n == name)
            .expect("generated θ is estimated");
        let init = block.model.default_params.theta[i];
        assert!(
            (block_fit.theta[i] - init).abs() > 1e-6,
            "{name} never moved from its initial estimate — the covariate effect \
             is not being estimated, and the comparison below would be vacuous"
        );
    }

    assert_eq!(hand_fit.ofv, block_fit.ofv);
    assert_eq!(hand_fit.theta, block_fit.theta);
    assert_eq!(hand_fit.omega, block_fit.omega);
    assert_eq!(hand_fit.sigma, block_fit.sigma);
    assert_eq!(hand_fit.se_theta, block_fit.se_theta);
    assert!(
        hand_fit.se_theta.is_some(),
        "the covariance step must have run — SEs are half of what this compares"
    );
}
