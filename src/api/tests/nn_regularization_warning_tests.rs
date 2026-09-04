//! The pre-run "regularization not applied" warning for `nn_l2` / `nn_smooth`.
//!
//! The parser's `unsupported_keys_warnings` catches a model-file user who sets
//! the keys under SAEM, but ferx-r sets the two `FitOptions` fields directly —
//! `user_set_keys` stays empty — so `fit()` has to check the final stage itself.
//! These pin that check: it fires only when a λ is positive, the model has a
//! `[covariate_nn]` block, and the chain's *final* stage is not FOCE-family.

use super::{applies_nn_regularization, nn_regularization_unapplied_warning};
use crate::parser::model_parser::parse_full_model;
use crate::types::{CompiledModel, EstimationMethod, FitOptions};

const DCM: &str = include_str!("../../../tests/fixtures/two_cpt_dcm_regularized.ferx");

fn dcm_model() -> CompiledModel {
    parse_full_model(DCM).expect("DCM fixture parses").model
}

/// A model with no `[covariate_nn]` block, for the "nothing to regularize" arm.
fn plain_model() -> CompiledModel {
    parse_full_model(
        r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("plain model parses")
    .model
}

#[test]
fn foce_family_applies_the_penalty_and_the_rest_does_not() {
    for m in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::Laplace,
        EstimationMethod::FoceGn,
        EstimationMethod::FoceGnHybrid,
    ] {
        assert!(applies_nn_regularization(m), "{m:?} must apply the penalty");
    }
    for m in [
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
        EstimationMethod::Bayes,
        EstimationMethod::Vi,
    ] {
        assert!(
            !applies_nn_regularization(m),
            "{m:?} must not apply the penalty"
        );
    }
}

#[test]
fn warns_only_for_a_regularized_nn_model_whose_final_stage_is_not_foce_family() {
    let model = dcm_model();
    let mut opts = FitOptions::default();

    // λ = 0: nothing to warn about, whatever the method.
    assert!(
        nn_regularization_unapplied_warning(&model, &opts, &[EstimationMethod::Saem]).is_none()
    );

    opts.nn_l2_lambda = 1e-2;
    // FOCE-family final stage: applied, no warning — including after a SAEM
    // stage, since the regularized stage is the one reported.
    for chain in [
        vec![EstimationMethod::FoceI],
        vec![EstimationMethod::FoceGn],
        vec![EstimationMethod::Saem, EstimationMethod::FoceI],
    ] {
        assert!(
            nn_regularization_unapplied_warning(&model, &opts, &chain).is_none(),
            "{chain:?} must not warn"
        );
    }
    // Non-FOCE final stage: warn, naming the option and the stage.
    let w = nn_regularization_unapplied_warning(&model, &opts, &[EstimationMethod::Saem])
        .expect("saem final stage must warn");
    assert!(w.contains("nn_l2 is set"), "{w}");
    assert!(w.contains("`SAEM`"), "{w}");
    // Both keys set: both named, plural verb.
    opts.nn_smooth_lambda = 1e-1;
    let w = nn_regularization_unapplied_warning(
        &model,
        &opts,
        &[EstimationMethod::FoceI, EstimationMethod::Imp],
    )
    .expect("imp final stage must warn even after a focei stage");
    assert!(w.contains("nn_l2 / nn_smooth are set"), "{w}");
    assert!(w.contains("`IMP`"), "{w}");

    // No `[covariate_nn]` block: the penalty is a no-op regardless of method,
    // so there is nothing to warn about.
    assert!(
        nn_regularization_unapplied_warning(&plain_model(), &opts, &[EstimationMethod::Saem])
            .is_none()
    );
}
