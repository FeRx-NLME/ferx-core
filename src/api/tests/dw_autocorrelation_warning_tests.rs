//! Exactly what the pooled-IWRES autocorrelation warning says (#1285, #1350
//! row 10a).
//!
//! The assertions are **exact string equality**, not `contains`. The sentence
//! removed in this change — `" For ODE models, SDE process noise may also
//! help."` — is a recommendation, so a `!msg.contains("SDE")` assertion would
//! pass on any rewording that still sends the user into `[diffusion]` for
//! residual autocorrelation, which the EKF cannot supply while it never
//! corrects the state mean with the observed data. Nothing tested the suffix at
//! all, which is how it outlived the filing of #1285.
//!
//! Classification is *not* re-asserted here: `crate::types::classify_warning`
//! keys on "autocorrelation" / "durbin", and `types_tests.rs` already pins both
//! of these messages to the `dw_autocorrelation` category. A second copy of
//! that assertion here would cover for the first one.
//!
//! The string tests below pin the **helper**. What `fit_inner` actually pushes
//! is a separate property, and pinning only the helper left the pre-PR
//! user-visible behaviour reachable: re-appending the sentence *at the call
//! site* killed none of them (measured on review of `1d816920`). That is what
//! `fit_pushes_exactly_the_helpers_message` below exists for.

use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

/// Positive autocorrelation, character for character.
///
/// Mutations that must redden this: re-appending the SDE sentence inside the
/// helper (the state of the tree before #1350 row 10a); rewording the three
/// remedies the warning does still name. It does **not** see a sentence
/// appended by a caller — `fit_pushes_exactly_the_helpers_message` is the test
/// for that.
#[test]
fn positive_autocorrelation_message_is_exact() {
    let msg = dw_autocorrelation_warning(1.20).expect("DW = 1.20 is below the 1.5 threshold");
    assert_eq!(
        msg,
        "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20). \
         Structural model may be missing dynamics. Consider a transit \
         absorption model, additional compartment, or IOV on ka/F."
    );
}

/// Negative autocorrelation, character for character. Untouched by #1285 — the
/// SDE suffix was only ever appended to the positive branch — and pinned here
/// so a future edit to the shared helper cannot quietly move it either.
#[test]
fn negative_autocorrelation_message_is_exact() {
    let msg = dw_autocorrelation_warning(2.80).expect("DW = 2.80 is above the 2.5 threshold");
    assert_eq!(
        msg,
        "Negative IWRES autocorrelation detected (Durbin-Watson = 2.80). \
         Possible over-parameterization or misspecified error model."
    );
}

/// The quiet band, edges included: `< 1.5` and `> 2.5` are strict, so 1.5 and
/// 2.5 themselves warn about nothing.
///
/// Mutation that must redden this: swapping the two thresholds (`< 2.5` /
/// `> 1.5`), which turns every DW in the band into a warning.
#[test]
fn no_warning_inside_the_band() {
    for dw in [1.5, 2.0, 2.5] {
        assert_eq!(
            dw_autocorrelation_warning(dw),
            None,
            "DW = {dw} is inside the quiet band"
        );
    }
}

/// `dw_statistic` is `NaN` when no subject has two finite IWRES values, and the
/// comparisons above are both false on `NaN` — but relying on that leaves the
/// guard deletable, so the non-finite cases are asserted directly.
#[test]
fn non_finite_dw_never_warns() {
    for dw in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            dw_autocorrelation_warning(dw),
            None,
            "a non-finite DW ({dw}) is a missing statistic, not a diagnosis"
        );
    }
}

/// A one-compartment ODE model whose fixed effects cannot follow a
/// two-compartment truth: `CL/V = 0.1` against a biexponential.
///
/// No `omega`, deliberately. With a random effect on `CL` the inner loop moves
/// each subject's elimination rate and absorbs part of the very drift this
/// fixture exists to create, which would make the Durbin-Watson value depend on
/// the EBE search rather than on the fixture.
fn one_cpt_model() -> CompiledModel {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
"#;
    parse_model_string(src).expect("parse")
}

/// Observations from a two-compartment truth, `100·(0.6·e^{-0.5t} + 0.4·e^{-0.05t})`.
/// Against the model's single `e^{-0.1t}` the residuals are a smooth U — strongly
/// positively autocorrelated, which is the `DW < 1.5` branch.
fn biexponential_subject(id: &str, scale: f64) -> Subject {
    let obs_times: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0];
    let observations: Vec<f64> = obs_times
        .iter()
        .map(|t| scale * 100.0 * (0.6 * (-0.5f64 * t).exp() + 0.4 * (-0.05f64 * t).exp()))
        .collect();
    let n = obs_times.len();
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// What `fit_inner` pushes is the helper's message, unmodified.
///
/// The four tests above pin `dw_autocorrelation_warning` in isolation, and that
/// left the pre-PR user-visible behaviour reachable through the *caller*:
/// re-appending `" For ODE models, SDE process noise may also help."` to `msg`
/// inside `fit_inner` passed the entire `ferx-core/ci` lib suite (measured on
/// review of `1d816920`). Nothing under `tests/` or `crates/` looked at the text
/// either. This is the wiring assertion that was missing — equality against the
/// helper's own return, so any caller-side edit fails it: a suffix, a prefix, or
/// a second `push` of a reworded copy.
///
/// The fixture is an **ODE** model on purpose. The removed suffix was gated on
/// `model.ode_spec.is_some()`, so an analytic model cannot observe its return,
/// and it was appended only below 1.5 — so the fixture has to land on that side
/// of the gate. Both facts are asserted rather than assumed: a fixture that
/// drifted into the quiet band would otherwise leave this test passing on an
/// empty search.
#[test]
fn fit_pushes_exactly_the_helpers_message() {
    let model = one_cpt_model();
    assert!(
        model.ode_spec.is_some(),
        "the removed suffix was gated on `ode_spec`; an analytic fixture cannot observe it"
    );

    let pop = Population {
        subjects: vec![
            biexponential_subject("1", 1.0),
            biexponential_subject("2", 1.1),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    // One outer iteration: the statistic comes from the post-fit residual pass,
    // and the misfit is structural, so there is nothing to gain by converging.
    let opts = FitOptions {
        outer_maxiter: 1,
        run_covariance_step: false,
        ..Default::default()
    };

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");

    assert!(
        result.dw_statistic.is_finite() && result.dw_statistic < 1.5,
        "fixture must land on the branch the SDE sentence was appended to; DW = {}",
        result.dw_statistic
    );

    let expected = dw_autocorrelation_warning(result.dw_statistic)
        .expect("a DW below 1.5 must produce a message");
    let emitted: Vec<&String> = result
        .warnings
        .iter()
        .filter(|w| w.contains("autocorrelation"))
        .collect();
    assert_eq!(
        emitted.len(),
        1,
        "expected exactly one autocorrelation warning, got {emitted:?}"
    );
    assert_eq!(
        *emitted[0], expected,
        "fit_inner must push the helper's message unmodified"
    );
}
