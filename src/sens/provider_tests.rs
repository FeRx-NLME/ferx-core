use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::pk::compute_predictions_with_tv;
use crate::types::{test_helpers, DoseEvent, Subject};
use std::collections::HashMap;

/// Regression: an analytic `pk one_cpt_ig` / `two_cpt_ig` model must take the
/// **exact `Dual2` analytic** gradient path, not the finite-difference fallback —
/// the whole point of the #790 closed form. This pins `slot_to_dim` covering the
/// IG `mat`/`cv2` slots (11/12); without those arms `analytical_supported_core`'s
/// `pk_indices.iter().all(slot_to_dim(s).is_some())` fails and the model silently
/// falls back to FD (equivalence/anchor OFV tests would still pass on FD, so only
/// this assertion catches it).
#[test]
fn ig_models_use_the_analytic_gradient_path() {
    const ONE_CPT_IG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  MAT = TVMAT
  CV2 = TVCV2
[structural_model]
  pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    const TWO_CPT_IG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(10.0, 0.1, 200.0)
  theta TVV2(100.0, 5.0, 1000.0)
  theta TVMAT(1.5, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  MAT = TVMAT
  CV2 = TVCV2
[structural_model]
  pk two_cpt_ig(cl=CL, v1=V1, q=Q, v2=V2, mat=MAT, cv2=CV2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    for (label, src) in [("one_cpt_ig", ONE_CPT_IG), ("two_cpt_ig", TWO_CPT_IG)] {
        let m = parse_model_string(src).unwrap_or_else(|e| panic!("[{label}] parse: {e}"));
        // The IG structural slots must be recognised as differentiable.
        assert!(
            slot_to_dim(PK_IDX_MAT).is_some() && slot_to_dim(PK_IDX_CV2).is_some(),
            "[{label}] slot_to_dim must map MAT/CV2"
        );
        assert!(
            analytical_supported(&m),
            "[{label}] must use the exact Dual2 analytic gradient path, not FD"
        );
    }
}

#[test]
fn analytic_outer_gradient_available_tracks_scope_and_fd() {
    // Analytical PK model with `gradient = auto` → analytic outer gradient.
    assert!(analytic_outer_gradient_available(
        &test_helpers::analytical_model(GradientMethod::Auto)
    ));
    // `gradient = fd` forces FD even for an analytical model.
    assert!(!analytic_outer_gradient_available(
        &test_helpers::analytical_model(GradientMethod::Fd)
    ));
    // An ODE model is outside the analytic scope.
    assert!(!analytic_outer_gradient_available(
        &test_helpers::ode_model(GradientMethod::Auto)
    ));
    // A closed-form `iiv_on_ruv` model is analytic (#474)…
    let mut ruv = test_helpers::analytical_model(GradientMethod::Auto);
    ruv.residual_error_eta = Some(0);
    assert!(analytic_outer_gradient_available(&ruv));
    // …and closed-form M3 BLOQ + `iiv_on_ruv` is now analytic too (#4c — the
    // censored × residual-eta cross-terms are assembled). The ODE M3 +
    // `iiv_on_ruv` combo is analytic as well (#623), so every combination is served.
    let mut ruv_m3 = test_helpers::analytical_model(GradientMethod::Auto);
    ruv_m3.residual_error_eta = Some(0);
    ruv_m3.bloq_method = crate::types::BloqMethod::M3;
    assert!(analytic_outer_gradient_available(&ruv_m3));
    // Correlated residual (`block_sigma`) is now analytic on the outer loop (#627):
    // the dense assembly reduces to the scalar path fed correlation-aware `(r,d,d2)`.
    // (Previously this predicate short-circuited to FD on any residual correlation.)
    let block_sigma = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 1.00]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse block_sigma model");
    assert!(!block_sigma.residual_correlations.is_empty());
    assert!(analytic_outer_gradient_available(&block_sigma));
}

/// The `TIME` built-in makes a structural parameter piecewise/time-varying, so
/// the subject routes through the event-driven per-event PK walk rather than
/// dose superposition. As of #486 every analytic provider seeds the per-event time
/// and serves TIME: **closed-form non-IOV** (`analytical_supported` /
/// `tvcov_analytical_supported`), **closed-form IOV** (`iov_analytical_supported`),
/// **non-IOV ODE** (`ode_analytical_supported`), and **ODE IOV** (`ode_iov_supported`,
/// via the per-event stacked walk) — each validated against FD of production
/// (`time_builtin_provider_matches_fd_of_production`,
/// `iov_time_builtin_provider_matches_fd_of_predict_iov`,
/// `ode_time_builtin_provider_matches_fd_of_production`,
/// `ode_iov_time_builtin_provider_matches_fd_of_predict_iov`).
///
/// `TIME` composes with an η-dependent `ExpressionScale` obs_scale too — the
/// event-driven walk applies the subject-static scale quotient post-walk (validated
/// in `expression_scale_on_event_walk_matches_fd_closed_form`,
/// `ode_expression_scale_on_event_walk_matches_production`, and
/// `ode_iov_time_expression_scale_matches_fd_of_predict_iov`). The direct
/// `pk(...=TIME)` mapping is now desugared into a synthetic `__ferx_pktime_*`
/// individual parameter and served analytically too (see the `ANALYTICAL_TIME_DIRECT`
/// block below and `time_builtin_direct_pk_mapping_matches_fd_of_production`). `TIME`
/// combined with an ODE `init(...)` baseline (#662) or a built-in input-rate forcing
/// (#643) is now analytic on the event-driven walk too — the old model-level declines
/// were stale once those features landed (validated by `ode_time_builtin_with_init_*`
/// and `ode_time_builtin_with_first_order_*`); this test now pins the analytic route for
/// `TIME + init(...)` at the model gate. The pre-existing scale fallbacks (LTBS +
/// `ExpressionScale`; closed-form IOV + any scaling) are unchanged and independent of
/// `TIME`. The non-`TIME` twin of each model must stay supported, proving the guards
/// are specific (#486 / #610).
#[test]
fn time_builtin_indiv_params_analytic_routes() {
    // Analytical 1-cpt IV: a `$PK IF(TIME...)`-style switch on CL.
    const ANALYTICAL_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    const ANALYTICAL_NO_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    // IOV (n_kappa > 0) 1-cpt oral with the same switch.
    const IOV_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 24.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;
    // ODE 1-cpt with the switch in the individual parameters (so the
    // uses_time_builtin flag is set, not merely the ODE-RHS clock).
    const ODE_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    const ODE_NO_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let uses_time = crate::parser::model_parser::compiled_model_uses_time_builtin;
    let ode_supported = crate::sens::ode_provider::ode_analytical_supported;

    let ana_t = parse_model_string(ANALYTICAL_TIME).expect("parses analytical TIME");
    let ana_n = parse_model_string(ANALYTICAL_NO_TIME).expect("parses analytical control");
    assert!(
        uses_time(&ana_t),
        "TIME switch sets the uses_time_builtin flag"
    );
    assert!(!uses_time(&ana_n), "control model must not set the flag");
    // Closed-form non-IOV now serves TIME via the per-event walk (#486): the
    // model is admitted and routes through `tvcov_analytical_supported`.
    assert!(
        analytical_supported(&ana_t),
        "closed-form non-IOV now admits TIME (routed to the per-event walk)"
    );
    assert!(
        tvcov_analytical_supported(&ana_t),
        "TIME routes through the TV-cov event-driven walk"
    );
    assert!(
        analytical_supported(&ana_n),
        "non-TIME twin stays analytic (guard is specific)"
    );

    let iov_t = parse_model_string(IOV_TIME).expect("parses IOV TIME");
    let iov_n = parse_model_string(WARFARIN_IOV).expect("parses IOV control");
    assert!(
        iov_t.n_kappa > 0 && iov_n.n_kappa > 0,
        "both IOV models carry a kappa"
    );
    // Closed-form IOV now serves TIME (per-event stacked seeding in
    // `build_iov_sources`, #486).
    assert!(
        iov_analytical_supported(&iov_t),
        "closed-form IOV now serves TIME via the per-event walk"
    );
    assert!(
        iov_analytical_supported(&iov_n),
        "non-TIME IOV twin stays analytic"
    );

    let ode_t = parse_model_string(ODE_TIME).expect("parses ODE TIME");
    let ode_n = parse_model_string(ODE_NO_TIME).expect("parses ODE control");
    assert!(
        uses_time(&ode_t),
        "ODE indiv-param TIME switch sets the flag"
    );
    // Non-IOV ODE now serves TIME via the event-driven TV-cov walk (#486).
    assert!(
        ode_supported(&ode_t),
        "non-IOV ODE now serves indiv-param TIME via the per-event walk"
    );
    assert!(ode_supported(&ode_n), "non-TIME ODE twin stays analytic");

    // ODE **IOV** now serves TIME via the per-event stacked walk (#486). Build an
    // ODE + kappa + TIME model to pin `ode_iov_supported == true`.
    const ODE_IOV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;
    let ode_iov_t = parse_model_string(ODE_IOV_TIME).expect("parses ODE IOV TIME");
    assert!(ode_iov_t.n_kappa > 0 && uses_time(&ode_iov_t));
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&ode_iov_t),
        "ODE IOV now serves TIME via the per-event stacked walk (#486)"
    );

    // #486: a TIME + `init(...)` ODE model is now analytic at the model level. TIME forces
    // the event-driven walk (`integrate_tvcov_g`), which now seeds a non-zero `init(...)`
    // state (#662) alongside the per-event TIME seeding (#637) — so the model reports
    // analytic, not FD. (The old #637 round-2 decline assumed the walk seeded compartments
    // at zero, which was true before #662.)
    const ODE_TIME_INIT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(5.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = 1000.0 / V
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let ode_time_init = parse_model_string(ODE_TIME_INIT).expect("parses ODE TIME init");
    assert!(uses_time(&ode_time_init) && ode_time_init.ode_spec.is_some());
    assert!(
        ode_supported(&ode_time_init),
        "TIME + init(...) ODE is analytic on the event-driven walk now (#486)"
    );
    assert!(
        analytic_outer_gradient_available(&ode_time_init),
        "TIME + init(...) ODE outer route is analytic now (#486)"
    );

    // Direct `pk(...=TIME)` mapping (not an `[individual_parameters]` statement): the
    // parser now desugars the `=TIME` binding into a synthetic
    // `__ferx_pktime_<slot> = TIME` individual parameter (mirroring the #631 Form-C
    // readout desugaring), so the mapped slot enters the program's `pk_slots` and
    // rides the same per-event analytic walk as an `[individual_parameters]` TIME
    // switch — no longer FD (#486 direct-mapping follow-up). Use a 2-cpt `q=TIME`
    // mapping so the mapped slot is not a denominator (a `v=TIME` model divides by
    // `V = 0` at the `t = 0` dose — a user degeneracy, not a gate concern).
    const ANALYTICAL_TIME_DIRECT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=TIME, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let direct = parse_model_string(ANALYTICAL_TIME_DIRECT).expect("parses direct pk=TIME");
    assert!(
        uses_time(&direct),
        "direct pk(...=TIME) mapping sets the uses_time_builtin flag"
    );
    assert!(
        analytical_supported(&direct) && tvcov_analytical_supported(&direct),
        "direct pk(...=TIME) mapping is now served by the per-event analytic walk (desugared)"
    );
    assert!(
        analytic_outer_gradient_available(&direct),
        "direct pk(...=TIME) outer gradient route is now analytic"
    );
}

/// The closed-form non-IOV provider's exact value/∂η/∂²η/∂θ/∂²η∂θ for a model
/// whose structural parameter reads the `TIME` built-in must match central
/// finite differences of the production predictor `compute_predictions_with_tv`
/// (the independent f64 event-driven path that threads the same per-event TIME).
/// No TV covariates are present — the subject is routed to the per-event walk
/// purely by `uses_time_builtin` (#486 / #610). Covers a `$PK IF(TIME...)` switch
/// (1-cpt IV, 2-cpt IV) and a continuous `TVCL + c·TIME` term (1-cpt oral).
#[test]
fn time_builtin_provider_matches_fd_of_production() {
    // (a) 1-cpt IV: CL switches at TIME = 45 (NONMEM `IF (TIME.GE.45) CL=...`).
    const ONECPT_IV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(6.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    // (b) 1-cpt oral: CL varies continuously with TIME (`TVCL + 0.05·TIME`), so
    // both ∂CL/∂TVCL and the TIME-scaled ∂CL/∂ETA_CL are exercised per event.
    const ONECPT_ORAL_TIME: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.5, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = (TVCL + 0.05 * TIME) * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    // (c) 2-cpt IV: same TIME switch on CL, widening the dual to M = 7.
    const TWOCPT_IV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(6.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(5.0, 0.5, 50.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let iv_bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
    // Observations straddle the TIME = 45 switch so both arms of the `if` drive
    // at least one prediction (the switch is data-driven, so it is fixed under
    // the η/θ perturbation — the prediction is smooth in η/θ within each arm).
    let straddle = [10.0, 30.0, 50.0, 70.0, 90.0];

    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        {
            let m = parse_model_string(ONECPT_IV_TIME).expect("parse 1cpt iv TIME");
            let s = subject_with_doses_and_resets(vec![iv_bolus(0.0)], &straddle, Vec::new());
            (m, s, vec![10.0, 6.0, 50.0], vec![0.15, -0.10])
        },
        {
            let m = parse_model_string(ONECPT_ORAL_TIME).expect("parse 1cpt oral TIME");
            let s = subject_with_doses_and_resets(
                vec![iv_bolus(0.0)],
                &[1.0, 4.0, 10.0, 24.0, 48.0],
                Vec::new(),
            );
            (m, s, vec![1.0, 10.0, 1.5], vec![0.15, -0.10, 0.20])
        },
        {
            let m = parse_model_string(TWOCPT_IV_TIME).expect("parse 2cpt iv TIME");
            let s = subject_with_doses_and_resets(vec![iv_bolus(0.0)], &straddle, Vec::new());
            (m, s, vec![10.0, 6.0, 50.0, 5.0, 100.0], vec![0.15, -0.10])
        },
    ];

    for (m, s, theta, eta) in &cases {
        assert!(
            crate::parser::model_parser::compiled_model_uses_time_builtin(m),
            "fixture must read the TIME built-in"
        );
        assert!(
            !s.has_tv_covariates(),
            "fixture must stay on the no-TV path (routed purely by uses_time_builtin)"
        );
        assert!(
            tvcov_analytical_supported(m),
            "TIME model must be provider-supported via the per-event walk"
        );
        assert!(
            subject_sensitivities(m, s, theta, eta).is_some(),
            "TIME subject must take the analytic per-event provider"
        );
        check_full_provider_vs_fd(m, s, theta, eta);
    }
}

/// #486 (PR #665 review): a `TIME`-built-in structural parameter combined with
/// **LTBS** (`log(DV) ~ additive(...)`). Removing the `log_transform` bail from
/// [`tvcov_analytical_supported`] flips this combination from FD to the analytic
/// per-event walk, and `analytical_supported` short-circuits every
/// `uses_time_builtin` model to that gate — so the outer FOCE/FOCEI population
/// gradient is now served by the event-driven walk plus the post-walk `ln(f)`
/// transform ([`apply_ltbs_transform_outer`]). The other LTBS + TV-cov coverage
/// (`ltbs_tvcov_outer_matches_production`) uses a real time-varying covariate;
/// this test pins the *TIME-built-in* route, where the piecewise parameter is
/// seeded per event by [`ModelTimeGuard`] rather than by a covariate snapshot,
/// so the log transform must compose with that per-event seeding. Covers the
/// same three fixture shapes as [`time_builtin_provider_matches_fd_of_production`]
/// (IV switch, continuous `TVCL + c·TIME`, and a switch under an
/// `ExpressionScale obs_scale` to exercise scale-then-log `ln(f/s)`), each on the
/// no-TV path (routed purely by `uses_time_builtin`).
#[test]
fn ltbs_time_builtin_outer_matches_fd_of_production() {
    // (a) 1-cpt IV: CL switches at TIME = 45, LTBS error.
    const ONECPT_IV_TIME_LTBS: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(6.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma ADD_ERR ~ 0.04 (sd)
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  log(DV) ~ additive(ADD_ERR)
"#;
    // (b) 1-cpt oral: CL varies continuously with TIME, LTBS error.
    const ONECPT_ORAL_TIME_LTBS: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.5, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.10
  sigma ADD_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = (TVCL + 0.05 * TIME) * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  log(DV) ~ additive(ADD_ERR)
"#;
    // (c) 1-cpt oral: TIME switch on CL under an `ExpressionScale obs_scale`,
    // LTBS error — exercises the scale-then-log order `ln(f/s)` on the TIME route.
    const ONECPT_ORAL_TIME_SCALED_LTBS: &str = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVCL_LATE(0.6, 0.1, 50.0)
  theta TVV(10.0, 1.0, 200.0)
  theta TVKA(1.5, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.10
  sigma ADD_ERR ~ 0.04 (sd)
[individual_parameters]
  if (TIME > 24.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = 1000 / V
[error_model]
  log(DV) ~ additive(ADD_ERR)
"#;
    let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
    // Observations straddle the switch so both arms of the `if` drive at least
    // one prediction (the switch is data-driven, so it is fixed under the η/θ
    // perturbation — the prediction is smooth in η/θ within each arm).
    let straddle = [10.0, 30.0, 50.0, 70.0, 90.0];

    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        {
            let m = parse_model_string(ONECPT_IV_TIME_LTBS).expect("parse 1cpt iv TIME LTBS");
            let s = subject_with_doses_and_resets(vec![bolus(0.0)], &straddle, Vec::new());
            (m, s, vec![10.0, 6.0, 50.0], vec![0.15, -0.10])
        },
        {
            let m = parse_model_string(ONECPT_ORAL_TIME_LTBS).expect("parse 1cpt oral TIME LTBS");
            let s = subject_with_doses_and_resets(
                vec![bolus(0.0)],
                &[1.0, 4.0, 10.0, 24.0, 48.0],
                Vec::new(),
            );
            (m, s, vec![1.0, 10.0, 1.5], vec![0.15, -0.10, 0.20])
        },
        {
            let m = parse_model_string(ONECPT_ORAL_TIME_SCALED_LTBS)
                .expect("parse 1cpt oral TIME scaled LTBS");
            let s = subject_with_doses_and_resets(
                vec![bolus(0.0)],
                &[1.0, 4.0, 10.0, 24.0, 48.0],
                Vec::new(),
            );
            (m, s, vec![1.0, 0.6, 10.0, 1.5], vec![0.15, -0.10, 0.20])
        },
    ];

    for (m, s, theta, eta) in &cases {
        assert!(
            crate::parser::model_parser::compiled_model_uses_time_builtin(m),
            "fixture must read the TIME built-in"
        );
        assert!(m.log_transform, "fixture must be LTBS");
        assert!(
            !s.has_tv_covariates(),
            "fixture must stay on the no-TV path (routed purely by uses_time_builtin)"
        );
        assert!(
            tvcov_analytical_supported(m),
            "TIME + LTBS must be on the analytic OUTER path via the per-event walk (#486)"
        );
        // Outer analytic vs FD of the log-scale production predictor.
        check_full_provider_vs_fd(m, s, theta, eta);
        // Inner TIME + LTBS is now analytic too (Tier-1 follow-up — the event-driven
        // walk applies the `ln f` jet): inner η-gradient must equal the outer η-block.
        let full = subject_sensitivities(m, s, theta, eta).expect("outer");
        let light = subject_eta_grad(m, s, theta, eta).expect("inner TIME + LTBS");
        assert_eq!(full.obs.len(), light.len());
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-12);
            for k in 0..m.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-10,
                    epsilon = 1e-12
                );
            }
        }
    }
}

/// #486: the light `Dual1` inner η-gradient must equal the full `Dual2` outer
/// `df_deta` (η-block) for a `TIME`-built-in subject — both run the same
/// event-driven walk, and the outer is FD-validated by
/// [`time_builtin_provider_matches_fd_of_production`].
#[test]
fn time_builtin_eta_grad_matches_full() {
    const ONECPT_IV_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVCL_LATE(6.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(ONECPT_IV_TIME).expect("parse 1cpt iv TIME");
    let s = subject_with_doses_and_resets(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[10.0, 30.0, 50.0, 70.0, 90.0],
        Vec::new(),
    );
    let theta = [10.0, 6.0, 50.0];
    let eta = [0.15, -0.10];
    let full = subject_sensitivities(&m, &s, &theta, &eta).expect("full provider");
    let light = subject_eta_grad(&m, &s, &theta, &eta).expect("light inner provider");
    assert_eq!(full.obs.len(), light.len());
    for (o, g) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-12, epsilon = 1e-12);
        for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-10, epsilon = 1e-12);
        }
    }
}

/// #486: a **direct `pk(...=TIME)` structural mapping** (here `q=TIME` on a 2-cpt IV,
/// so the mapped slot is not a denominator — `v=TIME` would divide by `V = 0` at the
/// `t = 0` dose). The parser desugars it into a synthetic `__ferx_pktime_q = TIME`
/// individual parameter, so `Q` = event time per event (`∂Q/∂θ = ∂Q/∂η = 0`) while
/// `CL`/`V1`'s derivatives stay exact. value + ∂η + ∂²η + ∂θ + ∂²η∂θ must match central
/// FD of production (which threads the same per-event TIME through the desugared param).
#[test]
fn time_builtin_direct_pk_mapping_matches_fd_of_production() {
    const DIRECT_Q_TIME: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=TIME, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(DIRECT_Q_TIME).expect("parse direct q=TIME");
    assert!(crate::parser::model_parser::compiled_model_uses_time_builtin(&m));
    assert!(
        tvcov_analytical_supported(&m),
        "direct pk(q=TIME) desugars to an indiv-param → analytic"
    );
    let s = subject_with_doses_and_resets(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 4.0, 10.0, 24.0, 48.0],
        Vec::new(),
    );
    assert!(!s.has_tv_covariates());
    check_full_provider_vs_fd(&m, &s, &[10.0, 50.0, 100.0], &[0.10, -0.05]);
}

/// #486: the direct `pk(q=TIME)` desugaring must be *exactly* the explicit
/// `[individual_parameters] QT = TIME; pk(q=QT)` form — a pure rename into a synthetic
/// parameter. This ties the direct mapping to the `[individual_parameters]` TIME path
/// that #637 validated live against NONMEM (`METHOD=1 INTER`, ~5 sig figs), so the
/// direct form inherits that numerical validation by transitivity. All sensitivity
/// outputs (value + ∂η + ∂θ + 2nd-order) must agree to 1e-12.
#[test]
fn time_builtin_direct_pk_mapping_equivalent_to_explicit_indiv_param() {
    const DIRECT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=TIME, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    const EXPLICIT: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  V2 = TVV2
  QT = TIME
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=QT, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m_direct = parse_model_string(DIRECT).expect("parse direct");
    let m_explicit = parse_model_string(EXPLICIT).expect("parse explicit");
    let s = subject_with_doses_and_resets(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 4.0, 10.0, 24.0, 48.0],
        Vec::new(),
    );
    let theta = [10.0, 50.0, 100.0];
    let eta = [0.10, -0.05];
    let sd = subject_sensitivities(&m_direct, &s, &theta, &eta).expect("direct sens");
    let se = subject_sensitivities(&m_explicit, &s, &theta, &eta).expect("explicit sens");
    assert_eq!(sd.obs.len(), se.obs.len());
    for (a, b) in sd.obs.iter().zip(se.obs.iter()) {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-12, epsilon = 1e-13);
        for (x, y) in a.df_deta.iter().zip(b.df_deta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
        }
        for (x, y) in a.df_dtheta.iter().zip(b.df_dtheta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
        }
        for (x, y) in a.d2f_deta2.iter().zip(b.d2f_deta2.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
        }
        // The mixed η-θ second derivative feeds the FOCEI Laplace `log|H̃|`
        // θ-gradient this work targets, so it must match too.
        for (x, y) in a.d2f_deta_dtheta.iter().zip(b.d2f_deta_dtheta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-12, epsilon = 1e-13);
        }
    }
}

/// #486: an η-dependent `ExpressionScale` `obs_scale` on the **event-driven walk** —
/// for a `TIME` switch AND a time-varying covariate. The walk now applies the same
/// subject-static scale quotient (`apply_expression_scale_outer` / `_inner_dispatch`)
/// the dose-superposition path uses, so value/∂η/∂²η/∂θ/∂²η∂θ must match FD of
/// production and the light inner must track the outer η-block. (Closes the TV-cov +
/// expression-scale gap, not just the TIME one.)
#[test]
fn expression_scale_on_event_walk_matches_fd_closed_form() {
    const IV_TIME_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.05, 50.0)
  theta TVCL_LATE(0.5, 0.05, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  if (TIME > 45.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    const IV_TVCOV_SCALED: &str = r#"
[parameters]
  theta TVCL(1.0, 0.05, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[scaling]
  obs_scale = 1000 / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let check_parity = |m: &CompiledModel, s: &Subject, theta: &[f64], eta: &[f64]| {
        assert!(
            matches!(m.scaling, ScalingSpec::ExpressionScale { .. }),
            "fixture must carry an ExpressionScale obs_scale"
        );
        assert!(
            tvcov_analytical_supported(m),
            "ExpressionScale event-walk model must be provider-supported"
        );
        check_full_provider_vs_fd(m, s, theta, eta);
        // Light inner η-gradient must track the outer η-block (both apply the scale).
        let full = subject_sensitivities(m, s, theta, eta).expect("outer");
        let light = subject_eta_grad(m, s, theta, eta).expect("inner");
        for (o, g) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(o.f, g.f, max_relative = 1e-12, epsilon = 1e-12);
            for (a, b) in o.df_deta.iter().zip(g.df_deta.iter()) {
                approx::assert_relative_eq!(a, b, max_relative = 1e-10, epsilon = 1e-12);
            }
        }
    };
    // TIME switch + obs_scale (no TV covariates → routed by uses_time_builtin).
    let m_time = parse_model_string(IV_TIME_SCALED).expect("parse IV TIME scaled");
    let s_time = subject_with_doses_and_resets(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[10.0, 30.0, 50.0, 70.0, 90.0],
        Vec::new(),
    );
    check_parity(&m_time, &s_time, &[1.0, 0.5, 20.0], &[0.15, -0.10]);
    // Time-varying covariate + obs_scale (the broader gap this closes).
    let m_tv = parse_model_string(IV_TVCOV_SCALED).expect("parse IV tvcov scaled");
    let s_tv = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[70.0],
        &[1.0, 2.0, 4.0, 8.0, 24.0],
        &[70.0, 72.0, 80.0, 85.0, 90.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    assert!(s_tv.has_tv_covariates());
    check_parity(&m_tv, &s_tv, &[1.0, 20.0, 0.75], &[0.15, -0.10]);
}

/// A TTE (`[event_model]`) objective has no analytic outer gradient (the
/// provider only covers the structural PK/PD model), so the predicate must be
/// `false` even with `gradient = auto` — and `resolve_auto` must therefore
/// pick the derivative-free Bobyqa, not a gradient-based optimizer that would
/// stall on a gradient TTE cannot supply (#490 auto-optimizer × TTE).
#[cfg(feature = "survival")]
#[test]
fn analytic_outer_gradient_unavailable_for_tte() {
    use crate::types::Optimizer;
    const TTE: &str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA ~ 0.09

[event_model e]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA)
";
    let m = parse_model_string(TTE).expect("TTE model must parse");
    assert!(m.has_tte(), "model must register a TTE endpoint");
    assert!(
        !analytic_outer_gradient_available(&m),
        "TTE objective is FD-only: no analytic outer gradient"
    );
    assert_eq!(
        Optimizer::Auto.resolve_auto(&m, true),
        Optimizer::Bobyqa,
        "auto must resolve to derivative-free Bobyqa for a TTE model"
    );
}

const WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn oral_subject(times: &[f64]) -> Subject {
    let n = times.len();
    Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

const TWOCPT_IV: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const TWOCPT_ORAL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const THREECPT_IV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// ── [initial_conditions] analytic gradient fixtures (#524) ──────────────
// 1-cpt oral with a parameter-dependent CENTRAL baseline `A₀ = TVC0 · V`
// (IV-bolus impulse kernel). A₀ depends on θ_TVC0, θ_TVV, and η_V, so the
// analytic ∂C/∂A₀ · ∂A₀/∂(θ,η) chain is exercised.
const ONECPT_ORAL_INIT_CENTRAL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVC0(5.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 1-cpt oral with a pre-loaded DEPOT baseline (oral first-order kernel, F=1).
const ONECPT_ORAL_INIT_DEPOT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVD0(80.0, 0.01, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[initial_conditions]
  init(depot) = TVD0
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 2-cpt IV with a central baseline (2-cpt IV-bolus impulse kernel).
const TWOCPT_IV_INIT_CENTRAL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVC0(3.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[initial_conditions]
  init(central) = TVC0 * V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 3-cpt IV with a central baseline (3-cpt IV-bolus impulse kernel).
const THREECPT_IV_INIT_CENTRAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVC0(4.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)
[initial_conditions]
  init(central) = TVC0 * V1
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The analytic `[initial_conditions]` impulse (#524) and its `(θ, η)` jet
/// must match central finite differences of the production predictor
/// `compute_predictions_with_tv` (which layers the f64 init), across the
/// 1-/2-/3-cpt central kernels and the oral-depot kernel. The light inner
/// η-gradient provider must agree with the full outer one too.
#[test]
fn analytical_init_provider_matches_fd() {
    let cases: &[(&str, &str, Vec<f64>, Vec<f64>)] = &[
        (
            "1cpt oral central",
            ONECPT_ORAL_INIT_CENTRAL,
            vec![0.2, 10.0, 1.5, 5.0],
            vec![0.15, -0.10, 0.25],
        ),
        (
            "1cpt oral depot",
            ONECPT_ORAL_INIT_DEPOT,
            vec![0.2, 10.0, 1.5, 80.0],
            vec![0.15, -0.10, 0.25],
        ),
        (
            "2cpt iv central",
            TWOCPT_IV_INIT_CENTRAL,
            vec![10.0, 50.0, 15.0, 100.0, 3.0],
            vec![0.12, -0.08],
        ),
        (
            "3cpt iv central",
            THREECPT_IV_INIT_CENTRAL,
            vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 4.0],
            vec![0.12, -0.08],
        ),
    ];
    for (label, src, theta, eta) in cases {
        let m = parse_model_string(src).unwrap_or_else(|e| panic!("{label}: parse: {e}"));
        assert_eq!(m.analytical_init.len(), 1, "{label}: init parsed");
        assert!(
            analytical_supported(&m),
            "{label}: init model must use the analytic provider, not FD"
        );
        let s = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
        // Full outer jet (value, ∂η, ∂²η², ∂θ, ∂²η∂θ) vs central FD of the
        // init-aware production predictor.
        check_full_provider_vs_fd(&m, &s, theta, eta);
        // Light inner η-gradient must agree with the full provider.
        let full = subject_sensitivities(&m, &s, theta, eta).expect("full supported");
        let light = subject_eta_grad(&m, &s, theta, eta).expect("light supported");
        assert_eq!(light.len(), full.obs.len());
        for (lo, fo) in light.iter().zip(full.obs.iter()) {
            for k in 0..m.n_eta {
                approx::assert_relative_eq!(
                    lo.df_deta[k],
                    fo.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-12
                );
            }
        }
    }
}

// 1-cpt IV with WT-on-CL *and* an `[initial_conditions]` baseline. Regression
// for the #527/#524 review: the TV-cov event-driven walk does not layer the init
// impulse, so an init model must decline TV-cov analytic support and route its
// TV-cov subjects to FD — otherwise the analytic gradient omits the init baseline
// while the objective keeps it.
const ONECPT_IV_TVCOV_INIT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVC0(4.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[initial_conditions]
  init(central) = TVC0 * V
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// **Closed-form `init(...)` on the event-driven TV-cov walk** (#486, branch H). An
/// `init(central) = TVC0·V` baseline under a WT covariate on CL (TV-cov). Production's
/// `compute_predictions_with_tv` layers `A₀·kernel(t, pk)` with the subject-static `t = 0`
/// snapshot; the TV-cov provider now folds in the same impulse via `apply_tvcov_init_outer` /
/// `apply_tvcov_init_inner`. The full provider's value / `∂η` / `∂²η` / `∂θ` / `∂²η∂θ` must
/// match FD of the production predictor (which keeps the init), the inner `∂f/∂η` must equal
/// the outer η-block, and a static-covariate subject of the same model still takes the exact
/// dose-superposition init path.
#[test]
fn analytical_init_tvcov_subject_matches_fd() {
    let m = parse_model_string(ONECPT_IV_TVCOV_INIT).expect("parse");
    assert_eq!(m.analytical_init.len(), 1, "init parsed");
    assert!(
        tvcov_analytical_supported(&m),
        "init model is now analytic on the TV-cov walk (#486)"
    );

    let theta = vec![0.2, 10.0, 0.75, 4.0];
    let eta = vec![0.15, -0.10];

    // (a) TV-cov subject (WT changes across records): outer full provider matches FD of
    // production (which includes the init), and the inner η-gradient matches the outer.
    let tv = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[70.0],
        &[1.0, 2.0, 4.0, 8.0],
        &[70.0, 80.0, 90.0, 100.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    assert!(tv.has_tv_covariates(), "fixture must carry TV covariates");
    assert!(
        subject_sensitivities(&m, &tv, &theta, &eta).is_some(),
        "TV-cov init subject now takes the analytic provider (#486)"
    );
    check_full_provider_vs_fd(&m, &tv, &theta, &eta);
    let outer = subject_sensitivities_tvcov(&m, &tv, &theta, &eta).expect("outer tvcov init");
    let inner = subject_eta_grad_tvcov(&m, &tv, &theta, &eta).expect("inner tvcov init");
    assert_eq!(outer.obs.len(), inner.len());
    for (o, i) in outer.obs.iter().zip(inner.iter()) {
        approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-12, epsilon = 1e-12);
        for k in 0..m.n_eta {
            approx::assert_relative_eq!(
                o.df_deta[k],
                i.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-10
            );
        }
    }

    // (b) TV-cov + EVID 3/4 reset: the closed-form init impulse contributes only to
    // observations strictly before the first reset (production `add_analytical_init`), and
    // the reset itself is carried by the TV-cov walk — the combination must still match FD.
    let tv_reset = tvcov_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[70.0, 90.0],
        &[2.0, 6.0, 13.0, 18.0],
        &[70.0, 70.0, 90.0, 90.0],
        vec![12.0],
        Vec::new(),
        &[],
    );
    check_full_provider_vs_fd(&m, &tv_reset, &theta, &eta);

    // (c) A static-covariate subject of the SAME model still takes the exact analytic init
    // path (dose superposition + layered impulse).
    let mut stat = oral_subject(&[1.0, 2.0, 4.0, 8.0]);
    stat.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    stat.covariates = wt_map(70.0);
    assert!(!stat.has_tv_covariates(), "static-covariate subject");
    assert!(
        subject_sensitivities(&m, &stat, &theta, &eta).is_some(),
        "static-covariate init subject keeps the exact analytic init path"
    );
}

// 1-cpt IV with `init(central) = TVC0 * V` and a `TVC0` bound that admits zero.
// At `TVC0 = 0` the baseline amount `A₀` is exactly 0 but `∂A₀/∂TVC0 = V ≠ 0`.
const ONECPT_IV_INIT_ZERO: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVC0(0.0, -50.0, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// #527/#524 review (finding #2): when the init amount evaluates to exactly 0 but
/// has nonzero parameter sensitivity, the gradient must NOT be dropped. Evaluated
/// at `TVC0 = 0`, `A₀ = 0` yet `∂A₀/∂TVC0 = V`, so the init contributes 0 to the
/// value but `V·kernel` to `∂f/∂TVC0` (and the mixed `∂²f/∂η∂TVC0`). Before the
/// fix `add_init_impulse` skipped the whole impulse on `A₀ == 0`, zeroing those
/// derivatives; the FD of the objective sees the nonzero slope, so the full
/// provider jet (checked here) diverged from FD on the `TVC0` axis.
#[test]
fn analytical_init_zero_amount_keeps_gradient() {
    let m = parse_model_string(ONECPT_IV_INIT_ZERO).expect("parse");
    assert_eq!(m.analytical_init.len(), 1, "init parsed");
    assert!(
        analytical_supported(&m),
        "model must use the analytic provider"
    );
    let mut s = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
    s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    // TVC0 = 0 → A₀ = 0 exactly, the boundary the dropped-gradient bug lived on.
    check_full_provider_vs_fd(&m, &s, &[0.2, 10.0, 0.0], &[0.1, -0.1]);
}

// 2-cpt IV with an *additive* η on V1 (`V1 = TVV1 + ETA_V1`): a non-log-normal
// parameterization, so `∂V1/∂η = 1` (not `V1·sel`). Forces both providers down
// the compiled-program `∂p/∂η` path.
const TWOCPT_IV_ADDITIVE_V1: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 9.0
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 + ETA_V1
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const THREECPT_ORAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.5, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn subject_with_dose(dose: DoseEvent, times: &[f64]) -> Subject {
    let n = times.len();
    Subject {
        id: "1".to_string(),
        doses: vec![dose],
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Check the provider's `f` matches the production predictor exactly, and
/// `∂f/∂η`, `∂f/∂θ` match its finite differences.
fn check_provider_vs_production(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) {
    let sens = subject_sensitivities(model, subject, theta, eta).expect("supported");
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(model, subject, th, e)[j]
    };
    let n_eta = model.n_eta;
    let n_theta = theta.len();
    let he = 1e-6;
    for (j, obs) in sens.obs.iter().enumerate() {
        // f must equal the production prediction (the closed forms agree).
        approx::assert_relative_eq!(
            obs.f,
            pred(eta, theta, j),
            max_relative = 1e-9,
            epsilon = 1e-10
        );
        for k in 0..n_eta {
            let mut ep = eta.to_vec();
            ep[k] += he;
            let mut em = eta.to_vec();
            em[k] -= he;
            let g = (pred(&ep, theta, j) - pred(&em, theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 3e-4, epsilon = 1e-7);
        }
        for m in 0..n_theta {
            let h = he * (1.0 + theta[m].abs());
            let mut tp = theta.to_vec();
            tp[m] += h;
            let mut tm = theta.to_vec();
            tm[m] -= h;
            let g = (pred(eta, &tp, j) - pred(eta, &tm, j)) / (2.0 * h);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 3e-4, epsilon = 1e-7);
        }
    }
}

/// The light provider [`subject_eta_grad`] must return exactly the `f` and
/// `∂f/∂η` the full [`subject_sensitivities`] does — same generic PK source,
/// same log-normal η chain, just first-order. Checked across 1-/2-/3-cpt IV +
/// oral and a steady-state case.
#[test]
fn light_provider_matches_full_provider_eta_grad() {
    let times = [0.25, 1.0, 4.0, 12.0];
    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        {
            let m = parse_model_string(WARFARIN).unwrap();
            let s = oral_subject(&times);
            (m, s, vec![0.2, 10.0, 1.5], vec![0.15, -0.10, 0.25])
        },
        {
            let m = parse_model_string(TWOCPT_IV).unwrap();
            let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
            (m, s, vec![10.0, 50.0, 15.0, 100.0], vec![0.12, -0.08])
        },
        {
            let m = parse_model_string(THREECPT_IV).unwrap();
            let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
            (
                m,
                s,
                vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0],
                vec![0.12, -0.08],
            )
        },
        {
            let m = parse_model_string(THREECPT_ORAL).unwrap();
            let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0), &times);
            (
                m,
                s,
                vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5],
                vec![0.12, -0.08, 0.2],
            )
        },
        {
            // Non-log-normal η (additive on V1): exercises the program-path
            // `∂p/∂η` branch, not the closed-form `pk·sel` chain.
            let m = parse_model_string(TWOCPT_IV_ADDITIVE_V1).unwrap();
            let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
            (m, s, vec![10.0, 50.0, 15.0, 100.0], vec![0.1, 3.0])
        },
    ];
    for (m, s, theta, eta) in &cases {
        let full = subject_sensitivities(m, s, theta, eta).expect("full supported");
        let light = subject_eta_grad(m, s, theta, eta).expect("light supported");
        assert_eq!(full.obs.len(), light.len());
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
            for k in 0..m.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-12,
                    epsilon = 1e-14
                );
            }
        }
    }
}

/// A constant `obs_scale` divisor (`ScalarScale`) must flow through `f` and
/// every η/θ derivative — the provider divides the whole jet by `k`, and the
/// production predictor (`compute_predictions_with_tv` → `apply_scaling`)
/// divides its predictions by the same `k`, so they (and the FD derivatives)
/// must still agree.
#[test]
fn provider_scalar_scale_matches_production() {
    let scaled = WARFARIN.replace(
        "[error_model]",
        "[scaling]\n  obs_scale = 1000\n[error_model]",
    );
    let model = parse_model_string(&scaled).expect("parse");
    assert!(
        matches!(model.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-9),
        "model must carry the ScalarScale"
    );
    assert!(
        analytical_supported(&model),
        "ScalarScale must be supported"
    );
    let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    check_provider_vs_production(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);

    // The light η-provider must agree with the full provider under scaling too.
    let full = subject_sensitivities(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25])
        .expect("full");
    let light =
        subject_eta_grad(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]).expect("light");
    for (fo, lo) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                fo.df_deta[k],
                lo.df_deta[k],
                max_relative = 1e-12,
                epsilon = 1e-14
            );
        }
    }
}

#[test]
fn provider_2cpt_bolus_infusion_oral_match_production() {
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
    // 2-cpt IV: bolus (rate=0) and infusion (rate>0, dur=2).
    let iv = parse_model_string(TWOCPT_IV).expect("parse");
    let theta_iv = vec![10.0, 50.0, 15.0, 100.0];
    let eta_iv = vec![0.12, -0.08];
    let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
    let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
    check_provider_vs_production(&iv, &bolus, &theta_iv, &eta_iv);
    check_provider_vs_production(&iv, &infusion, &theta_iv, &eta_iv);

    // 2-cpt oral (first-order absorption).
    let oral_m = parse_model_string(TWOCPT_ORAL).expect("parse");
    let theta_or = vec![10.0, 50.0, 15.0, 100.0, 1.0];
    let eta_or = vec![0.12, -0.08, 0.2];
    let oral_s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
    check_provider_vs_production(&oral_m, &oral_s, &theta_or, &eta_or);
}

/// Regression: bioavailability `F` on an IV bolus / infusion must be applied
/// in the sensitivity path too. Production scales non-oral routes by `F` since
/// #327 (`route_f_scale`); before the `*_conc_g` post-multiply fix the sens IV
/// branch ignored `F`, so the analytic gradient/Jacobian was computed for a
/// different (unscaled) prediction surface than the FOCEI objective.
#[test]
fn provider_iv_with_bioavailability_matches_production() {
    const ONECPT_IV_F: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_F  ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  F  = TVF  * exp(ETA_F)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let times = [0.25, 1.0, 2.0, 4.0, 8.0];
    let m = parse_model_string(ONECPT_IV_F).expect("parse");
    let theta = vec![10.0, 50.0, 0.7];
    let eta = vec![0.1, -0.05, 0.2];
    // F on an IV *bolus* is a magnitude scale, so the analytic post-multiply
    // still matches production.
    let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
    check_provider_vs_production(&m, &bolus, &theta, &eta);
    // #419: an IV *infusion* under F ≠ 1 reshapes (rate held, window F·dur)
    // rather than scaling its magnitude, so the analytic `route_f_scale`
    // post-multiply no longer matches production — both providers decline it to
    // the FD gradient (whose `event_driven_predictions` applies the #419 rule).
    let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
    assert!(
        subject_sensitivities(&m, &infusion, &theta, &eta).is_none(),
        "F≠1 rate-defined infusion must decline to FD (full provider, #419)"
    );
    assert!(
        subject_eta_grad(&m, &infusion, &theta, &eta).is_none(),
        "F≠1 rate-defined infusion must decline to FD (light provider, #419)"
    );
}

/// Regression: a modeled-duration dose (`RATE=-2` → `D{cmt}`) is read with
/// unresolved `rate`/`duration` by the provider, so the analytic path must
/// decline (→ FD) rather than optimize a bolus/zero-input surrogate.
#[test]
fn provider_modeled_duration_dose_falls_back_to_fd() {
    let iv = parse_model_string(TWOCPT_IV).expect("parse");
    let mut dose = DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0);
    dose.rate_mode = crate::types::RateMode::ModeledDuration;
    let subj = subject_with_dose(dose, &[0.5, 2.0, 6.0]);
    let theta = vec![10.0, 50.0, 15.0, 100.0];
    let eta = vec![0.1, -0.05];
    assert!(
        subject_eta_grad(&iv, &subj, &theta, &eta).is_none(),
        "modeled-duration dose must fall back to FD (light provider)"
    );
    assert!(
        subject_sensitivities(&iv, &subj, &theta, &eta).is_none(),
        "modeled-duration dose must fall back to FD (full provider)"
    );
}

/// 1-cpt IV closed-form model with a **modeled-duration** dose (`RATE=-2` → `D1`),
/// `D1 = TVD1·exp(ETA_D1)`. The infusion window end `t_dose + D1` is a moving
/// boundary in `D1`; the event-driven walk carries it via the exact dual window
/// length (#486). Used by the analytic-vs-FD and routing tests below.
const ONECPT_IV_MODELED_DUR: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// An observation sampled **exactly at a moving infusion end** gets the *one-sided* analytic
/// derivative — the same value the ODE engine's jump/saltation sensitivities return (#486).
///
/// The walk propagates through the moving boundary, so its state at that instant is
/// `x(end(D))`. That is what later events need, but an observation's clock time does **not**
/// move with `D`: reading it there differentiates along the boundary and adds a spurious
/// `ẋ·∂end/∂D`. On this fixture that produced `−1.92`, which is not even a subgradient in
/// general. The walk now steps the read-out state back across the zero-length sliver
/// `Δ = end(D) − t_obs` (value 0, live jet) with the infusion still running, recovering
/// `∂x/∂D` at the sample's own fixed time.
///
/// The prediction is genuinely **kinked** in `D` here — above the coincidence the infusion is
/// still running at the sample (only the rate `amt/D` matters), below it the dose has finished
/// and a decay term appears — so no two-sided derivative exists (one-sided slopes `−8.57` and
/// `+2.26`; central FD returns their average `−3.15`). The strongest correct answer available
/// is a one-sided derivative, and the **ODE twin is the oracle**: it reaches the identical
/// value by an entirely independent route (RK45 + an explicit saltation injection, versus a
/// dual sliver on a closed form), so agreeing with it pins the convention *and* the value.
/// Central-FD parity is deliberately **not** asserted at the kink — it cannot hold.
#[test]
fn obs_on_modeled_infusion_end_matches_ode_twin() {
    const ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let cf = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse cf");
    let ode = parse_model_string(ODE_TWIN).expect("parse ode twin");
    let theta = [10.0, 50.0, 2.0];
    // η_D1 = 0 ⇒ D1 = TVD1 = 2.0 exactly, so the t = 2.0 sample sits ON the window end.
    let eta = [0.12, -0.08, 0.0];
    let dose = || {
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        )
    };

    let on_end = subject_with_dose(dose(), &[1.0, 2.0, 3.0]);
    let cf_sens = subject_sensitivities(&cf, &on_end, &theta, &eta)
        .expect("closed form must stay analytic on the kink (corrected read-out)");
    let ode_sens = subject_sensitivities(&ode, &on_end, &theta, &eta)
        .expect("ODE twin serves it via jump sensitivities");

    for (j, (c, o)) in cf_sens.obs.iter().zip(ode_sens.obs.iter()).enumerate() {
        approx::assert_relative_eq!(c.f, o.f, max_relative = 1e-6, epsilon = 1e-9);
        for m in 0..theta.len() {
            approx::assert_relative_eq!(
                c.df_dtheta[m],
                o.df_dtheta[m],
                max_relative = 1e-5,
                epsilon = 1e-7
            );
        }
        for k in 0..cf.n_eta {
            approx::assert_relative_eq!(
                c.df_deta[k],
                o.df_deta[k],
                max_relative = 1e-5,
                epsilon = 1e-7
            );
        }
        let _ = j;
    }

    // The inner (Dual1) walk must carry the same correction as the outer (Dual2) one — a
    // correction applied to one and not the other would silently split inner/outer scope.
    let inner = subject_eta_grad(&cf, &on_end, &theta, &eta).expect("inner analytic");
    for (o, i) in cf_sens.obs.iter().zip(inner.iter()) {
        for k in 0..cf.n_eta {
            approx::assert_relative_eq!(
                o.df_deta[k],
                i.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }

    // Away from the boundary the correction must be inert: still exact vs central FD.
    let off_end = subject_with_dose(dose(), &[1.0, 2.5, 3.0]);
    check_full_provider_vs_fd(&cf, &off_end, &theta, &eta);
}

/// A modeled-duration `RATE=-2` subject must now route to the event-driven walk
/// and be served analytically (the closed-form twin of the ODE #635 path), not
/// fall back to FD (#486). Both the full outer provider and the light inner
/// gradient must accept it.
#[test]
fn provider_modeled_duration_routes_to_event_walk() {
    let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        &[0.5, 1.5, 3.0, 6.0],
    );
    assert!(!subject.all_doses_fixed(), "dose must be modeled");
    assert!(
        subject_routes_to_event_walk(&model, &subject),
        "a modeled-RATE subject must route to the event-driven walk (#486)"
    );
    assert!(
        subject_sensitivities(&model, &subject, &[10.0, 50.0, 2.0], &[0.1, -0.05, 0.05]).is_some(),
        "modeled-duration dose must be served analytically by the walk (full provider)"
    );
    assert!(
        subject_eta_grad(&model, &subject, &[10.0, 50.0, 2.0], &[0.1, -0.05, 0.05]).is_some(),
        "modeled-duration dose must be served analytically (light inner provider)"
    );
}

/// The exact outer `(η, θ)` sensitivities of a modeled-duration `RATE=-2` dose must
/// match central finite differences of the production predictor
/// (`compute_predictions_with_tv`, which resolves `D1` per evaluation). Observations
/// straddle the window end (`D1 ≈ 2.1`: `t = 0.5, 1.5` inside; `t = 3, 6` after), so
/// the moving infusion-end boundary is genuinely exercised (#486).
#[test]
fn provider_modeled_duration_iv_matches_fd_of_production() {
    let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        &[0.5, 1.5, 3.0, 6.0],
    );
    check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05]);
}

/// Modeled **rate** (`RATE=-1` → `R1`), the sign-mirror: `duration = amt/R1`, so the
/// window end still moves with the PK param. Analytic outer sensitivities vs FD of
/// production (#486).
#[test]
fn provider_modeled_rate_iv_matches_fd_of_production() {
    const ONECPT_IV_MODELED_RATE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_R1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  R1 = TVR1 * exp(ETA_R1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(ONECPT_IV_MODELED_RATE).expect("parse");
    // amt=1000, R1≈500 → window ≈ 2h. Obs straddle it.
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledRate,
        ),
        &[0.5, 1.5, 3.0, 6.0],
    );
    check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 500.0], &[0.12, -0.08, 0.05]);
}

/// The light inner η-gradient (`subject_eta_grad`) for a modeled-duration dose must
/// match the `df_deta` block of the FD-validated full outer provider
/// (`subject_sensitivities`) — the inner and outer walks must agree (#486).
#[test]
fn provider_modeled_duration_inner_matches_outer() {
    let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        &[0.5, 1.5, 3.0, 6.0],
    );
    let theta = [10.0, 50.0, 2.0];
    let eta = [0.12, -0.08, 0.05];
    let outer = subject_sensitivities(&model, &subject, &theta, &eta).expect("outer some");
    let inner = subject_eta_grad(&model, &subject, &theta, &eta).expect("inner some");
    assert_eq!(outer.obs.len(), inner.len());
    for (o, i) in outer.obs.iter().zip(inner.iter()) {
        approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-9, epsilon = 1e-12);
        for (a, b) in o.df_deta.iter().zip(i.df_deta.iter()) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-12);
        }
    }
}

/// **Bit-parity cross-check vs the ODE twin.** The closed-form modeled-duration
/// walk and the ODE `[odes]` modeled-duration walk (already analytic + NONMEM-
/// anchored via #630/#635) are two independent implementations of the same
/// moving-boundary sensitivity. Their outer `(η, θ)` jets must agree to ODE
/// integration tolerance — the strongest confirmation the closed-form saltation is
/// right (#486).
#[test]
fn provider_modeled_duration_matches_ode_twin() {
    const ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let cf = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse cf");
    let ode = parse_model_string(ODE_TWIN).expect("parse ode twin");
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        &[0.5, 1.5, 3.0, 6.0],
    );
    let theta = [10.0, 50.0, 2.0];
    let eta = [0.12, -0.08, 0.05];
    let s_cf = subject_sensitivities(&cf, &subject, &theta, &eta).expect("cf some");
    let s_ode = subject_sensitivities(&ode, &subject, &theta, &eta).expect("ode some");
    assert_eq!(s_cf.obs.len(), s_ode.obs.len());
    for (a, b) in s_cf.obs.iter().zip(s_ode.obs.iter()) {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-6, epsilon = 1e-8);
        for (x, y) in a.df_deta.iter().zip(b.df_deta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
        }
        for (x, y) in a.df_dtheta.iter().zip(b.df_dtheta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
        }
        for (x, y) in a.d2f_deta2.iter().zip(b.d2f_deta2.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-4, epsilon = 1e-6);
        }
    }
}

/// **Closed-form modeled-duration dose × steady state** (#486 — the last modeled-dose
/// gap after #652). The SS trough is equilibrated by `equilibrate_ss_g`, which now threads
/// the modeled window dual `(rate_bare, dur_bare)` into each cycle's active/quiet split, so
/// the moving infusion-end jet (`∂D1`) flows through the SS trough exactly as it does
/// through the current pulse. `D1 ≈ 2.1 < II = 12`; obs straddle the window end within the
/// SS interval (0.5, 1.5 inside; 3, 6, 11 after). Validated vs central FD of the production
/// predictor (`compute_predictions_with_tv`, which resolves `D1` per evaluation and
/// equilibrates the same SS train).
#[test]
fn provider_modeled_duration_ss_matches_fd_of_production() {
    let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        ),
        &[0.5, 1.5, 3.0, 6.0, 11.0],
    );
    assert!(subject.has_periodic_ss_dose());
    assert!(
        subject_sensitivities(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05]).is_some(),
        "modeled-duration + SS must be served analytically now (#486)"
    );
    check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05]);
}

/// Modeled **rate** (`RATE=-1`) × steady state — the sign-mirror of the duration case,
/// same `equilibrate_ss_g` window-jet path (`duration = amt/R1` moves the SS window end).
#[test]
fn provider_modeled_rate_ss_matches_fd_of_production() {
    const ONECPT_IV_MODELED_RATE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(500.0, 10.0, 5000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_R1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  R1 = TVR1 * exp(ETA_R1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(ONECPT_IV_MODELED_RATE).expect("parse");
    // amt=1000, R1≈500 → window ≈ 2h < II = 12. Obs straddle the SS window end.
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledRate,
        ),
        &[0.5, 1.5, 3.0, 6.0, 11.0],
    );
    check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 500.0], &[0.12, -0.08, 0.05]);
}

/// **SS dose into a plain compartment on an absorption-declaring model stays analytic**
/// (#719 review, finding 4). The SS-into-input-rate-compartment FD decline in
/// `ode_tvcov_supported` is scoped to the *SS-dosed* compartment: a model that declares a
/// built-in `first_order` absorption forcing on the depot, but whose subject's `SS=1` dose
/// is an IV loading straight into **central** (a non-forcing compartment, so `R_in` is inert
/// for this subject), must keep the exact event-driven analytic dual walk rather than fall to
/// the FD fallback. Pins both halves: (a) the gate now admits it (`subject_sensitivities`
/// returns `Some` — it returned `None` under the old model-wide `!input_rate.is_empty()`
/// decline), and (b) the analytic jets match central FD of the production predictor.
#[test]
fn provider_ss_into_plain_cmt_with_inert_absorption_matches_fd() {
    use crate::types::DoseEvent;
    // Depot (cmt 1) carries a `first_order` input-rate forcing; central (cmt 2) is the
    // disposition compartment. The subject's only dose is an SS IV bolus into central, so
    // the depot forcing never fires — the effective system is a 1-cpt IV SS bolus.
    const DEPOT_FORCING_CENTRAL_SS: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = first_order(ka=KA) - KA*depot
  d/dt(central) = KA*depot - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let model = parse_model_string(DEPOT_FORCING_CENTRAL_SS).expect("parse");
    assert!(
        model
            .ode_spec
            .as_ref()
            .is_some_and(|o| !o.input_rate.is_empty()),
        "model must declare a built-in absorption forcing (on the depot)"
    );
    let n = 5usize;
    // SS=1 bolus into central (cmt 2), II = 12; the depot forcing (cmt 1) is inert here.
    let subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 2, 0.0, true, 12.0)],
        obs_times: vec![0.5, 2.0, 4.0, 8.0, 11.0],
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        obs_l2: Vec::new(),
    };
    assert!(subject.has_periodic_ss_dose());
    assert!(
        crate::sens::ode_provider::ode_tvcov_supported(&model, &subject),
        "an SS dose into a non-forcing compartment must stay on the analytic walk (#719)"
    );
    assert!(
        subject_sensitivities(&model, &subject, &[0.2, 10.0, 1.0], &[0.1, -0.05, 0.15]).is_some(),
        "analytic provider must serve SS-into-plain-cmt (returned None under the old \
             model-wide input-rate decline)"
    );
    check_full_provider_vs_fd(&model, &subject, &[0.2, 10.0, 1.0], &[0.1, -0.05, 0.15]);
}

/// **Bit-parity cross-check vs the ODE twin for SS + modeled-duration.** The ODE
/// `[odes]` SS + modeled-duration path is already analytic (#642) and independently
/// NONMEM-anchored, so its outer `(η, θ)` jets are a strong oracle for the closed-form
/// SS window-jet just added — they must agree to ODE integration tolerance (#486).
#[test]
fn provider_modeled_duration_ss_matches_ode_twin() {
    const ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let cf = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse cf");
    let ode = parse_model_string(ODE_TWIN).expect("parse ode twin");
    let subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        ),
        &[0.5, 1.5, 3.0, 6.0, 11.0],
    );
    let theta = [10.0, 50.0, 2.0];
    let eta = [0.12, -0.08, 0.05];
    let s_cf = subject_sensitivities(&cf, &subject, &theta, &eta).expect("cf some");
    let s_ode = subject_sensitivities(&ode, &subject, &theta, &eta).expect("ode some");
    assert_eq!(s_cf.obs.len(), s_ode.obs.len());
    for (a, b) in s_cf.obs.iter().zip(s_ode.obs.iter()) {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-6, epsilon = 1e-8);
        for (x, y) in a.df_deta.iter().zip(b.df_deta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
        }
        for (x, y) in a.df_dtheta.iter().zip(b.df_dtheta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-7);
        }
    }
}

/// **Closed-form modeled-duration × SS under IOV — κ-coupled window** (#486 review #1).
/// The relaxed `modeled_dose_analytic_gate` is shared by the closed-form **IOV** entry
/// (`build_iov_sources`), so a modeled SS dose is now analytic there too. Here `D1` is
/// κ-coupled (`D1 = TVD1·exp(ETA_D1 + KAPPA_D1)`), so each occasion's SS window length
/// differs — this pins that the per-occasion modeled-window jet threaded into
/// `equilibrate_ss_g` lands in the correct stacked-κ axis. FD of `predict_iov` is the
/// independent oracle.
#[test]
fn provider_modeled_duration_ss_iov_kappa_coupled_matches_fd_of_predict_iov() {
    const CF_IOV_MODELED_SS: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_D1 ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1 + KAPPA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let model = parse_model_string(CF_IOV_MODELED_SS).expect("parse cf IOV modeled SS");
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        ),
        DoseEvent::modeled(
            24.0,
            1000.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        ),
    ];
    let theta = [0.2, 10.0, 5.0];
    // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1]; κ on D1 → each occasion's SS window differs.
    let stacked = [0.12, -0.08, 0.05, 0.06, -0.11];
    assert!(
        subject_sensitivities_iov(&model, &subject, &theta, &stacked).is_some(),
        "closed-form IOV modeled-duration + SS must be analytic now (#486)"
    );
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
}

/// **Closed-form modeled-duration × SS under time-varying covariates** (#486 review #1).
/// The relaxed gate is also shared by the non-IOV TV-cov entry
/// (`subject_sensitivities_tvcov`); a covariate (`WT` on `CL`) varies across the records
/// of a single SS modeled dose. Validated vs central FD of `compute_predictions_with_tv`
/// (which resolves `D1` and equilibrates the SS train per evaluation).
#[test]
fn provider_modeled_duration_ss_tvcov_matches_fd_of_production() {
    const CF_TVCOV_MODELED_SS: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(2.0, 0.1, 24.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_D1 ~ 0.04
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(CF_TVCOV_MODELED_SS).expect("parse cf tvcov modeled SS");
    let mut subject = subject_with_dose(
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        ),
        &[1.0, 4.0, 6.0, 8.0, 11.0],
    );
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    subject.dose_covariates = vec![wt(60.0)];
    subject.obs_covariates = vec![wt(65.0), wt(70.0), wt(75.0), wt(80.0), wt(85.0)];
    assert!(subject.has_tv_covariates());
    assert!(
        subject_sensitivities(
            &model,
            &subject,
            &[10.0, 50.0, 2.0, 0.75],
            &[0.12, -0.08, 0.05]
        )
        .is_some(),
        "closed-form TV-cov modeled-duration + SS must be analytic now (#486)"
    );
    check_full_provider_vs_fd(
        &model,
        &subject,
        &[10.0, 50.0, 2.0, 0.75],
        &[0.12, -0.08, 0.05],
    );
}

/// **Reset regression (#486 review #3).** A modeled-duration dose that starts
/// *before* an EVID3/4 reset must not thread its window jet into the post-reset
/// sub-intervals: the walk's rate loop drops a pre-reset dose (`t_start <
/// reset_floor`), and `dual_pos` must drop it too — otherwise the post-reset window
/// length carries a spurious `∂D`. The first infusion (`t = 0`, `D1 ≈ 2.1`) ends
/// *after* the reset at `t = 1`, so its stale infusion-end break at `≈ 2.1` lands in
/// a post-reset interval that the obs at `t = 2.5` straddles; a fresh modeled dose at
/// the reset supplies the legitimate post-reset moving boundary. Validated against FD
/// of the production predictor (which cancels the pre-reset infusion at the reset).
#[test]
fn provider_modeled_duration_reset_matches_fd_of_production() {
    let model = parse_model_string(ONECPT_IV_MODELED_DUR).expect("parse");
    let doses = vec![
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        DoseEvent::modeled(
            1.0,
            800.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
    ];
    let subject = subject_with_doses_and_resets(doses, &[0.5, 1.5, 2.5, 4.0], vec![1.0]);
    assert!(subject.has_resets(), "subject must carry a reset");
    assert!(!subject.all_doses_fixed(), "doses must be modeled");
    check_full_provider_vs_fd(&model, &subject, &[10.0, 50.0, 2.0], &[0.12, -0.08, 0.05]);
}

/// **Separability guard (#486 review #2).** Two modeled-duration doses into
/// *distinct* compartments (`D1` into cmt 1, `D2` into cmt 2) whose infusion ends
/// coincide can't be represented by the single-`dt` walk — each end wants a different
/// per-compartment dual window length — so the subject declines to FD
/// (`subject_sensitivities` → `None`). Moving `D2` so the ends separate restores the
/// analytic path. Same-slot coincidences (identical jet) are unaffected.
#[test]
fn provider_modeled_distinct_slot_coincident_ends_decline() {
    const TWOCPT_IV_D1_D2: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(5.0, 0.5, 50.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVD1(2.0, 0.1, 24.0)
  theta TVD2(2.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  D1 = TVD1
  D2 = TVD2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(TWOCPT_IV_D1_D2).expect("parse");
    let doses = vec![
        DoseEvent::modeled(
            0.0,
            1000.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        DoseEvent::modeled(
            0.0,
            800.0,
            2,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
    ];
    // TVD1 == TVD2 == 2.0, both doses at t=0 → both infusions end at t=2 with
    // distinct backing slots ⇒ not separable ⇒ decline to FD.
    //
    // Sample times deliberately avoid every window end (2.0 / 3.0 below): an observation
    // *on* a moving infusion end takes the corrected one-sided read-out
    // (`obs_on_modeled_infusion_end_matches_ode_twin`), which would muddy the distinct-slot
    // separability this test is actually about.
    let coincident = subject_with_doses_and_resets(doses.clone(), &[0.5, 1.5, 3.5], Vec::new());
    assert!(
        subject_sensitivities(
            &model,
            &coincident,
            &[10.0, 50.0, 5.0, 100.0, 2.0, 2.0],
            &[0.1, -0.05]
        )
        .is_none(),
        "distinct-slot coincident infusion ends must decline to FD (#486 review #2)"
    );
    // TVD2 = 3.0 → ends at t=2 (cmt1) and t=3 (cmt2), separable ⇒ analytic path.
    assert!(
        subject_sensitivities(
            &model,
            &coincident,
            &[10.0, 50.0, 5.0, 100.0, 2.0, 3.0],
            &[0.1, -0.05]
        )
        .is_some(),
        "separable infusion ends must be served analytically"
    );
}

#[test]
fn provider_2cpt_steady_state_matches_production() {
    // SS bolus (II=12) and SS oral (II=24) — exercises the *_ss_g branches.
    let times = [0.5, 2.0, 6.0, 11.5];
    let iv = parse_model_string(TWOCPT_IV).expect("parse");
    let ss_bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 12.0), &times);
    check_provider_vs_production(&iv, &ss_bolus, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);

    let oral_m = parse_model_string(TWOCPT_ORAL).expect("parse");
    let ss_oral = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
        &[2.0, 6.0, 12.0, 23.0],
    );
    check_provider_vs_production(
        &oral_m,
        &ss_oral,
        &[10.0, 50.0, 15.0, 100.0, 1.0],
        &[0.1, -0.05, 0.15],
    );
}

#[test]
fn provider_2cpt_ss_infusion_matches_production() {
    // Non-overlapping SS infusion (rate=200, amt=1000 → dur=5; II=12 → dur<II).
    let iv = parse_model_string(TWOCPT_IV).expect("parse");
    let ss_inf = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
        &[1.0, 4.0, 6.0, 8.0, 11.0],
    );
    check_provider_vs_production(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);
}

#[test]
fn provider_1cpt_ss_infusion_matches_production() {
    let m = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
    let ss_inf = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
        &[1.0, 4.0, 6.0, 8.0, 11.0],
    );
    check_provider_vs_production(&m, &ss_inf, &[10.0, 50.0], &[0.1, -0.05]);
}

#[test]
fn provider_3cpt_bolus_infusion_oral_match_production() {
    let times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
    let iv = parse_model_string(THREECPT_IV).expect("parse");
    let theta_iv = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
    let eta_iv = vec![0.12, -0.08];
    let bolus = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
    let infusion = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
    check_provider_vs_production(&iv, &bolus, &theta_iv, &eta_iv);
    check_provider_vs_production(&iv, &infusion, &theta_iv, &eta_iv);

    let oral_m = parse_model_string(THREECPT_ORAL).expect("parse");
    let theta_or = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5];
    let eta_or = vec![0.12, -0.08, 0.2];
    let oral_s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), &times);
    check_provider_vs_production(&oral_m, &oral_s, &theta_or, &eta_or);
}

#[test]
fn provider_3cpt_steady_state_matches_production() {
    // SS bolus (II=12), SS oral (II=24), SS infusion (dur<II) — exercises
    // every *_ss_g branch for 3-cpt.
    let iv = parse_model_string(THREECPT_IV).expect("parse");
    let theta_iv = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
    let ss_bolus = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 12.0),
        &[0.5, 2.0, 6.0, 11.5],
    );
    check_provider_vs_production(&iv, &ss_bolus, &theta_iv, &[0.1, -0.05]);

    let ss_inf = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 12.0),
        &[1.0, 4.0, 6.0, 8.0, 11.0],
    );
    check_provider_vs_production(&iv, &ss_inf, &theta_iv, &[0.1, -0.05]);

    let oral_m = parse_model_string(THREECPT_ORAL).expect("parse");
    let theta_or = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5];
    let ss_oral = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0),
        &[2.0, 6.0, 12.0, 23.0],
    );
    check_provider_vs_production(&oral_m, &ss_oral, &theta_or, &[0.1, -0.05, 0.15]);
}

#[test]
fn provider_overlapping_ss_infusion_matches_production() {
    // Overlapping SS infusion (rate=200, amt=1000 → dur=5; II=2 → dur>II): the
    // provider now carries the same superposed closed form as production (#379),
    // so its value/η/θ sensitivities match FD of the production predictor.
    // Observations sampled within the dosing interval [0, II).
    let iv = parse_model_string(TWOCPT_IV).expect("parse");
    let ss_inf = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 2.0),
        &[0.3, 0.8, 1.2, 1.7],
    );
    assert!(
        subject_sensitivities(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]).is_some(),
        "overlapping SS infusion is now provider-supported"
    );
    check_provider_vs_production(&iv, &ss_inf, &[10.0, 50.0, 15.0, 100.0], &[0.1, -0.05]);

    // 1-cpt IV overlapping too (dur = 1000/200 = 5 > II = 2).
    let one = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
    let one_inf = subject_with_dose(
        DoseEvent::new(0.0, 1000.0, 1, 200.0, true, 2.0),
        &[0.3, 0.8, 1.2, 1.7],
    );
    check_provider_vs_production(&one, &one_inf, &[10.0, 50.0], &[0.1, -0.05]);
}

/// Build a subject carrying explicit doses and EVID=3/4 reset times (no
/// covariates, no IOV) for the reset-superposition tests.
fn subject_with_doses_and_resets(
    doses: Vec<DoseEvent>,
    times: &[f64],
    reset_times: Vec<f64>,
) -> Subject {
    let n = times.len();
    Subject {
        id: "1".to_string(),
        doses,
        obs_times: times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times,
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Oral models with an **infusion** dose (#350 depot-bypass central, RATE>0
/// into cmt 2; #400 zero-order into the depot, RATE>0 into cmt 1) route through
/// the state-propagating Dual2 walk — whose oral propagators now carry
/// `rate_central`/`rate_depot` — rather than dose superposition. The provider's
/// value/∂η/∂²η/∂θ/∂²η∂θ must match central FD of the production predictor
/// (`compute_predictions_with_tv`, the independent infusion-correct f64 path)
/// across 1-/2-/3-cpt and both infusion compartments.
#[test]
fn oral_infusion_provider_matches_fd_of_production() {
    const ONECPT_ORAL: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    const THREECPT_ORAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.0, 0.05, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    // Infusion of amt 1000 over 8 h (rate 125), then a later oral bolus.
    let inf = |cmt: usize| {
        vec![
            DoseEvent::new(0.0, 1000.0, cmt, 125.0, false, 0.0),
            DoseEvent::new(12.0, 500.0, 1, 0.0, false, 0.0),
        ]
    };
    let times = [1.0, 4.0, 7.0, 10.0, 14.0, 24.0];
    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        // 1-cpt oral, zero-order into the **depot** (cmt 1, #400).
        {
            let m = parse_model_string(ONECPT_ORAL).expect("parse 1cpt oral");
            let s = subject_with_doses_and_resets(inf(1), &times, Vec::new());
            (m, s, vec![10.0, 50.0, 1.0], vec![0.1, -0.05, 0.08])
        },
        // 1-cpt oral, depot-bypass infusion into **central** (cmt 2, #350).
        {
            let m = parse_model_string(ONECPT_ORAL).expect("parse 1cpt oral");
            let s = subject_with_doses_and_resets(inf(2), &times, Vec::new());
            (m, s, vec![10.0, 50.0, 1.0], vec![0.1, -0.05, 0.08])
        },
        // 2-cpt oral, zero-order into the depot (cmt 1).
        {
            let m = parse_model_string(TWOCPT_ORAL).expect("parse 2cpt oral");
            let s = subject_with_doses_and_resets(inf(1), &times, Vec::new());
            (
                m,
                s,
                vec![10.0, 50.0, 15.0, 100.0, 1.0],
                vec![0.1, -0.05, 0.08],
            )
        },
        // 3-cpt oral, zero-order into the depot (cmt 1).
        {
            let m = parse_model_string(THREECPT_ORAL).expect("parse 3cpt oral");
            let s = subject_with_doses_and_resets(inf(1), &times, Vec::new());
            (
                m,
                s,
                vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.0],
                vec![0.1, -0.05, 0.08],
            )
        },
    ];
    for (m, s, theta, eta) in &cases {
        assert!(
            subject_has_oral_infusion(m, s),
            "fixture must carry an oral infusion"
        );
        assert!(
            subject_sensitivities(m, s, theta, eta).is_some(),
            "oral-infusion subject must take the analytic provider (via the walk)"
        );
        check_provider_vs_production(m, s, theta, eta);
    }
}

/// Two infusion occasions on a 3-cpt IV model separated by an EVID=4 reset:
/// occasion-2 observations must rebuild from zero (no occasion-1 carryover).
/// The provider's reset-segment superposition must reproduce the production
/// event-driven predictor and its FD sensitivities.
#[test]
fn provider_3cpt_two_occasion_reset_matches_production() {
    let iv = parse_model_string(THREECPT_IV).expect("parse");
    let theta = vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0];
    let eta = vec![0.12, -0.08];
    // Occasion 1: infusion at t=0 (rate 200, amt 1000 → 5 h). Occasion 2:
    // same infusion at t=120, opened by an EVID=4 reset at t=120.
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 200.0, false, 0.0),
        DoseEvent::new(120.0, 1000.0, 1, 200.0, false, 0.0),
    ];
    let times = [2.0, 4.0, 8.0, 60.0, 122.0, 126.0, 150.0];
    let subject = subject_with_doses_and_resets(doses, &times, vec![120.0]);
    assert!(subject.has_resets(), "fixture must carry a reset");
    check_provider_vs_production(&iv, &subject, &theta, &eta);
}

/// A reset that lands mid-infusion (1-cpt IV): the ongoing infusion is turned
/// off and the compartment zeroed, so post-reset observations see only doses
/// from the new segment. Exercises the `dose.time < reset_floor` exclusion of
/// an in-flight infusion.
#[test]
fn provider_1cpt_reset_midinfusion_matches_production() {
    let m = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
    // Infusion 0–8 h (rate 125, amt 1000); reset at t=4 mid-infusion; a fresh
    // bolus opens the new segment at t=4.
    let doses = vec![
        DoseEvent::new(0.0, 1000.0, 1, 125.0, false, 0.0),
        DoseEvent::new(4.0, 500.0, 1, 0.0, false, 0.0),
    ];
    let times = [1.0, 3.0, 5.0, 7.0, 10.0];
    let subject = subject_with_doses_and_resets(doses, &times, vec![4.0]);
    check_provider_vs_production(&m, &subject, &[10.0, 50.0], &[0.1, -0.05]);
}

/// Provider's exact η/θ sensitivities (value, ∂/∂η, ∂²/∂η², ∂/∂θ, ∂²/∂η∂θ)
/// must match central finite differences of the production predictor
/// `compute_predictions_with_tv`. Shared by the natural-scale and LTBS checks —
/// for an LTBS model the production predictor returns `ln(f)`, and the provider
/// applies the matching `g = ln(f)` jet transform, so the same FD check covers
/// the log-scale value, gradient, and Hessian.
fn check_full_provider_vs_fd(model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64]) {
    let n_eta = model.n_eta;
    let n_theta = theta.len();

    let sens = subject_sensitivities(model, subject, theta, eta).expect("supported");

    // FD helpers over the full prediction vector (returns obs j's value).
    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(model, subject, th, e)[j]
    };
    let he = 1e-6; // first-derivative step
    let ht = 1e-6;
    let heh = 1e-4; // second-derivative step (4-point central is roundoff-prone)

    for (j, obs) in sens.obs.iter().enumerate() {
        // value
        let f0 = pred(&eta, &theta, j);
        approx::assert_relative_eq!(obs.f, f0, max_relative = 1e-9, epsilon = 1e-12);

        // ∂f/∂η and ∂²f/∂η²
        for k in 0..n_eta {
            let mut ep = eta.to_vec();
            ep[k] += he;
            let mut em = eta.to_vec();
            em[k] -= he;
            let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-4, epsilon = 1e-7);
            for l in 0..n_eta {
                let mut pp = eta.to_vec();
                pp[k] += heh;
                pp[l] += heh;
                let mut pm = eta.to_vec();
                pm[k] += heh;
                pm[l] -= heh;
                let mut mp = eta.to_vec();
                mp[k] -= heh;
                mp[l] += heh;
                let mut mm = eta.to_vec();
                mm[k] -= heh;
                mm[l] -= heh;
                let hh = (pred(&pp, &theta, j) - pred(&pm, &theta, j) - pred(&mp, &theta, j)
                    + pred(&mm, &theta, j))
                    / (4.0 * heh * heh);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * n_eta + l],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }

        // ∂f/∂θ
        for m in 0..n_theta {
            let mut tp = theta.to_vec();
            tp[m] += ht * (1.0 + theta[m].abs());
            let mut tm = theta.to_vec();
            tm[m] -= ht * (1.0 + theta[m].abs());
            let step = ht * (1.0 + theta[m].abs());
            let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * step);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-4, epsilon = 1e-7);
        }

        // ∂²f/∂η∂θ (mixed 4-point)
        for k in 0..n_eta {
            for m in 0..n_theta {
                let s = heh * (1.0 + theta[m].abs());
                let mut ep = eta.to_vec();
                ep[k] += heh;
                let mut em = eta.to_vec();
                em[k] -= heh;
                let mut tp = theta.to_vec();
                tp[m] += s;
                let mut tm = theta.to_vec();
                tm[m] -= s;
                let hh = (pred(&ep, &tp, j) - pred(&ep, &tm, j) - pred(&em, &tp, j)
                    + pred(&em, &tm, j))
                    / (4.0 * heh * s);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * n_theta + m],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }
    }
}

/// Provider's exact η/θ sensitivities must match central finite differences
/// of the production predictor `compute_predictions_with_tv`.
#[test]
fn provider_matches_fd_of_production_predictor() {
    let model = parse_model_string(WARFARIN).expect("parse");
    let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    check_full_provider_vs_fd(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);
}

// ── #860 Phase A6: closed-form MR analytic gradient ──────────────────────────
//
// `subject_sensitivities`/`subject_eta_grad` now try the no-integration MR
// superposition before the ODE provider for the static subjects `mr_scope`
// admits. Each test below both (a) confirms the fast path actually fired —
// `mr_subject_sensitivities` returning `Some` directly — so a scope regression
// that silently falls back to the (still-correct, just slower) ODE provider
// would be caught, and (b) exercises the full FD-parity / cross-provider
// agreement check through the same public entry points production uses.

const MR_MIXED_1CPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.5, 0.05, 24.0)
  theta TVDUR(2.0, 0.1, 12.0)
  theta TVLAG(1.0, 0.001, 6.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  FZO = TVFZO
  FZO1 = 1 - TVFZO
  KA = TVKA
  DUR = TVDUR
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA, lag=LAG) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

const MR_PARALLEL_2CPT: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(10.0, 0.1, 100.0)
  theta TVV2(100.0, 5.0, 1000.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central, periph])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V1*central - Q/V1*central + Q/V2*periph
  d/dt(periph) = Q/V1*central - Q/V2*periph
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// 1-cpt mixed `first_order` (per-route-lagged) + `zero_order`: the closed-form
/// outer/inner MR gradient must match central FD of the production predictor,
/// via the SAME public `subject_sensitivities` entry point production uses —
/// and must actually take the MR fast path, not silently fall back to ODE.
#[test]
fn mr_1cpt_mixed_provider_matches_fd_and_takes_fast_path() {
    let model = parse_model_string(MR_MIXED_1CPT).expect("parse");
    // Deliberately avoids exact coincidence with TVDUR (2.0) / TVLAG (1.0) —
    // at t == dur or t == lag exactly, central-FD-of-θ perturbs the boundary
    // past a *fixed* observation time, mixing two branches (a secant across a
    // kink, not a one-sided derivative); see the kernel-level
    // `zero_order_dual2_matches_fd_near_dur_boundary` test for the isolated
    // case. No real observation time lands exactly on a fitted dur/lag either.
    let subject = oral_subject(&[0.5, 1.3, 2.7, 3.6, 4.4, 8.2, 12.0]);
    let theta = [5.0, 50.0, 0.4, 1.5, 2.0, 1.0];
    let eta = [0.1, -0.05];
    assert!(
        crate::pk::modified_release::mr_subject_sensitivities(&model, &subject, &theta, &eta)
            .is_some(),
        "expected the MR closed-form fast path to admit this static 1-cpt mixed subject"
    );
    check_full_provider_vs_fd(&model, &subject, &theta, &eta);
}

/// 2-cpt parallel `first_order` routes: exercises the macro-rate (`α`/`β`) and
/// `v2 = q/k21` division inside `recover_disp_params_g` under `Dual2` — the
/// riskiest single spot for the outer chain (a reciprocal/quotient under a
/// second-order dual) — via the same public entry point.
#[test]
fn mr_2cpt_parallel_provider_matches_fd_and_takes_fast_path() {
    let model = parse_model_string(MR_PARALLEL_2CPT).expect("parse");
    let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]);
    let theta = [5.0, 50.0, 10.0, 100.0, 0.6, 1.5, 0.3];
    let eta = [0.1];
    assert!(
        crate::pk::modified_release::mr_subject_sensitivities(&model, &subject, &theta, &eta)
            .is_some(),
        "expected the MR closed-form fast path to admit this static 2-cpt parallel subject"
    );
    check_full_provider_vs_fd(&model, &subject, &theta, &eta);
}

/// The MR closed-form jet and the ODE-integrated jet are two different schemes
/// computing the same "analytic" gradient for the same subject
/// (`ode_analytical_supported` already reports this model analytic;
/// `mr_scope` is a narrower, faster alternative for it) — they must agree to
/// solver tolerance, directly, not just both agreeing with FD (FD's own
/// tolerance is far looser than solver `ode_reltol`/`ode_abstol`, so this is
/// the tighter regression guard against the two scopes silently drifting apart).
#[test]
fn mr_subject_sensitivities_matches_ode_provider_directly() {
    let model = parse_model_string(MR_MIXED_1CPT).expect("parse");
    let subject = oral_subject(&[0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0]);
    let theta = [5.0, 50.0, 0.4, 1.5, 2.0, 1.0];
    let eta = [0.1, -0.05];
    let mr = crate::pk::modified_release::mr_subject_sensitivities(&model, &subject, &theta, &eta)
        .expect("MR fast path admits this subject");
    let ode = crate::sens::ode_provider::ode_subject_sensitivities(&model, &subject, &theta, &eta)
        .expect("ODE provider also serves this subject analytically");
    assert_eq!(mr.obs.len(), ode.obs.len());
    for (j, (a, b)) in mr.obs.iter().zip(&ode.obs).enumerate() {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-6, epsilon = 1e-9);
        for k in 0..a.df_deta.len() {
            approx::assert_relative_eq!(
                a.df_deta[k],
                b.df_deta[k],
                max_relative = 1e-4,
                epsilon = 1e-8
            );
        }
        for m in 0..a.df_dtheta.len() {
            approx::assert_relative_eq!(
                a.df_dtheta[m],
                b.df_dtheta[m],
                max_relative = 1e-4,
                epsilon = 1e-8
            );
        }
        for idx in 0..a.d2f_deta2.len() {
            approx::assert_relative_eq!(
                a.d2f_deta2[idx],
                b.d2f_deta2[idx],
                max_relative = 1e-3,
                epsilon = 1e-6
            );
        }
        for idx in 0..a.d2f_deta_dtheta.len() {
            approx::assert_relative_eq!(
                a.d2f_deta_dtheta[idx],
                b.d2f_deta_dtheta[idx],
                max_relative = 1e-3,
                epsilon = 1e-6
            );
        }
        let _ = j;
    }
}

/// Inner light `Dual1` η-gradient counterpart: `mr_subject_eta_grad` must match
/// `ode_subject_eta_grad` too — the shared per-subject scope contract requires
/// inner and outer to serve exactly the same subjects, so their `f`/`df_deta`
/// values (which both must ALSO match the outer path's) are checked directly
/// against the ODE inner provider.
#[test]
fn mr_subject_eta_grad_matches_ode_provider_directly() {
    let model = parse_model_string(MR_MIXED_1CPT).expect("parse");
    let subject = oral_subject(&[0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0]);
    let theta = [5.0, 50.0, 0.4, 1.5, 2.0, 1.0];
    let eta = [0.1, -0.05];
    let mr = crate::pk::modified_release::mr_subject_eta_grad(&model, &subject, &theta, &eta)
        .expect("MR fast path admits this subject");
    let ode = crate::sens::ode_provider::ode_subject_eta_grad(&model, &subject, &theta, &eta)
        .expect("ODE provider also serves this subject analytically");
    assert_eq!(mr.len(), ode.len());
    for (a, b) in mr.iter().zip(&ode) {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-6, epsilon = 1e-9);
        for k in 0..a.df_deta.len() {
            approx::assert_relative_eq!(
                a.df_deta[k],
                b.df_deta[k],
                max_relative = 1e-4,
                epsilon = 1e-8
            );
        }
    }
}

// Regression: an ODE model may compose a Form-C `y = <expr>` readout WITH a
// separate `obs_scale = <const>` divisor on top (the parser explicitly allows
// this for ODE models — "ODE keeps its historical behaviour" — unlike
// analytical models, where it is rejected). `ode_subject_sensitivities`
// applies that divisor per observation via `apply_output_transform`, inside
// `resolve_obs_readout` — a site `mr_scope` does not go through at all. An
// earlier version of the MR fast path had no equivalent step, so it silently
// returned the *unscaled* jet for any `ScalarScale`-composed MR model instead
// of either applying the divisor or declining to the ODE provider.
const MR_MIXED_1CPT_SCALAR_SCALE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.5, 0.05, 24.0)
  theta TVDUR(2.0, 0.1, 12.0)
  theta TVLAG(1.0, 0.001, 6.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  FZO = TVFZO
  FZO1 = 1 - TVFZO
  KA = TVKA
  DUR = TVDUR
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA, lag=LAG) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
  obs_scale = 2.5
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn mr_subject_sensitivities_with_scalar_scale_matches_ode_provider() {
    let model = parse_model_string(MR_MIXED_1CPT_SCALAR_SCALE).expect("parse");
    let subject = oral_subject(&[0.5, 1.3, 2.7, 3.6, 4.4, 8.2, 12.0]);
    let theta = [5.0, 50.0, 0.4, 1.5, 2.0, 1.0];
    let eta = [0.1, -0.05];
    let mr = crate::pk::modified_release::mr_subject_sensitivities(&model, &subject, &theta, &eta)
        .expect("MR fast path must admit a ScalarScale-composed model, not silently misfire");
    let ode = crate::sens::ode_provider::ode_subject_sensitivities(&model, &subject, &theta, &eta)
        .expect("ODE provider also serves this subject analytically");
    assert_eq!(mr.obs.len(), ode.obs.len());
    for (a, b) in mr.obs.iter().zip(&ode.obs) {
        approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-6, epsilon = 1e-9);
        for k in 0..a.df_deta.len() {
            approx::assert_relative_eq!(
                a.df_deta[k],
                b.df_deta[k],
                max_relative = 1e-4,
                epsilon = 1e-8
            );
        }
        for m in 0..a.df_dtheta.len() {
            approx::assert_relative_eq!(
                a.df_dtheta[m],
                b.df_dtheta[m],
                max_relative = 1e-4,
                epsilon = 1e-8
            );
        }
    }
    check_full_provider_vs_fd(&model, &subject, &theta, &eta);
}

// Regression: `PerCmt` scaling is the one `ScalingSpec` variant
// `ode_analytical_supported` declines outright (routes to FD) — `mr_scope`
// must decline it too via the same `ode_scaling_supported` gate, not silently
// admit it with a plain (wrong, unscaled-per-CMT) jet.
#[test]
fn mr_subject_sensitivities_declines_percmt_scaling() {
    const MR_PERCMT_SCALE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.5, 0.05, 24.0)
  theta TVDUR(2.0, 0.1, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  FZO = TVFZO
  FZO1 = 1 - TVFZO
  KA = TVKA
  DUR = TVDUR
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central
[scaling]
  y = central / V
  obs_scale[CMT=1] = 2.5
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = parse_model_string(MR_PERCMT_SCALE).expect("parse");
    let subject = oral_subject(&[0.5, 1.3, 2.7]);
    let theta = [5.0, 50.0, 0.4, 1.5, 2.0];
    let eta = [0.1];
    assert!(
        crate::pk::modified_release::mr_subject_sensitivities(&model, &subject, &theta, &eta)
            .is_none(),
        "PerCmt scaling must decline the MR fast path, matching ode_scaling_supported"
    );
    assert!(
        crate::pk::modified_release::mr_subject_eta_grad(&model, &subject, &theta, &eta).is_none(),
        "PerCmt scaling must decline the MR inner fast path too"
    );
}

// ── analytic Form C readout (#650) exact sensitivities ───────────────────

/// A nonlinear analytic Form C readout — a saturable protein-binding total
/// concentration `y = C + BMAX·C/(KD + C)` with `C = central/V` — must be
/// differentiated exactly by the provider: value, `∂/∂η`, `∂²/∂η²`, `∂/∂θ`,
/// and `∂²/∂η∂θ` all match central FD of the readout-aware production
/// predictor. The readout carries η through both `C` (via CL, V) and the
/// `central/V` amount→conc map, and θ through BMAX/KD/CL/V. The `.expect`
/// inside the harness also asserts the analytic path is taken (not FD).
const ONECPT_IV_BINDING_READOUT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + BMAX * (central / V) / (KD + central / V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn form_c_binding_readout_provider_matches_fd() {
    let m = parse_model_string(ONECPT_IV_BINDING_READOUT).expect("parse binding readout");
    // Must be in the analytic Dual2 scope (not routed to FD).
    assert!(
        analytical_supported(&m),
        "central-only dual-evaluable Form C readout must stay analytic"
    );
    let s = subject_with_dose(
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        &[0.5, 2.0, 6.0, 12.0],
    );
    let theta = [0.2, 10.0, 3.0, 2.0];
    let eta = [0.12, -0.08];
    check_full_provider_vs_fd(&m, &s, &theta, &eta);

    // Light inner η-gradient must agree with the full provider's η block.
    let full = subject_sensitivities(&m, &s, &theta, &eta).expect("supported");
    // The θ-gradient for the non-structural BMAX (θ index 2) and KD (θ index 3)
    // must be non-zero — proving they are first-class differentiable params
    // (#650 basis extension), not aliased onto CL. Before the fix they read the
    // CL slot, so ∂y/∂θ_BMAX / ∂y/∂θ_KD were identically zero.
    let max_bmax = full
        .obs
        .iter()
        .map(|o| o.df_dtheta[2].abs())
        .fold(0.0_f64, f64::max);
    let max_kd = full
        .obs
        .iter()
        .map(|o| o.df_dtheta[3].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_bmax > 1e-6,
        "∂y/∂BMAX must be non-zero (BMAX is a real readout param, not aliased to CL)"
    );
    assert!(
        max_kd > 1e-6,
        "∂y/∂KD must be non-zero (KD is a real readout param, not aliased to CL)"
    );
    let light = subject_eta_grad(&m, &s, &theta, &eta).expect("light supported");
    assert_eq!(light.len(), full.obs.len());
    for (lo, fo) in light.iter().zip(full.obs.iter()) {
        for k in 0..m.n_eta {
            approx::assert_relative_eq!(
                lo.df_deta[k],
                fo.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

/// The fluconazole case: a readout gated on a **per-row** covariate (`FREE`)
/// makes the subject a time-varying-covariate subject, so it routes to the
/// event-walk provider. The readout is served analytically there too (#650):
/// value + all η/θ first/second derivatives match FD of the (event-walk)
/// production predictor, and BMAX/KD (non-structural, in their allocated slots)
/// stay differentiable through the walk.
#[test]
fn form_c_binding_readout_tvcov_matches_fd() {
    let m = parse_model_string(ONECPT_IV_BINDING_READOUT_FREE).expect("parse");
    let mut s = subject_with_dose(
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        &[0.5, 2.0, 6.0, 12.0],
    );
    // Alternating per-row FREE flag → per-observation covariate snapshots, so
    // the subject routes to the event-walk (TV-cov) provider.
    s.obs_covariates = vec![
        HashMap::from([("FREE".to_string(), 0.0)]),
        HashMap::from([("FREE".to_string(), 1.0)]),
        HashMap::from([("FREE".to_string(), 0.0)]),
        HashMap::from([("FREE".to_string(), 1.0)]),
    ];
    assert!(s.has_tv_covariates(), "fixture must be a TV-cov subject");
    assert!(
        readout_tvcov_supported(&m),
        "central-only binding readout fits PkDual slots"
    );
    let theta = [0.2, 10.0, 3.0, 2.0];
    let eta = [0.12, -0.08];
    check_full_provider_vs_fd(&m, &s, &theta, &eta);
}

const ONECPT_IV_BINDING_READOUT_FREE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = if (FREE == 0) central / V + BMAX * (central / V) / (KD + central / V) else central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// An **oral** Form C readout `y = central / V` (central at state slot 1,
/// depot slot 0 unreferenced) must also differentiate exactly.
#[test]
fn form_c_oral_readout_provider_matches_fd() {
    let src = WARFARIN.replace(
        "[error_model]",
        "[scaling]\n  y = central / V\n[error_model]",
    );
    let m = parse_model_string(&src).expect("parse oral readout");
    assert!(
        analytical_supported(&m),
        "oral central-only readout stays analytic"
    );
    let s = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]);
    check_full_provider_vs_fd(&m, &s, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);
}

/// A readout that references the oral **depot** amount is out of the static
/// jet's scope (the depot amount isn't reconstructed as a dual), so the model
/// routes to the FD gradient — `analytical_supported` reports that honestly.
#[test]
fn form_c_depot_readout_routes_to_fd() {
    let src = WARFARIN.replace(
        "[error_model]",
        "[scaling]\n  y = central / V + depot / V\n[error_model]",
    );
    let m = parse_model_string(&src).expect("parse depot readout");
    assert!(
        !analytical_supported(&m),
        "a depot-referencing analytic readout must fall back to FD (no dual depot amount)"
    );
    // The FD path still predicts correctly (readout applied in the f64 predictor).
    let s = oral_subject(&[1.0, 4.0]);
    let preds = compute_predictions_with_tv(&m, &s, &[0.2, 10.0, 1.5], &[0.0, 0.0, 0.0]);
    assert!(preds.iter().all(|p| p.is_finite()));
}

/// #650 review: a readout whose **non-structural** parameter depends on a
/// time-varying covariate (`BMAX = TVBMAX·WT/70`, WT per row) must be read
/// **per observation** by the f64 predictor, matching the per-observation
/// snapshot the event-walk provider differentiates. Before the fix the
/// predictor froze `BMAX` at the t=0 covariate while the analytic gradient used
/// the per-row value, so `check_full_provider_vs_fd` (analytic vs FD-of-predictor)
/// diverged. The `.expect` inside the harness also asserts the analytic path is
/// taken (BMAX→slot 2, KD→slot 3, both ≤ V3, so `readout_tvcov_supported`).
#[test]
fn form_c_tvcov_readout_param_matches_fd() {
    const SRC: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX * (WT / 70)
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + BMAX * (central / V) / (KD + central / V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(SRC).expect("parse tv-cov readout param");
    let mut s = subject_with_dose(
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        &[0.5, 2.0, 6.0, 12.0],
    );
    // Per-row body weight → time-varying covariate → event-walk provider, and a
    // readout parameter (BMAX) that actually moves between rows.
    s.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 60.0)]),
        HashMap::from([("WT".to_string(), 80.0)]),
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 95.0)]),
    ];
    assert!(s.has_tv_covariates(), "fixture must be a TV-cov subject");
    assert!(readout_tvcov_supported(&m), "BMAX/KD fit the PkDual slots");
    check_full_provider_vs_fd(&m, &s, &[0.2, 10.0, 3.0, 2.0], &[0.12, -0.08]);
}

/// #822 review #1 (outer): a readout that references a non-structural individual
/// parameter which is itself a `(θ, η)`-*constant* (`HILL = 2.0`, no theta/eta on
/// its RHS) — forced past `PK_IDX_V3` into a genuinely non-structural slot by
/// referencing enough other non-structural params first to exhaust the low free
/// slots (`allocate_readout_extra_slots` hands out the free `one_cpt_iv` slots
/// `[Q, V2, KA, Q3, V3]` before spilling to slot 9) — must still read HILL's real
/// *value* on the TV-cov walk, not substitute zero.
///
/// This pins the invariant review finding #822-#1 hinged on: the parser slots
/// *every* readout-referenced individual parameter into `pk_assignment_mapping`
/// regardless of whether its value is differentiated (`slot_row[s]` is set purely
/// by name resolution, not by whether the row's gradient is nonzero), so a
/// parameter's own slot is always a member of `ro_slots` whenever the readout
/// actually reads it — `ro_slots` cannot be empty while a live high-slot read
/// exists — confirmed by instrumenting `ro_slots` under this exact fixture: it is
/// `[9]` (HILL's spilled slot), never empty, so `ro_extra` always takes the branch
/// with the correct `pk.values[9] = 2.0` fallback — the finding's premise (a
/// constant param read while `ro_slots` is empty) does not occur.
#[test]
fn form_c_tvcov_readout_constant_param_matches_fd() {
    const SRC: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVP1(1.0, 0.01, 100.0)
  theta TVP2(1.0, 0.01, 100.0)
  theta TVP3(1.0, 0.01, 100.0)
  theta TVP4(1.0, 0.01, 100.0)
  theta TVP5(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  P1   = TVP1
  P2   = TVP2
  P3   = TVP3
  P4   = TVP4
  P5   = TVP5
  HILL = 2.0
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = (P1 + P2 + P3 + P4 + P5) * 0.0 + HILL * (central / V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let m = parse_model_string(SRC).expect("parse tv-cov constant readout param");
    let mut s = subject_with_dose(
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        &[0.5, 2.0, 6.0, 12.0],
    );
    // A per-row covariate (unreferenced by any individual parameter) is enough to
    // route the subject onto the event-driven TV-cov walk (`has_tv_covariates` is a
    // subject-level property); it need not be the thing that makes HILL constant.
    s.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 60.0)]),
        HashMap::from([("WT".to_string(), 80.0)]),
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 95.0)]),
    ];
    assert!(s.has_tv_covariates(), "fixture must be a TV-cov subject");
    assert!(
        readout_tvcov_supported(&m),
        "P1..P5/HILL fit the PkDual overflow slots"
    );
    let theta = [0.2, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let eta = [0.12, -0.08];
    let sens = subject_sensitivities(&m, &s, &theta, &eta).expect("supported");
    assert!(
        sens.obs.iter().all(|o| o.f != 0.0),
        "HILL must not be silently zeroed by the outer readout substitution"
    );
    check_full_provider_vs_fd(&m, &s, &theta, &eta);
}

/// #650 review: a readout that references the oral **depot** amount is rejected
/// on a subject carrying an EVID=3/4 reset (superposition can't restart the
/// depot across the reset), rather than silently reading a zero depot. A
/// `central`-only readout on the same reset subject is accepted (the central
/// concentration is already reset-correct).
#[test]
fn depot_readout_with_reset_is_rejected() {
    let depot_src = WARFARIN.replace(
        "[error_model]",
        "[scaling]\n  y = central / V + depot / V\n[error_model]",
    );
    let depot_m = parse_model_string(&depot_src).expect("parse depot readout");
    assert!(
        depot_m
            .analytic_readout
            .as_ref()
            .expect("analytic readout")
            .references_depot(),
        "readout must be detected as depot-referencing"
    );
    let reset_subj = subject_with_doses_and_resets(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[1.0, 4.0, 8.0],
        vec![5.0],
    );
    let pop = crate::types::Population {
        subjects: vec![reset_subj],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    assert!(
        crate::api::check_analytic_readout_support(&depot_m, &pop).is_some(),
        "depot readout + reset subject must be rejected"
    );

    // Central-only readout is fine on the very same reset subject.
    let central_src = WARFARIN.replace(
        "[error_model]",
        "[scaling]\n  y = central / V\n[error_model]",
    );
    let central_m = parse_model_string(&central_src).expect("parse central readout");
    assert!(
        crate::api::check_analytic_readout_support(&central_m, &pop).is_none(),
        "central-only readout must be accepted on a reset subject"
    );
}

/// Regression for #455/#456: an analytical model whose `[individual_parameters]`
/// block has **intermediate** assignments before the structural PK outputs must
/// drive the exact program-based sensitivity path even on a **static-covariate**
/// subject (the non-TV `subject_sensitivities` provider). Before the fix, those
/// gates compared `prog.pk_slots().len() == model.pk_indices.len()`; the
/// intermediate rows make `pk_indices` longer, so the gate rejected the program
/// path and fell back to the log-normal closed form keyed by `pk_indices`, whose
/// slot-0-aliased intermediate rows overwrote CL's seed and silently zeroed
/// `∂f/∂η_CL`. With the gate keyed on the required structural slots, the program
/// path runs and matches FD exactly.
#[test]
fn static_cov_intermediate_params_uses_program_path_and_matches_fd() {
    let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
    let m =
        parse_model_string(TWOCPT_IV_TVCOV_INTERMEDIATE).expect("parse 2cpt iv tvcov intermediate");

    // Static WT (a single `subject.covariates` snapshot, no per-event covariate
    // vectors) → NOT a TV-covariate subject, so this routes through the non-TV
    // `subject_sensitivities` / `subject_eta_grad` gates that the fix repaired.
    let mut s = subject_with_dose(bolus(0.0), &[0.5, 2.0, 6.0, 12.0]);
    s.covariates = wt_map(70.0);
    assert!(
        !s.has_tv_covariates(),
        "fixture must be a static-covariate subject so the non-TV provider runs"
    );

    // The fixture genuinely exposes intermediate rows (the precondition that the
    // old `len == len` gate tripped on), and the repaired gate admits it.
    let prog = m
        .indiv_param_partials
        .indiv_param_program
        .as_ref()
        .expect("compiled individual program");
    assert!(
        m.pk_indices.len() > prog.pk_slots().len(),
        "fixture must expose intermediate individual-parameter rows"
    );
    assert!(
        prog_covers_required_pk_slots(&m, prog),
        "repaired gate must admit the intermediate-parameter program path"
    );

    let theta = vec![10.0, 50.0, 15.0, 100.0, 0.75];
    let eta = vec![0.12, -0.08];

    // Sanity: the η_CL gradient must be non-zero — the exact symptom the old
    // mis-seeded fallback produced was a zeroed CL gradient.
    let sens = subject_sensitivities(&m, &s, &theta, &eta).expect("supported");
    let max_cl_grad = sens
        .obs
        .iter()
        .map(|o| o.df_deta[0].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_cl_grad > 1e-6,
        "∂f/∂η_CL must be non-zero (was silently zeroed by the mis-seeded fallback)"
    );

    // Full provider (and, via the harness's reference, the production predictor)
    // must match central finite differences exactly.
    check_full_provider_vs_fd(&m, &s, &theta, &eta);

    // The light inner η-gradient provider must agree with the full provider too.
    let light = subject_eta_grad(&m, &s, &theta, &eta).expect("light supported");
    assert_eq!(light.len(), sens.obs.len());
    for (lo, fo) in light.iter().zip(sens.obs.iter()) {
        for k in 0..m.n_eta {
            approx::assert_relative_eq!(
                lo.df_deta[k],
                fo.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

/// LTBS (`log(DV) ~ additive(...)`): the production predictor returns `ln(f)`,
/// and the provider applies the matching `g = ln(f)` jet transform. The full
/// value/gradient/Hessian must still match FD of the (log-scale) production
/// predictor, and the light η-provider must agree with the full one — over
/// 1-/2-/3-cpt so the second-order `g_kl = f_kl/f − f_k·f_l/f²` chain is covered.
#[test]
fn provider_ltbs_matches_production() {
    let ltbs = |src: &str| {
        src.replace(
            "[error_model]\n  DV ~ proportional(PROP_ERR)",
            "[error_model]\n  log(DV) ~ additive(PROP_ERR)",
        )
    };
    let times = [0.25, 1.0, 4.0, 12.0];
    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        {
            let m = parse_model_string(&WARFARIN.replace(
                "[error_model]\n  DV ~ proportional(PROP_ERR)",
                "[error_model]\n  log(DV) ~ additive(PROP_ERR)",
            ))
            .expect("parse warfarin LTBS");
            assert!(m.log_transform, "LTBS flag must be set");
            (
                m,
                oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0, 24.0]),
                vec![0.2, 10.0, 1.5],
                vec![0.15, -0.10, 0.25],
            )
        },
        {
            let m = parse_model_string(&ltbs(TWOCPT_IV)).expect("parse 2cpt LTBS");
            let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0), &times);
            (m, s, vec![10.0, 50.0, 15.0, 100.0], vec![0.12, -0.08])
        },
        {
            let m = parse_model_string(&ltbs(THREECPT_ORAL)).expect("parse 3cpt LTBS");
            let s = subject_with_dose(DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 24.0), &times);
            (
                m,
                s,
                vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5],
                vec![0.12, -0.08, 0.2],
            )
        },
    ];
    for (m, s, theta, eta) in &cases {
        assert!(analytical_supported(m), "LTBS must be provider-supported");
        check_full_provider_vs_fd(m, s, theta, eta);

        // Light η-provider must equal the full provider's log-scale f and ∂g/∂η.
        let full = subject_sensitivities(m, s, theta, eta).expect("full");
        let light = subject_eta_grad(m, s, theta, eta).expect("light");
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
            for k in 0..m.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-12,
                    epsilon = 1e-14
                );
            }
        }
    }
}

/// **LTBS combined with time-varying covariates** (#486). LTBS on a TV-cov subject
/// previously routed BOTH loops to FD; the outer θ/Ω/σ gradient is now analytic on the
/// event-driven walk (`subject_sensitivities_tvcov` applies the shared post-walk `ln f`
/// transform LAST, after scaling — the same helper the dose-superposition outer uses).
/// Validated over plain LTBS and LTBS + an `ExpressionScale` `obs_scale = 1000/V` divisor
/// (production's scale-then-log order `ln(f/s)` is reproduced post-walk) against central
/// FD of the log-scale production predictor. The inner EBE gradient stays on FD for LTBS
/// (covariance stability), asserted below.
#[test]
fn ltbs_tvcov_outer_matches_production() {
    let ltbs = |src: &str| {
        src.replace(
            "[error_model]\n  DV ~ proportional(PROP_ERR)",
            "[error_model]\n  log(DV) ~ additive(PROP_ERR)",
        )
    };
    // ExpressionScale (`obs_scale = 1000/V`) + LTBS + TV-cov: exercises scale-then-log.
    const ONECPT_ORAL_TVCOV_EXPR: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = 1000 / V
[covariates]
  WT continuous
[error_model]
  log(DV) ~ additive(PROP_ERR)
"#;
    let subject = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[70.0],
        &[1.0, 2.0, 4.0, 8.0, 24.0],
        &[70.0, 72.0, 80.0, 85.0, 90.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    let theta = vec![0.2, 10.0, 1.5, 0.75];
    let eta = vec![0.15, -0.10, 0.25];

    for src in [ltbs(ONECPT_ORAL_TVCOV), ONECPT_ORAL_TVCOV_EXPR.to_string()] {
        let model = parse_model_string(&src).expect("parse LTBS TV-cov");
        assert!(model.log_transform && subject.has_tv_covariates());
        assert!(
            tvcov_analytical_supported(&model),
            "LTBS + TV-cov must be on the analytic OUTER path now (#486)"
        );
        // Outer analytic vs FD of the log-scale production predictor.
        check_full_provider_vs_fd(&model, &subject, &theta, &eta);
        // Inner LTBS + TV-cov is now analytic too (Tier-1 follow-up): the light inner
        // provider's η-gradient must equal the full outer provider's df_deta η-block.
        let full = subject_sensitivities(&model, &subject, &theta, &eta).expect("outer");
        let light = subject_eta_grad(&model, &subject, &theta, &eta).expect("inner LTBS + TV-cov");
        assert_eq!(full.obs.len(), light.len());
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-12);
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-10,
                    epsilon = 1e-12
                );
            }
        }
    }
}

// 1-cpt oral with a log-normal dose lagtime (`LAGTIME = TVLAG·exp(ETA_LAG)`).
const ONECPT_ORAL_LAG: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The exact ODE twin of [`ONECPT_ORAL_LAG`] (depot → central, `LAGTIME` on the
/// depot bolus). Same θ/Ω/σ and parameterization, so the ODE provider's full
/// `SubjectSens` (value, `∂f/∂η`, `∂f/∂θ`, and the **2nd-order** blocks) must equal
/// the closed-form analytical provider's to RK45 accuracy — validating the
/// event-time (saltation) sensitivity, including its Hessian, against an independent
/// path (#439 lagtime).
const ONECPT_ORAL_LAG_ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;

#[test]
fn ode_lagtime_full_sens_matches_analytical_twin() {
    use crate::types::DoseEvent;
    let ana = parse_model_string(ONECPT_ORAL_LAG).expect("parse analytical oral lag");
    let ode = parse_model_string(ONECPT_ORAL_LAG_ODE_TWIN).expect("parse ODE oral lag");
    assert!(
        analytical_supported(&ana),
        "analytical lag must be supported"
    );
    assert!(
        crate::sens::ode_provider::ode_analytical_supported(&ode),
        "ODE bare-lag must be supported"
    );
    let n = 6usize;
    let subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0],
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    // θ = [TVCL, TVV, TVKA, TVLAG]; η = [ETA_CL, ETA_V, ETA_KA, ETA_LAG] (lag carries IIV).
    let theta = [0.2, 10.0, 1.5, 0.75];
    let eta = [0.1, -0.05, 0.15, 0.08];
    let a = subject_sensitivities(&ana, &subject, &theta, &eta).expect("analytical sens supported");
    let o = crate::sens::ode_provider::ode_subject_sensitivities(&ode, &subject, &theta, &eta)
        .expect("ODE sens supported");
    assert_eq!(a.obs.len(), o.obs.len());
    for (oa, oo) in a.obs.iter().zip(o.obs.iter()) {
        approx::assert_relative_eq!(oa.f, oo.f, max_relative = 1e-6, epsilon = 1e-9);
        for (x, y) in oa.df_deta.iter().zip(oo.df_deta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-8);
        }
        for (x, y) in oa.df_dtheta.iter().zip(oo.df_dtheta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-8);
        }
        for (x, y) in oa.d2f_deta2.iter().zip(oo.d2f_deta2.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-4, epsilon = 1e-7);
        }
        for (x, y) in oa.d2f_deta_dtheta.iter().zip(oo.d2f_deta_dtheta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-4, epsilon = 1e-7);
        }
    }
}

// 1-cpt IV bolus with a log-normal lagtime (`alag=` alias, IV route).
const ONECPT_IV_LAG: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVLAG(1.0, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V, alag=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 2-cpt oral with a log-normal lagtime.
const TWOCPT_ORAL_LAG: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta TVKA(1.0, 0.05, 20.0)
  theta TVLAG(0.6, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V1, q=Q, v2=V2, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Dose lagtime is now a differentiated PK slot: it enters every dose through
/// the elapsed-time argument (`∂elapsed/∂lagtime = −1`), seeded as its own dual
/// axis. The provider's exact value/∂η/∂²η/∂θ/∂η∂θ must match FD of the
/// production predictor for IV bolus, 1-/2-cpt oral, and — crucially — a
/// steady-state oral dose with an observation in the pre-arrival window
/// `[dose.time, dose.time + lagtime)`, which exercises the SS tail wrap.
#[test]
fn provider_lagtime_matches_production() {
    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        {
            let m = parse_model_string(ONECPT_ORAL_LAG).expect("parse 1cpt oral lag");
            assert!(m.has_lagtime(), "model must carry a lagtime");
            (
                m,
                oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]),
                vec![0.2, 10.0, 1.5, 0.75],
                vec![0.15, -0.10, 0.25, 0.12],
            )
        },
        {
            let m = parse_model_string(ONECPT_IV_LAG).expect("parse 1cpt iv lag");
            let s = subject_with_dose(
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                &[1.5, 3.0, 6.0, 12.0],
            );
            (m, s, vec![10.0, 50.0, 1.0], vec![0.1, -0.05, 0.2])
        },
        {
            let m = parse_model_string(TWOCPT_ORAL_LAG).expect("parse 2cpt oral lag");
            (
                m,
                oral_subject(&[1.0, 2.0, 6.0, 12.0, 24.0]),
                vec![10.0, 50.0, 15.0, 100.0, 1.0, 0.6],
                vec![0.12, -0.08, 0.15, 0.1],
            )
        },
        {
            // Steady-state oral with an observation at t=0.5 inside the
            // pre-arrival window (lagtime ≈ 0.8 h): the SS tail wrap branch.
            let m = parse_model_string(ONECPT_ORAL_LAG).expect("parse 1cpt oral lag ss");
            let s = subject_with_dose(
                DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
                &[0.5, 2.0, 6.0, 12.0, 23.0],
            );
            (
                m,
                s,
                vec![0.2, 10.0, 1.5, 0.75],
                vec![0.15, -0.10, 0.25, 0.12],
            )
        },
    ];
    for (m, s, theta, eta) in &cases {
        assert!(
            analytical_supported(m),
            "lagtime must be provider-supported"
        );
        check_full_provider_vs_fd(m, s, theta, eta);

        // Light η-provider must equal the full provider's f and ∂f/∂η.
        let full = subject_sensitivities(m, s, theta, eta).expect("full");
        let light = subject_eta_grad(m, s, theta, eta).expect("light");
        for (fo, lo) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-10, epsilon = 1e-12);
            for k in 0..m.n_eta {
                approx::assert_relative_eq!(
                    fo.df_deta[k],
                    lo.df_deta[k],
                    max_relative = 1e-9,
                    epsilon = 1e-11
                );
            }
        }
    }
}

/// Reset + lagtime: a dose recorded *before* a reset but *arriving after* it
/// (via lagtime) must contribute to the post-reset segment, exactly as the
/// production event-driven walk applies it. The reset exclusion keys on the
/// lagged arrival `dose.time + lag`, not the record time (PR #381 review #2).
/// Dose at t=4 with lag≈0.75 arrives ≈4.75, past the reset at t=4.5; the
/// earlier t=0 dose (arrives ≈0.75) is correctly washed out. Validated against
/// `compute_predictions_with_tv` via `check_full_provider_vs_fd` (value 1e-9).
#[test]
fn provider_reset_with_lagged_post_reset_dose_matches_production() {
    let m = parse_model_string(ONECPT_ORAL_LAG).expect("parse 1cpt oral lag");
    let s = subject_with_doses_and_resets(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(4.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[5.0, 6.0, 8.0, 12.0],
        vec![4.5],
    );
    // eta_lag = 0 → LAGTIME = TVLAG = 0.75; arrival of the t=4 dose is 4.75 > 4.5.
    let theta = vec![0.2, 10.0, 1.5, 0.75];
    let eta = vec![0.1, -0.05, 0.2, 0.0];
    check_full_provider_vs_fd(&m, &s, &theta, &eta);
}

/// `[scaling] obs_scale = 1000 / V` with `V = TVV·exp(ETA_V)`: an
/// η/θ-dependent `ExpressionScale`. The provider divides the whole jet by the
/// scale via the differentiable scale program (quotient rule), so its exact
/// value/∂η/∂²η/∂θ/∂²η∂θ must match FD of the production predictor (which
/// applies the same scale through `apply_scaling`).
#[test]
fn provider_expression_scale_matches_production() {
    let src = WARFARIN.replace(
        "[error_model]\n  DV ~ proportional(PROP_ERR)",
        "[error_model]\n  DV ~ proportional(PROP_ERR)\n[scaling]\n  obs_scale = 1000 / V",
    );
    let model = parse_model_string(&src).expect("scaling model parses");
    assert!(
        matches!(
            model.scaling,
            ScalingSpec::ExpressionScale { deriv: Some(_), .. }
        ),
        "model must carry a differentiable scale program"
    );
    assert!(
        analytical_supported(&model),
        "η/θ-dependent ExpressionScale must be provider-supported"
    );
    let subject = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
    check_full_provider_vs_fd(&model, &subject, &[0.2, 10.0, 1.5], &[0.15, -0.10, 0.25]);
}

/// The **light** inner η-provider (`subject_eta_grad`) must carry the same
/// `ExpressionScale` η-only quotient rule as the full provider for the
/// `obs_scale = 1000 / V` model — `apply_expression_scale_inner` is the η-block of
/// `apply_expression_scale`. Since `provider_expression_scale_matches_production`
/// already pins the full provider's scaled `f`/`∂f/∂η` to FD of the production
/// predictor, light ≡ full here transitively validates the inner gradient against
/// production. Guards the inner EBE loop (the BFGS gradient and the H-matrix Jacobian
/// both read `subject_eta_grad`), which previously reverted `ExpressionScale` to FD.
#[test]
fn light_provider_expression_scale_matches_full() {
    let src = WARFARIN.replace(
        "[error_model]\n  DV ~ proportional(PROP_ERR)",
        "[error_model]\n  DV ~ proportional(PROP_ERR)\n[scaling]\n  obs_scale = 1000 / V",
    );
    let model = parse_model_string(&src).expect("scaling model parses");
    assert!(
        matches!(
            model.scaling,
            ScalingSpec::ExpressionScale { deriv: Some(_), .. }
        ),
        "model must carry a differentiable scale program"
    );
    // The model-level inner gate must now serve `ExpressionScale` analytically
    // (no longer a common bail).
    assert!(
        crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(&model),
        "ExpressionScale inner gradient must be in analytic scope"
    );
    let subject = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = [0.2, 10.0, 1.5];
    let eta = [0.15, -0.10, 0.25];
    let full = subject_sensitivities(&model, &subject, &theta, &eta).expect("full");
    let light = subject_eta_grad(&model, &subject, &theta, &eta).expect("light supported");
    assert_eq!(full.obs.len(), light.len());
    for (fo, lo) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                fo.df_deta[k],
                lo.df_deta[k],
                max_relative = 1e-12,
                epsilon = 1e-14
            );
        }
    }
}

/// Regression for the finding-3 stack-buffer bound (#534 audit): a scale program's
/// `var_to_pk_slot().len()` is the number of `[individual_parameters]` vars, NOT the
/// axis count — a model may declare more than `MAX_SCALE_AXES` individual parameters.
/// `apply_expression_scale_inner` must not panic on the fixed-size buffer there (it
/// falls back to a heap `Vec`). Build a chain of 18 vars (> 16) and confirm the light
/// inner gradient runs and still matches the full provider.
#[test]
fn light_provider_expression_scale_many_indiv_params_no_panic() {
    let mut ip = String::from("[individual_parameters]\n  A0 = 1.0\n");
    for i in 1..16 {
        ip.push_str(&format!("  A{i} = A{}\n", i - 1));
    }
    // 16 A-vars (A0..A15) + CL + V = 18 individual-parameter vars > MAX_SCALE_AXES.
    ip.push_str("  CL = TVCL * exp(ETA_CL) * A15\n  V = TVV * exp(ETA_V)\n");
    let src = format!(
        "[parameters]\n  theta TVCL(0.13,0.01,1.0)\n  theta TVV(8.0,1.0,50.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.05\n{ip}\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n"
    );
    let model = parse_model_string(&src).expect("parse many-param scaling model");
    assert!(
        matches!(
            model.scaling,
            ScalingSpec::ExpressionScale { deriv: Some(_), .. }
        ),
        "model must carry a differentiable scale program"
    );
    let subject = oral_subject(&[0.5, 1.0, 2.0, 4.0, 8.0]);
    let theta = [0.13, 8.0];
    let eta = [0.1, -0.05];
    let full = subject_sensitivities(&model, &subject, &theta, &eta).expect("full");
    let light = subject_eta_grad(&model, &subject, &theta, &eta).expect("light supported");
    assert_eq!(full.obs.len(), light.len());
    for (fo, lo) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-14);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                fo.df_deta[k],
                lo.df_deta[k],
                max_relative = 1e-12,
                epsilon = 1e-14
            );
        }
    }
}

/// Unit-covers `apply_ltbs_transform_inner` directly: the `g = ln(f)`, `∂g/∂η = ∂f/∂η/f`
/// normal branch; the below-`LTBS_FLOOR` clamp (grads zeroed, value floored); and the
/// no-op when `log_transform` is false.
#[test]
fn apply_ltbs_transform_inner_branches() {
    let mut out = vec![
        ObsGrad {
            f: 2.0,
            df_deta: vec![4.0, -2.0],
        },
        ObsGrad {
            f: crate::pk::LTBS_FLOOR * 0.5, // below the floor → clamp
            df_deta: vec![9.0, 9.0],
        },
    ];
    apply_ltbs_transform_inner(&mut out, true);
    // Row 0: f > FLOOR → g = ln f, df_deta /= f.
    approx::assert_relative_eq!(out[0].f, 2.0_f64.ln(), epsilon = 1e-12);
    approx::assert_relative_eq!(out[0].df_deta[0], 2.0, epsilon = 1e-12);
    approx::assert_relative_eq!(out[0].df_deta[1], -1.0, epsilon = 1e-12);
    // Row 1: below floor → grads zeroed, value = floored log.
    assert_eq!(out[1].df_deta, vec![0.0, 0.0]);
    approx::assert_relative_eq!(
        out[1].f,
        crate::pk::ltbs_log_g(crate::pk::LTBS_FLOOR * 0.5),
        epsilon = 1e-12
    );
    // No-op when `log_transform` is false.
    let mut out2 = vec![ObsGrad {
        f: 3.0,
        df_deta: vec![1.0, 2.0],
    }];
    apply_ltbs_transform_inner(&mut out2, false);
    approx::assert_relative_eq!(out2[0].f, 3.0, epsilon = 1e-12);
    assert_eq!(out2[0].df_deta, vec![1.0, 2.0]);
}

/// Plain closed-form LTBS now takes the analytic **inner** gradient (PR #665): the
/// light provider applies the same `g = ln(f)` jet as the outer, so its η-gradient must
/// equal the full outer provider's `df_deta` η-block. (The covariance step reconverges
/// these EBEs at the tighter `cov_inner_tol` so the `ln`-amplified EBE offset does not
/// corrupt the SEs — see `FitOptions::effective_cov_inner_tol`.)
#[test]
fn ltbs_plain_inner_eta_grad_matches_outer() {
    let src = WARFARIN.replace(
        "[error_model]\n  DV ~ proportional(PROP_ERR)",
        "[error_model]\n  log(DV) ~ additive(PROP_ERR)",
    );
    let model = parse_model_string(&src).expect("parse LTBS warfarin");
    assert!(model.log_transform, "fixture must be LTBS");
    let subject = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = [0.2, 10.0, 1.5];
    let eta = [0.15, -0.10, 0.25];
    // Plain LTBS (no scale) is now in analytic inner scope.
    assert!(
        crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(&model),
        "plain LTBS must now be served by the analytic inner gradient (PR #665)"
    );
    let full = subject_sensitivities(&model, &subject, &theta, &eta).expect("outer");
    let light = subject_eta_grad(&model, &subject, &theta, &eta)
        .expect("inner analytic gradient serves plain LTBS");
    assert_eq!(full.obs.len(), light.len());
    for (fo, lo) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-12);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                fo.df_deta[k],
                lo.df_deta[k],
                max_relative = 1e-10,
                epsilon = 1e-12
            );
        }
    }
}

/// LTBS combined with an η-dependent `ExpressionScale obs_scale` now takes the analytic
/// **inner** gradient too (Tier-1 follow-up to #665): the η-quotient is applied, then the
/// `g = ln(f)` jet LAST (reproducing `ln(f/s)`), so the inner η-gradient must equal the
/// full outer provider's `df_deta` η-block.
#[test]
fn ltbs_plus_expression_scale_inner_matches_outer() {
    let src = WARFARIN.replace(
        "[error_model]\n  DV ~ proportional(PROP_ERR)",
        "[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  log(DV) ~ additive(PROP_ERR)",
    );
    let model = parse_model_string(&src).expect("parse");
    assert!(model.log_transform, "fixture must be LTBS");
    assert!(
        matches!(model.scaling, ScalingSpec::ExpressionScale { .. }),
        "fixture must carry an η-dependent ExpressionScale obs_scale"
    );
    let subject = oral_subject(&[1.0, 2.0, 4.0, 8.0, 24.0]);
    let theta = [0.2, 10.0, 1.5];
    let eta = [0.15, -0.10, 0.25];
    assert!(
        crate::estimation::inner_optimizer::analytic_inner_grad_supported_model(&model),
        "LTBS + ExpressionScale is now in analytic inner scope"
    );
    let full = subject_sensitivities(&model, &subject, &theta, &eta).expect("outer");
    let light = subject_eta_grad(&model, &subject, &theta, &eta)
        .expect("inner analytic gradient serves LTBS + ExpressionScale");
    assert_eq!(full.obs.len(), light.len());
    for (fo, lo) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(fo.f, lo.f, max_relative = 1e-12, epsilon = 1e-12);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                fo.df_deta[k],
                lo.df_deta[k],
                max_relative = 1e-10,
                epsilon = 1e-12
            );
        }
    }
}

// ── Time-varying covariate analytic sensitivities ─────────────────

// Allometric WT-on-CL, the canonical time-varying covariate: `WT` changes
// across a subject's records, so `CL = TVCL·(WT/70)^THETA_WT·exp(ETA_CL)`
// switches mid-decay. θ = [TVCL, TVV, TVKA, THETA_WT].
const ONECPT_ORAL_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 1-cpt oral with WT-on-CL **and a constant `obs_scale` divisor** — the scale
// is covariate-independent, so the whole jet divides by it. θ = [TVCL, TVV,
// TVKA, THETA_WT].
const ONECPT_ORAL_TVCOV_SCALED: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = 1000
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 1-cpt IV with WT-on-CL, used for the **steady-state + TV-cov** case. θ =
// [TVCL, TVV, THETA_WT].
const ONECPT_IV_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 2-cpt IV with WT-on-CL. θ = [TVCL, TVV1, TVQ, TVV2, THETA_WT].
const TWOCPT_IV_TVCOV: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 2-cpt IV with WT-on-CL through intermediate individual-parameter assignments.
// Regression for #455: the TV-cov Dual2 gate must look at the compiled PK
// outputs (`prog.pk_slots()`), not `model.pk_indices`, because `pk_indices` is
// parallel to all unconditional assignments and contains intermediate rows.
const TWOCPT_IV_TVCOV_INTERMEDIATE: &str = r#"
[parameters]
  theta TVCL(10.0, 1.0, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(15.0, 1.0, 100.0)
  theta TVV2(100.0, 10.0, 1000.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  WTREL = WT / 70
  WTCL  = WTREL ^ THETA_WT
  BASECL = TVCL * WTCL
  CL = BASECL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  QBASE = TVQ
  Q  = QBASE
  V2BASE = TVV2
  V2 = V2BASE
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// 3-cpt oral with WT-on-CL. θ = [TVCL, TVV1, TVQ2, TVV2, TVQ3, TVV3, TVKA,
// THETA_WT].
const THREECPT_ORAL_TVCOV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV1(10.0, 1.0, 100.0)
  theta TVQ2(2.0, 0.1, 20.0)
  theta TVV2(20.0, 2.0, 200.0)
  theta TVQ3(1.5, 0.1, 20.0)
  theta TVV3(30.0, 3.0, 300.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn wt_map(wt: f64) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert("WT".to_string(), wt);
    m
}

/// Build a single-subject TV-covariate fixture with per-event `WT` snapshots.
/// `dose_wts`/`obs_wts`/`pk_only_wts` are parallel to `doses`/`obs_times`/
/// `pk_only_times`; populating `dose_covariates`/`obs_covariates` is what makes
/// `has_tv_covariates()` true (and routes production + provider through the
/// event-driven walk).
#[allow(clippy::too_many_arguments)]
fn tvcov_subject(
    doses: Vec<DoseEvent>,
    dose_wts: &[f64],
    obs_times: &[f64],
    obs_wts: &[f64],
    reset_times: Vec<f64>,
    pk_only_times: Vec<f64>,
    pk_only_wts: &[f64],
) -> Subject {
    let n = obs_times.len();
    Subject {
        id: "1".to_string(),
        doses,
        obs_times: obs_times.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: wt_map(obs_wts[0]),
        dose_covariates: dose_wts.iter().map(|&w| wt_map(w)).collect(),
        obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
        pk_only_times,
        pk_only_covariates: pk_only_wts.iter().map(|&w| wt_map(w)).collect(),
        reset_times,
        cens: vec![0; n],
        occasions: vec![1; n],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

// ---- Lagtime on the event-driven walk (#486) ----------------------------------------
//
// Before #486 a closed-form model carrying *any* lagtime declined to FD as soon as the
// subject routed to the event walk (time-varying covariates, the `TIME` built-in, or IOV) —
// `tvcov_analytical_supported` / `iov_analytical_supported` both bailed on `has_lagtime()`.
// Lagtime + IOV and lagtime + TV-covariates are bread-and-butter popPK, so this was the
// most-hit FD cell in the library. The walk now threads each dose *arrival* `t + ALAG` as a
// moving dual boundary — the same mechanism a modeled infusion end already used — so the
// bolus saltation `−A·Φ(t−τ)·b` falls out of the closed-form flow to all orders, with no
// explicit saltation injection (contrast the ODE path, which must inject one because RK45
// needs an f64 timeline).
//
// `LAGTIME = TVLAG * exp(ETA_LAG)` throughout: the lag carries **both** a θ and an η jet,
// so these exercise the moving boundary in the outer *and* inner walks.

/// `ONECPT_ORAL_LAG` + a weight covariate on CL, so a subject with time-varying WT routes
/// to the event-driven walk (and previously fell to FD purely because of the lagtime).
const ONECPT_ORAL_LAG_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The headline cell: an **estimated lagtime on the TV-covariate event walk** is analytic,
/// matching FD of the production predictor in value, `∂f/∂η`, `∂²f/∂η²` and `∂f/∂θ`.
/// The observation times deliberately straddle the lagged arrivals (some pre-arrival, where
/// the prediction is still zero/flat, some after), so the moving boundary is genuinely
/// exercised rather than sitting in a region where `∂/∂ALAG` vanishes.
#[test]
fn lagtime_tvcov_walk_matches_fd_of_production() {
    let model = parse_model_string(ONECPT_ORAL_LAG_TVCOV).expect("parse lag + tvcov");
    assert!(
        model.has_lagtime(),
        "fixture must carry a lagtime (the point of the test)"
    );
    assert!(
        tvcov_analytical_supported(&model),
        "an estimated lagtime must no longer take the TV-cov walk out of analytic scope"
    );

    // WT moves across the record, so the subject routes to the event walk.
    let subject = tvcov_subject(
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        &[70.0, 82.0],
        &[0.25, 0.5, 1.0, 3.0, 8.0, 24.4, 26.0, 30.0],
        &[70.0, 70.0, 74.0, 74.0, 78.0, 82.0, 82.0, 86.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    assert!(subject.has_tv_covariates());
    assert!(subject_routes_to_event_walk(&model, &subject));

    let theta = [0.22, 11.0, 1.4, 0.7, 0.8];
    let eta = [0.12, -0.08, 0.15, 0.10];
    check_full_provider_vs_fd(&model, &subject, &theta, &eta);
}

/// The inner (`Dual1`, η-only) twin of the walk must agree with the outer `Dual2` walk on
/// `∂f/∂η` — the two are separate monomorphizations of the same moving-boundary logic, and
/// a lag jet threaded into one but not the other would silently split inner/outer scope.
#[test]
fn lagtime_tvcov_inner_matches_outer_eta_grad() {
    let model = parse_model_string(ONECPT_ORAL_LAG_TVCOV).expect("parse lag + tvcov");
    let subject = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[70.0],
        &[0.25, 1.0, 4.0, 12.0],
        &[70.0, 74.0, 78.0, 82.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    let theta = [0.22, 11.0, 1.4, 0.7, 0.8];
    let eta = [0.12, -0.08, 0.15, 0.10];

    let outer = subject_sensitivities(&model, &subject, &theta, &eta).expect("outer analytic");
    let inner = subject_eta_grad(&model, &subject, &theta, &eta).expect("inner analytic");
    assert_eq!(outer.obs.len(), inner.len());
    for (o, i) in outer.obs.iter().zip(inner.iter()) {
        approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-10, epsilon = 1e-12);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                o.df_deta[k],
                i.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

/// A **fixed-duration infusion** under a lagtime: the infusion *end* rides the arrival's
/// jet (`τ + D` with `D` constant), so both window boundaries move together. This is the
/// case the pre-#486 walk could not express at all — it threaded a jet onto the modeled end
/// only, and the dose start "never moved".
#[test]
fn lagtime_with_fixed_infusion_matches_fd_of_production() {
    let model = parse_model_string(ONECPT_ORAL_LAG_TVCOV).expect("parse lag + tvcov");
    // Infusion into the depot (cmt 1) over 2 h, arriving at `t + ALAG`.
    let subject = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)],
        &[70.0],
        &[0.5, 1.5, 2.5, 4.0, 10.0],
        &[70.0, 72.0, 76.0, 80.0, 84.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    let theta = [0.22, 11.0, 1.4, 0.7, 0.8];
    let eta = [0.12, -0.08, 0.15, 0.10];
    check_full_provider_vs_fd(&model, &subject, &theta, &eta);
}

/// **SS × lagtime still declines to FD**, per-subject. Production overlays the pre-arrival
/// steady-state tail for observations in `[t, t + ALAG)` (`ss_state_at_phase_event_driven`)
/// and the dual walk has no twin of that overlay, so serving it would disagree with
/// production in *value*, not just in derivative. Deliberately a hard decline, not an
/// approximation — a lagged non-SS subject on the same model stays analytic.
#[test]
fn lagtime_with_ss_dose_declines_to_fd() {
    let model = parse_model_string(ONECPT_ORAL_LAG_TVCOV).expect("parse lag + tvcov");
    let ss = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
        &[70.0],
        &[1.0, 4.0, 9.0],
        &[70.0, 74.0, 78.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    assert!(ss.doses.iter().any(|d| d.ss));
    let theta = [0.22, 11.0, 1.4, 0.7, 0.8];
    let eta = [0.12, -0.08, 0.15, 0.10];
    assert!(
        subject_sensitivities(&model, &ss, &theta, &eta).is_none(),
        "SS + lagtime on the walk must decline to FD (no dual twin of the SS pre-arrival overlay)"
    );
    assert!(
        subject_eta_grad(&model, &ss, &theta, &eta).is_none(),
        "inner must decline in lockstep with the outer (no split scope)"
    );
}

/// An observation sampled **exactly at a lagged bolus arrival** gets the one-sided analytic
/// derivative, matching the ODE twin (#486).
///
/// This is the sharper sibling of the infusion-end case. Here the prediction is not merely
/// kinked in `ALAG` — it is **discontinuous**: for a lag just below the coincidence the dose
/// has landed by the sample, for a lag just above it has not, so the value jumps by the whole
/// bolus. Production resolves that by ordering `Dose` before `Obs` (`kind_order`), which makes
/// the prediction *left*-continuous — the sample sees the dose. The derivative consistent with
/// the model's own semantics is therefore the derivative of that branch, and it is finite.
///
/// A finite-difference gradient is **actively worse** here: a central difference straddles the
/// jump and returns `≈ −jump / 2h`, an enormous bogus value that would wreck a line search. So
/// declining to FD (the earlier behaviour) was the wrong instinct. The walk instead lifts the
/// just-deposited bolus off the state, steps back across the sliver under the pre-arrival rate
/// set, and puts the bolus back — the bolus does not decay across a zero-length window.
///
/// The ODE twin is the oracle: it reaches the same value from the same convention by an
/// entirely different route (it captures the observation *before* injecting the arrival's
/// saltation, since `K_OBS` sorts before the infusion/rate events and `K_DOSE` before `K_OBS`).
/// Central-FD parity is deliberately not asserted — it cannot hold across a discontinuity.
#[test]
fn obs_on_lagged_bolus_arrival_matches_ode_twin() {
    const ODE_TWIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  LAGTIME = TVLAG * exp(ETA_LAG)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let cf = parse_model_string(ONECPT_ORAL_LAG_TVCOV).expect("parse cf lag + tvcov");
    let ode = parse_model_string(ODE_TWIN).expect("parse ode twin");
    let theta = [0.22, 11.0, 1.4, 0.7, 0.8];
    // η_LAG = 0 ⇒ ALAG = TVLAG = 0.7 exactly, so the t = 0.7 sample sits on the arrival of
    // the t = 0 bolus.
    let eta = [0.12, -0.08, 0.15, 0.0];
    let subject = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[70.0],
        &[0.7, 2.0, 6.0],
        &[70.0, 74.0, 78.0],
        Vec::new(),
        Vec::new(),
        &[],
    );

    let cf_sens = subject_sensitivities(&cf, &subject, &theta, &eta)
        .expect("closed form must stay analytic on a coincident arrival (corrected read-out)");
    let ode_sens = subject_sensitivities(&ode, &subject, &theta, &eta)
        .expect("ODE twin serves it via jump sensitivities");

    // At the coincident sample the *value* is zero (the depot bolus has only just landed, so
    // central is still empty) while ∂f/∂ALAG is emphatically not — it is the rate at which
    // central would have filled had the dose arrived earlier. The closed form used to return
    // zero here, which looks perfectly innocent and is wrong.
    for (c, o) in cf_sens.obs.iter().zip(ode_sens.obs.iter()) {
        approx::assert_relative_eq!(c.f, o.f, max_relative = 1e-5, epsilon = 1e-8);
        for m in 0..theta.len() {
            approx::assert_relative_eq!(
                c.df_dtheta[m],
                o.df_dtheta[m],
                max_relative = 1e-4,
                epsilon = 1e-6
            );
        }
        for k in 0..cf.n_eta {
            approx::assert_relative_eq!(
                c.df_deta[k],
                o.df_deta[k],
                max_relative = 1e-4,
                epsilon = 1e-6
            );
        }
    }

    // Inner must carry the same correction as outer — no split scope.
    let inner = subject_eta_grad(&cf, &subject, &theta, &eta).expect("inner analytic");
    for (o, i) in cf_sens.obs.iter().zip(inner.iter()) {
        for k in 0..cf.n_eta {
            approx::assert_relative_eq!(
                o.df_deta[k],
                i.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

const ONECPT_TRANSIT_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.0, 30.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;

/// A `one_cpt_transit` subject with time-varying covariates is served by the model's ODE
/// `transit()` equivalent (`effective_for` routes it there), NOT the closed form — the
/// closed form assumes constant parameters over each absorption window, and the
/// event-driven walk can't state-propagate the continuous-`N` Gamma absorption
/// (`subject_routes_to_event_walk` omits transit for that reason). The gradient is
/// analytic (a non-`None` `SubjectSens`, non-zero — the old silent all-zero bug is gone);
/// its numerical agreement with the production predictor is covered end-to-end by the ODE
/// equivalence suite (`tests/transit_analytic_equivalence.rs`). A constant-parameter
/// transit subject keeps the fast closed form (`effective_for` returns the model itself).
#[test]
fn transit_with_tvcov_routes_to_ode_equivalent() {
    let m = parse_model_string(ONECPT_TRANSIT_MODEL).expect("parse transit");
    assert_eq!(m.pk_model, PkModel::OneCptTransit);
    assert!(
        m.absorption_ode_equivalent.is_some(),
        "a plain transit model carries an ODE equivalent"
    );
    let theta = [5.0, 50.0, 1.0, 3.0];
    let eta = [0.1, -0.05];

    // TV-cov subject → routed to the ODE equivalent, analytic and non-zero.
    let tv = tvcov_subject(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[70.0],
        &[1.0, 2.0, 4.0, 8.0, 24.0],
        &[70.0, 72.0, 80.0, 85.0, 90.0],
        Vec::new(),
        Vec::new(),
        &[],
    );
    assert!(tv.has_tv_covariates());
    assert!(
        std::ptr::eq(
            m.effective_for(&tv),
            m.absorption_ode_equivalent.as_ref().unwrap().get_or_build()
        ),
        "TV-cov transit subject must be served by the ODE equivalent"
    );
    assert!(
        !subject_routes_to_event_walk(&m, &tv),
        "transit never routes to the closed-form event walk"
    );
    let sens = subject_sensitivities(&m, &tv, &theta, &eta)
        .expect("transit + TV-cov is analytic via the ODE equivalent (was silent zeros)");
    assert!(
        sens.obs.iter().any(|o| o.f.abs() > 1e-6),
        "predictions must be non-zero"
    );
    assert!(
        subject_eta_grad(&m, &tv, &theta, &eta).is_some(),
        "inner analytic too"
    );

    // Constant-parameter subject → keeps the fast closed form (served by the model itself).
    let flat = subject_with_doses_and_resets(
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        &[1.0, 2.0, 4.0, 8.0, 24.0],
        Vec::new(),
    );
    assert!(!flat.has_tv_covariates());
    assert!(
        std::ptr::eq(m.effective_for(&flat), &m),
        "constant-parameter transit subject keeps the closed form"
    );
}

/// The TV-covariate provider's exact value/∂η/∂²η/∂θ/∂²η∂θ must match central
/// finite differences of the production predictor `compute_predictions_with_tv`
/// (the independent f64 event-driven path), across 1-/2-/3-cpt and the three
/// scenarios the walk must cover: (a) the covariate changing at observations,
/// (b) a covariate breakpoint carried by an EVID=2 (`pk_only`) record between
/// observations, and (c) a covariate change combined with an EVID=4 reset.
#[test]
fn tvcov_provider_matches_fd_of_production() {
    let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        // (a) 1-cpt oral, WT changing at each observation.
        {
            let m = parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[1.0, 2.0, 4.0, 8.0, 24.0],
                &[70.0, 72.0, 80.0, 85.0, 90.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
        },
        // (b) 1-cpt oral, covariate breakpoint at an EVID=2 record (t=3) that
        // falls between observations — the WT jumps 70→95 there, switching CL
        // mid-decay with no observation at the breakpoint.
        {
            let m = parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov pkonly");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[1.0, 2.0, 4.0, 8.0],
                &[70.0, 70.0, 95.0, 95.0],
                Vec::new(),
                vec![3.0],
                &[95.0],
            );
            (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.12, 0.08, -0.15])
        },
        // (c) 1-cpt oral, WT change combined with an EVID=4 reset at t=12.
        {
            let m = parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov reset");
            let s = tvcov_subject(
                vec![bolus(0.0), bolus(12.0)],
                &[70.0, 90.0],
                &[1.0, 3.0, 13.0, 15.0, 18.0],
                &[70.0, 70.0, 90.0, 90.0, 90.0],
                vec![12.0],
                Vec::new(),
                &[],
            );
            (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.10, -0.05, 0.20])
        },
        // (d) 2-cpt IV bolus, WT changing at each observation.
        {
            let m = parse_model_string(TWOCPT_IV_TVCOV).expect("parse 2cpt iv tvcov");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[0.5, 2.0, 6.0, 12.0, 24.0],
                &[70.0, 75.0, 82.0, 88.0, 95.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
        },
        // (e) 2-cpt IV with intermediate individual-parameter assignments and
        // an EVID=2-style covariate breakpoint. `model.pk_indices` contains
        // extra intermediate rows here; the TV-cov path must use
        // `prog.pk_slots()` to seed/scatter the four structural PK outputs.
        {
            let m = parse_model_string(TWOCPT_IV_TVCOV_INTERMEDIATE)
                .expect("parse 2cpt iv tvcov intermediate");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[0.5, 2.0, 6.0, 12.0],
                &[70.0, 70.0, 95.0, 95.0],
                Vec::new(),
                vec![3.0],
                &[95.0],
            );
            assert!(
                m.pk_indices.len()
                    > m.indiv_param_partials
                        .indiv_param_program
                        .as_ref()
                        .expect("compiled individual program")
                        .pk_slots()
                        .len(),
                "fixture must expose intermediate individual-parameter rows"
            );
            (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
        },
        // (f) 3-cpt oral, WT changing at each observation (widest dual, M=11).
        {
            let m = parse_model_string(THREECPT_ORAL_TVCOV).expect("parse 3cpt oral tvcov");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[1.0, 2.0, 6.0, 12.0, 24.0],
                &[70.0, 73.0, 80.0, 86.0, 92.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (
                m,
                s,
                vec![5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.5, 0.75],
                vec![0.15, -0.10, 0.25],
            )
        },
        // (g) 1-cpt oral with a constant `obs_scale = 1000` divisor — the whole
        // jet divides by the (covariate-independent) scale.
        {
            let m =
                parse_model_string(ONECPT_ORAL_TVCOV_SCALED).expect("parse 1cpt oral tvcov scaled");
            assert!(
                matches!(m.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-9),
                "model must carry a constant ScalarScale"
            );
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[1.0, 2.0, 4.0, 8.0, 24.0],
                &[70.0, 72.0, 80.0, 85.0, 90.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
        },
        // (h) 1-cpt IV **steady-state** bolus (II=24) with WT changing across
        // observations: the walk equilibrates the SS state per-event at the
        // dose's covariate snapshot, then the covariate switches the decay.
        {
            let m = parse_model_string(ONECPT_IV_TVCOV).expect("parse 1cpt iv tvcov ss");
            let s = tvcov_subject(
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
                &[70.0],
                &[1.0, 6.0, 12.0, 18.0, 23.0],
                &[70.0, 78.0, 86.0, 92.0, 98.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            assert!(
                s.doses.iter().any(|d| d.ss),
                "fixture must carry an SS dose"
            );
            (m, s, vec![0.2, 10.0, 0.75], vec![0.12, -0.09])
        },
    ];
    for (m, s, theta, eta) in &cases {
        assert!(
            tvcov_analytical_supported(m),
            "TV-cov model must be provider-supported"
        );
        assert!(s.has_tv_covariates(), "fixture must carry TV covariates");
        assert!(
            subject_sensitivities(m, s, theta, eta).is_some(),
            "TV-cov subject must take the analytic provider"
        );
        check_full_provider_vs_fd(m, s, theta, eta);
    }
}

/// #447: the light `Dual1` inner η-gradient ([`subject_eta_grad_tvcov`]) must
/// equal the full `Dual2` outer `df_deta` (η-block) for TV-cov subjects — both
/// run the same event-driven walk, and the outer is FD-validated above. Covers
/// 1-cpt oral, 2-cpt IV, and a steady-state bolus.
#[test]
fn tvcov_eta_grad_matches_full() {
    let bolus = |t: f64| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0);
    let cases: Vec<(CompiledModel, Subject, Vec<f64>, Vec<f64>)> = vec![
        {
            let m = parse_model_string(ONECPT_ORAL_TVCOV).expect("parse 1cpt oral tvcov");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[1.0, 2.0, 4.0, 8.0, 24.0],
                &[70.0, 72.0, 80.0, 85.0, 90.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
        },
        {
            let m = parse_model_string(TWOCPT_IV_TVCOV).expect("parse 2cpt iv tvcov");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[0.5, 2.0, 6.0, 12.0, 24.0],
                &[70.0, 75.0, 82.0, 88.0, 95.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
        },
        {
            let m = parse_model_string(TWOCPT_IV_TVCOV_INTERMEDIATE)
                .expect("parse 2cpt iv tvcov intermediate");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[0.5, 2.0, 6.0, 12.0],
                &[70.0, 70.0, 95.0, 95.0],
                Vec::new(),
                vec![3.0],
                &[95.0],
            );
            (m, s, vec![10.0, 50.0, 15.0, 100.0, 0.75], vec![0.12, -0.08])
        },
        {
            let m = parse_model_string(ONECPT_IV_TVCOV).expect("parse 1cpt iv tvcov ss");
            let s = tvcov_subject(
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
                &[70.0],
                &[1.0, 6.0, 12.0, 18.0, 23.0],
                &[70.0, 78.0, 86.0, 92.0, 98.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (m, s, vec![0.2, 10.0, 0.75], vec![0.12, -0.09])
        },
        {
            // Constant `ScalarScale` (`obs_scale = 1000`) on the TV-cov **inner**:
            // exercises `run_obs_grad_tvcov`'s `∂(f/k)/∂η = (∂f/∂η)/k` division,
            // which the other inner cases (no output scaling) leave uncovered
            // (#451 / #449 review #10).
            let m =
                parse_model_string(ONECPT_ORAL_TVCOV_SCALED).expect("parse 1cpt oral tvcov scaled");
            let s = tvcov_subject(
                vec![bolus(0.0)],
                &[70.0],
                &[1.0, 2.0, 4.0, 8.0, 24.0],
                &[70.0, 72.0, 80.0, 85.0, 90.0],
                Vec::new(),
                Vec::new(),
                &[],
            );
            (m, s, vec![0.2, 10.0, 1.5, 0.75], vec![0.15, -0.10, 0.25])
        },
    ];
    for (model, subject, theta, eta) in &cases {
        let full = subject_sensitivities_tvcov(model, subject, theta, eta).expect("outer tvcov");
        let light = subject_eta_grad_tvcov(model, subject, theta, eta).expect("light tvcov inner");
        assert_eq!(full.obs.len(), light.len());
        for (a, b) in full.obs.iter().zip(light.iter()) {
            approx::assert_relative_eq!(a.f, b.f, max_relative = 1e-12, epsilon = 1e-12);
            for k in 0..model.n_eta {
                approx::assert_relative_eq!(
                    a.df_deta[k],
                    b.df_deta[k],
                    max_relative = 1e-10,
                    epsilon = 1e-11
                );
            }
        }
    }
}

// ── IOV analytic sensitivities ───────────────────────────────────

const WARFARIN_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// M3 BLOQ + IOV scope (#580): a closed-form IOV model with M3 BLOQ is analytic
/// (the censored coefficients ride the stacked `[η_bsv, κ]` layout). The triple
/// **M3 + IOV + `iiv_on_ruv`** is analytic too as of #591 — the closed-form assembly
/// already carried the censored residual-eta cross coefficients `(C·z, C·m)`, so
/// `iov_analytical_supported` admits it (and `analytic_outer_gradient_available`
/// follows). The ODE IOV triple is analytic too (#486), as is the *non-IOV ODE* triple
/// (#623). Plain IOV and IOV + `iiv_on_ruv` (no M3) stay analytic.
#[test]
fn iov_analytical_supported_admits_m3_but_not_the_ruv_triple() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    // Plain IOV: analytic.
    assert!(iov_analytical_supported(&model));
    // M3 + IOV (no iiv_on_ruv): analytic as of #580.
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(iov_analytical_supported(&model));
    assert!(analytic_outer_gradient_available(&model));
    // M3 + IOV + iiv_on_ruv (the triple): analytic as of #591 (the closed-form
    // assembly carried the censored residual-eta cross coefficients all along).
    model.residual_error_eta = Some(0);
    assert!(iov_analytical_supported(&model));
    assert!(analytic_outer_gradient_available(&model));
    // IOV + iiv_on_ruv without M3: analytic (#4b).
    model.bloq_method = crate::types::BloqMethod::Drop;
    assert!(iov_analytical_supported(&model));
}

/// Two-occasion IOV subject: a dose + observations in occasion 1, then a dose +
/// observations in occasion 2 (no washout — carryover spans the boundary).
fn iov_subject() -> Subject {
    let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
    let occasions = vec![1u32, 1, 1, 2, 2, 2];
    let n = obs_times.len();
    Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Closed-form **IOV × lagtime** (#486). Before this the CF IOV walk declined *any*
/// lagtime outright (`iov_analytical_supported` bailed on `has_lagtime()`), so every
/// oral-with-lag IOV model — a completely standard popPK object — ran on FD. The lag here
/// carries an occasion κ (`KAPPA_LAG`), so each occasion's doses arrive at a *different*
/// lagged time and the moving boundary is seeded per dose from that dose's own occasion
/// snapshot.
const ONECPT_ORAL_LAG_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta TVLAG(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_LAG ~ 0.05
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
  LAGTIME = TVLAG * exp(KAPPA_LAG)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;

/// IOV × lagtime is analytic on the closed-form walk, matching FD of `predict_iov` across
/// the stacked `(η_bsv, κ₁..κ_K)` basis. The κ sits **on the lag itself**, so occasion 2's
/// doses arrive at a different offset than occasion 1's — the strongest form of the moving
/// boundary (a per-occasion arrival), and the one that falls out of per-dose seeding.
#[test]
fn lagtime_iov_walk_matches_fd_of_predict_iov() {
    let model = parse_model_string(ONECPT_ORAL_LAG_IOV).expect("parse IOV + lag");
    assert_eq!(model.n_kappa, 1);
    assert!(model.has_lagtime());
    assert!(
        iov_analytical_supported(&model),
        "an estimated lagtime must no longer take the CF IOV walk out of analytic scope"
    );
    // stacked = [η_cl, η_v, κ_lag(occ1), κ_lag(occ2)].
    check_iov_provider_vs_fd(
        &model,
        &iov_subject(),
        &[0.22, 11.0, 1.4, 0.7],
        &[0.12, -0.08, 0.06, -0.09],
    );
}

// 1-cpt IV closed-form IOV with an `init(central) = TVC0·V` baseline (#486, branch
// G-closed-form). `CL = TVCL·exp(ETA_CL + KAPPA_CL)` carries the occasion κ, so the init
// *decay* kernel depends on κ (via each observation's occasion clearance) while the init
// *amount* `A₀ = TVC0·V` is BSV-only (no κ) — exactly production's split in `predict_iov`
// (`add_analytical_init_with(amount_pk = BSV, Some(obs_params))`).
const WARFARIN_IOV_INIT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVC0(5.0, 0.1, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[initial_conditions]
  init(central) = TVC0 * V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// **`init(...)` × IOV on the closed-form walk** (#486, branch G-closed-form). The IOV
/// provider now layers the `A₀·kernel(t, pk)` init impulse per occasion: the amount from the
/// BSV snapshot (κ = 0), the decay from each observation's occasion clearance (κ carried).
/// The full provider's value / `∂(stacked-η)` / Hessian / `∂θ` must match FD of `predict_iov`
/// (which includes the init), and the inner η-gradient must equal the outer η-block.
#[test]
fn iov_init_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_INIT).expect("parse warfarin IOV+init");
    assert_eq!(model.n_kappa, 1);
    assert_eq!(model.analytical_init.len(), 1, "init parsed");
    assert!(model.ode_spec.is_none(), "closed-form model");
    assert!(
        iov_analytical_supported(&model),
        "closed-form IOV + init must be analytic (#486)"
    );
    let subject = iov_subject();
    // θ = [TVCL, TVV, TVC0]; stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn iov_init_inner_eta_grad_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_INIT).expect("parse warfarin IOV+init"),
        &iov_subject(),
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

const WARFARIN_IOV_2CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// Shared FD-vs-predict_iov check for a two-occasion IOV model: value, gradient,
/// and Hessian over the stacked random-effects vector `[η_bsv, κ_g0, κ_g1]` and
/// the θ block must match central differences of the production `predict_iov`
/// (an independent f64 path), validating the whole walk + (η,κ,θ) chain.
fn check_iov_provider_vs_fd(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked: &[f64],
) {
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let n_theta = theta.len();
    let k_groups = crate::stats::likelihood::iov_occasion_groups(subject).len();
    assert_eq!(
        stacked.len(),
        n_eta + k_groups * n_kappa,
        "stacked vector must match IOV occasion groups"
    );
    let sens = subject_sensitivities_iov(model, subject, theta, stacked).expect("supported");

    // Map a stacked-η vector to predict_iov's (η_bsv, kappas-per-group) form.
    let pred = |st: &[f64], th: &[f64], j: usize| -> f64 {
        let eta_bsv = st[..n_eta].to_vec();
        let kappas: Vec<Vec<f64>> = (0..k_groups)
            .map(|g| {
                let base = n_eta + g * n_kappa;
                st[base..base + n_kappa].to_vec()
            })
            .collect();
        crate::pk::predict_iov(model, subject, th, &eta_bsv, &kappas)[j]
    };

    let theta = theta.to_vec();
    let stacked = stacked.to_vec();
    let n_st = stacked.len();
    let he = 1e-6;
    let heh = 1e-4;
    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(
            obs.f,
            pred(&stacked, &theta, j),
            max_relative = 1e-9,
            epsilon = 1e-12
        );
        // ∂f/∂stacked and ∂²f/∂stacked².
        for k in 0..n_st {
            let mut sp = stacked.clone();
            sp[k] += he;
            let mut sm = stacked.clone();
            sm[k] -= he;
            let g = (pred(&sp, &theta, j) - pred(&sm, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-4, epsilon = 1e-7);
            for l in 0..n_st {
                let mut pp = stacked.clone();
                pp[k] += heh;
                pp[l] += heh;
                let mut pm = stacked.clone();
                pm[k] += heh;
                pm[l] -= heh;
                let mut mp = stacked.clone();
                mp[k] -= heh;
                mp[l] += heh;
                let mut mm = stacked.clone();
                mm[k] -= heh;
                mm[l] -= heh;
                let hh = (pred(&pp, &theta, j) - pred(&pm, &theta, j) - pred(&mp, &theta, j)
                    + pred(&mm, &theta, j))
                    / (4.0 * heh * heh);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * n_st + l],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }
        // ∂f/∂θ and ∂²f/∂stacked∂θ.
        for m in 0..n_theta {
            let s = he * (1.0 + theta[m].abs());
            let mut tp = theta.clone();
            tp[m] += s;
            let mut tm = theta.clone();
            tm[m] -= s;
            let g = (pred(&stacked, &tp, j) - pred(&stacked, &tm, j)) / (2.0 * s);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-4, epsilon = 1e-7);
            for k in 0..n_st {
                let sh = heh * (1.0 + theta[m].abs());
                let mut ep = stacked.clone();
                ep[k] += heh;
                let mut em = stacked.clone();
                em[k] -= heh;
                let mut tp2 = theta.clone();
                tp2[m] += sh;
                let mut tm2 = theta.clone();
                tm2[m] -= sh;
                let hh = (pred(&ep, &tp2, j) - pred(&ep, &tm2, j) - pred(&em, &tp2, j)
                    + pred(&em, &tm2, j))
                    / (4.0 * heh * sh);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * n_theta + m],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }
    }
}

/// 1-cpt oral IOV: provider == FD of `predict_iov` over `[η_bsv, κ_g0, κ_g1]`.
#[test]
fn iov_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    assert_eq!(model.n_kappa, 1, "model must carry one kappa");
    assert!(
        iov_analytical_supported(&model),
        "warfarin IOV must be IOV-provider supported"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

// 1-cpt IV closed-form IOV with a saturable-binding analytic Form C readout
// `y = C + BMAX·C/(KD + C)`, `C = central/V` (#655). `CL = TVCL·exp(ETA_CL + KAPPA_CL)`
// carries the occasion κ through the concentration, while production applies the
// readout with the κ = 0 (BSV-only) PK params (`apply_analytic_readout(..., eta_bsv)`)
// — so the readout param jet (V, BMAX, KD) is κ-less and only `C` carries κ, exactly
// the split `run_obs_iov` seeds from `readout_obs`.
const WARFARIN_IOV_BINDING_READOUT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + BMAX * (central / V) / (KD + central / V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// **Analytic Form C readout × IOV on the closed-form walk** (#655). The IOV provider
/// now replaces each observation's concentration jet with `y = <expr>`, seeding the
/// readout PK params from the κ = 0 (BSV-only) per-obs snapshot while the walk's
/// concentration carries κ — matching production `predict_iov`'s
/// `apply_analytic_readout(..., eta_bsv, ...)`. The full provider's value /
/// `∂(stacked-η)` / Hessian / `∂θ` must match FD of `predict_iov` (which applies the
/// readout), and the non-structural BMAX/KD must be first-class differentiable.
#[test]
fn iov_form_c_binding_readout_provider_matches_fd() {
    let model =
        parse_model_string(WARFARIN_IOV_BINDING_READOUT).expect("parse warfarin IOV readout");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_none(), "closed-form model");
    assert!(model.analytic_readout.is_some(), "Form C readout parsed");
    assert!(
        iov_analytical_supported(&model),
        "closed-form IOV + central-only dual-evaluable Form C readout must be analytic (#655)"
    );
    let subject = iov_subject();
    // θ = [TVCL, TVV, TVBMAX, TVKD]; stacked = [η_cl, η_v, κ_g0, κ_g1].
    let theta = [0.2, 10.0, 3.0, 2.0];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);

    // BMAX (θ 2) / KD (θ 3) are non-structural readout params (basis extension) — their
    // θ-gradient must be non-zero, proving they are differentiated under IOV, not aliased.
    let full = subject_sensitivities_iov(&model, &subject, &theta, &stacked).expect("supported");
    let max_bmax = full
        .obs
        .iter()
        .map(|o| o.df_dtheta[2].abs())
        .fold(0.0_f64, f64::max);
    let max_kd = full
        .obs
        .iter()
        .map(|o| o.df_dtheta[3].abs())
        .fold(0.0_f64, f64::max);
    assert!(max_bmax > 1e-6, "∂y/∂BMAX must be non-zero under IOV");
    assert!(max_kd > 1e-6, "∂y/∂KD must be non-zero under IOV");
}

#[test]
fn iov_form_c_binding_readout_inner_eta_grad_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_BINDING_READOUT).expect("parse"),
        &iov_subject(),
        &[0.2, 10.0, 3.0, 2.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

// As `WARFARIN_IOV_BINDING_READOUT`, but the readout is gated on a **per-row** covariate
// (`FREE`) — the free-vs-total assay pattern (#650/#655). A per-observation `FREE` flag
// makes the subject a TV-covariate subject, so `build_iov_sources` takes the per-event
// branch and builds one κ = 0 readout snapshot per observation (`readout_obs`), each at
// that observation's covariate — the branch the plain (static-covariate) case above does
// not exercise.
const WARFARIN_IOV_BINDING_READOUT_FREE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = if (FREE == 0) central / V + BMAX * (central / V) / (KD + central / V) else central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// Covariate-gated Form C readout × IOV: the per-row `FREE` flag routes the subject
/// through the per-event `readout_obs` branch (one κ = 0 snapshot per observation, at
/// that obs's covariate). Value + all stacked-η/θ first/second derivatives must match
/// FD of `predict_iov`, and the inner η-gradient must equal the outer η-block.
#[test]
fn iov_form_c_readout_tvcov_gated_matches_fd() {
    let model = parse_model_string(WARFARIN_IOV_BINDING_READOUT_FREE).expect("parse");
    let mut subject = iov_subject();
    // Alternating per-row FREE flag → per-observation covariate snapshots.
    subject.obs_covariates = (0..subject.obs_times.len())
        .map(|j| HashMap::from([("FREE".to_string(), (j % 2) as f64)]))
        .collect();
    assert!(
        subject.has_tv_covariates(),
        "FREE flag makes it a TV-cov subject"
    );
    assert!(
        iov_analytical_supported(&model),
        "covariate-gated Form C readout × IOV must be analytic (#655)"
    );
    let theta = [0.2, 10.0, 3.0, 2.0];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    check_iov_inner_matches_outer(&model, &subject, &theta, &stacked);
}

// A readout that references the `TIME` builtin directly (`y = central/V + BETA·TIME`).
// `TIME` resolves `Op::PushTime` from the model-time thread-local, so both the analytic
// readout jet and production `apply_analytic_readout` must enter each observation's time
// (Copilot #670 review). `BETA = TVBETA` is a non-structural readout param, so
// `∂y/∂TVBETA = TIME` — a direct probe that the observation time (not a stale 0) is used.
const WARFARIN_IOV_TIME_READOUT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBETA(0.05, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV  * exp(ETA_V)
  BETA = TVBETA
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + BETA * TIME
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// A `TIME`-referencing Form C readout under IOV must evaluate `TIME` at each
/// observation. FD-of-`predict_iov` parity ties the analytic path to the (now equally
/// guarded) production predictor, and `∂y/∂TVBETA` must equal each observation's time
/// (it would be 0 everywhere if the readout read a stale model-time of 0).
#[test]
fn iov_form_c_time_readout_evaluated_at_obs_time() {
    let model = parse_model_string(WARFARIN_IOV_TIME_READOUT).expect("parse");
    assert!(
        iov_analytical_supported(&model),
        "TIME readout × IOV analytic (#655)"
    );
    let subject = iov_subject();
    let theta = [0.2, 10.0, 0.05];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    let full = subject_sensitivities_iov(&model, &subject, &theta, &stacked).expect("supported");
    for (j, o) in full.obs.iter().enumerate() {
        // ∂(central/V + TVBETA·TIME)/∂TVBETA = TIME at observation j.
        approx::assert_relative_eq!(
            o.df_dtheta[2],
            subject.obs_times[j],
            max_relative = 1e-6,
            epsilon = 1e-9
        );
    }
}

// A baseline-subtracted readout `y = central/V - E0` that is negative at every observation
// (E0 exceeds the concentration). Production clamps the *concentration* to ≥ 0 before the
// readout and returns the readout output **unclamped**; the provider must do the same
// rather than zeroing a negative output (user #670 review finding #1).
const WARFARIN_IOV_NEG_READOUT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVE0(100.0, 0.01, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  E0 = TVE0
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V - E0
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// A negative-valued Form C readout under IOV: the readout output is returned unclamped
/// (matching production), so the provider value/gradient match FD of `predict_iov` on
/// rows where `y < 0` — the case the inverted clamp (zeroing the output) got wrong.
#[test]
fn iov_form_c_negative_output_readout_not_clamped() {
    let model = parse_model_string(WARFARIN_IOV_NEG_READOUT).expect("parse");
    assert!(iov_analytical_supported(&model));
    let subject = iov_subject();
    let theta = [0.2, 10.0, 100.0];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    let full = subject_sensitivities_iov(&model, &subject, &theta, &stacked).expect("supported");
    assert!(
        full.obs.iter().all(|o| o.f < 0.0),
        "readout must stay negative (unclamped), not zeroed"
    );
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
    check_iov_inner_matches_outer(&model, &subject, &theta, &stacked);
}

// Non-IOV counterpart of the TIME-readout probe — exercises the static `run_obs` /
// `run_obs_grad` readout guard (and the shared production guard) added in the #670 review.
const ONECPT_IV_TIME_READOUT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBETA(0.05, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BETA = TVBETA
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + BETA * TIME
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn form_c_time_readout_evaluated_at_obs_time() {
    let m = parse_model_string(ONECPT_IV_TIME_READOUT).expect("parse");
    assert!(analytical_supported(&m), "TIME readout stays analytic");
    let s = subject_with_dose(
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        &[0.5, 2.0, 6.0, 12.0],
    );
    let theta = [0.2, 10.0, 0.05];
    let eta = [0.12, -0.08];
    check_full_provider_vs_fd(&m, &s, &theta, &eta);
    let full = subject_sensitivities(&m, &s, &theta, &eta).expect("supported");
    let light = subject_eta_grad(&m, &s, &theta, &eta).expect("light supported");
    for (j, o) in full.obs.iter().enumerate() {
        approx::assert_relative_eq!(
            o.df_dtheta[2],
            s.obs_times[j],
            max_relative = 1e-6,
            epsilon = 1e-9
        );
    }
    for (lo, fo) in light.iter().zip(full.obs.iter()) {
        for k in 0..m.n_eta {
            approx::assert_relative_eq!(
                lo.df_deta[k],
                fo.df_deta[k],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }
}

/// Closed-form **IOV + modeled-duration** dose (`RATE=-2` → `D1`, #486). The
/// per-occasion `D1` sets each infusion window; the walk resolves it from the
/// occasion's stacked PK jet and the moving infusion-end boundary carries `∂/∂D1`
/// over θ, η, and κ. Doses at 0 and 24 (`D1 ≈ 5`), obs straddling each window end
/// (1 inside / 6 after; 25 inside / 30 after). Validated vs central FD of
/// `predict_iov` (which resolves `D1` per occasion) — the closed-form twin of the
/// ODE `ode_iov_modeled_duration_provider_matches_fd_of_predict_iov` (#635).
const WARFARIN_IOV_MODELED_DUR: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;

fn iov_modeled_dur_subject() -> Subject {
    let mut s = iov_subject();
    s.doses = vec![
        DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        DoseEvent::modeled(
            24.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
    ];
    s
}

#[test]
fn iov_modeled_duration_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_MODELED_DUR).expect("parse modeled-dur IOV");
    assert_eq!(model.n_kappa, 1);
    assert_eq!(model.n_eta, 3);
    assert!(model.ode_spec.is_none(), "must be a closed-form model");
    let subject = iov_modeled_dur_subject();
    assert!(!subject.all_doses_fixed(), "doses must be modeled");
    // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
    assert!(
        subject_sensitivities_iov(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.05, -0.10],
        )
        .is_some(),
        "modeled-duration closed-form IOV subject (no SS) must be served analytically (#486)"
    );
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, 0.05, -0.10],
    );
}

/// Inner (`Dual1`) IOV modeled-duration walk must match the FD-validated outer.
#[test]
fn iov_modeled_duration_inner_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_MODELED_DUR).expect("parse modeled-dur IOV"),
        &iov_modeled_dur_subject(),
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, 0.05, -0.10],
    );
}

/// The closed-form **inner** IOV walk (`Dual1`, `subject_eta_grad_iov_analytical`)
/// must produce the same per-observation value and `∂f/∂(stacked-η)` as the
/// **outer** walk (`Dual2`, `subject_sensitivities_iov`) — whose `df_deta` is
/// already validated against FD of `predict_iov`. Confirms the new first-order
/// analytical IOV inner agrees with the FD-validated second-order path (#439
/// closed-form IOV inner).
#[test]
fn analytical_iov_inner_eta_grad_matches_outer() {
    // 1-/2-/3-cpt oral closed-form IOV — the new Dual1 inner must track the
    // FD-validated Dual2 outer's first-order block on each.
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV).expect("parse 1cpt"),
        &iov_subject(),
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_2CPT).expect("parse 2cpt"),
        &iov_subject(),
        &[0.2, 10.0, 0.5, 20.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_3CPT).expect("parse 3cpt"),
        &iov_subject(),
        &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

// ------------------------------------------------------------------------------------
// Closed-form IOV + `ExpressionScale` `obs_scale` (#486): the closed-form twin of the
// ODE-IOV path (#575/#590). The scale divisor is a per-occasion-group post-walk quotient
// on the stacked `(θ, η_bsv, κ)` axes; each group's divisor rides its κ through the PK
// params. Validated against central FD of `predict_iov` (the independent production f64
// path), inner-vs-outer parity, and bit-parity with the ODE-IOV twin.
// ------------------------------------------------------------------------------------

/// 1-cpt oral closed-form IOV (κ on CL) with an η-dependent `obs_scale = V` divisor.
const WARFARIN_IOV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// 2-cpt oral closed-form IOV (κ on CL) with an `obs_scale = V` divisor — exercises the
/// per-group quotient over the wider PK-param chain.
const WARFARIN_IOV_EXPRSCALE_2CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// 1-cpt oral closed-form IOV + **WT-on-CL time-varying covariate** + `obs_scale = CL`.
/// The scale references a covariate-carrying PK param, but the divisor is still built at
/// the subject-static covariate snapshot (`t = 0`) — matching production `predict_iov`.
const WARFARIN_IOV_TVCOV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[scaling]
  obs_scale = CL
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// 1-cpt IV closed-form IOV (κ on CL) with `obs_scale = V` — the closed-form half of the
/// bit-parity twin with the ODE-IOV path. Closed-form `one_cpt_iv` yields `A/V`
/// (concentration), so `obs_scale = V` gives `A/V²`.
const WARFARIN_IOV_IV_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// The gate admits a closed-form IOV model carrying a differentiable `ExpressionScale`
/// `obs_scale`, but keeps declining LTBS, `ScalarScale`, and lagtime (out of scope for
/// the post-walk quotient) — the closed-form mirror of `ode_iov_expr_scale_supported_and_gated`.
#[test]
fn iov_analytical_expr_scale_supported_and_gated() {
    let model = parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse expr-scale IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(
        matches!(model.scaling, ScalingSpec::ExpressionScale { .. }),
        "fixture must use an expression obs_scale"
    );
    assert!(
        iov_analytical_supported(&model),
        "closed-form IOV + ExpressionScale obs_scale must be on the analytic path (#486)"
    );
    assert!(analytic_outer_gradient_available(&model));
    // + LTBS is served on the OUTER gradient now (#486): `subject_sensitivities_iov`
    // applies the `ln(f)` jet after the in-walk scale quotient, reproducing `ln(f/s)`.
    // (The inner EBE gradient still declines via `analytic_inner_common_bail`.)
    let mut ltbs = parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse");
    ltbs.log_transform = true;
    assert!(
        iov_analytical_supported(&ltbs),
        "ExpressionScale + LTBS under closed-form IOV is served on the outer gradient (#486)"
    );
    // A constant `ScalarScale` under IOV is now analytic too (#486 IOV-scope parity):
    // `f/k` is the trivial covariate/η/κ-independent case of the post-walk quotient,
    // divided uniformly into every jet entry — matching the non-IOV `f_scaled = f/k`.
    let mut scalar = parse_model_string(WARFARIN_IOV).expect("parse");
    scalar.scaling = ScalingSpec::ScalarScale(2.0);
    assert!(
        iov_analytical_supported(&scalar),
        "ScalarScale under closed-form IOV is now on the analytic path (#486 parity)"
    );
}

/// **Closed-form IOV + constant `ScalarScale` `obs_scale`** (#486 IOV-scope parity). A
/// uniform output divisor `f/k` under IOV, validated (value + gradient + Hessian over the
/// stacked `[η, κ]` vector + θ) against central FD of `predict_iov` (whose `apply_scaling`
/// applies the same `pred /= k`). Before this fix, `ScalarScale × IOV` declined to FD even
/// though the constant divide is strictly simpler than the `ExpressionScale` quotient the
/// IOV walk already applies.
#[test]
fn iov_analytical_scalar_scale_matches_fd() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse WARFARIN_IOV");
    model.scaling = ScalingSpec::ScalarScale(2.5);
    assert!(iov_analytical_supported(&model));
    check_iov_provider_vs_fd(
        &model,
        &iov_subject(),
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// The closed-form IOV **inner** η-gradient (`Dual1`, `run_obs_iov_eta`) with a constant
/// `ScalarScale` divisor must track the FD-validated **outer** walk's first-order block —
/// the inner twin of `iov_analytical_scalar_scale_matches_fd`, which only exercises the
/// outer `Dual2` path. Without this the inner `ScalarScale` divide (`f/k`, `∂f/∂η ÷ k`)
/// carries no coverage (#486 IOV-scope parity, review follow-up). The `k = 1.0` case
/// additionally pins the identity-scale no-op branch of the shared scaling helpers.
#[test]
fn iov_analytical_scalar_scale_inner_eta_grad_matches_outer() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse WARFARIN_IOV");
    model.scaling = ScalingSpec::ScalarScale(2.5);
    assert!(iov_analytical_supported(&model));
    check_iov_inner_matches_outer(
        &model,
        &iov_subject(),
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
    // Identity scale `k = 1.0`: the divide is a no-op on both walks, so inner still
    // tracks outer (covers the `k == 1.0` early-return arm of `scale_obs_sens_*`).
    model.scaling = ScalingSpec::ScalarScale(1.0);
    check_iov_inner_matches_outer(
        &model,
        &iov_subject(),
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// An **invalid** constant `ScalarScale` (`k <= 0` or non-finite) must NaN-out the whole
/// analytic IOV jet — value AND every derivative, on both the outer and inner walks —
/// matching `pk::apply_scaling` / `build_obs_scale_array`, which encode an invalid scale
/// as a `NaN` prediction (loud-failure semantics). Guards against the analytic path
/// emitting an `inf`/sign-flipped jet that would diverge from the production predictor the
/// FD oracle differentiates (Copilot review, #663).
#[test]
fn iov_scalar_scale_invalid_k_nans_the_jet() {
    let mut model = parse_model_string(WARFARIN_IOV).expect("parse WARFARIN_IOV");
    model.scaling = ScalingSpec::ScalarScale(-2.0);
    assert!(iov_analytical_supported(&model));
    let theta = [0.2, 10.0, 1.5];
    let stacked = [0.12, -0.08, 0.20, 0.05, -0.10];
    let outer =
        subject_sensitivities_iov(&model, &iov_subject(), &theta, &stacked).expect("supported");
    assert!(
        outer.obs.iter().all(|o| o.f.is_nan()
            && o.df_deta.iter().all(|x| x.is_nan())
            && o.df_dtheta.iter().all(|x| x.is_nan())
            && o.d2f_deta2.iter().all(|x| x.is_nan())
            && o.d2f_deta_dtheta.iter().all(|x| x.is_nan())),
        "invalid ScalarScale must NaN-out the whole outer jet"
    );
    let inner = subject_eta_grad_iov(&model, &iov_subject(), &theta, &stacked).expect("supported");
    assert!(
        inner
            .iter()
            .all(|o| o.f.is_nan() && o.df_deta.iter().all(|x| x.is_nan())),
        "invalid ScalarScale must NaN-out the whole inner jet"
    );
}

/// **ODE IOV + constant `ScalarScale` `obs_scale`** (#486 IOV-scope parity — the ODE twin
/// of `iov_analytical_scalar_scale_matches_fd`). On the ODE path the divide is even
/// simpler than closed-form: `resolve_obs_readout` → `apply_output_transform` already
/// divides the in-walk readout `p/k` over the stacked `(θ, η, κ)` dual (the same in-walk
/// step the non-IOV walk uses), so admitting `ScalarScale` in the ODE-IOV gate needs no
/// run-loop change. Validated vs central FD of `predict_iov`.
#[test]
fn ode_iov_scalar_scale_matches_fd_of_predict_iov() {
    const ODE_IOV_SCALARSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V)*central
[scaling]
  obs_scale = 40
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(ODE_IOV_SCALARSCALE).expect("parse ODE IOV ScalarScale");
    assert!(
        matches!(model.scaling, ScalingSpec::ScalarScale(_)),
        "fixture must use a constant obs_scale"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + constant ScalarScale must be on the analytic path (#486 parity)"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
    check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.06, -0.11]);
}

/// Outer packed sensitivities of a closed-form IOV + `obs_scale = V` model must match
/// central FD of `predict_iov` (value, `∂f/∂stacked`, `∂f/∂θ`, and both Hessian blocks)
/// over `[η_bsv, κ_g0, κ_g1]` + θ — 1-cpt and 2-cpt oral.
#[test]
fn iov_analytical_expr_scale_outer_matches_fd() {
    check_iov_provider_vs_fd(
        &parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse 1cpt expr-scale IOV"),
        &iov_subject(),
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
    check_iov_provider_vs_fd(
        &parse_model_string(WARFARIN_IOV_EXPRSCALE_2CPT).expect("parse 2cpt expr-scale IOV"),
        &iov_subject(),
        &[0.2, 10.0, 0.5, 20.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// Closed-form IOV + **TV covariate** + `obs_scale = CL`: the scale divisor stays
/// subject-static (built at `t = 0`) even though the walk seeds each event per-covariate.
/// Value/grad/Hessian over `[η_bsv, κ_g0, κ_g1]` + θ (incl. `THETA_WT`) vs FD of `predict_iov`.
#[test]
fn iov_analytical_tvcov_expr_scale_outer_matches_fd() {
    let model =
        parse_model_string(WARFARIN_IOV_TVCOV_EXPRSCALE).expect("parse TV-cov expr-scale IOV");
    let subject = iov_tvcov_subject(false);
    assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5, 0.75],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// The closed-form IOV **inner** η-gradient (`Dual1`) with an `obs_scale` divisor must
/// track the FD-validated **outer** walk's first-order block — plain and TV-cov.
#[test]
fn iov_analytical_expr_scale_inner_eta_grad_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_EXPRSCALE).expect("parse expr-scale IOV"),
        &iov_subject(),
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_TVCOV_EXPRSCALE).expect("parse TV-cov expr-scale IOV"),
        &iov_tvcov_subject(false),
        &[0.2, 10.0, 1.5, 0.75],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// **Bit-parity with the ODE-IOV twin (#575/#590).** Closed-form `one_cpt_iv` + `obs_scale
/// = V` computes `A/V²`; the same physical model as a user `[odes]` model reaches `A/V²`
/// two independent, already-analytic ways: an `obs_scale = V*V` divisor (the ODE half of
/// the very quotient this PR ports) and a Form-C readout `y = central/(V*V)` (the in-walk
/// division). Per-observation value and `∂f/∂stacked` must agree across all three paths —
/// confirming the closed-form quotient reproduces the NONMEM-validated ODE result.
#[test]
fn iov_analytical_expr_scale_equals_ode_twin() {
    const ODE_OBS_SCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V * V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    const ODE_FORMC: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / (V * V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let cf = parse_model_string(WARFARIN_IOV_IV_EXPRSCALE).expect("parse closed-form IV");
    let ode_scale = parse_model_string(ODE_OBS_SCALE).expect("parse ODE obs_scale twin");
    let ode_formc = parse_model_string(ODE_FORMC).expect("parse ODE Form-C twin");
    assert!(iov_analytical_supported(&cf));
    assert!(crate::sens::ode_provider::ode_iov_supported(&ode_scale));
    assert!(crate::sens::ode_provider::ode_iov_supported(&ode_formc));
    let subject = iov_subject();
    let theta = [0.2, 10.0];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    let a = subject_sensitivities_iov(&cf, &subject, &theta, &stacked).expect("closed-form");
    for twin in [&ode_scale, &ode_formc] {
        let b = subject_sensitivities_iov(twin, &subject, &theta, &stacked).expect("ODE twin");
        assert_eq!(a.obs.len(), b.obs.len());
        for (oa, ob) in a.obs.iter().zip(b.obs.iter()) {
            approx::assert_relative_eq!(oa.f, ob.f, max_relative = 1e-6, epsilon = 1e-9);
            for (x, y) in oa.df_deta.iter().zip(ob.df_deta.iter()) {
                approx::assert_relative_eq!(x, y, max_relative = 1e-5, epsilon = 1e-8);
            }
        }
    }
}

/// 1-cpt IV IOV written as a user `[odes]` model (κ on CL, Form-C readout
/// `y = central/V`). Routes through `subject_sensitivities_iov` → the ODE IOV
/// `y = central/V`). Routes through `subject_sensitivities_iov` → the ODE IOV
/// provider, validated against central FD of the production `predict_iov` (which
/// integrates the same ODE via `ode_predictions_event_driven`). This is the ODE
/// counterpart of `iov_provider_matches_fd_of_predict_iov`, proving the
/// per-occasion κ-axis seeding + event-driven dual walk compose the exact stacked
/// (η_bsv, κ, θ) gradient (#439 ODE IOV).
const WARFARIN_IOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_iov_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    assert_eq!(model.n_kappa, 1, "model must carry one kappa");
    assert!(model.ode_spec.is_some(), "must be an ODE model");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "1-cpt IV ODE IOV must be ODE-IOV-provider supported"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
    check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
}

/// #486: a `TIME`-switched CL on an **ODE IOV** model (Form-C `y = central/V`
/// readout). With no TV covariates the per-event stacked walk is reached purely by
/// `uses_time_builtin`; the value/∂/∂² over `[η_bsv, κ_g0, κ_g1]` + θ must match
/// central FD of `predict_iov` (which threads the same per-event TIME through the
/// ODE event-driven predictor).
#[test]
fn ode_iov_time_builtin_provider_matches_fd_of_predict_iov() {
    const WARFARIN_IOV_ODE_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 20.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(WARFARIN_IOV_ODE_TIME).expect("parse ODE IOV TIME");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    assert!(!iov_subject().has_tv_covariates());
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV TIME (Form-C readout) must be provider-supported"
    );
    // obs [1,6,12 | 25,30,36] straddle TIME=20. stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &iov_subject(),
        &[0.2, 0.1, 10.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn ode_iov_above_legacy_axis_cap_stays_analytic() {
    // A wide `Dual2<M>` (M = 29 here) carries an M×M Hessian per value, and the ODE walk
    // holds several frames at once — more than the 2 MiB default test-thread stack once the
    // fixture clears the raised 24-axis cap. Production already runs fits on a 32 MiB Rayon
    // stack for exactly this reason (`api::FIT_RAYON_STACK_SIZE`), so mirror that here
    // rather than shrinking the fixture back under the cap it exists to exceed.
    std::thread::Builder::new()
        .stack_size(crate::api::FIT_RAYON_STACK_SIZE)
        .spawn(ode_iov_above_legacy_axis_cap_body)
        .expect("spawn wide-stack test thread")
        .join()
        .expect("wide ODE IOV sensitivity test panicked");
}

fn ode_iov_above_legacy_axis_cap_body() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    // Wide enough that the stacked `(θ, η, κ…)` axis count clears the ordinary ODE axis
    // cap (raised 16 → 24 in #486) while staying inside `MAX_ODE_IOV_AXES` — the point of
    // the test is that a high-occasion IOV subject rides the *IOV* cap, not the plain one.
    let n_occ = 25;
    let obs_times: Vec<f64> = (0..n_occ).map(|i| i as f64 * 24.0 + 1.0).collect();
    let occasions: Vec<u32> = (1..=n_occ as u32).collect();
    let doses: Vec<DoseEvent> = (0..n_occ)
        .map(|i| DoseEvent::new(i as f64 * 24.0, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let n = obs_times.len();
    let subject = Subject {
        id: "wide-iov".to_string(),
        doses,
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: (1..=n_occ as u32).collect(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let theta = vec![0.2, 10.0];
    let mut stacked = vec![0.0; model.n_eta + n_occ * model.n_kappa];
    stacked[0] = 0.12;
    stacked[1] = -0.08;
    for g in 0..n_occ {
        stacked[model.n_eta + g] = 0.02 * (g as f64 - 7.0);
    }
    let m_dim = model.n_theta + stacked.len();
    assert!(
        m_dim > crate::sens::ode_provider::MAX_ODE_AXES,
        "fixture must exceed the ordinary ODE axis cap"
    );
    assert!(
        m_dim <= crate::sens::ode_provider::MAX_ODE_IOV_AXES,
        "fixture must stay within the widened IOV cap"
    );
    let full = crate::sens::ode_provider::ode_subject_sensitivities_iov(
        &model, &subject, &theta, &stacked,
    )
    .expect("wide ODE IOV outer gradient should be analytic");
    let light =
        crate::sens::ode_provider::ode_subject_eta_grad_iov(&model, &subject, &theta, &stacked)
            .expect("wide ODE IOV inner gradient should be analytic");
    assert_eq!(full.obs.len(), light.len());
    for (outer, inner) in full.obs.iter().zip(light.iter()) {
        approx::assert_relative_eq!(outer.f, inner.f, max_relative = 1e-10, epsilon = 1e-12);
        for k in 0..stacked.len() {
            approx::assert_relative_eq!(
                outer.df_deta[k],
                inner.df_deta[k],
                max_relative = 1e-8,
                epsilon = 1e-10
            );
        }
    }
}

/// Regression guard for the ODE IOV worker-stack overflow (#601): a PNA-scale,
/// 86-occasion subject yields a `Dual2<90>` (90×90 Hessian per dual) whose
/// event-walk frames overflow the platform-default (~2 MiB) Rayon worker stack. The
/// gradient is run on [`crate::api::default_fit_pool`] — the *same* pool `fit()` uses
/// by default — so dropping the 32 MiB stack from that pool re-introduces the crash
/// here. Heavy (full wide-`M` sensitivity through RK45), so it is gated to the
/// nightly slow-tests tier rather than the fast per-PR job.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn fit_rayon_stack_handles_pna_scale_ode_iov_gradient() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    let n_occ = 86;
    let obs_times: Vec<f64> = (0..n_occ).map(|i| i as f64 * 24.0 + 1.0).collect();
    let occasions: Vec<u32> = (1..=n_occ as u32).collect();
    let doses: Vec<DoseEvent> = (0..n_occ)
        .map(|i| DoseEvent::new(i as f64 * 24.0, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let n = obs_times.len();
    let subject = Subject {
        id: "pna-scale-iov".to_string(),
        doses,
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: (1..=n_occ as u32).collect(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let theta = vec![0.2, 10.0];
    let stacked = vec![0.0; model.n_eta + n_occ * model.n_kappa];
    let m_dim = model.n_theta + stacked.len();
    assert_eq!(m_dim, 90, "fixture mirrors the PNA-scale occasion width");

    // Run on the actual default fit pool, so a regression that drops the big stack
    // from `default_fit_pool` (or `fit_thread_pool_builder`) overflows here.
    let pool = crate::api::default_fit_pool().expect("ferx default fit pool");
    pool.install(|| {
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model, &subject, &theta, &stacked,
        )
        .expect("PNA-scale ODE IOV gradient should fit on ferx worker stack");
    });
}

/// A dose in an occasion that carries no sampled observations still gets its own κ
/// axis. That kappa can affect later observations through carryover, so the ODE IOV
/// provider must keep the subject on the analytic path rather than falling back to FD.
#[test]
fn ode_iov_dose_only_occasion_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    let mut subject = iov_subject();
    // Add a dose between the two observed occasions. Occasion 3 has no observations, but
    // its CL kappa affects the post-dose amount that carries into occasion 2.
    subject
        .doses
        .push(DoseEvent::new(18.0, 100.0, 1, 0.0, false, 0.0));
    subject.dose_occasions.push(3);
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0],
        &[0.12, -0.08, 0.05, -0.10, 0.08],
    );
}

/// **ODE IOV + EVID 3/4 reset.** A two-occasion ODE IOV subject with a washout reset
/// (+ re-dose) at the occasion boundary. The event-driven walk zeros the dual state at
/// the reset (no cross-occasion carryover) and the per-occasion κ seeding continues on
/// the post-reset occasion. Validated vs FD of `predict_iov` (#439 IOV × reset).
#[test]
fn ode_iov_reset_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let subject = iov_reset_subject();
    assert!(subject.has_resets());
    check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
}

/// **ODE IOV + infusion.** Two-occasion IOV with finite-duration infusions; the
/// event-driven walk applies the per-occasion `F·rate` forcing over each window.
/// Validated vs FD of `predict_iov` (#439 IOV × infusion).
#[test]
fn ode_iov_infusion_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 50.0, false, 0.0),
    ];
    assert!(subject.doses[0].is_infusion());
    check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
}

/// 1-cpt IV ODE IOV with a **modeled-duration** dose (`RATE=-2` → `D1`). `D1` is a
/// structural individual parameter (`D1 = TVD1·exp(ETA_D1)`), so the infusion window
/// end `t_dose + D1` is a moving boundary in `D1`; the per-occasion rate-off saltation
/// carries its derivative on the IOV stacked axes exactly as on the non-IOV TV-cov walk
/// (#486 / #530). κ rides on CL here (the modeled slot itself is η-only).
const WARFARIN_IOV_ODE_MODELED_DUR: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **ODE IOV + modeled-duration dose** (#486, design A). A two-occasion IOV subject
/// whose per-occasion modeled `D1` sets each infusion's window length. The walk resolves
/// `D1` from the per-occasion stacked PK jet (`inf_eff` → `pk_at_dose[k][slot]`) and the
/// moving infusion-end saltation carries `∂/∂D1` over θ, η, and κ. Validated vs central
/// FD of `predict_iov` (which resolves `D1` per occasion). Observations straddle each
/// window end (`D1 ≈ 5`: obs at 1 inside, 6 after the first window; 25 inside, 30 after
/// the second) so the moving boundary is genuinely exercised.
#[test]
fn ode_iov_modeled_duration_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE_MODELED_DUR).expect("parse modeled-dur IOV");
    assert_eq!(model.n_kappa, 1);
    assert_eq!(model.n_eta, 3);
    assert!(model.ode_spec.is_some());
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        DoseEvent::modeled(
            24.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
    ];
    assert!(
        !subject.all_doses_fixed(),
        "doses must be modeled, not fixed"
    );
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model,
            &subject,
            &[0.2, 10.0, 5.0],
            &[0.12, -0.08, 0.05, 0.05, -0.10],
        )
        .is_some(),
        "modeled-duration ODE IOV subject (no SS) must be served analytically (#486)"
    );
    // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, 0.05, -0.10],
    );
}

/// **ODE IOV + κ-coupled modeled-duration dose** (#486, design A — the κ-coupling guard).
/// Here the modeled window itself varies by occasion: `D1 = TVD1·exp(ETA_D1 + KAPPA_D1)`,
/// so each occasion's infusion has a *different* length. This pins the concern that the
/// `∂/∂D1` moving-boundary column lands in the correct κ-group axis — `inf_eff` reads the
/// per-occasion `seed_pk_dual2_iov` jet, and central FD of `predict_iov` (which rebuilds
/// `D1` from each occasion's κ) is the independent oracle.
#[test]
fn ode_iov_modeled_duration_kappa_coupled_matches_fd_of_predict_iov() {
    const KCOUPLED: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_D1 ~ 0.04
  kappa KAPPA_D1 ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  D1 = TVD1 * exp(ETA_D1 + KAPPA_D1)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(KCOUPLED).expect("parse kappa-coupled modeled-dur IOV");
    assert_eq!(model.n_kappa, 1);
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
        DoseEvent::modeled(
            24.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        ),
    ];
    // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1]; κ on D1 → each occasion's window differs.
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, 0.06, -0.11],
    );
}

/// **Still-FD edge: modeled-duration dose + steady-state** (#486, design B not yet done).
/// The dual SS equilibration reads a fixed per-cycle `t_inf` with no modeled-window jet,
/// so a modeled + SS subject must route to FD on BOTH the outer sensitivity walk and the
/// inner η-gradient (scope parity). Pins the `has_ss` arm of the relaxed IOV gate.
/// **ODE IOV + modeled-duration dose × steady-state is now analytic (#486, PR3
/// sub-case (d)).** `equilibrate_ss_state_g` threads the per-occasion `inf_eff` jet
/// (`D1` seeded per occasion group, same as the non-SS modeled-duration IOV test) into
/// its per-cycle active/quiet split. Validated vs central FD of `predict_iov` (both
/// outer and inner, scope parity).
#[test]
fn ode_iov_modeled_duration_ss_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE_MODELED_DUR).expect("parse modeled-dur IOV");
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::modeled(
            0.0,
            100.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        ),
        DoseEvent::modeled(
            24.0,
            100.0,
            1,
            true,
            12.0,
            crate::types::RateMode::ModeledDuration,
        ),
    ];
    let theta = [0.2, 10.0, 5.0];
    let stacked = [0.12, -0.08, 0.05, 0.05, -0.10];
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model, &subject, &theta, &stacked
        )
        .is_some(),
        "modeled-duration + SS must be analytic now (outer, #486)"
    );
    assert!(
        crate::sens::ode_provider::ode_subject_eta_grad_iov(&model, &subject, &theta, &stacked)
            .is_some(),
        "modeled-duration + SS must be analytic now (inner, scope parity)"
    );
    // stacked = [η_cl, η_v, η_d1, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
}

const ZERO_ORDER_IOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVDUR(5.0, 0.1, 24.0)
  omega ETA_CL  ~ 0.09
  omega ETA_V   ~ 0.04
  omega ETA_DUR ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL + KAPPA_CL)
  V   = TVV  * exp(ETA_V)
  DUR = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **ODE IOV + `zero_order(dur)` absorption** (#486). The last zero-order gap after
/// #653 ported the moving-boundary window to the non-IOV event-driven walk: the IOV
/// walk is the *same* `integrate_tvcov_g`, so once `ode_iov_supported` admits a
/// `ZeroOrder` forcing the κ-coupled window rides through — its rate `F·amt/dur` and
/// window end `t_dose + dur` are rebuilt from each dose's own per-occasion stacked PK
/// jet (`pk_at_dose[k]`, seeded by `seed_pk_dual2_iov`). Validated vs central FD of
/// `predict_iov`. `DUR ≈ 5`, so obs at 1 (inside) / 6 (after) straddle the first window
/// end and 25 / 30 the second, exercising the rate-off saltation per occasion.
#[test]
fn ode_iov_zero_order_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(ZERO_ORDER_IOV_ODE).expect("parse zero_order IOV");
    assert_eq!(model.n_kappa, 1);
    assert_eq!(model.n_eta, 3);
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "zero_order model must be admitted under IOV (#486)"
    );
    let subject = iov_subject();
    let theta = [0.2, 10.0, 5.0];
    // stacked = [η_cl, η_v, η_dur, κ_g0, κ_g1] (n_eta = 3, n_kappa = 1, K = 2).
    let stacked = [0.12, -0.08, 0.05, 0.05, -0.10];
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model, &subject, &theta, &stacked
        )
        .is_some(),
        "zero_order ODE IOV subject (no SS) must be served analytically (#486)"
    );
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
}

/// **ODE IOV + κ-coupled `zero_order(dur)`** (#486 — the κ-axis-placement guard). Here
/// the window itself varies by occasion (`DUR = TVDUR·exp(ETA_DUR + KAPPA_DUR)`), so each
/// occasion's zero-order window has a different length. This pins that the `∂/∂DUR`
/// rate-off column lands in the correct κ-group axis of the stacked vector (central FD of
/// `predict_iov`, which rebuilds `DUR` from each occasion's κ, is the independent oracle).
#[test]
fn ode_iov_zero_order_kappa_coupled_matches_fd_of_predict_iov() {
    const KCOUPLED: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVDUR(5.0, 0.1, 24.0)
  omega ETA_CL  ~ 0.09
  omega ETA_V   ~ 0.04
  omega ETA_DUR ~ 0.04
  kappa KAPPA_DUR ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  DUR = TVDUR * exp(ETA_DUR + KAPPA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(KCOUPLED).expect("parse κ-coupled zero_order IOV");
    let subject = iov_subject();
    // stacked = [η_cl, η_v, η_dur, κ_g0, κ_g1]; κ on DUR → each occasion's window differs.
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, 0.06, -0.11],
    );
}

const MIXED_IOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVFZO(0.4, 0.05, 0.95)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVDUR(5.0, 0.1, 24.0)
  omega ETA_CL  ~ 0.09
  omega ETA_DUR ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL + KAPPA_CL)
  V    = TVV
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR * exp(ETA_DUR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **ODE IOV + `mixed` (first-order + zero-order) absorption** (#486). Exercises both
/// admitted forcing kinds together: the pointwise first-order `R_in` and the zero-order
/// window are delivered concurrently per occasion, and the `frac` (FZO / FZO1) multiplier
/// rides each rate. Validated vs central FD of `predict_iov`.
#[test]
fn ode_iov_mixed_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(MIXED_IOV_ODE).expect("parse mixed IOV");
    assert_eq!(model.n_kappa, 1);
    assert_eq!(model.n_eta, 2);
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "mixed (first_order + zero_order) must be admitted under IOV (#486)"
    );
    let subject = iov_subject();
    let theta = [0.2, 10.0, 0.4, 1.0, 5.0];
    // stacked = [η_cl, η_dur, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
    let stacked = [0.12, 0.05, 0.06, -0.11];
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
}

/// **Still-FD edge: `zero_order` window + steady state under IOV** (#486, narrowed by #835).
/// The dual SS input-rate equilibration (`equilibrate_ss_input_rate_state_g`) now spreads a
/// periodic *smooth-kernel* `R_in` tail (first_order/transit/igd/weibull) over the cycle
/// analytically, but a `zero_order` spanning window is not built by the pointwise fixed point,
/// so a `zero_order` + SS subject still routes to FD on BOTH loops. (The `weibull` + lagtime FD
/// case is pinned separately by `ode_iov_weibull_lagtime_falls_back_to_fd`.)
#[test]
fn ode_iov_zero_order_ss_falls_back_to_fd() {
    let model = parse_model_string(ZERO_ORDER_IOV_ODE).expect("parse zero_order IOV");
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, true, 12.0),
    ];
    let theta = [0.2, 10.0, 5.0];
    let stacked = [0.12, -0.08, 0.05, 0.05, -0.10];
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model, &subject, &theta, &stacked
        )
        .is_none(),
        "zero_order + SS must route to FD under IOV (outer, #486)"
    );
    assert!(
        crate::sens::ode_provider::ode_subject_eta_grad_iov(&model, &subject, &theta, &stacked)
            .is_none(),
        "zero_order + SS must route to FD under IOV (inner, scope parity)"
    );
}

/// **ODE IOV + `parallel` dual-pathway absorption** (#486 review — the multi-forcing
/// case). Two concurrent `first_order` forcings (`FR1·first_order(ka1) +
/// FR2·first_order(ka2)`) feed the central compartment. The gate admits it (both kinds
/// are `FirstOrder`); this pins that the per-occasion accumulation of two `R_in`
/// contributions stays analytic and matches central FD of `predict_iov`.
#[test]
fn ode_iov_parallel_provider_matches_fd_of_predict_iov() {
    const PARALLEL_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  theta TVFR1(0.6, 0.05, 0.95)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL + KAPPA_CL)
  V   = TVV  * exp(ETA_V)
  KA1 = TVKA1
  KA2 = TVKA2
  FR1 = TVFR1
  FR2 = 1 - TVFR1
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(PARALLEL_IOV).expect("parse parallel IOV");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "parallel (two first_order) must be admitted under IOV (#486)"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5, 0.3, 0.6],
        &[0.12, -0.08, 0.06, -0.11],
    );
}

/// **ODE IOV × a logit-transformed absorption fraction** — inter-occasion variability on the
/// pathway split itself. A parallel model where the immediate-release fraction carries both
/// IIV and IOV on the **logit** scale, `FR1 = inv_logit(logit(TVFR1) + ETA_FR + KAPPA_FR)`
/// (so `FR1 ∈ (0,1)` for every occasion draw and `FR2 = 1 − FR1` keeps the split valid — an
/// `exp(κ)` fraction would let `κ > 0` push `FR1 > 1`). This pins that `∂f/∂κ` **through the
/// pathway fraction** (`frac_slot` seeded per occasion, then the logit inverse's chain rule)
/// is exact against central FD of `predict_iov` — the fraction's occasion sensitivity, not
/// just clearance's.
#[test]
fn ode_iov_logit_fraction_provider_matches_fd_of_predict_iov() {
    const LOGIT_FRAC_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  theta TVFR1(0.6, 0.05, 0.95)
  omega ETA_CL ~ 0.09
  omega ETA_FR ~ 0.04
  kappa KAPPA_FR ~ 0.02
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA1 = TVKA1
  KA2 = TVKA2
  FR1 = inv_logit(logit(TVFR1) + ETA_FR + KAPPA_FR)
  FR2 = 1 - FR1
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(LOGIT_FRAC_IOV).expect("parse logit-fraction IOV");
    assert_eq!(model.n_kappa, 1, "one IOV kappa (on the fraction)");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "IOV on a logit fraction must be admitted (κ is an ordinary axis on frac_slot)"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_fr, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5, 0.3, 0.6],
        &[0.12, -0.08, 0.06, -0.11],
    );
}

/// **ODE IOV + `first_order` absorption × estimated (bare) lagtime** (#486 review). The
/// non-IOV path admits `first_order` + lagtime (#643 onset saltation `Δr = R_in(0⁺)`);
/// the IOV walk is the same, so it must too. Pins that the per-occasion rate-on onset
/// saltation at the lagged arrival stays analytic under κ (FD of `predict_iov` is the
/// oracle). Uses a **bare** `LAGTIME`; the compartment-indexed `ALAG{cmt}` case is now
/// analytic too and covered by `ode_iov_indexed_lag_and_f_match_fd_of_predict_iov`.
#[test]
fn ode_iov_first_order_lagtime_matches_fd_of_predict_iov() {
    const FO_LAG_IOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL + KAPPA_CL)
  V       = TVV  * exp(ETA_V)
  KA      = TVKA
  LAGTIME = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(FO_LAG_IOV).expect("parse first_order+lag IOV");
    assert!(model.has_lagtime());
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "first_order + lagtime must be admitted under IOV (#486)"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.0, 0.5],
        &[0.12, -0.08, 0.06, -0.11],
    );
}

/// **ODE IOV + `zero_order` absorption × EVID 3/4 reset** (#486 review). Resets are
/// admitted by `ode_iov_subject_supported` and the zero-order window is kind-agnostic to
/// the `reset_floor` cutoff (#653), so the combination stays analytic under IOV. A reset
/// at `t = 15` (after the first window, before the second dose) zeros the state; central
/// FD of `predict_iov` (which cancels the pre-reset trajectory) is the oracle.
#[test]
fn ode_iov_zero_order_reset_matches_fd_of_predict_iov() {
    let model = parse_model_string(ZERO_ORDER_IOV_ODE).expect("parse zero_order IOV");
    let mut subject = iov_subject();
    subject.reset_times = vec![15.0];
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 5.0],
        &[0.12, -0.08, 0.05, 0.05, -0.10],
    );
}

// #486 IOV-scope parity: build a 1-cpt-depot ODE IOV model whose depot is fed by one
// smooth-density absorption forcing `<forcing>`, κ on CL. These were declined under IOV
// before the gate mirrored the non-IOV `supported_over_dual()` allowlist.
fn smooth_forcing_iov_model(forcing: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVP(1.5, 0.05, 24.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
  PP = TVP
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = {forcing} - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#
    )
}

/// **ODE IOV + `igd` / `transit` / `weibull` absorption** (#486 IOV-scope parity). The
/// non-IOV gate admits every input-rate kind via the kind-agnostic `supported_over_dual()`;
/// the IOV walk is the *same* `integrate_tvcov_g`, so it must too. Before this parity fix
/// the IOV gate hard-restricted to `{ZeroOrder, FirstOrder}`, declining these smooth
/// densities for no technical reason. Each is validated (value + gradient + Hessian over
/// the stacked `[η, κ]` vector) against central FD of `predict_iov`.
#[test]
fn ode_iov_smooth_density_forcings_match_fd_of_predict_iov() {
    for forcing in [
        "igd(mat=PP, cv2=KA)",
        "transit(n=PP, mtt=KA)",
        "weibull(td=KA, beta=PP)",
    ] {
        let src = smooth_forcing_iov_model(forcing);
        let model = parse_model_string(&src).unwrap_or_else(|e| panic!("parse {forcing}: {e}"));
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&model),
            "{forcing} must be admitted under IOV (#486 parity)"
        );
        let subject = iov_subject();
        // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
        check_iov_provider_vs_fd(
            &model,
            &subject,
            &[0.2, 10.0, 1.0, 1.5],
            &[0.12, -0.08, 0.06, -0.11],
        );
    }
}

/// **`weibull` + estimated lagtime stays FD under IOV** (#486 parity — the one input-rate
/// kind that stays FD with lagtime on every path, its onset diverging for shape `β < 1`).
/// Mirrors the non-IOV decline; pins that broadening the IOV input-rate allowlist did not
/// accidentally admit this divergent combination.
#[test]
fn ode_iov_weibull_lagtime_falls_back_to_fd() {
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVTD(2.0, 0.05, 24.0)
  theta TVBETA(1.5, 0.1, 10.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL + KAPPA_CL)
  V       = TVV  * exp(ETA_V)
  TD      = TVTD
  BETA    = TVBETA
  LAGTIME = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let model = parse_model_string(src).expect("parse weibull+lag IOV");
    assert!(model.has_lagtime());
    assert!(
        !crate::sens::ode_provider::ode_iov_supported(&model),
        "weibull + lagtime must stay FD under IOV (β<1 onset divergence, #486)"
    );
}

// ─── #877: analytic per-route absorption-lag gradients under IOV ─────────────────────────────
// #859/#875 made per-route lag (`fn(..., lag=L)`) analytic on the NON-IOV event-driven walk;
// #877 lifts the IOV decline so a route-lag model with inter-occasion variability gets exact
// analytic FOCE/FOCEI gradients too. The IOV walk is the *same* `integrate_tvcov_g` — the gate
// flip admits `first_order`/`zero_order`/`transit`/`igd` route lags (weibull stays FD), and the
// onset saltation + its #880/#883 δ² onset-slope curvature carry κ through the per-occasion
// `pk_at_dose[k]` jet exactly as η/θ. Each fixture is validated value + gradient + **Hessian**
// against central FD of `predict_iov` (`check_iov_provider_vs_fd`). Crucially these are NOT
// happy-path: the δ² onset-slope term is `dr_dtad_value · (∂lag/∂axis_i)(∂lag/∂axis_j)`, so it
// lands on a checked stacked (η/κ) Hessian axis ONLY when a lag carries that axis's jet — a
// fixed-θ lag would hide the term in the unchecked `d2f_dtheta2` block. So the lag itself carries
// η and/or κ here, the axis where the FOCEI Hessian would be wrong without #883.

/// #877 teeth: `first_order` per-route lag with **both η and κ on the lag** (`LAG` carries
/// `ETA_LAG + KAPPA_LAG`). The δlag² onset-slope curvature (`½·∂Δr/∂tad`, #880/#883) therefore
/// lands on the checked `d2f_deta2` blocks — `κ×κ`, `κ×η_LAG`, `η_LAG×η_LAG` — the exact
/// FOCEI-Hessian axes that were wrong before #883. Value + gradient + Hessian must match central
/// FD of `predict_iov` over the stacked `[η_CL, η_LAG, κ_g0, κ_g1]`.
#[test]
fn ode_iov_first_order_route_lag_iiv_iov_on_lag_matches_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.04
  kappa KAPPA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG * exp(ETA_LAG + KAPPA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(src).expect("parse first_order route-lag IOV (η+κ on lag)");
    assert_eq!(model.n_kappa, 1);
    let ode = model.ode_spec.as_ref().expect("ode spec");
    assert_eq!(
        ode.input_rate
            .iter()
            .filter(|f| f.lag_slot.is_some())
            .count(),
        1,
        "the forcing carries a per-route lag"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: first_order route lag must now be analytic under IOV"
    );
    let subject = iov_subject();
    // stacked = [η_CL, η_LAG, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2). Values chosen so
    // each occasion's onset (t_dose + LAG) stays ≥ 0.4 h from every observation — no obs
    // crosses an onset under the FD steps, so the Hessian FD stays kink-free.
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[1.0, 20.0, 1.2, 1.5],
        &[0.1, 0.05, 0.03, -0.04],
    );
}

/// #877 × #880/#883 onset snapshot: `first_order` route lag with a **TV covariate on the onset
/// kernel `KA`** (`KA = TVKA·exp(KAPPA_KA + KA_WT·(WT−70))`) crossing the onset, plus κ on KA.
/// The onset saltation reads `ka`/`frac`/post-Jacobian from the POST-arrival record snapshot
/// (#883), not `last_params`; under a TV cov those diverge, a several-percent gradient error on
/// the **κ axis** here. Guards that the onset-snapshot fix carries κ correctly under IOV.
#[test]
fn ode_iov_first_order_route_lag_tvcov_on_kernel_matches_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  theta KA_WT(0.01, -0.1, 0.1)
  omega ETA_CL ~ 0.09
  kappa KAPPA_KA ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA * exp(KAPPA_KA + KA_WT * (WT - 70))
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(src).expect("parse first_order route-lag IOV + TV cov");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: first_order route lag + TV cov must be analytic under IOV"
    );
    // Per-record WT so the onset kernel KA jumps across the route onset (dose record WT=70,
    // observations 60..85) — the case the onset-segment snapshot fixes.
    let mut subject = iov_subject();
    subject.dose_covariates = vec![
        std::collections::HashMap::from([("WT".to_string(), 70.0)]),
        std::collections::HashMap::from([("WT".to_string(), 70.0)]),
    ];
    subject.obs_covariates = (0..subject.obs_times.len())
        .map(|i| std::collections::HashMap::from([("WT".to_string(), 60.0 + 5.0 * (i as f64))]))
        .collect();
    assert!(subject.has_tv_covariates(), "WT must be time-varying");
    // stacked = [η_CL, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[1.0, 20.0, 1.2, 1.5, 0.01],
        &[0.1, 0.03, -0.04],
    );
}

/// #877: `zero_order` per-route lag with **κ on the lag**. A route-lagged zero-order window
/// shifts BOTH boundaries with `lag_route` — the rate-on saltation at `K_ROUTE_ONSET` and the
/// rate-off at `K_ZO_END` — so both must carry the κ jet. Its onset slope is zero (constant
/// window rate), so this isolates the moving-window κ sensitivity (distinct from the
/// first_order δ² term). Value + gradient + Hessian vs FD of `predict_iov`.
#[test]
fn ode_iov_zero_order_route_lag_kappa_on_lag_matches_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVDUR(2.0, 0.1, 12.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  DUR = TVDUR
  LAG = TVLAG * exp(KAPPA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(src).expect("parse zero_order route-lag IOV");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: zero_order route lag must be analytic under IOV"
    );
    let subject = iov_subject();
    // stacked = [η_CL, κ_g0, κ_g1]. Windows [onset, onset+2] at ~1.5 and ~25.5 clear the
    // observation grid's kink-free zones.
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[1.0, 20.0, 2.0, 1.5],
        &[0.1, 0.03, -0.04],
    );
}

/// #877: `transit` per-route lag with **κ on the lag**. Unlike `first_order`/`zero_order`, the
/// transit onset is *continuous* (`R_in → 0` smoothly at `t_dose + LAG`), so `K_ROUTE_ONSET`
/// injects a **zero-magnitude** saltation (`rate_at_zero` = 0 for `Transit`) — the κ-sensitivity
/// rides entirely through the continuous `∂R_in/∂lag_route` of the shifted gamma density, over the
/// per-occasion `pk_at_dose[k]` jet. This is the direct Dual2-vs-FD parity the gate's admission of
/// `Transit` under IOV requires (the `route_lag_analytic()` classifier admits it; only
/// `first_order`/`zero_order` had a κ-on-lag fixture before). Value + gradient + Hessian vs FD.
#[test]
fn ode_iov_transit_route_lag_kappa_on_lag_matches_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVN(5.0, 1.0, 30.0)
  theta TVMTT(2.0, 0.1, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  LAG = TVLAG * exp(KAPPA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = transit(n=N, mtt=MTT, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let model = parse_model_string(src).expect("parse transit route-lag IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(model.has_route_absorption_lag(), "per-route lag present");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: transit route lag must be analytic under IOV"
    );
    let subject = iov_subject();
    // stacked = [η_CL, κ_g0, κ_g1]. Onsets t_dose + LAG ≈ 1.5 / 25.5 clear every obs.
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[1.0, 20.0, 5.0, 2.0, 1.5],
        &[0.1, 0.03, -0.04],
    );
}

/// #877: `igd` (inverse-Gaussian density) per-route lag with **κ on the lag**. Like `transit`, the
/// IG onset is continuous (an essential singularity drives `R_in → 0` at `t_dose + LAG`), so the
/// `K_ROUTE_ONSET` saltation is zero-magnitude and κ rides the continuous `∂R_in/∂lag_route` of the
/// shifted IG density over the per-occasion jet. Direct FD parity for the fourth kernel the gate
/// admits under IOV. Value + gradient + Hessian vs FD of `predict_iov`.
#[test]
fn ode_iov_igd_route_lag_kappa_on_lag_matches_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVMAT(2.0, 0.1, 20.0)
  theta TVCV2(0.5, 0.01, 10.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MAT = TVMAT
  CV2 = TVCV2
  LAG = TVLAG * exp(KAPPA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-11
  ode_abstol = 1e-13
"#;
    let model = parse_model_string(src).expect("parse igd route-lag IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(model.has_route_absorption_lag(), "per-route lag present");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: igd route lag must be analytic under IOV"
    );
    let subject = iov_subject();
    // stacked = [η_CL, κ_g0, κ_g1]. Onsets t_dose + LAG ≈ 1.5 / 25.5 clear every obs.
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[1.0, 20.0, 2.0, 0.5, 1.5],
        &[0.1, 0.03, -0.04],
    );
}

/// #877: **compartment lagtime × per-route lag** composition under IOV, with η on the
/// compartment lag (`LAGTIME`) and κ on the route lag (`LAG`). The onset shift is
/// `∂/∂(lag_cmt + lag_route)`, so the δlag² jet carries both `η_LAGC` and `κ_LAGR` — cross terms
/// `η_LAGC×κ_LAGR` in the checked Hessian. Guards that the combined onset jet is exact under κ.
#[test]
fn ode_iov_compartment_plus_route_lag_matches_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAGC(0.8, 0.0, 10.0)
  theta TVLAGR(0.7, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_LAGC ~ 0.03
  kappa KAPPA_LAGR ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KA      = TVKA
  LAGTIME = TVLAGC * exp(ETA_LAGC)
  LAGR    = TVLAGR * exp(KAPPA_LAGR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAGR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(src).expect("parse comp-lag × route-lag IOV");
    assert!(model.has_lagtime(), "compartment LAGTIME present");
    assert!(model.has_route_absorption_lag(), "per-route lag present");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: compartment lag × route lag must be analytic under IOV"
    );
    let subject = iov_subject();
    // stacked = [η_CL, η_LAGC, κ_g0, κ_g1]. Combined onset t_dose + LAGC + LAGR ≈ 1.5 / 25.5.
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[1.0, 20.0, 1.2, 0.8, 0.7],
        &[0.1, 0.05, 0.03, -0.04],
    );
}

/// #877: parallel **IR + DR** absorption (two `first_order` arms, only the DR arm route-lagged)
/// under IOV — the `first_order` composition the issue names. Fractions `FR1 + FR2 = 1` split the
/// dose across an immediate and a delayed pathway; only the DR arm carries `lag=LAGR`, with κ on
/// it. Guards that the per-forcing onset summation (the DR onset saltation, the IR one at
/// `K_DOSE`) composes correctly under κ. Value + gradient + Hessian vs FD.
#[test]
fn ode_iov_parallel_ir_dr_route_lag_matches_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA1(1.6, 0.05, 20.0)
  theta TVKA2(0.6, 0.05, 20.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVLAGR(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_LAGR ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA1  = TVKA1
  KA2  = TVKA2
  FR1  = TVFR1
  FR2  = 1 - TVFR1
  LAGR = TVLAGR * exp(KAPPA_LAGR)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2, lag=LAGR) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#;
    // Two absorption phases (ka1=1.6, ka2=0.6) make the prediction's 4th derivative larger, so
    // the 4-point FD **Hessian** hits the ODE noise floor sooner than the single-arm models —
    // tighten the solver to 1e-12/1e-14 so the FD cross-Hessian is clean at the harness's fixed
    // 1e-4 step. (Verified: the analytic term is unchanged across tolerances; only the FD
    // reference converges onto it, i.e. this is FD-vs-ODE-precision, not an analytic error.)
    let model = parse_model_string(src).expect("parse parallel IR+DR route-lag IOV");
    let ode = model.ode_spec.as_ref().expect("ode spec");
    assert_eq!(ode.input_rate.len(), 2, "two first_order arms (IR + DR)");
    assert_eq!(
        ode.input_rate
            .iter()
            .filter(|f| f.lag_slot.is_some())
            .count(),
        1,
        "only the DR arm is route-lagged"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: parallel IR+DR route lag must be analytic under IOV"
    );
    let subject = iov_subject();
    // stacked = [η_CL, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[1.0, 20.0, 1.6, 0.6, 0.6, 1.5],
        &[0.1, 0.03, -0.04],
    );
}

/// #877 routing (kept FD): a `weibull` per-route lag stays on finite differences under IOV, its
/// onset diverging for shape `β < 1` (an integrable spike, no finite rate-on saltation) — exactly
/// as on the non-IOV path and as the compartment-lagtime `weibull` gate. The lifted IOV gate uses
/// the per-kernel `route_lag_analytic()` classifier, so broadening it must NOT admit `weibull`.
#[test]
fn ode_iov_weibull_route_lag_falls_back_to_fd() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVTD(2.0, 0.05, 24.0)
  theta TVBETA(1.5, 0.1, 10.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  TD  = TVTD
  BETA = TVBETA
  LAG = TVLAG * exp(KAPPA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let model = parse_model_string(src).expect("parse weibull route-lag IOV");
    assert!(model.has_route_absorption_lag(), "per-route lag present");
    assert!(
        !crate::sens::ode_provider::ode_iov_supported(&model),
        "#877: weibull route lag must stay FD under IOV (β<1 onset divergence)"
    );
}

/// #877 subject routing: an in-scope route-lag IOV subject resolves to analytic ODE-IOV
/// sensitivities, while the SAME model with a **steady-state** dose into the route-lagged
/// compartment declines to FD (`ss_absorption_out_of_scope`'s `lag_slot` operand — the SS dual
/// seed does not carry the per-route onset). Belt on the per-subject gate, not just the model gate.
#[test]
fn ode_iov_route_lag_subject_gate_admits_transient_declines_ss() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(20.0, 1.0, 200.0)
  theta TVKA(1.2, 0.05, 20.0)
  theta TVLAG(1.5, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_LAG ~ 0.02
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  LAG = TVLAG * exp(KAPPA_LAG)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(src).expect("parse route-lag IOV");
    let theta = [1.0, 20.0, 1.2, 1.5];
    // Transient (non-SS) route-lag subject → analytic.
    let transient = iov_subject();
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model,
            &transient,
            &theta,
            &[0.1, 0.03, -0.04],
        )
        .is_some(),
        "#877: a transient route-lag IOV subject must be analytic"
    );
    // SS dose into the route-lagged compartment → declines to FD (out of scope).
    assert!(
        model.ss_absorption_out_of_scope(&Subject {
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
            ..iov_subject()
        }),
        "SS + route lag must stay FD (lag_slot operand of ss_absorption_out_of_scope)"
    );
}

/// #835 review: `CompiledModel::ss_absorption_out_of_scope` is the single source of truth for
/// the SS-into-absorption FD decline shared by `ode_tvcov_supported`,
/// `ode_iov_subject_supported`, and `iov_fd_reason`. Pin all three out-of-scope operands
/// directly on the helper — a `zero_order` window, a compartment absorption lagtime on the
/// dosed compartment, and a per-route `lag=` on the forcing feeding it (#859) — plus the
/// in-scope smooth-kernel-without-lag case, so a future edit to any single gate site cannot
/// silently diverge from the others, and so the helper keeps mirroring the upstream
/// `check_absorption_dosing` reject (whose `has_ss_lag` covers a per-route lag via
/// `route_lag_cmts`). Without the `lag_slot` operand a `first_order` route lag — which #859
/// makes analytic — would slip the belt-and-suspenders decline (its onset is NOT carried by
/// the SS seed). (At the gate call sites the lagtime/route-lag operands are short-circuited by
/// `f.kind == ZeroOrder` for a zero-order forcing, so this is their only direct coverage.)
#[test]
fn ss_absorption_out_of_scope_covers_all_operands() {
    let ss_into = |cmt: usize| Subject {
        doses: vec![DoseEvent::new(0.0, 100.0, cmt, 0.0, true, 12.0)],
        ..iov_subject()
    };
    // Operand 1 — SS into a `zero_order` window → out of scope (FD).
    let zo = parse_model_string(ZERO_ORDER_IOV_ODE).expect("parse zero_order IOV");
    assert!(zo.ss_absorption_out_of_scope(&ss_into(1)));
    // Operand 2 — SS + an absorption lagtime on the dosed compartment → out of scope (FD).
    let weibull_lag = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVTD(2.0, 0.05, 24.0)
  theta TVBETA(1.5, 0.1, 10.0)
  theta TVLAG(0.3, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL      = TVCL * exp(ETA_CL + KAPPA_CL)
  V       = TVV  * exp(ETA_V)
  TD      = TVTD
  BETA    = TVBETA
  LAGTIME = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let wl = parse_model_string(weibull_lag).expect("parse weibull+lag IOV");
    assert!(wl.has_lagtime_on_cmt(1));
    assert!(wl.ss_absorption_out_of_scope(&ss_into(1)));
    // In scope — the same smooth kernel WITHOUT a lagtime is analytic under #835; the helper
    // must NOT decline it (guards against over-declining the supported path).
    let weibull_nolag = weibull_lag
        .replace("  theta TVLAG(0.3, 0.01, 5.0)\n", "")
        .replace("  LAGTIME = TVLAG\n", "");
    let wn = parse_model_string(&weibull_nolag).expect("parse weibull IOV");
    assert!(!wn.has_lagtime_on_cmt(1));
    assert!(!wn.ss_absorption_out_of_scope(&ss_into(1)));
    // Operand 3 (#859) — SS into a `first_order` forcing carrying a per-route `lag=` → out of
    // scope (FD). The per-route lag lives on the forcing's `lag_slot`, NOT the compartment-lag
    // machinery, so `has_lagtime_on_cmt` is FALSE here — the decline must come from the
    // `lag_slot` operand alone, mirroring the upstream `route_lag_cmts` reject. Without it, a
    // route-lagged SS dose (which #859 makes analytic) would take the analytic SS seed, which
    // does not route the dose through the lagged onset.
    let route_lag = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.05, 20.0)
  theta TVLAG(0.3, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL + KAPPA_CL)
  V   = TVV  * exp(ETA_V)
  KA  = TVKA
  LAG = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = first_order(ka=KA, lag=LAG) - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
"#;
    let rl = parse_model_string(route_lag).expect("parse first_order route-lag IOV");
    assert!(
        !rl.has_lagtime_on_cmt(1),
        "a per-route lag is NOT a compartment lagtime"
    );
    assert!(
        rl.has_route_absorption_lag(),
        "the forcing carries a lag_slot"
    );
    assert!(
        rl.ss_absorption_out_of_scope(&ss_into(1)),
        "SS + per-route lag must decline to FD via the lag_slot operand (#859)"
    );
    // A bolus (ii = 0, not steady state) into the same compartment is never out of scope.
    let bolus = Subject {
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        ..iov_subject()
    };
    assert!(!wl.ss_absorption_out_of_scope(&bolus));
    // No ODE spec (a closed-form/analytical model) → the helper's early return: never out of
    // scope, whatever the dose (there is no built-in input-rate forcing to consume it).
    let analytic = parse_model_string(
            "[parameters]\n  theta TVCL(10.0,1.0,100.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse analytic one_cpt_iv");
    assert!(analytic.ode_spec.is_none());
    assert!(!analytic.ss_absorption_out_of_scope(&ss_into(1)));
}

/// **ODE IOV + compartment-indexed `ALAG{cmt}` / `F{cmt}`** (#486 IOV-scope parity). The
/// old gate declined indexed lag/F under IOV on the (mistaken) premise that the walk uses
/// a single `PK_IDX_LAGTIME`/F slot; in fact `integrate_tvcov_readout` resolves each dose's
/// own compartment slot (`f_bio_slot`/`lag_slot`), and the non-IOV path already serves
/// both. An oral (depot→central) model with `ALAG1` (lag on the depot) and `F1` (depot
/// bioavailability), κ on CL, validated against central FD of `predict_iov`.
#[test]
fn ode_iov_indexed_lag_and_f_match_fd_of_predict_iov() {
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVLAG(0.4, 0.01, 5.0)
  theta TVF1(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL + KAPPA_CL)
  V     = TVV  * exp(ETA_V)
  KA    = TVKA
  ALAG1 = TVLAG
  F1    = TVF1
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) =  KA*depot - (CL/V)*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(src).expect("parse indexed-lag/F IOV");
    assert!(model.has_lagtime());
    assert!(
        model
            .active_dose_attr_map()
            .has_indexed_attr(crate::types::DoseAttr::Lag),
        "model must declare an indexed ALAG1"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "indexed ALAG1/F1 must be admitted under IOV (#486 parity)"
    );
    let mut subject = iov_subject();
    // Dose into the depot (cmt 1) so ALAG1/F1 apply.
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
    ];
    // stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.0, 0.4, 0.7],
        &[0.12, -0.08, 0.06, -0.11],
    );
}

/// Regression (#575 review): a plain `ScalingSpec::None` IOV ODE model under LTBS
/// (`log_transform`, no `obs_scale`) must stay on FD. The #575 gate rewrite replaced
/// the `|| model.log_transform` bail with a `match`, and the `None` arm initially
/// admitted LTBS — re-routing IOV + LTBS onto the analytic IOV walk, whose in-PK-param
/// log can't reproduce the production scale-then-log order. The `None` arm carries an
/// explicit `if !model.log_transform` guard; this pins it.
#[test]
fn ode_iov_ltbs_no_scale_falls_back_to_fd() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    assert!(
        matches!(model.scaling, ScalingSpec::None),
        "base model must have no obs_scale (None scaling)"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "non-LTBS None-scaling IOV ODE is analytic"
    );
    let mut ltbs = model;
    ltbs.log_transform = true;
    assert!(
        !crate::sens::ode_provider::ode_iov_supported(&ltbs),
        "IOV + LTBS (None scaling) must fall back to FD"
    );
}

/// 1-cpt IV ODE IOV with an η-dependent `ExpressionScale` `obs_scale = V` divisor
/// (κ on CL, `ObsCmt` readout on the amount). The post-walk per-occasion-group
/// quotient (#575) must reproduce central FD of `predict_iov`, which applies the
/// same divisor per occasion (κ-aware). The ExpressionScale counterpart of
/// `ode_iov_provider_matches_fd_of_predict_iov`.
const WARFARIN_IOV_ODE_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_iov_expr_scale_supported_and_gated() {
    let model = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse expr-scale IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + ExpressionScale obs_scale must be on the analytic path (#575)"
    );
    // + LTBS still routes to FD (the in-walk log transform is not composed with the
    // post-walk quotient on the IOV path).
    let mut ltbs = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse");
    ltbs.log_transform = true;
    assert!(
        !crate::sens::ode_provider::ode_iov_supported(&ltbs),
        "ExpressionScale + LTBS under IOV must fall back to FD"
    );
}

/// ODE M3 BLOQ + IOV + `iiv_on_ruv` scope (#486): the ODE-path counterpart of
/// [`iov_analytical_supported_admits_m3_but_not_the_ruv_triple`]. After the gate flips,
/// `ode_iov_supported` admits M3, `iiv_on_ruv`, and the full **triple** M3 + IOV +
/// `iiv_on_ruv` — all provider-agnostic over the stacked `[η_bsv, κ]` layout (the ODE
/// walk emits a zero `∂f/∂η_ruv` column; the shared assembly applies the variance
/// scaling and the residual-eta column). The **non-IOV** ODE M3 + `iiv_on_ruv` combo is
/// analytic as well (#623), so every combination is served. LTBS still declines.
#[test]
fn ode_iov_supported_admits_m3_and_the_ruv_triple() {
    let mut model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some(), "must be an ODE model");
    // Plain ODE IOV: analytic.
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    // ODE IOV + iiv_on_ruv (no M3): analytic as of #486.
    model.residual_error_eta = Some(1);
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + iiv_on_ruv must be on the analytic path (#486)"
    );
    // The full triple M3 + ODE IOV + iiv_on_ruv: analytic as of #486.
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + M3 + iiv_on_ruv (the triple) must be analytic (#486)"
    );
    assert!(
        iov_sens_supported(&model),
        "iov_sens_supported follows ode_iov_supported"
    );
    // LTBS still declines (the in-walk transform is not composed with the post-walk
    // quotient on the IOV path).
    let mut ltbs = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    ltbs.residual_error_eta = Some(1);
    ltbs.bloq_method = crate::types::BloqMethod::M3;
    ltbs.log_transform = true;
    assert!(
        !crate::sens::ode_provider::ode_iov_supported(&ltbs),
        "ODE IOV + M3 + iiv_on_ruv + LTBS stays FD"
    );
}

#[test]
fn ode_iov_expr_scale_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse expr-scale IOV");
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1] (n_eta = 2, n_kappa = 1, K = 2).
    check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
}

/// #486: ODE IOV + `TIME` switch + η-dependent `ExpressionScale` `obs_scale = V`.
/// The per-event stacked walk (TIME) composes with the per-occasion post-walk scale
/// quotient (built at `t = 0`, matching production's `apply_scaling`); value/∂/∂²
/// over `[η_bsv, κ_g0, κ_g1]` + θ must match FD of `predict_iov`.
#[test]
fn ode_iov_time_expression_scale_matches_fd_of_predict_iov() {
    const M: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 20.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(M).expect("parse ODE IOV TIME expr-scale");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    // obs [1,6,12 | 25,30,36] straddle TIME=20. stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &iov_subject(),
        &[0.2, 0.1, 10.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn ode_iov_expr_scale_inner_eta_grad_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse expr-scale IOV"),
        &iov_subject(),
        &[0.2, 10.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// The `obs_scale = V` divisor form is a numerical twin of the Form-C readout
/// `y = central / V` (already analytic + FD-validated): both compute `central/V`
/// and its exact stacked-(η,κ,θ) sensitivities, the divisor post-walk and the
/// readout in-walk. Per-observation value and `∂f/∂stacked` must agree (#575).
#[test]
fn ode_iov_expr_scale_equals_formc_readout() {
    let divisor = parse_model_string(WARFARIN_IOV_ODE_EXPRSCALE).expect("parse divisor");
    let formc = parse_model_string(WARFARIN_IOV_ODE).expect("parse Form-C");
    let subject = iov_subject();
    let theta = [0.2, 10.0];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    let a = subject_sensitivities_iov(&divisor, &subject, &theta, &stacked).expect("divisor");
    let b = subject_sensitivities_iov(&formc, &subject, &theta, &stacked).expect("formc");
    assert_eq!(a.obs.len(), b.obs.len());
    for (oa, ob) in a.obs.iter().zip(b.obs.iter()) {
        approx::assert_relative_eq!(oa.f, ob.f, max_relative = 1e-8, epsilon = 1e-10);
        for (x, y) in oa.df_deta.iter().zip(ob.df_deta.iter()) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-7, epsilon = 1e-9);
        }
    }
}

/// **ODE IOV + steady-state bolus.** Each occasion's SS dose equilibrates with that
/// occasion's κ-seeded params (dual SS-equilibration), then the per-occasion walk
/// continues. Validated vs FD of `predict_iov` (#439 IOV × SS).
#[test]
fn ode_iov_ss_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, true, 12.0),
    ];
    assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
    check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
}

/// **ODE IOV + rate-defined infusion under bioavailability `F ≠ 1`** (#419 × IOV). The
/// bioavailable window length `F·amt/rate` is a moving rate-off boundary per occasion;
/// the event-driven walk carries it with the rate held. Validated vs FD of `predict_iov`.
#[test]
fn ode_iov_rate_defined_infusion_under_f_matches_fd_of_predict_iov() {
    // WARFARIN IOV ODE with a bioavailability parameter `F`.
    const WARFARIN_IOV_F_ODE: &str = r#"
[parameters]
  theta TVCL(0.13, 0.01, 1.0)
  theta TVV(8.0, 1.0, 50.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  iov_column OCC
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV * exp(ETA_V)
  F  = TVF
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(WARFARIN_IOV_F_ODE).expect("parse IOV+F ODE");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 50.0, false, 0.0),
    ];
    assert!(subject.has_rate_defined_infusion());
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.13, 8.0, 0.7],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// **IOV × estimated lagtime.** 1-cpt IV IOV `[odes]` model (κ on CL) with a bare
/// `LAGTIME`. The dose arrives per occasion at `t_dose + lag`; the lag sensitivity is
/// the event-time saltation injected at each dose and propagated through the
/// occasion-switching event-driven walk (`integrate_tvcov_g`, shared with the TV-cov
/// path). Validates the full stacked-η + θ (incl. the `TVLAG` column) gradient and
/// Hessian against FD of `predict_iov`, which handles IOV + lagtime in production
/// (#439 lagtime × IOV).
const WARFARIN_IOV_LAG_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAG(0.5, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  LAGTIME = TVLAG
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_iov_lagtime_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_LAG_ODE).expect("parse ODE IOV+lag");
    assert_eq!(model.n_kappa, 1);
    assert!(model.has_lagtime());
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + bare lagtime must be supported"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVLAG].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// **IOV × steady-state × estimated lagtime is now analytic (#486, PR3 sub-case (a)).**
/// Each occasion's SS dose gets its own `K_SS_SEED` pre-arrival seed (phase `II − lag`,
/// `ss_state_at_phase_g` seeded with that occasion's κ) — validated against the dense
/// (non-event-driven) predictor for this exact combination in
/// `ode_provider_ss_lagtime_matches_production`/`..._infusion_...` (including an
/// observation strictly inside the pre-arrival window). `predict_iov`, this test's
/// oracle, has no such seed of its own (a pre-existing, orthogonal gap — it would read
/// zero for a pre-arrival observation regardless of gate/gradient correctness), so this
/// test uses the default `iov_subject()` times (post-arrival only) to isolate the SS
/// dose's own event-time saltation under IOV.
#[test]
fn ode_iov_ss_lagtime_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_LAG_ODE).expect("parse ODE IOV+lag");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + bare lagtime must be supported"
    );
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, true, 12.0),
    ];
    assert!(subject.doses[0].ss && subject.doses[0].ii > 0.0);
    let theta = [0.2, 10.0, 0.5];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model, &subject, &theta, &stacked,
        )
        .is_some(),
        "SS + lagtime IOV subject must be analytic now (#486)"
    );
    // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVLAG].
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
}

/// **IOV × rate-defined SS infusion under bioavailability `F ≠ 1` is now analytic**
/// (#486, PR3 sub-case (b)). `equilibrate_ss_state_g` reads the per-occasion `inf_eff`
/// jet (window `F·duration`, rate held) instead of the fixed raw duration.
#[test]
fn ode_iov_ss_rate_defined_infusion_under_f_matches_fd_of_predict_iov() {
    const WARFARIN_IOV_SS_F_ODE: &str = r#"
[parameters]
  theta TVCL(0.13, 0.01, 1.0)
  theta TVV(8.0, 1.0, 50.0)
  theta TVF(0.7, 0.05, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  iov_column OCC
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV * exp(ETA_V)
  F  = TVF
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let model = parse_model_string(WARFARIN_IOV_SS_F_ODE).expect("parse IOV+SS+F ODE");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let mut subject = iov_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, true, 12.0),
        DoseEvent::new(24.0, 100.0, 1, 50.0, true, 12.0),
    ];
    assert!(subject.doses[0].ss && subject.has_rate_defined_infusion());
    let theta = [0.13, 8.0, 0.7];
    let stacked = [0.12, -0.08, 0.05, -0.10];
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model, &subject, &theta, &stacked,
        )
        .is_some(),
        "rate-defined SS infusion under F IOV subject must be analytic now (#486)"
    );
    check_iov_provider_vs_fd(&model, &subject, &theta, &stacked);
}

/// **IOV × lagtime × infusion × reset** — the combined path through `integrate_tvcov_g`
/// (both rate-on/off saltations, the `reset_floor` guard, and per-occasion κ seeding) is
/// otherwise covered only piecewise. Full stacked-η + θ gradient and Hessian vs FD of
/// `predict_iov` (#472 review round 2 #5).
#[test]
fn ode_iov_lagtime_infusion_reset_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_LAG_ODE).expect("parse ODE IOV+lag");
    assert!(model.has_lagtime());
    let mut subject = iov_reset_subject(); // 2 occasions, EVID=4 reset at t=24
                                           // Per-occasion infusions (rate>0): occ-1 window starts at 0, occ-2 re-dose at the
                                           // reset; the lagtime shifts both windows.
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 50.0, false, 0.0),
    ];
    assert!(subject.has_resets() && subject.doses[0].is_infusion());
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// 1-cpt IV IOV `[odes]` model with a WT covariate on CL (`(WT/70)^θ_WT`) under
/// **time-varying covariates**: each event's PK params are seeded at its own
/// (occasion, WT-snapshot), so the individual CL switches both by κ and by WT.
/// Validates the per-event IOV+TV-cov seeding vs FD of `predict_iov` (#439 ODE IOV).
const WARFARIN_IOV_TVCOV_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// Same as `WARFARIN_IOV_TVCOV_ODE`, but with the readout expressed as a
/// post-walk divisor instead of Form C. The subject has time-varying covariates
/// in the event walk, and the `obs_scale = CL` quotient references **both** the
/// TV covariate `WT` and `KAPPA_CL` — so the scale jet would *differ* if it read
/// per-event covariates (it must use the subject-static snapshot, like production
/// `predict_iov`'s `apply_scaling`) and *differs per occasion group* via κ. A
/// covariate- and κ-free scale (`obs_scale = V`) could not distinguish those (#590).
const WARFARIN_IOV_TVCOV_ODE_EXPRSCALE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = CL
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_iov_tvcov_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse ODE IOV+TV-cov");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "1-cpt IV ODE IOV+TV-cov must be ODE-IOV-provider supported"
    );
    let subject = iov_tvcov_subject(false);
    assert!(
        subject.has_tv_covariates(),
        "subject must carry TV covariates"
    );
    // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, THETA_WT].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.75],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn ode_iov_tvcov_expr_scale_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV_ODE_EXPRSCALE)
        .expect("parse ODE IOV+TV-cov+expr-scale");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    assert!(
        matches!(model.scaling, ScalingSpec::ExpressionScale { .. }),
        "fixture must use an expression obs_scale"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + TV covariates + ExpressionScale must stay analytic (#590)"
    );
    let subject = iov_tvcov_subject(false);
    assert!(
        subject.has_tv_covariates(),
        "subject must carry TV covariates"
    );
    // θ = [TVCL, TVV, THETA_WT]; stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.75],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn ode_iov_tvcov_expr_scale_inner_eta_grad_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_TVCOV_ODE_EXPRSCALE)
            .expect("parse ODE IOV+TV-cov+expr-scale"),
        &iov_tvcov_subject(false),
        &[0.2, 10.0, 0.75],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn ode_iov_tvcov_pkonly_breakpoint_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse ODE IOV+TV-cov");
    let subject = iov_tvcov_subject(true);
    assert!(
        !subject.pk_only_times.is_empty(),
        "fixture must carry an EVID=2 covariate breakpoint"
    );
    assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "model-level ODE IOV gate must admit the fixture"
    );
    assert!(
        subject_sensitivities_iov(
            &model,
            &subject,
            &[0.2, 10.0, 0.75],
            &[0.12, -0.08, 0.05, -0.10],
        )
        .is_some(),
        "EVID=2 breakpoint must stay on the analytic ODE IOV path (#590)"
    );
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.75],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn ode_iov_tvcov_pkonly_inner_eta_grad_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse ODE IOV+TV-cov"),
        &iov_tvcov_subject(true),
        &[0.2, 10.0, 0.75],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

// 1-cpt IV ODE IOV + TV-cov with a **κ-coupled `init(...)` baseline**: `init(central) =
// BASE` where `BASE = TVBASE·exp(ETA_V + KAPPA_CL)` carries η_v *and* the occasion κ, so the
// event-driven IOV walk's `tvcov_init_state` seed (built from the first-record snapshot's
// stacked `(θ, η_bsv, κ)` duals) must fold `∂init/∂κ` onto the correct occasion axis (#486).
const WARFARIN_IOV_ODE_INIT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVBASE(40.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL   = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V    = TVV  * exp(ETA_V)
  BASE = TVBASE * exp(ETA_V + KAPPA_CL)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE
  d/dt(central) = -(CL/V) * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **`init(...)` × IOV on the ODE walk** (#486, branch G-ODE). A κ-coupled `init(central) =
/// BASE` baseline (`BASE = TVBASE·exp(ETA_V + KAPPA_CL)`) on a 1-cpt IV ODE IOV + TV-cov
/// subject. The IOV outer/inner run through the same `integrate_tvcov_readout` the non-IOV
/// walk uses, so `tvcov_init_state` already seeds `init` from the first-record snapshot's
/// stacked `(θ, η_bsv, κ)` duals — the Taylor deltas carry `∂init/∂κ` onto the correct
/// occasion axis with no IOV-specific code. Validated against FD of `predict_iov` over the
/// full stacked layout, plus inner/outer η-parity.
#[test]
fn ode_iov_init_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE_INIT).expect("parse ODE IOV+init");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    assert!(
        model.ode_spec.as_ref().unwrap().init_fn.is_some(),
        "fixture must declare an init(...)"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + init must be analytic (#486)"
    );
    let subject = iov_tvcov_subject(false);
    // θ = [TVCL, TVV, THETA_WT, TVBASE]; stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.75, 40.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

#[test]
fn ode_iov_init_inner_eta_grad_matches_outer() {
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_ODE_INIT).expect("parse ODE IOV+init"),
        &iov_tvcov_subject(false),
        &[0.2, 10.0, 0.75, 40.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

// ODE IOV with a `first_order` input-rate forcing AND a κ-coupled `init(central) = BASE`
// baseline — the composition the #660 (`FirstOrder` under IOV) and #486 (`init` under IOV)
// gate relaxations jointly admit. Both ride the same `integrate_tvcov_readout` walk: the
// dose feeds `R_in = KA·(remaining)` and the init baseline is seeded from the first-record
// stacked snapshot, so the two are orthogonal. This pins that they compose correctly.
const WARFARIN_IOV_ODE_INIT_FO: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta TVBASE(40.0, 1.0, 500.0)
  theta TVKA(1.2, 0.01, 20.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL   = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V    = TVV  * exp(ETA_V)
  KA   = TVKA
  BASE = TVBASE * exp(ETA_V + KAPPA_CL)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central) = BASE
  d/dt(central)  = first_order(ka=KA) - (CL/V) * central
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// **`init(...)` × `first_order` input-rate forcing × IOV on the ODE walk** (#486, the
/// combination the #660 and #486 IOV relaxations jointly admit). Validated against FD of
/// `predict_iov` over the stacked layout, plus inner/outer parity.
#[test]
fn ode_iov_init_first_order_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE_INIT_FO).expect("parse ODE IOV+init+FO");
    assert!(!model.ode_spec.as_ref().unwrap().input_rate.is_empty());
    assert!(model.ode_spec.as_ref().unwrap().init_fn.is_some());
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "ODE IOV + init + first_order must be analytic"
    );
    let subject = iov_tvcov_subject(false);
    // θ = [TVCL, TVV, THETA_WT, TVBASE, TVKA]; stacked = [η_cl, η_v, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.75, 40.0, 1.2],
        &[0.12, -0.08, 0.05, -0.10],
    );
    check_iov_inner_matches_outer(
        &model,
        &subject,
        &[0.2, 10.0, 0.75, 40.0, 1.2],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// **Static-covariate** ODE IOV subject with an EVID=2 pk-only breakpoint. The TV-cov
/// pk-only tests above always take `seed_iov_events`' per-event (TV) branch; a subject
/// with no dose/obs TV covariates **and an empty `pk_only_covariates`** instead takes the
/// static-cov else-branch — the pk-only event is seeded once at the subject-static snapshot
/// and shared (`vec![seeded; len]`). This is reachable in production (e.g. a pk-only
/// breakpoint whose covariate was pruned as irrelevant while its time remained), so it must
/// stay analytic and match FD of `predict_iov` (#598 review — covers the else-branch).
#[test]
fn ode_iov_static_cov_pkonly_breakpoint_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    let mut subject = iov_subject();
    // EVID=2 breakpoint at t=18 (occasion 2), no covariate snapshot → static-cov path.
    subject.pk_only_times = vec![18.0];
    assert!(
        subject.pk_only_covariates.is_empty() && !subject.has_tv_covariates(),
        "fixture must hit the static-cov pk-only branch (no TV cov, empty pk_only_covariates)"
    );
    assert!(
        crate::sens::ode_provider::ode_subject_sensitivities_iov(
            &model,
            &subject,
            &[0.2, 10.0],
            &[0.12, -0.08, 0.05, -0.10],
        )
        .is_some(),
        "static-cov EVID=2 breakpoint must stay on the analytic ODE IOV path"
    );
    check_iov_provider_vs_fd(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
}

/// Inner η-gradient parity for the same static-cov pk-only subject — exercises the
/// `Dual1` seeder's static-cov else-branch (#598 review).
#[test]
fn ode_iov_static_cov_pkonly_inner_eta_grad_matches_outer() {
    let model = parse_model_string(WARFARIN_IOV_ODE).expect("parse ODE IOV");
    let mut subject = iov_subject();
    subject.pk_only_times = vec![18.0];
    check_iov_inner_matches_outer(&model, &subject, &[0.2, 10.0], &[0.12, -0.08, 0.05, -0.10]);
}

/// 2-cpt IV IOV `[odes]` model (κ on CL) — higher state/axis coverage for the ODE
/// IOV walk: stacked dual width M = n_θ(4) + n_η(2) + K(2)·n_κ(1) = 8 (#439 ODE IOV).
const WARFARIN_IOV_2CPT_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V) * central - (Q/V) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_iov_2cpt_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_2CPT_ODE).expect("parse 2cpt ODE IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVQ, TVV2].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5, 20.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// 1-cpt **oral** IOV `[odes]` model (depot → central first-order absorption, κ on
/// CL): bolus into the depot (cmt 1), readout `central/V`. Exercises the multi-state
/// absorption ODE under per-occasion κ seeding (#439 ODE IOV).
const WARFARIN_IOV_ORAL_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  ode(states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_iov_oral_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_ORAL_ODE).expect("parse oral ODE IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let subject = iov_subject();
    // stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1]; θ = [TVCL, TVV, TVKA].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// 3-cpt IV IOV `[odes]` model (κ on CL) — highest state/axis coverage for the ODE
/// IOV walk: 3 ODE states, dual width M = n_θ(6) + n_η(2) + K(2)·n_κ(1) = 10
/// (#439 ODE IOV).
const WARFARIN_IOV_3CPT_ODE: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ2(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVQ3(0.3, 0.001, 50.0)
  theta TVV3(50.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
[structural_model]
  ode(states=[central, periph1, periph2])
[odes]
  d/dt(central) = -(CL/V)*central - (Q/V)*central + (Q/V2)*periph1 - (Q3/V)*central + (Q3/V3)*periph2
  d/dt(periph1) =  (Q/V)*central - (Q/V2)*periph1
  d/dt(periph2) =  (Q3/V)*central - (Q3/V3)*periph2
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

#[test]
fn ode_iov_3cpt_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_3CPT_ODE).expect("parse 3cpt ODE IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVQ2, TVV2, TVQ3, TVV3].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// 1-cpt IV IOV `[odes]` model whose Form-C readout references a θ (`TVBASE`) **and**
/// an η (`ETA_CL`) directly (#486). The parser desugars each bare θ/η into a synthetic
/// individual parameter (`__ferx_ro_*`); under IOV those synthetics ride the stacked
/// `(θ, η_bsv, κ)` chain like any other individual parameter. `ETA_CL` is the BSV η that
/// also carries the per-occasion κ, so the readout's `∂y/∂η_cl` couples the explicit
/// `(1 + ETA_CL)` term with the κ-driven state — exercising the synthetic-param seeding
/// on the IOV walk (the path the #631 review flagged as previously FD-only).
const WARFARIN_IOV_ODE_DIRECT_THETA_ETA: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBASE(0.5, 0.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V * (1.0 + ETA_CL) + TVBASE
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// #486 + #439: a direct-θ/η Form-C readout under **IOV** must take the analytic ODE
/// IOV path and match FD of `predict_iov` (value, η-Hessian, θ-grad, η×θ cross). Before
/// #631 such a readout was not `dual_evaluable`, so `ode_iov_supported` returned false
/// and the subject fell back to finite differences; this confirms the synthetic
/// readout parameters seed the stacked `(θ, η_bsv, κ)` chain correctly on the IOV walk.
#[test]
fn ode_iov_form_c_direct_theta_eta_matches_production() {
    let model =
        parse_model_string(WARFARIN_IOV_ODE_DIRECT_THETA_ETA).expect("parse ODE IOV direct θ/η");
    assert_eq!(model.n_kappa, 1);
    assert!(model.ode_spec.is_some());
    // 2 real (CL, V) + 2 synthetic (__ferx_ro_th2, __ferx_ro_eta0) individual params,
    // and no new omega for the direct η reference.
    assert_eq!(
        model.n_eta, 2,
        "direct ETA_CL reuses the existing BSV η (no new omega)"
    );
    assert_eq!(
        model.pk_indices.len(),
        4,
        "CL, V + 2 synthetic readout params"
    );
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&model),
        "direct-θ/η Form-C readout under IOV should be analytic (#486)"
    );
    let subject = iov_subject();
    // stacked = [η_cl, η_v, κ_g0, κ_g1]; θ = [TVCL, TVV, TVBASE].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// The light **inner** IOV walk (`Dual1`, via the `subject_eta_grad_iov` dispatch —
/// ODE or closed-form) must produce the same per-observation value and
/// `∂f/∂(stacked-η)` as the **outer** walk (`Dual2`, `subject_sensitivities_iov`),
/// whose `df_deta` is already validated against FD of `predict_iov`. A
/// bit-for-bit-close cross-check that the first-order seeding/readout is consistent
/// across the two dual orders, for both providers (#439 IOV inner).
fn check_iov_inner_matches_outer(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked: &[f64],
) {
    let outer =
        subject_sensitivities_iov(model, subject, theta, stacked).expect("outer IOV supported");
    let inner = subject_eta_grad_iov(model, subject, theta, stacked).expect("inner IOV supported");
    assert_eq!(outer.obs.len(), inner.len());
    for (o, i) in outer.obs.iter().zip(inner.iter()) {
        approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-12, epsilon = 1e-12);
        assert_eq!(o.df_deta.len(), i.df_deta.len());
        for (a, b) in o.df_deta.iter().zip(i.df_deta.iter()) {
            approx::assert_relative_eq!(a, b, max_relative = 1e-9, epsilon = 1e-11);
        }
    }
}

#[test]
fn ode_iov_inner_eta_grad_matches_outer() {
    // 1-cpt IV, oral, and TV-cov variants — the inner Dual1 walk must track the
    // FD-validated outer Dual2 walk's first-order block on each.
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_ODE).expect("parse"),
        &iov_subject(),
        &[0.2, 10.0],
        &[0.12, -0.08, 0.05, -0.10],
    );
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_ORAL_ODE).expect("parse"),
        &iov_subject(),
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
    check_iov_inner_matches_outer(
        &parse_model_string(WARFARIN_IOV_TVCOV_ODE).expect("parse"),
        &iov_tvcov_subject(false),
        &[0.2, 10.0, 0.75],
        &[0.12, -0.08, 0.05, -0.10],
    );
}

/// Two-occasion IOV subject with a washout: an EVID=4 reset at t=24 zeros the
/// state and opens occasion 2, so there is NO carryover across the boundary
/// (the complement of `iov_subject`, which carries occasion-1 amounts forward).
/// Exercises the walk's reset handling under per-occasion κ seeding.
fn iov_reset_subject() -> Subject {
    let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
    let occasions = vec![1u32, 1, 1, 2, 2, 2];
    let n = obs_times.len();
    Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: vec![24.0],
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// 1-cpt oral IOV **with an EVID=4 washout reset** at the occasion boundary:
/// the provider's value/grad/Hessian over `[η_bsv, κ_g0, κ_g1]` + θ must still
/// match FD of `predict_iov` (which routes the reset through the same
/// event-driven walk). Confirms ungating resets in `subject_sensitivities_iov`
/// keeps the (η, κ, θ) chain exact across the reset.
#[test]
fn iov_provider_with_reset_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV).expect("parse warfarin IOV");
    let subject = iov_reset_subject();
    assert!(subject.has_resets(), "fixture must carry a reset");
    assert!(
        subject_sensitivities_iov(
            &model,
            &subject,
            &[0.2, 10.0, 1.5],
            &[0.1, 0.0, 0.1, 0.0, 0.0]
        )
        .is_some(),
        "IOV + reset subject must be analytic-supported"
    );
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// 2-cpt oral IOV: same FD check, exercising the generic 2-cpt event-driven
/// sensitivity walk (eigen-decomposition propagators) under occasion carryover.
#[test]
fn iov_provider_2cpt_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_2CPT).expect("parse 2cpt warfarin IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(
        iov_analytical_supported(&model),
        "2-cpt warfarin IOV must be IOV-provider supported"
    );
    let subject = iov_subject();
    // θ = [TVCL, TVV, TVQ, TVV2, TVKA]; stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5, 20.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// #486: a `TIME`-switched CL on a **closed-form IOV** model. With no TV
/// covariates the subject routes to the per-event stacked walk purely by
/// `uses_time_builtin`; the value/∂/∂² over `[η_bsv, κ_g0, κ_g1]` + θ must match
/// central FD of `predict_iov` (which threads the same per-event TIME).
#[test]
fn iov_time_builtin_provider_matches_fd_of_predict_iov() {
    const IOV_ORAL_TIME: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVCL_LATE(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  if (TIME > 20.0) {
    CL = TVCL_LATE * exp(ETA_CL + KAPPA_CL)
  } else {
    CL = TVCL * exp(ETA_CL + KAPPA_CL)
  }
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;
    let model = parse_model_string(IOV_ORAL_TIME).expect("parse IOV oral TIME");
    assert_eq!(model.n_kappa, 1);
    let subject = iov_subject(); // obs [1,6,12 | 25,30,36] straddle TIME=20
    assert!(
        !subject.has_tv_covariates(),
        "no TV cov: routed by uses_time only"
    );
    assert!(
        iov_analytical_supported(&model),
        "closed-form IOV TIME must be provider-supported"
    );
    // θ = [TVCL, TVCL_LATE, TVV, TVKA]; stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 0.1, 10.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

const WARFARIN_IOV_3CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ2(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVQ3(0.3, 0.001, 50.0)
  theta TVV3(50.0, 0.1, 1000.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// 3-cpt oral IOV: same FD check, exercising the generic 3-cpt eigenmode
/// event-driven sensitivity walk under occasion carryover.
#[test]
fn iov_provider_3cpt_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_3CPT).expect("parse 3cpt warfarin IOV");
    assert_eq!(model.n_kappa, 1);
    assert!(
        iov_analytical_supported(&model),
        "3-cpt warfarin IOV must be IOV-provider supported"
    );
    let subject = iov_subject();
    // θ = [TVCL,TVV,TVQ2,TVV2,TVQ3,TVV3,TVKA]; stacked = [η_cl,η_v,η_ka,κ_g0,κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0, 1.5],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

// ── IOV combined with time-varying covariates ────────────────────────
//
// These models carry BOTH a kappa (IOV) and a WT-on-CL covariate that varies
// within the subject. Each event's PK-param duals must be seeded at that
// event's covariate snapshot *and* at the right occasion's κ — the per-event
// `sources` refactor in `subject_sensitivities_iov`. The FD reference is the
// production `predict_iov`, which already seeds per-event covariates, so the
// check validates the merged (η_bsv, κ, θ, WT) chain end to end.

const WARFARIN_IOV_TVCOV: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

const WARFARIN_IOV_TVCOV_2CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

const WARFARIN_IOV_TVCOV_3CPT: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVQ2(0.5, 0.001, 50.0)
  theta TVV2(20.0, 0.1, 500.0)
  theta TVQ3(0.3, 0.001, 50.0)
  theta TVV3(50.0, 0.1, 1000.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  Q  = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk three_cpt_oral(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3, ka=KA)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;

/// Two-occasion IOV subject with a WT covariate that varies across records
/// (occasion-1 doses/obs at a lighter weight, occasion-2 heavier), so the
/// individual `CL` switches both by κ (occasion) and by WT (covariate). When
/// `pk_only` is set, an EVID=2 covariate breakpoint (WT jump at t=18, no
/// occasion) sits between the occasion-2 observations — exercising the κ=0
/// `pk_only` source on the IOV+TV-cov path.
fn iov_tvcov_subject(pk_only: bool) -> Subject {
    let obs_times = vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0];
    let occasions = vec![1u32, 1, 1, 2, 2, 2];
    let obs_wts = [70.0, 72.0, 78.0, 88.0, 90.0, 95.0];
    let n = obs_times.len();
    let (pk_only_times, pk_only_covariates) = if pk_only {
        (vec![18.0], vec![wt_map(85.0)])
    } else {
        (Vec::new(), Vec::new())
    };
    Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        covariates: wt_map(70.0),
        dose_covariates: vec![wt_map(70.0), wt_map(85.0)],
        obs_covariates: obs_wts.iter().map(|&w| wt_map(w)).collect(),
        pk_only_times,
        pk_only_covariates,
        reset_times: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// 1-cpt oral IOV **+ WT-on-CL time-varying covariate**: the provider's
/// value/grad/Hessian over `[η_bsv, κ_g0, κ_g1]` + θ (now including `THETA_WT`)
/// must match FD of `predict_iov`, which seeds each event at its own covariate
/// snapshot and occasion κ. Validates the per-event `sources` merge.
#[test]
fn iov_tvcov_provider_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV).expect("parse warfarin IOV+TVcov");
    assert_eq!(model.n_kappa, 1);
    assert!(iov_analytical_supported(&model));
    let subject = iov_tvcov_subject(false);
    assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
    // θ = [TVCL, TVV, TVKA, THETA_WT]; stacked = [η_cl, η_v, η_ka, κ_g0, κ_g1].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5, 0.75],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// 1-cpt oral IOV + TV-cov **with an EVID=2 covariate breakpoint**: a WT jump
/// carried by a `pk_only` record (no occasion → κ fixed at 0) between the
/// occasion-2 observations. Exercises the new `pk_only` source on the IOV path,
/// which the previous code bailed out of.
#[test]
fn iov_tvcov_pkonly_breakpoint_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV).expect("parse warfarin IOV+TVcov");
    let subject = iov_tvcov_subject(true);
    assert!(
        !subject.pk_only_times.is_empty(),
        "fixture must carry EVID=2"
    );
    assert!(subject.has_tv_covariates(), "fixture must carry TV cov");
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 1.5, 0.75],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// 2-cpt oral IOV + WT-on-CL TV covariate: same FD check through the generic
/// 2-cpt event-driven sensitivity walk under per-event covariate seeding.
#[test]
fn iov_tvcov_2cpt_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV_2CPT).expect("parse 2cpt warfarin IOV+TVcov");
    assert_eq!(model.n_kappa, 1);
    let subject = iov_tvcov_subject(false);
    // θ = [TVCL, TVV, TVQ, TVV2, TVKA, THETA_WT].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5, 20.0, 1.5, 0.75],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}

/// 3-cpt oral IOV + WT-on-CL TV covariate: same FD check through the generic
/// 3-cpt eigenmode walk (widest dual on the IOV+TV-cov path).
#[test]
fn iov_tvcov_3cpt_matches_fd_of_predict_iov() {
    let model = parse_model_string(WARFARIN_IOV_TVCOV_3CPT).expect("parse 3cpt warfarin IOV+TVcov");
    assert_eq!(model.n_kappa, 1);
    let subject = iov_tvcov_subject(false);
    // θ = [TVCL, TVV, TVQ2, TVV2, TVQ3, TVV3, TVKA, THETA_WT].
    check_iov_provider_vs_fd(
        &model,
        &subject,
        &[0.2, 10.0, 0.5, 20.0, 0.3, 50.0, 1.5, 0.75],
        &[0.12, -0.08, 0.20, 0.05, -0.10],
    );
}
