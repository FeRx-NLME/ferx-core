//! Integration test for the `[scaling]` block (Phase 1).
//!
//! Tier 2: exercises the full parser → compiled model → prediction pipeline
//! via the public API without running a fit to convergence. Verifies that:
//!
//! - Form A (`obs_scale = 1000`) yields predictions that are 1/1000 of the
//!   unscaled baseline.
//! - Form C (ODE `y = central / V`) on an amount-based ODE produces the same
//!   per-observation values as a concentration-baked ODE with `obs_cmt=central`.
//!
//! These tests run on every PR (`cargo check --tests`) and exit fast — no
//! `fit()` calls — so they don't need to be gated behind `slow-tests`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population, Subject};
use ferx_core::{predict, ScalingSpec};
use std::collections::HashMap;

fn one_subject_pop() -> Population {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
    let n_obs = obs_times.len();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
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
        }],
    }
}

const ANALYTICAL_BASE: &str = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
";

#[test]
fn form_a_scalar_scale_divides_predictions_by_exactly_k() {
    let baseline = parse_model_string(ANALYTICAL_BASE).expect("baseline parses");
    let mut scaled_src = String::from(ANALYTICAL_BASE);
    scaled_src.push_str("\n[scaling]\n  obs_scale = 1000\n");
    let scaled = parse_model_string(&scaled_src).expect("scaled parses");

    assert!(matches!(baseline.scaling, ScalingSpec::None));
    assert!(matches!(scaled.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-12));

    let pop = one_subject_pop();
    let pop_preds = predict(&baseline, &pop, &baseline.default_params);
    let scaled_preds = predict(&scaled, &pop, &scaled.default_params);

    assert_eq!(pop_preds.len(), scaled_preds.len());
    assert!(!pop_preds.is_empty(), "must have predictions to compare");

    for (a, b) in pop_preds.iter().zip(scaled_preds.iter()) {
        assert!(a.pred > 0.0, "baseline pred must be positive");
        let ratio = a.pred / b.pred;
        assert!(
            (ratio - 1000.0).abs() < 1e-9,
            "scalar scaling must divide by exactly 1000: baseline={} scaled={} ratio={}",
            a.pred,
            b.pred,
            ratio
        );
    }
}

const ODE_CONCENTRATION_FORM: &str = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
";

const ODE_AMOUNT_FORM_C: &str = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
";

#[test]
fn form_b_expression_uses_individual_parameter() {
    // Phase 1.5: Form B can reference individual parameters. We construct
    // two equivalent analytical models on the same data:
    //   (a) baseline with no scaling
    //   (b) `obs_scale = V` — divides every prediction by V
    // and check (a).pred / V = (b).pred. With V = TVV = 50 and the test
    // population having a single subject (η = 0), V is deterministic.
    let baseline = parse_model_string(ANALYTICAL_BASE).expect("baseline parses");
    let mut scaled_src = String::from(ANALYTICAL_BASE);
    scaled_src.push_str("\n[scaling]\n  obs_scale = V\n");
    let scaled = parse_model_string(&scaled_src).expect("Form B with indiv-param parses");

    let pop = one_subject_pop();
    let base_preds = predict(&baseline, &pop, &baseline.default_params);
    let scaled_preds = predict(&scaled, &pop, &scaled.default_params);

    assert_eq!(base_preds.len(), scaled_preds.len());
    assert!(!base_preds.is_empty());

    // V = TVV = 50 (no eta on V in the template).
    let v = 50.0;
    for (b, s) in base_preds.iter().zip(scaled_preds.iter()) {
        assert!(b.pred > 0.0);
        let expected = b.pred / v;
        let rel = (s.pred - expected).abs() / expected.abs().max(1e-12);
        assert!(
            rel < 1e-9,
            "Form B with indiv-param V: baseline={} scaled={} expected={}",
            b.pred,
            s.pred,
            expected
        );
    }
}

#[test]
fn form_c_amount_ode_matches_concentration_ode() {
    let conc = parse_model_string(ODE_CONCENTRATION_FORM).expect("concentration form parses");
    let amt = parse_model_string(ODE_AMOUNT_FORM_C).expect("Form C parses");

    // ODE spec sanity: concentration form keeps ObsCmt readout; Form C
    // swaps to a Single output_fn readout.
    let ode_conc = conc.ode_spec.as_ref().expect("conc ODE present");
    let ode_amt = amt.ode_spec.as_ref().expect("amt ODE present");
    assert!(matches!(
        ode_conc.readout,
        ferx_core::ode::OdeReadout::ObsCmt(_)
    ));
    assert!(matches!(
        ode_amt.readout,
        ferx_core::ode::OdeReadout::Single(_)
    ));

    let pop = one_subject_pop();
    let conc_preds = predict(&conc, &pop, &conc.default_params);
    let amt_preds = predict(&amt, &pop, &amt.default_params);

    assert_eq!(conc_preds.len(), amt_preds.len());
    assert!(!conc_preds.is_empty(), "must have predictions to compare");

    // Same physical system written two ways → identical predictions to
    // ODE-solver tolerance. Use a tight relative tolerance; the absolute
    // floor handles observations very near zero.
    for (a, b) in conc_preds.iter().zip(amt_preds.iter()) {
        let denom = a.pred.abs().max(1e-12);
        let rel = (a.pred - b.pred).abs() / denom;
        assert!(
            rel < 1e-4,
            "Form C must match concentration-baked ODE: t={} conc={} form_c={} rel={}",
            a.time,
            a.pred,
            b.pred,
            rel
        );
    }
}

/// Two-subject population with observations split across two CMTs — used
/// to exercise the multi-analyte (PerCmt) dispatch end-to-end through
/// `predict()`. Each subject has 3 obs on CMT=1 and 2 obs on CMT=2.
fn two_cmt_pop() -> Population {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
    let obs_cmts = vec![1, 1, 1, 2, 2];
    let n_obs = obs_times.len();
    let mk_subject = |id: &str| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: obs_times.clone(),
        observations: vec![0.0; n_obs],
        obs_cmts: obs_cmts.clone(),
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; n_obs],
        occasions: Vec::new(),
        dose_occasions: Vec::new(),
    };
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        subjects: vec![mk_subject("1"), mk_subject("2")],
    }
}

#[test]
fn per_cmt_scaling_dispatches_per_observation_through_predict() {
    // Same baseline model, two scaling configurations:
    //   (a) no scaling — predictions match the analytical 1-cpt IV bolus
    //       formula at every obs.
    //   (b) per-CMT: CMT=1 /1000, CMT=2 /2.
    // Verify the scaled predictions differ from the baseline by exactly
    // the right factor for each CMT.
    let baseline = parse_model_string(ANALYTICAL_BASE).expect("baseline parses");

    let mut scaled_src = String::from(ANALYTICAL_BASE);
    scaled_src.push_str("\n[scaling]\n  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=2] = 2\n");
    let scaled = parse_model_string(&scaled_src).expect("per-CMT scaling parses");

    let pop = two_cmt_pop();
    let base_preds = predict(&baseline, &pop, &baseline.default_params);
    let scaled_preds = predict(&scaled, &pop, &scaled.default_params);

    assert_eq!(base_preds.len(), scaled_preds.len());

    for (subj_i, subj) in pop.subjects.iter().enumerate() {
        let n = subj.obs_times.len();
        let base_off = subj_i * n;
        for j in 0..n {
            let cmt = subj.obs_cmts[j];
            let expected_scale = match cmt {
                1 => 1000.0,
                2 => 2.0,
                other => panic!("unexpected CMT {}", other),
            };
            let base = base_preds[base_off + j].pred;
            let scaled = scaled_preds[base_off + j].pred;
            assert!(base > 0.0);
            let ratio = base / scaled;
            assert!(
                (ratio - expected_scale).abs() < 1e-9,
                "subj {} obs {} CMT={}: expected scale {}, got ratio {}",
                subj.id,
                j,
                cmt,
                expected_scale,
                ratio
            );
        }
    }
}

#[test]
fn per_cmt_scaling_missing_cmt_errors_at_fit() {
    // PerCmt map covers CMT=1 only, but the population has obs on CMT=2.
    // fit() must reject this with a clear error naming the missing CMT.
    use ferx_core::{fit, FitOptions};

    let mut src = String::from(ANALYTICAL_BASE);
    src.push_str("\n[scaling]\n  obs_scale[CMT=1] = 1000\n");
    let model = parse_model_string(&src).expect("PerCmt-with-only-CMT-1 parses");

    let pop = two_cmt_pop(); // observes both CMT=1 and CMT=2

    let mut opts = FitOptions::default();
    opts.verbose = false;
    let err = fit(&model, &pop, &model.default_params, &opts)
        .expect_err("missing per-CMT entry must error at fit() entry");
    let msg = err.to_string();
    assert!(
        msg.contains("[2]") || (msg.contains("missing") && msg.contains("CMT")),
        "expected missing-CMT error, got: {}",
        msg
    );
}

#[test]
fn per_cmt_scaling_works_with_ad_gradient() {
    // Phase 2.5: PerCmt + gradient = ad now goes through the per-observation
    // scale array in `inner_optimizer::build_scale_array_for_ad`. The fit
    // must complete (no parse error, no runtime panic), produce a finite
    // OFV, AND actually use AD (not silently fall back to FD).
    //
    // `fit()` does NOT propagate `opts.gradient_method` to
    // `model.gradient_method` — the inner loop reads `model.gradient_method`
    // which is set by the parser from the `[fit_options]` block. So the
    // model source must hardcode `gradient = ad`. (Caught by Copilot
    // review on PR #85.)
    use ferx_core::{fit, FitOptions};

    let mut src = String::from(ANALYTICAL_BASE).replace("gradient = fd", "gradient = ad");
    src.push_str("\n[scaling]\n  obs_scale[CMT=1] = 1000\n  obs_scale[CMT=2] = 2\n");
    let model = parse_model_string(&src).expect("PerCmt + AD model parses");

    let pop = two_cmt_pop();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let res = fit(&model, &pop, &model.default_params, &opts)
        .expect("PerCmt + AD fit must complete without errors");
    assert!(
        res.ofv.is_finite(),
        "PerCmt + AD OFV must be finite, got {}",
        res.ofv
    );
    // The inner-loop gradient should report Enzyme AD when the autodiff
    // feature is on. Under `--features ci` (no autodiff) it falls back to
    // FD and the check below is skipped; we want the explicit assertion
    // when CI runs the autodiff feature in a separate job.
    #[cfg(feature = "autodiff")]
    assert_eq!(
        res.gradient_method_inner, "Enzyme AD",
        "expected inner-loop AD, got {}",
        res.gradient_method_inner
    );
}

#[test]
fn form_b_expression_scaling_works_with_ad_gradient() {
    // Phase 2.5: ExpressionScale + gradient = ad also lifts (same
    // machinery — subject-static per-obs scale array). Verifies the
    // wiring on the Form B path. See `per_cmt_scaling_works_with_ad_gradient`
    // for why the model string must hardcode `gradient = ad`.
    use ferx_core::{fit, FitOptions};

    let mut src = String::from(ANALYTICAL_BASE).replace("gradient = fd", "gradient = ad");
    src.push_str("\n[scaling]\n  obs_scale = TVV / 10\n");
    let model = parse_model_string(&src).expect("ExpressionScale + AD model parses");

    let pop = one_subject_pop();
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let res = fit(&model, &pop, &model.default_params, &opts)
        .expect("ExpressionScale + AD fit must complete without errors");
    assert!(
        res.ofv.is_finite(),
        "ExpressionScale + AD OFV must be finite, got {}",
        res.ofv
    );
    #[cfg(feature = "autodiff")]
    assert_eq!(
        res.gradient_method_inner, "Enzyme AD",
        "expected inner-loop AD, got {}",
        res.gradient_method_inner
    );
}
