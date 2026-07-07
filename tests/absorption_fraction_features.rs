//! #588 feature-interaction probes: the pathway-fraction / absorption input-rate
//! validation must behave identically on `predict()` / `simulate()` and `fit()`
//! (`check_model_data`) when the model *also* uses bioavailability `F`, a dose
//! `lagtime`, time-varying covariates, or IOV (`kappa`). These are the features
//! that bit earlier absorption work (silent depot-init drop #611; transit + TV-cov
//! silent zeros / IOV scope gaps #663), so a valid feature-rich model must not
//! spuriously trip the new guard, and a genuinely malformed one must still be
//! caught with those features present.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::Population;
use ferx_core::{check_model_data, predict, simulate_with_seed, DoseEvent};

/// Does `check_model_data` raise any absorption diagnostic for this model+pop?
fn has_absorption_err(model: &ferx_core::CompiledModel, pop: &Population) -> bool {
    check_model_data(model, pop)
        .iter()
        .any(|d| d.code.starts_with("E_ABSORPTION"))
}

/// Biphasic IG into central (CL=5, V=50), two declared fractions, plus optional
/// extra `[individual_parameters]` lines (F1 / LAGTIME1 / covariate / kappa refs)
/// and extra `[parameters]` (thetas / kappa). `fr1`/`fr2` are the fraction defs.
fn biphasic(extra_params: &str, indiv: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(5.0,    0.1, 100.0)
  theta TVV(50.0,    5.0, 500.0)
  theta TVMAT1(1.0, 0.05,  24.0)
  theta TVMAT2(4.0, 0.05,  24.0)
  theta TVCV2_1(0.3, 0.001, 10.0)
  theta TVCV2_2(0.5, 0.001, 10.0)
  theta TVFR1(0.6, 0.001, 0.999)
{extra_params}
  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  MAT1  = TVMAT1
  MAT2  = TVMAT2
  CV2_1 = TVCV2_1
  CV2_2 = TVCV2_2
{indiv}

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) + FR2*igd(mat=MAT2, cv2=CV2_2) - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#
    )
}

fn pop_plain(cov_names: Vec<String>) -> Population {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let subj = common::subject("1", vec![dose], obs_times, vec![0.0; n], vec![1; n]);
    Population {
        covariate_names: cov_names,
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subj],
    }
}

#[test]
fn fraction_validation_valid_with_f_and_lagtime() {
    // Bioavailability `F1` scales the delivered mass and `LAGTIME1` shifts tad;
    // neither touches the pathway-fraction slot. A valid split (FR1+FR2=1) must pass
    // `check_model_data` AND not trip the predict/simulate guard.
    let src = biphasic(
        "  theta TVF(0.7, 0.01, 1.0)\n  theta TVLAG(1.5, 0.001, 12.0)",
        "  FR1 = TVFR1\n  FR2 = 1 - TVFR1\n  F1 = TVF\n  LAGTIME1 = TVLAG",
    );
    let model = parse_full_model(&src)
        .expect("biphasic igd + F + lag parses")
        .model;
    let pop = pop_plain(vec![]);
    assert!(
        !has_absorption_err(&model, &pop),
        "valid F+lag flagged by fit-check"
    );
    // Must not panic:
    let preds = predict(&model, &pop, &model.default_params);
    assert!(
        preds.iter().all(|p| p.pred.is_finite()),
        "non-finite pred with F+lag"
    );
    let _ = simulate_with_seed(&model, &pop, &model.default_params, 1, 7);
}

#[test]
#[should_panic(expected = "absorption input-rate machinery cannot honour")]
fn fraction_error_still_caught_with_f_and_lagtime() {
    // Same F+lag model but a malformed split (both fractions 0.6 → Σ=1.2). The
    // features must not mask the error: predict() must still panic.
    let src = biphasic(
        "  theta TVF(0.7, 0.01, 1.0)\n  theta TVLAG(1.5, 0.001, 12.0)",
        "  FR1 = TVFR1\n  FR2 = TVFR1\n  F1 = TVF\n  LAGTIME1 = TVLAG",
    );
    let model = parse_full_model(&src).expect("model parses").model;
    let pop = pop_plain(vec![]);
    assert!(
        has_absorption_err(&model, &pop),
        "fit-check should flag Σ≠1 too"
    );
    let _ = predict(&model, &pop, &model.default_params);
}

#[test]
fn fraction_validation_valid_with_time_varying_covariate() {
    // A fraction driven by a covariate that is *time-varying* in the data. The
    // validation evaluates typical values at the baseline covariate (TIME=0), so a
    // subject whose COVF baseline gives FR1=0.5 (⇒ FR2=0.5, Σ=1) is valid — the same
    // as fit() sees. This is the transit+TV-cov class (#663): the guard must handle a
    // `has_tv_covariates()==true` subject without choking.
    let src = biphasic("", "  FR1 = COVF\n  FR2 = 1 - COVF");
    let model = parse_full_model(&src)
        .expect("biphasic igd + covariate fraction parses")
        .model;

    let mut pop = pop_plain(vec!["COVF".into()]);
    let subj = &mut pop.subjects[0];
    // Baseline COVF = 0.5 → typical FR1 = 0.5 (⇒ FR2 = 0.5, Σ = 1). The per-observation
    // snapshots below make has_tv_covariates() == true (COVF drifts 0.50 → 0.60), but the
    // check reads the baseline value, exactly as fit() does.
    subj.covariates.insert("COVF".into(), 0.5);
    let n = subj.obs_times.len();
    subj.obs_covariates = (0..n)
        .map(|i| {
            let mut m = std::collections::HashMap::new();
            m.insert("COVF".to_string(), 0.50 + 0.02 * i as f64);
            m
        })
        .collect();
    assert!(
        subj.has_tv_covariates(),
        "probe must exercise the TV-cov path"
    );

    assert!(
        !has_absorption_err(&model, &pop),
        "valid TV-cov fraction flagged"
    );
    let preds = predict(&model, &pop, &model.default_params);
    assert!(
        preds.iter().all(|p| p.pred.is_finite()),
        "non-finite pred with TV-cov"
    );
    let _ = simulate_with_seed(&model, &pop, &model.default_params, 1, 7);
}

#[test]
fn fraction_validation_valid_with_iov() {
    // IOV on a fraction: FR1 = TVFR1 * exp(KAPPA_FR). The check zeros kappa (as fit
    // does), so it evaluates the typical FR1 = 0.6 (⇒ FR2 = 0.4, Σ = 1) — valid. The
    // guard must not spuriously fire on a fitted IOV absorption model.
    let src = format!(
        r#"
[parameters]
  theta TVCL(5.0,    0.1, 100.0)
  theta TVV(50.0,    5.0, 500.0)
  theta TVMAT1(1.0, 0.05,  24.0)
  theta TVMAT2(4.0, 0.05,  24.0)
  theta TVCV2_1(0.3, 0.001, 10.0)
  theta TVCV2_2(0.5, 0.001, 10.0)
  theta TVFR1(0.6, 0.001, 0.999)
  omega ETA_CL ~ 0.04
  kappa KAPPA_FR ~ 0.02
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  MAT1  = TVMAT1
  MAT2  = TVMAT2
  CV2_1 = TVCV2_1
  CV2_2 = TVCV2_2
  FR1   = TVFR1 * exp(KAPPA_FR)
  FR2   = 1 - TVFR1

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) + FR2*igd(mat=MAT2, cv2=CV2_2) - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#
    );
    let model = parse_full_model(&src)
        .expect("biphasic igd + IOV parses")
        .model;
    assert!(model.n_kappa > 0, "probe must have IOV");

    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let mut subj = common::subject("1", vec![dose], obs_times, vec![0.0; n], vec![1; n]);
    subj.occasions = vec![1; n];
    subj.dose_occasions = vec![1];

    let pop = Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subj],
    };
    assert!(
        !has_absorption_err(&model, &pop),
        "valid IOV fraction flagged"
    );
    let preds = predict(&model, &pop, &model.default_params);
    assert!(
        preds.iter().all(|p| p.pred.is_finite()),
        "non-finite pred with IOV"
    );
}
