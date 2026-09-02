//! Tier-3 (slow) integration test for GAM covariate screening (#1114).
//!
//! Fits a base (no-covariate) two-compartment oral model to the
//! `two_cpt_oral_cov` dataset, then runs GAM screening and asserts that
//! WT and CRCL rank as the top covariates for ETA_CL — as expected, because
//! the data were generated *with* WT and CRCL effects on CL.
//!
//! The test must converge to a minimum to produce meaningful EBEs, so it
//! runs to full convergence and is gated behind `--features slow-tests`.

use std::io::Write;

use ferx_core::{fit, prepare_run};
use ferx_tools::gam::{gam_screen, GamOptions};

/// Two-compartment oral base model: no covariate effects, but WT and CRCL
/// are *declared* in [covariates] so the data reader populates
/// `pop.subjects[i].covariates` and `pop.covariate_names`.
const BASE_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)

  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_Q  ~ 0.08
  omega ETA_V2 ~ 0.08
  omega ETA_KA ~ 0.20

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 300
  covariance = false
"#;

const DATA: &str = "../../data/two_cpt_oral_cov.csv";

/// Number of uninformative covariates injected alongside WT and CRCL.
///
/// The dataset carries exactly two covariate columns, so a "WT and CRCL rank
/// top-2" assertion against it alone is vacuous — they are the *only* two
/// candidates, and the assertion holds however the screen ranks them. Decoys
/// give the ranking something to be wrong about.
const N_DECOYS: usize = 4;

/// Add `N_DECOYS` deterministic, structurally unrelated covariates to every
/// subject.
///
/// The values are a fixed function of the subject's position and the decoy
/// index — reproducible run to run, and carrying no information about the
/// simulated CL, so a correct screen must rank every one of them below WT and
/// CRCL.
fn inject_decoy_covariates(pop: &mut ferx_core::Population) {
    for (i, subject) in pop.subjects.iter_mut().enumerate() {
        for d in 0..N_DECOYS {
            let phase = (d as f64 + 1.0) * 0.7;
            let value = 50.0 + ((i as f64 * phase + 0.31).sin() * 12.0);
            subject
                .covariates
                .insert(format!("DECOY_{d}"), (value * 1e6).round() / 1e6);
        }
    }
    for d in 0..N_DECOYS {
        pop.covariate_names.push(format!("DECOY_{d}"));
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gam_ranks_wt_and_crcl_for_eta_cl() {
    // Write the base model to a temp file.
    let mut model_file = tempfile::Builder::new()
        .suffix(".ferx")
        .tempfile()
        .expect("create temp model file");
    model_file
        .write_all(BASE_MODEL.as_bytes())
        .expect("write base model");
    let model_path = model_file.path().to_str().unwrap().to_string();

    // Parse model and read data.
    let mut p = prepare_run(&model_path, Some(DATA)).expect("prepare_run for base model");

    // Silence output and disable the covariance step.
    p.parsed.fit_options.verbose = false;
    p.parsed.fit_options.run_covariance_step = false;
    p.parsed.fit_options.threads = Some(4);

    let fit_result = fit(
        &p.parsed.model,
        &p.population,
        &p.init_params,
        &p.parsed.fit_options,
    )
    .expect("base model fit should succeed");

    assert!(
        fit_result.converged,
        "base model should converge; OFV = {}",
        fit_result.ofv
    );

    // Screen WT and CRCL against decoys that carry no signal, so the ranking
    // assertions below can actually fail.
    inject_decoy_covariates(&mut p.population);

    // Run GAM screening with default options.
    let gam = gam_screen(&fit_result, &p.population, &GamOptions::default());

    // No error-level warnings (high shrinkage is possible for some ETAs, just
    // advisory; convergence / NaN warnings would be a real failure).
    let hard_warnings: Vec<&str> = gam
        .warnings
        .iter()
        .map(|s| s.as_str())
        .filter(|w| !w.contains("shrinkage"))
        .collect();
    assert!(
        hard_warnings.is_empty(),
        "unexpected hard warnings: {hard_warnings:?}"
    );

    // Find the ETA_CL result.
    let eta_cl = gam
        .eta_results
        .iter()
        .find(|r| r.eta_name == "ETA_CL")
        .expect("ETA_CL should be in results");

    // Both WT and CRCL should appear in the screening output.
    let cov_names: Vec<&str> = eta_cl
        .covariate_scores
        .iter()
        .map(|s| s.covariate.as_str())
        .collect();
    assert!(
        cov_names.contains(&"WT"),
        "WT should be screened for ETA_CL; got {cov_names:?}"
    );
    assert!(
        cov_names.contains(&"CRCL"),
        "CRCL should be screened for ETA_CL; got {cov_names:?}"
    );

    // Every decoy must be screened too, otherwise the comparison below is
    // between WT/CRCL and nothing.
    assert_eq!(
        eta_cl.covariate_scores.len(),
        2 + N_DECOYS,
        "all covariates should be screened; got {cov_names:?}"
    );

    // WT and CRCL must be *the* top two for ETA_CL — the dataset was generated
    // with both effects on CL, and the decoys have none.
    let top2: Vec<&str> = eta_cl
        .covariate_scores
        .iter()
        .take(2)
        .map(|s| s.covariate.as_str())
        .collect();
    assert!(
        top2.contains(&"WT") && top2.contains(&"CRCL"),
        "WT and CRCL should be the top-2 for ETA_CL, ahead of {N_DECOYS} decoys; \
         full ranking: {:?}",
        eta_cl
            .covariate_scores
            .iter()
            .map(|s| (s.covariate.as_str(), s.delta_aic))
            .collect::<Vec<_>>()
    );

    // And they must be ahead of the decoys by a real margin, not a tie-break.
    let delta = |name: &str| {
        eta_cl
            .covariate_scores
            .iter()
            .find(|s| s.covariate == name)
            .unwrap_or_else(|| panic!("{name} should be screened"))
            .delta_aic
    };
    let best_decoy: f64 = (0..N_DECOYS)
        .map(|d| delta(&format!("DECOY_{d}")))
        .fold(f64::NEG_INFINITY, f64::max);
    for name in ["WT", "CRCL"] {
        assert!(
            delta(name) > best_decoy + 1.0,
            "{name} (Δ AIC = {:.2}) should beat the best decoy \
             (Δ AIC = {best_decoy:.2}) by more than 1",
            delta(name)
        );
    }

    // At least one should show a clearly positive ΔAIC (informative covariate).
    let max_delta = delta("WT").max(delta("CRCL"));
    assert!(
        max_delta > 1.0,
        "WT or CRCL should have Δ AIC > 1 for ETA_CL; got max Δ AIC = {max_delta:.2}"
    );
}
