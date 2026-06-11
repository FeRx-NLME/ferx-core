//! FREM end-to-end integration tests using the warfarin one-compartment oral model.
//!
//! Tier 2 tests: call the public API (`prepare_frem` → `fit`) but limit outer
//! iterations so they finish quickly.  The slow-gated test runs to convergence.

use ferx_core::{
    fit, parse_model_file, prepare_frem, read_nonmem_csv, FitOptions, FremPrepareResult,
};
use std::io::Write;
use std::path::Path;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Warfarin base model (one-cpt oral, proportional error, 3 etas).
const BASE_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 300
"#;

/// Write the warfarin CSV with synthetic WT and AGE covariates appended.
///
/// WT and AGE are assigned per subject (constant across rows).  The values
/// are chosen to give non-trivial sample variance so the FREM covariate
/// omega diagonal has clear targets to hit.
fn write_warfarin_with_covariates(dir: &Path) -> std::path::PathBuf {
    // (subject_id, WT, AGE) — 10 subjects
    let cov_table: &[(u32, f64, f64)] = &[
        (1, 70.0, 35.0),
        (2, 80.0, 45.0),
        (3, 65.0, 28.0),
        (4, 90.0, 55.0),
        (5, 55.0, 22.0),
        (6, 75.0, 40.0),
        (7, 85.0, 50.0),
        (8, 60.0, 30.0),
        (9, 72.0, 38.0),
        (10, 68.0, 33.0),
    ];

    let base_csv = include_str!("../data/warfarin.csv");
    let data_path = dir.join("warfarin_frem.csv");
    let mut f = std::fs::File::create(&data_path).unwrap();

    for (i, line) in base_csv.lines().enumerate() {
        if i == 0 {
            writeln!(f, "{},WT,AGE", line).unwrap();
            continue;
        }
        let id: u32 = line.split(',').next().unwrap().parse().unwrap();
        let (_, wt, age) = cov_table.iter().find(|(sid, _, _)| *sid == id).unwrap();
        writeln!(f, "{},{},{}", line, wt, age).unwrap();
    }
    data_path
}

/// Run `prepare_frem` in a tempdir and return the result + paths.
fn setup_frem(dir: &Path) -> FremPrepareResult {
    let model_path = dir.join("warfarin_base.ferx");
    std::fs::write(&model_path, BASE_MODEL).unwrap();

    let data_path = write_warfarin_with_covariates(dir);

    prepare_frem(
        &model_path,
        &data_path,
        &["WT".to_string(), "AGE".to_string()],
        None, // no categoricals
        None, // default output model path
        None, // default output data path
    )
    .expect("prepare_frem should succeed")
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// FREM preparation produces correct omega dimensions and covariate metadata.
#[test]
fn frem_prepare_produces_correct_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());

    // 3 PK etas + 2 covariate etas = 5 total.
    assert_eq!(result.n_total_etas, 5);

    // FREMTYPE mapping.
    assert_eq!(result.fremtype_map.len(), 2);
    assert_eq!(result.fremtype_map[0], ("WT".to_string(), 100));
    assert_eq!(result.fremtype_map[1], ("AGE".to_string(), 200));

    // Covariate means: WT mean = (70+80+65+90+55+75+85+60+72+68)/10 = 72.0
    let wt_mean = result
        .covariate_means
        .iter()
        .find(|(n, _)| n == "WT")
        .unwrap()
        .1;
    assert!((wt_mean - 72.0).abs() < 0.01, "WT mean = {wt_mean}");

    // AGE mean = (35+45+28+55+22+40+50+30+38+33)/10 = 37.6
    let age_mean = result
        .covariate_means
        .iter()
        .find(|(n, _)| n == "AGE")
        .unwrap()
        .1;
    assert!((age_mean - 37.6).abs() < 0.01, "AGE mean = {age_mean}");

    // Check output files exist.
    assert!(result.model_path.exists(), "FREM model file not written");
    assert!(result.data_path.exists(), "FREM data file not written");
}

/// Generated FREM model parses successfully and has correct parameter counts.
#[test]
fn frem_generated_model_parses() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());

    let model = parse_model_file(&result.model_path).expect("FREM model should parse");

    // 3 base thetas + 2 fixed covariate thetas = 5 thetas.
    assert_eq!(model.n_theta, 5, "expected 5 thetas, got {}", model.n_theta);

    // 3 PK etas + 2 covariate etas = 5 etas.
    assert_eq!(model.n_eta, 5, "expected 5 etas, got {}", model.n_eta);

    // Covariate thetas should be fixed.
    assert!(model.default_params.theta_fixed[3], "TV_WT should be fixed");
    assert!(
        model.default_params.theta_fixed[4],
        "TV_AGE should be fixed"
    );

    // Omega should be 5x5.
    let omega = &model.default_params.omega;
    assert_eq!(omega.matrix.nrows(), 5);
    assert_eq!(omega.matrix.ncols(), 5);

    // FREM config should be present.
    assert!(model.frem_config.is_some(), "frem_config should be set");
    let fc = model.frem_config.as_ref().unwrap();
    assert_eq!(fc.fremtype_to_indices.len(), 2);
}

/// FREM dataset has the right number of rows (original + covariate pseudo-obs).
#[test]
fn frem_dataset_row_count() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());

    let content = std::fs::read_to_string(&result.data_path).unwrap();
    let n_lines = content.lines().count();

    // Original: 10 subjects × (1 dose + 11 obs) = 120 rows + 1 header = 121 lines.
    // FREM adds: 10 subjects × 2 covariates = 20 pseudo-obs.
    // Total: 120 + 20 = 140 data rows + 1 header = 141 lines.
    assert_eq!(n_lines, 141, "expected 141 lines (header + 140 rows)");

    // Verify FREMTYPE column exists and has correct values.
    let header = content.lines().next().unwrap();
    let ft_col = header
        .split(',')
        .position(|h| h == "FREMTYPE")
        .expect("FREMTYPE column missing");

    let mut ft_100_count = 0;
    let mut ft_200_count = 0;
    for line in content.lines().skip(1) {
        let ft: u16 = line.split(',').nth(ft_col).unwrap().parse().unwrap();
        match ft {
            100 => ft_100_count += 1,
            200 => ft_200_count += 1,
            _ => {}
        }
    }
    assert_eq!(ft_100_count, 10, "should have 10 WT pseudo-obs");
    assert_eq!(ft_200_count, 10, "should have 10 AGE pseudo-obs");
}

/// FREM fit completes (fast, 3 outer iterations) with finite OFV and correct omega size.
#[test]
fn frem_fit_completes_with_finite_ofv() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());

    let model = parse_model_file(&result.model_path).unwrap();
    let pop = read_nonmem_csv(&result.data_path, None, None).unwrap();

    let mut opts = FitOptions::default();
    opts.outer_maxiter = 3; // fast — just verify it doesn't crash
    opts.run_covariance_step = false;
    opts.verbose = false;

    let fit_result =
        fit(&model, &pop, &model.default_params, &opts).expect("FREM fit should not error");

    assert!(
        fit_result.ofv.is_finite(),
        "OFV should be finite, got {}",
        fit_result.ofv
    );

    // Final omega should be 5x5.
    let omega = &fit_result.omega;
    assert_eq!(omega.nrows(), 5);
    assert_eq!(omega.ncols(), 5);

    // PK omega diagonal should be positive.
    for i in 0..3 {
        assert!(omega[(i, i)] > 0.0, "PK omega[{i},{i}] should be positive");
    }

    // Covariate omega diagonal should be positive and in the right ballpark.
    // WT sample variance ≈ 111.6, AGE sample variance ≈ 99.4
    // After only 3 iterations these won't match exactly, but they should be
    // positive and non-trivial (the initial values from prepare_frem are the
    // sample variances, so they start close).
    for i in 3..5 {
        assert!(
            omega[(i, i)] > 1.0,
            "Covariate omega[{i},{i}] should be > 1.0, got {}",
            omega[(i, i)]
        );
    }
}

/// FREM covariate omega diagonals converge to sample variances.
///
/// This is the key FREM correctness check: covariate omega diagonals should
/// approximately equal the sample variance of each covariate, since the
/// covariate thetas are fixed at the sample mean and the only "data" for
/// covariate observations is the subject's own value.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn frem_covariate_omega_matches_sample_variance() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());

    let model = parse_model_file(&result.model_path).unwrap();
    let pop = read_nonmem_csv(&result.data_path, None, None).unwrap();

    // Run to convergence with SAEM (natural choice for large block omega).
    let mut opts = FitOptions::default();
    opts.method = ferx_core::EstimationMethod::Saem;
    opts.saem_n_exploration = 500;
    opts.saem_n_convergence = 800;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let fit_result =
        fit(&model, &pop, &model.default_params, &opts).expect("FREM SAEM fit should succeed");

    assert!(fit_result.ofv.is_finite(), "OFV should be finite");

    let omega = &fit_result.omega;

    // WT sample variance: Var([70,80,65,90,55,75,85,60,72,68]) = 111.56
    let wt_var = omega[(3, 3)];
    let wt_expected = 111.56;
    let wt_pct = ((wt_var - wt_expected) / wt_expected * 100.0).abs();
    assert!(
        wt_pct < 15.0,
        "WT omega diag ({wt_var:.2}) should be within 15% of sample var ({wt_expected:.2}), got {wt_pct:.1}%"
    );

    // AGE sample variance: Var([35,45,28,55,22,40,50,30,38,33]) = 99.38
    let age_var = omega[(4, 4)];
    let age_expected = 99.38;
    let age_pct = ((age_var - age_expected) / age_expected * 100.0).abs();
    assert!(
        age_pct < 15.0,
        "AGE omega diag ({age_var:.2}) should be within 15% of sample var ({age_expected:.2}), got {age_pct:.1}%"
    );
}
