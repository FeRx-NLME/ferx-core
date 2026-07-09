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

[covariates]
  WT  continuous
  AGE continuous


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
        None, // default missing value (-99)
        None, // no prior fit to seed inits from
    )
    .expect("prepare_frem should succeed")
}

/// Rewrite a CSV's header line, renaming TIME→TAFD and DV→CONC so a `[data]`
/// column mapping is required to read it. Body rows are positional, unchanged.
fn rename_time_dv_headers(src: &Path, dst: &Path) {
    let text = std::fs::read_to_string(src).unwrap();
    let mut lines = text.lines();
    let header = lines
        .next()
        .unwrap()
        .replace("TIME", "TAFD")
        .replace("DV", "CONC");
    let body: Vec<&str> = lines.collect();
    let out = std::iter::once(header.as_str())
        .chain(body.iter().copied())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(dst, out).unwrap();
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// #730 regression: FREM prep must honour the model's `[data]` column mapping.
/// A dataset with TAFD/CONC headers is only readable once TIME/DV are mapped;
/// before the fix `prepare_frem` read via the unmapped reader and failed with
/// `Missing TIME column`.
#[test]
fn frem_prepare_honours_data_block_column_mapping() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Dataset with WT/AGE covariates, then TIME→TAFD / DV→CONC in the header.
    let cov_csv = write_warfarin_with_covariates(dir);
    let data_path = dir.join("warfarin_frem_tafd.csv");
    rename_time_dv_headers(&cov_csv, &data_path);

    // Model carries the mapping in its `[data]` block. `path` is required by the
    // parser but ignored here (the explicit data_path argument overrides it).
    let model_src =
        format!("{BASE_MODEL}\n[data]\n  path = ignored.csv\n  TIME = TAFD\n  DV = CONC\n");
    let model_path = dir.join("warfarin_base_mapped.ferx");
    std::fs::write(&model_path, model_src).unwrap();

    let result = prepare_frem(
        &model_path,
        &data_path,
        &["WT".to_string(), "AGE".to_string()],
        None,
        None,
        None,
        None,
        None,
    )
    .expect("prepare_frem should read the mapped TAFD/CONC dataset");

    // Mapping fed real observations through: covariate metadata is well-formed
    // and the renamed roles did not leak in as covariates.
    assert_eq!(result.n_total_etas, 5);
    assert_eq!(result.fremtype_map.len(), 2);
    assert!(result.covariate_means.iter().any(|(n, _)| n == "WT"));
    assert!(!result.covariate_means.iter().any(|(n, _)| n == "TAFD"));
    assert!(!result.covariate_means.iter().any(|(n, _)| n == "CONC"));
}

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

/// IMPMAP on a FREM model exercises the Rao-Blackwellised E-step
/// (`subject_is_draws_frem_rb`): the covariate etas are integrated analytically
/// and only the PK etas are importance-sampled (#406). A few iterations must
/// produce a finite OFV, a 5x5 omega, and the covariate-omega cc-block ≈ the
/// covariate sample covariance (which RB reconstructs exactly as `d dᵀ`).
#[test]
fn frem_impmap_rao_blackwell_runs_finite() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());

    let model = parse_model_file(&result.model_path).unwrap();
    let pop = read_nonmem_csv(&result.data_path, None, None).unwrap();

    let mut opts = FitOptions::default();
    opts.method = ferx_core::EstimationMethod::Impmap;
    opts.impmap_iterations = 3; // fast — just exercise the RB E-step + M-step
    opts.impmap_samples = 200;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let fit_result =
        fit(&model, &pop, &model.default_params, &opts).expect("FREM IMPMAP fit should not error");

    assert!(
        fit_result.ofv.is_finite(),
        "IMPMAP OFV should be finite, got {}",
        fit_result.ofv
    );
    let omega = &fit_result.omega;
    assert_eq!(omega.nrows(), 5);
    assert_eq!(omega.ncols(), 5);
    for i in 0..5 {
        assert!(
            omega[(i, i)] > 0.0 && omega[(i, i)].is_finite(),
            "omega[{i},{i}] should be positive finite, got {}",
            omega[(i, i)]
        );
    }
    // Covariate cc-block ≈ sample covariance (WT var ≈ 111.6, AGE var ≈ 99.4);
    // RB sets it from d dᵀ so it stays in the right ballpark even after 3 iters.
    assert!(
        omega[(3, 3)] > 50.0 && omega[(4, 4)] > 50.0,
        "covariate omega diagonals should be near the sample variances, got {} / {}",
        omega[(3, 3)],
        omega[(4, 4)]
    );
}

/// The defensive mixture (`imp_defensive_alpha > 0`, issue #528) must also apply
/// on the FREM Rao-Blackwell E-step — its covering component is the conditional
/// PK prior `N(μ, P_pp⁻¹)`. A few IMPMAP iterations with the mixture enabled must
/// still produce a finite OFV and a well-formed 5×5 omega (regression for the
/// finding that `subject_is_draws_frem_rb` ignored `imp_defensive_alpha`).
#[test]
fn frem_impmap_rao_blackwell_defensive_mixture_runs_finite() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());

    let model = parse_model_file(&result.model_path).unwrap();
    let pop = read_nonmem_csv(&result.data_path, None, None).unwrap();

    let mut opts = FitOptions::default();
    opts.method = ferx_core::EstimationMethod::Impmap;
    opts.impmap_iterations = 3;
    opts.impmap_samples = 200;
    opts.imp_defensive_alpha = 0.1; // exercise the RB mixture branch
    opts.run_covariance_step = false;
    opts.verbose = false;

    let fit_result = fit(&model, &pop, &model.default_params, &opts)
        .expect("FREM IMPMAP fit with defensive mixture should not error");

    assert!(
        fit_result.ofv.is_finite(),
        "IMPMAP OFV with defensive mixture should be finite, got {}",
        fit_result.ofv
    );
    let omega = &fit_result.omega;
    assert_eq!(omega.nrows(), 5);
    assert_eq!(omega.ncols(), 5);
    for i in 0..5 {
        assert!(
            omega[(i, i)] > 0.0 && omega[(i, i)].is_finite(),
            "omega[{i},{i}] should be positive finite, got {}",
            omega[(i, i)]
        );
    }
}

/// Regression guard for the FREM covariate-marginal 2π bookkeeping.
///
/// `log_p_d` once included the covariate-obs `nc·ln(2π)` normalizer that the
/// rest of the OFV (and NONMEM's "without constant") drops, inflating the RB
/// OFV by `Σ nc·ln(2π)`. Here that offset would be 2 covariates × 10 subjects ×
/// ln(2π) ≈ 36.8 — far above the MC tolerance. Do not compare this against the
/// full-dimensional FREM sampler: with EPSCOV near zero, brute-force sampling of
/// covariate etas has effectively zero ESS at this fast-test sample count.
#[test]
fn frem_rao_blackwell_marginal_drops_covariate_2pi_constant() {
    let tmp = tempfile::tempdir().unwrap();
    let result = setup_frem(tmp.path());
    let model = parse_model_file(&result.model_path).unwrap();
    let pop = read_nonmem_csv(&result.data_path, None, None).unwrap();

    // Evaluate the IS marginal at the fixed initial params (no estimation), so
    // both paths score the identical point.
    let mut base = FitOptions::default();
    base.method = ferx_core::EstimationMethod::Imp;
    base.imp_eval_only = true;
    base.imp_samples = 6000;
    base.imp_seed = Some(20240619);
    base.run_covariance_step = false;
    base.verbose = false;

    let mut opts_rb = base.clone();
    opts_rb.frem_rao_blackwell = true;
    let rb = fit(&model, &pop, &model.default_params, &opts_rb)
        .expect("RB eval should not error")
        .importance_sampling
        .expect("eval-only IMP must populate importance_sampling")
        .minus2_log_likelihood;

    assert!(rb.is_finite(), "RB marginal must be finite, got {rb}");
    // Fixed seed and fast-but-stable K=6000. Tolerance is comfortably below the
    // ~36.8 bug offset and above the observed MC/platform noise.
    let expected = 19781.127590682896_f64;
    assert!(
        (rb - expected).abs() < 12.0,
        "RB marginal {rb} should stay near the seeded no-constant reference \
         {expected} (Δ={})",
        (rb - expected).abs()
    );
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

/// Regression test: under SAEM, adding FREM covariates must NOT shrink the PK
/// residual error. Before the FREM-aware residual override, SAEM scored the
/// covariate pseudo-observations with the PK error model; their near-zero
/// residuals dragged PROP_ERR toward zero. Here we fit the same PK model with
/// and without FREM (both SAEM, same seed) and require the PK residual error to
/// be essentially unchanged.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn frem_saem_does_not_collapse_pk_residual_error() {
    let tmp = tempfile::tempdir().unwrap();

    let saem_opts = || {
        let mut o = FitOptions::default();
        o.method = ferx_core::EstimationMethod::Saem;
        o.saem_n_exploration = 500;
        o.saem_n_convergence = 800;
        o.saem_seed = Some(20260611);
        o.run_covariance_step = false;
        o.verbose = false;
        o
    };

    let prop_err = |fit: &ferx_core::FitResult| -> f64 {
        let idx = fit
            .sigma_names
            .iter()
            .position(|n| n == "PROP_ERR")
            .expect("PROP_ERR sigma must exist");
        fit.sigma[idx]
    };

    // Base PK fit (no FREM).
    let base_model_path = tmp.path().join("warfarin_base.ferx");
    std::fs::write(&base_model_path, BASE_MODEL).unwrap();
    let base_data = write_warfarin_with_covariates(tmp.path());
    let base_model = parse_model_file(&base_model_path).unwrap();
    let base_pop = read_nonmem_csv(&base_data, None, None).unwrap();
    let base_fit = fit(
        &base_model,
        &base_pop,
        &base_model.default_params,
        &saem_opts(),
    )
    .expect("base SAEM fit should succeed");
    let base_prop = prop_err(&base_fit);

    // FREM fit (same PK model + 2 covariates).
    let frem = setup_frem(tmp.path());
    let frem_model = parse_model_file(&frem.model_path).unwrap();
    let frem_pop = read_nonmem_csv(&frem.data_path, None, None).unwrap();
    let frem_fit = fit(
        &frem_model,
        &frem_pop,
        &frem_model.default_params,
        &saem_opts(),
    )
    .expect("FREM SAEM fit should succeed");
    let frem_prop = prop_err(&frem_fit);

    assert!(base_prop.is_finite() && base_prop > 0.0);
    // The PK residual error must not collapse: it should stay close to the base
    // fit (the bug drove it to <0.3x). Generous band to tolerate SAEM noise.
    let ratio = frem_prop / base_prop;
    assert!(
        (0.7..1.4).contains(&ratio),
        "FREM PK PROP_ERR ({frem_prop:.4}) should be ~ base ({base_prop:.4}); ratio {ratio:.2} \
         — a collapse toward 0 indicates covariate rows are scored with the PK error model"
    );
}

/// IMPMAP with `impmap_mceta = 3` runs to completion on a FREM model and
/// produces a finite OFV. This exercises the multi-start MAP helper
/// (`run_map_multistart`) in a high-dimensional setting (5 ETAs).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn frem_impmap_mceta_produces_finite_ofv() {
    let tmp = tempfile::tempdir().unwrap();
    let frem = setup_frem(tmp.path());
    let model = parse_model_file(&frem.model_path).unwrap();
    let pop = read_nonmem_csv(&frem.data_path, None, None).unwrap();

    let mut opts = FitOptions::default();
    opts.method = ferx_core::EstimationMethod::Impmap;
    opts.impmap_iterations = 20;
    opts.impmap_samples = 50;
    opts.impmap_mceta = 3;
    opts.impmap_seed = Some(42);
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP+MCETA fit should succeed");
    assert!(
        result.ofv.is_finite(),
        "OFV should be finite, got {}",
        result.ofv
    );
}

/// Cross-method comparison on the 5-ETA FREM warfarin model: FOCEI, SAEM,
/// IMPMAP (mceta=0), IMPMAP (mceta=3).  Prints a comparison table.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn frem_estimation_method_comparison() {
    let tmp = tempfile::tempdir().unwrap();
    let frem = setup_frem(tmp.path());
    let model = parse_model_file(&frem.model_path).unwrap();
    let pop = read_nonmem_csv(&frem.data_path, None, None).unwrap();

    // Expected sample variances (the FREM ground truth for covariate omegas).
    let wt_var_expected = 111.56;
    let age_var_expected = 99.38;

    struct MethodResult {
        name: String,
        ofv: f64,
        theta: Vec<f64>,
        omega_diag: Vec<f64>,
        sigma: Vec<f64>,
    }

    let mut results: Vec<MethodResult> = Vec::new();

    // ---- FOCEI ----
    {
        let mut opts = FitOptions::default();
        opts.method = ferx_core::EstimationMethod::FoceI;
        opts.outer_maxiter = 300;
        opts.run_covariance_step = false;
        opts.verbose = false;
        let r = fit(&model, &pop, &model.default_params, &opts).expect("FOCEI fit should succeed");
        let n = r.omega.nrows();
        results.push(MethodResult {
            name: "FOCEI".to_string(),
            ofv: r.ofv,
            theta: r.theta.clone(),
            omega_diag: (0..n).map(|i| r.omega[(i, i)]).collect(),
            sigma: r.sigma.clone(),
        });
    }

    // ---- SAEM ----
    {
        let mut opts = FitOptions::default();
        opts.method = ferx_core::EstimationMethod::Saem;
        opts.saem_n_exploration = 500;
        opts.saem_n_convergence = 800;
        opts.saem_seed = Some(20260614);
        opts.run_covariance_step = false;
        opts.verbose = false;
        let r = fit(&model, &pop, &model.default_params, &opts).expect("SAEM fit should succeed");
        let n = r.omega.nrows();
        results.push(MethodResult {
            name: "SAEM".to_string(),
            ofv: r.ofv,
            theta: r.theta.clone(),
            omega_diag: (0..n).map(|i| r.omega[(i, i)]).collect(),
            sigma: r.sigma.clone(),
        });
    }

    // ---- IMPMAP (mceta=0) ----
    {
        let mut opts = FitOptions::default();
        opts.method = ferx_core::EstimationMethod::Impmap;
        opts.impmap_iterations = 200;
        opts.impmap_samples = 300;
        opts.impmap_mceta = 0;
        opts.impmap_seed = Some(12345);
        opts.run_covariance_step = false;
        opts.verbose = false;
        let r = fit(&model, &pop, &model.default_params, &opts)
            .expect("IMPMAP mceta=0 fit should succeed");
        let n = r.omega.nrows();
        results.push(MethodResult {
            name: "IMPMAP".to_string(),
            ofv: r.ofv,
            theta: r.theta.clone(),
            omega_diag: (0..n).map(|i| r.omega[(i, i)]).collect(),
            sigma: r.sigma.clone(),
        });
    }

    // ---- IMPMAP (mceta=3) ----
    {
        let mut opts = FitOptions::default();
        opts.method = ferx_core::EstimationMethod::Impmap;
        opts.impmap_iterations = 200;
        opts.impmap_samples = 300;
        opts.impmap_mceta = 3;
        opts.impmap_seed = Some(12345);
        opts.run_covariance_step = false;
        opts.verbose = false;
        let r = fit(&model, &pop, &model.default_params, &opts)
            .expect("IMPMAP mceta=3 fit should succeed");
        let n = r.omega.nrows();
        results.push(MethodResult {
            name: "IMPMAP+MCETA3".to_string(),
            ofv: r.ofv,
            theta: r.theta.clone(),
            omega_diag: (0..n).map(|i| r.omega[(i, i)]).collect(),
            sigma: r.sigma.clone(),
        });
    }

    // ---- Print comparison table ----
    let theta_names = ["TVCL", "TVV", "TVKA"];
    let omega_names = ["w2_CL", "w2_V", "w2_KA", "w2_WT", "w2_AGE"];

    eprintln!("\n=== FREM Warfarin Estimation Method Comparison (5 ETAs, 10 subjects) ===\n");
    eprint!("{:<16}", "Parameter");
    for r in &results {
        eprint!("{:>16}", r.name);
    }
    eprintln!();
    eprintln!("{}", "-".repeat(16 + 16 * results.len()));

    // OFV
    eprint!("{:<16}", "OFV");
    for r in &results {
        eprint!("{:>16.2}", r.ofv);
    }
    eprintln!();

    // Thetas (first 3 only, 4-5 are FIX)
    for (i, name) in theta_names.iter().enumerate() {
        eprint!("{:<16}", name);
        for r in &results {
            eprint!("{:>16.4}", r.theta[i]);
        }
        eprintln!();
    }

    // Omega diagonals
    for (i, name) in omega_names.iter().enumerate() {
        eprint!("{:<16}", name);
        for r in &results {
            eprint!("{:>16.4}", r.omega_diag[i]);
        }
        eprintln!();
    }

    // Sigma
    eprint!("{:<16}", "sigma_PROP");
    for r in &results {
        eprint!("{:>16.6}", r.sigma[0]);
    }
    eprintln!();

    eprintln!(
        "\nExpected covariate variances: WT = {:.2}, AGE = {:.2}",
        wt_var_expected, age_var_expected
    );

    // Basic sanity assertions
    for r in &results {
        assert!(r.ofv.is_finite(), "{} OFV not finite: {}", r.name, r.ofv);
    }
}
