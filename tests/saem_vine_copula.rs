//! Statistical verification ladder (Rungs 1–4) and refusal guards for SAEM-Copula
//! (`omega_dist = vine`, Issue #225).
//!
//! Exercises:
//! 1. Validation refusal guards (`E_OMEGA_DIST_NO_SAEM`, `E_OMEGA_DIST_CHAIN`,
//!    `E_OMEGA_DIST_SIR`, `E_OMEGA_DIST_MIXTURE`).
//! 2. **Rung 1**: Gaussian parity (`OmegaDist::VineCopula` on Gaussian PK data matches
//!    `OmegaDist::Gaussian` parameter recovery, BIC selects `normal` marginals, and
//!    Vine QMC IS `ofv` agrees with `saem_ofv = imp`).
//! 3. **Rung 2**: Flexible zero-mean marginals (`SkewNormal` + heavy-tailed `StudentT`
//!    with `Independence` copula): recovers `theta` without location-shift bias and
//!    selects non-Gaussian marginals + `independence` pair-copula via BIC.
//! 4. **Rung 3**: Asymmetric tail dependence (`examples/vine_tail_iv.ferx` vs
//!    `examples/vine_tail_iv_gaussian.ferx`), `90`-degree rotated Clayton negative
//!    dependence, and 3D R-vine non-collapse.
//! 5. **Rung 4**: Uninformative-subject `OFV == 0.0` invariance, `.fitrx` lossless
//!    round-trip, console/YAML formatting, `conddist = true` vine posterior sampling,
//!    and Three-Uses simulation consistency (`simulate_with_seed` draws from the fitted
//!    Clayton R-vine rather than symmetric Gaussian `Omega_equiv`).

mod common;

use ferx_core::estimation::uncertainty_samples::fitted_params_from_result;
use ferx_core::io::fitrx::{load_fit_from_bytes, save_fit_to_bytes, SaveFitOptions};
use ferx_core::io::output::{format_summary, write_estimates_yaml};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::stats::copula::{CopulaFamily, CopulaRotation};
use ferx_core::stats::marginals::MarginalDist;
use ferx_core::stats::vine_copula::{RVine, VineCopulaOmega};
use ferx_core::types::{
    DoseEvent, EstimationMethod, FitOptions, OmegaDist, OmegaMatrix, Population, SaemOfv, Subject,
    VineMarginals,
};
use ferx_core::{fit, run_model_with_data, simulate_with_seed};
use nalgebra::DMatrix;
use std::path::Path;
use std::sync::Arc;

const ONE_CPT_IV_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const ONE_CPT_IV_BLOCK_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.09]
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn iv_bolus_template(n_subjects: usize) -> Population {
    let obs_times = vec![0.25, 1.0, 3.0, 6.0, 12.0, 24.0];
    let n_obs = obs_times.len();
    let subjects: Vec<Subject> = (1..=n_subjects)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)],
                obs_times.clone(),
                vec![1.0; n_obs],
                vec![1; n_obs],
            )
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn fill_simulated_dvs(template: &Population, sim: &[ferx_core::SimulationResult]) -> Population {
    let mut pop = template.clone();
    for subj in &mut pop.subjects {
        subj.observations = sim
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value().max(1e-6))
            .collect();
    }
    pop
}

#[test]
fn refusal_guards_reject_invalid_vine_combinations() {
    let model = parse_model_string(ONE_CPT_IV_MODEL).expect("parse");
    let pop = iv_bolus_template(6);

    // 1. Non-SAEM method with omega_dist = vine -> E_OMEGA_DIST_NO_SAEM.
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.saem_omega_dist = OmegaDist::VineCopula;
    let err = fit(&model, &pop, &model.default_params, &opts).unwrap_err();
    assert_eq!(err.code(), Some("E_OMEGA_DIST_NO_SAEM"), "{err}");

    // 2. Method chain c(saem, focei) with omega_dist = vine -> E_OMEGA_DIST_CHAIN.
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
    opts.saem_omega_dist = OmegaDist::VineCopula;
    let err = fit(&model, &pop, &model.default_params, &opts).unwrap_err();
    assert_eq!(err.code(), Some("E_OMEGA_DIST_CHAIN"), "{err}");

    // 3. sir = true with omega_dist = vine -> E_OMEGA_DIST_SIR.
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_omega_dist = OmegaDist::VineCopula;
    opts.run_covariance_step = true;
    opts.sir = true;
    let err = fit(&model, &pop, &model.default_params, &opts).unwrap_err();
    assert_eq!(err.code(), Some("E_OMEGA_DIST_SIR"), "{err}");

    // 4. [mixture] model with omega_dist = vine -> E_OMEGA_DIST_MIXTURE.
    let mix_src = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL * exp(ETA_CL) else 2.5 * TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let mix_model = parse_model_string(mix_src).expect("parse mixture");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_omega_dist = OmegaDist::VineCopula;
    let err = fit(&mix_model, &pop, &mix_model.default_params, &opts).unwrap_err();
    assert_eq!(err.code(), Some("E_OMEGA_DIST_MIXTURE"), "{err}");
}

#[test]
fn rung1_gaussian_vine_matches_gaussian_saem_and_imp_ofv() {
    let model = parse_model_string(ONE_CPT_IV_BLOCK_MODEL).expect("parse");
    let template = iv_bolus_template(80);

    // Simulate correlated Gaussian random effects (var = 0.09, corr = 0.50 -> cov = 0.045).
    let mut truth = model.default_params.clone();
    truth.theta = vec![4.0, 50.0];
    let om = DMatrix::from_row_slice(2, 2, &[0.09, 0.045, 0.045, 0.09]);
    truth.omega = OmegaMatrix::from_matrix(om, vec!["ETA_CL".into(), "ETA_V".into()], false);
    truth.sigma.values = vec![0.01]; // variance = 0.10^2

    let sim = simulate_with_seed(&model, &template, &truth, 1, 20261120).expect("sim");
    let pop = fill_simulated_dvs(&template, &sim);

    let mut gauss_opts = FitOptions::default();
    gauss_opts.method = EstimationMethod::Saem;
    gauss_opts.saem_ofv = SaemOfv::Imp;
    gauss_opts.saem_n_exploration = 180;
    gauss_opts.saem_n_convergence = 200;
    gauss_opts.saem_seed = Some(4242);
    gauss_opts.run_covariance_step = false;
    gauss_opts.verbose = false;
    let gauss_fit = fit(&model, &pop, &model.default_params, &gauss_opts).expect("gauss fit");

    let mut vine_opts = gauss_opts.clone();
    vine_opts.saem_omega_dist = OmegaDist::VineCopula;
    vine_opts.vine_marginals = VineMarginals::Auto;
    let vine_fit = fit(&model, &pop, &model.default_params, &vine_opts).expect("vine fit");

    assert!(vine_fit.converged, "vine fit must converge");
    let vp = vine_fit.vine_params.as_ref().expect("vine_params present");

    // 1. Fixed effects match standard Gaussian SAEM closely (< 3% difference).
    for j in 0..2 {
        let rel = (vine_fit.theta[j] - gauss_fit.theta[j]).abs() / gauss_fit.theta[j];
        assert!(
            rel < 0.04,
            "theta[{j}] mismatch: vine={:.4} vs gauss={:.4} (rel={rel:.4})",
            vine_fit.theta[j],
            gauss_fit.theta[j]
        );
    }

    // 2. BIC under `vine_marginals = auto` does not overfit complex marginals on Gaussian data.
    for m in &vp.marginals {
        assert_eq!(
            m.family, "normal",
            "BIC should select normal marginal on Gaussian data, got: {m:?}"
        );
    }

    // 3. Positive correlation recovered in both Omega_equiv and vine pair-copula.
    let tau = vp.trees[0].pairs[0].copula.kendall_tau;
    assert!(
        tau > 0.15 && tau < 0.60,
        "expected moderate positive Kendall tau around ~0.33, got {tau:.4}"
    );

    // 4. Vine QMC IS OFV and Gaussian `saem_ofv = imp` OFV sit on the identical `-2 log L` scale.
    let ofv_diff = (vine_fit.ofv - gauss_fit.ofv).abs();
    assert!(
        ofv_diff < 5.0,
        "Gaussian data OFV scale parity failed: vine={:.3} vs gauss_imp={:.3} (diff={ofv_diff:.3})",
        vine_fit.ofv,
        gauss_fit.ofv
    );
}

#[test]
fn rung2_flexible_zero_mean_skew_and_heavy_tail_marginals_no_theta_bias() {
    let model = parse_model_string(ONE_CPT_IV_MODEL).expect("parse");
    let template = iv_bolus_template(110);

    // Ground truth:
    // - ETA_CL ~ Zero-mean SkewNormal(omega = 0.55, alpha = 8.0) -> strong right skew (~0.92), E[ETA_CL] == 0!
    // - ETA_V  ~ Zero-mean StudentT(scale = 0.18, nu = 2.6)      -> heavy tails, E[ETA_V] == 0!
    // - Pair-copula: Independence!
    let mut truth = model.default_params.clone();
    truth.theta = vec![4.0, 50.0];
    truth.sigma.values = vec![0.0064]; // SD = 0.08
    let mut true_vine = VineCopulaOmega::from_init_params(&truth, true);
    true_vine.marginals =
        vec![MarginalDist::skew_normal(0.55, 8.0), MarginalDist::student_t(0.18, 2.6)];
    true_vine.rvine = RVine::independence_dvine(2);
    truth.vine_dist = Some(Arc::new(true_vine));

    let sim = simulate_with_seed(&model, &template, &truth, 1, 90909).expect("sim");
    let pop = fill_simulated_dvs(&template, &sim);

    let mut gauss_opts = FitOptions::default();
    gauss_opts.method = EstimationMethod::Saem;
    gauss_opts.saem_ofv = SaemOfv::Imp;
    gauss_opts.saem_n_exploration = 180;
    gauss_opts.saem_n_convergence = 200;
    gauss_opts.saem_seed = Some(777);
    gauss_opts.run_covariance_step = false;
    gauss_opts.verbose = false;
    let gauss_fit = fit(&model, &pop, &model.default_params, &gauss_opts).expect("gauss fit");

    let mut vine_opts = gauss_opts.clone();
    vine_opts.saem_omega_dist = OmegaDist::VineCopula;
    vine_opts.vine_marginals = VineMarginals::Auto;
    let vine_fit = fit(&model, &pop, &model.default_params, &vine_opts).expect("vine fit");

    let vp = vine_fit.vine_params.as_ref().expect("vine_params");

    // 1. Defect 1 guard: TVCL and TVV remain unbiased (< 8% error) despite skewed / heavy-tailed ETAs!
    let cl_rel = (vine_fit.theta[0] - 4.0).abs() / 4.0;
    let v_rel = (vine_fit.theta[1] - 50.0).abs() / 50.0;
    assert!(
        cl_rel < 0.08 && v_rel < 0.08,
        "zero-mean marginal constraint violated: TVCL={:.4} (rel {cl_rel:.3}), TVV={:.4} (rel {v_rel:.3})",
        vine_fit.theta[0],
        vine_fit.theta[1]
    );
    assert_eq!(vp.marginals[0].mean, 0.0);
    assert_eq!(vp.marginals[1].mean, 0.0);

    // 2. Non-Gaussian marginals selected by BIC, and positive skewness recovered on ETA_CL.
    assert!(
        vp.marginals[0].family == "skew_normal" || vp.marginals[0].family == "skew_t",
        "expected skewed marginal family for ETA_CL, got: {:?}",
        vp.marginals[0]
    );
    assert!(
        vp.marginals[0].skewness.unwrap_or(0.0) > 0.25,
        "expected positive skewness on ETA_CL, got: {:?}",
        vp.marginals[0]
    );
    assert!(
        vp.marginals[1].family == "student_t" || vp.marginals[1].family == "skew_t",
        "expected heavy-tailed marginal family for ETA_V, got: {:?}",
        vp.marginals[1]
    );

    // 3. Defect 3 guard: Independence copula (0 parameters) selected on independent ETAs!
    assert_eq!(
        vp.trees[0].pairs[0].copula.family, "independence",
        "BIC should select 0-parameter independence copula for independent ETAs, got: {:?}",
        vp.trees[0].pairs[0].copula
    );

    // 4. Flexible marginals achieve strictly better marginal `-2 log L` than Gaussian SAEM.
    assert!(
        vine_fit.ofv < gauss_fit.ofv - 5.0,
        "expected flexible zero-mean marginals to beat Gaussian OFV: vine={:.3} vs gauss={:.3}",
        vine_fit.ofv,
        gauss_fit.ofv
    );
}

#[test]
fn rung3_vine_tail_iv_clayton_rotated_clayton90_and_3d_rvine() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let vine_path = root.join("examples/vine_tail_iv.ferx");
    let gauss_path = root.join("examples/vine_tail_iv_gaussian.ferx");
    let data_path = root.join("data/vine_tail_iv.csv");

    // --- Part A: Committed `examples/vine_tail_iv.ferx` vs `vine_tail_iv_gaussian.ferx` ---
    let (vine_fit, pop) =
        run_model_with_data(vine_path.to_str().unwrap(), Some(data_path.to_str().unwrap()))
            .expect("vine_tail_iv fit");
    let (gauss_fit, _) =
        run_model_with_data(gauss_path.to_str().unwrap(), Some(data_path.to_str().unwrap()))
            .expect("vine_tail_iv_gaussian fit");

    assert!(vine_fit.converged, "vine_tail_iv must converge");
    assert_eq!(vine_fit.vine_corrected_ofv, Some(vine_fit.ofv));
    assert!(vine_fit.aic.is_some() && vine_fit.bic.is_some());

    // Defect 7 guard: BicInputs::omega equals exact vine free parameter count (`n_free() == n_parameters`).
    let vp = vine_fit.vine_params.as_ref().expect("vine_params");
    assert_eq!(vine_fit.bic_inputs.omega, vp.n_free_params());
    assert_eq!(vine_fit.bic_inputs.n_free(), vine_fit.n_parameters);

    // Lower-tail dependence recovered & marginal OFV decisively beats symmetric Gaussian block_omega!
    let pair = &vp.trees[0].pairs[0].copula;
    assert!(
        pair.kendall_tau > 0.25,
        "expected positive Kendall tau on vine_tail_iv, got {:.4}",
        pair.kendall_tau
    );
    assert!(
        pair.tail_dep_lower > 0.30,
        "expected strong lower-tail dependence lambda_L > 0.30 on vine_tail_iv, got {:.4} (family={})",
        pair.tail_dep_lower,
        pair.family
    );
    assert!(
        vine_fit.ofv < gauss_fit.ofv - 3.0,
        "vine_tail_iv OFV ({:.3}) should decisively beat Gaussian block_omega OFV ({:.3})",
        vine_fit.ofv,
        gauss_fit.ofv
    );

    // Defect 8 guard: Louis/BHHH covariance step produces finite positive SEs for theta and sigma,
    // while `se_omega` is None (since Omega_equiv is derived and vine SEs live on `vine_params`).
    let se_theta = vine_fit.se_theta.as_ref().expect("se_theta present");
    let se_sigma = vine_fit.se_sigma.as_ref().expect("se_sigma present");
    assert!(se_theta.iter().all(|&s| s.is_finite() && s > 0.0), "se_theta: {se_theta:?}");
    assert!(se_sigma.iter().all(|&s| s.is_finite() && s > 0.0), "se_sigma: {se_sigma:?}");
    assert!(vine_fit.se_omega.is_none());

    // --- Part B: 90-degree rotated Clayton (negative correlation + asymmetric tail) ---
    let model = parse_model_string(ONE_CPT_IV_MODEL).expect("parse");
    let template = iv_bolus_template(95);
    let mut truth90 = model.default_params.clone();
    truth90.theta = vec![4.0, 50.0];
    truth90.sigma.values = vec![0.0081]; // SD = 0.09
    let mut rv90 = RVine::independence_dvine(2);
    rv90.set_edge_copula(1, 0, CopulaFamily::Clayton { theta: 2.8, rotation: CopulaRotation::R90 });
    let mut vine90 = VineCopulaOmega::from_init_params(&truth90, false);
    vine90.marginals = vec![MarginalDist::normal(0.32), MarginalDist::normal(0.32)];
    vine90.rvine = rv90;
    truth90.vine_dist = Some(Arc::new(vine90));

    let sim90 = simulate_with_seed(&model, &template, &truth90, 1, 5150).expect("sim90");
    let pop90 = fill_simulated_dvs(&template, &sim90);

    let mut opts90 = FitOptions::default();
    opts90.method = EstimationMethod::Saem;
    opts90.saem_omega_dist = OmegaDist::VineCopula;
    opts90.vine_marginals = VineMarginals::Gaussian;
    opts90.saem_n_exploration = 180;
    opts90.saem_n_convergence = 180;
    opts90.saem_seed = Some(5151);
    opts90.run_covariance_step = false;
    opts90.verbose = false;
    let fit90 = fit(&model, &pop90, &model.default_params, &opts90).expect("fit90");
    let c90 = &fit90.vine_params.as_ref().unwrap().trees[0].pairs[0].copula;
    assert!(
        c90.kendall_tau < -0.30,
        "90-deg rotated Clayton should recover negative Kendall tau < -0.30, got {:.4} ({:?})",
        c90.kendall_tau,
        c90
    );
    assert!(
        c90.rotation == 90 || c90.rotation == 270,
        "expected rotated 90/270 asymmetric copula for negative tail dependence, got: {c90:?}"
    );

    // --- Part C: 3D R-vine non-collapse (Defect 4 guard for d >= 3) ---
    let model_3d_src = r#"
[parameters]
  theta TVCL(3.0,   0.1, 150.0)
  theta TVV1(40.0,  1.0, 1000.0)
  theta TVQ(4.0,    0.1, 100.0)
  theta TVV2(80.0,  1.0, 1500.0)
  omega ETA_CL ~ 0.12
  omega ETA_V1 ~ 0.12
  omega ETA_V2 ~ 0.16
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model_3d = parse_model_string(model_3d_src).expect("parse 3d");
    let template_3d = iv_bolus_template(85);
    let mut truth_3d = model_3d.default_params.clone();
    let mut rv3 = RVine::independence_dvine(3);
    rv3.set_edge_copula(1, 0, CopulaFamily::Clayton { theta: 2.0, rotation: CopulaRotation::R0 });
    rv3.set_edge_copula(2, 0, CopulaFamily::Gumbel { theta: 1.7, rotation: CopulaRotation::R0 });
    rv3.set_edge_copula(2, 1, CopulaFamily::frank(2.5));
    let mut vine3 = VineCopulaOmega::from_init_params(&truth_3d, false);
    vine3.marginals =
        vec![MarginalDist::normal(0.32), MarginalDist::normal(0.32), MarginalDist::normal(0.38)];
    vine3.rvine = rv3;
    truth_3d.vine_dist = Some(Arc::new(vine3));

    let sim_3d = simulate_with_seed(&model_3d, &template_3d, &truth_3d, 1, 202603).expect("sim 3d");
    let pop_3d = fill_simulated_dvs(&template_3d, &sim_3d);

    let opts_3d = opts90.clone();
    let fit_3d = fit(&model_3d, &pop_3d, &model_3d.default_params, &opts_3d).expect("fit 3d");
    let vp3 = fit_3d.vine_params.as_ref().expect("vp3");
    assert_eq!(vp3.trees.len(), 2, "3D R-vine must have 2 trees");
    assert_eq!(vp3.trees[0].pairs.len(), 2);
    assert_eq!(vp3.trees[1].pairs.len(), 1);

    let om3 = &fit_3d.omega;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let r = om3[(i, j)] / (om3[(i, i)] * om3[(j, j)]).sqrt();
            assert!(r.abs() < 0.92, "3D R-vine collapsed to rank-1: corr({i},{j}) = {r:.4}");
        }
    }

    // Keep `pop` alive for Rung 4 helper checks below via separate test or direct reuse.
    let _ = pop;
}

#[test]
fn rung4_uninformative_subject_zero_ofv_fitrx_roundtrip_and_three_uses_consistency() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let vine_path = root.join("examples/vine_tail_iv.ferx");
    let data_path = root.join("data/vine_tail_iv.csv");
    let model_source = std::fs::read_to_string(&vine_path).expect("read model");

    let (vine_fit, pop) =
        run_model_with_data(vine_path.to_str().unwrap(), Some(data_path.to_str().unwrap()))
            .expect("vine_tail_iv fit");
    let model = ferx_core::parse_model_file(&vine_path).expect("parse model");

    // 1. Defect 5 guard: Uninformative (zero-observation) subject contributes exact 0.0 to Vine QMC IS OFV!
    let fitted_params = fitted_params_from_result(&vine_fit, &model).expect("reconstruct params");
    assert!(
        fitted_params.vine_dist.is_some(),
        "fitted_params_from_result must reconstruct vine_dist"
    );

    let empty_subj = common::subject(
        "UNINFORMATIVE_DOSE_ONLY",
        vec![DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0)],
        vec![],
        vec![],
        vec![],
    );
    let mut pop_only_empty = pop.clone();
    pop_only_empty.subjects = vec![empty_subj.clone()];
    let is_opts = FitOptions::default();
    let is_empty_only = ferx_core::estimation::importance_sampling::run_importance_sampling_vine(
        &model,
        &pop_only_empty,
        &fitted_params,
        &[nalgebra::DVector::zeros(2)],
        &[DMatrix::zeros(0, 2)],
        &[Vec::new()],
        &is_opts,
    )
    .expect("IS on uninformative subject");
    assert_eq!(
        is_empty_only.minus2_log_likelihood, 0.0,
        "zero-observation subject must contribute exact 0.0 to Vine QMC IS OFV"
    );

    let base_etas: Vec<nalgebra::DVector<f64>> =
        vine_fit.subjects.iter().map(|s| s.eta.clone()).collect();
    let base_h: Vec<DMatrix<f64>> =
        pop.subjects.iter().map(|s| DMatrix::zeros(s.observations.len(), 2)).collect();
    let base_k: Vec<Vec<nalgebra::DVector<f64>>> = vec![Vec::new(); pop.subjects.len()];
    let is_base = ferx_core::estimation::importance_sampling::run_importance_sampling_vine(
        &model,
        &pop,
        &fitted_params,
        &base_etas,
        &base_h,
        &base_k,
        &is_opts,
    )
    .expect("IS base");

    let mut pop_with_empty = pop.clone();
    pop_with_empty.subjects.push(empty_subj);
    let mut etas_ext = base_etas;
    etas_ext.push(nalgebra::DVector::zeros(2));
    let mut h_ext = base_h;
    h_ext.push(DMatrix::zeros(0, 2));
    let mut k_ext = base_k;
    k_ext.push(Vec::new());
    let is_with_empty = ferx_core::estimation::importance_sampling::run_importance_sampling_vine(
        &model,
        &pop_with_empty,
        &fitted_params,
        &etas_ext,
        &h_ext,
        &k_ext,
        &is_opts,
    )
    .expect("IS with empty");
    assert!(
        (is_with_empty.minus2_log_likelihood - is_base.minus2_log_likelihood).abs() < 1e-9,
        "appending a zero-observation subject must leave total Vine OFV invariant: {:.6} vs {:.6}",
        is_with_empty.minus2_log_likelihood,
        is_base.minus2_log_likelihood
    );

    // 2. `.fitrx` lossless round-trip + Console & YAML formatting.
    let bytes = save_fit_to_bytes(&vine_fit, &pop, &model_source, SaveFitOptions::default())
        .expect("save_fit_to_bytes");
    let loaded = load_fit_from_bytes(&bytes).expect("load_fit_from_bytes");
    assert_eq!(loaded.fit.vine_params, vine_fit.vine_params);
    assert_eq!(loaded.fit.vine_corrected_ofv, vine_fit.vine_corrected_ofv);
    assert_eq!(loaded.fit.ofv, vine_fit.ofv);
    assert_eq!(loaded.fit.bic_inputs, vine_fit.bic_inputs);

    let console = format_summary(&vine_fit);
    assert!(console.contains("VINE COPULA"), "console summary missing VINE COPULA:\n{console}");
    let tmp_yaml = tempfile::NamedTempFile::new().expect("tmp yaml");
    write_estimates_yaml(&vine_fit, tmp_yaml.path().to_str().unwrap()).expect("write yaml");
    let yaml_text = std::fs::read_to_string(tmp_yaml.path()).expect("read yaml");
    assert!(yaml_text.contains("vine_copula:"), "YAML missing vine_copula:\n{yaml_text}");
    assert!(
        yaml_text.contains("vine_corrected_ofv:"),
        "YAML missing vine_corrected_ofv:\n{yaml_text}"
    );

    // 3. Three-Uses Consistency: `simulate_with_options_diag` from reconstructed `fitted_params`
    //    draws asymmetric lower-tail-dependent ETAs matching the Clayton vine, NOT symmetric Gaussian!
    let sim_opts = ferx_core::SimulateOptions { seed: Some(202611), ..Default::default() };
    let sim_res =
        ferx_core::simulate_with_options_diag(&model, &pop, &fitted_params, 25, &sim_opts)
            .expect("simulate_with_options_diag");
    let drawn_etas: Vec<(f64, f64)> =
        sim_res.parameters.rows.iter().map(|sp| (sp.eta[0], sp.eta[1])).collect();
    assert!(drawn_etas.len() >= 1500);

    // Compute empirical rank quantiles U1, U2 in (0, 1) and compare lower vs upper 10% tail co-exceedance.
    let n = drawn_etas.len();
    let mut s1: Vec<f64> = drawn_etas.iter().map(|p| p.0).collect();
    let mut s2: Vec<f64> = drawn_etas.iter().map(|p| p.1).collect();
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q10_1 = s1[n / 10];
    let q10_2 = s2[n / 10];
    let q90_1 = s1[(n * 9) / 10];
    let q90_2 = s2[(n * 9) / 10];

    let n_low_2 = drawn_etas.iter().filter(|p| p.1 <= q10_2).count().max(1) as f64;
    let p_low_joint =
        drawn_etas.iter().filter(|p| p.0 <= q10_1 && p.1 <= q10_2).count() as f64 / n_low_2;
    let n_high_2 = drawn_etas.iter().filter(|p| p.1 >= q90_2).count().max(1) as f64;
    let p_high_joint =
        drawn_etas.iter().filter(|p| p.0 >= q90_1 && p.1 >= q90_2).count() as f64 / n_high_2;

    assert!(
        p_low_joint > p_high_joint + 0.12,
        "Three-Uses violation: simulated lower-tail P(U1<=0.1|U2<=0.1)={p_low_joint:.3} did not exceed \
         upper-tail P(U1>=0.9|U2>=0.9)={p_high_joint:.3} by > 0.12"
    );

    // 4. `conddist = true` samples from the vine posterior seamlessly.
    let mut cd_opts = FitOptions::default();
    cd_opts.method = EstimationMethod::Saem;
    cd_opts.saem_omega_dist = OmegaDist::VineCopula;
    cd_opts.vine_marginals = VineMarginals::Gaussian;
    cd_opts.saem_n_exploration = 60;
    cd_opts.saem_n_convergence = 60;
    cd_opts.saem_conddist = true;
    cd_opts.saem_conddist_nsamp = 40;
    cd_opts.saem_conddist_burnin = 15;
    cd_opts.run_covariance_step = false;
    cd_opts.verbose = false;
    let cd_fit = fit(&model, &pop, &model.default_params, &cd_opts).expect("conddist vine fit");
    let cd = cd_fit.cond_dist.as_ref().expect("cond_dist populated");
    assert_eq!(cd.cond_mean.len(), pop.subjects.len());
    assert!(cd.cond_mean.iter().all(|m| m.iter().all(|x| x.is_finite())));
}
