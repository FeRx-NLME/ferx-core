//! Tier-2 integration test for the SAEM conditional-distribution pass (#257).
//!
//! Exercises the public `fit()` boundary: a short SAEM run with
//! `saem_conddist = true` must populate `FitResult.cond_dist` with per-subject
//! conditional means/SDs of the right shape, and `keep_samples` must retain the
//! requested number of draws. This is a smoke test — it runs only a handful of
//! outer iterations and a small sampling budget, not to convergence.
//!
//! The dataset is fully synthetic (simulated from the model's own parameters);
//! no proprietary data is used.

use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population, Subject};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};
use std::collections::HashMap;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  1.0, 500.0)
  omega ETA_CL ~ 0.10
  omega ETA_V  ~ 0.10
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn template_population(n: usize) -> Population {
    let times = [0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0];
    let subjects: Vec<Subject> = (1..=n)
        .map(|i| Subject {
            id: format!("{i}"),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.to_vec(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; times.len()],
            obs_cmts: vec![1; times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; times.len()],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn simulate_into(model: &ferx_core::types::CompiledModel, template: &Population) -> Population {
    let truth = model.default_params.clone();
    let sim = simulate_with_seed(model, template, &truth, 1, 424242);
    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        let dv: Vec<f64> = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value().max(1e-6))
            .collect();
        subj.observations = dv;
    }
    pop
}

fn short_saem_opts() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 8;
    opts.saem_n_convergence = 8;
    opts.saem_omega_burnin = 4;
    opts.saem_seed = Some(11);
    opts.run_covariance_step = false;
    opts.verbose = false;
    // Conditional-distribution pass: small budget keeps the test fast.
    opts.saem_conddist = true;
    opts.saem_conddist_nsamp = 40;
    opts.saem_conddist_burnin = 10;
    opts
}

#[test]
fn saem_conddist_populates_per_subject_mean_and_sd() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let template = template_population(12);
    let population = simulate_into(&model, &template);

    let opts = short_saem_opts();
    let result = fit(&model, &population, &model.default_params, &opts).expect("SAEM fit succeeds");

    let cd = result
        .cond_dist
        .as_ref()
        .expect("cond_dist must be populated when saem_conddist = true");

    let n_subjects = population.subjects.len();
    let n_eta = result.eta_names.len();
    assert_eq!(cd.cond_mean.len(), n_subjects);
    assert_eq!(cd.cond_sd.len(), n_subjects);
    assert_eq!(cd.shrinkage.len(), n_eta);
    assert_eq!(cd.nsamp, opts.saem_conddist_nsamp);

    for i in 0..n_subjects {
        assert_eq!(cd.cond_mean[i].len(), n_eta);
        assert_eq!(cd.cond_sd[i].len(), n_eta);
        for j in 0..n_eta {
            assert!(
                cd.cond_mean[i][j].is_finite(),
                "subject {i} eta {j}: cond_mean not finite"
            );
            assert!(
                cd.cond_sd[i][j].is_finite() && cd.cond_sd[i][j] >= 0.0,
                "subject {i} eta {j}: cond_sd invalid ({})",
                cd.cond_sd[i][j]
            );
        }
    }

    // Samples are not retained by default.
    assert!(cd.samples.iter().all(|s| s.is_empty()));
}

#[test]
fn saem_conddist_disabled_by_default_is_none() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let template = template_population(8);
    let population = simulate_into(&model, &template);

    let mut opts = short_saem_opts();
    opts.saem_conddist = false; // explicit: the pass must not run

    let result = fit(&model, &population, &model.default_params, &opts).expect("SAEM fit succeeds");
    assert!(
        result.cond_dist.is_none(),
        "cond_dist must be None when saem_conddist = false"
    );
}

#[test]
fn saem_conddist_keep_samples_retains_draws() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let template = template_population(6);
    let population = simulate_into(&model, &template);

    let mut opts = short_saem_opts();
    opts.saem_conddist_keep_samples = true;

    let result = fit(&model, &population, &model.default_params, &opts).expect("SAEM fit succeeds");
    let cd = result.cond_dist.as_ref().expect("cond_dist present");

    let n_eta = result.eta_names.len();
    for i in 0..population.subjects.len() {
        assert_eq!(cd.samples[i].len(), opts.saem_conddist_nsamp);
        assert_eq!(cd.samples[i][0].len(), n_eta);
    }
}

/// #675: `.fitrx` bundles must round-trip `cond_dist` (cond_mean/cond_sd) via
/// `conddist.csv`, not silently drop it as `None` on load.
#[test]
fn saem_conddist_roundtrips_through_fitrx_bundle() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let template = template_population(10);
    let population = simulate_into(&model, &template);

    let opts = short_saem_opts();
    let result = fit(&model, &population, &model.default_params, &opts).expect("SAEM fit succeeds");
    assert!(
        result.cond_dist.is_some(),
        "fixture fit must have run conddist"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conddist_fit.fitrx");
    save_fit(
        &result,
        &population,
        MODEL,
        &path,
        SaveFitOptions::default(),
    )
    .unwrap();

    let loaded = load_fit(&path).unwrap();
    let cd = loaded
        .fit
        .cond_dist
        .expect("cond_dist must round-trip through .fitrx");
    let orig = result.cond_dist.as_ref().unwrap();

    assert_eq!(cd.cond_mean.len(), orig.cond_mean.len());
    for (a, b) in cd.cond_mean.iter().zip(orig.cond_mean.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "cond_mean {x} vs {y}");
        }
    }
    for (a, b) in cd.cond_sd.iter().zip(orig.cond_sd.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "cond_sd {x} vs {y}");
        }
    }

    assert!(
        loaded.manifest.entries.iter().any(|e| e == "conddist.csv"),
        "manifest must list conddist.csv"
    );
}

/// #675: a fit that never ran the conddist pass must not gain a `conddist.csv`
/// entry, and must still load back with `cond_dist: None`.
#[test]
fn saem_no_conddist_omits_fitrx_entry() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let template = template_population(6);
    let population = simulate_into(&model, &template);

    let mut opts = short_saem_opts();
    opts.saem_conddist = false;

    let result = fit(&model, &population, &model.default_params, &opts).expect("SAEM fit succeeds");
    assert!(result.cond_dist.is_none());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_conddist_fit.fitrx");
    save_fit(
        &result,
        &population,
        MODEL,
        &path,
        SaveFitOptions::default(),
    )
    .unwrap();

    let loaded = load_fit(&path).unwrap();
    assert!(loaded.fit.cond_dist.is_none());
    assert!(!loaded.manifest.entries.iter().any(|e| e == "conddist.csv"));
}
