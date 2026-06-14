//! Integration tests for propensity-score-matched simulation
//! (`simulate_with_options` with `propensity_match = true`).
//!
//! The matching *math* (Mahalanobis + optimal assignment) is unit-tested in
//! `src/propensity_match.rs`; these tests exercise the public-API wiring on a
//! real compiled model: shape, finiteness, the error path, reproducibility of
//! the unmatched path, and that the matched path is actually distinct.

use ferx_core::{
    parse_model_string, simulate_with_options, simulate_with_seed, DoseEvent, Population,
    SimulateOptions, Subject,
};

mod common;

const MODEL: &str = r#"
[parameters]
  theta TVCL(8.66, 0.1, 150.0)
  theta TVV(100.0, 1.0, 2000.0)
  omega ETA_CL ~ 0.25
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn template_population(n: usize) -> Population {
    let grid = [0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let subjects: Vec<Subject> = (1..=n)
        .map(|i| {
            // Phase-shift the (5-point) grid per subject so designs vary.
            let times: Vec<f64> = (0..5).map(|j| grid[(i + j) % grid.len()]).collect();
            let n_obs = times.len();
            common::subject(
                &format!("{i}"),
                vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
                times,
                vec![0.0; n_obs],
                vec![1; n_obs],
            )
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

/// Fill in observations by simulating once from the true params, so the
/// population looks like real observed data (needed for posthoc etas).
fn observed_population(model: &ferx_core::types::CompiledModel, n: usize) -> Population {
    let template = template_population(n);
    let sim = simulate_with_seed(model, &template, &model.default_params, 1, 42);
    let mut pop = template;
    for subj in pop.subjects.iter_mut() {
        subj.observations = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value().max(1e-6))
            .collect();
    }
    pop
}

#[test]
fn matched_simulation_has_expected_shape_and_finite_dvs() {
    let model = parse_model_string(MODEL).expect("model parses");
    let pop = observed_population(&model, 12);
    let n_sim = 4;
    let n_obs: usize = pop.subjects.iter().map(|s| s.obs_times.len()).sum();

    let opts = SimulateOptions {
        seed: Some(2024),
        propensity_match: true,
    };
    let rows = simulate_with_options(&model, &pop, &model.default_params, n_sim, &opts)
        .expect("matched simulation succeeds");

    assert_eq!(rows.len(), n_obs * n_sim, "one row per obs per replicate");
    assert!(rows
        .iter()
        .all(|r| r.outcome.continuous_value().is_finite()));
    // Replicate indices span 1..=n_sim.
    let mut sims: Vec<usize> = rows.iter().map(|r| r.sim).collect();
    sims.sort_unstable();
    sims.dedup();
    assert_eq!(sims, (1..=n_sim).collect::<Vec<_>>());
}

#[test]
fn matched_simulation_is_reproducible_and_distinct_from_unmatched() {
    let model = parse_model_string(MODEL).expect("model parses");
    let pop = observed_population(&model, 10);

    let matched = SimulateOptions {
        seed: Some(7),
        propensity_match: true,
    };
    let a = simulate_with_options(&model, &pop, &model.default_params, 3, &matched).unwrap();
    let b = simulate_with_options(&model, &pop, &model.default_params, 3, &matched).unwrap();
    let dv = |rs: &[ferx_core::SimulationResult]| {
        rs.iter()
            .map(|r| r.outcome.continuous_value())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        dv(&a),
        dv(&b),
        "matched path must be reproducible under a seed"
    );

    // The matched path takes a different branch (pool draw + reassignment), so
    // its output must differ from the unmatched path on the same seed.
    let unmatched = SimulateOptions {
        seed: Some(7),
        propensity_match: false,
    };
    let c = simulate_with_options(&model, &pop, &model.default_params, 3, &unmatched).unwrap();
    assert_ne!(dv(&a), dv(&c), "matched should differ from unmatched");
}

#[test]
fn unmatched_options_path_equals_simulate_with_seed() {
    let model = parse_model_string(MODEL).expect("model parses");
    let pop = observed_population(&model, 8);

    let opts = SimulateOptions {
        seed: Some(99),
        propensity_match: false,
    };
    let via_opts = simulate_with_options(&model, &pop, &model.default_params, 2, &opts).unwrap();
    let via_seed = simulate_with_seed(&model, &pop, &model.default_params, 2, 99);

    let dv = |rs: &[ferx_core::SimulationResult]| {
        rs.iter()
            .map(|r| r.outcome.continuous_value())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        dv(&via_opts),
        dv(&via_seed),
        "unmatched simulate_with_options must reproduce simulate_with_seed byte-for-byte"
    );
}

#[test]
fn matching_requires_observations() {
    let model = parse_model_string(MODEL).expect("model parses");
    // A subject with no observation records → posthoc eta undefined.
    let obsless = common::subject(
        "1",
        vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let pop = Population {
        subjects: vec![obsless],
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let opts = SimulateOptions {
        seed: Some(1),
        propensity_match: true,
    };
    let err = simulate_with_options(&model, &pop, &model.default_params, 1, &opts)
        .expect_err("matching without observations must error");
    assert!(
        err.contains("observations"),
        "error should mention the missing observations: {err}"
    );
}
