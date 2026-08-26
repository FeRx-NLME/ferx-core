//! `theta NAME[COL, ...]` binding (#1064).

use super::*;
use crate::parser::model_parser::parse_full_model;
use crate::types::{CompiledModel, DoseEvent, FitOptions, Population, Subject};

/// An MBMA-shaped model: an unstructured placebo effect per (STUDY, TIME) cell,
/// added to a typical value that also carries between-study variability.
fn mbma_model(contrast: &str) -> String {
    let modifier = if contrast.is_empty() {
        String::new()
    } else {
        format!(", contrast = {contrast}")
    };
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[STUDY, TIME{modifier}](0.0, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL) + PLACEBO
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#
    )
}

/// A factor with no random effect sharing its scale — the plain
/// unstructured-effect case, which takes global sum-to-zero.
fn no_eta_model() -> String {
    r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_V ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL + PLACEBO
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#
    .to_string()
}

/// One subject per study, `n_times` observations each, `STUDY` carried as a
/// subject-level covariate.
fn population(n_studies: usize, n_times: usize) -> Population {
    let subjects = (0..n_studies)
        .map(|s| {
            let mut covariates = HashMap::new();
            covariates.insert("STUDY".to_string(), (s + 1) as f64);
            Subject {
                id: format!("{}", s + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: (1..=n_times).map(|t| t as f64).collect(),
                obs_raw_times: Vec::new(),
                observations: vec![1.0; n_times],
                obs_cmts: vec![1; n_times],
                covariates,
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; n_times],
                occasions: Vec::new(),
                obs_l2: Vec::new(),
                dose_occasions: Vec::new(),
                fremtype: Vec::new(),
                obs_records: Vec::new(),
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["STUDY".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Bind `text` against `pop`, returning the re-parsed model.
fn bind(text: &str, pop: &mut Population) -> Result<CompiledModel, String> {
    let mut parsed = parse_full_model(text)?;
    crate::api::bind_theta_levels(&mut parsed, text, pop)?;
    Ok(parsed.model)
}

fn theta_names(model: &CompiledModel) -> Vec<String> {
    model.theta_names.clone()
}

#[test]
fn an_unbound_level_block_declares_no_thetas_and_refuses_to_fit() {
    let parsed = parse_full_model(&mbma_model("")).unwrap();
    assert_eq!(parsed.model.n_theta, 2, "only TVCL and TVV exist yet");
    assert_eq!(
        parsed.model.theta_blocks.unbound_level_blocks(),
        &["PLACEBO".to_string()]
    );
    let err = crate::api::fit(
        &parsed.model,
        &population(2, 2),
        &parsed.model.default_params,
        &FitOptions::default(),
    )
    .unwrap_err();
    assert!(
        err.contains("never bound to data"),
        "unexpected error: {err}"
    );
}

#[test]
fn binding_expands_to_one_theta_per_observed_combination() {
    let mut pop = population(3, 4);
    let model = bind(&no_eta_model(), &mut pop).unwrap();
    // 3 studies x 4 timepoints = 12 levels, global sum-to-zero drops one.
    assert_eq!(model.n_theta, 2 + 11);
    let names = theta_names(&model);
    assert_eq!(names[0], "TVCL");
    assert_eq!(names[1], "PLACEBO[STUDY=1,TIME=1]");
    assert_eq!(names[11], "PLACEBO[STUDY=3,TIME=3]");
    assert_eq!(
        names[12], "TVV",
        "the block is contiguous; later scalars follow it"
    );
    assert!(
        !names.contains(&"PLACEBO[STUDY=3,TIME=4]".to_string()),
        "the last level is the sum-to-zero contrast, not a parameter"
    );
}

#[test]
fn only_observed_combinations_become_levels() {
    // Study 1 is observed at t = 1, 2; study 2 only at t = 1. The unobserved
    // (2, 2) cell must not become an estimable parameter.
    let mut pop = population(2, 2);
    pop.subjects[1].obs_times.truncate(1);
    pop.subjects[1].observations.truncate(1);
    pop.subjects[1].obs_cmts.truncate(1);
    pop.subjects[1].cens.truncate(1);
    let model = bind(&no_eta_model(), &mut pop).unwrap();
    let names = theta_names(&model);
    assert!(!names.iter().any(|n| n == "PLACEBO[STUDY=2,TIME=2]"));
    // 3 observed levels, global sum-to-zero drops one.
    assert_eq!(model.n_theta, 2 + 2);
}

#[test]
fn sum_to_zero_levels_sum_to_exactly_zero() {
    let mut pop = population(2, 3);
    let model = bind(&no_eta_model(), &mut pop).unwrap();
    // 6 levels, 5 free θ.
    let mut theta = vec![2.0, 0.7, -0.3, 1.1, 0.25, -0.9, 10.0];
    assert_eq!(theta.len(), model.n_theta);
    let mut total = 0.0;
    for level in 1..=6usize {
        let mut covs = HashMap::new();
        covs.insert("__level_PLACEBO".to_string(), level as f64);
        let p = (model.pk_param_fn)(&theta, &[0.0], &covs, 0.0);
        total += p.values[0] - 2.0;
    }
    assert!(
        total.abs() < 1e-12,
        "sum-to-zero contrast must sum to 0, got {total}"
    );
    // And the dependent level really is minus the sum of the free ones.
    theta[1] = 5.0;
    let mut covs = HashMap::new();
    covs.insert("__level_PLACEBO".to_string(), 6.0);
    let p = (model.pk_param_fn)(&theta, &[0.0], &covs, 0.0);
    let free_sum: f64 = theta[1..6].iter().sum();
    assert!((p.values[0] - 2.0 + free_sum).abs() < 1e-12);
}

#[test]
fn an_eta_at_the_leading_grouping_selects_within_group_sum_to_zero() {
    // `[STUDY, TIME]` plus an η on the same additive scale is
    // over-parameterised globally: that study's η *is* the mean of its own
    // timepoint levels. The default must constrain within study.
    let mut pop = population(3, 4);
    let model = bind(&mbma_model(""), &mut pop).unwrap();
    // 12 levels, one dependent level per study → 9 free θ.
    assert_eq!(model.n_theta, 2 + 9);
    let names = theta_names(&model);
    for study in 1..=3 {
        assert!(
            !names
                .iter()
                .any(|n| n == &format!("PLACEBO[STUDY={study},TIME=4]")),
            "study {study}'s last level is its own contrast"
        );
    }
}

#[test]
fn within_group_sum_to_zero_sums_to_zero_inside_each_group() {
    let mut pop = population(2, 3);
    let model = bind(&mbma_model(""), &mut pop).unwrap();
    // 6 levels, 2 groups → 4 free θ.
    assert_eq!(model.n_theta, 2 + 4);
    let theta = vec![2.0, 0.4, -1.1, 0.9, 0.2, 10.0];
    let placebo = |level: usize| -> f64 {
        let mut covs = HashMap::new();
        covs.insert("__level_PLACEBO".to_string(), level as f64);
        (model.pk_param_fn)(&theta, &[0.0], &covs, 0.0).values[0] - 2.0
    };
    let study1: f64 = (1..=3).map(placebo).sum();
    let study2: f64 = (4..=6).map(placebo).sum();
    assert!(study1.abs() < 1e-12, "study 1 sums to {study1}");
    assert!(study2.abs() < 1e-12, "study 2 sums to {study2}");
}

#[test]
fn without_a_shared_eta_the_default_is_global_sum_to_zero() {
    let mut pop = population(2, 3);
    let model = bind(&no_eta_model(), &mut pop).unwrap();
    // A single global group → 5 free θ, not 4. Grouping within study here
    // would force both studies' mean placebo effects equal, which is a
    // modelling assumption, not a normalization.
    assert_eq!(model.n_theta, 2 + 5);
}

#[test]
fn global_sum_to_zero_against_a_nested_eta_is_rejected() {
    let mut pop = population(2, 3);
    let err = bind(&mbma_model("sum_to_zero"), &mut pop).unwrap_err();
    assert!(
        err.contains("is not identified") && err.contains("sum_to_zero_within"),
        "unexpected error: {err}"
    );
}

#[test]
fn reference_level_against_a_nested_eta_is_rejected() {
    let mut pop = population(2, 3);
    let err = bind(&mbma_model("ref"), &mut pop).unwrap_err();
    assert!(err.contains("is not identified"), "unexpected error: {err}");
}

#[test]
fn reference_level_pins_the_first_level_of_each_group_at_zero() {
    let mut pop = population(2, 3);
    let model = bind(
        &no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = ref]"),
        &mut pop,
    )
    .unwrap();
    assert_eq!(model.n_theta, 2 + 5);
    let theta = vec![2.0, 0.4, -1.1, 0.9, 0.2, 0.3, 10.0];
    let mut covs = HashMap::new();
    covs.insert("__level_PLACEBO".to_string(), 1.0);
    let p = (model.pk_param_fn)(&theta, &[0.0], &covs, 0.0);
    assert!(
        (p.values[0] - 2.0).abs() < 1e-12,
        "level 1 is the reference and contributes 0"
    );
    let names = theta_names(&model);
    assert!(!names.iter().any(|n| n == "PLACEBO[STUDY=1,TIME=1]"));
    assert_eq!(names[1], "PLACEBO[STUDY=1,TIME=2]");
}

#[test]
fn unconstrained_estimates_every_level() {
    let mut pop = population(2, 3);
    let model = bind(
        &no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = none]"),
        &mut pop,
    )
    .unwrap();
    assert_eq!(model.n_theta, 2 + 6);
}

#[test]
fn a_single_level_block_matches_a_plain_theta() {
    // Degenerate oracle: with one level and no constraint, the block is a
    // scalar θ and must behave exactly like one.
    let mut pop = population(1, 1);
    let model = bind(
        &no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = none]"),
        &mut pop,
    )
    .unwrap();
    assert_eq!(model.n_theta, 3);

    let plain = parse_full_model(&no_eta_model().replace(
        "theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)",
        "theta PLACEBO(0.0, -10.0, 10.0)",
    ))
    .unwrap()
    .model;

    let theta = vec![2.0, 0.375, 10.0];
    let mut covs = HashMap::new();
    covs.insert("__level_PLACEBO".to_string(), 1.0);
    let a = (model.pk_param_fn)(&theta, &[0.0], &covs, 0.0);
    let b = (plain.pk_param_fn)(&theta, &[0.0], &HashMap::new(), 0.0);
    assert_eq!(
        a.values[0], b.values[0],
        "a one-level block must be bit-identical to the scalar it degenerates to"
    );
}

#[test]
fn a_single_level_block_under_sum_to_zero_is_rejected() {
    let mut pop = population(1, 1);
    let err = bind(&no_eta_model(), &mut pop).unwrap_err();
    assert!(err.contains("single level"), "unexpected error: {err}");
}

#[test]
fn the_index_column_is_written_onto_every_subject() {
    let mut pop = population(2, 3);
    bind(&no_eta_model(), &mut pop).unwrap();
    for subject in &pop.subjects {
        assert!(subject.covariates.contains_key("__level_PLACEBO"));
        // The level moves with the timepoint, so per-observation snapshots
        // must exist and carry distinct indices.
        assert_eq!(subject.obs_covariates.len(), 3);
        let idx: Vec<f64> = subject
            .obs_covariates
            .iter()
            .map(|m| m["__level_PLACEBO"])
            .collect();
        assert_eq!(idx.len(), 3);
        assert!(idx[0] < idx[1] && idx[1] < idx[2]);
    }
    assert!(pop.covariate_names.iter().any(|n| n == "__level_PLACEBO"));
}

#[test]
fn a_subject_constant_index_does_not_engage_time_varying_machinery() {
    // `factor(STUDY)` alone is constant within a subject, so no per-event
    // snapshots are needed and the model keeps whatever fast path it had.
    let text = no_eta_model().replace("[STUDY, TIME]", "[STUDY]");
    let mut pop = population(3, 4);
    let model = bind(&text, &mut pop).unwrap();
    assert_eq!(model.n_theta, 2 + 2, "3 studies, global sum-to-zero");
    for subject in &pop.subjects {
        assert!(
            !subject.has_tv_covariates(),
            "a subject-constant index must not make the subject time-varying"
        );
        assert!(subject.covariates.contains_key("__level_PLACEBO"));
    }
}

#[test]
fn level_map_round_trips_the_labels() {
    let mut pop = population(2, 2);
    let model = bind(&no_eta_model(), &mut pop).unwrap();
    let map = crate::api::theta_level_map(&model);
    assert_eq!(
        map["PLACEBO"],
        vec![
            "STUDY=1,TIME=1".to_string(),
            "STUDY=1,TIME=2".to_string(),
            "STUDY=2,TIME=1".to_string(),
        ],
        "the dependent level carries no θ, so it is not in the map"
    );
}

#[test]
fn a_missing_level_column_is_a_loud_error() {
    let text = no_eta_model().replace("[STUDY, TIME]", "[REGION, TIME]");
    let mut pop = population(2, 2);
    let err = bind(&text, &mut pop).unwrap_err();
    assert!(
        err.contains("`REGION` is not in the data"),
        "unexpected error: {err}"
    );
}

#[test]
fn level_columns_are_registered_as_required_data_columns() {
    let parsed = parse_full_model(&no_eta_model()).unwrap();
    assert!(
        parsed
            .model
            .referenced_covariates
            .iter()
            .any(|c| c == "STUDY"),
        "STUDY must be read from the CSV: {:?}",
        parsed.model.referenced_covariates
    );
    assert!(
        !parsed
            .model
            .referenced_covariates
            .iter()
            .any(|c| c.eq_ignore_ascii_case("TIME")),
        "TIME is the record time, not a covariate"
    );
}

#[test]
fn a_level_block_rejects_an_unknown_modifier() {
    let text = no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, grouping = x]");
    let err = parse_full_model(&text).err().unwrap();
    assert!(err.contains("unknown modifier"), "got: {err}");
}

#[test]
fn a_level_block_rejects_an_unknown_contrast() {
    let text = no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = qq]");
    let err = parse_full_model(&text).err().unwrap();
    assert!(err.contains("unknown `contrast = qq`"), "got: {err}");
}

#[test]
fn empty_brackets_name_both_forms_in_the_error() {
    let text = no_eta_model().replace("[STUDY, TIME]", "[]");
    let err = parse_full_model(&text).err().unwrap();
    assert!(
        err.contains("PLACEBO[800]") && err.contains("PLACEBO[STUDY, TIME]"),
        "the error must show both bracket forms, got: {err}"
    );
}

#[test]
fn a_contrast_modifier_with_no_columns_is_rejected() {
    let text = no_eta_model().replace("[STUDY, TIME]", "[contrast = ref]");
    let err = parse_full_model(&text).err().unwrap();
    assert!(err.contains("name at least one data column"), "got: {err}");
}

#[test]
fn a_digit_only_bracket_is_a_level_count_not_a_column() {
    // The one place the two bracket forms could collide. A data column named
    // `800` is not referenceable anywhere else in the DSL either, so digits
    // always mean a count.
    let text = no_eta_model().replace("[STUDY, TIME]", "[3]");
    // The model reads `PLACEBO` bare, which only the column form supports — so
    // the error itself proves `[3]` parsed as a count of 3 levels rather than
    // as a column named `3`.
    let err = parse_full_model(&text).err().unwrap();
    assert!(
        err.contains("is a vector of 3 θ levels"),
        "digits must mean a level count, got: {err}"
    );

    // And with an explicit index it is an ordinary counted block: three levels,
    // no data binding needed.
    let indexed = text.replace("CL = TVCL + PLACEBO", "CL = TVCL + PLACEBO[PLA_IDX]");
    let parsed = parse_full_model(&indexed).unwrap();
    assert!(
        parsed.model.theta_blocks.unbound_level_blocks().is_empty(),
        "a counted block needs no data binding"
    );
    assert_eq!(parsed.model.n_theta, 2 + 3);
    assert_eq!(parsed.model.theta_names[1], "PLACEBO[1]");
}

#[test]
fn eta_sharing_is_detected_through_an_intermediate_assignment() {
    // The idiomatic two-line form must be recognised, not just the single-line
    // one — otherwise the identifiability check silently picks the wrong
    // convention for the exact model this feature serves.
    let text = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  BASE = TVCL + PLACEBO
  CL   = BASE * exp(ETA_CL)
  V    = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(text).unwrap();
    assert!(
        parsed.model.theta_blocks.level_blocks()[0].shares_scale_with_eta(),
        "taint must propagate through BASE"
    );
    let mut pop = population(2, 3);
    let model = bind(text, &mut pop).unwrap();
    assert_eq!(model.n_theta, 2 + 4, "within-study sum-to-zero");
}

// ── #1064: the loud half of the gather index policy ─────────────────────────
mod theta_gather_index_check {
    use crate::api::validation::check_theta_gather_indices;
    use crate::types::{DoseEvent, Population, Subject};
    use std::collections::HashMap;

    fn model(levels: usize) -> crate::types::CompiledModel {
        let content = format!(
            r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta PLACEBO[{levels}](0.5, -10.0, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL + PLACEBO[PLA_IDX]
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#
        );
        crate::parser::model_parser::parse_full_model(&content)
            .unwrap()
            .model
    }

    /// One subject whose `PLA_IDX` takes each of `indices` in turn.
    fn population(indices: &[f64]) -> Population {
        let n = indices.len();
        let mut covariates = HashMap::new();
        covariates.insert("PLA_IDX".to_string(), indices[0]);
        let subject = Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: (1..=n).map(|t| t as f64).collect(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n],
            obs_cmts: vec![1; n],
            covariates: covariates.clone(),
            dose_covariates: vec![covariates],
            obs_covariates: indices
                .iter()
                .map(|&i| HashMap::from([("PLA_IDX".to_string(), i)]))
                .collect(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: Vec::new(),
        };
        Population {
            subjects: vec![subject],
            covariate_names: vec!["PLA_IDX".to_string()],
            dv_column: "DV".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn valid_indices_produce_no_diagnostic() {
        let diags = check_theta_gather_indices(&model(4), &population(&[1.0, 2.0, 3.0, 4.0]));
        assert!(diags.is_empty(), "unexpected: {diags:?}");
    }

    #[test]
    fn a_zero_based_index_column_is_caught() {
        // The single most likely user error, and one the NaN guard alone would
        // report as "the fit diverged".
        let diags = check_theta_gather_indices(&model(4), &population(&[0.0, 1.0, 2.0, 3.0]));
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, "E_THETA_GATHER_INDEX_RANGE");
        assert!(diags[0].message.contains("1-based"), "{}", diags[0].message);
    }

    #[test]
    fn an_index_past_the_declared_level_count_is_caught() {
        let diags = check_theta_gather_indices(&model(3), &population(&[1.0, 2.0, 3.0, 4.0]));
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, "E_THETA_GATHER_INDEX_RANGE");
        assert!(
            diags[0].message.contains("has 3 levels"),
            "{}",
            diags[0].message
        );
    }

    #[test]
    fn a_non_integer_index_is_caught() {
        let diags = check_theta_gather_indices(&model(4), &population(&[1.0, 2.5, 3.0]));
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, "E_THETA_GATHER_INDEX_RANGE");
    }

    #[test]
    fn a_missing_index_column_is_caught() {
        let mut pop = population(&[1.0, 2.0]);
        for m in pop.subjects[0].obs_covariates.iter_mut() {
            m.remove("PLA_IDX");
        }
        pop.subjects[0].covariates.remove("PLA_IDX");
        pop.subjects[0].dose_covariates[0].remove("PLA_IDX");
        let diags = check_theta_gather_indices(&model(4), &pop);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, "E_THETA_GATHER_INDEX_MISSING");
    }

    #[test]
    fn one_diagnostic_per_subject_not_one_per_row() {
        // A mis-coded index column is wrong on every row; 100k copies of the
        // same finding help nobody.
        let diags = check_theta_gather_indices(&model(2), &population(&[7.0; 50]));
        assert_eq!(diags.len(), 1);
    }

    #[test]
    fn a_model_with_no_blocks_short_circuits() {
        let plain = crate::parser::model_parser::parse_full_model(
            r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#,
        )
        .unwrap()
        .model;
        assert!(plain.theta_blocks.is_empty());
        assert!(check_theta_gather_indices(&plain, &population(&[1.0])).is_empty());
    }
}
