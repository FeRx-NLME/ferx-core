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
                reset_covariates: Vec::new(),
                cens: vec![0; n_times],
                occasions: Vec::new(),
                obs_l2: Vec::new(),
                dose_occasions: Vec::new(),
                reset_occasions: Vec::new(),
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
        parsed.model.theta_blocks().unbound_level_blocks(),
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
fn the_simulate_paths_report_an_unbound_level_block_and_name_the_binder() {
    // `E_THETA_LEVELS_UNBOUND` is what every `simulate()` meets through
    // `check_simulation_data` (#1384). Both sides of the gate in one test: the
    // unbound model reports it, the same model bound reports nothing.
    let parsed = parse_full_model(&no_eta_model()).unwrap();
    let diags = crate::api::validation::check_simulation_data(&parsed.model, &population(2, 2));
    let unbound: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "E_THETA_LEVELS_UNBOUND")
        .collect();
    assert_eq!(unbound.len(), 1, "{diags:?}");
    let d = unbound[0];
    assert_eq!(d.block.as_deref(), Some("parameters"));
    // One assertion per sentence of the message and the suggestion.
    assert!(
        d.message
            .contains("`theta PLACEBO[...]` was never bound to data, so it has no levels"),
        "{}",
        d.message
    );
    assert!(
        d.message.contains("every value gathered from it is NaN"),
        "{}",
        d.message
    );
    let s = d.suggestion.as_deref().expect("a suggestion");
    assert!(
        s.contains("`bind_theta_levels(&mut parsed, &model_text, &mut population)`")
            && s.contains("(`read_population_for_simulation`) before simulating"),
        "the public binder, and when to call it: {s}"
    );
    assert!(
        s.contains(
            "`run_model_simulate` (`ferx --simulate`) binds for you, against the [simulation] design"
        ),
        "the one simulating entry point that binds on its own: {s}"
    );
    assert!(
        s.contains("`theta PLACEBO[N](...)` and index it with your own column"),
        "the counted-form alternative: {s}"
    );

    let mut pop = population(2, 2);
    let bound = bind(&no_eta_model(), &mut pop).unwrap();
    let diags = crate::api::validation::check_simulation_data(&bound, &pop);
    assert!(
        !diags.iter().any(|d| d.code == "E_THETA_LEVELS_UNBOUND"),
        "a bound model must not report it: {diags:?}"
    );
}

#[test]
fn predict_reports_an_unbound_level_block_and_names_predicts_binder() {
    // #1644. Before, `predict_diag` ran only the covariate check, which named the
    // synthesized `__level_PLACEBO` as "not found in data" — a column the user
    // never wrote. Both sides of the entry-point gate in one test: predict's
    // refusal names `predict`, simulate's suggestion is unchanged.
    let parsed = parse_full_model(&no_eta_model()).unwrap();
    let pop = population(2, 2);
    let err = crate::api::predict_diag(&parsed.model, &pop, &parsed.model.default_params)
        .err()
        .expect("an unbound model must not predict");
    // One assertion per sentence (and clause) of the message and the suggestion.
    assert!(
        err.starts_with(
            "`theta PLACEBO[...]` was never bound to data, so it has no levels and every \
             value gathered from it is NaN. "
        ),
        "the shared E_THETA_LEVELS_UNBOUND message: {err}"
    );
    // That the advice is also *true* — following it reproduces the fit's predictions on
    // new data — is `from_fit::following_the_unbound_predict_refusal_gives_the_fits_predictions`.
    assert!(
        err.contains(
            "With a fit's θ, call `bind_theta_levels_from_fit(&mut parsed, &model_text, &mut \
             population, &fitted_levels)` on the population you pass to `predict`"
        ),
        "the from-fit binder, on the population predict reads: {err}"
    );
    assert!(
        err.contains(
            ", where `fitted_levels` is the `parsed.bindings.levels` kept from binding the fit \
             data"
        ),
        "where the fit's bindings come from: {err}"
    );
    assert!(
        err.contains(", and predict with the model it re-parses into `parsed`."),
        "the bound model is a re-parse, not the one in hand: {err}"
    );
    assert!(
        err.contains(
            "`bind_theta_levels` on that population fits only a θ laid out for the levels it \
             discovers, such as the model's own `default_params`."
        ),
        "when the other binder is the right one: {err}"
    );
    assert!(
        err.ends_with(
            "Or declare the block explicitly as `theta PLACEBO[N](...)` and index it with \
             your own column."
        ),
        "the counted-form alternative: {err}"
    );
    for absent in [
        "__level_",
        "not found in data",
        "before simulating",
        "read_population_for_simulation",
        "run_model_simulate",
    ] {
        assert!(
            !err.contains(absent),
            "`{absent}` in predict's refusal: {err}"
        );
    }

    // The simulate side of the gate: the suggestion still says when to bind for a
    // simulation (its sentences are pinned in
    // `the_simulate_paths_report_an_unbound_level_block_and_name_the_binder`),
    // and the simulate `Err` is still the bare message.
    let diags = crate::api::validation::check_simulation_data(&parsed.model, &pop);
    let sim = diags
        .iter()
        .find(|d| d.code == "E_THETA_LEVELS_UNBOUND")
        .expect("simulate reports it");
    assert!(
        sim.suggestion
            .as_deref()
            .unwrap()
            .contains("before simulating"),
        "{sim:?}"
    );
    let sim_err = crate::api::simulate_with_options_diag(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        1,
        &Default::default(),
    )
    .err()
    .expect("an unbound model must not simulate");
    assert_eq!(sim_err, sim.message);

    // Bound, the same model predicts.
    let mut pop = population(2, 2);
    let mut bound = parse_full_model(&no_eta_model()).unwrap();
    crate::api::bind_theta_levels(&mut bound, &no_eta_model(), &mut pop).unwrap();
    crate::api::predict_diag(&bound.model, &pop, &bound.model.default_params)
        .expect("a bound model predicts");
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
fn a_leading_group_shared_by_subjects_uses_global_sum_to_zero() {
    // Eta is subject-scoped. When two subjects share a STUDY value, neither
    // subject eta can represent that study's common mean, so STUDY does not
    // identify the eta grouping and the within-study contrast is invalid.
    let mut pop = population(4, 3);
    for (subject, study) in pop.subjects.iter_mut().zip([1.0, 1.0, 2.0, 2.0]) {
        subject.covariates.insert("STUDY".to_string(), study);
    }
    let model = bind(&mbma_model(""), &mut pop).unwrap();
    // Six observed cells, one global dependent level -> five free coefficients.
    assert_eq!(model.n_theta, 2 + 5);

    let mut explicit_pop = population(4, 3);
    for (subject, study) in explicit_pop.subjects.iter_mut().zip([1.0, 1.0, 2.0, 2.0]) {
        subject.covariates.insert("STUDY".to_string(), study);
    }
    bind(&mbma_model("sum_to_zero"), &mut explicit_pop)
        .expect("global contrast is identified when STUDY is shared");
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
    // ...and nowhere else. `covariate_names` is the data's columns as users and
    // downstream tools read them (`FitResult::covariate_names`, GAM, the FREM
    // CSV header); the synthesized column is engine plumbing (#1644).
    assert_eq!(pop.covariate_names, vec!["STUDY".to_string()]);
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
            "STUDY=2,TIME=2".to_string(),
        ],
        "the map includes the dependent level even though it carries no free θ"
    );
}

#[test]
fn constrained_blocks_reject_unrepresentable_broadcast_initializers() {
    let mut pop = population(2, 2);
    let nonzero = no_eta_model().replace(
        "PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)",
        "PLACEBO[STUDY, TIME](0.5, -10.0, 10.0)",
    );
    let err = bind(&nonzero, &mut pop).unwrap_err();
    assert!(err.contains("requires init = 0"), "unexpected error: {err}");

    let mut pop = population(2, 2);
    let excludes_zero = no_eta_model().replace(
        "PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)",
        "PLACEBO[STUDY, TIME](0.0, 0.1, 10.0)",
    );
    let err = bind(&excludes_zero, &mut pop).unwrap_err();
    assert!(
        err.contains("bounds must include 0"),
        "unexpected error: {err}"
    );
}

#[test]
fn unconstrained_blocks_keep_broadcast_init_and_fix_semantics() {
    let text = no_eta_model()
        .replace("[STUDY, TIME]", "[STUDY, TIME, contrast = none]")
        .replace(
            "PLACEBO[STUDY, TIME, contrast = none](0.0, -10.0, 10.0)",
            "PLACEBO[STUDY, TIME, contrast = none](0.5, -10.0, 10.0, FIX)",
        );
    let mut pop = population(2, 2);
    let model = bind(&text, &mut pop).unwrap();
    assert_eq!(&model.default_params.theta[1..5], &[0.5; 4]);
    assert!(model.default_params.theta_fixed[1..5].iter().all(|f| *f));
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
        parsed
            .model
            .theta_blocks()
            .unbound_level_blocks()
            .is_empty(),
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
        parsed.model.theta_blocks().level_blocks()[0].shares_scale_with_eta(),
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
            reset_covariates: Vec::new(),
            cens: vec![0; n],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
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
        assert!(plain.theta_blocks().is_empty());
        assert!(check_theta_gather_indices(&plain, &population(&[1.0])).is_empty());
    }
}

// ── #1064: binder helpers, branch by branch ────────────────────────────────
mod binder_helpers {
    use super::super::{cmp_levels, contrast_token, format_level_value, free_count, Level};
    use super::*;

    #[test]
    fn level_values_render_without_a_trailing_point_zero() {
        // Study ids and visit numbers are integers almost always, and
        // `PLACEBO[STUDY=7,TIME=4]` reads better than `STUDY=7.0`.
        assert_eq!(format_level_value(7.0), "7");
        assert_eq!(format_level_value(-3.0), "-3");
        assert_eq!(format_level_value(0.0), "0");
        // A genuinely fractional level keeps its value rather than truncating
        // two distinct cells onto one label.
        assert_eq!(format_level_value(0.5), "0.5");
        assert_eq!(format_level_value(-1.25), "-1.25");
        // Past the `abs() < 1e15` guard the integer path would saturate through
        // `as i64` and collapse every large level onto `i64::MAX`. The fallback
        // must keep the value instead.
        assert_ne!(format_level_value(1e300), i64::MAX.to_string());
        assert_eq!(format_level_value(1e300).parse::<f64>().unwrap(), 1e300);
    }

    fn level(values: &[f64]) -> Level {
        Level {
            values: values.to_vec(),
        }
    }

    #[test]
    fn levels_sort_lexicographically_by_their_tuple() {
        use std::cmp::Ordering;
        assert_eq!(
            cmp_levels(&level(&[1.0, 4.0]), &level(&[1.0, 12.0])),
            Ordering::Less
        );
        assert_eq!(
            cmp_levels(&level(&[2.0, 1.0]), &level(&[1.0, 99.0])),
            Ordering::Greater
        );
        assert_eq!(
            cmp_levels(&level(&[1.0, 4.0]), &level(&[1.0, 4.0])),
            Ordering::Equal
        );
    }

    #[test]
    fn level_ordering_stays_total_under_nan() {
        // `total_cmp`, not `partial_cmp`: the binding has to be reproducible
        // run to run even if a column carries a NaN, rather than depending on
        // the sort's comparison order.
        use std::cmp::Ordering;
        let nan = level(&[f64::NAN]);
        let one = level(&[1.0]);
        assert_eq!(cmp_levels(&nan, &nan), Ordering::Equal);
        assert_ne!(cmp_levels(&nan, &one), Ordering::Equal);
        assert_eq!(cmp_levels(&nan, &one), cmp_levels(&nan, &one));
    }

    /// The free-θ count does not depend on `assign_groups` handing group ids out
    /// contiguously (#1654 review). Ids `[0, 1, 0]` are two groups of sizes 2
    /// and 1, so one free θ under a group-wise contrast; `none` frees all three.
    ///
    /// Mutation — drop the sort before `dedup`: the non-adjacent repeat of id 0
    /// survives, three ids are counted, and the count drops to 0, a false
    /// "estimates nothing".
    #[test]
    fn the_free_count_does_not_assume_contiguous_group_ids() {
        use crate::parser::model_parser::LevelContrast;
        for c in [
            LevelContrast::SumToZero,
            LevelContrast::SumToZeroWithin,
            LevelContrast::Ref,
        ] {
            assert_eq!(free_count(&[0, 1, 0], c), 1, "{c:?}");
            assert_eq!(free_count(&[0, 0, 1], c), 1, "{c:?} contiguous control");
        }
        assert_eq!(free_count(&[0, 1, 0], LevelContrast::Unconstrained), 3);
    }

    #[test]
    fn every_contrast_has_a_diagnostic_token() {
        assert_eq!(contrast_token(LevelContrast::Auto), "auto");
        assert_eq!(contrast_token(LevelContrast::SumToZero), "sum_to_zero");
        assert_eq!(
            contrast_token(LevelContrast::SumToZeroWithin),
            "sum_to_zero_within"
        );
        assert_eq!(contrast_token(LevelContrast::Ref), "ref");
        assert_eq!(contrast_token(LevelContrast::Unconstrained), "none");
    }

    #[test]
    fn a_non_finite_level_column_is_rejected() {
        let mut pop = population(2, 2);
        pop.subjects[0]
            .covariates
            .insert("STUDY".to_string(), f64::NAN);
        let err = bind(&no_eta_model(), &mut pop).unwrap_err();
        assert!(err.contains("non-finite"), "unexpected error: {err}");
    }

    #[test]
    fn a_population_with_no_observations_is_rejected() {
        let mut pop = population(2, 2);
        for s in pop.subjects.iter_mut() {
            s.obs_times.clear();
            s.observations.clear();
            s.obs_cmts.clear();
            s.cens.clear();
        }
        let err = bind(&no_eta_model(), &mut pop).unwrap_err();
        assert!(
            err.contains("no observation rows"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn dose_rows_carry_the_level_of_the_most_recent_observation() {
        // A level is a property of an *observation*, so dose and EVID=2 rows
        // take the preceding observation's level (the first one when they
        // precede every observation). This only matters for a model whose
        // gathered parameter also drives the dosing dynamics.
        let mut pop = population(1, 3);
        let subject = &mut pop.subjects[0];
        // Observations at t = 1, 2, 3; doses at t = 0 (before all) and t = 2.5.
        subject
            .doses
            .push(crate::types::DoseEvent::new(2.5, 50.0, 1, 0.0, false, 0.0));
        bind(&no_eta_model().replace("[STUDY, TIME]", "[TIME]"), &mut pop).unwrap();

        let subject = &pop.subjects[0];
        let obs: Vec<f64> = subject
            .obs_covariates
            .iter()
            .map(|m| m["__level_PLACEBO"])
            .collect();
        assert_eq!(obs, vec![1.0, 2.0, 3.0], "one level per timepoint");
        let doses: Vec<f64> = subject
            .dose_covariates
            .iter()
            .map(|m| m["__level_PLACEBO"])
            .collect();
        assert_eq!(
            doses,
            vec![1.0, 2.0],
            "t=0 precedes every observation so takes the first level; \
             t=2.5 takes the t=2 level"
        );
    }

    #[test]
    fn level_map_carries_the_dependent_level_the_theta_vector_omits() {
        // The dependent contrast level has no θ, so it is absent from
        // `theta_names` — but the *binding* still has to name it, or a caller
        // cannot tell an 800-level block from a 799-level one.
        let mut pop = population(2, 2);
        let model = bind(&no_eta_model(), &mut pop).unwrap();
        let map = crate::api::theta_level_map(&model);
        let labels = &map["PLACEBO"];
        assert_eq!(labels.len(), 4, "all four observed levels: {labels:?}");
        assert!(
            labels.contains(&"STUDY=2,TIME=2".to_string()),
            "the sum-to-zero level must still be named: {labels:?}"
        );
        let named: Vec<&String> = model
            .theta_names
            .iter()
            .filter(|n| n.starts_with("PLACEBO["))
            .collect();
        assert_eq!(named.len(), 3, "but only three carry a θ: {named:?}");
    }
}

/// `bind_theta_levels_from_fit` (#1614): a simulation design bound against the fit's
/// level bindings. Analytic one-compartment IV throughout; no gradient path.
mod from_fit {
    use super::*;
    use crate::api::{bind_theta_levels_from_fit, simulate_with_seed};
    use crate::parser::model_parser::{LevelBindings, LevelContrast};
    use crate::types::{ModelParameters, ParsedModel};

    /// Bind `text` against the fit data `pop` and return the parsed model, whose
    /// `bindings.levels` is what a caller keeps after the fit.
    fn bind_fit(text: &str, pop: &mut Population) -> ParsedModel {
        let mut parsed = parse_full_model(text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, text, pop).expect("bind the fit data");
        parsed
    }

    fn bind_design(
        text: &str,
        design: &mut Population,
        fitted: &LevelBindings,
    ) -> Result<ParsedModel, String> {
        let mut parsed = parse_full_model(text).unwrap();
        bind_theta_levels_from_fit(&mut parsed, text, design, fitted)?;
        Ok(parsed)
    }

    /// [`population`] with every subject replicated `reps` times under fresh ids —
    /// more subjects per study than the fit had, the shape of a VPC design.
    fn replicated(n_studies: usize, n_times: usize, reps: usize) -> Population {
        let mut pop = population(n_studies, n_times);
        let base = pop.subjects.clone();
        pop.subjects = (0..reps)
            .flat_map(|r| {
                base.iter().map(move |s| Subject {
                    id: format!("{}-{r}", s.id),
                    ..s.clone()
                })
            })
            .collect();
        pop
    }

    /// The fit's θ (its own initial estimates) as parameters for `model`.
    fn fit_theta_params(model: &CompiledModel, fit_theta: &[f64]) -> ModelParameters {
        let mut params = model.default_params.clone();
        params.theta = fit_theta.to_vec();
        params
    }

    /// T5. The contrast travels from the fit. One subject per study makes the
    /// fit's `STUDY` identify subjects, so the η-sharing block resolves to
    /// `sum_to_zero_within` (6 θ). A design with two subjects per study would
    /// resolve to global sum-to-zero (7 θ) if the contrast were re-resolved on it
    /// — measured on `a1cd1b5b`: every ipred 0.
    ///
    /// Mutation — call `resolve_contrast` / `assign_groups` on the design instead
    /// of taking `fitted` verbatim: `n_theta` goes to 7 and this dies on it.
    #[test]
    fn the_contrast_travels_from_the_fit_not_the_design() {
        let text = mbma_model("");
        let mut fit_pop = population(2, 3);
        let fit = bind_fit(&text, &mut fit_pop);
        assert_eq!(fit.model.n_theta, 6, "TVCL, TVV, 2 studies x (3 - 1)");
        assert_eq!(
            fit.bindings.levels["PLACEBO"].contrast,
            LevelContrast::SumToZeroWithin
        );

        // The control: the design re-bound on its own does re-resolve.
        let mut own = replicated(2, 3, 2);
        let rebound = bind(&text, &mut own).unwrap();
        assert_eq!(rebound.n_theta, 7, "the design alone resolves globally");

        let mut design = replicated(2, 3, 2);
        let parsed = bind_design(&text, &mut design, &fit.bindings.levels).expect("bind");
        assert_eq!(parsed.model.n_theta, 6);
        assert_eq!(parsed.model.theta_names, fit.model.theta_names);
        let b = &parsed.bindings.levels["PLACEBO"];
        assert_eq!(b.contrast, LevelContrast::SumToZeroWithin);
        assert_eq!(b.groups, fit.bindings.levels["PLACEBO"].groups);

        let mut theta = fit.model.default_params.theta.clone();
        for (i, t) in theta.iter_mut().enumerate().skip(1).take(4) {
            *t = 0.05 * i as f64;
        }
        let params = fit_theta_params(&parsed.model, &theta);
        let rows = simulate_with_seed(&parsed.model, &design, &params, 1, 3).expect("simulate");
        assert_eq!(rows.len(), 12);
        assert!(
            rows.iter().all(|r| r.ipred.is_finite() && r.ipred > 0.0),
            "{:?}",
            rows.iter().map(|r| r.ipred).collect::<Vec<_>>()
        );
    }

    /// Every synthesized index snapshot on `pop`, flattened in a fixed order.
    fn index_snapshots(pop: &Population) -> Vec<(String, &'static str, f64)> {
        let col = "__level_PLACEBO";
        let mut out = Vec::new();
        for s in &pop.subjects {
            out.push((s.id.clone(), "subject", s.covariates[col]));
            for (kind, maps) in [
                ("obs", &s.obs_covariates),
                ("dose", &s.dose_covariates),
                ("pk_only", &s.pk_only_covariates),
                ("reset", &s.reset_covariates),
            ] {
                for m in maps.iter() {
                    out.push((s.id.clone(), kind, m[col]));
                }
            }
        }
        out
    }

    /// T8, the other side of the gate: when the design *is* the fit data, the
    /// from-fit binder is `bind_theta_levels` — every index snapshot, the θ layout,
    /// and the simulated rows bit for bit. Both conventions: global sum-to-zero and
    /// `sum_to_zero_within`. The fixture carries an EVID=2 row and a reset row, so
    /// the shared writer's LOCF arms are on the compared path.
    ///
    /// Mutation — the from-fit table written 0-based (`(level, i)` for
    /// `(level, i + 1)`) dies here on the first index snapshot. A change inside the
    /// shared writer moves both sides alike and is the Tier-2 `PLA_IDX` tests' job.
    #[test]
    fn on_the_fit_data_it_is_bind_theta_levels_bit_for_bit() {
        for text in [no_eta_model(), mbma_model("")] {
            let mut data = population(2, 3);
            data.subjects[0].pk_only_times = vec![2.5];
            data.subjects[0].reset_times = vec![3.5];
            let mut fit_pop = data.clone();
            let fit = bind_fit(&text, &mut fit_pop);
            let mut design = data.clone();
            let parsed = bind_design(&text, &mut design, &fit.bindings.levels).expect("bind");

            assert_eq!(parsed.model.n_theta, fit.model.n_theta);
            assert_eq!(parsed.model.theta_names, fit.model.theta_names);
            let (a, b) = (index_snapshots(&fit_pop), index_snapshots(&design));
            assert!(a.iter().any(|(_, k, _)| *k == "pk_only"));
            assert!(a.iter().any(|(_, k, _)| *k == "reset"));
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.0, y.0);
                assert_eq!(x.1, y.1);
                assert_eq!(x.2.to_bits(), y.2.to_bits(), "{} {}", x.0, x.1);
            }
            assert_eq!(fit_pop.covariate_names, design.covariate_names);

            let mut theta = fit.model.default_params.theta.clone();
            for (i, t) in theta.iter_mut().enumerate().skip(1) {
                if fit.model.theta_names[i].starts_with("PLACEBO[") {
                    *t = 0.03 * i as f64;
                }
            }
            let pa = fit_theta_params(&fit.model, &theta);
            let pb = fit_theta_params(&parsed.model, &theta);
            let ra = simulate_with_seed(&fit.model, &fit_pop, &pa, 2, 21).expect("fit side");
            let rb = simulate_with_seed(&parsed.model, &design, &pb, 2, 21).expect("design side");
            assert_eq!(ra.len(), 12);
            assert_eq!(ra.len(), rb.len());
            for (x, y) in ra.iter().zip(&rb) {
                assert!(x.ipred.is_finite() && x.ipred > 0.0, "{}", x.ipred);
                assert_eq!(x.ipred.to_bits(), y.ipred.to_bits());
            }
        }
    }

    /// The fit's bindings must describe this model's blocks: one it lacks is
    /// refused naming the block, one it has extra is refused naming that one, and
    /// a binding whose groups are not parallel to its labels is refused. A model
    /// with no level block and empty bindings is a no-op.
    ///
    /// Mutations — drop the extra-block check and the second arm goes `Ok`; drop
    /// the length check and the third binds a malformed layout.
    #[test]
    fn bindings_that_do_not_match_the_model_are_refused_by_name() {
        let text = no_eta_model();
        let mut fit_pop = population(2, 2);
        let fit = bind_fit(&text, &mut fit_pop);

        let err = bind_design(&text, &mut population(2, 2), &LevelBindings::new())
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("theta PLACEBO[STUDY, TIME]")
                && err.contains("the fit's level bindings carry no `PLACEBO`")
                && err.contains("was the model edited since the fit?"),
            "{err}"
        );

        let mut extra = fit.bindings.levels.clone();
        extra.insert("OTHER".to_string(), extra["PLACEBO"].clone());
        let err = bind_design(&text, &mut population(2, 2), &extra)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("carry the block(s) `OTHER`, which this model does not declare")
                && err.contains("belong to a different model"),
            "{err}"
        );
        // A model with no level block at all still refuses bindings it cannot use.
        let plain = no_eta_model()
            .replace("theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)", "")
            .replace(" + PLACEBO", "");
        let err = bind_design(&plain, &mut population(2, 2), &fit.bindings.levels)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("`PLACEBO`, which this model does not declare"),
            "{err}"
        );
        let mut untouched = population(2, 2);
        let before = untouched.covariate_names.clone();
        bind_design(&plain, &mut untouched, &LevelBindings::new()).expect("no-op");
        assert_eq!(untouched.covariate_names, before);

        let mut skewed = fit.bindings.levels.clone();
        skewed.get_mut("PLACEBO").unwrap().groups.pop();
        let err = bind_design(&text, &mut population(2, 2), &skewed)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("has 4 labels but 3 groups; they must be parallel"),
            "{err}"
        );
    }

    /// The TIME-grid sentence is conditional on the block being keyed on `TIME`;
    /// both sides in one test so a gate stuck on either branch dies. Both
    /// refusals leave the design untouched — no index column is written unless
    /// every block binds.
    ///
    /// Mutations — drop the TIME sentence and the first arm dies; emit it
    /// unconditionally and the second does; write the index before checking for
    /// unseen labels and the untouched-population assertions die.
    #[test]
    fn the_time_grid_sentence_appears_only_for_a_time_keyed_block() {
        // A denser grid than the fit's: TIME 3 and 4 were never fitted.
        let text = no_eta_model();
        let fit = bind_fit(&text, &mut population(2, 2));
        let mut design = population(2, 4);
        let err = bind_design(&text, &mut design, &fit.bindings.levels)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("the design has 4 level(s) the fit estimated no theta for")
                && err.contains(
                    "`STUDY=1,TIME=3`, `STUDY=1,TIME=4`, `STUDY=2,TIME=3`, `STUDY=2,TIME=4`"
                ),
            "{err}"
        );
        assert!(
            err.contains(
                "`TIME` is a level column of this block, so the design can only be simulated \
                 at the fit's observation times"
            ),
            "{err}"
        );
        assert!(!design
            .covariate_names
            .iter()
            .any(|c| c == "__level_PLACEBO"));
        assert!(design.subjects[0].obs_covariates.is_empty());

        // A block keyed on STUDY alone: a new study is refused, with no TIME claim.
        let by_study = no_eta_model().replace("[STUDY, TIME]", "[STUDY]");
        let fit = bind_fit(&by_study, &mut population(2, 2));
        let mut design = population(3, 2);
        let err = bind_design(&by_study, &mut design, &fit.bindings.levels)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("theta PLACEBO[STUDY]: the design has 1 level(s)")
                && err.contains("`STUDY=3`"),
            "{err}"
        );
        assert!(
            !err.contains("TIME"),
            "no TIME-grid claim for a STUDY block: {err}"
        );
        // The advice names actions, not a Rust function (#1623).
        assert!(
            err.ends_with(
                " Either simulate only the fit's levels, or simulate the design without the \
                 fit's theta, from a theta vector for the design's own levels (the model's \
                 initial estimates, for example)."
            ),
            "{err}"
        );
        assert!(!err.contains("bind_theta_levels"), "{err}");
        assert!(!design
            .covariate_names
            .iter()
            .any(|c| c == "__level_PLACEBO"));
    }

    /// The θ-length gate's level-block hint, both sides of its gate in one test:
    /// a bound level-block model names the block and the fit's level bindings; the
    /// same model without its block does not. Neither names a Rust function: the R
    /// wrapper reaches this message too (#1623).
    ///
    /// Mutations — drop the hint and the first arm dies; emit it unconditionally
    /// and the second does; put a binder's name back and the negative assertion dies.
    #[test]
    fn the_theta_length_gate_names_the_fit_bindings_only_for_a_level_block() {
        let text = no_eta_model();
        let mut pop = population(2, 2);
        let model = bind(&text, &mut pop).unwrap();
        let mut params = model.default_params.clone();
        params.theta.pop();
        let err = simulate_with_seed(&model, &pop, &params, 1, 1).unwrap_err();
        assert!(
            err.contains("the supplied theta has 4 values but this model has 5"),
            "{err}"
        );
        assert!(
            err.contains("declares the theta level block(s) `PLACEBO`"),
            "{err}"
        );
        assert!(
            err.contains("whose theta count is set by the data the model was bound against."),
            "why the count moves: {err}"
        );
        assert!(
            err.contains(
                "A fit's theta fits only a design bound against that fit's level bindings, \
                 which give the design the fit's theta layout"
            ),
            "what a fit's theta needs: {err}"
        );
        assert!(
            !err.contains("bind_theta_levels"),
            "no Rust function in a message a wrapper reaches: {err}"
        );

        let plain = no_eta_model()
            .replace("theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)", "")
            .replace(" + PLACEBO", "");
        let model = parse_full_model(&plain).unwrap().model;
        let mut params = model.default_params.clone();
        params.theta.pop();
        let err = simulate_with_seed(&model, &population(2, 2), &params, 1, 1).unwrap_err();
        assert!(
            err.contains("the supplied theta has 1 values but this model has 2"),
            "{err}"
        );
        assert!(!err.contains("level block"), "{err}");
        assert!(!err.contains("level bindings"), "{err}");
    }

    /// T9. The unseen-level refusal's second action works, not only its wording:
    /// on a design the refusal rejects, binding the design on its own levels and
    /// simulating from the model's initial estimates succeeds. From Rust the action
    /// is `bind_theta_levels` + `default_params`, as the `bind_theta_levels_from_fit`
    /// rustdoc says.
    ///
    /// Mutations — re-parse in `bind_theta_levels` but keep the unbound model's
    /// `default_params` (the initial estimates no longer fit the design's layout),
    /// or drop the rebound model: the simulation is refused and this dies.
    #[test]
    fn the_refusals_second_action_simulates_the_design_on_its_own_levels() {
        let text = no_eta_model();
        let fit = bind_fit(&text, &mut population(2, 2));
        let mut design = population(3, 4);
        let err = bind_design(&text, &mut design, &fit.bindings.levels)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.contains("the design has 8 level(s) the fit estimated no theta for")
                && err.contains("simulate the design without the fit's theta"),
            "the refusal under test offers the action: {err}"
        );

        let mut parsed = parse_full_model(&text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, &text, &mut design)
            .expect("the design binds on its own levels");
        let model = &parsed.model;
        assert_eq!(model.n_theta, 13, "TVCL, TVV, 12 levels - 1 (sum_to_zero)");
        assert_eq!(model.default_params.theta.len(), 13);
        let rows = simulate_with_seed(model, &design, &model.default_params, 2, 7)
            .expect("the design simulates from the model's initial estimates");
        assert_eq!(rows.len(), 24, "2 sims x 3 subjects x 4 observations");
        assert!(
            rows.iter().all(|r| r.ipred.is_finite()),
            "every ipred finite: {:?}",
            rows.iter().map(|r| r.ipred).collect::<Vec<_>>()
        );
    }

    /// #1644 review round 1, row 1: the unbound-predict refusal's advice must be *true*,
    /// not only present. A caller predicting new data (here: two of the fit's three
    /// studies) with a fit's θ follows the binder the refusal names and must get the
    /// fit's own predictions for those subjects, bit for bit.
    ///
    /// The first wording named `bind_theta_levels` on the predict population. That
    /// re-discovers the levels (5 θ here, against the fit's 7), and `predict_diag` has no
    /// θ-length guard (#1615), so the fit's θ came back `Ok` read at the wrong positions —
    /// measured on this fixture at `74078e0c`: subject 3 at t = 1 predicted 2.455468 against
    /// the fit's 7.557837, subject 2 at t = 2 0.182376 against 6.250023.
    #[test]
    fn following_the_unbound_predict_refusal_gives_the_fits_predictions() {
        let text = no_eta_model();
        let mut fit_pop = population(3, 2);
        let fit = bind_fit(&text, &mut fit_pop);
        assert_eq!(
            fit.model.n_theta, 7,
            "TVCL, TVV, 3 x 2 levels - 1 (sum_to_zero)"
        );
        // Distinct, non-cancelling level effects, so a misplaced read moves a prediction.
        let mut theta = fit.model.default_params.theta.clone();
        for (i, t) in theta.iter_mut().enumerate().skip(1).take(5) {
            *t = 0.3 * i as f64 - 0.7;
        }
        let rows = |model: &CompiledModel, pop: &Population| -> Vec<(String, u64, u64)> {
            crate::api::predict_diag(model, pop, &fit_theta_params(model, &theta))
                .expect("predict")
                .results
                .into_iter()
                .filter(|r| r.id != "1")
                .map(|r| (r.id, r.time.to_bits(), r.pred.to_bits()))
                .collect()
        };
        let want = rows(&fit.model, &fit_pop);
        assert_eq!(want.len(), 4, "studies 2 and 3, two times each");

        let mut new_data = population(3, 2);
        new_data.subjects.remove(0);
        let unbound = parse_full_model(&text).unwrap();
        let err =
            crate::api::predict_diag(&unbound.model, &new_data, &unbound.model.default_params)
                .err()
                .expect("an unbound model must not predict");
        assert!(
            err.contains("`bind_theta_levels_from_fit("),
            "the refusal names the binder this test follows: {err}"
        );

        let mut followed = new_data.clone();
        let parsed = bind_design(&text, &mut followed, &fit.bindings.levels).expect("bind");
        assert_eq!(rows(&parsed.model, &followed), want);

        // Why the advice cannot be `bind_theta_levels`: on this population it lays θ out
        // for the levels it discovers, which is not the fit's layout.
        let mut own = new_data.clone();
        assert_eq!(bind(&text, &mut own).unwrap().n_theta, 5);
    }
}

/// `theta_level_values` (#1623): every level's value, free and dependent, read through
/// the engine's own gather. The fixtures' θ are distinct and non-cancelling
/// (`θₖ = 0.1k + 0.01k²`), and every expected value is written as its closed form
/// over those θ, never by calling the gather.
mod level_values {
    use super::*;
    use crate::api::{theta_level_map, theta_level_values, ThetaLevelValue};

    fn distinct_theta(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.1 * k as f64 + 0.01 * (k * k) as f64)
            .collect()
    }

    /// The values of `PLACEBO`, after checking the labels are `theta_level_map`'s
    /// and every free level's θ name is `PLACEBO[label]` at its index.
    fn placebo_values(model: &CompiledModel, theta: &[f64]) -> Vec<ThetaLevelValue> {
        let map = theta_level_values(model, theta).expect("a bound model and its own theta");
        assert_eq!(map.len(), 1, "only PLACEBO is a level block: {map:?}");
        let values = map["PLACEBO"].clone();
        let labels: Vec<String> = values.iter().map(|v| v.label.clone()).collect();
        assert_eq!(labels, theta_level_map(model)["PLACEBO"]);
        for v in &values {
            if let Some(k) = v.theta_index {
                assert_eq!(model.theta_names[k], format!("PLACEBO[{}]", v.label));
            }
        }
        values
    }

    /// `(value, theta_index)` per level, for a compact comparison against the
    /// closed form. Values compare with `==` on purpose: the gather sums the group
    /// left to right, exactly as the closed forms below are written.
    fn pairs(values: &[ThetaLevelValue]) -> Vec<(f64, Option<usize>)> {
        values.iter().map(|v| (v.value, v.theta_index)).collect()
    }

    /// T1. Each contrast on a 2 study × 3 time design: the dependent levels are the
    /// negated sum of their group's free θ (or 0 for `ref`), the free levels their
    /// own θ, and `theta_index` is set exactly for the free ones.
    ///
    /// Mutations — flip `NegSum`'s sign in `eval_gather`; take `theta_index` from the
    /// label's position instead of the level's rule; evaluate level `i` instead of
    /// `i + 1`; read the k-th label as `theta[k]` (wrong under `ref` and the
    /// within-group contrast, where positions shift); skip dependent levels. Each
    /// changes one of the vectors below.
    #[test]
    fn every_contrast_reports_the_closed_form_of_each_level() {
        // sum_to_zero (no η shares the study scale): one group, L6 dependent.
        let mut pop = population(2, 3);
        let model = bind(&no_eta_model(), &mut pop).unwrap();
        let t = distinct_theta(model.n_theta);
        assert_eq!(model.n_theta, 7);
        assert_eq!(
            pairs(&placebo_values(&model, &t)),
            vec![
                (t[1], Some(1)),
                (t[2], Some(2)),
                (t[3], Some(3)),
                (t[4], Some(4)),
                (t[5], Some(5)),
                (-((((t[1] + t[2]) + t[3]) + t[4]) + t[5]), None),
            ],
            "sum_to_zero"
        );

        // sum_to_zero_within (η on the study): one dependent level per study.
        let mut pop = population(2, 3);
        let model = bind(&mbma_model(""), &mut pop).unwrap();
        let t = distinct_theta(model.n_theta);
        assert_eq!(model.n_theta, 6);
        let within = pairs(&placebo_values(&model, &t));
        assert_eq!(
            within,
            vec![
                (t[1], Some(1)),
                (t[2], Some(2)),
                (-(t[1] + t[2]), None),
                (t[3], Some(3)),
                (t[4], Some(4)),
                (-(t[3] + t[4]), None),
            ],
            "sum_to_zero_within"
        );
        // The planning probe's numbers, so a fixture change cannot quietly move them.
        assert!((within[2].0 + 0.35).abs() < 1e-12 && (within[5].0 + 0.95).abs() < 1e-12);

        // ref: the first level is the reference, pinned at 0.
        let mut pop = population(2, 3);
        let model = bind(
            &no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = ref]"),
            &mut pop,
        )
        .unwrap();
        let t = distinct_theta(model.n_theta);
        assert_eq!(
            pairs(&placebo_values(&model, &t)),
            vec![
                (0.0, None),
                (t[1], Some(1)),
                (t[2], Some(2)),
                (t[3], Some(3)),
                (t[4], Some(4)),
                (t[5], Some(5)),
            ],
            "ref"
        );

        // none: every level is its own θ.
        let mut pop = population(2, 3);
        let model = bind(
            &no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = none]"),
            &mut pop,
        )
        .unwrap();
        let t = distinct_theta(model.n_theta);
        assert_eq!(
            pairs(&placebo_values(&model, &t)),
            (1..=6).map(|k| (t[k], Some(k))).collect::<Vec<_>>(),
            "none"
        );
    }

    /// T2. The degenerate groups: a study observed at one time only, under the
    /// within-group contrast, reads 0; a one-level `none` block reads its θ.
    ///
    /// Mutations — as T1's sign / skip mutations; and a `NegSum(a, a)` that reads
    /// `theta[a]` instead of the empty sum: the singleton then reports the next θ
    /// (here `TVV`), not 0.
    #[test]
    fn a_group_with_no_free_theta_reports_zero() {
        let mut pop = population(2, 3);
        let s2 = &mut pop.subjects[1];
        s2.obs_times.truncate(1);
        s2.observations.truncate(1);
        s2.obs_cmts.truncate(1);
        s2.cens.truncate(1);
        let model = bind(&mbma_model(""), &mut pop).unwrap();
        let t = distinct_theta(model.n_theta);
        assert_eq!(model.n_theta, 4, "TVCL, TVV, study 1's two free levels");
        let values = placebo_values(&model, &t);
        assert_eq!(values[3].label, "STUDY=2,TIME=1");
        assert_eq!(
            pairs(&values),
            vec![
                (t[1], Some(1)),
                (t[2], Some(2)),
                (-(t[1] + t[2]), None),
                (0.0, None),
            ]
        );

        // A one-level block under `ref` / `sum_to_zero_within` used to bind here
        // with no free θ; it is now refused (#1624, `contrast_refusals`).

        let mut pop = population(1, 1);
        let model = bind(
            &no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = none]"),
            &mut pop,
        )
        .unwrap();
        let t = distinct_theta(model.n_theta);
        assert_eq!(pairs(&placebo_values(&model, &t)), vec![(t[1], Some(1))]);
    }

    /// T3. A θ of the wrong length is refused, naming both counts; the model's own
    /// length is accepted.
    ///
    /// Mutation — delete the length check: the short θ returns `Ok` (its last free
    /// level reading `NaN`) and the first assertion dies.
    #[test]
    fn a_theta_of_the_wrong_length_is_refused() {
        let mut pop = population(2, 3);
        let model = bind(&no_eta_model(), &mut pop).unwrap();
        let t = distinct_theta(model.n_theta);
        let err = theta_level_values(&model, &t[..6]).unwrap_err();
        assert!(
            err.contains("the supplied theta has 6 values but this model has 7"),
            "{err}"
        );
        let mut long = t.clone();
        long.push(1.0);
        let err = theta_level_values(&model, &long).unwrap_err();
        assert!(
            err.contains("the supplied theta has 8 values but this model has 7"),
            "{err}"
        );
        assert!(theta_level_values(&model, &t).is_ok());
    }

    /// T4. Only bound level blocks are reported: an unbound one and a counted
    /// `theta NAME[N]` block give an empty map. T1 is the bound side of this gate.
    ///
    /// Mutations — drop the unbound-block filter (an unbound block still has its
    /// gather, so the map becomes `Ok({"PLACEBO": []})`), or walk every gather
    /// instead of the level blocks (the counted block appears): either way these die.
    #[test]
    fn unbound_and_counted_blocks_are_not_reported() {
        let unbound = parse_full_model(&no_eta_model()).unwrap().model;
        let t = distinct_theta(unbound.n_theta);
        assert_eq!(theta_level_values(&unbound, &t), Ok(HashMap::new()));

        let counted = parse_full_model(
            &no_eta_model()
                .replace("PLACEBO[STUDY, TIME]", "PLACEBO[4]")
                .replace("TVCL + PLACEBO", "TVCL + PLACEBO[STUDY]"),
        )
        .unwrap()
        .model;
        assert_eq!(counted.n_theta, 6, "TVCL, TVV and four counted levels");
        let t = distinct_theta(counted.n_theta);
        assert_eq!(theta_level_values(&counted, &t), Ok(HashMap::new()));
    }

    /// T5. The compact rule counts **free** coefficients, on a real binding: a
    /// 21-level `sum_to_zero` block (20 free θ) is compacted, a 20-level one (19
    /// free) is not, and a 20-level `none` block (20 free) is.
    ///
    /// Mutations — `>=` to `>`, or the threshold 20 to 21: the 21-level case goes
    /// empty and this dies.
    #[test]
    fn the_compact_rule_counts_free_coefficients_not_levels() {
        use crate::io::output::{compact_theta_blocks, THETA_BLOCK_COMPACT_MIN};
        assert_eq!(THETA_BLOCK_COMPACT_MIN, 20);

        let mut pop = population(1, 21);
        let model = bind(&no_eta_model(), &mut pop).unwrap();
        assert_eq!(
            compact_theta_blocks(&model.theta_names),
            vec![("PLACEBO".to_string(), 1..21)],
            "21 levels, 20 free coefficients"
        );

        let mut pop = population(1, 20);
        let model = bind(&no_eta_model(), &mut pop).unwrap();
        assert!(
            compact_theta_blocks(&model.theta_names).is_empty(),
            "20 levels, 19 free coefficients"
        );

        let mut pop = population(1, 20);
        let model = bind(
            &no_eta_model().replace("[STUDY, TIME]", "[STUDY, TIME, contrast = none]"),
            &mut pop,
        )
        .unwrap();
        assert_eq!(
            compact_theta_blocks(&model.theta_names),
            vec![("PLACEBO".to_string(), 1..21)],
            "20 levels, 20 free coefficients"
        );
    }

    /// T8. The report is what the model applies, read off the model's own
    /// `pk_param_fn` rather than off any closed form: with `CL = TVCL + PLACEBO`
    /// and η = 0, each level's `CL` is `TVCL + value`, bit for bit, under every
    /// contrast that has a dependent level. T1 pins the values against closed
    /// forms, which agree with the gather by construction while the report calls
    /// it; this test does not depend on how the report computes them.
    ///
    /// Mutation — re-derive the value locally (`Free(k) => theta[k]`,
    /// `NegSum(a, b) => -theta[a..b].sum()`) and flip `NegSum`'s sign in
    /// `eval_gather`: T1–T5 stay green, and this dies on the first dependent level.
    #[test]
    fn every_reported_value_is_what_the_model_applies_at_that_level() {
        for contrast in ["sum_to_zero", "ref", "sum_to_zero_within"] {
            let mut pop = population(2, 3);
            let text = no_eta_model().replace(
                "[STUDY, TIME]",
                &format!("[STUDY, TIME, contrast = {contrast}]"),
            );
            let model = bind(&text, &mut pop).unwrap();
            let mut t = distinct_theta(model.n_theta);
            t[0] = 2.0; // TVCL, so CL does not collapse onto the level's value
            let values = placebo_values(&model, &t);
            assert_eq!(values.len(), 6, "{contrast}: 2 studies x 3 times");
            assert!(
                values.iter().any(|v| v.theta_index.is_none()),
                "{contrast}: has a dependent level, so the gather's NegSum arm is read"
            );
            let eta = vec![0.0; model.n_eta];
            for (i, v) in values.iter().enumerate() {
                let mut covs = HashMap::new();
                covs.insert("STUDY".to_string(), if i < 3 { 1.0 } else { 2.0 });
                covs.insert("__level_PLACEBO".to_string(), (i + 1) as f64);
                let cl = (model.pk_param_fn)(&t, &eta, &covs, 0.0).values[crate::types::PK_IDX_CL];
                assert_eq!(
                    cl.to_bits(),
                    (t[0] + v.value).to_bits(),
                    "{contrast}, level {} (`{}`): the model applies CL = {cl}, the report \
                     says TVCL + {}",
                    i + 1,
                    v.label,
                    v.value
                );
            }
        }
    }
}

// ── #1642: `contrast = auto` sees the readout, and an η through a variable ──
//
// No NONMEM spelling exists for an automatic contrast choice, so the oracles are
// the twin (auto ≡ explicit `sum_to_zero_within`, bit for bit), the closed-form
// count `L − G` (18 − 3 = 15) against `L − 1` (17), and the exact group sums.
mod readout_share {
    use super::*;
    use crate::api::{theta_level_values, ThetaLevelValue};
    use crate::parser::model_parser::LevelContrast;
    use crate::types::ParsedModel;

    /// `n_studies × per_study` subjects on the time grid `times`; `STUDY` is
    /// the subject's block of `per_study`.
    pub(super) fn cf_pop(n_studies: usize, per_study: usize, times: &[f64]) -> Population {
        let mut pop = population(n_studies * per_study, times.len());
        for (k, s) in pop.subjects.iter_mut().enumerate() {
            s.covariates
                .insert("STUDY".into(), (k / per_study + 1) as f64);
            s.obs_times = times.to_vec();
            s.observations = vec![1.0; times.len()];
            s.obs_cmts = vec![1; times.len()];
            s.cens = vec![0; times.len()];
        }
        pop
    }

    /// A compartment-free placebo/Emax model: `ip` is the
    /// `[individual_parameters]` body, `y` the readout.
    pub(super) fn cf_model(contrast: &str, columns: &str, ip: &str, y: &str) -> String {
        let modifier = if contrast.is_empty() {
            String::new()
        } else {
            format!(", contrast = {contrast}")
        };
        format!(
            r#"
[parameters]
  theta TVE0(1.5, -10.0, 10.0)
  theta PLACEBO[{columns}{modifier}](0.0, -10.0, 10.0)
  theta TVEMAX(3.0, 0.1, 20.0)
  theta TVET50(1.5, 0.1, 20.0)
  omega ETA_E0 ~ 0.1
  sigma ADD ~ 0.1
[individual_parameters]
{ip}
[structural_model]
  y = {y}
[error_model]
  DV ~ additive(ADD)
"#
        )
    }

    pub(super) const T6: [f64; 6] = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
    pub(super) const EMAXY: &str = "EMAX * TIME / (TIME + ET50)";
    pub(super) const BASE: &str = "  EMAX = TVEMAX\n  ET50 = TVET50\n";

    /// H2: the η sits on `E0`, the block is read only in the readout.
    pub(super) fn h2(contrast: &str) -> String {
        cf_model(
            contrast,
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + ETA_E0"),
            &format!("E0 + PLACEBO + {EMAXY}"),
        )
    }

    /// H7 (control): the η sits on a parameter `y` never reads.
    fn h7(contrast: &str) -> String {
        cf_model(
            contrast,
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0\n  Z = TVE0 * exp(ETA_E0)"),
            &format!("E0 + PLACEBO + {EMAXY}"),
        )
    }

    pub(super) fn bind_parsed(text: &str, pop: &mut Population) -> ParsedModel {
        let mut parsed = parse_full_model(text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, text, pop).expect("bind");
        parsed
    }

    /// The resolved contrast and the block's free-θ count on 3 studies × `T6`,
    /// `per_study` subjects each.
    fn layout(text: &str, per_study: usize) -> (LevelContrast, usize) {
        let mut pop = cf_pop(3, per_study, &T6);
        let parsed = bind_parsed(text, &mut pop);
        let free = parsed
            .model
            .theta_names
            .iter()
            .filter(|n| n.starts_with("PLACEBO["))
            .count();
        (parsed.bindings.levels["PLACEBO"].contrast, free)
    }

    const WITHIN: (LevelContrast, usize) = (LevelContrast::SumToZeroWithin, 15);
    const GLOBAL: (LevelContrast, usize) = (LevelContrast::SumToZero, 17);

    /// T1, the twin. On H2, `auto` must be explicit `sum_to_zero_within` bit
    /// for bit: names, binding, and every level's value at a distinct θ. In the
    /// same test, H7's auto must *differ* from its explicit within, so the twin
    /// straddles the gate and cannot become a tautology.
    ///
    /// Mutation — pass an empty readout to the predicate (the pre-#1642 code):
    /// H2's auto binds 17 free θ against within's 15, and the names differ.
    #[test]
    fn auto_on_a_readout_block_is_bit_identical_to_explicit_within() {
        let mut pa = cf_pop(3, 1, &T6);
        let auto = bind_parsed(&h2(""), &mut pa);
        let mut pw = cf_pop(3, 1, &T6);
        let within = bind_parsed(&h2("sum_to_zero_within"), &mut pw);

        assert_eq!(auto.model.n_theta, 3 + 15, "18 levels − 3 studies");
        assert_eq!(auto.model.theta_names, within.model.theta_names);
        let (a, w) = (
            &auto.bindings.levels["PLACEBO"],
            &within.bindings.levels["PLACEBO"],
        );
        assert_eq!(a.labels, w.labels);
        assert_eq!(a.groups, w.groups);
        assert_eq!(a.contrast, w.contrast);
        assert_eq!(a.contrast, LevelContrast::SumToZeroWithin);

        let theta: Vec<f64> = (0..auto.model.n_theta)
            .map(|k| 0.1 * k as f64 + 0.013 * (k * k) as f64)
            .collect();
        let va = theta_level_values(&auto.model, &theta).unwrap();
        let vw = theta_level_values(&within.model, &theta).unwrap();
        let bits = |v: &ThetaLevelValue| (v.label.clone(), v.value.to_bits(), v.theta_index);
        assert_eq!(
            va["PLACEBO"].iter().map(bits).collect::<Vec<_>>(),
            vw["PLACEBO"].iter().map(bits).collect::<Vec<_>>()
        );
        // The exact group sums: each study's six levels sum to 0.
        for g in 0..3 {
            let sum: f64 = va["PLACEBO"][6 * g..6 * g + 6]
                .iter()
                .map(|v| v.value)
                .sum();
            assert!(sum.abs() < 1e-12, "study {} sums to {sum}", g + 1);
        }

        // The straddle: H7's η never reaches `y`, so auto stays global and
        // differs from its explicit within.
        assert_eq!(layout(&h7(""), 1), GLOBAL);
        assert_eq!(layout(&h7("sum_to_zero_within"), 1), WITHIN);
    }

    /// T2. The η reaches the block's expression through a variable, inside
    /// `[individual_parameters]` alone (H3), next to the single-line form (H1).
    ///
    /// Mutation — the one-colour taint (track "reads the block" only): H3 → 17.
    #[test]
    fn eta_reaches_the_block_through_a_variable() {
        let h1 = cf_model(
            "",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + PLACEBO + ETA_E0"),
            &format!("E0 + {EMAXY}"),
        );
        let h3 = cf_model(
            "",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + ETA_E0\n  E1 = E0 + PLACEBO"),
            &format!("E1 + {EMAXY}"),
        );
        assert_eq!(layout(&h1, 1), WITHIN, "H1");
        assert_eq!(layout(&h3, 1), WITHIN, "H3");
    }

    pub(super) fn scaling_model(ip: &str) -> String {
        format!(
            r#"
[parameters]
  theta TVE0(1.5, -10.0, 10.0)
  theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)
  theta TVEMAX(3.0, 0.1, 20.0)
  theta TVET50(1.5, 0.1, 20.0)
  omega ETA_E0 ~ 0.1
  sigma ADD ~ 0.1
[individual_parameters]
{ip}
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = central / V + E0 + PLACEBO
[error_model]
  DV ~ additive(ADD)
"#
        )
    }

    /// T3. The readout spellings that share a scale (H4 bare η in `y`, H5
    /// multiplicative, H6 η only on `EMAX`, S1 η on `V` read by a PK readout)
    /// against the two that do not (S2 η reaching `y` only through the state,
    /// H7 η `y` never reads) — both sides of the gate in one test.
    ///
    /// Mutations — η taint ignores `Variable` (H6, S1 red); the readout taints
    /// the state `central` (S2 red); return `Some` unconditionally (H7 red).
    #[test]
    fn readout_spellings_share_a_scale() {
        let cases = [
            (
                "H4",
                cf_model(
                    "",
                    "STUDY, TIME",
                    &format!("{BASE}  E0 = TVE0"),
                    &format!("E0 + ETA_E0 + PLACEBO + {EMAXY}"),
                ),
                WITHIN,
            ),
            (
                "H5",
                cf_model(
                    "",
                    "STUDY, TIME",
                    &format!("{BASE}  E0 = TVE0 * exp(ETA_E0)"),
                    &format!("E0 * exp(PLACEBO) + {EMAXY}"),
                ),
                WITHIN,
            ),
            (
                "H6",
                cf_model(
                    "",
                    "STUDY, TIME",
                    "  EMAX = TVEMAX + ETA_E0\n  ET50 = TVET50\n  E0 = TVE0",
                    &format!("E0 + PLACEBO + {EMAXY}"),
                ),
                WITHIN,
            ),
            (
                "S1",
                scaling_model("  CL = TVEMAX\n  V = TVET50 * exp(ETA_E0)\n  E0 = TVE0"),
                WITHIN,
            ),
            (
                "S2",
                scaling_model("  CL = TVEMAX * exp(ETA_E0)\n  V = TVET50\n  E0 = TVE0"),
                GLOBAL,
            ),
            ("H7", h7(""), GLOBAL),
        ];
        for (tag, text, want) in cases {
            assert_eq!(layout(&text, 1), want, "{tag}");
        }
    }

    /// T4. A named intermediate carries the block into `y`: a `[scaling]`
    /// intermediate, and the compartment-free `[structural_model]` one.
    ///
    /// Mutation — skip `inline_scaling_intermediates` in `readout_y_exprs`:
    /// `BASE` / `EFF` are then unknown names and both bind 17.
    #[test]
    fn a_named_intermediate_carries_the_block_into_y() {
        let scaling = scaling_model("  CL = TVEMAX\n  V = TVET50\n  E0 = TVE0 + ETA_E0").replace(
            "  y = central / V + E0 + PLACEBO",
            "  BASE = E0 + PLACEBO\n  y = central / V + BASE",
        );
        assert_eq!(layout(&scaling, 1), WITHIN, "[scaling] intermediate");
        let cf = cf_model(
            "",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + ETA_E0"),
            &format!("EFF + {EMAXY}"),
        )
        .replace("  y = EFF", "  EFF = E0 + PLACEBO\n  y = EFF");
        assert_eq!(layout(&cf, 1), WITHIN, "compartment-free intermediate");
    }

    /// T5. Four subjects per study: `STUDY` no longer identifies a subject, so
    /// no subject's η can carry a study's mean and both spellings stay global.
    ///
    /// Mutation — drop `&& nested` from `Auto`: both go to 15.
    #[test]
    fn four_subjects_per_study_stay_global() {
        let h1 = cf_model(
            "",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + PLACEBO + ETA_E0"),
            &format!("E0 + {EMAXY}"),
        );
        assert_eq!(layout(&h1, 4), GLOBAL, "H1");
        assert_eq!(layout(&h2(""), 4), GLOBAL, "H2");
    }

    /// T10 (#1636 interplay). Since #1636 the parsed readout no longer carries
    /// the gather: it is lifted into the synthetic parameter `__ferx_ro_g0`, and
    /// the readout reads that variable. On every engine, the block read in `y`
    /// next to an η on `E0` must still take within-study sum-to-zero, bit for
    /// bit the explicit contrast's names. The control on the same engine moves
    /// the η to `CL`, which reaches `y` only through the state, so the gather is
    /// lifted on both sides of the gate and only the η path differs.
    ///
    /// Each case first asserts that the desugar really ran (`__ferx_ro_g0` is an
    /// individual parameter). Without it, this would test the pre-#1636 readout.
    ///
    /// Mutations — drop the readout on the ODE engine only: the ODE share case
    /// binds 17, and no other test here runs an ODE model. Feed the predicate
    /// the desugared readout *and* skip the appended `__ferx_ro_*` statements:
    /// the block becomes invisible and the share cases bind 17 (T1–T8 die too).
    /// Feeding the desugared readout alone is equivalent, since the appended
    /// `__ferx_ro_g0 = PLACEBO` statement carries the block's taint to it.
    #[test]
    fn a_block_lifted_out_of_the_readout_still_shares_a_scale() {
        const ODE: &str = "  ode(states=[central])\n\n[odes]\n  d/dt(central) = -CL / V * central";
        let share_ip = "  CL = TVEMAX\n  V = TVET50\n  E0 = TVE0 + ETA_E0";
        let state_ip = "  CL = TVEMAX * exp(ETA_E0)\n  V = TVET50\n  E0 = TVE0";
        let engine = |name: &str, ip: &str, contrast: &str| -> String {
            let analytic = scaling_model(ip);
            let text = match name {
                "analytical" => analytic,
                "ode" => analytic.replace("  pk one_cpt_iv(cl=CL, v=V)", ODE),
                _ => unreachable!(),
            };
            if contrast.is_empty() {
                text
            } else {
                text.replace(
                    "PLACEBO[STUDY, TIME]",
                    &format!("PLACEBO[STUDY, TIME, contrast = {contrast}]"),
                )
            }
        };
        let mut cases: Vec<(String, String, String, (LevelContrast, usize))> = Vec::new();
        for name in ["analytical", "ode"] {
            cases.push((
                format!("{name} share"),
                engine(name, share_ip, ""),
                engine(name, share_ip, "sum_to_zero_within"),
                WITHIN,
            ));
            cases.push((
                format!("{name} state-only control"),
                engine(name, state_ip, ""),
                engine(name, state_ip, "sum_to_zero"),
                GLOBAL,
            ));
        }
        cases.push((
            "compartment-free share".into(),
            h2(""),
            h2("sum_to_zero_within"),
            WITHIN,
        ));

        for (tag, auto, explicit, want) in cases {
            let mut pa = cf_pop(3, 1, &T6);
            let a = bind_parsed(&auto, &mut pa);
            assert!(
                a.model
                    .indiv_param_names
                    .iter()
                    .any(|n| n == "__ferx_ro_g0"),
                "[{tag}] the readout gather was not lifted: {:?}",
                a.model.indiv_param_names
            );
            assert_eq!(layout(&auto, 1), want, "[{tag}] auto");
            let mut pe = cf_pop(3, 1, &T6);
            let e = bind_parsed(&explicit, &mut pe);
            assert_eq!(
                a.model.theta_names, e.model.theta_names,
                "[{tag}] auto ≡ explicit"
            );
        }
    }
}

// ── #1642 / #1624: the binder's refusals, enumerated per cell ───────────────
//
// Each refusal's sentences are asserted one by one, so deleting any of them
// reddens a test here (the PR's message table names which).
mod contrast_refusals {
    use super::readout_share::{bind_parsed, cf_model, cf_pop, h2, BASE, EMAXY, T6};
    use super::*;
    use crate::api::{bind_theta_levels_from_fit, theta_level_values};
    use crate::parser::model_parser::{LevelBinding, LevelBindings, LevelContrast};

    fn refusal(text: &str, mut pop: Population) -> String {
        bind(text, &mut pop).expect_err("must be refused")
    }

    fn has(err: &str, parts: &[&str]) {
        for p in parts {
            assert!(err.contains(p), "missing {p:?} in: {err}");
        }
    }

    fn lacks(err: &str, parts: &[&str]) {
        for p in parts {
            assert!(!err.contains(p), "must not say {p:?}: {err}");
        }
    }

    const NOT_IDENTIFIED: &str = "the two are the same quantity, so the model is not identified.";
    const USE_WITHIN: &str = "Use `contrast = sum_to_zero_within` (the default for this shape), \
                              or drop the random effect.";

    /// T6 (M1, M2). A block read in the readout next to an η on `E0` is
    /// refused under every global contrast, naming the readout and `E0`; the
    /// η-first `[individual_parameters]` spelling names `E1`.
    ///
    /// Mutation — revert to the IP-only predicate: H2 binds under all three.
    #[test]
    fn explicit_global_contrasts_on_a_readout_block_are_refused() {
        for c in ["sum_to_zero", "ref", "none"] {
            let err = refusal(&h2(c), cf_pop(3, 1, &T6));
            has(
                &err,
                &[
                    "theta PLACEBO[STUDY, TIME]: ",
                    &format!("`contrast = {c}` leaves each STUDY group's mean free"),
                    "but the `y` readout reads this block and a random effect (through `E0`) at \
                     that grouping",
                    NOT_IDENTIFIED,
                    USE_WITHIN,
                ],
            );
            lacks(&err, &["individual parameter", "remove the block"]);
        }

        // M1: the share is an individual parameter, the η read through `E0`.
        let h3 = cf_model(
            "sum_to_zero",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + ETA_E0\n  E1 = E0 + PLACEBO"),
            &format!("E1 + {EMAXY}"),
        );
        let err = refusal(&h3, cf_pop(3, 1, &T6));
        has(
            &err,
            &[
                "but the individual parameter `E1` reads this block and carries a random effect \
                 (through `E0`) at that grouping",
                NOT_IDENTIFIED,
                USE_WITHIN,
            ],
        );
        lacks(&err, &["readout"]);

        // M1, direct: the parameter reads the η itself, so no `through`.
        let err = refusal(&mbma_model("ref"), population(2, 3));
        has(
            &err,
            &["but the individual parameter `CL` reads this block and carries a random effect at"],
        );
        lacks(&err, &["through"]);
    }

    /// M7. The η refusal on a block whose groups are all single levels must
    /// not recommend `sum_to_zero_within`, which would leave nothing to
    /// estimate; it says to remove the block.
    ///
    /// Mutation — always give the within advice: this dies on `lacks`.
    #[test]
    fn the_eta_refusal_on_singleton_groups_says_remove_the_block() {
        let err = refusal(&h2("sum_to_zero"), cf_pop(3, 1, &[1.0]));
        has(
            &err,
            &[
                "`contrast = sum_to_zero` leaves each STUDY group's mean free",
                NOT_IDENTIFIED,
                "Every STUDY group has a single level, so the random effect already carries each \
                 group's value: remove the block, or drop the random effect.",
            ],
        );
        lacks(&err, &["sum_to_zero_within"]);
    }

    /// T8 (#1624). A block left with no free θ is refused under every
    /// contrast, with or without an η; a block with *some* free θ binds even
    /// when one of its groups has none. Both sides of the gate in one test.
    ///
    /// Mutations — restore the old `levels.len() == 1 && SumToZero` check (the
    /// `ref` / within cells bind); count free θ per group instead of per block
    /// (the partial-singleton control is refused); run the zero-free check
    /// ahead of the η refusal (the one-level η + `sum_to_zero` cell gets M6's
    /// text instead of the η refusal's).
    #[test]
    fn a_block_with_no_free_theta_is_refused_whatever_the_contrast() {
        let one_level = |c: &str| {
            no_eta_model().replace("[STUDY, TIME]", &format!("[STUDY, TIME, contrast = {c}]"))
        };
        let none = "Use `contrast = none` if a single constant is what you meant.";

        // M3: one level, global sum-to-zero (auto and explicit).
        for text in [no_eta_model(), one_level("sum_to_zero")] {
            let err = refusal(&text, population(1, 1));
            has(
                &err,
                &[
                    "theta PLACEBO[STUDY, TIME]: the data carries a single level, which \
                     sum-to-zero pins at 0. ",
                    none,
                ],
            );
            lacks(&err, &["random effect"]);
        }

        // M4: one level, within-group sum-to-zero, no η.
        let err = refusal(&one_level("sum_to_zero_within"), population(1, 1));
        has(
            &err,
            &[
                "the data carries a single level, which the within-group sum-to-zero pins at 0. ",
                none,
            ],
        );
        lacks(&err, &["`contrast = sum_to_zero`", "random effect"]);

        // M4b: three studies at one time each, within, no η.
        let err = refusal(&one_level("sum_to_zero_within"), population(3, 1));
        has(
            &err,
            &[
                "every STUDY group has a single level, which the within-group sum-to-zero pins \
                 at 0, so the block estimates nothing.",
                "Use `contrast = sum_to_zero` to estimate the levels around their common mean, \
                 or `contrast = none`.",
            ],
        );
        lacks(&err, &["sum_to_zero_within", "random effect"]);

        // M5: one level, reference.
        let err = refusal(&one_level("ref"), population(1, 1));
        has(
            &err,
            &[
                "the data carries a single level, which is the reference level, held at 0. ",
                none,
            ],
        );
        lacks(&err, &["sum-to-zero", "random effect"]);

        // M6: one level carried by an η — auto and explicit within.
        for text in [mbma_model(""), mbma_model("sum_to_zero_within"), h2("")] {
            let err = refusal(&text, population(1, 1));
            has(
                &err,
                &[
                    "every STUDY group has a single level, and ",
                    " reads this block and ",
                    " — the random effect already carries each group's value, so the block \
                     estimates nothing. Remove the block.",
                ],
            );
            lacks(
                &err,
                &[
                    "`contrast = none`",
                    "`contrast = sum_to_zero",
                    "`contrast = ref`",
                ],
            );
        }
        // M6 on all-singleton groups, the readout site named.
        let err = refusal(&h2(""), cf_pop(3, 1, &[1.0]));
        has(
            &err,
            &["every STUDY group has a single level, and the `y` readout reads this block"],
        );

        // The η refusal runs first: one level, η, `sum_to_zero` is the η cell.
        let err = refusal(&mbma_model("sum_to_zero"), population(1, 1));
        has(
            &err,
            &["leaves each STUDY group's mean free", NOT_IDENTIFIED],
        );

        // Controls. `none` on one level binds with its one θ.
        let mut pop = population(1, 1);
        let model = bind(&one_level("none"), &mut pop).unwrap();
        assert_eq!(model.n_theta, 3);
        // One study observed once, next to one observed three times: the block
        // keeps study 1's two free θ, and the singleton reads exactly 0.
        let mut pop = population(2, 3);
        let s2 = &mut pop.subjects[1];
        s2.obs_times.truncate(1);
        s2.observations.truncate(1);
        s2.obs_cmts.truncate(1);
        s2.cens.truncate(1);
        let model = bind(&mbma_model(""), &mut pop).unwrap();
        assert_eq!(model.n_theta, 4, "TVCL, TVV, study 1's two free levels");
        let theta = [2.0, 0.3, -0.7, 10.0];
        let values = theta_level_values(&model, &theta).unwrap();
        assert_eq!(values["PLACEBO"][3].label, "STUDY=2,TIME=1");
        assert_eq!(values["PLACEBO"][3].value.to_bits(), 0.0f64.to_bits());
    }

    /// T7. A fit bound before #1642 (H2 under global sum-to-zero, 17 free θ)
    /// still drives a design: `bind_theta_levels_from_fit` takes the stored
    /// contrast, never re-resolving it, so the old layout survives.
    ///
    /// Mutation — re-resolve the contrast in `bind_theta_levels_from_fit`:
    /// `n_theta` goes to 18.
    #[test]
    fn a_fit_bound_before_1642_rebinds_its_own_layout() {
        let text = h2("");
        let mut pop = cf_pop(3, 1, &T6);
        let today = bind_parsed(&text, &mut pop);
        assert_eq!(today.model.n_theta, 18, "today's layout: within, 15 free");
        let labels = today.bindings.levels["PLACEBO"].labels.clone();
        assert_eq!(labels.len(), 18);
        let mut old = LevelBindings::new();
        old.insert(
            "PLACEBO".to_string(),
            LevelBinding {
                labels,
                groups: vec![0; 18],
                contrast: LevelContrast::SumToZero,
            },
        );

        let mut design = cf_pop(3, 1, &T6);
        let mut parsed = parse_full_model(&text).unwrap();
        bind_theta_levels_from_fit(&mut parsed, &text, &mut design, &old).expect("rebind");
        assert_eq!(
            parsed.model.n_theta, 20,
            "TVE0, TVEMAX, TVET50 and 17 free levels"
        );
        let theta: Vec<f64> = (0..20).map(|k| 0.05 * k as f64 + 0.01).collect();
        let values = theta_level_values(&parsed.model, &theta).unwrap();
        let v = &values["PLACEBO"];
        for (k, level) in v.iter().take(17).enumerate() {
            assert_eq!(level.value.to_bits(), theta[k + 1].to_bits(), "level {k}");
        }
        let neg_sum = -theta[1..18].iter().fold(0.0, |a, t| a + t);
        assert_eq!(v[17].value.to_bits(), neg_sum.to_bits());
    }
}

/// #1621: the parse records the data-derived bindings it compiled the model from.
mod data_bindings {
    use super::*;
    use crate::api::{bind_covariate_stats, bind_theta_levels};
    use crate::parser::model_parser::{DataBindings, LevelContrast};

    /// A level block **and** a symbolic covariate centre, so both halves of
    /// `DataBindings` are live.
    fn model() -> String {
        no_eta_model()
            .replace("theta PLACEBO[STUDY, TIME]", "theta PLACEBO[STUDY]")
            .replace(
                "[structural_model]",
                "[covariates]\n  WT continuous\n  STUDY categorical\n\n\
                 [covariate_model]\n  V ~ WT power(center = median) => THETA_V_WT(0.9, 0.01, 5.0)\n\n\
                 [structural_model]",
            )
    }

    /// [`population`] with three studies and a subject weight that differs per
    /// study (median 70.0).
    fn pop() -> Population {
        let mut pop = population(3, 2);
        for (s, wt) in pop.subjects.iter_mut().zip([60.0, 70.0, 90.0]) {
            s.covariates.insert("WT".to_string(), wt);
        }
        pop.covariate_names.push("WT".to_string());
        pop
    }

    /// T1. `CompiledModel::data_bindings()` is exactly what the final parse was
    /// bound with, in both bind orders, and empty on an unbound parse.
    ///
    /// Mutations — stamp only `levels` (the stats half comes back empty); stamp
    /// `DataBindings::default()` (both halves empty); stamp from the bindings of
    /// the first, unbound parse (empty again). Each dies on the non-empty /
    /// equality assertions below.
    #[test]
    fn the_parse_stamps_both_halves_in_either_bind_order() {
        let text = model();
        let unbound = parse_full_model(&text).unwrap();
        assert!(unbound.model.data_bindings().is_empty());
        assert_eq!(*unbound.model.data_bindings(), DataBindings::default());

        let mut seen: Vec<DataBindings> = Vec::new();
        for levels_first in [true, false] {
            let mut data = pop();
            let mut parsed = parse_full_model(&text).unwrap();
            if levels_first {
                bind_theta_levels(&mut parsed, &text, &mut data).unwrap();
                bind_covariate_stats(&mut parsed, &text, &data).unwrap();
            } else {
                bind_covariate_stats(&mut parsed, &text, &data).unwrap();
                bind_theta_levels(&mut parsed, &text, &mut data).unwrap();
            }
            let stamped = parsed.model.data_bindings();
            assert_eq!(stamped.levels, parsed.bindings.levels, "{levels_first}");
            assert_eq!(
                stamped.covariate_stats, parsed.bindings.covariate_stats,
                "{levels_first}"
            );
            let placebo = &stamped.levels["PLACEBO"];
            assert_eq!(placebo.labels, ["STUDY=1", "STUDY=2", "STUDY=3"]);
            assert_eq!(placebo.groups, [0, 0, 0]);
            assert_eq!(placebo.contrast, LevelContrast::SumToZero);
            assert_eq!(stamped.covariate_stats["WT"].median, 70.0);
            assert!(!stamped.is_empty());
            seen.push(stamped.clone());
        }
        assert_eq!(seen[0], seen[1]);
    }
}

/// #1621 T7: a fitted binding that lists a level more than once is refused.
mod from_fit_repeated_labels {
    use super::*;
    use crate::api::bind_theta_levels_from_fit;
    use crate::parser::model_parser::LevelBindings;

    /// Two level blocks: `PLACEBO[STUDY, TIME]` is block 1 (declared first),
    /// `EFF[STUDY]` block 2.
    fn two_block_model() -> String {
        no_eta_model()
            .replace(
                "theta TVV(10.0, 0.1, 500.0)",
                "theta TVV(10.0, 0.1, 500.0)\n  theta EFF[STUDY](0.0, -10.0, 10.0)",
            )
            .replace("CL = TVCL + PLACEBO", "CL = TVCL + PLACEBO + EFF")
    }

    fn fitted(text: &str) -> LevelBindings {
        let mut pop = population(2, 2);
        let mut parsed = parse_full_model(text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, text, &mut pop).unwrap();
        parsed.bindings.levels
    }

    fn bind(text: &str, design: &mut Population, fitted: &LevelBindings) -> Result<usize, String> {
        let mut parsed = parse_full_model(text).unwrap();
        bind_theta_levels_from_fit(&mut parsed, text, design, fitted)?;
        Ok(parsed.model.n_theta)
    }

    /// Repeat `label` (with its group) at the end of `block`'s labels.
    fn repeat(b: &mut LevelBindings, block: &str, label: &str) {
        let binding = b.get_mut(block).unwrap();
        let i = binding.labels.iter().position(|l| l == label).unwrap();
        let g = binding.groups[i];
        binding.labels.push(label.to_string());
        binding.groups.push(g);
    }

    /// The refusal must blame the bindings, never the design.
    fn assert_not_about_the_design(err: &str) {
        assert!(
            !err.contains("design") && !err.contains("never observed"),
            "{err}"
        );
    }

    /// T7. One repeated label is refused naming the block, its columns and the
    /// label; the same bindings without the repeat bind (the control); several
    /// repeats list each label once; a repeat in block 2 of 2 names block 2 only
    /// and writes nothing to the population.
    ///
    /// Mutations — delete the repeat check (the bind goes `Ok` with 6 θ against the
    /// fit's 5, one per label) and every arm dies on `unwrap_err`. Delete either sentence of
    /// the message and the substring assertions die: the first sentence carries
    /// the block, the count and the labels, the second the cause.
    #[test]
    fn a_repeated_label_in_the_fits_bindings_is_refused() {
        let text = no_eta_model();
        let clean = fitted(&text);
        assert_eq!(bind(&text, &mut population(2, 2), &clean), Ok(5));

        let mut once = clean.clone();
        repeat(&mut once, "PLACEBO", "STUDY=2,TIME=1");
        let err = bind(&text, &mut population(2, 2), &once).unwrap_err();
        assert!(
            err.contains(
                "theta PLACEBO[STUDY, TIME]: the fit's level bindings list 1 level(s) more \
                 than once: `STUDY=2,TIME=1`."
            ),
            "{err}"
        );
        assert!(
            err.contains(
                "Each level has exactly one fitted theta, so the bindings are malformed — \
                 they are not the ones the fit recorded."
            ),
            "{err}"
        );
        assert_not_about_the_design(&err);

        let mut several = clean.clone();
        repeat(&mut several, "PLACEBO", "STUDY=2,TIME=2");
        repeat(&mut several, "PLACEBO", "STUDY=1,TIME=1");
        repeat(&mut several, "PLACEBO", "STUDY=2,TIME=2");
        let err = bind(&text, &mut population(2, 2), &several).unwrap_err();
        assert!(
            err.contains(
                "list 2 level(s) more than once: `STUDY=2,TIME=2`, `STUDY=1,TIME=1`. Each"
            ),
            "{err}"
        );
        assert_not_about_the_design(&err);

        let text2 = two_block_model();
        let clean2 = fitted(&text2);
        assert!(bind(&text2, &mut population(2, 2), &clean2).is_ok());
        let mut second = clean2.clone();
        repeat(&mut second, "EFF", "STUDY=1");
        let mut design = population(2, 2);
        let before = design.clone();
        let err = bind(&text2, &mut design, &second).unwrap_err();
        assert!(
            err.starts_with("theta EFF[STUDY]: the fit's level bindings list 1"),
            "{err}"
        );
        assert!(err.contains("`STUDY=1`."), "{err}");
        assert!(!err.contains("PLACEBO"), "{err}");
        assert_not_about_the_design(&err);
        assert_eq!(design.covariate_names, before.covariate_names);
        for (a, b) in design.subjects.iter().zip(&before.subjects) {
            assert_eq!(a.covariates.len(), b.covariates.len(), "{}", a.id);
            assert!(!a.covariates.contains_key("__level_PLACEBO"), "{}", a.id);
        }
    }
}

// ── #1649 / #1650: when a level block absorbs a random effect ───────────────
//
// The labels of every cell come from an oracle, not from hand: the part of each
// subject's η sensitivity that no fixed effect can reproduce, `P⊥_X Z`, with
// `X` every θ column under the contrast and `Z` the per-subject η columns. A
// rank of 0 where the block-free baseline is positive means ω is not informed
// by the data at all — the case the binder must refuse.
mod absorption {
    use super::readout_share::{cf_model, cf_pop, scaling_model, BASE, EMAXY, T6};
    use super::*;
    use crate::api::bind_theta_levels_from_fit;
    use crate::parser::model_parser::{LevelBindings, LevelContrast};
    use nalgebra::DMatrix;

    /// Rank threshold on singular values normalised by `‖Z‖_F`; the gap it
    /// sits in is measured and asserted by `binder_agrees_with_the_jacobian_oracle`.
    const RANK_TOL: f64 = 1e-6;

    /// `[STUDY, VISIT]`: VISIT 1 holds TIME 0, 1, 2 and VISIT 2 holds 4, 8, 12,
    /// so the block does not resolve a subject's observations.
    pub(super) fn visit_pop(n: usize, per: usize) -> Population {
        let mut p = cf_pop(n, per, &T6);
        for s in p.subjects.iter_mut() {
            s.obs_covariates = T6
                .iter()
                .map(|t| {
                    let mut m = HashMap::new();
                    m.insert("VISIT".to_string(), if *t < 3.0 { 1.0 } else { 2.0 });
                    m
                })
                .collect();
        }
        p
    }

    /// The model shapes of the #1649 plan. Every one carries `ETA_E0`.
    const SHAPES: [&str; 8] = ["H1", "G", "H2", "H5", "H6", "S1", "S2", "H7"];

    /// Shape `tag` with the block on `cols` and `contrast` (`""` = auto).
    pub(super) fn shape(tag: &str, cols: &str, contrast: &str) -> String {
        let cf = |ip: String, y: String| cf_model(contrast, cols, &ip, &y);
        let pk = |ip: &str| {
            let modifier = if contrast.is_empty() {
                String::new()
            } else {
                format!(", contrast = {contrast}")
            };
            scaling_model(ip).replace(
                "PLACEBO[STUDY, TIME]",
                &format!("PLACEBO[{cols}{modifier}]"),
            )
        };
        match tag {
            "H1" => cf(
                format!("{BASE}  E0 = TVE0 + PLACEBO + ETA_E0"),
                format!("E0 + {EMAXY}"),
            ),
            "G" => cf(
                "  EMAX = TVEMAX + PLACEBO + ETA_E0\n  ET50 = TVET50\n  E0 = TVE0".into(),
                format!("E0 + {EMAXY}"),
            ),
            "H2" => cf(
                format!("{BASE}  E0 = TVE0 + ETA_E0"),
                format!("E0 + PLACEBO + {EMAXY}"),
            ),
            "H5" => cf(
                format!("{BASE}  E0 = TVE0 * exp(ETA_E0)"),
                format!("E0 * exp(PLACEBO) + {EMAXY}"),
            ),
            "H6" => cf(
                "  EMAX = TVEMAX + ETA_E0\n  ET50 = TVET50\n  E0 = TVE0".into(),
                format!("E0 + PLACEBO + {EMAXY}"),
            ),
            "S1" => pk("  CL = TVEMAX\n  V = TVET50 * exp(ETA_E0)\n  E0 = TVE0"),
            "S2" => pk("  CL = TVEMAX * exp(ETA_E0)\n  V = TVET50\n  E0 = TVE0"),
            "H7" => cf(
                format!("{BASE}  E0 = TVE0\n  Z = TVE0 * exp(ETA_E0)"),
                format!("E0 + PLACEBO + {EMAXY}"),
            ),
            _ => unreachable!("{tag}"),
        }
    }

    /// Bind `text` against a copy of `pop`: the resolved contrast and the
    /// block's free-θ count, or the refusal.
    pub(super) fn try_bind(text: &str, pop: &Population) -> Result<(LevelContrast, usize), String> {
        let mut p = pop.clone();
        let mut parsed = parse_full_model(text)?;
        crate::api::bind_theta_levels(&mut parsed, text, &mut p)?;
        let free = parsed
            .model
            .theta_names
            .iter()
            .filter(|n| n.starts_with("PLACEBO["))
            .count();
        Ok((parsed.bindings.levels["PLACEBO"].contrast, free))
    }

    /// The Jacobian pieces at the initial θ and η = 0, by central FD of the
    /// f64 predictor: the block's per-level columns (bound under `none`), the
    /// other θ columns, the per-subject η columns, and the level labels.
    ///
    /// The layout is taken from an η-free twin bound under `none`, then imposed
    /// on the real model with [`bind_theta_levels_from_fit`], so a model the
    /// binder refuses can still be measured.
    pub(super) struct Jac {
        xb: DMatrix<f64>,
        xo: DMatrix<f64>,
        z: DMatrix<f64>,
        labels: Vec<String>,
        one_column: bool,
    }

    pub(super) fn jacobian(text: &str, pop0: &Population) -> Jac {
        let twin = text
            .replace("ETA_E0", "0.0")
            .replace("omega 0.0", "omega ETA_E0");
        let mut p_twin = pop0.clone();
        let mut pt = parse_full_model(&twin).unwrap();
        crate::api::bind_theta_levels(&mut pt, &twin, &mut p_twin).expect("η-free twin binds");
        let fitted: LevelBindings = pt.bindings.levels.clone();
        let mut pop = pop0.clone();
        let mut parsed = parse_full_model(text).unwrap();
        bind_theta_levels_from_fit(&mut parsed, text, &mut pop, &fitted).expect("from_fit");
        let m = &parsed.model;
        let theta0 = m.default_params.theta.clone();
        let eta0 = vec![0.0; m.n_eta];
        let preds = |th: &[f64], s: usize, et: &[f64]| {
            crate::pk::compute_predictions_with_tv(m, &pop.subjects[s], th, et)
        };
        let ns = pop.subjects.len();
        let lens: Vec<usize> = (0..ns).map(|s| preds(&theta0, s, &eta0).len()).collect();
        let nrow: usize = lens.iter().sum();
        let offs: Vec<usize> = lens
            .iter()
            .scan(0, |a, n| {
                let o = *a;
                *a += n;
                Some(o)
            })
            .collect();
        let theta_col = |k: usize| -> Vec<f64> {
            let h = 1e-6 * theta0[k].abs().max(1.0);
            let (mut tp, mut tm) = (theta0.clone(), theta0.clone());
            tp[k] += h;
            tm[k] -= h;
            let mut col = vec![0.0; nrow];
            for s in 0..ns {
                let (a, b) = (preds(&tp, s, &eta0), preds(&tm, s, &eta0));
                for j in 0..a.len() {
                    col[offs[s] + j] = (a[j] - b[j]) / (2.0 * h);
                }
            }
            col
        };
        let block: Vec<usize> = (0..m.n_theta)
            .filter(|&i| m.theta_names[i].starts_with("PLACEBO["))
            .collect();
        let other: Vec<usize> = (0..m.n_theta).filter(|i| !block.contains(i)).collect();
        let mat = |cols: Vec<Vec<f64>>| DMatrix::from_fn(nrow, cols.len(), |r, c| cols[c][r]);
        let mut zc = Vec::new();
        for e in 0..m.n_eta {
            for s in 0..ns {
                let h = 1e-6;
                let (mut ep, mut em) = (eta0.clone(), eta0.clone());
                ep[e] += h;
                em[e] -= h;
                let (a, b) = (preds(&theta0, s, &ep), preds(&theta0, s, &em));
                let mut col = vec![0.0; nrow];
                for j in 0..a.len() {
                    col[offs[s] + j] = (a[j] - b[j]) / (2.0 * h);
                }
                zc.push(col);
            }
        }
        let labels = fitted["PLACEBO"].labels.clone();
        Jac {
            xb: mat(block.iter().map(|&k| theta_col(k)).collect()),
            xo: mat(other.iter().map(|&k| theta_col(k)).collect()),
            z: mat(zc),
            one_column: !labels[0].contains(','),
            labels,
        }
    }

    /// The coding matrix of `contrast` over the per-level columns.
    fn coding(j: &Jac, contrast: LevelContrast) -> DMatrix<f64> {
        let l = j.labels.len();
        // A one-column block is a single group (`assign_groups`).
        let grp: Vec<&str> = j
            .labels
            .iter()
            .map(|s| {
                if j.one_column {
                    ""
                } else {
                    s.split(',').next().unwrap()
                }
            })
            .collect();
        let unit = |k: usize, minus: Option<usize>| {
            let mut c = vec![0.0; l];
            c[k] = 1.0;
            if let Some(m) = minus {
                c[m] = -1.0;
            }
            c
        };
        let cols: Vec<Vec<f64>> = match contrast {
            LevelContrast::Unconstrained => (0..l).map(|k| unit(k, None)).collect(),
            LevelContrast::Ref => (1..l).map(|k| unit(k, None)).collect(),
            LevelContrast::SumToZero => (0..l.saturating_sub(1))
                .map(|k| unit(k, Some(l - 1)))
                .collect(),
            LevelContrast::SumToZeroWithin => (0..l)
                .filter_map(|k| {
                    let last = (0..l).rev().find(|&i| grp[i] == grp[k]).unwrap();
                    (k != last).then(|| unit(k, Some(last)))
                })
                .collect(),
            LevelContrast::Auto => unreachable!(),
        };
        DMatrix::from_fn(l, cols.len(), |r, c| cols[c][r])
    }

    /// Singular values of the part of `z` outside `col(x)`, normalised by `‖z‖_F`.
    fn residual_sv(z: &DMatrix<f64>, x: &DMatrix<f64>) -> Vec<f64> {
        let zn = z.norm();
        assert!(zn.is_finite(), "non-finite η sensitivity");
        if zn == 0.0 {
            return Vec::new();
        }
        let r = if x.ncols() == 0 {
            z.clone()
        } else {
            let svd = x.clone().svd(true, false);
            let u = svd.u.unwrap();
            let s = &svd.singular_values;
            assert!(s.iter().all(|v| v.is_finite()), "non-finite θ sensitivity");
            let mx = s.iter().copied().fold(0.0, f64::max);
            let keep: Vec<usize> = (0..s.len()).filter(|&i| s[i] > 1e-10 * mx).collect();
            let q = DMatrix::from_fn(x.nrows(), keep.len(), |r, c| u[(r, keep[c])]);
            z - &q * (q.transpose() * z)
        };
        r.svd(false, false)
            .singular_values
            .iter()
            .map(|v| v / zn)
            .collect()
    }

    fn rank(sv: &[f64]) -> usize {
        sv.iter().filter(|&&v| v > RANK_TOL).count()
    }

    /// `(baseline rank, rank under contrast, free θ)`. Every singular value is
    /// pushed onto `seen`, so the caller can report the gap around `RANK_TOL`.
    pub(super) fn oracle(
        j: &Jac,
        contrast: LevelContrast,
        seen: &mut Vec<f64>,
    ) -> (usize, usize, usize) {
        let base = residual_sv(&j.z, &j.xo);
        let xb = &j.xb * coding(j, contrast);
        let mut x = DMatrix::zeros(j.xo.nrows(), j.xo.ncols() + xb.ncols());
        x.columns_mut(0, j.xo.ncols()).copy_from(&j.xo);
        x.columns_mut(j.xo.ncols(), xb.ncols()).copy_from(&xb);
        let under = residual_sv(&j.z, &x);
        seen.extend(base.iter().chain(&under));
        (rank(&base), rank(&under), xb.ncols())
    }

    /// The designs of the grid: `(tag, block columns, data)`.
    fn designs() -> Vec<(&'static str, &'static str, Population)> {
        vec![
            ("1-col 1/study", "STUDY", cf_pop(3, 1, &T6)),
            ("1-col 2/study", "STUDY", cf_pop(3, 2, &T6)),
            ("[STUDY,TIME] 1/study", "STUDY, TIME", cf_pop(3, 1, &T6)),
            ("[STUDY,TIME] 2/study", "STUDY, TIME", cf_pop(3, 2, &T6)),
            ("[STUDY,VISIT] 1/study", "STUDY, VISIT", visit_pop(3, 1)),
        ]
    }

    pub(super) const EXPLICIT: [(LevelContrast, &str); 4] = [
        (LevelContrast::SumToZero, "sum_to_zero"),
        (LevelContrast::SumToZeroWithin, "sum_to_zero_within"),
        (LevelContrast::Ref, "ref"),
        (LevelContrast::Unconstrained, "none"),
    ];

    /// T1. Over designs × shapes × contrasts, the binder refuses an explicit
    /// contrast exactly when the oracle says the block absorbs the random
    /// effect under it (rank 0 against a positive baseline). Cells whose
    /// contrast leaves no free θ are #1624's and skipped. `auto` must resolve
    /// to a contrast that does not absorb, global whenever global does not, and
    /// be refused only when every contrast absorbs.
    #[test]
    fn binder_agrees_with_the_jacobian_oracle() {
        let mut wrong: Vec<String> = Vec::new();
        let mut seen: Vec<f64> = Vec::new();
        let (mut absorbed_cells, mut free_cells) = (0usize, 0usize);
        for (dtag, cols, pop) in designs() {
            for tag in SHAPES {
                // G reads the block through `EMAX * TIME / (TIME + ET50)`, which is 0
                // at TIME = 0: on a block keyed on TIME that level has no effect on
                // `y` under any contrast, with or without the η, so the oracle
                // measures an unidentified level rather than an absorbed η.
                if tag == "G" && cols.contains(',') {
                    continue;
                }
                let jac = jacobian(&shape(tag, cols, "none"), &pop);
                let mut absorbs = HashMap::new();
                for (c, token) in EXPLICIT {
                    let (base, under, free) = oracle(&jac, c, &mut seen);
                    if free == 0 {
                        continue;
                    }
                    let absorbed = base > 0 && under == 0;
                    absorbed_cells += usize::from(absorbed);
                    free_cells += usize::from(!absorbed);
                    absorbs.insert(token, absorbed);
                    let got = try_bind(&shape(tag, cols, token), &pop);
                    if got.is_err() != absorbed {
                        wrong.push(format!(
                            "{dtag} {tag} {token}: oracle rank {under} of {base} \
                             ⇒ refuse={absorbed}, binder {got:?}"
                        ));
                    }
                }
                let all_absorb = absorbs.values().all(|a| *a);
                match try_bind(&shape(tag, cols, ""), &pop) {
                    Err(e) if !all_absorb => wrong.push(format!(
                        "{dtag} {tag} auto: refused, but some contrast does not absorb: {e}"
                    )),
                    Err(_) => {}
                    Ok(_) if all_absorb => wrong.push(format!(
                        "{dtag} {tag} auto: bound, but every contrast absorbs"
                    )),
                    Ok((resolved, _)) => {
                        let token = EXPLICIT.iter().find(|(c, _)| *c == resolved).unwrap().1;
                        if absorbs.get(token).copied().unwrap_or(false) {
                            wrong.push(format!("{dtag} {tag} auto → {token}, which absorbs"));
                        }
                        if absorbs.get("sum_to_zero") == Some(&false)
                            && resolved != LevelContrast::SumToZero
                        {
                            wrong.push(format!(
                                "{dtag} {tag} auto → {token}, but global does not absorb"
                            ));
                        }
                    }
                }
            }
        }
        let zero = seen
            .iter()
            .copied()
            .filter(|v| *v <= RANK_TOL)
            .fold(0.0, f64::max);
        let live = seen
            .iter()
            .copied()
            .filter(|v| *v > RANK_TOL)
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "oracle: largest zero {zero:.3e}, smallest live {live:.3e}; \
             {absorbed_cells} absorbed / {free_cells} identified cells"
        );
        assert!(
            zero < 1e-8 && live > 1e-3,
            "oracle gap collapsed: {zero:e} / {live:e}"
        );
        // Both sides of the gate must be exercised.
        assert!(
            absorbed_cells >= 20 && free_cells >= 20,
            "{absorbed_cells} absorbed / {free_cells} identified cells"
        );
        assert!(
            wrong.is_empty(),
            "{} cells disagree:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }

    /// T2, the differential pair. One model, one subject per study against two:
    /// the one-column block is refused at one and binds its 2 free θ at two.
    /// And on `[STUDY, TIME]` at one subject per study, an η that reaches `y`
    /// only through the state (S2) takes the within-study contrast, while one
    /// that never reaches `y` (H7) stays global. Before #1649 both one-column
    /// sides bound and S2 was global, so the pair straddles the gate.
    #[test]
    fn one_subject_per_level_straddles_the_gate() {
        let h1 = shape("H1", "STUDY", "");
        let err = try_bind(&h1, &cf_pop(3, 1, &T6)).expect_err("1 subject/study");
        assert!(err.starts_with("theta PLACEBO[STUDY]: "), "{err}");
        assert_eq!(
            try_bind(&h1, &cf_pop(3, 2, &T6)),
            Ok((LevelContrast::SumToZero, 2)),
            "2 subjects/study"
        );

        let one = cf_pop(3, 1, &T6);
        assert_eq!(
            try_bind(&shape("S2", "STUDY, TIME", ""), &one),
            Ok((LevelContrast::SumToZeroWithin, 15)),
            "S2: η reaches y through the state"
        );
        assert_eq!(
            try_bind(&shape("H7", "STUDY, TIME", ""), &one),
            Ok((LevelContrast::SumToZero, 17)),
            "H7: η never reaches y"
        );
    }

    /// The model-side half, per shape: the funnel's site and the η's route.
    /// H1/G funnel through an individual parameter, H2/H5 through the readout;
    /// H6/S1 (η only in a time-varying term) and S2 (η only in the state) have
    /// none; H7's η never reaches `y`. The default readout of a model with no
    /// `[scaling]` block (the `no_eta_model` shape) reaches `y` through `V`.
    ///
    /// `mbma_model` adds a parameter funnel reached only through the state.
    #[test]
    fn couplings_per_shape() {
        use crate::parser::model_parser::{EtaCoupling, EtaRoute, ScaleShare};
        fn coupling(text: &str) -> EtaCoupling {
            let parsed = parse_full_model(text).unwrap();
            let decl = &parsed.model.theta_blocks().level_blocks()[0];
            assert_eq!(decl.eta_couplings.len(), 1, "one random effect");
            decl.eta_couplings[0].clone()
        }
        let site = |param: Option<&str>, via: Option<&str>| ScaleShare {
            param: param.map(str::to_string),
            eta_via: via.map(str::to_string),
        };
        let via = |v: &str| Some(EtaRoute::Via(v.to_string()));
        let cases: [(&str, Option<ScaleShare>, Option<EtaRoute>); 8] = [
            ("H1", Some(site(Some("E0"), None)), via("E0")),
            ("G", Some(site(Some("EMAX"), None)), via("EMAX")),
            ("H2", Some(site(None, Some("E0"))), via("E0")),
            ("H5", Some(site(None, Some("E0"))), via("E0")),
            ("H6", None, via("EMAX")),
            ("S1", None, via("V")),
            ("S2", None, Some(EtaRoute::State)),
            ("H7", None, None),
        ];
        for (tag, funnel, reach) in cases {
            let c = coupling(&shape(tag, "STUDY", ""));
            assert_eq!(c.eta, "ETA_E0", "{tag}");
            assert!(!c.kappa, "{tag}");
            assert_eq!(c.funnels.first().map(|f| f.site.clone()), funnel, "{tag}");
            assert_eq!(c.reach, reach, "{tag}");
        }

        // No `[scaling]`: `y` is the amount over `V`.
        let ne = coupling(&no_eta_model());
        assert_eq!((ne.funnels.len(), ne.reach), (0, via("V")), "no_eta_model");
        // A parameter funnel reached only through the state: `CL` carries both,
        // and nothing else reads either.
        let mb = coupling(&mbma_model(""));
        assert_eq!(
            mb.funnels
                .iter()
                .map(|f| f.site.clone())
                .collect::<Vec<_>>(),
            vec![site(Some("CL"), None)],
            "mbma_model"
        );
        assert_eq!(mb.reach, Some(EtaRoute::State), "mbma_model");
    }
}
