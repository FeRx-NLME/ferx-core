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
///
/// The random effect sits on `Z`, which `y` never reads. With one subject per
/// study (`population`) the block takes a level at every observation, so a
/// random effect reaching `y` by any route, the state included, would be
/// absorbed (#1650): this fixture was `V = TVV * exp(ETA_V)` before, which the
/// Jacobian oracle measures as rank 0.
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
  V  = TVV
  Z  = TVV * exp(ETA_V)

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
    crate::api::bind_theta_levels(&mut parsed, text, pop).map_err(|e| e.to_string())?;
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
    // #1746 (review r1 #2): the folded text still carries the diagnostic. Mutation:
    // build it with `EngineError::from(text)` → the `code()` assert dies.
    assert_eq!(err.code(), Some("E_THETA_LEVELS_UNBOUND"), "{err}");
    // (review r1 #6) …and the advice is in `to_string()` once (asserted below) and in
    // `suggestion()`, never also in `message()`, so a renderer of `message()` +
    // `suggestion()` does not show it twice. Both sides of the split in one test.
    let advice = err.suggestion().expect("the refusal names its binder");
    assert!(advice.contains("bind_from_fit"), "{advice}");
    assert!(
        !err.message().contains("bind_from_fit"),
        "{}",
        err.message()
    );
    assert!(err.to_string().contains("bind_from_fit"), "{err}");
    // One assertion per sentence (and clause) of the message and the suggestion.
    assert!(
        err.to_string().starts_with(
            "`theta PLACEBO[...]` was never bound to data, so it has no levels and every \
             value gathered from it is NaN. "
        ),
        "the shared E_THETA_LEVELS_UNBOUND message: {err}"
    );
    // That the advice is also *true* — following it reproduces the fit's predictions on
    // new data — is `from_fit::following_the_unbound_predict_refusal_gives_the_fits_predictions`.
    assert!(
        err.to_string().contains(
            "With a fit's θ, call `bind_from_fit(&mut parsed, &model_text, &mut population, \
             &fit.data_bindings)` on the population you pass to `predict`"
        ),
        "the from-fit binder, on the population predict reads: {err}"
    );
    assert!(
        err.to_string()
            .contains(", with the `data_bindings` the fit (or its `.fitrx`) carries"),
        "where the fit's bindings come from: {err}"
    );
    assert!(
        err.to_string()
            .contains(", and predict with the model it re-parses into `parsed`."),
        "the bound model is a re-parse, not the one in hand: {err}"
    );
    assert!(
        err.to_string().contains(
            "`bind_theta_levels` on that population fits only a θ laid out for the levels it \
             discovers, such as the model's own `default_params`."
        ),
        "when the other binder is the right one: {err}"
    );
    assert!(
        err.to_string().ends_with(
            "Or declare the block explicitly as `theta PLACEBO[N](...)` and index it with \
             your own column."
        ),
        "the counted-form alternative: {err}"
    );
    for absent in [
        "__level_",
        "bind_theta_levels_from_fit",
        "parsed.bindings.levels",
        "not found in data",
        "before simulating",
        "read_population_for_simulation",
        "run_model_simulate",
    ] {
        assert!(
            !err.to_string().contains(absent),
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
    assert_eq!(sim_err.to_string(), sim.message);

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
#[allow(deprecated)] // the deprecated binder, kept as a control (#1619)
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
        bind_theta_levels_from_fit(&mut parsed, text, design, fitted).map_err(|e| e.to_string())?;
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
            err.to_string()
                .contains("the supplied theta has 4 values but this model has 5"),
            "{err}"
        );
        assert!(
            err.to_string()
                .contains("declares the theta level block(s) `PLACEBO`"),
            "{err}"
        );
        assert!(
            err.to_string()
                .contains("whose theta count is set by the data the model was bound against."),
            "why the count moves: {err}"
        );
        assert!(
            err.to_string().contains(
                "A fit's theta fits only a design bound against that fit's level bindings, \
                 which give the design the fit's theta layout"
            ),
            "what a fit's theta needs: {err}"
        );
        assert!(
            !err.to_string().contains("bind_theta_levels"),
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
            err.to_string()
                .contains("the supplied theta has 1 values but this model has 2"),
            "{err}"
        );
        assert!(!err.to_string().contains("level block"), "{err}");
        assert!(!err.to_string().contains("level bindings"), "{err}");
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
    /// re-discovers the levels (5 θ here, against the fit's 7), and `predict_diag` had no
    /// θ-length guard then (#1615), so the fit's θ came back `Ok` read at the wrong positions
    /// — measured on this fixture at `74078e0c`: subject 3 at t = 1 predicted 2.455468
    /// against the fit's 7.557837, subject 2 at t = 2 0.182376 against 6.250023. Since
    /// #1615 that mismatched count is `E_THETA_LENGTH`; a same-count mismatch is not.
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
            err.to_string().contains("`bind_from_fit("),
            "the refusal names the binder this test follows: {err}"
        );

        let mut followed = new_data.clone();
        let mut parsed = parse_full_model(&text).unwrap();
        crate::api::bind_from_fit(&mut parsed, &text, &mut followed, fit.model.data_bindings())
            .expect("bind");
        assert_eq!(rows(&parsed.model, &followed), want);

        // Why the advice cannot be `bind_theta_levels`: on this population it lays θ out
        // for the levels it discovers, which is not the fit's layout.
        let mut own = new_data.clone();
        assert_eq!(bind(&text, &mut own).unwrap().n_theta, 5);
    }

    // ── #1647: a bound model on a population that was never bound ──────────────

    const NEVER_BOUND: &str = "`theta PLACEBO[...]` is bound, but this population was never \
        bound for it, so its records carry no index into the block's levels. ";
    const RUN_BINDER: &str = "Bind the population with `bind_from_fit(&mut parsed, \
        &model_text, &mut population, &fit.data_bindings)`";
    const RUN_WHICH_BINDINGS: &str = ", passing the bindings the θ you run was laid out on: \
        the fit's `data_bindings`, or, for the model's own θ, a clone of \
        `parsed.model.data_bindings()` taken before the call.";
    const RUN_WHICH_MODEL: &str = " Then run the model it re-parses into `parsed`.";
    const FIT_BINDER: &str = "To fit this population, parse the model text again and bind it \
        with `bind_theta_levels(&mut parsed, &model_text, &mut population)`, which lays θ out \
        for the levels it holds.";

    /// The refusal a θ-running entry point gives, one assertion per sentence (and clause).
    fn assert_run_refusal(err: &str, entry: &str) {
        assert!(err.starts_with(NEVER_BOUND), "{entry}: the cause: {err}");
        assert!(err.contains(RUN_BINDER), "{entry}: the binder: {err}");
        assert!(
            err.contains(RUN_WHICH_BINDINGS),
            "{entry}: which bindings to pass: {err}"
        );
        assert!(
            err.ends_with(RUN_WHICH_MODEL),
            "{entry}: which model to run: {err}"
        );
        // The other cell's binder: a θ already laid out must not be re-laid out.
        assert!(!err.contains(FIT_BINDER), "{entry}: {err}");
        assert!(!err.contains("`bind_theta_levels("), "{entry}: {err}");
        assert_no_engine_column(err, entry);
    }

    fn assert_no_engine_column(err: &str, entry: &str) {
        for absent in [
            "__level_",
            "not found in data",
            "Available covariate columns",
        ] {
            assert!(!err.contains(absent), "{entry}: `{absent}` in: {err}");
        }
    }

    /// #1647. Fit, then predict / simulate / npde / fit on new data that was never bound:
    /// the model's block is bound, so `E_THETA_LEVELS_UNBOUND` passes, and before this the
    /// covariate check refused with "covariate(s) not found in data: __level_PLACEBO" — a
    /// column the user never wrote. Measured at `712cd47b` with the check removed: that text
    /// on `predict_diag`, `simulate_with_options_diag` and `fit`, and `compute_npde_npd`
    /// returning `Ok` with every npd `NaN`.
    ///
    /// The cells (model × population × entry), and the sentence each gets:
    ///
    /// | Model | Population | Entry | Gets |
    /// |---|---|---|---|
    /// | unbound | any | predict / simulate | `E_THETA_LEVELS_UNBOUND` (#1644, pinned above) |
    /// | unbound | any | `fit` | `fit()`'s own refusal (pinned above) |
    /// | bound (`bind_theta_levels`) | never bound | predict, simulate, npde | `NEVER_BOUND` + `RUN_*` |
    /// | bound from a fit (`bind_from_fit`) | never bound | the same | the same — here, second arm |
    /// | bound | never bound | `fit`, `ferx check` | `NEVER_BOUND` + `FIT_BINDER` |
    /// | bound | bound by the same binding | any | runs — here, the control |
    /// | bound on A | bound on B, θ count differs | predict, simulate | `E_THETA_LENGTH` (#1615) |
    /// | bound on A | bound on B, same θ count | any | **runs, silently on A's levels** — measured, follow-up |
    ///
    /// Both sides of the entry gate in one test (`Run` vs `Fit`) and both sides of the
    /// population gate (never bound vs bound).
    ///
    /// Mutations — delete `check_level_index_columns` from `predict_diag`,
    /// `check_simulation_data`, `check_model_data_rule` or `compute_npde_npd` and that arm
    /// goes `Ok` (the covariate check no longer names the column), naming itself; swap the
    /// two binder sentences and both sides die; delete any one sentence or clause and its
    /// own assertion dies; drop the level-column exclusion from `check_covariates` and the
    /// `check_model_data` arm dies on `E_MISSING_COVARIATE`.
    #[test]
    fn a_bound_model_on_a_population_never_bound_names_the_binder_for_its_entry() {
        let text = no_eta_model();
        let mut fit_pop = population(3, 2);
        let fit = bind_fit(&text, &mut fit_pop);
        let params = fit.model.default_params.clone();
        let never = population(3, 2);

        // The control: the population it was bound on runs.
        crate::api::predict_diag(&fit.model, &fit_pop, &params).expect("bound population");

        let err = crate::api::predict_diag(&fit.model, &never, &params)
            .err()
            .expect("predict_diag");
        assert_run_refusal(&err.to_string(), "predict_diag");
        let err = crate::api::predict(&fit.model, &never, &params)
            .err()
            .expect("predict");
        assert_run_refusal(&err.to_string(), "predict");
        let err = crate::api::simulate_with_options_diag(
            &fit.model,
            &never,
            &params,
            1,
            &Default::default(),
        )
        .err()
        .expect("simulate_with_options_diag");
        assert_run_refusal(&err.to_string(), "simulate_with_options_diag");
        let err = crate::stats::npde::compute_npde_npd(&fit.model, &never, &params, 20, Some(1))
            .err()
            .expect("compute_npde_npd");
        assert_run_refusal(&err.to_string(), "compute_npde_npd");

        let err = crate::api::fit(&fit.model, &never, &params, &FitOptions::default())
            .err()
            .expect("fit");
        assert!(err.starts_with(NEVER_BOUND), "fit: the cause: {err}");
        assert!(err.ends_with(FIT_BINDER), "fit: the binder: {err}");
        assert!(!err.contains("bind_from_fit"), "fit: {err}");
        assert_no_engine_column(&err, "fit");

        // The diagnostic itself, as `ferx check` and #1746's typed errors carry it.
        let diags = crate::api::check_model_data(&fit.model, &never);
        let codes: Vec<&str> = diags.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E_THETA_LEVELS_DATA_UNBOUND"), "{codes:?}");
        assert!(!codes.contains(&"E_MISSING_COVARIATE"), "{codes:?}");
        let sim = crate::api::validation::check_simulation_data(&fit.model, &never);
        let d = sim
            .iter()
            .find(|d| d.code == "E_THETA_LEVELS_DATA_UNBOUND")
            .expect("simulate's bundle");
        assert_eq!(d.block.as_deref(), Some("parameters"));
        assert!(!sim.iter().any(|d| d.code == "E_MISSING_COVARIATE"));

        // A model laid out on the fit's bindings (`bound_from_fit`) gets the same refusal:
        // `bind_from_fit` is still the binder, and the one `bind_theta_levels` would refuse.
        let mut design = population(2, 2);
        let mut from_fit = parse_full_model(&text).unwrap();
        crate::api::bind_from_fit(&mut from_fit, &text, &mut design, fit.model.data_bindings())
            .expect("bind the design");
        let p = fit_theta_params(&from_fit.model, &params.theta);
        let err = crate::api::predict_diag(&from_fit.model, &never, &p)
            .err()
            .expect("predict_diag, from-fit model");
        assert_run_refusal(&err.to_string(), "predict_diag, from-fit model");
    }

    /// #1647: each refusal's advice is *true*. Following the `Run` sentence gives the fit's
    /// own predictions on the new data, bit for bit, both ways it offers: with the fit's
    /// bindings, and — literally as written, on the running model's own `parsed` — with a
    /// clone of `parsed.model.data_bindings()` taken before the call. Following the `Fit`
    /// sentence leaves no fatal model/data finding for `fit()`.
    ///
    /// Review r1 row 1: the first wording passed `parsed.model.data_bindings()` straight into
    /// a call taking `&mut parsed`, which does not compile (E0502, measured); this test then
    /// followed it with a different object, so the sentence as written was never exercised.
    ///
    /// Mutation — name `bind_theta_levels` in the `Run` sentence instead: that binder lays
    /// θ out for the new data's 5 levels, so the fit's 7-value θ is refused (`n_theta`
    /// assertion below shows why); and a sentence naming `layout_from_fit` (which writes no
    /// column) would leave the population refused again.
    #[test]
    fn following_the_never_bound_population_refusals_works() {
        let text = no_eta_model();
        let mut fit_pop = population(3, 2);
        let fit = bind_fit(&text, &mut fit_pop);
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
        assert_eq!(want.len(), 4);

        let mut new_data = population(3, 2);
        new_data.subjects.remove(0);
        crate::api::predict_diag(&fit.model, &new_data, &fit_theta_params(&fit.model, &theta))
            .err()
            .expect("the refusal under test");
        // "the fit's `data_bindings`": a fresh parse bound on the fit's.
        let mut followed = new_data.clone();
        let mut parsed = parse_full_model(&text).unwrap();
        crate::api::bind_from_fit(&mut parsed, &text, &mut followed, fit.model.data_bindings())
            .expect("bind");
        assert_eq!(rows(&parsed.model, &followed), want);

        // "for the model's own θ, a clone of `parsed.model.data_bindings()` taken before the
        // call": the running model's own `parsed`, re-bound in place.
        let mut followed = new_data.clone();
        let mut parsed = fit;
        let own = parsed.model.data_bindings().clone();
        crate::api::bind_from_fit(&mut parsed, &text, &mut followed, &own).expect("bind");
        assert_eq!(rows(&parsed.model, &followed), want);

        let mut refit = new_data.clone();
        let mut parsed = parse_full_model(&text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, &text, &mut refit).expect("bind");
        assert_eq!(
            parsed.model.n_theta, 5,
            "laid out for the new data's own levels"
        );
        let fatal: Vec<_> = crate::api::check_model_data(&parsed.model, &refit)
            .into_iter()
            .filter(|d| d.severity == crate::diagnostics::Severity::Error)
            .collect();
        assert!(fatal.is_empty(), "{fatal:?}");
    }

    /// Review r1 row 2: an **unbound** model is never told its block "is bound". The gate's
    /// own filter keeps unbound blocks out; `check_model_data` and `compute_npde_npd` run no
    /// unbound-block check ahead of it, so without the filter they would say exactly that.
    /// Both sides of the filter in one test: the unbound model reports nothing, the same
    /// block bound (on a population never bound) reports it.
    ///
    /// Mutation — drop the `unbound_level_blocks` filter in `bound_level_columns`: this test
    /// dies, on its first (direct-call) arm; the `check_model_data` and npde arms state the
    /// same through the entry points the review named. The mutation survived 257 lib tests
    /// before this test existed (review r1).
    #[test]
    fn an_unbound_model_is_never_told_its_block_is_bound() {
        use crate::api::{check_level_index_columns, LevelDataEntry};
        let text = no_eta_model();
        let unbound = parse_full_model(&text).unwrap().model;
        assert!(!unbound.theta_blocks().unbound_level_blocks().is_empty());
        let pop = population(2, 2);
        for entry in [LevelDataEntry::Run, LevelDataEntry::Fit] {
            assert!(
                check_level_index_columns(&unbound, &pop, entry).is_empty(),
                "{entry:?}"
            );
        }
        let codes: Vec<String> = crate::api::check_model_data(&unbound, &pop)
            .into_iter()
            .map(|d| d.code)
            .collect();
        assert!(
            !codes.iter().any(|c| c == "E_THETA_LEVELS_DATA_UNBOUND"),
            "{codes:?}"
        );
        // npde has no unbound-model refusal of its own (#1763); whatever it returns, it must
        // not be this one.
        if let Err(e) = crate::stats::npde::compute_npde_npd(
            &unbound,
            &pop,
            &unbound.default_params,
            20,
            Some(1),
        ) {
            assert!(!e.to_string().contains("is bound, but"), "{e}");
        }

        // The other side: bound, on a population never bound, it is reported.
        let bound = bind_fit(&text, &mut population(2, 2)).model;
        for entry in [LevelDataEntry::Run, LevelDataEntry::Fit] {
            assert_eq!(
                check_level_index_columns(&bound, &pop, entry).len(),
                1,
                "{entry:?}"
            );
        }
    }

    // ── #1633: "nothing is written to `population` unless every block binds" ──────

    /// Two level blocks: `EFFT` (keyed on `TIME`) first, `EFFS` (keyed on `STUDY`) second,
    /// so a design with a new study binds the first and is refused on the second.
    fn two_block_model() -> String {
        no_eta_model()
            .replace(
                "theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)",
                "theta EFFT[TIME](0.0, -10.0, 10.0)\n  theta EFFS[STUDY](0.0, -10.0, 10.0)",
            )
            .replace("CL = TVCL + PLACEBO", "CL = TVCL + EFFT + EFFS")
    }

    /// Everything a binder may write to a population, in a fixed order with values as bits:
    /// the column lists and every subject's five covariate channels, keys sorted.
    fn population_snapshot(pop: &Population) -> Vec<String> {
        let map = |m: &HashMap<String, f64>| {
            let mut kv: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{k}={:x}", v.to_bits()))
                .collect();
            kv.sort();
            kv.join(",")
        };
        let mut out = vec![
            format!("names {:?}", pop.covariate_names),
            format!("inputs {:?}", pop.input_columns),
        ];
        for s in &pop.subjects {
            out.push(format!("{} subject {}", s.id, map(&s.covariates)));
            for (kind, maps) in [
                ("obs", &s.obs_covariates),
                ("dose", &s.dose_covariates),
                ("pk_only", &s.pk_only_covariates),
                ("reset", &s.reset_covariates),
            ] {
                out.push(format!("{} {kind} n={}", s.id, maps.len()));
                for (j, m) in maps.iter().enumerate() {
                    out.push(format!("{} {kind}[{j}] {}", s.id, map(m)));
                }
            }
        }
        out
    }

    /// #1633. `bind_from_fit` and the deprecated `bind_theta_levels_from_fit` promise
    /// "nothing is written to `population` unless every block binds". On a two-block model
    /// whose first block binds and whose second is refused for an unseen level, the design
    /// population — and `parsed.model`'s θ layout — must be exactly as passed in.
    ///
    /// The two-block geometry is the point: a single block cannot tell "nothing written"
    /// from "written block by block, refused before the first write". The `Ok` control on
    /// the same fixture shows both columns *are* written when every block binds, so a
    /// snapshot that cannot see a write (say, of the wrong population) fails there.
    ///
    /// Mutations (each kills its binder's arm, which names itself):
    /// 1. on the unseen-level refusal, write a `__level_EFFT` value into the population
    ///    before returning `Err`;
    /// 2. bind block by block — build one block's table and write its column inside the
    ///    loop — so `EFFT`'s column is on the population when `EFFS` is refused.
    #[test]
    #[allow(deprecated)]
    fn a_refusal_on_the_second_block_writes_nothing_to_the_population() {
        let text = two_block_model();
        let mut fit_pop = population(2, 2);
        let fit = bind_fit(&text, &mut fit_pop);
        assert_eq!(
            fit.model.theta_blocks().level_blocks().len(),
            2,
            "two level blocks"
        );
        let blocks: Vec<&str> = fit
            .model
            .theta_blocks()
            .level_blocks()
            .iter()
            .map(|d| d.name())
            .collect();
        assert_eq!(blocks, ["EFFT", "EFFS"], "block 1 binds, block 2 refuses");

        type Binder =
            fn(&mut ParsedModel, &str, &mut Population, &ParsedModel) -> Result<(), String>;
        let binders: [(&str, Binder); 2] = [
            ("bind_from_fit", |p, t, pop, fit| {
                crate::api::bind_from_fit(p, t, pop, fit.model.data_bindings())
                    .map_err(|e| e.to_string())
            }),
            ("bind_theta_levels_from_fit", |p, t, pop, fit| {
                bind_theta_levels_from_fit(p, t, pop, &fit.bindings.levels)
                    .map_err(|e| e.to_string())
            }),
        ];
        for (name, binder) in binders {
            // Refused: study 3 was never fitted; both TIME levels were.
            let mut design = population(3, 2);
            let before = population_snapshot(&design);
            let mut parsed = parse_full_model(&text).unwrap();
            let n_theta_before = parsed.model.n_theta;
            let err = binder(&mut parsed, &text, &mut design, &fit).unwrap_err();
            assert!(
                err.contains("theta EFFS[STUDY]") && err.contains("`STUDY=3`"),
                "{name}: refused on the second block: {err}"
            );
            let after = population_snapshot(&design);
            assert_eq!(
                before.len(),
                after.len(),
                "{name}: population shape changed"
            );
            for (b, a) in before.iter().zip(&after) {
                assert_eq!(b, a, "{name}: the refusal wrote to the population");
            }
            assert_eq!(
                parsed.model.n_theta, n_theta_before,
                "{name}: parsed.model moved"
            );

            // The control: a design whose every level was fitted binds, and both columns land.
            let mut ok = population(2, 2);
            let before = population_snapshot(&ok);
            let mut parsed = parse_full_model(&text).unwrap();
            binder(&mut parsed, &text, &mut ok, &fit).expect("every block binds");
            assert_ne!(before, population_snapshot(&ok), "{name}: nothing written");
            for col in ["__level_EFFT", "__level_EFFS"] {
                assert!(
                    ok.subjects.iter().all(|s| s.covariates.contains_key(col)),
                    "{name}: {col} not written"
                );
            }
            assert_eq!(parsed.model.n_theta, fit.model.n_theta, "{name}");
        }
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

    /// T3. The readout spellings a block taking a level at every observation
    /// absorbs (H4 bare η in `y`, H5 multiplicative, H6 η only on `EMAX`, S1 η
    /// on `V` read by a PK readout, and since #1650 S2, η reaching `y` only
    /// through the state) against the one it does not (H7, η `y` never reads)
    /// — both sides of the gate in one test.
    ///
    /// Mutations — η taint ignores `Variable` (H6, S1 red); states carry no
    /// taint (S2 red); count every declared η (H7 red).
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
                WITHIN,
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
    /// bit the explicit contrast's names. So must an η on `CL`, which reaches `y`
    /// only through the state (#1650): on the ODE engine that route is read off
    /// the `[odes]` line. The control on the same engine moves the η to `Z`,
    /// which `y` never reads, so the gather is lifted on both sides of the gate
    /// and only the η path differs.
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
        let unread_ip = "  CL = TVEMAX\n  V = TVET50\n  E0 = TVE0\n  Z = TVE0 * exp(ETA_E0)";
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
                format!("{name} state-only"),
                engine(name, state_ip, ""),
                engine(name, state_ip, "sum_to_zero_within"),
                WITHIN,
            ));
            cases.push((
                format!("{name} unread-η control"),
                engine(name, unread_ip, ""),
                engine(name, unread_ip, "sum_to_zero"),
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
#[allow(deprecated)] // the deprecated binder, kept as a control (#1619)
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
#[allow(deprecated)] // the deprecated binder, kept as a control (#1619)
mod from_fit_repeated_labels {
    use super::*;
    use crate::api::bind_theta_levels_from_fit;
    use crate::parser::model_parser::LevelBindings;

    /// Two level blocks: `PLACEBO[STUDY, TIME]` is block 1 (declared first),
    /// `EFF[STUDY]` block 2.
    pub(super) fn two_block_model() -> String {
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
        bind_theta_levels_from_fit(&mut parsed, text, design, fitted).map_err(|e| e.to_string())?;
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

/// `bind_from_fit` (#1619, #1672): the refusal table, cell by cell. Each refusal's
/// sentences are asserted one by one, and every refusal leaves the population
/// untouched. Analytic one-compartment IV; nothing is fitted, only bound.
mod bind_from_fit {
    use super::*;
    use crate::api::bind_from_fit;
    use crate::parser::model_parser::{DataBindings, LevelContrast};
    use crate::types::ParsedModel;

    /// `no_eta_model`, with and without its level block, with and without a
    /// `center = median` relation on `WT`.
    pub(super) fn model(level: bool, median: bool) -> String {
        let mut text = no_eta_model();
        if !level {
            text = text
                .replace("theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)", "")
                .replace(" + PLACEBO", "");
        }
        if median {
            text.push_str(
                "\n[covariates]\n  WT continuous\n\n[covariate_model]\n  V ~ WT \
                 power(center = median) => THETA_V_WT(0.6, 0.01, 5.0)\n",
            );
        }
        text
    }

    /// [`population`] with a subject-level `WT` of `base + 10·study`.
    pub(super) fn weighed(n_studies: usize, n_times: usize, base: f64) -> Population {
        let mut pop = population(n_studies, n_times);
        for (s, subject) in pop.subjects.iter_mut().enumerate() {
            subject
                .covariates
                .insert("WT".to_string(), base + 10.0 * s as f64);
        }
        pop.covariate_names.push("WT".to_string());
        pop
    }

    /// The bindings a fit of `text` on `weighed(3, 2, 60)` records.
    pub(super) fn fitted(text: &str) -> DataBindings {
        let mut pop = weighed(3, 2, 60.0);
        let mut parsed = parse_full_model(text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, text, &mut pop).unwrap();
        crate::api::bind_covariate_stats(&mut parsed, text, &pop).unwrap();
        parsed.model.data_bindings().clone()
    }

    /// Bind a fresh parse of `text` on `design`.
    fn bind(text: &str, design: &mut Population, b: &DataBindings) -> Result<ParsedModel, String> {
        let mut parsed = parse_full_model(text).unwrap();
        bind_from_fit(&mut parsed, text, design, b).map_err(|e| e.to_string())?;
        Ok(parsed)
    }

    /// The refusal, after checking it wrote nothing to the design.
    fn refusal(text: &str, b: &DataBindings) -> String {
        let mut design = weighed(3, 2, 80.0);
        let before = format!("{design:?}");
        let err = bind(text, &mut design, b)
            .map(|_| ())
            .expect_err("must be refused");
        assert_eq!(
            format!("{design:?}"),
            before,
            "a refusal wrote to the design"
        );
        err
    }

    /// Both halves come from the fit, at once: the design's own median (90) is not
    /// the fit's (70), its level layout is the fit's, and the model records exactly
    /// the fit's bindings. `bind_covariate_stats` afterwards is then a no-op (T3's
    /// unit form).
    ///
    /// Mutations — stamp only `levels`, or skip the re-parse when the model has no
    /// level block: the `Median`-only arm keeps an unresolved relation and is refused.
    #[test]
    fn both_halves_come_from_the_fit() {
        for (level, median) in [(true, true), (false, true), (true, false)] {
            let text = model(level, median);
            let b = fitted(&text);
            let mut design = weighed(3, 2, 80.0);
            let mut parsed = bind(&text, &mut design, &b).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                parsed.model.data_bindings(),
                &b,
                "level {level} median {median}"
            );
            if median {
                assert_eq!(b.covariate_stats["WT"].median, 70.0);
            }
            let n_theta = parsed.model.n_theta;
            crate::api::bind_covariate_stats(&mut parsed, &text, &design).unwrap();
            assert_eq!(
                parsed.model.data_bindings(),
                &b,
                "a later stats bind is a no-op"
            );
            assert_eq!(parsed.model.n_theta, n_theta);
        }
        // A plain model with empty bindings binds as itself.
        let text = model(false, false);
        let mut design = weighed(3, 2, 80.0);
        let parsed = bind(&text, &mut design, &DataBindings::default()).unwrap();
        assert_eq!(parsed.model.n_theta, 2);
    }

    /// R1–R3: empty bindings on a model that needs them. Each half's clause is
    /// asserted present exactly when the model has that half, all three cells in one
    /// test, so forcing either clause on or off reddens it.
    #[test]
    fn empty_bindings_name_each_half_the_model_has() {
        for (level, median) in [(true, false), (false, true), (true, true)] {
            let err = refusal(&model(level, median), &DataBindings::default());
            let cell = format!("level {level} median {median}: {err}");
            assert!(
                err.starts_with(
                    "this fit carries no data-derived bindings, so the model cannot be \
                     rebuilt the way it was fitted: "
                ),
                "{cell}"
            );
            assert_eq!(
                err.contains(
                    "its theta level block(s) `PLACEBO[STUDY, TIME]` take their level layout \
                     from the data it was fitted on"
                ),
                level,
                "{cell}"
            );
            assert_eq!(
                err.contains(
                    "its [covariate_model] relations state a statistic of `WT` symbolically, \
                     so their centres come from the data it was fitted on"
                ),
                median,
                "{cell}"
            );
            assert_eq!(err.contains(", and its"), level && median, "{cell}");
            assert!(
                err.ends_with(
                    ". The fit is an older `.fitrx` bundle, or was made before ferx recorded \
                     these bindings with a fit. Refit the model to record them."
                ),
                "{cell}"
            );
            for absent in ["design", "edited"] {
                assert!(!err.contains(absent), "`{absent}` in {cell}");
            }
            assert_eq!(err.contains("statistic"), median, "{cell}");
        }
    }

    /// R5 / R6: the statistics half is validated against the relations, never
    /// filled in from the design.
    ///
    /// Mutations — delete either check: R5's cell binds on the design's median, or
    /// is refused by `assert_covariate_model_bound` instead; R6's cells bind `Ok`.
    #[test]
    fn statistics_must_match_the_relations() {
        let text = model(true, true);
        let mut b = fitted(&text);
        b.covariate_stats.clear();
        assert_eq!(
            refusal(&text, &b),
            "[covariate_model] relations state a statistic of `WT` symbolically, but the \
             fit's covariate statistics carry no entry for it: they are not the statistics \
             this model was fitted with."
        );

        let with_stats = fitted(&model(false, true));
        for text in [model(false, false), model(true, false)] {
            let mut b = fitted(&text);
            b.covariate_stats = with_stats.covariate_stats.clone();
            assert_eq!(
                refusal(&text, &b),
                "the fit's covariate statistics carry `WT`, which no [covariate_model] \
                 relation of this model reads: the bindings belong to a different model."
            );
        }
    }

    /// R7 / R8 (#1672): the two shapes a fit never writes. The `debug_assert!` in
    /// `bind_theta_levels` measures that claim on every level fixture of the suite.
    ///
    /// Mutations — delete either check: the cell re-parses into a layout no fit had
    /// and binds `Ok`.
    #[test]
    fn a_split_group_or_an_auto_contrast_is_malformed() {
        let text = model(true, false);
        let mut b = fitted(&text);
        let placebo = b.levels.get_mut("PLACEBO").unwrap();
        assert_eq!(placebo.groups, vec![0; 6], "global sum_to_zero: one group");
        placebo.groups = vec![0, 0, 1, 1, 0, 0];
        assert_eq!(
            refusal(&text, &b),
            "theta PLACEBO[STUDY, TIME]: the fit's level bindings are malformed: the levels \
             of contrast group 0 are split. A fit records each group's levels contiguously, \
             since a group's free theta occupy one contiguous range."
        );

        let mut b = fitted(&text);
        b.levels.get_mut("PLACEBO").unwrap().contrast = LevelContrast::Auto;
        assert_eq!(
            refusal(&text, &b),
            "theta PLACEBO[STUDY, TIME]: the fit's level bindings are malformed: they record \
             the contrast `auto`, and a fit never records `auto`, only the contrast it \
             resolved to."
        );

        // A contiguous multi-group binding is not split: the check fires on a group's
        // return, not on a group change.
        let mut b = fitted(&text);
        let placebo = b.levels.get_mut("PLACEBO").unwrap();
        placebo.groups = vec![0, 0, 0, 1, 1, 1];
        placebo.contrast = LevelContrast::SumToZeroWithin;
        let mut design = weighed(3, 2, 80.0);
        bind(&text, &mut design, &b).expect("contiguous groups bind");
    }

    /// #1680 review r1, finding 1: a recorded contrast that is not the one the block
    /// declares. `ref` and `sum_to_zero` free the same number of θ, so an edited
    /// `.fitrx` passes every count check and reads the fitted θ (levels 2..n against
    /// level 1) as levels 1..n−1 with the last at minus their sum. Both directions,
    /// and the control: under a declared `auto`, any resolved contrast is the fit's.
    ///
    /// Mutation — drop the declared-contrast comparison: both edited cells bind `Ok`.
    #[test]
    fn a_contrast_other_than_the_declared_one_is_refused() {
        let text = model(true, false).replace(
            "theta PLACEBO[STUDY, TIME](",
            "theta PLACEBO[STUDY, TIME, contrast = ref](",
        );
        let b = fitted(&text);
        assert_eq!(b.levels["PLACEBO"].contrast, LevelContrast::Ref);
        let mut design = weighed(3, 2, 80.0);
        bind(&text, &mut design, &b).expect("the fit's own contrast binds");

        let mut edited = b.clone();
        edited.levels.get_mut("PLACEBO").unwrap().contrast = LevelContrast::SumToZero;
        assert_eq!(
            refusal(&text, &edited),
            "theta PLACEBO[STUDY, TIME]: the fit's level bindings record the contrast \
             `sum_to_zero`, but this block declares `contrast = ref`. A fit records the \
             contrast its block declares, so the bindings are malformed or belong to a \
             different model."
        );

        let text = model(true, false).replace(
            "theta PLACEBO[STUDY, TIME](",
            "theta PLACEBO[STUDY, TIME, contrast = sum_to_zero](",
        );
        let mut edited = fitted(&text);
        edited.levels.get_mut("PLACEBO").unwrap().contrast = LevelContrast::Ref;
        assert!(
            refusal(&text, &edited).contains(
                "record the contrast `ref`, but this block declares `contrast = sum_to_zero`."
            ),
            "the other direction"
        );

        // Declared `auto`: the binding records what `auto` resolved to, and any
        // resolved contrast passes the shape check.
        let text = model(true, false);
        let mut b = fitted(&text);
        assert_eq!(b.levels["PLACEBO"].contrast, LevelContrast::SumToZero);
        b.levels.get_mut("PLACEBO").unwrap().contrast = LevelContrast::Ref;
        let mut design = weighed(3, 2, 80.0);
        bind(&text, &mut design, &b).expect("under `auto`, the recorded contrast is the fit's");
    }

    /// #1680 review r1, finding 6: a refusal from the re-parse itself — here a fitted
    /// `WT` median of 0, which `power` divides by — leaves the design without index
    /// columns and `parsed` as it was. On a model with a level block, so there are
    /// columns to leave behind.
    ///
    /// Mutation — write the index columns before the re-parse (the r1 order): the
    /// design gains `__level_PLACEBO` and `refusal`'s untouched-design check dies.
    #[test]
    fn a_refusal_from_the_re_parse_writes_nothing() {
        let text = model(true, true);
        let mut b = fitted(&text);
        b.covariate_stats.get_mut("WT").unwrap().median = 0.0;
        let err = refusal(&text, &b);
        assert!(err.contains("WT"), "the parser's own refusal: {err}");

        let mut parsed = parse_full_model(&text).unwrap();
        let before = parsed.model.n_theta;
        let mut design = weighed(3, 2, 80.0);
        bind_from_fit(&mut parsed, &text, &mut design, &b).unwrap_err();
        assert_eq!(parsed.model.n_theta, before, "parsed is not replaced");
        assert!(parsed.bindings.levels.is_empty(), "nor its bindings");
    }

    /// `text` parsed and bound to `design` the way `prepare_run` binds a model to its
    /// own data: every level block and every symbolic statistic resolved there.
    pub(super) fn pre_bound(text: &str, design: &mut Population) -> ParsedModel {
        let mut parsed = parse_full_model(text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, text, design).unwrap();
        crate::api::bind_covariate_stats(&mut parsed, text, design).unwrap();
        parsed
    }

    /// #1686: a model already bound to other data still needs the fit's bindings.
    /// The pre-bound twin's relations are resolved (on the design's median, 90), so
    /// a gate that reads `parsed.model` sees nothing to refuse; the unbound twin's
    /// are not. The straddle is asserted, then both twins must give the same
    /// empty-bindings refusal, with nothing written.
    ///
    /// Mutation — compute `symbolic` from `parsed.model` again, the pre-#1686 gate:
    /// the pre-bound `Median`-only twin binds `Ok` on the design's median.
    #[test]
    fn a_model_bound_to_other_data_still_needs_the_fits_bindings() {
        use crate::api::covariate_stats::symbolic_covariates;
        for (level, median) in [(false, true), (true, true), (true, false)] {
            let text = model(level, median);
            let cell = format!("level {level} median {median}");
            let mut design = weighed(3, 2, 80.0);
            let mut prebound = pre_bound(&text, &mut design);
            let unbound = parse_full_model(&text).unwrap();
            // The straddle: the old gate read the resolution state.
            assert_eq!(
                symbolic_covariates(&unbound.model).is_empty(),
                !median,
                "{cell}"
            );
            assert!(
                symbolic_covariates(&prebound.model).is_empty(),
                "{cell}: pre-binding resolves every relation"
            );
            if median {
                assert_eq!(prebound.bindings.covariate_stats["WT"].median, 90.0);
            }

            let want = refusal(&text, &DataBindings::default());
            let before_model = prebound.model.data_bindings().clone();
            let before = format!("{design:?}");
            let err = bind_from_fit(&mut prebound, &text, &mut design, &DataBindings::default())
                .expect_err(&cell)
                .to_string();
            assert_eq!(err, want, "{cell}: the twins refuse alike");
            assert_eq!(format!("{design:?}"), before, "{cell}");
            assert_eq!(prebound.model.data_bindings(), &before_model, "{cell}");
        }

        // The bound cell: full bindings rebind a pre-bound model on the fit's
        // median, not the design's.
        let text = model(true, true);
        let b = fitted(&text);
        let mut design = weighed(3, 2, 80.0);
        let mut prebound = pre_bound(&text, &mut design);
        let mut fresh = weighed(3, 2, 80.0);
        bind_from_fit(&mut prebound, &text, &mut fresh, &b).unwrap();
        assert_eq!(prebound.model.data_bindings(), &b);
        assert_eq!(prebound.bindings.covariate_stats["WT"].median, 70.0);
    }

    /// #1686, the partial cell: a pre-bound model given a fit whose statistics lack
    /// `WT` hears the statistics refusal, as the unbound model does, not the
    /// generic unbound-model one the re-parse would otherwise end in.
    ///
    /// Mutation — validate the statistics against `parsed.model`: the pre-bound
    /// model finds nothing missing and is refused by `assert_covariate_model_bound`.
    #[test]
    fn a_pre_bound_model_names_the_statistic_the_fit_lacks() {
        let text = model(true, true);
        let mut b = fitted(&text);
        b.covariate_stats.clear();
        let mut design = weighed(3, 2, 80.0);
        let mut prebound = pre_bound(&text, &mut design);
        let err = bind_from_fit(&mut prebound, &text, &mut design, &b)
            .unwrap_err()
            .to_string();
        assert_eq!(err, refusal(&text, &b));
        assert!(
            err.starts_with("[covariate_model] relations state a statistic of `WT`"),
            "{err}"
        );
    }

    /// #1686, the level half: binding stamps the contrast `auto` resolved to on the
    /// block's declaration, so a model pre-bound to a design "declares" the design's
    /// contrast. A fit whose `auto` resolved otherwise on its own data (here
    /// `sum_to_zero_within`, against the design's `sum_to_zero`) is the fit's
    /// layout, and must bind on the pre-bound model as on the unbound one. The
    /// straddle — the pre-bound declaration is no longer `auto` — is asserted.
    ///
    /// Mutation — read the level blocks from `parsed.model`: the pre-bound model is
    /// refused, told the block "declares `contrast = sum_to_zero`", which it does not.
    #[test]
    fn a_pre_bound_model_takes_the_contrast_auto_resolved_to_on_the_fit() {
        let text = model(true, false);
        let mut b = fitted(&text);
        let placebo = b.levels.get_mut("PLACEBO").unwrap();
        placebo.groups = vec![0, 0, 0, 1, 1, 1];
        placebo.contrast = LevelContrast::SumToZeroWithin;

        let mut design = weighed(3, 2, 80.0);
        let mut prebound = pre_bound(&text, &mut design);
        let declared = |p: &ParsedModel| p.model.theta_blocks().level_blocks()[0].contrast();
        assert_eq!(declared(&prebound), LevelContrast::SumToZero);
        assert_eq!(
            declared(&parse_full_model(&text).unwrap()),
            LevelContrast::Auto
        );

        let mut own = weighed(3, 2, 80.0);
        let unbound = bind(&text, &mut own, &b).expect("the unbound model binds");
        let mut fresh = weighed(3, 2, 80.0);
        bind_from_fit(&mut prebound, &text, &mut fresh, &b)
            .unwrap_or_else(|e| panic!("the pre-bound model binds: {e}"));
        assert_eq!(shape(&prebound.model), shape(&unbound.model));
    }

    /// The shape of a model: what `layout_from_fit` promises to match.
    type Shape = (
        usize,
        Vec<String>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<bool>,
        DataBindings,
    );

    fn shape(m: &crate::types::CompiledModel) -> Shape {
        let p = &m.default_params;
        (
            m.n_theta,
            p.theta_names.clone(),
            p.theta.clone(),
            p.theta_lower.clone(),
            p.theta_upper.clone(),
            p.theta_fixed.clone(),
            m.data_bindings().clone(),
        )
    }

    /// #1703: `layout_from_fit` lays the model out exactly as `bind_from_fit` does on
    /// the fit's own data — θ count, names, inits, bounds, `FIX` flags and recorded
    /// bindings — with no population, on every (level, median) cell and under a
    /// `ref` contrast.
    ///
    /// Mutations — drop the statistics from the re-parse's bindings in
    /// `lay_out_on_fit`: the `Median` cells' recorded bindings lose `WT` here, and
    /// `both_halves_come_from_the_fit` dies on the same edit through `bind_from_fit`.
    /// Skip `apply` in `layout_from_fit`: every cell keeps the unbound layout.
    #[test]
    fn layout_from_fit_is_bind_from_fit_without_the_population() {
        let ref_text = model(true, true).replace(
            "theta PLACEBO[STUDY, TIME](",
            "theta PLACEBO[STUDY, TIME, contrast = ref](",
        );
        for text in [
            model(true, true),
            model(false, true),
            model(true, false),
            ref_text,
        ] {
            let b = fitted(&text);
            let mut own = weighed(3, 2, 60.0);
            let bound = bind(&text, &mut own, &b).unwrap();
            let mut laid = parse_full_model(&text).unwrap();
            crate::api::layout_from_fit(&mut laid, &text, &b).unwrap();
            assert_eq!(shape(&laid.model), shape(&bound.model), "{text}");
            assert_eq!(laid.model.name, bound.model.name);
            assert_ne!(
                shape(&laid.model),
                shape(&parse_full_model(&text).unwrap().model),
                "the layout moved: {text}"
            );
        }
        // A plain model with empty bindings is laid out as itself.
        let text = model(false, false);
        let mut laid = parse_full_model(&text).unwrap();
        crate::api::layout_from_fit(&mut laid, &text, &DataBindings::default()).unwrap();
        assert_eq!(
            shape(&laid.model),
            shape(&parse_full_model(&text).unwrap().model)
        );
    }

    /// #1703: `layout_from_fit` refuses what `bind_from_fit` refuses, word for word,
    /// and leaves `parsed` as it was: a repeated label, a split (non-contiguous)
    /// group, missing statistics, empty bindings — on an unbound and on a pre-bound
    /// model (#1686).
    ///
    /// Mutation — delete `validate_fitted_levels` or `validate_fitted_stats` from
    /// `lay_out_on_fit`: the repeated label lays out `Ok`, or the split group and
    /// the missing statistics are refused by the parser or the bound assert in
    /// other words.
    #[test]
    fn layout_from_fit_refuses_what_bind_from_fit_refuses() {
        let text = model(true, true);
        let repeated = {
            let mut b = fitted(&text);
            let p = b.levels.get_mut("PLACEBO").unwrap();
            p.labels[1] = p.labels[0].clone();
            b
        };
        let split = {
            let mut b = fitted(&text);
            b.levels.get_mut("PLACEBO").unwrap().groups = vec![0, 0, 1, 1, 0, 0];
            b
        };
        let no_stats = {
            let mut b = fitted(&text);
            b.covariate_stats.clear();
            b
        };
        let cases = [
            ("repeated", repeated, "level(s) more than once"),
            ("split", split, "are split"),
            ("no stats", no_stats, "carry no entry for it"),
            (
                "empty",
                DataBindings::default(),
                "carries no data-derived bindings",
            ),
        ];
        for (what, b, says) in cases {
            let want = refusal(&text, &b);
            assert!(want.contains(says), "{what}: {want}");
            for pre in [false, true] {
                let mut parsed = if pre {
                    pre_bound(&text, &mut weighed(3, 2, 80.0))
                } else {
                    parse_full_model(&text).unwrap()
                };
                let before = shape(&parsed.model);
                let err = crate::api::layout_from_fit(&mut parsed, &text, &b)
                    .expect_err(&format!("{what} pre {pre}"))
                    .to_string();
                assert_eq!(err, want, "{what} pre {pre}");
                assert_eq!(shape(&parsed.model), before, "{what} pre {pre}: untouched");
            }
        }
    }

    /// R4 and the unseen-level refusal pass through unchanged, and still write
    /// nothing.
    #[test]
    fn the_level_refusals_of_the_old_binder_still_apply() {
        let text = model(true, true);
        let mut b = fitted(&text);
        let placebo = b.levels.remove("PLACEBO").unwrap();
        assert!(refusal(&text, &b).contains("the fit's level bindings carry no `PLACEBO`"));
        b.levels.insert("EXTRA".to_string(), placebo);
        assert!(refusal(&text, &b).contains("carry the block(s) `EXTRA`"));

        let b = fitted(&text);
        let mut design = weighed(4, 2, 80.0);
        let before = format!("{design:?}");
        let err = bind(&text, &mut design, &b).map(|_| ()).unwrap_err();
        assert!(
            err.contains("the design has 2 level(s) the fit estimated no theta for"),
            "{err}"
        );
        assert_eq!(format!("{design:?}"), before);
    }
}

// ── #1649 / #1650: when a level block absorbs a random effect ───────────────
//
// The labels of every cell come from an oracle, not from hand: the part of each
// subject's η sensitivity that no fixed effect can reproduce, `P⊥_X Z`, with
// `X` every θ column under the contrast and `Z` the per-subject η columns. A
// rank of 0 where the block-free baseline is positive means ω is not informed
// by the data at all — the case the binder must refuse.
#[allow(deprecated)] // the deprecated binder, kept as a control (#1619)
mod absorption {
    use super::readout_share::{cf_model, cf_pop, scaling_model, BASE, EMAXY, T6};
    use super::*;
    use crate::api::bind_theta_levels_from_fit;
    use crate::parser::model_parser::{LevelBindings, LevelContrast};
    use nalgebra::DMatrix;

    /// Rank threshold on singular values normalised by `‖Z‖_F`; the gap it
    /// sits in is measured and asserted by `binder_agrees_with_the_jacobian_oracle`.
    const RANK_TOL: f64 = 1e-6;

    /// `T6` with 0.5 in place of 0: every level of a `TIME`-keyed block holds a
    /// record after the dose, so none is dead by construction (#1679).
    pub(super) const NO0: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

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

    /// The model shapes of the #1649 plan, plus three that hold an individual
    /// parameter reading both the block and the η without being a funnel: the
    /// block read again in `y` (H8), the η read again (H9), and `TIME` read
    /// inside it (H10); two readout funnels reached through a unary function
    /// (H11) and a power (H12); and an η scaled by `TIME` inside the funnel
    /// candidate (H13). H6 and H13 are also the `TIME` twins of the `TAD` /
    /// `TAFD` shapes in `the_clocks_vary_like_time`. Every one carries `ETA_E0`.
    const SHAPES: [&str; 14] = [
        "H1", "G", "H2", "H5", "H6", "S1", "S2", "H7", "H8", "H9", "H10", "H11", "H12", "H13",
    ];

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
            "H8" => cf(
                format!("{BASE}  E0 = TVE0 + PLACEBO + ETA_E0"),
                "E0 + (EMAX + PLACEBO) * TIME / (TIME + ET50)".into(),
            ),
            "H9" => cf(
                "  EMAX = TVEMAX + ETA_E0\n  ET50 = TVET50\n  E0 = TVE0 + PLACEBO + ETA_E0".into(),
                format!("E0 + {EMAXY}"),
            ),
            "H10" => cf(
                format!("{BASE}  E0 = TVE0 + PLACEBO * TIME + ETA_E0"),
                format!("E0 + {EMAXY}"),
            ),
            "H11" => cf(
                format!("{BASE}  E0 = TVE0 + ETA_E0"),
                format!("exp((E0 + PLACEBO + {EMAXY}) / 10)"),
            ),
            "H12" => cf(
                format!("{BASE}  E0 = TVE0 + ETA_E0"),
                format!("(E0 + PLACEBO + {EMAXY}) ^ 2"),
            ),
            "H13" => cf(
                format!("{BASE}  E0 = TVE0 + PLACEBO + ETA_E0 * TIME"),
                format!("E0 + {EMAXY}"),
            ),
            _ => unreachable!("{tag}"),
        }
    }

    /// Bind `text` against a copy of `pop`: the resolved contrast and the
    /// block's free-θ count, or the refusal.
    pub(super) fn try_bind(text: &str, pop: &Population) -> Result<(LevelContrast, usize), String> {
        let mut p = pop.clone();
        let mut parsed = parse_full_model(text)?;
        crate::api::bind_theta_levels(&mut parsed, text, &mut p).map_err(|e| e.to_string())?;
        let free = parsed
            .model
            .theta_names
            .iter()
            .filter(|n| n.starts_with("PLACEBO["))
            .count();
        Ok((parsed.bindings.levels["PLACEBO"].contrast, free))
    }

    /// The Jacobian pieces at the initial θ and η = κ = 0 ([`jacobian_at`] for
    /// another point), by central FD of the f64 predictor: the block's
    /// per-level columns (bound under `none`), the other θ columns, the
    /// per-subject η columns, the per-unit columns of each random effect (a
    /// subject for an η, an occasion group of a subject for a kappa), and the
    /// level labels.
    ///
    /// The layout is the `none` layout of the levels the data shows
    /// ([`none_layout`]), imposed on the model with [`bind_theta_levels_from_fit`],
    /// so a model the binder refuses can still be measured.
    pub(super) struct Jac {
        xb: DMatrix<f64>,
        xo: DMatrix<f64>,
        z: DMatrix<f64>,
        /// Per random effect: its name, whether it is a kappa, its unit columns.
        res: Vec<(String, bool, DMatrix<f64>)>,
        labels: Vec<String>,
        one_column: bool,
    }

    /// Every block of `text` under `contrast = none`, on the levels `pop` shows.
    pub(super) fn none_layout(text: &str, pop: &Population) -> LevelBindings {
        let parsed = parse_full_model(text).unwrap();
        parsed
            .model
            .theta_blocks()
            .level_blocks()
            .iter()
            .map(|d| {
                let levels = discover_levels(d, pop).unwrap();
                let binding = crate::parser::model_parser::LevelBinding {
                    labels: levels.iter().map(|l| l.label(d.columns())).collect(),
                    groups: vec![0; levels.len()],
                    contrast: LevelContrast::Unconstrained,
                };
                (d.name().to_string(), binding)
            })
            .collect()
    }

    pub(super) fn jacobian(text: &str, pop0: &Population) -> Jac {
        jacobian_at(text, pop0, Point::ZERO)
    }

    /// Where [`jacobian_at`] evaluates the random effects: η_{s,e} =
    /// `scale`·`EV[(s + 2e) % 6]` and, when `per_occasion`, κ_{s,g,q} =
    /// `scale`·`CV[(3s + g + q) % 5]`, which differs between the occasion
    /// groups `g` of a subject; otherwise κ_{s,g,q} = `scale`·`CV[(3s + q) % 5]`,
    /// one nonzero value on every occasion of the subject.
    #[derive(Clone, Copy)]
    pub(super) struct Point {
        scale: f64,
        per_occasion: bool,
    }

    impl Point {
        pub(super) const ZERO: Point = Point {
            scale: 0.0,
            per_occasion: false,
        };
    }

    /// [`jacobian`] at the random effects of `at`, the θ still at its initial value.
    pub(super) fn jacobian_at(text: &str, pop0: &Population, at: Point) -> Jac {
        const EV: [f64; 6] = [0.4, -0.3, 0.7, -0.6, 0.2, 0.5];
        const CV: [f64; 5] = [0.6, -0.9, 0.3, 1.0, -0.4];
        let fitted = none_layout(text, pop0);
        let mut pop = pop0.clone();
        let mut parsed = parse_full_model(text).unwrap();
        bind_theta_levels_from_fit(&mut parsed, text, &mut pop, &fitted).expect("from_fit");
        let m = &parsed.model;
        let theta0 = m.default_params.theta.clone();
        let eta0 = |s: usize| -> Vec<f64> {
            (0..m.n_eta)
                .map(|e| at.scale * EV[(s + 2 * e) % 6])
                .collect()
        };
        let n_kappa = m.n_kappa;
        let groups: Vec<usize> = pop
            .subjects
            .iter()
            .map(|s| {
                crate::stats::likelihood::iov_occasion_groups(s)
                    .len()
                    .max(1)
            })
            .collect();
        let kappa0 = |s: usize| -> Vec<Vec<f64>> {
            (0..groups[s])
                .map(|g| {
                    let g = if at.per_occasion { g } else { 0 };
                    (0..n_kappa)
                        .map(|q| at.scale * CV[(3 * s + g + q) % 5])
                        .collect()
                })
                .collect()
        };
        let preds = |th: &[f64], s: usize, et: &[f64], ka: &[Vec<f64>]| {
            if n_kappa > 0 {
                crate::pk::predict_iov(m, &pop.subjects[s], th, et, ka)
            } else {
                crate::pk::compute_predictions_with_tv(m, &pop.subjects[s], th, et)
            }
        };
        let ns = pop.subjects.len();
        let lens: Vec<usize> = (0..ns)
            .map(|s| preds(&theta0, s, &eta0(s), &kappa0(s)).len())
            .collect();
        let nrow: usize = lens.iter().sum();
        let offs: Vec<usize> = lens
            .iter()
            .scan(0, |a, n| {
                let o = *a;
                *a += n;
                Some(o)
            })
            .collect();
        // One FD column: the difference of the predictions at two points, laid
        // into the rows of subject `s` (or of every subject when `s` is `None`).
        let fd = |only: Option<usize>, plus: &dyn Fn(usize) -> (Vec<f64>, Vec<f64>), h: f64| {
            let mut col = vec![0.0; nrow];
            for s in (0..ns).filter(|s| only.is_none_or(|o| o == *s)) {
                let (a, b) = plus(s);
                for j in 0..a.len() {
                    col[offs[s] + j] = (a[j] - b[j]) / (2.0 * h);
                }
            }
            col
        };
        let theta_col = |k: usize| -> Vec<f64> {
            let h = 1e-6 * theta0[k].abs().max(1.0);
            let (mut tp, mut tm) = (theta0.clone(), theta0.clone());
            tp[k] += h;
            tm[k] -= h;
            fd(
                None,
                &|s| {
                    (
                        preds(&tp, s, &eta0(s), &kappa0(s)),
                        preds(&tm, s, &eta0(s), &kappa0(s)),
                    )
                },
                h,
            )
        };
        let block: Vec<usize> = (0..m.n_theta)
            .filter(|&i| m.theta_names[i].starts_with("PLACEBO["))
            .collect();
        let other: Vec<usize> = (0..m.n_theta).filter(|i| !block.contains(i)).collect();
        let mat = |cols: &[Vec<f64>]| DMatrix::from_fn(nrow, cols.len(), |r, c| cols[c][r]);
        let h = 1e-6;
        let mut res = Vec::new();
        let mut zc = Vec::new();
        for e in 0..m.n_eta {
            let cols: Vec<Vec<f64>> = (0..ns)
                .map(|s| {
                    fd(
                        Some(s),
                        &|s| {
                            let (mut ep, mut em) = (eta0(s), eta0(s));
                            ep[e] += h;
                            em[e] -= h;
                            (
                                preds(&theta0, s, &ep, &kappa0(s)),
                                preds(&theta0, s, &em, &kappa0(s)),
                            )
                        },
                        h,
                    )
                })
                .collect();
            res.push((m.eta_names[e].clone(), false, mat(&cols)));
            zc.extend(cols);
        }
        for q in 0..n_kappa {
            let mut cols = Vec::new();
            for s in 0..ns {
                for g in 0..groups[s] {
                    let (mut kp, mut km) = (kappa0(s), kappa0(s));
                    kp[g][q] += h;
                    km[g][q] -= h;
                    cols.push(fd(
                        Some(s),
                        &|s| {
                            (
                                preds(&theta0, s, &eta0(s), &kp),
                                preds(&theta0, s, &eta0(s), &km),
                            )
                        },
                        h,
                    ));
                }
            }
            res.push((m.kappa_names[q].clone(), true, mat(&cols)));
        }
        let labels = fitted["PLACEBO"].labels.clone();
        Jac {
            xb: mat(&block.iter().map(|&k| theta_col(k)).collect::<Vec<_>>()),
            xo: mat(&other.iter().map(|&k| theta_col(k)).collect::<Vec<_>>()),
            z: mat(&zc),
            res,
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

    /// `(baseline rank, rank under contrast, free θ)`. The largest singular
    /// value of each side, the one that decides "rank 0", is pushed onto `seen`
    /// so the caller can report the gap around `RANK_TOL`.
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
        let largest = |sv: &[f64]| sv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        seen.extend(
            [largest(&base), largest(&under)]
                .into_iter()
                .filter(|v| v.is_finite()),
        );
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

    /// The clause every dead-level refusal with live levels left carries.
    const DEAD_LEVELS: &str =
        "has no effect on the predictions or the residual error at any of its records";

    /// Whether the block coded by `contrast` loses rank on its own: some free θ
    /// moves only levels with no effect (#1679, #1702 review F1). A dead level
    /// costs nothing where the coding reaches it only through the other levels —
    /// the reference under `ref`, one level per group under sum-to-zero.
    pub(super) fn block_loses_rank(j: &Jac, contrast: LevelContrast) -> bool {
        let xb = &j.xb * coding(j, contrast);
        rank(&residual_sv(&xb, &DMatrix::zeros(xb.nrows(), 0))) < xb.ncols()
    }

    /// T1. Over designs × shapes × contrasts, the binder refuses an explicit
    /// contrast exactly when the oracle does: the block absorbs the random
    /// effect under it (rank 0 against a positive baseline), or the coded block
    /// loses rank on its own ([`block_loses_rank`]). Cells whose contrast leaves
    /// no free θ are #1624's and skipped. `auto` must resolve to a contrast the
    /// oracle accepts, global whenever global is accepted, and be refused only
    /// when every contrast is. Exactly 4 cells have a level with no effect
    /// (`rank X_B < L`); on those the verdict straddles — the within-group
    /// contrast binds where no absorbed random effect takes the dead level's
    /// slack (H10, and G at 2 subjects per study) and is refused where one does
    /// (G at one) — and the dead cells are checked by the same oracle, not
    /// skipped.
    ///
    /// Mutations — `dead_levels` returns nothing (the dead cells bind under
    /// `none`); the dead failure ignores the contrast (within is refused on H10);
    /// the slack clause dropped (G within binds at one subject per study).
    #[test]
    fn binder_agrees_with_the_jacobian_oracle() {
        let mut wrong: Vec<String> = Vec::new();
        let mut dead_cells: Vec<String> = Vec::new();
        let (mut dead_bound, mut dead_refused) = (0usize, 0usize);
        let mut seen: Vec<f64> = Vec::new();
        let (mut absorbed_cells, mut free_cells) = (0usize, 0usize);
        for (dtag, cols, pop) in designs() {
            for tag in SHAPES {
                let jac = jacobian(&shape(tag, cols, "none"), &pop);
                // A shape that reads the block only through a factor of `TIME` (G,
                // H10) gives a block keyed on TIME a level with no effect on `y` at
                // TIME = 0, η or not (#1679).
                let dead = rank(&residual_sv(&jac.xb, &DMatrix::zeros(jac.xb.nrows(), 0)))
                    < jac.xb.ncols();
                if dead {
                    dead_cells.push(format!("{dtag} {tag}"));
                }
                let mut absorbs = HashMap::new();
                for (c, token) in EXPLICIT {
                    let (base, under, free) = oracle(&jac, c, &mut seen);
                    if free == 0 {
                        continue;
                    }
                    let absorbed = (base > 0 && under == 0) || block_loses_rank(&jac, c);
                    absorbed_cells += usize::from(absorbed);
                    free_cells += usize::from(!absorbed);
                    if dead {
                        dead_refused += usize::from(absorbed);
                        dead_bound += usize::from(!absorbed);
                    }
                    absorbs.insert(token, absorbed);
                    let got = try_bind(&shape(tag, cols, token), &pop);
                    if got.is_err() != absorbed {
                        wrong.push(format!(
                            "{dtag} {tag} {token}: oracle rank {under} of {base}, block rank \
                             loss {} ⇒ refuse={absorbed}, binder {got:?}",
                            block_loses_rank(&jac, c)
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
        assert_eq!(
            dead_cells,
            [
                "[STUDY,TIME] 1/study G",
                "[STUDY,TIME] 1/study H10",
                "[STUDY,TIME] 2/study G",
                "[STUDY,TIME] 2/study H10",
            ],
            "cells whose block has an unidentified level"
        );
        // The dead cells straddle: some contrasts carry their dead level, some
        // do not (measured: within binds on 3 of the 4, and every other
        // contrast is refused).
        assert_eq!(
            (dead_bound, dead_refused),
            (3, 13),
            "dead cells: bound / refused"
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

        // No `[scaling]`: `y` is the amount over `V`. The pre-#1650 `no_eta_model`.
        let ne = coupling(&no_eta_model().replace(
            "  V  = TVV\n  Z  = TVV * exp(ETA_V)",
            "  V  = TVV * exp(ETA_V)",
        ));
        assert_eq!((ne.funnels.len(), ne.reach), (0, via("V")), "η on V");
        assert_eq!(coupling(&no_eta_model()).reach, None, "no_eta_model");
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
        // An ODE model whose state reads a parameter with no canonical PK name:
        // only the `[odes]` line connects `KE` to the state.
        let ode = shape("S2", "STUDY", "")
            .replace("  CL = TVEMAX * exp(ETA_E0)", "  KE = TVEMAX * exp(ETA_E0)")
            .replace(
                "  pk one_cpt_iv(cl=CL, v=V)",
                "  ode(states=[central])\n\n[odes]\n  d/dt(central) = -KE * central",
            );
        assert_eq!(coupling(&ode).reach, Some(EtaRoute::State), "ODE via KE");
        // R1-2 (#1675 review): a parameter only the engine reads by name reaches
        // the states (an ODE model's bare `F`, an analytical `D1`); an unread one
        // with a canonical PK name (`KA` on `one_cpt_iv`) is dead, like `ZZ`.
        let pk_ip = |extra: &str| {
            scaling_model(&format!(
                "  CL = TVEMAX\n  V = TVET50\n  E0 = TVE0\n{extra}"
            ))
            .replace("PLACEBO[STUDY, TIME]", "PLACEBO[STUDY]")
        };
        for (tag, extra, want) in [
            ("unread KA", "  KA = TVEMAX * exp(ETA_E0)", None),
            ("unread ZZ", "  ZZ = TVEMAX * exp(ETA_E0)", None),
            (
                "analytical D1",
                "  D1 = TVEMAX * exp(ETA_E0)",
                Some(EtaRoute::State),
            ),
        ] {
            assert_eq!(coupling(&pk_ip(extra)).reach, want, "{tag}");
        }
        let ode_f = pk_ip("  F = TVEMAX * exp(ETA_E0)").replace(
            "  pk one_cpt_iv(cl=CL, v=V)",
            "  ode(states=[central])\n\n[odes]\n  d/dt(central) = -CL / V * central",
        );
        assert_eq!(coupling(&ode_f).reach, Some(EtaRoute::State), "ODE bare F");
        // ... and only those: an analytical model reads `F` only through `pk(...)`,
        // and an ODE model reads no other canonical name by itself.
        assert_eq!(
            coupling(&pk_ip("  F = TVEMAX * exp(ETA_E0)")).reach,
            None,
            "analytical unread F"
        );
        let ode_ka = pk_ip("  KA = TVEMAX * exp(ETA_E0)").replace(
            "  pk one_cpt_iv(cl=CL, v=V)",
            "  ode(states=[central])\n\n[odes]\n  d/dt(central) = -CL / V * central",
        );
        assert_eq!(coupling(&ode_ka).reach, None, "ODE unread KA");
        // R1-3: across readouts the most direct route wins. One reads the η
        // itself, the other only through the state.
        let routes = shape("S2", "STUDY", "")
            .replace(
                "  pk one_cpt_iv(cl=CL, v=V)",
                "  ode(states=[central])\n\n[odes]\n  d/dt(central) = -CL / V * central",
            )
            .replace(
                "  y = central / V + E0 + PLACEBO",
                "  y[CMT=1] = central / V + E0 + ETA_E0\n  y[CMT=2] = central / V + PLACEBO",
            )
            .replace(
                "  DV ~ additive(ADD)",
                "  CMT=1: DV ~ additive(ADD)\n  CMT=2: DV ~ additive(ADD)",
            );
        assert_eq!(
            coupling(&routes).reach,
            Some(EtaRoute::Direct),
            "Direct beats State"
        );
        // The binder side of the KA/ZZ pair, on the saturating block.
        let sat = |extra: &str| pk_ip(extra).replace("PLACEBO[STUDY]", "PLACEBO[STUDY, TIME]");
        let one = cf_pop(3, 1, &T6);
        for extra in ["  KA = TVEMAX * exp(ETA_E0)", "  ZZ = TVEMAX * exp(ETA_E0)"] {
            assert_eq!(
                try_bind(&sat(extra), &one),
                Ok((LevelContrast::SumToZero, 17)),
                "{extra}"
            );
        }
        // A random effect read only by an `if` condition still reaches `y`.
        let cond = cf_model(
            "",
            "STUDY",
            &format!("{BASE}  E0 = TVE0\n  if (ETA_E0 > 0) {{ E0 = TVE0 + 1 }}"),
            &format!("E0 + PLACEBO + {EMAXY}"),
        );
        assert_eq!(coupling(&cond).reach, via("E0"), "η in a condition");
        // Two readouts (an ODE model, which per-CMT error models need): the first has a readout funnel (`E0 + PLACEBO`), the
        // second reads the block time-varyingly, so no single expression carries
        // every route and there is no funnel.
        let two = scaling_model("  CL = TVEMAX\n  V = TVET50\n  E0 = TVE0 + ETA_E0")
            .replace("PLACEBO[STUDY, TIME]", "PLACEBO[STUDY]")
            .replace(
                "  pk one_cpt_iv(cl=CL, v=V)",
                "  ode(states=[central])\n\n[odes]\n  d/dt(central) = -CL / V * central",
            )
            .replace(
                "  y = central / V + E0 + PLACEBO",
                "  y[CMT=1] = central / V + E0 + PLACEBO\n  y[CMT=2] = central / V + PLACEBO * TIME",
            )
            .replace(
                "  DV ~ additive(ADD)",
                "  CMT=1: DV ~ additive(ADD)\n  CMT=2: DV ~ additive(ADD)",
            );
        let c = coupling(&two);
        assert_eq!((c.funnels.len(), c.reach), (0, via("E0")), "two readouts");
        // The control: the first readout alone has the funnel.
        let one_y = two
            .replace("\n  y[CMT=2] = central / V + PLACEBO * TIME", "")
            .replace("\n  CMT=2: DV ~ additive(ADD)", "");
        assert_eq!(coupling(&one_y).funnels.len(), 1, "one readout");
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

    const ONE_COLUMN_HEAD: &str =
        "theta PLACEBO[STUDY]: each `STUDY` level belongs to a single subject, and ";
    const ONE_COLUMN_WHY: &str = ", so a level and that subject's random effect are the same \
                                  quantity: the model is not identified under any contrast.";
    const ONE_COLUMN_FIX: &str = " Remove the block, or drop the random effect.";
    /// What a one-column refusal must not say: it has no groups, and no
    /// contrast rescues it.
    const ONE_COLUMN_NEVER: [&str; 5] = [
        "sum_to_zero_within",
        "group's mean",
        "leading",
        "nested",
        "` leaves",
    ];

    /// T3 (C1, C2, C3). A one-column block with one subject per level is
    /// refused under every contrast that leaves it a free θ, auto included,
    /// naming the funnel: an individual parameter (C1, with the variable that
    /// carried the η when there is one), or the readout (C2). On a single
    /// level, the plain single-level text, with no random-effect clause (C3).
    #[test]
    fn the_one_column_refusal_names_its_site() {
        let pop = cf_pop(3, 1, &T6);
        let h3 = |c: &str| {
            cf_model(
                c,
                "STUDY",
                &format!("{BASE}  E0 = TVE0 + ETA_E0\n  E1 = E0 + PLACEBO"),
                &format!("E1 + {EMAXY}"),
            )
        };
        for c in ["", "sum_to_zero", "sum_to_zero_within", "ref", "none"] {
            let tag = if c.is_empty() { "auto" } else { c };
            // C1: an individual parameter reads the block and the η.
            let err = try_bind(&shape("H1", "STUDY", c), &pop).expect_err(tag);
            has(
                &err,
                &[
                    &format!(
                        "{ONE_COLUMN_HEAD}the individual parameter `E0` reads this block and \
                         carries a random effect{ONE_COLUMN_WHY}"
                    ),
                    ONE_COLUMN_FIX,
                ],
            );
            lacks(&err, &ONE_COLUMN_NEVER);
            lacks(&err, &["(through"]);
            // C1 through a variable.
            let err = try_bind(&h3(c), &pop).expect_err(tag);
            has(
                &err,
                &[&format!(
                    "{ONE_COLUMN_HEAD}the individual parameter `E1` reads this block and \
                     carries a random effect (through `E0`){ONE_COLUMN_WHY}{ONE_COLUMN_FIX}"
                )],
            );
            // C2: the readout.
            let err = try_bind(&shape("H2", "STUDY", c), &pop).expect_err(tag);
            has(
                &err,
                &[&format!(
                    "{ONE_COLUMN_HEAD}the `y` readout reads this block and a random effect \
                     (through `E0`){ONE_COLUMN_WHY}{ONE_COLUMN_FIX}"
                )],
            );
            lacks(&err, &ONE_COLUMN_NEVER);
            lacks(&err, &["individual parameter"]);
        }

        // C3: one study, so a single level; every contrast but `none` leaves it
        // no free θ.
        let one = cf_pop(1, 1, &T6);
        let none = "Use `contrast = none` if a single constant is what you meant.";
        for (c, text) in [
            ("", "which sum-to-zero pins at 0. "),
            ("sum_to_zero", "which sum-to-zero pins at 0. "),
            ("ref", "which is the reference level, held at 0. "),
            (
                "sum_to_zero_within",
                "which the within-group sum-to-zero pins at 0. ",
            ),
        ] {
            let err = try_bind(&shape("H1", "STUDY", c), &one).expect_err(c);
            has(
                &err,
                &[
                    &format!("theta PLACEBO[STUDY]: the data carries a single level, {text}"),
                    none,
                ],
            );
            lacks(
                &err,
                &["random effect", "every  group", "every STUDY group"],
            );
        }
    }

    const USE_WITHIN: &str = " Use `contrast = sum_to_zero_within` (the default for this \
                              shape), or drop the random effect.";
    const SAME_QUANTITY: &str = " — the two are the same quantity, so the model is not identified.";
    const EVERY_OBSERVATION: &str = ", and the block takes a level at every observation, so \
                                     it can reproduce any effect that random effect has";

    /// T4 (C4). On `[STUDY, TIME]` at one subject per study, a random effect that
    /// never meets the block in one expression is still absorbed, and an
    /// explicit global contrast is refused naming its route: through the states
    /// (S2), through an individual parameter the readout reads (η on `V`, no
    /// `[scaling]`), or directly. Within-group sum-to-zero binds. With every
    /// group a single level, the zero-free refusal names the route too.
    #[test]
    fn a_state_route_refusal_names_the_state() {
        let pop = cf_pop(3, 1, &T6);
        let s2 = |c: &str| shape("S2", "STUDY, TIME", c);
        let via_v = |c: &str| {
            no_eta_model()
                .replace(
                    "  V  = TVV\n  Z  = TVV * exp(ETA_V)",
                    "  V  = TVV * exp(ETA_V)",
                )
                .replace(
                    "[STUDY, TIME]",
                    &if c.is_empty() {
                        "[STUDY, TIME]".to_string()
                    } else {
                        format!("[STUDY, TIME, contrast = {c}]")
                    },
                )
        };
        let direct = |c: &str| {
            shape("S2", "STUDY, TIME", c)
                .replace(
                    "  CL = TVEMAX * exp(ETA_E0)",
                    "  CL = TVEMAX * exp(PLACEBO)",
                )
                .replace(
                    "  y = central / V + E0 + PLACEBO",
                    "  y = central / V + E0 + ETA_E0",
                )
        };
        type Text = Box<dyn Fn(&str) -> String>;
        let cases: [(&str, Text, Population); 3] = [
            (
                "the random effect `ETA_E0` reaches `y` through the model's states",
                Box::new(s2),
                pop.clone(),
            ),
            (
                "the random effect `ETA_V` reaches `y` through `V`",
                Box::new(via_v),
                population(3, 6),
            ),
            (
                "the random effect `ETA_E0` reaches `y` directly",
                Box::new(direct),
                // Off `TIME = 0`: `CL` has no effect on a bolus read at its own dose
                // time, so on `T6` this block has dead levels (#1679), refused first.
                cf_pop(3, 1, &NO0),
            ),
        ];
        for (route, text, pop) in &cases {
            for c in ["sum_to_zero", "ref", "none"] {
                let err = try_bind(&text(c), pop).expect_err(c);
                has(
                    &err,
                    &[
                        &format!(
                            "theta PLACEBO[STUDY, TIME]: `contrast = {c}` leaves each STUDY \
                             group's mean free, but {route}{EVERY_OBSERVATION}{SAME_QUANTITY}"
                        ),
                        USE_WITHIN,
                    ],
                );
                lacks(&err, &["reads this block", "individual parameter"]);
            }
            let (contrast, free) = try_bind(&text(""), pop).expect("auto binds");
            assert_eq!(contrast, LevelContrast::SumToZeroWithin, "{route}");
            assert!(free > 0, "{route}");
        }

        // An η that shares an expression with the block but has no funnel (H6) is
        // named by that expression, not by its route.
        let err = try_bind(&shape("H6", "STUDY, TIME", "sum_to_zero"), &pop).expect_err("H6");
        has(
            &err,
            &[
                "but the `y` readout reads this block and a random effect (through `EMAX`) at \
               that grouping",
            ],
        );
        lacks(&err, &["reaches `y`", EVERY_OBSERVATION]);

        // Every group a single level: the zero-free refusal names the route.
        let err = try_bind(&s2(""), &cf_pop(3, 1, &[1.0])).expect_err("single time");
        has(
            &err,
            &[
                "theta PLACEBO[STUDY, TIME]: every STUDY group has a single level, and the \
               random effect `ETA_E0` reaches `y` through the model's states — the random \
               effect already carries each group's value, so the block estimates nothing. \
               Remove the block.",
            ],
        );
    }

    /// T5. A funnel holds only when the covariates it reads are constant within
    /// every subject. `E0 = (TVE0 + PLACEBO) * (WT / 70) + ETA_E0` on a
    /// one-column block: with `WT` constant per subject the η and the level are
    /// proportional and the block is refused; with `WT` changing between a
    /// subject's records they are not, and it binds. The oracle agrees on both
    /// sides, in one test.
    #[test]
    fn a_funnel_covariate_must_be_subject_constant() {
        let text = |c: &str| {
            cf_model(
                c,
                "STUDY",
                &format!("{BASE}  E0 = (TVE0 + PLACEBO) * (WT / 70) + ETA_E0"),
                &format!("E0 + {EMAXY}"),
            )
        };
        let mut constant = cf_pop(3, 1, &T6);
        for (k, s) in constant.subjects.iter_mut().enumerate() {
            s.covariates.insert("WT".into(), 60.0 + 10.0 * k as f64);
        }
        let mut varying = cf_pop(3, 1, &T6);
        for (k, s) in varying.subjects.iter_mut().enumerate() {
            s.obs_covariates = T6
                .iter()
                .map(|t| HashMap::from([("WT".to_string(), 60.0 + 5.0 * t + k as f64)]))
                .collect();
        }
        // A covariate read only by an `if` condition counts too: the branches
        // scale the level differently, so with `WT` crossing 65 inside a subject
        // the level and the η stop being proportional.
        let branch = |c: &str| {
            cf_model(
                c,
                "STUDY",
                &format!(
                    "{BASE}  E0 = TVE0 + PLACEBO + ETA_E0\n  \
                     if (WT > 65) {{ E0 = TVE0 + 2 * PLACEBO + ETA_E0 }}"
                ),
                &format!("E0 + {EMAXY}"),
            )
        };
        let cases: [(&str, &dyn Fn(&str) -> String, &Population, bool); 4] = [
            ("constant WT", &text, &constant, true),
            ("varying WT", &text, &varying, false),
            ("constant WT, if", &branch, &constant, true),
            ("varying WT, if", &branch, &varying, false),
        ];
        for (tag, text, pop, refused) in cases {
            let jac = jacobian(&text("none"), pop);
            let (base, under, _) = oracle(&jac, LevelContrast::SumToZero, &mut Vec::new());
            assert!(base > 0, "{tag}: the η is identified without the block");
            assert_eq!(under == 0, refused, "{tag}: oracle rank {under} of {base}");
            let got = try_bind(&text(""), pop);
            assert_eq!(got.is_err(), refused, "{tag}: {got:?}");
        }
    }

    /// T6. The fixtures that stand for "no random effect on the block's scale"
    /// are identified by the oracle: `no_eta_model` on one subject per study,
    /// and the integration fixture of `tests/theta_level_blocks.rs` on its own
    /// design. Under every explicit contrast leaving a free θ the rank equals
    /// the block-free baseline and the binder accepts; `auto` is global.
    #[test]
    fn the_no_eta_controls_are_identified() {
        let integration = no_eta_model()
            .replace(
                "theta TVCL(2.0, 0.001, 10.0)",
                "theta TVCL(2.0, 0.001, 20.0)",
            )
            .replace(
                "PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)",
                "PLACEBO[STUDY, TIME](0.0, -5.0, 5.0)",
            );
        let mut design = population(2, 3);
        for s in design.subjects.iter_mut() {
            s.obs_times = vec![1.0, 4.0, 12.0];
        }
        for (tag, base_text, pop) in [
            ("no_eta_model", no_eta_model(), population(3, 6)),
            ("integration", integration, design),
        ] {
            let with = |c: &str| {
                base_text.replace("[STUDY, TIME]", &format!("[STUDY, TIME, contrast = {c}]"))
            };
            let jac = jacobian(&with("none"), &pop);
            for (c, token) in EXPLICIT {
                let (base, under, free) = oracle(&jac, c, &mut Vec::new());
                if free == 0 {
                    continue;
                }
                assert_eq!(under, base, "{tag} {token}: the block absorbs the η");
                assert!(try_bind(&with(token), &pop).is_ok(), "{tag} {token}");
            }
            assert_eq!(
                try_bind(&base_text, &pop).map(|(c, _)| c),
                Ok(LevelContrast::SumToZero),
                "{tag} auto"
            );
        }
    }

    /// T7. A fit bound before #1650 resolved S2 to global sum-to-zero (17 free
    /// θ); bound now it is within (15). Its saved bindings still drive it:
    /// [`bind_theta_levels_from_fit`] imposes the stored layout and never
    /// re-resolves the contrast.
    #[test]
    fn a_fit_bound_before_1649_rebinds_its_own_layout() {
        use crate::api::theta_level_values;
        use crate::parser::model_parser::LevelBinding;
        let text = shape("S2", "STUDY, TIME", "");
        let today = try_bind(&text, &cf_pop(3, 1, &T6));
        assert_eq!(today, Ok((LevelContrast::SumToZeroWithin, 15)), "today");
        let mut p = cf_pop(3, 1, &T6);
        let mut parsed = parse_full_model(&text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, &text, &mut p).unwrap();
        let labels = parsed.bindings.levels["PLACEBO"].labels.clone();
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

    /// `[block | X]` as one matrix.
    fn hcat(parts: &[&DMatrix<f64>]) -> DMatrix<f64> {
        let nrows = parts.first().map_or(0, |p| p.nrows());
        let ncols = parts.iter().map(|p| p.ncols()).sum();
        let mut x = DMatrix::zeros(nrows, ncols);
        let mut at = 0;
        for p in parts {
            x.columns_mut(at, p.ncols()).copy_from(p);
            at += p.ncols();
        }
        x
    }

    /// The rank random effect `name` keeps outside the other θ, the random
    /// effects at a coarser-or-equal unit (every η for a kappa; the other η for
    /// an η), and — under `Some(contrast)` — the block coded by `contrast`.
    pub(super) fn unit_rank(j: &Jac, name: &str, contrast: Option<LevelContrast>) -> usize {
        rank(&unit_sv(j, name, contrast))
    }

    /// The singular values behind [`unit_rank`], normalised by `‖Z_name‖_F`.
    fn unit_sv(j: &Jac, name: &str, contrast: Option<LevelContrast>) -> Vec<f64> {
        let (r, (_, kappa, zr)) = j
            .res
            .iter()
            .enumerate()
            .find(|(_, (n, ..))| n == name)
            .unwrap_or_else(|| panic!("no random effect {name}"));
        let mut parts: Vec<&DMatrix<f64>> = vec![&j.xo];
        let xb = contrast.map(|c| &j.xb * coding(j, c));
        parts.extend(xb.as_ref());
        parts.extend(
            j.res
                .iter()
                .enumerate()
                .filter(|(i, (_, k, _))| *i != r && (*kappa || !*k))
                .map(|(_, (.., z))| z),
        );
        residual_sv(zr, &hcat(&parts))
    }

    /// The joint oracle (#1678, #1696): the random effects `contrast` absorbs —
    /// each `r` with `rank P⊥(X_c, Z_coarser) Z_r = 0 < rank P⊥(X_o, Z_coarser) Z_r` —
    /// and the block's free θ under it. With one random effect this is [`oracle`].
    pub(super) fn joint_oracle(j: &Jac, contrast: LevelContrast) -> (Vec<String>, usize) {
        let free = coding(j, contrast).ncols();
        let absorbed = j
            .res
            .iter()
            .filter(|(name, ..)| {
                unit_rank(j, name, None) > 0 && unit_rank(j, name, Some(contrast)) == 0
            })
            .map(|(name, ..)| name.clone())
            .collect();
        (absorbed, free)
    }

    /// Whether the binder agrees with [`joint_oracle`] on `text(contrast)` over
    /// `pop`, by T1's rule: an explicit contrast leaving a free θ is refused
    /// exactly when the oracle absorbs something under it, and `auto` resolves
    /// to a contrast that absorbs nothing — global whenever global absorbs
    /// nothing — and is refused only when every contrast absorbs. Disagreements
    /// are pushed onto `wrong`.
    pub(super) fn agrees_with_joint_oracle(
        tag: &str,
        text: &dyn Fn(&str) -> String,
        pop: &Population,
        wrong: &mut Vec<String>,
    ) {
        let jac = jacobian(&text("none"), pop);
        let mut absorbs = HashMap::new();
        for (c, token) in EXPLICIT {
            let (absorbed, free) = joint_oracle(&jac, c);
            if free == 0 {
                continue;
            }
            let refuse = !absorbed.is_empty() || block_loses_rank(&jac, c);
            absorbs.insert(token, refuse);
            let got = try_bind(&text(token), pop);
            if got.is_err() != refuse {
                wrong.push(format!(
                    "{tag} {token}: oracle absorbs {absorbed:?}, binder {got:?}"
                ));
            }
        }
        let all_absorb = absorbs.values().all(|a| *a);
        match try_bind(&text(""), pop) {
            Err(e) if !all_absorb => wrong.push(format!(
                "{tag} auto: refused, but some contrast absorbs nothing: {e}"
            )),
            Err(_) => {}
            Ok(_) if all_absorb => {
                wrong.push(format!("{tag} auto: bound, but every contrast absorbs"))
            }
            Ok((resolved, _)) => {
                let token = EXPLICIT.iter().find(|(c, _)| *c == resolved).unwrap().1;
                if absorbs.get(token).copied().unwrap_or(false) {
                    wrong.push(format!("{tag} auto → {token}, which absorbs"));
                }
                if absorbs.get("sum_to_zero") == Some(&false)
                    && resolved != LevelContrast::SumToZero
                {
                    wrong.push(format!("{tag} auto → {token}, but global absorbs nothing"));
                }
            }
        }
    }

    /// `cf_pop` with `occ` occasions per subject, also written as an `OCC`
    /// column. As **arms** every occasion carries the whole grid (the MBMA
    /// layout: records at one time, one per arm); as **periods** the occasions
    /// split the grid into consecutive runs.
    pub(super) fn occ_pop(
        n: usize,
        per: usize,
        times: &[f64],
        occ: usize,
        arms: bool,
    ) -> Population {
        let mut p = cf_pop(n, per, times);
        for s in p.subjects.iter_mut() {
            let rows: Vec<(f64, u32)> = if arms {
                times
                    .iter()
                    .flat_map(|&t| (1..=occ as u32).map(move |o| (t, o)))
                    .collect()
            } else {
                let run = times.len().div_ceil(occ);
                times
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| (t, (i / run + 1) as u32))
                    .collect()
            };
            let n_obs = rows.len();
            s.obs_times = rows.iter().map(|r| r.0).collect();
            s.occasions = rows.iter().map(|r| r.1).collect();
            s.obs_covariates = rows
                .iter()
                .map(|r| HashMap::from([("OCC".to_string(), r.1 as f64)]))
                .collect();
            s.observations = vec![1.0; n_obs];
            s.obs_cmts = vec![1; n_obs];
            s.cens = vec![0; n_obs];
        }
        p
    }

    /// The kappa shapes of the #1678 grid, on `KAPPA_E0`, with `ETA_E0` on `E0`
    /// when `eta` (else on an unread `Z`): K-E0 (kappa and block on `E0`), K-y
    /// (kappa on `E0`, block in `y`), K-EMAX (kappa on `EMAX`, block on `E0`),
    /// K-unread (kappa on a parameter `y` never reads).
    pub(super) fn kshape(tag: &str, eta: bool, cols: &str, contrast: &str) -> String {
        let e = if eta { " + ETA_E0" } else { "" };
        let unread = if eta {
            ""
        } else {
            "\n  Z = TVE0 * exp(ETA_E0)"
        };
        let (ip, y) = match tag {
            "K-E0" => (
                format!("{BASE}  E0 = TVE0 + PLACEBO + KAPPA_E0{e}"),
                format!("E0 + {EMAXY}"),
            ),
            "K-y" => (
                format!("{BASE}  E0 = TVE0 + KAPPA_E0{e}"),
                format!("E0 + PLACEBO + {EMAXY}"),
            ),
            "K-EMAX" => (
                format!("  EMAX = TVEMAX + KAPPA_E0\n  ET50 = TVET50\n  E0 = TVE0 + PLACEBO{e}"),
                format!("E0 + {EMAXY}"),
            ),
            "K-unread" => (
                format!("{BASE}  E0 = TVE0 + PLACEBO{e}\n  ZK = TVE0 * exp(KAPPA_E0)"),
                format!("E0 + {EMAXY}"),
            ),
            _ => unreachable!("{tag}"),
        };
        cf_model(contrast, cols, &format!("{ip}{unread}"), &y).replace(
            "  omega ETA_E0 ~ 0.1\n",
            "  omega ETA_E0 ~ 0.1\n  kappa KAPPA_E0 ~ 0.1\n",
        )
    }

    /// The K3′ sentence: a kappa that is its η by another name.
    const KAPPA_IS_ETA: &str = "every subject has a single occasion, so `KAPPA_E0` cannot be \
                                told apart from `ETA_E0` with or without this block: the model \
                                is not identified.";

    /// K1 (#1678). One model per verdict class of the kappa grid, against the
    /// joint oracle: a kappa is absorbed when the block's levels nest in its
    /// occasions, which a one-occasion subject, a period layout and a
    /// `[STUDY, OCC]` block give, and an arm layout or two subjects per study do
    /// not. The full grid is `the_full_kappa_grid_agrees_with_the_joint_oracle`.
    ///
    /// Mutations — the #1642 rule restored for a kappa (a nested block with a
    /// shared expression: the arms cells are refused under global); the
    /// subject as a kappa's unit (the period cells bind global, the arm cells
    /// go within).
    #[test]
    fn a_kappa_counts_when_the_levels_nest_in_its_occasions() {
        type Cell = (&'static str, &'static str, bool, &'static str, Population);
        let st = "STUDY, TIME";
        let so = "STUDY, OCC";
        let cells: Vec<Cell> = vec![
            (
                "K-E0",
                "STUDY",
                false,
                "1-col 1 occ",
                occ_pop(3, 1, &T6, 1, false),
            ),
            (
                "K-E0",
                "STUDY",
                false,
                "1-col 2 periods",
                occ_pop(3, 1, &T6, 2, false),
            ),
            (
                "K-E0",
                "STUDY",
                true,
                "1-col 3 arms",
                occ_pop(3, 1, &T6, 3, true),
            ),
            (
                "K-E0",
                st,
                false,
                "1/st 2 periods",
                occ_pop(3, 1, &T6, 2, false),
            ),
            (
                "K-E0",
                st,
                true,
                "1/st 2 periods",
                occ_pop(3, 1, &T6, 2, false),
            ),
            (
                "K-E0",
                st,
                false,
                "1/st 2 arms",
                occ_pop(3, 1, &T6, 2, true),
            ),
            ("K-E0", st, true, "1/st 2 arms", occ_pop(3, 1, &T6, 2, true)),
            (
                "K-E0",
                so,
                false,
                "1/st 2 periods",
                occ_pop(3, 1, &T6, 2, false),
            ),
            ("K-E0", so, true, "1/st 3 arms", occ_pop(3, 1, &T6, 3, true)),
            (
                "K-E0",
                so,
                false,
                "2/st 2 periods",
                occ_pop(3, 2, &T6, 2, false),
            ),
            (
                "K-y",
                st,
                false,
                "1/st 2 periods",
                occ_pop(3, 1, &T6, 2, false),
            ),
            (
                "K-EMAX",
                st,
                false,
                "1/st 3 periods",
                occ_pop(3, 1, &T6, 3, false),
            ),
            (
                "K-EMAX",
                st,
                true,
                "1/st 1 occ",
                occ_pop(3, 1, &T6, 1, false),
            ),
            (
                "K-unread",
                st,
                false,
                "1/st 2 periods",
                occ_pop(3, 1, &T6, 2, false),
            ),
            (
                "K-unread",
                st,
                true,
                "1/st 2 arms",
                occ_pop(3, 1, &T6, 2, true),
            ),
            ("K-E0", st, true, "2/st 2 arms", occ_pop(3, 2, &T6, 2, true)),
        ];
        let mut wrong = Vec::new();
        let mut refused = 0;
        for (shape, cols, eta, design, pop) in &cells {
            let tag = format!("{shape} η={eta} [{cols}] {design}");
            let text = |c: &str| kshape(shape, *eta, cols, c);
            agrees_with_joint_oracle(&tag, &text, pop, &mut wrong);
            refused += usize::from(try_bind(&text("sum_to_zero"), pop).is_err());
        }
        assert!(
            wrong.is_empty(),
            "{} cells disagree:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
        // Both sides of the gate: global is refused on some cells and not others.
        assert!(
            (4..cells.len() - 4).contains(&refused),
            "{refused} refused under global"
        );
    }

    /// K2 (#1678, #1696). The differential pair: η and κ on `E0`, one subject
    /// per study, arms. Keyed on `[STUDY, OCC]` the levels nest in the occasions
    /// too, so both random effects are absorbed and the block is refused under
    /// every contrast; keyed on `[STUDY, TIME]` each level holds every arm, so
    /// only the η is, and the block takes the within-study contrast. Before the
    /// fix both took within. The oracle straddles: κ⊥η keeps 3 of 3 under
    /// within on `[STUDY, TIME]` and 0 on `[STUDY, OCC]`.
    ///
    /// Mutation — drop the joint clause: `[STUDY, OCC]` binds within.
    #[test]
    fn a_kappa_and_an_eta_absorbed_together_are_refused() {
        let pop = occ_pop(3, 1, &T6, 2, true);
        let within = Some(LevelContrast::SumToZeroWithin);
        let by_occ = |c: &str| kshape("K-E0", true, "STUDY, OCC", c);
        let by_time = |c: &str| kshape("K-E0", true, "STUDY, TIME", c);
        let (jo, jt) = (
            jacobian(&by_occ("none"), &pop),
            jacobian(&by_time("none"), &pop),
        );
        assert_eq!(
            unit_rank(&jo, "KAPPA_E0", None),
            3,
            "κ⊥η base, [STUDY, OCC]"
        );
        assert_eq!(
            unit_rank(&jo, "KAPPA_E0", within),
            0,
            "κ⊥η within, [STUDY, OCC]"
        );
        assert_eq!(
            unit_rank(&jt, "KAPPA_E0", None),
            3,
            "κ⊥η base, [STUDY, TIME]"
        );
        assert_eq!(
            unit_rank(&jt, "KAPPA_E0", within),
            3,
            "κ⊥η within, [STUDY, TIME]"
        );
        for c in ["", "sum_to_zero", "sum_to_zero_within", "ref", "none"] {
            let err = try_bind(&by_occ(c), &pop).expect_err(c);
            assert!(
                err.contains("the levels absorb `ETA_E0` and `KAPPA_E0` together"),
                "{err}"
            );
        }
        assert_eq!(
            try_bind(&by_time(""), &pop),
            Ok((LevelContrast::SumToZeroWithin, 15))
        );
    }

    /// K3 (#1678). A kappa's funnel holds when the covariates it reads are
    /// constant within each **occasion**: `E0 = TVE0 + PLACEBO * X + KAPPA_E0` on
    /// `[STUDY, OCC]`, one subject per study, 3 periods. With `X` constant within
    /// each occasion (but not within the subject) the kappa is absorbed and auto
    /// goes within; with `X` varying per record it is not, and auto is global.
    /// The oracle agrees on both sides.
    ///
    /// Mutation — `constant_within_units` reads the subject for a kappa: the
    /// occasion-constant `X` changes within the subject, the funnel fails, and
    /// that side binds global.
    #[test]
    fn a_kappa_funnel_is_measured_per_occasion() {
        let text = |c: &str| {
            cf_model(
                c,
                "STUDY, OCC",
                &format!("{BASE}  E0 = TVE0 + PLACEBO * X + KAPPA_E0\n  Z = TVE0 * exp(ETA_E0)"),
                &format!("E0 + {EMAXY}"),
            )
            .replace(
                "  omega ETA_E0 ~ 0.1\n",
                "  omega ETA_E0 ~ 0.1\n  kappa KAPPA_E0 ~ 0.1\n",
            )
        };
        let with_x = |per_record: bool| {
            let mut p = occ_pop(3, 1, &T6, 3, false);
            for s in p.subjects.iter_mut() {
                let occ = s.occasions.clone();
                for (j, m) in s.obs_covariates.iter_mut().enumerate() {
                    let x = if per_record {
                        1.0 + 0.3 * j as f64
                    } else {
                        1.0 + occ[j] as f64
                    };
                    m.insert("X".to_string(), x);
                }
            }
            p
        };
        for (tag, per_record, want) in [
            ("occasion-constant X", false, LevelContrast::SumToZeroWithin),
            ("per-record X", true, LevelContrast::SumToZero),
        ] {
            let pop = with_x(per_record);
            let jac = jacobian(&text("none"), &pop);
            let (absorbed, _) = joint_oracle(&jac, LevelContrast::SumToZero);
            assert_eq!(
                absorbed.is_empty(),
                per_record,
                "{tag}: oracle {absorbed:?}"
            );
            assert_eq!(try_bind(&text(""), &pop).map(|(c, _)| c), Ok(want), "{tag}");
        }
    }

    /// K5 (#1678). Occasions derived by `iov_occasion = time(3)` count exactly as
    /// the same occasions read from a column: on `[STUDY, TIME]`, one subject per
    /// study, two periods, the kappa is absorbed either way and auto goes
    /// within. With no occasions at all (the column twin on data that carries
    /// none) the kappa counts nowhere, and auto is global — `fit()` refuses that
    /// model later for its missing occasions.
    ///
    /// Mutation — skip the derivation in `bind_theta_levels`: the derived twin
    /// has no occasions and binds global.
    #[test]
    fn derived_occasions_count_like_a_column() {
        let base = kshape("K-E0", false, "STUDY, TIME", "");
        let derived = format!("{base}\n[fit_options]\n  iov_occasion = time(3)\n");
        let column = format!("{base}\n[fit_options]\n  iov_column = OCC\n");
        let periods = occ_pop(3, 1, &T6, 2, false);
        let bare = cf_pop(3, 1, &T6);
        assert!(bare.subjects.iter().all(|s| s.occasions.is_empty()));
        let within = Ok((LevelContrast::SumToZeroWithin, 15));
        assert_eq!(try_bind(&column, &periods), within, "column");
        assert_eq!(try_bind(&derived, &bare), within, "derived");
        assert_eq!(
            try_bind(&column, &bare),
            Ok((LevelContrast::SumToZero, 17)),
            "no occasions"
        );
    }

    /// N1 (#1696a). Two subjects per study on disjoint times: the `STUDY` key
    /// does not identify subjects, but every `(STUDY, TIME)` level lies in one
    /// subject, so the η is absorbed — global sum-to-zero is refused and auto
    /// goes within. With the two subjects sharing the times no level nests and
    /// the block binds global. The oracle agrees on both.
    ///
    /// Mutation — restore `subject_key_identifies_subjects`: the disjoint
    /// design binds global.
    #[test]
    fn levels_nested_in_subjects_absorb_the_eta() {
        let text = |c: &str| shape("H1", "STUDY, TIME", c);
        let mut disjoint = cf_pop(3, 2, &T6);
        for (k, s) in disjoint.subjects.iter_mut().enumerate() {
            s.obs_times = if k % 2 == 0 {
                vec![0.0, 1.0, 2.0]
            } else {
                vec![4.0, 8.0, 12.0]
            };
            s.observations = vec![1.0; 3];
            s.obs_cmts = vec![1; 3];
            s.cens = vec![0; 3];
        }
        let shared = cf_pop(3, 2, &T6);
        for (tag, pop, absorbed) in [("disjoint", &disjoint, true), ("shared", &shared, false)] {
            let jac = jacobian(&text("none"), pop);
            let (got, _) = joint_oracle(&jac, LevelContrast::SumToZero);
            assert_eq!(!got.is_empty(), absorbed, "{tag}: oracle {got:?}");
            assert_eq!(
                try_bind(&text("sum_to_zero"), pop).is_err(),
                absorbed,
                "{tag}"
            );
        }
        assert_eq!(
            try_bind(&text(""), &disjoint),
            Ok((LevelContrast::SumToZeroWithin, 15))
        );
        assert_eq!(
            try_bind(&text(""), &shared),
            Ok((LevelContrast::SumToZero, 17))
        );
    }

    /// J1 (#1696b). Two η absorbed together: `ETA_E0` on `E0` and `ETA_EM` on
    /// `EMAX`, `[STUDY, TIME]`, one subject per study. Each alone takes within,
    /// but within leaves each subject's mean to one of them while the levels
    /// reproduce the other (`ETA_EM` beyond `ETA_E0`: 2 under no block, 0 under
    /// within), so every contrast is refused. Two subjects per study: no level
    /// nests, and the block binds global.
    ///
    /// Mutation — drop the joint clause: one subject per study binds within.
    #[test]
    fn two_etas_absorbed_together_are_refused() {
        let text = |c: &str| {
            cf_model(
                c,
                "STUDY, TIME",
                "  EMAX = TVEMAX + ETA_EM\n  ET50 = TVET50\n  E0 = TVE0 + PLACEBO + ETA_E0",
                &format!("E0 + {EMAXY}"),
            )
            .replace(
                "  omega ETA_E0 ~ 0.1\n",
                "  omega ETA_E0 ~ 0.1\n  omega ETA_EM ~ 0.1\n",
            )
        };
        let one = cf_pop(3, 1, &T6);
        let jac = jacobian(&text("none"), &one);
        assert_eq!(unit_rank(&jac, "ETA_EM", None), 2);
        assert_eq!(
            unit_rank(&jac, "ETA_EM", Some(LevelContrast::SumToZeroWithin)),
            0
        );
        for c in ["", "sum_to_zero", "sum_to_zero_within", "ref", "none"] {
            let err = try_bind(&text(c), &one).expect_err(c);
            has(&err, &["the levels absorb `ETA_E0` and `ETA_EM` together"]);
        }
        assert_eq!(
            try_bind(&text(""), &cf_pop(3, 2, &T6)),
            Ok((LevelContrast::SumToZero, 17))
        );
        // Each alone is fine under within: the refusal is the pair's.
        let within = Ok((LevelContrast::SumToZeroWithin, 15));
        let alone_e0 = text("").replace("TVEMAX + ETA_EM", "TVEMAX");
        let alone_em = text("").replace("PLACEBO + ETA_E0", "PLACEBO");
        assert_eq!(try_bind(&alone_e0, &one), within, "ETA_E0 alone");
        assert_eq!(try_bind(&alone_em, &one), within, "ETA_EM alone");
    }

    /// The kappa and joint refusals, sentence by sentence (message cells K1,
    /// K2, K2′, K3, K3′ of the #1679 plan). Each conditional sentence has both
    /// sides of its gate here: the one-column kappa against its η twin, K2
    /// against K2′, K3′ against the same model on two occasions and against a
    /// kappa on another parameter.
    #[test]
    fn the_kappa_refusals_name_the_occasion() {
        const OCCASION_HEAD: &str =
            "theta PLACEBO[STUDY]: each `STUDY` level lies within a single occasion of one \
             subject, and the individual parameter `E0` reads this block and carries a random \
             effect, so a level and that occasion's random effect are the same quantity: the \
             model is not identified under any contrast.";
        // K1: one-column, one occasion, kappa only — against its η twin.
        let one_occ = occ_pop(3, 1, &T6, 1, false);
        for c in ["", "sum_to_zero", "ref", "none"] {
            let err = try_bind(&kshape("K-E0", false, "STUDY", c), &one_occ).expect_err(c);
            has(&err, &[OCCASION_HEAD, ONE_COLUMN_FIX]);
            lacks(&err, &["belongs to a single subject", "that subject's"]);
            let eta = try_bind(&shape("H1", "STUDY", c), &one_occ).expect_err(c);
            has(
                &eta,
                &[
                    "belongs to a single subject",
                    "that subject's random effect",
                ],
            );
            lacks(&eta, &["occasion"]);
        }

        // K2: nested, kappa alone, explicit global — within is the advice.
        let periods = occ_pop(3, 1, &T6, 2, false);
        for c in ["sum_to_zero", "ref", "none"] {
            let err = try_bind(&kshape("K-E0", false, "STUDY, TIME", c), &periods).expect_err(c);
            has(
                &err,
                &[
                    &format!(
                        "theta PLACEBO[STUDY, TIME]: `contrast = {c}` lets the levels reproduce \
                         `KAPPA_E0` at every occasion: the individual parameter `E0` reads this \
                         block and carries a random effect at that grouping{SAME_QUANTITY}"
                    ),
                    USE_WITHIN,
                ],
            );
            lacks(&err, &["group's mean free"]);
        }
        // K2′: the same, with a single time per study, so within leaves nothing.
        let single = occ_pop(3, 1, &[1.0], 1, false);
        let err = try_bind(
            &kshape("K-E0", false, "STUDY, TIME", "sum_to_zero"),
            &single,
        )
        .expect_err("single time");
        has(
            &err,
            &[
                "lets the levels reproduce `KAPPA_E0` at every occasion",
                " Every STUDY group has a single level, so the random effect already carries \
                 each group's value: remove the block, or drop the random effect.",
            ],
        );
        lacks(&err, &["Use `contrast"]);

        // K3: an η and a kappa absorbed together, on two periods.
        for c in ["", "sum_to_zero", "sum_to_zero_within"] {
            let err = try_bind(&kshape("K-E0", true, "STUDY, TIME", c), &periods).expect_err(c);
            has(
                &err,
                &[
                    "theta PLACEBO[STUDY, TIME]: the levels absorb `ETA_E0` and `KAPPA_E0` \
                     together (the individual parameter `E0` reads this block and carries a \
                     random effect).",
                    " `contrast = sum_to_zero_within` leaves each subject's mean to one of them, \
                     and the levels reproduce the other, so the model is not identified under \
                     any contrast.",
                    " Drop one of the random effects, or remove the block.",
                ],
            );
            lacks(&err, &["Use `contrast", "single occasion"]);
        }
        // K3′: the same model on one occasion — the kappa is the η by another name.
        let err = try_bind(&kshape("K-E0", true, "STUDY, TIME", ""), &one_occ).expect_err("K3′");
        has(
            &err,
            &[KAPPA_IS_ETA, " Drop one of the two random effects."],
        );
        lacks(&err, &["absorb", "remove the block", "within"]);
        // ... but not when the kappa sits on another parameter: then the block is
        // what confuses them.
        let err =
            try_bind(&kshape("K-EMAX", true, "STUDY, TIME", ""), &one_occ).expect_err("K-EMAX");
        has(
            &err,
            &["the levels absorb `ETA_E0` and `KAPPA_E0` together ("],
        );
        lacks(&err, &["single occasion"]);
    }

    /// The labels of `[STUDY, TIME]`'s `TIME = 0` levels on `n` studies.
    fn time_zero_labels(n: usize) -> Vec<String> {
        (1..=n).map(|s| format!("`STUDY={s},TIME=0`")).collect()
    }

    /// G without a random effect on the block's scale: `EMAX` reads the block,
    /// and `y` reads `EMAX` only through a factor of `TIME`.
    fn g_no_eta(cols: &str, c: &str) -> String {
        cf_model(
            c,
            cols,
            "  EMAX = TVEMAX + PLACEBO\n  ET50 = TVET50\n  E0 = TVE0\n  Z = TVE0 * exp(ETA_E0)",
            &format!("E0 + {EMAXY}"),
        )
    }

    /// D1 (#1679, #1702 review F1). G with no η on `[STUDY, TIME]`, three
    /// studies: on `T6` each study's `TIME = 0` level has no effect (the oracle:
    /// `rank X_B` 15 of 18, against 18 of 18 on the "no 0" grid). What that
    /// costs depends on the contrast. Global sum-to-zero (three dead levels in
    /// its one group), `ref` (two outside the reference) and `none` lose rank
    /// and are refused; within (one per group) carries them, and auto takes it.
    /// On the "no 0" grid auto is global. Before the fix both grids bound
    /// global (17), so the pair straddles. And a one-column `[TIME]` block, whose
    /// one dead level is the reference and the sum-to-zero's derived level at
    /// once, binds under `ref` where its `[STUDY, TIME]` twin is refused.
    ///
    /// Every spelling that carries the block to `y` is held to the same oracle,
    /// over every contrast and on both grids — an exponential time course
    /// (G-exp), an ODE state at its initial value, a PK `CL` at its own dose
    /// time, `[TIME]` — so a check that only looked for a product with `TIME`
    /// fails on all but G. The PK cell is dead by the time-varying covariate
    /// convention (a record's value governs the interval ending at it), which
    /// oracle and binder share; the NONMEM anchor nearest to it is
    /// `tests/tvcov_intermediate_nonmem.rs` (an `EVID=2` covariate change on
    /// ADVAN3), which pins that convention for an analytical model, not for a
    /// level block.
    ///
    /// Mutations — check only products with `TIME` (G-exp, ODE, PK bind under
    /// `none`); `dead_levels` returns nothing (every `none` cell binds); the
    /// failure ignores the contrast (within refused, `[TIME]` + `ref` refused).
    #[test]
    fn a_level_with_no_effect_at_time_zero_is_refused() {
        let t6 = cf_pop(3, 1, &T6);
        let no0 = cf_pop(3, 1, &NO0);
        let rank_xb = |pop: &Population| {
            let jac = jacobian(&g_no_eta("STUDY, TIME", "none"), pop);
            rank(&residual_sv(&jac.xb, &DMatrix::zeros(jac.xb.nrows(), 0)))
        };
        assert_eq!(
            (rank_xb(&t6), rank_xb(&no0)),
            (15, 18),
            "the oracle straddles"
        );
        let g = |c: &str| g_no_eta("STUDY, TIME", c);
        assert_eq!(
            try_bind(&g(""), &t6),
            Ok((LevelContrast::SumToZeroWithin, 15)),
            "T6"
        );
        assert_eq!(
            try_bind(&g(""), &no0),
            Ok((LevelContrast::SumToZero, 17)),
            "no 0"
        );
        for c in ["sum_to_zero", "ref", "none"] {
            let err = try_bind(&g(c), &t6).expect_err(c);
            has(&err, &[DEAD_LEVELS, &time_zero_labels(3).join(", ")]);
        }
        // The `ref` pair: the dead level is `[TIME]`'s reference.
        assert_eq!(
            try_bind(&g_no_eta("TIME", "ref"), &t6),
            Ok((LevelContrast::Ref, 5)),
            "[TIME], ref"
        );

        let g_exp = |c: &str| g(c).replace(EMAXY, "EMAX * (1 - exp(-TIME / ET50))");
        let ode = |c: &str| {
            cf_model(
                c,
                "STUDY, TIME",
                "  KE = TVET50\n  PB = PLACEBO\n  E0 = TVE0\n  Z = TVE0 * exp(ETA_E0)",
                "E",
            )
            .replace(
                "[structural_model]\n  y = E\n",
                "[structural_model]\n  ode(states=[E])\n\n[odes]\n  d/dt(E) = (1 + PB) - KE * E\n\n[scaling]\n  y = E\n",
            )
        };
        let pk = |c: &str| {
            shape("S2", "STUDY, TIME", c)
                .replace(
                    "  CL = TVEMAX * exp(ETA_E0)",
                    "  CL = TVEMAX * exp(PLACEBO)",
                )
                .replace("  y = central / V + E0 + PLACEBO", "  y = central / V + E0")
                .replace("  E0 = TVE0", "  E0 = TVE0\n  Z = TVE0 * exp(ETA_E0)")
                // k = 0.2 rather than 2: at k = 2 the `TIME = 12` level moves
                // `y` by ~1e-8, live to the bitwise check (D4) but under the FD
                // oracle's rank threshold, so the oracle would measure itself.
                .replace("  V = TVET50", "  V = 10 * TVET50")
        };
        type Text<'a> = &'a dyn Fn(&str) -> String;
        let one_col = |c: &str| g_no_eta("TIME", c);
        let cases: [(&str, Text, String); 5] = [
            ("G", &g, time_zero_labels(3).join(", ")),
            ("G-exp", &g_exp, time_zero_labels(3).join(", ")),
            ("ODE", &ode, time_zero_labels(3).join(", ")),
            (
                "PK CL at the dose time",
                &pk,
                time_zero_labels(3).join(", "),
            ),
            ("[TIME]", &one_col, "`TIME=0`".to_string()),
        ];
        let mut wrong = Vec::new();
        for (tag, text, labels) in cases {
            agrees_with_joint_oracle(&format!("{tag} T6"), text, &t6, &mut wrong);
            agrees_with_joint_oracle(&format!("{tag} no 0"), text, &no0, &mut wrong);
            let err = try_bind(&text("none"), &t6).expect_err(tag);
            has(&err, &[DEAD_LEVELS, &labels]);
            assert!(try_bind(&text("none"), &no0).is_ok(), "{tag}: none on no 0");
        }
        assert!(
            wrong.is_empty(),
            "{} cells disagree:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }

    /// Every partial dead-level refusal ends with this caveat (#1702 review F3).
    const CHECK_SEES: &str = " The check sees only the initial estimates and a point near \
                              them, so a θ that switches the block off there (a lag or a \
                              threshold) makes a live level look dead: check that θ's initial \
                              estimate.";

    /// The dead-level refusals, sentence by sentence (message cells D1–D4 of
    /// the #1679 plan, and the contrast-aware cells of the #1702 review), each
    /// conditional sentence with both sides of its gate here: the `TIME = 0`
    /// advice (every listed level at `TIME = 0`, every one at `TIME = 4`, a
    /// mixed set); the consequence (a contrast that carries the levels named,
    /// or none exists); a rank loss against a slack (an absorbed η taking the
    /// within-group constraint); and the absorption refusal's advice (within
    /// named, or not when it cannot carry the dead levels either). Then every
    /// level dead (D3) and more labels than are shown (D4).
    #[test]
    fn the_dead_level_refusal_says_what_the_levels_hold() {
        const AT_ZERO: &str = " Every one of them holds only records at `TIME = 0`.";
        const ZERO_FIX: &str = " Key the block so those records share a level with a later \
                                time, or read the block where it acts at `TIME = 0`.";
        const GENERAL_FIX: &str =
            " Key the block so those records share a level with records it acts on.";
        const WITHIN_CARRIES: &str = " Under `contrast = sum_to_zero` that leaves θ the data \
                                      cannot estimate, which `contrast = sum_to_zero_within` \
                                      does not.";
        const NO_CONTRAST: &str = " That leaves θ the data cannot estimate under any contrast.";
        const MEASURED: &str = "at any of its records, at the initial estimates or at a nearby \
                                point: ";
        let t6 = cf_pop(3, 1, &T6);
        let at = |factor: &str, c: &str| {
            cf_model(
                c,
                "STUDY, TIME",
                &format!("{BASE}  E0 = TVE0 + PLACEBO * {factor}\n  Z = TVE0 * exp(ETA_E0)"),
                &format!("E0 + {EMAXY}"),
            )
        };
        // D1: every listed level at TIME = 0; within carries them.
        let err = try_bind(&at("TIME", "sum_to_zero"), &t6).expect_err("TIME");
        has(
            &err,
            &[
                &format!(
                    "theta PLACEBO[STUDY, TIME]: each of these levels {DEAD_LEVELS}, at the \
                     initial estimates or at a nearby point: {}.",
                    time_zero_labels(3).join(", ")
                ),
                &format!("{WITHIN_CARRIES}{AT_ZERO}{ZERO_FIX}{CHECK_SEES}"),
            ],
        );
        lacks(&err, &[GENERAL_FIX, NO_CONTRAST]);
        assert!(err.ends_with(CHECK_SEES), "{err}");
        // D2: every listed level at TIME = 4, under `none`; within carries them.
        let err = try_bind(&at("(TIME - 4)", "none"), &t6).expect_err("TIME - 4");
        has(
            &err,
            &[
                "`STUDY=1,TIME=4`, `STUDY=2,TIME=4`, `STUDY=3,TIME=4`.",
                " Under `contrast = none` that leaves θ the data cannot estimate, which \
                 `contrast = sum_to_zero_within` does not.",
                &format!("{GENERAL_FIX}{CHECK_SEES}"),
            ],
        );
        lacks(&err, &["TIME = 0", "later time", NO_CONTRAST]);
        // A mixed set, two per study: no contrast carries them, auto included.
        let err = try_bind(&at("TIME * (TIME - 4)", ""), &t6).expect_err("mixed");
        has(
            &err,
            &[
                MEASURED,
                "`STUDY=1,TIME=0`, `STUDY=1,TIME=4`, `STUDY=2,TIME=0`, `STUDY=2,TIME=4`, \
                 `STUDY=3,TIME=0` and 1 more.",
                &format!("{NO_CONTRAST}{GENERAL_FIX}{CHECK_SEES}"),
            ],
        );
        lacks(&err, &["TIME = 0", "Under `contrast"]);
        // D4: seven studies, five labels shown.
        let err = try_bind(&at("TIME", "sum_to_zero"), &cf_pop(7, 1, &T6)).expect_err("7");
        has(
            &err,
            &[&format!("{} and 2 more.", time_zero_labels(5).join(", "))],
        );
        lacks(&err, &["`STUDY=6,TIME=0`"]);

        // The slack: G with its η on `EMAX`, one subject per study. The η is
        // absorbed, so within is the only candidate, and the dead `TIME = 0`
        // level — which the η does not read either — takes each study's
        // constraint. H10 (the η on `E0`, read at `TIME = 0`) is the other side.
        let err = try_bind(&shape("G", "STUDY, TIME", ""), &t6).expect_err("G slack");
        has(
            &err,
            &[
                MEASURED,
                &time_zero_labels(3).join(", "),
                " Under `contrast = sum_to_zero_within` such a level takes its group's \
                 sum-to-zero constraint, so the other levels reproduce the individual parameter \
                 `EMAX` reads this block and carries a random effect — the two are the same \
                 quantity, so the model is not identified.",
                &format!("{AT_ZERO}{ZERO_FIX}{CHECK_SEES}"),
            ],
        );
        lacks(&err, &["leaves θ the data cannot estimate"]);
        assert_eq!(
            try_bind(&shape("H10", "STUDY, TIME", ""), &t6),
            Ok((LevelContrast::SumToZeroWithin, 15)),
            "H10"
        );
        // The absorption refusal no longer advises a within that fails too.
        let err = try_bind(&shape("G", "STUDY, TIME", "sum_to_zero"), &t6).expect_err("G s2z");
        has(
            &err,
            &[
                " `contrast = sum_to_zero_within` does not help: some levels have no effect at \
               any of their records, which that contrast cannot carry either. Drop the random \
               effect, or key the block so every level acts on its records.",
            ],
        );
        lacks(&err, &[USE_WITHIN]);
        let err = try_bind(&shape("H1", "STUDY, TIME", "sum_to_zero"), &t6).expect_err("H1");
        has(&err, &[USE_WITHIN]);
        lacks(&err, &["does not help"]);

        // D3: every level dead — the SLOPE0 twin with the slope fixed at 0.
        let err = try_bind(&slope0(true), &t6).expect_err("every level dead");
        assert_eq!(
            err,
            "theta PLACEBO[STUDY, TIME]: no level of the block affects the likelihood at the \
             initial estimates or at a nearby point; the block estimates nothing. Check the \
             expression that reads it."
        );
    }

    /// SLOPE0: the block enters `E0` through a slope initialised at 0, so it
    /// has no effect at the initial estimates. `fixed` fixes the slope there.
    fn slope0(fixed: bool) -> String {
        let tvs = if fixed {
            "(0.0, -10.0, 10.0, FIX)"
        } else {
            "(0.0, -10.0, 10.0)"
        };
        cf_model(
            "",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + PLACEBO * TVS\n  Z = TVE0 * exp(ETA_E0)"),
            &format!("E0 + {EMAXY}"),
        )
        .replace(
            "  theta TVET50(1.5, 0.1, 20.0)\n",
            &format!("  theta TVET50(1.5, 0.1, 20.0)\n  theta TVS{tvs}\n"),
        )
    }

    /// D2 (#1679). A block dead only at the initial estimates binds: the
    /// jittered point moves the slope off 0. Fixed at 0, the slope stays there
    /// at both points, and the block is refused.
    ///
    /// Mutations — check the initial estimates only (SLOPE0 refused); jitter a
    /// `FIX` θ too (the twin binds).
    #[test]
    fn a_level_dead_only_at_the_initial_estimates_binds() {
        let pop = cf_pop(3, 1, &NO0);
        assert_eq!(
            try_bind(&slope0(false), &pop),
            Ok((LevelContrast::SumToZero, 17))
        );
        assert!(try_bind(&slope0(true), &pop).is_err());
    }

    /// D3 (#1679). A block read only in the residual magnitude
    /// (`DV ~ proportional(PROP * ERRSCALE[STUDY])`) has no effect on any
    /// prediction, but it is live: the check compares the residual magnitudes
    /// too.
    ///
    /// Mutation — drop `ruv_obs_mult` from the compared values: refused as
    /// every level dead.
    #[test]
    fn a_level_read_only_in_the_residual_magnitude_binds() {
        let text = r#"
[parameters]
  theta TVCL(2.0, 0.01, 20.0)
  theta TVV(8.0, 0.1, 500.0)
  theta ERRSCALE[STUDY, contrast = none](2.0, 0.1, 10.0)
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.02 FIX
[individual_parameters]
  CL = TVCL
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR * ERRSCALE)
"#;
        let mut pop = population(3, 6);
        let mut parsed = parse_full_model(text).unwrap();
        crate::api::bind_theta_levels(&mut parsed, text, &mut pop).expect("binds");
        assert_eq!(parsed.model.n_theta, 2 + 3);
    }

    /// D4 (#1679). The check steps finitely and compares per record: on the
    /// "direct" fixture (`CL = TVEMAX * exp(PLACEBO)`, a bolus at 0) the level at
    /// `TIME = 12` moves `y` by about 1e-8 against `E0 = 1.5` (∂y/∂level ≈ 6e-8),
    /// and must stay live.
    ///
    /// Mutation — compare the summed objective under a 1e-6 relative step: that
    /// change is lost in the sum's rounding, and the level is refused.
    #[test]
    fn the_dead_check_steps_finitely_and_per_record() {
        let text = shape("S2", "STUDY, TIME", "none")
            .replace(
                "  CL = TVEMAX * exp(ETA_E0)",
                "  CL = TVEMAX * exp(PLACEBO)",
            )
            .replace("  y = central / V + E0 + PLACEBO", "  y = central / V + E0")
            .replace("  E0 = TVE0", "  E0 = TVE0\n  Z = TVE0 * exp(ETA_E0)");
        let pop = cf_pop(3, 1, &[12.0]);
        let jac = jacobian(&text, &pop);
        let effect = jac.xb.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            effect > 0.0 && effect < 1e-6,
            "∂y/∂level at TIME = 12: {effect:e}"
        );
        assert_eq!(try_bind(&text, &pop), Ok((LevelContrast::Unconstrained, 3)));
    }

    /// D5 (#1679). A point where anything evaluates non-finite is inconclusive,
    /// and the level counts as live: here the jitter drives `TVN` negative, so
    /// `TVN ^ 0.5` is NaN on every record at the second point (`log` would not
    /// do: it is floored), and the `TIME = 0` levels, dead at the first, bind
    /// under global sum-to-zero. With `TVN` positive at both points the same
    /// levels are refused there — both sides of the gate.
    ///
    /// Mutation — compare NaN bits like any others: the NaN point reads as
    /// unchanged and the levels are refused.
    #[test]
    fn a_non_finite_evaluation_is_inconclusive() {
        let text = |bounds: &str| {
            cf_model(
                "sum_to_zero",
                "STUDY, TIME",
                &format!(
                    "{BASE}  E0 = TVE0 + TVN ^ 0.5 + PLACEBO * TIME\n  Z = TVE0 * exp(ETA_E0)"
                ),
                &format!("E0 + {EMAXY}"),
            )
            .replace(
                "  theta TVET50(1.5, 0.1, 20.0)\n",
                &format!("  theta TVET50(1.5, 0.1, 20.0)\n  theta TVN{bounds}\n"),
            )
        };
        let pop = cf_pop(3, 1, &T6);
        assert_eq!(
            try_bind(&text("(0.05, -1.0, 0.1)"), &pop),
            Ok((LevelContrast::SumToZero, 17)),
            "NaN at the jittered point"
        );
        let err = try_bind(&text("(0.05, 0.01, 1.0)"), &pop).expect_err("finite at both");
        has(&err, &[DEAD_LEVELS]);
    }

    /// D6 (#1679). The check skips a model with a likelihood channel it does not
    /// read: a block read only by a hazard parameter has no effect on any
    /// prediction, but the event model reads it, so the model binds. The same
    /// block with the event model removed is refused — both sides in one test.
    ///
    /// Mutation — delete `dead_check_reads_every_channel`: the joint model is
    /// refused as every level dead.
    #[cfg(feature = "survival")]
    #[test]
    fn channels_the_check_does_not_read_are_skipped() {
        let joint = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(20.0, 0.1, 500.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta PLACEBO[STUDY](0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  H0   = TVH0 * exp(PLACEBO)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[event_model]
  cmt    = 3
  hazard = H0
[error_model]
  DV ~ proportional(PROP)
"#;
        let pop = cf_pop(3, 2, &NO0);
        assert_eq!(try_bind(joint, &pop), Ok((LevelContrast::SumToZero, 2)));
        let pk_only = joint.replace("[event_model]\n  cmt    = 3\n  hazard = H0\n", "");
        let err = try_bind(&pk_only, &pop).expect_err("no event model");
        has(&err, &["no level of the block affects the likelihood"]);
    }

    /// #1702 review F2. A θ initialised at the centre of tight bounds still
    /// steps: `PLACEBO[STUDY, TIME](0.0, -0.1, 0.1)` with the block read
    /// directly in `E0` is live everywhere and binds. Before the fix the step
    /// fell back to the midpoint — the θ itself — at both points, and the block
    /// was refused as estimating nothing. With `(0.0, -0.5, 0.5)` it bound
    /// either way: the pair straddles the old fallback.
    ///
    /// Mutation — restore the midpoint fallback: the ±0.1 block is refused.
    #[test]
    fn a_theta_at_the_centre_of_tight_bounds_still_steps() {
        let pop = cf_pop(3, 1, &NO0);
        for bounds in ["(0.0, -0.1, 0.1)", "(0.0, -0.5, 0.5)"] {
            let text = cf_model(
                "",
                "STUDY, TIME",
                &format!("{BASE}  E0 = TVE0 + PLACEBO\n  Z = TVE0 * exp(ETA_E0)"),
                &format!("E0 + {EMAXY}"),
            )
            .replace("(0.0, -10.0, 10.0)", bounds);
            assert!(text.contains(bounds), "{bounds}");
            assert_eq!(
                try_bind(&text, &pop),
                Ok((LevelContrast::SumToZero, 17)),
                "{bounds}"
            );
        }
        // Bounds that leave no room at all: the θ cannot step, which measures
        // nothing, so the levels count as live and the block binds. Read as "no
        // change", every level would be dead.
        let pinned = cf_model(
            "",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + PLACEBO\n  Z = TVE0 * exp(ETA_E0)"),
            &format!("E0 + {EMAXY}"),
        )
        .replace("(0.0, -10.0, 10.0)", "(0.0, 0.0, 0.0)");
        assert_eq!(
            try_bind(&pinned, &pop),
            Ok((LevelContrast::SumToZero, 17)),
            "no room"
        );
        // The helper itself: never onto a bound, and back to `x` only with no room.
        assert!((toward_interior(0.0, -0.1, 0.1, 0.13) - 0.05).abs() < 1e-15);
        assert!((toward_interior(0.09, -0.1, 0.1, 0.25) + 0.005).abs() < 1e-15);
        assert_eq!(toward_interior(1.0, 1.0, 1.0, 0.1), 1.0);
    }

    /// #1702 review F3. A θ that switches the block off near its initial
    /// estimate makes a live level look dead at both points the check uses:
    /// `E0 = TVE0 + PLACEBO * max(0, TIME - TLAG)` with `TLAG` at 2 leaves the
    /// levels at `TIME` 0.5, 1 and 2 unmoved there, though any `TLAG` below
    /// their time reads them. The check cannot see past that, so the refusal
    /// says what it measured and points at such a θ.
    ///
    /// Mutation — drop the caveat sentence: this test and the message test die.
    #[test]
    fn a_theta_gated_read_names_what_the_check_saw() {
        let text = cf_model(
            "none",
            "STUDY, TIME",
            &format!("{BASE}  E0 = TVE0 + PLACEBO * max(0, TIME - TLAG)\n  Z = TVE0 * exp(ETA_E0)"),
            &format!("E0 + {EMAXY}"),
        )
        .replace(
            "  theta TVET50(1.5, 0.1, 20.0)\n",
            "  theta TVET50(1.5, 0.1, 20.0)\n  theta TLAG(2.0, 0.0, 24.0)\n",
        );
        let err = try_bind(&text, &cf_pop(3, 1, &NO0)).expect_err("TLAG");
        has(
            &err,
            &[
                "at the initial estimates or at a nearby point: `STUDY=1,TIME=0.5`, \
                 `STUDY=1,TIME=1`, `STUDY=1,TIME=2`, `STUDY=2,TIME=0.5`, `STUDY=2,TIME=1` and \
                 4 more.",
                CHECK_SEES,
            ],
        );
    }

    /// #1702 review F4. K3′ is decided by the parameters that read each random
    /// effect, not by the diagnostic clause. A kappa and an η in one
    /// `CL = TVEMAX * exp(ETA_E0 + KAPPA_E0)`, reaching `y` only through the
    /// states, are one random effect by two names on one occasion: K3′. A kappa
    /// on `CL` and an η on `V` share the route but not the parameter: K3.
    ///
    /// Mutations — compare the clauses again: the route twin takes K3 (its
    /// clauses name each random effect). Compare the routes only: κ-`CL` /
    /// η-`V` takes K3′. Skip `if` branches when collecting the readers: the
    /// branched κ-`CL` / η-`V` takes K3′.
    #[test]
    fn a_route_twin_is_named_by_its_parameter() {
        let one_occ = occ_pop(3, 1, &T6, 1, false);
        let route = |cl: &str, v: &str| {
            shape("S2", "STUDY, TIME", "")
                .replace("  CL = TVEMAX * exp(ETA_E0)", &format!("  CL = {cl}"))
                .replace("  V = TVET50", &format!("  V = {v}"))
                .replace(
                    "  omega ETA_E0 ~ 0.1\n",
                    "  omega ETA_E0 ~ 0.1\n  kappa KAPPA_E0 ~ 0.1\n",
                )
        };
        let twin = route("TVEMAX * exp(ETA_E0 + KAPPA_E0)", "TVET50");
        let err = try_bind(&twin, &one_occ).expect_err("twin");
        has(&err, &[KAPPA_IS_ETA]);
        let apart = route("TVEMAX * exp(KAPPA_E0)", "TVET50 * exp(ETA_E0)");
        let err = try_bind(&apart, &one_occ).expect_err("apart");
        has(
            &err,
            &["the levels absorb `ETA_E0` and `KAPPA_E0` together ("],
        );
        lacks(&err, &["single occasion"]);
        // The same two, each read only inside an `if` branch: the readers walk
        // the branches, or both would read as read by nothing — and equal.
        let branched = route("TVEMAX", "TVET50").replace(
            "  E0 = TVE0",
            "  E0 = TVE0\n  if (STUDY > 0) {\n    CL = TVEMAX * exp(KAPPA_E0)\n    \
             V = TVET50 * exp(ETA_E0)\n  }",
        );
        let err = try_bind(&branched, &one_occ).expect_err("branched");
        has(
            &err,
            &["the levels absorb `ETA_E0` and `KAPPA_E0` together ("],
        );
        lacks(&err, &["single occasion"]);
    }

    /// #1702 review F5. On a one-column block K3′ adds that the block has to go
    /// as well: dropping either random effect leaves one that each level is the
    /// same quantity as. On `[STUDY, TIME]` it does not.
    ///
    /// Mutation — drop the clause: the one-column side dies.
    #[test]
    fn a_kappa_twin_on_a_one_column_block_also_removes_the_block() {
        const ALSO: &str =
            " Each `STUDY` level also lies within a single subject, so remove the block as well.";
        let one_occ = occ_pop(3, 1, &T6, 1, false);
        let err = try_bind(&kshape("K-E0", true, "STUDY", ""), &one_occ).expect_err("1-col");
        has(
            &err,
            &[
                KAPPA_IS_ETA,
                &format!(" Drop one of the two random effects.{ALSO}"),
            ],
        );
        let err = try_bind(&kshape("K-E0", true, "STUDY, TIME", ""), &one_occ).expect_err("nested");
        has(&err, &[KAPPA_IS_ETA]);
        lacks(&err, &["remove the block"]);
    }

    /// #1702 review F6. Two kappas absorbed together leave each **occasion's**
    /// mean to one of them under within; an η among them leaves each subject's.
    /// The verdict is the joint oracle's on both.
    ///
    /// Mutation — "subject's" for every set: the kappa pair dies.
    #[test]
    fn two_kappas_absorbed_together_name_the_occasion() {
        let periods = occ_pop(3, 1, &T6, 2, false);
        let two_kappas = cf_model(
            "",
            "STUDY, TIME",
            "  EMAX = TVEMAX + KAPPA_EM\n  ET50 = TVET50\n  E0 = TVE0 + PLACEBO + KAPPA_E0\n  \
             Z = TVE0 * exp(ETA_E0)",
            &format!("E0 + {EMAXY}"),
        )
        .replace(
            "  omega ETA_E0 ~ 0.1\n",
            "  omega ETA_E0 ~ 0.1\n  kappa KAPPA_E0 ~ 0.1\n  kappa KAPPA_EM ~ 0.1\n",
        );
        let jac = jacobian(
            &two_kappas.replace(
                "PLACEBO[STUDY, TIME]",
                "PLACEBO[STUDY, TIME, contrast = none]",
            ),
            &periods,
        );
        let (absorbed, _) = joint_oracle(&jac, LevelContrast::SumToZeroWithin);
        assert_eq!(absorbed.len(), 2, "oracle: {absorbed:?}");
        let err = try_bind(&two_kappas, &periods).expect_err("κ + κ");
        has(&err, &["leaves each occasion's mean to one of them"]);
        let err = try_bind(&kshape("K-E0", true, "STUDY, TIME", ""), &periods).expect_err("η + κ");
        has(&err, &["leaves each subject's mean to one of them"]);
    }

    /// The whole #1678 grid, slow-gated: block columns `[STUDY]`, `[STUDY,
    /// TIME]`, `[STUDY, OCC]` × 1–2 subjects per study × 1–3 occasions as arms
    /// or periods × the four kappa shapes × with or without an η, against the
    /// joint oracle. The κ ≡ η cells (one occasion, η and kappa at the same
    /// site, levels nested) are unidentified with or without the block: the
    /// oracle, which measures what the block adds, does not refuse them, and
    /// the binder must, with the K3′ sentence.
    ///
    /// Gated on measured cost (minutes in a local debug build, and CI runs the
    /// levels tests 6–22× slower); the per-PR K1, K2 and K3 die on the same
    /// mutations.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn the_full_kappa_grid_agrees_with_the_joint_oracle() {
        let mut wrong = Vec::new();
        let (mut cells, mut twins) = (0usize, 0usize);
        for cols in ["STUDY", "STUDY, TIME", "STUDY, OCC"] {
            for per in [1, 2] {
                for occ in [1, 2, 3] {
                    for arms in [true, false] {
                        if occ == 1 && !arms {
                            continue; // one occasion: arms and periods coincide
                        }
                        let pop = occ_pop(3, per, &T6, occ, arms);
                        for tag in ["K-E0", "K-y", "K-EMAX", "K-unread"] {
                            for eta in [false, true] {
                                cells += 1;
                                let layout = if arms { "arms" } else { "periods" };
                                let cell =
                                    format!("{tag} η={eta} [{cols}] {per}/st {occ} {layout}");
                                let text = |c: &str| kshape(tag, eta, cols, c);
                                let twin = occ == 1 && eta && matches!(tag, "K-E0" | "K-y");
                                if twin {
                                    if let Err(e) = try_bind(&text(""), &pop) {
                                        if e.contains(KAPPA_IS_ETA) {
                                            twins += 1;
                                            continue;
                                        }
                                    }
                                }
                                agrees_with_joint_oracle(&cell, &text, &pop, &mut wrong);
                            }
                        }
                    }
                }
            }
        }
        eprintln!("kappa grid: {cells} cells, {twins} κ ≡ η");
        // Measured: K-E0 and K-y on each of the three block shapes, one subject per
        // study, one occasion. The `[STUDY]` pair is also refused by the oracle
        // (the η alone absorbs a one-column block); the other four are the plan's.
        assert_eq!(twins, 6, "κ ≡ η cells");
        assert!(
            wrong.is_empty(),
            "{} of {cells} cells disagree:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }

    /// R1-1 (#1675 review). `TAD` and `TAFD` vary within a subject exactly as
    /// `TIME` does, so a shape reading them must bind as its `TIME` twin does,
    /// under every design and contrast of the oracle grid. With the only dose
    /// at time 0 the three clocks are equal on these data, so the twin's oracle
    /// cells in `binder_agrees_with_the_jacobian_oracle` (H6, H13) label these
    /// too. The f64 predictor cannot label them directly: it reads `TAD` as 0
    /// in a readout.
    ///
    /// Before the fix, `expr_constant` filed both clocks as data covariates
    /// that were missing everywhere, so they read as constant: H6-with-`TAD`
    /// was refused on one column and went to within on `[STUDY, VISIT]`.
    #[test]
    fn the_clocks_vary_like_time() {
        let readout = |cols: &str, c: &str| {
            cf_model(
                c,
                cols,
                "  EMAX = TVEMAX + ETA_E0\n  ET50 = TVET50\n  E0 = TVE0",
                "E0 + PLACEBO + EMAX * TAD / (TAD + ET50)",
            )
        };
        let param = |cols: &str, c: &str| {
            cf_model(
                c,
                cols,
                &format!("{BASE}  E0 = TVE0 + PLACEBO + ETA_E0 * TAFD"),
                &format!("E0 + {EMAXY}"),
            )
        };
        let tad = |cols: &str, c: &str| readout(cols, c).replace("TAD", "tad");
        type Shape<'a> = (&'a str, &'a dyn Fn(&str, &str) -> String, &'a str);
        let cases: [Shape; 3] = [
            ("TAD in y", &readout, "TAD"),
            ("tad in y", &tad, "tad"),
            ("TAFD in a parameter", &param, "TAFD"),
        ];
        let mut straddle = 0;
        for (dtag, cols, pop) in designs() {
            for (tag, text, clock) in cases {
                for c in ["", "sum_to_zero", "sum_to_zero_within", "ref", "none"] {
                    let clocked = text(cols, c);
                    assert!(clocked.contains(clock), "{tag}");
                    let twin = clocked.replace(clock, "TIME");
                    let (got, want) = (try_bind(&clocked, &pop), try_bind(&twin, &pop));
                    assert_eq!(
                        got.is_ok(),
                        want.is_ok(),
                        "{dtag} {tag} {c:?}: {got:?} vs {want:?}"
                    );
                    if let (Ok(g), Ok(w)) = (&got, &want) {
                        assert_eq!(g, w, "{dtag} {tag} {c:?}");
                    }
                    straddle += usize::from(want.is_ok());
                }
            }
        }
        // The twins bind in most cells, so a clock read as constant (refused, or
        // pushed to within) is visible.
        assert!(straddle >= 40, "only {straddle} twin cells bind");
    }

    /// T1 (#1708). A kappa inside an η's shared expression is read as constant
    /// within the subject (`expr_constant`'s `Eta(_)` arm). Two shapes on the
    /// MBMA arms layout pin both sides of what that reading means, against the
    /// joint oracle evaluated away from η = κ = 0 ([`jacobian_at`]):
    ///
    /// - **R1**, `(TVE0 + PLACEBO) * exp(KAPPA_E0) / (1 + ETA_E0 * 0.5)` (the
    ///   issue's example): `exp(KAPPA_E0)` is a common factor and cancels, so
    ///   the block absorbs `ETA_E0` at every κ and the refusal is exact.
    /// - **M**, `TVE0 + PLACEBO * exp(KAPPA_E0) + ETA_E0`: the kappa does not
    ///   separate from the block. The block absorbs `ETA_E0` wherever each
    ///   subject's kappas are equal — κ = 0, or one nonzero κ on every occasion —
    ///   and does not once they differ between a subject's occasions; the
    ///   separating singular value grows with that spread. The binder refuses
    ///   as if the spread were zero: an over-refusal, never a bind the oracle
    ///   would refuse.
    ///
    /// A future κ-aware funnel that lets M bind is expected to redden the M
    /// half; update this test with the docs (`#### Kappas` in
    /// `docs/model-file/parameters.qmd`).
    ///
    /// Mutations — `Eta(_)` read as varying in `expr_constant` (R1 binds); the
    /// evaluation point ignored by `jacobian_at` (M per-occasion absorbs, the
    /// straddle collapses).
    #[test]
    fn a_kappa_in_an_eta_funnel_is_read_as_constant_within_the_subject() {
        let model = |e0: &str, c: &str| {
            cf_model(
                c,
                "STUDY",
                &format!("{BASE}  E0 = {e0}"),
                &format!("E0 + {EMAXY}"),
            )
            .replace(
                "  omega ETA_E0 ~ 0.1\n",
                "  omega ETA_E0 ~ 0.1\n  kappa KAPPA_E0 ~ 0.1\n",
            )
        };
        let r1 = "(TVE0 + PLACEBO) * exp(KAPPA_E0) / (1 + ETA_E0 * 0.5)";
        let mm = "TVE0 + PLACEBO * exp(KAPPA_E0) + ETA_E0";
        let pop = occ_pop(4, 1, &T6, 3, true);
        let global = LevelContrast::SumToZero;
        let constant = Point {
            scale: 0.1,
            per_occasion: false,
        };
        let per_occ = Point {
            scale: 0.1,
            per_occasion: true,
        };
        let top = |j: &Jac| {
            let sv = unit_sv(j, "ETA_E0", Some(global));
            assert!(sv.iter().all(|v| v.is_finite()), "non-finite σ: {sv:?}");
            sv.iter().copied().fold(0.0, f64::max)
        };
        for (tag, e0) in [("R1", r1), ("M", mm)] {
            // The binder refuses under every contrast, and under auto, naming the η.
            for c in ["", "sum_to_zero", "sum_to_zero_within", "ref", "none"] {
                match try_bind(&model(e0, c), &pop) {
                    Err(e) => assert!(
                        e.contains("`E0` reads this block and carries a random effect"),
                        "{tag} {c:?}: {e}"
                    ),
                    Ok(r) => panic!("{tag} {c:?}: bound {r:?}"),
                }
            }
            // Where each subject's kappas are equal, the oracle absorbs the η.
            for (ptag, at) in [("κ = 0", Point::ZERO), ("κ constant", constant)] {
                let j = jacobian_at(&model(e0, "none"), &pop, at);
                // Four subjects, less the direction `TVE0` shares with them.
                assert_eq!(unit_rank(&j, "ETA_E0", None), 3, "{tag} {ptag}: baseline");
                assert_eq!(
                    joint_oracle(&j, global).0,
                    ["ETA_E0"],
                    "{tag} {ptag}: σ {:.1e}",
                    top(&j)
                );
            }
        }
        // R1 is exact: the common factor cancels at every κ.
        let j = jacobian_at(&model(r1, "none"), &pop, per_occ);
        assert_eq!(
            joint_oracle(&j, global).0,
            ["ETA_E0"],
            "R1 per occasion: σ {:.1e}",
            top(&j)
        );
        // M is not: the spread of a subject's kappas identifies the η.
        let jp = jacobian_at(&model(mm, "none"), &pop, per_occ);
        let jc = jacobian_at(&model(mm, "none"), &pop, constant);
        let (sp, sc) = (top(&jp), top(&jc));
        eprintln!("M: σ per occasion {sp:.3e}, constant within subject {sc:.3e}");
        assert!(
            joint_oracle(&jp, global).0.is_empty(),
            "M per occasion absorbs: σ {sp:.1e}"
        );
        assert!(sp > 1e-3, "M per occasion: σ {sp:.1e}");
        // The straddle itself: the separating direction is the κ spread, not
        // the κ magnitude.
        assert!(sp >= 100.0 * sc, "M straddle: {sp:.1e} vs {sc:.1e}");
    }
}

// ── #1730: re-binding a model already bound ─────────────────────────────────
//
// No fit runs: these are binder states, so the oracle is the **unbound twin** — a
// fresh parse bound once on the same data, which is what the declaration means.
#[allow(deprecated)] // the deprecated binder is one of the provenance setters
mod rebind {
    use super::absorption::shape as absorption_shape;
    use super::bind_from_fit::{fitted, model, pre_bound, weighed};
    use super::from_fit_repeated_labels::two_block_model;
    use super::readout_share::{cf_pop, T6};
    use super::*;
    use crate::api::covariate_stats::symbolic_covariates;
    use crate::api::{bind_covariate_stats, bind_from_fit, bind_theta_levels_from_fit};
    use crate::parser::model_parser::{DataBindings, LevelContrast};
    use crate::types::ParsedModel;

    /// Everything a binder decides: θ count, names, inits, bounds and `FIX` flags,
    /// the recorded bindings, each block's contrast, and the provenance.
    type Twin = (
        usize,
        Vec<String>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<bool>,
        DataBindings,
        Vec<LevelContrast>,
        bool,
    );

    fn twin(p: &ParsedModel) -> Twin {
        let m = &p.model;
        let d = &m.default_params;
        assert_eq!(&p.bindings.levels, &m.data_bindings().levels);
        assert_eq!(
            &p.bindings.covariate_stats,
            &m.data_bindings().covariate_stats
        );
        (
            m.n_theta,
            d.theta_names.clone(),
            d.theta.clone(),
            d.theta_lower.clone(),
            d.theta_upper.clone(),
            d.theta_fixed.clone(),
            m.data_bindings().clone(),
            m.theta_blocks()
                .level_blocks()
                .iter()
                .map(|b| b.contrast())
                .collect(),
            m.bound_from_fit(),
        )
    }

    /// `pop` with every covariate map sorted, so two populations compare by content:
    /// their `Debug` lists a `HashMap` in its own iteration order.
    fn canon(pop: &Population) -> String {
        use std::collections::BTreeMap;
        let sorted =
            |m: &HashMap<String, f64>| format!("{:?}", m.iter().collect::<BTreeMap<_, _>>());
        let all = |v: &[HashMap<String, f64>]| v.iter().map(sorted).collect::<Vec<_>>();
        let mut p = pop.clone();
        let maps: Vec<_> = p
            .subjects
            .iter_mut()
            .map(|s| {
                let m = (
                    sorted(&s.covariates),
                    all(&s.dose_covariates),
                    all(&s.obs_covariates),
                    all(&s.pk_only_covariates),
                    all(&s.reset_covariates),
                );
                s.covariates.clear();
                s.dose_covariates.clear();
                s.obs_covariates.clear();
                s.pk_only_covariates.clear();
                s.reset_covariates.clear();
                m
            })
            .collect();
        format!("{p:?} {maps:?}")
    }

    /// The `prepare_run` pair: levels, then statistics.
    fn both(parsed: &mut ParsedModel, text: &str, pop: &mut Population) {
        crate::api::bind_theta_levels(parsed, text, pop).unwrap_or_else(|e| panic!("{e}"));
        bind_covariate_stats(parsed, text, pop).unwrap_or_else(|e| panic!("{e}"));
    }

    /// `pop` with a subject-level `WT` of `base + 10·k` on subject `k`.
    fn with_wt(mut pop: Population, base: f64) -> Population {
        for (k, s) in pop.subjects.iter_mut().enumerate() {
            s.covariates
                .insert("WT".to_string(), base + 10.0 * k as f64);
        }
        pop.covariate_names.push("WT".to_string());
        pop
    }

    /// Shape `tag` on `[STUDY, TIME]` with `auto`, plus a `center = median`
    /// relation on `WT` for a parameter the shape reads.
    fn shape_with_median(tag: &str) -> String {
        let param = if tag == "S2" { "V" } else { "ET50" };
        format!(
            "{}\n[covariates]\n  WT continuous\n\n[covariate_model]\n  {param} ~ WT \
             power(center = median) => THETA_WT(0.6, 0.01, 5.0)\n",
            absorption_shape(tag, "STUDY, TIME", "")
        )
    }

    /// T1 (M1–M6): a model bound to data A and re-bound on B by the `prepare_run`
    /// pair is the unbound model bound on B — both halves, both directions of the
    /// `auto` flip (one subject per study ↔ two), on a PK shape (S2) and a
    /// compartment-free one (H1). The straddle is asserted: A's stamped contrast and
    /// median both differ from B's, so a binder reading either from `parsed.model`
    /// lands on A's.
    ///
    /// Mutations — read `decls` from `parsed.model` in `bind_theta_levels`: the
    /// A = 1/study cells keep `SumToZeroWithin`, the A = 2/study cells are refused for
    /// a contrast nobody wrote. (The statistics half of the pair is pinned by T2:
    /// here the level bind drops A's median first, so the statistics bind finds the
    /// relation unresolved either way.)
    #[test]
    fn a_pre_bound_pair_rebinds_as_the_unbound_pair() {
        for tag in ["S2", "H1"] {
            let text = shape_with_median(tag);
            for (a_per, b_per) in [(1, 2), (2, 1)] {
                let cell = format!("{tag} A {a_per}/study → B {b_per}/study");
                let a = with_wt(cf_pop(3, a_per, &T6), 80.0);
                let b = with_wt(cf_pop(3, b_per, &T6), 60.0);

                let mut on_a = a.clone();
                let mut prebound = parse_full_model(&text).unwrap();
                both(&mut prebound, &text, &mut on_a);

                let mut want_pop = b.clone();
                let mut unbound = parse_full_model(&text).unwrap();
                both(&mut unbound, &text, &mut want_pop);

                // The straddle.
                let (ta, tb) = (twin(&prebound), twin(&unbound));
                assert_ne!(ta.7, tb.7, "{cell}: the contrast flips between A and B");
                assert!(ta.7.iter().all(|c| *c != LevelContrast::Auto), "{cell}");
                assert_ne!(
                    ta.6.covariate_stats["WT"].median, tb.6.covariate_stats["WT"].median,
                    "{cell}: the median differs between A and B"
                );

                let mut got_pop = b.clone();
                both(&mut prebound, &text, &mut got_pop);
                assert_eq!(twin(&prebound), tb, "{cell}");
                assert_eq!(canon(&got_pop), canon(&want_pop), "{cell}");
            }
        }
    }

    /// T1b: the dead-level check runs on the statistics the level bind keeps, not
    /// on `parsed`'s. Under `linear(center = c)` at θ = 0.05 the factor on `CL` is
    /// exactly 0 for a subject 20 below `c`, so that subject's levels are dead.
    /// Straddle: statistics bound on C (WT 60, 80, 100; median 80) reach the check,
    /// and the level bind on C is refused. A model bound to A (WT 70, 80, 90; median
    /// 80, no subject 20 below) and re-bound on B (WT 60, 70, 80; median 70) drops
    /// A's median and binds as the unbound model does.
    ///
    /// Mutation — measure the dead levels on `parsed.bindings`: A's median kills B's
    /// WT-60 subject, and the re-bind is refused.
    #[test]
    fn the_dead_level_check_reads_the_statistics_the_bind_keeps() {
        let text = format!(
            "{}\n[covariates]\n  WT continuous\n\n[covariate_model]\n  CL ~ WT \
             linear(center = median) => THETA_CL_WT(0.05, -1.0, 1.0)\n",
            model(true, false).replace("CL = TVCL + PLACEBO", "CL = TVCL * exp(PLACEBO)")
        );
        let weights = |w: [f64; 3]| {
            let mut pop = weighed(3, 2, 0.0);
            for (s, wt) in pop.subjects.iter_mut().zip(w) {
                s.covariates.insert("WT".to_string(), wt);
            }
            pop
        };
        let mut c = weights([60.0, 80.0, 100.0]);
        let mut parsed = parse_full_model(&text).unwrap();
        bind_covariate_stats(&mut parsed, &text, &c).unwrap();
        let err = crate::api::bind_theta_levels(&mut parsed, &text, &mut c)
            .expect_err("C's own median makes its WT-60 subject's levels dead")
            .to_string();
        assert!(
            err.contains("each of these levels has no effect on the predictions"),
            "{err}"
        );

        let mut prebound = pre_bound(&text, &mut weights([70.0, 80.0, 90.0]));
        assert_eq!(prebound.bindings.covariate_stats["WT"].median, 80.0);
        let mut got = weights([60.0, 70.0, 80.0]);
        crate::api::bind_theta_levels(&mut prebound, &text, &mut got)
            .unwrap_or_else(|e| panic!("A's median is not B's: {e}"));
        let mut want = weights([60.0, 70.0, 80.0]);
        let mut unbound = parse_full_model(&text).unwrap();
        crate::api::bind_theta_levels(&mut unbound, &text, &mut want).unwrap();
        assert_eq!(twin(&prebound), twin(&unbound));
        assert_eq!(canon(&got), canon(&want));
    }

    /// M8: binding once — `prepare_run`'s pair, levels then statistics — reads each
    /// declaration from the model in hand, with no parse beyond the two re-parses it
    /// always made. Straddle: the same model, asked for a half it has bound,
    /// re-parses.
    ///
    /// Mutation — key the `Stats` read on every binding (`data_bindings().is_empty()`,
    /// the plan's first draft): the statistics bind after a level bind re-parses on
    /// every fit of a model with both halves.
    #[test]
    fn binding_once_reads_the_declaration_without_a_parse() {
        use crate::api::levels::{declared_model, Declared, Reads};
        let text = model(true, true);
        let borrowed = |p: &ParsedModel, r: Reads| {
            matches!(declared_model(p, &text, r).unwrap(), Declared::Borrowed(_))
        };
        let mut pop = weighed(3, 2, 60.0);
        let mut parsed = parse_full_model(&text).unwrap();
        assert!(borrowed(&parsed, Reads::Levels), "fresh: the level bind");
        crate::api::bind_theta_levels(&mut parsed, &text, &mut pop).unwrap();
        assert!(
            borrowed(&parsed, Reads::Stats),
            "levels bound: the statistics bind"
        );
        assert!(
            !borrowed(&parsed, Reads::Levels),
            "the bound half re-parses"
        );
        assert!(!borrowed(&parsed, Reads::Both), "the bound half re-parses");
    }

    /// T2 (M5): `bind_covariate_stats` alone on a model bound to A's median (90)
    /// re-centres it on B's (70), as on the unbound model.
    ///
    /// Mutation — gate on `parsed.model`'s relations: the A-bound model sees nothing
    /// unresolved and keeps 90.
    #[test]
    fn statistics_alone_rebind_on_the_new_data() {
        let text = model(false, true);
        let a = weighed(3, 2, 80.0);
        let b = weighed(3, 2, 60.0);
        let mut prebound = parse_full_model(&text).unwrap();
        bind_covariate_stats(&mut prebound, &text, &a).unwrap();
        assert_eq!(prebound.bindings.covariate_stats["WT"].median, 90.0);

        let mut unbound = parse_full_model(&text).unwrap();
        bind_covariate_stats(&mut unbound, &text, &b).unwrap();
        assert_eq!(unbound.bindings.covariate_stats["WT"].median, 70.0);

        bind_covariate_stats(&mut prebound, &text, &b).unwrap();
        assert_eq!(twin(&prebound), twin(&unbound));
    }

    /// T3a: `bind_theta_levels` alone on B drops A's statistics, which are no
    /// longer the data's, leaving the relation for `bind_covariate_stats`: the
    /// unbound levels-only twin. Control, in the same test: statistics bound on B
    /// itself survive a level bind on B.
    ///
    /// Mutations — always keep the statistics: the first arm keeps A's median.
    /// Always drop them: the control loses B's.
    #[test]
    fn a_level_bind_keeps_the_statistics_only_while_they_are_the_datas() {
        let text = model(true, true);
        let mut prebound = pre_bound(&text, &mut weighed(3, 2, 80.0));
        let mut got = weighed(3, 2, 60.0);
        crate::api::bind_theta_levels(&mut prebound, &text, &mut got).unwrap();
        let mut want = weighed(3, 2, 60.0);
        let mut unbound = parse_full_model(&text).unwrap();
        crate::api::bind_theta_levels(&mut unbound, &text, &mut want).unwrap();
        assert_eq!(symbolic_covariates(&prebound.model), vec!["WT".to_string()]);
        assert_eq!(twin(&prebound), twin(&unbound));
        assert_eq!(canon(&got), canon(&want));

        // Control: statistics first, on B, then levels on B.
        let mut on_b = weighed(3, 2, 60.0);
        let mut parsed = parse_full_model(&text).unwrap();
        bind_covariate_stats(&mut parsed, &text, &on_b).unwrap();
        crate::api::bind_theta_levels(&mut parsed, &text, &mut on_b).unwrap();
        assert_eq!(parsed.bindings.covariate_stats["WT"].median, 70.0);
        assert!(symbolic_covariates(&parsed.model).is_empty());
    }

    /// T3b: `bind_covariate_stats` alone on B, whose subjects carry no level index
    /// column, drops A's level layout, leaving the block for `bind_theta_levels`:
    /// the unbound statistics-only twin. Control, in the same test: levels bound on
    /// B itself (the `prepare_run` order) survive a statistics bind on B.
    ///
    /// The third cell (#1735 review r1, finding 1): B's subjects carry index columns,
    /// but written for B's own levels, which are not A's (`STUDY=1,TIME=3` is B's
    /// index 3, A's `STUDY=2,TIME=1`). A's layout is dropped there too.
    ///
    /// Mutations — always keep the levels: the first arm keeps A's layout. Always
    /// drop them: the control loses B's. Check only that the columns are present:
    /// the third cell keeps A's layout over B's columns.
    #[test]
    fn a_statistics_bind_keeps_the_levels_only_while_they_are_written_on_the_data() {
        let text = model(true, true);
        let mut prebound = pre_bound(&text, &mut weighed(3, 2, 80.0));
        assert!(!prebound.bindings.levels.is_empty());
        let b = weighed(3, 2, 60.0);
        bind_covariate_stats(&mut prebound, &text, &b).unwrap();
        let mut unbound = parse_full_model(&text).unwrap();
        bind_covariate_stats(&mut unbound, &text, &b).unwrap();
        assert!(prebound.bindings.levels.is_empty());
        assert_eq!(twin(&prebound), twin(&unbound));

        // Control: levels first, on B, then statistics on B.
        let mut on_b = weighed(3, 2, 60.0);
        let mut parsed = parse_full_model(&text).unwrap();
        both(&mut parsed, &text, &mut on_b);
        assert!(parsed.bindings.levels.contains_key("PLACEBO"));
        assert_eq!(parsed.bindings.covariate_stats["WT"].median, 70.0);

        // Columns present, written for other levels: 2 studies × 3 times.
        let mut other = weighed(2, 3, 60.0);
        let mut own = parse_full_model(&text).unwrap();
        crate::api::bind_theta_levels(&mut own, &text, &mut other).unwrap();
        let mut prebound = pre_bound(&text, &mut weighed(3, 2, 80.0));
        // The straddle: same level count, so only the labels tell them apart.
        assert_eq!(
            own.bindings.levels["PLACEBO"].labels.len(),
            prebound.bindings.levels["PLACEBO"].labels.len()
        );
        assert_ne!(
            own.bindings.levels["PLACEBO"].labels,
            prebound.bindings.levels["PLACEBO"].labels
        );
        bind_covariate_stats(&mut prebound, &text, &other).unwrap();
        let mut unbound = parse_full_model(&text).unwrap();
        bind_covariate_stats(&mut unbound, &text, &other).unwrap();
        assert!(
            prebound.bindings.levels.is_empty(),
            "A's layout over B's columns"
        );
        assert_eq!(twin(&prebound), twin(&unbound));
    }

    /// #1736, the documented scope of a statistics-only re-bind: a kept level layout
    /// keeps its contrast. A (one subject per study) resolves the `auto` block on
    /// `[STUDY, TIME]` to `SumToZeroWithin`. B (two per study) shows the same 18
    /// levels and carries their index columns, but resolves `SumToZero`, two more
    /// free θ. `bind_covariate_stats` on B keeps A's coding; `bind_theta_levels` on B
    /// (the documented remedy) then gives the unbound twin. Measured on S2 and H1:
    /// kept 19 θ, B's own 21; PRED agrees at the default θ (every level init is 0).
    ///
    /// Two cells per shape: A bound by levels alone (the statistics bind then borrows
    /// the stamped model, as in `prepare_run`), and by both halves (it re-parses the
    /// declaration). The straddle is asserted: same labels, different contrast.
    ///
    /// Mutations — always drop the levels in `bind_covariate_stats`: the levels-only
    /// cell's `kept` assertion sees `Auto`. Read `decls` from `parsed.model` in
    /// `bind_theta_levels`: the remedy keeps `SumToZeroWithin`.
    #[test]
    fn a_statistics_only_rebind_keeps_the_contrast_and_a_level_bind_resolves_it() {
        for tag in ["S2", "H1"] {
            let text = shape_with_median(tag);
            // B's own resolution, with its index columns written: levels alone, so
            // the reference does not lean on the statistics binder under test.
            let mut b = with_wt(cf_pop(3, 2, &T6), 60.0);
            let mut own = parse_full_model(&text).unwrap();
            crate::api::bind_theta_levels(&mut own, &text, &mut b).unwrap();
            let mut stats_on_b = parse_full_model(&text).unwrap();
            bind_covariate_stats(&mut stats_on_b, &text, &b).unwrap();
            for pair in [false, true] {
                let cell = format!("{tag}, A bound by {}", if pair { "both" } else { "levels" });
                let mut a = with_wt(cf_pop(3, 1, &T6), 80.0);
                let mut prebound = parse_full_model(&text).unwrap();
                crate::api::bind_theta_levels(&mut prebound, &text, &mut a).unwrap();
                if pair {
                    bind_covariate_stats(&mut prebound, &text, &a).unwrap();
                }
                // The straddle.
                let (kept, theirs) = (
                    &prebound.bindings.levels["PLACEBO"],
                    &own.bindings.levels["PLACEBO"],
                );
                assert_eq!(kept.labels, theirs.labels, "{cell}: B shows A's levels");
                assert_eq!(kept.contrast, LevelContrast::SumToZeroWithin, "{cell}");
                assert_eq!(theirs.contrast, LevelContrast::SumToZero, "{cell}");

                bind_covariate_stats(&mut prebound, &text, &b).unwrap();
                let got = twin(&prebound);
                assert_eq!(got.7, vec![LevelContrast::SumToZeroWithin], "{cell}: kept");
                assert_eq!(
                    got.6.covariate_stats, stats_on_b.bindings.covariate_stats,
                    "{cell}: B's statistics"
                );
                assert_eq!((got.0, own.model.n_theta), (19, 21), "{cell}: θ count");

                let mut want_pop = b.clone();
                let mut unbound = parse_full_model(&text).unwrap();
                both(&mut unbound, &text, &mut want_pop);
                crate::api::bind_theta_levels(&mut prebound, &text, &mut b.clone()).unwrap();
                assert_eq!(twin(&prebound), twin(&unbound), "{cell}: the remedy");

                // The order the docs give (#1745 review r1, finding 3): from the
                // A-bound model, levels first, then statistics, both on B.
                let mut a = with_wt(cf_pop(3, 1, &T6), 80.0);
                let mut ordered = parse_full_model(&text).unwrap();
                crate::api::bind_theta_levels(&mut ordered, &text, &mut a).unwrap();
                if pair {
                    bind_covariate_stats(&mut ordered, &text, &a).unwrap();
                }
                assert_eq!(twin(&ordered).7, vec![LevelContrast::SumToZeroWithin]);
                both(&mut ordered, &text, &mut b.clone());
                let got = twin(&ordered);
                assert_eq!(got.7, vec![LevelContrast::SumToZero], "{cell}: in order");
                assert_eq!(got, twin(&unbound), "{cell}: in order");
            }
        }
    }

    /// T4: provenance, both arms in one test. The two models carry the **same**
    /// bindings — the fit's, median 70 — and differ only in where those came from.
    /// Laid out on the fit, a statistics bind on a design (median 90) is a no-op:
    /// the fitted θ was estimated against the fit's centres (#1619's T3). Bound to
    /// the fit's data, the same call re-centres on the design.
    ///
    /// Mutations — drop the flag check in `bind_covariate_stats`: the first arm
    /// re-centres on 90 (and so do `both_halves_come_from_the_fit` and the Tier-2
    /// `bind_from_fit_keeps_the_fits_covariate_centres_on_a_design`). Treat every
    /// model as fit-bound: the second arm keeps 70.
    #[test]
    fn the_fits_centres_stay_and_a_datas_centres_move() {
        let text = model(false, true);
        let b = fitted(&text);
        let design = weighed(3, 2, 80.0);

        let mut from_fit = parse_full_model(&text).unwrap();
        bind_from_fit(&mut from_fit, &text, &mut design.clone(), &b).unwrap();
        let mut from_data = parse_full_model(&text).unwrap();
        bind_covariate_stats(&mut from_data, &text, &weighed(3, 2, 60.0)).unwrap();
        // The straddle: the bindings are the same; only the provenance differs.
        assert_eq!(from_fit.model.data_bindings(), &b);
        assert_eq!(from_data.model.data_bindings(), &b);
        assert!(from_fit.model.bound_from_fit() && !from_data.model.bound_from_fit());

        bind_covariate_stats(&mut from_fit, &text, &design).unwrap();
        assert_eq!(from_fit.bindings.covariate_stats["WT"].median, 70.0);
        bind_covariate_stats(&mut from_data, &text, &design).unwrap();
        assert_eq!(from_data.bindings.covariate_stats["WT"].median, 90.0);
    }

    fn fit_bound_refusal(blocks: &str) -> String {
        format!(
            "{blocks}: this model is laid out on a fit's levels, so binding it to the \
             levels of this data would read the fitted theta at other positions. To run \
             this data on the fit's theta, bind it from the fit's bindings instead; to use \
             this data's own levels, parse the model again and bind the new parse."
        )
    }

    /// T5: `bind_theta_levels` on a model laid out by **each** provenance setter —
    /// `bind_from_fit`, `layout_from_fit` and the deprecated binder — is refused with
    /// the same text, and writes nothing to the population or to `parsed`. Every
    /// block is named once. A fit-bound model with no level block binds `Ok` and
    /// stays as it is.
    ///
    /// Mutations — skip the flag in `FitLayout::apply`: the `bind_from_fit` and
    /// `layout_from_fit` cells bind. Skip it in the deprecated binder: that cell
    /// binds. Delete any sentence of the message: every cell's text differs.
    #[test]
    fn a_model_laid_out_on_a_fit_refuses_its_datas_own_levels() {
        type Setter = fn(&mut ParsedModel, &str, &DataBindings);
        let setters: [(&str, Setter); 3] = [
            ("bind_from_fit", |p, t, b| {
                bind_from_fit(p, t, &mut weighed(3, 2, 80.0), b).unwrap()
            }),
            ("layout_from_fit", |p, t, b| {
                crate::api::layout_from_fit(p, t, b).unwrap()
            }),
            ("bind_theta_levels_from_fit", |p, t, b| {
                bind_theta_levels_from_fit(p, t, &mut weighed(3, 2, 80.0), &b.levels).unwrap()
            }),
        ];
        let cases = [
            (model(true, true), "theta PLACEBO[STUDY, TIME]"),
            (
                two_block_model(),
                "theta PLACEBO[STUDY, TIME], theta EFF[STUDY]",
            ),
        ];
        for (text, blocks) in &cases {
            let want = fit_bound_refusal(blocks);
            for (name, set) in setters {
                let cell = format!("{name}, {blocks}");
                let b = if text.contains("WT") {
                    fitted(text)
                } else {
                    let mut pop = population(3, 2);
                    let mut p = parse_full_model(text).unwrap();
                    crate::api::bind_theta_levels(&mut p, text, &mut pop).unwrap();
                    p.model.data_bindings().clone()
                };
                let mut parsed = parse_full_model(text).unwrap();
                set(&mut parsed, text, &b);
                let before = twin(&parsed);
                let mut pop = weighed(3, 2, 60.0);
                let before_pop = format!("{pop:?}");
                let err = crate::api::bind_theta_levels(&mut parsed, text, &mut pop)
                    .expect_err(&format!("{cell}: must be refused"))
                    .to_string();
                assert_eq!(err, want, "{cell}");
                for absent in ["differ", "bound to data", "_from_fit"] {
                    assert!(!err.contains(absent), "{cell}: `{absent}` in {err}");
                }
                assert_eq!(twin(&parsed), before, "{cell}: parsed untouched");
                assert_eq!(
                    format!("{pop:?}"),
                    before_pop,
                    "{cell}: population untouched"
                );
            }
        }

        // No level block: nothing to refuse.
        let text = model(false, true);
        let mut parsed = parse_full_model(&text).unwrap();
        bind_from_fit(&mut parsed, &text, &mut weighed(3, 2, 80.0), &fitted(&text)).unwrap();
        let before = twin(&parsed);
        crate::api::bind_theta_levels(&mut parsed, &text, &mut weighed(3, 2, 60.0)).unwrap();
        assert_eq!(twin(&parsed), before);
    }

    /// T6 (M7): the deprecated binder on a model bound to data A takes the fit's
    /// layout as the unbound model does — here a `sum_to_zero_within` the fit
    /// resolved, where A's `auto` stamped `sum_to_zero`. Its statistics stay the
    /// caller's, as they always were for this binder: the twin is the sequence it
    /// was documented with, a fresh parse with the same statistics installed by
    /// hand (the Tier-2 `a_fits_data_bindings_survive_fitrx_*` runs it on the fit's).
    ///
    /// Mutations — read `decls` from `parsed.model`: refused for a contrast the block
    /// does not declare. Drop `parsed`'s statistics: the median is gone, and the
    /// Tier-2 test's simulate is refused for an unbound relation.
    #[test]
    fn the_deprecated_binder_reads_the_declaration_too() {
        let text = model(true, true);
        let mut b = fitted(&text);
        let placebo = b.levels.get_mut("PLACEBO").unwrap();
        placebo.groups = vec![0, 0, 0, 1, 1, 1];
        placebo.contrast = LevelContrast::SumToZeroWithin;

        let mut prebound = pre_bound(&text, &mut weighed(3, 2, 80.0));
        assert_eq!(
            prebound.model.theta_blocks().level_blocks()[0].contrast(),
            LevelContrast::SumToZero
        );
        assert_eq!(prebound.bindings.covariate_stats["WT"].median, 90.0);

        let mut got = weighed(3, 2, 80.0);
        bind_theta_levels_from_fit(&mut prebound, &text, &mut got, &b.levels)
            .unwrap_or_else(|e| panic!("the pre-bound model binds: {e}"));
        let mut want = weighed(3, 2, 80.0);
        let mut by_hand = parse_full_model(&text).unwrap();
        by_hand.bindings.covariate_stats = prebound.bindings.covariate_stats.clone();
        by_hand.model =
            crate::parser::model_parser::parse_full_model_with(&text, &by_hand.bindings)
                .unwrap()
                .model;
        bind_theta_levels_from_fit(&mut by_hand, &text, &mut want, &b.levels).unwrap();
        assert_eq!(prebound.bindings.covariate_stats["WT"].median, 90.0);
        assert!(symbolic_covariates(&prebound.model).is_empty());
        assert_eq!(
            prebound.model.theta_blocks().level_blocks()[0].contrast(),
            LevelContrast::SumToZeroWithin
        );
        assert_eq!(twin(&prebound), twin(&by_hand));
        assert_eq!(canon(&got), canon(&want));
    }

    /// #1735 review r1, finding 2: a model laid out by the deprecated binder with no
    /// statistics installed carries the fit's levels but no fit centre. A
    /// `bind_covariate_stats` on the design then refuses, naming the relation and
    /// what to do, rather than returning `Ok` with the relation unresolved (the
    /// failure would surface later, at `simulate` or `fit`) or centring on the
    /// design (#1619). Straddle, same test: with the fit's statistics installed
    /// first, the same call is the no-op that keeps the fit's median.
    ///
    /// Mutations — return `Ok` on every fit-bound model: the first arm binds `Ok`.
    /// Drop the flag check: the first arm centres on the design, and the second
    /// re-centres on it. Delete either sentence of the message: the text differs.
    #[test]
    fn a_fit_bound_model_without_the_fits_statistics_is_refused() {
        let text = model(true, true);
        let b = fitted(&text);
        let mut design = weighed(3, 2, 80.0);
        let mut parsed = parse_full_model(&text).unwrap();
        bind_theta_levels_from_fit(&mut parsed, &text, &mut design, &b.levels).unwrap();
        assert!(parsed.model.bound_from_fit());
        assert_eq!(symbolic_covariates(&parsed.model), vec!["WT".to_string()]);
        let before = twin(&parsed);
        let err = bind_covariate_stats(&mut parsed, &text, &design)
            .expect_err("no fit centre to keep, and the design's is not the fit's")
            .to_string();
        assert_eq!(
            err,
            "[covariate_model] relations still need data-derived statistics:\n  \
             V ~ WT power(center = median) => THETA_V_WT(0.6, 0.01, 5.0)\n\
             This model is laid out on a fit's levels, so its centres must be the \
             fit's too, not those of the data at hand. Bind it with `bind_from_fit` \
             and the fit's `data_bindings`, which carry both, or install the fit's \
             covariate statistics before binding its levels."
        );
        assert_eq!(twin(&parsed), before, "parsed untouched");

        // The fit's statistics installed first: the no-op that keeps them.
        let mut design = weighed(3, 2, 80.0);
        let mut parsed = parse_full_model(&text).unwrap();
        parsed.bindings.covariate_stats = b.covariate_stats.clone();
        parsed.model = crate::parser::model_parser::parse_full_model_with(&text, &parsed.bindings)
            .unwrap()
            .model;
        bind_theta_levels_from_fit(&mut parsed, &text, &mut design, &b.levels).unwrap();
        bind_covariate_stats(&mut parsed, &text, &design).unwrap();
        assert_eq!(parsed.bindings.covariate_stats["WT"].median, 70.0);
        assert!(symbolic_covariates(&parsed.model).is_empty());
    }

    /// T7: re-binding on the data a model is already bound to changes nothing —
    /// each binder alone and the pair — so the "keep the other half" predicates are
    /// not over-eager.
    ///
    /// Mutations — drop the statistics in `bind_theta_levels` or the levels in
    /// `bind_covariate_stats` unconditionally: that binder's cell loses a half.
    #[test]
    fn re_binding_on_the_same_data_is_idempotent() {
        let text = model(true, true);
        let mut once_pop = weighed(3, 2, 80.0);
        let once = pre_bound(&text, &mut once_pop);
        type Rebind = fn(&mut ParsedModel, &str, &mut Population);
        let rebinds: [(&str, Rebind); 3] = [
            ("levels", |p, t, pop| {
                crate::api::bind_theta_levels(p, t, pop).unwrap()
            }),
            ("statistics", |p, t, pop| {
                bind_covariate_stats(p, t, pop).unwrap()
            }),
            ("pair", both),
        ];
        for (name, rebind) in rebinds {
            let mut pop = weighed(3, 2, 80.0);
            let mut parsed = pre_bound(&text, &mut pop);
            rebind(&mut parsed, &text, &mut pop);
            assert_eq!(twin(&parsed), twin(&once), "{name}");
            assert_eq!(canon(&pop), canon(&once_pop), "{name}");
        }
    }
}

/// #1773: the binders return `EngineError` carrying the code `ferx check` reports
/// for the same refusal. Each code is assigned in one helper per half
/// (`level_binding_error`, `stats_binding_error`), which both the binders and
/// `bind_for_check` go through.
mod binding_codes {
    #![allow(deprecated)]
    use super::bind_from_fit::{fitted, model, weighed};
    use super::*;
    use crate::api::validation::bind_for_check;
    use crate::api::{
        bind_covariate_stats, bind_from_fit, bind_theta_levels, bind_theta_levels_from_fit,
        layout_from_fit,
    };
    use crate::diagnostics::{Diagnostic, EngineError};
    use crate::parser::model_parser::{DataBindings, LevelContrast};

    const LEVEL: (&str, &str) = ("E_THETA_LEVEL_BINDING", "parameters");
    const STATS: (&str, &str) = ("E_COVARIATE_STATS_BINDING", "covariate_model");

    fn assert_code(e: &EngineError, (code, block): (&str, &str), what: &str) {
        assert_eq!(e.code(), Some(code), "{what}: {e}");
        assert_eq!(e.block(), Some(block), "{what}: {e}");
    }

    /// `ferx check`'s binding step on a fresh parse of `text`.
    fn check(text: &str, pop: &mut Population) -> Result<(), Diagnostic> {
        let mut parsed = parse_full_model(text).unwrap();
        bind_for_check(&mut parsed, text, pop)
    }

    /// The binder's error and the check's diagnostic are one refusal: code, block,
    /// suggestion and text.
    fn assert_same(e: &EngineError, d: &Diagnostic, what: &str) {
        assert_eq!(e.code(), Some(d.code.as_str()), "{what}");
        assert_eq!(e.block(), d.block.as_deref(), "{what}");
        assert_eq!(e.suggestion(), d.suggestion.as_deref(), "{what}");
        assert_eq!(e.to_string(), d.message, "{what}");
    }

    /// A refusal `ferx check` can reach, through each binder that reaches it, is
    /// the check's own diagnostic: a level column missing on subject `1` (the
    /// binder on the data's levels, and `bind_from_fit` on a design), and a
    /// symbolic `WT` with no value in the data.
    ///
    /// Mutations — tag either half at the `pub` wrapper only and let
    /// `bind_for_check` assign its own code: the codes drift and `assert_same` dies;
    /// swap the two helpers' codes: both `assert_code`s die.
    #[test]
    fn a_binder_refusal_check_can_reach_has_check_code_block_and_text() {
        let text = no_eta_model();
        let no_study = || {
            let mut pop = population(2, 2);
            pop.subjects[0].covariates.remove("STUDY");
            pop
        };
        let d = check(&text, &mut no_study()).expect_err("check refuses");
        assert!(
            d.message
                .contains("column `STUDY` is not in the data (subject 1)"),
            "{d:?}"
        );
        let mut parsed = parse_full_model(&text).unwrap();
        let e = bind_theta_levels(&mut parsed, &text, &mut no_study()).expect_err("own levels");
        assert_code(&e, LEVEL, "bind_theta_levels");
        assert_same(&e, &d, "bind_theta_levels");

        let mut fit_pop = population(2, 2);
        let mut fit = parse_full_model(&text).unwrap();
        bind_theta_levels(&mut fit, &text, &mut fit_pop).unwrap();
        let mut parsed = parse_full_model(&text).unwrap();
        let e = bind_from_fit(
            &mut parsed,
            &text,
            &mut no_study(),
            fit.model.data_bindings(),
        )
        .expect_err("design side");
        assert_code(&e, LEVEL, "bind_from_fit");
        assert_same(&e, &d, "bind_from_fit");

        let text = model(false, true);
        let no_wt = population(3, 2);
        let d = check(&text, &mut no_wt.clone()).expect_err("check refuses");
        assert!(d.message.contains("no non-missing value"), "{d:?}");
        let mut parsed = parse_full_model(&text).unwrap();
        let e = bind_covariate_stats(&mut parsed, &text, &no_wt).expect_err("no WT");
        assert_code(&e, STATS, "bind_covariate_stats");
        assert_same(&e, &d, "bind_covariate_stats");
    }

    /// A refusal only a from-fit binder can reach carries the level code: a design
    /// level the fit never estimated (through `bind_from_fit` and the deprecated
    /// `bind_theta_levels_from_fit`), malformed bindings (`layout_from_fit`), and a
    /// model laid out on a fit handed to `bind_theta_levels`. The straddle that
    /// makes the first one from-fit-only is asserted: `ferx check` binds the same
    /// design to its own levels and accepts it, so the docs' "no `ferx check`
    /// counterpart" sentence stays true.
    ///
    /// Mutation — drop the tag at the level boundary (`fitted_level_tables` in
    /// `bind_from_fit_on`, the deprecated binder's wrapper, `validate_fitted_levels`
    /// in `lay_out_on_fit`, `level_binding_error` on `bind_levels_on_data`): that
    /// cell's code is the other half's or none.
    #[test]
    fn a_from_fit_only_refusal_carries_the_level_code() {
        let text = no_eta_model();
        let mut fit_pop = population(2, 2);
        let mut fit = parse_full_model(&text).unwrap();
        bind_theta_levels(&mut fit, &text, &mut fit_pop).unwrap();
        let b = fit.model.data_bindings().clone();

        check(&text, &mut population(3, 2)).expect("check binds the design on its own levels");

        let mut parsed = parse_full_model(&text).unwrap();
        let e = bind_from_fit(&mut parsed, &text, &mut population(3, 2), &b).expect_err("unseen");
        assert_code(&e, LEVEL, "bind_from_fit");
        assert!(
            e.to_string().starts_with(
                "theta PLACEBO[STUDY, TIME]: the design has 2 level(s) the fit estimated no \
                 theta for"
            ),
            "{e}"
        );
        let mut parsed = parse_full_model(&text).unwrap();
        let e = bind_theta_levels_from_fit(&mut parsed, &text, &mut population(3, 2), &b.levels)
            .expect_err("unseen");
        assert_code(&e, LEVEL, "bind_theta_levels_from_fit");

        let mut auto = b.clone();
        auto.levels.get_mut("PLACEBO").unwrap().contrast = LevelContrast::Auto;
        let mut parsed = parse_full_model(&text).unwrap();
        let e = layout_from_fit(&mut parsed, &text, &auto).expect_err("auto contrast");
        assert_code(&e, LEVEL, "layout_from_fit");
        assert!(e.to_string().contains("record the contrast `auto`"), "{e}");

        let mut laid = parse_full_model(&text).unwrap();
        layout_from_fit(&mut laid, &text, &b).unwrap();
        let e = bind_theta_levels(&mut laid, &text, &mut population(2, 2)).expect_err("fit-bound");
        assert_code(&e, LEVEL, "bind_theta_levels on a fit-bound model");
        assert!(e.to_string().contains("laid out on a fit's levels"), "{e}");
    }

    /// A refusal of the statistics half carries the statistics code even on a model
    /// with a level block: a fit whose statistics lack `WT`, a fit carrying a
    /// statistic no relation reads, and `bind_covariate_stats` on a model the
    /// deprecated binder laid out without the fit's centres.
    ///
    /// Mutation — tag `validate_fitted_stats` with the level helper, or
    /// `bind_stats_on_data` with it: the stats cells die.
    #[test]
    fn a_stats_refusal_carries_the_stats_code() {
        let text = model(true, true);
        let mut b = fitted(&text);
        b.covariate_stats.clear();
        for (name, run) in [("bind_from_fit", true), ("layout_from_fit", false)] {
            let mut parsed = parse_full_model(&text).unwrap();
            let e = if run {
                bind_from_fit(&mut parsed, &text, &mut weighed(3, 2, 80.0), &b)
            } else {
                layout_from_fit(&mut parsed, &text, &b)
            }
            .expect_err(name);
            assert_code(&e, STATS, name);
            assert!(
                e.to_string().contains("carry no entry for it"),
                "{name}: {e}"
            );
        }

        let level_only = model(true, false);
        let mut extra = fitted(&level_only);
        extra.covariate_stats = fitted(&text).covariate_stats;
        let mut parsed = parse_full_model(&level_only).unwrap();
        let e = layout_from_fit(&mut parsed, &level_only, &extra).expect_err("extra statistic");
        assert_code(&e, STATS, "an extra statistic");
        assert!(
            e.to_string().contains("no [covariate_model] relation"),
            "{e}"
        );

        let mut design = weighed(3, 2, 80.0);
        let mut parsed = parse_full_model(&text).unwrap();
        bind_theta_levels_from_fit(&mut parsed, &text, &mut design, &fitted(&text).levels).unwrap();
        let e = bind_covariate_stats(&mut parsed, &text, &design).expect_err("no fit centre");
        assert_code(&e, STATS, "bind_covariate_stats on a fit-bound model");
    }

    /// A fit with no bindings at all, on a model that needs them: the refusal names
    /// every half the model has, and carries the level code whenever the model has
    /// a level block — the half every binder checks first — and the statistics
    /// code only when it has none. Both sides of that gate in one test.
    ///
    /// Mutation — key the code on "a symbolic statistic is present" instead of "a
    /// level block is present": the both-halves cell gets the statistics code.
    #[test]
    fn a_two_half_refusal_is_the_level_code() {
        for (level, median, want) in [
            (true, true, LEVEL),
            (true, false, LEVEL),
            (false, true, STATS),
        ] {
            let text = model(level, median);
            let what = format!("level {level}, median {median}");
            let mut parsed = parse_full_model(&text).unwrap();
            let e = layout_from_fit(&mut parsed, &text, &DataBindings::default()).expect_err(&what);
            assert_code(&e, want, &what);
            assert!(
                e.to_string().contains("carries no data-derived bindings"),
                "{what}: {e}"
            );
            let mut parsed = parse_full_model(&text).unwrap();
            let e = bind_from_fit(
                &mut parsed,
                &text,
                &mut weighed(3, 2, 80.0),
                &DataBindings::default(),
            )
            .expect_err(&what);
            assert_code(&e, want, &what);
        }
    }
}
