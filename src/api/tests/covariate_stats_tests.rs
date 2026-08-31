//! Binding `[covariate_model]` statistics to data (#1111).

use std::collections::HashMap;

use crate::parser::model_parser::parse_full_model;
use crate::types::{CompiledModel, DoseEvent, Population, Subject};

/// A one-compartment model whose `[covariate_model]` block is supplied by the
/// caller, so each test states only the relation it is about.
fn model(covariates: &str, covariate_model: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)

  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[covariates]
{covariates}

[covariate_model]
{covariate_model}

[error_model]
  DV ~ proportional(PROP_ERR)
"#
    )
}

/// One subject per value of `WT`, each with one observation — so the summary is
/// exactly the values listed.
fn population(name: &str, values: &[f64]) -> Population {
    let subjects = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut covariates = HashMap::new();
            covariates.insert(name.to_string(), *v);
            Subject {
                id: format!("{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0],
                obs_raw_times: Vec::new(),
                observations: vec![1.0],
                obs_cmts: vec![1],
                covariates,
                cens: vec![0],
                ..Default::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec![name.to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Bind `text` against `pop`, returning the re-parsed model.
fn bind(text: &str, pop: &Population) -> Result<CompiledModel, String> {
    let mut parsed = parse_full_model(text)?;
    crate::api::bind_covariate_stats(&mut parsed, text, pop)?;
    Ok(parsed.model)
}

/// The desugared `CL = ...` line of a bound model.
fn cl_line(model: &CompiledModel) -> String {
    model
        .covariate_model
        .as_ref()
        .expect("the block is recorded")
        .desugared_individual_parameters
        .iter()
        .find(|l| l.trim_start().starts_with("CL "))
        .expect("CL is assigned")
        .trim()
        .to_string()
}

#[test]
fn a_symbolic_median_resolves_to_the_data_median() {
    let text = model("  WT continuous", "  CL ~ WT power(center = median)");
    let pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    let bound = bind(&text, &pop).expect("binding should succeed");
    assert_eq!(
        cl_line(&bound),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * exp(ETA_CL)"
    );
    let rel = &bound.covariate_model.as_ref().unwrap().relations[0];
    assert_eq!(rel.resolved_center, Some(70.0));
    // The source form survives alongside the resolved value, so a run launched
    // symbolically is still reproducible from the fit output.
    assert_eq!(rel.center.unwrap().label(), "median");
}

#[test]
fn each_statistic_resolves_to_its_own_summary() {
    for (keyword, expected) in [("mean", 70.0), ("min", 50.0), ("max", 90.0)] {
        let text = model(
            "  WT continuous",
            &format!("  CL ~ WT power(center = {keyword})"),
        );
        let pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
        let bound = bind(&text, &pop).expect("binding should succeed");
        assert_eq!(
            bound.covariate_model.as_ref().unwrap().relations[0].resolved_center,
            Some(expected),
            "center = {keyword}"
        );
    }
}

#[test]
fn the_median_weights_one_value_per_subject() {
    // A subject with many records must not drag the median toward their own
    // covariate value — PsN's weighting, and the reason this walks subjects
    // rather than observations.
    let mut pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    pop.subjects[0].obs_times = vec![1.0; 40];
    pop.subjects[0].observations = vec![1.0; 40];
    pop.subjects[0].obs_cmts = vec![1; 40];
    pop.subjects[0].cens = vec![0; 40];
    let text = model("  WT continuous", "  CL ~ WT power(center = median)");
    let bound = bind(&text, &pop).expect("binding should succeed");
    assert_eq!(
        bound.covariate_model.as_ref().unwrap().relations[0].resolved_center,
        Some(70.0)
    );
}

#[test]
fn a_time_varying_covariate_contributes_each_distinct_value() {
    let mut pop = population("WT", &[50.0, 90.0]);
    // Subject 1's weight drifts across the admission.
    let mut later = HashMap::new();
    later.insert("WT".to_string(), 70.0);
    pop.subjects[0].obs_covariates = vec![later];
    let text = model("  WT continuous", "  CL ~ WT power(center = median)");
    let bound = bind(&text, &pop).expect("binding should succeed");
    // Values are 50, 70, 90 → median 70, not the 70 the two-subject static case
    // would have averaged to.
    assert_eq!(
        bound.covariate_model.as_ref().unwrap().relations[0].resolved_center,
        Some(70.0)
    );
}

#[test]
fn the_mode_is_the_reference_level_of_a_categorical_relation() {
    let text = model(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical(ref = mode)",
    );
    let pop = population("SEX", &[0.0, 1.0, 1.0, 1.0]);
    let bound = bind(&text, &pop).expect("binding should succeed");
    let rel = &bound.covariate_model.as_ref().unwrap().relations[0];
    assert_eq!(rel.resolved_center, Some(1.0));
    // …so the θ contrasts the *other* level.
    assert_eq!(rel.thetas.len(), 1);
    assert_eq!(rel.thetas[0].name, "THETA_CL_SEX_0");
}

#[test]
fn auto_levels_are_discovered_from_the_data() {
    let text = model(
        "  SEX categorical(levels = auto)",
        "  CL ~ SEX categorical(ref = 0)",
    );
    let pop = population("SEX", &[0.0, 1.0, 2.0, 2.0]);
    let bound = bind(&text, &pop).expect("binding should succeed");
    let names: Vec<String> = bound.covariate_model.as_ref().unwrap().relations[0]
        .thetas
        .iter()
        .map(|t| t.name.clone())
        .collect();
    assert_eq!(names, vec!["THETA_CL_SEX_1", "THETA_CL_SEX_2"]);
}

#[test]
fn linear_bounds_follow_the_psn_table() {
    // PsN state 2: init 0.001/(med−min), lower 1/(med−max), upper 1/(med−min).
    // These bounds are what keep `1 + θ(COV − med)` positive over the observed
    // range, so they are adopted verbatim rather than re-invented.
    let text = model("  WT continuous", "  CL ~ WT linear(center = median)");
    let pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    let bound = bind(&text, &pop).expect("binding should succeed");
    let theta = &bound.covariate_model.as_ref().unwrap().relations[0].thetas[0];
    assert!((theta.init - 0.001 / 20.0).abs() < 1e-15);
    assert!((theta.lower - 1.0 / -20.0).abs() < 1e-15);
    assert!((theta.upper - 1.0 / 20.0).abs() < 1e-15);
}

#[test]
fn hockey_bounds_follow_the_psn_table() {
    let text = model("  WT continuous", "  CL ~ WT hockey(breakpoint = median)");
    let pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    let bound = bind(&text, &pop).expect("binding should succeed");
    let thetas = &bound.covariate_model.as_ref().unwrap().relations[0].thetas;
    assert_eq!(thetas.len(), 2);
    assert_eq!(thetas[0].name, "THETA_CL_WT_LO");
    assert!((thetas[0].lower + 1e6).abs() < 1e-9);
    assert!((thetas[0].upper - 1.0 / 20.0).abs() < 1e-15);
    assert_eq!(thetas[1].name, "THETA_CL_WT_HI");
    assert!((thetas[1].lower - 1.0 / -20.0).abs() < 1e-15);
    assert!((thetas[1].upper - 1e6).abs() < 1e-9);
}

/// A covariate value carried on a non-observation record still reaches the
/// evaluator, so it has to reach the summary (#1111 review).
///
/// Snapshots live on four vectors — observations, doses (EVID=1), covariate
/// change markers (EVID=2) and resets (EVID=3/4). Summarising only the first
/// two sources understates the spread, which is what the default θ bounds and
/// `levels = auto` are built from.
#[test]
fn every_event_snapshot_feeds_the_summary_not_only_the_observations() {
    for source in ["dose", "pk_only", "reset"] {
        let text = model("  WT continuous", "  CL ~ WT power(center = max)");
        // Two subjects at 50 and 60 by the static fallback; the largest value
        // in the dataset, 90, exists *only* on the event record under test.
        let mut pop = population("WT", &[50.0, 60.0]);
        let extreme = HashMap::from([("WT".to_string(), 90.0)]);
        let s = &mut pop.subjects[0];
        match source {
            "dose" => s.dose_covariates = vec![extreme],
            "pk_only" => {
                s.pk_only_times = vec![0.5];
                s.pk_only_covariates = vec![extreme];
            }
            _ => {
                s.reset_times = vec![0.5];
                s.reset_covariates = vec![extreme];
            }
        }
        let bound = bind(&text, &pop).expect("binding should succeed");
        let rel = &bound.covariate_model.as_ref().unwrap().relations[0];
        assert_eq!(
            rel.resolved_center,
            Some(90.0),
            "`max` must see the {source} snapshot"
        );
    }
}

/// A centre outside the observed range emits `lower > upper` under the PsN
/// default bounds (`center = 100` over 50..90 gives `(0.1, 0.02)`), so it is
/// rejected rather than handed to the optimiser (#1111 review).
#[test]
fn a_centre_outside_the_observed_range_is_rejected() {
    let text = model("  WT continuous", "  CL ~ WT linear(center = 100)");
    let pop = population("WT", &[50.0, 70.0, 90.0]);
    let e = bind(&text, &pop).expect_err("an out-of-range centre has no ordered default bounds");
    assert!(e.contains("lie strictly inside"), "{e}");
}

/// `power` and `linear_relative` divide by the centre, and division by zero
/// underflows to `0.0` in this engine — the covariate factor would collapse
/// silently. Both are rejected at parse time (#1111 review).
#[test]
fn a_centre_the_form_divides_by_must_be_usable() {
    for (form, needle) in [
        ("power(center = 0)", "positive centre"),
        ("power(center = -70)", "positive centre"),
        ("linear_relative(center = 0)", "non-zero centre"),
    ] {
        let text = model("  WT continuous", &format!("  CL ~ WT {form}"));
        let e = parse_full_model(&text)
            .err()
            .unwrap_or_else(|| panic!("`{form}` must be rejected"));
        assert!(e.contains(needle), "{form}: {e}");
    }
    // …while `linear`, which subtracts, is untouched by the same centre.
    let text = model("  WT continuous", "  CL ~ WT linear(center = 0)");
    let pop = population("WT", &[-10.0, 0.0, 10.0]);
    bind(&text, &pop).expect("`linear` may centre on zero");
}

/// `linear_relative` scales its bounds by the centre, so a negative centre
/// reverses them. They must come back ordered (#1111 review).
#[test]
fn a_negative_relative_centre_still_emits_ordered_bounds() {
    let text = model(
        "  TEMP continuous",
        "  CL ~ TEMP linear_relative(center = -5)",
    );
    let pop = population("TEMP", &[-10.0, -5.0, 10.0]);
    let bound = bind(&text, &pop).expect("binding should succeed");
    let theta = &bound.covariate_model.as_ref().unwrap().relations[0].thetas[0];
    assert!(
        theta.lower < theta.upper,
        "lower {} must be below upper {}",
        theta.lower,
        theta.upper
    );
    assert!(
        theta.init > theta.lower && theta.init < theta.upper,
        "init {} must lie inside ({}, {})",
        theta.init,
        theta.lower,
        theta.upper
    );
}

/// A categorical value the block never declared takes the same factor as the
/// reference level — the fit would silently model it as reference. The data
/// check refuses it (#1111 review).
#[test]
fn a_categorical_value_outside_the_declared_levels_is_rejected() {
    let text = model(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical(ref = 0)",
    );
    let parsed = parse_full_model(&text).expect("model should parse");

    // Declared levels only: clean.
    let clean = population("SEX", &[0.0, 1.0, 0.0]);
    assert!(
        crate::api::check_model_data(&parsed.model, &clean)
            .iter()
            .all(|d| d.code != "E_COV_LEVEL_UNKNOWN"),
        "a dataset inside the declared levels must pass"
    );

    // A third code in the data — the silent-reference case.
    let dirty = population("SEX", &[0.0, 1.0, 2.0]);
    let diags = crate::api::check_model_data(&parsed.model, &dirty);
    let hit = diags
        .iter()
        .find(|d| d.code == "E_COV_LEVEL_UNKNOWN")
        .unwrap_or_else(|| panic!("{diags:?}"));
    assert!(hit.message.contains("2.0"), "{}", hit.message);
}

/// The echoed relation table has to be machine-readable on its own: which
/// level each categorical θ belongs to, and the body of an `expr(...)`
/// relation, are otherwise recoverable only by re-parsing the model file
/// (#1111 review).
#[test]
fn the_echoed_relation_table_keeps_level_and_expression() {
    let text = model(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical(ref = 0)\n  V ~ SEX expr(\"1 + 0.1 * SEX\")",
    );
    let parsed = parse_full_model(&text).expect("model should parse");
    let names = parsed.model.theta_names.clone();
    let theta = parsed.model.default_params.theta.clone();
    let fixed = vec![false; theta.len()];
    let rels =
        crate::api::covariate_relation_estimates(&parsed.model, &names, &theta, None, &fixed);

    let cat = rels
        .iter()
        .find(|r| r.form == "categorical")
        .expect("the categorical relation is echoed");
    assert_eq!(cat.thetas.len(), 1, "one contrast against the reference");
    assert_eq!(cat.thetas[0].level, Some(1.0));
    assert_eq!(cat.expression, None);

    let expr = rels
        .iter()
        .find(|r| r.form == "expr")
        .expect("the expr relation is echoed");
    assert_eq!(expr.expression.as_deref(), Some("1 + 0.1 * SEX"));
    assert!(expr.thetas.is_empty());
}

#[test]
fn a_constant_covariate_is_rejected_rather_than_given_infinite_bounds() {
    let text = model("  WT continuous", "  CL ~ WT linear(center = median)");
    let pop = population("WT", &[70.0, 70.0, 70.0]);
    let e = bind(&text, &pop).expect_err("a constant covariate has no spread to bound against");
    assert!(e.contains("lie strictly inside"), "{e}");
}

#[test]
fn a_covariate_absent_from_the_data_is_rejected() {
    let text = model("  AGE continuous", "  CL ~ AGE power(center = median)");
    let pop = population("WT", &[50.0, 70.0, 90.0]);
    let e = bind(&text, &pop).expect_err("no value to summarise");
    assert!(e.contains("no non-missing value"), "{e}");
}

#[test]
fn a_literal_centred_model_needs_no_binding_at_all() {
    // The common case: nothing symbolic, so `bind` is a no-op and the model was
    // already fully desugared at parse time.
    let text = model(
        "  WT continuous",
        "  CL ~ WT power(center = 70) => T(0.75, 0.1, 1.5)",
    );
    let parsed = parse_full_model(&text).expect("model should parse");
    assert!(parsed
        .model
        .covariate_model
        .as_ref()
        .unwrap()
        .unresolved()
        .is_empty());
    assert!(crate::api::assert_covariate_model_bound(&parsed.model).is_ok());
}

#[test]
fn an_unbound_symbolic_statistic_is_a_hard_error() {
    // The failure this guards against is silent: an unresolved relation is
    // simply absent from the compiled expression, and a missing covariate
    // divides to 0.0 rather than inf here, so nothing downstream would notice.
    let text = model("  WT continuous", "  CL ~ WT power(center = median)");
    let parsed = parse_full_model(&text).expect("model should parse");
    let e = crate::api::assert_covariate_model_bound(&parsed.model)
        .expect_err("an unbound symbolic statistic must not reach a fit");
    assert!(e.contains("data-derived statistics"), "{e}");
    assert!(e.contains("bind_covariate_stats"), "{e}");
    // …and the check the fit path runs reports it under its own code.
    let pop = population("WT", &[50.0, 70.0, 90.0]);
    let diags = crate::api::check_model_data(&parsed.model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_COVSTAT_UNRESOLVED"),
        "{diags:?}"
    );
}

#[test]
fn binding_preserves_a_level_block_binding_made_first() {
    // A model may declare both a `theta NAME[COL]` level block (#1064) and a
    // symbolic covariate statistic (#1111). Each binder re-parses, so the
    // second must carry the first one's bindings — otherwise binding the
    // statistics would silently un-bind the level counts.
    let text = r#"
[parameters]
  theta TVCL(2.0, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta PLACEBO[STUDY](0.0, -10.0, 10.0)

  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV + PLACEBO

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[covariates]
  WT continuous
  STUDY continuous

[covariate_model]
  CL ~ WT power(center = median)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let mut pop = population("WT", &[50.0, 70.0, 90.0]);
    pop.covariate_names.push("STUDY".to_string());
    for (i, subject) in pop.subjects.iter_mut().enumerate() {
        subject
            .covariates
            .insert("STUDY".to_string(), (i + 1) as f64);
    }

    let mut parsed = parse_full_model(text).expect("model should parse");
    crate::api::bind_theta_levels(&mut parsed, text, &mut pop).expect("levels bind");
    crate::api::bind_covariate_stats(&mut parsed, text, &pop).expect("statistics bind");

    // The level block is still bound to the three observed studies — two free θ
    // under the default sum-to-zero contrast, which derives the third.
    assert_eq!(
        parsed
            .model
            .theta_names
            .iter()
            .filter(|n| n.starts_with("PLACEBO["))
            .count(),
        2,
        "{:?}",
        parsed.model.theta_names
    );
    // … and the covariate statistic resolved.
    assert_eq!(
        parsed.model.covariate_model.as_ref().unwrap().relations[0].resolved_center,
        Some(70.0)
    );
}
