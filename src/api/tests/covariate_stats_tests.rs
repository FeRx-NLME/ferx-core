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

/// `categorical2(ref = mode)` binds like `categorical(ref = mode)` (#1312) —
/// the reference is the most common level and the θ are the rest. This is also
/// the round-trip of what `Relation::render()` writes for the form.
#[test]
fn categorical2_binds_its_reference_to_the_mode() {
    let text = model(
        "  SEX categorical(levels = auto)",
        "  CL ~ SEX categorical2(ref = mode)",
    );
    // 2 is the mode, so it is the reference and 0 / 1 carry the θ.
    let pop = population("SEX", &[0.0, 1.0, 2.0, 2.0]);
    let bound = bind(&text, &pop).expect("binding should succeed");
    let rel = &bound.covariate_model.as_ref().unwrap().relations[0];
    assert_eq!(rel.resolved_center, Some(2.0));
    let names: Vec<String> = rel.thetas.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["THETA_CL_SEX_0", "THETA_CL_SEX_1"]);
    // …and the bound factor is the `cat2` shape, not the `cat` one.
    assert!(
        cl_line(&bound).contains("(if (SEX == 0) THETA_CL_SEX_0 else "),
        "{}",
        cl_line(&bound)
    );
    assert!(
        !cl_line(&bound).contains("1 + THETA"),
        "{}",
        cl_line(&bound)
    );
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

/// The same silent-reference trap on `categorical2` (#1312).
///
/// The fallthrough of the generated chain is still the reference branch — `1`
/// — so an undeclared code is modelled as reference in exactly the same way,
/// and the check has to fire for both forms. It keys on `is_categorical()`, so
/// this is the test that a new categorical variant cannot slip past it.
#[test]
fn a_categorical2_value_outside_the_declared_levels_is_rejected_too() {
    let text = model(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical2(ref = 0)",
    );
    let parsed = parse_full_model(&text).expect("model should parse");

    let clean = population("SEX", &[0.0, 1.0, 0.0]);
    assert!(
        crate::api::check_model_data(&parsed.model, &clean)
            .iter()
            .all(|d| d.code != "E_COV_LEVEL_UNKNOWN"),
        "a dataset inside the declared levels must pass"
    );

    let dirty = population("SEX", &[0.0, 1.0, 2.0]);
    let diags = crate::api::check_model_data(&parsed.model, &dirty);
    let hit = diags
        .iter()
        .find(|d| d.code == "E_COV_LEVEL_UNKNOWN")
        .unwrap_or_else(|| panic!("{diags:?}"));
    assert!(hit.message.contains("2.0"), "{}", hit.message);
    // …and it names the form that was actually written, not `categorical`.
    assert!(hit.message.contains("categorical2(...)"), "{}", hit.message);
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

// ── #1740: an unlisted categorical level at every entry point ───────────────

/// `CL ~ GRP categorical`, levels `[1, 2, 3]`, reference 2, written literally.
fn grp_literal() -> String {
    model(
        "  GRP categorical(levels = [1, 2, 3])",
        "  CL ~ GRP categorical(ref = 2)",
    )
}

/// The same relation with the levels and reference read off the data.
fn grp_auto() -> String {
    model(
        "  GRP categorical(levels = auto)",
        "  CL ~ GRP categorical(ref = mode)",
    )
}

/// The model's default parameters with every `THETA_CL_GRP_*` contrast set to
/// 0.5, so each non-reference level has a factor visibly different from the
/// reference's 1 — at the default `0.0` every level predicts alike and no
/// bit-equality below could tell a level from the reference.
fn live_params(m: &CompiledModel) -> crate::types::ModelParameters {
    let mut p = m.default_params.clone();
    let mut n = 0;
    for (i, name) in m.theta_names.iter().enumerate() {
        if name.starts_with("THETA_CL_GRP_") {
            p.theta[i] = 0.5;
            n += 1;
        }
    }
    assert_eq!(n, 2, "two contrasts: {:?}", m.theta_names);
    p
}

/// PRED per subject, one observation each.
fn preds(m: &CompiledModel, pop: &Population) -> Result<Vec<f64>, String> {
    let p = live_params(m);
    crate::api::predict(m, pop, &p).map(|r| {
        let v: Vec<f64> = r.iter().map(|r| r.pred).collect();
        assert!(v.iter().all(|x| x.is_finite()), "{v:?}");
        v
    })
}

/// #1740 T1. `predict` refuses a categorical value outside the relation's levels
/// instead of scoring it as the reference. The differential pair straddles the
/// gate: subject 3 is `4` in one design and `2` (the reference) in its twin.
/// Without the gate both are `Ok` with the same PRED bits (measured on `202bea5e`),
/// so the twin must stay `Ok` and the `4` design must turn `Err`.
///
/// The message is the literal-levels cell: each sentence is asserted — the facts
/// (relation, levels, reference, the unseen value), the consequence, and the advice
/// a literal model can act on — and the from-fit wording is asserted absent.
#[test]
fn predict_refuses_a_categorical_value_outside_the_levels() {
    let m = parse_full_model(&grp_literal()).expect("parse").model;
    assert!(!m.bound_from_fit());
    let twin = population("GRP", &[1.0, 2.0, 2.0]);
    let unseen = population("GRP", &[1.0, 2.0, 4.0]);

    let ok = preds(&m, &twin).expect("listed levels only must predict");
    // The pair is live: level 1 has its own factor, so a level is not the reference.
    assert_ne!(ok[0].to_bits(), ok[1].to_bits(), "{ok:?}");
    assert_eq!(ok[1].to_bits(), ok[2].to_bits(), "{ok:?}");

    let e = preds(&m, &unseen).expect_err("an unlisted level must be refused");
    for needle in [
        "`CL ~ GRP categorical(...)` has levels [1.0, 2.0, 3.0] (reference 2)",
        "`GRP` takes [4.0] in this data",
        "has no θ of its own and takes the reference level's factor",
        "modelled as the reference",
        "Add the value to `GRP categorical(levels = [...])` (and refit)",
        "`levels = auto`",
        "drop or recode those rows",
    ] {
        assert!(e.contains(needle), "missing {needle:?}: {e}");
    }
    for absent in ["the fit's levels", "The fit estimated no θ"] {
        assert!(!e.contains(absent), "{absent:?} is the from-fit cell: {e}");
    }
}

/// #1740 T2. Both sides of the advice's `bound_from_fit()` gate in one test. A
/// model laid out on an `auto` fit's bindings has the fit's θ vector, so the
/// literal advice ("Add the value", `levels = auto`) is not the repair there — the
/// message must say the fit estimated no θ for the value and leave it out. (A
/// written-out relation in a from-fit model is T2b.) The literal model on the
/// same design gets the literal advice. Forcing the branch either way reddens one
/// half.
#[test]
fn the_advice_for_an_unseen_level_depends_on_whether_the_model_came_from_a_fit() {
    let text = grp_auto();
    // The fit's data: levels [1, 2, 3], mode 2.
    let fit_pop = population("GRP", &[1.0, 2.0, 2.0, 3.0]);
    let bindings = bind(&text, &fit_pop).expect("bind").data_bindings().clone();
    let mut parsed = parse_full_model(&text).expect("parse");
    let mut design = population("GRP", &[1.0, 2.0, 4.0]);
    crate::api::bind_from_fit(&mut parsed, &text, &mut design, &bindings).expect("from fit");
    let from_fit = parsed.model;
    assert!(from_fit.bound_from_fit());

    let e = preds(&from_fit, &design).expect_err("from fit: unlisted level refused");
    for needle in [
        "has the fit's levels [1.0, 2.0, 3.0] (reference 2)",
        "`GRP` takes [4.0] in this data",
        "modelled as the reference",
        "The fit estimated no θ for these values",
        "drop or recode those rows, or refit on data that carries them",
    ] {
        assert!(e.contains(needle), "from fit, missing {needle:?}: {e}");
    }
    for absent in ["levels = auto", "Add the value"] {
        assert!(
            !e.contains(absent),
            "from fit, {absent:?} is wrong here: {e}"
        );
    }

    let literal = parse_full_model(&grp_literal()).expect("parse").model;
    let e = preds(&literal, &design).expect_err("literal: unlisted level refused");
    assert!(e.contains("has levels [1.0, 2.0, 3.0]"), "{e}");
    assert!(e.contains("`levels = auto`"), "{e}");
    assert!(!e.contains("The fit estimated no θ"), "{e}");

    // The listed twin of the design predicts from the fit's layout.
    let mut twin = population("GRP", &[1.0, 2.0, 2.0]);
    let mut parsed = parse_full_model(&text).expect("parse");
    crate::api::bind_from_fit(&mut parsed, &text, &mut twin, &bindings).expect("from fit");
    preds(&parsed.model, &twin).expect("from fit: listed levels predict");
}

/// #1740 T2b (review r1 #1/#3). `bound_from_fit()` is model-wide: a model is laid
/// out on a fit because of *any* data-derived relation — here `V ~ WT power(center =
/// median)` — so a relation whose levels are **written out** is in the from-fit
/// cell too. There, refitting on data that carries the value is not enough on its
/// own: the refit is refused the same way until the value is added to `levels =
/// [...]`, so the from-fit advice has to say so. Asserted from both ends: the
/// from-fit message names that repair, and the refit it would otherwise send the
/// reader to is refused, with the literal advice.
#[test]
fn the_from_fit_advice_covers_a_relation_whose_levels_are_written_out() {
    let text = model(
        "  GRP categorical(levels = [1, 2, 3])\n  WT continuous",
        "  CL ~ GRP categorical(ref = 2)\n  V ~ WT power(center = median)",
    );
    let with_wt = |grp: &[f64]| {
        let mut pop = population("GRP", grp);
        for (i, s) in pop.subjects.iter_mut().enumerate() {
            s.covariates
                .insert("WT".to_string(), 50.0 + 10.0 * i as f64);
        }
        pop.covariate_names.push("WT".to_string());
        pop
    };
    let bindings = bind(&text, &with_wt(&[1.0, 2.0, 2.0, 3.0]))
        .expect("bind")
        .data_bindings()
        .clone();
    let mut parsed = parse_full_model(&text).expect("parse");
    let mut design = with_wt(&[1.0, 2.0, 4.0]);
    crate::api::bind_from_fit(&mut parsed, &text, &mut design, &bindings).expect("from fit");
    assert!(
        parsed.model.bound_from_fit(),
        "the WT relation lays the model out"
    );

    let e = preds(&parsed.model, &design).expect_err("from fit: unlisted level refused");
    assert!(e.contains("The fit estimated no θ for these values"), "{e}");
    assert!(
        e.contains("adding them to `levels = [...]` first if the levels are written out"),
        "the written-out case needs its repair: {e}"
    );

    // The refit on the design, as the advice's "refit" half would have it.
    let refit = bind(&text, &design).expect("the design binds its own WT median");
    let diags = crate::api::check_model_data(&refit, &design);
    let hit = diags
        .iter()
        .find(|d| d.code == "E_COV_LEVEL_UNKNOWN")
        .unwrap_or_else(|| panic!("the refit is refused too: {diags:?}"));
    assert!(hit.message.contains("Add the value"), "{}", hit.message);
}

/// #1740 T3. The check reads every record a value can arrive on, the same set the
/// summary does. A literal model whose unlisted `4` exists **only on a dose
/// record** is refused — a check reading the static covariates alone would pass
/// it, and the dose would be given under the reference factor. The `auto` twin on
/// the same population discovers `4` from that dose record and binds a θ for it,
/// so its check is clean: the "auto bound on the data at hand" cell cannot reach
/// the refusal.
#[test]
fn a_level_seen_only_on_a_dose_record_is_checked_like_any_other() {
    let mut pop = population("GRP", &[1.0, 2.0, 2.0, 3.0]);
    pop.subjects[0].dose_covariates = vec![HashMap::from([("GRP".to_string(), 4.0)])];

    let literal = parse_full_model(&grp_literal()).expect("parse").model;
    let hit = crate::api::check_covariate_levels(&literal, &pop);
    assert_eq!(hit.len(), 1, "{hit:?}");
    assert!(hit[0].message.contains("[4.0]"), "{}", hit[0].message);

    let auto = bind(&grp_auto(), &pop).expect("auto binds 4 from the dose record");
    let levels: Vec<f64> = auto.covariate_model.as_ref().unwrap().relations[0]
        .thetas
        .iter()
        .filter_map(|t| t.level)
        .collect();
    assert!(levels.contains(&4.0), "{levels:?}");
    assert!(
        crate::api::check_covariate_levels(&auto, &pop).is_empty(),
        "auto on its own data has no unseen level"
    );
}

/// #1740 T4. A missing value is not a level: it keeps the documented neutral
/// branch, which is numerically the reference factor, and is not refused. The
/// NaN subject's PRED is bit-equal to its reference twin's, and level 1 is not —
/// so the equality is the neutral branch, not a flat model.
#[test]
fn a_missing_categorical_value_is_neutral_not_refused() {
    let m = parse_full_model(&grp_literal()).expect("parse").model;
    let pop = population("GRP", &[1.0, 2.0, f64::NAN]);
    assert!(crate::api::check_covariate_levels(&m, &pop).is_empty());
    let p = preds(&m, &pop).expect("a missing value predicts");
    assert_eq!(p[2].to_bits(), p[1].to_bits(), "{p:?}");
    assert_ne!(p[0].to_bits(), p[1].to_bits(), "{p:?}");
}

/// #1740 T5. `run_covariance` and `run_sir` with a supplied population: the same
/// subject IDs pass `check_subjects`, so a recoded covariate column reached the
/// re-scored objective as the reference level. Both now refuse it, prefixed by the
/// entry point. The control: the fit's own population runs `run_covariance` to
/// `Ok`, so the refusal is the level, not the call.
#[test]
fn run_covariance_and_run_sir_refuse_an_unseen_level_in_a_supplied_population() {
    let m = parse_full_model(&grp_literal()).expect("parse").model;
    let fit_pop = population("GRP", &[1.0, 2.0, 2.0, 3.0]);
    let opts = crate::types::FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..crate::types::FitOptions::default()
    };
    let fit = crate::api::fit(&m, &fit_pop, &m.default_params, &opts).expect("fit");
    let recoded = population("GRP", &[1.0, 2.0, 2.0, 4.0]);

    for (entry, err) in [
        (
            "run_covariance",
            crate::run_covariance(&fit, Some(&m), Some(&recoded), &opts).expect_err("refused"),
        ),
        (
            "run_sir",
            crate::run_sir(&fit, Some(&m), Some(&recoded), &opts).expect_err("refused"),
        ),
    ] {
        assert!(err.starts_with(&format!("{entry}: ")), "{err}");
        assert!(
            err.contains("`GRP` takes [4.0] in this data"),
            "{entry}: {err}"
        );
    }
    crate::run_covariance(&fit, Some(&m), Some(&fit_pop), &opts)
        .expect("the fit's own population is not refused");
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
    // #1619: the same text reaches a caller holding a fit's θ, for whom the
    // fit-time binder is the wrong one (it centres on the data at hand). Each
    // clause of the from-fit remedy, asserted on its own.
    assert!(
        e.contains(
            " To run the model with a fit's θ instead (a simulation, a prediction, SIR or a \
             covariance step), bind it with `ferx_core::api::bind_from_fit` and the fit's \
             `data_bindings`:"
        ),
        "the from-fit binder: {e}"
    );
    assert!(
        e.ends_with(
            " statistics taken from the data at hand would centre the relations on that data, \
             not on the data the θ was estimated from."
        ),
        "why not the fit-time binder: {e}"
    );
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

// ── `ferx check` (`validate_model_file`) ───────────────────────────────────

/// The model text written to a `.ferx` temp file, for the entry points that
/// take a path rather than a string.
fn temp_model(src: &str) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::Builder::new()
        .suffix(".ferx")
        .tempfile()
        .expect("create temp model");
    f.write_all(src.as_bytes()).expect("write temp model");
    f.flush().expect("flush temp model");
    f
}

#[test]
fn check_without_data_warns_that_the_statistics_are_still_unbound() {
    // `center = median` cannot be resolved from the file alone, so the relation
    // is parsed and recorded but not desugared. Without a dataset that is not an
    // error — the model is fine, just not buildable yet — but it has to be said:
    // the desugared echo is suppressed in this state, and a silent "no errors"
    // on a model whose covariate effects are all still pending reads as "there
    // are none".
    let text = model("  WT continuous", "  CL ~ WT power(center = median)");
    let f = temp_model(&text);
    let report = crate::api::validate_model_file(f.path().to_str().expect("utf-8 temp path"), None);

    let hit = report
        .diagnostics
        .iter()
        .find(|d| d.code == "W_COVSTAT_UNBOUND")
        .unwrap_or_else(|| {
            panic!(
                "unbound statistics must be reported: {:?}",
                report.diagnostics
            )
        });
    assert_eq!(hit.severity, crate::Severity::Warning);
    assert!(hit.message.contains("`CL ~ WT`"), "{}", hit.message);
    assert!(
        hit.suggestion
            .as_deref()
            .is_some_and(|s| s.contains("--data")),
        "the suggestion must point at re-running with data: {:?}",
        hit.suggestion
    );
    // A warning does not invalidate the report …
    assert!(report.valid, "an unbound statistic is not a rejection");
    // … but the echo stays empty: those lines are the block *without* the
    // pending covariate effect, so printing them would state the opposite of
    // the truth.
    assert!(
        report.desugared_individual_parameters.is_empty(),
        "{:?}",
        report.desugared_individual_parameters
    );
}

#[test]
fn check_with_data_binds_the_statistics_and_echoes_the_desugared_block() {
    // The same path with a dataset: `bind_covariate_stats` runs, nothing is left
    // unresolved, and the report carries the expression that was actually built
    // — the text to diff against a NONMEM control stream.
    let report = crate::api::validate_model_file(
        "examples/two_cpt_oral_covmodel.ferx",
        Some("data/two_cpt_oral_cov.csv"),
    );
    assert!(report.valid, "{:?}", report.diagnostics);
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|d| d.code == "W_COVSTAT_UNBOUND" || d.code == "E_COVSTAT_UNRESOLVED"),
        "statistics must be bound once data is supplied: {:?}",
        report.diagnostics
    );
    let cl = report
        .desugared_individual_parameters
        .iter()
        .find(|l| l.trim_start().starts_with("CL "))
        .unwrap_or_else(|| {
            panic!(
                "the desugared block must be echoed: {:?}",
                report.desugared_individual_parameters
            )
        });
    assert!(cl.contains("present(WT)"), "{cl}");
    assert!(cl.contains("^THETA_CL_WT"), "{cl}");
}

/// One covariate column written as a NONMEM CSV, one dose and one observation per
/// subject, for the entry points that read a data file. A `None` cell is `.`.
fn temp_cov_csv(name: &str, values: &[Option<f64>]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("create temp data");
    writeln!(f, "ID,TIME,DV,EVID,AMT,CMT,MDV,{name}").unwrap();
    for (i, v) in values.iter().enumerate() {
        let g = v.map_or(".".to_string(), |v| format!("{v}"));
        writeln!(f, "{},0,.,1,100,1,1,{g}", i + 1).unwrap();
        writeln!(f, "{},1,2.0,0,.,1,0,{g}", i + 1).unwrap();
    }
    f.flush().expect("flush temp data");
    f
}

/// #1739 T9. A `[covariate_model]` statistic the data cannot bind is reported as
/// `E_COVARIATE_STATS_BINDING` on block `covariate_model` — not as the level-block
/// code `E_THETA_LEVEL_BINDING` on `parameters`, which sent the reader to the wrong
/// block. Two binder sources, so the code cannot depend on which one fired: an
/// `auto` relation whose data carry one level ("nothing to estimate"), and a
/// covariate with no non-missing value. The level-block side keeps its code; that
/// is `tests/theta_level_blocks.rs::check_reports_level_binding_errors_directly`.
#[test]
fn a_covariate_statistic_bind_failure_carries_its_own_code_and_block() {
    let text = model(
        "  GRP categorical(levels = auto)",
        "  CL ~ GRP categorical(ref = mode)",
    );
    let m = temp_model(&text);
    for (label, values, needle) in [
        ("single level", vec![Some(2.0); 4], "nothing to estimate"),
        ("all missing", vec![None; 4], "no non-missing value"),
    ] {
        let d = temp_cov_csv("GRP", &values);
        let report = crate::api::validate_model_file(
            m.path().to_str().expect("utf-8 temp path"),
            Some(d.path().to_str().expect("utf-8 temp path")),
        );
        assert!(!report.valid, "{label}: {:?}", report.diagnostics);
        let binds: Vec<_> = report
            .diagnostics
            .iter()
            .filter(|d| d.code.ends_with("_BINDING"))
            .collect();
        assert_eq!(binds.len(), 1, "{label}: {:?}", report.diagnostics);
        let hit = binds[0];
        assert_eq!(hit.code, "E_COVARIATE_STATS_BINDING", "{label}: {hit:?}");
        assert_eq!(
            hit.block.as_deref(),
            Some("covariate_model"),
            "{label}: {hit:?}"
        );
        assert!(hit.message.contains(needle), "{label}: {}", hit.message);
    }
}

/// #1743. The three ways the check's binding step fails each carry their own code
/// and block, asserted in one test so no arm can borrow another's: a failed
/// re-read of the model file is `E_MODEL_REREAD` with **no** block, even on a
/// model with no level block (the arm was `E_THETA_LEVEL_BINDING` on
/// `parameters`); a level block that cannot bind keeps `E_THETA_LEVEL_BINDING` on
/// `parameters`; a statistic that cannot bind keeps `E_COVARIATE_STATS_BINDING`
/// on `covariate_model`.
///
/// Mutations — restore the old code or block on the re-read arm: the first cell.
/// Delete either sentence of its message: the `contains` on that sentence.
/// Swap the two binders' codes: the second and third cells.
#[test]
fn each_binding_failure_of_the_check_carries_its_own_code() {
    use crate::api::validation::bind_for_check;
    let bind = |text: &str, read: std::io::Result<String>, pop: &mut Population| {
        let mut parsed = parse_full_model(text).unwrap();
        bind_for_check(&mut parsed, "m.ferx", read, pop).expect_err("the bind fails")
    };
    // No level block: the re-read arm is no level-block error.
    let text = grp_auto();
    let reread = bind(
        &text,
        Err(std::io::Error::other("gone")),
        &mut population("GRP", &[1.0, 2.0]),
    );
    assert_eq!(reread.code, "E_MODEL_REREAD", "{reread:?}");
    assert_eq!(reread.block, None, "{reread:?}");
    assert!(
        reread
            .message
            .contains("Failed to re-read the model file `m.ferx` to bind it to the data: gone."),
        "{}",
        reread.message
    );
    assert!(
        reread.message.contains(
            "The check had already read it, so it was moved, deleted or made unreadable \
             while the check ran."
        ),
        "{}",
        reread.message
    );

    let level = text
        .replace(
            "  theta TVV(40.0, 1.0, 500.0)",
            "  theta TVV(40.0, 1.0, 500.0)\n  theta PLACEBO[GRP](0.5, -5.0, 5.0)",
        )
        .replace("V  = TVV", "V  = TVV * exp(PLACEBO)");
    let levels = bind(
        &level,
        Ok(level.clone()),
        &mut population("GRP", &[1.0, 2.0]),
    );
    assert_eq!(levels.code, "E_THETA_LEVEL_BINDING", "{levels:?}");
    assert_eq!(levels.block.as_deref(), Some("parameters"), "{levels:?}");

    let stats = bind(&text, Ok(text.clone()), &mut population("GRP", &[2.0, 2.0]));
    assert_eq!(stats.code, "E_COVARIATE_STATS_BINDING", "{stats:?}");
    assert_eq!(stats.block.as_deref(), Some("covariate_model"), "{stats:?}");
    assert!(
        stats.message.contains("nothing to estimate"),
        "{}",
        stats.message
    );
}

/// #1743, end to end: a `validate_model_file` run whose model file is deleted
/// after the parse, while the data are read, reports `E_MODEL_REREAD` with no
/// block, and no binding code. The data file is a FIFO, so the writer controls
/// when the read finishes: it deletes the model while the reader blocks on the
/// pipe, then writes the rows. Control, in the same test: the same files with the
/// model left in place validate clean of every binding code.
///
/// Mutations — restore the old code on the re-read arm: the `code` assertion.
/// Restore its `parameters` block: the `block` assertion.
#[cfg(unix)]
#[test]
fn a_model_file_gone_by_the_rebind_is_its_own_error() {
    use std::io::Write;
    let text = grp_auto();
    let rows = {
        let d = temp_cov_csv("GRP", &[Some(1.0), Some(2.0), Some(1.0), Some(2.0)]);
        std::fs::read_to_string(d.path()).unwrap()
    };
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.ferx");
    let binding_codes = |r: &crate::diagnostics::CheckReport| -> Vec<_> {
        r.diagnostics
            .iter()
            .filter(|d| d.code.ends_with("_BINDING") || d.code == "E_MODEL_REREAD")
            .cloned()
            .collect()
    };

    for gone in [true, false] {
        std::fs::write(&model, &text).unwrap();
        let fifo = dir.path().join(format!("data-{gone}.csv"));
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("run mkfifo");
        assert!(made.success(), "mkfifo");
        let writer = {
            let (fifo, model, rows) = (fifo.clone(), model.clone(), rows.clone());
            std::thread::spawn(move || {
                // Opening blocks until the check opens the data: the parse is done.
                let mut f = std::fs::OpenOptions::new().write(true).open(&fifo).unwrap();
                if gone {
                    std::fs::remove_file(&model).unwrap();
                }
                f.write_all(rows.as_bytes()).unwrap();
            })
        };
        let report =
            crate::api::validate_model_file(model.to_str().unwrap(), Some(fifo.to_str().unwrap()));
        writer.join().unwrap();
        let hits = binding_codes(&report);
        if gone {
            assert_eq!(hits.len(), 1, "{:?}", report.diagnostics);
            assert_eq!(hits[0].code, "E_MODEL_REREAD", "{:?}", hits[0]);
            assert_eq!(hits[0].block, None, "{:?}", hits[0]);
        } else {
            assert!(hits.is_empty(), "control: {:?}", report.diagnostics);
        }
    }
}

/// The #1738 fixture: `CL ~ WT power` with an explicit `=> THETA_CL_WT(...)`,
/// centred on `center`, plus a genuinely unused `UNUSED_T` as the control.
fn unused_census_model(center: &str) -> String {
    model(
        "  WT continuous",
        &format!("  CL ~ WT power(center = {center}) => THETA_CL_WT(0.6, 0.01, 5.0)"),
    )
    .replace(
        "  theta TVV(",
        "  theta UNUSED_T(1.0, 0.1, 10.0)\n  theta TVV(",
    )
}

/// WT 55, 58, …, 88: twelve subjects, median 71.5.
fn census_weights() -> Vec<f64> {
    (0..12).map(|i| 55.0 + 3.0 * f64::from(i)).collect()
}

/// The θ names a parse warns about as "not referenced", in warning order.
fn unreferenced(warnings: &[String]) -> Vec<String> {
    warnings
        .iter()
        .filter(|w| w.contains("not referenced"))
        .map(|w| w.split('\'').nth(1).unwrap_or(w).to_string())
        .collect()
}

/// #1738 T10. A θ that only a not-yet-resolved relation reads is not unused: the
/// relation emits its expression as soon as it is bound, and the census must not
/// say otherwise in the meantime. The symbolic model, unbound and bound, warns
/// about exactly what the literal twin centred on the same value (the data's
/// median, 71.5) warns about — `UNUSED_T`, the control, and nothing else. The
/// control is what keeps the fix from being "count every `[covariate_model]` θ"
/// or "stop warning": `UNUSED_T` must keep warning in every cell.
#[test]
fn a_theta_read_only_by_an_unresolved_relation_is_not_reported_unused() {
    let pop = population("WT", &census_weights());
    let literal = parse_full_model(&unused_census_model("71.5")).expect("parse");
    let symbolic_text = unused_census_model("median");
    let symbolic = parse_full_model(&symbolic_text).expect("parse");
    // The fixture is the unresolved case, or it tests nothing.
    assert_eq!(
        symbolic
            .model
            .covariate_model
            .as_ref()
            .expect("block recorded")
            .unresolved()
            .len(),
        1
    );
    let bound = bind(&symbolic_text, &pop).expect("bind");
    assert_eq!(
        bound.covariate_model.as_ref().unwrap().relations[0].resolved_center,
        Some(71.5),
        "the literal twin must sit on the bound centre"
    );

    let want = vec!["UNUSED_T".to_string()];
    assert_eq!(unreferenced(&literal.model.parse_warnings), want, "literal");
    assert_eq!(
        unreferenced(&symbolic.model.parse_warnings),
        want,
        "symbolic, unbound"
    );
    assert_eq!(unreferenced(&bound.parse_warnings), want, "symbolic, bound");
}

/// #1738 T11. The same through `ferx check`, with and without `--data`: the
/// symbolic and the literal model report the same `W_UNUSED_PARAM` set.
#[test]
fn check_reports_the_same_unused_parameters_for_a_symbolic_and_a_literal_centre() {
    let unused = |center: &str, with_data: bool| -> Vec<String> {
        let m = temp_model(&unused_census_model(center));
        let d = temp_cov_csv(
            "WT",
            &census_weights().into_iter().map(Some).collect::<Vec<_>>(),
        );
        let report = crate::api::validate_model_file(
            m.path().to_str().expect("utf-8 temp path"),
            with_data.then(|| d.path().to_str().expect("utf-8 temp path")),
        );
        let mut hits: Vec<String> = report
            .diagnostics
            .iter()
            .filter(|d| d.code == "W_UNUSED_PARAM")
            .map(|d| {
                d.message
                    .split('\'')
                    .nth(1)
                    .unwrap_or(&d.message)
                    .to_string()
            })
            .collect();
        hits.sort();
        hits
    };
    for with_data in [false, true] {
        let literal = unused("71.5", with_data);
        assert_eq!(literal, vec!["UNUSED_T".to_string()], "data = {with_data}");
        assert_eq!(unused("median", with_data), literal, "data = {with_data}");
    }
}

/// #1729 T6. The comparator behind the empty-bindings check names the first field
/// that differs, in the order median, mean, min, max, mode, levels. Each row makes
/// its field **and every later one** differ, so the earliest is the one named:
/// reordering two fields, or dropping one from the comparison, names the wrong
/// field (or none) on some row.
#[test]
fn the_stat_comparison_names_the_first_field_that_differs() {
    use crate::types::CovariateSummary;
    let base = CovariateSummary {
        median: 70.0,
        mean: 71.5,
        min: 50.0,
        max: 95.0,
        mode: 60.0,
        levels: vec![50.0, 60.0, 95.0],
    };
    assert_eq!(super::first_difference(&base, &base.clone()), None);

    let fields = ["median", "mean", "min", "max", "mode", "levels"];
    for (i, want) in fields.iter().enumerate() {
        let mut other = base.clone();
        for f in &fields[i..] {
            match *f {
                "median" => other.median += 1.0,
                "mean" => other.mean += 1.0,
                "min" => other.min += 1.0,
                "max" => other.max += 1.0,
                "mode" => other.mode += 1.0,
                _ => other.levels.push(99.0),
            }
        }
        let (field, model, data) =
            super::first_difference(&base, &other).unwrap_or_else(|| panic!("{want}: no diff"));
        assert_eq!(field, *want);
        if *want == "levels" {
            // A categorical `levels = auto` can differ in its level set alone.
            assert_eq!(model, "[50.0, 60.0, 95.0]");
            assert_eq!(data, "[50.0, 60.0, 95.0, 99.0]");
        } else {
            assert_ne!(model, data, "{want}: both values are reported");
        }
    }
    // The values are the model's then the data's, formatted plainly.
    let mut shifted = base.clone();
    shifted.median = 92.105;
    assert_eq!(
        super::first_difference(&base, &shifted),
        Some(("median", "70".to_string(), "92.105".to_string()))
    );
}

/// #1729 T6. `check_stats_on` re-summarises with the binder's own `summarize`, so a
/// model bound on a population passes on that population bit for bit, fails on
/// another naming the first covariate in name order, and carries a covariate the
/// population lacks as the summariser's error.
///
/// Mutations — compare against the model's own statistics: the second population
/// passes; drop the name sort: the named covariate follows `HashMap` order.
#[test]
fn check_stats_on_re_summarises_the_population() {
    let text = model(
        "  WT continuous\n  AGE continuous",
        "  CL ~ WT power(center = median)\n  CL ~ AGE linear(center = median)",
    );
    let mut pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    for (s, age) in pop.subjects.iter_mut().zip([20.0, 30.0, 40.0, 50.0, 60.0]) {
        s.covariates.insert("AGE".to_string(), age);
    }
    pop.covariate_names.push("AGE".to_string());
    let bound = bind(&text, &pop).expect("binds");
    assert_eq!(super::check_stats_on(&bound, &pop), Ok(None));

    let mut other = pop.clone();
    for s in &mut other.subjects {
        *s.covariates.get_mut("WT").unwrap() *= 1.3;
        *s.covariates.get_mut("AGE").unwrap() += 1.0;
    }
    // Each bind builds a fresh `HashMap` with its own random hasher, so without the
    // sort `WT` comes first in about half of these and the run fails with
    // probability 1 − 2⁻⁸.
    for _ in 0..8 {
        let bound = bind(&text, &pop).expect("binds");
        let got = super::check_stats_on(&bound, &other)
            .unwrap()
            .expect("differs");
        assert_eq!(
            (
                got.covariate.as_str(),
                got.field,
                got.model.as_str(),
                got.data.as_str()
            ),
            ("AGE", "median", "40", "41")
        );
    }

    let mut absent = pop.clone();
    for s in &mut absent.subjects {
        s.covariates.remove("AGE");
    }
    let err = super::check_stats_on(&bound, &absent).unwrap_err();
    assert!(err.contains("covariate `AGE`"), "{err}");
}
