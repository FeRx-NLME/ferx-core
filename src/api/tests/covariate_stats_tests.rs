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

/// The PsN bounds encode the positivity of a *factor*, so they must not be
/// reused for a relation that adds a term (#1316 review).
///
/// The regression this exists to catch: `linear_family_thetas` applied
/// `[1/(c−max), 1/(c−min)]` under both operators. Those bounds cap the whole
/// additive term at ~±1 absolute unit over the covariate's range, which on a
/// realistic weight range is a slope bound of ±0.04 — tighter than the 0.05 the
/// docs use and the 0.04 the NONMEM anchor uses, both of which sit *at or
/// outside* it. A covsearch additive candidate (which always takes these
/// defaults) would then be estimated against a pinned bound, and its ΔOFV would
/// under-detect the effect with no warning.
///
/// A differential pair: the two models differ by one character, and the
/// multiplicative arm is asserted too, so the test cannot pass by widening both.
#[test]
fn additive_default_bounds_are_not_the_multiplicative_positivity_bounds() {
    // The span of `data/two_cpt_oral_cov.csv`, median 70 — so the bound this
    // compares against is the measured one, `θ ∈ [−0.0422, 0.0400]`.
    let pop = population("WT", &[45.0, 60.0, 70.0, 80.0, 93.7]);
    let mul = bind(
        &model("  WT continuous", "  CL ~ WT linear(center = median)"),
        &pop,
    )
    .expect("binding should succeed");
    let add = bind(
        &model("  WT continuous", "  CL ~ WT linear(center = median) +"),
        &pop,
    )
    .expect("binding should succeed");
    let mul = mul.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone();
    let add = add.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone();

    // The multiplicative arm keeps PsN's table, unchanged.
    assert!((mul.lower - 1.0 / (70.0 - 93.7)).abs() < 1e-15, "{mul:?}");
    assert!((mul.upper - 1.0 / (70.0 - 45.0)).abs() < 1e-15, "{mul:?}");

    // The additive arm is scale-free: null at 0, effectively unbounded.
    assert!((add.lower + 1e6).abs() < 1e-9, "{add:?}");
    assert!((add.upper - 1e6).abs() < 1e-9, "{add:?}");
    assert!(add.init > 0.0 && add.init < 0.01, "{add:?}");

    // The straddle, stated as the property rather than as two numbers: the
    // slope this feature's own docs and anchor use is *outside* the
    // multiplicative bound and inside the additive one. Without it the two arms
    // could drift to any pair of different numbers and still pass.
    let doc_slope = 0.05;
    assert!(
        doc_slope > mul.upper,
        "the multiplicative bound must exclude {doc_slope}, else this pair proves nothing: {mul:?}"
    );
    assert!(doc_slope < add.upper, "{add:?}");
}

/// `linear_relative` is `linear` reparameterized as `θ_rel = c·θ_abs`, and that
/// identity has to survive the operator switch: both spellings must start the
/// optimiser at the same covariate effect.
///
/// The **bounds** deliberately do *not* follow the init through that scaling,
/// unlike the multiplicative twin where they must (there they are `1/(c − min)`
/// and `1/(c − max)`, in the same units as the init). Under `+` they are the
/// fixed scale-free sentinels, so both spellings carry the same pair — asserted
/// here, since the reparameterization is the obvious reason to scale them and
/// the init assertion alone would not notice.
#[test]
fn the_additive_init_is_the_same_effect_in_both_linear_parameterizations() {
    let pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    let abs = bind(
        &model("  WT continuous", "  CL ~ WT linear(center = median) +"),
        &pop,
    )
    .expect("binding should succeed");
    let rel = bind(
        &model(
            "  WT continuous",
            "  CL ~ WT linear_relative(center = median) +",
        ),
        &pop,
    )
    .expect("binding should succeed");
    let abs = abs.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone();
    let rel = rel.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone();
    // θ_rel = 70 · θ_abs.
    assert!(
        (rel.init - 70.0 * abs.init).abs() < 1e-15,
        "{:?} vs {:?}",
        rel,
        abs
    );
    // The bounds are the sentinels on both sides — not `70 ×` them.
    for theta in [&abs, &rel] {
        assert_eq!(theta.lower, -1e6, "{theta:?}");
        assert_eq!(theta.upper, 1e6, "{theta:?}");
    }
}

/// Additive `categorical` shifts the parameter by `θ_k` outright, so PsN's
/// `θ_k > −1` — the bound that keeps the factor `1 + θ_k` positive — does not
/// apply to it either.
#[test]
fn additive_categorical_bounds_drop_the_factor_positivity_floor() {
    let pop = population("SEX", &[0.0, 1.0, 1.0, 1.0]);
    let mul = bind(
        &model(
            "  SEX categorical(levels = [0, 1])",
            "  CL ~ SEX categorical(ref = 1)",
        ),
        &pop,
    )
    .expect("binding should succeed");
    let add = bind(
        &model(
            "  SEX categorical(levels = [0, 1])",
            "  CL ~ SEX categorical(ref = 1) +",
        ),
        &pop,
    )
    .expect("binding should succeed");
    let mul = mul.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone();
    let add = add.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone();
    assert!((mul.lower + 1.0).abs() < 1e-15, "{mul:?}");
    assert!((mul.upper - 5.0).abs() < 1e-15, "{mul:?}");
    // A clearance shift of −2 L/h is representable additively; −1 is not the
    // floor it is multiplicatively.
    assert!(add.lower < -1.0, "{add:?}");
    assert!(add.upper > 5.0, "{add:?}");
}

/// `θ_cat2 = 1 + θ_cat` is an exact reparameterization, so it has to survive the
/// operator switch: `categorical2`'s additive defaults must be the image of
/// `categorical`'s, not a second hand-written pair.
///
/// The two axes are independent — the operator picks the bounds, the shape maps
/// them — and this is what pins that they compose. `relation_effect` emits
/// `θ_k − 1` under `+` where `categorical` emits `θ_k`, so an unmapped bound
/// would put the two spellings on different models.
#[test]
fn additive_categorical2_defaults_stay_the_image_of_categorical() {
    let pop = population("SEX", &[0.0, 1.0, 1.0, 1.0]);
    let of = |form: &str, op: &str| {
        let text = model(
            "  SEX categorical(levels = [0, 1])",
            &format!("  CL ~ SEX {form}(ref = 1) {op}"),
        );
        let bound = bind(&text, &pop).expect("binding should succeed");
        bound.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone()
    };

    // Multiplicatively, the documented image: (−0.001, −1, 5) → (0.999, 0, 6).
    let (cat, cat2) = (of("categorical", "*"), of("categorical2", "*"));
    assert!((cat2.init - (cat.init + 1.0)).abs() < 1e-15, "{cat2:?}");
    assert!((cat2.lower - (cat.lower + 1.0)).abs() < 1e-15, "{cat2:?}");
    assert!((cat2.upper - (cat.upper + 1.0)).abs() < 1e-15, "{cat2:?}");
    assert!((cat2.lower - 0.0).abs() < 1e-15, "{cat2:?}");
    assert!((cat2.upper - 6.0).abs() < 1e-15, "{cat2:?}");

    // …and additively, the same map applied to the scale-free bounds. Asserted
    // against the multiplicative arm as well, so a change that made the additive
    // branch ignore the shape (or the shape branch ignore the operator) fails
    // here rather than passing on one axis.
    let (cat, cat2) = (of("categorical", "+"), of("categorical2", "+"));
    assert!((cat2.init - (cat.init + 1.0)).abs() < 1e-15, "{cat2:?}");
    assert!((cat2.lower - (cat.lower + 1.0)).abs() < 1e-15, "{cat2:?}");
    assert!((cat2.upper - (cat.upper + 1.0)).abs() < 1e-15, "{cat2:?}");
    assert!(cat.lower < -1.0, "the operator axis still applies: {cat:?}");
    assert!(cat2.lower < 0.0, "…and it survives the shape map: {cat2:?}");
}

/// The centre gate is part of the same positivity story, so it must not reject
/// an additive relation (#1316 review).
///
/// `center = 0` is the natural spelling of an uncentred additive slope `θ·WT`.
/// It is rejected multiplicatively — correctly, the PsN bounds are unordered
/// there — and the shared gate rejected it under `+` too, citing a factor the
/// additive form does not have.
///
/// The single-slope forms only: `hockey` needs its breakpoint inside the range
/// for a reason that is about support rather than positivity, and so keeps the
/// requirement under `+` — see
/// `an_additive_hockey_breakpoint_outside_the_data_is_rejected`.
#[test]
fn a_centre_outside_the_observed_range_is_legal_for_an_additive_relation() {
    let pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    let err = bind(
        &model("  WT continuous", "  CL ~ WT linear(center = 0)"),
        &pop,
    )
    .expect_err("a centre below the range has no ordered multiplicative bounds");
    assert!(err.contains("strictly inside"), "{err}");

    let ok = bind(
        &model("  WT continuous", "  CL ~ WT linear(center = 0) +"),
        &pop,
    )
    .expect("an additive relation has no factor to keep positive");
    let theta = ok.covariate_model.as_ref().unwrap().relations[0].thetas[0].clone();
    assert!((theta.lower + 1e6).abs() < 1e-9, "{theta:?}");
}

/// …but the *degenerate* case is still an error, for its own reason: a constant
/// covariate makes `θ·(c₀ − c)` a constant shift, confounded with the typical
/// value. Dropping the whole gate under `+` would have lost this.
#[test]
fn a_constant_covariate_is_still_rejected_for_an_additive_relation() {
    let pop = population("WT", &[70.0, 70.0, 70.0]);
    // A symbolic centre. The literal-centre arm — which resolves at parse time
    // and so never meets the data on its own — is
    // `a_constant_covariate_is_rejected_for_a_literal_additive_centre`.
    let err = bind(
        &model("  WT continuous", "  CL ~ WT linear(center = median) +"),
        &pop,
    )
    .expect_err("a constant covariate leaves nothing to estimate");
    assert!(err.contains("constant"), "{err}");
    assert!(err.contains("confounded"), "{err}");
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

/// …and which *operator* each relation used (#1316 review).
///
/// The two spellings generate different models — one multiplies the parameter,
/// the other shifts it — and echoed identically, so a caller reading the table
/// back (ferx-r, `ferx allometry`) reported an additive relation as
/// multiplicative. Both arms are asserted, and against each other, so the field
/// cannot pass by being a constant.
#[test]
fn the_echoed_relation_table_keeps_the_operator() {
    let text = model(
        "  WT continuous",
        "  CL ~ WT linear(center = 70)\n  V ~ WT linear(center = 70) +",
    );
    let parsed = parse_full_model(&text).expect("model should parse");
    let names = parsed.model.theta_names.clone();
    let theta = parsed.model.default_params.theta.clone();
    let fixed = vec![false; theta.len()];
    let rels =
        crate::api::covariate_relation_estimates(&parsed.model, &names, &theta, None, &fixed);

    let cl = rels.iter().find(|r| r.parameter == "CL").expect("echoed");
    let v = rels.iter().find(|r| r.parameter == "V").expect("echoed");
    assert_eq!(cl.op, "*");
    assert_eq!(v.op, "+");
    // Everything else about the two rows is identical, which is exactly why the
    // operator has to be carried: without it these two are the same record.
    assert_eq!(cl.form, v.form);
    assert_eq!(cl.center, v.center);
    assert_ne!(cl.op, v.op);
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

/// A `hockey` breakpoint outside the observed range is rejected under `+` too.
///
/// The additive defaults drop the multiplicative centre-inside-range rule
/// because that rule keeps a *factor* positive, and `+` has no factor. The
/// hockey rule survives the drop for a different reason: outside the range one
/// arm holds no subjects, so its θ has an identically zero gradient and the
/// covariance step is singular. Dropping both together (#1316 review) accepted
/// `hockey(breakpoint = 0) +` on a realistic weight range.
///
/// A differential pair on the **form**, at the same out-of-range centre, so the
/// relaxation the additive path does keep is asserted next to the one it must
/// not: `linear(center = 0) +` is the uncentred slope and is fine.
#[test]
fn an_additive_hockey_breakpoint_outside_the_data_is_rejected() {
    let pop = population("WT", &[50.0, 60.0, 70.0, 80.0, 90.0]);
    let err = bind(
        &model("  WT continuous", "  CL ~ WT hockey(breakpoint = 0) +"),
        &pop,
    )
    .expect_err("one arm holds no subjects");
    assert!(err.contains("outside the observed range"), "{err}");
    assert!(
        err.contains("[50, 90]"),
        "the message states the range: {err}"
    );

    // Same operator, same centre, single slope: accepted.
    bind(
        &model("  WT continuous", "  CL ~ WT linear(center = 0) +"),
        &pop,
    )
    .expect("an uncentred additive slope needs no split, so no support rule applies");

    // Same form, same operator, inside the range: accepted. Without this the
    // check could pass by rejecting every additive hockey.
    bind(
        &model("  WT continuous", "  CL ~ WT hockey(breakpoint = 70) +"),
        &pop,
    )
    .expect("a breakpoint inside the range splits the data");

    // And the multiplicative twin still fails, from its own bounds rule.
    assert!(
        bind(
            &model("  WT continuous", "  CL ~ WT hockey(breakpoint = 0)"),
            &pop,
        )
        .is_err(),
        "the multiplicative rule is unchanged"
    );
}

/// A constant covariate is rejected for an additive relation with a **literal**
/// centre — the case the check could not previously see.
///
/// `linear(center = 70) +` has scale-free defaults, so it is fully built at
/// parse time, `needs_data()` is false, and `bind_covariate_stats` used to
/// return before summarising anything. The relation then reached the fit with
/// `θ·(70 − 70) ≡ 0`: a θ the optimiser cannot move, on data the multiplicative
/// twin refuses. Which is why the check now lives on the data path rather than
/// in the θ builder.
#[test]
fn a_constant_covariate_is_rejected_for_a_literal_additive_centre() {
    let pop = population("WT", &[70.0, 70.0, 70.0, 70.0]);
    let err = bind(
        &model("  WT continuous", "  CL ~ WT linear(center = 70) +"),
        &pop,
    )
    .expect_err("θ·(70 − 70) is identically zero");
    assert!(err.contains("has nothing to estimate"), "{err}");
    assert!(err.contains("constant"), "{err}");

    // An explicit θ does not rescue it: the defect is the design, not the bounds.
    assert!(
        bind(
            &model(
                "  WT continuous",
                "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)",
            ),
            &pop,
        )
        .is_err(),
        "an explicit bound cannot make a constant shift identifiable"
    );

    // The symbolic-centre arm, which the parse-time check did cover, still fails.
    assert!(
        bind(
            &model("  WT continuous", "  CL ~ WT linear(center = median) +"),
            &pop,
        )
        .is_err(),
        "a symbolic centre on constant data is unchanged"
    );

    // The straddle: the same relation on data that varies is accepted, so the
    // test cannot pass by rejecting every additive linear relation.
    bind(
        &model("  WT continuous", "  CL ~ WT linear(center = 70) +"),
        &population("WT", &[50.0, 70.0, 90.0]),
    )
    .expect("a covariate that varies has something to estimate");
}
