//! Tier-1 tests for the `[covariate_model]` block (#1111).
//!
//! Every test here asserts on the **generated text**: the block is sugar, so
//! what it is worth is exactly the expression it desugars to. A test that only
//! checked the parsed relation would pass while the emitted model said
//! something else.

use super::*;
use crate::parser::model_parser::parse_full_model;
use crate::types::CovariateModelSpec;

/// A one-compartment model with the given `[covariates]` / `[covariate_model]`
/// blocks spliced in, and `CL` carrying an η so the insertion-point partition
/// has something to place the factor against.
fn model_with(covariates: &str, covariate_model: &str) -> String {
    format!(
        "[parameters]\n\
        \x20 theta TVCL(4.0, 0.1, 100.0)\n\
        \x20 theta TVV(40.0, 1.0, 500.0)\n\
        \x20 omega ETA_CL ~ 0.09\n\
        \x20 sigma PROP_ERR ~ 0.02 (sd)\n\
        \n\
        [individual_parameters]\n\
        \x20 CL = TVCL * exp(ETA_CL)\n\
        \x20 V  = TVV\n\
        \n\
        [structural_model]\n\
        \x20 pk one_cpt_iv(cl=CL, v=V)\n\
        \n\
        [covariates]\n{covariates}\n\
        \n\
        [covariate_model]\n{covariate_model}\n\
        \n\
        [error_model]\n\
        \x20 DV ~ proportional(PROP_ERR)\n"
    )
}

fn spec(covariates: &str, covariate_model: &str) -> CovariateModelSpec {
    parse_full_model(&model_with(covariates, covariate_model))
        .expect("model should parse")
        .model
        .covariate_model
        .expect("[covariate_model] should be recorded on the model")
}

fn err(covariates: &str, covariate_model: &str) -> String {
    parse_err(&model_with(covariates, covariate_model))
}

/// `parse_full_model`'s error, with the `Ok` side dropped — `ParsedModel` holds
/// closures and so is not `Debug`, which `expect_err` would require.
fn parse_err(text: &str) -> String {
    parse_full_model(text)
        .map(|_| ())
        .expect_err("model should be rejected")
}

/// The desugared `CL = ...` line.
fn cl_line(spec: &CovariateModelSpec) -> String {
    spec.desugared_individual_parameters
        .iter()
        .find(|l| l.trim_start().starts_with("CL "))
        .expect("CL is assigned")
        .trim()
        .to_string()
}

// ── One test per form: the generated factor ────────────────────────────────

#[test]
fn power_generates_the_classical_allometric_factor() {
    let s = spec("  WT continuous", "  CL ~ WT power(center = 70)");
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * exp(ETA_CL)"
    );
    assert_eq!(
        s.generated_thetas,
        vec!["  theta THETA_CL_WT(0.001, -100, 1000000)"]
    );
}

#[test]
fn exponential_generates_an_exp_of_the_centred_covariate() {
    let s = spec("  WT continuous", "  CL ~ WT exponential(center = 70)");
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) exp(THETA_CL_WT * (WT - 70)) else 1.0) * exp(ETA_CL)"
    );
}

#[test]
fn linear_needs_data_and_stays_unresolved_without_it() {
    // PsN's linear bounds are `1/(median − max) .. 1/(median − min)`, which only
    // a dataset can supply — so the relation parses, records itself, and is
    // deliberately left undesugared until the statistics are bound.
    let s = spec("  WT continuous", "  CL ~ WT linear(center = 70)");
    assert_eq!(cl_line(&s), "CL = TVCL * exp(ETA_CL)");
    assert!(s.generated_thetas.is_empty());
    assert_eq!(s.unresolved().len(), 1);
}

#[test]
fn hockey_generates_two_slopes_around_the_breakpoint() {
    let s = spec(
        "  WT continuous",
        "  CL ~ WT hockey(breakpoint = 70) => T_LO(0.01, -1, 1), T_HI(0.02, -1, 1)",
    );
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (if (WT <= 70) 1 + T_LO * (WT - 70) else 1 + T_HI * (WT - 70)) \
         else 1.0) * exp(ETA_CL)"
    );
}

#[test]
fn categorical_contrasts_every_non_reference_level() {
    let s = spec(
        "  SEX categorical(levels = [0, 1, 2])",
        "  CL ~ SEX categorical(ref = 0)",
    );
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(SEX)) (if (SEX == 1) 1 + THETA_CL_SEX_1 else \
         if (SEX == 2) 1 + THETA_CL_SEX_2 else 1) else 1.0) * exp(ETA_CL)"
    );
    // PsN's categorical θ is null at zero, so the bounds straddle it.
    assert_eq!(
        s.generated_thetas,
        vec![
            "  theta THETA_CL_SEX_1(-0.001, -1, 5)",
            "  theta THETA_CL_SEX_2(-0.001, -1, 5)",
        ]
    );
}

#[test]
fn none_declares_no_theta_and_leaves_the_expression_alone() {
    // A search writes `none` to record "tested, rejected"; the generated model
    // must round-trip through the parser unchanged.
    let s = spec("  WT continuous", "  CL ~ WT none");
    assert_eq!(cl_line(&s), "CL = TVCL * exp(ETA_CL)");
    assert!(s.generated_thetas.is_empty());
    assert!(s.unresolved().is_empty());
}

#[test]
fn expr_is_emitted_verbatim() {
    let s = spec("  WT continuous", "  CL ~ WT expr(\"(WT/70)^0.75\")");
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) ((WT/70)^0.75) else 1.0) * exp(ETA_CL)"
    );
    assert!(s.generated_thetas.is_empty());
}

#[test]
fn linear_relative_is_dimensionless_in_theta() {
    let s = spec(
        "  WT continuous",
        "  CL ~ WT linear_relative(center = 70) => T(0.1, -1, 1)",
    );
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (1 + T * (WT / 70 - 1)) else 1.0) * exp(ETA_CL)"
    );
}

#[test]
fn fix_pins_the_theta_and_declares_it_fix() {
    let s = spec(
        "  WT continuous",
        "  CL ~ WT power(center = 70, fix = 0.75)",
    );
    assert_eq!(
        s.generated_thetas,
        vec!["  theta THETA_CL_WT(0.75, -100, 1000000) FIX"]
    );
}

// ── The missing-value guard ────────────────────────────────────────────────

#[test]
fn a_missing_covariate_contributes_exactly_one() {
    // The guard is `COV == COV`, which is false for NaN. Without it the factor
    // would be `(NaN/70)^θ`, and a division by a missing value underflows to
    // `0.0` here rather than blowing up — a silent zero, not a loud failure.
    let parsed = parse_full_model(&model_with(
        "  WT continuous",
        "  CL ~ WT power(center = 70)",
    ))
    .expect("model should parse");
    let theta_idx = parsed
        .model
        .theta_names
        .iter()
        .position(|n| n == "THETA_CL_WT")
        .expect("the generated θ is a real θ");
    let mut theta = vec![4.0; parsed.model.theta_names.len()];
    theta[theta_idx] = 0.75;
    let eta = vec![0.0; parsed.model.eta_names.len()];

    let mut covariates = std::collections::HashMap::new();
    covariates.insert("WT".to_string(), f64::NAN);
    let missing = (parsed.model.pk_param_fn)(&theta, &eta, &covariates, 0.0);

    covariates.insert("WT".to_string(), 70.0);
    let at_centre = (parsed.model.pk_param_fn)(&theta, &eta, &covariates, 0.0);

    // At the centring weight the factor is 1, so a missing weight must give the
    // same CL — and, crucially, not 0.
    assert_eq!(missing.values[0], at_centre.values[0]);
    assert!(
        missing.values[0] > 0.0,
        "missing covariate must not zero the parameter"
    );
}

// ── The insertion point ────────────────────────────────────────────────────

#[test]
fn the_factor_lands_before_the_first_eta_bearing_factor() {
    // Not at the end of the RHS: a factor after `exp(ETA)` is numerically
    // identical but takes the typical value out of the shape the SAEM
    // mu-reference detector reads (#619).
    let s = spec("  WT continuous", "  CL ~ WT power(center = 70)");
    let line = cl_line(&s);
    let factor = line.find("(if (present(WT))").expect("factor is present");
    let eta = line.find("exp(ETA_CL)").expect("η factor is present");
    assert!(
        factor < eta,
        "covariate factor must precede exp(ETA_CL): {line}"
    );
}

#[test]
fn a_parameter_with_no_eta_takes_the_factor_at_the_end() {
    let s = spec("  WT continuous", "  V ~ WT power(center = 70)");
    let v = s
        .desugared_individual_parameters
        .iter()
        .find(|l| l.trim_start().starts_with("V "))
        .expect("V is assigned")
        .trim()
        .to_string();
    assert_eq!(
        v,
        "V  = TVV * (if (present(WT)) (WT / 70)^THETA_V_WT else 1.0)"
    );
}

#[test]
fn several_relations_on_one_parameter_all_land_in_the_non_eta_group() {
    let s = spec(
        "  WT continuous\n  CRCL continuous",
        "  CL ~ WT power(center = 70)\n  CL ~ CRCL power(center = 100)",
    );
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * \
         (if (present(CRCL)) (CRCL / 100)^THETA_CL_CRCL else 1.0) * exp(ETA_CL)"
    );
}

#[test]
fn a_non_product_right_hand_side_is_a_hard_error() {
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)")
        .replace("CL = TVCL * exp(ETA_CL)", "CL = TVCL + exp(ETA_CL)");
    let e = parse_err(&text);
    assert!(e.contains("not a top-level product"), "{e}");
    assert!(
        e.contains("COV_CL"),
        "the error must name the explicit handle: {e}"
    );
}

#[test]
fn a_sum_that_also_carries_a_product_is_a_hard_error() {
    // `split_top_level(rhs, '*')` returns two parts here, so a guard that only
    // looked at the whole RHS when it had *no* top-level `*` skipped this shape
    // entirely — and the factor was multiplied into the first addend alone,
    // leaving `TVV` uncovered. Every part has to be a plain factor.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)")
        .replace("CL = TVCL * exp(ETA_CL)", "CL = TVCL * exp(ETA_CL) + TVV");
    let e = parse_err(&text);
    assert!(e.contains("not a top-level product"), "{e}");
    assert!(e.contains("COV_CL"), "{e}");
}

#[test]
fn a_conditional_factor_in_a_product_is_a_hard_error() {
    // `if (...) a else b * TVCL` parses as one product whose first part is a
    // conditional: appending the covariate factor would attach it to the `else`
    // branch, not to the parameter.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)").replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = if (TVV > 1) 1.2 else 1.0 * TVCL * exp(ETA_CL)",
    );
    let e = parse_err(&text);
    assert!(e.contains("not a top-level product"), "{e}");
}

#[test]
fn an_eta_reached_through_an_intermediate_still_takes_the_factor_first() {
    // `first_random` classifies a factor by the η/κ *names* it mentions, so
    // `CLI` would look η-free and the factor would land after it — the #619
    // placement, silently. The transitive closure over block-local assignments
    // is what keeps it in the non-η group.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)").replace(
        "CL = TVCL * exp(ETA_CL)",
        "CLI = exp(ETA_CL)\n  CL = TVCL * CLI",
    );
    let s = parse_full_model(&text)
        .expect("model should parse")
        .model
        .covariate_model
        .expect("recorded");
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * CLI"
    );
}

#[test]
fn a_log_transformed_right_hand_side_is_a_hard_error() {
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)")
        .replace("CL = TVCL * exp(ETA_CL)", "CL = exp(TVCL + ETA_CL)");
    let e = parse_err(&text);
    assert!(e.contains("not a top-level product"), "{e}");
}

// ── Validation ─────────────────────────────────────────────────────────────

#[test]
fn an_undeclared_covariate_is_an_error() {
    // Unlike the lenient classical path, which warns and reads the covariate
    // anyway: the declaration is what states the kind and the levels the
    // generated θ vector depends on.
    let e = err("  WT continuous", "  CL ~ AGE power(center = 40)");
    assert!(e.contains("not declared in [covariates]"), "{e}");
}

#[test]
fn a_relation_on_an_unknown_parameter_is_an_error() {
    let e = err("  WT continuous", "  KA ~ WT power(center = 70)");
    assert!(
        e.contains("not a top-level [individual_parameters] name"),
        "{e}"
    );
}

#[test]
fn a_duplicate_parameter_covariate_pair_is_an_error() {
    let e = err(
        "  WT continuous",
        "  CL ~ WT power(center = 70)\n  CL ~ WT linear(center = 70)",
    );
    assert!(e.contains("declared more than once"), "{e}");
}

#[test]
fn an_explicit_theta_name_that_collides_with_parameters_is_an_error() {
    let e = err(
        "  WT continuous",
        "  CL ~ WT power(center = 70) => TVCL(0.5, -1, 1)",
    );
    assert!(e.contains("already declared in [parameters]"), "{e}");
}

#[test]
fn an_auto_theta_name_defers_to_an_existing_declaration() {
    // The modeller who declared the θ stated the init and bounds they want;
    // re-emitting it would declare the same θ twice.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)").replace(
        "  theta TVV(40.0, 1.0, 500.0)",
        "  theta TVV(40.0, 1.0, 500.0)\n  theta THETA_CL_WT(0.75, 0.1, 1.5)",
    );
    let parsed = parse_full_model(&text).expect("model should parse");
    assert_eq!(
        parsed
            .model
            .theta_names
            .iter()
            .filter(|n| *n == "THETA_CL_WT")
            .count(),
        1
    );
    let s = parsed.model.covariate_model.expect("recorded");
    assert!(s.generated_thetas.is_empty());
}

#[test]
fn a_categorical_form_on_a_continuous_covariate_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT categorical(ref = 0)");
    assert!(e.contains("declares continuous"), "{e}");
}

#[test]
fn a_continuous_form_on_a_categorical_covariate_is_an_error() {
    let e = err(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX power(center = 1)",
    );
    assert!(e.contains("categorical"), "{e}");
}

#[test]
fn a_categorical_relation_without_levels_is_an_error() {
    let e = err("  SEX categorical", "  CL ~ SEX categorical(ref = 0)");
    assert!(e.contains("needs its levels"), "{e}");
    assert!(
        e.contains("levels = auto"),
        "the error must offer the data-derived opt-in: {e}"
    );
}

#[test]
fn a_reference_level_outside_the_declared_levels_is_an_error() {
    let e = err(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical(ref = 2)",
    );
    assert!(e.contains("is not one of the declared levels"), "{e}");
}

#[test]
fn an_unknown_form_offers_a_suggestion() {
    let e = err("  WT continuous", "  CL ~ WT powr(center = 70)");
    assert!(e.contains("did you mean `power`"), "{e}");
}

#[test]
fn an_unknown_keyword_argument_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT power(centre = 70)");
    assert!(e.contains("unknown argument"), "{e}");
}

#[test]
fn the_wrong_centring_keyword_for_the_form_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT power(breakpoint = 70)");
    assert!(e.contains("centres on `center`"), "{e}");
}

#[test]
fn a_theta_clause_of_the_wrong_arity_is_an_error() {
    let e = err(
        "  WT continuous",
        "  CL ~ WT hockey(breakpoint = 70) => T_LO(0.01, -1, 1)",
    );
    assert!(e.contains("generates 2 θ"), "{e}");
}

#[test]
fn a_misspelled_block_name_is_still_rejected() {
    // The registry is closed-world; `[covariate_model]` joining it must not
    // open a door for its neighbours.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)")
        .replace("[covariate_model]", "[covariate_models]");
    let e = parse_err(&text);
    assert!(e.contains("covariate_models"), "{e}");
}

// ── The `[covariates]` levels clause ───────────────────────────────────────

#[test]
fn declared_levels_are_read_off_the_covariates_block() {
    let parsed = parse_full_model(&model_with(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical(ref = 0)",
    ))
    .expect("model should parse");
    let decls = parsed.covariate_decls.expect("declared");
    assert_eq!(
        decls[0].levels,
        Some(crate::types::CovariateLevels::Declared(vec![0.0, 1.0]))
    );
}

#[test]
fn auto_levels_leave_the_relation_unresolved() {
    let s = spec(
        "  SEX categorical(levels = auto)",
        "  CL ~ SEX categorical(ref = 0)",
    );
    assert_eq!(s.unresolved().len(), 1);
    assert!(s.generated_thetas.is_empty());
}

#[test]
fn a_single_level_categorical_says_it_has_no_contrast() {
    // One level spends zero θ, which `needs_data()` would otherwise read as
    // *unresolved* — sending the user to `--data`, which can never help.
    let e = err(
        "  SEX categorical(levels = [1])",
        "  CL ~ SEX categorical(ref = 1)",
    );
    assert!(e.contains("has nothing to estimate"), "{e}");
    assert!(e.contains("`SEX` has 1 level (1)"), "{e}");
    assert!(
        !e.contains("needs data-derived statistics"),
        "must not be reported as unresolved: {e}"
    );
}

#[test]
fn fix_on_a_form_that_declares_no_theta_is_an_error() {
    // `none` spends no θ, so `apply_fix` would run over an empty list: the
    // argument would be accepted and then do nothing.
    let e = err("  WT continuous", "  CL ~ WT none(fix = 0.5)");
    assert!(e.contains("takes no `fix`"), "{e}");
}

#[test]
fn levels_on_a_continuous_covariate_is_an_error() {
    let e = err(
        "  WT continuous(levels = [0, 1])",
        "  CL ~ WT power(center = 70)",
    );
    assert!(
        e.contains("only meaningful for a categorical covariate"),
        "{e}"
    );
}

// ── Text helpers ───────────────────────────────────────────────────────────

#[test]
fn top_level_split_ignores_operators_inside_parentheses() {
    let parts = split_top_level("A * f(B * C) * D", '*');
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[1].trim(), "f(B * C)");
}

#[test]
fn an_exponent_sign_is_not_a_top_level_sum() {
    assert!(!has_top_level_additive("1.5e-3 * TVCL"));
    assert!(has_top_level_additive("TVCL + 1"));
    assert!(!has_top_level_additive("-TVCL"));
}

#[test]
fn a_commented_assignment_desugars_on_its_expression_alone() {
    // The block extractor strips comments before any block is read, so the RHS
    // split here never carries one. Pinned because that split is
    // parenthesis-aware, not comment-aware: were comments ever preserved, a `#`
    // would silently become part of the last factor.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)").replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = TVCL * exp(ETA_CL)   # allometric on purpose",
    );
    let s = parse_full_model(&text)
        .expect("model should parse")
        .model
        .covariate_model
        .expect("recorded");
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * exp(ETA_CL)"
    );
}

#[test]
fn a_parameter_assigned_only_inside_a_conditional_is_not_addressable() {
    // There is no single right-hand side to multiply into, so the relation must
    // be rejected rather than silently landing on one branch.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)").replace(
        "CL = TVCL * exp(ETA_CL)",
        "if (WT > 70) {\n    CL = TVCL * exp(ETA_CL)\n  } else {\n    CL = TVCL\n  }",
    );
    let e = parse_err(&text);
    assert!(
        e.contains("not a top-level [individual_parameters] name"),
        "{e}"
    );
}

#[test]
fn a_kappa_bearing_factor_counts_as_a_random_effect_factor() {
    // IOV κ is a random effect too: the covariate factor belongs before it, for
    // the same mu-referencing reason as η.
    let text = model_with("  WT continuous", "  CL ~ WT power(center = 70)")
        .replace(
            "  omega ETA_CL ~ 0.09",
            "  omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.05",
        )
        .replace(
            "CL = TVCL * exp(ETA_CL)",
            "CL = TVCL * exp(ETA_CL) * exp(KAPPA_CL)",
        );
    let s = parse_full_model(&text)
        .expect("model should parse")
        .model
        .covariate_model
        .expect("recorded");
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * exp(ETA_CL) * exp(KAPPA_CL)"
    );
}
