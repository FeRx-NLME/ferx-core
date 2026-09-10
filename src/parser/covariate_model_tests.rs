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

/// [`model_with`] plus a `[fit_options] method`, for the warnings that are
/// scoped to the methods that actually read `mu_refs`.
fn model_with_method(covariates: &str, covariate_model: &str, method: &str) -> String {
    format!(
        "{}\n[fit_options]\n\x20 method = {method}\n",
        model_with(covariates, covariate_model)
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

// ── Malformed lines: one assert per diagnostic ─────────────────────────────
//
// These are the messages a user actually meets when a hand-written or
// search-generated block is wrong, so each is pinned on the phrase that makes
// it actionable rather than on the whole string.

#[test]
fn a_line_without_a_tilde_names_the_expected_shape() {
    let e = err("  WT continuous", "  CL WT power(center = 70)");
    assert!(e.contains("expected `PARAM ~ COV form(...)`"), "{e}");
}

#[test]
fn a_line_with_no_parameter_before_the_tilde_is_an_error() {
    let e = err("  WT continuous", "  ~ WT power(center = 70)");
    assert!(e.contains("missing parameter name"), "{e}");
}

#[test]
fn a_relation_that_states_no_form_lists_the_forms() {
    let e = err("  WT continuous", "  CL ~ WT");
    assert!(e.contains("states no form"), "{e}");
    assert!(e.contains("linear_relative"), "{e}");
}

#[test]
fn two_relations_generating_the_same_theta_name_is_an_error() {
    // Each relation owns its θ; sharing one would couple two lines and defeat
    // the point of a line-oriented block.
    let e = err(
        "  WT continuous\n  CRCL continuous",
        "  CL ~ WT power(center = 70) => SHARED(0.75, -10, 10)\n\
         \x20 CL ~ CRCL power(center = 100) => SHARED(0.75, -10, 10)",
    );
    assert!(e.contains("both generate the θ `SHARED`"), "{e}");
}

#[test]
fn an_unbalanced_form_parenthesis_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT power(center = 70");
    assert!(e.contains("unbalanced parentheses"), "{e}");
}

#[test]
fn an_unquoted_or_empty_expr_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT expr((WT/70)^0.75)");
    assert!(e.contains("takes a quoted expression"), "{e}");

    let e = err("  WT continuous", "  CL ~ WT expr(\"  \")");
    assert!(e.contains("states no expression"), "{e}");
}

#[test]
fn a_keyword_argument_without_a_value_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT power(median)");
    assert!(e.contains("takes `key = value` arguments"), "{e}");
}

#[test]
fn a_centring_value_that_is_neither_a_number_nor_a_statistic_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT power(center = middling)");
    assert!(e.contains("neither a number nor one of"), "{e}");
}

#[test]
fn a_repeated_keyword_argument_is_an_error() {
    let e = err(
        "  WT continuous",
        "  CL ~ WT power(center = 70, center = 80)",
    );
    assert!(e.contains("is given more than once"), "{e}");
}

#[test]
fn a_symbolic_fix_value_is_an_error() {
    // `fix` pins θ at a number; a data-derived statistic is not one.
    let e = err(
        "  WT continuous",
        "  CL ~ WT power(center = 70, fix = median)",
    );
    assert!(e.contains("`fix` takes a number"), "{e}");
}

#[test]
fn min_and_max_accept_their_long_spellings() {
    // `minimum`/`maximum` are the spellings PsN users reach for; both resolve
    // to the same statistic as the short form.
    let s = spec("  WT continuous", "  CL ~ WT linear(center = minimum)");
    assert_eq!(s.relations[0].center, Some(CovariateStat::Min));
    let s = spec("  WT continuous", "  CL ~ WT linear(center = maximum)");
    assert_eq!(s.relations[0].center, Some(CovariateStat::Max));
}

#[test]
fn a_non_finite_literal_centre_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT power(center = inf)");
    assert!(e.contains("not a finite number"), "{e}");
}

#[test]
fn a_malformed_theta_clause_is_an_error() {
    let e = err(
        "  WT continuous",
        "  CL ~ WT power(center = 70) => THETA_WT",
    );
    assert!(e.contains("expected `=> NAME(init, lower, upper)`"), "{e}");

    let e = err(
        "  WT continuous",
        "  CL ~ WT power(center = 70) => T(0.75, low, 10)",
    );
    assert!(e.contains("`low` is not a number"), "{e}");

    let e = err(
        "  WT continuous",
        "  CL ~ WT power(center = 70) => T(0.75, -10)",
    );
    assert!(e.contains("needs exactly (init, lower, upper)"), "{e}");

    let e = err("  WT continuous", "  CL ~ WT power(center = 70) => ");
    assert!(e.contains("empty `=>` clause"), "{e}");
}

// ── The additive operator (#1313) ──────────────────────────────────────────

/// `[CL, V]` as the compiled model computes them, for a given θ override and
/// covariate map — the first two slots of the fixed PK-parameter layout.
fn pk_values(model_text: &str, thetas: &[(&str, f64)], covariates: &[(&str, f64)]) -> Vec<f64> {
    let parsed = parse_full_model(model_text).expect("model should parse");
    let mut theta = vec![1.0; parsed.model.theta_names.len()];
    for (name, value) in thetas {
        let i = parsed
            .model
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("θ `{name}` is declared: {:?}", parsed.model.theta_names));
        theta[i] = *value;
    }
    let eta = vec![0.0; parsed.model.eta_names.len()];
    let map: std::collections::HashMap<String, f64> = covariates
        .iter()
        .map(|(n, v)| ((*n).to_string(), *v))
        .collect();
    (parsed.model.pk_param_fn)(&theta, &eta, &map, 0.0).values[..2].to_vec()
}

#[test]
fn the_additive_operator_drops_the_leading_one_from_every_form() {
    // The choice #1313 had to settle. Pharmpy reuses the *multiplicative*
    // template under `+`, so its additive linear effect adds
    // `1 + θ·(WT − 70)` and a subject at the centring weight has `1` added to
    // their clearance. ferx drops that `1`, so θ = 0 and COV = centre are both
    // "no effect". Every expected string below carries no leading `1`, and a
    // switch to the Pharmpy-verbatim convention reddens all six.
    let cases: [(&str, &str); 6] = [
        (
            "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)",
            "(THETA_CL_WT * (WT - 70))",
        ),
        (
            "  CL ~ WT linear_relative(center = 70) + => THETA_CL_WT(0.02, -1, 1)",
            "(THETA_CL_WT * (WT / 70 - 1))",
        ),
        (
            "  CL ~ WT exponential(center = 70) +",
            "(exp(THETA_CL_WT * (WT - 70)) - 1)",
        ),
        (
            "  CL ~ WT power(center = 70) +",
            "((WT / 70)^THETA_CL_WT - 1)",
        ),
        (
            "  CL ~ WT hockey(breakpoint = 70) + => T_LO(0.01, -1, 1), T_HI(0.02, -1, 1)",
            "(if (WT <= 70) T_LO * (WT - 70) else T_HI * (WT - 70))",
        ),
        ("  CL ~ WT expr(\"0.5 * (WT - 70)\") +", "(0.5 * (WT - 70))"),
    ];
    for (line, term) in cases {
        let s = spec("  WT continuous", line);
        assert_eq!(
            cl_line(&s),
            format!("CL = TVCL * exp(ETA_CL) + (if (present(WT)) {term} else 0.0)"),
            "additive desugar of `{line}`"
        );
        assert!(
            !term.contains("1 +"),
            "the additive template must not carry Pharmpy's leading `1`: {term}"
        );
    }
}

#[test]
fn an_additive_categorical_contributes_zero_at_the_reference_level() {
    let s = spec(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical(ref = 0) +",
    );
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * exp(ETA_CL) + \
         (if (present(SEX)) (if (SEX == 1) THETA_CL_SEX_1 else 0) else 0.0)"
    );
    // The multiplicative twin on the same form and levels keeps its `1`s: the
    // two templates differ, and one is not the other with a different join.
    let mul = spec(
        "  SEX categorical(levels = [0, 1])",
        "  CL ~ SEX categorical(ref = 0)",
    );
    assert_eq!(
        cl_line(&mul),
        "CL = TVCL * (if (present(SEX)) (if (SEX == 1) 1 + THETA_CL_SEX_1 else 1) else 1.0) \
         * exp(ETA_CL)"
    );
}

#[test]
fn theta_at_zero_and_a_covariate_at_its_centre_are_both_no_effect() {
    // The property the null-at-zero choice buys, asserted on numbers rather
    // than on text: under Pharmpy's convention each of these would be
    // `TVCL + 1`, not `TVCL`.
    let text = model_with(
        "  WT continuous",
        "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)",
    );
    let tvcl = 4.0;

    let at_centre = pk_values(
        &text,
        &[("TVCL", tvcl), ("THETA_CL_WT", 0.02)],
        &[("WT", 70.0)],
    );
    assert_eq!(at_centre[0], tvcl, "a covariate at its centre adds nothing");

    let theta_zero = pk_values(
        &text,
        &[("TVCL", tvcl), ("THETA_CL_WT", 0.0)],
        &[("WT", 120.0)],
    );
    assert_eq!(theta_zero[0], tvcl, "θ = 0 adds nothing");

    // …and the effect is live off-centre, so the two assertions above are not
    // passing because the term was dropped altogether.
    let off_centre = pk_values(
        &text,
        &[("TVCL", tvcl), ("THETA_CL_WT", 0.02)],
        &[("WT", 120.0)],
    );
    assert!(
        (off_centre[0] - (tvcl + 0.02 * 50.0)).abs() < 1e-12,
        "expected TVCL + θ*(120 - 70), got {}",
        off_centre[0]
    );
}

#[test]
fn the_missing_covariate_guard_is_per_operator_on_one_model() {
    // Both relations read the same column on the same model, so a regression
    // that shares one neutral element across the operators cannot pass by
    // getting one of them right.
    let text = model_with(
        "  WT continuous",
        "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)\n  V ~ WT power(center = 70)",
    );
    let s = parse_full_model(&text)
        .expect("model should parse")
        .model
        .covariate_model
        .expect("recorded");
    assert!(cl_line(&s).ends_with("else 0.0)"), "{}", cl_line(&s));
    let v = s
        .desugared_individual_parameters
        .iter()
        .find(|l| l.trim_start().starts_with("V "))
        .expect("V is assigned")
        .clone();
    assert!(v.trim().ends_with("else 1.0)"), "{v}");

    let thetas = [
        ("TVCL", 4.0),
        ("TVV", 40.0),
        ("THETA_CL_WT", 0.02),
        ("THETA_V_WT", 0.75),
    ];
    let missing = pk_values(&text, &thetas, &[("WT", f64::NAN)]);
    // A missing covariate must leave BOTH parameters exactly as they were: a
    // shared `1.0` guard would make CL 5.0 instead of 4.0, and a shared `0.0`
    // guard would zero V.
    assert_eq!(missing[0], 4.0, "additive: a missing covariate must add 0");
    assert_eq!(
        missing[1], 40.0,
        "multiplicative: a missing covariate must multiply by 1"
    );
}

#[test]
fn a_multiplicative_and_an_additive_relation_on_one_parameter_both_land() {
    let s = spec(
        "  WT continuous\n  AGE continuous",
        "  CL ~ WT power(center = 70)\n  CL ~ AGE linear(center = 40) + => THETA_CL_AGE(0.1, -1, 1)",
    );
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * exp(ETA_CL) \
         + (if (present(AGE)) (THETA_CL_AGE * (AGE - 40)) else 0.0)"
    );
}

#[test]
fn a_multiplicative_relation_declared_after_an_additive_one_still_joins_the_product() {
    // The `is_simple_factor` case that motivated `non_product_error`: if the
    // additive term were appended first, the factor would be multiplied into
    // one addend of the sum — or the relation would be rejected outright. The
    // product is rebuilt from the *original* right-hand side, so declaration
    // order cannot reach either.
    let s = spec(
        "  WT continuous\n  AGE continuous",
        "  CL ~ AGE linear(center = 40) + => THETA_CL_AGE(0.1, -1, 1)\n  CL ~ WT power(center = 70)",
    );
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * (if (present(WT)) (WT / 70)^THETA_CL_WT else 1.0) * exp(ETA_CL) \
         + (if (present(AGE)) (THETA_CL_AGE * (AGE - 40)) else 0.0)"
    );

    // And the arithmetic is what the text says: the factor scales only the
    // typical value, the term is added to the whole thing.
    let text = model_with(
        "  WT continuous\n  AGE continuous",
        "  CL ~ AGE linear(center = 40) + => THETA_CL_AGE(0.1, -1, 1)\n  CL ~ WT power(center = 70)",
    );
    let v = pk_values(
        &text,
        &[("TVCL", 4.0), ("THETA_CL_WT", 1.0), ("THETA_CL_AGE", 0.1)],
        &[("WT", 140.0), ("AGE", 60.0)],
    );
    assert!(
        (v[0] - (4.0 * 2.0 + 0.1 * 20.0)).abs() < 1e-12,
        "expected TVCL*(140/70)^1 + 0.1*(60-40) = 10, got {}",
        v[0]
    );
}

#[test]
fn an_additive_relation_needs_no_top_level_product() {
    // A shape `non_product_error` rejects for a multiplicative relation is
    // legal for an additive one: there is nothing to multiply into.
    let text = model_with(
        "  WT continuous",
        "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)",
    )
    .replace("CL = TVCL * exp(ETA_CL)", "CL = exp(TVCL + ETA_CL)");
    let s = parse_full_model(&text)
        .expect("an additive relation on a log-transformed RHS must parse")
        .model
        .covariate_model
        .expect("recorded");
    assert_eq!(
        cl_line(&s),
        "CL = exp(TVCL + ETA_CL) + (if (present(WT)) (THETA_CL_WT * (WT - 70)) else 0.0)"
    );
}

#[test]
fn a_conditional_right_hand_side_is_parenthesised_before_a_term_is_appended() {
    // `if (c) a else b + t` binds the term inside the `else` arm. The taken
    // branch here is the `then` one, so an unparenthesised append would drop
    // the covariate effect entirely — silently, and only for the subjects on
    // that branch.
    let text = model_with(
        "  WT continuous",
        "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)",
    )
    .replace(
        "CL = TVCL * exp(ETA_CL)",
        "CL = if (TVV > 1) 1.2 else TVCL * exp(ETA_CL)",
    );
    let s = parse_full_model(&text)
        .expect("model should parse")
        .model
        .covariate_model
        .expect("recorded");
    assert_eq!(
        cl_line(&s),
        "CL = (if (TVV > 1) 1.2 else TVCL * exp(ETA_CL)) \
         + (if (present(WT)) (THETA_CL_WT * (WT - 70)) else 0.0)"
    );

    // TVV is 40, so the `then` branch is taken and the term must still be
    // added: 1.2 + 0.02*(120 - 70) = 2.2, not 1.2.
    let v = pk_values(
        &text,
        &[("TVV", 40.0), ("THETA_CL_WT", 0.02)],
        &[("WT", 120.0)],
    );
    assert!(
        (v[0] - 2.2).abs() < 1e-12,
        "the term must be added to the conditional, not inside its else arm: {}",
        v[0]
    );
}

#[test]
fn mu_referencing_is_off_for_an_additive_relation_and_on_for_its_multiplicative_twin() {
    // The #619 classification, pinned rather than incidental. The two models
    // differ in exactly one character — the trailing `+` — so they sit on
    // opposite sides of `detect_mu_refs`' predicate, and a change that made the
    // detector match a sum (or stopped it matching a product) reddens this.
    let mul = parse_full_model(&model_with_method(
        "  WT continuous",
        "  CL ~ WT linear(center = 70) => THETA_CL_WT(0.02, -1, 1)",
        "saem",
    ))
    .expect("model should parse");
    assert!(
        mul.model.mu_refs.contains_key("ETA_CL"),
        "a multiplicative relation keeps the mu-ref anchor: {:?}",
        mul.model.mu_refs
    );
    assert!(
        !mul.model
            .parse_warnings
            .iter()
            .any(|w| w.contains("additive (`+`) [covariate_model]")),
        "no additive warning on a multiplicative model: {:?}",
        mul.model.parse_warnings
    );

    let add = parse_full_model(&model_with_method(
        "  WT continuous",
        "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)",
        "saem",
    ))
    .expect("model should parse");
    assert!(
        !add.model.mu_refs.contains_key("ETA_CL"),
        "an additive term makes the typical value a sum, which is not a mu-ref shape: {:?}",
        add.model.mu_refs
    );
    let warning = add
        .model
        .parse_warnings
        .iter()
        .find(|w| w.contains("additive (`+`) [covariate_model]"))
        .unwrap_or_else(|| panic!("expected a warning: {:?}", add.model.parse_warnings));
    assert!(warning.contains("CL"), "{warning}");
    assert!(warning.contains("numerical M-step"), "{warning}");
}

#[test]
fn a_parameter_with_no_eta_gets_no_mu_reference_warning() {
    // The warning names a performance cliff that only exists for an η the
    // M-step would otherwise shift; `V = TVV` carries none.
    // Under `saem`, so the warning is suppressed by the missing η and not by
    // the method gate — with `focei` here the assertion could not fail.
    let parsed = parse_full_model(&model_with_method(
        "  WT continuous",
        "  V ~ WT linear(center = 70) + => THETA_V_WT(0.02, -1, 1)",
        "saem",
    ))
    .expect("model should parse");
    assert!(
        !parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("additive (`+`) [covariate_model]")),
        "{:?}",
        parsed.model.parse_warnings
    );
}

/// The mu-referencing warning describes the SAEM/IMP M-step, so it must not
/// fire for a method that never reads `mu_refs` (#1316 review).
///
/// Its own last sentence says FOCE/FOCEI are unaffected, yet it was pushed
/// unconditionally at parse time — a paragraph on every FOCEI fit about a path
/// that run does not take. A differential pair on the *method*, with the
/// additive relation held fixed, so it straddles the gate that was added.
#[test]
fn the_mu_reference_warning_is_scoped_to_the_methods_that_read_mu_refs() {
    let relation = "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.02, -1, 1)";
    let fired = |method: &str| {
        let parsed = parse_full_model(&model_with_method("  WT continuous", relation, method))
            .expect("model should parse");
        // The classification itself is method-independent — only the warning is
        // scoped — so a gate that silently disabled the detection would not
        // pass here either.
        assert!(
            !parsed.model.mu_refs.contains_key("ETA_CL"),
            "the sum is not a mu-ref shape under {method}: {:?}",
            parsed.model.mu_refs
        );
        parsed
            .model
            .parse_warnings
            .iter()
            .find(|w| w.contains("additive (`+`) [covariate_model]"))
            .cloned()
    };

    let saem = fired("saem").expect("SAEM uses the mu-ref M-step, so the cliff is real");
    assert!(
        saem.contains("SAEM"),
        "the warning names the method: {saem}"
    );
    for method in ["imp", "impmap", "bayes"] {
        assert!(
            fired(method).is_some(),
            "{method} reads mu_refs and must be warned"
        );
    }
    for method in ["focei", "foce", "laplace"] {
        assert!(
            fired(method).is_none(),
            "{method} never reads mu_refs, so the warning is noise"
        );
    }
}

#[test]
fn the_operator_token_is_optional_and_explicit_star_is_the_default() {
    let implicit = spec("  WT continuous", "  CL ~ WT power(center = 70)");
    let explicit = spec("  WT continuous", "  CL ~ WT power(center = 70) *");
    assert_eq!(cl_line(&implicit), cl_line(&explicit));
    assert_eq!(implicit.relations[0].op, CovariateOp::Multiply);
    assert_eq!(explicit.relations[0].op, CovariateOp::Multiply);

    let add = spec("  WT continuous", "  CL ~ WT power(center = 70) +");
    assert_eq!(add.relations[0].op, CovariateOp::Add);
}

#[test]
fn the_operator_is_read_before_the_theta_clause() {
    let s = spec(
        "  WT continuous",
        "  CL ~ WT linear(center = 70) + => THETA_CL_WT(0.01, -1, 1)",
    );
    assert_eq!(s.relations[0].op, CovariateOp::Add);
    assert_eq!(s.generated_thetas, vec!["  theta THETA_CL_WT(0.01, -1, 1)"]);
    assert_eq!(
        cl_line(&s),
        "CL = TVCL * exp(ETA_CL) + (if (present(WT)) (THETA_CL_WT * (WT - 70)) else 0.0)"
    );
}

#[test]
fn an_operator_with_no_form_is_an_error() {
    let e = err("  WT continuous", "  CL ~ WT +");
    assert!(e.contains("states no form"), "{e}");

    // …and with no covariate either, the operator split is what catches it.
    let e = err("  WT continuous", "  CL ~ +");
    assert!(e.contains("states the `+` operator but no form"), "{e}");
}

#[test]
fn the_two_operators_are_still_one_line_per_pair() {
    // `CL ~ WT linear` and `CL ~ WT linear +` are *competing* models of the
    // same pair, not two effects to be combined — the block's one-line-per-pair
    // rule is what says so, and it must not be read as operator-scoped.
    let e = err(
        "  WT continuous",
        "  CL ~ WT linear(center = 70)\n  CL ~ WT linear(center = 70) +",
    );
    assert!(e.contains("declared more than once"), "{e}");
}
