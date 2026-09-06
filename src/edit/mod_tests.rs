//! Unit tests for `ferx-core::edit` (#1176).
//!
//! Every [`ModelEdit`] variant is pinned twice: once on the text it renders,
//! here, and once on the *fit* it produces, in
//! `tests/model_edit_equivalence.rs`. The rendered-text assertion is the one
//! that localises a defect (it says which character moved); the fit assertion
//! is the one that says the edited model is the model the user would have
//! written by hand.

use super::*;
use crate::types::{CovariateForm, CovariateStat};

/// The model every edit test starts from: one-compartment oral, three θ, three
/// η, a proportional error model — deliberately comment- and alignment-heavy,
/// so an edit that reflows the file fails loudly.
const BASE: &str = "\
# One-compartment oral PK model
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[covariates]
  WT continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 300
";

/// The two-compartment model the narrowing tests start from. `SetStructural`
/// to `one_cpt_oral` must take `Q`, `V2`, `TVQ`, `TVV2`, `ETA_Q` and `ETA_V2`
/// with it.
const TWO_CPT: &str = "\
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_Q  ~ 0.08
  omega ETA_V2 ~ 0.08
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Narrow a two-compartment model to `one_cpt_oral` — the edit every pruning
/// test is about.
fn narrow_to_one_cpt() -> ModelEdit<'static> {
    ModelEdit::SetStructural(StructuralSpec {
        template: "one_cpt_oral".into(),
        bindings: vec![
            ("cl".into(), "CL".into()),
            ("v".into(), "V1".into()),
            ("ka".into(), "KA".into()),
        ],
        new_parameters: vec![],
    })
}

/// [`BASE`] with `ETA_CL` and `ETA_V` blocked, written over several lines the
/// way `examples/warfarin_frem.ferx` writes its block.
fn multi_line_block_base() -> String {
    BASE.replace("  omega ETA_CL ~ 0.09\n", "").replace(
        "  omega ETA_V  ~ 0.04\n",
        "  block_omega (ETA_CL, ETA_V) = [\n    0.09,\n    0.006, 0.04\n  ]\n",
    )
}

fn base() -> ModelText {
    ModelText::parse(BASE).expect("the base model must parse")
}

fn apply(text: &mut ModelText, edit: ModelEdit<'_>) {
    text.apply(edit).expect("edit must apply");
}

// ── ModelText: the round-trip that makes everything else safe ───────────────

#[test]
fn render_is_byte_identical_when_unedited() {
    assert_eq!(base().render(), BASE);
}

#[test]
fn render_preserves_crlf_and_a_missing_final_newline() {
    let src = "[parameters]\r\n  theta TVCL(1.0)\r\n\r\n[structural_model]\r\n  pk one_cpt_iv(cl=CL, v=V)";
    assert_eq!(ModelText::parse(src).unwrap().render(), src);
}

#[test]
fn a_source_with_no_block_header_is_rejected() {
    let err = ModelText::parse("theta TVCL(1.0)\n").unwrap_err();
    assert!(err.contains("no `[block]` header"), "{err}");
}

#[test]
fn block_headers_are_found_through_comments_and_case() {
    let text = ModelText::parse("[Parameters]   # the block\n  theta TVCL(1.0)\n").unwrap();
    assert_eq!(text.block_names(), vec!["parameters".to_string()]);
    assert_eq!(text.block_lines("parameters"), vec!["theta TVCL(1.0)"]);
}

#[test]
fn a_bracketed_expression_is_not_mistaken_for_a_block_header() {
    // `states=[central]` sits inside a line; only a whole-line `[name]` opens a
    // block. Getting this wrong would put every ODE line in a phantom block.
    let src = "[structural_model]\n  ode(obs_cmt=1, states=[central])\n";
    let text = ModelText::parse(src).unwrap();
    assert_eq!(text.block_names(), vec!["structural_model".to_string()]);
}

#[test]
fn appending_to_the_last_block_does_not_glue_onto_a_missing_final_newline() {
    // `[fit_options]` is very often the block that ends the file, and a file
    // may end without a terminator — appending straight after that line would
    // render `method = focei  seed = 1` as one line.
    let src = "[parameters]\n  theta TVCL(1.0)\n\n[fit_options]\n  method = focei";
    let mut text = ModelText::parse(src).unwrap();
    apply(
        &mut text,
        ModelEdit::SetFitOption {
            key: "seed".into(),
            value: "1".into(),
        },
    );
    assert_eq!(
        text.render(),
        "[parameters]\n  theta TVCL(1.0)\n\n[fit_options]\n  method = focei\n  seed = 1\n"
    );
}

// ── canonical_hash ─────────────────────────────────────────────────────────

#[test]
fn canonical_hash_ignores_comments_blank_lines_and_spacing() {
    let reflowed = "\
[parameters]

    # a different comment entirely
    theta TVCL(0.2,0.001,10.0)
    theta TVV(10.0, 0.1, 500.0)
    theta TVKA(1.5, 0.01, 50.0)
    omega ETA_CL~0.09
    omega ETA_V ~ 0.04
    omega ETA_KA ~ 0.30
    sigma PROP_ERR ~ 0.02 (sd)
[INDIVIDUAL_PARAMETERS]
    CL = TVCL*exp(ETA_CL)
    V = TVV*exp(ETA_V)
    KA = TVKA*exp(ETA_KA)
[structural_model]
    pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
    WT continuous
[error_model]
    DV ~ proportional(PROP_ERR)
[fit_options]
    method = foce
    maxiter = 300
";
    assert_eq!(
        base().canonical_hash(),
        ModelText::parse(reflowed).unwrap().canonical_hash()
    );
}

#[test]
fn canonical_hash_changes_for_every_semantic_edit() {
    let base_hash = base().canonical_hash();
    let mut seen = std::collections::HashSet::new();
    seen.insert(base_hash);

    let edits: Vec<ModelEdit<'static>> = vec![
        ModelEdit::DropIiv { param: "KA".into() },
        ModelEdit::SetFitOption {
            key: "maxiter".into(),
            value: "301".into(),
        },
        ModelEdit::SetOmegaBlock(vec!["ETA_CL".into(), "ETA_V".into()]),
        ModelEdit::AddCovariateRelation(Relation {
            parameter: "CL".into(),
            covariate: "WT".into(),
            form: CovariateForm::Power,
            center: Some(CovariateStat::Literal(70.0)),
            fix: None,
            thetas: vec![],
        }),
    ];
    for edit in edits {
        let mut text = base();
        apply(&mut text, edit);
        assert!(
            seen.insert(text.canonical_hash()),
            "an edit left the canonical hash unchanged:\n{}",
            text.canonical_form()
        );
    }
}

#[test]
fn canonical_form_keeps_a_space_between_two_words() {
    // `theta TVCL` must not collapse to `thetaTVCL`, or a declaration and a
    // typo'd identifier would hash alike.
    let text = ModelText::parse("[parameters]\n  theta   TVCL( 1.0 , 0.1 )\n").unwrap();
    assert_eq!(text.canonical_form(), "[parameters]\ntheta TVCL(1.0,0.1)\n");
}

#[test]
fn canonical_hash_keeps_the_bytes_inside_a_quoted_value() {
    // Whitespace inside quotes is content, not spacing: these name two
    // different files, and one canonical form for both would have the fit
    // cache answer one candidate with the other's result.
    let spaced = ModelText::parse("[data]\n  path = \"cohort / A.csv\"\n").unwrap();
    let tight = ModelText::parse("[data]\n  path = \"cohort/A.csv\"\n").unwrap();
    assert_ne!(spaced.canonical_hash(), tight.canonical_hash());
    // Spacing *outside* the quotes still normalises away.
    let reflowed = ModelText::parse("[data]\n  path=\"cohort / A.csv\"\n").unwrap();
    assert_eq!(spaced.canonical_hash(), reflowed.canonical_hash());
}

#[test]
fn canonical_hash_keeps_a_comment_marker_inside_a_quoted_value() {
    // `squeeze` copies a quoted string byte for byte, but it never sees one
    // that `code_of` already cut: a naive "cut at the first `#` or `//`" ends
    // both of these at `path = "s3:`, and the search then fits one candidate
    // and reports its criterion for the other under `duplicate_of`.
    for (a, b) in [
        ("\"s3://bucket/a.csv\"", "\"s3://bucket/b.csv\""),
        ("\"cohort#A.csv\"", "\"cohort#B.csv\""),
        ("'s3://bucket/a.csv'", "'s3://bucket/b.csv'"),
    ] {
        let one = ModelText::parse(&format!("[data]\n  path = {a}\n")).unwrap();
        let two = ModelText::parse(&format!("[data]\n  path = {b}\n")).unwrap();
        assert_ne!(
            one.canonical_hash(),
            two.canonical_hash(),
            "`{a}` and `{b}` canonicalised alike:\n{}",
            one.canonical_form()
        );
        // And the straddle the pair exists to test: the marker must actually be
        // past the opening quote, or both spellings survive a naive cut too.
        assert!(a.find("//").or_else(|| a.find('#')).unwrap() > 0);
    }
}

#[test]
fn code_of_still_strips_a_comment_that_follows_a_quoted_value() {
    // The quote closes before the marker, so the marker is a comment again —
    // otherwise the fix would trade a wrong hash for an uncommentable line.
    let with = ModelText::parse("[data]\n  path = \"a.csv\"  # the cohort\n").unwrap();
    let without = ModelText::parse("[data]\n  path = \"a.csv\"\n").unwrap();
    assert_eq!(with.canonical_hash(), without.canonical_hash());
    let slashes = ModelText::parse("[data]\n  path = \"a.csv\"  // the cohort\n").unwrap();
    assert_eq!(slashes.canonical_hash(), without.canonical_hash());
}

// ── SetFitOption ───────────────────────────────────────────────────────────

#[test]
fn set_fit_option_replaces_a_value_and_keeps_the_alignment() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetFitOption {
            key: "method".into(),
            value: "focei".into(),
        },
    );
    assert!(
        text.render().contains("  method     = focei\n"),
        "{}",
        text.render()
    );
}

#[test]
fn set_fit_option_appends_an_absent_key() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetFitOption {
            key: "gradient".into(),
            value: "fd".into(),
        },
    );
    assert_eq!(
        text.block_lines("fit_options"),
        vec!["method     = foce", "maxiter    = 300", "gradient = fd"]
    );
}

#[test]
fn set_fit_option_creates_the_block_when_it_is_missing() {
    let mut text = ModelText::parse("[parameters]\n  theta TVCL(1.0)\n").unwrap();
    apply(
        &mut text,
        ModelEdit::SetFitOption {
            key: "method".into(),
            value: "focei".into(),
        },
    );
    assert_eq!(
        text.render(),
        "[parameters]\n  theta TVCL(1.0)\n\n[fit_options]\n  method = focei\n"
    );
}

#[test]
fn set_fit_option_refuses_a_comment_marker() {
    // `code_of` strips the comment before the parser sees the line, so an
    // option written with one would read back truncated.
    for value in ["1 # nope", "1 // nope"] {
        let err = base()
            .apply(ModelEdit::SetFitOption {
                key: "seed".into(),
                value: value.into(),
            })
            .unwrap_err();
        assert!(err.contains("comment marker"), "{err}");
    }
}

// ── AddIiv / DropIiv ───────────────────────────────────────────────────────

#[test]
fn drop_iiv_removes_the_factor_and_the_omega_line() {
    let mut text = base();
    apply(&mut text, ModelEdit::DropIiv { param: "KA".into() });
    assert!(text
        .block_lines("individual_parameters")
        .contains(&"KA = TVKA".to_string()));
    assert!(!text.render().contains("ETA_KA"));
    // …and nothing else moved.
    assert!(text.render().contains("  CL = TVCL * exp(ETA_CL)\n"));
    assert!(text.render().contains("  omega ETA_V  ~ 0.04\n"));
}

#[test]
fn add_iiv_is_the_inverse_of_drop_iiv() {
    let mut text = base();
    apply(&mut text, ModelEdit::DropIiv { param: "KA".into() });
    apply(
        &mut text,
        ModelEdit::AddIiv {
            param: "KA".into(),
            form: IivForm::Exponential {
                eta: "ETA_KA".into(),
                variance: 0.30,
            },
        },
    );
    // The η comes back at the end of [parameters] rather than in its original
    // slot, so the files differ — but the *model* is the same one.
    assert!(text
        .block_lines("individual_parameters")
        .contains(&"KA = TVKA * exp(ETA_KA)".to_string()));
    assert!(text
        .block_lines("parameters")
        .contains(&"omega ETA_KA ~ 0.3".to_string()));
}

#[test]
fn add_iiv_rejects_a_parameter_that_already_has_one() {
    let err = base()
        .apply(ModelEdit::AddIiv {
            param: "CL".into(),
            form: IivForm::Exponential {
                eta: "ETA_CL2".into(),
                variance: 0.1,
            },
        })
        .unwrap_err();
    assert!(
        err.contains("already carries the random effect `ETA_CL`"),
        "{err}"
    );
}

#[test]
fn drop_iiv_on_a_non_canonical_expression_is_a_hard_error_naming_the_parameter() {
    let src = BASE.replace(
        "  CL = TVCL * exp(ETA_CL)",
        "  CL = TVCL * (1 + ETA_CL) + 0.5",
    );
    let err = ModelText::parse(&src)
        .unwrap()
        .apply(ModelEdit::DropIiv { param: "CL".into() })
        .unwrap_err();
    assert!(err.contains("`CL ="), "{err}");
    assert!(err.contains("canonical form"), "{err}");
}

#[test]
fn drop_iiv_refuses_to_break_a_block_omega() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetOmegaBlock(vec!["ETA_CL".into(), "ETA_V".into()]),
    );
    let err = text
        .apply(ModelEdit::DropIiv { param: "CL".into() })
        .unwrap_err();
    assert!(err.contains("block_omega"), "{err}");
}

#[test]
fn a_rejected_edit_leaves_the_text_untouched() {
    let mut text = base();
    let before = text.render();
    assert!(text
        .apply(ModelEdit::DropIiv {
            param: "NOPE".into()
        })
        .is_err());
    assert_eq!(text.render(), before);
}

#[test]
fn add_iiv_refuses_an_eta_a_block_omega_already_declares() {
    // The duplicate guard reads `omega NAME ~ …`, which does not match a
    // `block_omega` line — without a second look the same η is declared twice.
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetOmegaBlock(vec!["ETA_CL".into(), "ETA_V".into()]),
    );
    apply(&mut text, ModelEdit::DropIiv { param: "KA".into() });
    let err = text
        .apply(ModelEdit::AddIiv {
            param: "KA".into(),
            form: IivForm::Exponential {
                eta: "ETA_CL".into(),
                variance: 0.1,
            },
        })
        .unwrap_err();
    assert!(err.contains("already declared by `block_omega"), "{err}");
}

#[test]
fn add_iiv_tells_a_float_exponent_from_an_identifier_that_ends_in_e() {
    // `TVCL_BASE-1` is a difference, not a product: appending `* exp(η)` to it
    // re-associates the arithmetic silently. Reading one character back turns
    // the `E` into a float exponent and lets exactly that through.
    let src = BASE.replace("CL = TVCL * exp(ETA_CL)", "CL = TVCL_BASE-1");
    let err = ModelText::parse(&src)
        .unwrap()
        .apply(ModelEdit::AddIiv {
            param: "CL".into(),
            form: IivForm::Exponential {
                eta: "ETA_X".into(),
                variance: 0.1,
            },
        })
        .unwrap_err();
    assert!(err.contains("not a top-level product"), "{err}");

    // A real exponent is still part of the number.
    let src = BASE.replace("CL = TVCL * exp(ETA_CL)", "CL = TVCL * 1e-5");
    let mut text = ModelText::parse(&src).unwrap();
    apply(
        &mut text,
        ModelEdit::AddIiv {
            param: "CL".into(),
            form: IivForm::Exponential {
                eta: "ETA_X".into(),
                variance: 0.1,
            },
        },
    );
    assert!(
        text.block_lines("individual_parameters")
            .contains(&"CL = TVCL * 1e-5 * exp(ETA_X)".to_string()),
        "{:?}",
        text.block_lines("individual_parameters")
    );
}

#[test]
fn drop_iiv_keeps_an_omega_a_second_parameter_still_uses() {
    // Two parameters may legally share one η. Dropping it from `CL` must not
    // take the declaration with it, or `V` is left reading a random effect the
    // model never declares.
    let src = BASE
        .replace("V  = TVV  * exp(ETA_V)", "V  = TVV  * exp(ETA_CL)")
        .replace("  omega ETA_V  ~ 0.04\n", "");
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, ModelEdit::DropIiv { param: "CL".into() });
    let params = text.block_lines("parameters");
    assert!(
        params.contains(&"omega ETA_CL ~ 0.09".to_string()),
        "{params:?}"
    );
    let indiv = text.block_lines("individual_parameters");
    assert!(indiv.contains(&"CL = TVCL".to_string()), "{indiv:?}");
    assert!(
        indiv.contains(&"V  = TVV  * exp(ETA_CL)".to_string()),
        "{indiv:?}"
    );
    // The last user of the η does take the declaration with it.
    apply(&mut text, ModelEdit::DropIiv { param: "V".into() });
    assert!(
        !text
            .block_lines("parameters")
            .iter()
            .any(|l| l.contains("ETA_CL")),
        "{:?}",
        text.block_lines("parameters")
    );
}

#[test]
fn a_multi_line_block_omega_is_one_declaration_to_the_duplicate_guards() {
    // `examples/warfarin_frem.ferx` writes its block over seven lines, and the
    // parser rejoins it before matching. An editor reading physical lines sees
    // no `block_omega` at all.
    let mut text = ModelText::parse(&multi_line_block_base()).unwrap();
    let err = text
        .apply(ModelEdit::DropIiv { param: "CL".into() })
        .unwrap_err();
    assert!(err.contains("block_omega"), "{err}");
    let err = text
        .apply(ModelEdit::AddIiv {
            param: "KA".into(),
            form: IivForm::Exponential {
                eta: "ETA_V".into(),
                variance: 0.1,
            },
        })
        .unwrap_err();
    assert!(err.contains("already declared by `block_omega"), "{err}");
}

// ── SetOmegaBlock ──────────────────────────────────────────────────────────

#[test]
fn set_omega_block_replaces_the_diagonal_declarations_in_place() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetOmegaBlock(vec!["ETA_CL".into(), "ETA_V".into()]),
    );
    let params = text.block_lines("parameters");
    // 0.1 × sqrt(0.09 × 0.04) = 0.006.
    assert!(
        params.contains(&"block_omega (ETA_CL, ETA_V) = [0.09, 0.006, 0.04]".to_string()),
        "{params:?}"
    );
    assert!(!params.iter().any(|l| l.starts_with("omega ETA_CL")));
    assert!(!params.iter().any(|l| l.starts_with("omega ETA_V ")));
    // The η that was not blocked is untouched.
    assert!(params.contains(&"omega ETA_KA ~ 0.30".to_string()));
}

#[test]
fn set_omega_block_reads_an_sd_declaration_on_its_own_scale() {
    let src = BASE.replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.3 (sd)");
    let mut text = ModelText::parse(&src).unwrap();
    apply(
        &mut text,
        ModelEdit::SetOmegaBlock(vec!["ETA_CL".into(), "ETA_V".into()]),
    );
    assert!(
        text.block_lines("parameters")
            .contains(&"block_omega (ETA_CL, ETA_V) = [0.09, 0.006, 0.04]".to_string()),
        "{:?}",
        text.block_lines("parameters")
    );
}

#[test]
fn set_omega_block_rejects_a_single_eta_and_an_unknown_one() {
    assert!(base()
        .apply(ModelEdit::SetOmegaBlock(vec!["ETA_CL".into()]))
        .unwrap_err()
        .contains("at least two"));
    assert!(base()
        .apply(ModelEdit::SetOmegaBlock(vec![
            "ETA_CL".into(),
            "ETA_NOPE".into()
        ]))
        .unwrap_err()
        .contains("no `omega ETA_NOPE"));
}

#[test]
fn set_omega_block_lists_the_eta_in_declaration_order() {
    // The block takes the place of the *first* declaration, so writing the η
    // in the caller's order would permute the omega matrix against the
    // `eta_names` the model had before the edit.
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetOmegaBlock(vec!["ETA_V".into(), "ETA_CL".into()]),
    );
    assert!(
        text.block_lines("parameters")
            .contains(&"block_omega (ETA_CL, ETA_V) = [0.09, 0.006, 0.04]".to_string()),
        "{:?}",
        text.block_lines("parameters")
    );
}

#[test]
fn set_omega_block_carries_fix_and_refuses_a_mixed_block() {
    // `FIX` is a block-level flag. Dropping it silently would turn two fixed
    // variances into estimated ones — a different model, with two parameters
    // the search does not know it added.
    let src = BASE
        .replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.09 FIX")
        .replace("omega ETA_V  ~ 0.04", "omega ETA_V  ~ 0.04 FIX");
    let mut text = ModelText::parse(&src).unwrap();
    apply(
        &mut text,
        ModelEdit::SetOmegaBlock(vec!["ETA_CL".into(), "ETA_V".into()]),
    );
    assert!(
        text.block_lines("parameters")
            .contains(&"block_omega (ETA_CL, ETA_V) = [0.09, 0.006, 0.04] FIX".to_string()),
        "{:?}",
        text.block_lines("parameters")
    );

    // One flag cannot say "fix these two and estimate that one".
    let src = BASE.replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.09 FIX");
    let err = ModelText::parse(&src)
        .unwrap()
        .apply(ModelEdit::SetOmegaBlock(vec![
            "ETA_CL".into(),
            "ETA_V".into(),
        ]))
        .unwrap_err();
    assert!(err.contains("FIX"), "{err}");
    assert!(err.contains("ETA_V"), "{err}");
}

// ── [covariate_model] ──────────────────────────────────────────────────────

fn power_on_wt() -> Relation {
    Relation {
        parameter: "CL".into(),
        covariate: "WT".into(),
        form: CovariateForm::Power,
        center: Some(CovariateStat::Median),
        fix: None,
        thetas: vec![RelationTheta {
            name: "THETA_CL_WT".into(),
            init: 0.75,
            lower: 0.01,
            upper: 5.0,
        }],
    }
}

#[test]
fn add_covariate_relation_creates_the_block_and_writes_the_line() {
    let mut text = base();
    apply(&mut text, ModelEdit::AddCovariateRelation(power_on_wt()));
    assert_eq!(
        text.block_lines("covariate_model"),
        vec!["CL ~ WT power(center = median) => THETA_CL_WT(0.75, 0.01, 5.0)"]
    );
}

#[test]
fn relation_rendering_uses_the_keyword_each_form_actually_takes() {
    let mut hockey = power_on_wt();
    hockey.form = CovariateForm::Hockey;
    hockey.thetas.clear();
    assert_eq!(hockey.render(), "CL ~ WT hockey(breakpoint = median)");

    let mut categorical = power_on_wt();
    categorical.form = CovariateForm::Categorical;
    categorical.center = Some(CovariateStat::Mode);
    categorical.thetas.clear();
    assert_eq!(categorical.render(), "CL ~ WT categorical(ref = mode)");

    let mut none = power_on_wt();
    none.form = CovariateForm::None;
    none.thetas.clear();
    assert_eq!(none.render(), "CL ~ WT none");

    let mut expr = power_on_wt();
    expr.form = CovariateForm::Expr("(WT/70)^0.75".into());
    expr.thetas.clear();
    assert_eq!(expr.render(), "CL ~ WT expr(\"(WT/70)^0.75\")");

    let mut fixed = power_on_wt();
    fixed.fix = Some(0.75);
    fixed.thetas.clear();
    assert_eq!(fixed.render(), "CL ~ WT power(center = median, fix = 0.75)");
}

#[test]
fn adding_the_same_relation_twice_is_rejected() {
    let mut text = base();
    apply(&mut text, ModelEdit::AddCovariateRelation(power_on_wt()));
    let err = text
        .apply(ModelEdit::AddCovariateRelation(power_on_wt()))
        .unwrap_err();
    assert!(err.contains("already declares `CL ~ WT`"), "{err}");
}

#[test]
fn drop_covariate_relation_removes_the_line_and_its_orphaned_theta() {
    let mut text = base();
    apply(&mut text, ModelEdit::AddCovariateRelation(power_on_wt()));
    // A θ the block would defer to, declared classically.
    apply(
        &mut text,
        ModelEdit::SetFitOption {
            key: "maxiter".into(),
            value: "1".into(),
        },
    );
    apply(
        &mut text,
        ModelEdit::DropCovariateRelation {
            param: "CL".into(),
            cov: "WT".into(),
        },
    );
    assert!(text.block_lines("covariate_model").is_empty());
    // The rest of the model is intact.
    assert!(text
        .block_lines("individual_parameters")
        .contains(&"CL = TVCL * exp(ETA_CL)".to_string()));
}

#[test]
fn dropping_a_relation_that_is_not_there_names_the_pair() {
    let err = base()
        .apply(ModelEdit::DropCovariateRelation {
            param: "CL".into(),
            cov: "WT".into(),
        })
        .unwrap_err();
    assert!(err.contains("no `CL ~ WT` relation"), "{err}");
}

#[test]
fn dropping_a_relation_prunes_only_its_own_generated_theta_family() {
    // `THETA_CL_WT2` starts with `THETA_CL_WT` but belongs to a different
    // covariate: a raw prefix test keeps it alive after its own relation is
    // dropped, and the candidate reports a parameter it never estimates.
    let src = BASE
        .replace(
            "[covariates]",
            "[covariate_model]\n  CL ~ WT power(center = median)\n  CL ~ WT2 power(center = \
             median)\n\n[covariates]",
        )
        .replace(
            "  theta TVKA(1.5, 0.01, 50.0)",
            "  theta TVKA(1.5, 0.01, 50.0)\n  theta THETA_CL_WT(0.75, 0.01, 5.0)\n  theta \
             THETA_CL_WT2(0.5, 0.01, 5.0)",
        );
    let mut text = ModelText::parse(&src).unwrap();
    apply(
        &mut text,
        ModelEdit::DropCovariateRelation {
            param: "CL".into(),
            cov: "WT2".into(),
        },
    );
    let params = text.block_lines("parameters");
    assert!(
        params.iter().any(|l| l.starts_with("theta THETA_CL_WT(")),
        "{params:?}"
    );
    assert!(
        !params.iter().any(|l| l.contains("THETA_CL_WT2")),
        "{params:?}"
    );
}

// ── SetStructural ──────────────────────────────────────────────────────────

#[test]
fn set_structural_widens_to_two_compartments_and_declares_the_new_parameters() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetStructural(StructuralSpec {
            template: "two_cpt_oral".into(),
            bindings: vec![
                ("cl".into(), "CL".into()),
                ("v1".into(), "V".into()),
                ("q".into(), "Q".into()),
                ("v2".into(), "V2".into()),
                ("ka".into(), "KA".into()),
            ],
            new_parameters: vec![
                NewParameter {
                    name: "Q".into(),
                    theta: "TVQ".into(),
                    init: 2.0,
                    lower: 0.01,
                    upper: 100.0,
                    iiv: None,
                    fixed: false,
                },
                NewParameter {
                    name: "V2".into(),
                    theta: "TVV2".into(),
                    init: 20.0,
                    lower: 0.1,
                    upper: 500.0,
                    iiv: Some(("ETA_V2".into(), 0.1)),
                    fixed: false,
                },
            ],
        }),
    );
    assert_eq!(
        text.block_lines("structural_model"),
        vec!["pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)"]
    );
    let indiv = text.block_lines("individual_parameters");
    assert!(indiv.contains(&"Q = TVQ".to_string()), "{indiv:?}");
    assert!(
        indiv.contains(&"V2 = TVV2 * exp(ETA_V2)".to_string()),
        "{indiv:?}"
    );
    let params = text.block_lines("parameters");
    assert!(
        params.contains(&"theta TVQ(2.0, 0.01, 100.0)".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"theta TVV2(20.0, 0.1, 500.0)".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"omega ETA_V2 ~ 0.1".to_string()),
        "{params:?}"
    );
}

#[test]
fn set_structural_declares_a_fixed_new_parameter_with_fix() {
    // A structural constant stated as a parameter — modelsearch's
    // `TRANSITS(3)` (#1181) — so the count has a name the ODE twin can bind,
    // unlike a literal `n=3`. `SeedInits` skips `FIX` lines, so a later
    // carry-over cannot turn the constant into an estimate.
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetStructural(StructuralSpec {
            template: "one_cpt_transit".into(),
            bindings: vec![
                ("cl".into(), "CL".into()),
                ("v".into(), "V".into()),
                ("n".into(), "NTR".into()),
                ("mtt".into(), "MTT".into()),
            ],
            new_parameters: vec![
                NewParameter {
                    name: "NTR".into(),
                    theta: "TVNTR".into(),
                    init: 3.0,
                    lower: 0.0,
                    upper: 64.0,
                    iiv: None,
                    fixed: true,
                },
                NewParameter {
                    name: "MTT".into(),
                    theta: "TVMTT".into(),
                    init: 0.5,
                    lower: 0.0,
                    upper: 100.0,
                    iiv: None,
                    fixed: false,
                },
            ],
        }),
    );
    let params = text.block_lines("parameters");
    assert!(
        params.contains(&"theta TVNTR(3.0, 0.0, 64.0) FIX".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"theta TVMTT(0.5, 0.0, 100.0)".to_string()),
        "{params:?}"
    );
    // The mutation this pins: dropping the `FIX` suffix makes the count a
    // free θ, which an evaluation would not notice but a fit would estimate.
    assert!(!params.iter().any(|l| l == "theta TVNTR(3.0, 0.0, 64.0)"));
}

#[test]
fn set_structural_prunes_the_parameters_the_new_template_no_longer_uses() {
    // Start from the two-compartment model and narrow it: `Q`, `V2`, `TVQ`,
    // `TVV2`, `ETA_Q` and `ETA_V2` must all go — three blocks, one edit.
    let mut text = ModelText::parse(TWO_CPT).unwrap();
    apply(&mut text, narrow_to_one_cpt());
    let params = text.block_lines("parameters");
    let indiv = text.block_lines("individual_parameters");
    for gone in ["TVQ", "TVV2", "ETA_Q", "ETA_V2"] {
        assert!(
            !params.iter().any(|l| l.contains(gone)),
            "`{gone}` survived: {params:?}"
        );
    }
    assert_eq!(
        indiv,
        vec![
            "CL = TVCL * exp(ETA_CL)",
            "V1 = TVV1 * exp(ETA_V1)",
            "KA = TVKA * exp(ETA_KA)",
        ]
    );
    // σ is untouched by structural pruning — it belongs to the error model.
    assert!(params.contains(&"sigma PROP_ERR ~ 0.04 (sd)".to_string()));
}

#[test]
fn set_structural_rejects_an_unknown_template_and_an_undeclared_binding() {
    assert!(base()
        .apply(ModelEdit::SetStructural(StructuralSpec {
            template: "four_cpt_oral".into(),
            bindings: vec![("cl".into(), "CL".into())],
            new_parameters: vec![],
        }))
        .unwrap_err()
        .contains("not a known `pk` template"));

    let err = base()
        .apply(ModelEdit::SetStructural(StructuralSpec {
            template: "two_cpt_oral".into(),
            bindings: vec![
                ("cl".into(), "CL".into()),
                ("v1".into(), "V".into()),
                ("q".into(), "Q".into()),
                ("v2".into(), "V2".into()),
                ("ka".into(), "KA".into()),
            ],
            new_parameters: vec![],
        }))
        .unwrap_err();
    assert!(err.contains("`Q`"), "{err}");
    assert!(err.contains("new_parameters"), "{err}");
}

#[test]
fn set_structural_needs_a_pk_line() {
    let ode = "[structural_model]\n  ode(obs_cmt=1, states=[central])\n";
    let err = ModelText::parse(ode)
        .unwrap()
        .apply(ModelEdit::SetStructural(StructuralSpec {
            template: "one_cpt_iv".into(),
            bindings: vec![("cl".into(), "CL".into())],
            new_parameters: vec![],
        }))
        .unwrap_err();
    assert!(err.contains("pk NAME(...)"), "{err}");
}

#[test]
fn set_structural_shrinks_a_block_omega_around_the_eta_that_survive() {
    // A blocked η is declared nowhere else, so a block left whole after its
    // parameter is pruned leaves a random effect nothing uses: a flat omega
    // direction and one parameter too many in the BIC the search ranks on.
    let src = TWO_CPT
        .replace("  omega ETA_CL ~ 0.15\n", "")
        .replace("  omega ETA_V1 ~ 0.15\n", "")
        .replace(
            "  omega ETA_V2 ~ 0.08\n",
            "  block_omega (ETA_CL, ETA_V1, ETA_V2) = [0.15, 0.05, 0.15, 0.02, 0.03, 0.08]\n",
        );
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, narrow_to_one_cpt());
    let params = text.block_lines("parameters");
    // The surviving 2×2 sub-block, variances and covariance unchanged.
    assert!(
        params.contains(&"block_omega (ETA_CL, ETA_V1) = [0.15, 0.05, 0.15]".to_string()),
        "{params:?}"
    );
    assert!(!params.iter().any(|l| l.contains("ETA_V2")), "{params:?}");
}

#[test]
fn set_structural_collapses_a_block_omega_with_one_survivor_to_a_diagonal_omega() {
    let src = TWO_CPT.replace("  omega ETA_CL ~ 0.15\n", "").replace(
        "  omega ETA_V2 ~ 0.08\n",
        "  block_omega (ETA_CL, ETA_V2) = [0.15, 0.03, 0.08]\n",
    );
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, narrow_to_one_cpt());
    let params = text.block_lines("parameters");
    // A 1×1 block is not a form the parser accepts, so the survivor goes back
    // to the diagonal declaration it came from.
    assert!(
        params.contains(&"omega ETA_CL ~ 0.15".to_string()),
        "{params:?}"
    );
    assert!(
        !params.iter().any(|l| l.contains("block_omega")),
        "{params:?}"
    );
}

#[test]
fn set_structural_deletes_a_block_omega_whose_eta_all_die() {
    let src = TWO_CPT.replace("  omega ETA_Q  ~ 0.08\n", "").replace(
        "  omega ETA_V2 ~ 0.08\n",
        "  block_omega (ETA_Q, ETA_V2) = [0.08, 0.01, 0.08]\n",
    );
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, narrow_to_one_cpt());
    let params = text.block_lines("parameters");
    assert!(
        !params
            .iter()
            .any(|l| l.contains("ETA_Q") || l.contains("ETA_V2")),
        "{params:?}"
    );
}

#[test]
fn set_structural_refuses_to_prune_through_a_conditional_individual_parameters_block() {
    // Reading a branch's assignments as roots instead would keep `TVQ` and
    // `ETA_Q` alive for ever — an orphaned θ and a phantom η. Removing a line
    // from inside a branch is a guess about the branch, so the edit is refused
    // and the text left exactly as it was found.
    let src = TWO_CPT.replace(
        "  Q  = TVQ  * exp(ETA_Q)\n",
        "  if (WT > 70) {\n    Q = TVQ * 1.2 * exp(ETA_Q)\n  } else {\n    Q = TVQ * exp(ETA_Q)\n  }\n",
    );
    let mut text = ModelText::parse(&src).unwrap();
    let before = text.render();
    let err = text.apply(narrow_to_one_cpt()).unwrap_err();
    assert!(err.contains("`Q`"), "{err}");
    assert!(err.contains("if"), "{err}");
    assert_eq!(text.render(), before);
}

#[test]
fn set_structural_shrinks_a_multi_line_block_omega() {
    // The shipped multi-line form, narrowed: the block is one declaration, so
    // it is rewritten as one and its continuation lines go with it.
    let src = TWO_CPT
        .replace("  omega ETA_CL ~ 0.15\n", "")
        .replace("  omega ETA_V1 ~ 0.15\n", "")
        .replace("  omega ETA_Q  ~ 0.08\n", "")
        .replace(
            "  omega ETA_V2 ~ 0.08\n",
            "  block_omega (ETA_CL, ETA_V1, ETA_Q, ETA_V2) = [\n    0.15,\n    0.05, 0.15,\n              0.0, 0.0, 0.08,\n    0.0, 0.0, 0.02, 0.08\n  ]\n",
        );
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, narrow_to_one_cpt());
    let params = text.block_lines("parameters");
    assert!(
        params.contains(&"block_omega (ETA_CL, ETA_V1) = [0.15, 0.05, 0.15]".to_string()),
        "{params:?}"
    );
    // No orphaned continuation line survives anywhere in the file.
    let rendered = text.render();
    for gone in ["ETA_Q", "ETA_V2", "0.02"] {
        assert!(!rendered.contains(gone), "`{gone}` survived:\n{rendered}");
    }
}

// ── SetErrorModel ──────────────────────────────────────────────────────────

#[test]
fn set_error_model_swaps_the_statement_and_reconciles_the_sigmas() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetErrorModel(ErrorSpecText {
            endpoint: "DV".into(),
            form: ErrorForm::Combined,
            sigmas: vec![
                SigmaDecl {
                    name: "ADD_ERR".into(),
                    init: 0.1,
                    as_sd: true,
                },
                SigmaDecl {
                    name: "PROP_ERR".into(),
                    init: 0.02,
                    as_sd: true,
                },
            ],
        }),
    );
    assert_eq!(
        text.block_lines("error_model"),
        vec!["DV ~ combined(ADD_ERR, PROP_ERR)"]
    );
    let params = text.block_lines("parameters");
    // The σ that already existed keeps its own declaration verbatim…
    assert!(
        params.contains(&"sigma PROP_ERR ~ 0.02 (sd)".to_string()),
        "{params:?}"
    );
    // …and the new one is declared *before* it, because a single-endpoint
    // [error_model] consumes its σ positionally from the declaration order.
    let sigmas: Vec<&String> = params.iter().filter(|l| l.starts_with("sigma ")).collect();
    assert_eq!(
        sigmas,
        vec!["sigma ADD_ERR ~ 0.1 (sd)", "sigma PROP_ERR ~ 0.02 (sd)"],
        "the σ declarations must be in the order `combined(...)` names them"
    );
}

#[test]
fn set_error_model_drops_a_sigma_nothing_references_any_more() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetErrorModel(ErrorSpecText {
            endpoint: "DV".into(),
            form: ErrorForm::Additive,
            sigmas: vec![SigmaDecl {
                name: "ADD_ERR".into(),
                init: 0.5,
                as_sd: true,
            }],
        }),
    );
    assert!(!text.render().contains("PROP_ERR"), "{}", text.render());
}

#[test]
fn set_error_model_refuses_a_per_cmt_block() {
    let src = BASE.replace(
        "  DV ~ proportional(PROP_ERR)",
        "  CMT=1: DV ~ proportional(PROP_ERR)\n  CMT=2: DV ~ additive(PROP_ERR)",
    );
    let err = ModelText::parse(&src)
        .unwrap()
        .apply(ModelEdit::SetErrorModel(ErrorSpecText {
            endpoint: "DV".into(),
            form: ErrorForm::Additive,
            sigmas: vec![SigmaDecl {
                name: "ADD_ERR".into(),
                init: 0.5,
                as_sd: true,
            }],
        }))
        .unwrap_err();
    assert!(err.contains("per-CMT"), "{err}");
}

#[test]
fn set_error_model_checks_the_sigma_count() {
    let err = base()
        .apply(ModelEdit::SetErrorModel(ErrorSpecText {
            endpoint: "DV".into(),
            form: ErrorForm::Combined,
            sigmas: vec![SigmaDecl {
                name: "ADD_ERR".into(),
                init: 0.5,
                as_sd: true,
            }],
        }))
        .unwrap_err();
    assert!(err.contains("takes 2 sigma(s), 1 given"), "{err}");
}

#[test]
fn set_error_model_refuses_a_block_sigma() {
    // σ are reconciled positionally against `sigma NAME ~ …` declarations; a
    // correlated block matches none of them, so the new σ would be appended
    // beside it and the model would carry more σ than the error model names.
    let src = BASE.replace(
        "  sigma PROP_ERR ~ 0.02 (sd)",
        "  block_sigma (PROP_ERR, ADD_ERR) = [0.02, 0.0, 0.1]",
    );
    let err = ModelText::parse(&src)
        .unwrap()
        .apply(ModelEdit::SetErrorModel(ErrorSpecText {
            endpoint: "DV".into(),
            form: ErrorForm::Additive,
            sigmas: vec![SigmaDecl {
                name: "ADD_ERR".into(),
                init: 0.5,
                as_sd: true,
            }],
        }))
        .unwrap_err();
    assert!(err.contains("block_sigma"), "{err}");
}

// ── num ─────────────────────────────────────────────────────────────────────

#[test]
fn num_drops_last_ulp_packing_noise_and_keeps_every_typed_value() {
    use super::spec::num;
    // The noise a log/exp round trip leaves on an evaluation's estimates
    // (measured on #1256's vancomycin dry run).
    assert_eq!(num(3.0000000000000004), "3.0");
    assert_eq!(num(49.99999999999999), "50.0");
    assert_eq!(num(0.19999999999999998), "0.2");
    assert_eq!(num(0.1 + 0.2), "0.3");
    // Values a user types are untouched.
    assert_eq!(num(0.2), "0.2");
    assert_eq!(num(1e-5), "0.00001");
    assert_eq!(num(1e6), "1000000.0");
    assert_eq!(num(0.132695), "0.132695");
    assert_eq!(num(-286.00421948870667), "-286.004219488707");
    assert_eq!(num(0.0), "0.0");
    assert_eq!(num(-0.5), "-0.5");
    // Fifteen significant digits, not fewer: the mutation this pins is a
    // rounding coarse enough to move a real estimate.
    assert_eq!(num(0.123456789012345), "0.123456789012345");
    assert_ne!(
        num(0.1234567890123456),
        "0.123456789012346".to_string() + "0"
    );
    // Non-finite values pass through to `finite()`'s rejection unchanged.
    assert_eq!(num(f64::INFINITY), "inf");
}

// ── SeedInits ──────────────────────────────────────────────────────────────

/// A minimal `FitResult` carrying only what `SeedInits` reads.
fn fit_with(
    theta_names: &[&str],
    theta: &[f64],
    eta_names: &[&str],
    omega_diag: &[f64],
    sigma_names: &[&str],
    sigma: &[f64],
) -> crate::types::FitResult {
    let n = eta_names.len();
    let mut omega = nalgebra::DMatrix::zeros(n, n);
    for (i, v) in omega_diag.iter().enumerate() {
        omega[(i, i)] = *v;
    }
    crate::types::FitResult {
        theta: theta.to_vec(),
        theta_names: theta_names.iter().map(|s| s.to_string()).collect(),
        eta_names: eta_names.iter().map(|s| s.to_string()).collect(),
        omega,
        sigma: sigma.to_vec(),
        sigma_names: sigma_names.iter().map(|s| s.to_string()).collect(),
        ..crate::types::test_helpers::empty_fit_result()
    }
}

#[test]
fn seed_inits_carries_estimates_over_by_name_not_position() {
    // The fit's θ are in a different order from the file's, and it carries one
    // the model does not have — the exact mismatch a candidate model creates.
    let fit = fit_with(
        &["TVKA", "TVCL", "TVQ"],
        &[2.5, 0.35, 9.9],
        &["ETA_V", "ETA_CL"],
        &[0.11, 0.22],
        &["PROP_ERR"],
        &[0.05],
    );
    let mut text = base();
    apply(&mut text, ModelEdit::SeedInits(&fit));
    let params = text.block_lines("parameters");
    assert!(
        params.contains(&"theta TVCL(0.35, 0.001, 10.0)".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"theta TVKA(2.5, 0.01, 50.0)".to_string()),
        "{params:?}"
    );
    // Untouched: the fit says nothing about TVV.
    assert!(
        params.contains(&"theta TVV(10.0, 0.1, 500.0)".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"omega ETA_CL ~ 0.22".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"omega ETA_V  ~ 0.11".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"omega ETA_KA ~ 0.30".to_string()),
        "{params:?}"
    );
    // σ is stored on the SD scale and the declaration says `(sd)`, so it is
    // written through unchanged.
    assert!(
        params.contains(&"sigma PROP_ERR ~ 0.05 (sd)".to_string()),
        "{params:?}"
    );
}

#[test]
fn seed_inits_floors_a_collapsed_variance_at_the_smallest_startable_one() {
    // A parent whose η collapsed to the optimizer's rail (6.1e-6 measured on
    // #1181) would hand its child a start the engine refuses. The seed is
    // floored at 1e-5, on the declaration's own scale; a healthy variance
    // and the θ are carried unchanged.
    let fit = fit_with(
        &["TVCL"],
        &[0.35],
        &["ETA_CL", "ETA_V", "ETA_KA"],
        &[6.14421235332821e-6, 0.11, 1e-9],
        &[],
        &[],
    );
    let src = BASE.replace("omega ETA_KA ~ 0.30", "omega ETA_KA ~ 0.5477 (sd)");
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, ModelEdit::SeedInits(&fit));
    let params = text.block_lines("parameters");
    assert!(
        params.contains(&"omega ETA_CL ~ 0.00001".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"omega ETA_V  ~ 0.11".to_string()),
        "{params:?}"
    );
    // `(sd)`: the floor is a variance, written as its square root.
    let ka = params
        .iter()
        .find(|l| l.starts_with("omega ETA_KA"))
        .unwrap();
    assert_eq!(ka, "omega ETA_KA ~ 0.00316227766016838 (sd)", "{params:?}");
    assert!(
        params.contains(&"theta TVCL(0.35, 0.001, 10.0)".to_string()),
        "{params:?}"
    );
    // A block's diagonal takes the same floor; its off-diagonal does not.
    let src = BASE.replace(
        "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]",
    );
    let mut fit = fit_with(&[], &[], &["ETA_CL", "ETA_V"], &[1e-9, 0.11], &[], &[]);
    fit.omega[(1, 0)] = 1e-7;
    fit.omega[(0, 1)] = 1e-7;
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, ModelEdit::SeedInits(&fit));
    let params = text.block_lines("parameters");
    assert!(
        params.contains(&"block_omega (ETA_CL, ETA_V) = [0.00001, 0.0000001, 0.11]".to_string()),
        "{params:?}"
    );
}

#[test]
fn seed_inits_writes_each_declaration_on_the_scale_it_was_written_in() {
    let src = BASE.replace("sigma PROP_ERR ~ 0.02 (sd)", "sigma PROP_ERR ~ 0.0004");
    let src = src.replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.3 (sd)");
    let fit = fit_with(&[], &[], &["ETA_CL"], &[0.25], &["PROP_ERR"], &[0.05]);
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, ModelEdit::SeedInits(&fit));
    let params = text.block_lines("parameters");
    // ω 0.25 (variance) written into an `(sd)` declaration is 0.5.
    assert!(
        params.contains(&"omega ETA_CL ~ 0.5 (sd)".to_string()),
        "{params:?}"
    );
    // σ 0.05 (SD) written into a variance declaration is 0.0025 — and not
    // `0.0025000000000000005`, the floating-point square: `num` writes 15
    // significant digits.
    assert!(
        params.contains(&"sigma PROP_ERR ~ 0.0025".to_string()),
        "{params:?}"
    );
}

#[test]
fn seed_inits_leaves_a_fixed_parameter_alone() {
    let src = BASE.replace("theta TVCL(0.2, 0.001, 10.0)", "theta TVCL(0.2, FIX)");
    let fit = fit_with(&["TVCL"], &[9.9], &[], &[], &[], &[]);
    let mut text = ModelText::parse(&src).unwrap();
    apply(&mut text, ModelEdit::SeedInits(&fit));
    assert!(text
        .block_lines("parameters")
        .contains(&"theta TVCL(0.2, FIX)".to_string()));
}

#[test]
fn seed_inits_fills_a_block_omega_from_the_fitted_submatrix() {
    let mut text = base();
    apply(
        &mut text,
        ModelEdit::SetOmegaBlock(vec!["ETA_CL".into(), "ETA_V".into()]),
    );
    let mut fit = fit_with(&[], &[], &["ETA_CL", "ETA_V"], &[0.5, 0.6], &[], &[]);
    fit.omega[(1, 0)] = 0.25;
    fit.omega[(0, 1)] = 0.25;
    apply(&mut text, ModelEdit::SeedInits(&fit));
    assert!(
        text.block_lines("parameters")
            .contains(&"block_omega (ETA_CL, ETA_V) = [0.5, 0.25, 0.6]".to_string()),
        "{:?}",
        text.block_lines("parameters")
    );
}

#[test]
fn seed_inits_fills_a_multi_line_block_omega() {
    let mut text = ModelText::parse(&multi_line_block_base()).unwrap();
    let mut fit = fit_with(&[], &[], &["ETA_CL", "ETA_V"], &[0.5, 0.6], &[], &[]);
    fit.omega[(1, 0)] = 0.25;
    fit.omega[(0, 1)] = 0.25;
    apply(&mut text, ModelEdit::SeedInits(&fit));
    assert!(
        text.block_lines("parameters")
            .contains(&"block_omega (ETA_CL, ETA_V) = [0.5, 0.25, 0.6]".to_string()),
        "{:?}",
        text.block_lines("parameters")
    );
    // The old triangle's continuation lines are gone, not left below the
    // rewritten declaration.
    assert!(!text.render().contains("0.006"), "{}", text.render());
}

// ── Number formatting ──────────────────────────────────────────────────────

#[test]
fn a_non_finite_initial_value_is_refused_rather_than_written() {
    let err = base()
        .apply(ModelEdit::AddIiv {
            param: "CL".into(),
            form: IivForm::Exponential {
                eta: "ETA_X".into(),
                variance: f64::NAN,
            },
        })
        .unwrap_err();
    assert!(
        err.contains("cannot be written as a `.ferx` number"),
        "{err}"
    );
}
