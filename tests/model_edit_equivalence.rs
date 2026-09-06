//! An edited model ≡ the hand-written model it stands for (#1176).
//!
//! `ferx-core::edit` generates candidates by rewriting `.ferx` text. The
//! rendered-text assertions in `src/edit/mod_tests.rs` say *which characters*
//! each edit writes; they cannot say that the result is the model a
//! pharmacometrician would have typed. That is what this file checks, the same
//! way `tests/covariate_model_equivalence.rs` validates the `[covariate_model]`
//! desugar: for every variant, the edited model and its hand-written twin must
//! agree
//!
//! - on `predict()` at a fixed parameter vector, **bit for bit**, and
//! - on the OFV after a couple of outer iterations from that same start.
//!
//! Bit-for-bit rather than to-tolerance is the point. An edit that produced an
//! equivalent-but-differently-associated expression — `A * exp(η) * f` where
//! the author wrote `A * f * exp(η)` — would agree to 1e-12 and still be the
//! wrong text to hand a user, and would move the SAEM mu-reference detector's
//! answer (#619). Only equality catches that.
//!
//! No NONMEM anchor is added here, and none is needed: every model this
//! module can generate is a model the hand-written path could already state,
//! and those forms are NONMEM-anchored already. The anchor for *this* feature
//! is that the two spellings are the same model.

use std::path::Path;

use ferx_core::edit::{
    ErrorForm, ErrorSpecText, IivForm, ModelEdit, ModelText, NewParameter, Relation, RelationTheta,
    SigmaDecl, StructuralSpec,
};
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{CovariateForm, CovariateStat, FitOptions};
use ferx_core::{fit, predict, read_nonmem_csv};

const DATA: &str = "data/two_cpt_oral_cov.csv";

/// The parent model every edit starts from: one-compartment oral, three η, a
/// proportional error model, on a dataset that also carries `WT` and `CRCL`.
const PARENT: &str = "\
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)

  omega ETA_CL ~ 0.15
  omega ETA_V  ~ 0.15
  omega ETA_KA ~ 0.20

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";

/// Apply `edits` to [`PARENT`] and return the rendered candidate.
fn edited(edits: Vec<ModelEdit<'_>>) -> String {
    let mut text = ModelText::parse(PARENT).expect("the parent model must parse");
    for edit in edits {
        text.apply(edit).expect("edit must apply");
    }
    text.render()
}

/// The two sources must be the same model: same θ vector, bit-identical
/// predictions, and the same objective function after two outer iterations.
///
/// Tier-2 throughout — `outer_maxiter = 2` exercises the fit path without
/// running a convergence loop.
fn assert_same_model(generated: &str, hand_written: &str) {
    let gen = parse_full_model(generated)
        .unwrap_or_else(|e| panic!("the generated model must parse: {e}\n---\n{generated}"));
    let hand = parse_full_model(hand_written)
        .unwrap_or_else(|e| panic!("the hand-written model must parse: {e}"));

    assert_eq!(
        gen.model.theta_names, hand.model.theta_names,
        "the two models declare different θ"
    );
    assert_eq!(
        gen.model.eta_names, hand.model.eta_names,
        "the two models declare different η"
    );
    assert_eq!(
        gen.model.default_params.theta, hand.model.default_params.theta,
        "the two models start from different θ values"
    );

    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("dataset must load");
    let gen_pred = predict(&gen.model, &pop, &gen.model.default_params);
    let hand_pred = predict(&hand.model, &pop, &hand.model.default_params);
    assert_eq!(gen_pred.len(), hand_pred.len());
    for (g, h) in gen_pred.iter().zip(&hand_pred) {
        assert_eq!(g.id, h.id);
        assert_eq!(g.time, h.time);
        assert_eq!(g.pred, h.pred, "PRED differs for subject {}", g.id);
    }

    let opts = FitOptions {
        outer_maxiter: 2,
        ..FitOptions::default()
    };
    let gen_fit = fit(&gen.model, &pop, &gen.model.default_params, &opts)
        .expect("short fit of the generated model must not error");
    let hand_fit = fit(&hand.model, &pop, &hand.model.default_params, &opts)
        .expect("short fit of the hand-written model must not error");
    assert_eq!(
        gen_fit.ofv, hand_fit.ofv,
        "the generated model is a different objective function"
    );
    assert_eq!(gen_fit.theta, hand_fit.theta);
}

// ── SetStructural ──────────────────────────────────────────────────────────

#[test]
fn widening_to_two_compartments_is_the_hand_written_two_compartment_model() {
    let generated = edited(vec![ModelEdit::SetStructural(StructuralSpec {
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
                init: 8.0,
                lower: 0.1,
                upper: 100.0,
                iiv: None,
                fixed: false,
            },
            NewParameter {
                name: "V2".into(),
                theta: "TVV2".into(),
                init: 80.0,
                lower: 1.0,
                upper: 500.0,
                iiv: Some(("ETA_V2".into(), 0.08)),
                fixed: false,
            },
        ],
    })]);

    // The θ and η the edit appends land at the end of their blocks, so the
    // hand-written twin declares them in that same order — the two parameter
    // vectors have to line up element for element to be comparable at all.
    let hand = PARENT
        .replace(
            "  KA = TVKA * exp(ETA_KA)\n",
            "  KA = TVKA * exp(ETA_KA)\n  Q = TVQ\n  V2 = TVV2 * exp(ETA_V2)\n",
        )
        .replace(
            "  sigma PROP_ERR ~ 0.04 (sd)\n",
            "  sigma PROP_ERR ~ 0.04 (sd)\n  theta TVQ(8.0, 0.1, 100.0)\n  \
             theta TVV2(80.0, 1.0, 500.0)\n  omega ETA_V2 ~ 0.08\n",
        )
        .replace(
            "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
            "pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)",
        );
    assert_same_model(&generated, &hand);
}

#[test]
fn narrowing_back_to_one_compartment_restores_the_parent() {
    // Widen, then narrow: the coupled θ/η/expression deletions across three
    // blocks must land exactly back on the model we started from. This is the
    // property a stepwise search depends on when it backs out of a step.
    let generated = edited(vec![
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
                    init: 8.0,
                    lower: 0.1,
                    upper: 100.0,
                    iiv: None,
                    fixed: false,
                },
                NewParameter {
                    name: "V2".into(),
                    theta: "TVV2".into(),
                    init: 80.0,
                    lower: 1.0,
                    upper: 500.0,
                    iiv: Some(("ETA_V2".into(), 0.08)),
                    fixed: false,
                },
            ],
        }),
        ModelEdit::SetStructural(StructuralSpec {
            template: "one_cpt_oral".into(),
            bindings: vec![
                ("cl".into(), "CL".into()),
                ("v".into(), "V".into()),
                ("ka".into(), "KA".into()),
            ],
            new_parameters: vec![],
        }),
    ]);
    assert_same_model(&generated, PARENT);
    assert_eq!(
        ModelText::parse(&generated).unwrap().canonical_hash(),
        ModelText::parse(PARENT).unwrap().canonical_hash(),
        "a widen/narrow round-trip must land on the same candidate identity"
    );
}

// ── AddCovariateRelation / DropCovariateRelation ───────────────────────────

fn wt_on_cl() -> Relation {
    Relation {
        parameter: "CL".into(),
        covariate: "WT".into(),
        form: CovariateForm::Power,
        center: Some(CovariateStat::Literal(70.0)),
        fix: None,
        thetas: vec![RelationTheta {
            name: "THETA_CL_WT".into(),
            init: 0.6,
            lower: 0.01,
            upper: 5.0,
        }],
    }
}

#[test]
fn adding_a_covariate_relation_is_the_hand_written_block() {
    let generated = edited(vec![ModelEdit::AddCovariateRelation(wt_on_cl())]);
    let hand = format!(
        "{PARENT}
[covariate_model]
  CL ~ WT power(center = 70) => THETA_CL_WT(0.6, 0.01, 5.0)
"
    );
    assert_same_model(&generated, &hand);
}

#[test]
fn dropping_a_relation_restores_the_model_without_it() {
    let generated = edited(vec![
        ModelEdit::AddCovariateRelation(wt_on_cl()),
        ModelEdit::DropCovariateRelation {
            param: "CL".into(),
            cov: "WT".into(),
        },
    ]);
    assert_same_model(&generated, PARENT);
}

// ── AddIiv / DropIiv ───────────────────────────────────────────────────────

#[test]
fn dropping_an_eta_is_the_hand_written_model_without_it() {
    let generated = edited(vec![ModelEdit::DropIiv { param: "KA".into() }]);
    let hand = PARENT
        .replace("  omega ETA_KA ~ 0.20\n", "")
        .replace("KA = TVKA * exp(ETA_KA)", "KA = TVKA");
    assert_same_model(&generated, &hand);
}

#[test]
fn adding_an_eta_is_the_hand_written_model_with_it() {
    // Drop then re-add: the η comes back at the end of [parameters], so the
    // hand-written twin declares it there too — same model, different file.
    let generated = edited(vec![
        ModelEdit::DropIiv { param: "KA".into() },
        ModelEdit::AddIiv {
            param: "KA".into(),
            form: IivForm::Exponential {
                eta: "ETA_KA".into(),
                variance: 0.20,
            },
        },
    ]);
    let hand = PARENT.replace("  omega ETA_KA ~ 0.20\n", "").replace(
        "  sigma PROP_ERR ~ 0.04 (sd)\n",
        "  sigma PROP_ERR ~ 0.04 (sd)\n  omega ETA_KA ~ 0.2\n",
    );
    assert_same_model(&generated, &hand);
}

// ── SetOmegaBlock ──────────────────────────────────────────────────────────

#[test]
fn blocking_two_etas_is_the_hand_written_block_omega() {
    let generated = edited(vec![ModelEdit::SetOmegaBlock(vec![
        "ETA_CL".into(),
        "ETA_V".into(),
    ])]);
    // 0.1 × sqrt(0.15 × 0.15) = 0.015.
    let hand = PARENT.replace(
        "  omega ETA_CL ~ 0.15\n  omega ETA_V  ~ 0.15\n",
        "  block_omega (ETA_CL, ETA_V) = [0.15, 0.015, 0.15]\n",
    );
    assert_same_model(&generated, &hand);
}

// ── SetErrorModel ──────────────────────────────────────────────────────────

#[test]
fn switching_to_a_combined_error_model_is_the_hand_written_one() {
    let generated = edited(vec![ModelEdit::SetErrorModel(ErrorSpecText {
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
                init: 0.04,
                as_sd: true,
            },
        ],
    })]);
    // The σ of a single-endpoint `[error_model]` are consumed *positionally*
    // from the `[parameters]` declaration order, so `ADD_ERR` has to be
    // declared first — appending it would not parse. This is the case where a
    // wrong edit is most dangerous: for `combined`, a σ order that happens to
    // parse still swaps which of the two is the proportional component.
    let hand = PARENT
        .replace(
            "  sigma PROP_ERR ~ 0.04 (sd)\n",
            "  sigma ADD_ERR ~ 0.1 (sd)\n  sigma PROP_ERR ~ 0.04 (sd)\n",
        )
        .replace(
            "DV ~ proportional(PROP_ERR)",
            "DV ~ combined(ADD_ERR, PROP_ERR)",
        );
    assert_same_model(&generated, &hand);
}

// ── SeedInits ──────────────────────────────────────────────────────────────

#[test]
fn seeding_inits_starts_the_child_where_the_parent_finished() {
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("dataset must load");
    let parent = parse_full_model(PARENT).expect("parent must parse");
    let opts = FitOptions {
        outer_maxiter: 2,
        ..FitOptions::default()
    };
    let parent_fit = fit(&parent.model, &pop, &parent.model.default_params, &opts)
        .expect("short parent fit must not error");

    // The child is a *different* model — it has one η fewer — so its θ vector
    // is not the parent's. Name-keyed carry-over is the only thing that can
    // seed it; a positional copy would put `TVKA`'s estimate on `TVV`.
    let child = edited(vec![
        ModelEdit::DropIiv { param: "KA".into() },
        ModelEdit::SeedInits(&parent_fit),
    ]);
    let child = parse_full_model(&child).expect("the seeded child must parse");

    for (name, init) in child
        .model
        .theta_names
        .iter()
        .zip(&child.model.default_params.theta)
    {
        let k = parent_fit
            .theta_names
            .iter()
            .position(|n| n == name)
            .expect("every child θ is a parent θ here");
        assert!(
            (init - parent_fit.theta[k]).abs() <= 1e-9 * parent_fit.theta[k].abs().max(1.0),
            "`{name}` was seeded with {init}, not the parent's {}",
            parent_fit.theta[k]
        );
    }
    for (i, name) in child.model.eta_names.iter().enumerate() {
        let k = parent_fit
            .eta_names
            .iter()
            .position(|n| n == name)
            .expect("every child η is a parent η here");
        let got = child.model.default_params.omega.matrix[(i, i)];
        let want = parent_fit.omega[(k, k)];
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "`{name}` was seeded with {got}, not the parent's {want}"
        );
    }
}
