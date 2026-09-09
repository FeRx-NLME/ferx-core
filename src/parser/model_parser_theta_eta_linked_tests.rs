//! `CompiledModel::theta_eta_linked` — the per-theta Delattre class the mixed
//! BIC penalises on (#1177). Mirrors Pharmpy's `_categorize_parameters`.

use crate::parser::model_parser::parse_model_string;

fn linked(indiv: &str, extra_params: &str, extra_blocks: &str) -> Vec<bool> {
    let text = format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
{extra_params}
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
{indiv}
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
{extra_blocks}"#
    );
    let model = parse_model_string(&text).expect("model parses");
    assert_eq!(model.theta_eta_linked.len(), model.n_theta);
    model.theta_eta_linked
}

#[test]
fn direct_eta_bearing_parameters_link_their_theta() {
    let l = linked(
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA\n",
        "",
        "",
    );
    assert_eq!(l, [true, true, false]);
}

#[test]
fn linkage_is_transitive_through_intermediate_assignments() {
    // pheno: TVV = THETA(2)*WGT*(1+THETA(3)); V = TVV*EXP(ETA(2)) — both
    // thetas in the intermediate are random-class.
    let l = linked(
        "  TVCL_WT = TVCL * WT\n  TVV_WT = TVV * WT * (1 + TVKA)\n  CL = TVCL_WT * exp(ETA_CL)\n  V = TVV_WT * exp(ETA_V)\n  KA = 1.0\n",
        "",
        "[covariates]\n  WT continuous\n",
    );
    assert_eq!(l, [true, true, true]);
}

#[test]
fn a_theta_in_any_eta_bearing_expansion_is_random_class() {
    // TVKA is a scale factor on KA (no eta) *and* on CL (eta): random.
    let l = linked(
        "  CL = TVCL * TVKA * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA\n",
        "",
        "",
    );
    assert_eq!(l, [true, true, true]);
}

#[test]
fn if_conditions_contribute_to_every_parameter_assigned_inside() {
    let l = linked(
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  if (WT > TVKA) {\n    KA = 2.0 * exp(ETA_CL)\n  } else {\n    KA = 1.0\n  }\n",
        "",
        "[covariates]\n  WT continuous\n",
    );
    assert_eq!(l, [true, true, true]);
}

#[test]
fn covariate_model_thetas_are_classified_after_desugaring() {
    let l = linked(
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA\n",
        "",
        "[covariates]\n  WT continuous\n\n[covariate_model]\n  CL ~ WT power(center = 70) => THETA_CL_WT(0.75, 0.01, 5.0)\n  KA ~ WT power(center = 70) => THETA_KA_WT(0.5, 0.01, 5.0)\n",
    );
    // Declared thetas first, then the two covariate-model thetas.
    assert_eq!(l, [true, true, false, true, false]);
}

#[test]
fn iov_kappas_count_as_random_effects() {
    let l = linked(
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(KAPPA_KA)\n",
        "  kappa KAPPA_KA ~ 0.01",
        "",
    );
    assert_eq!(l, [true, true, true]);
}

/// A sum-to-zero level group's dependent level is `NegSum(a, b)` over the
/// half-open `θ[a..b]`; `b` is already one past the group. Reading it as
/// inclusive dragged the *next* declared θ into the group's class, so a θ used
/// only in an η-free parameter was random-class whenever it happened to be
/// declared right after a level block on an η-bearing one.
#[test]
fn negsum_level_group_is_half_open_and_does_not_capture_the_next_theta() {
    use super::{classify_theta_eta_linked, BinOp, Expression, GatherSpec, LevelRule, Statement};
    use std::sync::Arc;

    // θ0, θ1 are the free coefficients of a 3-level sum-to-zero block read by
    // an η-bearing parameter; θ2 is the next declaration, η-free.
    let spec = Arc::new(GatherSpec {
        name: "SITE".into(),
        levels: vec![
            LevelRule::Free(0),
            LevelRule::Free(1),
            LevelRule::NegSum(0, 2),
        ],
    });
    let gather = Expression::ThetaGather {
        spec,
        idx: Box::new(Expression::Literal(1.0)),
    };
    let stmts = vec![
        Statement::Assign(
            "CL".into(),
            Expression::BinOp(Box::new(gather), BinOp::Add, Box::new(Expression::Eta(0))),
        ),
        Statement::Assign("KA".into(), Expression::Theta(2)),
    ];
    assert_eq!(
        classify_theta_eta_linked(&stmts, 3, &[]),
        [true, true, false],
        "θ2 is declared after the level group and never meets an η"
    );
    // A reference level is an empty `NegSum(a, a)`: contributes nothing.
    let spec = Arc::new(GatherSpec {
        name: "SITE".into(),
        levels: vec![LevelRule::NegSum(1, 1), LevelRule::Free(1)],
    });
    let gather = Expression::ThetaGather {
        spec,
        idx: Box::new(Expression::Literal(1.0)),
    };
    let stmts = vec![Statement::Assign(
        "CL".into(),
        Expression::BinOp(Box::new(gather), BinOp::Mul, Box::new(Expression::Eta(0))),
    )];
    assert_eq!(
        classify_theta_eta_linked(&stmts, 3, &[]),
        [false, true, false]
    );
}

/// A `[covariate_nn]`'s weights are registered as θ after the declared ones,
/// and `TYPICAL_PK.CL * exp(ETA_CL)` reads them through `Expression::NnOutput`,
/// which names no θ index itself. The shared hidden-layer weights and the CL
/// head's row + bias are random-class; the V head's row + bias (V carries no
/// η) and the declared `TVKA` stay fixed-class.
#[cfg(feature = "nn")]
#[test]
fn covariate_nn_weights_follow_the_eta_bearing_output_head() {
    let text = r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V]
  layers = [3]
  activation = tanh
  output = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;
    let model = parse_model_string(text).expect("nn model parses");
    // 1 declared θ + 17 weights (2→3→2: W_1 6 + b_1 3 + W_2 6 + b_2 2).
    assert_eq!(model.theta_names.len(), 18);
    assert_eq!(model.covariate_nns[0].weights_offset, 1);
    let mut want = vec![false; 18];
    for i in 1..=9 {
        want[i] = true; // shared hidden layer
    }
    for i in [10, 11, 12, 16] {
        want[i] = true; // CL head: W_2 row 0 and b_2[0]
    }
    assert_eq!(model.theta_eta_linked, want, "{:?}", model.theta_names);
    // And the names agree with the indices: every random-class weight is a
    // layer-1 weight or a layer-2 `_1_*` (CL) entry.
    for (name, &linked) in model.theta_names.iter().zip(&model.theta_eta_linked) {
        let is_cl_head = name.starts_with("W_TYPICAL_PK_2_1_") || name == "B_TYPICAL_PK_2_1";
        let is_hidden = name.starts_with("W_TYPICAL_PK_1_") || name.starts_with("B_TYPICAL_PK_1_");
        assert_eq!(linked, is_cl_head || is_hidden, "{name}");
    }
}

/// `[event_model]` accepts ETA directly, and the shipped TTE examples use it
/// with no `[individual_parameters]` at all (`examples/tte_weibull.ferx`). A θ
/// that meets its η only there is random-class — the same answer as writing
/// the line in `[individual_parameters]`.
#[cfg(feature = "survival")]
#[test]
fn event_model_theta_meeting_its_eta_there_is_random_class() {
    // The compact TTE-only spelling of `examples/tte_weibull.ferx`.
    let compact = r#"
[parameters]
  theta TVSCALE(20.0, 0.1, 500.0)
  theta TVSHAPE(1.5,  0.1, 10.0)
  omega ETA_SCALE ~ 0.04

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE * exp(ETA_SCALE)
  shape  = TVSHAPE
"#;
    let model = parse_model_string(compact).expect("model parses");
    assert_eq!(model.theta_eta_linked, [true, false]);

    // The legacy dummy-PK spelling (`examples/tte_exponential.ferx`), once with
    // the η-bearing line in `[event_model]` and once routed through an
    // `[individual_parameters]` name: the same θ vector, and the same class
    // either way.
    let legacy = |scale_line: &str, indiv_line: &str| -> Vec<bool> {
        let text = format!(
            r#"
[parameters]
  theta TVSCALE(20.0, 0.1, 500.0)
  theta TVSHAPE(1.5,  0.1, 10.0)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  omega ETA_SCALE ~ 0.04
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
{indiv_line}
  CL = DUMMY_CL
  V  = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = weibull
{scale_line}
  shape  = TVSHAPE
"#
        );
        parse_model_string(&text)
            .expect("model parses")
            .theta_eta_linked
    };
    let direct = legacy("  scale = TVSCALE * exp(ETA_SCALE)", "");
    let via_indiv = legacy("  scale = SCALE", "  SCALE = TVSCALE * exp(ETA_SCALE)");
    assert_eq!(direct, [true, false, false, false]);
    assert_eq!(direct, via_indiv);
}

/// `[binary_model]`'s `logit` is the other block that takes ETA directly.
#[cfg(feature = "survival")]
#[test]
fn binary_model_logit_theta_meeting_its_eta_there_is_random_class() {
    let text = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta B0(0.0, -10.0, 10.0)
  theta B_SLOPE(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_B0 ~ 0.1
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[binary_model]
  cmt   = 3
  logit = B0 + ETA_B0 + B_SLOPE * CL
"#;
    let model = parse_model_string(text).expect("model parses");
    // TVCL: via CL. B0: meets ETA_B0 in the logit. B_SLOPE: multiplies CL,
    // which carries ETA_CL — random by the same expansion rule as
    // `[individual_parameters]`. TVKA: η-free.
    assert_eq!(model.theta_eta_linked, [true, true, false, true, true]);
}
