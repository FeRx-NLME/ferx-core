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
