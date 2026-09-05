//! Tier-1 tests for the allometry tool (#1180): what the edit writes, and
//! how the options are read. No fits — `run_allometry` is exercised
//! end-to-end in `tests/covsearch_end_to_end.rs`.

use std::path::Path;

use ferx_core::edit::ModelText;
use ferx_core::prepare_run;

use super::*;
use crate::search::SearchConfig;

const EXAMPLES: &str = "../../examples";
const DATA: &str = "../../data/two_cpt_oral_cov.csv";

fn base(model: &str) -> BaseModel {
    let path = Path::new(EXAMPLES).join(model);
    let prepared = prepare_run(path.to_str().unwrap(), Some(DATA)).expect("base model + data");
    let text = ModelText::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    BaseModel { prepared, text }
}

/// The two-compartment example with no covariate model: CL, Q are
/// clearances and V1, V2 volumes; KA is neither.
const TWO_CPT: &str = "\
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  maxiter = 2
";

fn two_cpt() -> BaseModel {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_cpt.ferx");
    std::fs::write(&path, TWO_CPT).unwrap();
    let prepared = prepare_run(path.to_str().unwrap(), Some(DATA)).expect("base model + data");
    BaseModel {
        prepared,
        text: ModelText::parse(TWO_CPT).unwrap(),
    }
}

#[test]
fn default_scaling_is_075_on_clearances_and_1_on_volumes_fixed() {
    let built = allometric_model(&two_cpt(), &AllometryOptions::default()).unwrap();
    let got: Vec<(&str, f64, bool)> = built
        .scalings
        .iter()
        .map(|s| (s.parameter.as_str(), s.exponent, s.fixed))
        .collect();
    assert_eq!(
        got,
        vec![
            ("CL", 0.75, true),
            ("Q", 0.75, true),
            ("V1", 1.0, true),
            ("V2", 1.0, true)
        ]
    );
    let lines = built.model.block_lines("covariate_model");
    assert_eq!(
        lines,
        vec![
            "CL ~ WT power(center = 70, fix = 0.75)",
            "Q ~ WT power(center = 70, fix = 0.75)",
            "V1 ~ WT power(center = 70, fix = 1.0)",
            "V2 ~ WT power(center = 70, fix = 1.0)",
        ]
    );
    assert!(built.notes.is_empty());
    // The scaled model compiles against the data.
    ferx_core::parser::model_parser::parse_full_model(&built.model.render())
        .expect("the allometric model compiles");
}

#[test]
fn estimated_exponents_declare_a_bounded_theta_per_parameter() {
    let options = AllometryOptions {
        fixed: false,
        parameters: Some(vec!["CL".into(), "V1".into()]),
        ..AllometryOptions::default()
    };
    let built = allometric_model(&two_cpt(), &options).unwrap();
    assert_eq!(
        built.model.block_lines("covariate_model"),
        vec![
            "CL ~ WT power(center = 70) => THETA_CL_WT(0.75, 0.0, 2.0)",
            "V1 ~ WT power(center = 70) => THETA_V1_WT(1.0, 0.0, 2.0)",
        ]
    );
    assert_eq!(built.scalings[0].theta.as_deref(), Some("THETA_CL_WT"));
}

#[test]
fn explicit_parameters_and_exponents_override_the_roles() {
    let options = AllometryOptions {
        covariate: "CRCL".into(),
        reference: 100.0,
        parameters: Some(vec!["KA".into(), "CL".into()]),
        exponents: Some(vec![0.5, 0.6]),
        ..AllometryOptions::default()
    };
    let built = allometric_model(&two_cpt(), &options).unwrap();
    assert_eq!(
        built.model.block_lines("covariate_model"),
        vec![
            "KA ~ CRCL power(center = 100, fix = 0.5)",
            "CL ~ CRCL power(center = 100, fix = 0.6)",
        ]
    );
}

#[test]
fn a_parameter_without_a_role_needs_an_explicit_exponent() {
    let options = AllometryOptions {
        parameters: Some(vec!["KA".into()]),
        ..AllometryOptions::default()
    };
    let e = allometric_model(&two_cpt(), &options).unwrap_err();
    assert!(
        e.contains("`KA` is bound to neither a clearance nor a volume role"),
        "{e}"
    );
}

#[test]
fn unknown_parameter_and_unknown_covariate_are_named() {
    let options = AllometryOptions {
        parameters: Some(vec!["VMAX".into()]),
        exponents: Some(vec![1.0]),
        ..AllometryOptions::default()
    };
    let e = allometric_model(&two_cpt(), &options).unwrap_err();
    assert!(e.contains("`VMAX` is not an individual parameter"), "{e}");

    let options = AllometryOptions {
        covariate: "BMI".into(),
        ..AllometryOptions::default()
    };
    let e = allometric_model(&two_cpt(), &options).unwrap_err();
    assert!(e.contains("`BMI` is not a covariate"), "{e}");
}

#[test]
fn a_parameter_already_scaled_on_the_covariate_is_left_alone() {
    // The shipped example declares CL ~ WT and V1 ~ WT (and CL ~ CRCL).
    let built = allometric_model(
        &base("two_cpt_oral_covmodel.ferx"),
        &AllometryOptions::default(),
    )
    .unwrap();
    let scaled: Vec<&str> = built
        .scalings
        .iter()
        .map(|s| s.parameter.as_str())
        .collect();
    assert_eq!(scaled, vec!["Q", "V2"]);
    assert_eq!(built.notes.len(), 2, "{:?}", built.notes);
    assert!(built.notes[0].contains("`CL` already carries a relation on `WT`"));
    // The original lines are untouched and the new ones appended.
    let lines = built.model.block_lines("covariate_model");
    assert_eq!(lines.len(), 5);
    assert!(lines[0].starts_with("CL ~ WT"));
    assert_eq!(lines[3], "Q ~ WT power(center = 70, fix = 0.75)");
}

#[test]
fn nothing_to_add_is_an_error_not_a_base_refit() {
    let options = AllometryOptions {
        parameters: Some(vec!["CL".into(), "V1".into()]),
        ..AllometryOptions::default()
    };
    let e = allometric_model(&base("two_cpt_oral_covmodel.ferx"), &options).unwrap_err();
    assert!(e.contains("nothing to add"), "{e}");
}

#[test]
fn options_validate_their_shape() {
    let bad = AllometryOptions {
        parameters: Some(vec!["CL".into()]),
        exponents: Some(vec![0.75, 1.0]),
        ..AllometryOptions::default()
    };
    assert!(bad
        .validate()
        .unwrap_err()
        .contains("1 parameter but 2 exponents"));
    let bad = AllometryOptions {
        exponents: Some(vec![0.75]),
        ..AllometryOptions::default()
    };
    assert!(bad.validate().unwrap_err().contains("needs `parameters`"));
    let bad = AllometryOptions {
        reference: 0.0,
        ..AllometryOptions::default()
    };
    assert!(bad.validate().unwrap_err().contains("positive number"));
    let bad = AllometryOptions {
        fixed: false,
        lower: 2.0,
        upper: 1.0,
        ..AllometryOptions::default()
    };
    assert!(bad.validate().unwrap_err().contains("must be below"));
}

fn config(mfl: &str, section: &str) -> Result<AllometryOptions, String> {
    let text = format!(
        "base = \"two_cpt_oral_covmodel.ferx\"\ndata = \"../data/two_cpt_oral_cov.csv\"\n\
         [space]\nmfl = \"{mfl}\"\n{section}"
    );
    let cfg = SearchConfig::from_str(&text, Path::new(EXAMPLES))?;
    AllometryOptions::from_config(&cfg)
}

#[test]
fn from_config_reads_the_allometry_statement_and_section() {
    let o = config("ALLOMETRY(WT, 70)", "").unwrap();
    assert_eq!(o, AllometryOptions::default());

    let o = config(
        "ALLOMETRY(CRCL, 100)",
        "[allometry]\nparameters = [\"CL\"]\nexponents = [0.6]\nfixed = false\nupper = 3.0\n",
    )
    .unwrap();
    assert_eq!(o.covariate, "CRCL");
    assert_eq!(o.reference, 100.0);
    assert_eq!(o.parameters.as_deref(), Some(&["CL".to_string()][..]));
    assert!(!o.fixed);
    assert_eq!(o.upper, 3.0);

    // The reference may be omitted in MFL; the section's (or the default)
    // value then applies.
    let o = config("ALLOMETRY(WT)", "[allometry]\nreference = 75\n").unwrap();
    assert_eq!(o.reference, 75.0);
}

#[test]
fn from_config_refuses_a_space_without_or_with_two_allometry_statements() {
    let e = config("COVARIATE?(CL, WT, pow)", "").unwrap_err();
    assert!(e.contains("needs one `ALLOMETRY(WT, 70)` statement"), "{e}");
    let e = config("ALLOMETRY(WT, 70); ALLOMETRY(CRCL, 100)", "").unwrap_err();
    assert!(e.contains("more than one ALLOMETRY"), "{e}");
    let e = config("ALLOMETRY(WT, 70)", "[allometry]\nfix = true\n").unwrap_err();
    assert!(e.contains("[allometry]") && e.contains("fix"), "{e}");
}
