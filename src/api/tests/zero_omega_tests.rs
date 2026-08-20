//! Validation-layer tests for fixed-effects-only models (`n_eta = 0`, #989).
//!
//! Two capabilities are pinned here and must not regress independently:
//!   * a model may declare **no** random effects (Ω is 0×0), and
//!   * an identifier that *reads* like a random effect but was never declared is
//!     still caught — the safety net that the removed parse-time
//!     `No omega parameters defined` rejection used to provide.

use super::*;

/// A `Population` carrying exactly the named covariate columns and no subjects.
/// `check_covariates` only reads `covariate_names`, so nothing else matters here.
fn population_with_covariates(names: &[&str]) -> Population {
    Population {
        subjects: Vec::new(),
        covariate_names: names.iter().map(|s| s.to_string()).collect(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn model_referencing(names: &[&str]) -> CompiledModel {
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    model.referenced_covariates = names.iter().map(|s| s.to_string()).collect();
    model
}

// --- is_random_effect_shaped ------------------------------------------------

#[test]
fn random_effect_shaped_accepts_eta_and_kappa_prefixes() {
    for name in ["ETA_CL", "eta_v", "Eta1", "ETA", "KAPPA_CL", "kappa_ka"] {
        assert!(
            crate::api::validation::is_random_effect_shaped(name),
            "{name} should read as a random effect"
        );
    }
}

#[test]
fn random_effect_shaped_rejects_ordinary_covariates() {
    // `ETHNIC` is the near-miss that matters: it shares `ET` with `ETA` but is a
    // real covariate name, so a substring test rather than a prefix test would
    // misfire on it.
    for name in ["WT", "AGE", "CRCL", "ETHNIC", "SEX", "KAP", "TVCL"] {
        assert!(
            !crate::api::validation::is_random_effect_shaped(name),
            "{name} must not read as a random effect"
        );
    }
}

#[test]
fn undeclared_random_effect_names_filters_to_eta_shaped_only() {
    // `ETHNIC` is in the input on purpose: it shares a prefix with `ETA` up to two
    // characters, so a looser prefix test would sweep a real covariate in here.
    let model = model_referencing(&["WT", "ETA_CL", "ETHNIC", "AGE", "KAPPA_V"]);
    assert_eq!(
        crate::api::validation::undeclared_random_effect_names(&model),
        vec!["ETA_CL", "KAPPA_V"]
    );
}

// --- check_covariates split -------------------------------------------------

#[test]
fn check_covariates_reports_undeclared_eta_before_missing_covariate() {
    // Both an eta-shaped and an ordinary name are unresolved. The eta diagnostic
    // must come FIRST, because `fit()` reports `first_error` and the actionable
    // message is the eta one — "covariate not found in data" sends the user to
    // the CSV for what is really a missing `[parameters]` line.
    let model = model_referencing(&["ETA_CL", "WT"]);
    let pop = population_with_covariates(&[]);
    let diags = check_covariates(&model, &pop);

    assert_eq!(
        diags.len(),
        2,
        "expected one eta + one covariate diagnostic"
    );
    assert_eq!(diags[0].code, "E_ETA_NOT_DECLARED");
    assert_eq!(diags[1].code, "E_MISSING_COVARIATE");
    assert!(diags[0].message.contains("ETA_CL"));
    assert!(diags[0].message.contains("omega ETA_CL ~ 0.09"));
    // The generic diagnostic must list only the genuinely-covariate name, so the
    // user is not told to add an `ETA_CL` column.
    assert!(diags[1].message.contains("WT"));
    assert!(
        !diags[1].message.contains("ETA_CL"),
        "eta name must not leak into the covariate message: {}",
        diags[1].message
    );
}

#[test]
fn check_covariates_message_unchanged_when_nothing_is_eta_shaped() {
    // Regression pin: the historical `E_MISSING_COVARIATE` text is byte-for-byte
    // what `fit()` has always returned, so the #989 split must not perturb it for
    // an ordinary missing covariate.
    let model = model_referencing(&["WT"]);
    let pop = population_with_covariates(&["AGE"]);
    let diags = check_covariates(&model, &pop);

    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code, "E_MISSING_COVARIATE");
    assert_eq!(
        diags[0].message,
        "Model references covariate(s) not found in data (case-sensitive): WT. \
         Available covariate columns: AGE."
    );
}

#[test]
fn check_covariates_silent_when_an_eta_named_column_exists() {
    // A model may legitimately read a NONMEM-exported `ETA_CL` column as a
    // covariate. The heuristic must never fire when the name actually resolves.
    let model = model_referencing(&["ETA_CL"]);
    let pop = population_with_covariates(&["ETA_CL"]);
    assert!(check_covariates(&model, &pop).is_empty());
}

// --- SAEM guard -------------------------------------------------------------

fn opts_with_methods(methods: Vec<EstimationMethod>) -> FitOptions {
    FitOptions {
        methods,
        ..Default::default()
    }
}

#[test]
fn check_model_options_rejects_saem_without_random_effects() {
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    model.n_eta = 0;
    let diags = check_model_options(&model, &opts_with_methods(vec![EstimationMethod::Saem]));
    let hit = diags
        .iter()
        .find(|d| d.code == "E_SAEM_NO_RANDOM_EFFECTS")
        .expect("saem at n_eta = 0 must be rejected");
    assert!(hit.message.contains("at least one random effect"));
}

#[test]
fn check_model_options_rejects_saem_anywhere_in_a_chain() {
    // The guard must fire on a chained fit too, up front — otherwise
    // `methods = [saem, focei]` runs the whole SAEM stage before failing.
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    model.n_eta = 0;
    let diags = check_model_options(
        &model,
        &opts_with_methods(vec![EstimationMethod::Saem, EstimationMethod::FoceI]),
    );
    assert!(diags.iter().any(|d| d.code == "E_SAEM_NO_RANDOM_EFFECTS"));
}

#[test]
fn check_model_options_allows_saem_with_random_effects() {
    let model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    assert!(model.n_eta > 0, "fixture must carry random effects");
    let diags = check_model_options(&model, &opts_with_methods(vec![EstimationMethod::Saem]));
    assert!(!diags.iter().any(|d| d.code == "E_SAEM_NO_RANDOM_EFFECTS"));
}

#[test]
fn check_model_options_allows_focei_without_random_effects() {
    // The estimators that *do* reduce exactly at n_eta = 0 must stay unguarded.
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    model.n_eta = 0;
    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::Laplace,
    ] {
        let diags = check_model_options(&model, &opts_with_methods(vec![method]));
        assert!(
            !diags.iter().any(|d| d.code == "E_SAEM_NO_RANDOM_EFFECTS"),
            "{method:?} must not trip the SAEM guard"
        );
    }
}
