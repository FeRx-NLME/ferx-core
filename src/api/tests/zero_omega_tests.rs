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

#[test]
fn check_covariates_reports_only_the_eta_diagnostic_when_nothing_else_is_missing() {
    // The actual shape of the #989 bug report: an `omega` line was deleted and its
    // `exp(ETA_CL)` left behind, with every real covariate present. The tests above
    // all pair the eta with a genuinely-missing covariate, so this — the common
    // case — is the one that exercises the `plain.is_empty()` early return, and
    // it is the one that must NOT also emit `E_MISSING_COVARIATE` for a name the
    // user never meant as a covariate.
    let model = model_referencing(&["ETA_CL", "WT"]);
    let pop = population_with_covariates(&["WT"]);
    let diags = check_covariates(&model, &pop);

    assert_eq!(
        diags.len(),
        1,
        "expected the eta diagnostic alone: {diags:?}"
    );
    assert_eq!(diags[0].code, "E_ETA_NOT_DECLARED");
    assert_eq!(diags[0].block.as_deref(), Some("parameters"));
}

#[test]
fn undeclared_random_effect_message_agrees_in_number() {
    // Singular and plural are separate arms; a model can easily lose a whole
    // `omega` block and strand several etas at once, so both must read correctly.
    let one = crate::api::validation::undeclared_random_effect_message(&["ETA_CL"]);
    assert!(
        one.contains("references ETA_CL as a random effect"),
        "{one}"
    );
    assert!(one.contains("declaration defines it"), "{one}");

    let many = crate::api::validation::undeclared_random_effect_message(&["ETA_CL", "KAPPA_V"]);
    assert!(
        many.contains("references ETA_CL, KAPPA_V as random effects"),
        "{many}"
    );
    assert!(many.contains("declaration defines them"), "{many}");
    // The worked example is drawn from the first name in both arms.
    assert!(many.contains("omega ETA_CL ~ 0.09"), "{many}");
}

// --- validate_model_file (`ferx check`) -------------------------------------

/// Write `src` to a `.ferx` temp file. The suffix matters: `parse_full_model_file`
/// is reached through the path, and the model stem names the report.
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

/// `MINIMAL_MODEL`-shaped continuous model whose `omega` line was deleted but
/// whose `exp(ETA_CL)` term was left behind.
const FORGOT_OMEGA: &str = "\
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
fn check_without_data_warns_about_an_undeclared_random_effect() {
    // Without a dataset ferx cannot know whether an `ETA_CL` column exists, so this
    // can only be a warning — but it must still be said. Before #989 the parser
    // rejected this model outright; now it parses, and a silent "ok" would ship a
    // model whose eta had quietly become a covariate lookup.
    let f = temp_model(FORGOT_OMEGA);
    let report = validate_model_file(f.path().to_str().expect("utf-8 temp path"), None);

    let hit = report
        .diagnostics
        .iter()
        .find(|d| d.code == "W_ETA_NOT_DECLARED")
        .expect("a stranded eta must be reported without --data");
    assert_eq!(hit.severity, crate::Severity::Warning);
    assert!(hit.message.contains("ETA_CL"), "{}", hit.message);
    assert!(
        hit.suggestion
            .as_deref()
            .is_some_and(|s| s.contains("--data")),
        "the suggestion must point at re-running with data: {:?}",
        hit.suggestion
    );
    // Warnings alone keep the report valid — this is a warning, not a rejection.
    assert!(report.valid, "a warning must not invalidate the report");
}

#[test]
fn check_with_data_escalates_the_undeclared_random_effect_to_an_error() {
    // With a dataset in hand the column is known to be absent, so the same finding
    // is an error and the report is invalid. This also pins the `data_path.is_none()`
    // branch: the warning must NOT be emitted as well, or the user sees the finding
    // twice at two severities.
    let f = temp_model(FORGOT_OMEGA);
    let report = validate_model_file(
        f.path().to_str().expect("utf-8 temp path"),
        Some("data/one_cpt_iv.csv"),
    );

    assert!(
        !report.valid,
        "a stranded eta with data present is an error"
    );
    assert!(report
        .diagnostics
        .iter()
        .any(|d| d.code == "E_ETA_NOT_DECLARED"));
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|d| d.code == "W_ETA_NOT_DECLARED"),
        "the warning must not double up with the error: {:?}",
        report.diagnostics
    );
}

#[test]
fn check_is_clean_for_a_deliberately_fixed_effects_only_model() {
    // The negative control for both tests above, and the whole point of #989: a
    // model that drops the `omega` line AND the `exp(ETA_…)` term is simply valid.
    let report = validate_model_file("examples/one_cpt_iv_pooled.ferx", None);
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|d| d.code == "W_ETA_NOT_DECLARED"),
        "a fixed-effects-only model must not be warned about: {:?}",
        report.diagnostics
    );
    assert!(
        report.valid,
        "unexpected findings: {:?}",
        report.diagnostics
    );
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

// --- imp / impmap / bayes guard (#1007) ---------------------------------------

#[test]
fn check_model_options_rejects_imp_impmap_bayes_without_random_effects() {
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    model.n_eta = 0;
    for (method, name) in [
        (EstimationMethod::Imp, "imp"),
        (EstimationMethod::Impmap, "impmap"),
        (EstimationMethod::Bayes, "bayes"),
    ] {
        let diags = check_model_options(&model, &opts_with_methods(vec![method]));
        let hit = diags
            .iter()
            .find(|d| d.code == "E_METHOD_NO_RANDOM_EFFECTS")
            .unwrap_or_else(|| panic!("{name} at n_eta = 0 must be rejected: {diags:?}"));
        assert_eq!(hit.severity, crate::diagnostics::Severity::Error);
        assert!(
            hit.message.contains(&format!("method = {name}")),
            "message must name the offending method: {}",
            hit.message
        );
        assert!(
            !diags.iter().any(|d| d.code == "E_SAEM_NO_RANDOM_EFFECTS"),
            "{name} must not reuse the stable SAEM code"
        );
    }
}

#[test]
fn check_model_options_rejects_imp_anywhere_in_a_chain() {
    // `methods = [focei, imp]` previously ran the whole FOCEI stage before the
    // IMP stage errored at run time (#1007); it must now fail up front.
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    model.n_eta = 0;
    let diags = check_model_options(
        &model,
        &opts_with_methods(vec![EstimationMethod::FoceI, EstimationMethod::Imp]),
    );
    assert!(diags.iter().any(|d| d.code == "E_METHOD_NO_RANDOM_EFFECTS"));
}

#[test]
fn check_model_options_allows_imp_impmap_bayes_with_random_effects() {
    let model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    assert!(model.n_eta > 0, "fixture must carry random effects");
    for method in [
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
        EstimationMethod::Bayes,
    ] {
        let diags = check_model_options(&model, &opts_with_methods(vec![method]));
        assert!(
            !diags.iter().any(|d| d.code == "E_METHOD_NO_RANDOM_EFFECTS"),
            "{method:?} at n_eta > 0 must not be rejected: {diags:?}"
        );
    }
}

// --- pure-GN warning (#1006) ---------------------------------------------------

#[test]
fn check_model_options_warns_gn_without_random_effects() {
    let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    model.n_eta = 0;
    let diags = check_model_options(&model, &opts_with_methods(vec![EstimationMethod::FoceGn]));
    let hit = diags
        .iter()
        .find(|d| d.code == "W_GN_NO_RANDOM_EFFECTS")
        .expect("gn at n_eta = 0 must warn");
    // Warning, not error: `gn` does reach the optimum from a good start, so a
    // clean `ferx check` run must stay `valid: true` for this model.
    assert_eq!(hit.severity, crate::diagnostics::Severity::Warning);
    assert!(hit.message.contains("gn_hybrid"));
}

#[test]
fn check_model_options_does_not_warn_gn_hybrid_or_mixed_effects_gn() {
    // gn_hybrid's FOCEI polish recovers the optimum, and gn with random effects
    // has the inner EBE loop to absorb a poor start — neither may warn.
    let mut pooled = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    pooled.n_eta = 0;
    let diags = check_model_options(
        &pooled,
        &opts_with_methods(vec![EstimationMethod::FoceGnHybrid]),
    );
    assert!(
        !diags.iter().any(|d| d.code == "W_GN_NO_RANDOM_EFFECTS"),
        "gn_hybrid must not warn at n_eta = 0: {diags:?}"
    );

    let mixed = crate::types::test_helpers::analytical_model(GradientMethod::Fd);
    assert!(mixed.n_eta > 0);
    let diags = check_model_options(&mixed, &opts_with_methods(vec![EstimationMethod::FoceGn]));
    assert!(
        !diags.iter().any(|d| d.code == "W_GN_NO_RANDOM_EFFECTS"),
        "gn at n_eta > 0 must not warn: {diags:?}"
    );
}
