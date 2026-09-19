//! A compartment-free (`$PRED`-equivalent, #811) model applies no dose, so a dose
//! record in its dataset is dropped from the prediction without a word from the
//! model-blind reader. #1443 makes that visible once (`W_COMPARTMENT_FREE_DOSES`)
//! and keeps the coded-`RATE` gate from asking for a `D{n}` / `R{n}` parameter the
//! model cannot consume.
//!
//! The model is *parsed*, not hand-built: `is_algebraic()` is a marker the parser
//! sets from the `[structural_model]` shape, and a hand-built fixture would pin the
//! predicate against whatever the test author believed the marker to be.
use crate::api::validation::{
    check_dose_compartments, check_model_data, check_model_data_warnings, check_modeled_dose_rates,
};
use crate::parser::model_parser::parse_model_string;
use crate::types::{CompiledModel, DoseEvent, Population, RateMode, Subject};
use std::collections::HashMap;

/// An Emax time-course with no compartments — the #811 base case.
fn compartment_free_model() -> CompiledModel {
    let model = parse_model_string(
        "[parameters]\n\
         \x20 theta TVE0(10.0, 0.1, 100.0)\n\
         \x20 theta TVEMAX(5.0, 0.1, 100.0)\n\
         \x20 sigma PROP ~ 0.02 (sd)\n\n\
         [individual_parameters]\n\
         \x20 E0 = TVE0\n\
         \x20 EMAX = TVEMAX\n\n\
         [structural_model]\n\
         \x20 y = E0 - EMAX * TIME / (2.0 + TIME)\n\n\
         [error_model]\n\
         \x20 DV ~ proportional(PROP)\n",
    )
    .expect("the #811 base case parses");
    assert!(model.is_algebraic(), "fixture must be compartment-free");
    model
}

/// One subject per dose list; each subject also scores one observation so the
/// population is otherwise ordinary.
fn population(dose_lists: Vec<Vec<DoseEvent>>) -> Population {
    Population {
        subjects: dose_lists
            .into_iter()
            .enumerate()
            .map(|(i, doses)| Subject {
                id: format!("{}", i + 1),
                doses,
                obs_times: vec![1.0],
                obs_raw_times: Vec::new(),
                observations: vec![8.0],
                obs_cmts: vec![1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                reset_covariates: Vec::new(),
                cens: vec![0],
                occasions: Vec::new(),
                obs_l2: Vec::new(),
                dose_occasions: Vec::new(),
                reset_occasions: Vec::new(),
                fremtype: Vec::new(),
                obs_records: vec![],
            })
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// A plain bolus into `cmt`. The fixtures deliberately use compartments the
/// placeholder `one_cpt_iv` does **not** have (`cmt = 2`) as well as its one real
/// one: `cmt = 1` is the single value on which the placeholder's range check and
/// "no compartments at all" happen to agree, so a fixture built only from it
/// cannot see a validator that still reads the placeholder (#1454 review).
fn bolus(cmt: usize) -> DoseEvent {
    DoseEvent::new(0.0, 100.0, cmt, 0.0, false, 0.0)
}

/// A coded `RATE=-1` row into `cmt`. `cmt = 0` is what an infusion into a
/// compartment the placeholder cannot infuse into looks like to
/// `check_dose_compartments`.
fn coded_rate(cmt: usize) -> DoseEvent {
    DoseEvent::modeled(0.0, 100.0, cmt, false, 0.0, RateMode::ModeledRate)
}

fn fatal_codes(model: &CompiledModel, pop: &Population) -> Vec<String> {
    check_model_data(model, pop)
        .into_iter()
        .filter(|d| d.is_error())
        .map(|d| d.code)
        .collect()
}

fn warning_codes(model: &CompiledModel, pop: &Population) -> Vec<String> {
    check_model_data_warnings(model, pop, &model.default_params)
        .into_iter()
        .filter(|d| !d.is_error())
        .map(|d| d.code)
        .collect()
}

/// The regression: two subjects with three dose events between them — into
/// compartments 1 **and 2** — are reported once, with both counts, and the message
/// names the model class rather than the data. A third, dose-free subject must not
/// be counted (it is what makes "2 subject(s)" differ from the population size).
#[test]
fn compartment_free_model_with_dose_records_warns_once_with_counts() {
    let model = compartment_free_model();
    let pop = population(vec![vec![bolus(1), bolus(2)], vec![bolus(2)], vec![]]);
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    let hits: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "W_COMPARTMENT_FREE_DOSES")
        .collect();
    assert_eq!(hits.len(), 1, "one model-wide warning, got {diags:?}");
    let msg = &hits[0].message;
    assert!(
        msg.contains("3 dose event(s) (after ADDL expansion) across 2 subject(s)"),
        "counts dose events and dosed subjects, not all subjects: {msg}"
    );
    assert!(
        msg.contains("compartment-free") && msg.contains("no dose is applied"),
        "names the model class and what happens to the rows: {msg}"
    );
    // And the same rows draw **no** fatal diagnostic of any kind: the `CMT=2` bolus
    // used to be `E_DOSE_CMT_OUT_OF_RANGE` "…the analytical `one_cpt_iv` model has
    // only 1 compartment(s)" from `check_dose_compartments` reading the placeholder,
    // which `fit()` consumes through `first_error` *before* the warnings loop — so
    // the user got the analytical error and never saw this warning. The whole list,
    // not a code prefix: a prefix filter is how the first version of this test
    // missed that validator.
    let fatal = fatal_codes(&model, &pop);
    assert!(
        fatal.is_empty(),
        "a compartment-free model has no dose to reject: {fatal:?}"
    );
}

/// `fit()` pushes the message without its `W_` code, so the warning's category on
/// the R / YAML side comes from `classify_warning`'s phrase match. Pinned the way
/// `ss_absolute_time_tests` pins its neighbour: the category, and that removing the
/// routing phrase changes it — otherwise the assertion holds for any message the
/// `DataQuality` arm happens to accept.
#[test]
fn the_message_reaches_the_data_quality_category_through_its_phrase() {
    use crate::types::{classify_warning, WarningCode};
    let model = compartment_free_model();
    let pop = population(vec![vec![bolus(1)]]);
    let msg = check_model_data_warnings(&model, &pop, &model.default_params)
        .into_iter()
        .find(|d| d.code == "W_COMPARTMENT_FREE_DOSES")
        .expect("must warn")
        .message;
    assert_eq!(
        classify_warning(&msg).category,
        WarningCode::DataQuality,
        "must classify as data_quality: {msg}"
    );
    let defanged = msg.replace("no dose is applied", "doses are not used");
    assert_ne!(
        classify_warning(&defanged).category,
        WarningCode::DataQuality,
        "`no dose is applied` is what routes this message; if it classifies without the \
         phrase, this test is not pinning the route it claims to"
    );
}

/// The normal shape of this model class — a dose-free dataset — draws nothing.
/// Without this control the warning could fire on the mere presence of the
/// compartment-free marker and the test above would still pass.
#[test]
fn compartment_free_model_without_dose_records_is_silent() {
    let model = compartment_free_model();
    let pop = population(vec![vec![], vec![]]);
    assert!(
        !warning_codes(&model, &pop)
            .iter()
            .any(|c| c == "W_COMPARTMENT_FREE_DOSES"),
        "no dose records, no warning"
    );
}

/// A coded `RATE=-1` row used to be rejected with `E_MODELED_RATE_NO_PARAM`
/// ("requires an `R1` parameter … but none is declared") — advice for a parameter
/// a compartment-free model cannot consume, and wrong on its face when the user
/// *had* declared `R1` (#1442 made that parse). The coded row is now covered by
/// the model-wide warning above and nothing else.
///
/// Two validators, two rows, each assertion named for the one it kills:
/// - `cmt = 1` → `check_modeled_dose_rates`; dies when its early return is removed.
/// - `cmt = 0` → `check_dose_compartments` (`E_DOSE_CMT_NOT_INFUSABLE` "…the
///   analytical `one_cpt_iv` model can only infuse into…"); dies when *its* early
///   return is removed. Asserted on the whole fatal list so a third validator that
///   starts reading the placeholder cannot hide behind a code-prefix filter.
#[test]
fn compartment_free_model_coded_rate_is_not_a_missing_parameter_error() {
    let model = compartment_free_model();
    let pop = population(vec![vec![coded_rate(1), coded_rate(0)]]);
    let modeled = check_modeled_dose_rates(&model, &pop);
    assert!(
        modeled.is_empty(),
        "no D{{n}}/R{{n}} advice for a model that applies no dose, got {modeled:?}"
    );
    let routing = check_dose_compartments(&model, &pop);
    assert!(
        routing.is_empty(),
        "no placeholder-topology routing check for a model that applies no dose, got {routing:?}"
    );
    let fatal = fatal_codes(&model, &pop);
    assert!(fatal.is_empty(), "got {fatal:?}");
    assert!(
        warning_codes(&model, &pop)
            .iter()
            .any(|c| c == "W_COMPARTMENT_FREE_DOSES"),
        "the coded rows are still reported, as ignored doses"
    );
}

/// Both gates are keyed on `is_algebraic()`, not on the absence of a dose-attribute
/// slot or of an `ode_spec`: an analytical `pk(...)` model with the same rows keeps
/// `E_MODELED_RATE_NO_PARAM` for the `cmt = 1` coded row and
/// `E_DOSE_CMT_OUT_OF_RANGE` for a `cmt = 2` bolus, so neither early return can be
/// widened into "no ODE spec ⇒ skip".
#[test]
fn analytical_model_coded_rate_without_parameter_still_errors() {
    let model = parse_model_string(
        "[parameters]\n\
         \x20 theta TVCL(1.0, 0.01, 10.0)\n\
         \x20 theta TVV(10.0, 0.1, 100.0)\n\
         \x20 sigma PROP ~ 0.02 (sd)\n\n\
         [individual_parameters]\n\
         \x20 CL = TVCL\n\
         \x20 V = TVV\n\n\
         [structural_model]\n\
         \x20 pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n\
         \x20 DV ~ proportional(PROP)\n",
    )
    .expect("parses");
    assert!(!model.is_algebraic());
    let pop = population(vec![vec![coded_rate(1), bolus(2)]]);
    let codes: Vec<String> = check_modeled_dose_rates(&model, &pop)
        .into_iter()
        .map(|d| d.code)
        .collect();
    assert_eq!(codes, vec!["E_MODELED_RATE_NO_PARAM".to_string()]);
    let routing: Vec<String> = check_dose_compartments(&model, &pop)
        .into_iter()
        .map(|d| d.code)
        .collect();
    assert_eq!(routing, vec!["E_DOSE_CMT_OUT_OF_RANGE".to_string()]);
    assert!(
        !warning_codes(&model, &pop)
            .iter()
            .any(|c| c == "W_COMPARTMENT_FREE_DOSES"),
        "the ignored-doses warning is for compartment-free models only"
    );
}
