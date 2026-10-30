//! Tier-1 tests for #1788: the refusals `fit()` used to raise without a code — the
//! inline checks of `fit_unstamped` and the file entry points' — carry one, and their
//! text is what it was.
//!
//! One test per code, through `fit()` (or the file entry point that raises it),
//! asserting `code()`, `block()` and `to_string()`. The mutation for each: revert the
//! site to `EngineError::uncoded`, and its `code()` turns `None`. Where `ferx check`
//! reports the same refusal, a second assertion holds it to the same code, so the two
//! cannot disagree.
//!
//! Tier 1: `fit()` refuses every input here before its first iteration.

use super::*;
use crate::diagnostics::EngineError;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, FitOptions, Population, Subject};

fn population_of(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Three subjects, a bolus into compartment 1 and three Gaussian observations each.
fn gaussian_population() -> Population {
    population_of(
        (1..=3)
            .map(|i| Subject {
                id: format!("{i}"),
                obs_times: vec![1.0, 4.0, 12.0],
                observations: vec![1.0, 0.8, 0.3],
                obs_cmts: vec![1; 3],
                cens: vec![0; 3],
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                ..Default::default()
            })
            .collect(),
    )
}

const ONE_CPT_IV: &str = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  \
    theta TVV(20.0, 0.001, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n";

/// Two classes; `omega(2)` gives class 2 its own Ω (a per-class override).
const MIXTURE: &str = "[parameters]\n  theta TVCL1(1.0, 0.01, 100.0)\n  \
    theta TVCL2(3.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
    theta P1(0.5, 0.001, 0.999)\n  omega ETA_CL ~ 0.09\n  sigma EPS ~ 0.04\n\
    [mixture]\n  nsub = 2\n  p(1) = P1\n\
    [individual_parameters]\n  \
    CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(EPS)\n";

fn fit_refusal(
    model: &CompiledModel,
    pop: &Population,
    params: &ModelParameters,
    opts: &FitOptions,
) -> EngineError {
    fit(model, pop, params, opts)
        .err()
        .expect("the fixture is refused before the fit")
}

/// The code, the block and the text of one refusal, named by `label` on failure.
fn assert_coded(err: &EngineError, label: &str, code: &str, block: Option<&str>, text: &str) {
    assert_eq!(err.code(), Some(code), "{label}: {err}");
    assert_eq!(err.block(), block, "{label}: {err}");
    assert_eq!(err.to_string(), text, "{label}");
}

fn write_model(text: &str) -> tempfile::NamedTempFile {
    let f = tempfile::Builder::new().suffix(".ferx").tempfile().unwrap();
    std::fs::write(f.path(), text).unwrap();
    f
}

/// The codes `ferx check` (no data) reports for a model file.
fn check_codes(text: &str) -> Vec<String> {
    let f = write_model(text);
    crate::api::validate_model_file(f.path().to_str().unwrap(), None)
        .diagnostics
        .into_iter()
        .map(|d| d.code)
        .collect()
}

// ── [mixture] ────────────────────────────────────────────────────────────────

/// `init_params` with no per-class Ω/Σ is `E_PARAM_SHAPE`, the code every entry point
/// gives parameters that do not fit the model; the method refusals are
/// `E_MIXTURE_METHOD` on `fit_options`, and `ferx check` reports those too.
#[test]
fn mixture_refusals_are_coded() {
    let model = parse_model_string(MIXTURE).expect("parse");
    let pop = gaussian_population();
    let foce = FitOptions {
        method: EstimationMethod::Foce,
        ..FitOptions::default()
    };

    let mut no_classes = model.default_params.clone();
    no_classes.mixture = None;
    assert_coded(
        &fit_refusal(&model, &pop, &no_classes, &foce),
        "mixture: None",
        "E_PARAM_SHAPE",
        None,
        "mixture model ([mixture] block, #977) requires per-class Omega/Sigma in \
         `init_params.mixture`; pass the parsed model's `default_params` (or rebuild \
         them from the model file) rather than params with `mixture: None`",
    );

    let quadrature = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        n_agq: 3,
        ..FitOptions::default()
    };
    assert_coded(
        &fit_refusal(&model, &pop, &model.default_params, &quadrature),
        "n_agq > 1",
        "E_MIXTURE_METHOD",
        Some("fit_options"),
        "mixture models do not yet support FOCEI quadrature (n_agq > 1); use n_agq = 1 \
         for mixture FOCEI. The mixture objective does not evaluate the quadrature grid.",
    );

    let laplace = FitOptions {
        method: EstimationMethod::Laplace,
        ..FitOptions::default()
    };
    assert_coded(
        &fit_refusal(&model, &pop, &model.default_params, &laplace),
        "laplace",
        "E_MIXTURE_METHOD",
        Some("fit_options"),
        "mixture models (#977) currently support FOCE / FOCEI / SAEM / Bayes / IMP / \
         IMPMAP; the LAPLACE method is not yet wired for mixtures (#985)",
    );

    let overrides = parse_model_string(
        &MIXTURE.replace("  p(1) = P1\n", "  p(1) = P1\n  omega(2) ETA_CL ~ 0.4\n"),
    )
    .expect("parse with an omega(2) override");
    let bayes = FitOptions {
        method: EstimationMethod::Bayes,
        ..FitOptions::default()
    };
    assert_coded(
        &fit_refusal(&overrides, &pop, &overrides.default_params, &bayes),
        "bayes + omega(2)",
        "E_MIXTURE_METHOD",
        Some("fit_options"),
        "Bayesian estimation (method = bayes) does not yet support per-class \
         Omega/Sigma overrides (omega(k)/sigma(k)) in a mixture; Omega/Sigma are \
         shared across classes. Fit with FOCE/FOCEI, or drop the per-class \
         overrides (#985).",
    );

    // `ferx check` on the model file: the same code for the same option, and none
    // for the accepted one.
    let with = |opts: &str| format!("{MIXTURE}[fit_options]\n  {opts}\n");
    assert!(
        check_codes(&with("method = laplace")).contains(&"E_MIXTURE_METHOD".to_string()),
        "ferx check misses the mixture method"
    );
    assert!(
        !check_codes(&with("method = foce")).contains(&"E_MIXTURE_METHOD".to_string()),
        "ferx check refuses an accepted mixture method"
    );
}

// ── LTBS on a hand-built model ───────────────────────────────────────────────

/// The parser refuses all three for a model file, so only a `CompiledModel` built in
/// code reaches them. One fixture, an additive-error model with `log_transform` set,
/// and one field changed per cell. `check_model_options` reports the same code.
#[test]
fn ltbs_refusals_are_coded() {
    let pop = gaussian_population();
    let ltbs = || {
        let mut m = parse_model_string(&ONE_CPT_IV.replace("proportional(PROP)", "additive(PROP)"))
            .expect("parse");
        assert!(matches!(m.error_model, ErrorModel::Additive));
        m.log_transform = true;
        m
    };
    let mut proportional = ltbs();
    proportional.error_model = ErrorModel::Proportional;
    let mut per_cmt = ltbs();
    per_cmt.error_spec = ErrorSpec::PerCmt(std::collections::HashMap::new());
    let mut sde = ltbs();
    sde.diffusion_theta_start = Some(0);
    for (label, model, want) in [
        (
            "proportional",
            proportional,
            "LTBS (`log_transform = true`) requires `error_model = Additive`; \
             proportional/combined error on the log scale is not supported",
        ),
        (
            "per-CMT error",
            per_cmt,
            "LTBS (`log_transform = true`) is not supported with per-CMT \
             (`ErrorSpec::PerCmt`) error models",
        ),
        (
            "SDE",
            sde,
            "LTBS (`log_transform = true`) is not supported with an SDE \
             model (`diffusion_theta_start = Some(_)`)",
        ),
    ] {
        let err = fit_refusal(&model, &pop, &model.default_params, &FitOptions::default());
        assert_coded(&err, label, "E_LTBS_UNSUPPORTED", Some("error_model"), want);
        assert!(
            crate::api::check_model_options(&model, &FitOptions::default())
                .iter()
                .any(|d| d.code == "E_LTBS_UNSUPPORTED"),
            "{label}: check_model_options misses it"
        );
    }
    assert!(
        crate::api::validation::check_ltbs_model(&ltbs()).is_empty(),
        "LTBS on an additive single-endpoint model is accepted"
    );
}

// ── iiv_on_ruv ───────────────────────────────────────────────────────────────

fn iiv_on_ruv_model() -> String {
    std::fs::read_to_string("examples/iiv_on_ruv.ferx").expect("example")
}

/// A shared η is `E_IIV_ON_RUV_SHARED_ETA` on `error_model`; a non-interaction method
/// is `E_IIV_ON_RUV_METHOD` on `fit_options`. Both are reported by `ferx check`, which
/// before #1788 called both files valid.
#[test]
fn iiv_on_ruv_refusals_are_coded() {
    let pop = gaussian_population();
    let src = iiv_on_ruv_model();
    let focei = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        ..FitOptions::default()
    };

    let shared_src = src.replace("iiv_on_ruv = ETA_RUV", "iiv_on_ruv = ETA_V");
    let shared = parse_model_string(&shared_src).expect("parse a shared iiv_on_ruv eta");
    assert_coded(
        &fit_refusal(&shared, &pop, &shared.default_params, &focei),
        "shared eta",
        "E_IIV_ON_RUV_SHARED_ETA",
        Some("error_model"),
        "[error_model] iiv_on_ruv = ETA_V: this eta is also used in \
         [individual_parameters]; the residual-error random effect must be a \
         dedicated omega not shared with a structural parameter",
    );
    assert!(check_codes(&shared_src).contains(&"E_IIV_ON_RUV_SHARED_ETA".to_string()));

    let model = parse_model_string(&src).expect("parse");
    let foce = FitOptions {
        method: EstimationMethod::Foce,
        interaction: false,
        ..FitOptions::default()
    };
    assert_coded(
        &fit_refusal(&model, &pop, &model.default_params, &foce),
        "foce",
        "E_IIV_ON_RUV_METHOD",
        Some("fit_options"),
        "IIV on residual error (iiv_on_ruv) requires an interaction or \
         Monte-Carlo method: use method = focei, imp, impmap, or saem (got Foce \
         with interaction = false). NONMEM `Y = IPRED + EPS*EXP(ETA)` is an \
         INTERACTION model.",
    );
    let foce_src = src.replace("method     = focei", "method     = foce");
    assert!(check_codes(&foce_src).contains(&"E_IIV_ON_RUV_METHOD".to_string()));
    // The example as written (focei, a dedicated η) is neither.
    let clean = check_codes(&src);
    assert!(
        !clean.iter().any(|c| c.starts_with("E_IIV_ON_RUV")),
        "{clean:?}"
    );
}

// ── AGQ grid under IOV, single occasion ─────────────────────────────────────

fn iov_model() -> CompiledModel {
    parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.04\n  sigma PROP ~ 0.1 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ proportional(PROP)\n",
    )
    .expect("parse an IOV model")
}

/// Six occasions per subject: `d = n_eta + 6·n_kappa = 7`, so `n_agq = 6` needs
/// `6^7 = 279,936` nodes — over the cap, while the η-only grid (`6^1`) is not. The code
/// is the one `check_model_options` gives the η-only grid.
#[test]
fn agq_grid_under_iov_is_coded() {
    let model = iov_model();
    let mut pop = gaussian_population();
    for s in &mut pop.subjects {
        s.obs_times = (1..=6).map(f64::from).collect();
        s.observations = vec![1.0; 6];
        s.obs_cmts = vec![1; 6];
        s.cens = vec![0; 6];
        s.occasions = (1..=6).collect();
        s.dose_occasions = vec![1];
    }
    let laplace = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: 6,
        ..FitOptions::default()
    };
    let err = fit_refusal(&model, &pop, &model.default_params, &laplace);
    assert_eq!(err.code(), Some("E_AGQ_GRID_TOO_LARGE"), "{err}");
    assert_eq!(err.block(), Some("fit_options"), "{err}");
    assert!(
        err.to_string().starts_with(
            "method = laplace with n_agq = 6 needs a 6^7 = 279936-node tensor grid per \
             subject per iteration — over the 100000 limit. Under IOV"
        ),
        "{err}"
    );
    // The η-only grid passes `check_model_options`, so the population is what tripped it.
    assert!(!crate::api::check_model_options(&model, &laplace)
        .iter()
        .any(|d| d.code == "E_AGQ_GRID_TOO_LARGE"));
}

/// A derived `iov_occasion` rule that leaves every subject one occasion is
/// `E_IOV_SINGLE_OCCASION` on `fit_options`.
#[test]
fn single_derived_occasion_is_coded() {
    let model = iov_model();
    let opts = FitOptions {
        iov_occasion: IovOccasionRule::TimeWindows(vec![100.0]),
        ..FitOptions::default()
    };
    assert_coded(
        &fit_refusal(&model, &gaussian_population(), &model.default_params, &opts),
        "time(100)",
        "E_IOV_SINGLE_OCCASION",
        Some("fit_options"),
        "The `iov_occasion` rule produced only a single occasion for every subject, \
         so the per-occasion kappa (IOV) cannot be separated from between-subject \
         variability (it is unidentifiable). Use `iov_occasion = time(...)` with \
         breakpoints inside the sampling window, `iov_occasion = dose` on a \
         multiple-dose design, or supply an `iov_column`.",
    );
}

/// Review r1: the occasions a derived `iov_occasion` rule gives are not on the population
/// the caller passes in, so the AGQ cap must count them after the rule has run. Here the
/// population carries no labels at all, and `time(1.5, …, 5.5)` cuts the six observations
/// into six windows: `d = 1 + 6 = 7`, `6^7` nodes. Counted on the unlabelled input it would
/// be `6^1` and pass. A second population carries one stale column label per record,
/// which the rule replaces. `fit()` and the helper `ferx check --data` calls agree.
///
/// Mutation — count `group_count` on `population` instead of the derived one → both
/// cells are accepted (or refused later with another code) and the test names the cell.
#[test]
fn agq_grid_under_iov_counts_derived_occasions() {
    let model = iov_model();
    let six_obs = |labels: Vec<u32>, dose_labels: Vec<u32>| {
        let mut pop = gaussian_population();
        for s in &mut pop.subjects {
            s.obs_times = (1..=6).map(f64::from).collect();
            s.observations = vec![1.0; 6];
            s.obs_cmts = vec![1; 6];
            s.cens = vec![0; 6];
            s.occasions = labels.clone();
            s.dose_occasions = dose_labels.clone();
        }
        pop
    };
    let laplace = FitOptions {
        method: EstimationMethod::Laplace,
        n_agq: 6,
        iov_occasion: IovOccasionRule::TimeWindows(vec![1.5, 2.5, 3.5, 4.5, 5.5]),
        ..FitOptions::default()
    };
    for (label, pop) in [
        ("unlabelled", six_obs(Vec::new(), Vec::new())),
        ("stale column labels", six_obs(vec![1; 6], vec![1])),
    ] {
        // The helper first: a regression here otherwise lets `fit()` start the
        // 279936-node fit and the test hangs instead of failing.
        assert!(
            crate::api::validation::check_agq_grid_iov(&model, &pop, &laplace)
                .iter()
                .any(|d| d.code == "E_AGQ_GRID_TOO_LARGE"),
            "{label}: the ferx check helper misses it"
        );
        let err = fit_refusal(&model, &pop, &model.default_params, &laplace);
        assert_eq!(err.code(), Some("E_AGQ_GRID_TOO_LARGE"), "{label}: {err}");
        assert!(
            err.to_string()
                .starts_with("method = laplace with n_agq = 6 needs a 6^7 = 279936-node"),
            "{label}: {err}"
        );
    }
    // The control: a rule giving two windows (`d = 3`, `6^3 = 216`) is under the cap.
    let two = FitOptions {
        iov_occasion: IovOccasionRule::TimeWindows(vec![3.5]),
        ..laplace.clone()
    };
    assert!(crate::api::validation::check_agq_grid_iov(
        &model,
        &six_obs(Vec::new(), Vec::new()),
        &two
    )
    .is_empty());
}

/// Review r1: `ferx check --data` reports the single-occasion refusal `fit()` raises, with
/// the same code, block and message, from the model file's own `iov_occasion` rule. A
/// rule whose breakpoint lies inside the sampling window passes both.
///
/// Mutation — drop `check_single_derived_occasion` from `check_model_data_rule_at` → the
/// check side reports nothing and `fit()` accepts the model, so both halves go red.
#[test]
fn ferx_check_reports_a_single_derived_occasion() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("d.csv");
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT\n");
    for id in 1..=3 {
        csv.push_str(&format!("{id},0,0,100,1,1\n"));
        for t in [1, 4, 12] {
            csv.push_str(&format!("{id},{t},1.0,0,0,1\n"));
        }
    }
    std::fs::write(&data, csv).unwrap();
    let model_src = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  \
        theta TVV(20.0, 0.001, 500.0)\n  omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.04\n  \
        sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  \
        V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
        DV ~ proportional(PROP)\n[fit_options]\n  iov_occasion = time(RULE)\n";
    let check = |rule: &str| {
        let f = write_model(&model_src.replace("RULE", rule));
        crate::api::validate_model_file(f.path().to_str().unwrap(), data.to_str())
    };

    let report = check("100");
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.code == "E_IOV_SINGLE_OCCASION")
        .unwrap_or_else(|| panic!("ferx check misses it: {:?}", report.diagnostics));
    assert!(!report.valid);
    assert_eq!(d.block.as_deref(), Some("fit_options"));

    let model = iov_model();
    let opts = FitOptions {
        iov_occasion: IovOccasionRule::TimeWindows(vec![100.0]),
        ..FitOptions::default()
    };
    let err = fit_refusal(&model, &gaussian_population(), &model.default_params, &opts);
    assert_eq!(
        err.to_string(),
        d.message,
        "fit() and ferx check say the same"
    );

    let inside = check("6");
    assert!(
        !inside
            .diagnostics
            .iter()
            .any(|d| d.code == "E_IOV_SINGLE_OCCASION"),
        "{:?}",
        inside.diagnostics
    );
}

// ── endpoint records ─────────────────────────────────────────────────────────

/// A repeated-event subject whose records go back in time is `E_RTTE_RECORDS` on
/// `event_model`, from `fit()` and from `ferx check --data`'s rule list.
#[cfg(feature = "survival")]
#[test]
fn rtte_records_out_of_order_are_coded() {
    use crate::types::{EventType, ObsRecord};
    let src = std::fs::read_to_string("examples/rtte_exponential.ferx").expect("example");
    let model = parse_model_string(&src).expect("parse");
    let event = |time: f64, event_type: EventType| ObsRecord::Event {
        time,
        event_type,
        entry_time: 0.0,
        cmt: 2,
    };
    let pop = population_of(vec![Subject {
        id: "1".to_string(),
        obs_records: vec![
            event(5.0, EventType::Exact),
            event(3.0, EventType::Exact),
            event(20.0, EventType::RightCensored),
        ],
        ..Default::default()
    }]);
    let want = "Subject '1': RTTE endpoint CMT=2 has records out of time order \
                (t=3 follows t=5). Repeated-event rows must be sorted by TIME \
                per subject so the cumulative hazard accumulates correctly.";
    assert_coded(
        &fit_refusal(&model, &pop, &model.default_params, &FitOptions::default()),
        "rtte",
        "E_RTTE_RECORDS",
        Some("event_model"),
        want,
    );
    let check = crate::api::check_model_data(&model, &pop);
    assert!(
        check.iter().any(|d| d.code == "E_RTTE_RECORDS"),
        "{check:?}"
    );
}

/// A binary record with DV 2 is `E_BINARY_STATE` on `binary_model`.
#[cfg(feature = "survival")]
#[test]
fn binary_state_outside_zero_one_is_coded() {
    use crate::types::ObsRecord;
    let src = std::fs::read_to_string("examples/binary_logistic.ferx").expect("example");
    let model = parse_model_string(&src).expect("parse");
    let state = |time: f64, state: usize| ObsRecord::DiscreteState {
        time,
        raw_time: time,
        state,
        cmt: 3,
    };
    let mut pop = population_of(vec![Subject {
        id: "1".to_string(),
        obs_records: vec![state(1.0, 0), state(2.0, 2)],
        covariates: [("X".to_string(), 1.0)].into(),
        ..Default::default()
    }]);
    pop.covariate_names = vec!["X".to_string()];
    assert_coded(
        &fit_refusal(&model, &pop, &model.default_params, &FitOptions::default()),
        "binary",
        "E_BINARY_STATE",
        Some("binary_model"),
        "[binary_model] cmt = 3: observed DV must be 0 or 1 (Bernoulli), got \
         2. For an ordered response with more than two categories use an \
         ordinal endpoint (not yet supported).",
    );
    let check = crate::api::check_model_data(&model, &pop);
    assert!(
        check.iter().any(|d| d.code == "E_BINARY_STATE"),
        "{check:?}"
    );
}

/// An undeclared state and decreasing times are `E_CTMM_RECORDS` on `markov_model`. An
/// intensity reading a model state on a model with no ODE system stays refused, by the
/// #1918 fail-closed check that runs first (`E_ENDPOINT_TIME_DEPENDENT_PK`, same block),
/// so `fit_unstamped`'s own guard for it was removed rather than coded.
#[cfg(feature = "markov")]
#[test]
fn ctmm_refusals_are_coded() {
    use crate::types::{EndpointLikelihood, ObsRecord};
    let src = std::fs::read_to_string("examples/ctmm_2state.ferx").expect("example");
    let model = parse_model_string(&src).expect("parse");
    let state = |time: f64, state: usize| ObsRecord::DiscreteState {
        time,
        raw_time: time,
        state,
        cmt: 5,
    };
    let pop_with = |records: Vec<ObsRecord>| {
        population_of(vec![Subject {
            id: "1".to_string(),
            obs_records: records,
            ..Default::default()
        }])
    };
    let undeclared = pop_with(vec![state(1.0, 0), state(2.0, 7)]);
    assert_coded(
        &fit_refusal(
            &model,
            &undeclared,
            &model.default_params,
            &FitOptions::default(),
        ),
        "undeclared state",
        "E_CTMM_RECORDS",
        Some("markov_model"),
        "[markov_model] cmt = 5: observed DV 7 is not one of the declared \
         states [0, 1]. Every DV value on a [markov_model] CMT must match a `states` code.",
    );
    let unsorted = pop_with(vec![state(2.0, 0), state(1.0, 1)]);
    assert_coded(
        &fit_refusal(
            &model,
            &unsorted,
            &model.default_params,
            &FitOptions::default(),
        ),
        "unsorted times",
        "E_CTMM_RECORDS",
        Some("markov_model"),
        "[markov_model] cmt = 5: observation times must be non-decreasing \
         within a subject, but 2 is followed by 1. Sort each subject's \
         CTMM rows by TIME (the datareader does not reorder observations).",
    );
    assert!(crate::api::check_model_data(&model, &unsorted)
        .iter()
        .any(|d| d.code == "E_CTMM_RECORDS"));

    let mut no_ode = parse_model_string(&src).expect("parse");
    for ep in no_ode.endpoints.values_mut() {
        if let EndpointLikelihood::Ctmm {
            generator_states, ..
        } = ep
        {
            generator_states.push(("central".to_string(), 0));
        }
    }
    let ok = pop_with(vec![state(1.0, 0), state(2.0, 1)]);
    let err = fit_refusal(&no_ode, &ok, &no_ode.default_params, &FitOptions::default());
    assert_eq!(err.code(), Some("E_ENDPOINT_TIME_DEPENDENT_PK"), "{err}");
    assert_eq!(err.block(), Some("markov_model"), "{err}");
}

// ── file entry points ────────────────────────────────────────────────────────

/// No data path and no `[data]` block is `E_DATA_PATH_MISSING`, with no block, on both
/// file entry points that resolve one.
#[test]
fn a_missing_data_path_is_coded() {
    let f = write_model(ONE_CPT_IV);
    let path = f.path().to_str().unwrap();
    let want = "no dataset specified — pass a data path, or add a `[data]` block \
                (`path = ...`) to the model file";
    let errs = [
        (
            "fit_from_files",
            crate::api::fit_from_files(path, None, None, None).expect_err("no data"),
        ),
        (
            "prepare_run",
            crate::api::prepare_run(path, None).err().expect("no data"),
        ),
    ];
    for (entry, err) in &errs {
        assert_coded(err, entry, "E_DATA_PATH_MISSING", None, want);
    }
}

/// `run_model_simulate`'s `[simulation]` design refusals: no block is
/// `E_MISSING_BLOCK`, a design with nothing to observe `E_SIMULATION_DESIGN`, a covariate
/// the design does not state `E_MISSING_COVARIATE`, all on block `simulation`.
#[test]
fn run_model_simulate_design_refusals_are_coded() {
    let simulate = |text: &str| {
        let f = write_model(text);
        crate::api::run_model_simulate(f.path().to_str().unwrap())
            .err()
            .expect("the design is refused")
    };
    assert_coded(
        &simulate(ONE_CPT_IV),
        "no [simulation]",
        "E_MISSING_BLOCK",
        Some("simulation"),
        "Model file has no [simulation] block — use --data instead",
    );

    let no_times = format!(
        "{ONE_CPT_IV}[simulation]\n  n_subjects = 2\n  dose_amt = 100\n  dose_cmt = 1\n  \
         seed = 7\n  horizon = 10\n"
    );
    assert_coded(
        &simulate(&no_times),
        "no times",
        "E_SIMULATION_DESIGN",
        Some("simulation"),
        "[simulation] has no `times`, but the model has a continuous \
         (residual-error) endpoint that needs observation times — add \
         `times = [...]` (a joint PK + TTE design needs both `times` and a \
         `horizon`)",
    );

    let reads_wt = format!(
        "{}[simulation]\n  n_subjects = 2\n  dose_amt = 100\n  dose_cmt = 1\n  seed = 7\n  \
         times = [1, 2]\n",
        ONE_CPT_IV.replace("V  = TVV\n", "V  = TVV * WT / 70\n")
    );
    let err = simulate(&reads_wt);
    assert_eq!(err.code(), Some("E_MISSING_COVARIATE"), "{err}");
    assert_eq!(err.block(), Some("simulation"), "{err}");
    assert!(
        err.to_string().starts_with(
            "[simulation]: the model references covariate(s) `WT` that the simulation design \
             does not supply."
        ),
        "{err}"
    );
}

/// A TTE endpoint with no `horizon` is `E_SIMULATE_HORIZON`, the code the `simulate*`
/// entry points give a missing horizon; a binary endpoint with no `times`, and a model
/// with no endpoint to observe at the `horizon`, are `E_SIMULATION_DESIGN`.
#[cfg(feature = "survival")]
#[test]
fn run_model_simulate_survival_design_refusals_are_coded() {
    let simulate = |text: &str| {
        let f = write_model(text);
        crate::api::run_model_simulate(f.path().to_str().unwrap())
            .err()
            .expect("the design is refused")
    };
    let rtte = std::fs::read_to_string("examples/rtte_exponential.ferx").expect("example");
    let err = simulate(&rtte.replace("  horizon    = 20\n", "  times = [1, 2]\n"));
    assert_coded(
        &err,
        "TTE without horizon",
        "E_SIMULATE_HORIZON",
        None,
        "[simulation] with a TTE endpoint requires `horizon = <t>` \
         (the administrative censoring time at which event-free subjects are \
         right-censored)",
    );

    let binary = std::fs::read_to_string("examples/binary_logistic.ferx").expect("example");
    let err = simulate(&format!(
        "{binary}\n[simulation]\n  n_subjects = 2\n  dose_amt = 0\n  dose_cmt = 1\n  \
         seed = 7\n  horizon = 10\n"
    ));
    assert_eq!(err.code(), Some("E_SIMULATION_DESIGN"), "{err}");
    assert_eq!(err.block(), Some("simulation"), "{err}");
    assert!(
        err.to_string().starts_with(
            "[simulation] has no `times`, but the model has a [binary_model] endpoint."
        ),
        "{err}"
    );
}
