//! Tier-1 tests for #1774: the predict / simulate / adaptive refusals `ferx check` cannot
//! reach carry a code, one per cause, assigned in `api/entry_refusals.rs`.
//!
//! One row per refusal **text**, through the public entry point that raises it. Each row
//! asserts three things, and each has its own mutation:
//!
//! | Assertion | Regression it catches |
//! |---|---|
//! | `code()` | the site still returns its old `String` (`code()` is `None`), or names another cause's constructor |
//! | `block()` | a constructor keyed on the wrong block, or a block dropped |
//! | `to_string()`, the whole text | a sentence of the message deleted or reworded while it was moved; every sentence of every refusal is pinned here once, so deleting any one of them kills its row |
//!
//! The two refusals `ferx check` *does* reach (`E_ENDPOINT_TV_COVARIATE`,
//! `E_READOUT_DEPOT_UNSUPPORTED`) are pinned against `ferx check` itself in
//! `entry_point_errors_tests::each_precondition_family_is_an_err_carrying_its_own_message`
//! and in `ferx_check_reports_what_fit_refuses` below.
//!
//! Tier 1: no row runs a simulation past its preconditions, except the propensity rows,
//! which stop before the inner EBE pass.

use super::*;
use crate::parser::model_parser::{parse_full_model, parse_model_string};
use crate::types::{DoseEvent, Population, Subject};

/// One refusal: the `Err` an entry point returned, and what it must carry.
struct Row {
    name: &'static str,
    got: Result<(), crate::EngineError>,
    code: &'static str,
    block: Option<&'static str>,
    text: &'static str,
}

fn row<T>(
    name: &'static str,
    got: Result<T, crate::EngineError>,
    code: &'static str,
    block: Option<&'static str>,
    text: &'static str,
) -> Row {
    Row {
        name,
        got: got.map(|_| ()),
        code,
        block,
        text,
    }
}

/// Every mismatch at once, so one run names every row that moved rather than the first.
fn check_rows(rows: Vec<Row>) {
    let mut bad = Vec::new();
    for r in rows {
        let Err(e) = r.got else {
            bad.push(format!("{}: accepted", r.name));
            continue;
        };
        if e.code() != Some(r.code) {
            bad.push(format!("{}: code {:?}, want {}", r.name, e.code(), r.code));
        }
        if e.block() != r.block {
            bad.push(format!(
                "{}: block {:?}, want {:?}",
                r.name,
                e.block(),
                r.block
            ));
        }
        if e.to_string() != r.text {
            bad.push(format!("{}: text\n  got:  {e}\n  want: {}", r.name, r.text));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn population_of(subjects: Vec<Subject>, covariate_names: &[&str]) -> Population {
    Population {
        subjects,
        covariate_names: covariate_names.iter().map(|s| s.to_string()).collect(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn observed(id: &str, dvs: &[f64]) -> Subject {
    let n = dvs.len();
    Subject {
        id: id.to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: (1..=n).map(|t| t as f64).collect(),
        observations: dvs.to_vec(),
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        ..Default::default()
    }
}

fn opts(horizon: Option<f64>) -> SimulateOptions {
    SimulateOptions {
        seed: Some(1),
        horizon,
        ..Default::default()
    }
}

const ONE_CPT_IV: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  \
    theta TVV(40.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.02\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n";

const ODE_IOV: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  \
    theta TVV(40.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.04\n  \
    sigma PROP ~ 0.02\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  \
    V  = TVV\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
    d/dt(central) = -CL/V * central\n[error_model]\n  DV ~ proportional(PROP)\n";

/// `E_SIMULATE_HORIZON` (the `SimulateOptions` value), `E_PROPENSITY_INPUT` and
/// `E_PARAM_SHAPE` on a missing Ω~IOV~ — the rows with no endpoint feature.
#[test]
fn simulate_option_and_input_refusals_carry_their_codes() {
    let iv = parse_model_string(ONE_CPT_IV).expect("parse");
    let p = &iv.default_params;
    let matched = |pop: &Population| {
        let o = SimulateOptions {
            seed: Some(1),
            match_method: Some(crate::propensity_match::MatchMethod::Optimal),
            ..Default::default()
        };
        simulate_with_options(&iv, pop, p, 1, &o)
    };
    let iov = parse_model_string(ODE_IOV).expect("parse");
    let mut no_iov = iov.default_params.clone();
    no_iov.omega_iov = None;
    let mut dose_only = observed("7", &[]);
    dose_only.obs_cmts.clear();
    check_rows(vec![
        row(
            "horizon NaN",
            simulate_with_options(
                &iv,
                &population_of(vec![observed("1", &[1.0])], &[]),
                p,
                1,
                &opts(Some(f64::NAN)),
            ),
            "E_SIMULATE_HORIZON",
            None,
            "SimulateOptions.horizon must be finite and > 0 (got NaN)",
        ),
        row(
            "horizon zero",
            simulate_with_options(
                &iv,
                &population_of(vec![observed("1", &[1.0])], &[]),
                p,
                1,
                &opts(Some(0.0)),
            ),
            "E_SIMULATE_HORIZON",
            None,
            "SimulateOptions.horizon must be finite and > 0 (got 0)",
        ),
        row(
            "propensity, empty population",
            matched(&population_of(Vec::new(), &[])),
            "E_PROPENSITY_INPUT",
            None,
            "propensity-score matching requires a non-empty observed population",
        ),
        row(
            "propensity, a subject with no observations",
            matched(&population_of(vec![observed("1", &[1.0]), dose_only], &[])),
            "E_PROPENSITY_INPUT",
            None,
            "propensity-score matching requires observations for every subject (to \
             compute posthoc etas); subject '7' has none",
        ),
        row(
            "propensity, a non-finite DV",
            matched(&population_of(
                vec![observed("1", &[1.0]), observed("8", &[f64::NAN])],
                &[],
            )),
            "E_PROPENSITY_INPUT",
            None,
            "propensity-score matching requires finite observations for every subject (to \
             compute posthoc etas); subject '8' has non-finite DV values. A `DV = .` \
             design template carries NaN placeholders — match against the observed \
             dataset instead",
        ),
        row(
            "Ω_IOV missing (#1019)",
            simulate(
                &iov,
                &population_of(vec![observed("1", &[1.0])], &[]),
                &no_iov,
                1,
            ),
            "E_PARAM_SHAPE",
            None,
            "model declares 1 kappa (IOV) but the supplied parameters carry no omega_iov; \
             simulation draws one kappa vector per occasion from it. Rebuild the \
             parameters with the fitted IOV covariance (see `fitted_params_from_result`) \
             before simulating",
        ),
    ]);
}

/// `E_TTE_NOT_SIMULATABLE` and the TTE half of `E_SIMULATE_HORIZON`: every text of
/// `validate_tte_simulatable`, plus the horizon below an entry time.
#[cfg(feature = "survival")]
#[test]
fn tte_simulation_refusals_carry_their_codes() {
    use crate::types::{EventType, ObsRecord};
    let read = |path: &str| std::fs::read_to_string(path).expect("example");
    let event = |cmt: usize, entry_time: f64| ObsRecord::Event {
        time: 20.0,
        event_type: EventType::RightCensored,
        entry_time,
        cmt,
    };
    let tte_subject = |records: Vec<ObsRecord>| Subject {
        id: "4".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_records: records,
        ..Default::default()
    };
    let with_reset = |mut s: Subject| {
        s.reset_times = vec![5.0];
        s.reset_covariates = vec![Default::default()];
        s.reset_occasions = vec![1];
        s
    };
    let one = |s: Subject| population_of(vec![s], &[]);

    let rtte = parse_model_string(&read("examples/rtte_exponential.ferx")).expect("parse");
    let competing = read("examples/tte_competing_risks.ferx");
    let two_rtte = parse_model_string(&competing.replace(
        "family = exponential",
        "type   = rtte\n  family = exponential",
    ))
    .expect("parse");
    let rtte_and_single = parse_model_string(&competing.replacen(
        "family = exponential",
        "type   = rtte\n  family = exponential",
        1,
    ))
    .expect("parse");
    let single = parse_model_string(&competing).expect("parse");
    let ode_tte = parse_model_string(&read("examples/pktte_joint.ferx")).expect("parse");

    let sim = |m: &CompiledModel, pop: Population, h: Option<f64>| {
        simulate_with_options(m, &pop, &m.default_params, 1, &opts(h))
    };
    check_rows(vec![
        row(
            "RTTE without a horizon",
            sim(&rtte, one(tte_subject(vec![event(2, 0.0)])), None),
            "E_SIMULATE_HORIZON",
            None,
            "Repeated time-to-event (RTTE, `type = rtte`) simulation requires a finite, \
             positive administrative horizon: set `[simulation] horizon` (or \
             `SimulateOptions.horizon`). The recurrent event stream is generated up to \
             that horizon.",
        ),
        row(
            "RTTE on two CMTs",
            sim(
                &two_rtte,
                one(tte_subject(vec![event(2, 0.0), event(3, 0.0)])),
                Some(20.0),
            ),
            "E_TTE_NOT_SIMULATABLE",
            Some("event_model"),
            "RTTE simulation supports one recurrent stream per subject, but subject '4' \
             has RTTE endpoints on multiple CMTs ({2, 3}); multi-cause RTTE simulation is \
             a later slice",
        ),
        row(
            "RTTE beside a single-event cause",
            sim(
                &rtte_and_single,
                one(tte_subject(vec![event(2, 0.0), event(3, 0.0)])),
                Some(20.0),
            ),
            "E_TTE_NOT_SIMULATABLE",
            Some("event_model"),
            "RTTE simulation does not support an RTTE cause combined with a competing \
             single-event TTE cause on subject '4'; simulate them separately",
        ),
        row(
            "RTTE with a reset",
            sim(
                &rtte,
                one(with_reset(tte_subject(vec![event(2, 0.0)]))),
                Some(20.0),
            ),
            "E_TTE_NOT_SIMULATABLE",
            Some("event_model"),
            "RTTE simulation does not support EVID=3/4 resets (a reset would disturb the \
             hazard clock mid-stream; selective per-state reset is a later slice); \
             subject '4' has resets",
        ),
        row(
            "ODE-accumulated TTE without a horizon",
            sim(&ode_tte, one(tte_subject(vec![event(3, 0.0)])), None),
            "E_SIMULATE_HORIZON",
            None,
            "ODE-accumulated TTE (joint PK-TTE) simulation requires a finite, positive \
             administrative horizon: set `[simulation] horizon` (or \
             `SimulateOptions.horizon`). A drug-driven hazard can vanish, so there is no \
             implicit observation window.",
        ),
        row(
            "ODE-accumulated TTE with a reset",
            sim(
                &ode_tte,
                one(with_reset(tte_subject(vec![event(3, 0.0)]))),
                Some(20.0),
            ),
            "E_TTE_NOT_SIMULATABLE",
            Some("event_model"),
            "ODE-accumulated TTE simulation does not support EVID=3/4 resets (a full \
             reset zeros the cumulative hazard mid-flight; selective per-state reset is a \
             later slice); subject '4' has resets",
        ),
        row(
            "ODE-accumulated TTE, left truncated",
            sim(&ode_tte, one(tte_subject(vec![event(3, 2.0)])), Some(20.0)),
            "E_TTE_NOT_SIMULATABLE",
            Some("event_model"),
            "ODE-accumulated TTE simulation does not support left truncation \
             (entry_time=2 for subject '4'); conditional sampling past entry is a \
             deferred follow-up",
        ),
        row(
            "horizon below an entry time",
            sim(&single, one(tte_subject(vec![event(2, 5.0)])), Some(2.0)),
            "E_SIMULATE_HORIZON",
            None,
            "SimulateOptions.horizon (2) is below subject '4' entry_time (5); the \
             administrative horizon must be ≥ every subject's entry time",
        ),
    ]);
}

/// `E_CTMM_UNSUPPORTED` from both entry points that have no CTMM path.
#[cfg(feature = "markov")]
#[test]
fn ctmm_refusals_carry_their_code() {
    let src = std::fs::read_to_string("examples/ctmm_2state.ferx").expect("example");
    let model = parse_model_string(&src).expect("parse");
    let pop = population_of(
        vec![Subject {
            id: "1".to_string(),
            ..Default::default()
        }],
        &[],
    );
    let p = &model.default_params;
    check_rows(vec![
        row(
            "predict, CTMM only",
            predict(&model, &pop, p),
            "E_CTMM_UNSUPPORTED",
            Some("markov_model"),
            "predict() does not support a [markov_model] (CTMM) endpoint yet, and this \
             population has no continuous observations either — so the call would return \
             an empty vec rather than an occupancy π(t). State-occupancy prediction is a \
             later slice (#820).",
        ),
        row(
            "simulate, CTMM",
            simulate(&model, &pop, p, 1),
            "E_CTMM_UNSUPPORTED",
            Some("markov_model"),
            "simulate()/predict() does not support a [markov_model] (CTMM) endpoint yet — \
             its discrete-state trajectory has no simulation path, so the Gaussian \
             emitter would produce meaningless all-zero observations. CTMM simulation is \
             a later slice.",
        ),
    ]);
}

const ODE_ADAPTIVE: &str = "[parameters]\n  theta TVCL(4.0, 0.5, 20.0)\n  \
    theta TVV(50.0, 5.0, 200.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.10 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    ode(states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n[scaling]\n  \
    y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n";

/// `E_ADAPTIVE_UNSUPPORTED`, `E_ADAPTIVE_OPTIONS`, `E_ADAPTIVE_SPEC` and the reused
/// `E_MISSING_COVARIATE`, through `simulate_adaptive` and `simulate_adaptive_from_spec`.
#[test]
fn adaptive_refusals_carry_their_codes() {
    use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};
    let ode = parse_model_string(ODE_ADAPTIVE).expect("parse");
    let iv = parse_model_string(ONE_CPT_IV).expect("parse");
    let pop = population_of(vec![observed("1", &[1.0])], &[]);
    let bolus = || |_: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let programmatic = |m: &CompiledModel, decision_times: Vec<f64>| {
        let o = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times,
            ..Default::default()
        };
        simulate_adaptive(m, &pop, &m.default_params, 1, bolus, &o)
    };
    // The TDM example's spec, edited one field at a time.
    let tdm = std::fs::read_to_string("examples/adaptive_tdm_titration.ferx").expect("example");
    let spec = parse_full_model(&tdm)
        .expect("parse")
        .adaptive_dosing
        .expect("has [adaptive_dosing]");
    let from_spec = |m: &CompiledModel,
                     pop: &Population,
                     edit: &dyn Fn(&mut crate::sim::adaptive::AdaptiveDosingSpec),
                     o: AdaptiveSimulateOptions| {
        let mut s = spec.clone();
        edit(&mut s);
        simulate_adaptive_from_spec(m, pop, &m.default_params, 1, &s, &o)
    };
    let none = |_: &mut crate::sim::adaptive::AdaptiveDosingSpec| {};
    let seeded = || AdaptiveSimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let mut reset_subject = observed("1", &[1.0]);
    reset_subject.reset_times = vec![30.0];
    reset_subject.reset_covariates = vec![Default::default()];
    reset_subject.reset_occasions = vec![1];
    check_rows(vec![
        row(
            "simulate_adaptive, no [odes]",
            programmatic(&iv, vec![0.0]),
            "E_ADAPTIVE_UNSUPPORTED",
            None,
            "simulate_adaptive requires an ODE model (the reactive driver runs on the ODE \
             engine); this model has no [odes] block",
        ),
        row(
            "simulate_adaptive, empty decision_times",
            programmatic(&ode, Vec::new()),
            "E_ADAPTIVE_OPTIONS",
            None,
            "simulate_adaptive requires a non-empty decision schedule \
             (`AdaptiveSimulateOptions::decision_times`); with no decision times the \
             controller is never consulted and no dose is ever issued",
        ),
        row(
            "simulate_adaptive, decision_times not increasing",
            programmatic(&ode, vec![12.0, 0.0]),
            "E_ADAPTIVE_OPTIONS",
            None,
            "[adaptive_dosing]: `decision_times` must be strictly increasing",
        ),
        row(
            "simulate_adaptive, decision_times not finite",
            programmatic(&ode, vec![0.0, f64::NAN]),
            "E_ADAPTIVE_OPTIONS",
            None,
            "[adaptive_dosing]: `decision_times` must all be finite",
        ),
        row(
            "from_spec, decision_times also set",
            from_spec(
                &ode,
                &pop,
                &none,
                AdaptiveSimulateOptions {
                    decision_times: vec![0.0],
                    ..seeded()
                },
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "simulate_adaptive_from_spec takes its decision schedule from the \
             [adaptive_dosing] block's `at`; leave `opts.decision_times` empty",
        ),
        row(
            "from_spec, monitors also set",
            from_spec(
                &ode,
                &pop,
                &none,
                AdaptiveSimulateOptions {
                    monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)],
                    ..seeded()
                },
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "simulate_adaptive_from_spec takes its monitor from the [adaptive_dosing] \
             block's `observe`; leave `opts.monitors` empty",
        ),
        row(
            "from_spec, no [odes]",
            from_spec(&iv, &pop, &none, seeded()),
            "E_ADAPTIVE_UNSUPPORTED",
            None,
            "simulate_adaptive_from_spec: [adaptive_dosing] requires an ODE model (the \
             analytical engine is a follow-up)",
        ),
        row(
            "from_spec, spec invariant (no assay_cmt)",
            from_spec(&ode, &pop, &|s| s.assay_cmt = None, seeded()),
            "E_ADAPTIVE_SPEC",
            Some("adaptive_dosing"),
            "simulate_adaptive_from_spec: [adaptive_dosing]: `with_assay_error = true` \
             requires `assay_cmt = N` to name which model output is measured",
        ),
        row(
            "from_spec, assay_cmt out of range",
            from_spec(&ode, &pop, &|s| s.assay_cmt = Some(3), seeded()),
            "E_ADAPTIVE_SPEC",
            Some("adaptive_dosing"),
            "simulate_adaptive_from_spec: [adaptive_dosing]: assay_cmt = 3 is out of \
             range (model has 1 compartment(s))",
        ),
        row(
            "from_spec, observe does not compile",
            from_spec(
                &ode,
                &pop,
                &|s| {
                    s.with_assay_error = false;
                    s.assay_cmt = None;
                    s.observe = Some("central /".to_string());
                },
                seeded(),
            ),
            "E_ADAPTIVE_SPEC",
            Some("adaptive_dosing"),
            "simulate_adaptive_from_spec: [adaptive_dosing] observe: Unexpected end of \
             expression",
        ),
        row(
            "from_spec, observe names a covariate the data lack",
            from_spec(
                &ode,
                &pop,
                &|s| {
                    s.with_assay_error = false;
                    s.assay_cmt = None;
                    s.observe = Some("central / WT".to_string());
                    s.observe_declared_covariates = vec!["WT".to_string()];
                },
                seeded(),
            ),
            "E_MISSING_COVARIATE",
            Some("adaptive_dosing"),
            "simulate_adaptive_from_spec: [adaptive_dosing] `observe` references \
             covariate(s) not found in data (case-sensitive): WT. Available covariate \
             columns: (none).",
        ),
        row(
            "from_spec, auc_target on a reset subject",
            from_spec(
                &ode,
                &population_of(vec![reset_subject], &[]),
                &|s| s.auc_target = Some((300.0, 500.0)),
                seeded(),
            ),
            "E_ADAPTIVE_UNSUPPORTED",
            None,
            "adaptive-dosing `auc_target` is not yet supported for \
             time-varying-covariate, TIME-in-PK, IOV (`kappa`), or system-reset (EVID=3) \
             subjects. For a drifting covariate or a per-occasion κ the exposure metric \
             integrates its dense grid from a single frozen PK snapshot, which is \
             silently wrong when the PK changes across the horizon. For a reset the dense \
             solver *does* zero the state, but the per-window trapezoid grid carries no \
             node at the reset instant, so the one window straddling the reset would \
             integrate that discontinuity inaccurately. Drop `auc_target` (all other \
             outputs remain per-event / per-occasion / reset aware), or track \
             #700/#701/#716 for a per-event, reset-node-aware AUC.",
        ),
    ]);
}

/// Two of the three model refusals the adaptive entry points share, through the helpers
/// they call: their whole text and code. The third, a residual `correlation` line, is
/// pinned whole beside its fixture in `tests.rs`.
#[test]
fn adaptive_model_refusal_texts() {
    let selected = ODE_ADAPTIVE
        .replace("  sigma PROP ~ 0.10 (sd)", "  sigma PROP ~ 0.10 (sd)\n  sigma PROP2 ~ 0.10 (sd)")
        .replace(
            "  DV ~ proportional(PROP)",
            "  if (FREE == 0) {\n    DV ~ proportional(PROP)\n  } else {\n    DV ~ proportional(PROP2)\n  }\n[covariates]\n  FREE continuous",
        );
    // An SDE takes no Form C readout, so the observable is the state itself.
    let sde = ODE_ADAPTIVE
        .replace(
            "ode(states=[central])",
            "ode(obs_cmt=central, states=[central])",
        )
        .replace("[scaling]\n  y = central / V\n", "")
        .replace(
            "[error_model]",
            "[diffusion]\n  central ~ 0.01\n[error_model]",
        );
    let selected = parse_model_string(&selected).expect("parse");
    let sde = parse_model_string(&sde).expect("parse");
    let pop = population_of(vec![observed("1", &[1.0])], &[]);
    check_rows(vec![
        row(
            "covariate-selected [error_model]",
            reject_selected_error_for_adaptive(&selected),
            "E_ADAPTIVE_UNSUPPORTED",
            Some("error_model"),
            "adaptive-dosing simulation does not support a covariate-selected \
             `[error_model]` (`if (COV …) { … } else { … }`): the assay keys residual \
             error by the monitored compartment, but a selected error model keys \
             endpoints by covariate branch. Use a single-endpoint error model for the \
             monitored signal.",
        ),
        row(
            "SDE",
            reject_unsupported_adaptive(&sde, &pop),
            "E_ADAPTIVE_UNSUPPORTED",
            Some("diffusion"),
            "adaptive-dosing simulation does not support stochastic (`[diffusion]` / SDE) \
             models: the reactive integrator is deterministic and would silently drop the \
             process-noise term. Use the deterministic ODE model for adaptive dosing.",
        ),
    ]);
}

const DEPOT_READOUT: &str = "[parameters]\n  theta TVCL(5.0, 0.1, 50.0)\n  \
    theta TVV(50.0, 5.0, 500.0)\n  theta TVKA(1.0, 0.01, 10.0)\n  omega ETA_CL ~ 0.09\n  \
    sigma PROP ~ 0.01 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  \
    KA = TVKA\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[scaling]\n  \
    y = (central + depot) / V\n[error_model]\n  DV ~ proportional(PROP)\n";

/// `E_READOUT_DEPOT_UNSUPPORTED`'s two texts, one per cause of a depot the superposition
/// cannot reconstruct, through `predict`. The code and block are also pinned against
/// `ferx check` (`each_precondition_family_is_an_err_carrying_its_own_message`); the
/// whole text is pinned here, since that test compares `predict` with `fit()`, which share
/// the text and so agree on any deletion.
#[test]
fn readout_refusal_texts() {
    let model = parse_model_string(DEPOT_READOUT).expect("parse");
    let p = &model.default_params;
    let mut reset = observed("1", &[1.0]);
    reset.obs_cmts = vec![2];
    reset.reset_times = vec![0.5];
    reset.reset_covariates = vec![Default::default()];
    reset.reset_occasions = vec![1];
    let mut central_dose = observed("2", &[1.0]);
    central_dose.obs_cmts = vec![2];
    central_dose.doses = vec![DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0)];
    check_rows(vec![
        row(
            "depot readout across a reset (#650)",
            predict(&model, &population_of(vec![reset], &[]), p),
            "E_READOUT_DEPOT_UNSUPPORTED",
            Some("scaling"),
            "[scaling] y: an analytic Form C readout that references the oral `depot` \
             amount is not supported on a subject with an EVID=3/4 reset (subject 1): the \
             depot amount is reconstructed by dose superposition, which is invalid across \
             a reset. Reference only the `central` amount, or use an ODE model \
             (`ode(states=[depot, central])`) which integrates the depot across resets. \
             See issue #650.",
        ),
        row(
            "depot readout, dose into a non-default compartment (#375)",
            predict(&model, &population_of(vec![central_dose], &[]), p),
            "E_READOUT_DEPOT_UNSUPPORTED",
            Some("scaling"),
            "[scaling] y: an analytic Form C readout that references the oral `depot` \
             amount is not supported on a subject dosing a non-default compartment \
             (subject 2): the depot amount is reconstructed by dose superposition, which \
             places every bolus in compartment 1 and every infusion in central regardless \
             of `CMT`, so a dose elsewhere would contribute a phantom depot amount to \
             `PRED` (and to the objective). Reference only the `central` amount, dose the \
             default compartment, or use an ODE model (`ode(states=[depot, central])`) \
             which addresses compartments directly. See issue #375.",
        ),
    ]);
}

/// `ferx check` now reports the two refusals `fit()` already made (#1774), through
/// [`validate_model_file`] on files: before, it called the pair `valid: true`.
///
/// Mutation — drop either `extend` from `check_model_data_rule_at` and its arm reports
/// `valid: true` with no such code. The other arm, a data file `fit()` accepts, must stay
/// valid, so a check that refused every readout model would die too.
#[test]
fn ferx_check_reports_what_fit_refuses() {
    // Removed on drop, so a failing assertion leaves nothing behind either.
    let dir = tempfile::tempdir().expect("temp dir");
    let model = dir.path().join("readout.ferx");
    std::fs::write(
        &model,
        "[parameters]\n  theta TVCL(5.0, 0.1, 50.0)\n  theta TVV(50.0, 5.0, 500.0)\n  \
         theta TVKA(1.0, 0.01, 10.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  KA = TVKA\n\
         [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[scaling]\n  \
         y = (central + depot) / V\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("write model");
    let header = "ID,TIME,DV,EVID,AMT,CMT,MDV\n";
    let rows = "1,0,.,1,100,1,1\n1,1,2.0,0,.,2,0\n1,4,1.0,0,.,2,0\n";
    let reset = "1,2,.,3,.,.,1\n";
    for (name, data, want) in [
        ("no reset", format!("{header}{rows}"), None),
        (
            "a reset",
            format!("{header}{rows}{reset}"),
            Some("E_READOUT_DEPOT_UNSUPPORTED"),
        ),
    ] {
        let path = dir.path().join(format!("{}.csv", name.replace(' ', "_")));
        std::fs::write(&path, data).expect("write data");
        let report = validate_model_file(
            model.to_str().expect("utf-8 path"),
            Some(path.to_str().expect("utf-8 path")),
        );
        let codes: Vec<&str> = report
            .diagnostics
            .iter()
            .filter(|d| d.is_error())
            .map(|d| d.code.as_str())
            .collect();
        match want {
            None => assert!(report.valid, "{name}: {:?}", report.diagnostics),
            Some(code) => {
                assert!(!report.valid, "{name}: {:?}", report.diagnostics);
                assert_eq!(codes, [code], "{name}");
                let d = report.diagnostics.iter().find(|d| d.code == code).unwrap();
                assert_eq!(d.block.as_deref(), Some("scaling"), "{name}");
                assert!(
                    d.line.is_some(),
                    "{name}: the [scaling] header line is attached"
                );
            }
        }
    }
}
