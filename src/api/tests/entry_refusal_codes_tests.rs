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
    let unc = |m: &CompiledModel, pop: Population| {
        let fit = super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params);
        let o = SimulateUncertaintyOptions {
            n_uncertainty_draws: 0,
            n_sim_per_draw: 1,
            seed: Some(1),
            ..Default::default()
        };
        simulate_with_uncertainty(m, &pop, &fit, &o)
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
        // `simulate_with_uncertainty` takes no horizon (review r1 #4): the same two
        // horizon-required cases are not an option to set there, so they are
        // `E_TTE_NOT_SIMULATABLE`, naming the entry point that can simulate the model.
        row(
            "uncertainty, RTTE (no horizon option)",
            unc(&rtte, one(tte_subject(vec![event(2, 0.0)]))),
            "E_TTE_NOT_SIMULATABLE",
            Some("event_model"),
            "Repeated time-to-event (RTTE, `type = rtte`) simulation requires a finite, \
             positive administrative horizon, and `simulate_with_uncertainty` takes none \
             (`SimulateUncertaintyOptions` has no horizon). The recurrent event stream is \
             generated up to that horizon. Simulate this model with \
             `simulate_with_options` and `SimulateOptions.horizon` instead.",
        ),
        row(
            "uncertainty, ODE-accumulated TTE (no horizon option)",
            unc(&ode_tte, one(tte_subject(vec![event(3, 0.0)]))),
            "E_TTE_NOT_SIMULATABLE",
            Some("event_model"),
            "ODE-accumulated TTE (joint PK-TTE) simulation requires a finite, positive \
             administrative horizon, and `simulate_with_uncertainty` takes none \
             (`SimulateUncertaintyOptions` has no horizon). A drug-driven hazard can \
             vanish, so there is no implicit observation window. Simulate this model with \
             `simulate_with_options` and `SimulateOptions.horizon` instead.",
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

/// Two states, residual error on CMT 2 only: a `Dv` reading of CMT 1 has no σ.
const ODE_ERR_ON_CMT2: &str = "[parameters]\n  theta TVCL(4.0, 0.5, 20.0)\n  \
    theta TVV(50.0, 5.0, 200.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.10 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    ode(obs_cmt=central, states=[central, periph])\n[odes]\n  d/dt(central) = -(CL / V) * central\n  \
    d/dt(periph) = 0\n[error_model]\n  CMT=2: DV ~ proportional(PROP)\n";

/// A subject with observations and no dose: the controller is its only source of drug.
fn dose_free(id: &str, obs_times: &[f64]) -> Subject {
    let n = obs_times.len();
    Subject {
        id: id.to_string(),
        obs_times: obs_times.to_vec(),
        observations: vec![1.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        ..Default::default()
    }
}

/// The next `f64` above `t`: within the driver's 1e-15 dedup of `t`, but not bit-equal.
fn next_up(t: f64) -> f64 {
    f64::from_bits(t.to_bits() + 1)
}

/// The adaptive driver's refusals that are a function of the inputs alone, moved to the
/// entry by #1983: their code, block and whole text, and — the reason they moved — that
/// each is raised **before any subject is simulated**. The controller factory is called
/// once per (subject, replicate) just before the driver runs, so a row whose factory count
/// is not 0 reached a subject first. That is what the driver-side check did: on the
/// base × reset row subject '1' is valid and would run, and only subject '2' is refused.
///
/// Each predicate is also asserted from its accepting side, in the same test, so a check
/// stuck on "refuse" (or a gate deleted) reddens it.
#[test]
fn adaptive_entry_refusals_simulate_no_subject() {
    use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};
    use std::cell::Cell;
    let ode = parse_model_string(ODE_ADAPTIVE).expect("parse");
    let iov = parse_model_string(ODE_IOV).expect("parse");
    let cmt2_err = parse_model_string(ODE_ERR_ON_CMT2).expect("parse");
    let made = Cell::new(0usize);
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    // `simulate_adaptive`, counting controllers made; each row resets the count.
    let mut run =
        |name: &'static str, m: &CompiledModel, pop: &Population, o: AdaptiveSimulateOptions| {
            made.set(0);
            let r = simulate_adaptive(
                m,
                pop,
                &m.default_params,
                1,
                || {
                    made.set(made.get() + 1);
                    |_: &ControllerCtx| vec![DoseAction::Hold]
                },
                &o,
            );
            counts.push((name, made.get()));
            r
        };
    let schedule = |decision_times: Vec<f64>| AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times,
        ..Default::default()
    };
    let monitored = |monitors: Vec<MonitorSpec>| AdaptiveSimulateOptions {
        monitors,
        ..schedule(vec![0.0])
    };
    let one = population_of(vec![dose_free("1", &[12.0])], &[]);
    // Subject '1' is valid; subject '2' carries a base dose and an EVID=3 reset.
    let mut base_reset = observed("2", &[1.0, 40.0]);
    base_reset.reset_times = vec![30.0];
    let two = population_of(vec![observed("1", &[1.0, 40.0]), base_reset], &[]);

    // The spec path: no factory to count, so `n_sim = 0` — a refusal that still fires there
    // cannot be the driver's, which runs only inside a replicate.
    let tdm = std::fs::read_to_string("examples/adaptive_tdm_titration.ferx").expect("example");
    let mut spec = parse_full_model(&tdm)
        .expect("parse")
        .adaptive_dosing
        .expect("has [adaptive_dosing]");
    assert!(spec.with_assay_error, "the TDM example draws an assay");
    let mut spec_on = |cmt: usize| {
        spec.assay_cmt = Some(cmt);
        simulate_adaptive_from_spec(
            &cmt2_err,
            &one,
            &cmt2_err.default_params,
            0,
            &spec,
            &AdaptiveSimulateOptions::default(),
        )
    };
    let spec_dv_cmt1 = spec_on(1);
    let spec_dv_cmt2 = spec_on(2);

    let rows = vec![
        row(
            "base regimen × EVID=3 under IOV",
            run("base × reset", &iov, &two, schedule(vec![0.0, 24.0])),
            "E_ADAPTIVE_UNSUPPORTED",
            None,
            "subject '2': a pre-scheduled base regimen combined with system resets (EVID=3/4) \
             is not yet supported under time-varying covariates or inter-occasion \
             variability; #932 supports base × reset on the constant-covariate path only \
             (time-varying-covariate / IOV base × reset is a follow-up)",
        ),
        row(
            "max_decisions",
            run(
                "max_decisions",
                &ode,
                &one,
                AdaptiveSimulateOptions {
                    max_decisions: 1,
                    ..schedule(vec![0.0, 24.0])
                },
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "decision schedule has 2 points, exceeding max_decisions = 1 (runaway guard); \
             raise `max_decisions` in the simulate options if the schedule is intentional",
        ),
        row(
            "monitor past the states",
            run(
                "monitor cmt 2",
                &ode,
                &one,
                monitored(vec![MonitorSpec::new("A", 2, ObserveMode::Ipred)]),
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "monitor 'A' observes compartment 2 but the model has 1 state(s)",
        ),
        row(
            "monitor on compartment 0",
            run(
                "monitor cmt 0",
                &ode,
                &one,
                monitored(vec![MonitorSpec::new("A", 0, ObserveMode::Ipred)]),
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "monitor 'A' observes compartment 0 but the model has 1 state(s)",
        ),
        row(
            "Dv monitor, no [error_model] on its compartment",
            run(
                "Dv cmt 1",
                &cmt2_err,
                &one,
                monitored(vec![MonitorSpec::new("A", 1, ObserveMode::Dv)]),
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "monitor 'A' requests DV observation on compartment 1 but no [error_model] \
             defines residual error there",
        ),
        row(
            "from_spec, assay_cmt with no [error_model]",
            spec_dv_cmt1,
            "E_ADAPTIVE_SPEC",
            Some("adaptive_dosing"),
            "simulate_adaptive_from_spec: [adaptive_dosing]: `with_assay_error = true` draws \
             the assay from the [error_model] of assay_cmt = 1, but no [error_model] defines \
             residual error there",
        ),
    ];

    // The accepting side of each predicate, one input changed from a refused row.
    let accepted = [
        (
            "base × reset without IOV (constant path, #932)",
            run("ok: base × reset", &ode, &two, schedule(vec![0.0, 24.0])).err(),
        ),
        (
            "two decisions under max_decisions = 2",
            run(
                "ok: max_decisions",
                &ode,
                &one,
                AdaptiveSimulateOptions {
                    max_decisions: 2,
                    ..schedule(vec![0.0, 24.0])
                },
            )
            .err(),
        ),
        (
            "a monitor on the model's last state",
            run(
                "ok: monitor cmt 1",
                &ode,
                &one,
                monitored(vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)]),
            )
            .err(),
        ),
        (
            "a Dv monitor on the compartment with the [error_model]",
            run(
                "ok: Dv cmt 2",
                &cmt2_err,
                &one,
                monitored(vec![MonitorSpec::new("A", 2, ObserveMode::Dv)]),
            )
            .err(),
        ),
        (
            "an Ipred monitor on the compartment without one",
            run(
                "ok: Ipred cmt 1",
                &cmt2_err,
                &one,
                monitored(vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)]),
            )
            .err(),
        ),
        (
            "from_spec, assay_cmt with an [error_model]",
            spec_dv_cmt2.err(),
        ),
    ];
    for (what, err) in &accepted {
        assert!(err.is_none(), "{what} must run: {err:?}");
    }

    // Refused rows made no controller; accepted ones made one per subject, which is what
    // makes the zero above a measurement rather than a constant.
    for (name, n) in &counts {
        let want = match *name {
            "ok: base × reset" => 2,
            n if n.starts_with("ok: ") => 1,
            _ => 0,
        };
        assert_eq!(*n, want, "{name}: controllers made");
    }
    check_rows(rows);
}

/// The refusals the adaptive driver still makes once it runs (#1983): they need the resolved
/// doses, the break timeline or the controller's own output. One row per text, through
/// `simulate_adaptive`, each keeping the run loop's `subject '…' (sim N):` prefix. Each
/// fixture differs from a run that succeeds in the one input named in the row.
#[test]
fn adaptive_driver_refusals_carry_their_codes() {
    use crate::sim::adaptive::{ControllerCtx, DoseAction};
    let ode = parse_model_string(ODE_ADAPTIVE).expect("parse");
    let iov = parse_model_string(ODE_IOV).expect("parse");
    let lagged =
        parse_model_string(&ODE_ADAPTIVE.replace("  V  = TVV\n", "  V  = TVV\n  LAGTIME = 2.0\n"))
            .expect("parse");
    let input_rate = parse_model_string(
        &ODE_ADAPTIVE
            .replace(
                "  theta TVV(50.0, 5.0, 200.0)\n",
                "  theta TVV(50.0, 5.0, 200.0)\n  theta TVKA(1.0, 0.05, 24.0)\n",
            )
            .replace("  V  = TVV\n", "  V  = TVV\n  KA = TVKA\n")
            .replace(
                "d/dt(central) = -(CL / V) * central",
                "d/dt(central) = first_order(ka=KA) - (CL / V) * central",
            ),
    )
    .expect("parse");
    let nan_lag = parse_model_string(
        &ODE_ADAPTIVE.replace("  V  = TVV\n", "  V  = TVV\n  LAGTIME = exp(1000.0)\n"),
    )
    .expect("parse");
    let tad = parse_model_string(&ODE_ADAPTIVE.replace(
        "d/dt(central) = -(CL / V) * central",
        "d/dt(central) = -(CL / V) * central * (1.0 + 0.01 * TAD)",
    ))
    .expect("parse");
    let run = |m: &CompiledModel,
               pop: &Population,
               decision_times: Vec<f64>,
               actions: Vec<DoseAction>| {
        let o = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times,
            ..Default::default()
        };
        simulate_adaptive(
            m,
            pop,
            &m.default_params,
            1,
            || {
                let a = actions.clone();
                move |_: &ControllerCtx| a.clone()
            },
            &o,
        )
    };
    let bolus = |amt: f64, cmt: usize| vec![DoseAction::Bolus { amt, cmt }];
    let one = population_of(vec![dose_free("1", &[12.0])], &[]);
    let mut ss_base = observed("1", &[6.0, 30.0]);
    ss_base.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)];
    let mut reset_at_1 = dose_free("1", &[2.0]);
    reset_at_1.reset_times = vec![1.0];
    check_rows(vec![
        row(
            "base SS dose under IOV",
            run(
                &iov,
                &population_of(vec![ss_base], &[]),
                vec![0.0, 24.0],
                bolus(100.0, 1),
            ),
            "E_ADAPTIVE_UNSUPPORTED",
            None,
            "subject '1' (sim 1): ode_predictions_adaptive: a pre-scheduled base dose (index 0) \
             combined with time-varying covariates or inter-occasion variability must be a \
             plain fixed bolus or infusion into a plain compartment; steady-state, lagged, \
             built-in input-rate (transit / zero-order absorption), and modeled-RATE base \
             doses under a time-varying covariate or IOV are a #930/#931 follow-up",
        ),
        row(
            "infinite base-dose lag",
            run(
                &nan_lag,
                &population_of(vec![observed("1", &[12.0])], &[]),
                vec![0.0],
                vec![DoseAction::Hold],
            ),
            "E_ADAPTIVE_UNSUPPORTED",
            None,
            "subject '1' (sim 1): ode_predictions_adaptive: a non-finite break time \
             (NaN/infinite dose lagtime, route lag, or infusion duration) — the subject's \
             timeline cannot be ordered",
        ),
        row(
            "IOV: decision within 1e-15 of an observation",
            run(
                &iov,
                &population_of(vec![dose_free("1", &[1.0])], &[]),
                vec![next_up(1.0)],
                bolus(100.0, 1),
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "subject '1' (sim 1): adaptive-dosing (time-varying) event time 1.0000000000000002 \
             lies within 1e-15 of another break time but is not bit-identical, so its \
             per-event PK / decision lookup would be silently dropped. Align decision, \
             observation, and EVID=2 times to identical values (integer-valued time grids are \
             unaffected).",
        ),
        row(
            "constant path: decision within 1e-15 of a reset",
            run(
                &ode,
                &population_of(vec![reset_at_1], &[]),
                vec![next_up(1.0)],
                bolus(100.0, 1),
            ),
            "E_ADAPTIVE_OPTIONS",
            None,
            "subject '1' (sim 1): adaptive-dosing decision time 1.0000000000000002 lies within \
             1e-15 of a system-reset or base-regimen (dose / infusion-end) break time but is \
             not bit-identical, so its decision lookup would be silently dropped. Align \
             decision, reset (EVID=3), and base-dose times to identical values \
             (integer-valued time grids are unaffected).",
        ),
        row(
            "controller dose past the states",
            run(&ode, &one, vec![0.0], bolus(100.0, 2)),
            "E_ADAPTIVE_CONTROLLER_ACTION",
            None,
            "subject '1' (sim 1): decision 0: dose into compartment 2 but the model has 1 \
             state(s)",
        ),
        row(
            "controller dose into an input-rate compartment",
            run(&input_rate, &one, vec![0.0], bolus(100.0, 1)),
            "E_ADAPTIVE_CONTROLLER_ACTION",
            None,
            "subject '1' (sim 1): decision 0: compartment 1 is fed by a built-in input-rate \
             (absorption) function; controller dosing into an input-rate compartment is not \
             supported",
        ),
        row(
            "controller dose into a lagged compartment",
            run(&lagged, &one, vec![0.0], bolus(100.0, 1)),
            "E_ADAPTIVE_CONTROLLER_ACTION",
            None,
            "subject '1' (sim 1): decision 0: compartment 1 declares a dose lag time (2); \
             lagged controller dosing is not supported",
        ),
        row(
            "controller dose into compartment 0",
            run(&ode, &one, vec![0.0], bolus(100.0, 0)),
            "E_ADAPTIVE_CONTROLLER_ACTION",
            None,
            "subject '1' (sim 1): decision 0 at t=0: dose target compartment is 0, but CMT is \
             1-based",
        ),
        row(
            "controller dose with a negative amount",
            run(&ode, &one, vec![0.0], bolus(-1.0, 1)),
            "E_ADAPTIVE_CONTROLLER_ACTION",
            None,
            "subject '1' (sim 1): decision 0 at t=0: dose amount must be finite and \
             non-negative; got -1",
        ),
        row(
            "controller infusion with a zero rate",
            run(
                &ode,
                &one,
                vec![0.0],
                vec![DoseAction::Infuse {
                    amt: 100.0,
                    cmt: 1,
                    rate: 0.0,
                }],
            ),
            "E_ADAPTIVE_CONTROLLER_ACTION",
            None,
            "subject '1' (sim 1): decision 0 at t=0: Infuse requires a positive, finite rate; \
             got 0",
        ),
        row(
            "controller Stop before another action",
            run(
                &ode,
                &one,
                vec![0.0],
                vec![DoseAction::Stop, DoseAction::Bolus { amt: 100.0, cmt: 1 }],
            ),
            "E_ADAPTIVE_CONTROLLER_ACTION",
            None,
            "subject '1' (sim 1): decision 0 at t=0: Stop must be the final action, but 1 \
             action(s) follow it",
        ),
        row(
            "TAD read before any dose",
            run(
                &tad,
                &population_of(vec![dose_free("1", &[9.0])], &[]),
                vec![6.0, 12.0],
                vec![DoseAction::Hold],
            ),
            "E_ADAPTIVE_UNSUPPORTED",
            None,
            "subject '1' (sim 1): ode_predictions_adaptive: the segment (9, 12] integrated to \
             a non-finite state, and the [odes] RHS reads `TAD`, which has no referent there: \
             no dose has been given in this run. No observation is read off that segment, but \
             the state integrated there carries into every later read. Where the record \
             contains a dose, the static engines anchor such a window at the first one — a \
             dose a reactive run has not decided yet, and may never decide — so this driver \
             refuses the window rather than read one out of the future. Give the subject a \
             pre-scheduled base regimen, or have the controller dose at a decision at or \
             before the subject's first record; a controller that never doses leaves a model \
             reading `TAD` unanchored for the whole run.",
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
