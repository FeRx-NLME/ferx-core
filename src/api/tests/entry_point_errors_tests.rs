//! Tier-1 tests for #898: a model/data precondition failure on a non-`fit()` entry point is
//! an `Err` carrying **`fit()`'s own text**, not a panic. PR 1 flipped the `_diag` /
//! `_with_options` forms; PR 2 flipped `predict` / `simulate` / `simulate_with_seed`.
//!
//! # What was measured on `main` (`eb0bcf9a`) before this
//!
//! The #375 fixture (an infusion into `CMT=0` of a `two_cpt_oral` model) through four entry
//! points under `catch_unwind`: `fit` → `Err`; `predict_diag` → **panic**;
//! `simulate_with_options` → **panic**; `simulate_with_options_diag` → **panic**. The last two
//! return `Result<_, String>` — a caller matching on the `Err` arm never saw it.
//!
//! # The contract pinned here
//!
//! | Input | Must say | Must not say |
//! |---|---|---|
//! | a failed precondition, `Result` form | exactly what `fit()` says | the old wrapper sentences |
//! | the same, `predict` / `simulate` / `simulate_with_seed` | the same string, as an `Err` | a second wording, a panic |
//! | accepted model | nothing — rows unchanged | — |
//! | `simulate_with_uncertainty`, flip-flop draw | run stays `Ok`, draw skipped | an `Err` naming the draw |
//!
//! # Tiering
//!
//! Tier 1. `fit()` is called, but only on inputs it refuses before its first iteration.

use super::*;
use crate::parser::model_parser::{parse_full_model, parse_model_string};
use crate::types::{DoseEvent, FitOptions, Population, RateMode, Subject};
use std::collections::HashMap;

// ── fixtures ─────────────────────────────────────────────────────────────────

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

/// One subject, three observations on compartment `obs_cmt`, the given doses.
fn subject_with(doses: Vec<DoseEvent>, obs_cmt: usize) -> Subject {
    Subject {
        id: "1".to_string(),
        obs_times: vec![1.0, 4.0, 12.0],
        observations: vec![1.0, 0.8, 0.3],
        obs_cmts: vec![obs_cmt; 3],
        cens: vec![0; 3],
        doses,
        ..Default::default()
    }
}

fn bolus() -> DoseEvent {
    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)
}

const ONE_CPT_IV: &str = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  \
    theta TVV(20.0, 0.001, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n";

/// The #375 fixture: `two_cpt_oral`, an infusion into `CMT=0`. An infusion has no "default
/// compartment" to fall back on, so nothing can route it.
const TWO_CPT_ORAL: &str = "[parameters]\n  theta TVCL(5.0, 0.1, 50.0)\n  \
    theta TVV(50.0, 5.0, 500.0)\n  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)\n  \
    theta TVKA(1.0, 0.01, 10.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n  Q  = TVQ\n  V2 = TVV2\n  \
    KA = TVKA\n[structural_model]\n  pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)\n\
    [error_model]\n  DV ~ proportional(PROP)\n";

fn unroutable() -> (CompiledModel, Population) {
    let model = parse_model_string(TWO_CPT_ORAL).expect("parse");
    let infusion_into_cmt0 = DoseEvent::new(0.0, 100.0, 0, 20.0, false, 0.0);
    let pop = population_of(vec![subject_with(vec![infusion_into_cmt0], 2)], &[]);
    (model, pop)
}

/// What `fit()` says about this input — the reference every other entry point is held to.
fn fit_err(model: &CompiledModel, pop: &Population, params: &ModelParameters) -> String {
    match fit(model, pop, params, &FitOptions::default()) {
        Err(e) => e,
        Ok(_) => panic!("fixture is supposed to be refused by fit()"),
    }
}

/// The three sentences that existed only in the deleted `assert_*` wrappers. None may survive
/// into an `Err`.
fn assert_no_wrapper_text(msg: &str) {
    for banned in [
        "predict()/simulate() received",
        "rather than panicking",
        "panicked",
    ] {
        assert!(!msg.contains(banned), "wrapper text {banned:?} in: {msg}");
    }
}

// ── T1: every flipped entry point returns fit()'s text ───────────────────────

/// The #898 defect on the #375 fixture, entry point by entry point.
///
/// Regression this catches: a precondition checked by `panic!` inside a `Result`-returning
/// function, or reported under a second wording. Mutation — restore a panicking assert at any
/// one site and that arm dies on the unwind; prefix any one `Err` and its `assert_eq!` dies.
/// Each arm names its entry point in the failure so the dead one is identifiable.
#[test]
fn every_flipped_entry_point_returns_the_text_fit_returns() {
    let (model, pop) = unroutable();
    let params = &model.default_params;
    let want = fit_err(&model, &pop, params);
    assert!(
        want.contains("subject 1, time 0: infusion into compartment 0"),
        "fixture no longer trips the #375 check: {want}"
    );
    assert_no_wrapper_text(&want);

    let got = predict_diag(&model, &pop, params).expect_err("predict_diag");
    assert_eq!(got, want, "predict_diag");

    let opts = SimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    let got = simulate_with_options(&model, &pop, params, 1, &opts).expect_err("options");
    assert_eq!(got, want, "simulate_with_options");
    let got = simulate_with_options_diag(&model, &pop, params, 1, &opts).expect_err("diag");
    assert_eq!(got, want, "simulate_with_options_diag");

    let got = crate::suggest_start::inits_from_nca(&model, &pop, crate::NcaInit::Nca)
        .expect_err("inits_from_nca");
    assert_eq!(got, want, "inits_from_nca");

    #[cfg(feature = "survival")]
    {
        let got = predict_survival(&model, &pop, params, &[1.0, 2.0]).expect_err("survival");
        assert_eq!(got, want, "predict_survival");
    }
}

/// `predict_categorical` runs neither dose check, so the #375 fixture cannot reach it; its two
/// preconditions are the time-varying-covariate and endpoint-routing ones. The routing fixture
/// is the repo's own binary example read by the model-blind reader.
#[cfg(feature = "survival")]
#[test]
fn predict_categorical_returns_the_text_fit_returns() {
    let src = std::fs::read_to_string("examples/binary_logistic.ferx").expect("example");
    let model = parse_full_model(&src).expect("parses").model;
    let pop = crate::read_nonmem_csv(std::path::Path::new("data/binary_logistic.csv"), None, None)
        .expect("model-blind read");
    let params = &model.default_params;
    let want = fit_err(&model, &pop, params);
    assert!(want.contains("E_ENDPOINT_UNROUTED"), "{want}");
    let got = predict_categorical(&model, &pop, params).expect_err("predict_categorical");
    assert_eq!(got, want);
    assert_no_wrapper_text(&got);
}

// ── T2: one fixture per check family, through predict_diag ───────────────────

/// A fixture that trips exactly one of `predict_diag`'s precondition families.
struct Family {
    name: &'static str,
    /// A phrase only this family's check message contains.
    phrase: &'static str,
    model: CompiledModel,
    pop: Population,
    /// `None` → the model's defaults. The flip-flop family is a function of θ.
    theta: Option<Vec<f64>>,
}

const TWINLESS_TRANSIT: &str = "[parameters]\n  theta TVCL(0.5, 0.001, 50.0)\n  \
    theta TVV(4.0, 0.1, 500.0)\n  theta TVNTR(3.0, 0.0, 20.0)\n  theta TVMTT(20.0, 0.05, 200.0)\n  \
    omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n[individual_parameters]\n  \
    CL = TVCL * exp(ETA_CL)\n  V = TVV\n  NTR = TVNTR\n  MTT = TVMTT\n[structural_model]\n  \
    pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n[scaling]\n  obs_scale = 1\n\
    [error_model]\n  DV ~ proportional(PROP)\n";

const DEPOT_READOUT: &str = "[parameters]\n  theta TVCL(5.0, 0.1, 50.0)\n  \
    theta TVV(50.0, 5.0, 500.0)\n  theta TVKA(1.0, 0.01, 10.0)\n  omega ETA_CL ~ 0.09\n  \
    sigma PROP ~ 0.01 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  \
    KA = TVKA\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[scaling]\n  \
    y = (central + depot) / V\n[error_model]\n  DV ~ proportional(PROP)\n";

/// Two `igd()` pathways whose fractions are both `TVFR1 = 0.6`, so Σ = 1.2: structurally valid,
/// value-malformed (#588).
const FRACTIONS_SUM_TO_1_2: &str = "[parameters]\n  theta TVCL(5.0, 0.1, 100.0)\n  \
    theta TVV(50.0, 5.0, 500.0)\n  theta TVMAT1(1.0, 0.05, 24.0)\n  \
    theta TVMAT2(4.0, 0.05, 24.0)\n  theta TVCV2_1(0.3, 0.001, 10.0)\n  \
    theta TVCV2_2(0.5, 0.001, 10.0)\n  theta TVFR1(0.6, 0.001, 0.999)\n  omega ETA_CL ~ 0.09\n  \
    sigma PROP ~ 0.01 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV\n  \
    MAT1 = TVMAT1\n  MAT2 = TVMAT2\n  CV2_1 = TVCV2_1\n  CV2_2 = TVCV2_2\n  FR1 = TVFR1\n  \
    FR2 = TVFR1\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
    d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) + FR2*igd(mat=MAT2, cv2=CV2_2) - CL/V*central\n\
    [error_model]\n  DV ~ proportional(PROP)\n";

const UNBOUND_COVARIATE_MODEL: &str = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  \
    theta TVV(40.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.02\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    pk one_cpt_iv(cl=CL, v=V)\n[covariates]\n  WT continuous\n[covariate_model]\n  \
    CL ~ WT power(center = median)\n[error_model]\n  DV ~ proportional(PROP)\n";

const READS_WT: &str = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  \
    theta TVV(20.0, 0.001, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n\
    [individual_parameters]\n  CL = TVCL * (WT / 70) * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n";

fn families() -> Vec<Family> {
    let iv = || parse_model_string(ONE_CPT_IV).expect("parse");
    let one = |s: Subject| population_of(vec![s], &[]);
    let mut out = vec![
        Family {
            name: "modeled dose rates (#324)",
            phrase: "RATE=-2 (modeled infusion duration) into compartment 1",
            model: iv(),
            pop: one(subject_with(
                vec![DoseEvent::modeled(
                    0.0,
                    100.0,
                    1,
                    false,
                    0.0,
                    RateMode::ModeledDuration,
                )],
                1,
            )),
            theta: None,
        },
        Family {
            name: "covariates present (#1028)",
            phrase: "Model references covariate(s) not found in data (case-sensitive): WT",
            model: parse_model_string(READS_WT).expect("parse"),
            pop: one(subject_with(vec![bolus()], 1)),
            theta: None,
        },
        Family {
            name: "covariate model bound (#1111)",
            phrase: "bind_covariate_stats",
            model: parse_full_model(UNBOUND_COVARIATE_MODEL)
                .expect("parse")
                .model,
            pop: {
                let mut s = subject_with(vec![bolus()], 1);
                s.covariates = HashMap::from([("WT".to_string(), 70.0)]);
                population_of(vec![s], &["WT"])
            },
            theta: None,
        },
        {
            let (model, pop) = unroutable();
            Family {
                name: "dose compartments (#375)",
                phrase: "infusion into compartment 0",
                model,
                pop,
                theta: None,
            }
        },
        Family {
            name: "absorption closed-form support",
            phrase: "does not support infusion doses in this form (subject 1)",
            model: parse_model_string(TWINLESS_TRANSIT).expect("parse"),
            pop: one(subject_with(
                vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)],
                1,
            )),
            theta: None,
        },
        Family {
            // ke = CL/V = 2/4 = 0.5 ≥ KTR = (n+1)/MTT = 4/20 = 0.2: the flip-flop regime, on a
            // model the inert `[scaling]` block leaves without an ODE twin to reroute to.
            name: "flip-flop, no twin (#776)",
            phrase: "flip-flop",
            model: parse_model_string(TWINLESS_TRANSIT).expect("parse"),
            pop: one(subject_with(vec![bolus()], 1)),
            theta: Some(vec![2.0, 4.0, 3.0, 20.0]),
        },
        Family {
            name: "analytic readout support (#650)",
            phrase: "EVID=3/4 reset",
            model: parse_model_string(DEPOT_READOUT).expect("parse"),
            pop: {
                let mut s = subject_with(vec![bolus()], 2);
                s.reset_times = vec![2.0];
                s.reset_covariates = vec![HashMap::new()];
                s.reset_occasions = vec![1];
                one(s)
            },
            theta: None,
        },
        Family {
            name: "absorption dosing (#588)",
            phrase: "Pathway fractions on compartment",
            model: parse_full_model(FRACTIONS_SUM_TO_1_2).expect("parse").model,
            pop: one(subject_with(vec![bolus()], 1)),
            theta: None,
        },
    ];
    #[cfg(feature = "survival")]
    out.extend(survival_families());
    out
}

#[cfg(feature = "survival")]
fn survival_families() -> Vec<Family> {
    use crate::types::{EventType, ObsRecord};
    // A time-varying covariate on an analytic hazard (#741): it would be frozen at baseline.
    let tv_model = parse_model_string(
        "[parameters]\n  theta TVLAMBDA(0.05, 0.001, 10.0)\n  theta TVBETA(0.1, -5.0, 5.0)\n  \
         omega ETA_LAMBDA ~ 0.09\n[event_model]\n  cmt    = 2\n  family = exponential\n  \
         scale  = TVLAMBDA * exp(ETA_LAMBDA)\n  loghr  = TVBETA * CRCL\n",
    )
    .expect("parse");
    let tv_subject = Subject {
        id: "1".to_string(),
        covariates: HashMap::from([("CRCL".to_string(), 100.0)]),
        obs_covariates: vec![
            HashMap::from([("CRCL".to_string(), 100.0)]),
            HashMap::from([("CRCL".to_string(), 60.0)]),
        ],
        obs_records: vec![ObsRecord::Event {
            time: 30.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 2,
        }],
        ..Default::default()
    };
    let src = std::fs::read_to_string("examples/binary_logistic.ferx").expect("example");
    vec![
        Family {
            name: "survival time-varying covariate (#741)",
            phrase: "#741",
            model: tv_model,
            pop: population_of(vec![tv_subject], &["CRCL"]),
            theta: None,
        },
        Family {
            name: "endpoint routing (#1199)",
            phrase: "E_ENDPOINT_UNROUTED",
            model: parse_full_model(&src).expect("parses").model,
            pop: crate::read_nonmem_csv(
                std::path::Path::new("data/binary_logistic.csv"),
                None,
                None,
            )
            .expect("model-blind read"),
            theta: None,
        },
    ]
}

/// Ten precondition families, ten `?` lines in `predict_diag`. Each fixture trips one of them,
/// and each must come back as that family's own message — which is also what `fit()` says.
///
/// Regression this catches: a family's check dropped from `predict_diag`, so the input reaches
/// the predictor (wrong rows, or a panic from deep inside the walk). Mutation — delete any one
/// `?` line and that family's row returns `Ok` (or a different family's text); the failure
/// names the family.
#[test]
fn each_precondition_family_is_an_err_carrying_its_own_message() {
    let families = families();
    #[cfg(feature = "survival")]
    assert_eq!(families.len(), 10, "one fixture per `?` in predict_diag");
    // Without `survival` two of the ten `?` lines cannot fire at all, so no fixture can exist
    // for them: the time-varying-covariate one is `#[cfg]`-gated at the call, and the
    // endpoint-routing one is *not* gated but calls `check_endpoint_routing`, whose
    // non-`survival` definition returns an empty list. Deleting either `?` is invisible to a
    // plain-`ci` build because it is inert there, not because it is untested.
    #[cfg(not(feature = "survival"))]
    assert_eq!(
        families.len(),
        8,
        "the two families whose checks are inert without `survival`"
    );

    for f in &families {
        let mut params = f.model.default_params.clone();
        if let Some(theta) = &f.theta {
            params.theta = theta.clone();
        }
        let got = match predict_diag(&f.model, &f.pop, &params) {
            Err(e) => e,
            Ok(out) => panic!("{}: accepted, {} rows", f.name, out.results.len()),
        };
        assert!(got.contains(f.phrase), "{}: {got}", f.name);
        assert_no_wrapper_text(&got);
        assert_eq!(
            got,
            fit_err(&f.model, &f.pop, &params),
            "{} vs fit()",
            f.name
        );
    }
}

// ── which entry point checks which family (docs/warnings.qmd#entry-point-errors) ──

/// Does each narrower entry point refuse this family? `None` = do not call it (see the one
/// use). Order: `simulate_with_options`, `inits_from_nca`, `predict_survival`,
/// `predict_categorical`.
fn checked_by(family: &str) -> [Option<bool>; 4] {
    const Y: Option<bool> = Some(true);
    const N: Option<bool> = Some(false);
    match family {
        "modeled dose rates (#324)" => [Y, Y, N, N],
        "covariates present (#1028)" => [Y, N, N, N],
        "covariate model bound (#1111)" => [Y, N, N, N],
        "dose compartments (#375)" => [Y, Y, Y, N],
        "absorption closed-form support" => [Y, N, N, N],
        "flip-flop, no twin (#776)" => [Y, N, N, N],
        // `simulate*` has never run `check_analytic_readout_support`, and on this fixture the
        // emitter then trips a `debug_assert!` in `pk::analytical_state_at_times` (measured; in
        // release it returns rows instead). Pre-existing and filed separately — not called
        // here, so this test does not pin a panic as the contract.
        "analytic readout support (#650)" => [None, N, N, N],
        "absorption dosing (#588)" => [Y, N, N, N],
        "survival time-varying covariate (#741)" => [Y, N, Y, Y],
        "endpoint routing (#1199)" => [Y, N, N, Y],
        other => panic!("no row for family {other:?}"),
    }
}

/// `docs/warnings.qmd#entry-point-errors` tells a caller which preconditions each entry point
/// checks — and therefore which ones it does **not**, where a bad input comes back as `Ok`.
/// That table was first written as "all nine entry points return `Err`", which was false for
/// four of them (review of #1487); this is its enumerated input space, entry point × family,
/// measured rather than read.
///
/// Regression this catches, both directions: a check added to a narrow entry point flips an
/// `Ok` cell to `Err` (the docs then under-promise), and a check dropped flips an `Err` cell to
/// `Ok` (they over-promise). Mutation — add `check_modeled_dose_rates` to
/// `predict_categorical`, or drop `check_dose_compartments` from `inits_from_nca`; either dies
/// here naming the cell. And a `Y` cell is held to the family's `phrase`, so a refusal through
/// some *other* check is not mistaken for this one — mutation: keep the simulate flip-flop
/// check but replace its message.
#[test]
fn each_narrow_entry_point_refuses_exactly_the_families_the_docs_list() {
    let opts = SimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    for f in families() {
        let mut params = f.model.default_params.clone();
        if let Some(theta) = &f.theta {
            params.theta = theta.clone();
        }
        // A `Y` cell must refuse **for this family's own reason**: `is_err()` alone is green
        // when the entry point refuses the fixture through some other check (review r2, V4).
        let cell = |entry: &str, want: bool, got: Option<String>| {
            assert_eq!(got.is_some(), want, "{entry} × {}: {got:?}", f.name);
            if let Some(msg) = got {
                assert!(msg.contains(f.phrase), "{entry} × {}: {msg}", f.name);
            }
        };
        let [sim, nca, surv, cat] = checked_by(f.name);
        if let Some(want) = sim {
            let got = simulate_with_options(&f.model, &f.pop, &params, 1, &opts).err();
            cell("simulate_with_options", want, got);
        }
        if let Some(want) = nca {
            let got =
                crate::suggest_start::inits_from_nca(&f.model, &f.pop, crate::NcaInit::Nca).err();
            cell("inits_from_nca", want, got);
        }
        #[cfg(feature = "survival")]
        {
            if let Some(want) = surv {
                let got = predict_survival(&f.model, &f.pop, &params, &[1.0, 2.0]).err();
                cell("predict_survival", want, got);
            }
            if let Some(want) = cat {
                let got = predict_categorical(&f.model, &f.pop, &params).err();
                cell("predict_categorical", want, got);
            }
        }
        #[cfg(not(feature = "survival"))]
        let _ = (surv, cat);
    }
}

// ── T3: simulate_with_uncertainty ────────────────────────────────────────────

/// A fit result good enough to draw parameter sets from: the model's defaults as the
/// estimate, a small diagonal covariance.
fn synthetic_fit(model: &CompiledModel) -> FitResult {
    super::simulate_with_uncertainty_tests::synthetic_fit(&model.default_params)
}

/// (a) of T3: a precondition failure reaches the caller as an `Err`, from a function that
/// already returned `Result` and used to panic at the per-draw chokepoint instead.
///
/// Mutation — drop the `?` on the chokepoint call in `simulate_with_uncertainty` and this no
/// longer compiles; swallow the `Err` instead (`unwrap_or_default()`) and it returns `Ok`.
/// (b) — a flip-flop *draw* is still skipped rather than failing the run — is pinned by
/// `uncertainty_skips_flip_flop_draws_without_panicking`, which dies if the #786 check moves
/// behind that `?`.
#[test]
fn uncertainty_precondition_failure_is_an_err() {
    let (model, pop) = unroutable();
    let fit_result = synthetic_fit(&model);
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 3,
        n_sim_per_draw: 1,
        seed: Some(7),
        ..Default::default()
    };
    let got = simulate_with_uncertainty(&model, &pop, &fit_result, &opts)
        .expect_err("an unroutable dose must fail the run");
    assert_eq!(got, fit_err(&model, &pop, &model.default_params));
}

// ── T4: the formerly `Vec`-returning forms return the same `Err` ───────────────

/// `predict` / `simulate` / `simulate_with_seed` return `Result` since #898 PR 2; they used to
/// re-raise the `Err` text as a panic. Now they return it — the same string as the `_diag` /
/// `_with_options` forms, so there is one wording per condition and no panic on the way.
///
/// Mutation — reintroduce `unwrap_or_else(|e| panic!("{e}"))` in any wrapper and this test dies
/// on the unwind; prefix the message (`Err(format!("simulate: {e}"))`) and the `assert_eq!` dies.
#[test]
fn convenience_forms_return_exactly_the_err_text() {
    let (model, pop) = unroutable();
    let params = &model.default_params;
    let want = predict_diag(&model, &pop, params).expect_err("fixture is refused");

    let got = predict(&model, &pop, params).expect_err("predict");
    assert_eq!(got, want, "predict");
    let got = simulate(&model, &pop, params, 1).expect_err("simulate");
    assert_eq!(got, want, "simulate");
    let got = simulate_with_seed(&model, &pop, params, 1, 3).expect_err("simulate_with_seed");
    assert_eq!(got, want, "simulate_with_seed");
}

/// The rule after #898 PR 2: no library entry point turns a precondition `Err` back into a
/// panic. Two idioms did that — the `unwrap_or_else(|e| panic!("{e}"))` re-raise the old
/// `Vec` wrappers used, and an internal caller unwrapping one of the flipped functions (the
/// `inits_from_nca` sweep did, and would have panicked out of a `Result`-returning function).
/// Scans every non-test source under `src/` except the dev binaries in `src/bin`.
///
/// Mutation — put `.unwrap_or_else(|e| panic!("{e}"))` back in `predict`, or `.unwrap()` on the
/// sweep's `predict(..)?`, and this names the file and line.
#[test]
fn no_library_code_re_raises_a_precondition_err_as_a_panic() {
    fn rust_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("readable dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources(&root.join("src"), &mut files);

    let unwrapped_call =
        regex::Regex::new(r"(^|[^\w.])(predict|simulate|simulate_with_seed)\(.*\)\.unwrap\(\)")
            .expect("valid regex");
    let mut scanned = 0;
    let mut hits = Vec::new();
    for file in &files {
        let rel = file
            .strip_prefix(&root)
            .expect("under the manifest dir")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.ends_with("_tests.rs") || rel.contains("/tests/") || rel.starts_with("src/bin/") {
            continue;
        }
        scanned += 1;
        let text = std::fs::read_to_string(file).expect("readable source");
        // Stop at an inline `mod tests {` — test code may unwrap freely.
        for (n, line) in text
            .lines()
            .take_while(|l| !(l.starts_with("mod ") && l.contains("test") && l.ends_with('{')))
            .enumerate()
        {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            if code.contains("panic!(\"{e}\")") || unwrapped_call.is_match(code) {
                hits.push(format!("{rel}:{}: {code}", n + 1));
            }
        }
    }
    assert!(
        scanned > 100,
        "the scan saw only {scanned} files — it is measuring the walk, not the code"
    );
    assert!(
        hits.is_empty(),
        "precondition Err re-raised as a panic:\n{}",
        hits.join("\n")
    );
}

// ── T5: an accepted model's rows do not move ─────────────────────────────────

fn accepted_population() -> Population {
    let subjects = (0..3)
        .map(|i| Subject {
            id: format!("{}", i + 1),
            ..subject_with(vec![bolus()], 1)
        })
        .collect();
    population_of(subjects, &[])
}

const ONE_CPT_ODE: &str = "[parameters]\n  theta TVCL(1.0, 0.1, 50.0)\n  \
    theta TVV(10.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.04\n  sigma PROP ~ 0.04\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n\
    [scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP)\n";

/// On an accepted model the change is invisible: the `Result` forms are `Ok`, and their rows
/// are the rows the convenience forms return, bit for bit. `predict_diag` is additionally held to
/// the predictor called directly, so a row perturbed inside `predict_diag` — which the wrapper
/// would inherit — still reddens this. Analytic and `[odes]`, since the two take different
/// paths through both entry points.
///
/// What this cannot see is a change inside the shared simulate chokepoint, which both simulate
/// forms inherit; the PR records a one-off before/after `to_bits` measurement for that.
#[test]
fn accepted_model_rows_are_bit_identical_across_the_result_and_vec_forms() {
    for src in [ONE_CPT_IV, ONE_CPT_ODE] {
        let model = parse_model_string(src).expect("parse");
        let pop = accepted_population();
        let params = &model.default_params;

        let diag = predict_diag(&model, &pop, params)
            .expect("accepted")
            .results;
        let plain = predict(&model, &pop, params).unwrap();
        let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
        let direct: Vec<f64> = pop
            .subjects
            .iter()
            .flat_map(|s| {
                crate::pk::compute_predictions_with_tv(&model, s, &params.theta, &zero_eta)
            })
            .collect();
        assert_eq!(diag.len(), 9);
        assert_eq!(plain.len(), diag.len());
        assert_eq!(direct.len(), diag.len());
        for ((d, p), e) in diag.iter().zip(&plain).zip(&direct) {
            assert!(d.pred.is_finite() && d.pred > 0.0, "{}", d.pred);
            assert_eq!(d.pred.to_bits(), p.pred.to_bits());
            assert_eq!(d.pred.to_bits(), e.to_bits());
        }

        let opts = SimulateOptions {
            seed: Some(11),
            ..Default::default()
        };
        let via_result = simulate_with_options(&model, &pop, params, 2, &opts).expect("accepted");
        let via_vec = simulate_with_seed(&model, &pop, params, 2, 11).unwrap();
        assert_eq!(via_result.len(), 18);
        assert_eq!(via_vec.len(), via_result.len());
        for (a, b) in via_result.iter().zip(&via_vec) {
            let (SimOutcome::Continuous { value: va }, SimOutcome::Continuous { value: vb }) =
                (&a.outcome, &b.outcome)
            else {
                panic!("continuous rows expected");
            };
            assert!(va.is_finite());
            assert_eq!(va.to_bits(), vb.to_bits());
            assert_eq!(a.ipred.to_bits(), b.ipred.to_bits());
        }
    }
}
