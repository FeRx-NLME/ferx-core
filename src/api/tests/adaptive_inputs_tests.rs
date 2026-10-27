//! Supplied per-subject η on the adaptive path (#1594 PR-B): `SuppliedEtas`,
//! `AdaptiveInputs`, `UnmatchedSubjects`.
//!
//! The end-to-end oracle against a fit's own IPRED (B1) needs `fit()` and lives in
//! `tests/adaptive_supplied_etas.rs`. These are the fast pieces: the η stream (B2),
//! replicates and the assay σ (B3), the κ stream (B4) and the refusals (B5).

use super::*;
use crate::diagnostics::EngineError;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};
use std::collections::HashMap;

// 1-cpt IV ODE with BSV on CL, so a subject's IPRED reads its η.
const ODE_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
"#;

// IOV model whose IPRED reads κ only: ETA_V is declared but unread, so a supplied η
// cannot move the prediction and any IPRED change is a change of κ.
const ODE_IOV_ETA_UNREAD: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(KAPPA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
"#;

const DECISIONS: [f64; 2] = [0.0, 24.0];
const OBS: [f64; 4] = [4.0, 20.0, 28.0, 44.0];

/// One subject on the base regimen: 100 at each decision time.
fn subject(id: &str) -> Subject {
    let doses: Vec<DoseEvent> = DECISIONS
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    let n = OBS.len();
    let n_dose = doses.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.to_string(),
        doses,
        obs_times: OBS.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: vec![1u32; n],
        obs_l2: Vec::new(),
        dose_occasions: vec![1u32; n_dose],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
    }
}

fn population(ids: &[&str]) -> Population {
    Population {
        subjects: ids.iter().map(|id| subject(id)).collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn hold_all() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |_ctx: &ControllerCtx| vec![DoseAction::Hold]
}

fn options(seed: u64, mode: ObserveMode) -> AdaptiveSimulateOptions {
    let mut o = AdaptiveSimulateOptions::default();
    o.seed = Some(seed);
    o.decision_times = DECISIONS.to_vec();
    o.monitors = vec![MonitorSpec::new("CONC", 1, mode)];
    o
}

fn etas(rows: &[(&str, f64)]) -> SuppliedEtas {
    SuppliedEtas::from_table(
        rows.iter().map(|(id, _)| id.to_string()).collect(),
        vec!["ETA_CL".to_string()],
        rows.iter().map(|&(_, e)| vec![e]).collect(),
    )
    .expect("valid table")
}

/// IPRED bits of `id`'s rows in replicate `sim`.
fn ipred(res: &AdaptiveSimulationResult, id: &str, sim: usize) -> Vec<u64> {
    let v: Vec<u64> = res
        .trajectories
        .iter()
        .filter(|r| r.id == id && r.sim == sim)
        .map(|r| r.ipred.to_bits())
        .collect();
    assert_eq!(v.len(), OBS.len(), "rows of subject {id}, replicate {sim}");
    v
}

/// B2: a supplied subject still consumes its draw from the η stream. Subjects 1 and
/// 3 must get the η an all-drawn run with the same seed gives them, in both
/// replicates, while subject 2 (supplied) must not.
#[test]
fn supplied_subject_keeps_the_eta_stream_of_the_others() {
    let model = parse_model_string(ODE_IIV).expect("parse");
    let pop = population(&["1", "2", "3"]);
    let opts = options(5, ObserveMode::Ipred);
    let drawn = simulate_adaptive(&model, &pop, &model.default_params, 2, hold_all, &opts)
        .expect("all drawn");
    let e = etas(&[("2", 0.5)]);
    let inputs = AdaptiveInputs::new(&model.default_params)
        .with_etas(&e)
        .with_unmatched(UnmatchedSubjects::Draw);
    let mixed =
        simulate_adaptive_with_inputs(&model, &pop, &inputs, 2, hold_all, &opts).expect("mixed");

    for sim in 1..=2 {
        for id in ["1", "3"] {
            assert_eq!(
                ipred(&mixed, id, sim),
                ipred(&drawn, id, sim),
                "subject {id}, replicate {sim}: a supplied subject 2 shifted the η stream"
            );
        }
        // Straddle: subject 2 is the one that changed, so the comparison above is not
        // between two runs that ignore the supplied η.
        assert_ne!(
            ipred(&mixed, "2", sim),
            ipred(&drawn, "2", sim),
            "subject 2, replicate {sim}: its supplied η was not used"
        );
    }
    let flags: Vec<(&str, usize, bool)> = mixed
        .metrics
        .iter()
        .map(|m| (m.subject.as_str(), m.sim, m.eta_supplied))
        .collect();
    assert_eq!(
        flags,
        vec![
            ("1", 1, false),
            ("2", 1, true),
            ("3", 1, false),
            ("1", 2, false),
            ("2", 2, true),
            ("3", 2, false),
        ]
    );
    assert!(drawn.metrics.iter().all(|m| !m.eta_supplied));
}

// Two η, on CL and V, so a row read in the wrong slot order moves the prediction.
const ODE_TWO_ETAS: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
"#;

/// The supplied η is the subject's η, slot by slot: a run at (η_CL, η_V) = (0.5,
/// −0.3) equals the 1-cpt bolus closed form at those values. Read in the wrong slot
/// order, CL and V swap their η and the prediction moves.
#[test]
fn supplied_eta_is_the_eta_the_run_uses() {
    let model = parse_model_string(ODE_TWO_ETAS).expect("parse");
    let pop = population(&["1"]);
    let e = SuppliedEtas::from_table(
        vec!["1".into()],
        vec!["ETA_CL".into(), "ETA_V".into()],
        vec![vec![0.5, -0.3]],
    )
    .expect("table");
    let inputs = AdaptiveInputs::new(&model.default_params).with_etas(&e);
    let res = simulate_adaptive_with_inputs(
        &model,
        &pop,
        &inputs,
        1,
        hold_all,
        &options(1, ObserveMode::Ipred),
    )
    .expect("run");
    let (cl, v) = (5.0 * 0.5f64.exp(), 50.0 * (-0.3f64).exp());
    let k = cl / v;
    assert_eq!(res.trajectories.len(), OBS.len());
    for r in &res.trajectories {
        let want: f64 = DECISIONS
            .iter()
            .filter(|&&td| td <= r.time)
            .map(|&td| 100.0 / v * (-k * (r.time - td)).exp())
            .sum();
        assert!(r.ipred.is_finite(), "t = {}: IPRED {}", r.time, r.ipred);
        // Measured worst relative error 2.1e-16 (macOS, 2026-10-19); the bound keeps
        // ~4 decades of headroom.
        let rel = (r.ipred - want).abs() / want;
        assert!(
            rel <= 1e-12,
            "t = {}: IPRED {} vs closed form {} at (η_CL, η_V) = (0.5, -0.3)",
            r.time,
            r.ipred,
            want
        );
    }
}

/// B3: a supplied η stays fixed across replicates while the assay is redrawn, and
/// the assay reads σ from the run's `params`, not the model's initial σ.
#[test]
fn supplied_eta_is_fixed_across_replicates_and_the_assay_reads_params_sigma() {
    let model = parse_model_string(ODE_IIV).expect("parse");
    let pop = population(&["1"]);
    let e = etas(&[("1", 0.3)]);
    let run = |sigma: f64, mode: ObserveMode| {
        let mut params = model.default_params.clone();
        params.sigma.values = vec![sigma];
        let inputs = AdaptiveInputs::new(&params).with_etas(&e);
        simulate_adaptive_with_inputs(&model, &pop, &inputs, 2, hold_all, &options(9, mode))
            .expect("run")
    };
    let a = run(0.04, ObserveMode::Dv);
    assert_eq!(
        ipred(&a, "1", 1),
        ipred(&a, "1", 2),
        "the supplied η was redrawn between replicates"
    );
    let signal = |res: &AdaptiveSimulationResult, sim: usize| -> Vec<f64> {
        res.decisions
            .iter()
            .filter(|d| d.sim == sim)
            .map(|d| d.observed_signals[0].value)
            .collect()
    };
    // Straddle: the assay is redrawn, so the equality above is not between two copies
    // of one replicate.
    assert_ne!(signal(&a, 1), signal(&a, 2), "the assay was not redrawn");

    // σ: the same seed draws the same standard normal z on the assay substream, and
    // the proportional assay noise `value − f` is linear in σ, so tripling σ triples
    // it. The initial σ is 0.04, so a run that read it instead of `params` would give
    // a ratio of 1, not 3.
    let latent = run(0.04, ObserveMode::Ipred);
    let b = run(0.12, ObserveMode::Dv);
    let mut compared = 0;
    let mut worst = 0.0f64;
    for sim in 1..=2 {
        for ((f, va), vb) in signal(&latent, sim)
            .iter()
            .zip(signal(&a, sim))
            .zip(signal(&b, sim))
        {
            if *f == 0.0 {
                continue; // before the first dose: nothing to scale
            }
            let ratio = (vb - f) / (va - f);
            assert!(ratio.is_finite(), "replicate {sim}: ratio {ratio}");
            worst = worst.max((ratio - 3.0).abs());
            compared += 1;
        }
    }
    // Measured: 2 decisions compared, worst |ratio − 3| = 4.4e-15 (macOS, 2026-10-19).
    assert_eq!(compared, 2, "one dosed decision per replicate");
    assert!(worst < 1e-12, "assay noise ratio off 3 by {worst}");
}

/// B4: on an IOV model the κ stream does not move when η is supplied. The model's
/// IPRED reads κ only, so equal IPRED is equal κ.
#[test]
fn supplied_eta_leaves_the_kappa_stream_alone() {
    let model = parse_model_string(ODE_IOV_ETA_UNREAD).expect("parse");
    let pop = population(&["1", "2"]);
    let opts = options(13, ObserveMode::Ipred);
    let drawn =
        simulate_adaptive(&model, &pop, &model.default_params, 2, hold_all, &opts).expect("drawn");
    let e = SuppliedEtas::from_table(
        vec!["1".into(), "2".into()],
        vec!["ETA_V".into()],
        vec![vec![0.7], vec![-0.4]],
    )
    .expect("table");
    let inputs = AdaptiveInputs::new(&model.default_params).with_etas(&e);
    let supplied =
        simulate_adaptive_with_inputs(&model, &pop, &inputs, 2, hold_all, &opts).expect("run");
    for sim in 1..=2 {
        for id in ["1", "2"] {
            assert_eq!(
                ipred(&supplied, id, sim),
                ipred(&drawn, id, sim),
                "subject {id}, replicate {sim}: κ moved when η was supplied"
            );
        }
    }
    // Straddle: κ is live, so equal IPRED is not two κ-free runs agreeing.
    assert_ne!(ipred(&drawn, "1", 1), ipred(&drawn, "1", 2), "κ not drawn");
}

/// The table accessors: IDs in the order given, a lookup by ID.
#[test]
fn from_table_keeps_order_and_looks_up_by_id() {
    let e = etas(&[("7", 0.2), ("3", -0.1)]);
    assert_eq!(e.ids(), ["7".to_string(), "3".to_string()]);
    assert_eq!(e.names(), ["ETA_CL".to_string()]);
    assert_eq!(e.get("3"), Some(&[-0.1][..]));
    assert_eq!(e.get("7"), Some(&[0.2][..]));
    assert_eq!(e.get("1"), None);
}

fn refusal(r: Result<impl std::fmt::Debug, EngineError>) -> (String, String) {
    let e = r.expect_err("refused");
    let code = e.diagnostic().map(|d| d.code.clone()).unwrap_or_default();
    (code, e.to_string())
}

/// B5: every refusal, by code and by its full text, so deleting any sentence of a
/// message fails here.
#[test]
fn supplied_eta_refusals_carry_their_code_and_text() {
    let model = parse_model_string(ODE_IIV).expect("parse");
    let names = || vec!["ETA_CL".to_string()];
    let code = "E_SUPPLIED_ETAS".to_string();

    assert_eq!(
        refusal(SuppliedEtas::from_table(
            vec!["1".into(), "2".into()],
            names(),
            vec![vec![0.1]]
        )),
        (
            code.clone(),
            "2 subject ID(s) but 1 row(s) of η".to_string()
        )
    );
    assert_eq!(
        refusal(SuppliedEtas::from_table(
            vec!["1".into()],
            names(),
            vec![vec![0.1, 0.2]]
        )),
        (
            code.clone(),
            "subject '1' has 2 η but 1 are named; a row gives every η".to_string()
        )
    );
    assert_eq!(
        refusal(SuppliedEtas::from_table(
            vec!["1".into()],
            names(),
            vec![vec![f64::NAN]]
        )),
        (
            code.clone(),
            "subject '1' has a non-finite ETA_CL = NaN".to_string()
        )
    );
    assert_eq!(
        refusal(SuppliedEtas::from_table(
            vec!["1".into(), "1".into()],
            names(),
            vec![vec![0.1], vec![0.2]]
        )),
        (
            code.clone(),
            "subject ID '1' appears more than once".to_string()
        )
    );
    // `from_fit`'s own refusal (a fit of another model) is in
    // `tests/adaptive_supplied_etas.rs`, which has a fit to hand it.

    let pop = population(&["1", "2", "3"]);
    let opts = options(1, ObserveMode::Ipred);
    let run = |e: &SuppliedEtas, policy: UnmatchedSubjects| {
        let inputs = AdaptiveInputs::new(&model.default_params)
            .with_etas(e)
            .with_unmatched(policy);
        simulate_adaptive_with_inputs(&model, &pop, &inputs, 1, hold_all, &opts)
    };
    let wrong_names =
        SuppliedEtas::from_table(vec!["1".into()], vec!["ETA_V".into()], vec![vec![0.1]])
            .expect("table");
    assert_eq!(
        refusal(run(&wrong_names, UnmatchedSubjects::Draw)),
        (
            code.clone(),
            "the supplied η are named [ETA_V] but the model's are [ETA_CL]; supply every η \
             of the model, in the model's order"
                .to_string()
        )
    );

    // Unmatched: refused by default, both sides of the policy in one test.
    let only_2 = etas(&[("2", 0.1), ("9", 0.0)]);
    assert_eq!(
        refusal(run(&only_2, UnmatchedSubjects::default())),
        (
            code.clone(),
            "2 simulated subject(s) have no supplied η: '1', '3'. Supply their η, or pass \
             `UnmatchedSubjects::Draw` to draw theirs from Ω"
                .to_string()
        )
    );
    // A supplied subject absent from the simulation ('9') is ignored.
    assert!(run(&only_2, UnmatchedSubjects::Draw).is_ok());

    // The from_spec entry refuses through the same gate.
    let spec_src = format!(
        "{ODE_IIV}[adaptive_dosing]\n  observe = central / V\n  at = [0, 24]\n  \
         start_dose = 100\n  route = bolus(cmt=1)\n  dose_bounds = [0, 1000]\n  \
         when signal >= 0 : hold\n"
    );
    let parsed = crate::parser::model_parser::parse_full_model(&spec_src).expect("parse");
    let spec = parsed.adaptive_dosing.as_ref().expect("block");
    let mut spec_opts = AdaptiveSimulateOptions::default();
    spec_opts.seed = Some(1);
    let inputs = AdaptiveInputs::new(&parsed.model.default_params).with_etas(&only_2);
    let (c, _) = refusal(simulate_adaptive_from_spec_with_inputs(
        &parsed.model,
        &pop,
        &inputs,
        1,
        spec,
        &spec_opts,
    ));
    assert_eq!(c, code);
}

// Two-class mixture on CL. Class 2 clears 4× faster, so a class-2 subject's η̂
// simulated under class 1 is a wrong number, not a rounding difference.
const ODE_MIXTURE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta MIXL(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[mixture]
  nsub = 2
  logit(1) = MIXL
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL * exp(ETA_CL) else 4 * TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
"#;

/// #2162 review, finding 1: the adaptive run draws each subject's mixture class (#2163)
/// rather than taking the class its η̂ was estimated under, so supplied η on a
/// `[mixture]` model are refused. Both sides of the gate: the same η on the same
/// population run on the class-free model.
#[test]
fn supplied_eta_on_a_mixture_model_is_refused() {
    let mixture = parse_model_string(ODE_MIXTURE).expect("parse mixture");
    assert!(mixture.mixture.is_some(), "the fixture is a mixture model");
    let plain = parse_model_string(ODE_IIV).expect("parse");
    let pop = population(&["1"]);
    let e = etas(&[("1", 0.2)]);
    let opts = options(1, ObserveMode::Ipred);
    let run = |m: &CompiledModel| {
        let inputs = AdaptiveInputs::new(&m.default_params).with_etas(&e);
        simulate_adaptive_with_inputs(m, &pop, &inputs, 1, hold_all, &opts)
    };
    assert_eq!(
        refusal(run(&mixture)),
        (
            "E_SUPPLIED_ETAS".to_string(),
            "supplied η are not supported on a [mixture] model. Each subject's η was \
             estimated under its own mixture class, and an adaptive run draws each \
             subject's class from the mixing probabilities, so the η could be simulated \
             under another class"
                .to_string()
        )
    );
    assert!(run(&plain).is_ok(), "the class-free model takes supplied η");
}

/// More than ten unmatched subjects: the list is cut at ten and the rest counted.
#[test]
fn unmatched_list_is_capped_at_ten() {
    let model = parse_model_string(ODE_IIV).expect("parse");
    let ids: Vec<String> = (1..=12).map(|i| i.to_string()).collect();
    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let pop = population(&id_refs);
    let e = etas(&[("99", 0.0)]);
    let inputs = AdaptiveInputs::new(&model.default_params).with_etas(&e);
    let (_, text) = refusal(simulate_adaptive_with_inputs(
        &model,
        &pop,
        &inputs,
        1,
        hold_all,
        &options(1, ObserveMode::Ipred),
    ));
    assert_eq!(
        text,
        "12 simulated subject(s) have no supplied η: '1', '2', '3', '4', '5', '6', '7', \
         '8', '9', '10' and 2 more. Supply their η, or pass `UnmatchedSubjects::Draw` to \
         draw theirs from Ω"
    );
}
