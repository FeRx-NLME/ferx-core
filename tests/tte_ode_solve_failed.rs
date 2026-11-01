//! #2087: an ODE-accumulated TTE event-time solve that fails for one simulated subject is
//! handled for that subject — censored at the observation window and named in a
//! `W_TTE_SOLVE_FAILED` warning — instead of panicking and taking the whole simulation down.
//!
//! Fixture: `d/dt(blowup) = K·blowup²` from `blowup(0) = 1` diverges at `t = 1/K`. Subject 1
//! has `K = 0.5` (diverges at t = 2, inside the horizon of 30); subject 2 has `K = 0.01`
//! (diverges at t = 100, past it). The hazard `H0` does not read `blowup`, and is small
//! enough that subject 1 essentially never fires before t = 2, so the solve fails on the
//! diverged unmonitored state, the input class of the original report.
//!
//! Engine: the `f64` value path only (`simulate_with_options_diag` → `simulate_tte` →
//! `ode_solve_until_chz_threshold`). Simulation has no dual or FD path. No NONMEM anchor:
//! NONMEM has no simulation of an ODE-accumulated TTE latent time that can fail this way.
#![cfg(feature = "survival")]

use ferx_core::api::{read_population_for, SimulationOutput, SimulationResult};
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, Population, SimOutcome};
use std::io::Write;

const HORIZON: f64 = 30.0;

const MODEL: &str = r"
[parameters]
  theta TVH0(0.001, FIX)
  omega ETA_H ~ 0.01
  sigma ADD ~ 0.01

[individual_parameters]
  H0 = TVH0 * exp(ETA_H)
  K  = KC

[structural_model]
  ode(obs_cmt=blowup, states=[blowup])

[odes]
  init(blowup) = 1.0
  d/dt(blowup) = K * blowup * blowup

[event_model]
  cmt    = 2
  hazard = H0

[error_model]
  DV ~ additive(ADD)
";

/// Competing risks: the ODE-accumulated cause on CMT 2, an analytic exponential cause on
/// CMT 3. The ODE cause's record comes first, so the analytic latent is drawn *after* the
/// failing solve — which is what lets T4 see a cause loop that stops drawing on failure.
const MODEL_CR: &str = r"
[parameters]
  theta TVH0(0.001, FIX)
  theta TVLAM(0.03, FIX)
  omega ETA_H ~ 0.01
  sigma ADD ~ 0.01

[individual_parameters]
  H0 = TVH0 * exp(ETA_H)
  K  = KC

[structural_model]
  ode(obs_cmt=blowup, states=[blowup])

[odes]
  init(blowup) = 1.0
  d/dt(blowup) = K * blowup * blowup

[event_model drug]
  cmt    = 2
  hazard = H0

[event_model other]
  cmt    = 3
  family = exponential
  scale  = TVLAM

[error_model]
  DV ~ additive(ADD)
";

/// One PK row and one censored TTE record per subject; `kc1` is subject 1's `K`.
fn data(kc1: f64) -> String {
    format!(
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II,KC\n\
         1,0.5,1,0,.,1,0,0,0,0,{kc1}\n\
         1,30,0,0,0,2,0,0,0,0,{kc1}\n\
         2,0.5,1,0,.,1,0,0,0,0,0.01\n\
         2,30,0,0,0,2,0,0,0,0,0.01\n"
    )
}

/// As [`data`], with a censored record for each of the two competing causes.
fn data_cr(kc1: f64) -> String {
    format!(
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II,KC\n\
         1,0.5,1,0,.,1,0,0,0,0,{kc1}\n\
         1,30,0,0,0,2,0,0,0,0,{kc1}\n\
         1,30,0,0,0,3,0,0,0,0,{kc1}\n\
         2,0.5,1,0,.,1,0,0,0,0,0.01\n\
         2,30,0,0,0,2,0,0,0,0,0.01\n\
         2,30,0,0,0,3,0,0,0,0,0.01\n"
    )
}

fn population(model: &CompiledModel, csv: &str) -> Population {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    read_population_for(
        model,
        &None,
        f.path().to_str().unwrap(),
        None,
        None,
        None,
        &[],
    )
    .expect("read population")
    .0
}

fn simulate(model_src: &str, csv: &str, n_sim: usize) -> SimulationOutput {
    let model = parse_model_string(model_src).expect("parse");
    let pop = population(&model, csv);
    let opts = ferx_core::SimulateOptions {
        horizon: Some(HORIZON),
        seed: Some(20872087),
        match_method: None,
        iov_occasion: None,
    };
    ferx_core::api::simulate_with_options_diag(&model, &pop, &model.default_params, n_sim, &opts)
        .expect("simulation must succeed with one subject's event-time solve failing")
}

fn solve_failed(out: &SimulationOutput) -> Vec<&String> {
    out.warnings
        .iter()
        .filter(|w| w.starts_with("W_TTE_SOLVE_FAILED"))
        .collect()
}

fn rows_of<'a>(out: &'a SimulationOutput, id: &str) -> Vec<&'a SimulationResult> {
    out.results.iter().filter(|r| r.id == id).collect()
}

fn event(r: &SimulationResult) -> Option<(f64, bool)> {
    match r.outcome {
        SimOutcome::Event { time, observed } => Some((time, observed)),
        _ => None,
    }
}

/// Every field of a row, as bits, so two rows compare equal only when bit-identical
/// (`NaN` included: the TTE row's `ipred` is `NaN`, which `==` would call unequal).
fn row_bits(r: &SimulationResult) -> (usize, usize, String, u64, usize, u64, String) {
    let outcome = match r.outcome {
        SimOutcome::Continuous { value } => format!("cont:{:x}", value.to_bits()),
        SimOutcome::Event { time, observed } => format!("event:{:x}:{observed}", time.to_bits()),
        ref other => format!("{other:?}"),
    };
    (
        r.draw,
        r.sim,
        r.id.clone(),
        r.time.to_bits(),
        r.cmt,
        r.ipred.to_bits(),
        outcome,
    )
}

/// T1: the diverging subject no longer panics the run. Every replicate keeps both subjects'
/// rows, the failed subject's TTE row is censored at the window, and one warning per failed
/// replicate names it.
///
/// Mutations that redden it: restore the `panic!` (the call panics); censor at `f64::MAX`
/// instead of the window (censor-time assert); drop the failed subject's TTE row
/// (row-count assert); drop the warning (warning-count assert).
#[test]
fn diverging_subject_is_censored_and_the_run_continues() {
    let n_sim = 3;
    let out = simulate(MODEL, &data(0.5), n_sim);

    let s1 = rows_of(&out, "1");
    let s1_tte: Vec<_> = s1.iter().filter(|r| r.cmt == 2).collect();
    assert_eq!(
        s1.len(),
        2 * n_sim,
        "subject 1 keeps a PK and a TTE row per replicate"
    );
    assert_eq!(s1_tte.len(), n_sim);
    for r in &s1_tte {
        let (t, observed) = event(r).expect("CMT 2 is the event row");
        assert!(!observed, "a failed solve is censored, never an event");
        assert_eq!(
            t.to_bits(),
            HORIZON.to_bits(),
            "censored at the window, got {t}"
        );
        assert_eq!(r.time.to_bits(), HORIZON.to_bits());
    }
    assert_eq!(
        rows_of(&out, "2").len(),
        2 * n_sim,
        "the healthy subject still simulates"
    );
    assert_eq!(solve_failed(&out).len(), n_sim, "{:#?}", out.warnings);
}

/// T2: the differential pair. Run A has subject 1 diverging (K = 0.5), run B has it healthy
/// (K = 0.01); same seed. Subject 2's rows must be bit-identical across the pair, so the
/// failure consumes exactly the RNG a successful solve does — and subject 1's outcome must
/// differ (A warns, B does not), asserted so the pair cannot quietly become a tautology.
///
/// Mutations that redden it: skip the `u` draw on the failing path, or any other change to
/// the RNG consumed by a failed subject (subject 2's PK value shifts); emit no rows for the
/// failed subject (subject 1 row-count straddle).
#[test]
fn healthy_subject_is_bit_identical_whether_or_not_another_diverges() {
    let n_sim = 3;
    let a = simulate(MODEL, &data(0.5), n_sim);
    let b = simulate(MODEL, &data(0.01), n_sim);

    // The straddle: A fails on subject 1, B does not.
    assert_eq!(solve_failed(&a).len(), n_sim, "{:#?}", a.warnings);
    assert!(solve_failed(&b).is_empty(), "{:#?}", b.warnings);
    assert_eq!(rows_of(&a, "1").len(), rows_of(&b, "1").len());

    let a2: Vec<_> = rows_of(&a, "2").into_iter().map(row_bits).collect();
    let b2: Vec<_> = rows_of(&b, "2").into_iter().map(row_bits).collect();
    assert_eq!(a2.len(), 2 * n_sim);
    assert_eq!(
        a2, b2,
        "subject 2's rows must not depend on subject 1's solve failing"
    );
    // The comparison has teeth only if subject 2's rows carry RNG-drawn values.
    assert!(rows_of(&a, "2").iter().any(|r| matches!(
        r.outcome,
        SimOutcome::Continuous { value } if value.is_finite() && value != r.ipred
    )));
}

/// T3: the warning names the subject, the CMT, the diverged `[odes]` state by its name, the
/// censoring time, and the rest of the population — and nothing it should not.
///
/// Mutations that redden it: drop the label mapping (prints `state 0`); render the injected
/// accumulator by its internal name (`__chz`); name the wrong subject; reuse the
/// degenerate-hazard message.
#[test]
fn warning_names_the_subject_the_state_and_the_censoring() {
    let out = simulate(MODEL, &data(0.5), 1);
    let w = solve_failed(&out);
    assert_eq!(w.len(), 1, "{:#?}", out.warnings);
    let w = w[0];
    for want in [
        "W_TTE_SOLVE_FAILED: subject '1' (CMT=2)",
        "state `blowup` became non-finite",
        "The [odes] right-hand side diverged",
        "censored at the observation window (t=30)",
        "not simulated past the failure",
        "The rest of the population still simulates.",
    ] {
        assert!(w.contains(want), "missing {want:?} in: {w}");
    }
    for unwanted in ["state 0", "__chz", "subject '2'", "degenerate"] {
        assert!(!w.contains(unwanted), "unexpected {unwanted:?} in: {w}");
    }
    assert_eq!(out.warnings.len(), 1, "{:#?}", out.warnings);
}

/// T4: competing risks. A failed ODE cause censors **every** cause of the subject at the
/// shared window, one warning names only the ODE cause's CMT, and the healthy subject is
/// bit-identical to the run where nothing diverges.
///
/// Mutations that redden it: leave the analytic cause to resolve on its own (its row may
/// fire, or carry a time other than the window); stop drawing causes after the failure
/// (subject 2 shifts, since the analytic cause is drawn after the ODE one); name the
/// analytic CMT.
#[test]
fn competing_risks_failure_censors_every_cause() {
    let n_sim = 3;
    let a = simulate(MODEL_CR, &data_cr(0.5), n_sim);
    let b = simulate(MODEL_CR, &data_cr(0.01), n_sim);

    let w = solve_failed(&a);
    assert_eq!(w.len(), n_sim, "{:#?}", a.warnings);
    for w in &w {
        assert!(w.contains("subject '1' (CMT=2)"), "{w}");
        assert!(!w.contains("CMT=3"), "{w}");
        assert!(
            w.contains("every cause of the subject is censored at"),
            "{w}"
        );
    }
    assert!(solve_failed(&b).is_empty(), "{:#?}", b.warnings);

    for cmt in [2, 3] {
        let tte: Vec<_> = rows_of(&a, "1")
            .into_iter()
            .filter(|r| r.cmt == cmt)
            .collect();
        assert_eq!(tte.len(), n_sim, "CMT {cmt}");
        for r in tte {
            let (t, observed) = event(r).expect("event row");
            assert!(!observed, "CMT {cmt} must be censored");
            assert_eq!(t.to_bits(), HORIZON.to_bits(), "CMT {cmt} censored at {t}");
        }
    }

    let a2: Vec<_> = rows_of(&a, "2").into_iter().map(row_bits).collect();
    let b2: Vec<_> = rows_of(&b, "2").into_iter().map(row_bits).collect();
    assert_eq!(a2.len(), 3 * n_sim);
    assert_eq!(
        a2, b2,
        "subject 2's rows must not depend on subject 1's solve failing"
    );
}
