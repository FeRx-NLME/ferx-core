//! #1235 — a data-level typical-value check must evaluate at the `$PK` snapshots the
//! **engine** reads that quantity at, not at one frozen `(baseline covariates, TIME = 0)`
//! snapshot per subject.
//!
//! Before this, `check_dose_attr_finiteness` and `check_absorption_dosing` both called
//! `(pk_param_fn)(θ, 0, &subject.covariates, 0.0)` once. **Both arguments were wrong**
//! relative to what the engine does, and they are wrong along two *independent* axes —
//! which is why every arm below is paired with the minimal edit it exists to kill:
//!
//! | arm | escapes the old check via | dies if the fix widens only … |
//! |---|---|---|
//! | A | a time-varying covariate that is benign at dose 1 and overflows at dose 2 | the `TIME` axis |
//! | B | a `TIME`-reading `$PK` on a subject with **no covariates at all** | the covariate axis, or a `has_tv_covariates()` gate |
//!
//! Arm A is also the de-dup mutation: both of its doses are `CMT=1`, and the *finite*
//! one comes first, so the original `seen: BTreeSet<usize>` per-compartment skip would
//! `continue` past the bad one and pass the widened check. Keep `seen` keyed on `cmt`
//! alone and arm A goes green.
//!
//! The false-positive control is the other half of the contract: the engine reads a lag
//! at `pk_at_dose[k]` **only**, so a lag that goes non-finite at an *observation*
//! snapshot is a value the engine never applies and must **not** be rejected — while a
//! built-in absorption parameter, rebuilt per segment from the last event's snapshot, is
//! read at every record and **must** be. Two checks, two snapshot sets, asserted against
//! each other so a shared helper cannot quietly collapse them into one.

mod common;

use ferx_core::api::check_model_data;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, Population, RateMode, Subject};
use ferx_core::{predict, Diagnostic};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// models
// ---------------------------------------------------------------------------

/// Arm A: the depot lag is `TVLAG · exp(WT)`, finite for a small `WT` and `+inf` once
/// `WT` overflows the exponential. `exp` is the one DSL arithmetic with no domain guard
/// (`/` by ~0 returns 0, `ln`/`sqrt` floor their argument), so this is the reachable
/// route to a non-finite attribute at typical values.
const TV_COV_LAG_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAG(0.3, 0.0, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  ALAG1 = TVLAG * exp(WT)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Arm B: the same lag driven by the `TIME` built-in, with **no covariate anywhere**.
/// `pk::subject_needs_per_event_pk` fires on `model_uses_time_builtin` alone, so the
/// engine resolves `$PK` per dose while the old check froze it at `TIME = 0`.
const TIME_LAG_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAG(0.3, 0.0, 12.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  ALAG1 = TVLAG * exp(TIME)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// A modeled infusion **duration** (`RATE=-2` → `D1`, #324) driven by the same
/// overflowing covariate. `D1` is a dose attribute the engine applies at the dose event,
/// read from the same per-dose snapshot as the lag
/// (`resolve_subject_doses_with`'s `pk_for_dose(k)`), and until #1235 nothing checked its
/// **value** — `check_modeled_dose_rates` validates only that the slot exists.
const MODELED_DURATION_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVD(2.0, 0.01, 48.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD * exp(WT)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// The `RATE=-1` mirror: a modeled infusion **rate** `R1`.
const MODELED_RATE_MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVR(50.0, 0.01, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR * exp(WT)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Transit absorption whose mean transit time is `TVMTT − WT`: in domain at `WT = 0`,
/// `≤ 0` once `WT` exceeds it. Unlike a lag, the engine rebuilds this forcing **per
/// segment** from the last event's snapshot, so a value reached only at an observation
/// is one the engine really does apply.
const TV_COV_TRANSIT_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.05, 24.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.1, 30.0)
  omega ETA_CL ~ 0.0 FIX
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  MTT = TVMTT - WT
  NTR = TVN
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Parallel dual first-order absorption whose fast-pathway fraction is `TVFR1 + WT`:
/// in `(0, 1]` at `WT = 0`, out of range once `WT` grows. The `E_ABSORPTION_FRACTION`
/// counterpart of the model above, on the same snapshot set.
const TV_COV_PARALLEL_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVFR1(0.6, 0.05, 0.95)
  theta TVKA1(1.5, 0.05, 24.0)
  theta TVKA2(0.3, 0.01, 24.0)
  omega ETA_CL ~ 0.0 FIX
  sigma PROP_ERR ~ 0.01 (sd)
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  FR1 = TVFR1 + WT
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

fn wt(v: f64) -> HashMap<String, f64> {
    HashMap::from([("WT".to_string(), v)])
}

fn population(subject: Subject, covariate_names: &[&str]) -> Population {
    Population {
        covariate_names: covariate_names.iter().map(|s| s.to_string()).collect(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subject],
    }
}

/// Two doses into the **same** compartment with per-dose `WT` snapshots, plus two
/// observations. `wt_baseline` is the subject-level fallback the *old* check read, so a
/// fixture can be benign there and bad at a later dose — which is the whole escape.
fn two_dose_tv_subject(wt_baseline: f64, wt_dose: [f64; 2], wt_obs: [f64; 2]) -> Subject {
    let mut s = common::subject(
        "1",
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        vec![1.0, 48.0],
        vec![0.0; 2],
        vec![1; 2],
    );
    s.covariates = wt(wt_baseline);
    s.dose_covariates = wt_dose.iter().map(|&v| wt(v)).collect();
    s.obs_covariates = wt_obs.iter().map(|&v| wt(v)).collect();
    s
}

fn codes(diags: &[Diagnostic]) -> Vec<&str> {
    diags.iter().map(|d| d.code.as_str()).collect()
}

fn find<'a>(diags: &'a [Diagnostic], code: &str) -> &'a Diagnostic {
    diags
        .iter()
        .find(|d| d.code == code)
        .unwrap_or_else(|| panic!("expected {code}, got {:?}", codes(diags)))
}

// ---------------------------------------------------------------------------
// arm A — the covariate axis (and the de-dup mutation)
// ---------------------------------------------------------------------------

/// A lag that is finite at dose 1 and `+inf` at dose 2 — same compartment, and the
/// *baseline* covariate the old check read is the benign one — must be rejected, naming
/// the record.
///
/// Kills three distinct minimal edits: widening only the `TIME` axis (the bad value is
/// reached by a covariate, at `TIME = 24`, and `exp(24)` is finite so the time axis
/// alone finds nothing); keeping the per-compartment `seen` de-dup (dose 1 claims
/// `cmt = 1` and dose 2 is skipped); and evaluating only the *first* dose snapshot.
#[test]
fn arm_a_a_covariate_that_overflows_at_a_later_dose_is_rejected() {
    let model = parse_full_model(TV_COV_LAG_MODEL)
        .expect("the exp-lag model parses")
        .model;
    // The fixture has to be live on both sides: benign where the old check looked,
    // non-finite where the engine actually reads it. Assert both, so this cannot pass
    // on a fixture that was never bad (or was bad everywhere).
    assert!((0.3f64 * 0.5f64.exp()).is_finite());
    assert!((0.3f64 * 1000.0f64.exp()).is_infinite());
    // `exp(TIME)` at the bad dose's own time is finite, so a time-only widening has
    // nothing to find here.
    assert!(24.0f64.exp().is_finite());

    let pop = population(two_dose_tv_subject(0.5, [0.5, 1000.0], [0.5, 0.5]), &["WT"]);
    let diags = check_model_data(&model, &pop);
    let hit = find(&diags, "E_DOSE_ATTR_NONFINITE");
    assert!(
        hit.message.contains("subject 1"),
        "must name the subject: {}",
        hit.message
    );
    assert!(
        hit.message.contains("lag time"),
        "must name the attribute: {}",
        hit.message
    );
    assert!(
        hit.message.contains("dose 2 at TIME=24"),
        "must name the record whose snapshot went bad — that is what tells the user \
         where to look: {}",
        hit.message
    );
}

// ---------------------------------------------------------------------------
// arm B — the time axis, with no covariates at all
// ---------------------------------------------------------------------------

/// The same rejection with **no covariate in the model or the data**: the lag reads the
/// `TIME` built-in and the second dose is late enough to overflow it.
///
/// Kills the covariate-only widening, and kills a gate spelled `has_tv_covariates()` —
/// this subject has neither `dose_covariates` nor `obs_covariates`, so that paraphrase
/// drops it to the frozen `TIME = 0` snapshot where `exp(0) = 1` is perfectly finite.
#[test]
fn arm_b_a_time_reading_lag_is_rejected_with_no_covariates_at_all() {
    let model = parse_full_model(TIME_LAG_MODEL)
        .expect("the TIME-lag model parses")
        .model;
    let mut s = common::subject(
        "1",
        vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(1000.0, 100.0, 1, 0.0, false, 0.0),
        ],
        vec![1.0, 1001.0],
        vec![0.0; 2],
        vec![1; 2],
    );
    s.covariates = HashMap::new();
    // The gate under test is `subject_needs_per_event_pk`, not `has_tv_covariates()`.
    // Assert the discriminator rather than trusting the constructor: if these ever go
    // non-empty the arm stops testing the `TIME` axis.
    assert!(s.dose_covariates.is_empty() && s.obs_covariates.is_empty());
    assert!(!s.has_tv_covariates(), "arm B must have no TV covariates");
    assert!((0.3f64 * 0.0f64.exp()).is_finite());
    assert!((0.3f64 * 1000.0f64.exp()).is_infinite());

    let pop = population(s, &[]);
    let diags = check_model_data(&model, &pop);
    let hit = find(&diags, "E_DOSE_ATTR_NONFINITE");
    assert!(
        hit.message.contains("dose 2 at TIME=1000"),
        "must name the record: {}",
        hit.message
    );
}

// ---------------------------------------------------------------------------
// the false-positive control — the other side of the contract
// ---------------------------------------------------------------------------

/// A lag that is finite at **every dose** and `+inf` at a later *observation* must
/// **not** be rejected, and the model must still predict finite values.
///
/// Without this the widening would pass its own arms by rejecting everything: the
/// engine resolves a lag from `pk_at_dose[k]` and nowhere else, so an observation
/// snapshot is a value it never applies. This is what pins `SnapshotSet::Doses` for the
/// dose-attribute check — swap it to `AllRecords` and this test is the one that dies.
#[test]
fn a_lag_non_finite_only_at_an_observation_is_not_rejected_and_still_predicts() {
    let model = parse_full_model(TV_COV_LAG_MODEL)
        .expect("the exp-lag model parses")
        .model;
    // Doses benign, second observation catastrophic.
    let pop = population(two_dose_tv_subject(0.5, [0.5, 0.5], [0.5, 1000.0]), &["WT"]);
    let diags = check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code == "E_DOSE_ATTR_NONFINITE"),
        "a lag the engine never reads there must not be rejected, got {:?}",
        codes(&diags)
    );
    // And it is not merely un-rejected: it still works. If this ever returns NaN, the
    // premise above ("the engine reads a lag at dose snapshots only") is wrong and the
    // snapshot set is the thing to revisit.
    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .iter()
        .map(|p| p.pred)
        .collect();
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "the subject must still predict finite values, got {preds:?}"
    );
}

// ---------------------------------------------------------------------------
// modeled D{n} / R{n} — the value check, which silently served a bolus
// ---------------------------------------------------------------------------

/// Build the modeled-`RATE` fixture: one coded dose into compartment 1 per entry of
/// `wt_per_dose`, dosed 24 h apart, with that entry's `WT` on the dose record. A
/// single-element slice gives the single-dose shape; two give the multi-dose one.
fn modeled_dose_pop(mode: RateMode, wt_per_dose: &[f64]) -> Population {
    let doses = (0..wt_per_dose.len())
        .map(|k| DoseEvent::modeled(24.0 * k as f64, 100.0, 1, false, 0.0, mode))
        .collect();
    let mut s = common::subject("1", doses, vec![1.0, 4.0, 8.0], vec![0.0; 3], vec![1; 3]);
    s.covariates = wt(0.0);
    s.dose_covariates = wt_per_dose.iter().map(|&v| wt(v)).collect();
    s.obs_covariates = vec![wt(0.0); 3];
    population(s, &["WT"])
}

/// A non-finite modeled infusion duration / rate is rejected — **single dose and multi
/// dose**, because they test different things and neither implies the other.
///
/// *Single dose* pins that the value is read at all: `check_modeled_dose_rates` validated
/// only that the `D1`/`R1` slot **exists**, which this fixture satisfies, so before #1235
/// nothing looked at what the slot held. Unlike a lag it never produced a `NaN` either —
/// `resolve_rate` derives `rate = amt / duration`, so `D1 = +inf` gives `rate = 0`, no
/// longer an infusion at all, and the engine serves an instantaneous bolus.
///
/// *Multi dose* pins that it is read **per dose**: the first dose is finite and the
/// second is not, both into `CMT=1`. That is the shape the original per-compartment
/// de-dup swallowed, and the shape a fix that evaluates only `pk_at_dose[0]` passes. A
/// single-dose fixture alone cannot fail either mutation.
#[test]
fn a_non_finite_modeled_duration_is_rejected_at_one_dose_and_at_a_later_one() {
    let model = parse_full_model(MODELED_DURATION_MODEL)
        .expect("the modeled-duration model parses")
        .model;
    for (shape, wts, at) in [
        ("single dose", &[1000.0][..], "dose 1 at TIME=0"),
        ("multi dose", &[0.0, 1000.0][..], "dose 2 at TIME=24"),
    ] {
        let pop = modeled_dose_pop(RateMode::ModeledDuration, wts);
        let diags = check_model_data(&model, &pop);
        let hit = find(&diags, "E_DOSE_ATTR_NONFINITE");
        assert!(
            hit.message.contains("modeled infusion duration"),
            "{shape}: must name the attribute: {}",
            hit.message
        );
        assert!(
            hit.message.contains(at),
            "{shape}: must name the offending record ({at}): {}",
            hit.message
        );
    }
}

/// The `RATE=-1` mirror, on both shapes for the same reasons. `D` and `R` read different
/// slots through different `DoseAttr`s, so neither arm stands in for the other.
#[test]
fn a_non_finite_modeled_rate_is_rejected_at_one_dose_and_at_a_later_one() {
    let model = parse_full_model(MODELED_RATE_MODEL)
        .expect("the modeled-rate model parses")
        .model;
    for (shape, wts, at) in [
        ("single dose", &[1000.0][..], "dose 1 at TIME=0"),
        ("multi dose", &[0.0, 1000.0][..], "dose 2 at TIME=24"),
    ] {
        let pop = modeled_dose_pop(RateMode::ModeledRate, wts);
        let diags = check_model_data(&model, &pop);
        let hit = find(&diags, "E_DOSE_ATTR_NONFINITE");
        assert!(
            hit.message.contains("modeled infusion rate"),
            "{shape}: must name the attribute: {}",
            hit.message
        );
        assert!(
            hit.message.contains(at),
            "{shape}: must name the offending record ({at}): {}",
            hit.message
        );
    }
}

/// The multi-dose control for both: **every** dose finite must not be rejected, and must
/// still predict. Without it the arms above are satisfied by a check that rejects any
/// subject with more than one coded-`RATE` dose.
#[test]
fn a_multi_dose_modeled_regimen_with_every_dose_finite_is_not_rejected() {
    for (name, src, mode) in [
        ("D1", MODELED_DURATION_MODEL, RateMode::ModeledDuration),
        ("R1", MODELED_RATE_MODEL, RateMode::ModeledRate),
    ] {
        let model = parse_full_model(src).expect("parses").model;
        let pop = modeled_dose_pop(mode, &[0.0, 0.0]);
        let diags = check_model_data(&model, &pop);
        assert!(
            !diags.iter().any(|d| d.code == "E_DOSE_ATTR_NONFINITE"),
            "{name}: a finite multi-dose regimen must not be rejected, got {:?}",
            codes(&diags)
        );
        let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
            .iter()
            .map(|p| p.pred)
            .collect();
        assert!(
            preds.iter().all(|p| p.is_finite()),
            "{name}: must still predict finite values, got {preds:?}"
        );
    }
}

/// The control, and the non-degeneracy proof for the two arms above: a **finite** `D1`
/// is accepted, and the curve it produces is genuinely an infusion — materially
/// different from the bolus curve `100/V · exp(−ke·t)` that the non-finite arms silently
/// served. Without this the two arms could pass on a fixture where infusion and bolus
/// coincide.
#[test]
fn a_finite_modeled_duration_is_accepted_and_is_not_the_bolus_curve() {
    let model = parse_full_model(MODELED_DURATION_MODEL)
        .expect("the modeled-duration model parses")
        .model;
    let pop = modeled_dose_pop(RateMode::ModeledDuration, &[0.0]);
    let diags = check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code == "E_DOSE_ATTR_NONFINITE"),
        "a finite modeled duration must not be rejected, got {:?}",
        codes(&diags)
    );
    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .iter()
        .map(|p| p.pred)
        .collect();
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "the control must predict finite values, got {preds:?}"
    );
    // ke = CL/V = 0.1, V = 10 ⇒ the bolus readout is 10·exp(−0.1·t).
    let bolus_at_1 = 10.0 * (-0.1f64).exp();
    assert!(
        (preds[0] - bolus_at_1).abs() > 1.0,
        "the fixture must distinguish an infusion from a bolus, or the arms above could \
         pass on a degenerate model: infusion {} vs bolus {bolus_at_1}",
        preds[0]
    );
}

// ---------------------------------------------------------------------------
// the absorption pair — the OTHER snapshot set
// ---------------------------------------------------------------------------

/// A built-in absorption parameter driven out of domain at an **observation** snapshot
/// is rejected — the opposite verdict to the lag on the same kind of record, because the
/// engine rebuilds this forcing per segment from the last event's snapshot and so does
/// apply the bad value.
///
/// This is the assertion that stops the shared helper from collapsing the two snapshot
/// sets: point `check_absorption_dosing` at `SnapshotSet::Doses` and only this side
/// dies; point `check_dose_attr_finiteness` at `AllRecords` and only the false-positive
/// control dies.
#[test]
fn an_absorption_parameter_out_of_domain_only_at_an_observation_is_rejected() {
    let model = parse_full_model(TV_COV_TRANSIT_MODEL)
        .expect("the TV-covariate transit model parses")
        .model;
    let mut s = common::subject(
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        vec![1.0, 8.0],
        vec![0.0; 2],
        vec![2; 2],
    );
    s.covariates = wt(0.0);
    s.dose_covariates = vec![wt(0.0)];
    s.obs_covariates = vec![wt(0.0), wt(10.0)];
    // MTT = 1 − WT: in domain at the dose and the first observation, ≤ 0 at the second.
    let pop = population(s, &["WT"]);
    let diags = check_model_data(&model, &pop);
    let hit = find(&diags, "E_ABSORPTION_DOMAIN");
    assert!(
        hit.message.contains("observation 2 at TIME=8"),
        "must name the record: {}",
        hit.message
    );
}

/// The `E_ABSORPTION_FRACTION` counterpart, so both codes on this snapshot set carry
/// their own control rather than one standing in for the other.
#[test]
fn a_pathway_fraction_out_of_range_only_at_an_observation_is_rejected() {
    let model = parse_full_model(TV_COV_PARALLEL_MODEL)
        .expect("the TV-covariate parallel model parses")
        .model;
    let mut s = common::subject(
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        vec![1.0, 8.0],
        vec![0.0; 2],
        vec![1; 2],
    );
    s.covariates = wt(0.0);
    s.dose_covariates = vec![wt(0.0)];
    s.obs_covariates = vec![wt(0.0), wt(5.0)];
    // FR1 = 0.6 + WT: in (0, 1] at the dose and the first observation, 5.6 at the second.
    let pop = population(s, &["WT"]);
    let diags = check_model_data(&model, &pop);
    let hit = find(&diags, "E_ABSORPTION_FRACTION");
    assert!(
        hit.message.contains("observation 2 at TIME=8"),
        "must name the record: {}",
        hit.message
    );
}

/// **A stated behaviour change on a public entry point.** `predict()` / `simulate()` run
/// no `check_model_data`, but they *do* call `assert_absorption_dosing_supported`, which
/// panics on the first error `check_absorption_dosing` returns. Widening that check's
/// snapshot set therefore widens what these entry points abort on: a model that returned
/// numbers before now panics. Deliberate — the numbers it returned came from an
/// out-of-domain forcing the engine really does apply at that record — and pinned here
/// so the change cannot happen twice by accident (#898 is the issue for turning these
/// aborts into diagnostics).
///
/// `E_DOSE_ATTR_NONFINITE` has no such `assert_*` twin, so the widening leaves
/// `predict()` unchanged for it. That gap is #1280 / #898, not something to close by
/// adding an eleventh panic wrapper here.
#[test]
#[should_panic(expected = "absorption input-rate machinery cannot honour")]
fn predict_now_aborts_on_an_absorption_domain_error_reached_only_at_an_observation() {
    let model = parse_full_model(TV_COV_TRANSIT_MODEL)
        .expect("the TV-covariate transit model parses")
        .model;
    let mut s = common::subject(
        "1",
        vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        vec![1.0, 8.0],
        vec![0.0; 2],
        vec![2; 2],
    );
    s.covariates = wt(0.0);
    s.dose_covariates = vec![wt(0.0)];
    s.obs_covariates = vec![wt(0.0), wt(10.0)];
    let pop = population(s, &["WT"]);
    let _ = predict(&model, &pop, &model.default_params);
}
