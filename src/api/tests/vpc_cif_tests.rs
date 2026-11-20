//! Tier-1 tests for the cumulative-incidence VPC driver (#1887): the single-cause twin
//! with `vpc_tte` (plan T7), the cause collector (T8, T8b), and every refusal's whole
//! text (T6).

use super::super::vpc_tte::{vpc_tte, TteVpc, VpcTteOptions};
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{EventType, Subject};

const EXP_TTE: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Two causes, λ = 0.05 on cmt 2 and 0.03 on cmt 3.
const COMPETING: &str = crate::test_models::MODEL_2;

/// Two causes, cause A (cmt 2) ten times as fast as cause B (cmt 3).
const FAST_SLOW: &str = r"
[parameters]
  theta TVLAMBDA_A(0.2, 0.001, 10.0)
  theta TVLAMBDA_B(0.02, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B
";

/// A single-event cause on cmt 2 beside a repeated-events endpoint on cmt 3.
const WITH_RTTE: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.15, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  type   = rtte
  family = exponential
  scale  = TVLAMBDA_B
";

const PK_ONLY: &str = crate::test_models::ONE_CPT_IV_22;

fn model(src: &str) -> CompiledModel {
    parse_model_string(src).expect("model parses")
}

fn rec(time: f64, event_type: EventType, cmt: usize) -> ObsRecord {
    ObsRecord::Event { time, event_type, entry_time: 0.0, cmt }
}

fn subj(id: &str, records: Vec<ObsRecord>) -> Subject {
    Subject { id: id.into(), obs_records: records, ..Default::default() }
}

fn pop(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// A two-cause subject: `cause` 0 censored at `t` on both cmts, 2 or 3 an event there.
fn two_cause(id: &str, t: f64, cause: usize) -> Subject {
    let et = |c: usize| if cause == c { EventType::Exact } else { EventType::RightCensored };
    subj(id, vec![rec(t, et(2), 2), rec(t, et(3), 3)])
}

fn opts(n_sim: usize) -> VpcCifOptions {
    VpcCifOptions {
        n_sim,
        sim: SimulateOptions { seed: Some(7), ..Default::default() },
        ..Default::default()
    }
}

fn run(src: &str, p: &Population, o: &VpcCifOptions) -> Result<CifVpc, EngineError> {
    let m = model(src);
    vpc_cif(&m, p, &m.default_params, o)
}

// ── T7: the single-cause twin with vpc_tte ───────────────────────────────────

/// 40 subjects on cmt 2: events at `0.9·i` except every third subject, censored there.
fn single_cause_pop() -> Population {
    pop((1..=40)
        .map(|i| {
            let et = if i % 3 == 0 { EventType::RightCensored } else { EventType::Exact };
            subj(&i.to_string(), vec![rec(0.9 * i as f64, et, 2)])
        })
        .collect())
}

fn twin(censoring: VpcCensoring) -> (CifVpc, TteVpc) {
    let grid = VpcGrid::Times(vec![0.0, 3.0, 9.0, 18.0, 27.0, 33.0, 36.0]);
    let sim = SimulateOptions { seed: Some(11), ..Default::default() };
    let c = VpcCifOptions {
        n_sim: 40,
        grid: grid.clone(),
        censoring,
        sim: sim.clone(),
        ..Default::default()
    };
    let t = VpcTteOptions { n_sim: 40, grid, censoring, sim, ..Default::default() };
    let m = model(EXP_TTE);
    let p = single_cause_pop();
    (
        vpc_cif(&m, &p, &m.default_params, &c).unwrap(),
        vpc_tte(&m, &p, &m.default_params, &t).unwrap(),
    )
}

/// T7. With one cause, `vpc_cif` is `vpc_tte` turned upside down: the same replicates
/// (the simulator call and the censoring draws, keyed by row index, are shared), so the
/// all-cause band is `vpc_tte`'s **bit for bit**, `n_used` is equal, and the cause band
/// is `1 − KM` with lo and hi swapped. Under [`VpcCensoring::Observed`] the event
/// subjects draw their censoring times, so a uniform keyed to a different row index
/// changes the survival band and reddens the bit-equality; a collector that drops the
/// winner flag makes every `F` zero and reddens the cause assertions. Measured on macOS
/// arm64: worst |F − (1 − S)| over the observed curve and the three band rows =
/// 6.661e-16; the bound 4e-15 keeps 6× headroom.
#[test]
fn single_cause_is_one_minus_vpc_tte() {
    let mut worst = 0.0_f64;
    for censoring in [VpcCensoring::Observed, VpcCensoring::Horizon] {
        let (c, t) = twin(censoring);
        assert_eq!(c.cmts, [2]);
        assert_eq!(c.horizon, t.horizon);
        let (cs, ts) = (&c.strata[0], &t.strata[0]);
        assert_eq!(cs.grid, ts.grid);
        assert_eq!(cs.survival.n_used, ts.n_used, "{censoring:?}: n_used");
        assert_eq!(cs.survival.obs_on_grid, ts.obs_on_grid, "{censoring:?}: observed S");
        assert_eq!(
            (&cs.survival.lo, &cs.survival.med, &cs.survival.hi),
            (&ts.lo, &ts.med, &ts.hi),
            "{censoring:?}: survival band"
        );
        let f = &cs.causes[0].band;
        assert_eq!(f.n_used, ts.n_used, "{censoring:?}: cause n_used");
        let rows = [
            (&f.obs_on_grid, &ts.obs_on_grid, "observed"),
            (&f.lo, &ts.hi, "lo vs 1 − hi"),
            (&f.med, &ts.med, "med"),
            (&f.hi, &ts.lo, "hi vs 1 − lo"),
        ];
        for (got, km, what) in rows {
            for (g, (a, b)) in got.iter().zip(km).enumerate() {
                let at = format!("{censoring:?}, {what}, t = {}", cs.grid[g]);
                assert_eq!(a.is_some(), b.is_some(), "{at}: definedness");
                if let (Some(a), Some(b)) = (a, b) {
                    let gap = (a - (1.0 - b)).abs();
                    assert!(gap.is_finite(), "{at}: non-finite");
                    worst = worst.max(gap);
                    assert!(gap <= 4e-15, "{at}: F = {a}, 1 − S = {}", 1.0 - b);
                }
            }
        }
        // The fixture must have something to compare: a defined band with F away from 0.
        assert!(f.med.iter().flatten().any(|&m| m > 0.2), "{censoring:?}: F stays near 0");
    }
    eprintln!("single-cause twin: worst |F − (1 − S)| = {worst:.3e}");
}

// ── T8 / T8b: the cause collector ────────────────────────────────────────────

/// 40 subjects: 24 with cause A (cmt 2), 8 with cause B (cmt 3), 8 censored, at distinct
/// times `0.5·i`.
fn fast_slow_pop() -> (Population, Vec<f64>, Vec<usize>) {
    let mut subjects = Vec::new();
    let (mut t, mut c) = (Vec::new(), Vec::new());
    for i in 1..=40 {
        let cause = match i % 5 {
            0 => 0,
            1 => 3,
            _ => 2,
        };
        let time = 0.5 * i as f64;
        subjects.push(two_cause(&i.to_string(), time, cause));
        t.push(time);
        c.push(match cause {
            2 => 1,
            3 => 2,
            _ => 0,
        });
    }
    (pop(subjects), t, c)
}

/// T8. The observed curves are [`aalen_johansen`] on the hand rows, with cmt 2 as cause 1
/// and cmt 3 as cause 2; and the replicates, drawn with cause A ten times as fast as B,
/// put the larger median incidence on cmt 2 at every grid time past 0. Swapping the
/// collector's cmt → cause map moves the replicate incidence to the wrong band and
/// reddens the median comparison; swapping it in the observed rows reddens the first.
#[test]
fn causes_are_read_by_cmt_in_data_and_replicates() {
    let (p, t, c) = fast_slow_pop();
    let mut o = opts(40);
    o.sim.horizon = Some(20.0);
    o.grid = VpcGrid::Times(vec![2.0, 5.0, 10.0]);
    let v = run(FAST_SLOW, &p, &o).unwrap();
    assert_eq!(v.cmts, [2, 3]);
    let s = &v.strata[0];
    assert_eq!(s.observed, aalen_johansen(&t, &c, 2).unwrap());
    assert_eq!(s.observed.n_event[0].iter().sum::<usize>(), 24);
    assert_eq!(s.causes.iter().map(|b| b.cmt).collect::<Vec<_>>(), [2, 3]);
    for g in 0..3 {
        let (a, b) = (s.causes[0].band.med[g].unwrap(), s.causes[1].band.med[g].unwrap());
        assert!(a > 2.0 * b, "t = {}: median F_A = {a}, F_B = {b}", s.grid[g]);
    }
    // At t = 10 the closed form gives F_A = 0.2/0.22·(1 − e^{−2.2}) = 0.810.
    let fa = s.causes[0].band.med[2].unwrap();
    assert!((fa - 0.810).abs() < 0.08, "median F_A at 10 = {fa}");
}

/// T8b, the `competing_event_is_not_an_observed_censoring` analogue. Every subject has
/// cause 2 at t = 1, so its cmt-3 record is censored at 1 — a competing event, not a
/// dropout. With no observed censoring the replicates keep the horizon, and the bands
/// are defined at 5. Treating any censored cause record as the subject's own censoring
/// time censors every replicate at 1 and makes the counts at 5 zero.
#[test]
fn competing_event_is_not_an_observed_censoring() {
    let p = pop((1..=30).map(|i| two_cause(&i.to_string(), 1.0, 2)).collect());
    let mut o = opts(20);
    o.sim.horizon = Some(10.0);
    o.grid = VpcGrid::Times(vec![0.5, 5.0]);
    let v = run(COMPETING, &p, &o).unwrap();
    let s = &v.strata[0];
    assert_eq!(v.censoring, VpcCensoring::Observed);
    assert_eq!(s.survival.n_used, [20, 20]);
    assert!(s.causes.iter().all(|b| b.band.n_used == [20, 20]));
}

/// The data are read on `[0, H]`: a subject with an event after `H` is censored at `H`
/// for every cause, as `vpc_tte`'s cut does. With `H` = 10 the events at 12–20 are
/// censorings at 10, so the observed cause-1 count drops from 24 and the observed curves
/// end at 10.
#[test]
fn observed_curves_read_the_data_cut_at_the_horizon() {
    let (p, t, c) = fast_slow_pop();
    let mut o = opts(20);
    o.sim.horizon = Some(10.0);
    let v = run(FAST_SLOW, &p, &o).unwrap();
    let cut: (Vec<f64>, Vec<usize>) =
        t.iter().zip(&c).map(|(&t, &c)| if t > 10.0 { (10.0, 0) } else { (t, c) }).unzip();
    assert_eq!(v.strata[0].observed, aalen_johansen(&cut.0, &cut.1, 2).unwrap());
    assert_eq!(v.strata[0].observed.all_cause.time.last(), Some(&10.0));
}

/// Stratified by a covariate, each stratum reads only its own subjects.
#[test]
fn each_stratum_uses_its_own_subjects() {
    let (mut p, ..) = fast_slow_pop();
    for (i, s) in p.subjects.iter_mut().enumerate() {
        s.covariates.insert("ARM".into(), (i % 2) as f64);
    }
    let mut o = opts(20);
    o.stratify = VpcStratify::Covariate("ARM".into());
    let v = run(FAST_SLOW, &p, &o).unwrap();
    assert_eq!(v.strata.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), ["ARM=0", "ARM=1"]);
    assert!(v.strata.iter().all(|s| s.n_subjects == 20));
}

// ── T6: refusals, whole text ─────────────────────────────────────────────────

/// The refusal `got` must carry `E_VPC_TTE_INPUT` and exactly `text`; the text must name
/// `vpc_cif` and not another entry point, so a shared arm that hardcodes one fails.
fn assert_refusal<T: std::fmt::Debug>(got: Result<T, EngineError>, text: &str) {
    let e = got.expect_err("must refuse");
    assert_eq!(e.code(), Some("E_VPC_TTE_INPUT"), "{e}");
    let msg = e.to_string();
    assert!(msg.starts_with("vpc_cif: ") && !msg.contains("vpc_tte"), "{msg}");
    assert_eq!(msg, text);
}

#[test]
fn refuses_no_tte_endpoint() {
    assert_refusal(
        run(PK_ONLY, &pop(vec![two_cause("1", 1.0, 2)]), &opts(20)),
        "vpc_cif: the model declares no TTE endpoint ([event_model]).",
    );
}

/// The RTTE refusal names the endpoint and `vpc_rtte`, and never asks for a `cmt`.
#[test]
fn refuses_a_repeated_events_endpoint() {
    let p = pop(vec![two_cause("1", 1.0, 2)]);
    let e = run(WITH_RTTE, &p, &opts(20));
    assert!(!format!("{e:?}").contains("set `cmt`"));
    assert_refusal(
        e,
        "vpc_cif: cmt 3 is a repeated-events (RTTE) endpoint. A cumulative incidence takes \
         single-event causes; `vpc_rtte` draws the Kaplan–Meier and mean-cumulative-function \
         VPC of a repeated-events endpoint.",
    );
}

#[test]
fn refuses_a_subject_without_a_record_on_every_cause() {
    let p = pop(vec![two_cause("1", 1.0, 2), subj("2", vec![rec(3.0, EventType::Exact, 2)])]);
    assert_refusal(
        run(COMPETING, &p, &opts(20)),
        "vpc_cif: subject 2 has no record on cmt 3. The simulator competes only the causes a \
         subject has records on, so a subject needs one record on every cause (cmt 2, 3).",
    );
}

#[test]
fn refuses_two_records_on_one_cause() {
    let mut s = two_cause("4", 1.0, 0);
    s.obs_records.push(rec(2.0, EventType::Exact, 3));
    assert_refusal(
        run(COMPETING, &pop(vec![s]), &opts(20)),
        "vpc_cif: subject 4 has 2 records on cmt 3. A cause takes one record per subject.",
    );
}

#[test]
fn refuses_cause_records_at_different_times() {
    let s = subj("5", vec![rec(2.0, EventType::Exact, 2), rec(3.5, EventType::RightCensored, 3)]);
    assert_refusal(
        run(COMPETING, &pop(vec![s]), &opts(20)),
        "vpc_cif: subject 5 has time 2 on cmt 2 but 3.5 on cmt 3. Every cause is followed to \
         the same time, the first event or the censoring.",
    );
}

#[test]
fn refuses_events_on_two_causes() {
    let s = subj("6", vec![rec(2.0, EventType::Exact, 2), rec(2.0, EventType::Exact, 3)]);
    assert_refusal(
        run(COMPETING, &pop(vec![s]), &opts(20)),
        "vpc_cif: subject 6 has an event on cmt 2 and on cmt 3. Under competing risks a \
         subject has one cause, the first.",
    );
}

/// The shared record checks with `vpc_cif`'s own wording: no "Kaplan–Meier".
#[test]
fn refuses_interval_censoring_and_left_truncation() {
    let ic = EventType::IntervalCensored { left: 1.0, right: 2.0 };
    let s = subj("7", vec![rec(2.0, ic, 2), rec(2.0, EventType::RightCensored, 3)]);
    let e = run(COMPETING, &pop(vec![s]), &opts(20));
    assert!(!format!("{e:?}").contains("Kaplan"));
    assert_refusal(
        e,
        "vpc_cif: subject 7 has an interval-censored record on cmt 2. A cumulative incidence \
         curve needs exact or right-censored times.",
    );
    let lt = ObsRecord::Event { time: 2.0, event_type: EventType::Exact, entry_time: 0.5, cmt: 3 };
    let s = subj("8", vec![rec(2.0, EventType::RightCensored, 2), lt]);
    assert_refusal(
        run(COMPETING, &pop(vec![s]), &opts(20)),
        "vpc_cif: subject 8 has entry time 0.5 on cmt 3. Left truncation (delayed entry) is \
         not supported yet.",
    );
}

#[test]
fn shared_refusals_name_vpc_cif() {
    let dup = pop(vec![two_cause("9", 1.0, 2), two_cause("9", 2.0, 3)]);
    assert_refusal(
        run(COMPETING, &dup, &opts(20)),
        "vpc_cif: subject id 9 appears more than once in the population. Simulated rows are \
         matched to subjects by id.",
    );
    assert_refusal(
        run(COMPETING, &pop(vec![subj("10", vec![])]), &opts(20)),
        "vpc_cif: no subject has a record on cmt 2, 3.",
    );
    let p = pop(vec![two_cause("1", 1.0, 2)]);
    assert_refusal(
        run(COMPETING, &p, &opts(19)),
        "vpc_cif: `n_sim = 19` is too few. The band at probabilities [0.05, 0.5, 0.95] needs at \
         least 20 replicates.",
    );
    let mut o = opts(20);
    o.probs = [0.5, 0.05, 0.95];
    assert_refusal(
        run(COMPETING, &p, &o),
        "vpc_cif: band probabilities [0.5, 0.05, 0.95] must be strictly inside (0, 1) and \
         strictly increasing.",
    );
    let mut o = opts(20);
    o.grid = VpcGrid::Times(vec![-1.0]);
    assert_refusal(
        run(COMPETING, &p, &o),
        "vpc_cif: grid time -1 is not a finite, non-negative number.",
    );
    let mut o = opts(20);
    o.stratify = VpcStratify::Covariate("WT".into());
    assert_refusal(
        run(COMPETING, &p, &o),
        "vpc_cif: subject 1 has no covariate `WT` to stratify on.",
    );
    let bad = pop(vec![subj("11", vec![rec(f64::NAN, EventType::Exact, 2)])]);
    assert_refusal(
        run(COMPETING, &bad, &opts(20)),
        "vpc_cif: subject 11 has time NaN on cmt 2, which is not a finite, non-negative number.",
    );
}

/// A model with one TTE endpoint is accepted (its incidence is `1 − KM`), and a subject
/// with no cause record at all is skipped, not refused.
#[test]
fn accepts_a_single_cause_and_skips_subjects_without_cause_records() {
    let mut p = single_cause_pop();
    p.subjects.push(subj("pk-only", vec![]));
    let v = run(EXP_TTE, &p, &opts(20)).unwrap();
    assert_eq!(v.strata[0].n_subjects, 40);
    assert_eq!(v.strata[0].causes.len(), 1);
}
