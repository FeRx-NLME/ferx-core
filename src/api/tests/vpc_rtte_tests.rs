//! Tier-1 tests for the repeated-events VPC driver (#1886 PR2): the per-order risk
//! sets on the plan's fixture R1 (T6, T7), the follow-up of the replicates (T8), the
//! one-stream-per-subject contract of the simulator (T-sim), the order range (T9),
//! strata (T10) and every refusal's whole text (T11).

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{EventType, Subject};

/// A homogeneous Poisson process of rate 0.15 with no between-subject variability, so
/// `E[N(t)] = 0.15·t` exactly.
const RTTE: &str = r"
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)

[event_model]
  cmt    = 2
  type   = rtte
  clock  = forward
  family = exponential
  scale  = TVLAMBDA
";

const SINGLE: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";

const TWO_ENDPOINTS: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B
";

const PK_ONLY: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
";

fn model(src: &str) -> CompiledModel {
    parse_model_string(src).expect("model parses")
}

fn rec(time: f64, event_type: EventType) -> ObsRecord {
    ObsRecord::Event {
        time,
        event_type,
        entry_time: 0.0,
        cmt: 2,
    }
}

fn subj(id: &str, records: Vec<ObsRecord>) -> Subject {
    Subject {
        id: id.into(),
        obs_records: records,
        ..Default::default()
    }
}

/// A subject followed to `end` with events at `events`: one exact record per event and,
/// unless the last event is at `end` itself, a closing right-censored record.
fn stream(id: &str, end: f64, events: &[f64]) -> Subject {
    let mut records: Vec<ObsRecord> = events.iter().map(|&t| rec(t, EventType::Exact)).collect();
    if events.last() != Some(&end) {
        records.push(rec(end, EventType::RightCensored));
    }
    subj(id, records)
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

/// Fixture R1 (#1886 PR2 plan §0). A tie across subjects at 2, an event at the
/// subject's own follow-up end (B at 8, E at 4), and a subject with no event (C).
fn r1() -> Vec<Subject> {
    vec![
        stream("A", 10.0, &[2.0, 5.0, 9.0]),
        stream("B", 8.0, &[3.0, 8.0]),
        stream("C", 6.0, &[]),
        stream("D", 10.0, &[2.0, 7.0]),
        stream("E", 4.0, &[4.0]),
        stream("F", 5.0, &[1.0]),
    ]
}

/// `n` subjects, ids from `first`, each followed to `end` with events at `events`.
fn cohort(first: usize, n: usize, end: f64, events: &[f64]) -> Vec<Subject> {
    (first..first + n)
        .map(|i| stream(&i.to_string(), end, events))
        .collect()
}

fn opts(n_sim: usize) -> VpcRtteOptions {
    VpcRtteOptions {
        n_sim,
        sim: SimulateOptions {
            seed: Some(7),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn run(src: &str, p: &Population, o: &VpcRtteOptions) -> Result<RtteVpc, EngineError> {
    let m = model(src);
    vpc_rtte(&m, p, &m.default_params, o)
}

fn assert_near(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            g.is_finite() && (g - w).abs() < 1e-15,
            "{got:?} vs {want:?}"
        );
    }
}

// ── T6: the per-order risk sets on R1 ────────────────────────────────────────

/// R1 through the entry point against `survfit(Surv(t, d) ~ 1)` per order (survival
/// 3.8.12, re-measured at the commit adding this test). Absolute order 2: every subject,
/// censored at its follow-up end. Gap order 2: only the five subjects with an event,
/// E censored at gap 0 because its first event is at its follow-up end. Gap order 3:
/// A, B, D. Putting every subject in the gap risk set makes `n_risk` start at 6;
/// censoring at `cᵢ` instead of `cᵢ − T₁` moves F's censoring from 4 to 5; using the
/// event time as the gap moves A's event from 3 to 5.
#[test]
fn order_curves_match_survfit_on_r1() {
    let mut o = opts(20);
    o.max_order = Some(3);
    let v = run(RTTE, &pop(r1()), &o).unwrap();
    assert_eq!((v.cmt, v.horizon, v.max_order), (2, 10.0, 3));
    let s = &v.strata[0];
    assert_eq!(s.n_subjects, 6);
    // The mean cumulative function sees all six follow-up ends, C's included.
    assert_eq!(s.mcf.observed.n_risk, [6, 6, 6, 6, 5, 4, 3, 3, 2, 2]);
    assert_eq!(s.mcf.observed.n_event, [1, 2, 1, 1, 1, 0, 1, 1, 1, 0]);
    assert!((s.mcf.observed.mcf[9] - 11.0 / 5.0).abs() < 1e-15);

    assert_eq!(
        s.absolute.iter().map(|b| b.order).collect::<Vec<_>>(),
        [1, 2, 3]
    );
    let a2 = &s.absolute[1];
    assert_eq!(a2.n_subjects, 6);
    assert_eq!(a2.observed.time, [4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(a2.observed.n_risk, [6, 5, 3, 2, 1]);
    assert_eq!(a2.observed.n_event, [0, 1, 0, 1, 1]);
    assert_eq!(a2.observed.n_censor, [1, 1, 1, 0, 0]);
    assert_near(&a2.observed.surv, &[1.0, 0.8, 0.8, 0.4, 0.0]);

    assert_eq!(s.gap.iter().map(|b| b.order).collect::<Vec<_>>(), [2, 3]);
    let g2 = &s.gap[0];
    assert_eq!(g2.n_subjects, 5);
    assert_eq!(g2.observed.time, [0.0, 3.0, 4.0, 5.0]);
    assert_eq!(g2.observed.n_risk, [5, 4, 3, 2]);
    assert_eq!(g2.observed.n_event, [0, 1, 0, 2]);
    assert_eq!(g2.observed.n_censor, [1, 0, 1, 0]);
    assert_near(&g2.observed.surv, &[1.0, 0.75, 0.75, 0.0]);
    let g3 = &s.gap[1];
    assert_eq!(g3.n_subjects, 3);
    assert_eq!(g3.observed.time, [0.0, 3.0, 4.0]);
    assert_eq!(g3.observed.n_risk, [3, 2, 1]);
    assert_eq!(g3.observed.n_event, [0, 0, 1]);
    assert_eq!(g3.observed.n_censor, [1, 1, 0]);
    assert_near(&g3.observed.surv, &[1.0, 1.0, 0.0]);

    // Every curve's parallel vectors have the grid's length and a finite band.
    let lens = |g: &[f64], parts: [usize; 5]| parts.iter().all(|&n| n == g.len());
    assert!(lens(
        &s.mcf.grid,
        [
            s.mcf.obs_on_grid.len(),
            s.mcf.lo.len(),
            s.mcf.med.len(),
            s.mcf.hi.len(),
            s.mcf.n_used.len()
        ]
    ));
    for b in s.absolute.iter().chain(&s.gap) {
        assert!(lens(
            &b.grid,
            [
                b.obs_on_grid.len(),
                b.lo.len(),
                b.med.len(),
                b.hi.len(),
                b.n_used.len()
            ]
        ));
        assert!(b
            .lo
            .iter()
            .chain(&b.med)
            .chain(&b.hi)
            .flatten()
            .all(|x| x.is_finite()));
    }
    // The overlay reads the observed curve at the grid's observed times.
    let k = a2.grid.iter().position(|&t| t == 7.0).expect("7 on grid");
    assert_eq!(a2.obs_on_grid[k], a2.observed.at(7.0));
    let k = s
        .mcf
        .grid
        .iter()
        .position(|&t| t == 9.0)
        .expect("9 on grid");
    assert_eq!(s.mcf.obs_on_grid[k], s.mcf.observed.at(9.0));
}

/// `order_mark` itself, on A (`{2, 5, 9}`, end 10) and E (`{4}`, end 4). Gap order 1 is
/// absolute order 1, which is why the driver returns gap curves from order 2.
#[test]
fn order_mark_reads_one_subject() {
    let a = [2.0, 5.0, 9.0];
    assert_eq!(order_mark(&a, 10.0, 1, Clock::Absolute), Some((2.0, true)));
    assert_eq!(order_mark(&a, 10.0, 3, Clock::Absolute), Some((9.0, true)));
    assert_eq!(
        order_mark(&a, 10.0, 4, Clock::Absolute),
        Some((10.0, false))
    );
    assert_eq!(order_mark(&a, 10.0, 2, Clock::Gap), Some((3.0, true)));
    assert_eq!(order_mark(&a, 10.0, 4, Clock::Gap), Some((1.0, false)));
    assert_eq!(order_mark(&a, 10.0, 5, Clock::Gap), None);
    assert_eq!(order_mark(&[4.0], 4.0, 2, Clock::Gap), Some((0.0, false)));
    assert_eq!(order_mark(&[], 6.0, 2, Clock::Gap), None);
    for (events, end) in [(&a[..], 10.0), (&[][..], 6.0)] {
        assert_eq!(
            order_mark(events, end, 1, Clock::Gap),
            order_mark(events, end, 1, Clock::Absolute)
        );
    }
}

// ── T7: two events at one time in one subject ────────────────────────────────

/// R2 is R1 with A's events at `{2, 5, 5}` (allowed under `clock = forward`). The mean
/// cumulative function counts both events at 5; A's second and third events step the
/// order-2 and order-3 absolute curves at the same time; and A's third gap is an event
/// at 0, beside B censored at 0 (its second event is at its follow-up end). Skipping a
/// zero gap removes the event from the row at 0.
#[test]
fn two_events_at_one_time_step_two_orders_and_give_a_zero_gap() {
    let mut subjects = r1();
    subjects[0] = stream("A", 10.0, &[2.0, 5.0, 5.0]);
    let mut o = opts(20);
    o.max_order = Some(3);
    let v = run(RTTE, &pop(subjects), &o).unwrap();
    let s = &v.strata[0];
    let k = s.mcf.observed.time.iter().position(|&t| t == 5.0).unwrap();
    assert_eq!(s.mcf.observed.n_event[k], 2);
    assert!((s.mcf.observed.mcf[k] - 37.0 / 30.0).abs() < 1e-15);
    for order in [2, 3] {
        let c = &s.absolute[order - 1].observed;
        let k = c.time.iter().position(|&t| t == 5.0).expect("a row at 5");
        assert_eq!(c.n_event[k], 1, "order {order} steps at 5");
    }
    let g3 = &s.gap[1].observed;
    assert_eq!(g3.time, [0.0, 3.0]);
    assert_eq!(g3.n_risk, [3, 1]);
    assert_eq!(g3.n_event, [1, 0]);
    assert_eq!(g3.n_censor, [1, 1]);
    assert_near(&g3.surv, &[2.0 / 3.0, 2.0 / 3.0]);
}

// ── T8: follow-up of the replicates ──────────────────────────────────────────

/// `replicate_follow_up` on one subject: under `Observed` the events after its observed
/// follow-up end are dropped and follow-up ends there, or at `H` when `H` is earlier;
/// under `Horizon` everything is kept to `H`. An event exactly at the observed end stays.
#[test]
fn replicate_follow_up_cuts_at_the_observed_end() {
    let ev = [1.0, 4.0, 6.0, 9.0];
    let obs = VpcCensoring::Observed;
    assert_eq!(replicate_follow_up(&ev, 4.0, 20.0, obs), (&ev[..2], 4.0));
    assert_eq!(replicate_follow_up(&ev, 5.0, 20.0, obs), (&ev[..2], 5.0));
    assert_eq!(replicate_follow_up(&ev, 30.0, 20.0, obs), (&ev[..], 20.0));
    assert_eq!(
        replicate_follow_up(&ev, 4.0, 20.0, VpcCensoring::Horizon),
        (&ev[..], 20.0)
    );
}

/// 30 subjects followed to 4 (ARM 0) beside 30 followed to 20 (ARM 1).
fn short_and_long() -> Population {
    let mut subjects = cohort(1, 30, 4.0, &[1.0]);
    subjects.extend(cohort(31, 30, 20.0, &[1.0, 12.0]));
    for (i, s) in subjects.iter_mut().enumerate() {
        s.covariates
            .insert("ARM".into(), if i < 30 { 0.0 } else { 1.0 });
    }
    pop(subjects)
}

/// Through the entry point, by arm. Under `Observed` no ARM-0 replicate is followed
/// past 4, so at t = 5 none of its curves is defined, and with every replicate step on
/// the grid none of its times is past 4. Under `Horizon` both arms run to `H = 20`.
/// With `sim.horizon = 2`, below every follow-up end, `H` wins in both arms. Ending
/// follow-up at `H` under `Observed` makes the first ARM-0 count at 5 twenty.
#[test]
fn observed_follow_up_ends_each_replicate_subject_at_its_own_time() {
    let p = short_and_long();
    let mut o = opts(20);
    o.max_order = Some(2);
    o.stratify = VpcStratify::Covariate("ARM".into());
    o.grid = VpcGrid::Times(vec![3.0, 5.0]);
    let v = run(RTTE, &p, &o).unwrap();
    assert_eq!((v.horizon, v.censoring), (20.0, VpcCensoring::Observed));
    assert_eq!(v.strata[0].label, "ARM=0");
    for (arm, want) in [(0, [20, 0]), (1, [20, 20])] {
        let s = &v.strata[arm];
        assert_eq!(s.mcf.n_used, want, "MCF, ARM {arm}");
        assert_eq!(s.absolute[1].n_used, want, "order 2, ARM {arm}");
    }
    assert_eq!(v.strata[0].mcf.lo[1], None);

    o.grid = VpcGrid::AllSteps;
    let v = run(RTTE, &p, &o).unwrap();
    let last = |g: &[f64]| *g.last().unwrap();
    assert_eq!(last(&v.strata[0].mcf.grid), 4.0);
    assert_eq!(last(&v.strata[0].absolute[0].grid), 4.0);
    assert_eq!(last(&v.strata[1].mcf.grid), 20.0);
    assert!(
        v.strata[0].mcf.grid.len() > 20,
        "replicate steps on the grid"
    );

    o.grid = VpcGrid::Times(vec![3.0, 5.0]);
    o.censoring = VpcCensoring::Horizon;
    let v = run(RTTE, &p, &o).unwrap();
    assert_eq!(v.strata[0].mcf.n_used, [20, 20]);
    assert_eq!(v.strata[0].absolute[1].n_used, [20, 20]);

    o.censoring = VpcCensoring::Observed;
    o.sim.horizon = Some(2.0);
    o.grid = VpcGrid::Times(vec![1.5, 3.0]);
    let v = run(RTTE, &p, &o).unwrap();
    assert_eq!(v.horizon, 2.0);
    for s in &v.strata {
        assert_eq!(s.mcf.n_used, [20, 0], "{}", s.label);
    }
    // The observed curves are not cut at `H`.
    assert_eq!(v.strata[1].mcf.observed.time.last(), Some(&20.0));
}

/// One stratum holding both cohorts, rate 0.15 with no variability, so the replicate
/// MCF estimates `0.15·t` at every `t` when the events and the risk set end together.
/// Keeping a short subject's events after its follow-up end, with the risk set still
/// cut at 4, credits the 30 long subjects with all 60 subjects' events past 4: the
/// median at 20 becomes about `0.6 + 2·2.4 = 5.4` against 3.0.
#[test]
fn replicate_events_after_the_follow_up_end_are_dropped() {
    let mut o = opts(60);
    o.max_order = Some(1);
    o.grid = VpcGrid::Times(vec![4.0, 20.0]);
    let v = run(RTTE, &short_and_long(), &o).unwrap();
    let m = &v.strata[0].mcf;
    assert_eq!(m.n_used, [60, 60]);
    for (k, want) in [(0, 0.6), (1, 3.0)] {
        let med = m.med[k].expect("60 replicates resolve the band");
        assert!(
            med.is_finite() && (med - want).abs() < 0.3,
            "median {med} at grid point {k}"
        );
        let (lo, hi) = (m.lo[k].unwrap(), m.hi[k].unwrap());
        assert!(lo <= med && med <= hi && lo < hi, "{lo} {med} {hi}");
    }
}

// ── T-sim: one simulated stream per subject per replicate ────────────────────

/// A subject observed with four records gets one regenerated stream per replicate, not
/// one per record: the simulator's rows for it hold exactly one censoring row, at `H`,
/// and its event times increase below `H`. Then through the entry point, 30 such
/// subjects under `Horizon`: the replicate MCF at 10 estimates `0.15·10 = 1.5`. Reading
/// a stream once per observed record would give 6.
#[test]
fn simulator_gives_one_stream_per_subject_per_replicate() {
    let p = pop(cohort(1, 30, 20.0, &[3.0, 8.0, 15.0]));
    assert_eq!(p.subjects[0].obs_records.len(), 4);
    let m = model(RTTE);
    let sim_opts = SimulateOptions {
        seed: Some(7),
        horizon: Some(20.0),
        ..Default::default()
    };
    let out = simulate_with_options_impl(&m, &p, &m.default_params, 3, &sim_opts, false).unwrap();
    for sim in 1..=3 {
        let rows: Vec<(f64, bool)> = out
            .results
            .iter()
            .filter(|r| r.sim == sim && r.id == "1" && r.cmt == 2)
            .map(|r| match r.outcome {
                SimOutcome::Event { time, observed } => (time, observed),
                _ => panic!("an RTTE row is an event outcome"),
            })
            .collect();
        let censorings: Vec<f64> = rows.iter().filter(|r| !r.1).map(|r| r.0).collect();
        assert_eq!(censorings, [20.0], "replicate {sim}: one stream");
        let events: Vec<f64> = rows.iter().filter(|r| r.1).map(|r| r.0).collect();
        assert!(events.windows(2).all(|w| w[0] < w[1]) && events.iter().all(|&t| t < 20.0));
    }

    let mut o = opts(60);
    o.max_order = Some(1);
    o.censoring = VpcCensoring::Horizon;
    o.grid = VpcGrid::Times(vec![10.0]);
    let v = run(RTTE, &p, &o).unwrap();
    let med = v.strata[0].mcf.med[0].unwrap();
    assert!(med.is_finite() && (med - 1.5).abs() < 0.3, "median {med}");
    assert!(v.warnings.is_empty(), "{:?}", v.warnings);
}

// ── T9: which orders ─────────────────────────────────────────────────────────

/// Ten subjects have three or more events and nine have four, so the default highest
/// order is 3 (an "at least 9" rule gives 4, "more than 10" gives 1). A given
/// `max_order` is used as is. A gap order nobody is at risk for (order 6 needs five
/// events) has an empty observed curve and no band, and is not an error.
#[test]
fn default_max_order_needs_ten_subjects_and_a_given_one_is_honoured() {
    let mut subjects = cohort(1, 9, 30.0, &[1.0, 2.0, 3.0, 4.0]);
    subjects.extend(cohort(10, 1, 30.0, &[1.0, 2.0, 3.0]));
    subjects.extend(cohort(11, 5, 30.0, &[]));
    let p = pop(subjects);
    let v = run(RTTE, &p, &opts(20)).unwrap();
    assert_eq!(v.max_order, 3);
    assert_eq!((v.strata[0].absolute.len(), v.strata[0].gap.len()), (3, 2));

    let mut o = opts(20);
    o.max_order = Some(6);
    let v = run(RTTE, &p, &o).unwrap();
    assert_eq!(v.max_order, 6);
    let s = &v.strata[0];
    assert_eq!((s.absolute.len(), s.gap.len()), (6, 5));
    // Absolute order 6: every subject is in it, censored.
    assert_eq!(s.absolute[5].n_subjects, 15);
    assert_eq!(s.absolute[5].observed.n_event, [0]);
    // Gap order 5: the nine subjects with four events.
    assert_eq!((s.gap[3].order, s.gap[3].n_subjects), (5, 9));
    let g6 = &s.gap[4];
    assert_eq!((g6.order, g6.n_subjects), (6, 0));
    assert_eq!(g6.observed, KmCurve::default());
    assert!(!g6.grid.is_empty());
    assert!(g6.obs_on_grid.iter().all(Option::is_none));
    assert!(g6
        .lo
        .iter()
        .chain(&g6.med)
        .chain(&g6.hi)
        .all(Option::is_none));
    assert!(g6.n_used.iter().all(|&n| n == 0));

    // Fewer than ten subjects with any event: order 1 still.
    let few = pop(cohort(1, 4, 30.0, &[1.0, 2.0]));
    assert_eq!(run(RTTE, &few, &opts(20)).unwrap().max_order, 1);
}

// ── T10: strata ──────────────────────────────────────────────────────────────

/// R1 as ARM 0 beside three long-followed subjects as ARM 1. Each stratum's curves use
/// its own subjects: ARM 0's are R1's (pooling the follow-up ends would start its risk
/// set at 9), and no ARM-0 replicate is followed past 10 while ARM 1's run to 20.
#[test]
fn each_stratum_uses_its_own_subjects() {
    let mut subjects = r1();
    subjects.push(stream("G", 20.0, &[1.0]));
    subjects.push(stream("H", 20.0, &[1.0, 2.0]));
    subjects.push(stream("I", 20.0, &[]));
    for (i, s) in subjects.iter_mut().enumerate() {
        s.covariates
            .insert("ARM".into(), if i < 6 { 0.0 } else { 1.0 });
    }
    let mut o = opts(20);
    o.max_order = Some(2);
    o.stratify = VpcStratify::Covariate("ARM".into());
    o.grid = VpcGrid::Times(vec![5.0, 15.0]);
    let v = run(RTTE, &pop(subjects), &o).unwrap();
    assert_eq!(v.horizon, 20.0);
    let (a, b) = (&v.strata[0], &v.strata[1]);
    assert_eq!(
        (a.label.as_str(), a.value, a.n_subjects),
        ("ARM=0", Some(0.0), 6)
    );
    assert_eq!(
        (b.label.as_str(), b.value, b.n_subjects),
        ("ARM=1", Some(1.0), 3)
    );
    assert_eq!(a.mcf.observed.n_risk, [6, 6, 6, 6, 5, 4, 3, 3, 2, 2]);
    assert_eq!(a.absolute[1].observed.n_risk, [6, 5, 3, 2, 1]);
    assert_eq!(a.gap[0].n_subjects, 5);
    assert_eq!(b.mcf.observed.time, [1.0, 2.0, 20.0]);
    assert_eq!(b.mcf.observed.n_risk, [3, 3, 3]);
    assert_eq!(b.mcf.observed.n_event, [2, 1, 0]);
    assert_eq!(b.absolute[0].n_subjects, 3);
    assert_eq!(b.gap[0].n_subjects, 2);
    assert_eq!(a.mcf.n_used, [20, 0]);
    assert_eq!(b.mcf.n_used, [20, 20]);
}

/// Without stratification there is one stratum, `all`; by dose the label is the first
/// dose amount.
#[test]
fn strata_labels() {
    let v = run(RTTE, &pop(r1()), &opts(20)).unwrap();
    assert_eq!(v.strata.len(), 1);
    assert_eq!(
        (v.strata[0].label.as_str(), v.strata[0].value),
        ("all", None)
    );
    let mut subjects = r1();
    for s in subjects.iter_mut().take(3) {
        s.doses = vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    }
    let mut o = opts(20);
    o.stratify = VpcStratify::Dose;
    let v = run(RTTE, &pop(subjects), &o).unwrap();
    let labels: Vec<&str> = v.strata.iter().map(|s| s.label.as_str()).collect();
    assert_eq!(labels, ["dose=0", "dose=100"]);
}

// ── T11: refusals, whole text ────────────────────────────────────────────────

/// The refusal `got` must carry `E_VPC_TTE_INPUT` and exactly `text`, so deleting or
/// rewording any sentence of it fails here, and so does another entry point's prefix.
fn assert_refusal<T: std::fmt::Debug>(got: Result<T, EngineError>, text: &str) {
    let e = got.expect_err("must refuse");
    assert_eq!(e.code(), Some("E_VPC_TTE_INPUT"), "{e}");
    assert_eq!(e.to_string(), text);
}

#[test]
fn refuses_a_single_event_endpoint() {
    let p = pop(vec![subj("1", vec![rec(1.0, EventType::Exact)])]);
    assert_refusal(
        run(SINGLE, &p, &opts(20)),
        "vpc_rtte: cmt 2 is a single-event TTE endpoint. Use `vpc_tte` for its Kaplan–Meier \
         VPC.",
    );
}

#[test]
fn refuses_no_tte_endpoint_and_a_cmt_that_is_not_one() {
    assert_refusal(
        run(PK_ONLY, &pop(r1()), &opts(20)),
        "vpc_rtte: the model declares no TTE endpoint ([event_model]).",
    );
    let mut o = opts(20);
    o.cmt = Some(1);
    assert_refusal(
        run(RTTE, &pop(r1()), &o),
        "vpc_rtte: `cmt = 1` is not a TTE endpoint. The model's TTE endpoints are on cmt 2.",
    );
}

/// No competing-risks sentence here: that one is about reading a single-event curve.
#[test]
fn refuses_several_tte_endpoints_without_cmt() {
    assert_refusal(
        run(TWO_ENDPOINTS, &pop(r1()), &opts(20)),
        "vpc_rtte: the model has TTE endpoints on cmt 2, 3; set `cmt` to the one to check.",
    );
}

#[test]
fn refuses_left_truncation() {
    let p = pop(vec![subj(
        "7",
        vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 1.5,
            cmt: 2,
        }],
    )]);
    assert_refusal(
        run(RTTE, &p, &opts(20)),
        "vpc_rtte: subject 7 has entry time 1.5 on cmt 2. Left truncation (delayed entry) is \
         not supported yet.",
    );
}

#[test]
fn refuses_interval_censoring() {
    let p = pop(vec![subj(
        "8",
        vec![rec(
            5.0,
            EventType::IntervalCensored {
                left: 2.0,
                right: 5.0,
            },
        )],
    )]);
    assert_refusal(
        run(RTTE, &p, &opts(20)),
        "vpc_rtte: subject 8 has an interval-censored record on cmt 2. The repeated-events \
         curves need exact or right-censored times.",
    );
}

#[test]
fn refuses_a_bad_observed_time() {
    for (t, shown) in [(f64::NAN, "NaN"), (-1.0, "-1")] {
        let p = pop(vec![
            stream("1", 5.0, &[2.0]),
            subj("9", vec![rec(t, EventType::Exact)]),
        ]);
        assert_refusal(
            run(RTTE, &p, &opts(20)),
            &format!(
                "vpc_rtte: subject 9 has time {shown} on cmt 2, which is not a finite, \
                 non-negative number."
            ),
        );
    }
}

/// Tied records are in order; only a strictly earlier time is refused.
#[test]
fn refuses_records_out_of_time_order() {
    let p = pop(vec![subj(
        "3",
        vec![
            rec(4.0, EventType::Exact),
            rec(2.5, EventType::Exact),
            rec(9.0, EventType::RightCensored),
        ],
    )]);
    assert_refusal(
        run(RTTE, &p, &opts(20)),
        "vpc_rtte: subject 3 has records on cmt 2 out of time order (time 2.5 follows 4). \
         Sort each subject's records by time.",
    );
    let tied = pop(vec![subj(
        "3",
        vec![rec(4.0, EventType::Exact), rec(4.0, EventType::Exact)],
    )]);
    assert!(run(RTTE, &tied, &opts(20)).is_ok());
}

#[test]
fn refuses_duplicate_ids_and_no_records() {
    let p = pop(vec![stream("5", 5.0, &[1.0]), stream("5", 6.0, &[2.0])]);
    assert_refusal(
        run(RTTE, &p, &opts(20)),
        "vpc_rtte: subject id 5 appears more than once in the population. Simulated rows are \
         matched to subjects by id.",
    );
    assert_refusal(
        run(RTTE, &pop(vec![subj("6", vec![])]), &opts(20)),
        "vpc_rtte: no subject has a record on cmt 2.",
    );
}

#[test]
fn refuses_max_order_zero() {
    let mut o = opts(20);
    o.max_order = Some(0);
    assert_refusal(
        run(RTTE, &pop(r1()), &o),
        "vpc_rtte: `max_order = 0` asks for no event order. Set it to at least 1, or leave it \
         unset to take it from the data.",
    );
}

#[test]
fn refuses_bad_band_options() {
    let p = pop(r1());
    let mut o = opts(20);
    o.probs = [0.5, 0.05, 0.95];
    assert_refusal(
        run(RTTE, &p, &o),
        "vpc_rtte: band probabilities [0.5, 0.05, 0.95] must be strictly inside (0, 1) and \
         strictly increasing.",
    );
    assert_refusal(
        run(RTTE, &p, &opts(19)),
        "vpc_rtte: `n_sim = 19` is too few. The band at probabilities [0.05, 0.5, 0.95] needs \
         at least 20 replicates.",
    );
    let mut o = opts(20);
    o.grid = VpcGrid::Times(vec![1.0, -2.0]);
    assert_refusal(
        run(RTTE, &p, &o),
        "vpc_rtte: grid time -2 is not a finite, non-negative number.",
    );
}

#[test]
fn refuses_bad_strata() {
    let mut o = opts(20);
    o.stratify = VpcStratify::Covariate("WT".into());
    assert_refusal(
        run(RTTE, &pop(r1()), &o),
        "vpc_rtte: subject A has no covariate `WT` to stratify on.",
    );
    let mut subjects = cohort(1, 11, 5.0, &[1.0]);
    for (i, s) in subjects.iter_mut().enumerate() {
        s.covariates.insert("WT".into(), 50.0 + i as f64);
    }
    assert_refusal(
        run(RTTE, &pop(subjects), &o),
        "vpc_rtte: covariate `WT` has 11 distinct values, more than the 10 strata a VPC can \
         show. Bin it into a categorical covariate and stratify on that.",
    );
}
