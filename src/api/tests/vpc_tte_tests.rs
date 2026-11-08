//! Tier-1 tests for the Kaplan–Meier VPC driver (#1886): the post-censoring step
//! (plan T7), the pipeline around it, and every refusal's whole text (T11).

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Subject};

const EXP_TTE: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

const COMPETING: &str = r"
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

const RTTE: &str = "\n[parameters]\n  theta TVLAMBDA(0.15, 0.001, 10.0)\n\n[event_model]\n  cmt    = 2\n  type   = rtte\n  family = exponential\n  scale  = TVLAMBDA\n";

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

fn rec(time: f64, event_type: EventType, cmt: usize) -> ObsRecord {
    ObsRecord::Event {
        time,
        event_type,
        entry_time: 0.0,
        cmt,
    }
}

fn subj(id: &str, records: Vec<ObsRecord>) -> Subject {
    Subject {
        id: id.into(),
        obs_records: records,
        ..Default::default()
    }
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

/// `n` subjects on cmt 2, every one censored at `c` (`event = false`) or with an event
/// at `c` (`event = true`).
fn uniform_pop(n: usize, c: f64, event: bool) -> Population {
    let et = if event {
        EventType::Exact
    } else {
        EventType::RightCensored
    };
    pop((1..=n)
        .map(|i| subj(&i.to_string(), vec![rec(c, et.clone(), 2)]))
        .collect())
}

fn opts(n_sim: usize) -> VpcTteOptions {
    VpcTteOptions {
        n_sim,
        sim: SimulateOptions {
            seed: Some(7),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn run(src: &str, p: &Population, o: &VpcTteOptions) -> Result<TteVpc, EngineError> {
    let m = model(src);
    vpc_tte(&m, p, &m.default_params, o)
}

// ── T7: the post-censoring step ──────────────────────────────────────────────

/// A draw past the subject's censoring time is censored there; a draw at or before it
/// is kept. Skipping the step breaks the first two assertions.
#[test]
fn post_censor_censors_draws_past_the_censoring_time() {
    assert_eq!(post_censor(6.0, true, 4.0), (4.0, false));
    assert_eq!(post_censor(6.0, false, 4.0), (4.0, false));
    assert_eq!(post_censor(3.0, true, 4.0), (3.0, true));
    assert_eq!(post_censor(4.0, true, 4.0), (4.0, true));
}

/// The conditional draw on a hand-computed reverse Kaplan–Meier curve. Observed rows:
/// event at 1, censored at 2, event at 3, censored at 4, censored at 6, so
/// `G(2) = 3/4`, `G(4) = 3/8`, `G(6) = 0`. From `t = 1` (`G = 1`) a uniform `u` maps to
/// the first censoring time with `G ≤ u`; from `t = 3` (`G = 3/4`) to the first after 3
/// with `G / (3/4) ≤ u`, so never back to 2. Dropping the division by `G(t)` returns
/// 6 for `u = 0.6`; reading the censoring curve the wrong way round changes `g.surv`.
#[test]
fn draw_censoring_inverts_the_conditional_censoring_distribution() {
    let rows: Vec<ObsRow> = [
        (1.0, None),
        (2.0, Some(2.0)),
        (3.0, None),
        (4.0, Some(4.0)),
        (6.0, Some(6.0)),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (time, censor_at))| ObsRow {
        id: i.to_string(),
        time,
        event: censor_at.is_none(),
        censor_at,
        stratum: 0,
    })
    .collect();
    let g = censoring_curve(rows.iter());
    assert_eq!(g.surv, [1.0, 0.75, 0.75, 0.375, 0.0]);
    assert_eq!(draw_censoring(&g, 1.0, 0.8, 9.0), 2.0);
    assert_eq!(draw_censoring(&g, 1.0, 0.5, 9.0), 4.0);
    assert_eq!(draw_censoring(&g, 1.0, 0.2, 9.0), 6.0);
    assert_eq!(draw_censoring(&g, 3.0, 0.6, 9.0), 4.0);
    assert_eq!(draw_censoring(&g, 3.0, 0.4, 9.0), 6.0);
    assert_eq!(draw_censoring(&g, 3.0, 0.99, 9.0), 4.0);
    // No censoring mass past t: the horizon.
    assert_eq!(draw_censoring(&g, 6.0, 0.5, 9.0), 9.0);
    let none = kaplan_meier(&[1.0, 2.0], &[false, false]).unwrap();
    assert_eq!(draw_censoring(&none, 1.0, 0.0, 9.0), 9.0);
}

/// The censoring uniforms are in `[0, 1)`, depend on every key, and look uniform.
#[test]
fn censoring_uniform_is_keyed_and_uniform() {
    let u: Vec<f64> = (0..10_000)
        .map(|i| censoring_uniform(3, 1 + i / 100, i % 100))
        .collect();
    assert!(u.iter().all(|x| (0.0..1.0).contains(x)));
    let mean = u.iter().sum::<f64>() / u.len() as f64;
    assert!((mean - 0.5).abs() < 0.01, "{mean}");
    assert_ne!(censoring_uniform(3, 1, 0), censoring_uniform(4, 1, 0));
    assert_ne!(censoring_uniform(3, 1, 0), censoring_uniform(3, 2, 0));
    assert_ne!(censoring_uniform(3, 1, 0), censoring_uniform(3, 1, 1));
}

/// Through the pipeline: every subject censored at 2, horizon 10. Under `Observed`
/// no replicate is followed past 2, so at t = 5 no curve is defined; under `Horizon`
/// all 20 are. Skipping the post-censoring makes the first count 20.
#[test]
fn observed_censoring_stops_replicates_at_each_subjects_own_time() {
    let p = uniform_pop(30, 2.0, false);
    let mut o = opts(20);
    o.sim.horizon = Some(10.0);
    o.grid = VpcGrid::Times(vec![1.0, 5.0]);
    let v = run(EXP_TTE, &p, &o).unwrap();
    assert_eq!(v.horizon, 10.0);
    assert_eq!(v.strata[0].n_used, [20, 0]);
    assert_eq!(v.strata[0].lo[1], None);
    o.censoring = VpcCensoring::Horizon;
    let v = run(EXP_TTE, &p, &o).unwrap();
    assert_eq!(v.strata[0].n_used, [20, 20]);
    assert!(v.strata[0].lo[1].unwrap().is_finite());
}

/// 15 subjects censored at 2 and 15 with an event at 1, horizon 10. The censoring
/// distribution puts all its mass at 2, so each event subject draws `c = 2` and no
/// replicate is followed past 2: at t = 5 no curve is defined. Letting event subjects
/// keep the horizon (#1886 plan D4) follows them to 10 and makes the count 20.
#[test]
fn event_subjects_draw_censoring_from_the_observed_censorings() {
    let subjects = (1..=30)
        .map(|i| {
            let r = if i <= 15 {
                rec(2.0, EventType::RightCensored, 2)
            } else {
                rec(1.0, EventType::Exact, 2)
            };
            subj(&i.to_string(), vec![r])
        })
        .collect();
    let mut o = opts(20);
    o.sim.horizon = Some(10.0);
    o.grid = VpcGrid::Times(vec![1.5, 5.0]);
    let v = run(EXP_TTE, &pop(subjects), &o).unwrap();
    assert_eq!(v.strata[0].n_used, [20, 0]);
}

/// Two arms with different dropout (review r1 row 1). ARM = 0: 15 censored at 2 and 15
/// events at 1. ARM = 1: 15 censored at 8 and 15 events at 1. Horizon 10. Within ARM 0
/// the censoring distribution puts all its mass at 2, so its event subjects draw `c = 2`
/// and no ARM-0 replicate is followed past 2: at t = 5 none is defined. A censoring
/// distribution pooled across the arms lets ARM-0 event subjects draw ARM 1's 8, and the
/// ARM-0 count at 5 becomes 20 (a band under another arm's dropout).
#[test]
fn censoring_distribution_is_per_stratum() {
    let subjects = (1..=60)
        .map(|i| {
            let arm = if i <= 30 { 0.0 } else { 1.0 };
            let censored_at = if arm == 0.0 { 2.0 } else { 8.0 };
            let r = if i % 2 == 0 {
                rec(censored_at, EventType::RightCensored, 2)
            } else {
                rec(1.0, EventType::Exact, 2)
            };
            let mut s = subj(&i.to_string(), vec![r]);
            s.covariates.insert("ARM".into(), arm);
            s
        })
        .collect();
    let mut o = opts(20);
    o.sim.horizon = Some(10.0);
    o.grid = VpcGrid::Times(vec![1.5, 5.0]);
    o.stratify = VpcStratify::Covariate("ARM".into());
    let v = run(EXP_TTE, &pop(subjects), &o).unwrap();
    assert_eq!(v.strata[0].label, "ARM=0");
    assert_eq!(
        v.strata[0].n_used,
        [20, 0],
        "ARM 0 is never followed past 2"
    );
    assert_eq!(v.strata[1].n_used, [20, 20], "ARM 1 is followed to 8");
}

/// Competing risks under the default `Observed` censoring (review r1 row 2). Every subject
/// had cause 2 at time 1, so its cause-3 record is right-censored at 1. That censoring is
/// the competing event, not an observed dropout: the subject was never seen censored
/// (no observed censoring at all, so the horizon applies), and the simulator draws the
/// competing cause itself. So replicates are followed to 10 and at t = 5 curves are
/// defined. Treating the cause-3 record as the subject's own censoring time censors every
/// replicate at 1 a second time and makes the count at 5 zero.
#[test]
fn competing_event_is_not_an_observed_censoring() {
    let subjects = (1..=30)
        .map(|i| {
            subj(
                &i.to_string(),
                vec![
                    rec(1.0, EventType::Exact, 2),
                    rec(1.0, EventType::RightCensored, 3),
                ],
            )
        })
        .collect();
    let mut o = opts(20);
    o.cmt = Some(3);
    o.sim.horizon = Some(10.0);
    o.grid = VpcGrid::Times(vec![0.5, 5.0]);
    let v = run(COMPETING, &pop(subjects), &o).unwrap();
    assert_eq!(v.censoring, VpcCensoring::Observed);
    assert_eq!(v.strata[0].n_used, [20, 20]);
}

/// Subjects observed with an event at 1, and no censoring observed at all: the
/// censoring distribution has no mass, so every replicate keeps the horizon. Using the
/// event time as a censoring time would end every replicate at 1 and make the count at 5
/// zero.
#[test]
fn event_subjects_keep_the_horizon_without_observed_censoring() {
    let p = uniform_pop(30, 1.0, true);
    let mut o = opts(20);
    o.sim.horizon = Some(10.0);
    o.grid = VpcGrid::Times(vec![5.0]);
    let v = run(EXP_TTE, &p, &o).unwrap();
    assert_eq!(v.strata[0].n_used, [20]);
}

/// Without a horizon, `H` is the latest recorded time; with one below a subject's own
/// censoring time, `H` wins: no replicate is followed past it, and the automatic grid
/// stops at it.
#[test]
fn horizon_defaults_to_latest_record_and_wins_when_below_it() {
    let mut subjects: Vec<Subject> = (1..=25)
        .map(|i| subj(&i.to_string(), vec![rec(i as f64, EventType::Exact, 2)]))
        .collect();
    subjects.push(subj("26", vec![rec(30.0, EventType::RightCensored, 2)]));
    let p = pop(subjects);
    let v = run(EXP_TTE, &p, &opts(20)).unwrap();
    assert_eq!(v.horizon, 30.0);
    let s = &v.strata[0];
    assert_eq!(s.n_subjects, 26);
    assert_eq!(s.label, "all");
    assert_eq!(s.value, None);
    assert_eq!(s.grid.first(), Some(&0.0));
    assert_eq!(s.grid.last(), Some(&30.0));
    // The observed times are on the grid, and the overlay reads the observed curve.
    for t in [1.0, 13.0, 25.0] {
        let k = s
            .grid
            .iter()
            .position(|&g| g == t)
            .expect("observed time on grid");
        assert_eq!(s.obs_on_grid[k], s.observed.at(t));
    }
    // 101 even points plus 26 observed times, minus those that coincide.
    assert!(
        s.grid.len() > 101 && s.grid.len() <= 101 + 26,
        "{}",
        s.grid.len()
    );
    assert!(s.n_used.iter().all(|&n| n <= 20));

    let mut o = opts(20);
    o.sim.horizon = Some(5.0);
    let v = run(EXP_TTE, &p, &o).unwrap();
    assert_eq!(v.horizon, 5.0);
    assert_eq!(v.strata[0].grid.last(), Some(&5.0));
    // No replicate is followed past H = 5, although subject 26 is censored at 30 and
    // every event subject's drawn censoring time is 30: at 6 no curve is defined, at 4
    // all are. Simulating to the subject's own 30 instead makes the count at 6 twenty.
    o.grid = VpcGrid::Times(vec![4.0, 6.0]);
    let v = run(EXP_TTE, &p, &o).unwrap();
    assert_eq!(v.strata[0].n_used, [20, 0]);
}

/// `AllSteps` puts every replicate step time on the grid, so it is at least as long as
/// the observed times and contains each replicate curve's times.
#[test]
fn all_steps_grid_holds_every_replicate_step() {
    let p = uniform_pop(30, 40.0, false);
    let mut o = opts(20);
    o.grid = VpcGrid::AllSteps;
    let v = run(EXP_TTE, &p, &o).unwrap();
    let s = &v.strata[0];
    // 30 subjects × 20 replicates draw distinct continuous times below 40.
    assert!(s.grid.len() > 200, "{}", s.grid.len());
    assert!(s.grid.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(s.grid.last(), Some(&40.0));
}

/// Strata by covariate, ordered by value, each with its own observed curve.
#[test]
fn stratify_by_covariate_splits_subjects_by_exact_value() {
    let subjects = (1..=30)
        .map(|i| {
            let mut s = subj(&i.to_string(), vec![rec(i as f64, EventType::Exact, 2)]);
            s.covariates
                .insert("SEX".into(), if i % 3 == 0 { 1.0 } else { 0.0 });
            s
        })
        .collect();
    let mut o = opts(20);
    o.stratify = VpcStratify::Covariate("SEX".into());
    let v = run(EXP_TTE, &pop(subjects), &o).unwrap();
    let labels: Vec<&str> = v.strata.iter().map(|s| s.label.as_str()).collect();
    assert_eq!(labels, ["SEX=0", "SEX=1"]);
    assert_eq!(v.strata[0].value, Some(0.0));
    assert_eq!(v.strata[0].n_subjects, 20);
    assert_eq!(v.strata[1].n_subjects, 10);
    assert_eq!(
        v.strata[1].observed.time,
        [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
    );
}

/// Strata by first dose amount; a subject with no dose is `dose=0`.
#[test]
fn stratify_by_dose_reads_the_first_dose() {
    let subjects = (1..=30)
        .map(|i| {
            let mut s = subj(&i.to_string(), vec![rec(i as f64, EventType::Exact, 2)]);
            if i > 10 {
                // The first dose decides: the second is the same for everyone.
                let amt = if i > 20 { 200.0 } else { 100.0 };
                s.doses = vec![
                    DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0),
                    DoseEvent::new(1.0, 50.0, 1, 0.0, false, 0.0),
                ];
            }
            s
        })
        .collect();
    let mut o = opts(20);
    o.stratify = VpcStratify::Dose;
    let v = run(EXP_TTE, &pop(subjects), &o).unwrap();
    let labels: Vec<&str> = v.strata.iter().map(|s| s.label.as_str()).collect();
    assert_eq!(labels, ["dose=0", "dose=100", "dose=200"]);
    assert!(v.strata.iter().all(|s| s.n_subjects == 10));
}

/// Competing risks with `cmt` named: the curve is that cause's, with the other cause's
/// events as censoring.
#[test]
fn competing_risks_with_cmt_reads_that_cause() {
    let subjects = (1..=30)
        .map(|i| {
            let (a, b) = if i % 2 == 0 {
                (EventType::Exact, EventType::RightCensored)
            } else {
                (EventType::RightCensored, EventType::Exact)
            };
            subj(
                &i.to_string(),
                vec![rec(i as f64, a, 2), rec(i as f64, b, 3)],
            )
        })
        .collect();
    let mut o = opts(20);
    o.cmt = Some(3);
    o.n_sim = 60;
    o.censoring = VpcCensoring::Horizon;
    o.grid = VpcGrid::Times(vec![20.0]);
    let v = run(COMPETING, &pop(subjects), &o).unwrap();
    assert_eq!(v.cmt, 3);
    assert_eq!(v.strata[0].observed.n_event.iter().sum::<usize>(), 15);
    // The replicate median at 20 estimates cause 3's survival exp(−0.03·20) = 0.549.
    // Reading both causes' rows instead doubles the risk set at each event and gives
    // about exp(−0.04·20) = 0.449.
    let med = v.strata[0].med[0].unwrap();
    assert!((med - (-0.6f64).exp()).abs() < 0.05, "median {med}");
}

// ── T11: refusals, whole text ────────────────────────────────────────────────

/// The refusal `got` must carry `E_VPC_TTE_INPUT` and exactly `text`, so deleting or
/// rewording any sentence of it fails here.
fn assert_refusal<T: std::fmt::Debug>(got: Result<T, EngineError>, text: &str) {
    let e = got.expect_err("must refuse");
    assert_eq!(e.code(), Some("E_VPC_TTE_INPUT"), "{e}");
    assert_eq!(e.to_string(), text);
}

#[test]
fn refuses_two_tte_endpoints_without_cmt() {
    let p = pop(vec![subj(
        "1",
        vec![
            rec(1.0, EventType::Exact, 2),
            rec(1.0, EventType::RightCensored, 3),
        ],
    )]);
    assert_refusal(
        run(COMPETING, &p, &opts(20)),
        "vpc_tte: the model has TTE endpoints on cmt 2, 3; set `cmt` to the one to check. \
         Under competing risks 1 − KM of one cause is not its cumulative incidence: the curve \
         is that cause's survival with the other causes as censoring.",
    );
}

#[test]
fn refuses_repeated_events() {
    assert_refusal(
        run(RTTE, &uniform_pop(3, 1.0, true), &opts(20)),
        "vpc_tte: cmt 2 is a repeated-events (RTTE) endpoint. A Kaplan–Meier VPC of repeated \
         events is not supported yet.",
    );
}

#[test]
fn refuses_no_tte_endpoint_and_a_cmt_that_is_not_one() {
    assert_refusal(
        run(PK_ONLY, &uniform_pop(3, 1.0, true), &opts(20)),
        "vpc_tte: the model declares no TTE endpoint ([event_model]).",
    );
    let mut o = opts(20);
    o.cmt = Some(1);
    assert_refusal(
        run(EXP_TTE, &uniform_pop(3, 1.0, true), &o),
        "vpc_tte: `cmt = 1` is not a TTE endpoint. The model's TTE endpoints are on cmt 2.",
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
        run(EXP_TTE, &p, &opts(20)),
        "vpc_tte: subject 7 has entry time 1.5 on cmt 2. Left truncation (delayed entry) is not \
         supported yet.",
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
            2,
        )],
    )]);
    assert_refusal(
        run(EXP_TTE, &p, &opts(20)),
        "vpc_tte: subject 8 has an interval-censored record on cmt 2. A Kaplan–Meier curve \
         needs exact or right-censored times.",
    );
}

/// The floor is `min_curves_for(probs) + 1`: 20 at 5/95, 40 at 2.5/97.5.
#[test]
fn refuses_too_few_replicates() {
    let p = uniform_pop(3, 1.0, true);
    assert_refusal(
        run(EXP_TTE, &p, &opts(19)),
        "vpc_tte: `n_sim = 19` is too few. The band at probabilities [0.05, 0.5, 0.95] needs \
         at least 20 replicates.",
    );
    let mut o = opts(39);
    o.probs = [0.025, 0.5, 0.975];
    assert_refusal(
        run(EXP_TTE, &p, &o),
        "vpc_tte: `n_sim = 39` is too few. The band at probabilities [0.025, 0.5, 0.975] needs \
         at least 40 replicates.",
    );
}

#[test]
fn refuses_bad_probs() {
    let mut o = opts(20);
    o.probs = [0.5, 0.05, 0.95];
    assert_refusal(
        run(EXP_TTE, &uniform_pop(3, 1.0, true), &o),
        "vpc_tte: band probabilities [0.5, 0.05, 0.95] must be strictly inside (0, 1) and \
         strictly increasing.",
    );
}

#[test]
fn refuses_too_many_strata() {
    let subjects = (1..=11)
        .map(|i| {
            let mut s = subj(&i.to_string(), vec![rec(1.0, EventType::Exact, 2)]);
            s.covariates.insert("WT".into(), 50.0 + i as f64);
            s
        })
        .collect();
    let mut o = opts(20);
    o.stratify = VpcStratify::Covariate("WT".into());
    assert_refusal(
        run(EXP_TTE, &pop(subjects), &o),
        "vpc_tte: covariate `WT` has 11 distinct values, more than the 10 strata a VPC can \
         show. Bin it into a categorical covariate and stratify on that.",
    );
}

#[test]
fn refuses_a_missing_stratification_covariate() {
    let mut o = opts(20);
    o.stratify = VpcStratify::Covariate("WT".into());
    assert_refusal(
        run(EXP_TTE, &uniform_pop(3, 1.0, true), &o),
        "vpc_tte: subject 1 has no covariate `WT` to stratify on.",
    );
}

#[test]
fn refuses_several_records_duplicate_ids_and_no_records() {
    let p = pop(vec![subj(
        "4",
        vec![
            rec(1.0, EventType::RightCensored, 2),
            rec(2.0, EventType::Exact, 2),
        ],
    )]);
    assert_refusal(
        run(EXP_TTE, &p, &opts(20)),
        "vpc_tte: subject 4 has 2 records on cmt 2. A single-event Kaplan–Meier curve takes \
         one per subject.",
    );
    let p = pop(vec![
        subj("5", vec![rec(1.0, EventType::Exact, 2)]),
        subj("5", vec![rec(2.0, EventType::Exact, 2)]),
    ]);
    assert_refusal(
        run(EXP_TTE, &p, &opts(20)),
        "vpc_tte: subject id 5 appears more than once in the population. Simulated rows are \
         matched to subjects by id.",
    );
    assert_refusal(
        run(EXP_TTE, &pop(vec![subj("6", vec![])]), &opts(20)),
        "vpc_tte: no subject has a record on cmt 2.",
    );
}

/// A hand-built population can carry a time the data reader would have skipped (review r1
/// row 5); it is refused, not panicked on.
#[test]
fn refuses_a_bad_observed_time() {
    for (t, shown) in [(f64::NAN, "NaN"), (-1.0, "-1")] {
        let p = pop(vec![
            subj("1", vec![rec(2.0, EventType::Exact, 2)]),
            subj("9", vec![rec(t, EventType::Exact, 2)]),
        ]);
        assert_refusal(
            run(EXP_TTE, &p, &opts(20)),
            &format!(
                "vpc_tte: subject 9 has time {shown} on cmt 2, which is not a finite, \
                 non-negative number."
            ),
        );
    }
}

#[test]
fn refuses_a_bad_grid_time() {
    let mut o = opts(20);
    o.grid = VpcGrid::Times(vec![1.0, -2.0]);
    assert_refusal(
        run(EXP_TTE, &uniform_pop(3, 1.0, true), &o),
        "vpc_tte: grid time -2 is not a finite, non-negative number.",
    );
}
