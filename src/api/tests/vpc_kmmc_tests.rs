//! Tier-1 tests for the Kaplan–Meier mean covariate VPC driver (#1886 PR3): the
//! replicate censoring (K6), the `t = 0` identity (K7), `level` (K7b), the driver's
//! at-risk floor (K7c) and every message's text (K8).

use super::super::vpc_tte::{replicate_censoring, ObsRow, RepStratum, VpcGrid};
use super::*;
use crate::api::SimulateOptions;
use crate::parser::model_parser::parse_model_string;
use crate::stats::km::KmCurve;
use crate::types::{EventType, ObsRecord};

/// Exponential single-event TTE with no random effect; `TVLAMBDA` is the rate.
fn exp_model(lambda: f64) -> CompiledModel {
    parse_model_string(&format!(
        "\n[parameters]\n  theta TVLAMBDA({lambda}, 1e-9, 10.0)\n\n[event_model]\n  cmt    = 2\n  \
         family = exponential\n  scale  = TVLAMBDA\n"
    ))
    .expect("model parses")
}

const RTTE: &str = "\n[parameters]\n  theta TVLAMBDA(0.15, 0.001, 10.0)\n\n[event_model]\n  cmt    = 2\n  type   = rtte\n  family = exponential\n  scale  = TVLAMBDA\n";

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

fn rec(time: f64, event_type: EventType, cmt: usize) -> ObsRecord {
    ObsRecord::Event {
        time,
        event_type,
        entry_time: 0.0,
        cmt,
    }
}

/// A subject with one record on cmt 2 and covariate `Z = z`.
fn subj(id: &str, time: f64, event: bool, z: f64) -> Subject {
    let et = if event {
        EventType::Exact
    } else {
        EventType::RightCensored
    };
    let mut s = Subject {
        id: id.into(),
        obs_records: vec![rec(time, et, 2)],
        ..Default::default()
    };
    s.covariates.insert("Z".into(), z);
    s
}

fn pop(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: vec!["Z".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn opts(n_sim: usize) -> VpcKmmcOptions {
    let mut o = VpcKmmcOptions::new("Z");
    o.vpc.n_sim = n_sim;
    o.vpc.sim = SimulateOptions {
        seed: Some(11),
        ..Default::default()
    };
    o
}

fn run(m: &CompiledModel, p: &Population, o: &VpcKmmcOptions) -> Result<KmmcVpc, EngineError> {
    vpc_kmmc(m, p, &m.default_params, o)
}

// ── K6: the replicate censoring ──────────────────────────────────────────────

/// A censoring distribution with one step, to 0 at `c0`.
fn one_step_g(c0: f64) -> KmCurve {
    KmCurve {
        time: vec![c0],
        n_risk: vec![1],
        n_event: vec![1],
        n_censor: vec![0],
        surv: vec![0.0],
        cumhaz: vec![1.0],
    }
}

fn obs_row(time: f64, event: bool) -> ObsRow {
    ObsRow {
        id: "1".into(),
        time,
        event,
        censor_at: (!event).then_some(time),
        stratum: 0,
    }
}

/// K6, unit level. `DrawAll` (KMMC) draws every subject's censoring time from `G`,
/// unconditionally: a subject observed censored at 3 gets `G`'s only censoring time 7,
/// never its own 3, and an event subject at 9 (past every censoring) still gets 7,
/// where `KeepObserved` (`vpc_tte`) keeps 3 and gives the event subject `H`. Using
/// `KeepObserved` for KMMC makes the first assertion 3.
#[test]
fn draw_all_ignores_the_subjects_own_censoring_time() {
    let g = one_step_g(7.0);
    let h = 50.0;
    for u in [0.0, 0.3, 0.999] {
        let censored = obs_row(3.0, false);
        let event = obs_row(9.0, true);
        assert_eq!(
            replicate_censoring(TteCensorRule::DrawAll, &censored, &g, u, h),
            7.0
        );
        assert_eq!(
            replicate_censoring(TteCensorRule::DrawAll, &event, &g, u, h),
            7.0,
            "an event subject draws unconditionally, not given c > t"
        );
        assert_eq!(
            replicate_censoring(TteCensorRule::KeepObserved, &censored, &g, u, h),
            3.0
        );
        assert_eq!(
            replicate_censoring(TteCensorRule::KeepObserved, &event, &g, u, h),
            h,
            "vpc_tte's rule: no censoring mass past 9"
        );
    }
}

/// Ten subjects: censored at 2 (rows 0–2) and at 8 (rows 3–5), events at 5 (rows 6–9).
/// Their reverse-KM `G` puts mass only on 2 and 8.
fn two_censoring_pop() -> Population {
    let mut s = Vec::new();
    for i in 0..10 {
        let (t, e) = match i {
            0..=2 => (2.0, false),
            3..=5 => (8.0, false),
            _ => (5.0, true),
        };
        s.push(subj(&(i + 1).to_string(), t, e, i as f64));
    }
    pop(s)
}

/// Every replicate time of observed row `row`, over all replicates.
fn times_of(reps: &[Vec<RepStratum>], row: usize) -> Vec<f64> {
    reps.iter()
        .flat_map(|r| {
            r[0].rows
                .iter()
                .zip(&r[0].times)
                .filter(move |(&i, _)| i == row)
                .map(|(_, &t)| t)
        })
        .collect()
}

/// K6, pipeline level. With a hazard of 1e-6 nobody has an event before `H = 8`, so a
/// replicate time is the censoring time. Under `DrawAll` a subject observed censored
/// at 2 is censored at 8 in some replicates and an event subject at 2 in some; under
/// `KeepObserved` the first is always 2 and the second never 2. `Horizon` censors no
/// one before `H`. The `DrawAll` draws come from the same `(seed, sim, subject)`
/// stream as `vpc_tte`'s.
#[test]
fn kmmc_replicates_draw_every_censoring_time_from_g() {
    let m = exp_model(1e-6);
    let p = two_censoring_pop();
    let o = opts(40);
    let (rows, _, _) = observed_rows(Entry::Kmmc, &p, 2, &VpcStratify::None).unwrap();
    let run_rule = |rule, censoring| {
        let mut v = o.vpc.clone();
        v.censoring = censoring;
        simulate_replicates(&m, &p, &m.default_params, &v, 2, &rows, 1, 8.0, rule)
            .unwrap()
            .0
    };
    let draw = run_rule(TteCensorRule::DrawAll, VpcCensoring::Observed);
    let censored = times_of(&draw, 0);
    let event = times_of(&draw, 6);
    assert_eq!(censored.len(), 40);
    assert!(
        censored.iter().all(|t| *t == 2.0 || *t == 8.0),
        "{censored:?}"
    );
    assert!(
        censored.contains(&8.0),
        "a censored subject is not kept at 2"
    );
    assert!(
        event.contains(&2.0),
        "an event subject draws below its event time"
    );

    let keep = run_rule(TteCensorRule::KeepObserved, VpcCensoring::Observed);
    assert!(times_of(&keep, 0).iter().all(|t| *t == 2.0));
    assert!(times_of(&keep, 6).iter().all(|t| *t == 8.0));

    let horizon = run_rule(TteCensorRule::DrawAll, VpcCensoring::Horizon);
    for row in 0..10 {
        assert!(times_of(&horizon, row).iter().all(|t| *t == 8.0));
    }

    // The driver itself must pick `DrawAll`. Under `KeepObserved` every replicate has
    // rows 3–9 at risk at t = 5 (the subjects censored at 2 keep 2, everyone else is
    // censored at 8), so every replicate mean there is (3 + … + 9)/7 = 6 and the band
    // has width 0. Under `DrawAll` who is at risk at 5 varies, and so does the mean.
    let mut ko = o.clone();
    ko.vpc.grid = VpcGrid::Times(vec![5.0]);
    ko.vpc.sim.horizon = Some(8.0);
    ko.min_at_risk = 1;
    let r = run(&m, &p, &ko).unwrap();
    let s = &r.strata[0];
    let (lo, hi) = (s.lo[0].unwrap(), s.hi[0].unwrap());
    assert!(
        hi > lo,
        "vpc_kmmc band at 5 is [{lo}, {hi}]: replicates kept their own censoring"
    );
}

// ── K7: the t = 0 identity ───────────────────────────────────────────────────

/// 30 subjects, events at 1..30, `Z = 3·i mod 7 + 0.1·i` (not sorted by time).
fn thirty() -> Population {
    pop((1..=30)
        .map(|i| {
            subj(
                &i.to_string(),
                i as f64,
                true,
                ((3 * i) % 7) as f64 + 0.1 * i as f64,
            )
        })
        .collect())
}

/// K7. At `t = 0` every subject is at risk in the data and in every replicate, so the
/// observed value and the whole band are the population mean (up to summation order:
/// realised worst 0 here; bound 4 ULPs at the mean). Later the band opens
/// (`hi − lo > 0` at t = 15), so the identity at 0 is not a degenerate band
/// everywhere. Reading the replicates' covariate from another source, or from the
/// curve's own time order, breaks the first assertion.
#[test]
fn band_collapses_to_the_population_mean_at_time_zero() {
    let p = thirty();
    let m = exp_model(0.05);
    let mut o = opts(40);
    o.vpc.grid = VpcGrid::Times(vec![0.0, 15.0]);
    let r = run(&m, &p, &o).unwrap();
    let mean: f64 = p.subjects.iter().map(|s| s.covariates["Z"]).sum::<f64>() / 30.0;
    let s = &r.strata[0];
    let tol = 4.0 * f64::EPSILON * mean;
    for (what, v) in [
        ("obs", s.obs_on_grid[0]),
        ("lo", s.lo[0]),
        ("med", s.med[0]),
        ("hi", s.hi[0]),
    ] {
        let v = v.unwrap_or_else(|| panic!("{what} at 0 is None"));
        assert!((v - mean).abs() <= tol, "{what}(0) = {v} vs mean {mean}");
    }
    assert_eq!(s.n_used[0], 40);
    let width = s.hi[1].unwrap() - s.lo[1].unwrap();
    assert!(width > 0.0, "band width at 15 is {width}");

    // Per stratum, with the strata interleaved in the population: each stratum's band at
    // 0 is that stratum's own mean. With one stratum a replicate lists its subjects in row
    // order, so reading a replicate subject's value by its position in the replicate
    // instead of by its row index passes above; here it gives stratum 1 stratum 0's
    // values.
    let mut p = thirty();
    for (i, s) in p.subjects.iter_mut().enumerate() {
        s.covariates.insert("W".into(), (i % 2) as f64);
    }
    o.vpc.stratify = VpcStratify::Covariate("W".into());
    let r = run(&m, &p, &o).unwrap();
    assert_eq!(r.strata.len(), 2);
    for (k, s) in r.strata.iter().enumerate() {
        let zs: Vec<f64> = p
            .subjects
            .iter()
            .filter(|x| x.covariates["W"] == k as f64)
            .map(|x| x.covariates["Z"])
            .collect();
        let mean = zs.iter().sum::<f64>() / zs.len() as f64;
        let tol = 4.0 * f64::EPSILON * mean;
        for (what, v) in [("lo", s.lo[0]), ("hi", s.hi[0])] {
            let v = v.unwrap_or_else(|| panic!("stratum {k} {what} at 0 is None"));
            assert!(
                (v - mean).abs() <= tol,
                "stratum {k} {what}(0) = {v} vs its mean {mean}"
            );
        }
    }
}

/// K7b. `level: Some(2)` gives the share of at-risk subjects at level 2, not the mean
/// of the codes: `Z` cycles 0, 1, 2 over 30 subjects with events at 1..30, so 10 of 30
/// are at 2 at t = 1, and at t = 29 the two subjects left are 29 (`Z` = 1) and 30
/// (`Z` = 2), a share of 1/2. Ignoring `level` gives 1 and 1.5.
#[test]
fn level_gives_the_share_at_that_level() {
    let p = pop((1..=30)
        .map(|i| subj(&i.to_string(), i as f64, true, (i % 3) as f64))
        .collect());
    let m = exp_model(0.05);
    let mut o = opts(20);
    o.level = Some(2.0);
    o.min_at_risk = 1;
    let r = run(&m, &p, &o).unwrap();
    let obs = &r.strata[0].observed;
    assert_eq!(obs.mean[0], 10.0 / 30.0);
    assert_eq!(obs.at(29.0, 1), Some(0.5));
    assert_eq!(r.level, Some(2.0));
}

/// K7c. The driver passes `min_at_risk` to the band and to the observed curve. At
/// t = 26 the data have 5 at risk and the replicates about 10, so with a floor of 10
/// `obs_on_grid` is `None` (it is `Some` with floor 1) and `n_used` counts only the
/// replicates with at least 10 at risk, recomputed here from the same replicates; the
/// straddle (fewer than with floor 1) is asserted so the check cannot pass vacuously.
/// Passing 1 to `vpc_band_kmmc` in the driver breaks the `n_used` assertion.
#[test]
fn driver_floors_the_band_and_the_observed_curve() {
    let p = thirty();
    let m = exp_model(0.042);
    let mut o = opts(60);
    o.vpc.grid = VpcGrid::Times(vec![0.0, 26.0]);
    let r = run(&m, &p, &o).unwrap();
    let s = &r.strata[0];
    assert_eq!(s.obs_on_grid[1], None);
    assert!(s.observed.at(26.0, 1).is_some());

    let (rows, _, _) = observed_rows(Entry::Kmmc, &p, 2, &VpcStratify::None).unwrap();
    let mut v = o.vpc.clone();
    v.sim.horizon = Some(r.horizon);
    let (reps, _) = simulate_replicates(
        &m,
        &p,
        &m.default_params,
        &o.vpc,
        2,
        &rows,
        1,
        r.horizon,
        TteCensorRule::DrawAll,
    )
    .unwrap();
    let z: Vec<f64> = rows
        .iter()
        .map(|row| {
            p.subjects
                .iter()
                .find(|s| s.id == row.id)
                .unwrap()
                .covariates["Z"]
        })
        .collect();
    let curves: Vec<KmmcCurve> = reps
        .iter()
        .map(|r| {
            let vals: Vec<f64> = r[0].rows.iter().map(|&i| z[i]).collect();
            kaplan_meier_mean_covariate(&r[0].times, &r[0].events, &vals).unwrap()
        })
        .collect();
    let floored = curves.iter().filter(|c| c.at(26.0, 10).is_some()).count();
    let unfloored = curves.iter().filter(|c| c.at(26.0, 1).is_some()).count();
    assert!(floored < unfloored, "straddle: {floored} vs {unfloored}");
    assert!(floored > 0, "some replicate keeps 10 at risk at 26");
    assert_eq!(s.n_used[1], floored);
    let _ = v;
}

// ── K8: messages ─────────────────────────────────────────────────────────────

fn assert_refusal<T: std::fmt::Debug>(got: Result<T, EngineError>, text: &str) {
    let e = got.expect_err("must refuse");
    assert_eq!(e.code(), Some("E_VPC_TTE_INPUT"), "{e}");
    assert_eq!(e.to_string(), text);
}

#[test]
fn refuses_repeated_events_and_names_vpc_rtte() {
    let m = parse_model_string(RTTE).unwrap();
    let p = thirty();
    let e = run(&m, &p, &opts(20)).unwrap_err().to_string();
    assert_eq!(
        e,
        "vpc_kmmc: cmt 2 is a repeated-events (RTTE) endpoint. A Kaplan–Meier mean \
         covariate by event order is not supported yet; `vpc_rtte` draws the Kaplan–Meier \
         and mean-cumulative-function VPC of a repeated-events endpoint."
    );
    assert!(!e.contains("Use `vpc_rtte` for its Kaplan–Meier mean covariate"));
}

/// `VpcKmmcOptions::default()` names no covariate; `vpc_kmmc` says so instead of
/// reporting that subject 1 lacks a covariate called ``.
#[test]
fn refuses_the_default_options_without_a_covariate() {
    let mut o = VpcKmmcOptions::default();
    o.vpc.n_sim = 20;
    let e = run(&exp_model(0.05), &thirty(), &o).unwrap_err();
    assert_eq!(e.code(), Some("E_VPC_TTE_INPUT"));
    assert_eq!(
        e.to_string(),
        "vpc_kmmc: `covariate` is empty. Name the covariate whose mean to take, e.g. \
         `VpcKmmcOptions::new(\"WT\")`."
    );
    assert_eq!(o.min_at_risk, 10);
    assert_eq!(o.level, None);
}

#[test]
fn refuses_a_subject_that_lacks_the_covariate() {
    let mut p = thirty();
    p.subjects[4].covariates.clear();
    assert_refusal(
        run(&exp_model(0.05), &p, &opts(20)),
        "vpc_kmmc: subject 5 has no covariate `Z` to take the mean of.",
    );
}

#[test]
fn refuses_a_missing_baseline_with_count_and_first_id() {
    let mut p = thirty();
    p.subjects[6].covariates.insert("Z".into(), f64::NAN);
    p.subjects[2].covariates.insert("Z".into(), f64::INFINITY);
    let e = run(&exp_model(0.05), &p, &opts(20))
        .unwrap_err()
        .to_string();
    assert_eq!(
        e,
        "vpc_kmmc: covariate `Z` is missing (every value blank or not finite) in 2 \
         subjects, first 3. The mean over a risk set needs a value for every subject in it."
    );
    assert!(!e.contains("lacks") && !e.contains("has no covariate"));
}

#[test]
fn refuses_a_level_no_subject_has() {
    let mut o = opts(20);
    o.level = Some(99.0);
    assert_refusal(
        run(&exp_model(0.05), &thirty(), &o),
        "vpc_kmmc: no subject on the endpoint has covariate `Z` = 99, the `level` asked for.",
    );
}

#[test]
fn refuses_stratifying_on_the_covariate_itself() {
    let mut o = opts(20);
    o.vpc.stratify = VpcStratify::Covariate("Z".into());
    assert_refusal(
        run(&exp_model(0.05), &thirty(), &o),
        "vpc_kmmc: stratifying on `Z`, the covariate whose mean is taken, makes the mean \
         constant within each stratum. Stratify on another covariate, or not at all.",
    );
}

#[test]
fn refuses_two_tte_endpoints_without_cmt() {
    let m = parse_model_string(COMPETING).unwrap();
    let e = run(&m, &thirty(), &opts(20)).unwrap_err().to_string();
    assert_eq!(
        e,
        "vpc_kmmc: the model has TTE endpoints on cmt 2, 3; set `cmt` to the one to check. \
         Under competing risks the risk set of one cause holds the subjects free of every \
         cause, so the other causes act as censorings."
    );
}

#[test]
fn shared_record_refusals_name_vpc_kmmc() {
    let m = exp_model(0.05);
    let mut p = thirty();
    p.subjects[0].obs_records = vec![rec(
        3.0,
        EventType::IntervalCensored {
            left: 1.0,
            right: 3.0,
        },
        2,
    )];
    let e = run(&m, &p, &opts(20)).unwrap_err().to_string();
    assert_eq!(
        e,
        "vpc_kmmc: subject 1 has an interval-censored record on cmt 2. A Kaplan–Meier mean \
         covariate needs exact or right-censored times."
    );
    assert!(!e.contains("vpc_tte:"));

    let mut p = thirty();
    p.subjects[0].obs_records = vec![ObsRecord::Event {
        time: 3.0,
        event_type: EventType::Exact,
        entry_time: 1.0,
        cmt: 2,
    }];
    assert_refusal(
        run(&m, &p, &opts(20)),
        "vpc_kmmc: subject 1 has entry time 1 on cmt 2. Left truncation (delayed entry) is \
         not supported yet.",
    );

    let mut p = thirty();
    p.subjects[0]
        .obs_records
        .push(rec(9.0, EventType::RightCensored, 2));
    let e = run(&m, &p, &opts(20)).unwrap_err().to_string();
    assert_eq!(
        e,
        "vpc_kmmc: subject 1 has 2 records on cmt 2. A single-event Kaplan–Meier curve \
         takes one per subject."
    );
    assert!(!e.contains("vpc_tte:"));
}

/// The time-varying warning names the covariate, counts the subjects whose value
/// changes, and says the first value is used. It is a warning: the curve is returned.
#[test]
fn warns_on_a_time_varying_covariate() {
    let mut p = thirty();
    for i in [3, 8] {
        let mut snap = std::collections::HashMap::new();
        snap.insert("Z".to_string(), 100.0);
        p.subjects[i].endpoint_covariates = vec![snap];
    }
    let r = run(&exp_model(0.05), &p, &opts(20)).unwrap();
    let w: Vec<&String> = r
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_VPC_KMMC_TIME_VARYING_COVARIATE"))
        .collect();
    assert_eq!(
        w,
        [
            "W_VPC_KMMC_TIME_VARYING_COVARIATE: covariate `Z` changes over time in 2 subjects. \
             The Kaplan–Meier mean covariate uses each subject's first value, not its value at \
             the risk time."
        ]
    );
    assert!(!w[0].contains("refused"));
    assert!(
        !run(&exp_model(0.05), &thirty(), &opts(20))
            .unwrap()
            .warnings
            .iter()
            .any(|w| w.starts_with("W_VPC_KMMC_TIME_VARYING_COVARIATE")),
        "no warning for a constant covariate"
    );
}
