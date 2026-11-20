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

const COMPETING: &str = crate::test_models::MODEL_2;

fn rec(time: f64, event_type: EventType, cmt: usize) -> ObsRecord {
    ObsRecord::Event { time, event_type, entry_time: 0.0, cmt }
}

/// A subject with one record on cmt 2 and covariate `Z = z`.
fn subj(id: &str, time: f64, event: bool, z: f64) -> Subject {
    let et = if event { EventType::Exact } else { EventType::RightCensored };
    let mut s =
        Subject { id: id.into(), obs_records: vec![rec(time, et, 2)], ..Default::default() };
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
    o.vpc.sim = SimulateOptions { seed: Some(11), ..Default::default() };
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
    ObsRow { id: "1".into(), time, event, censor_at: (!event).then_some(time), stratum: 0 }
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
        assert_eq!(replicate_censoring(TteCensorRule::DrawAll, &censored, &g, u, h), 7.0);
        assert_eq!(
            replicate_censoring(TteCensorRule::DrawAll, &event, &g, u, h),
            7.0,
            "an event subject draws unconditionally, not given c > t"
        );
        assert_eq!(replicate_censoring(TteCensorRule::KeepObserved, &censored, &g, u, h), 3.0);
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
            r[0].rows.iter().zip(&r[0].times).filter(move |(&i, _)| i == row).map(|(_, &t)| t)
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
        simulate_replicates(&m, &p, &m.default_params, &v, 2, &rows, 1, 8.0, rule).unwrap().0
    };
    let draw = run_rule(TteCensorRule::DrawAll, VpcCensoring::Observed);
    let censored = times_of(&draw, 0);
    let event = times_of(&draw, 6);
    assert_eq!(censored.len(), 40);
    assert!(censored.iter().all(|t| *t == 2.0 || *t == 8.0), "{censored:?}");
    assert!(censored.contains(&8.0), "a censored subject is not kept at 2");
    assert!(event.contains(&2.0), "an event subject draws below its event time");

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
    assert!(hi > lo, "vpc_kmmc band at 5 is [{lo}, {hi}]: replicates kept their own censoring");
}

// ── K7: the t = 0 identity ───────────────────────────────────────────────────

/// 30 subjects, events at 1..30, `Z = 3·i mod 7 + 0.1·i` (not sorted by time).
fn thirty() -> Population {
    pop((1..=30)
        .map(|i| subj(&i.to_string(), i as f64, true, ((3 * i) % 7) as f64 + 0.1 * i as f64))
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
    for (what, v) in
        [("obs", s.obs_on_grid[0]), ("lo", s.lo[0]), ("med", s.med[0]), ("hi", s.hi[0])]
    {
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
            assert!((v - mean).abs() <= tol, "stratum {k} {what}(0) = {v} vs its mean {mean}");
        }
    }
}

/// K7b. `level: Some(2)` gives the share of at-risk subjects at level 2, not the mean
/// of the codes: `Z` cycles 0, 1, 2 over 30 subjects with events at 1..30, so 10 of 30
/// are at 2 at t = 1, and at t = 29 the two subjects left are 29 (`Z` = 2) and 30
/// (`Z` = 0), a share of 1/2. Ignoring `level` gives the mean of the codes, 1 at both
/// times, so both assertions fail.
#[test]
fn level_gives_the_share_at_that_level() {
    let p = pop((1..=30).map(|i| subj(&i.to_string(), i as f64, true, (i % 3) as f64)).collect());
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
        .map(|row| p.subjects.iter().find(|s| s.id == row.id).unwrap().covariates["Z"])
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
    let e = run(&exp_model(0.05), &p, &opts(20)).unwrap_err().to_string();
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
    p.subjects[0].obs_records =
        vec![rec(3.0, EventType::IntervalCensored { left: 1.0, right: 3.0 }, 2)];
    let e = run(&m, &p, &opts(20)).unwrap_err().to_string();
    assert_eq!(
        e,
        "vpc_kmmc: subject 1 has an interval-censored record on cmt 2. A Kaplan–Meier mean \
         covariate needs exact or right-censored times."
    );
    assert!(!e.contains("vpc_tte:"));

    let mut p = thirty();
    p.subjects[0].obs_records =
        vec![ObsRecord::Event { time: 3.0, event_type: EventType::Exact, entry_time: 1.0, cmt: 2 }];
    assert_refusal(
        run(&m, &p, &opts(20)),
        "vpc_kmmc: subject 1 has entry time 1 on cmt 2. Left truncation (delayed entry) is \
         not supported yet.",
    );

    let mut p = thirty();
    p.subjects[0].obs_records.push(rec(9.0, EventType::RightCensored, 2));
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
    let w: Vec<&String> =
        r.warnings.iter().filter(|w| w.starts_with("W_VPC_KMMC_TIME_VARYING_COVARIATE")).collect();
    assert_eq!(
        w,
        ["W_VPC_KMMC_TIME_VARYING_COVARIATE: covariate `Z` changes over time in 2 subjects. \
             The Kaplan–Meier mean covariate uses each subject's first value, not its value at \
             the risk time."]
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

// ── #2955: the observed data are read on [0, H] ──────────────────────────────

const H_CUT: f64 = 10.0;

/// Fixture C on cmt 2 with `H = 10`: events at 1, 2, …, 8 (three subjects each), a
/// censoring at 5, `X10` an event exactly at `H`, `E12` and `E15` events past `H`, and
/// `C18`, `C20` censorings past `H` (so the censoring curve `G` has mass past `H`).
/// `Z` is distinct per subject and not sorted by time; `W` alternates, for two strata.
fn fixture_c() -> Population {
    let mut s: Vec<Subject> = Vec::new();
    let mut add = |id: String, t: f64, e: bool| {
        let i = s.len();
        let mut x = subj(&id, t, e, ((7 * i) % 13) as f64 + 0.1 * i as f64);
        x.covariates.insert("W".into(), (i % 2) as f64);
        s.push(x);
    };
    for t in 1..=8 {
        for j in 0..3 {
            add(format!("S{t}{j}"), t as f64, true);
        }
    }
    add("C5".into(), 5.0, false);
    add("X10".into(), 10.0, true);
    add("E12".into(), 12.0, true);
    add("E15".into(), 15.0, true);
    add("C18".into(), 18.0, false);
    add("C20".into(), 20.0, false);
    let mut p = pop(s);
    p.covariate_names.push("W".into());
    p
}

/// The data cut at `h` by hand, independently of the engine: a record after `h`
/// becomes a right-censored record at `h`; one at or before `h` is unchanged.
fn cut_by_hand(p: &Population, h: f64) -> Population {
    let mut q = p.clone();
    for r in q.subjects.iter_mut().flat_map(|s| s.obs_records.iter_mut()) {
        if let ObsRecord::Event { time, event_type, .. } = r {
            if *time > h {
                *time = h;
                *event_type = EventType::RightCensored;
            }
        }
    }
    q
}

fn record_times(p: &Population) -> Vec<f64> {
    p.subjects
        .iter()
        .flat_map(|s| s.obs_records.iter())
        .map(|r| match r {
            ObsRecord::Event { time, .. } => *time,
            _ => unreachable!("fixture C holds event records only"),
        })
        .collect()
}

/// #2955. `sim.horizon = H` below the data's follow-up equals calling on the data cut
/// at `H` first: the whole `KmmcVpc`, under both censorings, unstratified and in two
/// strata, on the automatic grid and on a `Times` grid with points past `H` where the
/// uncut observed curve is still defined. The straddle is asserted first: C has records
/// past `H`, the hand cut has none. The hand cut also differs from the engine's in `G`
/// at `H`: it makes `E12` and `E15` censorings there, the engine leaves their
/// `censor_at` empty, so equality also pins that `G` at `H` moves no draw.
///
/// Mutations that redden it (measured): no cut (`observed` runs to 20, `obs_on_grid`
/// at 15 is defined); removing the call from `vpc_kmmc`; a cut that keeps the event flag
/// (`E12`, `E15` become events at `H`); a cut to `H − 1`. Not `>=` for `>`: it also
/// makes `X10` a censoring in the hand cut, so both sides agree;
/// `kmmc_observed_curve_reads_the_data_cut_at_the_horizon` kills it.
#[test]
fn kmmc_horizon_below_follow_up_equals_the_data_cut_at_the_horizon() {
    let p = fixture_c();
    let cut = cut_by_hand(&p, H_CUT);
    assert!(record_times(&p).iter().any(|&t| t > H_CUT), "C has records past H");
    assert!(record_times(&cut).iter().all(|&t| t <= H_CUT), "the cut has none");

    let m = exp_model(0.08);
    for stratify in [VpcStratify::None, VpcStratify::Covariate("W".into())] {
        for censoring in [VpcCensoring::Observed, VpcCensoring::Horizon] {
            for grid in [VpcGrid::Auto, VpcGrid::Times(vec![3.0, 9.0, 10.0, 15.0])] {
                let mut o = opts(20);
                o.min_at_risk = 2;
                o.vpc.stratify = stratify.clone();
                o.vpc.censoring = censoring;
                o.vpc.grid = grid.clone();
                o.vpc.sim.horizon = Some(H_CUT);
                let got = run(&m, &p, &o).unwrap();
                let want = run(&m, &cut, &o).unwrap();
                assert_eq!(got, want, "{stratify:?}, {censoring:?}, {grid:?}");
            }
        }
    }
}

/// #2955, the readable twin. The observed curve of C at `H = 10` ends at `H`: there
/// `X10`, `E12`, `E15`, `C18` and `C20` are at risk, `X10`'s event is an event and the
/// other four are censorings, and the mean is theirs. At the grid point 15 the observed
/// value is empty; uncut, the curve ran to 20 and was defined there.
///
/// Mutations that redden it (measured): no cut (last row 20), in the helper or by
/// removing the call; `>=` for `>` (`X10`'s event at `H` becomes a censoring,
/// `n_event` 0); keeping the event flag (`n_event` 3); a cut to `H − 1`.
#[test]
fn kmmc_observed_curve_reads_the_data_cut_at_the_horizon() {
    let p = fixture_c();
    let mut o = opts(20);
    o.min_at_risk = 2;
    o.vpc.sim.horizon = Some(H_CUT);
    o.vpc.grid = VpcGrid::Times(vec![9.0, 15.0]);
    let v = run(&exp_model(0.08), &p, &o).unwrap();
    let s = &v.strata[0];
    assert_eq!(s.n_subjects, 30);
    let c = &s.observed;
    assert_eq!(c.time.last(), Some(&H_CUT), "the curve ends at H");
    let k = c.time.len() - 1;
    assert_eq!(
        (c.n_risk[k], c.n_event[k], c.n_censor[k]),
        (5, 1, 4),
        "at H: X10's event, and E12, E15, C18, C20 censored there"
    );
    let at_risk = ["X10", "E12", "E15", "C18", "C20"];
    let want: f64 = p
        .subjects
        .iter()
        .filter(|x| at_risk.contains(&x.id.as_str()))
        .map(|x| x.covariates["Z"])
        .sum::<f64>()
        / 5.0;
    assert!(
        (c.mean[k] - want).abs() <= 4.0 * f64::EPSILON * want,
        "mean at H {} vs {want}",
        c.mean[k]
    );
    assert!(s.obs_on_grid[0].is_some(), "9 is before H");
    assert_eq!(s.obs_on_grid[1], None, "15 is past H");
}

/// #2955, the reason the cut leaves the band alone, at the replicate level: under
/// `DrawAll` the replicates of C are the same, time for time, with the rows cut at `H`
/// or not. `G` below `H` reads only risk sets before `H`, which the cut does not touch,
/// so every draw under `H` is unchanged, and a draw at or past `H` censors no replicate
/// time (all `≤ H`). The straddle: uncut, two rows are censorings past `H`, so `G` has
/// mass there that the cut moves onto `H`.
///
/// Mutation that reddens its equality (measured): a cut to a time below `H`
/// (`r.time = horizon - 1.0`), which moves `G`'s mass under `H`. No cut at all reddens
/// only its straddle assertion, since both sides are then the uncut rows.
#[test]
fn kmmc_replicates_are_the_same_with_the_rows_cut_at_the_horizon_or_not() {
    let p = fixture_c();
    let m = exp_model(0.08);
    let (rows, _, _) = observed_rows(Entry::Kmmc, &p, 2, &VpcStratify::None).unwrap();
    let (mut cut, _, _) = observed_rows(Entry::Kmmc, &p, 2, &VpcStratify::None).unwrap();
    assert_eq!(horizon_and_cut(&mut cut, Some(H_CUT)), H_CUT);
    assert_eq!(
        rows.iter().filter(|r| r.time > H_CUT && r.censor_at.is_some()).count(),
        2,
        "C18, C20 put G's mass past H"
    );
    assert!(cut.iter().all(|r| r.time <= H_CUT));

    let mut v = opts(40).vpc;
    v.censoring = VpcCensoring::Observed;
    let sim = |rows: &[ObsRow]| {
        simulate_replicates(
            &m,
            &p,
            &m.default_params,
            &v,
            2,
            rows,
            1,
            H_CUT,
            TteCensorRule::DrawAll,
        )
        .unwrap()
        .0
    };
    let (a, b) = (sim(&rows), sim(&cut));
    let mut censored_below_h = 0;
    for (ra, rb) in a.iter().zip(&b) {
        let (sa, sb) = (&ra[0], &rb[0]);
        assert_eq!(sa.rows, sb.rows);
        assert_eq!(sa.events, sb.events);
        let bits = |t: &[f64]| t.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&sa.times), bits(&sb.times));
        censored_below_h +=
            sa.times.iter().zip(&sa.events).filter(|(t, e)| !**e && **t < H_CUT).count();
    }
    // Not a fixture where nobody is censored before H: the draws under H are live.
    assert!(censored_below_h > 0, "no replicate censored before H");
}

/// #2955, the helper: a record after `H` becomes a censoring at `H`, an event exactly
/// at `H` stays an event, a record before `H` is untouched, and `censor_at` is left as
/// it was. `>=` for `>` reddens the `X10` assertion; keeping the event flag reddens
/// `E12`'s; no cut reddens every row past `H`.
#[test]
fn horizon_and_cut_censors_past_h_and_keeps_an_event_at_h() {
    let p = fixture_c();
    let (mut rows, _, _) = observed_rows(Entry::Kmmc, &p, 2, &VpcStratify::None).unwrap();
    assert_eq!(horizon_and_cut(&mut rows, Some(H_CUT)), H_CUT);
    let row = |id: &str| {
        let r = rows.iter().find(|r| r.id == id).unwrap();
        (r.time, r.event, r.censor_at)
    };
    assert_eq!(row("S80"), (8.0, true, None));
    assert_eq!(row("C5"), (5.0, false, Some(5.0)));
    assert_eq!(row("X10"), (10.0, true, None));
    assert_eq!(row("E12"), (10.0, false, None));
    assert_eq!(row("E15"), (10.0, false, None));
    assert_eq!(row("C18"), (10.0, false, Some(18.0)));
}

/// #2955, the helper with `sim.horizon` unset: `H` is the latest record time (20) and no
/// row changes, the event `X20` at that time included. A fold that takes the earliest
/// time instead (`f64::min`) gives `H = 1` and cuts every later row; ignoring
/// `sim_horizon` reddens the `Some` assertion in the test above.
#[test]
fn horizon_and_cut_without_a_horizon_is_the_latest_record_and_cuts_nothing() {
    let mut p = fixture_c();
    let mut x = subj("X20", 20.0, true, 1.5);
    x.covariates.insert("W".into(), 0.0);
    p.subjects.push(x);
    let (rows, _, _) = observed_rows(Entry::Kmmc, &p, 2, &VpcStratify::None).unwrap();
    let (mut got, _, _) = observed_rows(Entry::Kmmc, &p, 2, &VpcStratify::None).unwrap();
    assert_eq!(horizon_and_cut(&mut got, None), 20.0);
    let key = |r: &ObsRow| (r.id.clone(), r.time.to_bits(), r.event, r.censor_at);
    assert_eq!(got.iter().map(key).collect::<Vec<_>>(), rows.iter().map(key).collect::<Vec<_>>());
    assert!(got.iter().any(|r| r.id == "X20" && r.event), "the event at H is kept");
}
