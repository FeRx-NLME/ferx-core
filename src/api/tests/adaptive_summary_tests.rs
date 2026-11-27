//! #946 PR2: `summarize_adaptive`, the per-draw population summary, its bands across
//! parameter draws, and the pooled prediction interval.
//!
//! The adaptive exception (AGENTS.md, epic #391): no NONMEM run. The engine-level oracle is
//! conditional: on a 1-cpt model with Ω FIX 0, an `Ipred` monitor and a fixed controller,
//! every run of a draw is the same patient, so each per-draw statistic is a closed form of
//! that draw's θ, read from `result.parameters`. The band must then be the type-7 quantile
//! of those closed forms, computed here without the engine's quantile helper. The kernel's
//! counting rules (missing values, pooling, NaN) are pinned on hand-built rows.

use super::adaptive_sim_tests::{fixed_bolus, population, subj};
use super::*;
use crate::estimation::uncertainty_samples::{fitted_params_from_result, UncertaintyMethod};
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    AdaptiveSubjectMetrics, DecisionLogEntry, DecisionOutcome, MonitorSpec, ObserveMode,
    ObservedSignal,
};
use AdaptiveSummaryMetric as M;
use AdaptiveSummaryStatistic as S;

// ── hand-built rows ────────────────────────────────────────────────────────────────────

fn metrics_row(draw: usize, sim: usize, id: &str, dose: f64) -> AdaptiveSubjectMetrics {
    AdaptiveSubjectMetrics {
        subject: id.to_string(),
        draw,
        sim,
        rep: 1,
        cumulative_dose: dose,
        n_doses: 1,
        n_increases: 0,
        n_decreases: 0,
        n_holds: 0,
        n_not_due: 0,
        discontinued: false,
        time_to_discontinuation: None,
        signal_min: None,
        signal_max: None,
        signal_mean: None,
        pct_readings_in_window: None,
        auc_target_attainment: None,
        eta_supplied: false,
        dose_oracle: false,
    }
}

fn decision(draw: usize, sim: usize, id: &str, idx: usize, latent: f64) -> DecisionLogEntry {
    DecisionLogEntry {
        subject: id.to_string(),
        draw,
        sim,
        rep: 1,
        decision_idx: idx,
        time: 24.0 * idx as f64,
        observed_signals: vec![ObservedSignal {
            name: "A".into(),
            value: latent,
            latent,
            mode: ObserveMode::Ipred,
        }],
        outcome: DecisionOutcome::Hold,
        interval: None,
    }
}

fn result_of(
    metrics: Vec<AdaptiveSubjectMetrics>, decisions: Vec<DecisionLogEntry>,
) -> AdaptiveSimulationResult {
    AdaptiveSimulationResult {
        trajectories: vec![],
        ledger: vec![],
        decisions,
        metrics,
        parameters: vec![],
        warnings: vec![],
    }
}

fn summary(
    r: &AdaptiveSimulationResult, opts: &AdaptiveSummaryOptions,
) -> AdaptivePopulationSummary {
    summarize_adaptive(r, opts).expect("summarises")
}

fn row<'a>(
    s: &'a AdaptivePopulationSummary, metric: &M, stat: S, draw: usize,
) -> &'a AdaptiveSummaryRow {
    s.rows
        .iter()
        .find(|r| &r.metric == metric && r.statistic == stat && r.draw == draw)
        .unwrap_or_else(|| panic!("no row {metric:?} {stat:?} draw {draw}"))
}

fn band<'a>(s: &'a AdaptivePopulationSummary, metric: &M, stat: S) -> &'a AdaptiveSummaryBand {
    s.bands.iter().find(|b| &b.metric == metric && b.statistic == stat).expect("band")
}

fn pi<'a>(s: &'a AdaptivePopulationSummary, metric: &M) -> &'a AdaptivePredictionInterval {
    s.prediction_intervals.iter().find(|p| &p.metric == metric).expect("prediction interval")
}

fn band_values(b: &AdaptiveSummaryBand) -> [Option<f64>; 3] {
    match b.status {
        AdaptiveBandStatus::Quantiles(q) => q.values,
        AdaptiveBandStatus::NotDrawn => panic!("band not drawn: {b:?}"),
    }
}

/// Type 7, written out here rather than through `quantile_sorted`, so the oracle does not
/// share the engine's quantile code.
fn type7(vals: &[f64], p: f64) -> f64 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = (v.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    v[lo] + (h - lo as f64) * (v[(lo + 1).min(v.len() - 1)] - v[lo])
}

// ── kernel rules on hand-built rows ────────────────────────────────────────────────────

/// Each run weighs the same within a draw, and draws are summarised apart.
///
/// Mutations: pooling the draws before taking the statistic (draw 1's mean becomes 4.5);
/// weighting by subject rather than by run.
#[test]
fn a_draw_is_summarised_over_its_own_runs_each_weighted_equally() {
    let r = result_of(
        vec![
            metrics_row(1, 1, "1", 1.0),
            metrics_row(1, 1, "2", 2.0),
            metrics_row(1, 2, "1", 6.0),
            metrics_row(2, 1, "1", 10.0),
        ],
        vec![],
    );
    let s = summary(&r, &AdaptiveSummaryOptions::default());
    assert_eq!(s.n_draws, 2);
    let d1 = row(&s, &M::CumulativeDose, S::Mean, 1);
    assert_eq!((d1.value, d1.n_runs, d1.n_missing), (Some(3.0), 3, 0));
    assert_eq!(row(&s, &M::CumulativeDose, S::Median, 1).value, Some(2.0));
    assert_eq!(row(&s, &M::CumulativeDose, S::Mean, 2).value, Some(10.0));
}

/// A metric a run does not carry is skipped and counted, never imputed; one no run
/// carries has `value = None`. A threshold on such a metric gives rows, not an error.
///
/// Mutations: imputing `None` as 0 (the mean becomes 0.55 and the fraction 2/4); dropping
/// the `n_missing` count; refusing the threshold.
#[test]
fn a_missing_metric_is_skipped_and_counted_never_imputed() {
    let mut rows: Vec<_> = (1..=4).map(|i| metrics_row(1, 1, &i.to_string(), 0.0)).collect();
    rows[0].pct_readings_in_window = Some(0.5);
    rows[1].pct_readings_in_window = Some(0.8);
    rows[2].pct_readings_in_window = Some(0.9);
    let opts = AdaptiveSummaryOptions {
        thresholds: vec![(M::PctReadingsInWindow, 0.8), (M::AucTargetAttainment, 0.5)],
        ..Default::default()
    };
    let s = summary(&result_of(rows, vec![]), &opts);
    let mean = row(&s, &M::PctReadingsInWindow, S::Mean, 1);
    assert_eq!((mean.n_runs, mean.n_missing), (4, 1));
    assert!((mean.value.unwrap() - 2.2 / 3.0).abs() < 1e-15, "{mean:?}");
    let frac = row(&s, &M::PctReadingsInWindow, S::FractionAtLeast(0.8), 1);
    assert_eq!((frac.value, frac.n_missing), (Some(2.0 / 3.0), 1));
    let none = row(&s, &M::AucTargetAttainment, S::FractionAtLeast(0.5), 1);
    assert_eq!((none.value, none.n_runs, none.n_missing), (None, 4, 4));
    let p = pi(&s, &M::AucTargetAttainment);
    assert_eq!((p.quantiles.values, p.n_missing), ([None; 3], 4));
}

/// The per-decision signal: a run that stopped before decision 2 has no reading there and
/// is counted missing at that `decision_idx` only.
///
/// Mutations: imputing the absent reading as 0; keying by decision time instead of index
/// (two keys collide); dropping the analyte from the key.
#[test]
fn the_signal_at_a_decision_counts_runs_that_ended_before_it_as_missing() {
    let r = result_of(
        vec![metrics_row(1, 1, "1", 0.0), metrics_row(1, 1, "2", 0.0)],
        vec![decision(1, 1, "1", 0, 2.0), decision(1, 1, "1", 1, 4.0), decision(1, 1, "2", 0, 6.0)],
    );
    let s = summary(&r, &AdaptiveSummaryOptions::default());
    let at = |idx| M::Signal { analyte: "A".into(), decision_idx: idx };
    let d0 = row(&s, &at(0), S::Mean, 1);
    assert_eq!((d0.value, d0.n_missing), (Some(4.0), 0));
    let d1 = row(&s, &at(1), S::Mean, 1);
    assert_eq!((d1.value, d1.n_runs, d1.n_missing), (Some(4.0), 2, 1));
}

/// Supplied-η runs are summarised like any other (`eta_supplied` is not a filter), and
/// `Discontinued`'s mean is the fraction discontinued.
///
/// Mutation: leaving supplied subjects out of the summary.
#[test]
fn supplied_eta_runs_are_summarised_like_any_other() {
    let mut rows = vec![metrics_row(1, 1, "1", 0.0), metrics_row(1, 1, "2", 0.0)];
    rows[0].eta_supplied = true;
    rows[0].discontinued = true;
    let s = summary(&result_of(rows, vec![]), &AdaptiveSummaryOptions::default());
    let d = row(&s, &M::Discontinued, S::Mean, 1);
    assert_eq!((d.value, d.n_runs), (Some(0.5), 2));
}

/// A non-finite value propagates into every statistic, band and interval it enters; it is
/// never folded away. `≥` and a sort would each drop it silently.
///
/// Mutations: filtering non-finite values; computing the fraction without the guard (it
/// returns 0.5); sorting the `NaN` to the end and reading the lower quantiles.
#[test]
fn a_non_finite_value_propagates_and_is_never_folded_away() {
    let mut rows: Vec<_> = (1..=20).map(|d| metrics_row(d, 1, "1", d as f64)).collect::<Vec<_>>();
    rows.push(metrics_row(1, 2, "1", f64::NAN));
    let opts =
        AdaptiveSummaryOptions { thresholds: vec![(M::CumulativeDose, 0.5)], ..Default::default() };
    let s = summary(&result_of(rows, vec![]), &opts);
    for stat in [S::Mean, S::Median, S::FractionAtLeast(0.5)] {
        assert!(row(&s, &M::CumulativeDose, stat, 1).value.unwrap().is_nan(), "{stat:?}");
        assert!(band_values(band(&s, &M::CumulativeDose, stat))
            .iter()
            .all(|v| v.unwrap().is_nan()));
        // The other draws are finite: the NaN stays in draw 1's row.
        assert!(row(&s, &M::CumulativeDose, stat, 2).value.unwrap().is_finite());
    }
    assert!(pi(&s, &M::CumulativeDose).quantiles.values.iter().all(|v| v.unwrap().is_nan()));
}

/// The outer quantiles are read only from `⌈1/min(p, 1−p)⌉ − 1` values; the median from
/// one. At 18 draws the 5 % / 95 % ends are `None` and the median is not; at 19 all three.
///
/// Mutations: the whole-band gate of `vpc_band` (the median goes `None` at 18); no gate
/// (the ends are read from 18 values); dropping a probability level.
#[test]
fn each_quantile_needs_its_own_number_of_draws() {
    let at = |d: usize| {
        let rows = (1..=d).map(|k| metrics_row(k, 1, "1", k as f64)).collect();
        let s = summary(&result_of(rows, vec![]), &AdaptiveSummaryOptions::default());
        band_values(band(&s, &M::CumulativeDose, S::Mean))
    };
    assert_eq!(at(18), [None, Some(9.5), None]);
    let v: Vec<f64> = (1..=19).map(f64::from).collect();
    assert_eq!(at(19), [Some(type7(&v, 0.05)), Some(10.0), Some(type7(&v, 0.95))]);
}

/// One draw has no band: `NotDrawn`, never a zero-width band. Rows and a prediction
/// interval are still there.
///
/// Mutation: emitting `Quantiles` at D = 1 (all three equal).
#[test]
fn one_draw_has_no_band_but_keeps_rows_and_a_prediction_interval() {
    let rows = (1..=25).map(|i| metrics_row(1, 1, &i.to_string(), i as f64)).collect();
    let s = summary(&result_of(rows, vec![]), &AdaptiveSummaryOptions::default());
    assert_eq!(s.n_draws, 1);
    assert!(s.bands.iter().all(|b| b.status == AdaptiveBandStatus::NotDrawn));
    assert_eq!(row(&s, &M::CumulativeDose, S::Mean, 1).value, Some(13.0));
    let v: Vec<f64> = (1..=25).map(f64::from).collect();
    assert_eq!(
        pi(&s, &M::CumulativeDose).quantiles.values,
        [Some(type7(&v, 0.05)), Some(13.0), Some(type7(&v, 0.95))]
    );
}

/// Both refusals, whole text.
///
/// Mutations: either check deleted (the call succeeds); any sentence changed.
#[test]
fn bad_probabilities_and_thresholds_are_refused() {
    let r = result_of(vec![metrics_row(1, 1, "1", 1.0)], vec![]);
    let refuse = |opts: AdaptiveSummaryOptions| {
        let e = summarize_adaptive(&r, &opts).expect_err("refused");
        (e.diagnostic().map(|d| d.code.clone()).unwrap_or_default(), e.to_string())
    };
    let (code, msg) =
        refuse(AdaptiveSummaryOptions { probs: [0.5, 0.5, 0.95], ..Default::default() });
    assert_eq!(code, "E_ADAPTIVE_OPTIONS");
    assert!(
        msg.contains(
            "summarize_adaptive: probabilities [0.5, 0.5, 0.95] must be strictly inside (0, 1) \
             and strictly increasing"
        ),
        "{msg}"
    );
    let (code, msg) = refuse(AdaptiveSummaryOptions {
        thresholds: vec![(M::NDoses, f64::INFINITY)],
        ..Default::default()
    });
    assert_eq!(code, "E_ADAPTIVE_OPTIONS");
    assert!(
        msg.contains("summarize_adaptive: the threshold for N_DOSES is inf; it must be finite"),
        "{msg}"
    );
    let signal = M::Signal { analyte: "conc".into(), decision_idx: 3 };
    let (_, msg) = refuse(AdaptiveSummaryOptions {
        thresholds: vec![(signal, f64::NAN)],
        ..Default::default()
    });
    assert!(
        msg.contains(
            "summarize_adaptive: the threshold for SIGNAL (analyte 'conc', decision 3) is NaN; \
             it must be finite"
        ),
        "{msg}"
    );
}

/// The labels are ferx-r's column names and tokens, one per variant, unique per enum
/// (ferx-r#779 review finding 4). The `match`es without a wildcard stop compiling when a
/// variant is added; for the metric, that `match` is the expected-label table and the
/// variant list sits under it, and `SCALARS` is checked against that list, not built from
/// it, so a metric added to the enum but not to `SCALARS` fails here (#3213 r1 row 2).
///
/// Mutations: any label changed or duplicated; a metric dropped from `SCALARS`;
/// `from_label` not matching a scalar, or matching `"SIGNAL"`.
#[test]
fn every_metric_and_statistic_has_a_unique_stable_label() {
    let expected = |m: &M| match m {
        M::CumulativeDose => "CUM_DOSE",
        M::NDoses => "N_DOSES",
        M::NIncreases => "N_INCREASES",
        M::NDecreases => "N_DECREASES",
        M::NHolds => "N_HOLDS",
        M::NNotDue => "N_NOT_DUE",
        M::Discontinued => "DISCONTINUED",
        M::SignalMin => "SIGNAL_MIN",
        M::SignalMax => "SIGNAL_MAX",
        M::SignalMean => "SIGNAL_MEAN",
        M::PctReadingsInWindow => "PCT_READINGS_IN_WINDOW",
        M::AucTargetAttainment => "AUC_TARGET_ATTAINMENT",
        M::Signal { .. } => "SIGNAL",
    };
    // Every variant, one per arm above, scalars in output order.
    let scalars = [
        M::CumulativeDose,
        M::NDoses,
        M::NIncreases,
        M::NDecreases,
        M::NHolds,
        M::NNotDue,
        M::Discontinued,
        M::SignalMin,
        M::SignalMax,
        M::SignalMean,
        M::PctReadingsInWindow,
        M::AucTargetAttainment,
    ];
    let signal = M::Signal { analyte: "conc".into(), decision_idx: 0 };
    assert_eq!(M::SCALARS, scalars);
    let mut labels = std::collections::BTreeSet::new();
    for m in scalars.iter().chain(std::iter::once(&signal)) {
        assert_eq!(m.label(), expected(m), "{m:?}");
        assert!(labels.insert(m.label()), "{m:?}: label repeated");
    }
    for m in &scalars {
        assert_eq!(M::from_label(m.label()), Some(m.clone()), "{m:?}");
    }
    assert_eq!(M::from_label("SIGNAL"), None);
    assert_eq!(M::from_label("cum_dose"), None);

    let _exhaustive = |s: &S| match s {
        S::Mean | S::Median | S::FractionAtLeast(_) => (),
    };
    let stats = [S::Mean, S::Median, S::FractionAtLeast(0.8)].map(|s| s.label());
    assert_eq!(stats, ["mean", "median", "fraction_at_least"]);

    let _exhaustive = |b: &AdaptiveBandStatus| match b {
        AdaptiveBandStatus::NotDrawn | AdaptiveBandStatus::Quantiles(_) => (),
    };
    let q = AdaptiveQuantiles { probs: [0.05, 0.5, 0.95], values: [None; 3] };
    let bands = [AdaptiveBandStatus::NotDrawn, AdaptiveBandStatus::Quantiles(q)].map(|b| b.label());
    assert_eq!(bands, ["not_drawn", "quantiles"]);
}

/// Review r1 row 2: the signal rows read `latent`, not the assay-noised `value` the
/// controller saw (#946 PR1.5).
///
/// Mutation: the producer pushing `s.value`.
#[test]
fn the_signal_rows_read_the_latent_value_not_the_noised_reading() {
    let mut d = decision(1, 1, "1", 0, 4.0);
    d.observed_signals[0].value = 9.0;
    d.observed_signals[0].mode = ObserveMode::Dv;
    let s = summary(&result_of(vec![metrics_row(1, 1, "1", 0.0)], vec![d]), &Default::default());
    let at = M::Signal { analyte: "A".into(), decision_idx: 0 };
    assert_eq!(row(&s, &at, S::Mean, 1).value, Some(4.0));
}

/// Review r1 row 3: non-default probabilities reach both the band and the interval, and
/// are reported with them. At `[0.1, 0.5, 0.9]` the outer levels need 9 values.
///
/// Mutation: `opts.probs` replaced by the default.
#[test]
fn non_default_probabilities_reach_the_band_and_the_interval() {
    let probs = [0.1, 0.5, 0.9];
    let opts = AdaptiveSummaryOptions { probs, ..Default::default() };
    let rows = (1..=9).map(|d| metrics_row(d, 1, "1", d as f64)).collect();
    let s = summary(&result_of(rows, vec![]), &opts);
    let v: Vec<f64> = (1..=9).map(f64::from).collect();
    let want = probs.map(|p| Some(type7(&v, p)));
    match band(&s, &M::CumulativeDose, S::Mean).status {
        AdaptiveBandStatus::Quantiles(q) => assert_eq!((q.probs, q.values), (probs, want)),
        AdaptiveBandStatus::NotDrawn => panic!("nine draws"),
    }
    let q = pi(&s, &M::CumulativeDose).quantiles;
    assert_eq!((q.probs, q.values), (probs, want));
}

/// Review r1 rows 4 and 9: a draw in which no run carries the metric is counted in
/// `n_draws_missing`, and a band over a run with several draws is `Quantiles` even when
/// only one draw carries the metric: `NotDrawn` describes the run, not the metric.
///
/// Mutations: `n_draws_missing` fixed at 0; the status gated on draws present.
#[test]
fn a_draw_without_the_metric_is_counted_and_the_band_follows_the_run() {
    let mut rows = vec![metrics_row(1, 1, "1", 0.0), metrics_row(2, 1, "1", 0.0)];
    rows[0].signal_mean = Some(5.0);
    let s = summary(&result_of(rows, vec![]), &Default::default());
    let b = band(&s, &M::SignalMean, S::Mean);
    assert_eq!((b.n_draws_present, b.n_draws_missing), (1, 1));
    assert_eq!(band_values(b), [None, Some(5.0), None]);
    assert_eq!(row(&s, &M::SignalMean, S::Mean, 2).value, None);
}

/// Review r1 row 5: each metric reads its own field. One row with a distinct value in all
/// twelve, so a swapped arm shows. `Signal` is not a field of the metrics row.
///
/// Mutations: any two arms of `AdaptiveSummaryMetric::value` swapped or misread.
#[test]
fn each_metric_reads_its_own_field() {
    let mut m = metrics_row(1, 1, "1", 1.0);
    m.n_doses = 2;
    m.n_increases = 3;
    m.n_decreases = 4;
    m.n_holds = 5;
    m.n_not_due = 6;
    m.discontinued = true;
    m.signal_min = Some(8.0);
    m.signal_max = Some(9.0);
    m.signal_mean = Some(10.0);
    m.pct_readings_in_window = Some(0.11);
    m.auc_target_attainment = Some(0.12);
    let s = summary(&result_of(vec![m.clone()], vec![]), &Default::default());
    let want = [
        (M::CumulativeDose, 1.0),
        (M::NDoses, 2.0),
        (M::NIncreases, 3.0),
        (M::NDecreases, 4.0),
        (M::NHolds, 5.0),
        (M::NNotDue, 6.0),
        (M::Discontinued, 1.0),
        (M::SignalMin, 8.0),
        (M::SignalMax, 9.0),
        (M::SignalMean, 10.0),
        (M::PctReadingsInWindow, 0.11),
        (M::AucTargetAttainment, 0.12),
    ];
    for (metric, v) in want {
        assert_eq!(row(&s, &metric, S::Mean, 1).value, Some(v), "{metric:?}");
    }
    assert_eq!(M::Signal { analyte: "A".into(), decision_idx: 0 }.value(&m), None);
}

/// Review r1 rows 6 and 10: a threshold on a `Signal` no decision recorded (a misspelt
/// analyte, a `decision_idx` past the schedule) still gives rows, with `value = None`; and a
/// repeated threshold gives one set of rows, so a pivot on `(metric, statistic, draw)` keys
/// stays unique.
///
/// Mutations: the threshold metrics not seeded (no rows); the dedup removed (two rows).
#[test]
fn signal_thresholds_with_no_reading_give_rows_and_repeats_give_one() {
    let typo = M::Signal { analyte: "a".into(), decision_idx: 0 };
    let opts = AdaptiveSummaryOptions {
        thresholds: vec![(typo.clone(), 1.0), (M::NDoses, 1.0), (M::NDoses, 1.0)],
        ..Default::default()
    };
    let r = result_of(vec![metrics_row(1, 1, "1", 0.0)], vec![decision(1, 1, "1", 0, 4.0)]);
    let s = summary(&r, &opts);
    let t = row(&s, &typo, S::FractionAtLeast(1.0), 1);
    assert_eq!((t.value, t.n_runs, t.n_missing), (None, 1, 1));
    let n = s
        .rows
        .iter()
        .filter(|r| r.metric == M::NDoses && r.statistic == S::FractionAtLeast(1.0))
        .count();
    assert_eq!(n, 1);
    let n = s
        .bands
        .iter()
        .filter(|b| b.metric == M::NDoses && b.statistic == S::FractionAtLeast(1.0))
        .count();
    assert_eq!(n, 1);
}

// ── engine fixtures ────────────────────────────────────────────────────────────────────

/// A 1-cpt IV ODE; the cmt-1 monitor reads the central **amount**.
fn model(omega: &str) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  {omega}
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#
    ))
    .expect("parse")
}

const DECISIONS: [f64; 3] = [0.0, 24.0, 48.0];
const N_DRAWS: usize = 30;

/// The fit's packed covariance with variance on ln TVCL only (packed index 0); every
/// other coordinate gets the sampler's `1e-8` floor. The oracle reads both θ of each draw
/// back from `result.parameters`, so the floor's movement of V is accounted for.
fn cl_only_fit(model: &CompiledModel) -> FitResult {
    let mut fit = super::simulate_with_uncertainty_tests::synthetic_fit(&model.default_params);
    let cov = fit.covariance_matrix.as_mut().expect("synthetic fit has a matrix");
    cov.fill(0.0);
    cov[(0, 0)] = 0.04;
    fit
}

/// `n_sub` subjects × `n_sim` sims × `N_DRAWS` draws, 100 every 24 h, `Ipred` monitor.
fn run(model: &CompiledModel, n_sub: usize, n_sim: usize) -> AdaptiveSimulationResult {
    let fit = cl_only_fit(model);
    let params = fitted_params_from_result(&fit, model).expect("template");
    let pop = population(
        (1..=n_sub).map(|i| subj(&i.to_string(), vec![6.0, 30.0, 54.0], vec![])).collect(),
    );
    let opts = AdaptiveSimulateOptions {
        seed: Some(946),
        decision_times: DECISIONS.to_vec(),
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)],
        ..Default::default()
    };
    simulate_adaptive_with_inputs(
        model,
        &pop,
        &AdaptiveInputs::new(&params).with_uncertainty(
            &fit,
            UncertaintySpec { method: UncertaintyMethod::Asymptotic, n_draws: N_DRAWS },
        ),
        n_sim,
        fixed_bolus,
        &opts,
    )
    .expect("runs")
}

/// Review r1 row 1: two monitors with one name are refused by the run, whole text. Signals
/// are keyed by name, so the controller would read only the last, and the summary would
/// count each run twice (`n_runs − present` underflows). The straddle: two names run, and
/// each analyte is summarised once per run.
///
/// Mutation: the driver's check deleted (the run succeeds and the summary panics).
#[test]
fn two_monitors_with_one_name_are_refused_and_two_names_summarise_once() {
    let m = model("omega ETA_CL ~ 0.09");
    let pop = population(vec![subj("1", vec![6.0], vec![])]);
    let run = |names: [&str; 2]| {
        let opts = AdaptiveSimulateOptions {
            seed: Some(946),
            decision_times: DECISIONS.to_vec(),
            monitors: names.map(|n| MonitorSpec::new(n, 1, ObserveMode::Ipred)).to_vec(),
            ..Default::default()
        };
        simulate_adaptive(&m, &pop, &m.default_params, 1, fixed_bolus, &opts)
    };
    let e = run(["A", "A"]).expect_err("refused");
    assert_eq!(e.diagnostic().map(|d| d.code.as_str()), Some("E_ADAPTIVE_OPTIONS"));
    assert!(
        e.to_string().contains("two monitors are named 'A'; give each monitor its own name"),
        "{e}"
    );
    let s = summary(&run(["A", "B"]).expect("runs"), &Default::default());
    for analyte in ["A", "B"] {
        let at = M::Signal { analyte: analyte.into(), decision_idx: 1 };
        let r = row(&s, &at, S::Mean, 1);
        assert_eq!((r.n_runs, r.n_missing), (1, 0), "{analyte}");
    }
}

/// The pre-dose amounts at t = 0, 24, 48 of 100 every 24 h with `k = CL / V`.
fn readings(k: f64) -> [f64; 3] {
    let e = (-24.0 * k).exp();
    [0.0, 100.0 * e, 100.0 * (e * e + e)]
}

/// The conditional oracle (validation section of #946): Ω FIX 0, so every run of a draw is
/// the same patient and each per-draw mean is a closed form of θ_d. The band must equal the
/// type-7 quantiles of those closed forms across draws.
///
/// Realised gaps (macOS arm64), relative to the closed form: per-draw rows ≤ 5.8e-16,
/// band ≤ 3.4e-16. The tolerance, 1e-13, keeps about 170× headroom. The checks that use
/// `1e3 · tol` = 1e-10 have far more room: the closest two draws differ by 1.6e-3, and the
/// prediction interval's 5 % / 95 % ends sit 3.5e-2 to 1.6e-1 from the band's.
///
/// Mutations: quantiles across sims instead of across draws; pooling draws before the
/// statistic; swapping band and prediction interval; dropping a probability level.
#[test]
fn band_equals_type7_quantiles_of_the_per_draw_closed_forms() {
    let m = model("omega ETA_CL ~ 0.0 FIX");
    let r = run(&m, 2, 2);
    assert_eq!(r.parameters.len(), N_DRAWS);
    let s = summary(&r, &AdaptiveSummaryOptions::default());
    let tol = 1e-13;
    let close = |got: f64, want: f64| {
        assert!(got.is_finite() && want.is_finite(), "{got} vs {want}");
        (got - want).abs() <= tol * want.abs().max(1e-12)
    };
    let metrics = [
        (
            M::SignalMean,
            Box::new(|k| readings(k).iter().sum::<f64>() / 3.0) as Box<dyn Fn(f64) -> f64>,
        ),
        (M::Signal { analyte: "A".into(), decision_idx: 1 }, Box::new(|k| readings(k)[1])),
        (M::Signal { analyte: "A".into(), decision_idx: 2 }, Box::new(|k| readings(k)[2])),
    ];
    let mut spread = f64::INFINITY;
    for (metric, f) in &metrics {
        let mut per_draw = Vec::new();
        for dp in &r.parameters {
            let k = dp.parameters.theta[0] / dp.parameters.theta[1];
            let want = f(k);
            let got = row(&s, metric, S::Mean, dp.draw);
            assert_eq!((got.n_runs, got.n_missing), (4, 0), "{metric:?}");
            assert!(
                close(got.value.unwrap(), want),
                "{metric:?} draw {}: {got:?} vs {want}",
                dp.draw
            );
            per_draw.push(want);
        }
        let b = band(&s, metric, S::Mean);
        assert_eq!((b.n_draws_present, b.n_draws_missing), (N_DRAWS, 0));
        let got = band_values(b);
        for (k, p) in [0.05, 0.5, 0.95].into_iter().enumerate() {
            let want = type7(&per_draw, p);
            assert!(close(got[k].expect("30 draws read every level"), want), "{metric:?} p {p}");
        }
        // The interval pools the 4 identical runs of each draw. Its median is the band's (a
        // median ignores replication), but at 30 draws its 5 % / 95 % ends land on a draw
        // value where the band interpolates between two, so a swap shows.
        let pi_v = pi(&s, metric).quantiles.values;
        for k in [0, 2] {
            let (bk, pk) = (got[k].unwrap(), pi_v[k].unwrap());
            assert!(
                (bk - pk).abs() > 1e3 * tol * bk.abs(),
                "{metric:?}: band {bk} = interval {pk}"
            );
        }
        let mut sorted = per_draw.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in sorted.windows(2) {
            assert!(w[0].is_finite() && w[1].is_finite());
            spread = spread.min((w[1] - w[0]) / w[1]); // nan-ok: operands asserted finite
        }
    }
    assert!(spread > 1e3 * tol, "draws too close for the tolerance: {spread}");
}

/// The straddle pair: with Ω > 0 patients differ within a draw, so the pooled prediction
/// interval of a metric is wider than the band of its per-draw mean. Asserted on the
/// difference itself, with both computed independently from the returned rows. Measured
/// (macOS arm64, 20 subjects × 30 draws): interval width 19.48, band width 10.40, a ratio
/// of 1.87 against the asserted 1.5.
///
/// Mutations: swapping band and prediction interval; computing the interval from the
/// per-draw means (it collapses onto the band).
#[test]
fn band_and_prediction_interval_differ_when_patients_vary() {
    let m = model("omega ETA_CL ~ 0.09");
    let r = run(&m, 20, 1);
    let s = summary(&r, &AdaptiveSummaryOptions::default());
    let pooled: Vec<f64> = r.metrics.iter().map(|x| x.signal_mean.expect("Ipred signal")).collect();
    let per_draw: Vec<f64> = (1..=N_DRAWS)
        .map(|d| {
            let v: Vec<f64> =
                r.metrics.iter().filter(|x| x.draw == d).map(|x| x.signal_mean.unwrap()).collect();
            v.iter().sum::<f64>() / v.len() as f64
        })
        .collect();
    let b = band_values(band(&s, &M::SignalMean, S::Mean)).map(Option::unwrap);
    let p = pi(&s, &M::SignalMean).quantiles.values.map(Option::unwrap);
    for (k, q) in [0.05, 0.5, 0.95].into_iter().enumerate() {
        assert!((b[k] - type7(&per_draw, q)).abs() <= 1e-12 * b[k].abs(), "band p {q}");
        assert!((p[k] - type7(&pooled, q)).abs() <= 1e-12 * p[k].abs(), "interval p {q}");
    }
    let (band_w, pi_w) = (b[2] - b[0], p[2] - p[0]);
    assert!(pi_w > 1.5 * band_w, "interval width {pi_w} vs band width {band_w}");
}

/// Without uncertainty (D = 1) the band is `NotDrawn` and rows and intervals are present.
/// The result is only borrowed, so the type system already forbids writing back; the
/// `Debug` comparison below records that, it cannot fail on its own.
///
/// Mutation: a band at D = 1.
#[test]
fn without_uncertainty_there_is_no_band_and_the_run_is_unchanged() {
    let m = model("omega ETA_CL ~ 0.09");
    let pop = population((1..=3).map(|i| subj(&i.to_string(), vec![6.0, 30.0], vec![])).collect());
    let opts = AdaptiveSimulateOptions {
        seed: Some(946),
        decision_times: DECISIONS.to_vec(),
        monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)],
        ..Default::default()
    };
    let r = simulate_adaptive(&m, &pop, &m.default_params, 2, fixed_bolus, &opts).expect("runs");
    let before = format!("{r:?}");
    let s = summary(&r, &AdaptiveSummaryOptions::default());
    assert_eq!(format!("{r:?}"), before);
    assert_eq!(s.n_draws, 1);
    assert!(!s.bands.is_empty());
    assert!(s.bands.iter().all(|b| b.status == AdaptiveBandStatus::NotDrawn));
    assert_eq!(row(&s, &M::NDoses, S::Mean, 1).value, Some(3.0));
    assert_eq!(pi(&s, &M::NDoses).n_runs, 6);
    // Six runs: the median is read, the 5 % / 95 % ends need 19.
    let q = pi(&s, &M::SignalMean).quantiles.values;
    assert!(q[0].is_none() && q[1].is_some() && q[2].is_none(), "{q:?}");
}
