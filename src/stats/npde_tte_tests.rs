//! Tier-1 tests of the single-event TTE npde (#1889). Simulation has no dual or FD
//! path: everything here runs the `f64` value path.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::Subject;

/// Exponential hazard, no random effect: `F(t) = 1 − exp(−0.1 t)` for every subject.
const EXP_FIXED: &str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";

/// Exponential hazard with IIV: `λᵢ = 0.1·exp(ηᵢ)`, `ω² = 0.25`.
const EXP_IIV: &str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.25

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

fn rec(time: f64, event_type: EventType, entry_time: f64, cmt: usize) -> ObsRecord {
    ObsRecord::Event {
        time,
        event_type,
        entry_time,
        cmt,
    }
}

fn event(time: f64) -> Vec<ObsRecord> {
    vec![rec(time, EventType::Exact, 0.0, 2)]
}

fn censor(time: f64) -> Vec<ObsRecord> {
    vec![rec(time, EventType::RightCensored, 0.0, 2)]
}

fn pop(records: Vec<Vec<ObsRecord>>) -> Population {
    Population {
        subjects: records
            .into_iter()
            .enumerate()
            .map(|(i, obs_records)| Subject {
                id: (i + 1).to_string(),
                obs_records,
                ..Default::default()
            })
            .collect(),
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// The time-to-event warnings among `warnings`. The simulator drains the process-wide
/// steady-state warning sink (`dosing.rs`, #1289), so what `compute_tte_npde` returns
/// in a test can hold warnings written by unrelated tests running at the same time.
/// An assertion on the whole list fails beside the rest of the lib tests (review r2
/// #12); assert on this instead.
fn tte_warnings(warnings: Vec<String>) -> Vec<String> {
    warnings
        .into_iter()
        .filter(|w| w.starts_with("W_TTE_"))
        .collect()
}

fn scores(m: &CompiledModel, p: &Population, nsim: usize, seed: u64) -> Vec<Vec<TteNpde>> {
    compute_tte_npde(m, p, &m.default_params, nsim, Some(seed), None)
        .expect("scored")
        .0
}

fn refusal(m: &CompiledModel, p: &Population, nsim: usize) -> String {
    let err = compute_tte_npde(m, p, &m.default_params, nsim, Some(1), None).expect_err("refused");
    assert_eq!(
        err.diagnostic().map(|d| d.code.as_str()),
        Some("E_NPDE_TTE_INPUT"),
        "{err}"
    );
    err.to_string()
}

/// T1. The kernel on hand values.
///
/// Dies on: a censored row returning `u` (0.25, not 0.85); the clamp dropped (0, 1 and
/// the censored `n = K` row); `u` read on an event row.
#[test]
fn tte_pd_hand_values() {
    let k = 200;
    let (lo, hi) = (1.0 / 400.0, 1.0 - 1.0 / 400.0);
    assert_eq!(tte_pd(50, k, false, 0.9), 0.25);
    assert_eq!(tte_pd(0, k, false, 0.9), lo);
    assert_eq!(tte_pd(k, k, false, 0.9), hi);
    // Censored at F = 0.8 with u = 0.25: 0.8 + 0.25·0.2.
    assert!((tte_pd(160, k, true, 0.25) - 0.85).abs() < 1e-15);
    assert_eq!(tte_pd(k, k, true, 0.25), hi);
    // Censored at F = 0 with u = 0: the lower clamp, not 0.
    assert_eq!(tte_pd(0, k, true, 0.0), lo);
}

/// The counting rule on hand values: only an observed event strictly before `y`
/// counts. A replicate with no event never counts, whatever time it carries; this is
/// the only place that half can be seen, since the simulator reports such a replicate
/// at the horizon, which no scored time exceeds.
///
/// Dies on: `observed` dropped (the censored-below-`y` row); `<` made `<=` (the tie).
#[test]
fn event_before_hand_values() {
    assert!(event_before(2.0, true, 3.0));
    assert!(!event_before(3.0, true, 3.0));
    assert!(!event_before(4.0, true, 3.0));
    assert!(!event_before(2.0, false, 3.0));
    assert!(!event_before(f64::NAN, true, 3.0));
}

/// T2, the named test for the conditioning. Twenty subjects, all right-censored at the
/// `c` where `F(c) = 0.95`, no random effect. A censored score lies on `[F̂(c), 1]`, so
/// every one is at least 0.95 less four Monte-Carlo SDs of `F̂`.
///
/// Dies on: imputing from the unconditional distribution (`pd = u`), where each row
/// passes with probability about 0.08 and all twenty with about 1e-22; `imputed` not
/// set on a censored row.
#[test]
fn censored_scores_lie_above_the_censoring_quantile() {
    let m = model(EXP_FIXED);
    let c = -(0.05_f64).ln() / 0.1;
    let p = pop((0..20).map(|_| censor(c)).collect());
    let k = 1000;
    let floor = 0.95 - 4.0 * (0.95 * 0.05 / k as f64).sqrt();
    let out = scores(&m, &p, k, 11);
    assert_eq!(out.len(), 20);
    let mut lowest = f64::INFINITY;
    for s in &out {
        let [row] = s.as_slice() else {
            panic!("one score per subject, got {}", s.len())
        };
        assert!(row.imputed);
        assert!(row.pd.is_finite() && row.npde.is_finite());
        assert!(row.pd >= floor, "pd {} below {floor}", row.pd);
        assert!(row.pd <= 1.0 - 1.0 / (2.0 * k as f64));
        if row.pd < lowest {
            lowest = row.pd;
        }
    }
    eprintln!("T2: lowest censored pd {lowest:.4}, floor {floor:.4}");
    // The scores are not one repeated value: each subject has its own uniform.
    assert!(out.iter().any(|s| s[0].pd != out[0][0].pd));
}

/// T3. Event rows against closed forms: `1 − exp(−λt)` with no random effect, and the
/// quadrature values `F(3) = 0.2775050126`, `F(8) = 0.5567771936` for λ = 0.1,
/// ω² = 0.25 (R `integrate`, computed outside the engine).
///
/// The no-IIV fixture's last subject sits at the horizon (the latest time): every
/// replicate with no event is reported exactly there, so counting `time <= y` or
/// counting censored replicates sends its `pd` to the upper clamp.
///
/// Realised at K = 20000, seed 3 (macOS arm64, debug): printed below; the bound is
/// 4 Monte-Carlo SDs, `4·√(F(1−F)/K)`.
#[test]
fn event_scores_match_closed_forms() {
    let k = 20_000;
    let bound = |f: f64| 4.0 * (f * (1.0 - f) / k as f64).sqrt();

    let m = model(EXP_FIXED);
    let times = [1.0, 5.0, 12.0, 30.0];
    let p = pop(times.iter().map(|&t| event(t)).collect());
    let out = scores(&m, &p, k, 3);
    for (s, &t) in out.iter().zip(&times) {
        let want = 1.0 - (-0.1 * t).exp();
        let row = &s[0];
        assert!(!row.imputed);
        assert_eq!(row.time, t);
        assert_eq!(row.cmt, 2);
        assert!(row.pd.is_finite());
        let err = (row.pd - want).abs();
        eprintln!(
            "T3 fixed: t = {t}: pd {:.5}, closed form {want:.5}, error {err:.2e}, bound {:.2e}",
            row.pd,
            bound(want)
        );
        assert!(err < bound(want));
        assert_eq!(row.npde, normal_inv_cdf(row.pd));
    }
    // Earlier than predicted is negative, later is positive.
    assert!(out[0][0].npde < 0.0 && out[3][0].npde > 0.0);

    let m = model(EXP_IIV);
    let p = pop(vec![event(3.0), event(8.0)]);
    let out = scores(&m, &p, k, 3);
    for (s, want) in out.iter().zip([0.2775050126, 0.5567771936]) {
        let pd = s[0].pd;
        assert!(pd.is_finite());
        let err = (pd - want).abs();
        eprintln!(
            "T3 iiv: pd {pd:.5}, quadrature {want:.5}, error {err:.2e}, bound {:.2e}",
            bound(want)
        );
        assert!(err < bound(want));
    }
}

/// T7. Every refusal's whole text. A deleted sentence, a swapped message or a deleted
/// refusal each fail an `assert_eq!` here.
#[test]
fn refusals_say_exactly_this() {
    let one = || pop(vec![event(3.0), censor(5.0)]);
    assert_eq!(
        refusal(&model(EXP_FIXED), &one(), 0),
        "compute_tte_npde: `nsim` must be at least 1."
    );
    assert_eq!(
        refusal(&model(PK_ONLY), &one(), 100),
        "compute_tte_npde: the model declares no TTE endpoint ([event_model])."
    );
    assert_eq!(
        refusal(&model(COMPETING), &one(), 100),
        "compute_tte_npde: the model has TTE endpoints on cmt 2, 3 (competing risks). npde \
         under competing risks is not supported yet."
    );
    assert_eq!(
        refusal(&model(RTTE), &one(), 100),
        "compute_tte_npde: cmt 2 is a repeated events (RTTE) endpoint. npde for repeated \
         events is not supported yet."
    );
    let m = model(EXP_FIXED);
    let with = |records: Vec<ObsRecord>| pop(vec![event(3.0), records]);
    assert_eq!(
        refusal(&m, &with(vec![rec(4.0, EventType::Exact, 1.5, 2)]), 100),
        "compute_tte_npde: subject 2 has entry time 1.5 on cmt 2: left truncation (delayed \
         entry) is not supported yet."
    );
    assert_eq!(
        refusal(
            &m,
            &with(vec![rec(
                4.0,
                EventType::IntervalCensored {
                    left: 2.0,
                    right: 4.0,
                },
                0.0,
                2
            )]),
            100
        ),
        "compute_tte_npde: subject 2 has an interval-censored record on cmt 2; npde needs an \
         exact or a right-censored time."
    );
    assert_eq!(
        refusal(
            &m,
            &with(vec![
                rec(4.0, EventType::RightCensored, 0.0, 2),
                rec(6.0, EventType::Exact, 0.0, 2),
                rec(7.0, EventType::Exact, 0.0, 2)
            ]),
            100
        ),
        "compute_tte_npde: subject 2 has 3 records on cmt 2. A single-event endpoint takes \
         one per subject."
    );
    assert_eq!(
        refusal(
            &m,
            &with(vec![rec(f64::NAN, EventType::Exact, 0.0, 2)]),
            100
        ),
        "compute_tte_npde: subject 2 has time NaN on cmt 2, which is not a finite, \
         non-negative number."
    );
    assert_eq!(
        refusal(&m, &pop(vec![vec![], vec![]]), 100),
        "compute_tte_npde: no subject has a record on cmt 2."
    );
    let mut dup = one();
    dup.subjects[1].id = "1".into();
    assert_eq!(
        refusal(&m, &dup, 100),
        "compute_tte_npde: subject id 1 appears more than once in the population. Simulated \
         draws are matched to subjects by id."
    );
}

/// T8. Reproducible for a seed, different across seeds, and the batching is exact:
/// `nsim = 250` (two full batches and a partial one) draws exactly 250 replicates per
/// scored subject and none for a subject with no record, and the batches are not one
/// batch repeated.
///
/// Dies on: the last partial batch dropped (200 drawn); one seed reused for every
/// batch (the 200-replicate counts are twice the 100-replicate ones for every
/// subject); a subject with no record given a score.
#[test]
fn seeded_batches_are_reproducible_and_exact() {
    let m = model(EXP_IIV);
    let mut records: Vec<Vec<ObsRecord>> = (0..20)
        .map(|i| {
            if i % 2 == 0 {
                event(2.0 + i as f64)
            } else {
                censor(2.0 + i as f64)
            }
        })
        .collect();
    records.insert(3, vec![]);
    let p = pop(records);

    let a = scores(&m, &p, 250, 5);
    assert_eq!(a, scores(&m, &p, 250, 5));
    assert_ne!(a, scores(&m, &p, 250, 6));
    assert!(a[3].is_empty());
    assert_eq!(a.iter().map(Vec::len).sum::<usize>(), 20);

    let scored = scored_records(&p, 2).unwrap();
    let counts =
        |nsim| reference_counts(&m, &p, &m.default_params, nsim, 5, None, 2, &scored).unwrap();
    let (_, drawn, _) = counts(250);
    for (i, d) in drawn.iter().enumerate() {
        assert_eq!(*d, if i == 3 { 0 } else { 250 }, "subject index {i}");
    }
    let (first, ..) = counts(BATCH);
    let (two, ..) = counts(2 * BATCH);
    assert!(
        first.iter().zip(&two).any(|(a, b)| *b != 2 * a),
        "the second batch repeats the first: {first:?} / {two:?}"
    );
    assert!(first.iter().zip(&two).all(|(a, b)| b >= a));
}

/// The imputation uniform belongs to the subject: changing another subject's record
/// leaves a censored subject's `u` alone, and two subjects do not share one.
///
/// Dies on: one uniform for the whole population; the uniform drawn from a stream
/// that advances per scored subject.
#[test]
fn imputation_uniform_is_per_subject() {
    assert_eq!(impute_uniform(9, 4), impute_uniform(9, 4));
    assert_ne!(impute_uniform(9, 4), impute_uniform(9, 5));
    assert_ne!(impute_uniform(9, 4), impute_uniform(10, 4));
    let u = impute_uniform(9, 4);
    assert!((0.0..1.0).contains(&u));
}

/// Kolmogorov–Smirnov distance of `p` from U(0, 1).
fn ks_uniform(mut p: Vec<f64>) -> f64 {
    assert!(p.iter().all(|v| v.is_finite()), "a non-finite score");
    p.sort_by(f64::total_cmp);
    let n = p.len() as f64;
    let mut d = 0.0_f64;
    for (i, &v) in p.iter().enumerate() {
        for gap in [((i as f64 + 1.0) / n - v).abs(), (v - i as f64 / n).abs()] {
            if gap > d {
                d = gap;
            }
        }
    }
    d
}

/// The T4 design: N subjects with `λᵢ = 0.1·exp(ηᵢ)`, `ηᵢ ~ N(0, 0.25)`, an event time
/// `Tᵢ ~ Exp(λᵢ)` and an independent dropout time `Cᵢ ~ Exp(0.25)`, drawn here with
/// the test's own RNG and closed forms, not with the simulator under test.
fn dropout_design(n: usize, seed: u64) -> (Population, usize) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut censored = 0;
    let records = (0..n)
        .map(|_| {
            let (u1, u2): (f64, f64) = (rng.random(), rng.random());
            let z = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let lambda = 0.1 * (0.5 * z).exp();
            let t = -(1.0 - rng.random::<f64>()).ln() / lambda;
            let c = -(1.0 - rng.random::<f64>()).ln() / 0.25;
            if t <= c {
                event(t)
            } else {
                censored += 1;
                censor(c)
            }
        })
        .collect();
    (pop(records), censored)
}

/// T4. Under the model that generated the data the scores are U(0, 1), about 70 %
/// censoring included; under a hazard twice too large they are not. One test, both
/// twins, one bound between them.
///
/// The bound 0.0616 is the 0.1 % KS critical value at N = 1000. Measured outside the
/// engine (R, 200 datasets of this design): correct model at most 0.0531, hazard ×2 at
/// least 0.1017, unconditional imputation at least 0.1019. Realised here: printed.
///
/// Dies on: unconditional imputation (the correct twin moves past the bound); the
/// twins swapped (both halves fail); a censored time scored as an event.
#[test]
fn scores_are_uniform_under_the_true_model_and_not_under_a_wrong_one() {
    const BOUND: f64 = 0.0616;
    let n = 1000;
    let m = model(EXP_IIV);
    let (p, n_censored) = dropout_design(n, 1889);
    let frac = n_censored as f64 / n as f64;
    assert!((0.65..0.76).contains(&frac), "censored fraction {frac}");

    let pds = |params: &ModelParameters| -> Vec<f64> {
        let (out, warnings) = compute_tte_npde(&m, &p, params, 1000, Some(7), None).unwrap();
        let warnings = tte_warnings(warnings);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            out.iter().filter(|s| s[0].imputed).count(),
            n_censored,
            "one imputed score per censored record"
        );
        out.iter().map(|s| s[0].pd).collect()
    };
    let right = ks_uniform(pds(&m.default_params));
    let mut doubled = m.default_params.clone();
    doubled.theta[0] *= 2.0;
    let wrong = ks_uniform(pds(&doubled));
    eprintln!("T4: censored {frac:.3}; KS correct {right:.4}, hazard x2 {wrong:.4}, bound {BOUND}");
    assert!(right < BOUND, "correct model: KS {right} >= {BOUND}");
    assert!(wrong > BOUND, "hazard x2: KS {wrong} <= {BOUND}");
}

/// The warning fold: one string per code, the first message kept, the count stated
/// only when there is more than one.
///
/// Dies on: no fold (three strings for the repeated code); the count dropped; a single
/// warning given a count; codes merged.
#[test]
fn simulator_warnings_fold_to_one_per_code() {
    let folded = fold_warnings(
        vec![
            "W_A: subject '1' first".into(),
            "W_B: only".into(),
            "W_A: subject '2' second".into(),
            "W_A: subject '1' first".into(),
        ],
        250,
    );
    assert_eq!(
        folded,
        vec![
            "W_A: subject '1' first [3 `W_A` warnings were raised over the 250 npde reference \
             replicates; the first is shown.]"
                .to_string(),
            "W_B: only".to_string(),
        ]
    );
    assert!(fold_warnings(vec![], 10).is_empty());
}

/// Bounds of T5, set from the realised worst errors it prints (macOS arm64, R 4.6.1,
/// npde 3.5): `|pd − (1 − pd_R)|` and `|F̂(c) − (1 − ploq_R)|` are both 5.6e-17, one
/// rounding of `1 − x`; `|npde + npde_R|` is 2.1e-9, the gap between `normal_inv_cdf`
/// and R's `qnorm`. The bounds leave 18× and 5×.
const PD_BOUND: f64 = 1e-15;
const NPDE_BOUND: f64 = 1e-8;

fn reference_file(name: &str) -> Vec<Vec<String>> {
    let path = format!(
        "{}/tests/fixtures/npde_tte_reference/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect()
}

/// T5. The kernel against the R `npde` package (CRAN 3.5), on simulated times R drew
/// and committed (`tests/fixtures/npde_tte_reference/`, `make_reference.R`). The
/// package has no TTE mode; it is fed `−T` with right censoring as left censoring,
/// which is exact for event rows and for **one common censoring time** only (the
/// package applies a single LOQ to every subject), so the fixture censors
/// administratively at one time. R cannot reproduce the imputed uniform, so the
/// censored rows compare what is deterministic: `F̂(c)` and the interval.
///
/// - event rows: `pd = 1 − pd_R` and `npde = −npde_R` (R scores `−T`, so its sign is
///   the opposite of ours; asserted as such so a "fix" to R's sign cannot pass);
/// - censored rows: `F̂(c) = 1 − ploq_R`, and R's imputed `pd_R ≤ ploq_R`, the mirror
///   of our `pd ≥ F̂(c)`;
/// - subject 1 (an event before every simulated time) and subject 2 (after all of
///   them) sit on the two clamp ends.
///
/// The committed `sims.csv` holds the draws as the simulator reports them: a draw at
/// or past the horizon is `(horizon, no event)`. R scored the unbounded draws.
///
/// Dies on: the clamp dropped (the two edge rows); a no-event replicate counted
/// (`time <= y` at the horizon); one simulated time moved across its subject's `y`.
#[test]
fn kernel_matches_the_r_npde_package() {
    let obs = reference_file("obs.csv");
    let sims = reference_file("sims.csv");
    let reference = reference_file("reference.csv");
    assert_eq!(obs.len(), 40);
    assert_eq!(reference.len(), obs.len());
    let num = |s: &str| s.parse::<f64>().unwrap_or_else(|_| panic!("number {s:?}"));

    let (mut worst_pd, mut worst_npde, mut worst_f) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut n_event, mut n_censored) = (0, 0);
    for (o, r) in obs.iter().zip(&reference) {
        assert_eq!(o[0], r[0], "row order");
        let (y, is_event) = (num(&o[1]), o[2] == "1");
        let draws: Vec<(f64, bool)> = sims
            .iter()
            .filter(|s| s[0] == o[0])
            .map(|s| (num(&s[2]), s[3] == "1"))
            .collect();
        let k = draws.len();
        assert_eq!(k, 200, "subject {}", o[0]);
        let n = draws
            .iter()
            .filter(|(t, observed)| event_before(*t, *observed, y))
            .count();
        let (pd_r, ploq_r, npde_r, npd_r) = (num(&r[1]), num(&r[2]), num(&r[3]), num(&r[4]));
        assert_eq!(
            npde_r, npd_r,
            "one observation per subject: R's npde is its npd"
        );
        if is_event {
            n_event += 1;
            let pd = tte_pd(n, k, false, 0.0);
            let (e_pd, e_npde) = (
                (pd - (1.0 - pd_r)).abs(),
                (normal_inv_cdf(pd) + npde_r).abs(),
            );
            assert!(e_pd.is_finite() && e_npde.is_finite());
            if e_pd > worst_pd {
                worst_pd = e_pd;
            }
            if e_npde > worst_npde {
                worst_npde = e_npde;
            }
            match o[0].as_str() {
                "1" => assert_eq!(pd, 1.0 / 400.0, "before every draw: the lower clamp"),
                "2" => assert_eq!(pd, 1.0 - 1.0 / 400.0, "after every draw: the upper clamp"),
                _ => {}
            }
        } else {
            n_censored += 1;
            let f = n as f64 / k as f64;
            let e_f = (f - (1.0 - ploq_r)).abs();
            assert!(e_f.is_finite());
            if e_f > worst_f {
                worst_f = e_f;
            }
            // R imputes on [0, ploq]; we impute on [F̂(c), 1], its mirror.
            assert!(pd_r <= ploq_r.max(1.0 / 400.0), "subject {}", o[0]);
            for u in [0.0, 0.5, 0.999] {
                let pd = tte_pd(n, k, true, u);
                assert!(
                    1.0 - pd <= ploq_r.max(1.0 / 400.0) + 1e-15,
                    "subject {}",
                    o[0]
                );
            }
        }
    }
    assert_eq!((n_event, n_censored), (17, 23));
    eprintln!(
        "T5: worst |pd - (1 - pd_R)| {worst_pd:.3e}; worst |npde + npde_R| {worst_npde:.3e}; \
         worst |F(c) - (1 - ploq_R)| {worst_f:.3e}"
    );
    assert!(worst_pd < PD_BOUND, "{worst_pd}");
    assert!(worst_npde < NPDE_BOUND, "{worst_npde}");
    assert!(worst_f < PD_BOUND, "{worst_f}");
}

/// Joint PK-TTE data for `model`: per subject one dose, one concentration and (unless
/// `tte` is `None`) one record on cmt 3.
fn joint_population(model: &CompiledModel, tte: &[Option<(f64, bool)>]) -> Population {
    use std::io::Write;
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n");
    for (i, row) in tte.iter().enumerate() {
        let id = i + 1;
        csv.push_str(&format!("{id},0,.,1,100,1,0,1\n{id},2,5,0,.,2,0,0\n"));
        if let Some((y, is_event)) = row {
            csv.push_str(&format!("{id},{y},{},0,0,3,0,0\n", *is_event as u8));
        }
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::api::read_population_for(
        model,
        &None,
        f.path().to_str().unwrap(),
        None,
        None,
        None,
        &[],
    )
    .expect("population reads")
    .0
}

fn ode_hazard_model(file: &str) -> CompiledModel {
    let src = std::fs::read_to_string(format!(
        "{}/nonmem_anchor/{file}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    // IIV on CL, so the drug-driven hazard differs between replicates; the anchor's
    // own `[fit_options]` (tight tolerances for a NONMEM comparison) is dropped.
    let src = src
        .replace(
            "  sigma PROP_ERR",
            "  omega ETA_CL ~ 0.09\n  sigma PROP_ERR",
        )
        .replace("  CL   = TVCL", "  CL   = TVCL * exp(ETA_CL)");
    assert!(src.contains("ETA_CL ~ 0.09") && src.contains("TVCL * exp(ETA_CL)"));
    model(src.split("\n[fit_options]").next().unwrap())
}

/// T6. An ODE-accumulated hazard on joint PK-TTE subjects.
///
/// Constant arm (`hazard = H0`, integrated by the ODE engine): `pd` against
/// `1 − exp(−H0·t)`, four Monte-Carlo SDs. Drug-driven arm: one finite score per event
/// record, none for a subject with no TTE record, and `compute_npde_npd` on the same
/// population gives one Gaussian score per concentration row and none for the TTE
/// record.
///
/// Dies on: no horizon passed to the simulator (refused for an ODE hazard); a score
/// emitted per observation record; the concentration rows scored here.
#[test]
fn ode_hazard_and_joint_subjects() {
    let k = 2000;
    let m = ode_hazard_model("ss_chz_const_fit.ferx");
    let times = [10.0, 40.0, 80.0];
    let p = joint_population(&m, &times.map(|t| Some((t, true))));
    let (out, warnings) = compute_tte_npde(&m, &p, &m.default_params, k, Some(4), None).unwrap();
    let warnings = tte_warnings(warnings);
    assert!(warnings.is_empty(), "{warnings:?}");
    for (s, &t) in out.iter().zip(&times) {
        let want = 1.0 - (-0.02 * t).exp();
        let [row] = s.as_slice() else {
            panic!("one score per subject")
        };
        assert_eq!((row.cmt, row.time, row.imputed), (3, t, false));
        assert!(row.pd.is_finite());
        let (err, bound) = (
            (row.pd - want).abs(),
            4.0 * (want * (1.0 - want) / k as f64).sqrt(),
        );
        eprintln!("T6 const: t = {t}: pd {:.4}, closed form {want:.4}, error {err:.2e}, bound {bound:.2e}", row.pd);
        assert!(err < bound);
    }

    let m = ode_hazard_model("ss_chz_drug_fit.ferx");
    let mut rows: Vec<Option<(f64, bool)>> = (0..12)
        .map(|i| Some((4.0 + 3.0 * i as f64, i % 3 != 0)))
        .collect();
    rows[5] = None;
    let p = joint_population(&m, &rows);
    let gaussian = |p: &Population| {
        super::super::npde::compute_npde_npd(&m, p, &m.default_params, 200, Some(4), None)
            .unwrap()
            .into_iter()
            .map(|s| (s.npd, s.npde))
            .collect::<Vec<_>>()
    };
    let gaussian_scores = gaussian(&p);
    let (out, _) = compute_tte_npde(&m, &p, &m.default_params, 200, Some(4), None).unwrap();
    assert_eq!(out.len(), 12);
    for (i, (s, row)) in out.iter().zip(&rows).enumerate() {
        match row {
            None => assert!(s.is_empty(), "subject index {i} has no TTE record"),
            Some((y, is_event)) => {
                assert_eq!(s.len(), 1, "one score per event record, subject index {i}");
                assert_eq!((s[0].time, s[0].imputed), (*y, !*is_event));
                assert!(s[0].pd.is_finite() && s[0].npde.is_finite());
                assert!(s[0].pd > 0.0 && s[0].pd < 1.0);
            }
        }
    }
    // The concentration rows: one Gaussian score each, the TTE record none.
    assert!(gaussian_scores
        .iter()
        .all(|(npd, npde)| npd.len() == 1 && npde.len() == 1));
    assert!(gaussian_scores.iter().all(|(npd, _)| npd[0].is_finite()));
}

/// D6. A replicate with no event is in the denominator, and its per-subject simulator
/// warning comes back once, with a count, not once per replicate. A Gompertz hazard
/// with `gamma < 0` has a cure fraction: a replicate that never has an event is a
/// legitimate "never", which the simulator reports as `W_TTE_DEGENERATE_HAZARD`.
/// Measured: 493 raw warnings over 3 subjects × 250 replicates.
///
/// Dies on: the fold skipped (493 strings); a no-event replicate dropped from the
/// denominator (fewer than 250 drawn); the count in the folded text wrong.
#[test]
fn no_event_replicates_count_and_their_warnings_fold() {
    let m = model(
        "\n[parameters]\n  theta TVA(0.2, 0.001, 10.0)\n  theta TVG(-0.5, -5.0, 5.0)\n\n[event_model]\n  cmt    = 2\n  family = gompertz\n  alpha  = TVA\n  gamma  = TVG\n",
    );
    let p = pop(vec![event(1.0), censor(30.0), event(4.0)]);
    let scored = scored_records(&p, 2).unwrap();
    let (before, drawn, raw) =
        reference_counts(&m, &p, &m.default_params, 250, 5, None, 2, &scored).unwrap();
    assert_eq!(drawn, vec![250; 3]);
    let raw = tte_warnings(raw);
    assert!(raw.len() > 250, "{} raw warnings", raw.len());
    assert!(raw
        .iter()
        .all(|w| w.starts_with("W_TTE_DEGENERATE_HAZARD: ")));
    // The cure fraction is exp(−α/|γ|) = exp(−0.4) ≈ 0.67: about a third have an event.
    assert!(before[1] > 50 && before[1] < 120, "{before:?}");

    let (out, warnings) = compute_tte_npde(&m, &p, &m.default_params, 250, Some(5), None).unwrap();
    let warnings = tte_warnings(warnings);
    let [w] = warnings.as_slice() else {
        panic!("one folded warning, got {}", warnings.len())
    };
    assert!(w.starts_with(&raw[0]), "{w}");
    assert!(
        w.ends_with(&format!(
            " [{} `W_TTE_DEGENERATE_HAZARD` warnings were raised over the 250 npde reference \
             replicates; the first is shown.]",
            raw.len()
        )),
        "{w}"
    );
    assert!(out.iter().all(|s| s[0].pd.is_finite()));
    // The event at t = 1 is scored from its own count over all 250 replicates.
    assert_eq!(out[0][0].pd, before[0] as f64 / 250.0);
}

/// T16. A `TteObsDiagnostic` serialised before the `npde` field existed reads back
/// with `npde` NaN ("not computed"), and a present score round-trips.
///
/// Dies on: a plain `#[serde(default)]`, which gives `0.0`, a valid-looking score.
#[test]
fn tte_row_without_npde_deserialises_to_nan() {
    use crate::types::TteObsDiagnostic;
    let row = TteObsDiagnostic {
        cmt: 2,
        time: 5.0,
        dv: 1.0,
        coxsnell: 0.5,
        martingale: 0.5,
        deviance: 0.25,
        pit: 0.6,
        npde: -1.25,
    };
    let mut json = serde_json::to_value(&row).unwrap();
    let back: TteObsDiagnostic = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(back, row);
    assert!(json.as_object_mut().unwrap().remove("npde").is_some());
    let old: TteObsDiagnostic = serde_json::from_value(json).unwrap();
    assert!(old.npde.is_nan(), "{}", old.npde);
    assert_eq!((old.cmt, old.time, old.pit), (2, 5.0, 0.6));
}

/// `TteNpde::time` is the data TIME as written, not the engine clock (review r1 #3).
/// The second subject's record lies in a reset period that restarts `TIME`: engine
/// time 14, written 4. No other fixture has a reset period, so everywhere else the two
/// clocks are equal and a score reporting the engine time passes.
///
/// Dies on: `param_time_at` dropped (14, not 4).
#[test]
fn reported_time_is_the_written_time() {
    let m = model(EXP_FIXED);
    let mut p = pop(vec![event(3.0), event(14.0)]);
    p.subjects[1].reset_period_times = vec![(10.0, 0.0)];
    let out = scores(&m, &p, 100, 1);
    assert_eq!(out[0][0].time, 3.0);
    assert_eq!(out[1][0].time, 4.0);
    // The score itself is on the engine clock: F(14), not F(4).
    assert!(out[1][0].pd > 0.6, "{}", out[1][0].pd);
}

/// Records that are not on the scored endpoint are skipped, not scored and not
/// refused (review r1 #11): a discrete-endpoint record, and an event record on
/// another compartment.
///
/// Dies on: either skip removed (two records on the endpoint are refused, or the
/// other compartment's time is scored).
#[test]
fn records_off_the_endpoint_are_skipped() {
    let p = pop(vec![
        vec![
            ObsRecord::DiscreteState {
                time: 1.0,
                raw_time: 1.0,
                state: 1,
                cmt: 5,
            },
            rec(9.0, EventType::Exact, 0.0, 3),
            rec(4.0, EventType::RightCensored, 0.0, 2),
        ],
        vec![rec(9.0, EventType::Exact, 0.0, 3)],
    ]);
    let scored = scored_records(&p, 2).unwrap();
    assert_eq!(scored, vec![Some((4.0, 4.0, true)), None]);
}

/// What a fit keeps of the reference draws' warnings (review r1 #1): a failed solve,
/// reworded for where it happened; never the degenerate-hazard text, which every
/// cure-fraction replicate raises.
///
/// Dies on: the degenerate-hazard warning forwarded; the solve warning forwarded
/// unchanged, or dropped; the count clause lost.
#[test]
fn a_fit_keeps_only_failed_solves_of_the_reference_draws() {
    let kept = fit_time_warnings(vec![
        "W_TTE_DEGENERATE_HAZARD: subject '2' (CMT=2) drew a non-positive rate [493 more]".into(),
        "W_TTE_SOLVE_FAILED: subject '7' (CMT=3): the solve failed [12 such]".into(),
        "W_IOV_NO_OCCASIONS: something".into(),
    ]);
    assert_eq!(
        kept,
        vec![
            "W_TTE_SOLVE_FAILED: in the npde reference replicates of the time-to-event \
             records, where a failed solve counts as \"no event\": subject '7' (CMT=3): the \
             solve failed [12 such]"
                .to_string()
        ]
    );
    assert!(fit_time_warnings(vec![]).is_empty());
}

/// Every scored time at 0 (review r1 #9): no event lies strictly before time 0, so the
/// scores are exact without a simulation. An event sits at the lower clamp; a censored
/// time is its own uniform on (0, 1). Before, this input was refused with a message
/// about `SimulateOptions.horizon`, an option the caller never set.
///
/// Dies on: the early return removed (the simulator refuses a horizon of 0); a
/// censored record given the lower clamp too; a record-less subject given a score.
#[test]
fn all_scored_times_at_zero_are_scored_without_simulating() {
    let m = model(EXP_IIV);
    let p = pop(vec![event(0.0), censor(0.0), vec![], censor(0.0)]);
    let (out, warnings) =
        compute_tte_npde(&m, &p, &m.default_params, 200, Some(9), None).expect("scored");
    assert!(warnings.is_empty(), "{warnings:?}");
    let lo = 1.0 / 400.0;
    assert_eq!((out[0][0].pd, out[0][0].imputed), (lo, false));
    assert!(out[2].is_empty());
    for i in [1, 3] {
        let row = &out[i][0];
        assert!(row.imputed);
        assert_eq!(
            row.pd,
            clamp_prob(impute_uniform(9, i), 200),
            "subject index {i}"
        );
        assert!(row.npde.is_finite());
    }
    assert_ne!(out[1][0].pd, out[3][0].pd);
    assert_ne!(out[1][0].pd, lo);
}
