use super::*;

/// The survfit probe set (#1886 plan M2): a tie of two events at 3, an event and a
/// censoring at both 5 and 8, a censoring at 1.
const T: [f64; 10] = [1.0, 2.0, 3.0, 3.0, 5.0, 5.0, 7.0, 8.0, 8.0, 10.0];
const E: [u8; 10] = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0];

fn m2() -> KmCurve {
    let ev: Vec<bool> = E.iter().map(|&e| e == 1).collect();
    kaplan_meier(&T, &ev).unwrap()
}

/// A curve that stays at `s` from time 0 to `end`, its last record censored, so
/// [`KmCurve::at`] is `Some(s)` on `[0, end]` and `None` after it.
fn flat(s: f64, end: f64) -> KmCurve {
    KmCurve {
        time: vec![0.0, end],
        n_risk: vec![2, 1],
        n_event: vec![1, 0],
        n_censor: vec![0, 1],
        surv: vec![s, s],
        cumhaz: vec![0.0, 0.0],
    }
}

/// T1. The closed form under survfit's convention, computed with exact fractions
/// outside the engine (plan M3) and equal to survfit 3.8.12's rows (M2):
/// `S(5) = 5/9`, `n.risk(5) = 6`, `S(8) = 5/18`, `H(8) = 10/9`. Counting the
/// censoring at 5 before the event there gives `S(5) = 0.5333` instead, a gap of
/// 0.022 against a bound of 1e-15; dropping the censored subjects from their own
/// time's risk set is the same mutation.
#[test]
fn km_matches_survfit_tie_convention_closed_form() {
    let c = m2();
    let want_t = [1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0];
    let want_n = [10, 9, 8, 6, 4, 3, 1];
    let want_s = [
        1.0,
        8.0 / 9.0,
        2.0 / 3.0,
        5.0 / 9.0,
        5.0 / 12.0,
        5.0 / 18.0,
        5.0 / 18.0,
    ];
    let want_h = [
        0.0,
        1.0 / 9.0,
        13.0 / 36.0,
        19.0 / 36.0,
        7.0 / 9.0,
        10.0 / 9.0,
        10.0 / 9.0,
    ];
    assert_eq!(c.time, want_t);
    assert_eq!(c.n_risk, want_n, "risk set: censorings at t stay in it");
    for j in 0..want_t.len() {
        assert!(
            (c.surv[j] - want_s[j]).abs() < 1e-15,
            "S({}) = {} != {}",
            want_t[j],
            c.surv[j],
            want_s[j]
        );
        assert!(
            (c.cumhaz[j] - want_h[j]).abs() < 1e-15,
            "H({}) = {} != {}",
            want_t[j],
            c.cumhaz[j],
            want_h[j]
        );
    }
}

/// T2. Tied times form one row: one step at 3 carrying both events, and seven rows
/// for seven distinct times. Processing tied events one at a time leaves `S`
/// unchanged at the end of the tie but splits the row, so both are asserted.
#[test]
fn km_groups_ties_into_one_row() {
    let c = m2();
    assert_eq!(c.time.len(), 7);
    assert_eq!(c.n_event, [0, 1, 2, 1, 1, 1, 0]);
    assert_eq!(c.n_censor, [1, 0, 0, 1, 0, 1, 1]);
}

/// T3. Past the last time a curve is undefined unless it has reached zero. Always
/// carrying the last value forward breaks the first assertion; always returning
/// `None` breaks the second.
#[test]
fn km_at_past_last_time_is_none_unless_curve_reached_zero() {
    let censored_last = kaplan_meier(&[1.0, 2.0, 3.0], &[true, true, false]).unwrap();
    assert_eq!(censored_last.at(4.0), None);
    assert!((censored_last.at(3.0).unwrap() - 1.0 / 3.0).abs() < 1e-15);
    let hit_zero = kaplan_meier(&[1.0, 2.0], &[true, true]).unwrap();
    assert_eq!(hit_zero.at(3.0), Some(0.0));
    // Before the first time, and between steps.
    assert_eq!(hit_zero.at(0.5), Some(1.0));
    assert_eq!(hit_zero.at(1.5), Some(0.5));
    assert_eq!(hit_zero.at(f64::NAN), None);
    assert_eq!(KmCurve::default().at(1.0), None);
}

/// Bad input is refused, not panicked on.
#[test]
fn km_refuses_bad_input() {
    assert_eq!(
        kaplan_meier(&[1.0], &[true, false]),
        Err(KmInputError::LengthMismatch {
            times: 1,
            events: 2
        })
    );
    let err = kaplan_meier(&[1.0, f64::NAN], &[true, false]).unwrap_err();
    assert!(matches!(err, KmInputError::BadTime { index: 1, .. }));
    assert!(err.to_string().contains("index 1"), "{err}");
    let err = kaplan_meier(&[-1.0], &[true]).unwrap_err();
    assert_eq!(
        err,
        KmInputError::BadTime {
            index: 0,
            value: -1.0
        }
    );
    assert!(kaplan_meier(&[], &[]).unwrap().time.is_empty());
}

/// T4. A replicate past its last follow-up is left out of the band, not counted.
/// 22 curves reach t* = 5, 3 end at 2 with S > 0. Folding the 3 in as 1.0 or 0.0
/// raises `n_used` to 25 and moves every quantile.
#[test]
fn band_uses_only_curves_defined_at_the_grid_time() {
    let mut curves: Vec<KmCurve> = (0..22).map(|i| flat((i + 1) as f64 / 30.0, 10.0)).collect();
    curves.extend((0..3).map(|_| flat(0.9, 2.0)));
    let probs = [0.05, 0.5, 0.95];
    let band = vpc_band(&curves, &[1.0, 5.0], probs).unwrap();
    assert_eq!(band.n_used, [25, 22]);
    let mut v: Vec<f64> = (0..22).map(|i| (i + 1) as f64 / 30.0).collect();
    v.sort_by(f64::total_cmp);
    assert_eq!(band.lo[1], Some(quantile_sorted(&v, 0.05)));
    assert_eq!(band.med[1], Some(quantile_sorted(&v, 0.5)));
    assert_eq!(band.hi[1], Some(quantile_sorted(&v, 0.95)));
}

/// T5. A grid time reached by fewer than 19 curves (the 5 %/95 % floor) has no band
/// point; where there is one it is finite. Removing the floor makes the second
/// point `Some`.
#[test]
fn band_point_is_none_below_the_resolvable_floor() {
    let mut curves: Vec<KmCurve> = (0..18).map(|i| flat((i + 1) as f64 / 20.0, 10.0)).collect();
    curves.extend((0..7).map(|_| flat(0.5, 2.0)));
    let band = vpc_band(&curves, &[1.0, 5.0], [0.05, 0.5, 0.95]).unwrap();
    assert_eq!(band.n_used, [25, 18]);
    assert_eq!((band.lo[1], band.med[1], band.hi[1]), (None, None, None));
    for v in [band.lo[0], band.med[0], band.hi[0]] {
        assert!(v.expect("25 curves resolve the band").is_finite());
    }
}

/// The floor itself, `⌈1 / min(p, 1 − p)⌉ − 1`: an exact 20 from `1 / 0.05` must not
/// round up to 21.
#[test]
fn min_curves_for_reads_the_smallest_tail() {
    assert_eq!(min_curves_for([0.05, 0.5, 0.95]), 19);
    assert_eq!(min_curves_for([0.1, 0.5, 0.9]), 9);
    assert_eq!(min_curves_for([0.025, 0.5, 0.975]), 39);
    assert_eq!(min_curves_for([0.25, 0.5, 0.75]), 3);
    // The tail is the last probability's when it is the smaller one.
    assert_eq!(min_curves_for([0.2, 0.5, 0.95]), 19);
}

/// T6. The band is R's type-7 quantile. Against R 4.6 `quantile()`:
/// `c(.1,.4,.2,.9,.5)` at `.25/.5/.75` is `.20/.40/.50` (type 6: `.15/.40/.70`), and
/// `(1:20)/21` at `.05/.5/.95` is `0.0928571…/.5/0.9071428…` (type 6: `.05/.5/.95`).
/// At 5 curves the 5 %/95 % band is below its floor, so it is `None`.
#[test]
fn band_is_r_type7_quantile() {
    let five: Vec<KmCurve> = [0.1, 0.4, 0.2, 0.9, 0.5]
        .iter()
        .map(|&s| flat(s, 10.0))
        .collect();
    let b = vpc_band(&five, &[1.0], [0.25, 0.5, 0.75]).unwrap();
    let near = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 1e-15;
    assert!(
        near(b.lo[0], 0.2) && near(b.med[0], 0.4) && near(b.hi[0], 0.5),
        "{b:?}"
    );
    let b = vpc_band(&five, &[1.0], [0.05, 0.5, 0.95]).unwrap();
    assert_eq!(b.lo[0], None);

    let twenty: Vec<KmCurve> = (1..=20).map(|i| flat(i as f64 / 21.0, 10.0)).collect();
    let b = vpc_band(&twenty, &[1.0], [0.05, 0.5, 0.95]).unwrap();
    assert!(near(b.lo[0], 0.092_857_142_857_142_86), "{b:?}");
    assert!(near(b.med[0], 0.5));
    assert!(near(b.hi[0], 0.907_142_857_142_857_2), "{b:?}");
}

/// Probabilities outside (0, 1) or not increasing are refused, naming the values.
#[test]
fn band_refuses_bad_probs() {
    for probs in [
        [0.0, 0.5, 0.95],
        [0.05, 0.5, 1.0],
        [0.5, 0.5, 0.95],
        [0.95, 0.5, 0.05],
    ] {
        let err = vpc_band(&[], &[1.0], probs).unwrap_err();
        assert_eq!(err, KmInputError::BadProbs { probs });
        assert!(err.to_string().contains(&format!("{probs:?}")), "{err}");
    }
    assert!(!probs_valid([f64::NAN, 0.5, 0.95]));
}

// ── Mean cumulative function (#1886 PR2) ─────────────────────────────────────

/// Fixture R1 (#1886 PR2 plan §0): follow-up end and event times per subject. A tie
/// across subjects at 2, an event at the subject's own follow-up end (B at 8, E at 4)
/// and a subject with no event (C).
const R1: [(f64, &[f64]); 6] = [
    (10.0, &[2.0, 5.0, 9.0]), // A
    (8.0, &[3.0, 8.0]),       // B
    (6.0, &[]),               // C
    (10.0, &[2.0, 7.0]),      // D
    (4.0, &[4.0]),            // E
    (5.0, &[1.0]),            // F
];

/// The pooled event times and the follow-up ends of a fixture.
fn pooled(subjects: &[(f64, &[f64])]) -> (Vec<f64>, Vec<f64>) {
    let events = subjects.iter().flat_map(|s| s.1.iter().copied()).collect();
    let ends = subjects.iter().map(|s| s.0).collect();
    (events, ends)
}

/// R1 with a second event of A at 5: two events at one time within one subject.
fn r2() -> Vec<(f64, &'static [f64])> {
    let mut s = R1.to_vec();
    s[0] = (10.0, &[2.0, 5.0, 5.0]);
    s
}

/// An MCF that sits at `v` from time 0 to `end`, so [`McfCurve::at`] is `Some(v)` on
/// `[0, end]` and `None` after it.
fn flat_mcf(v: f64, end: f64) -> McfCurve {
    McfCurve {
        time: vec![0.0, end],
        n_risk: vec![1, 1],
        n_event: vec![1, 0],
        mcf: vec![v, v],
    }
}

/// PR2 T1. R1's rows against exact fractions computed outside the engine (Python
/// `fractions`) and equal to `survfit(Surv(start, stop, status) ~ 1, id = id)$cumhaz`
/// (survival 3.8.12), re-measured at the commit adding this test. A subject stays in
/// the risk set after its event: letting it leave gives 17/30 at t = 2 against 1/2, a
/// gap of 0.067 against a bound of 1e-15. An event at a subject's own follow-up end is
/// counted with that subject still at risk: `n_risk(4) = 6` and `n_risk(8) = 3`, where
/// counting the follow-up end first gives 5 and 2.
#[test]
fn mcf_matches_closed_form_and_keeps_subjects_at_risk_after_an_event() {
    let (events, ends) = pooled(&R1);
    let c = mean_cumulative(&events, &ends).unwrap();
    assert_eq!(c.time, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    assert_eq!(c.n_risk, [6, 6, 6, 6, 5, 4, 3, 3, 2, 2]);
    assert_eq!(c.n_event, [1, 2, 1, 1, 1, 0, 1, 1, 1, 0]);
    let want = [
        1.0 / 6.0,
        1.0 / 2.0,
        2.0 / 3.0,
        5.0 / 6.0,
        31.0 / 30.0,
        31.0 / 30.0,
        41.0 / 30.0,
        17.0 / 10.0,
        11.0 / 5.0,
        11.0 / 5.0,
    ];
    for j in 0..want.len() {
        assert!(c.mcf[j].is_finite());
        assert!(
            (c.mcf[j] - want[j]).abs() < 1e-15,
            "MCF({}) = {} != {}",
            c.time[j],
            c.mcf[j],
            want[j]
        );
    }
}

/// PR2 T2. Two events at one time in one subject both count: `n_event(5) = 2`,
/// `MCF(5) = 37/30` and `MCF(8) = 19/10` (exact closed form, and equals
/// `reda::mcf` 0.5.6; survfit refuses `stop == start`). Deduplicating event times
/// gives `n_event(5) = 1` and 31/30.
#[test]
fn mcf_counts_two_events_at_one_time_in_one_subject() {
    let (events, ends) = pooled(&r2());
    let c = mean_cumulative(&events, &ends).unwrap();
    let row = |t: f64| c.time.iter().position(|&x| x == t).expect("row");
    assert_eq!(c.n_event[row(5.0)], 2);
    assert_eq!(c.n_risk[row(5.0)], 5);
    assert!((c.mcf[row(5.0)] - 37.0 / 30.0).abs() < 1e-15, "{c:?}");
    assert!((c.mcf[row(8.0)] - 19.0 / 10.0).abs() < 1e-15, "{c:?}");
}

/// PR2 T3. `at`: 0 before the first row, the step value from it on, and `None` past the
/// last follow-up end (carrying 11/5 forward would give `Some`). The subject with no
/// event is in the risk set from the first row: without its follow-up end
/// `n_risk(1)` is 5.
#[test]
fn mcf_at_is_zero_before_the_first_row_and_none_past_the_last_follow_up_end() {
    let (events, ends) = pooled(&R1);
    let c = mean_cumulative(&events, &ends).unwrap();
    assert_eq!(c.n_risk[0], 6, "the subject with no event is at risk");
    assert_eq!(c.at(0.5), Some(0.0));
    assert_eq!(c.at(1.0), Some(c.mcf[0]));
    assert_eq!(c.at(2.5), Some(0.5));
    assert!((c.at(10.0).unwrap() - 11.0 / 5.0).abs() < 1e-15);
    assert_eq!(c.at(10.5), None);
    assert_eq!(c.at(f64::NAN), None);
    assert_eq!(McfCurve::default().at(1.0), None);
    // No subject and no event: an empty curve.
    assert_eq!(mean_cumulative(&[], &[]), Ok(McfCurve::default()));
    // Subjects with no event at all: a flat zero until the last follow-up end.
    let none = mean_cumulative(&[], &[3.0, 7.0]).unwrap();
    assert_eq!(none.mcf, [0.0, 0.0]);
    assert_eq!((none.at(7.0), none.at(7.5)), (Some(0.0), None));
}

/// PR2 T4. Bad input is refused with a message naming the value; an event later than
/// every follow-up end would divide by an empty risk set. An event exactly at the last
/// follow-up end is fine.
#[test]
fn mcf_refuses_bad_input() {
    let err = mean_cumulative(&[1.0, -2.0], &[5.0]).unwrap_err();
    assert_eq!(
        err,
        McfInputError::BadEventTime {
            index: 1,
            value: -2.0
        }
    );
    assert_eq!(
        err.to_string(),
        "mean_cumulative: event time -2 at index 1 is not a finite, non-negative number"
    );
    let err = mean_cumulative(&[1.0], &[5.0, f64::INFINITY]).unwrap_err();
    assert!(matches!(
        err,
        McfInputError::BadFollowUpEnd { index: 1, .. }
    ));
    assert_eq!(
        err.to_string(),
        "mean_cumulative: follow-up end inf at index 1 is not a finite, non-negative number"
    );
    assert!(matches!(
        mean_cumulative(&[f64::NAN], &[5.0]),
        Err(McfInputError::BadEventTime { index: 0, .. })
    ));
    assert!(matches!(
        mean_cumulative(&[1.0], &[-5.0]),
        Err(McfInputError::BadFollowUpEnd { index: 0, .. })
    ));
    let err = mean_cumulative(&[2.0, 5.0], &[4.0, 3.0]).unwrap_err();
    assert_eq!(err, McfInputError::EventAfterFollowUp { time: 5.0 });
    assert_eq!(
        err.to_string(),
        "mean_cumulative: the event at time 5 is later than every follow-up end, so no \
         subject is at risk for it"
    );
    let at_end = mean_cumulative(&[2.0, 4.0], &[4.0, 3.0]).unwrap();
    assert_eq!(at_end.n_risk, [2, 2, 1]);
    assert!(at_end.mcf.iter().all(|v| v.is_finite()));
    assert_eq!(at_end.mcf, [0.5, 0.5, 1.5]);
}

/// PR2 T5. `vpc_band_mcf` uses only the curves defined at a grid time (folding the 3
/// that ended in as 0 raises `n_used` to 25 and moves every quantile), and a point
/// reached by fewer than 19 curves has no band (removing the floor makes it `Some`).
#[test]
fn mcf_band_uses_only_defined_curves_and_keeps_the_floor() {
    let probs = [0.05, 0.5, 0.95];
    let mut curves: Vec<McfCurve> = (0..22)
        .map(|i| flat_mcf((i + 1) as f64 / 4.0, 10.0))
        .collect();
    curves.extend((0..3).map(|_| flat_mcf(0.9, 2.0)));
    let band = vpc_band_mcf(&curves, &[1.0, 5.0], probs).unwrap();
    assert_eq!(band.n_used, [25, 22]);
    let v: Vec<f64> = (0..22).map(|i| (i + 1) as f64 / 4.0).collect();
    assert_eq!(band.lo[1], Some(quantile_sorted(&v, 0.05)));
    assert_eq!(band.med[1], Some(quantile_sorted(&v, 0.5)));
    assert_eq!(band.hi[1], Some(quantile_sorted(&v, 0.95)));

    let mut curves: Vec<McfCurve> = (0..18)
        .map(|i| flat_mcf((i + 1) as f64 / 4.0, 10.0))
        .collect();
    curves.extend((0..7).map(|_| flat_mcf(0.5, 2.0)));
    let band = vpc_band_mcf(&curves, &[1.0, 5.0], probs).unwrap();
    assert_eq!(band.n_used, [25, 18]);
    assert_eq!((band.lo[1], band.med[1], band.hi[1]), (None, None, None));
    for v in [band.lo[0], band.med[0], band.hi[0]] {
        assert!(v.expect("25 curves resolve the band").is_finite());
    }
    assert_eq!(
        vpc_band_mcf(&curves, &[1.0], [0.5, 0.5, 0.95]),
        Err(KmInputError::BadProbs {
            probs: [0.5, 0.5, 0.95]
        })
    );
}
