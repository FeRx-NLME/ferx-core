use super::*;
use rand::{RngExt, SeedableRng};

/// The hand fixture of T1 (#1887 plan §3): 8 subjects, causes A = 1 and B = 2.
/// At t = 2 an A event, a B event and a censoring tie; at t = 4 two A events and a
/// censoring tie.
const T: [f64; 8] = [1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0];
const C: [usize; 8] = [1, 1, 2, 0, 0, 1, 1, 0];

fn t1() -> CifCurve {
    aalen_johansen(&T, &C, 2).unwrap()
}

/// T1. Exact fractions, by hand, under survfit's tie convention (events before
/// censorings at a tied time):
///
/// | t | n | d_A | d_B | c | S | F_A | F_B |
/// |---|---|---|---|---|---|---|---|
/// | 1 | 8 | 1 | 0 | 0 | 7/8  | 1/8 | 0   |
/// | 2 | 7 | 1 | 1 | 1 | 5/8  | 1/4 | 1/8 |
/// | 3 | 4 | 0 | 0 | 1 | 5/8  | 1/4 | 1/8 |
/// | 4 | 3 | 2 | 0 | 1 | 5/24 | 2/3 | 1/8 |
///
/// The drop at t = 2 is 1/4, split half to each cause; at t = 4 it is 5/12, all to A.
/// Mutations that redden it, each measured: form A with `S(t)` in place of `S(t⁻)`
/// (F_A(1) = 7/64); giving each cause the whole drop instead of its `d_k/d` share
/// (F_A(2) = 3/8); `1 − KM_A` with B censored (F_A(4) = 3/4, plan M4); writing a
/// cause-`c` event to row `c` instead of `c − 1` (out of bounds). Counting the censoring
/// at 2 before the events there is `kaplan_meier`'s convention, pinned by `km_tests`
/// and, through this estimator, by `tests/cif_survfit_reference.rs`.
#[test]
fn aj_matches_hand_fractions_with_mixed_ties() {
    let c = t1();
    assert_eq!(c.all_cause.time, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(c.all_cause.n_risk, [8, 7, 4, 3]);
    assert_eq!(c.n_event, [vec![1, 1, 0, 2], vec![0, 1, 0, 0]]);
    let want_s = [7.0 / 8.0, 5.0 / 8.0, 5.0 / 8.0, 5.0 / 24.0];
    let want_a = [1.0 / 8.0, 1.0 / 4.0, 1.0 / 4.0, 2.0 / 3.0];
    let want_b = [0.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0];
    for j in 0..4 {
        let at = format!("row {j} (t = {})", T[j]);
        assert!((c.all_cause.surv[j] - want_s[j]).abs() < 1e-15, "{at}: S");
        assert!((c.cif[0][j] - want_a[j]).abs() < 1e-15, "{at}: F_A = {}", c.cif[0][j]);
        assert!((c.cif[1][j] - want_b[j]).abs() < 1e-15, "{at}: F_B = {}", c.cif[1][j]);
    }
}

/// Seeded random `(times, causes)` with ties: times rounded to 0, 1 or 2 decimals,
/// sizes 5–400, about 30 % censored, causes uniform on `1..=n_causes` (plan M2/M3).
fn random_sets(n_causes: usize, sets: usize) -> Vec<(Vec<f64>, Vec<usize>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1887);
    (0..sets)
        .map(|_| {
            let n = [5usize, 20, 100, 400][rng.random_range(0..4)];
            let scale = [1.0, 10.0, 100.0][rng.random_range(0..3)];
            let times = (0..n)
                .map(|_| {
                    let u: f64 = rng.random_range(0.0..1.0);
                    (-(1.0 - u).ln() / 0.1 * scale).round() / scale
                })
                .collect();
            let causes = (0..n)
                .map(|_| {
                    if rng.random_range(0.0..1.0) < 0.3 {
                        0
                    } else {
                        rng.random_range(1..=n_causes)
                    }
                })
                .collect();
            (times, causes)
        })
        .collect()
}

/// T2. With one cause the cumulative incidence is `1 − S` (the sum telescopes), on
/// 2000 seeded tied datasets. Measured on macOS arm64: worst |F − (1 − S)| =
/// 1.776e-15 over 145 620 rows (Python model of the estimator, plan M2: 1.8e-15); the
/// bound 1e-14 keeps 5.6× headroom. Mutation-checked: form A with `S(t)` in place of
/// `S(t⁻)`, and a drop that never advances `S(t⁻)`, redden it. Giving each cause the
/// whole drop instead of its `d_k/d` share does not — under one cause the share is 1 —
/// so T1 owns that one.
#[test]
fn single_cause_cif_is_one_minus_km() {
    let mut worst = 0.0_f64;
    let mut rows = 0usize;
    for (times, causes) in random_sets(1, 2000) {
        let c = aalen_johansen(&times, &causes, 1).unwrap();
        for (f, s) in c.cif[0].iter().zip(&c.all_cause.surv) {
            let gap = (f - (1.0 - s)).abs();
            assert!(gap.is_finite(), "non-finite F = {f} or S = {s}");
            worst = worst.max(gap);
            rows += 1;
        }
    }
    eprintln!("single cause: worst |F − (1 − S)| = {worst:.3e} over {rows} rows");
    assert!(worst <= 1e-14, "worst |F − (1 − S)| = {worst:e}");
}

/// T3. With three causes `Σ F_k + S = 1` at every row, on 2000 seeded tied datasets.
/// Measured on macOS arm64: worst |Σ F + S − 1| = 5.551e-16 (Python model, plan M3:
/// 7.8e-16); the bound 4e-15 keeps 7.2× headroom. Mutation-checked: form A with `S(t)`
/// in place of `S(t⁻)`, each cause taking the whole drop, and `1 − KM_k` each break the
/// sum and redden it.
#[test]
fn cif_and_survival_sum_to_one() {
    let mut worst = 0.0_f64;
    for (times, causes) in random_sets(3, 2000) {
        let c = aalen_johansen(&times, &causes, 3).unwrap();
        for (j, s) in c.all_cause.surv.iter().enumerate() {
            let total: f64 = c.cif.iter().map(|f| f[j]).sum::<f64>() + s;
            assert!(total.is_finite(), "non-finite sum at row {j}");
            worst = worst.max((total - 1.0).abs());
        }
    }
    eprintln!("three causes: worst |Σ F + S − 1| = {worst:.3e}");
    assert!(worst <= 4e-15, "worst |Σ F + S − 1| = {worst:e}");
}

/// T4. [`CifCurve::at`] beside [`KmCurve::at`] of its own all-cause curve: defined
/// exactly where that one is. On the T1 fixture (last record censored, `S = 5/24`):
/// `0` before the first time, the step value at and between rows, and `None` past the
/// last time. Returning the final value there reddens the `None` assertion; returning
/// `1 − S` reddens the per-cause values, since `F_A ≠ 1 − S` once B has an event (at
/// t = 2.5 and 4, the last two grid times).
#[test]
fn cif_at_follows_all_cause_definedness() {
    let c = t1();
    let km = &c.all_cause;
    for (t, a, b) in
        [(0.5, 0.0, 0.0), (1.0, 0.125, 0.0), (2.5, 0.25, 0.125), (4.0, 2.0 / 3.0, 0.125)]
    {
        assert!(km.at(t).is_some(), "t = {t}: all-cause defined");
        let (fa, fb) = (c.at(0, t).unwrap(), c.at(1, t).unwrap());
        assert!((fa - a).abs() < 1e-15, "t = {t}: F_A = {fa}, want {a}");
        assert!((fb - b).abs() < 1e-15, "t = {t}: F_B = {fb}, want {b}");
    }
    for t in [2.5, 4.0] {
        let one_minus_s = 1.0 - km.at(t).unwrap();
        assert!((c.at(0, t).unwrap() - one_minus_s).abs() > 0.1, "t = {t}: F_A is 1 − S");
    }
    assert_eq!(km.at(4.5), None, "S = 5/24 > 0 past the last time");
    assert_eq!(c.at(0, 4.5), None, "F_A past the last time with S > 0");
    assert_eq!(c.at(1, 4.5), None, "F_B past the last time with S > 0");
    assert_eq!(c.at(0, f64::NAN), None, "NaN");
    assert_eq!(c.at(2, 1.0), None, "cause index 2 of 2 causes");
}

/// T4, the other end. When the last subject has an event, `S` reaches zero. At the last
/// time every `F_k` is at its final value (summing to 1); past it both curves are
/// `None`, also at zero, the #2930 convention of [`KmCurve::at`], so a band past the
/// replicates' follow-up gets no point mass at the final value. Returning the final value
/// past the end reddens the `None` assertions; returning `None` at the last time reddens
/// the values.
#[test]
fn cif_at_reaching_zero_is_final_at_the_last_time_and_none_after() {
    let c = aalen_johansen(&[1.0, 2.0, 3.0], &[1, 0, 2], 2).unwrap();
    assert_eq!(c.all_cause.at(3.0), Some(0.0));
    let (fa, fb) = (c.at(0, 3.0).unwrap(), c.at(1, 3.0).unwrap());
    // S(1) = 2/3, S(3) = 0 after the censoring at 2: F_A = 1/3, F_B = 2/3.
    assert!((fa - 1.0 / 3.0).abs() < 1e-15, "F_A = {fa}");
    assert!((fb - 2.0 / 3.0).abs() < 1e-15, "F_B = {fb}");
    assert_eq!(c.all_cause.at(9.0), None, "S past the last time, at zero");
    assert_eq!(c.at(0, 9.0), None, "F_A past the last time, S at zero");
    assert_eq!(c.at(1, 9.0), None, "F_B past the last time, S at zero");
    assert_eq!(c.n_causes(), 2);
}

/// Bad input is refused with a message naming the function and the fix; an empty
/// input gives an empty curve that is nowhere defined.
#[test]
fn aj_refuses_bad_input_and_empty_is_undefined() {
    let e = aalen_johansen(&[1.0, 2.0], &[1], 1).unwrap_err();
    assert_eq!(e, CifInputError::LengthMismatch { times: 2, causes: 1 });
    assert!(e.to_string().contains("aalen_johansen: 2 times but 1 cause labels"), "{e}");
    let e = aalen_johansen(&[1.0, -1.0], &[1, 0], 1).unwrap_err();
    assert_eq!(e, CifInputError::BadTime { index: 1, value: -1.0 });
    assert!(e.to_string().contains("time -1 at index 1"), "{e}");
    assert!(matches!(
        aalen_johansen(&[f64::NAN], &[1], 1),
        Err(CifInputError::BadTime { index: 0, .. })
    ));
    let e = aalen_johansen(&[1.0, 2.0], &[1, 3], 2).unwrap_err();
    assert_eq!(e, CifInputError::BadCause { index: 1, value: 3, n_causes: 2 });
    let msg = e.to_string();
    assert!(msg.contains("cause 3 at index 1 is not in 0..=2"), "{msg}");
    assert!(msg.contains("0 for a censoring"), "{msg}");
    let c = aalen_johansen(&[], &[], 2).unwrap();
    assert_eq!(c.n_causes(), 2);
    assert_eq!(c.at(0, 0.0), None);
}

/// Step 3. [`vpc_band_cif`] over 30 seeded two-cause curves keeps exactly the curves
/// whose all-cause curve is defined at each grid time — `n_used` equals
/// [`vpc_band`](crate::stats::km::vpc_band)'s over the `all_cause` curves — and its
/// quantiles are the type-7 quantiles of those `F_k` values, computed here outside
/// the band. The grid straddles the curves' last times (asserted), so a band that read
/// a curve past its end, or dropped one before it, changes `n_used`. A `k` that is not
/// a cause leaves every point empty.
#[test]
fn cif_band_keeps_the_all_cause_definedness() {
    use crate::stats::convergence::quantile_sorted;
    use crate::stats::km::{min_curves_for, vpc_band};
    let curves: Vec<CifCurve> =
        random_sets(2, 30).iter().map(|(t, c)| aalen_johansen(t, c, 2).unwrap()).collect();
    let km: Vec<KmCurve> = curves.iter().map(|c| c.all_cause.clone()).collect();
    let grid = [0.5, 5.0, 15.0, 30.0, 60.0];
    let probs = [0.05, 0.5, 0.95];
    let all = vpc_band(&km, &grid, probs).unwrap();
    assert!(
        all.n_used.iter().any(|&n| n > min_curves_for(probs) && n < curves.len()),
        "the grid must straddle some curves' last times: n_used = {:?}",
        all.n_used
    );
    for k in 0..2 {
        let band = vpc_band_cif(&curves, k, &grid, probs).unwrap();
        assert_eq!(band.n_used, all.n_used, "cause {k}: n_used");
        for (g, &t) in grid.iter().enumerate() {
            let mut v: Vec<f64> = curves.iter().filter_map(|c| c.at(k, t)).collect();
            v.sort_by(f64::total_cmp);
            let want = |p: f64| (v.len() >= min_curves_for(probs)).then(|| quantile_sorted(&v, p));
            assert_eq!(band.lo[g], want(probs[0]), "cause {k}, t = {t}: lo");
            assert_eq!(band.med[g], want(probs[1]), "cause {k}, t = {t}: med");
            assert_eq!(band.hi[g], want(probs[2]), "cause {k}, t = {t}: hi");
        }
    }
    let none = vpc_band_cif(&curves, 2, &grid, probs).unwrap();
    assert!(none.n_used.iter().all(|&n| n == 0) && none.med.iter().all(Option::is_none));
    assert!(vpc_band_cif(&curves, 0, &grid, [0.5, 0.5, 0.9]).is_err(), "bad probs");
}
