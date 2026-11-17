//! The resolution gate on a `|θ₀|`-floored finite-difference step (#2406).
//!
//! A small identity-packed θ is stepped relative to its declared magnitude
//! ([`fd_step_floor`](crate::estimation::parameterization::fd_step_floor), #2187): right when the
//! objective's sensitivity scales as `1/θ₀` (`V = TVV/k`), wrong when it is O(1) (a covariate
//! exponent declared near 0). There a step of `1e-4·θ₀` is too small for the re-solved probes
//! to follow: warm-started at the mode, they stop inside the inner tolerance τ, and the
//! difference is that of a part-reconverged objective. Which case a coordinate is in cannot be
//! read off the model, so it is measured: the probes are re-solved a second time at τ/100, and
//! the step is accepted only when the two differences agree. Otherwise it grows ×10, never past
//! the pre-#2187 step (floor 1).
//!
//! One loop, [`resolve_floored_step`], for the reconverged FD gradients' first differences and
//! the covariance R-matrix's second differences, so the two cannot drift.

use crate::estimation::inner_optimizer::{EbeResult, InnerLoopStats};

/// Largest relative disagreement between the τ and τ/100 differences that accepts a step.
///
/// Measured, not the 1e-4 first proposed: the small-step error is a bias, not noise, and two
/// stopped solves can share it. On a covariate exponent declared at `1e-2` and estimated at
/// `0.1`, the R-matrix diagonal at floor 0.1 agreed across τ and τ/100 to 5.6e-5 while both were
/// 5e-4 from the converged value; at 1e-4 the gate accepted there and SE(β) came out 3.9e-4 from
/// the analytic one. At 1e-5 every cell of the declared × estimated table (1e-10 … 1) lands
/// within 4.4e-5 of it, the fixture's floor-1 level, and the gradient's error falls from 2.8e-7
/// to 5.8e-9. A `V = TVV/k` volume is still accepted at its first step on both sides, its two
/// differences agreeing to 3e-9 (#2406).
pub(crate) const RESOLVED_REL: f64 = 1e-5;
/// Factor the step floor grows by when a step is not resolved.
pub(crate) const STEP_GROWTH: f64 = 10.0;
/// The tight twin's inner tolerance, relative to the stencil's own.
pub(crate) const TIGHT_FACTOR: f64 = 1e-2;
/// The τ/100 difference's numerator must exceed `NOISE_GUARD·√ε·|f|`, the noise floor of a
/// re-solved objective, for its agreement with the τ one to count.
///
/// The inner solve resolves η only to about √ε, since its objective is flat to second order at
/// the mode, while the marginal reads η at first order through `log det H`. Below that floor a
/// solve can stop at the same iterate under τ and τ/100, `converged`, and the two differences
/// agree because neither measured anything. On a covariate exponent declared at 1e-4 the
/// per-subject gradient did exactly that on Linux x86_64, at `numer/(√ε·|f|)` = 0.032, and was
/// accepted 13% off; an ε-scale guard (`1e3·ε·|f|`) passed it. Measured ratios: a stall that
/// agreed at 0.032, a population difference 2.2e-5 off at 0.20 and 2e-7 off at 1.8, the
/// smallest accepted real difference at 14 (that per-subject gradient at floor 0.1), and a
/// `V = TVV/k` volume at 2e5. So 1 sits 31× above the stall and 14× below the smallest
/// accepted difference (#2406).
pub(crate) const NOISE_GUARD: f64 = 1.0;
/// How far the τ/100 difference may move from one floor to the next, as a multiple of the two
/// floors' resolution errors `|τ − τ/100|`, before the move counts as truncation error and the
/// smaller floor's difference is kept.
///
/// Growing the step helps only while the difference is limited by how well the probes
/// resolve it; past that, the step's own truncation error grows as `h²` and growth makes the
/// difference worse. Near a stationary point the gradient is too small for the relative test
/// or the noise guard ever to pass, so without this the step grew until truncation made the
/// difference large enough to pass them: on a `V = TVV/k` volume at its stationary point the
/// accepted difference was `k·g = −1.1e-2` where the first floor's was `−4.4e-6` (#2406
/// review r1). Measured ratios `|t_k − t_{k−1}| / (e_{k−1} + e_k)`: at most 0.15 on the
/// covariate exponents the gate exists for (declared at 1e-10 and 1e-4, population, reporting
/// and per-subject), 1.28 on the covariance diagonal, and 2.3e5 on the volume at its
/// stationary point. So 100 sits 78× above the largest resolution-limited move and 2300×
/// below the truncation-limited one.
pub(crate) const TRUNCATION_FACTOR: f64 = 100.0;

/// The tight twin's inner tolerance for a stencil that re-solves at `tol`.
pub(crate) fn tight_tol(tol: f64) -> f64 {
    tol * TIGHT_FACTOR
}

/// Whether a population re-solve counts as resolved for the tight twin: no subject fell to
/// Nelder–Mead, was polished out of it, or stopped unconverged.
pub(crate) fn inner_resolved(stats: &InnerLoopStats) -> bool {
    stats.n_unconverged == 0 && stats.n_fallback == 0 && stats.n_nm_polished == 0
}

/// [`inner_resolved`] for one subject's solve.
pub(crate) fn ebe_resolved(ebe: &EbeResult) -> bool {
    ebe.converged && !ebe.used_fallback && !ebe.nm_polished
}

/// One coordinate's difference at one step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Difference {
    /// The difference quotient: `(f₊ − f₋)/(x₊ − x₋)`, or `(f₊ − 2f₀ + f₋)/h²`.
    pub(crate) value: f64,
    /// `|numerator|` of that quotient, for the noise guard.
    pub(crate) numer: f64,
    /// Magnitude of the objective at the probes, for the noise guard.
    pub(crate) scale: f64,
    /// The probes' span in the coordinate's units. Growth stops once the bounds clamp holds it.
    pub(crate) span: f64,
}

/// Why [`resolve_floored_step`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The τ and τ/100 differences agreed above the noise guard.
    Accepted,
    /// The next floor's τ/100 difference moved by more than [`TRUNCATION_FACTOR`] times the
    /// two floors' resolution errors: growth had crossed into truncation error, so the
    /// smaller floor's difference is kept.
    Truncation,
    /// Growth ran out — the floor-1 cap, or the bounds clamp holding the span — without
    /// either test deciding.
    RanOut,
}

/// Where [`resolve_floored_step`] stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Resolved {
    /// The difference to use: the τ/100 one where it exists, else the τ one.
    pub(crate) value: f64,
    /// The step floor it was taken at.
    pub(crate) floor: f64,
    /// Why growth stopped there.
    pub(crate) stop: Stop,
}

/// Whether the τ difference `loose` and the τ/100 difference `tight` resolve the step.
fn accepts(loose: &Difference, tight: &Difference) -> bool {
    (loose.value - tight.value).abs() <= RESOLVED_REL * tight.value.abs()
        && tight.numer > NOISE_GUARD * f64::EPSILON.sqrt() * tight.scale
}

/// Grow one floored coordinate's step from `floor0` until its difference is resolved.
///
/// `diff(floor, tight)` is the coordinate's difference at step floor `floor`, its probes
/// re-solved at the stencil's tolerance τ (`tight = false`) or at τ/100 (`true`); `None` when
/// a probe is missing — unusable, or at τ/100 a solve that fell to Nelder–Mead or did not
/// converge. The τ/100 side is evaluated only when the τ side exists.
///
/// Stops at the first floor where either test decides, returning:
///
/// - **accepted:** the τ/100 difference there, when its two differences agree
///   ([`RESOLVED_REL`]) above the noise guard ([`NOISE_GUARD`]);
/// - **truncation:** the previous floor's τ/100 difference, when this floor's moved from it by
///   more than [`TRUNCATION_FACTOR`] times the two floors' resolution errors `|τ − τ/100|`. A
///   τ and τ/100 difference that are the same number measured nothing (a stall), so a floor
///   whose two are bit-identical has no resolution error to compare with, and never stops this
///   way.
///
/// Growth otherwise stops at floor 1 — the pre-#2187 step — or once the bounds clamp keeps the
/// span from growing. There it returns the last τ/100 difference measured at any floor, so a
/// probe that goes missing only at the larger floors (a volume stepped through zero) keeps what
/// the smaller ones resolved. A τ difference alone is no evidence about the step: with no τ/100
/// difference at any floor it returns `floor0`'s τ difference, the ungated value, and does not
/// move the step. On a `V = TVV/k` volume at `k = 1e-6` with the twin missing everywhere,
/// returning the cap's τ difference instead gave `k·g = −2.85e13` against an exact 13.3
/// (#2406 review r2). `None` only when `floor0`'s τ probe is missing and no floor gave a τ/100
/// difference (the caller's own fallback for a missing probe takes over, at `floor0`).
pub(crate) fn resolve_floored_step(
    floor0: f64,
    mut diff: impl FnMut(f64, bool) -> Option<Difference>,
) -> Option<Resolved> {
    let mut floor = floor0;
    let mut prev_span = 0.0_f64;
    // `floor0`'s τ difference, the last τ/100 difference measured at any floor, and the
    // previous floor's τ/100 difference with its resolution error and floor (`None` where that
    // floor had no τ/100 difference).
    let mut first_loose: Option<Resolved> = None;
    let mut last_tight: Option<Resolved> = None;
    let mut prev_tight: Option<(f64, f64, f64)> = None;
    loop {
        let mut clamped = false;
        if let Some(l) = diff(floor, false) {
            if floor == floor0 {
                first_loose = Some(Resolved {
                    value: l.value,
                    floor,
                    stop: Stop::RanOut,
                });
            }
            match diff(floor, true) {
                Some(t) => {
                    // Bit-identical: the τ/100 solve stopped where the τ one did, so their
                    // agreement says nothing about the error.
                    let err = if l.value.to_bits() == t.value.to_bits() {
                        f64::INFINITY
                    } else {
                        (l.value - t.value).abs()
                    };
                    if let Some((t_prev, err_prev, floor_prev)) = prev_tight {
                        if (t.value - t_prev).abs() > TRUNCATION_FACTOR * (err_prev + err) {
                            return Some(Resolved {
                                value: t_prev,
                                floor: floor_prev,
                                stop: Stop::Truncation,
                            });
                        }
                    }
                    if accepts(&l, &t) {
                        return Some(Resolved {
                            value: t.value,
                            floor,
                            stop: Stop::Accepted,
                        });
                    }
                    prev_tight = Some((t.value, err, floor));
                    last_tight = Some(Resolved {
                        value: t.value,
                        floor,
                        stop: Stop::RanOut,
                    });
                }
                // A missing twin keeps the earlier τ/100 difference: this floor's τ one is
                // no evidence that the step should move.
                None => prev_tight = None,
            }
            clamped = l.span <= prev_span;
            prev_span = l.span;
        } else {
            // The next floor's move is compared with this one's, which has nothing to offer.
            prev_tight = None;
        }
        if floor >= 1.0 || clamped {
            return last_tight.or(first_loose);
        }
        floor = (floor * STEP_GROWTH).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A difference the noise guard passes: the numerator is far above `NOISE_GUARD·√ε·scale`.
    fn diff(value: f64, span: f64) -> Difference {
        Difference {
            value,
            numer: 1.0,
            scale: 1.0,
            span,
        }
    }

    /// Runs the gate on `at(floor, tight)`, returning where it stopped and every
    /// `(floor, tight)` it asked for, in order.
    fn run(
        floor0: f64,
        at: impl Fn(f64, bool) -> Option<Difference>,
    ) -> (Option<Resolved>, Vec<(f64, bool)>) {
        let mut asked = Vec::new();
        let r = resolve_floored_step(floor0, |fl, tight| {
            assert!(fl <= 1.0, "asked for floor {fl:e}, past the floor-1 cap");
            asked.push((fl, tight));
            at(fl, tight)
        });
        (r, asked)
    }

    /// The τ floors it asked for, in order.
    fn loose_floors(asked: &[(f64, bool)]) -> Vec<f64> {
        asked.iter().filter(|a| !a.1).map(|a| a.0).collect()
    }

    /// The τ difference is off by `1e-7/floor`, the τ/100 one by a hundredth of that: a
    /// relative disagreement of 5e-4 at floor 1e-4, 5e-5 at 1e-3 and 5e-6 at 1e-2 — the first
    /// inside `RESOLVED_REL`. The value returned there is the τ/100 one, not the τ one.
    #[test]
    fn accepts_the_first_floor_whose_twin_agrees_and_returns_the_tight_value() {
        let err = |fl: f64| 1e-7 / fl;
        let (r, asked) = run(1e-4, |fl, tight| {
            Some(diff(
                2.0 + if tight { err(fl) / 100.0 } else { err(fl) },
                fl,
            ))
        });
        let r = r.expect("a resolved difference");
        assert_eq!(r.stop, Stop::Accepted);
        assert_eq!(r.floor, 1e-2);
        assert_eq!(r.value, 2.0 + err(1e-2) / 100.0);
        assert_eq!(loose_floors(&asked), [1e-4, 1e-3, 1e-2]);
        assert_eq!(asked.len(), 6, "τ and τ/100 at each floor: {asked:?}");
    }

    /// A disagreement of `2·RESOLVED_REL` at every floor runs to the cap and returns the
    /// τ/100 difference there; one of `RESOLVED_REL/2` accepts at once.
    #[test]
    fn a_disagreement_past_the_bound_grows_to_the_cap() {
        let (r, asked) = run(1e-3, |fl, tight| {
            Some(diff(if tight { 1.0 } else { 1.0 + 2.0 * RESOLVED_REL }, fl))
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::RanOut);
        assert_eq!(
            (r.floor, r.value),
            (1.0, 1.0),
            "the cap returns the τ/100 difference"
        );
        assert_eq!(loose_floors(&asked), [1e-3, 1e-2, 1e-1, 1.0]);

        let (r, _) = run(1e-3, |fl, tight| {
            Some(diff(if tight { 1.0 } else { 1.0 + 0.5 * RESOLVED_REL }, fl))
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::Accepted);
        assert_eq!(r.floor, 1e-3);
    }

    /// Two differences whose numerator is below the re-solved objective's noise floor agree
    /// and resolve nothing — a stalled inner solve returns the same objective at τ and τ/100
    /// (the per-subject leg of #2406 on Linux) — so the step grows to the first floor whose
    /// numerator clears `NOISE_GUARD·√ε·|f|`. The stalled numerator is far above an ε-scale
    /// guard, so the guard has to be √ε-scale to reject it.
    #[test]
    fn the_noise_guard_rejects_an_agreement_below_the_resolve_noise() {
        let scale = 1e4;
        let noise = NOISE_GUARD * f64::EPSILON.sqrt() * scale;
        let stalled = 0.5 * noise;
        assert!(stalled > 1e6 * f64::EPSILON * scale);
        let (r, asked) = run(1e-4, |fl, _| {
            Some(Difference {
                value: 3.0,
                numer: if fl < 5e-3 { stalled } else { 2.0 * noise },
                scale,
                span: fl,
            })
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::Accepted);
        assert_eq!(r.floor, 1e-2);
        assert_eq!(loose_floors(&asked), [1e-4, 1e-3, 1e-2]);
    }

    /// The cap is floor 1 exactly, also from a floor that is not a power of ten.
    #[test]
    fn growth_is_capped_at_floor_one() {
        let (r, asked) = run(3e-3, |fl, tight| {
            Some(diff(if tight { 1.0 } else { 2.0 }, fl))
        });
        assert_eq!(r.unwrap().floor, 1.0);
        assert_eq!(loose_floors(&asked), [3e-3, 3e-2, 3e-1, 1.0]);
    }

    /// A τ/100 solve that is unresolved — Nelder–Mead, or not converged — is not an
    /// agreement: the step grows past it. With the twin unresolved at every floor the gate
    /// learned nothing, so it returns `floor0`'s τ difference, the ungated value, not the cap's
    /// (#2406 review r2). The τ difference moves with the floor, so the two are told apart.
    #[test]
    fn an_unresolved_twin_grows_the_step() {
        let (r, _) = run(1e-3, |fl, tight| {
            (!tight || fl > 5e-3).then(|| diff(5.0, fl))
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::Accepted);
        assert_eq!(r.floor, 1e-2);

        let (r, asked) = run(1e-3, |fl, tight| (!tight).then(|| diff(5.0 + fl, fl)));
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::RanOut);
        assert_eq!(
            (r.floor, r.value),
            (1e-3, 5.0 + 1e-3),
            "no twin anywhere keeps floor 0's τ difference"
        );
        assert_eq!(loose_floors(&asked), [1e-3, 1e-2, 1e-1, 1.0]);
    }

    /// #2406 review r2: a twin that goes missing at the larger floors, while their τ probes
    /// still exist, does not replace the τ/100 difference a smaller floor measured with a
    /// larger floor's τ one. Without the rule the cap's τ difference, 3.0, came back.
    #[test]
    fn a_twin_missing_at_the_larger_floors_keeps_the_earlier_tight_difference() {
        let (r, asked) = run(1e-3, |fl, tight| {
            if tight {
                (fl < 5e-2).then(|| diff(2.0, fl))
            } else {
                Some(diff(if fl < 5e-2 { 2.5 } else { 3.0 }, fl))
            }
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::RanOut);
        assert_eq!(
            (r.floor, r.value),
            (1e-2, 2.0),
            "the last τ/100 difference measured"
        );
        assert_eq!(loose_floors(&asked), [1e-3, 1e-2, 1e-1, 1.0]);
    }

    /// With `floor0`'s τ probe missing and no twin anywhere, a larger floor's τ difference is
    /// still no evidence: the gate returns `None`, and the caller's own missing-probe fallback
    /// runs at `floor0`, as on the ungated path.
    #[test]
    fn no_floor0_probe_and_no_twin_returns_none() {
        let (r, asked) = run(1e-3, |fl, tight| {
            (!tight && fl > 5e-3).then(|| diff(5.0, fl))
        });
        assert_eq!(r, None);
        assert_eq!(loose_floors(&asked), [1e-3, 1e-2, 1e-1, 1.0]);
    }

    /// Once the bounds clamp holds the probes' span, a larger floor is the same step:
    /// growth stops at the first floor that did not widen it, instead of re-solving up to
    /// the cap.
    #[test]
    fn growth_stops_once_the_bounds_clamp_holds_the_span() {
        let (r, asked) = run(1e-3, |fl, tight| {
            Some(diff(if tight { 1.0 } else { 2.0 }, fl.min(5e-3)))
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::RanOut);
        assert_eq!(r.floor, 1e-1);
        assert_eq!(loose_floors(&asked), [1e-3, 1e-2, 1e-1]);
    }

    /// A population re-solve is resolved only with no unconverged, fallback or NM-polished
    /// subject; each count alone makes it unresolved, so dropping any one test of the three
    /// lets a twin that fell to Nelder–Mead accept a step. A start-rejected subject is the
    /// outer guard's business, not this gate's.
    #[test]
    fn a_population_resolve_with_any_unresolved_subject_is_unresolved() {
        let clean = InnerLoopStats::default();
        assert!(inner_resolved(&clean));
        assert!(inner_resolved(&InnerLoopStats {
            n_start_rejected: 1,
            ..clean.clone()
        }));
        for (name, stats) in [
            (
                "unconverged",
                InnerLoopStats {
                    n_unconverged: 1,
                    ..clean.clone()
                },
            ),
            (
                "fallback",
                InnerLoopStats {
                    n_fallback: 1,
                    ..clean.clone()
                },
            ),
            (
                "nm_polished",
                InnerLoopStats {
                    n_nm_polished: 1,
                    ..clean.clone()
                },
            ),
        ] {
            assert!(!inner_resolved(&stats), "{name} counted as resolved");
        }
    }

    /// [`ebe_resolved`]: converged, not a fallback's point, not NM-polished — each flag alone
    /// makes one subject's solve unresolved.
    #[test]
    fn a_subject_solve_is_resolved_only_when_converged_without_nelder_mead() {
        use nalgebra::{DMatrix, DVector};
        let ebe = |converged, used_fallback, nm_polished| EbeResult {
            eta: DVector::zeros(1),
            h_matrix: DMatrix::zeros(1, 1),
            fd_h_redone: false,
            terminal_hessian: None,
            converged,
            used_fallback,
            nm_polished,
            grad_norm: 0.0,
            nll: 1.0,
            kappas: Vec::new(),
            hard_reject: false,
            screen_moved: false,
            twin_moved: false,
            ipreds: None,
        };
        assert!(ebe_resolved(&ebe(true, false, false)));
        assert!(!ebe_resolved(&ebe(false, false, false)), "unconverged");
        assert!(!ebe_resolved(&ebe(true, true, false)), "fallback");
        assert!(!ebe_resolved(&ebe(true, false, true)), "nm_polished");
    }

    /// A difference below the noise guard, as near a stationary point: neither test can
    /// accept it, so only the truncation test can stop growth.
    fn quiet(value: f64, span: f64) -> Difference {
        Difference {
            value,
            numer: 1e-12,
            scale: 1.0,
            span,
        }
    }

    /// #2406 review r1: growth into truncation error. Near a stationary point the relative
    /// test and the noise guard cannot pass, and the τ/100 difference then moves by ×100 per
    /// floor (truncation, `∝ h²`) while τ and τ/100 agree to 1e-6: the move is 5e7 resolution
    /// errors, so the gate keeps floor 0's difference after measuring floor 1. The twin is the
    /// resolution-limited move — the same disagreement, the value moving by at most 9e-6 per
    /// floor, under `TRUNCATION_FACTOR` times the two errors (2e-4) — which grows on to the
    /// cap: the straddle, so the test cannot pass on a gate that stops everywhere.
    #[test]
    fn growth_into_truncation_keeps_the_smaller_floors_difference() {
        let at = |t: fn(f64) -> f64| {
            move |fl: f64, tight: bool| Some(quiet(if tight { t(fl) } else { t(fl) + 1e-6 }, fl))
        };
        let (r, asked) = run(1e-4, at(|fl| 3.0 + (fl / 1e-4).powi(2)));
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::Truncation);
        assert_eq!(
            (r.floor, r.value),
            (1e-4, 4.0),
            "floor 0's τ/100 difference"
        );
        assert_eq!(loose_floors(&asked), [1e-4, 1e-3]);

        let (r, asked) = run(1e-4, at(|fl| 3.0 + 1e-5 * fl));
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::RanOut, "a resolution-limited move grows on");
        assert_eq!(loose_floors(&asked), [1e-4, 1e-3, 1e-2, 1e-1, 1.0]);
    }

    /// A floor whose τ and τ/100 differences are the same number measured nothing — the stall
    /// the noise guard exists for — so it has no resolution error to set against the next
    /// floor's move, which here is large: growth goes on, and the next floor's agreement
    /// accepts. Without the bit-identity test the stall's zero error made any move a
    /// truncation, and the stalled difference came back.
    #[test]
    fn a_stalled_floor_never_stops_growth_as_truncation() {
        let (r, asked) = run(1e-4, |fl, tight| {
            Some(if fl < 5e-4 {
                quiet(0.19, fl)
            } else {
                diff(if tight { 0.167 } else { 0.167 * (1.0 + 1e-7) }, fl)
            })
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::Accepted);
        assert_eq!((r.floor, r.value), (1e-3, 0.167));
        assert_eq!(loose_floors(&asked), [1e-4, 1e-3]);
    }

    /// The truncation test compares neighbouring floors only. A floor with no τ/100 difference,
    /// or no τ one, leaves nothing to compare with, so the next floor's move is not measured
    /// against an older floor two steps of growth away: here floor 1e-2 differs from floor
    /// 1e-4 by 1e5 resolution errors but agrees with itself, and is accepted. Both ways the
    /// middle floor can go missing, each of which compared across it returns floor 1e-4's
    /// difference as truncation.
    #[test]
    fn a_floor_with_a_missing_probe_starts_the_truncation_test_afresh() {
        for missing_tight in [true, false] {
            let (r, _) = run(1e-4, |fl, tight| {
                if (fl - 1e-3).abs() < 1e-12 && (tight == missing_tight || !missing_tight) {
                    return None;
                }
                Some(if fl < 5e-4 {
                    quiet(if tight { 1.0 } else { 1.0 + 1e-6 }, fl)
                } else {
                    diff(if tight { 1.1 } else { 1.1 * (1.0 + 1e-7) }, fl)
                })
            });
            let r = r.unwrap();
            assert_eq!(
                (r.stop, r.floor),
                (Stop::Accepted, 1e-2),
                "middle floor missing its {} probe",
                if missing_tight { "τ/100" } else { "τ" }
            );
        }
    }

    /// #2406 review r1, row 2: a probe that goes missing only at the larger floors — a
    /// `V = TVV/k` volume stepped through `V ≤ 0` — keeps the last difference the smaller
    /// floors measured, its floor with it, instead of returning nothing for the caller to
    /// replace with a one-sided difference at the cap.
    #[test]
    fn a_probe_missing_at_the_larger_floors_keeps_the_last_measured_difference() {
        let (r, asked) = run(1e-3, |fl, tight| {
            (fl < 5e-2).then(|| diff(if tight { 2.0 } else { 2.5 }, fl))
        });
        let r = r.unwrap();
        assert_eq!(r.stop, Stop::RanOut);
        assert_eq!(
            (r.floor, r.value),
            (1e-2, 2.0),
            "the last τ/100 difference measured"
        );
        assert_eq!(loose_floors(&asked), [1e-3, 1e-2, 1e-1, 1.0]);
    }

    /// With the τ probe missing at every floor there is nothing to return, and the caller's
    /// own missing-probe fallback takes over; the τ/100 twin is never asked for.
    #[test]
    fn a_missing_loose_probe_returns_none_without_the_twin() {
        let (r, asked) = run(1e-2, |_, _| None);
        assert_eq!(r, None);
        assert_eq!(asked, [(1e-2, false), (1e-1, false), (1.0, false)]);
    }
}
