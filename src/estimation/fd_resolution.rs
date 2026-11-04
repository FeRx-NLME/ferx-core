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
/// The difference's numerator must exceed `ULP_GUARD·ε·|f|`: below it the τ and τ/100
/// differences can agree because both are the same rounding of `f`, not because either
/// resolved anything.
pub(crate) const ULP_GUARD: f64 = 1e3;

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
    /// `|numerator|` of that quotient, for the ulp guard.
    pub(crate) numer: f64,
    /// Magnitude of the objective at the probes, for the ulp guard.
    pub(crate) scale: f64,
    /// The probes' span in the coordinate's units. Growth stops once the bounds clamp holds it.
    pub(crate) span: f64,
}

/// Where [`resolve_floored_step`] stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Resolved {
    /// The difference to use: the τ/100 one where it exists, else the τ one.
    pub(crate) value: f64,
    /// The step floor it was taken at.
    pub(crate) floor: f64,
    /// Whether the accept test passed there, rather than growth running out.
    pub(crate) accepted: bool,
}

/// Whether the τ difference `loose` and the τ/100 difference `tight` resolve the step.
fn accepts(loose: &Difference, tight: &Difference) -> bool {
    (loose.value - tight.value).abs() <= RESOLVED_REL * tight.value.abs()
        && tight.numer > ULP_GUARD * f64::EPSILON * tight.scale
}

/// Grow one floored coordinate's step from `floor0` until its difference is resolved.
///
/// `diff(floor, tight)` is the coordinate's difference at step floor `floor`, its probes
/// re-solved at the stencil's tolerance τ (`tight = false`) or at τ/100 (`true`); `None` when
/// a probe is missing — unusable, or at τ/100 a solve that fell to Nelder–Mead or did not
/// converge. The τ/100 side is evaluated only when the τ side exists.
///
/// Returns the τ/100 difference at the first floor whose two differences agree
/// ([`RESOLVED_REL`]) above the ulp guard ([`ULP_GUARD`]). Growth stops at floor 1 — the
/// pre-#2187 step — or once the bounds clamp keeps the span from growing; there the τ/100
/// difference is returned if it exists, else the τ one, else `None` (the caller's own
/// fallback for a missing probe takes over).
pub(crate) fn resolve_floored_step(
    floor0: f64,
    mut diff: impl FnMut(f64, bool) -> Option<Difference>,
) -> Option<Resolved> {
    let mut floor = floor0;
    let mut prev_span = 0.0_f64;
    loop {
        let mut last = None;
        let mut clamped = false;
        if let Some(l) = diff(floor, false) {
            last = Some(match diff(floor, true) {
                Some(t) if accepts(&l, &t) => {
                    return Some(Resolved {
                        value: t.value,
                        floor,
                        accepted: true,
                    })
                }
                Some(t) => t.value,
                None => l.value,
            });
            clamped = l.span <= prev_span;
            prev_span = l.span;
        }
        if floor >= 1.0 || clamped {
            return last.map(|value| Resolved {
                value,
                floor,
                accepted: false,
            });
        }
        floor = (floor * STEP_GROWTH).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A difference the ulp guard passes: the numerator is far above `ULP_GUARD·ε·scale`.
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
        assert!(r.accepted);
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
        assert!(!r.accepted);
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
        assert!(r.accepted);
        assert_eq!(r.floor, 1e-3);
    }

    /// Two differences that are the same rounding of `f` agree exactly and resolve nothing:
    /// the ulp guard rejects them (the θ₀ = 1e-10 cell of #2406), and the step grows to the
    /// first floor whose numerator clears `ULP_GUARD·ε·|f|`.
    #[test]
    fn the_ulp_guard_rejects_an_agreement_of_two_roundings() {
        let scale = 1e4;
        let (r, _) = run(1e-4, |fl, _| {
            let numer = if fl < 5e-3 {
                10.0 * f64::EPSILON * scale
            } else {
                1e-6
            };
            Some(Difference {
                value: 3.0,
                numer,
                scale,
                span: fl,
            })
        });
        let r = r.unwrap();
        assert!(r.accepted);
        assert_eq!(r.floor, 1e-2);
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
    /// agreement: the step grows past it, and at the cap the τ difference is returned.
    #[test]
    fn an_unresolved_twin_grows_the_step() {
        let (r, _) = run(1e-3, |fl, tight| {
            (!tight || fl > 5e-3).then(|| diff(5.0, fl))
        });
        let r = r.unwrap();
        assert!(r.accepted);
        assert_eq!(r.floor, 1e-2);

        let (r, _) = run(1e-3, |fl, tight| (!tight).then(|| diff(5.0, fl)));
        let r = r.unwrap();
        assert!(!r.accepted);
        assert_eq!(
            (r.floor, r.value),
            (1.0, 5.0),
            "the cap falls back to the τ difference"
        );
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
        assert!(!r.accepted);
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
        };
        assert!(ebe_resolved(&ebe(true, false, false)));
        assert!(!ebe_resolved(&ebe(false, false, false)), "unconverged");
        assert!(!ebe_resolved(&ebe(true, true, false)), "fallback");
        assert!(!ebe_resolved(&ebe(true, false, true)), "nm_polished");
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
