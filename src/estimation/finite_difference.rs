//! Bounded, noise-aware finite differences used by the outer and inner optimizers.
//!
//! This module deliberately operates in the caller's coordinates.  It knows
//! nothing about model parameters or penalties: an evaluator must report an
//! unusable objective value as `None`.
//!
//! There is **one** interval search ([`shi_central_search`]); the scalar and
//! vector entry points are thin wrappers over it, so the representability guard,
//! the non-finite policy and the shrink-on-rejected-probe policy cannot drift
//! between them.

use crate::types::OuterFdMethod;

/// Maximum halvings/doublings of the trial interval.
const MAX_SEARCH_ITERS: usize = 15;

/// Acceptance window for the scaled third-difference ratio
/// `r(h) = |Δ³f(h)| / (8·noise)`.
///
/// **The width is not a taste parameter.** `Δ³f(h) ≈ 8·h³·f'''(x)`, so `r ∝ h³`
/// and one doubling (or halving) of `h` moves `r` by a factor of **8**.  The
/// reachable ratios from any start therefore form the geometric set
/// `{r₀·8ᵏ}`, and a window narrower than a factor of 8 can sit entirely between
/// two consecutive members — the search then ping-pongs across it until the
/// iteration budget runs out and returns whatever the last interval gave.
///
/// Measured: the first version of this module used `[1.5, 6.0]` (a factor of 4).
/// On this module's own `shi_recovers_cubic_derivative` fixture — `f = x³`,
/// `x = 2`, `noise = 1e-8`, so `Δ³f = 48h³` and `r(h) = 6e8·h³` — that window is
/// straddled by `r(0.00125) = 1.1719` (below `1.5`, so double) and
/// `r(0.0025) = 9.375` (above `6.0`, so halve).  The search alternated between
/// those two `h` for all 15 iterations, 60 evaluations, and never accepted; the
/// test passed only because a central difference of a cubic is exact.
///
/// `[1.0, 8.0]` is a closed factor-8 window, so some `r₀·8ᵏ` always lands in it:
/// `RATIO_LOW ≤ r₀·8ᵏ ≤ RATIO_HIGH` has an integer solution for every `r₀ > 0`.
/// Acceptance can still be denied by the *bounds* (see [`FdEstimate::accepted`]),
/// never by the window's width.
const RATIO_LOW: f64 = 1.0;
const RATIO_HIGH: f64 = 8.0;

/// Acceptance window for Gill's one-sided second-difference ratio
/// `|f(x+2h) - 2f(x+h) + f(x)| / (4·noise)`.
///
/// The `4` matters and matches Shi's `8` above: it is the **worst-case noise
/// contribution** to the difference (`|1| + |−2| + |1| = 4` times the bound), so
/// `ratio ≥ 1` reads as "the curvature signal is at least everything the noise
/// could have contributed". Normalising by a bare `noise` instead — as the first
/// version did — makes `ratio ≈ 1` the *default* reading on noise-dominated
/// data, since a pure-noise second difference is already of order `noise`: the
/// search then accepts almost immediately and the curvature it hands to
/// `h = 2·√(noise/|f''|)` is a measurement of the noise. Measured on the
/// `gill_searches_for_a_resolvable_curvature` fixture (`f = x² + ripple(1e-8)`,
/// true `f'' = 2`): the un-normalised test accepted at `h = 4e-6` and reported
/// `f'' = 1727`.
///
/// A second difference scales as `h²`, so a doubling moves the ratio by a factor
/// of **4** and this is a closed factor-4 window, by the same reachability
/// argument as [`RATIO_LOW`].
const GILL_RATIO_LOW: f64 = 1.0;
const GILL_RATIO_HIGH: f64 = 4.0;
/// Worst-case noise contribution to a one-sided second difference (see
/// [`GILL_RATIO_LOW`]).
const GILL_NOISE_WEIGHT: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AxisBounds {
    pub lower: f64,
    pub upper: f64,
}

impl AxisBounds {
    /// Widest `h` for which the whole `±3h` centred stencil stays inside the box.
    fn central_cap(self, x: f64) -> f64 {
        ((x - self.lower).min(self.upper - x) / 3.0).max(0.0)
    }

    /// Widest `h` for which the one-sided `x, x+2h` stencil stays inside the box,
    /// together with the direction (`+1` / `-1`) that has the room for it.
    ///
    /// Picking the roomier side is what lets a coordinate sitting **at** its
    /// upper bound be differentiated at all: a forward-only rule has
    /// `upper - x == 0` there and can only return "no interval".
    fn one_sided_cap(self, x: f64) -> (f64, f64) {
        let up = (self.upper - x).max(0.0);
        let down = (x - self.lower).max(0.0);
        if up >= down {
            (up / 2.0, 1.0)
        } else {
            (down / 2.0, -1.0)
        }
    }
}

/// A finite-difference estimate together with the interval that produced it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FdEstimate {
    /// One entry per component of the evaluator's output (length 1 for scalars).
    pub derivative: Vec<f64>,
    /// The interval the returned derivative was computed on.
    pub h: f64,
    /// `true` only when the ratio test *accepted* `h`.
    ///
    /// `false` means the search ran out of room — the bounds capped the growth,
    /// the interval stopped being representable, or the iteration budget was
    /// spent — and the caller is looking at the best interval that was tried,
    /// not at one the noise/curvature balance endorsed.  A near-quadratic
    /// coordinate (`f''' ≈ 0`) always ends here: its ratio never reaches
    /// [`RATIO_LOW`] however wide `h` grows.
    pub accepted: bool,
}

/// Select a first-derivative interval and evaluate its derivative.
///
/// `noise` is an absolute bound in the evaluator's output units.  Adaptive
/// methods return `None` if no complete, representable stencil fits inside the
/// supplied bounds — the caller must then fall back to its own fixed stencil
/// rather than record a zero derivative.  `Fixed` always returns `None`: the
/// historical centred/clamped convention lives in the caller.
pub(crate) fn adaptive_first_derivative(
    method: OuterFdMethod,
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    eval: impl FnMut(f64) -> Option<f64>,
) -> Option<FdEstimate> {
    match method {
        OuterFdMethod::Fixed => None,
        OuterFdMethod::Shi => shi_first_derivative(x, initial_h, bounds, noise, eval),
        OuterFdMethod::Gill => gill_first_derivative(x, initial_h, bounds, noise, eval),
    }
}

/// Shi's centred interval search on a scalar evaluator.
pub(crate) fn shi_first_derivative(
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<FdEstimate> {
    shi_central_search(x, initial_h, bounds, noise, |probe| {
        eval(probe).map(|value| vec![value])
    })
}

/// Shi's centred search applied to one column of a vector-valued prediction.
/// A single interval is accepted only when every component's ratio is in range;
/// the largest component ratio drives the conservative search.
pub(crate) fn shi_central_vector_derivative(
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    eval: impl FnMut(f64) -> Option<Vec<f64>>,
) -> Option<FdEstimate> {
    shi_central_search(x, initial_h, bounds, noise, eval)
}

fn usable_h(h: f64) -> bool {
    h.is_finite() && h > 0.0
}

/// `x ± 3h` must be distinguishable from `x` in f64, or the stencil differences
/// nothing.
fn representable(x: f64, h: f64) -> bool {
    x + 3.0 * h != x && x - 3.0 * h != x
}

/// The single interval search behind every Shi entry point in this module.
fn shi_central_search(
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    mut eval: impl FnMut(f64) -> Option<Vec<f64>>,
) -> Option<FdEstimate> {
    if !usable_h(noise) {
        return None;
    }
    let cap = bounds.central_cap(x);
    let mut h = initial_h.min(cap);
    let mut best: Option<FdEstimate> = None;
    for _ in 0..MAX_SEARCH_ITERS {
        if !usable_h(h) || h > cap || !representable(x, h) {
            break;
        }
        // An unusable probe (a guard-rejected trial point, a non-finite value)
        // means this interval reaches somewhere the evaluator cannot score. Half
        // the interval and retry: abandoning the coordinate here is what makes a
        // caller record a fabricated zero derivative.
        let (Some(fp), Some(fm), Some(fp3), Some(fm3)) = (
            eval(x + h),
            eval(x - h),
            eval(x + 3.0 * h),
            eval(x - 3.0 * h),
        ) else {
            h /= 2.0;
            continue;
        };
        if fp.len() != fm.len() || fp.len() != fp3.len() || fp.len() != fm3.len() || fp.is_empty() {
            return None;
        }
        let mut derivative = Vec::with_capacity(fp.len());
        let mut ratio = 0.0_f64;
        let mut degenerate = false;
        for i in 0..fp.len() {
            let d = (fp[i] - fm[i]) / (2.0 * h);
            let r = (fp3[i] - 3.0 * fp[i] + 3.0 * fm[i] - fm3[i]).abs() / (8.0 * noise);
            if !d.is_finite() || !r.is_finite() {
                degenerate = true;
                break;
            }
            derivative.push(d);
            ratio = ratio.max(r);
        }
        if degenerate {
            h /= 2.0;
            continue;
        }
        let accepted = (RATIO_LOW..=RATIO_HIGH).contains(&ratio);
        best = Some(FdEstimate {
            derivative,
            h,
            accepted,
        });
        if accepted {
            return best;
        }
        let next = if ratio < RATIO_LOW { 2.0 * h } else { h / 2.0 };
        // Growing past the box is not an acceptance: the ratio test never
        // endorsed this interval, the bounds simply stopped the search. Leaving
        // `accepted = false` on `best` is what says so.
        if next > cap || next == h || !usable_h(next) {
            break;
        }
        h = next;
    }
    best
}

/// A bounded one-sided implementation of the elementary Gill et al.
/// curvature/noise balance.  It is intentionally separate from Shi: its chosen
/// interval is used only with a one-sided stencil, never re-used as a central
/// interval.  The full Gill classification search remains future work.
///
/// The curvature that sets `h = 2·√(noise/|f''|)` is **searched, not sampled
/// once**: in the noise-dominated regime this method targets, a single second
/// difference taken at `h₀` is mostly noise, and the resulting `h` is wrong by
/// the same factor the estimate is — too small and the derivative is noise, too
/// large and it is a chord across the box.  The loop below moves `h` until the
/// second difference is resolvable against `noise` ([`GILL_RATIO_LOW`]) before
/// the curvature it implies is used.
fn gill_first_derivative(
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<FdEstimate> {
    if !usable_h(noise) {
        return None;
    }
    let (cap, dir) = bounds.one_sided_cap(x);
    let mut h = initial_h.min(cap);
    let f0 = eval(x)?;
    if !f0.is_finite() {
        return None;
    }
    let mut curvature = None;
    for _ in 0..MAX_SEARCH_ITERS {
        if !usable_h(h) || h > cap || x + 2.0 * dir * h == x {
            break;
        }
        let (Some(f1), Some(f2)) = (eval(x + dir * h), eval(x + 2.0 * dir * h)) else {
            h /= 2.0;
            continue;
        };
        let second = f2 - 2.0 * f1 + f0;
        if !second.is_finite() {
            h /= 2.0;
            continue;
        }
        let ratio = second.abs() / (GILL_NOISE_WEIGHT * noise);
        let c = (second / (h * h)).abs();
        if c.is_finite() && c > 0.0 {
            curvature = Some(c);
        }
        if (GILL_RATIO_LOW..=GILL_RATIO_HIGH).contains(&ratio) {
            break;
        }
        let next = if ratio < GILL_RATIO_LOW {
            2.0 * h
        } else {
            h / 2.0
        };
        if next > cap || next == h || !usable_h(next) {
            break;
        }
        h = next;
    }
    // Balance truncation (`½·h·|f''|`) against noise (`2·noise/h`); the minimiser
    // is `h = 2·√(noise/|f''|)`. With no resolvable curvature the search's last
    // interval is the best information available.
    let step = match curvature {
        Some(c) => (2.0 * (noise / c).sqrt()).min(cap),
        None => h,
    };
    if !usable_h(step) || x + dir * step == x {
        return None;
    }
    let fh = eval(x + dir * step)?;
    let derivative = (fh - f0) / (dir * step);
    derivative.is_finite().then(|| FdEstimate {
        derivative: vec![derivative],
        h: step,
        accepted: curvature.is_some(),
    })
}

#[cfg(test)]
#[path = "finite_difference_tests.rs"]
mod tests;
