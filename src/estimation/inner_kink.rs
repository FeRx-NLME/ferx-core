//! Inner searches on a **non-smooth** individual objective (#3159).
//!
//! A prediction clipped at a threshold (`y = if (x − TH < 0) 0 else x − TH`) puts a kink in the
//! individual objective wherever a record crosses it. Two things go wrong there, and each has
//! its own remedy here, both behind `[fit_options] inner_kick` (a smooth ODE objective can
//! stall at a large gradient too, where the escape's moves made a BOBYQA fit non-smooth):
//!
//! * **A stall at a kink** ([`stall_escape`]). BFGS arrives on the kink's valley floor, where the
//!   gradient is large and points *across* the valley, every step along it raises the objective
//!   by more than the descent along the valley gains, and the objective-stall stop certifies the
//!   point at ‖g‖ ≈ 10. A large gradient is not what is wrong: the *true* minimum of such an
//!   objective is itself a kink minimum, certified at a large gradient. The right test is
//!   Clarke stationarity, `0 ∈ conv{∇f near x}` (gradient sampling, Burke–Lewis–Overton): when it
//!   holds the point is a minimum; when it does not, the hull's minimum-norm element is a descent
//!   direction along the valley.
//! * **A second mode** ([`kick_restarts`]). A clipped record that crosses the threshold
//!   between two points makes a second local minimum a barrier away from the first, close to it
//!   in distance (0.01) and not reachable by any descent from the incumbent. A descent from a
//!   small kick of each coordinate reaches it.

/// The gradient norm above which a stall is examined. The objective-stall stop's noise floor on
/// a smooth ODE objective is at most about 6e-3 (measured on a 40-subject fit); a kink's is
/// 5 and up, so nothing below this, which is every smooth objective, pays for a check.
pub(super) const STALL_GNORM: f64 = 1.0;
/// The radius of the gradient samples around a stalled point.
const SAMPLE_RADIUS: f64 = 1e-5;
/// The most escapes one inner solve attempts, so a pathological objective cannot spin.
pub(super) const MAX_STALL_ESCAPES: usize = 25;
/// Doublings of the first step along the descent direction: the longest step is
/// `SAMPLE_RADIUS · 2^17` ≈ 1.3 in η, a few prior SDs, so an escape cannot adopt a far-off point
/// that is not a local descent (#3159 review).
const MAX_DOUBLINGS: usize = 17;
/// A hull element this small contains zero for this purpose: the point is stationary.
const HULL_STATIONARY: f64 = 1e-6;
/// The kick of a restart, in prior standard deviations. 0.25 is the smallest radius that found
/// every second mode on the benchmark's clipped model; 0.5 and 1.0 found the same ones.
const KICK_FRAC: f64 = 0.25;
/// A coordinate whose prior SD is below this carries no information (a `0 FIX` placeholder).
const KICK_MIN_SD: f64 = 1e-3;
/// The most coordinates one subject's kicks visit (two descents each), the widest priors first,
/// so a model with many occasions cannot make a solve cost a descent per coordinate pair.
const MAX_KICK_COORDS: usize = 16;
/// The iterations of one kick's descent. A descent that reaches a lower mode does so within a
/// handful (12 found every second mode on the fixture; 6 missed one by 2e-4, 3 missed two),
/// and one that has not got there by then is not worth finishing.
pub(super) const KICK_MAX_ITER: usize = 12;

/// The minimum-norm point of the convex hull of `gs` (Frank–Wolfe with an exact line search).
pub(super) fn min_norm_hull(gs: &[Vec<f64>]) -> Vec<f64> {
    let n = gs[0].len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut v = gs[0].clone();
    for _ in 0..500 {
        let (mut best, mut best_dot) = (0, f64::INFINITY);
        for (i, g) in gs.iter().enumerate() {
            let d = dot(g, &v);
            if d < best_dot {
                (best, best_dot) = (i, d);
            }
        }
        let vv = dot(&v, &v);
        if best_dot >= vv - 1e-14 * (1.0 + vv) {
            break;
        }
        let diff: Vec<f64> = (0..n).map(|k| gs[best][k] - v[k]).collect();
        let dd = dot(&diff, &diff);
        if dd <= 0.0 {
            break;
        }
        let t = (-dot(&v, &diff) / dd).clamp(0.0, 1.0);
        for k in 0..n {
            v[k] += t * diff[k];
        }
    }
    v
}

/// Where a stall escape moved the search: the new point, its objective and its gradient.
pub(super) struct StallEscape {
    pub(super) x: Vec<f64>,
    pub(super) f: f64,
    pub(super) g: Vec<f64>,
}

/// The search stalled at `x` (objective `f0`), stepping from a point whose raw gradient was
/// `g_prev`. `None` when the gradient at `x` is below [`STALL_GNORM`], when `x` is stationary in
/// Clarke's sense (it is certified as before) or no lower point was found (likewise); otherwise
/// a lower point reached along the descent direction of the gradients sampled `±SAMPLE_RADIUS`
/// along each axis. The caller restarts its metric from it.
///
/// `‖g_prev‖ < STALL_GNORM` declines without a gradient evaluation, so a stall on a smooth
/// objective stays free; the gate itself is the stalled point's own raw ‖∇f(x)‖, the quantity
/// `STALL_GNORM` was calibrated on (#3159 review).
pub(super) fn stall_escape(
    obj: &dyn Fn(&[f64]) -> f64, grad: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], f0: f64,
    g_prev: &[f64],
) -> Option<StallEscape> {
    let l2 = |v: &[f64]| v.iter().map(|c| c * c).sum::<f64>().sqrt();
    if !(l2(g_prev) >= STALL_GNORM) {
        return None;
    }
    let g0 = grad(x);
    if !(l2(&g0) >= STALL_GNORM) {
        return None;
    }
    let n = x.len();
    let mut gs = vec![g0];
    for k in 0..n {
        for sgn in [1.0, -1.0] {
            let mut p = x.to_vec();
            p[k] += sgn * SAMPLE_RADIUS;
            gs.push(grad(&p));
        }
    }
    if gs.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let hull = min_norm_hull(&gs);
    let norm = l2(&hull);
    if norm < HULL_STATIONARY {
        return None;
    }
    let lower = |f: f64| f < f0 - 1e-9 * (1.0 + f0.abs());
    let at = |t: f64| -> Vec<f64> { (0..n).map(|k| x[k] - t * hull[k]).collect() };
    let (mut best_t, mut best_f) = (0.0, f0);
    let mut t = SAMPLE_RADIUS / norm;
    for _ in 0..MAX_DOUBLINGS {
        let f = obj(&at(t));
        if !f.is_finite() {
            break;
        }
        if lower(f) && f < best_f {
            (best_t, best_f) = (t, f);
        } else if best_t > 0.0 {
            break;
        }
        t *= 2.0;
    }
    (best_t > 0.0).then(|| {
        let p = at(best_t);
        let g = grad(&p);
        StallEscape { x: p, f: best_f, g }
    })
}

/// Restarts from a kick of each informative coordinate of the incumbent `x` (objective `nll`)
/// by `±KICK_FRAC` prior SDs, `probe`ing each with a short, capped descent (cheap: this runs
/// up to `2 · MAX_KICK_COORDS` times) and walking on from whichever drops the objective by more
/// than `1e-9`. At most [`MAX_KICK_COORDS`] coordinates are visited, the widest priors first.
///
/// `None` when no kick ever dropped the objective — the common case, and the only one that
/// costs nothing beyond the probes. Otherwise the walk's end point is handed to `finish`, an
/// **uncapped** re-descend to the search's own tolerance (#3159 review): `probe`'s
/// [`KICK_MAX_ITER`] cap can stop short of a converged mode, which would otherwise report the
/// adopted EBE `unconverged` and take its `H` off a point whose gradient has not settled, and a
/// candidate still short of full convergence is closer to the incumbent's own basin, where the
/// `1e-9` test is more exposed to the same non-move noise the guarded multi-start's distance
/// gate exists for. The final decision compares `finish`'s fully-converged objective against
/// `nll`, not `probe`'s: `None` when that comparison no longer clears `1e-9` either.
///
/// `E` is whatever `finish` reports about how its search ended — the `EbeResult::converged` /
/// `H` / `used_fallback` the caller reports are read off that call alone.
pub(super) fn kick_restarts<E>(
    x: &[f64], nll: f64, sds: &[f64], obj: &dyn Fn(&[f64]) -> f64,
    mut probe: impl FnMut(&mut Vec<f64>), mut finish: impl FnMut(&mut Vec<f64>) -> E,
) -> Option<(Vec<f64>, f64, E)> {
    let mut best: Option<(Vec<f64>, f64)> = None;
    let mut coords: Vec<(usize, f64)> =
        sds.iter().copied().enumerate().filter(|&(_, sd)| sd >= KICK_MIN_SD).collect();
    coords.sort_by(|a, b| b.1.total_cmp(&a.1));
    coords.truncate(MAX_KICK_COORDS);
    for (j, sd) in coords {
        for sgn in [1.0, -1.0] {
            let (from, f_from) = best.as_ref().map_or((x, nll), |(p, f)| (p.as_slice(), *f));
            let mut cand = from.to_vec();
            cand[j] += sgn * KICK_FRAC * sd;
            probe(&mut cand);
            if cand.iter().all(|v| v.is_finite()) {
                let f = obj(&cand);
                if f + 1e-9 < f_from {
                    best = Some((cand, f));
                }
            }
        }
    }
    let (mut p, _) = best?;
    let exit = finish(&mut p);
    let f = p.iter().all(|v| v.is_finite()).then(|| obj(&p))?;
    (f + 1e-9 < nll).then_some((p, f, exit))
}
