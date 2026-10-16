//! Exact propagation of a **linear time-invariant** `[odes]` segment (#852) — the ADVAN5
//! analogue.
//!
//! Inside one integration segment a certified-linear right-hand side is `du/dt = A·u + b`
//! with `A` and `b` constant: every PK parameter, infusion rate and zero-order input is fixed
//! between two break times, and the parser has proved the equations affine in the states
//! (`OdeSpec::linear_eligible`). Then
//!
//! ```text
//!   ⎡u(t+h)⎤          ⎛⎡A  b⎤  ⎞ ⎡u(t)⎤
//!   ⎣  1   ⎦  =  exp  ⎜⎣0  0⎦·h⎟ ⎣ 1  ⎦
//! ```
//!
//! exactly — the augmented form carries the constant input without ever inverting `A`, which
//! is singular whenever a compartment has no drain. So instead of stepping, the segment costs
//! one matrix exponential per distinct observation gap, with no stability limit (the stiff
//! eigenvalue of a fast liver compartment costs nothing) and an error at roundoff rather than
//! at `ode_reltol`.
//!
//! **Padé, never eigendecomposition.** [`expm_g`] is Higham's (2005) scaling-and-squaring
//! with the degree-13 diagonal Padé approximant. A transit chain at one rate `KTR` is a
//! *defective* matrix — one Jordan block, whose `t·e^{−kt}` / `t²·e^{−kt}` terms are what makes
//! the absorption Erlang-shaped — and an eigendecomposition is singular or catastrophically
//! ill-conditioned there. Padé never diagonalizes, so a repeated eigenvalue is a non-event.
//!
//! **One implementation for the value and the gradient.** Both functions are generic over
//! [`PkNum`]: at `T = f64` they are the prediction, at `T = Dual1`/`Dual2` the exact `∂/∂(θ,η)`
//! and its second order, by the same `*_g<T: PkNum>` rule the closed forms follow. Every
//! branch — the squaring count, LU pivoting, the self-checks below — reads [`PkNum::val`]
//! only, so the dual run differentiates the very computation the `f64` run reports.

use super::solver::SolPointG;
use crate::sens::linsolve::{lu_factor_in_place_g, lu_solve_in_place_g};
use crate::sens::num::PkNum;

/// Degree-13 Padé numerator coefficients `b_0..b_13` (Higham 2005, Table 10.4 / eq. 10.33).
const PADE13: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

/// `θ_13`: the largest `‖A‖₁` for which the degree-13 approximant meets unit roundoff in IEEE
/// double without scaling (Higham 2005, Table 2.3).
const THETA13: f64 = 5.371_920_351_148_152;

/// Upper bound on the number of squarings. `‖A‖₁ = θ₁₃·2⁶⁴` is far beyond any segment a PK
/// timeline produces (a rate of 1e6/h over 1e9 h); past it the input is treated as diverged
/// rather than looping on an `inf` norm — the same hang the markov `expm` guards against.
const MAX_SQUARINGS: u32 = 64;

/// Row-major `n×n` product `x·y`.
fn matmul<T: PkNum>(x: &[T], y: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::from_f64(0.0); n * n];
    for i in 0..n {
        for k in 0..n {
            let xik = x[i * n + k];
            for j in 0..n {
                out[i * n + j] = out[i * n + j] + xik * y[k * n + j];
            }
        }
    }
    out
}

/// `Σ_k c_k·M_k`, each `M_k` an `n×n` row-major matrix, plus `c_I·I` when `c_i` is nonzero.
fn lincomb<T: PkNum>(terms: &[(f64, &[T])], c_i: f64, n: usize) -> Vec<T> {
    let mut out = vec![T::from_f64(0.0); n * n];
    for &(c, m) in terms {
        let c = T::from_f64(c);
        for (o, &v) in out.iter_mut().zip(m) {
            *o = *o + c * v;
        }
    }
    if c_i != 0.0 {
        for i in 0..n {
            out[i * n + i] = out[i * n + i] + T::from_f64(c_i);
        }
    }
    out
}

/// Matrix exponential `exp(A)` of the row-major `n×n` matrix `a`, by scaling and squaring with
/// the degree-13 diagonal Padé approximant (Higham 2005).
///
/// Returns `None` when the value part is non-finite, the norm needs more than
/// [`MAX_SQUARINGS`] halvings, or the Padé denominator `V − U` is singular (it is not, for any
/// finite input in range — the guard is for a diverged parameter set).
///
/// The scaling exponent `s` is chosen from `‖A.val()‖₁`, so at `T = Dual2` the jets ride a
/// computation fixed by the value part and the result is the exact derivative of the
/// approximant the `f64` run returns. That approximant's Fréchet derivative is itself an
/// accurate approximation of `exp`'s (Al-Mohy & Higham 2009), which the FD parity tests pin.
pub(crate) fn expm_g<T: PkNum>(a: &[T], n: usize) -> Option<Vec<T>> {
    debug_assert_eq!(a.len(), n * n, "expm_g: matrix must be n×n row-major");
    if n == 0 {
        return Some(Vec::new());
    }
    if a.iter().any(|x| !x.val().is_finite()) {
        return None;
    }
    // ‖A‖₁ — max absolute column sum of the value part.
    let norm1 = (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].val().abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let s = if norm1 > THETA13 {
        let s = (norm1 / THETA13).log2().ceil();
        if !(s <= MAX_SQUARINGS as f64) {
            return None;
        }
        s as u32
    } else {
        0
    };
    let scale = T::from_f64(0.5_f64.powi(s as i32));
    let a1: Vec<T> = a.iter().map(|&x| x * scale).collect();

    let b = &PADE13;
    let a2 = matmul(&a1, &a1, n);
    let a4 = matmul(&a2, &a2, n);
    let a6 = matmul(&a4, &a2, n);

    // U = A·[A6·(b13·A6 + b11·A4 + b9·A2) + b7·A6 + b5·A4 + b3·A2 + b1·I]
    let u_inner = lincomb(&[(b[13], &a6), (b[11], &a4), (b[9], &a2)], 0.0, n);
    let u_inner = matmul(&a6, &u_inner, n);
    let u_poly = lincomb(
        &[(1.0, &u_inner), (b[7], &a6), (b[5], &a4), (b[3], &a2)],
        b[1],
        n,
    );
    let u = matmul(&a1, &u_poly, n);
    // V = A6·(b12·A6 + b10·A4 + b8·A2) + b6·A6 + b4·A4 + b2·A2 + b0·I
    let v_inner = lincomb(&[(b[12], &a6), (b[10], &a4), (b[8], &a2)], 0.0, n);
    let v_inner = matmul(&a6, &v_inner, n);
    let v = lincomb(
        &[(1.0, &v_inner), (b[6], &a6), (b[4], &a4), (b[2], &a2)],
        b[0],
        n,
    );

    // Solve (V − U)·R = (V + U), one column at a time against one factorization.
    let mut q: Vec<T> = v.iter().zip(&u).map(|(&vi, &ui)| vi - ui).collect();
    let p: Vec<T> = v.iter().zip(&u).map(|(&vi, &ui)| vi + ui).collect();
    let mut piv = vec![0usize; n];
    if !lu_factor_in_place_g(&mut q, &mut piv, n) {
        return None;
    }
    let mut r = vec![T::from_f64(0.0); n * n];
    let mut col = vec![T::from_f64(0.0); n];
    for j in 0..n {
        for i in 0..n {
            col[i] = p[i * n + j];
        }
        lu_solve_in_place_g(&q, &piv, n, &mut col);
        for i in 0..n {
            r[i * n + j] = col[i];
        }
    }
    for _ in 0..s {
        r = matmul(&r, &r, n);
    }
    if r.iter().any(|x| !x.val().is_finite() || !x.jets_finite()) {
        return None;
    }
    Some(r)
}

/// Relative tolerance of the per-segment self-checks in [`linear_segment_g`].
///
/// The checks compare the right-hand side evaluated directly against `A·u + b` reassembled
/// from the extracted columns. For an affine RHS the two differ only by the rounding of an
/// `n`-term sum — measured below 1e-15 relative on every fixture in this module's tests — so
/// `1e-9` sits six orders above the honest discrepancy and far below any real nonlinearity
/// or time dependence (a forcing switching on, a saturable term), which moves the RHS by
/// O(1) relative.
const SELF_CHECK_RTOL: f64 = 1e-9;

/// Propagate one segment `(t_span.0, t_span.1]` of a **linear time-invariant** RHS exactly,
/// returning the state at every `saveat` time and every `interp_at` time — the same
/// `(hard, soft)` contract as the stepping drivers, so a caller cannot tell which produced it.
///
/// `A` and `b` are read off the RHS itself at the segment midpoint: `b = f(0)` and column `j`
/// of `A` is `(f(s·e_j) − b)/s`, with `s` a power of two 2^20 above the forcing and the entry
/// state (values and derivative components). That keeps the dynamics in one place — the parser-built RHS, with
/// its infusion and zero-order wrapping — instead of a second symbolic copy of the model that
/// could drift from it.
///
/// Returns `None` — and the caller integrates the segment instead — when the extracted system
/// fails to reproduce the RHS at the entry state at two interior times, or at any state the
/// propagation reaches. That catches a segment whose forcing switches inside the span, or an
/// RHS that is not actually affine (an `OdeMethod::Linear` handed to a public `solve_ode*`
/// entry point with a closure the parser never saw). It is a backstop, not the certification:
/// the parser's structural classifier is what decides a model is linear, because a saturable
/// term whose `Km` is far above every state it reaches is affine to rounding at every checked
/// point — and then the exact answer is the right one to that same precision.
pub(crate) fn linear_segment_g<T: PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
) -> Option<(Vec<SolPointG<T>>, Vec<SolPointG<T>>)> {
    let n = u0.len();
    let (t0, t1) = t_span;
    let h = t1 - t0;
    if !(h > 0.0) || !h.is_finite() {
        return None;
    }
    let zero = T::from_f64(0.0);
    let t_mid = t0 + 0.5 * h;

    // ── Extract A (row-major n×n) and b at the midpoint: b = f(0), column j = (f(s·e_j) − b)/s.
    //
    // The difference cancels b, so it carries rounding of order ε·|b_i| — in the value and,
    // separately, in every derivative component — i.e. ε·|b_i|/(s·|A_ij|) relative to the
    // entry. A unit probe loses every digit wherever the forcing dwarfs the coefficient: an
    // infusion of 1e9 into k = 1e-3 reads A to ~1e-7, `du/dt = 1e20 − u` reads A = 0, and a
    // forcing whose *sensitivity* is large (`C·p` with C = 1e20) zeroes ∂A/∂p while the value
    // is fine. So the probe is set from the model's own magnitudes — the forcing, the entry
    // state, and the derivative components of both — times 2^20, rounded up to a power of
    // two so the multiply and divide are exact: `s·A_ij` then dominates `b_i` in every
    // component wherever `|A_ij|` is not itself tiny, and `fl(s·A + b) − b` is `s·A·(1 + δ)`.
    // Tying it to those magnitudes, rather than to one huge constant, keeps the RHS evaluated
    // within 2^20 of values it already produces, so a benign intermediate cannot overflow.
    // Whatever cancellation remains is not trusted: the self-check below compares the
    // derivative components too, and a column read wrongly in any of them declines the
    // segment rather than returning a wrong gradient.
    let mut probe = vec![zero; n];
    let mut b = vec![zero; n];
    rhs(&probe, params, t_mid, &mut b);
    let magnitude = b
        .iter()
        .chain(u0)
        .fold(1.0_f64, |m, x| m.max(x.val().abs()).max(x.jet_max_abs()));
    if !magnitude.is_finite() {
        return None;
    }
    let scale = 2f64.powi(magnitude.log2().ceil() as i32 + 20);
    if !scale.is_finite() {
        return None;
    }
    let mut a = vec![zero; n * n];
    let mut col = vec![zero; n];
    let inv_scale = T::from_f64(1.0 / scale);
    for j in 0..n {
        probe[j] = T::from_f64(scale);
        rhs(&probe, params, t_mid, &mut col);
        probe[j] = zero;
        for i in 0..n {
            a[i * n + j] = (col[i] - b[i]) * inv_scale;
        }
    }
    if a.iter().chain(&b).any(|x| !x.val().is_finite()) {
        return None;
    }

    // `f(x, t) == A·x + b` to rounding — the affine claim at one state and time.
    // In the value and in the derivative components alike: a column whose jets were lost to
    // cancellation reproduces the RHS's value exactly and its gradient not at all.
    let reproduces = |x: &[T], t: f64, direct: &mut [T]| -> bool {
        rhs(x, params, t, direct);
        (0..n).all(|i| {
            let mut assembled = b[i];
            let mut scale = b[i].val().abs();
            let mut jet_scale = b[i].jet_max_abs();
            for j in 0..n {
                let (aij, xj) = (a[i * n + j], x[j]);
                assembled = assembled + aij * xj;
                scale += (aij.val() * xj.val()).abs();
                jet_scale += aij.jet_max_abs() * xj.val().abs()
                    + aij.val().abs() * xj.jet_max_abs()
                    + aij.jet_max_abs() * xj.jet_max_abs();
            }
            let diff = direct[i] - assembled;
            direct[i].val().is_finite()
                && diff.val().abs() <= SELF_CHECK_RTOL * scale + 1e-300
                && diff.jet_max_abs() <= SELF_CHECK_RTOL * jet_scale + 1e-300
        })
    };

    // ── Self-check at the entry state, a quarter and three quarters through the span: catches
    // forcing that switches inside the span. Every propagated state is checked too, below.
    let mut direct = vec![zero; n];
    for t_check in [t0 + 0.25 * h, t0 + 0.75 * h] {
        if !reproduces(u0, t_check, &mut direct) {
            return None;
        }
    }

    // ── Augmented generator M = [[A, b/c], [0, 0]], (n+1)×(n+1); the propagated forcing
    // column is multiplied back by `c` below. Without `c` a forcing far above the rates sets
    // ‖M‖, and with it the squaring count: `du/dt = 1e20 − u` over unit time needs ~64
    // squarings and was declined. Scaling the column to the rates' magnitude leaves the
    // squaring count to `A`, where it belongs; `c` is a plain constant, so the jets are exact.
    let a_max = a.iter().fold(0.0_f64, |acc, x| acc.max(x.val().abs()));
    // Jets included: a forcing whose value is 0 but whose derivative is 1e292 leaves the
    // generator's value at zero (no squarings) while the Padé numerator multiplies that jet by
    // ~3e16 and overflows it.
    let b_max = b.iter().fold(0.0_f64, |acc, x| {
        acc.max(x.val().abs()).max(x.jet_max_abs())
    });
    let c = (b_max / a_max.max(1.0)).max(1.0);
    let inv_c = T::from_f64(1.0 / c);
    let m = n + 1;
    let mut gen = vec![zero; m * m];
    for i in 0..n {
        for j in 0..n {
            gen[i * m + j] = a[i * n + j];
        }
        gen[i * m + n] = b[i] * inv_c;
    }
    let c = T::from_f64(c);

    // Every requested time, clamped into the span, visited once in ascending order.
    let clamp = |t: f64| t.clamp(t0, t1);
    let mut times: Vec<f64> = saveat.iter().chain(interp_at).map(|&t| clamp(t)).collect();
    times.sort_by(|x, y| x.total_cmp(y));
    times.dedup();

    // Propagators keyed by the exact gap, so an evenly spaced grid pays one exponential.
    let mut cache: Vec<(u64, Vec<T>)> = Vec::new();
    let mut states: Vec<(f64, Vec<T>)> = Vec::with_capacity(times.len());
    let mut x = u0.to_vec();
    let mut t_prev = t0;
    for &t in &times {
        let dt = t - t_prev;
        if dt > 0.0 {
            let key = dt.to_bits();
            let idx = match cache.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    let scaled: Vec<T> = gen.iter().map(|&g| g * T::from_f64(dt)).collect();
                    cache.push((key, expm_g(&scaled, m)?));
                    cache.len() - 1
                }
            };
            let e = &cache[idx].1;
            let mut next = vec![zero; n];
            for i in 0..n {
                let mut acc = e[i * m + n] * c;
                for j in 0..n {
                    acc = acc + e[i * m + j] * x[j];
                }
                next[i] = acc;
            }
            x = next;
            t_prev = t;
            // The affine claim along the trajectory, at the extraction time (an endpoint could
            // read the next segment's forcing). The probes and `u0` alone leave a nonlinear
            // RHS undetected whenever it agrees with its secant there — `du/dt = u²` from
            // `u0 = 1` is reproduced exactly at both — but not at the states it then reaches.
            if !reproduces(&x, t_mid, &mut direct) {
                return None;
            }
        }
        states.push((t, x.clone()));
    }

    let lookup = |t: f64| -> Vec<T> {
        let tc = clamp(t);
        if tc <= t0 {
            return u0.to_vec();
        }
        // `times` is sorted and contains `tc` exactly.
        let i = states
            .binary_search_by(|(s, _)| s.total_cmp(&tc))
            .expect("every requested time was propagated");
        states[i].1.clone()
    };
    let at = |ts: &[f64]| -> Vec<SolPointG<T>> {
        ts.iter().map(|&t| SolPointG { t, u: lookup(t) }).collect()
    };
    Some((at(saveat), at(interp_at)))
}

#[cfg(test)]
#[path = "linear_tests.rs"]
mod tests;
