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
//! **Padé, never eigendecomposition.** [`Expm::exp`] is Higham's (2005) scaling-and-squaring
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

/// Row-major `n×n` product `x·y`, into `out`.
///
/// A factor that is a **structural** zero — value 0 and no derivative at all, no jet and no
/// tape node (`structural_zero`) — is skipped: its products add exactly 0 to every
/// component, so the result is bit-identical (up to the sign of a zero). A PK generator is
/// mostly such zeros (a transit chain is bidiagonal, its powers stay near-triangular), and in
/// a dual type every skipped product is a full jet multiply. A zero that is only *numerically*
/// zero — `θ − θ₀` at `θ = θ₀`, a taped or jet-carrying value — is not constant and is kept.
fn matmul_into<T: PkNum>(x: &[T], y: &[T], n: usize, out: &mut [T], row_finite: &mut [bool]) {
    use crate::sens::linsolve::{all_finite, structural_zero};
    // A skipped `x[i][k]` multiplies row `k` of `y`, which must be finite for the skipped
    // products to be exactly `±0` (`all_finite`).
    for (k, f) in row_finite.iter_mut().enumerate().take(n) {
        *f = y[k * n..(k + 1) * n].iter().all(|&v| all_finite(v));
    }
    if T::FUSED_FOLDS {
        // A taped type folds each entry as one tape node (`PkNum::dot_fold`): the same
        // products, in the same ascending-`k` order, as the loop below.
        let mut pairs: Vec<(T, T)> = Vec::with_capacity(n);
        for i in 0..n {
            for j in 0..n {
                pairs.clear();
                for k in 0..n {
                    let xik = x[i * n + k];
                    if row_finite[k] && structural_zero(xik) {
                        continue;
                    }
                    pairs.push((xik, y[k * n + j]));
                }
                out[i * n + j] = T::dot_fold(T::from_f64(0.0), &pairs);
            }
        }
        return;
    }
    out[..n * n].fill(T::from_f64(0.0));
    for i in 0..n {
        for k in 0..n {
            let xik = x[i * n + k];
            if row_finite[k] && structural_zero(xik) {
                continue;
            }
            for j in 0..n {
                out[i * n + j] = out[i * n + j] + xik * y[k * n + j];
            }
        }
    }
}

/// [`matmul_into`] into a new matrix.
#[cfg(test)]
fn matmul<T: PkNum>(x: &[T], y: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::from_f64(0.0); n * n];
    matmul_into(x, y, n, &mut out, &mut vec![false; n]);
    out
}

/// `Σ_k c_k·M_k`, each `M_k` an `n×n` row-major matrix, plus `c_I·I` when `c_i` is nonzero,
/// into `out`.
fn lincomb_into<T: PkNum>(terms: &[(f64, &[T])], c_i: f64, n: usize, out: &mut [T]) {
    if T::FUSED_FOLDS {
        // One linear tape node per entry (`PkNum::caxpy_fold`), the loop below's values.
        let mut ts: Vec<(f64, T)> = Vec::with_capacity(terms.len());
        for (e, o) in out[..n * n].iter_mut().enumerate() {
            ts.clear();
            ts.extend(terms.iter().map(|&(c, m)| (c, m[e])));
            *o = T::caxpy_fold(T::from_f64(0.0), &ts);
        }
    } else {
        out[..n * n].fill(T::from_f64(0.0));
        for &(c, m) in terms {
            let c = T::from_f64(c);
            for (o, &v) in out.iter_mut().zip(&m[..n * n]) {
                *o = *o + c * v;
            }
        }
    }
    if c_i != 0.0 {
        for i in 0..n {
            out[i * n + i] = out[i * n + i] + T::from_f64(c_i);
        }
    }
}

/// One Padé approximant `R` of a scaled matrix (`key`, empty when the type carries a tape and
/// the entry is never matched) and the squarings computed from it: `powers[k] = R^(2^k)`.
/// `powers` is empty when the approximant's denominator was singular.
struct PadeEntry<T> {
    key: Vec<T>,
    reusable: bool,
    powers: Vec<Vec<T>>,
}

/// Where [`Expm::exp`] left an exponential: valid until the next [`Expm::reset`].
#[derive(Clone, Copy)]
pub(crate) struct ExpmSlot {
    entry: usize,
    power: usize,
}

/// Matrix exponentials of `n×n` matrices sharing one workspace, so a segment's products
/// allocate once rather than once each.
///
/// Two exponentials whose *scaled* matrices `A·2^{−s}` are the same bits share their Padé
/// approximant, and the one with more squarings continues the other's: `exp(A·4)` is
/// `exp(A·2)` squared once whenever `A·2` is below `θ₁₃` and `A·4` is not. PK sampling
/// designs (0.5, 1, 2, 4, 8, 12, 24 h) are full of such gaps. The result is bit-identical to
/// computing each from scratch — the same operations on the same bits — and only types that
/// carry no tape are reused ([`PkNum::same_bits_untaped`]); a taped one keeps just the final
/// power of each result.
pub(crate) struct Expm<T> {
    n: usize,
    a1: Vec<T>,
    a2: Vec<T>,
    a4: Vec<T>,
    a6: Vec<T>,
    t1: Vec<T>,
    t2: Vec<T>,
    u: Vec<T>,
    v: Vec<T>,
    col: Vec<T>,
    piv: Vec<usize>,
    row_finite: Vec<bool>,
    cache: Vec<PadeEntry<T>>,
}

impl<T: PkNum> Expm<T> {
    pub(crate) fn new(n: usize) -> Self {
        let z = || vec![T::from_f64(0.0); n * n];
        Expm {
            n,
            a1: z(),
            a2: z(),
            a4: z(),
            a6: z(),
            t1: z(),
            t2: z(),
            u: z(),
            v: z(),
            col: vec![T::from_f64(0.0); n],
            piv: vec![0; n],
            row_finite: vec![false; n],
            cache: Vec::new(),
        }
    }

    /// Forget every cached approximant (and invalidate every [`ExpmSlot`]).
    pub(crate) fn reset(&mut self) {
        self.cache.clear();
    }

    /// Matrix exponential `exp(A)` of the row-major `n×n` matrix `a`, by scaling and squaring
    /// with the degree-13 diagonal Padé approximant (Higham 2005); read it with [`Self::get`].
    ///
    /// Returns `None` when the value part is non-finite, the norm needs more than
    /// [`MAX_SQUARINGS`] halvings, or the Padé denominator `V − U` is singular (it is not, for
    /// any finite input in range — the guard is for a diverged parameter set).
    ///
    /// The scaling exponent `s` is chosen from `‖A.val()‖₁`, so at `T = Dual2` the jets ride a
    /// computation fixed by the value part and the result is the exact derivative of the
    /// approximant the `f64` run returns. That approximant's Fréchet derivative is itself an
    /// accurate approximation of `exp`'s (Al-Mohy & Higham 2009), which the FD parity tests
    /// pin.
    pub(crate) fn exp(&mut self, a: &[T]) -> Option<ExpmSlot> {
        let n = self.n;
        debug_assert_eq!(a.len(), n * n, "expm: matrix must be n×n row-major");
        if n == 0 {
            return None;
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
            s as usize
        } else {
            0
        };
        let scale = T::from_f64(0.5_f64.powi(s as i32));
        for (o, &x) in self.a1.iter_mut().zip(a) {
            *o = x * scale;
        }
        let reusable = self.a1[0].same_bits_untaped(&self.a1[0]).is_some();
        let found = reusable
            .then(|| {
                self.cache.iter().position(|e| {
                    e.reusable
                        && e.key
                            .iter()
                            .zip(&self.a1)
                            .all(|(x, y)| x.same_bits_untaped(y) == Some(true))
                })
            })
            .flatten();
        let idx = match found {
            Some(i) => i,
            None => {
                let powers = self.pade().map(|r| vec![r]).unwrap_or_default();
                self.cache.push(PadeEntry {
                    key: if reusable {
                        self.a1.clone()
                    } else {
                        Vec::new()
                    },
                    reusable,
                    powers,
                });
                self.cache.len() - 1
            }
        };
        let Expm {
            cache, row_finite, ..
        } = self;
        let entry = &mut cache[idx];
        if entry.powers.is_empty() {
            return None;
        }
        if entry.reusable {
            while entry.powers.len() <= s {
                let last = entry.powers.last().expect("non-empty");
                let mut sq = vec![T::from_f64(0.0); n * n];
                matmul_into(last, last, n, &mut sq, row_finite);
                entry.powers.push(sq);
            }
        } else {
            // Never matched again: square in place and keep only the result.
            let mut sq = vec![T::from_f64(0.0); n * n];
            for _ in 0..s {
                matmul_into(&entry.powers[0], &entry.powers[0], n, &mut sq, row_finite);
                std::mem::swap(&mut entry.powers[0], &mut sq);
            }
        }
        let power = if entry.reusable { s } else { 0 };
        let r = &entry.powers[power];
        if r.iter().any(|x| !x.val().is_finite() || !x.jets_finite()) {
            return None;
        }
        Some(ExpmSlot { entry: idx, power })
    }

    /// The exponential [`Self::exp`] returned `slot` for.
    pub(crate) fn get(&self, slot: ExpmSlot) -> &[T] {
        &self.cache[slot.entry].powers[slot.power]
    }

    /// The degree-13 Padé approximant `R` of `self.a1`, or `None` when `V − U` is singular.
    fn pade(&mut self) -> Option<Vec<T>> {
        let n = self.n;
        let b = &PADE13;
        let rf = &mut self.row_finite;
        matmul_into(&self.a1, &self.a1, n, &mut self.a2, rf);
        matmul_into(&self.a2, &self.a2, n, &mut self.a4, rf);
        matmul_into(&self.a4, &self.a2, n, &mut self.a6, rf);

        // U = A·[A6·(b13·A6 + b11·A4 + b9·A2) + b7·A6 + b5·A4 + b3·A2 + b1·I]
        lincomb_into(
            &[(b[13], &self.a6), (b[11], &self.a4), (b[9], &self.a2)],
            0.0,
            n,
            &mut self.t1,
        );
        matmul_into(&self.a6, &self.t1, n, &mut self.t2, rf);
        lincomb_into(
            &[
                (1.0, &self.t2),
                (b[7], &self.a6),
                (b[5], &self.a4),
                (b[3], &self.a2),
            ],
            b[1],
            n,
            &mut self.t1,
        );
        matmul_into(&self.a1, &self.t1, n, &mut self.u, rf);
        // V = A6·(b12·A6 + b10·A4 + b8·A2) + b6·A6 + b4·A4 + b2·A2 + b0·I
        lincomb_into(
            &[(b[12], &self.a6), (b[10], &self.a4), (b[8], &self.a2)],
            0.0,
            n,
            &mut self.t1,
        );
        matmul_into(&self.a6, &self.t1, n, &mut self.t2, rf);
        lincomb_into(
            &[
                (1.0, &self.t2),
                (b[6], &self.a6),
                (b[4], &self.a4),
                (b[2], &self.a2),
            ],
            b[0],
            n,
            &mut self.v,
        );

        // Solve (V − U)·R = (V + U), one column at a time against one factorization.
        let (q, p) = (&mut self.t1, &mut self.t2);
        for i in 0..n * n {
            let (vi, ui) = (self.v[i], self.u[i]);
            q[i] = vi - ui;
            p[i] = vi + ui;
        }
        if !lu_factor_in_place_g(q, &mut self.piv, n) {
            return None;
        }
        let mut r = vec![T::from_f64(0.0); n * n];
        let col = &mut self.col;
        for j in 0..n {
            for i in 0..n {
                col[i] = p[i * n + j];
            }
            lu_solve_in_place_g(q, &self.piv, n, col);
            for i in 0..n {
                r[i * n + j] = col[i];
            }
        }
        Some(r)
    }
}

/// Most Padé approximants a thread's [`Expm`] keeps between segments. Each holds `s + 1`
/// matrices, so this bounds a thread's memory; past it the next segment forgets them all (and
/// recomputes, bit-identically).
const POOL_MAX_ENTRIES: usize = 64;
/// Most [`Expm`] sizes a thread's workspace keeps per number type.
const POOL_MAX_WORKSPACES: usize = 16;

/// Whether `T` carries no tape, so its Padé approximants are shared between segments
/// ([`PkNum::same_bits_untaped`]): a reused result would drop the records a taped type's
/// recomputation writes.
fn pade_shared<T: PkNum>() -> bool {
    T::from_f64(0.0)
        .same_bits_untaped(&T::from_f64(0.0))
        .is_some()
}

/// The Padé approximants this thread's workspace for `T` holds, over every size.
#[cfg(test)]
fn pooled_pade_entries<T: PkNum>() -> usize {
    LIN_WS.with(|p| {
        p.borrow()
            .iter()
            .filter_map(|b| b.downcast_ref::<LinWs<T>>())
            .flat_map(|w| w.expms.iter())
            .map(|e| e.cache.len())
            .sum()
    })
}

/// [`Expm::exp`] of one matrix.
#[cfg(test)]
pub(crate) fn expm_g<T: PkNum>(a: &[T], n: usize) -> Option<Vec<T>> {
    debug_assert_eq!(a.len(), n * n, "expm_g: matrix must be n×n row-major");
    if n == 0 {
        return Some(Vec::new());
    }
    let mut e = Expm::new(n);
    let slot = e.exp(a)?;
    Some(e.get(slot).to_vec())
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
    with_workspace(|ws: &mut LinWs<T>| segment_in(ws, rhs, u0, t_span, params, saveat, interp_at))
}

/// [`linear_segment_g`] on a reused workspace: every buffer a segment needs, so a fit's
/// thousands of segments allocate their outputs and nothing else.
fn segment_in<T: PkNum>(
    ws: &mut LinWs<T>,
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
    let m = n + 1;
    ws.size(n, zero);
    let LinWs {
        probe,
        b,
        a,
        col,
        direct,
        gen,
        scaled,
        next,
        x,
        pairs,
        params_d,
        a_d,
        b_d,
        x_d,
        times,
        props,
        n_props,
        states,
        expms,
    } = ws;

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
    rhs(probe, params, t_mid, b);
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
    for j in 0..n {
        probe[j] = T::from_f64(scale);
        rhs(probe, params, t_mid, col);
        probe[j] = zero;
        for i in 0..n {
            a[i * n + j] = T::scaled_diff(col[i], b[i], 1.0 / scale);
        }
    }
    if a.iter().chain(b.iter()).any(|x| !x.val().is_finite()) {
        return None;
    }

    // `f(x, t) == A·x + b` to rounding — the affine claim at one state and time.
    // In the value and in the derivative components alike: a column whose jets were lost to
    // cancellation reproduces the RHS's value exactly and its gradient not at all. It reads
    // values only, so it runs on detached copies (`PkNum::detach`): a taped type records
    // nothing for it.
    params_d.clear();
    params_d.extend(params.iter().map(|p| p.detach()));
    for (d, s) in a_d.iter_mut().zip(a.iter()) {
        *d = s.detach();
    }
    for (d, s) in b_d.iter_mut().zip(b.iter()) {
        *d = s.detach();
    }
    let mut reproduces = |x: &[T], t: f64, direct: &mut [T]| -> bool {
        for (d, s) in x_d.iter_mut().zip(x) {
            *d = s.detach();
        }
        let (a, b, x) = (&*a_d, &*b_d, &*x_d);
        rhs(x, params_d, t, direct);
        (0..n).all(|i| {
            let mut assembled = b[i];
            let mut scale = b[i].val().abs();
            let mut jet_scale = b[i].jet_max_abs();
            for j in 0..n {
                let (aij, xj) = (a[i * n + j], x[j]);
                // A structural zero against a finite state adds exactly 0 to the sum and to
                // both scales (the same skip, and the same guard, as the propagation).
                if crate::sens::linsolve::structural_zero(aij)
                    && crate::sens::linsolve::all_finite(xj)
                {
                    continue;
                }
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
    for t_check in [t0 + 0.25 * h, t0 + 0.75 * h] {
        if !reproduces(u0, t_check, direct) {
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
    for i in 0..n {
        for j in 0..n {
            gen[i * m + j] = a[i * n + j];
        }
        gen[i * m + n] = b[i] * inv_c;
    }
    let c = T::from_f64(c);

    // Every requested time, clamped into the span, visited once in ascending order.
    let clamp = |t: f64| t.clamp(t0, t1);
    times.clear();
    times.extend(saveat.iter().chain(interp_at).map(|&t| clamp(t)));
    times.sort_by(|x, y| x.total_cmp(y));
    times.dedup();

    // Propagators keyed by the exact gap, so an evenly spaced grid pays one exponential (and
    // `Expm` shares the Padé step between gaps whose scaled generators coincide, here and in
    // every other segment of the thread with the same generator and size: a steady-state
    // run-in's cycles, subjects with equal parameters).
    let expm = match expms.iter().position(|e| e.n == m) {
        Some(i) => &mut expms[i],
        None => {
            if expms.len() == POOL_MAX_WORKSPACES {
                expms.clear();
            }
            expms.push(Expm::new(m));
            expms.last_mut().expect("just pushed")
        }
    };
    // A type that carries a tape never matches an entry, so it starts afresh; an untaped one
    // keeps its approximants up to a bound on the thread's memory (forgetting them all
    // recomputes, bit-identically).
    if !pade_shared::<T>() || expm.cache.len() > POOL_MAX_ENTRIES {
        expm.reset();
    }
    *n_props = 0;
    states.clear();
    x.clear();
    x.extend_from_slice(u0);
    let mut t_prev = t0;
    for &t in times.iter() {
        let dt = t - t_prev;
        if dt > 0.0 {
            let key = dt.to_bits();
            let idx = match props[..*n_props].iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    let dt_t = T::from_f64(dt);
                    for (o, &g) in scaled.iter_mut().zip(gen.iter()) {
                        *o = g * dt_t;
                    }
                    let e = expm.exp(scaled)?;
                    if *n_props == props.len() {
                        props.push((0, Vec::new()));
                    }
                    let slot = &mut props[*n_props];
                    slot.0 = key;
                    slot.1.clear();
                    slot.1.extend_from_slice(expm.get(e));
                    *n_props += 1;
                    *n_props - 1
                }
            };
            let e = &props[idx].1;
            // `x` is finite here: `u0` passed the magnitude check, and every propagated state
            // passed `reproduces`, which declines a non-finite one
            // (`a_segment_whose_state_overflows_declines`). So a skipped term is exactly `±0`.
            for i in 0..n {
                let acc = e[i * m + n] * c;
                if T::FUSED_FOLDS {
                    // One tape node per state (`PkNum::dot_fold`), the loop's values.
                    pairs.clear();
                    for j in 0..n {
                        if !crate::sens::linsolve::structural_zero(e[i * m + j]) {
                            pairs.push((e[i * m + j], x[j]));
                        }
                    }
                    next[i] = T::dot_fold(acc, &pairs);
                    continue;
                }
                let mut acc = acc;
                for j in 0..n {
                    if crate::sens::linsolve::structural_zero(e[i * m + j]) {
                        continue;
                    }
                    acc = acc + e[i * m + j] * x[j];
                }
                next[i] = acc;
            }
            std::mem::swap(x, next);
            t_prev = t;
            // The affine claim along the trajectory, at the extraction time (an endpoint could
            // read the next segment's forcing). The probes and `u0` alone leave a nonlinear
            // RHS undetected whenever it agrees with its secant there — `du/dt = u²` from
            // `u0 = 1` is reproduced exactly at both — but not at the states it then reaches.
            if !reproduces(x, t_mid, direct) {
                return None;
            }
        }
        states.extend_from_slice(x);
    }

    let lookup = |t: f64| -> Vec<T> {
        let tc = clamp(t);
        if tc <= t0 {
            return u0.to_vec();
        }
        // `times` is sorted and contains `tc` exactly; `states` holds its state at the same
        // index.
        let i = times
            .binary_search_by(|s| s.total_cmp(&tc))
            .expect("every requested time was propagated");
        states[i * n..(i + 1) * n].to_vec()
    };
    let at = |ts: &[f64]| -> Vec<SolPointG<T>> {
        ts.iter().map(|&t| SolPointG { t, u: lookup(t) }).collect()
    };
    Some((at(saveat), at(interp_at)))
}

/// Every buffer [`segment_in`] reuses across segments, for one number type.
struct LinWs<T> {
    probe: Vec<T>,
    b: Vec<T>,
    a: Vec<T>,
    col: Vec<T>,
    direct: Vec<T>,
    gen: Vec<T>,
    scaled: Vec<T>,
    next: Vec<T>,
    x: Vec<T>,
    pairs: Vec<(T, T)>,
    params_d: Vec<T>,
    a_d: Vec<T>,
    b_d: Vec<T>,
    x_d: Vec<T>,
    times: Vec<f64>,
    /// Propagators by the bits of their gap; the first `n_props` are this segment's.
    props: Vec<(u64, Vec<T>)>,
    n_props: usize,
    /// The state at each of `times`, `n` values each.
    states: Vec<T>,
    /// One exponential workspace per size, its Padé approximants kept between segments.
    expms: Vec<Expm<T>>,
}

impl<T: PkNum> LinWs<T> {
    fn new() -> Self {
        LinWs {
            probe: Vec::new(),
            b: Vec::new(),
            a: Vec::new(),
            col: Vec::new(),
            direct: Vec::new(),
            gen: Vec::new(),
            scaled: Vec::new(),
            next: Vec::new(),
            x: Vec::new(),
            pairs: Vec::new(),
            params_d: Vec::new(),
            a_d: Vec::new(),
            b_d: Vec::new(),
            x_d: Vec::new(),
            times: Vec::new(),
            props: Vec::new(),
            n_props: 0,
            states: Vec::new(),
            expms: Vec::new(),
        }
    }
    /// Every per-segment buffer at its size for `n` states, filled with `zero` (the values a
    /// fresh `vec![zero; …]` held).
    fn size(&mut self, n: usize, zero: T) {
        let m = n + 1;
        for (v, len) in [
            (&mut self.probe, n),
            (&mut self.b, n),
            (&mut self.a, n * n),
            (&mut self.col, n),
            (&mut self.direct, n),
            (&mut self.gen, m * m),
            (&mut self.scaled, m * m),
            (&mut self.next, n),
            (&mut self.a_d, n * n),
            (&mut self.b_d, n),
            (&mut self.x_d, n),
        ] {
            v.clear();
            v.resize(len, zero);
        }
        self.pairs.clear();
    }
}

thread_local! {
    /// One [`LinWs`] per number type the thread has propagated.
    static LIN_WS: std::cell::RefCell<Vec<Box<dyn std::any::Any>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on this thread's workspace for `T`, taken out of the pool for the call (so a
/// nested segment gets its own) and put back after.
fn with_workspace<T: PkNum, R>(f: impl FnOnce(&mut LinWs<T>) -> R) -> R {
    let taken = LIN_WS.with(|p| {
        let mut p = p.borrow_mut();
        p.iter()
            .position(|w| w.is::<LinWs<T>>())
            .map(|i| p.swap_remove(i))
    });
    let mut ws: Box<LinWs<T>> = taken
        .and_then(|b| b.downcast::<LinWs<T>>().ok())
        .unwrap_or_else(|| Box::new(LinWs::new()));
    let r = f(&mut ws);
    LIN_WS.with(|p| p.borrow_mut().push(ws));
    r
}

#[cfg(test)]
#[path = "linear_tests.rs"]
mod tests;
