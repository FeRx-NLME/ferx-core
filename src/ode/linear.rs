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
//! [`PkNum`]: at `T = f64` they are the prediction, at `T = Dual2`/`Dual2` the exact `∂/∂(θ,η)`
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
        *f = crate::sens::linsolve::skips_zeros::<T>()
            && y[k * n..(k + 1) * n].iter().all(|&v| all_finite(v));
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
    // The common compartment counts as literals into the one inlined loop, so it unrolls at
    // each size; the same products in the same order, skips included (#1564).
    match n {
        2 => accumulate(x, y, 2, out, row_finite),
        3 => accumulate(x, y, 3, out, row_finite),
        4 => accumulate(x, y, 4, out, row_finite),
        _ => accumulate(x, y, n, out, row_finite),
    }
}

/// [`matmul_into`]'s accumulation: `out = x·y`, skipping a structural-zero `x[i][k]` whose
/// row `k` of `y` is finite. Inlined into each call so a literal `n` unrolls the loops.
#[inline(always)]
fn accumulate<T: PkNum>(x: &[T], y: &[T], n: usize, out: &mut [T], row_finite: &[bool]) {
    use crate::sens::linsolve::structural_zero;
    let (x, y, out) = (&x[..n * n], &y[..n * n], &mut out[..n * n]);
    out.fill(T::from_f64(0.0));
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
    piv: Vec<T>,
    row_finite: Vec<bool>,
    cache: Vec<PadeEntry<T>>,
    /// `n×n` buffers [`Self::reset`] released, handed out again for approximants, their
    /// squarings and keys (#2549): a fit's segments otherwise allocate and free several per
    /// new exponential.
    spare: Vec<Vec<T>>,
    /// Emptied `powers` lists, likewise.
    spare_powers: Vec<Vec<Vec<T>>>,
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
            piv: vec![T::from_f64(0.0); crate::sens::linsolve::lu_piv_len::<T>(n)],
            row_finite: vec![false; n],
            cache: Vec::new(),
            spare: Vec::new(),
            spare_powers: Vec::new(),
        }
    }

    /// An `n×n` buffer: a released one, or a new one. Its contents are unspecified; every
    /// caller writes all of it.
    fn buffer(spare: &mut Vec<Vec<T>>, n: usize) -> Vec<T> {
        let mut v = spare.pop().unwrap_or_default();
        v.clear();
        v.resize(n * n, T::from_f64(0.0));
        v
    }

    /// Forget every cached approximant (and invalidate every [`ExpmSlot`]).
    pub(crate) fn reset(&mut self) {
        for mut e in self.cache.drain(..) {
            self.spare.append(&mut e.powers);
            self.spare_powers.push(e.powers);
            if e.key.capacity() > 0 {
                self.spare.push(e.key);
            }
        }
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
        // The squaring count, on the values ([`pade_squarings`]; a recording type re-checks it
        // on every evaluation): `−1` for a non-finite value or a norm past `MAX_SQUARINGS`.
        let mut args = a[..n * n].to_vec();
        args.push(T::from_f64(n as f64));
        let host = T::host(crate::sens::num::HostFn::PadeSquarings, &args);
        let need = host[0];
        let out1 = host.get(1).copied().unwrap_or_else(|| T::from_f64(1.0));
        // A recording type (#2786) takes the count as data: the scale `2⁻ˢ` the analysis returns
        // with it, the squarings a run-time repeat call (`PkNum::repeat`) — the value path's
        // arithmetic at every count, checked only that the count exists.
        let (s, counted, scale) = if T::RECORDS_BRANCHES {
            if !need.br_cmp_c(crate::sens::num::BranchCmp::Ge, 0.0) {
                return None;
            }
            (0, Some(need), out1)
        } else {
            let s = need.br_value();
            if s < 0.0 {
                return None;
            }
            (s as usize, None, T::from_f64(0.5_f64.powi(s as i32)))
        };
        for (o, &x) in self.a1.iter_mut().zip(a) {
            *o = x * scale;
        }
        let reusable = counted.is_none() && self.a1[0].same_bits_untaped(&self.a1[0]).is_some();
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
                let mut powers = self.spare_powers.pop().unwrap_or_default();
                if let Some(r) = self.pade() {
                    powers.push(r);
                }
                let key = if reusable {
                    let mut k = Self::buffer(&mut self.spare, 0);
                    k.extend_from_slice(&self.a1);
                    k
                } else {
                    Vec::new()
                };
                self.cache.push(PadeEntry {
                    key,
                    reusable,
                    powers,
                });
                self.cache.len() - 1
            }
        };
        let Expm {
            cache,
            row_finite,
            spare,
            ..
        } = self;
        let entry = &mut cache[idx];
        if entry.powers.is_empty() {
            return None;
        }
        if entry.reusable {
            while entry.powers.len() <= s {
                let last = entry.powers.last().expect("non-empty");
                let mut sq = Self::buffer(spare, n);
                matmul_into(last, last, n, &mut sq, row_finite);
                entry.powers.push(sq);
            }
        } else {
            // Never matched again: square in place and keep only the result.
            let mut sq = Self::buffer(spare, n);
            if let Some(need) = counted {
                let n2 = n;
                let square = |x: &[T]| -> Vec<T> {
                    let mut y = vec![T::from_f64(0.0); n2 * n2];
                    let mut rf = vec![true; n2];
                    matmul_into(x, x, n2, &mut y, &mut rf);
                    y
                };
                match T::repeat(&entry.powers[0], need, &square) {
                    Some(Some(y)) => entry.powers[0].copy_from_slice(&y),
                    _ => return None,
                }
            }
            for _ in 0..s {
                matmul_into(&entry.powers[0], &entry.powers[0], n, &mut sq, row_finite);
                std::mem::swap(&mut entry.powers[0], &mut sq);
            }
            spare.push(sq);
        }
        let power = if entry.reusable { s } else { 0 };
        let r = &entry.powers[power];
        if r.iter().any(|x| !crate::sens::linsolve::all_finite(*x)) {
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
        let mut r = Self::buffer(&mut self.spare, n);
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

/// The [`HostFn::ForcingScale`](crate::sens::num::HostFn) analysis: `args = [A (n·n), b (n),
/// n]` → `[c]`, `c = max(1, max|b| / max(1, max|A|))`, the augmented generator's forcing
/// column scale on the values.
pub(crate) fn forcing_scale(args: &[f64], out: &mut Vec<f64>) {
    let n = args[args.len() - 1] as usize;
    let a_max = args[..n * n].iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let b_max = args[n * n..n * n + n]
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    out.clear();
    out.push((b_max / a_max.max(1.0)).max(1.0));
}

/// The [`HostFn::PadeSquarings`](crate::sens::num::HostFn) analysis: `args = [A (n·n), n]`
/// → `[s]`, the halvings that bring `‖A‖₁` under `θ₁₃`, or `−1` when a value is not finite or
/// more than [`MAX_SQUARINGS`] are needed.
pub(crate) fn pade_squarings(args: &[f64], out: &mut Vec<f64>) {
    let n = args[args.len() - 1] as usize;
    let a = &args[..n * n];
    out.clear();
    if a.iter().any(|x| !x.is_finite()) {
        out.push(-1.0);
        return;
    }
    // ‖A‖₁ — max absolute column sum of the value part.
    let norm1 = (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let s = if norm1 > THETA13 {
        let s = (norm1 / THETA13).log2().ceil();
        if !(s <= MAX_SQUARINGS as f64) {
            out.push(-1.0);
            return;
        }
        s
    } else {
        0.0
    };
    out.push(s);
    out.push(0.5_f64.powi(s as i32));
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
    rhs: &crate::ode::solver::RhsG<'_, T>,
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
) -> Option<(Vec<SolPointG<T>>, Vec<SolPointG<T>>)> {
    with_workspace(|ws: &mut LinWs<T>| segment_in(ws, rhs, u0, t_span, params, saveat, interp_at))
}

/// The spectral form of a segment's generator `A` (n×n, row-major): its eigenvalues `λ_i`
/// and Sylvester projectors `P_i = Π_{j≠i} (A − λ_j I)/(λ_i − λ_j)`, so that
/// `exp(A·h) = Σ_i e^{λ_i h} P_i` and `∫_0^h exp(A·s) ds = Σ_i φ(λ_i, h) P_i` with
/// `φ(λ, h) = (e^{λh} − 1)/λ`. Exact for a diagonalisable `A` with distinct eigenvalues,
/// in every number type: each eigenvalue holds the eigensolver's value (an acyclic
/// compartment's diagonal, exactly) and the exact implicit derivatives of its eigenpair
/// ([`spectral_of`]).
pub(super) struct Spectral<T> {
    pub(super) lambda: Vec<T>,
    /// `P_i` row-major, `n·n` each, concatenated.
    pub(super) proj: Vec<T>,
}

/// Relative eigenvalue separation below which the projectors' `1/(λ_i − λ_j)` amplifies
/// rounding past what Padé would deliver: a repeated or near-repeated eigenvalue (a
/// transit chain at one rate, `KA = CL/V`) takes the Padé path, which needs no
/// diagonalisation. A numerical-validity bound, checked again on the result below.
const SPECTRAL_MIN_REL_GAP: f64 = 1e-4;

/// Largest projector entry the spectral path accepts (see `spectral_of`).
const SPECTRAL_MAX_KAPPA: f64 = 32.0;
/// The projectors must reproduce `I = Σ P_i` and `A = Σ λ_i P_i` to this relative accuracy
/// (values and jets), or the segment takes Padé.
const SPECTRAL_CHECK_RTOL: f64 = 1e-11;
/// Chord steps for an eigenvalue's jets ([`spectral_of`]): each makes one more derivative
/// order exact, and the crate's deepest number type (the covariance step's nested dual)
/// carries four.
const SPECTRAL_JET_STEPS: usize = 5;

#[cfg(test)]
thread_local! {
    /// Test switch: propagate every segment by Padé, as before the spectral path.
    pub(crate) static SPECTRAL_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Segments the spectral path propagated on this thread.
    pub(super) static SPECTRAL_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
fn spectral_enabled() -> bool {
    #[cfg(test)]
    {
        !SPECTRAL_OFF.with(|c| c.get())
    }
    #[cfg(not(test))]
    {
        true
    }
}

pub(super) fn spectral<T: PkNum>(a: &[T], n: usize) -> Option<Spectral<T>> {
    if n == 0 || n > 6 {
        return None;
    }
    // The coupling graph: entry `r·n + c` is an edge unless it is a structural zero — value 0
    // and no derivative ([`crate::sens::linsolve::structural_zero`], the convention the Padé
    // kernels skip on). A rate whose value sits at 0 while it carries a jet is an edge: its
    // derivative couples the compartments even where its value does not.
    let present = a.iter().enumerate().fold(0u64, |m, (i, x)| {
        if crate::sens::linsolve::structural_zero(*x) {
            m
        } else {
            m | (1 << i)
        }
    });
    // The analysis runs on `A`'s values ([`spectral_analysis`]), as a host call: a recording
    // type (#2786) runs it again on every evaluation and re-checks its decisions there.
    let mut args: Vec<T> = a[..n * n].to_vec();
    args.push(T::from_f64(n as f64));
    args.push(T::from_f64(present as f64));
    let out = T::host(crate::sens::num::HostFn::Spectral, &args);
    if out[0].br_value() != 1.0 {
        return None;
    }
    spectral_of(a, n, &out)
}

/// The `f64` half of the spectral form — every decision and every quantity read off `A`'s
/// values (`args = [A (n·n), n, present]`) — into `out`, laid out as [`SpectralLayout`]: `ok`
/// (`1` when the spectral path applies, else `0` and nothing more), the eigenvalues, each one's
/// diagonal compartment (`−1` when it is not a diagonal), each projector's support masks, the
/// `f64` projectors, and per non-diagonal eigenvalue the chord system's `v₀` and `J₀⁻¹`.
pub(crate) fn spectral_analysis(args: &[f64], out: &mut Vec<f64>) {
    out.clear();
    let n = args[args.len() - 2] as usize;
    let present = args[args.len() - 1] as u64;
    let vals = &args[..n * n];
    match spectral_f64(vals, n, present) {
        None => out.push(0.0),
        Some(an) => {
            out.push(1.0);
            out.extend_from_slice(&an.lam);
            out.extend(an.exact.iter().map(|e| e.map_or(-1.0, |i| i as f64)));
            out.extend(an.masks.iter().flat_map(|&(r, c)| [r as f64, c as f64]));
            out.extend_from_slice(&an.pf);
            out.extend_from_slice(&an.v0);
            out.extend_from_slice(&an.jinv);
        }
    }
}

/// Where [`spectral_analysis`]'s outputs sit for an `n`-state generator.
struct SpectralLayout {
    n: usize,
}

impl SpectralLayout {
    fn lam(&self, k: usize) -> usize {
        1 + k
    }
    fn exact(&self, k: usize) -> usize {
        1 + self.n + k
    }
    fn mask(&self, k: usize) -> (usize, usize) {
        (1 + 2 * self.n + 2 * k, 2 + 2 * self.n + 2 * k)
    }
    fn pf(&self, i: usize) -> usize {
        1 + 4 * self.n + i
    }
    fn v0(&self, k: usize, r: usize) -> usize {
        1 + 4 * self.n + self.n * self.n * self.n + k * self.n + r
    }
    fn jinv(&self, k: usize, r: usize, c: usize) -> usize {
        let m = self.n + 1;
        1 + 4 * self.n + self.n * self.n * self.n + self.n * self.n + k * m * m + r * m + c
    }
}

/// [`spectral_analysis`]'s result.
struct SpectralF64 {
    lam: Vec<f64>,
    exact: Vec<Option<usize>>,
    masks: Vec<(u8, u8)>,
    pf: Vec<f64>,
    /// `v₀` per eigenvalue (`n` each; zeros for a diagonal one).
    v0: Vec<f64>,
    /// `J₀⁻¹` per eigenvalue (`(n+1)²` each, row-major; zeros for a diagonal one).
    jinv: Vec<f64>,
}

/// The eigenvalues of `[[a, b], [c, d]]`, when real, in a fixed order (the larger first when the
/// trace is positive, else the smaller): the larger-magnitude root from the stable sum
/// `m ± √disc`, the other from the determinant over it, so neither cancels. `None` for a
/// complex pair or a non-finite result.
fn eigenvalues_2x2(a: f64, b: f64, c: f64, d: f64) -> Option<(f64, f64)> {
    let m = 0.5 * (a + d);
    let h = 0.5 * (a - d);
    let disc = h * h + b * c;
    if !(disc > 0.0) {
        return None;
    }
    let s = disc.sqrt();
    let q = if m >= 0.0 { m + s } else { m - s };
    let det = a * d - b * c;
    let r = if q != 0.0 { det / q } else { return None };
    (q.is_finite() && r.is_finite()).then_some((q, r))
}

/// Whether two compartments that sit on no cycle of `A`'s coupling graph have (nearly) the same
/// diagonal. Such a compartment is a 1×1 diagonal block of `A`'s block-triangular form, so its
/// diagonal *is* an eigenvalue, exactly: a transit chain at one rate shows its repeated
/// eigenvalue here, without the eigensolve [`spectral_of`] would run only to decline it
/// (measured: that eigensolve made an all-declined transit fit 3% slower than Padé alone). The
/// gap is relative to the larger of the two diagonals, at most the spectral radius
/// `spectral_of` measures against, so this declines nothing `spectral_of` would accept.
fn repeated_singleton_eigenvalue(vals: &[f64], present: u64, n: usize) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= SPECTRAL_MIN_REL_GAP * x.abs().max(y.abs());
    // Only a pair of close diagonals can decline, so the coupling graph is walked only for one.
    if !(0..n).any(|i| (i + 1..n).any(|j| close(vals[i * n + i], vals[j * n + j]))) {
        return false;
    }
    let acyclic = acyclic_mask(present, n);
    let mut diag = [0.0_f64; 6];
    let mut m = 0;
    for i in (0..n).filter(|&i| acyclic & (1 << i) != 0) {
        diag[m] = vals[i * n + i];
        m += 1;
    }
    (0..m).any(|i| (i + 1..m).any(|j| close(diag[i], diag[j])))
}

/// `down[i]`: the compartments `i` reaches along `A`'s nonzero couplings, itself included
/// (Warshall over row bitmasks, `n ≤ 6`). Amount flows from column `c` to row `r` where
/// `A_rc ≠ 0`, so `r ∈ down[c]` is the reachability every power of `A` — and every function of
/// it, `exp(A·t)` and each projector included — is supported on.
fn downstream(present: u64, n: usize) -> [u8; 6] {
    let mut reach = [0u8; 6];
    for r in 0..n {
        for c in 0..n {
            if r != c && present & (1 << (r * n + c)) != 0 {
                reach[c] |= 1 << r;
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            if reach[i] & (1 << k) != 0 {
                reach[i] |= reach[k];
            }
        }
    }
    for (i, r) in reach.iter_mut().enumerate().take(n) {
        *r |= 1 << i;
    }
    reach
}

/// Bit `i` set when compartment `i` sits on no cycle of `A`'s coupling graph: a 1×1 block of
/// `A`'s block-triangular form, whose diagonal is therefore an eigenvalue, exactly.
fn acyclic_mask(present: u64, n: usize) -> u8 {
    let down = downstream(present, n);
    (0..n).fold(0u8, |m, i| {
        let cyclic = (0..n).any(|j| j != i && down[i] & (1 << j) != 0 && down[j] & (1 << i) != 0);
        if cyclic {
            m
        } else {
            m | (1 << i)
        }
    })
}

/// The Sylvester projectors `P_i = Π_{j≠i} (A − λ_j I)/(λ_i − λ_j)`, row-major and
/// concatenated. One body for the `f64` decision and the `T` result, so the `T` projectors'
/// values are the `f64` ones bit for bit: the same operations on the same values.
fn projectors<U: PkNum>(a: &[U], lambda: &[U], masks: &[(u8, u8)], n: usize) -> Vec<U> {
    let zero = U::from_f64(0.0);
    let one = U::from_f64(1.0);
    let mut proj = vec![zero; n * n * n];
    let mut tmp = vec![zero; n * n];
    for i in 0..n {
        let p = &mut proj[i * n * n..(i + 1) * n * n];
        for d in 0..n {
            p[d * n + d] = one;
        }
        for j in 0..n {
            if j == i {
                continue;
            }
            let inv = one / (lambda[i] - lambda[j]);
            // p ← p·(A − λ_j I)·inv
            for r in 0..n {
                for col in 0..n {
                    let mut s = zero;
                    for l in 0..n {
                        let alc = if l == col {
                            a[l * n + col] - lambda[j]
                        } else {
                            a[l * n + col]
                        };
                        s = s + p[r * n + l] * alc;
                    }
                    tmp[r * n + col] = s * inv;
                }
            }
            p.copy_from_slice(&tmp);
        }
        // Exact structural zeros, which the products above leave as rounding residue.
        let (rows, cols) = masks[i];
        for r in 0..n {
            for c in 0..n {
                if rows & (1 << r) == 0 || cols & (1 << c) == 0 {
                    p[r * n + c] = zero;
                }
            }
        }
    }
    proj
}

/// `A`'s spectral form, or `None` (the segment takes Padé). Decided in `f64` on `A`'s values
/// and its coupling graph (`present`: every entry but a structural zero): the eigenvalues,
/// the projectors' size and the identities they must reproduce never read a jet, so a
/// declined segment records nothing on a tape, and `f64` and every dual or tape type take the
/// same decision — unless an entry's value is 0 while it carries a derivative, which is an
/// edge for the dual run and not for the `f64` one.
fn spectral_f64(vals: &[f64], n: usize, present: u64) -> Option<SpectralF64> {
    // A host analysis runs before a compiled plan's guards are read, so it must end on any
    // input: nalgebra's Schur iteration has no cap and never converges on a `NaN`/`inf`.
    if vals.iter().any(|x| !x.is_finite()) {
        return None;
    }
    if repeated_singleton_eigenvalue(vals, present, n) {
        return None;
    }
    let scale = vals.iter().fold(0.0_f64, |s, x| s.max(x.abs())).max(1e-300);
    // Each eigenvalue belongs to one diagonal block of `A`'s block-triangular form: a strongly
    // connected set of compartments, and `A`'s spectrum is the union of the blocks' — so the
    // blocks are solved alone, never `A` whole. An acyclic compartment is a 1×1 block, its
    // diagonal an eigenvalue exactly: using the exact bits makes `A − λI` zero that
    // compartment's row (measured: a depot decayed to 1e-6 of its dose otherwise read 5.5e-10
    // off, the other modes leaking `ε·‖x‖` into it). A 2×2 block has a closed form
    // ([`eigenvalues_2x2`]); a larger one goes to the eigensolver, which sees a block and not
    // the whole matrix. The block fixes the projector's support — rows downstream of it,
    // columns upstream — which the products then keep as exact zeros rather than rounding
    // residue. (Entries a projector's support does include can still carry the modes'
    // algebraic cancellation: a stiff chain's `∂²u₁/∂k₂²`, exactly 0, reads ~1e-12 through
    // mode 0's `(A − λ₂I)/(λ₀ − λ₂)`.) `exact[k]`: the compartment eigenvalue `k` is the
    // diagonal of.
    //
    // The eigenvalues come in block order — the coupling graph's, fixed by the model — rather
    // than by value: two that cross as the parameters move keep their indices, so every
    // per-mode decision (`exact`, the masks) is the same at both points and a compiled plan's
    // checks on them hold (#2786).
    let down = downstream(present, n);
    let mut lam_f: Vec<f64> = Vec::with_capacity(n);
    let mut exact: Vec<Option<usize>> = Vec::with_capacity(n);
    let mut block: Vec<u8> = Vec::with_capacity(n);
    let mut done = 0u8;
    for i in 0..n {
        if done & (1 << i) != 0 {
            continue;
        }
        let scc = (0..n).fold(0u8, |m, j| {
            if down[i] & (1 << j) != 0 && down[j] & (1 << i) != 0 {
                m | (1 << j)
            } else {
                m
            }
        });
        done |= scc;
        let members: Vec<usize> = (0..n).filter(|&j| scc & (1 << j) != 0).collect();
        match members.len() {
            1 => {
                lam_f.push(vals[i * n + i]);
                exact.push(Some(i));
                block.push(scc);
            }
            2 => {
                let (p, q) = (members[0], members[1]);
                let (l0, l1) = eigenvalues_2x2(
                    vals[p * n + p],
                    vals[p * n + q],
                    vals[q * n + p],
                    vals[q * n + q],
                )?;
                for l in [l0, l1] {
                    lam_f.push(l);
                    exact.push(None);
                    block.push(scc);
                }
            }
            _ => {
                let sub =
                    nalgebra::DMatrix::<f64>::from_fn(members.len(), members.len(), |r, c| {
                        vals[members[r] * n + members[c]]
                    });
                for z in sub.complex_eigenvalues().iter() {
                    if !(z.re.is_finite() && z.im.is_finite()) || z.im.abs() > 1e-12 * scale {
                        return None;
                    }
                    lam_f.push(z.re);
                    exact.push(None);
                    block.push(scc);
                }
            }
        }
    }
    if lam_f.len() != n || !lam_f.iter().all(|l| l.is_finite()) {
        return None;
    }
    let lmax = lam_f
        .iter()
        .fold(0.0_f64, |s, x| s.max(x.abs()))
        .max(scale * 1e-12);
    let mut sorted = lam_f.clone();
    sorted.sort_by(|x, y| x.total_cmp(y));
    for w in sorted.windows(2) {
        if (w[1] - w[0]).abs() < SPECTRAL_MIN_REL_GAP * lmax {
            return None;
        }
    }
    let masks: Vec<(u8, u8)> = (0..n)
        .map(|k| {
            let b = block[k];
            let rows = (0..n)
                .filter(|&s| b & (1 << s) != 0)
                .fold(0u8, |m, s| m | down[s]);
            let cols = (0..n).fold(0u8, |m, c| if down[c] & b != 0 { m | (1 << c) } else { m });
            (rows, cols)
        })
        .collect();
    let pf = projectors(vals, &lam_f, &masks, n);
    // The projectors' size bounds how much they amplify rounding (×κ in values, ×κ² in a
    // second-order jet); Padé's error does not depend on it. A compartmental generator has
    // κ ≈ 1 (measured 0.88–1.27 on 2- and 3-state models); a transit chain is strongly
    // non-normal (94 to 1.9e5 on a 5-state one, whose order-2 jets then sat 1.2e-10 off
    // Padé's against 6.8e-12), so past `SPECTRAL_MAX_KAPPA` the segment takes Padé. A
    // non-finite entry declines too (`f64::max` would drop a NaN).
    if !pf.iter().all(|x| x.is_finite()) {
        return None;
    }
    let kappa = pf.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if !(kappa <= SPECTRAL_MAX_KAPPA) {
        return None;
    }
    // Validity: Σ P_i = I and Σ λ_i P_i = A.
    for r in 0..n {
        for col in 0..n {
            let (mut sp, mut sa) = (0.0, 0.0);
            for i in 0..n {
                let pij = pf[i * n * n + r * n + col];
                sp += pij;
                sa += lam_f[i] * pij;
            }
            let id = if r == col { 1.0 } else { 0.0 };
            if !((sp - id).abs() <= SPECTRAL_CHECK_RTOL
                && (sa - vals[r * n + col]).abs() <= SPECTRAL_CHECK_RTOL * scale)
            {
                return None;
            }
        }
    }
    let m = n + 1;
    let mut v0s = vec![0.0; n * n];
    let mut jinvs = vec![0.0; n * m * m];
    for (k, &l0) in lam_f.iter().enumerate() {
        if exact[k].is_some() {
            continue;
        }
        let pk = &pf[k * n * n..(k + 1) * n * n];
        let col = (0..n)
            .max_by(|&x, &y| {
                let nx: f64 = (0..n).map(|r| pk[r * n + x] * pk[r * n + x]).sum();
                let ny: f64 = (0..n).map(|r| pk[r * n + y] * pk[r * n + y]).sum();
                nx.total_cmp(&ny)
            })
            .unwrap_or(0);
        let norm = (0..n).map(|r| pk[r * n + col].powi(2)).sum::<f64>().sqrt();
        if !(norm > 0.0 && norm.is_finite()) {
            return None;
        }
        let v0: Vec<f64> = (0..n).map(|r| pk[r * n + col] / norm).collect();
        let jac = nalgebra::DMatrix::<f64>::from_fn(m, m, |r, c| match (r < n, c < n) {
            (true, true) => vals[r * n + c] - if r == c { l0 } else { 0.0 },
            (true, false) => -v0[r],
            (false, true) => v0[c],
            (false, false) => 0.0,
        });
        let jinv = jac.try_inverse()?;
        if !jinv.iter().all(|x| x.is_finite()) {
            return None;
        }
        v0s[k * n..(k + 1) * n].copy_from_slice(&v0);
        for r in 0..m {
            for c in 0..m {
                jinvs[k * m * m + r * m + c] = jinv[(r, c)];
            }
        }
    }
    Some(SpectralF64 {
        lam: lam_f,
        exact,
        masks,
        pf,
        v0: v0s,
        jinv: jinvs,
    })
}

/// The spectral form in `T` from [`spectral_analysis`]'s outputs `out` (each a `T`: a
/// constant, or a recording type's host-call result). The eigenvalue of an acyclic
/// compartment is its diagonal, exact in value and in every jet; any other carries the
/// analysis's value and the exact implicit derivatives of its eigenpair: each eigenpair
/// `(λ, v)` solves `(A − λI)v = 0, v₀ᵀv = 1`, with `v₀` the `f64` eigenvector (`P_k`'s largest
/// column). A chord step `z ← z − J₀⁻¹F(z)`, with `J₀` that system's exact `f64` Jacobian at
/// the eigensolver's `(λ, v₀)`, makes one more derivative order exact; its value part is
/// dropped ([`PkNum::value_only`]), so `λ` keeps the eigensolver's value bit for bit. No
/// characteristic polynomial is formed: its coefficients cancel on a stiff spectrum (rates
/// 10, 0.5, 0.005 left the slowest eigenvalue's jets 4e-9 off Padé).
fn spectral_of<T: PkNum>(a: &[T], n: usize, out: &[T]) -> Option<Spectral<T>> {
    let lay = SpectralLayout { n };
    let exact: Vec<Option<usize>> = (0..n)
        .map(|k| {
            let e = out[lay.exact(k)].br_value();
            (e >= 0.0).then_some(e as usize)
        })
        .collect();
    let masks: Vec<(u8, u8)> = (0..n)
        .map(|k| {
            let (r, c) = lay.mask(k);
            (out[r].br_value() as u8, out[c].br_value() as u8)
        })
        .collect();
    if a.iter().all(|x| x.is_constant()) {
        return Some(Spectral {
            lambda: (0..n).map(|k| out[lay.lam(k)]).collect(),
            proj: (0..n * n * n).map(|i| out[lay.pf(i)]).collect(),
        });
    }
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let m = n + 1;
    let mut lambda: Vec<T> = Vec::with_capacity(n);
    let mut v: Vec<T> = vec![zero; n];
    let mut fz: Vec<T> = vec![zero; m];
    for k in 0..n {
        if let Some(i) = exact[k] {
            // The diagonal itself: exact in value and in every jet.
            lambda.push(a[i * n + i]);
            continue;
        }
        for (r, vi) in v.iter_mut().enumerate() {
            *vi = out[lay.v0(k, r)];
        }
        let mut l = out[lay.lam(k)];
        // Step `s` is gated at order `s` (`PkNum::order_gate`): a recording's compiled
        // variant runs the steps its order needs (a reverse one more), a value forward none.
        for step in 1..=SPECTRAL_JET_STEPS {
            for r in 0..n {
                let mut acc = zero - l * v[r];
                for c in 0..n {
                    acc = acc + a[r * n + c] * v[c];
                }
                fz[r] = acc;
            }
            let mut dot = zero - one;
            for c in 0..n {
                dot = dot + out[lay.v0(k, c)] * v[c];
            }
            fz[n] = dot;
            for r in 0..m {
                let mut d = zero;
                for c in 0..m {
                    d = d + out[lay.jinv(k, r, c)] * fz[c];
                }
                let d = d - d.value_only();
                let g = step as u8;
                if r < n {
                    v[r] = T::order_gate(g, v[r], v[r] - d);
                } else {
                    l = T::order_gate(g, l, l - d);
                }
            }
        }
        lambda.push(l);
    }
    let proj = projectors(a, &lambda, &masks, n);
    Some(Spectral { lambda, proj })
}

/// `1/j` for `j < 48`, so the series below divides by no runtime value.
const RECIP: [f64; 48] = {
    let mut r = [0.0; 48];
    let mut j = 1;
    while j < 48 {
        r[j] = 1.0 / j as f64;
        j += 1;
    }
    r
};

/// The order-`m` derivative of `g(z) = (e^z − 1)/z` alone (`m ≤ 4`): what a compiled plan's or a
/// recording's `ExpRel(m)` reads — one accumulator, where [`exprel_derivs`] carries five — and
/// the same bits as [`exprel_derivs`]`(z, 4)[m]`.
pub(crate) fn exprel_k(z: f64, m: usize) -> f64 {
    debug_assert!(m <= 4);
    // Order 0's closed form `expm1(z)/z` cancels nowhere (`expm1` is accurate at small `z`):
    // it is the whole function but its removable point `z = 0`, where the series' one term is
    // the limit `1`.
    if m == 0 && z != 0.0 {
        return z.exp_m1() / z;
    }
    if z.abs() <= 2.0 {
        // `exprel_derivs`'s series for one order: the same terms, in the same order, with the
        // same stop (its `top = 4` bound), so the bits equal its entry `m`. Over a fixed range
        // with `m ≤ 4` known, so the loop carries no bounds check.
        let m = m.min(4);
        let (mut p, mut o) = (1.0, 0.0);
        for k in 0..43 {
            o += p * RECIP[k + m + 1];
            p *= z * RECIP[k + 1];
            if p.abs() < 1e-19 {
                break;
            }
        }
        return o;
    }
    exprel_closed(z, z.exp(), m)
}

/// `g(z) = (e^z − 1)/z` and its derivatives up to `orders` (≤ 4), the rest left 0, so
/// `φ(λ, Δ) = Δ·g(λΔ)` is lifted exactly through the jet orders a number type carries
/// ([`PkNum::lift4`]). Near `z = 0` the closed forms cancel (`g⁗` loses ~7 digits at `z = 0.1`),
/// so `|z| ≤ 2` sums `g⁽ᵐ⁾(z) = Σ_k (zᵏ/k!)/(k + m + 1)` in one pass over `k`; past it the
/// closed forms lose at most ~2. Each order accumulates on its own, so `g` is the same bits
/// whatever `orders` asks for: the `f64` run and a dual run read one value.
pub(crate) fn exprel_derivs(z: f64, orders: usize) -> [f64; 5] {
    let mut out = [0.0; 5];
    let top = orders.min(4);
    if z.abs() <= 2.0 {
        let mut p = 1.0; // zᵏ/k!
        let mut k = 0;
        loop {
            for (m, o) in out.iter_mut().enumerate().take(top + 1) {
                *o += p * RECIP[k + m + 1];
            }
            k += 1;
            p *= z * RECIP[k];
            // |z| ≤ 2: the next terms sum below 2·|p|, under an ulp of every order's value
            // (each is at least g⁽ᵐ⁾(−2) > 0.03).
            if p.abs() < 1e-19 || k + top + 1 >= RECIP.len() {
                break;
            }
        }
        if z != 0.0 {
            out[0] = z.exp_m1() / z;
        }
        return out;
    }
    out[0] = z.exp_m1() / z;
    let e = if top > 0 { z.exp() } else { 0.0 };
    for (m, o) in out.iter_mut().enumerate().take(top + 1).skip(1) {
        *o = exprel_closed(z, e, m);
    }
    out
}

/// `exprel`'s order-`m` closed form at `|z| > 2` (`1 ≤ m ≤ 4`; order 0 is `expm1(z)/z`, which
/// needs no `e`), `e = eᶻ`. Past `z ≈ −745` the exponential is 0 and the polynomials it
/// multiplies may overflow (`0·inf = NaN`): the term is dropped, leaving each closed form's
/// constant over `zᵐ⁺¹`.
fn exprel_closed(z: f64, e: f64, m: usize) -> f64 {
    let (z2, z3, z4) = (z * z, z * z * z, z * z * z * z);
    let e_term = |poly: f64| if e == 0.0 { 0.0 } else { e * poly };
    match m {
        1 => (e_term(z - 1.0) + 1.0) / z2,
        2 => (e_term(z2 - 2.0 * z + 2.0) - 2.0) / z3,
        3 => (e_term(z3 - 3.0 * z2 + 6.0 * z - 6.0) + 6.0) / z4,
        _ => (e_term(z4 - 4.0 * z3 + 12.0 * z2 - 24.0 * z + 24.0) - 24.0) / (z4 * z),
    }
}

/// [`linear_segment_g`] on a reused workspace: every buffer a segment needs, so a fit's
/// thousands of segments allocate their outputs and nothing else.
fn segment_in<T: PkNum>(
    ws: &mut LinWs<T>,
    rhs: &crate::ode::solver::RhsG<'_, T>,
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
) -> Option<(Vec<SolPointG<T>>, Vec<SolPointG<T>>)> {
    // The walk's moving span (`PkNum::span_hint`), or the values; handed back on a decline,
    // so the integration the walk falls back to takes it.
    let span = T::take_span();
    let r = segment_span(ws, rhs, u0, t_span, span, params, saveat, interp_at);
    if r.is_none() {
        if let Some((a, b)) = span {
            T::span_hint(a, b);
        }
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn segment_span<T: PkNum>(
    ws: &mut LinWs<T>,
    rhs: &crate::ode::solver::RhsG<'_, T>,
    u0: &[T],
    t_span: (f64, f64),
    span: Option<(T, T)>,
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
) -> Option<(Vec<SolPointG<T>>, Vec<SolPointG<T>>)> {
    use crate::sens::num::BranchCmp::{Eq, Gt, Lt};
    let n = u0.len();
    let (t0, t1) = t_span;
    let h = t1 - t0;
    let (t0_t, t1_t) = span.unwrap_or((T::from_f64(t0), T::from_f64(t1)));
    let h_t = t1_t - t0_t;
    if !h_t.br_cmp_c(Gt, 0.0) || !h_t.br_abs_lt(f64::INFINITY) {
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
    //
    // A recording type (#2786) reads them off its recording instead, exactly
    // ([`PkNum::affine_parts`]).
    if !T::affine_parts(rhs, params, t_mid, a, b) {
        rhs(probe, params, T::from_f64(t_mid), b);
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
            rhs(probe, params, T::from_f64(t_mid), col);
            probe[j] = zero;
            for i in 0..n {
                a[i * n + j] = T::scaled_diff(col[i], b[i], 1.0 / scale);
            }
        }
    }
    // `|x| < ∞`: the value finite (a `NaN` fails it too).
    if a.iter()
        .chain(b.iter())
        .any(|x| !x.br_abs_lt(f64::INFINITY))
    {
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
        rhs(x, params_d, T::from_f64(t), direct);
        (0..n).all(|i| {
            // A recording type (#2786) checks the values through its branch API, as the same
            // sums: its `A` is exact, so there is no jet cancellation to catch.
            if T::RECORDS_BRANCHES {
                let mut assembled = b[i];
                let mut scale = b[i].abs();
                for j in 0..n {
                    let (aij, xj) = (a[i * n + j], x[j]);
                    assembled = assembled + aij * xj;
                    scale = scale + (aij * xj).abs();
                }
                let tol = scale * T::from_f64(SELF_CHECK_RTOL) + T::from_f64(1e-300);
                return direct[i].br_abs_lt(f64::INFINITY)
                    && (direct[i] - assembled)
                        .abs()
                        .br_cmp(crate::sens::num::BranchCmp::Le, tol);
            }
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

    // Every requested time, clamped into the span, visited once in ascending order; `pos`
    // maps each request (`saveat`, then `interp_at`) to its time. The last `saveat` time is
    // the span's end, which moves with it.
    let n_save = saveat.len();
    let req: Vec<T> = saveat
        .iter()
        .chain(interp_at)
        .enumerate()
        .map(|(i, &t)| {
            let tt = if span.is_some() && i + 1 == n_save && t.to_bits() == t1.to_bits() {
                t1_t
            } else {
                T::from_f64(t)
            };
            if tt.br_cmp(Lt, t0_t) {
                t0_t
            } else if tt.br_cmp(Gt, t1_t) {
                t1_t
            } else {
                tt
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..req.len()).collect();
    if T::RECORDS_BRANCHES {
        order.sort_by(|&i, &j| {
            if req[i].br_cmp(Lt, req[j]) {
                std::cmp::Ordering::Less
            } else if req[j].br_cmp(Lt, req[i]) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    } else {
        order.sort_by(|&i, &j| req[i].val().total_cmp(&req[j].val()));
    }
    times.clear();
    let mut pos = vec![0usize; req.len()];
    for (k, &i) in order.iter().enumerate() {
        if k == 0 || !req[i].br_cmp(Eq, *times.last().expect("one pushed")) {
            times.push(req[i]);
        }
        pos[i] = times.len() - 1;
    }

    // ── Spectral fast path (#2706 follow-up): a diagonalisable `A` with real, distinct
    // eigenvalues — every 1/2/3-compartment disposition with first-order absorption — is
    // propagated as `Σ_i P_i (e^{λ_i Δ} x + φ(λ_i, Δ) b)`: `n` exponentials per gap instead of
    // a degree-13 Padé and its squarings. A repeated, near-repeated or complex spectrum (a
    // transit chain at one rate) declines to Padé below. Whether to take it is decided on
    // `A`'s values and coupling graph ([`spectral_of`]); each propagated state then passes the
    // same reproduction check as Padé's, which reads jets too, so a state that fails it
    // mid-segment sends that run (with the records it has made) to Padé, possibly on one
    // number type and not another — the two exact propagations agree to round-off.
    if spectral_enabled() {
        if let Some(sp) = spectral(a, n) {
            // The derivative orders `φ` needs: none on a constant generator; else the type's
            // (every type in the crate carries at most 4; one that reports none gets all).
            let jet_orders = if a.iter().all(|v| v.is_constant()) {
                0
            } else if T::JET_ORDER == 0 {
                4
            } else {
                T::JET_ORDER
            };
            // `P_i b`, once per segment.
            let mut pb = vec![zero; n * n];
            for k in 0..n {
                let pk = &sp.proj[k * n * n..(k + 1) * n * n];
                for r in 0..n {
                    let mut acc = zero;
                    if T::FUSED_FOLDS {
                        pairs.clear();
                        pairs.extend((0..n).map(|col| (pk[r * n + col], b[col])));
                        acc = T::dot_fold(acc, pairs);
                    } else {
                        for col in 0..n {
                            acc = acc + pk[r * n + col] * b[col];
                        }
                    }
                    pb[k * n + r] = acc;
                }
            }
            states.clear();
            x.clear();
            x.extend_from_slice(u0);
            let mut t_prev = t0_t;
            let mut ok = true;
            for &t in times.iter() {
                // One step a chunk (#2786): every step of every segment is the same code, so
                // a recording type shares one compiled step whatever the sampling design.
                T::chunk_mark();
                let dt_t = t - t_prev;
                if dt_t.br_cmp_c(Gt, 0.0) {
                    for v in next.iter_mut().take(n) {
                        *v = zero;
                    }
                    for k in 0..n {
                        let lam = sp.lambda[k];
                        let z = lam * dt_t;
                        let e = z.exp();
                        // φ(λ, Δ) = (e^{λΔ} − 1)/λ = Δ·g(λΔ), lifted exactly ([`exprel_derivs`]).
                        let phi = dt_t * z.exprel(jet_orders);
                        let pk = &sp.proj[k * n * n..(k + 1) * n * n];
                        for r in 0..n {
                            // One tape node per entry on a taped type (`PkNum::dot_fold`), as
                            // the Padé path's products.
                            let px = if T::FUSED_FOLDS {
                                pairs.clear();
                                pairs.extend((0..n).map(|col| (pk[r * n + col], x[col])));
                                T::dot_fold(zero, pairs)
                            } else {
                                let mut px = zero;
                                for col in 0..n {
                                    px = px + pk[r * n + col] * x[col];
                                }
                                px
                            };
                            next[r] = next[r] + e * px + phi * pb[k * n + r];
                        }
                    }
                    std::mem::swap(x, next);
                    t_prev = t;
                    // A recording type's generator is exact and its RHS proven affine
                    // ([`PkNum::affine_parts`]): there is nothing for a propagated state to
                    // catch.
                    if !T::RECORDS_BRANCHES && !reproduces(x, t_mid, direct) {
                        ok = false;
                        break;
                    }
                }
                states.extend_from_slice(x);
            }
            if ok {
                #[cfg(test)]
                SPECTRAL_HITS.with(|c| c.set(c.get() + 1));
                return Some(points(saveat, interp_at, &pos, states, n));
            }
        }
    }

    // ── Augmented generator M = [[A, b/c], [0, 0]], (n+1)×(n+1); the propagated forcing
    // column is multiplied back by `c` below. Without `c` a forcing far above the rates sets
    // ‖M‖, and with it the squaring count: `du/dt = 1e20 − u` over unit time needs ~64
    // squarings and was declined. Scaling the column to the rates' magnitude leaves the
    // squaring count to `A`, where it belongs; `c` is a plain constant, so the jets are exact.
    // Magnitudes only scale: `c` is a plain constant, any value of it exact.
    let (c, inv_c) = if T::RECORDS_BRANCHES {
        // A recording type (#2786) reads `c` off the values at run time ([`forcing_scale`]):
        // data, so the jets stay exact, and a plan's result is the same whatever point it
        // was recorded at — a scale fixed at recording would round differently elsewhere.
        let mut args: Vec<T> = a[..n * n].to_vec();
        args.extend_from_slice(&b[..n]);
        args.push(T::from_f64(n as f64));
        let c = T::host(crate::sens::num::HostFn::ForcingScale, &args)[0];
        (c, T::from_f64(1.0) / c)
    } else {
        let a_max = a.iter().fold(0.0_f64, |acc, x| acc.max(x.val().abs()));
        // Jets included: a forcing whose value is 0 but whose derivative is 1e292 leaves the
        // generator's value at zero (no squarings) while the Padé numerator multiplies that
        // jet by ~3e16 and overflows it.
        let b_max = b.iter().fold(0.0_f64, |acc, x| acc.max(x.magnitude_hint()));
        let c = (b_max / a_max.max(1.0)).max(1.0);
        (T::from_f64(c), T::from_f64(1.0 / c))
    };
    for i in 0..n {
        for j in 0..n {
            gen[i * m + j] = a[i * n + j];
        }
        gen[i * m + n] = b[i] * inv_c;
    }

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
    let mut t_prev = t0_t;
    for (step, &t) in times.iter().enumerate() {
        T::chunk_mark();
        let dt_t = t - t_prev;
        if dt_t.br_cmp_c(Gt, 0.0) {
            // A gap that moves with the inputs is its own exponential.
            let key = if dt_t.is_constant() {
                dt_t.val().to_bits()
            } else {
                u64::MAX - step as u64
            };
            let idx = match props[..*n_props].iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    // Each exponential its own chunk (#2786): its code depends only on the
                    // squaring count, so a recording type shares it across steps and segments.
                    T::chunk_mark();
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
            // The step itself: the same code whether or not it paid for a new exponential.
            T::chunk_mark();
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
            if !T::RECORDS_BRANCHES && !reproduces(x, t_mid, direct) {
                return None;
            }
        }
        states.extend_from_slice(x);
    }

    Some(points(saveat, interp_at, &pos, states, n))
}

/// The save points of a propagation: request `i` (`saveat`, then `interp_at`) reads the state
/// at its time's position `pos[i]` in `states` (`n` per time).
fn points<T: PkNum>(
    saveat: &[f64],
    interp_at: &[f64],
    pos: &[usize],
    states: &[T],
    n: usize,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    let at = |ts: &[f64], base: usize| -> Vec<SolPointG<T>> {
        ts.iter()
            .enumerate()
            .map(|(k, &t)| {
                let i = pos[base + k];
                SolPointG {
                    t,
                    u: states[i * n..(i + 1) * n].to_vec(),
                }
            })
            .collect()
    };
    (at(saveat, 0), at(interp_at, saveat.len()))
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
    times: Vec<T>,
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
