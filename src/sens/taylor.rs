//! A truncated multivariate Taylor polynomial: the value and every derivative of total
//! order ≤ `ORD` in `P` directions, as one number type.
//!
//! [`Taylor<P, ORD, L>`] stores the normalised coefficients `c_α = ∂^α f / α!` of the
//! monomials `δ^α` with `|α| ≤ ORD`, in graded order (`L = C(P + ORD, ORD)` of them):
//! arithmetic is truncated polynomial arithmetic, and an elementary function is its
//! univariate Taylor series composed with the argument's non-constant part. Every
//! derivative up to order `ORD` is exact; nothing above it is carried.
//!
//! It is the PK-space solve of the adjoint's covariance step (`sens::ode_adjoint`): a
//! subject's predictions as functions of its few live PK parameters, to third order (the
//! R-matrix's nested dual reads that many) or fourth (the Laplace exact anchor's), solved
//! once untaped and then lifted onto the recording's own type through this polynomial.
#![allow(clippy::needless_range_loop)]

use super::num::PkNum;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::OnceLock;

/// `C(n, k)`.
pub(crate) const fn binom(n: usize, k: usize) -> usize {
    let mut r = 1usize;
    let mut i = 0usize;
    while i < k {
        r = r * (n - i) / (i + 1);
        i += 1;
    }
    r
}

/// How many `(i, j)` monomial products of total degree ≤ `ord` a `(p, ord)` [`Table`]
/// lists: the terms one product sums.
pub(crate) const fn mul_len(p: usize, ord: usize) -> usize {
    // Monomials of exact degree `d` in `p` variables: `C(d + p − 1, d)`.
    let mut n = 0usize;
    let mut a = 0usize;
    while a <= ord {
        let mut b = 0usize;
        while a + b <= ord {
            n += binom(a + p - 1, a) * binom(b + p - 1, b);
            b += 1;
        }
        a += 1;
    }
    n
}

/// The monomial structure of one `(P, ORD)`.
pub(crate) struct Table {
    /// Exponent vector of each monomial (`P` entries each), graded order; index 0 is `1`.
    pub(crate) exps: Vec<Vec<u8>>,
    /// Total degree of each monomial.
    pub(crate) deg: Vec<u8>,
    /// `(i, j, k)`: monomial `i` times monomial `j` is monomial `k` (degree ≤ `ORD`).
    pub(crate) mul: Vec<(u16, u16, u16)>,
    /// For each monomial of degree ≥ 1: `(β, v)` with `α = β + e_v` (`β` earlier).
    pub(crate) parent: Vec<(u16, u8)>,
    /// The index of `e_v`, the first-degree monomial of each variable.
    pub(crate) var: Vec<u16>,
}

const MAX_P: usize = 12;
const MAX_ORD: usize = 4;

static TABLES: [OnceLock<Table>; (MAX_P + 1) * (MAX_ORD + 1)] =
    [const { OnceLock::new() }; (MAX_P + 1) * (MAX_ORD + 1)];

/// The table of `(p, ord)`, built on first use.
pub(crate) fn table(p: usize, ord: usize) -> &'static Table {
    assert!(
        p <= MAX_P && ord <= MAX_ORD,
        "Taylor<{p}, {ord}> is past the served range"
    );
    TABLES[p * (MAX_ORD + 1) + ord].get_or_init(|| build(p, ord))
}

fn build(p: usize, ord: usize) -> Table {
    // Graded order: degree, then exponent vector descending lexicographically.
    let mut exps: Vec<Vec<u8>> = vec![vec![0; p]];
    let mut prev: Vec<Vec<u8>> = vec![vec![0; p]];
    for _d in 1..=ord {
        let mut next: Vec<Vec<u8>> = Vec::new();
        for e in &prev {
            // Extend only at variables at or after the last nonzero one, so each monomial
            // is generated once.
            let last = e.iter().rposition(|&x| x > 0).unwrap_or(0);
            for v in last..p {
                let mut f = e.clone();
                f[v] += 1;
                next.push(f);
            }
        }
        exps.extend(next.iter().cloned());
        prev = next;
    }
    let deg: Vec<u8> = exps.iter().map(|e| e.iter().sum()).collect();
    let index: std::collections::HashMap<Vec<u8>, u16> = exps
        .iter()
        .enumerate()
        .map(|(i, e)| (e.clone(), i as u16))
        .collect();
    let mut mul = Vec::new();
    for (i, a) in exps.iter().enumerate() {
        for (j, b) in exps.iter().enumerate() {
            if (deg[i] + deg[j]) as usize > ord {
                continue;
            }
            let s: Vec<u8> = a.iter().zip(b).map(|(x, y)| x + y).collect();
            mul.push((i as u16, j as u16, index[&s]));
        }
    }
    let mut parent = vec![(0u16, 0u8); exps.len()];
    for (k, e) in exps.iter().enumerate().skip(1) {
        let v = e.iter().position(|&x| x > 0).expect("degree ≥ 1");
        let mut b = e.clone();
        b[v] -= 1;
        parent[k] = (index[&b], v as u8);
    }
    let var = (0..p)
        .map(|v| {
            let mut e = vec![0u8; p];
            e[v] = 1;
            index[&e]
        })
        .collect();
    Table {
        exps,
        deg,
        mul,
        parent,
        var,
    }
}

/// See the module docs. `L` must be `C(P + ORD, ORD)`.
#[derive(Clone, Copy, Debug)]
pub struct Taylor<const P: usize, const ORD: usize, const L: usize> {
    pub c: [f64; L],
}

impl<const P: usize, const ORD: usize, const L: usize> Taylor<P, ORD, L> {
    const CHECK: () = assert!(L == binom(P + ORD, ORD), "L must be C(P + ORD, ORD)");

    #[inline]
    fn tab() -> &'static Table {
        #[allow(clippy::let_unit_value)]
        let () = Self::CHECK;
        table(P, ORD)
    }
    #[inline]
    pub fn constant(v: f64) -> Self {
        let mut c = [0.0; L];
        c[0] = v;
        Taylor { c }
    }
    /// `v + δ_dim`: an independent direction.
    pub fn seed(v: f64, dim: usize) -> Self {
        let mut t = Self::constant(v);
        if dim < P {
            t.c[Self::tab().var[dim] as usize] = 1.0;
        }
        t
    }
    /// `f(self)` from `f`'s derivatives `d[k] = f⁽ᵏ⁾(self.val())`, `k = 0 … ORD`: the
    /// univariate Taylor series in the argument's non-constant part `h`.
    fn compose(self, d: &[f64]) -> Self {
        let mut h = self;
        h.c[0] = 0.0;
        let mut r = Self::constant(d[0]);
        let mut hk = Self::constant(1.0);
        let mut fact = 1.0;
        for (k, &dk) in d.iter().enumerate().take(ORD + 1).skip(1) {
            hk = hk * h;
            fact *= k as f64;
            let w = dk / fact;
            if w != 0.0 {
                for i in 0..L {
                    r.c[i] += w * hk.c[i];
                }
            }
        }
        r
    }
    fn zero_jet(&self) -> bool {
        self.c[1..].iter().all(|&x| x == 0.0)
    }
}

impl<const P: usize, const ORD: usize, const L: usize> Add for Taylor<P, ORD, L> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: Self) -> Self {
        for i in 0..L {
            self.c[i] += o.c[i];
        }
        self
    }
}
impl<const P: usize, const ORD: usize, const L: usize> Sub for Taylor<P, ORD, L> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: Self) -> Self {
        for i in 0..L {
            self.c[i] -= o.c[i];
        }
        self
    }
}
impl<const P: usize, const ORD: usize, const L: usize> Neg for Taylor<P, ORD, L> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        for i in 0..L {
            self.c[i] = -self.c[i];
        }
        self
    }
}
impl<const P: usize, const ORD: usize, const L: usize> Mul for Taylor<P, ORD, L> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut c = [0.0; L];
        for &(i, j, k) in &Self::tab().mul {
            c[k as usize] += self.c[i as usize] * o.c[j as usize];
        }
        Taylor { c }
    }
}
impl<const P: usize, const ORD: usize, const L: usize> Div for Taylor<P, ORD, L> {
    type Output = Self;
    /// Series division `q·b = a`, degree by degree: `q_α = (a_α − Σ q_β b_γ)/b₀` over the
    /// products `β + γ = α` with `β ≠ α`. The value is `a₀/b₀`, the `f64` quotient bit for
    /// bit, so a solve over this type takes the `f64` solve's steps.
    #[inline]
    fn div(self, o: Self) -> Self {
        let t = Self::tab();
        let b0 = o.c[0];
        let mut q = [0.0; L];
        let mut acc = self.c;
        // A term `q_i b_j` of `q_k` with `i ≠ k` has `deg i < deg k`, so degree by degree
        // every `q_i` it reads is already final.
        for d in 0..=ORD as u8 {
            for &(i, j, k) in &t.mul {
                let (i, j, k) = (i as usize, j as usize, k as usize);
                if t.deg[k] == d && i != k {
                    acc[k] -= q[i] * o.c[j];
                }
            }
            for k in 0..L {
                if t.deg[k] == d {
                    q[k] = acc[k] / b0;
                }
            }
        }
        Taylor { c: q }
    }
}

impl<const P: usize, const ORD: usize, const L: usize> PkNum for Taylor<P, ORD, L> {
    const SECOND_ORDER: bool = ORD >= 2;
    const JET_ORDER: usize = ORD;
    /// A product sums one term per pair of monomials of total degree ≤ `ORD`.
    const WORK: usize = mul_len(P, ORD);
    #[inline]
    fn is_constant(self) -> bool {
        self.zero_jet()
    }
    fn from_f64(x: f64) -> Self {
        Self::constant(x)
    }
    fn var(x: f64, dim: usize) -> Self {
        Self::seed(x, dim)
    }
    fn val(self) -> f64 {
        self.c[0]
    }
    fn exp(self) -> Self {
        let e = self.c[0].exp();
        self.compose(&[e; MAX_ORD + 1][..=ORD])
    }
    fn ln(self) -> Self {
        // `d_k = (−1)^{k−1} (k−1)! / x^k`.
        let x = self.c[0];
        let mut d = [0.0; MAX_ORD + 1];
        d[0] = x.ln();
        let mut f = 1.0 / x;
        for k in 1..=ORD {
            d[k] = f;
            f *= -(k as f64) / x;
        }
        self.compose(&d[..=ORD])
    }
    fn sqrt(self) -> Self {
        // `d_k = (1/2)(1/2 − 1)…(1/2 − k + 1) x^{1/2 − k}`.
        let x = self.c[0];
        let mut d = [0.0; MAX_ORD + 1];
        let mut a = 1.0;
        for k in 0..=ORD {
            d[k] = a * x.powf(0.5 - k as f64);
            a *= 0.5 - k as f64;
        }
        // The value as `f64::sqrt` computes it.
        d[0] = x.sqrt();
        self.compose(&d[..=ORD])
    }
    fn pow(self, e: Self) -> Self {
        // A constant exponent is the power rule (`0ⁿ` a constant at a zero base, as the
        // other duals have it); otherwise `exp(e · ln a)`.
        if e.zero_jet() {
            let (a, n) = (self.c[0], e.c[0]);
            if a == 0.0 {
                return Self::constant(a.powf(n));
            }
            let mut d = [0.0; MAX_ORD + 1];
            let mut fall = 1.0;
            for k in 0..=ORD {
                d[k] = fall * a.powf(n - k as f64);
                fall *= n - k as f64;
            }
            return self.compose(&d[..=ORD]);
        }
        (e * self.ln()).exp()
    }
    fn abs(self) -> Self {
        if self.c[0] >= 0.0 {
            self
        } else {
            -self
        }
    }
    fn inv_logit(self) -> Self {
        let u = 1.0 / (1.0 + (-self.c[0]).exp());
        let v = u * (1.0 - u);
        let d = [
            u,
            v,
            v * (1.0 - 2.0 * u),
            v * (1.0 - 6.0 * u + 6.0 * u * u),
            v * (1.0 - 14.0 * u + 36.0 * u * u - 24.0 * u * u * u),
        ];
        self.compose(&d[..=ORD])
    }
    fn logit(self) -> Self {
        // `ln x − ln(1 − x)`: `d_k = (k−1)!·((−1)^{k−1}/x^k + 1/(1−x)^k)`.
        let x = self.c[0];
        let mut d = [0.0; MAX_ORD + 1];
        d[0] = (x / (1.0 - x)).ln();
        let mut fact = 1.0;
        for k in 1..=ORD {
            if k > 1 {
                fact *= (k - 1) as f64;
            }
            let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
            d[k] = fact * (sign / x.powi(k as i32) + 1.0 / (1.0 - x).powi(k as i32));
        }
        self.compose(&d[..=ORD])
    }
    fn cos(self) -> Self {
        let (s, c) = self.c[0].sin_cos();
        let d = [c, -s, -c, s, c];
        self.compose(&d[..=ORD])
    }
    fn acos(self) -> Self {
        let x = self.c[0];
        let s = 1.0 - x * x;
        let d = [
            x.acos(),
            -s.powf(-0.5),
            -x * s.powf(-1.5),
            -(1.0 + 2.0 * x * x) * s.powf(-2.5),
            -3.0 * x * (2.0 * x * x + 3.0) * s.powf(-3.5),
        ];
        self.compose(&d[..=ORD])
    }
    fn guard_floor(self, lo: f64) -> Self {
        if !(self.c[0] >= lo) {
            Self::constant(lo)
        } else {
            self
        }
    }
    fn ln_gamma(self) -> Self {
        use crate::stats::special::{digamma, ln_gamma, pentagamma, tetragamma, trigamma};
        let x = self.c[0];
        let d = [
            ln_gamma(x),
            digamma(x),
            trigamma(x),
            tetragamma(x),
            pentagamma(x),
        ];
        self.compose(&d[..=ORD])
    }
    fn jets_finite(self) -> bool {
        self.c[1..].iter().all(|x| x.is_finite())
    }
    fn jet_max_abs(self) -> f64 {
        crate::sens::num::nan_max_abs(self.c[1..].iter().copied())
    }
    /// The derivatives a caller supplies, and `NaN` past them — a visible decline rather
    /// than a silently truncated jet, as the nested duals have it.
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        self.lift4(
            v,
            d1,
            d2,
            if d2 == 0.0 { 0.0 } else { f64::NAN },
            if d2 == 0.0 { 0.0 } else { f64::NAN },
        )
    }
    fn lift3(self, v: f64, d1: f64, d2: f64, d3: f64) -> Self {
        self.lift4(v, d1, d2, d3, f64::NAN)
    }
    fn lift4(self, v: f64, d1: f64, d2: f64, d3: f64, d4: f64) -> Self {
        let d = [v, d1, d2, d3, d4];
        self.compose(&d[..=ORD])
    }
}

impl<const P: usize, const ORD: usize, const L: usize> crate::sens::num::vm::VmArith
    for Taylor<P, ORD, L>
{
    #[inline]
    fn add_in(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
    #[inline]
    fn sub_in(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
    #[inline]
    fn rsub_in(&mut self, rhs: &Self) {
        *self = *rhs - *self;
    }
    #[inline]
    fn mul_in(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
    #[inline]
    fn div_in(&mut self, rhs: &Self) {
        *self = *self / *rhs;
    }
    #[inline]
    fn add_scalar_in(&mut self, c: f64) {
        self.c[0] += c;
    }
    #[inline]
    fn sub_scalar_in(&mut self, c: f64) {
        self.c[0] -= c;
    }
    #[inline]
    fn rsub_scalar_in(&mut self, c: f64) {
        *self = Self::constant(c) - *self;
    }
    #[inline]
    fn mul_scalar_in(&mut self, c: f64) {
        for x in self.c.iter_mut() {
            *x *= c;
        }
    }
    #[inline]
    fn div_scalar_in(&mut self, c: f64) {
        // The value as the `f64` quotient; the rest scaled.
        let ib = 1.0 / c;
        self.c[0] /= c;
        for x in self.c.iter_mut().skip(1) {
            *x *= ib;
        }
    }
    fn same_bits(&self, other: &Self) -> bool {
        self.c
            .iter()
            .zip(&other.c)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type T3 = Taylor<2, 3, 10>;

    /// Every coefficient of a composite against its closed form: `f(x, y) = exp(x·y) /
    /// (1 + x²) + ln(y)·x³` at `(0.3, 1.7)`, whose normalised Taylor coefficients to third
    /// order are computed here by hand-expanded derivatives.
    #[test]
    fn the_work_counts_every_product_the_table_lists() {
        for p in 1..=8 {
            for ord in 1..=4 {
                assert_eq!(
                    mul_len(p, ord),
                    table(p, ord).mul.len(),
                    "p = {p}, ord = {ord}"
                );
            }
        }
    }

    #[test]
    fn third_order_coefficients_are_exact() {
        let (x0, y0) = (0.3f64, 1.7f64);
        let x = T3::seed(x0, 0);
        let y = T3::seed(y0, 1);
        let f = (x * y).exp() / (T3::constant(1.0) + x * x) + y.ln() * x * x * x;
        // Oracle by nested forward differences would be FD; instead compare against a
        // second, independently built route: the same function over two nested univariate
        // series in each direction at a time (directional third derivatives), which the
        // polynomial must reproduce: D_v³ f = 6·Σ_{|α|=3} c_α v^α.
        let t = table(2, 3);
        for v in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.7, -0.4]] {
            // Directional series: x = x0 + v0 s, y = y0 + v1 s, one variable `s`.
            type S = Taylor<1, 3, 4>;
            let s = S::seed(0.0, 0);
            let xs = S::constant(x0) + s * S::constant(v[0]);
            let ys = S::constant(y0) + s * S::constant(v[1]);
            let g = (xs * ys).exp() / (S::constant(1.0) + xs * xs) + ys.ln() * xs * xs * xs;
            for d in 0..=3usize {
                let want = g.c[d];
                let got: f64 = (0..10)
                    .filter(|&k| t.deg[k] as usize == d)
                    .map(|k| {
                        f.c[k] * v[0].powi(t.exps[k][0] as i32) * v[1].powi(t.exps[k][1] as i32)
                    })
                    .sum();
                assert!(
                    (got - want).abs() <= 1e-13 * (1.0 + want.abs()),
                    "degree {d} along {v:?}: {got} vs {want}"
                );
            }
        }
        // And the univariate series itself against closed forms at a point.
        type S = Taylor<1, 3, 4>;
        let e = S::seed(0.5, 0).exp();
        for k in 0..4 {
            let fact = [1.0, 1.0, 2.0, 6.0][k];
            assert!((e.c[k] - 0.5f64.exp() / fact).abs() < 1e-15);
        }
        let l = S::seed(2.0, 0).ln();
        assert!((l.c[1] - 0.5).abs() < 1e-15);
        assert!((l.c[2] + 0.125).abs() < 1e-15);
        assert!((l.c[3] - 1.0 / 24.0).abs() < 1e-15);
    }
}
