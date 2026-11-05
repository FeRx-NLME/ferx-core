//! Copulas beyond the Gaussian for a group of random effects (#1716): the families
//! Copulas.jl provides, which is what Pumas's `@random` block takes through `SklarDist`.
//!
//! A copula family is written once, as its CDF `C(u)` (or, for the elliptical t, its
//! density), over [`Mlin`]: a multilinear dual number in the `d` coordinates of `u`, with
//! one coefficient per subset of them. Evaluating `C` at `u + ε` (each `uᵢ` carrying only
//! its own direction `εᵢ`, `εᵢ² = 0`) leaves the copula density `c = ∂ᵈC/∂u₁⋯∂u_d` as the
//! coefficient of the full subset, exactly. `Mlin` is generic over the crate's `PkNum`,
//! so the same evaluation carries the exact η- and θ-derivatives the estimators read (the
//! FOCEI curvature, the adjoint, Laplace's Hessian); nothing here is differentiated twice
//! or by finite differences.
//!
//! The same number answers the conditional CDFs the generic sampler inverts: the
//! coefficient of the subset `{1..k−1}` of `C(u₁..u_k, 1, …, 1)` is
//! `∂^{k−1}C/∂u₁⋯∂u_{k−1}`.

use crate::sens::num::PkNum;

/// The largest group a non-elliptical copula takes: its density has `2^d` coefficients
/// and the product is a subset convolution (`3^d`), so `d = 8` is 6 561 terms per product.
pub(crate) const MAX_COPULA_DIM: usize = 8;

/// A multilinear dual in `d` directions over `T`: `Σ_S c_S ε^S` for subsets `S` of
/// `{0..d}`, `εᵢ² = 0`. `c[mask]` is the coefficient of the subset `mask`.
#[derive(Clone, Debug)]
pub(crate) struct Mlin<T> {
    d: usize,
    c: Vec<T>,
    /// Whether any direction can be nonzero. A constant has none, and `apply` then needs
    /// `f(a₀)` alone: its derivatives may be infinite there (`0^θ` at 0, as `φ⁻¹(1)` is
    /// for Gumbel and Joe) and `∞ · 0` would poison every coefficient.
    live: bool,
}

fn k<T: PkNum>(v: f64) -> T {
    T::from_f64(v)
}

impl<T: PkNum> Mlin<T> {
    /// A constant (no direction).
    pub(crate) fn cst(d: usize, v: T) -> Self {
        let mut c = vec![k::<T>(0.0); 1 << d];
        c[0] = v;
        Self { d, c, live: false }
    }

    /// The coordinate `uᵢ = v + εᵢ`.
    pub(crate) fn var(d: usize, i: usize, v: T) -> Self {
        let mut m = Self::cst(d, v);
        m.c[1 << i] = k(1.0);
        m.live = true;
        m
    }

    /// The value (the empty subset).
    pub(crate) fn val(&self) -> T {
        self.c[0]
    }

    /// The coefficient of the subset `mask`.
    pub(crate) fn coeff(&self, mask: usize) -> T {
        self.c[mask]
    }

    /// The coefficient of every direction at once: `∂ᵈ/∂u₁⋯∂u_d`.
    pub(crate) fn top(&self) -> T {
        self.c[(1 << self.d) - 1]
    }

    fn zip(&self, o: &Self, f: impl Fn(T, T) -> T) -> Self {
        debug_assert_eq!(self.d, o.d);
        Self {
            d: self.d,
            c: self.c.iter().zip(&o.c).map(|(&a, &b)| f(a, b)).collect(),
            live: self.live || o.live,
        }
    }

    pub(crate) fn add(&self, o: &Self) -> Self {
        self.zip(o, |a, b| a + b)
    }

    pub(crate) fn sub(&self, o: &Self) -> Self {
        self.zip(o, |a, b| a - b)
    }

    pub(crate) fn neg(&self) -> Self {
        Self {
            d: self.d,
            c: self.c.iter().map(|&a| -a).collect(),
            live: self.live,
        }
    }

    /// `self · s` for a scalar `s`.
    pub(crate) fn scale(&self, s: T) -> Self {
        Self {
            d: self.d,
            c: self.c.iter().map(|&a| a * s).collect(),
            live: self.live,
        }
    }

    /// `self + s` for a scalar `s`.
    pub(crate) fn addc(&self, s: T) -> Self {
        let mut m = self.clone();
        m.c[0] = m.c[0] + s;
        m
    }

    /// The product: a subset convolution, `(ab)_S = Σ_{A ⊆ S} a_A b_{S∖A}`.
    pub(crate) fn mul(&self, o: &Self) -> Self {
        if !self.live || !o.live {
            // A constant factor scales the other: no convolution, and no `∞ · 0`.
            return if !self.live {
                o.scale(self.c[0])
            } else {
                self.scale(o.c[0])
            };
        }
        let n = 1usize << self.d;
        let mut c = vec![k::<T>(0.0); n];
        for (s, slot) in c.iter_mut().enumerate() {
            let mut acc = self.c[0] * o.c[s];
            let mut a = s;
            while a != 0 {
                acc = acc + self.c[a] * o.c[s ^ a];
                a = (a - 1) & s;
            }
            *slot = acc;
        }
        Self {
            d: self.d,
            c,
            live: true,
        }
    }

    /// `f(self)` for a scalar function with derivatives `f[j] = f⁽ʲ⁾(a₀)`, `j = 0..=d`:
    /// `Σ_j f[j]/j! · nʲ`, `n = self − a₀` (nilpotent: `n^{d+1} = 0`).
    pub(crate) fn apply(&self, f: &[T]) -> Self {
        if !self.live {
            return Self::cst(self.d, f[0]);
        }
        let mut n = self.clone();
        n.c[0] = k(0.0);
        let mut out = Self::cst(self.d, f[0]);
        let mut p = n.clone();
        let mut fact = 1.0;
        for (j, &fj) in f.iter().enumerate().skip(1).take(self.d) {
            fact *= j as f64;
            out = out.add(&p.scale(fj * k(1.0 / fact)));
            if j < self.d {
                p = p.mul(&n);
            }
        }
        out
    }

    pub(crate) fn exp(&self) -> Self {
        let e = self.val().exp();
        self.apply(&vec![e; self.d + 1])
    }

    pub(crate) fn ln(&self) -> Self {
        let a = self.val();
        let mut f = vec![a.ln()];
        let inv = k::<T>(1.0) / a;
        let mut p = inv;
        let mut fact = 1.0;
        for j in 1..=self.d {
            let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
            f.push(p * k(sign * fact));
            fact *= j as f64;
            p = p * inv;
        }
        self.apply(&f)
    }

    /// `ln(1 + self)`, accurate for a small value.
    pub(crate) fn ln1p(&self) -> Self {
        let a = self.val();
        let mut f = vec![super::kinds::ln1p_g(a)];
        let inv = k::<T>(1.0) / (k::<T>(1.0) + a);
        let mut p = inv;
        let mut fact = 1.0;
        for j in 1..=self.d {
            let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
            f.push(p * k(sign * fact));
            fact *= j as f64;
            p = p * inv;
        }
        self.apply(&f)
    }

    /// `exp(self) − 1`, accurate for a small value.
    pub(crate) fn expm1(&self) -> Self {
        let a = self.val();
        let e = a.exp();
        let mut f = vec![expm1_t(a)];
        f.extend(std::iter::repeat_n(e, self.d));
        self.apply(&f)
    }

    /// `self^p` for a scalar power `p`, `self > 0`.
    pub(crate) fn powf(&self, p: T) -> Self {
        let a = self.val();
        let v = a.pow(p);
        let inv = k::<T>(1.0) / a;
        let mut f = vec![v];
        let mut fall = p;
        let mut ip = inv;
        for j in 1..=self.d {
            f.push(fall * v * ip);
            fall = fall * (p - k(j as f64));
            ip = ip * inv;
        }
        self.apply(&f)
    }

    pub(crate) fn recip(&self) -> Self {
        let a = self.val();
        let inv = k::<T>(1.0) / a;
        let mut f = vec![inv];
        let mut p = inv * inv;
        let mut fact = 1.0;
        for j in 1..=self.d {
            fact *= j as f64;
            let sign = if j % 2 == 1 { -1.0 } else { 1.0 };
            f.push(p * k(sign * fact));
            p = p * inv;
        }
        self.apply(&f)
    }

    pub(crate) fn div(&self, o: &Self) -> Self {
        self.mul(&o.recip())
    }

    pub(crate) fn sqrt(&self) -> Self {
        self.powf(k(0.5))
    }
}

/// `exp(x) − 1` over `T`, by its series below `|x| = 1e-3` (where `exp(x) − 1` cancels).
pub(crate) fn expm1_t<T: PkNum>(x: T) -> T {
    if x.val().abs() < 1e-3 {
        let mut term = x;
        let mut sum = x;
        for j in 2..8 {
            term = term * x * k(1.0 / j as f64);
            sum = sum + term;
        }
        sum
    } else {
        x.exp() - k(1.0)
    }
}

/// A copula family other than the Gaussian (which `correction_g` evaluates in closed
/// form), with Copulas.jl's name and parameterisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CopulaFamily {
    /// Independence: `C(u) = Π uᵢ`.
    Independent,
    /// Clayton, `φ(t) = (1 + θt)^{−1/θ}`, `θ > 0`.
    Clayton,
    /// Frank, `φ(t) = −ln(1 − (1 − e^{−θ})e^{−t})/θ`, `θ > 0` (any `θ ≠ 0` for `d = 2`).
    Frank,
    /// Gumbel–Hougaard, `φ(t) = exp(−t^{1/θ})`, `θ ≥ 1`.
    Gumbel,
    /// Joe, `φ(t) = 1 − (1 − e^{−t})^{1/θ}`, `θ ≥ 1`.
    Joe,
    /// Ali–Mikhail–Haq, `φ(t) = (1 − θ)/(eᵗ − θ)`, `θ ∈ [0, 1)` (`[−1, 1)` for `d = 2`).
    Amh,
    /// Student t, `ν > 0` and the correlation of the group's Ω block.
    StudentT,
    /// Inverse Gaussian, `φ(t) = exp((1 − √(1 + 2θ²t))/θ)`, `θ ≥ 0`.
    InvGaussian,
    /// Gumbel–Barnett, `φ(t) = exp((1 − eᵗ)/θ)`, `θ ∈ [0, θ_d]` (`θ₂ = 1`).
    GumbelBarnett,
    /// BB1, `φ(t) = (1 + t^{1/δ})^{−1/θ}`, `θ > 0`, `δ ≥ 1`.
    Bb1,
    /// BB2, `φ(t) = (1 + ln(1 + t)/δ)^{−1/θ}`, `θ, δ > 0`.
    Bb2,
    /// BB3, `φ(t) = exp(−(ln(1 + t)/δ)^{1/θ})`, `θ ≥ 1`, `δ > 0`.
    Bb3,
    /// BB6, `φ(t) = 1 − (1 − exp(−t^{1/δ}))^{1/θ}`, `θ, δ ≥ 1`.
    Bb6,
    /// BB7, `φ(t) = 1 − (1 − (1 + t)^{−1/δ})^{1/θ}`, `θ ≥ 1`, `δ > 0`.
    Bb7,
    /// BB8, `φ(t) = (1 − (1 − ηe^{−t})^{1/ϑ})/δ`, `η = 1 − (1 − δ)^ϑ`, `ϑ ≥ 1`, `δ ∈ (0, 1]`.
    Bb8,
    /// BB9, `φ(t) = exp(1/δ − (t + δ^{−ϑ})^{1/ϑ})`, `ϑ ≥ 1`, `δ > 0`.
    Bb9,
    /// BB10, `φ(t) = ((1 − δ)/(eᵗ − δ))^{1/θ}`, `θ > 0`, `δ ∈ [0, 1)`.
    Bb10,
}

/// One dependence parameter's θ: its auto-declared name prefix, init and bounds, and the
/// open interval a declared θ's bounds must stay inside for a group of size `d`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CopulaParam {
    pub(crate) prefix: &'static str,
    pub(crate) init: f64,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) domain: (f64, f64),
}

impl CopulaFamily {
    pub(crate) const ALL: [CopulaFamily; 17] = [
        CopulaFamily::Independent,
        CopulaFamily::Clayton,
        CopulaFamily::Frank,
        CopulaFamily::Gumbel,
        CopulaFamily::Joe,
        CopulaFamily::Amh,
        CopulaFamily::StudentT,
        CopulaFamily::InvGaussian,
        CopulaFamily::GumbelBarnett,
        CopulaFamily::Bb1,
        CopulaFamily::Bb2,
        CopulaFamily::Bb3,
        CopulaFamily::Bb6,
        CopulaFamily::Bb7,
        CopulaFamily::Bb8,
        CopulaFamily::Bb9,
        CopulaFamily::Bb10,
    ];

    /// The model-file name (Copulas.jl's, lower-cased, without `Copula`).
    pub(crate) fn name(self) -> &'static str {
        match self {
            CopulaFamily::Independent => "independent",
            CopulaFamily::Clayton => "clayton",
            CopulaFamily::Frank => "frank",
            CopulaFamily::Gumbel => "gumbel",
            CopulaFamily::Joe => "joe",
            CopulaFamily::Amh => "amh",
            CopulaFamily::StudentT => "t",
            CopulaFamily::InvGaussian => "invgaussian",
            CopulaFamily::GumbelBarnett => "gumbelbarnett",
            CopulaFamily::Bb1 => "bb1",
            CopulaFamily::Bb2 => "bb2",
            CopulaFamily::Bb3 => "bb3",
            CopulaFamily::Bb6 => "bb6",
            CopulaFamily::Bb7 => "bb7",
            CopulaFamily::Bb8 => "bb8",
            CopulaFamily::Bb9 => "bb9",
            CopulaFamily::Bb10 => "bb10",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    /// Whether the family's dependence lives in the group's Ω block (a correlation),
    /// rather than in its own θs with a diagonal Ω.
    pub(crate) fn elliptical(self) -> bool {
        self == CopulaFamily::StudentT
    }

    /// The dependence parameters, for a group of `d` ETAs.
    pub(crate) fn params(self, d: usize) -> Vec<CopulaParam> {
        let p = |prefix, init, lower, upper, domain| CopulaParam {
            prefix,
            init,
            lower,
            upper,
            domain,
        };
        match self {
            CopulaFamily::Independent => vec![],
            CopulaFamily::Clayton => vec![p("THETA", 1.0, 0.01, 50.0, (0.0, f64::INFINITY))],
            CopulaFamily::Frank => {
                if d == 2 {
                    vec![p(
                        "THETA",
                        2.0,
                        -50.0,
                        50.0,
                        (f64::NEG_INFINITY, f64::INFINITY),
                    )]
                } else {
                    vec![p("THETA", 2.0, 0.01, 50.0, (0.0, f64::INFINITY))]
                }
            }
            CopulaFamily::Gumbel => vec![p("THETA", 1.5, 1.0, 50.0, (1.0, f64::INFINITY))],
            CopulaFamily::Joe => vec![p("THETA", 1.5, 1.0, 50.0, (1.0, f64::INFINITY))],
            CopulaFamily::Amh => {
                if d == 2 {
                    vec![p("THETA", 0.3, -1.0, 0.99, (-1.0, 1.0))]
                } else {
                    vec![p("THETA", 0.3, 0.0, 0.99, (0.0, 1.0))]
                }
            }
            CopulaFamily::StudentT => vec![p("NU", 10.0, 2.0, 100.0, (0.0, f64::INFINITY))],
            CopulaFamily::InvGaussian => vec![p("THETA", 0.5, 0.0, 50.0, (0.0, f64::INFINITY))],
            CopulaFamily::GumbelBarnett => {
                let hi = gumbel_barnett_max(d);
                vec![p("THETA", 0.5 * hi, 0.0, hi, (0.0, hi))]
            }
            CopulaFamily::Bb1 => vec![
                p("THETA", 1.0, 0.01, 50.0, (0.0, f64::INFINITY)),
                p("DELTA", 1.5, 1.0, 50.0, (1.0, f64::INFINITY)),
            ],
            CopulaFamily::Bb2 => vec![
                p("THETA", 1.0, 0.01, 50.0, (0.0, f64::INFINITY)),
                p("DELTA", 1.0, 0.01, 50.0, (0.0, f64::INFINITY)),
            ],
            CopulaFamily::Bb3 | CopulaFamily::Bb7 | CopulaFamily::Bb9 => vec![
                p("THETA", 1.5, 1.0, 50.0, (1.0, f64::INFINITY)),
                p("DELTA", 1.0, 0.01, 50.0, (0.0, f64::INFINITY)),
            ],
            CopulaFamily::Bb6 => vec![
                p("THETA", 1.5, 1.0, 50.0, (1.0, f64::INFINITY)),
                p("DELTA", 1.5, 1.0, 50.0, (1.0, f64::INFINITY)),
            ],
            CopulaFamily::Bb8 => vec![
                p("THETA", 1.5, 1.0, 50.0, (1.0, f64::INFINITY)),
                p("DELTA", 0.5, 0.01, 1.0, (0.0, 1.0)),
            ],
            CopulaFamily::Bb10 => vec![
                p("THETA", 1.0, 0.01, 50.0, (0.0, f64::INFINITY)),
                p("DELTA", 0.5, 0.0, 0.99, (0.0, 1.0)),
            ],
        }
    }

    /// Whether a parameter value sits exactly on the independence point, where the
    /// generator's own formula is `0/0` (Clayton and Frank at θ = 0, Gumbel and Joe at 1).
    fn independent_at(self, p: &[f64]) -> bool {
        match self {
            CopulaFamily::Independent => true,
            CopulaFamily::Clayton | CopulaFamily::Frank | CopulaFamily::Amh => p[0] == 0.0,
            CopulaFamily::Gumbel | CopulaFamily::Joe => p[0] == 1.0,
            CopulaFamily::InvGaussian | CopulaFamily::GumbelBarnett => p[0] == 0.0,
            CopulaFamily::StudentT
            | CopulaFamily::Bb1
            | CopulaFamily::Bb2
            | CopulaFamily::Bb3
            | CopulaFamily::Bb6
            | CopulaFamily::Bb7
            | CopulaFamily::Bb8
            | CopulaFamily::Bb9
            | CopulaFamily::Bb10 => false,
        }
    }

    /// `φ⁻¹(u)` of an Archimedean family, over `Mlin`; `p` its parameters.
    fn phi_inv<T: PkNum>(self, u: &Mlin<T>, p: &[T]) -> Mlin<T> {
        let th = p[0];
        let one = k::<T>(1.0);
        match self {
            // (u^{−θ} − 1)/θ = expm1(−θ ln u)/θ.
            CopulaFamily::Clayton => u.ln().scale(-th).expm1().scale(k::<T>(1.0) / th),
            // −ln(expm1(−θu)/expm1(−θ)).
            CopulaFamily::Frank => u
                .scale(-th)
                .expm1()
                .scale(k::<T>(1.0) / expm1_t(-th))
                .ln()
                .neg(),
            // (−ln u)^θ.
            CopulaFamily::Gumbel => u.ln().neg().powf(th),
            // −ln(1 − (1 − u)^θ).
            CopulaFamily::Joe => u.neg().addc(k(1.0)).powf(th).neg().ln1p().neg(),
            // ln((1 − θ + θu)/u).
            CopulaFamily::Amh => u.scale(th).addc(k::<T>(1.0) - th).ln().sub(&u.ln()),
            CopulaFamily::Independent => u.ln().neg(),
            // ((1 − θ ln u)² − 1)/(2θ²).
            CopulaFamily::InvGaussian => {
                let a = u.ln().scale(-th).addc(one);
                a.mul(&a).addc(-one).scale(one / (k::<T>(2.0) * th * th))
            }
            // ln1p(−θ ln u).
            CopulaFamily::GumbelBarnett => u.ln().scale(-th).ln1p(),
            // (u^{−θ} − 1)^δ.
            CopulaFamily::Bb1 => u.ln().scale(-th).expm1().powf(p[1]),
            // expm1(δ expm1(−θ ln u)).
            CopulaFamily::Bb2 => u.ln().scale(-th).expm1().scale(p[1]).expm1(),
            // expm1(δ (−ln u)^θ).
            CopulaFamily::Bb3 => u.ln().neg().powf(th).scale(p[1]).expm1(),
            // (−ln1p(−(1 − u)^θ))^δ.
            CopulaFamily::Bb6 => u.neg().addc(one).powf(th).neg().ln1p().neg().powf(p[1]),
            // w^{−δ} − 1, w = 1 − (1 − u)^θ; `ln(1 − u)`, which is −∞ at a constant
            // `u = 1` (the sampler's marginalised coordinates), where `ln1p_g` is NaN.
            CopulaFamily::Bb7 => u
                .neg()
                .addc(one)
                .ln()
                .scale(th)
                .expm1()
                .neg()
                .powf(-p[1])
                .addc(-one),
            // −ln((1 − (1 − δu)^ϑ)/η).
            CopulaFamily::Bb8 => {
                let eta = bb8_eta(th, p[1]);
                u.scale(-p[1])
                    .addc(one)
                    .powf(th)
                    .neg()
                    .addc(one)
                    .scale(one / eta)
                    .ln()
                    .neg()
            }
            // (1/δ − ln u)^ϑ − δ^{−ϑ}.
            CopulaFamily::Bb9 => u.ln().neg().addc(one / p[1]).powf(th).addc(-p[1].pow(-th)),
            // ln(δ + (1 − δ) u^{−θ}).
            CopulaFamily::Bb10 => u.ln().scale(-th).exp().scale(one - p[1]).addc(p[1]).ln(),
            CopulaFamily::StudentT => unreachable!("the t copula is not Archimedean"),
        }
    }

    /// `φ(s)` of an Archimedean family, over `Mlin`; `p` its parameters.
    fn phi<T: PkNum>(self, s: &Mlin<T>, p: &[T]) -> Mlin<T> {
        let one = k::<T>(1.0);
        let th = p[0];
        match self {
            CopulaFamily::Clayton => s.scale(th).addc(one).powf(-one / th),
            // −ln1p(e^{−s}·expm1(−θ))/θ.
            CopulaFamily::Frank => s.neg().exp().scale(expm1_t(-th)).ln1p().scale(-one / th),
            CopulaFamily::Gumbel => s.powf(one / th).neg().exp(),
            // 1 − (−expm1(−s))^{1/θ}.
            CopulaFamily::Joe => s.neg().expm1().neg().powf(one / th).neg().addc(one),
            CopulaFamily::Amh => s.exp().addc(-th).recip().scale(one - th),
            CopulaFamily::Independent => s.neg().exp(),
            CopulaFamily::InvGaussian => s
                .scale(k::<T>(2.0) * th * th)
                .addc(one)
                .sqrt()
                .neg()
                .addc(one)
                .scale(one / th)
                .exp(),
            // exp(−expm1(s)/θ).
            CopulaFamily::GumbelBarnett => s.expm1().scale(-one / th).exp(),
            // (1 + s^{1/δ})^{−1/θ}.
            CopulaFamily::Bb1 => s.powf(one / p[1]).ln1p().scale(-one / th).exp(),
            // exp(−ln1p(ln1p(s)/δ)/θ).
            CopulaFamily::Bb2 => s.ln1p().scale(one / p[1]).ln1p().scale(-one / th).exp(),
            // exp(−(ln1p(s)/δ)^{1/θ}).
            CopulaFamily::Bb3 => s.ln1p().scale(one / p[1]).powf(one / th).neg().exp(),
            // 1 − (1 − exp(−s^{1/δ}))^{1/θ}.
            CopulaFamily::Bb6 => s
                .powf(one / p[1])
                .neg()
                .expm1()
                .neg()
                .powf(one / th)
                .neg()
                .addc(one),
            // 1 − (1 − (1 + s)^{−1/δ})^{1/θ}.
            CopulaFamily::Bb7 => s
                .ln1p()
                .scale(-one / p[1])
                .expm1()
                .neg()
                .powf(one / th)
                .neg()
                .addc(one),
            // (1 − (1 − ηe^{−s})^{1/ϑ})/δ.
            CopulaFamily::Bb8 => {
                let eta = bb8_eta(th, p[1]);
                s.neg()
                    .exp()
                    .scale(-eta)
                    .addc(one)
                    .powf(one / th)
                    .neg()
                    .addc(one)
                    .scale(one / p[1])
            }
            // exp(1/δ − (s + δ^{−ϑ})^{1/ϑ}).
            CopulaFamily::Bb9 => s
                .addc(p[1].pow(-th))
                .powf(one / th)
                .neg()
                .addc(one / p[1])
                .exp(),
            // ((1 − δ)/(eˢ − δ))^{1/θ}.
            CopulaFamily::Bb10 => s
                .exp()
                .addc(-p[1])
                .ln()
                .neg()
                .addc(super::kinds::ln1p_g(-p[1]))
                .scale(one / th)
                .exp(),
            CopulaFamily::StudentT => unreachable!("the t copula is not Archimedean"),
        }
    }

    /// `C(u)` of a non-elliptical family, over `Mlin`; `p` its dependence parameters.
    pub(crate) fn cdf<T: PkNum>(self, u: &[Mlin<T>], p: &[T]) -> Mlin<T> {
        let pv: Vec<f64> = p.iter().map(|x| x.val()).collect();
        if self.independent_at(&pv) {
            let mut c = u[0].clone();
            for x in &u[1..] {
                c = c.mul(x);
            }
            return c;
        }
        if self == CopulaFamily::Bb2 {
            return bb2_cdf(u, p);
        }
        let mut s = self.phi_inv(&u[0], p);
        for x in &u[1..] {
            s = s.add(&self.phi_inv(x, p));
        }
        self.phi(&s, p)
    }

    /// `ln c(u)` of a non-elliptical family at `u ∈ (0, 1)^d`, over `T`; `−∞` where the
    /// density is not positive, or not representable (below `≈ e^{−700}`).
    pub(crate) fn log_density<T: PkNum>(self, u: &[T], p: &[T]) -> T {
        let d = u.len();
        let xs: Vec<Mlin<T>> = u
            .iter()
            .enumerate()
            .map(|(i, &v)| Mlin::var(d, i, v))
            .collect();
        let c = self.cdf(&xs, p).top();
        // Below `≈ e^{−700}` the density is not representable from its CDF (BB2 and BB3,
        // whose `φ⁻¹` grows like an exponential of a power, reach it at moderate `u`): the
        // value or its jets leave f64 range, and the copula reads as a wall (`−∞`) rather
        // than hand an estimator a finite value with a non-finite gradient.
        if c.val() > 0.0 && c.val().is_finite() && c.jets_finite() {
            let lc = c.ln();
            // A subnormal `c` has finite jets of its own but `ln c`'s are `g/c`.
            if lc.jets_finite() {
                return lc;
            }
        }
        k(f64::NEG_INFINITY)
    }

    /// One draw `u ∈ (0, 1)^d` of a non-elliptical family by conditional inversion:
    /// `u₁ ~ U(0, 1)`, then each `u_k` solves `C(u_k | u₁..u_{k−1}) = v_k` by bisection,
    /// the conditional CDF being the `{1..k−1}` coefficient of `C(u₁..u_k, 1, …, 1)`
    /// over that of `C(u₁..u_{k−1}, 1, …, 1)`'s density.
    pub(crate) fn sample(self, d: usize, p: &[f64], rng: &mut impl rand::Rng) -> Vec<f64> {
        use rand::RngExt;
        let mut u = vec![rng.random::<f64>().clamp(1e-300, 1.0 - 1e-16)];
        for kk in 1..d {
            let v: f64 = rng.random();
            // The conditioning coordinates carry directions 0..kk; u_k and the rest are
            // constants (the rest at 1, which marginalises them out).
            let cond = |x: f64| -> f64 {
                let args: Vec<Mlin<f64>> = (0..d)
                    .map(|i| {
                        if i < kk {
                            Mlin::var(kk, i, u[i])
                        } else if i == kk {
                            Mlin::cst(kk, x)
                        } else {
                            Mlin::cst(kk, 1.0)
                        }
                    })
                    .collect();
                self.cdf(&args, p).top()
            };
            let denom = cond(1.0);
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if cond(mid) / denom < v {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            u.push((0.5 * (lo + hi)).clamp(1e-300, 1.0 - 1e-16));
        }
        u
    }
}

/// BB2's CDF in log space: `C = (1 + ln1p(s)/δ)^{−1/θ}` with `s = Σ (e^{aᵢ} − 1)`,
/// `aᵢ = δ(u^{−θ} − 1)`. `e^{aᵢ}` overflows deep in a tail (`a = 750` at `u = 0.02`,
/// θ = 2), but `ln1p(s) = a_m + ln(Σ e^{aᵢ − a_m} − (d − 1)e^{−a_m})` at the largest `a_m`
/// does not.
fn bb2_cdf<T: PkNum>(u: &[Mlin<T>], p: &[T]) -> Mlin<T> {
    let (th, de) = (p[0], p[1]);
    let one = k::<T>(1.0);
    let a: Vec<Mlin<T>> = u
        .iter()
        .map(|x| x.ln().scale(-th).expm1().scale(de))
        .collect();
    let m = (0..a.len())
        .max_by(|&i, &j| a[i].val().val().total_cmp(&a[j].val().val()))
        .unwrap_or(0);
    let mut sum = a[0].sub(&a[m]).exp();
    for x in &a[1..] {
        sum = sum.add(&x.sub(&a[m]).exp());
    }
    let tail = a[m].neg().exp().scale(k(-((a.len() - 1) as f64)));
    let ln1p_s = a[m].add(&sum.add(&tail).ln());
    ln1p_s.scale(one / de).addc(one).powf(-one / th)
}

/// BB8's `η = 1 − (1 − δ)^ϑ`.
fn bb8_eta<T: PkNum>(th: T, de: T) -> T {
    -expm1_t(th * super::kinds::ln1p_g(-de))
}

/// Gumbel–Barnett's largest θ in `d` dimensions (Copulas.jl's
/// `_find_critical_value_gumbelbarnett`): above it the generator is not `d`-monotone.
pub(crate) fn gumbel_barnett_max(d: usize) -> f64 {
    match d {
        2 => 1.0,
        3 => 0.380,
        4 => 0.216,
        5 => 0.145,
        6 => 0.106,
        7 => 0.082,
        8 => 0.066,
        9 => 0.055,
        10 => 0.046,
        _ => {
            let df = d as f64;
            let c2 = df * (df - 1.0) / 2.0;
            let c3 = c2 * (df - 2.0) / 3.0;
            let c4 = c3 * (df - 3.0) / 4.0;
            let discr = c2 * c2 - (2.0 * df / (df - 1.0)) * (c3 + 3.0 * c4);
            if discr >= 0.0 {
                1.0 / (c2 / df + (df - 1.0) / df * discr.sqrt())
            } else {
                0.046
            }
        }
    }
}

/// `ln c(u)` of the Student-t copula with `ν` degrees of freedom and correlation `R`
/// (given as `R⁻¹` and `ln|R|`): the multivariate t log-density at `x = t_ν⁻¹(u)` minus
/// its margins' log-densities, over `T`.
pub(crate) fn t_copula_log_density<T: PkNum>(u: &[T], nu: T, r_inv: &[Vec<T>], ld_r: T) -> T {
    let d = u.len();
    let half = k::<T>(0.5);
    let x: Vec<T> = u.iter().map(|&ui| student_t_quantile(ui, nu)).collect();
    let mut q = k::<T>(0.0);
    for i in 0..d {
        for j in 0..d {
            q = q + r_inv[i][j] * x[i] * x[j];
        }
    }
    let df = k::<T>(d as f64);
    // ln t_{d,ν}(x; R) without the −(d/2)ln(νπ), which the margins' cancel term for term.
    let joint = ((nu + df) * half).ln_gamma()
        - (nu * half).ln_gamma()
        - half * ld_r
        - (nu + df) * half * super::kinds::ln1p_g(q / nu);
    let mut margins = k::<T>(0.0);
    for &xi in &x {
        margins = margins + ((nu + k(1.0)) * half).ln_gamma()
            - (nu * half).ln_gamma()
            - (nu + k(1.0)) * half * super::kinds::ln1p_g(xi * xi / nu);
    }
    joint - margins
}

/// The Student-t quantile `t_ν⁻¹(u)` over `T`: bisection in `f64` on the CDF, then
/// Newton steps in `T`, which carry the exact `∂/∂u` and `∂/∂ν`.
pub(crate) fn student_t_quantile<T: PkNum>(u: T, nu: T) -> T {
    use super::DensityKind;
    let one = k::<T>(1.0);
    let cdf_f = |x: f64| {
        super::kinds::eta_cdf::<f64>(DensityKind::StudentT, x, 1.0, &[nu.val()]).unwrap_or(f64::NAN)
    };
    let target = u.val();
    let (mut lo, mut hi) = (-1.0f64, 1.0f64);
    while cdf_f(lo) > target && lo > -1e12 {
        lo *= 2.0;
    }
    while cdf_f(hi) < target && hi < 1e12 {
        hi *= 2.0;
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if cdf_f(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut x = T::from_f64(0.5 * (lo + hi));
    for _ in 0..3 {
        let f =
            super::kinds::eta_cdf(DensityKind::StudentT, x, one, &[nu]).unwrap_or(x * k(f64::NAN));
        let lp = super::kinds::eta_log_pdf(DensityKind::StudentT, x, one, &[nu]);
        x = x - (f - u) / lp.exp();
    }
    x
}

/// One draw `u ∈ (0, 1)^d` of the Student-t copula: `x = √(ν/W)·L z`, `W ~ χ²_ν`,
/// `uᵢ = t_ν(xᵢ)`, with `L` the Cholesky factor of `R`.
pub(crate) fn sample_t_copula(
    nu: f64,
    r_chol: &nalgebra::DMatrix<f64>,
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    use rand_distr::{ChiSquared, Distribution, StandardNormal};
    let d = r_chol.nrows();
    let z = nalgebra::DVector::from_fn(d, |_, _| StandardNormal.sample(rng));
    let w = ChiSquared::new(nu)
        .map_or(f64::NAN, |c| c.sample(rng))
        .max(1e-300);
    let x = r_chol * z * (nu / w).sqrt();
    x.iter()
        .map(|&xi| {
            super::kinds::eta_cdf::<f64>(super::DensityKind::StudentT, xi, 1.0, &[nu])
                .unwrap_or(f64::NAN)
                .clamp(1e-300, 1.0 - 1e-16)
        })
        .collect()
}

#[cfg(test)]
#[path = "eta_density_copula_tests.rs"]
mod tests;
