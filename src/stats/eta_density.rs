//! Non-Gaussian random-effect distributions (#1716), declared on the `[parameters]` omega
//! line: `omega ETA ~ ω density KIND(SHAPE)`. After Pumas's `@random` distributions, the
//! random effect *is* the distributed quantity: `CL = TVCL * ETA_CL` under
//! `density gamma` gives a Gamma-distributed clearance of mean `TVCL`.
//!
//! The estimators work on an unconstrained **latent** `z`, which is what the inner loop,
//! the samplers and the sdtab ETA columns hold. The model reads the random effect
//! `η = m(z)` through the kind's **domain map** `m` (identity on ℝ, `exp` on (0, ∞),
//! `inv_logit` on (0, 1)), which the parser applies to every read of the ETA in
//! `[individual_parameters]`. The prior is then the exact latent density
//! `p_z(z) = p_η(m(z))·|m′(z)|`. A constrained effect thus never leaves its support, and
//! the Laplace / FOCE machinery runs in coordinates where the posterior is smooth (Pumas
//! maps a constrained random effect the same way).
//!
//! The kinds are every distribution Pumas's `@random` block takes for a scalar or vector
//! random effect (`DensityKind`), each with its latent density and parameterisation in
//! `eta_density_kinds`: on ℝ mean 0 and variance `ω`, on (0, ∞) mean 1 and variance `ω`,
//! on (0, 1) mean `μ` and variance `ω`, where the family has them. `truncated(KIND, lo,
//! hi)` restricts any univariate kind, through a bounded map so the latent stays
//! unconstrained.
//!
//! A block (`block_omega … density t(ν)`) is a multivariate t with one ν. The other kinds
//! are univariate: correlating them needs a copula, which is not built. A Box-Cox
//! "density" needs no kind of its own: on its latent scale it is exactly the Box-Cox
//! transform, so `density boxcox(λ)` is spelled as `transform boxcox(λ)`.
//!
//! Every consumer keeps its Gaussian prior `zᵀΩ⁻¹z + ln|Ω|` (ferx's `−2 ln p` convention,
//! which drops `n ln 2π`) and adds this module's **correction**, for each group `g`:
//!
//! ```text
//! c_g(z) = −2 ln p_g(z_g) − (z_gᵀ Ω_g⁻¹ z_g + ln|Ω_g| + n_g ln 2π)
//! ```
//!
//! The sum is then the exact `−2 ln p(z)`, and a Gaussian model adds nothing, so its
//! objective is unchanged bit for bit. Ω is block-diagonal across declared blocks, so
//! `Ω_g⁻¹` is `Ω⁻¹` restricted to the group, and for a univariate group `ω = 1/Ω⁻¹_aa`.
//!
//! The correction is written **once**, as `correction_g` over `PkNum`. `T = f64` is
//! the value, the hyper-dual seeds of `correction_jets_g` give its exact η-gradient and
//! η-Hessian, and the AD kernel's outer types carry `∂/∂θ` (the shape) and `∂/∂Ω` through
//! the same function.

use crate::sens::num::PkNum;
use crate::sens::rtape::DualT;
use nalgebra::DMatrix;

/// Which distribution a group of ETAs follows: every distribution Pumas's `@random`
/// block takes that a scalar or vector random effect can follow (`eta_density_kinds` has
/// each one's parameterisation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityKind {
    /// Student-t (multivariate over a block), one ν; scale² ω.
    StudentT,
    /// Logistic, variance ω.
    Logistic,
    /// Laplace (double exponential), variance ω.
    Laplace,
    /// Cauchy, scale² ω.
    Cauchy,
    /// Gumbel, mean 0, variance ω.
    Gumbel,
    /// Normal-inverse-Gaussian, mean 0, variance ω, shapes ζ and ρ.
    NormalInverseGaussian,
    /// p-generalized Gaussian, variance ω, shape p.
    PGeneralizedGaussian,
    /// Generalized extreme value, mean 0, variance ω, shape ξ.
    GeneralizedExtremeValue,
    /// Arcsine on a symmetric interval, variance ω.
    Arcsine,
    /// Uniform on a symmetric interval, variance ω.
    Uniform,
    /// Biweight kernel, variance ω.
    Biweight,
    /// Triweight kernel, variance ω.
    Triweight,
    /// Epanechnikov kernel, variance ω.
    Epanechnikov,
    /// Cosine kernel, variance ω.
    Cosine,
    /// Semicircle, variance ω.
    Semicircle,
    /// Symmetric triangular, variance ω.
    SymTriangular,
    /// Gamma, mean 1, variance ω.
    Gamma,
    /// Inverse gamma, mean 1, variance ω.
    InvGamma,
    /// Log-normal, mean 1, variance ω.
    LogNormal,
    /// Exponential, mean 1 (ω unused).
    Exponential,
    /// Erlang (gamma with shape k), mean 1 (ω unused).
    Erlang,
    /// Scaled chi-squared: the gamma with mean 1 and variance ω.
    ChiSq,
    /// Scaled chi (Nakagami), mean 1, variance ω.
    Chi,
    /// Rayleigh, mean 1 (ω unused).
    Rayleigh,
    /// Weibull, mean 1, variance ω.
    Weibull,
    /// Fréchet, mean 1, variance ω.
    Frechet,
    /// Inverse Gaussian, mean 1, variance ω.
    InverseGaussian,
    /// Beta prime, mean 1, variance ω.
    BetaPrime,
    /// Scaled noncentral chi-squared, mean 1, variance ω, k degrees of freedom.
    NoncentralChiSq,
    /// Kolmogorov (ω unused).
    Kolmogorov,
    /// Lévy, scale ω.
    Levy,
    /// Pareto, mean 1, variance ω.
    Pareto,
    /// Generalized Pareto, mean 1, variance ω.
    GeneralizedPareto,
    /// Log-uniform on `(a, b)` (ω unused).
    LogUniform,
    /// Beta, mean μ, variance ω.
    Beta,
    /// Logit-normal, location `logit μ`.
    LogitNormal,
}

/// How the model reads a latent `z`: `η = m(z)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DomainMap {
    /// `η = z`, on ℝ (a bounded kernel's support is enforced by its density).
    Identity,
    /// `η = exp(z)`, on (0, ∞).
    Exp,
    /// `η = inv_logit(z)`, on (0, 1).
    InvLogit,
}

impl DomainMap {
    /// The model-file function the parser wraps a read in, if any.
    pub(crate) fn function(self) -> Option<&'static str> {
        match self {
            DomainMap::Identity => None,
            DomainMap::Exp => Some("exp"),
            DomainMap::InvLogit => Some("inv_logit"),
        }
    }

    /// `m(z)`.
    pub(crate) fn apply(self, z: f64) -> f64 {
        match self {
            DomainMap::Identity => z,
            DomainMap::Exp => z.exp(),
            DomainMap::InvLogit => 1.0 / (1.0 + (-z).exp()),
        }
    }

    /// `m⁻¹(η)`.
    pub(crate) fn invert(self, eta: f64) -> f64 {
        match self {
            DomainMap::Identity => eta,
            DomainMap::Exp => eta.ln(),
            DomainMap::InvLogit => (eta / (1.0 - eta)).ln(),
        }
    }
}

/// A shape θ a kind takes: the auto-declared name's prefix, its `(init, lower, upper)`,
/// and the bounds a user-declared one must lie inside.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShapeSpec {
    pub(crate) prefix: &'static str,
    pub(crate) init: f64,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    /// `(lo, hi)`: a declared θ's bounds must sit inside, strictly where `strict`.
    pub(crate) domain: (f64, f64),
    pub(crate) strict: bool,
}

const fn shape(
    prefix: &'static str,
    init: f64,
    lower: f64,
    upper: f64,
    domain: (f64, f64),
    strict: bool,
) -> ShapeSpec {
    ShapeSpec {
        prefix,
        init,
        lower,
        upper,
        domain,
        strict,
    }
}

impl DensityKind {
    /// Every kind, for name lookup.
    pub(crate) const ALL: [DensityKind; 36] = {
        use DensityKind::*;
        [
            StudentT,
            Logistic,
            Laplace,
            Cauchy,
            Gumbel,
            NormalInverseGaussian,
            PGeneralizedGaussian,
            GeneralizedExtremeValue,
            Arcsine,
            Uniform,
            Biweight,
            Triweight,
            Epanechnikov,
            Cosine,
            Semicircle,
            SymTriangular,
            Gamma,
            InvGamma,
            LogNormal,
            Exponential,
            Erlang,
            ChiSq,
            Chi,
            Rayleigh,
            Weibull,
            Frechet,
            InverseGaussian,
            BetaPrime,
            NoncentralChiSq,
            Kolmogorov,
            Levy,
            Pareto,
            GeneralizedPareto,
            LogUniform,
            Beta,
            LogitNormal,
        ]
    };

    /// The model-file name after `density` (the first of [`Self::aliases`]).
    pub(crate) fn name(self) -> &'static str {
        self.aliases()[0]
    }

    /// The (lower-cased) model-file names of the kind; the first is canonical. Pumas's
    /// type names, lower-cased, are always among them.
    pub(crate) fn aliases(self) -> &'static [&'static str] {
        use DensityKind::*;
        match self {
            StudentT => &["t", "tdist", "studentt"],
            Logistic => &["logistic"],
            Laplace => &["laplace"],
            Cauchy => &["cauchy"],
            Gumbel => &["gumbel"],
            NormalInverseGaussian => &["nig", "normalinversegaussian"],
            PGeneralizedGaussian => &["pgeneralizedgaussian", "pgg", "generalizednormal"],
            GeneralizedExtremeValue => &["gev", "generalizedextremevalue"],
            Arcsine => &["arcsine"],
            Uniform => &["uniform"],
            Biweight => &["biweight"],
            Triweight => &["triweight"],
            Epanechnikov => &["epanechnikov"],
            Cosine => &["cosine"],
            Semicircle => &["semicircle"],
            SymTriangular => &["symtriangular", "symtriangulardist"],
            Gamma => &["gamma"],
            InvGamma => &["invgamma", "inversegamma"],
            LogNormal => &["lognormal"],
            Exponential => &["exponential"],
            Erlang => &["erlang"],
            ChiSq => &["chisq"],
            Chi => &["chi"],
            Rayleigh => &["rayleigh"],
            Weibull => &["weibull"],
            Frechet => &["frechet"],
            InverseGaussian => &["inversegaussian"],
            BetaPrime => &["betaprime"],
            NoncentralChiSq => &["noncentralchisq"],
            Kolmogorov => &["kolmogorov"],
            Levy => &["levy"],
            Pareto => &["pareto"],
            GeneralizedPareto => &["gpd", "generalizedpareto"],
            LogUniform => &["loguniform"],
            Beta => &["beta"],
            LogitNormal => &["logitnormal"],
        }
    }

    /// The kind a (lower-cased) name spells, if any.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.aliases().contains(&name))
    }

    /// The domain map the model reads the latent through.
    pub(crate) fn map(self) -> DomainMap {
        use DensityKind::*;
        match self {
            Gamma | InvGamma | LogNormal | Exponential | Erlang | ChiSq | Chi | Rayleigh
            | Weibull | Frechet | InverseGaussian | BetaPrime | NoncentralChiSq | Kolmogorov
            | Levy | Pareto | GeneralizedPareto | LogUniform => DomainMap::Exp,
            Beta | LogitNormal => DomainMap::InvLogit,
            _ => DomainMap::Identity,
        }
    }

    /// Whether a block of several ETAs can follow it (only the multivariate t).
    pub(crate) fn allows_block(self) -> bool {
        self == DensityKind::StudentT
    }

    /// Whether the kind has no free spread, so its `ω` is unused and must be `FIX`.
    pub(crate) fn omega_unused(self) -> bool {
        use DensityKind::*;
        matches!(
            self,
            Exponential | Erlang | Rayleigh | Kolmogorov | LogUniform
        )
    }

    /// The shape θs the kind takes, in argument order.
    pub(crate) fn shapes(self) -> &'static [ShapeSpec] {
        use DensityKind::*;
        const INF: f64 = f64::INFINITY;
        const T_NU: &[ShapeSpec] = &[shape("NU", 80.0, 3.0, 100.0, (0.0, INF), true)];
        const NIG: &[ShapeSpec] = &[
            shape("ZETA", 1.0, 0.01, 100.0, (0.0, INF), true),
            shape("RHO", 0.0, -0.95, 0.95, (-1.0, 1.0), true),
        ];
        const PGG: &[ShapeSpec] = &[shape("P", 2.0, 0.5, 20.0, (0.0, INF), true)];
        const GEV: &[ShapeSpec] = &[shape("XI", 0.1, -0.45, 0.45, (-INF, 0.5), true)];
        const ERLANG: &[ShapeSpec] = &[shape("K", 2.0, 1.0, 50.0, (0.0, INF), true)];
        const NCCHISQ: &[ShapeSpec] = &[shape("K", 2.0, 0.1, 100.0, (0.0, INF), true)];
        const LOGUNIF: &[ShapeSpec] = &[
            shape("A", 0.5, 0.001, 1.0, (0.0, INF), true),
            shape("B", 2.0, 1.0, 1000.0, (0.0, INF), true),
        ];
        const MU: &[ShapeSpec] = &[shape("MU", 0.5, 0.001, 0.999, (0.0, 1.0), false)];
        match self {
            StudentT => T_NU,
            NormalInverseGaussian => NIG,
            PGeneralizedGaussian => PGG,
            GeneralizedExtremeValue => GEV,
            Erlang => ERLANG,
            NoncentralChiSq => NCCHISQ,
            LogUniform => LOGUNIF,
            Beta | LogitNormal => MU,
            _ => &[],
        }
    }
}

/// One density group: the ETAs of one Ω block (one ETA unless the kind allows a block),
/// and its shape θ.
#[derive(Debug, Clone, PartialEq)]
pub struct EtaDensity {
    pub(crate) kind: DensityKind,
    /// η positions, ascending: every ETA of one Ω block.
    pub(crate) etas: Vec<usize>,
    /// θ indices of the shapes: the kind's ([`DensityKind::shapes`] order), then a
    /// truncation's θ-valued bounds.
    pub(crate) shape_thetas: Vec<usize>,
    /// `truncated(KIND, lo, hi)`: the kind restricted to `(lo, hi)`, renormalised.
    pub(crate) trunc: Option<Truncation>,
}

/// One end of a truncation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Bound {
    /// No bound on this side.
    Open,
    /// A fixed value.
    Value(f64),
    /// A θ: its position in [`EtaDensity::shape_thetas`].
    Shape(usize),
}

/// `truncated(KIND, lo, hi)` (#1716): the kind's density on `(lo, hi)`, divided by the mass
/// it puts there. The latent is unconstrained: the model reads `lo + (hi − lo)·inv_logit(z)`
/// with both bounds, `lo + exp(z)` or `hi − exp(z)` with one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Truncation {
    pub(crate) lo: Bound,
    pub(crate) hi: Bound,
}

impl EtaDensity {
    /// The model expression that reads the latent of η slot `eta` as the random effect, or
    /// `None` where the model reads the latent itself. `theta_slot(k)` is θ `k`'s
    /// expression.
    pub(crate) fn read_expr(
        &self,
        eta: crate::parser::model_parser::Expression,
    ) -> Option<crate::parser::model_parser::Expression> {
        use crate::parser::model_parser::{BinOp, Expression as E};
        let bound = |b: Bound| match b {
            Bound::Open => None,
            Bound::Value(v) => Some(E::Literal(v)),
            Bound::Shape(p) => Some(E::Theta(self.shape_thetas[p])),
        };
        let bx = |l: E, op: BinOp, r: E| E::BinOp(Box::new(l), op, Box::new(r));
        let call = |f: &str, a: E| E::UnaryFn(f.to_string(), Box::new(a));
        match self.trunc {
            Some(t) => match (bound(t.lo), bound(t.hi)) {
                (Some(lo), Some(hi)) => Some(bx(
                    lo.clone(),
                    BinOp::Add,
                    bx(bx(hi, BinOp::Sub, lo), BinOp::Mul, call("inv_logit", eta)),
                )),
                (Some(lo), None) => Some(bx(lo, BinOp::Add, call("exp", eta))),
                (None, Some(hi)) => Some(bx(hi, BinOp::Sub, call("exp", eta))),
                (None, None) => self.kind.map().function().map(|f| call(f, eta)),
            },
            None => self.kind.map().function().map(|f| call(f, eta)),
        }
    }
}

use kinds::ln1p_g;

#[path = "eta_density_kinds.rs"]
mod kinds;

/// `c_g(z)` of one group (see the module docs); `+∞` where the parameters leave the
/// kind's domain (a beta variance at or above `μ(1 − μ)`). `eta` is the full latent
/// vector, `oi(a, b)` reads `Ω⁻¹`, `theta(k)` reads θ.
fn group_correction_g<T: PkNum>(
    d: &EtaDensity,
    eta: &[T],
    oi: &impl Fn(usize, usize) -> T,
    theta: &impl Fn(usize) -> T,
) -> T {
    let c = |v: f64| T::from_f64(v);
    let ln_2pi = c(std::f64::consts::TAU.ln());
    let mut q = c(0.0);
    for &a in &d.etas {
        for &b in &d.etas {
            q = q + oi(a, b) * eta[a] * eta[b];
        }
    }
    if let Some(t) = d.trunc {
        // Univariate: the kind's η-density on (lo, hi), renormalised, read through the
        // truncation's own map; less the Gaussian term, as for every kind.
        let a = d.etas[0];
        let z = eta[a];
        let w = c(1.0) / oi(a, a);
        let shapes: Vec<T> = d.shape_thetas.iter().map(|&k| theta(k)).collect();
        let ks = &shapes[..d.kind.shapes().len()];
        let bound = |b: Bound| match b {
            Bound::Open => None,
            Bound::Value(v) => Some(c(v)),
            Bound::Shape(p) => Some(shapes[p]),
        };
        let (lo, hi) = (bound(t.lo), bound(t.hi));
        let (x, log_jac) = match (lo, hi) {
            (Some(lo), Some(hi)) => (
                lo + (hi - lo) * z.inv_logit(),
                (hi - lo).ln() - kinds::softplus_g(-z) - kinds::softplus_g(z),
            ),
            (Some(lo), None) => (lo + z.exp(), z),
            (None, Some(hi)) => (hi - z.exp(), z),
            (None, None) => (z, c(0.0)),
        };
        let cdf = |b: Option<T>, open: f64| match b {
            Some(v) => kinds::eta_cdf(d.kind, v, w, ks),
            None => Some(c(open)),
        };
        let (Some(f_lo), Some(f_hi)) = (cdf(lo, 0.0), cdf(hi, 1.0)) else {
            return c(f64::INFINITY);
        };
        let mass = f_hi - f_lo;
        if !(mass.val() > 0.0) {
            return c(f64::INFINITY);
        }
        let lp = kinds::eta_log_pdf(d.kind, x, w, ks) + log_jac - mass.ln();
        if !lp.val().is_finite() {
            return c(f64::INFINITY);
        }
        return c(-2.0) * lp - (q + w.ln() + ln_2pi);
    }
    if d.kind == DensityKind::StudentT {
        let n = d.etas.len() as f64;
        let nu = theta(d.shape_thetas[0]);
        let half = c(0.5);
        return (nu + c(n)) * ln1p_g(q * (c(1.0) / nu)) - q
            + c(n) * (nu * half).ln()
            + c(2.0) * ((nu * half).ln_gamma() - ((nu + c(n)) * half).ln_gamma());
    }
    // Univariate from here: `q = z²/ω`, and the Gaussian term to subtract is
    // `q + ln ω + ln 2π`.
    let a = d.etas[0];
    let z = eta[a];
    let w = c(1.0) / oi(a, a);
    let shapes: Vec<T> = d.shape_thetas.iter().map(|&k| theta(k)).collect();
    let neg2 = kinds::latent_neg2_logpdf(d.kind, z, w, &shapes);
    if !neg2.val().is_finite() {
        return c(f64::INFINITY);
    }
    neg2 - (q + w.ln() + ln_2pi)
}

/// `Σ_g c_g(η)`: what turns the Gaussian `ηᵀΩ⁻¹η + ln|Ω|` into the exact `−2 ln p(η)`.
/// Exactly `0` with no density groups.
pub(crate) fn correction_g<T: PkNum>(
    densities: &[EtaDensity],
    eta: &[T],
    oi: impl Fn(usize, usize) -> T,
    theta: impl Fn(usize) -> T,
) -> T {
    let mut s = T::from_f64(0.0);
    for d in densities {
        s = s + group_correction_g(d, eta, &oi, &theta);
    }
    s
}

/// [`correction_g`] at `f64`, with `Ω⁻¹` from its cached matrix.
pub(crate) fn correction(
    densities: &[EtaDensity],
    eta: &[f64],
    omega_inv: &DMatrix<f64>,
    theta: &[f64],
) -> f64 {
    correction_g(densities, eta, |a, b| omega_inv[(a, b)], |k| theta[k])
}

/// `½c(η)`: the density correction on the individual objective's `½(−2 ln p)` scale, the
/// share an importance-sampling weight's `−ln p(y | η) − ln N(η; 0, Ω)` is missing.
/// `0` without density groups, `+∞` outside a Box-Cox support. `eta` may be a stacked
/// `[η, κ…]` vector (BSV first) with `omega_inv` its block-diagonal prior precision.
pub(crate) fn half_correction(
    densities: &[EtaDensity],
    eta: &[f64],
    omega_inv: &DMatrix<f64>,
    theta: &[f64],
) -> f64 {
    if densities.is_empty() {
        return 0.0;
    }
    0.5 * correction(densities, eta, omega_inv, theta)
}

/// Add `½∇_η c` into `grad`: the density correction's share of the gradient of the
/// individual objective `½(data + prior)`. A no-op without density groups.
pub(crate) fn add_half_grad(
    densities: &[EtaDensity],
    eta: &[f64],
    omega_inv: &DMatrix<f64>,
    theta: &[f64],
    grad: &mut [f64],
) {
    if densities.is_empty() {
        return;
    }
    let (_, g, _) = correction_jets(densities, eta, omega_inv, theta, false);
    for (acc, gk) in grad.iter_mut().zip(&g) {
        *acc += 0.5 * gk;
    }
}

/// Add `½∇_η c` into `grad` and `½∇²_η c` into `hess`: [`add_half_grad`] with the
/// curvature, for a seed Hessian or `H̃`. A no-op without density groups.
pub(crate) fn add_half_grad_hess(
    densities: &[EtaDensity],
    eta: &[f64],
    omega_inv: &DMatrix<f64>,
    theta: &[f64],
    grad: &mut [f64],
    hess: &mut DMatrix<f64>,
) {
    if densities.is_empty() {
        return;
    }
    let (_, g, h) = correction_jets(densities, eta, omega_inv, theta, true);
    for (acc, gk) in grad.iter_mut().zip(&g) {
        *acc += 0.5 * gk;
    }
    if let Some(h) = h {
        *hess += h * 0.5;
    }
}

/// A density group's correction at an EBE (#1716), for the FOCE/FOCEI marginal: `c` joins
/// the prior term and `½∇²c` the prior's share of `H̃` (the individual objective is
/// `½(data + prior)`, whose prior curvature is `Ω⁻¹ + ½∇²c`).
#[derive(Debug, Clone)]
pub(crate) struct DensityAtMode {
    pub(crate) c: f64,
    pub(crate) half_hess: DMatrix<f64>,
}

impl DensityAtMode {
    /// The correction at `eta`, or `None` without density groups. `eta` may be longer than
    /// `n_η` (a stacked IOV vector, BSV first); only its BSV entries are read.
    pub(crate) fn at(
        densities: &[EtaDensity],
        eta: &[f64],
        omega_inv: &DMatrix<f64>,
        theta: &[f64],
    ) -> Option<Self> {
        if densities.is_empty() {
            return None;
        }
        let (c, _, h) = correction_jets(densities, eta, omega_inv, theta, true);
        Some(Self {
            c,
            half_hess: h.expect("requested") * 0.5,
        })
    }
}

/// `Ω⁻¹`, plus `½∇²c` under a density: the prior's curvature in `H̃`.
pub(crate) fn prior_curvature<'a>(
    omega_inv: &'a DMatrix<f64>,
    density: Option<&DensityAtMode>,
) -> std::borrow::Cow<'a, DMatrix<f64>> {
    match density {
        None => std::borrow::Cow::Borrowed(omega_inv),
        Some(d) => std::borrow::Cow::Owned(omega_inv + &d.half_hess),
    }
}

/// The correction's value, η-gradient and η-Hessian (full `n_η` layout, zero outside the
/// density groups) over any [`PkNum`] `T`. At `T = f64` these are plain numbers; over the
/// AD kernel's types they also carry `∂/∂θ` (ν, λ) and `∂/∂Ω`. Each entry comes from one
/// hyper-dual evaluation of [`correction_g`], seeded on a pair of a group's ETAs, so it is
/// the exact derivative of the one implementation. `want_hessian = false` skips the
/// second-order seeds.
pub(crate) fn correction_jets_g<T: PkNum>(
    densities: &[EtaDensity],
    eta: &[T],
    oi: impl Fn(usize, usize) -> T,
    theta: impl Fn(usize) -> T,
    want_hessian: bool,
) -> (T, Vec<T>, Option<Vec<Vec<T>>>) {
    type H<T> = DualT<1, DualT<1, T>>;
    let n = eta.len();
    let (zero, one) = (T::from_f64(0.0), T::from_f64(1.0));
    let lift = |v: T| -> H<T> {
        DualT {
            v: DualT { v, g: [zero] },
            g: [DualT { v: zero, g: [zero] }],
        }
    };
    let mut value = zero;
    let mut grad = vec![zero; n];
    let mut hess = want_hessian.then(|| vec![vec![zero; n]; n]);
    for d in densities {
        let group = std::slice::from_ref(d);
        // η_a on the inner lane, η_b on the outer one: `v.v` is c, `v.g[0]` is ∂c/∂η_a
        // and `g[0].g[0]` is ∂²c/∂η_a∂η_b.
        let at = |a: usize, b: usize| -> H<T> {
            let e: Vec<H<T>> = (0..n)
                .map(|k| {
                    let mut x = lift(eta[k]);
                    if k == a {
                        x.v.g[0] = one;
                    }
                    if k == b {
                        x.g[0].v = one;
                    }
                    x
                })
                .collect();
            correction_g(group, &e, |i, j| lift(oi(i, j)), |k| lift(theta(k)))
        };
        for (i, &a) in d.etas.iter().enumerate() {
            let pairs: &[usize] = if want_hessian {
                &d.etas[..=i]
            } else {
                &d.etas[i..=i]
            };
            for &b in pairs {
                let r = at(a, b);
                if a == b {
                    grad[a] = r.v.g[0];
                    if i == 0 {
                        value = value + r.v.v;
                    }
                }
                if let Some(h) = hess.as_mut() {
                    h[a][b] = r.g[0].g[0];
                    h[b][a] = r.g[0].g[0];
                }
            }
        }
    }
    (value, grad, hess)
}

/// [`correction_jets_g`] at `f64`, with `Ω⁻¹` from its cached matrix.
pub(crate) fn correction_jets(
    densities: &[EtaDensity],
    eta: &[f64],
    omega_inv: &DMatrix<f64>,
    theta: &[f64],
    want_hessian: bool,
) -> (f64, Vec<f64>, Option<DMatrix<f64>>) {
    let (v, g, h) = correction_jets_g(
        densities,
        eta,
        |a, b| omega_inv[(a, b)],
        |k| theta[k],
        want_hessian,
    );
    let n = eta.len();
    (v, g, h.map(|h| DMatrix::from_fn(n, n, |a, b| h[a][b])))
}

/// Turn a Gaussian simulation draw into the densities' draw (#1716), in place: `eta` holds
/// `η = L·z` from `N(0, Ω)`, and every density group's entries are replaced by a draw of
/// its latent. A t block rescales its Gaussian draw by `√(ν/W)`, `W ~ χ²_ν`, the
/// multivariate t's scale mixture. A univariate kind draws the random effect from its
/// distribution and stores the latent `m⁻¹(η)`. A Gaussian model draws nothing extra, so
/// its random stream is untouched.
/// A latent draw of a truncated group: `η` by the inverse CDF between `F(lo)` and `F(hi)`
/// (bisection on the kind's CDF, inside a bracket that widens until it holds the target),
/// then the truncation map's inverse.
fn sample_truncated(
    d: &EtaDensity,
    t: Truncation,
    w: f64,
    shapes: &[f64],
    rng: &mut impl rand::Rng,
) -> f64 {
    use rand_distr::{Distribution, Uniform};
    let ks = &shapes[..d.kind.shapes().len()];
    let val = |b: Bound| match b {
        Bound::Open => None,
        Bound::Value(v) => Some(v),
        Bound::Shape(p) => Some(shapes[p]),
    };
    let (lo, hi) = (val(t.lo), val(t.hi));
    let cdf = |x: f64| kinds::eta_cdf::<f64>(d.kind, x, w, ks).unwrap_or(f64::NAN);
    let f_lo = lo.map_or(0.0, cdf);
    let f_hi = hi.map_or(1.0, cdf);
    let u: f64 = Uniform::new(0.0f64, 1.0).map_or(0.5, |d| d.sample(rng));
    let target = f_lo + u * (f_hi - f_lo);
    let sd = w.sqrt().max(1e-6);
    let (mut a, mut b) = (lo.unwrap_or(-sd), hi.unwrap_or(sd));
    while lo.is_none() && cdf(a) > target {
        a -= (b - a).max(sd);
    }
    while hi.is_none() && cdf(b) < target {
        b += (b - a).max(sd);
    }
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if cdf(m) < target {
            a = m;
        } else {
            b = m;
        }
    }
    let x = 0.5 * (a + b);
    match (lo, hi) {
        (Some(lo), Some(hi)) => {
            let p = ((x - lo) / (hi - lo)).clamp(1e-300, 1.0 - 1e-16);
            (p / (1.0 - p)).ln()
        }
        (Some(lo), None) => (x - lo).max(1e-300).ln(),
        (None, Some(hi)) => (hi - x).max(1e-300).ln(),
        (None, None) => x,
    }
}

pub(crate) fn redraw_latents(
    densities: &[EtaDensity],
    omega: &DMatrix<f64>,
    theta: &[f64],
    eta: &mut [f64],
    rng: &mut impl rand::Rng,
) {
    use rand_distr::{ChiSquared, Distribution};
    for d in densities {
        let a = d.etas[0];
        let shapes: Vec<f64> = d.shape_thetas.iter().map(|&k| theta[k]).collect();
        if d.kind == DensityKind::StudentT {
            let nu = shapes[0];
            let scale = match ChiSquared::new(nu) {
                Ok(chi) => (nu / chi.sample(rng).max(1e-300)).sqrt(),
                Err(_) => f64::NAN,
            };
            for &k in &d.etas {
                eta[k] *= scale;
            }
            continue;
        }
        let w = omega[(a, a)];
        if let Some(t) = d.trunc {
            eta[a] = sample_truncated(d, t, w, &shapes, rng);
            continue;
        }
        let draw = kinds::sample_eta(d.kind, w, &shapes, rng);
        eta[a] = d.kind.map().invert(draw);
    }
}

/// What a `density` random effect is not carried beside (#1716), as `fit()`'s refusal:
/// FREM, whose covariate ETAs share one correlated Gaussian block with the parameter ETAs,
/// and a `[mixture]` class override of a density ETA's ω (a density reads the Ω every
/// class shares). `None` when the fit can run.
pub(crate) fn fit_refusal(
    model: &crate::types::CompiledModel,
    _options: &crate::types::FitOptions,
    _params: &crate::types::ModelParameters,
) -> Option<String> {
    if model.eta_densities.is_empty() {
        return None;
    }
    if model.frem_config.is_some() {
        return Some(
            "a `density` random effect cannot be combined with FREM: FREM's covariate ETAs \
             share one correlated Gaussian block with the parameter ETAs"
                .into(),
        );
    }
    if let Some(mix) = model.mixture.as_ref() {
        for o in &mix.omega_overrides {
            let hit = model
                .eta_names
                .iter()
                .position(|n| n == &o.name)
                .is_some_and(|k| model.eta_densities.iter().any(|d| d.etas.contains(&k)));
            if hit {
                return Some(format!(
                    "class {} overrides the omega of `{}`, which follows a `density`: a density \
                     reads one Ω shared by every class, so a per-class Ω for it is not supported",
                    o.class, o.name
                ));
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "eta_density_tests.rs"]
mod tests;
