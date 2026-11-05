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
//! are univariate; `lognormal` on a block is the multivariate log-normal, and `copula(…)`
//! correlates any univariate margins through a Gaussian copula. A Box-Cox
//! "density" needs no kind of its own: on its latent scale it is exactly the Box-Cox
//! transform, which `[eta_shape]` declares (`density boxcox` is refused).
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
use nalgebra::{DMatrix, DVector};

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
    /// Normal, mean 0, variance ω: the default, written out only as a copula margin.
    Normal,
    /// A Gaussian copula over a block, each ETA with its own margin
    /// (`EtaDensity::margins`); Ω's diagonal holds the margins' variances, its
    /// correlations the copula's.
    Copula,
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
    pub(crate) const ALL: [DensityKind; 38] = {
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
            Normal,
            Copula,
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
            Normal => &["normal", "normalcanon"],
            Copula => &["copula"],
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

    /// Whether its latent density has a kink or a hard edge where the density is still
    /// positive (Laplace's and the triangle's peak, the uniform's, arcsine's,
    /// log-uniform's and Pareto's edges, the GPD's upper edge for ω < 1). The
    /// FOCE-family curvature reads the prior's Hessian at the EBE, which is undefined at
    /// a kink and blind to an edge, so those methods refuse such a kind (#1721 review);
    /// IMP, SAEM, Bayes and VI score it exactly.
    pub(crate) fn has_kink_or_edge(self) -> bool {
        use DensityKind::*;
        matches!(
            self,
            Laplace | SymTriangular | Uniform | Arcsine | LogUniform | Pareto | GeneralizedPareto
        )
    }

    /// Whether a block of several ETAs can follow it: the multivariate t, the
    /// multivariate log-normal (`lognormal` on a block, Pumas's `MvLogNormal`) and the
    /// copula.
    pub(crate) fn allows_block(self) -> bool {
        matches!(
            self,
            DensityKind::StudentT | DensityKind::LogNormal | DensityKind::Copula
        )
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
    /// A copula's margin per ETA (in `etas` order); their shape θs sit in `shape_thetas`
    /// one margin after another. Empty for every other kind.
    pub(crate) margins: Vec<DensityKind>,
    /// Per ETA of the group, the θ whose log the latent shifts with (#1716): `Some(t)`
    /// when the model reads the ETA as `THETA_t * m(z)` with `m = exp` (or
    /// `THETA_t * exp(z)` on ℝ), the log mu-reference the parser finds on the rewritten
    /// AST and does not keep as a Gaussian mu-reference. Bayes's θ–z shift move
    /// (`log θ_t + δ`, `z − δ`) reads it; empty for a group with none.
    pub(crate) log_anchors: Vec<Option<usize>>,
    /// A copula group's family (#1716): `None` is the Gaussian copula (Ω carries its
    /// correlation, `correction_g`'s closed form). `Some` adds its own dependence θs after
    /// the margins' shapes in `shape_thetas`; a non-elliptical family's Ω block is
    /// diagonal (each margin's ω), an elliptical one's (the t) carries the correlation.
    pub(crate) family: Option<copula::CopulaFamily>,
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
    /// Whether the group's Ω block is diagonal by construction: a non-elliptical copula
    /// (#1716) joins ETAs declared on their own `omega` lines, its dependence lives in its
    /// own θs, and every update of the block must keep the off-diagonals at 0.
    pub(crate) fn diagonal_omega(&self) -> bool {
        self.family.is_some_and(|f| !f.elliptical())
    }

    /// The model expression that reads the latent `eta` of the group's `pos`-th ETA as the
    /// random effect, or `None` where the model reads the latent itself.
    pub(crate) fn read_expr(
        &self,
        pos: usize,
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
        if self.kind == DensityKind::Copula {
            return self.margins[pos].map().function().map(|f| call(f, eta));
        }
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

#[path = "eta_density_copula.rs"]
pub(crate) mod copula;

/// The random effect `m(z)` a reported distribution's model reads at latent `z`: the
/// kind's domain map, or a truncation's bounded map (`lo + (hi − lo)·inv_logit(z)`,
/// `lo + exp(z)`, `hi − exp(z)`), as the parser rewrites the reads. A transform's ETA is
/// itself Gaussian, so it maps to itself; an unknown kind gives `NaN`.
pub(crate) fn effect_of_latent(d: &crate::types::EtaDistribution, z: f64) -> f64 {
    if d.form != "density" {
        return z;
    }
    let Some(kind) = DensityKind::from_name(&d.kind) else {
        return f64::NAN;
    };
    match d.truncation {
        Some((Some(lo), Some(hi))) => lo + (hi - lo) / (1.0 + (-z).exp()),
        Some((Some(lo), None)) => lo + z.exp(),
        Some((None, Some(hi))) => hi - z.exp(),
        _ => kind.map().apply(z),
    }
}

/// The exact mean and variance of a reported distribution's effect (#1721 review), at
/// variance or scale `omega` and the kind's shape values: `kinds::effect_moments`.
/// `None` for a transform, a truncation (renormalised moments are not closed-form here)
/// and a kind without them.
pub(crate) fn effect_moments_of(
    d: &crate::types::EtaDistribution,
    omega: f64,
    shapes: &[f64],
) -> Option<(f64, f64)> {
    if d.form != "density" || d.truncation.is_some() {
        return None;
    }
    let kind = DensityKind::from_name(&d.kind)?;
    kinds::effect_moments(kind, omega, shapes).filter(|(m, v)| m.is_finite() && v.is_finite())
}

/// Effect-scale shrinkage for every density ETA (#1721 review), written over
/// `shrinkage[k]`: `1 − RMS(m(ẑ) − μ) / σ`, with `μ` and `σ²` the effect's exact prior
/// mean and variance (`effect_moments_of`). The Gaussian formula `1 − RMS(ẑ)/√ω` reads
/// the latent against the effect's variance and assumes a centred latent, neither of
/// which holds for a density. `NaN` when the moments have no closed form (a truncation, a
/// t with ν ≤ 2, …) or fewer than two subjects. Gaussian ETAs are left as they are.
pub(crate) fn effect_shrinkage(
    model: &crate::types::CompiledModel,
    theta: &[f64],
    omega: &DMatrix<f64>,
    subjects: &[crate::types::SubjectResult],
    shrinkage: &mut [f64],
) {
    for (k, name) in model.eta_names.iter().enumerate() {
        let Some(d) = model
            .eta_param_info
            .iter()
            .filter(|i| &i.eta_name == name)
            .find_map(|i| i.distribution.as_ref())
        else {
            continue;
        };
        if d.form != "density" || k >= shrinkage.len() || k >= omega.nrows() {
            continue;
        }
        let shapes: Option<Vec<f64>> = d
            .shape_thetas
            .iter()
            .map(|n| {
                let t = model.theta_names.iter().position(|x| x == n)?;
                theta.get(t).copied()
            })
            .collect();
        let moments = shapes
            .as_deref()
            .and_then(|s| effect_moments_of(d, omega[(k, k)], s))
            .filter(|&(_, v)| v > 0.0);
        shrinkage[k] = match moments {
            Some((mu, v)) if subjects.len() >= 2 => {
                let ms = subjects
                    .iter()
                    .map(|s| (effect_of_latent(d, s.eta[k]) - mu).powi(2))
                    .sum::<f64>()
                    / subjects.len() as f64;
                1.0 - ms.sqrt() / v.sqrt()
            }
            _ => f64::NAN,
        };
    }
}

/// The 5th, 50th and 95th percentiles of the random effect a reported distribution
/// describes, at variance (or scale²) `omega` and shape values `shapes` (the kind's own,
/// with a truncation's bounds already resolved in `d`) — what the fit
/// printout states for a non-Gaussian ETA, since its spread is no CV% or SD on the scales
/// the Gaussian labels describe, and some kinds (Cauchy, Lévy, t with ν ≤ 2) have no SD.
///
/// Exact up to bisection: a `transform` is monotone in η ~ N(0, ω), so its quantiles are
/// `h(√ω·Φ⁻¹(p))`; a `density` inverts its CDF on the latent scale, through the domain
/// map, so the bracket covers the whole support. `None` for an unknown kind or for
/// parameters that admit no distribution.
pub(crate) fn effect_quantiles(
    d: &crate::types::EtaDistribution,
    omega: f64,
    shapes: &[f64],
) -> Option<[f64; 3]> {
    const P: [f64; 3] = [0.05, 0.5, 0.95];
    if !(omega.is_finite() && omega > 0.0) {
        return None;
    }
    if d.form == "transform" {
        let kind = crate::parser::eta_shape::ShapeKind::from_name(&d.kind)?;
        let s = shapes.first().copied().unwrap_or(f64::NAN);
        let q = P.map(|p| {
            crate::parser::eta_shape::shape_f64(
                kind,
                crate::parser::eta_shape::ShapeOut::Value,
                omega.sqrt() * crate::stats::special::normal_inv_cdf(p),
                s,
            )
        });
        return q.iter().all(|v| v.is_finite()).then_some(q);
    }
    let kind = DensityKind::from_name(&d.kind)?;
    let cdf = |x: f64| kinds::eta_cdf::<f64>(kind, x, omega, shapes).filter(|f| f.is_finite());
    let map = kind.map();
    let at = |z: f64| match map {
        DomainMap::Identity => z,
        DomainMap::Exp => z.exp(),
        DomainMap::InvLogit => 1.0 / (1.0 + (-z).exp()),
    };
    // The latent bracket: the whole of ℝ mapped onto (0, ∞) / (0, 1) at ±700; on ℝ a
    // scale-relative ±40·√ω, widened until it holds every target (a heavy tail).
    let mut half = match map {
        DomainMap::Identity => 40.0 * omega.sqrt(),
        _ => 700.0,
    };
    let (f_lo, f_hi) = match d.truncation {
        Some((lo, hi)) => (lo.map_or(Some(0.0), cdf)?, hi.map_or(Some(1.0), cdf)?),
        None => (0.0, 1.0),
    };
    if !(f_hi > f_lo) {
        return None;
    }
    let targets = P.map(|p| f_lo + p * (f_hi - f_lo));
    for _ in 0..60 {
        if cdf(at(-half))? <= targets[0] && cdf(at(half))? >= targets[2] {
            break;
        }
        if map != DomainMap::Identity {
            return None;
        }
        half *= 2.0;
    }
    let mut q = [0.0; 3];
    for (qi, &t) in q.iter_mut().zip(&targets) {
        let (mut a, mut b) = (-half, half);
        for _ in 0..200 {
            let m = 0.5 * (a + b);
            if cdf(at(m))? < t {
                a = m;
            } else {
                b = m;
            }
        }
        *qi = at(0.5 * (a + b));
    }
    Some(q)
}

/// The inverse and log-determinant of a symmetric positive-definite matrix over `T`, by
/// Cholesky; `None` when it does not factor.
fn spd_inverse<T: PkNum>(a: &[Vec<T>]) -> Option<(Vec<Vec<T>>, T)> {
    let n = a.len();
    let z = T::from_f64(0.0);
    let mut l = vec![vec![z; n]; n];
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d = d - l[j][k] * l[j][k];
        }
        if !(d.val() > 0.0) {
            return None;
        }
        l[j][j] = d.sqrt();
        for i in j + 1..n {
            let mut s = a[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k];
            }
            l[i][j] = s / l[j][j];
        }
    }
    let mut ld = z;
    for (i, row) in l.iter().enumerate() {
        ld = ld + T::from_f64(2.0) * row[i].ln();
    }
    // L⁻¹ by forward substitution, A⁻¹ = L⁻ᵀL⁻¹.
    let mut li = vec![vec![z; n]; n];
    for i in 0..n {
        li[i][i] = T::from_f64(1.0) / l[i][i];
        for j in 0..i {
            let mut s = z;
            for k in j..i {
                s = s + l[i][k] * li[k][j];
            }
            li[i][j] = -s / l[i][i];
        }
    }
    let mut inv = vec![vec![z; n]; n];
    for a_ in 0..n {
        for b in 0..n {
            let mut s = z;
            for row in &li {
                s = s + row[a_] * row[b];
            }
            inv[a_][b] = s;
        }
    }
    Some((inv, ld))
}

/// `Φ⁻¹(u)` over `T`: the `f64` quantile, then three Newton steps over `T` (each doubles
/// the order to which the jets are the exact quantile's).
fn normal_quantile<T: PkNum>(u: T) -> T {
    let q0 = crate::stats::special::normal_inv_cdf(u.val().clamp(1e-300, 1.0 - 1e-16));
    let mut q = T::from_f64(q0);
    let inv_sqrt_2pi = 1.0 / std::f64::consts::TAU.sqrt();
    for _ in 0..3 {
        let phi = (q * q * T::from_f64(-0.5)).exp() * T::from_f64(inv_sqrt_2pi);
        q = q - (crate::stats::special::normal_cdf_g(q) - u) / phi;
    }
    q
}

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
    if d.kind == DensityKind::Copula || (d.kind == DensityKind::LogNormal && d.etas.len() > 1) {
        let n = d.etas.len();
        // Ω of the block, from Ω⁻¹'s block (block-diagonal across groups).
        let oi_g: Vec<Vec<T>> = (0..n)
            .map(|i| (0..n).map(|j| oi(d.etas[i], d.etas[j])).collect())
            .collect();
        let Some((om, ld_oi)) = spd_inverse(&oi_g) else {
            return c(f64::INFINITY);
        };
        let ld_om = -ld_oi;
        let gauss = q + ld_om + c(n as f64) * ln_2pi;
        let z: Vec<T> = d.etas.iter().map(|&a| eta[a]).collect();
        if d.kind == DensityKind::LogNormal {
            // MvLogNormal, mean 1: ln η ~ N(−diag(S)/2, S), S = ln(1 + Ω) elementwise.
            let sm: Vec<Vec<T>> = om
                .iter()
                .map(|r| r.iter().map(|&v| ln1p_g(v)).collect())
                .collect();
            let Some((si, ld_si)) = spd_inverse(&sm) else {
                return c(f64::INFINITY);
            };
            let r: Vec<T> = (0..n).map(|i| z[i] + sm[i][i] * c(0.5)).collect();
            let mut qs = c(0.0);
            for i in 0..n {
                for j in 0..n {
                    qs = qs + si[i][j] * r[i] * r[j];
                }
            }
            return qs + ld_si + c(n as f64) * ln_2pi - gauss;
        }
        // Gaussian copula: Σₖ −2 ln pₖ(zₖ; Ωₖₖ) + ln|R| + qᵀ(R⁻¹ − I)q, qₖ = Φ⁻¹(Fₖ(zₖ)),
        // R⁻¹ = D^{½}Ω⁻¹D^{½}, ln|R| = ln|Ω| − Σ ln Ωₖₖ.
        let shapes: Vec<T> = d.shape_thetas.iter().map(|&k| theta(k)).collect();
        let mut at = 0;
        let mut total = c(0.0);
        let mut qv = Vec::with_capacity(n);
        let mut uv = Vec::with_capacity(n);
        let mut ld_r = ld_om;
        for (i, &kind) in d.margins.iter().enumerate() {
            let m = kind.shapes().len();
            let sk = &shapes[at..at + m];
            at += m;
            let w = om[i][i];
            ld_r = ld_r - w.ln();
            let neg2 = if kind == DensityKind::StudentT {
                c(-2.0) * kinds::eta_log_pdf(kind, z[i], w, sk)
            } else {
                kinds::latent_neg2_logpdf(kind, z[i], w, sk)
            };
            let x = match kind.map() {
                DomainMap::Identity => z[i],
                DomainMap::Exp => z[i].exp(),
                DomainMap::InvLogit => z[i].inv_logit(),
            };
            let Some(u) = kinds::eta_cdf(kind, x, w, sk) else {
                return c(f64::INFINITY);
            };
            if !(neg2.val().is_finite() && u.val() > 0.0 && u.val() < 1.0) {
                return c(f64::INFINITY);
            }
            total = total + neg2;
            if d.family.is_none() {
                qv.push(normal_quantile(u));
            }
            uv.push(u);
        }
        // #1716: a copula family other than the Gaussian; its own θs follow the margins'.
        if let Some(fam) = d.family {
            let fp = &shapes[at..];
            let lc = if fam == copula::CopulaFamily::StudentT {
                // R⁻¹ = D^{½}Ω⁻¹D^{½}, ln|R| = ln|Ω| − Σ ln Ωₖₖ, as for the Gaussian.
                let r_inv: Vec<Vec<T>> = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| oi_g[i][j] * (om[i][i] * om[j][j]).sqrt())
                            .collect()
                    })
                    .collect();
                copula::t_copula_log_density(&uv, fp[0], &r_inv, ld_r)
            } else {
                fam.log_density(&uv, fp)
            };
            if !lc.val().is_finite() {
                return c(f64::INFINITY);
            }
            return total - c(2.0) * lc - gauss;
        }
        let mut quad = c(0.0);
        for i in 0..n {
            for j in 0..n {
                let rij = oi_g[i][j] * (om[i][i] * om[j][j]).sqrt();
                let id = if i == j { c(1.0) } else { c(0.0) };
                quad = quad + (rij - id) * qv[i] * qv[j];
            }
        }
        return total + ld_r + quad - gauss;
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

/// Every density group over a stacked `[η, κ₀, …, κ_{K−1}]` vector (#1716): the BSV
/// groups as declared, and each κ group once per occasion, at `n_η + g·n_κ + j`. Borrowed
/// when the model has no κ density, so the BSV-only path allocates nothing.
pub(crate) fn stacked_groups(
    model: &crate::types::CompiledModel,
    k_occasions: usize,
) -> std::borrow::Cow<'_, [EtaDensity]> {
    if model.kappa_densities.is_empty() || k_occasions == 0 {
        return std::borrow::Cow::Borrowed(&model.eta_densities);
    }
    let mut out = model.eta_densities.clone();
    for g in 0..k_occasions {
        for d in &model.kappa_densities {
            let mut dg = d.clone();
            dg.etas = d
                .etas
                .iter()
                .map(|&j| model.n_eta + g * model.n_kappa + j)
                .collect();
            out.push(dg);
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Every density group over a layout's vector (#1716): the BSV groups as declared, and
/// each κ group placed in every block `(κ indices, base)` of the layout that holds all its
/// κs (one per occasion for one level, one per group of the κs' level under nested
/// levels). A κ group's κs must share one level; one that spans two is never placed.
pub(crate) fn layout_groups(
    model: &crate::types::CompiledModel,
    blocks: &[(Vec<usize>, usize)],
) -> Vec<EtaDensity> {
    let mut out = model.eta_densities.clone();
    for (ks, base) in blocks {
        for d in &model.kappa_densities {
            let pos: Option<Vec<usize>> = d
                .etas
                .iter()
                .map(|j| ks.iter().position(|k| k == j).map(|p| base + p))
                .collect();
            if let Some(etas) = pos {
                let mut dg = d.clone();
                dg.etas = etas;
                out.push(dg);
            }
        }
    }
    out
}

/// The κ part of a nested subject's prior correction (#1716): every κ density group in
/// each level block of the compact vector `b`, against that level's `Ω_IOV⁻¹` block.
pub(crate) fn nested_kappa_correction(
    model: &crate::types::CompiledModel,
    blocks: &[(Vec<usize>, usize)],
    b: &[f64],
    omega_iov_inv: &DMatrix<f64>,
    theta: &[f64],
) -> f64 {
    let mut total = 0.0;
    for (ks, base) in blocks {
        let w = ks.len();
        let local = &b[*base..base + w];
        let inv = DMatrix::from_fn(w, w, |i, j| omega_iov_inv[(ks[i], ks[j])]);
        let groups: Vec<EtaDensity> = model
            .kappa_densities
            .iter()
            .filter_map(|d| {
                let etas: Option<Vec<usize>> = d
                    .etas
                    .iter()
                    .map(|j| ks.iter().position(|k| k == j))
                    .collect();
                etas.map(|etas| {
                    let mut dg = d.clone();
                    dg.etas = etas;
                    dg
                })
            })
            .collect();
        total += correction(&groups, local, &inv, theta);
    }
    total
}

/// [`half_correction`] over a stacked `[η, κ₀, …]` vector of any length: the occasion
/// count is read off the length (`(len − n_η)/n_κ`), so IS and VI draws of a BSV-only or
/// an IOV subject take one call.
pub(crate) fn half_correction_stacked(
    model: &crate::types::CompiledModel,
    eta: &[f64],
    omega_inv: &DMatrix<f64>,
    theta: &[f64],
) -> f64 {
    if !model.has_densities() {
        return 0.0;
    }
    let k = if model.n_kappa > 0 && eta.len() > model.n_eta {
        (eta.len() - model.n_eta) / model.n_kappa
    } else {
        0
    };
    half_correction(&stacked_groups(model, k), eta, omega_inv, theta)
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

/// `F⁻¹(u)` of a univariate kind at the random effect's scale, by bisection on its CDF in
/// a bracket that widens until it holds `u`.
fn margin_quantile(kind: DensityKind, u: f64, w: f64, s: &[f64]) -> f64 {
    let cdf = |x: f64| kinds::eta_cdf::<f64>(kind, x, w, s).unwrap_or(f64::NAN);
    let (mut a, mut b) = match kind.map() {
        DomainMap::Identity => (-w.sqrt(), w.sqrt()),
        DomainMap::Exp => (1e-12, 1.0),
        DomainMap::InvLogit => (1e-15, 1.0 - 1e-15),
    };
    if kind.map() == DomainMap::Identity {
        while cdf(a) > u {
            a -= 2.0 * (b - a);
        }
    }
    if kind.map() != DomainMap::InvLogit {
        while cdf(b) < u {
            b += 2.0 * (b - a);
        }
    }
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if cdf(m) < u {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
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
        if d.kind == DensityKind::LogNormal && d.etas.len() > 1 {
            // MvLogNormal: ln η ~ N(−diag(S)/2, S), S = ln(1 + Ω): redraw from S's factor.
            let n = d.etas.len();
            let sm = DMatrix::from_fn(n, n, |i, j| omega[(d.etas[i], d.etas[j])].ln_1p());
            let Some(l) = sm.clone().cholesky().map(|c| c.l()) else {
                continue;
            };
            let zs = DVector::from_fn(n, |_, _| {
                rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng)
            });
            let x = l * zs;
            for i in 0..n {
                eta[d.etas[i]] = -0.5 * sm[(i, i)] + x[i];
            }
            continue;
        }
        if d.kind == DensityKind::Copula {
            // The copula's uniforms: for the Gaussian, q ~ N(0, R) from the Gaussian draw
            // (η = L·z already correlated by Ω, so q = D^{-½}η) and u = Φ(q); for another
            // family, its own draw (#1716). Then each margin by its inverse CDF.
            let n = d.etas.len();
            let n_margin_shapes: usize = d.margins.iter().map(|k| k.shapes().len()).sum();
            let fam_u: Option<Vec<f64>> = d.family.map(|fam| {
                let fp = &shapes[n_margin_shapes..];
                if fam == copula::CopulaFamily::StudentT {
                    let r = DMatrix::from_fn(n, n, |i, j| {
                        let (a, b) = (d.etas[i], d.etas[j]);
                        omega[(a, b)] / (omega[(a, a)] * omega[(b, b)]).sqrt()
                    });
                    match r.cholesky() {
                        Some(ch) => copula::sample_t_copula(fp[0], &ch.l(), rng),
                        None => vec![f64::NAN; n],
                    }
                } else {
                    fam.sample(n, fp, rng)
                }
            });
            let mut at = 0;
            for (i, &kind) in d.margins.iter().enumerate() {
                let m = kind.shapes().len();
                let sk = &shapes[at..at + m];
                at += m;
                let a = d.etas[i];
                let w = omega[(a, a)];
                let u = match &fam_u {
                    Some(us) => us[i],
                    None => crate::stats::special::normal_cdf(eta[a] / w.sqrt())
                        .clamp(1e-300, 1.0 - 1e-16),
                };
                let x = margin_quantile(kind, u, w, sk);
                eta[a] = kind.map().invert(x);
            }
            continue;
        }
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

/// What a `density` random effect is not carried beside (#1716), as `fit()`'s refusal: a
/// kink or edge kind (`DensityKind::has_kink_or_edge`) under a FOCE-family or Laplace
/// method, and a κ density under nested IOV levels (which share κs across cells). `None`
/// when the fit can run.
pub(crate) fn fit_refusal(
    model: &crate::types::CompiledModel,
    options: &crate::types::FitOptions,
    _params: &crate::types::ModelParameters,
) -> Option<String> {
    if model.eta_densities.is_empty() && model.kappa_densities.is_empty() {
        return None;
    }
    // A kink or an edge under a method that reads the prior's curvature at the EBE.
    use crate::types::EstimationMethod as M;
    // VI's `vi_final_ofv = laplace` reports a Laplace objective too.
    let curvature = options.method_chain().into_iter().find(|m| {
        matches!(
            m,
            M::Foce | M::FoceI | M::FoceGn | M::FoceGnHybrid | M::Laplace
        ) || (*m == M::Vi && options.vi_final_ofv == crate::types::ViFinalOfv::Laplace)
    });
    if let Some(m) = curvature {
        for d in model.eta_densities.iter().chain(&model.kappa_densities) {
            let kinds = if d.kind == DensityKind::Copula {
                d.margins.clone()
            } else {
                vec![d.kind]
            };
            if let Some(k) = kinds.into_iter().find(|k| k.has_kink_or_edge()) {
                return Some(format!(
                    "`density {}` has a kink or an edge where its density is positive, \
                     so {} would read a prior curvature that does not exist there and \
                     print an objective that may be wrong; fit it with `method = imp`, \
                     `saem` or `bayes`, which score it exactly",
                    k.name(),
                    m.label()
                ));
            }
        }
    }
    // A κ density group spanning two nested levels is never placed in one block: refuse.
    if let Some(nl) = model.nested_levels() {
        for d in &model.kappa_densities {
            let lv: Vec<usize> = d.etas.iter().map(|&j| nl.kappa_level[j]).collect();
            if lv.windows(2).any(|w| w[0] != w[1]) {
                return Some("a κ `density` block must keep its κs at one nested level".into());
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "eta_density_tests.rs"]
mod tests;
