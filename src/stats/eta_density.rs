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
//! are univariate; `lognormal` / `lognormal_mean` on a block is the multivariate
//! log-normal of median / mean 1, and `copula(…)`
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
//! η-Hessian, and the likelihood kernel's compiled plans (#2786) record the same function,
//! so the adjoint's `∂/∂θ` (the shape) and `∂/∂Ω` are its exact derivatives too. Every
//! branch on a value goes through the branch API (`br_cmp`, `select_if`, `br_floor_div`),
//! so a recording guards it, and every special function is a recorded primitive (`JetFn`,
//! `gamma_pq`) or a `HostFn` root refined by Newton steps over `T`.

use crate::sens::jet::DualT;
use crate::sens::num::{BranchCmp as B, JetFn, PkNum};
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
    /// Log-normal, median 1: `ln η ~ N(0, ω)`, the classic `TV·exp(η)` random effect,
    /// whose latent prior is exactly Gaussian (correction 0). On a block, Pumas's
    /// `MvLogNormal` with `ln η ~ N(0, Ω)`.
    LogNormal,
    /// Log-normal, mean 1, variance ω (`ln η ~ N(−s²/2, s²)`, `s² = ln(1 + ω)`). On a
    /// block, `ln η ~ N(−diag(S)/2, S)` with `S = ln(1 + Ω)` elementwise.
    LogNormalMean,
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
    prefix: &'static str, init: f64, lower: f64, upper: f64, domain: (f64, f64), strict: bool,
) -> ShapeSpec {
    ShapeSpec { prefix, init, lower, upper, domain, strict }
}

impl DensityKind {
    /// Every kind, for name lookup.
    pub(crate) const ALL: [DensityKind; 39] = {
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
            LogNormalMean,
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
            StudentT => &["t", "studentt"],
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
            LogNormalMean => &["lognormal_mean"],
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
            Gamma | InvGamma | LogNormal | LogNormalMean | Exponential | Erlang | ChiSq | Chi
            | Rayleigh | Weibull | Frechet | InverseGaussian | BetaPrime | NoncentralChiSq
            | Kolmogorov | Levy | Pareto | GeneralizedPareto | LogUniform => DomainMap::Exp,
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
    /// multivariate log-normal (`lognormal` or `lognormal_mean` on a block, Pumas's
    /// `MvLogNormal`) and the
    /// copula.
    pub(crate) fn allows_block(self) -> bool {
        matches!(
            self,
            DensityKind::StudentT
                | DensityKind::LogNormal
                | DensityKind::LogNormalMean
                | DensityKind::Copula
        )
    }

    /// Whether the kind has no free spread, so its `ω` is unused and must be `FIX`.
    pub(crate) fn omega_unused(self) -> bool {
        use DensityKind::*;
        matches!(self, Exponential | Erlang | Rayleigh | Kolmogorov | LogUniform)
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
    /// The model expression that reads the latent `eta` of the group's `pos`-th ETA as the
    /// random effect, or `None` where the model reads the latent itself.
    pub(crate) fn read_expr(
        &self, pos: usize, eta: crate::parser::model_parser::Expression,
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

pub(crate) use kinds::implied_shape_root;
use kinds::ln1p_g;

#[path = "eta_density_kinds.rs"]
mod kinds;

/// `d` with its θ-valued truncation bounds at the values `theta_of(name)` gives (a fit's
/// estimates, or the parameters a prediction runs at), so the effect it maps to is the
/// model's at those parameters (#2669: they held the initial θ). A bound whose θ is not
/// found keeps its stored value.
pub(crate) fn resolve_distribution(
    d: &crate::types::EtaDistribution, theta_of: impl Fn(&str) -> Option<f64>,
) -> crate::types::EtaDistribution {
    let mut out = d.clone();
    if let Some((lo, hi)) = out.truncation.as_mut() {
        let (tlo, thi) = &d.truncation_thetas;
        if let Some(v) = tlo.as_deref().and_then(&theta_of) {
            *lo = Some(v);
        }
        if let Some(v) = thi.as_deref().and_then(&theta_of) {
            *hi = Some(v);
        }
    }
    out
}

/// Per BSV ETA of `model`, the `density` distribution its random effect follows, resolved
/// at `theta` ([`resolve_distribution`]); `None` for a Gaussian or transform ETA, whose
/// effect is the ETA itself. With [`effect_of_latent`] / [`latent_of_effect`] it maps
/// between the latent the estimators hold and the effect every output reports (#2669).
pub(crate) fn model_effect_maps(
    model: &crate::types::CompiledModel, theta: &[f64],
) -> Vec<Option<crate::types::EtaDistribution>> {
    model
        .eta_names
        .iter()
        .map(|name| {
            let d = model
                .eta_param_info
                .iter()
                .filter(|i| &i.eta_name == name)
                .find_map(|i| i.distribution.as_ref())
                .filter(|d| d.form == "density")?;
            Some(resolve_distribution(d, |n| {
                let t = model.theta_names.iter().position(|x| x == n)?;
                theta.get(t).copied()
            }))
        })
        .collect()
}

/// The latent `z` a reported distribution's model reads as the random effect `x`: the
/// inverse of [`effect_of_latent`]. `NaN` for an `x` outside the effect's support (a
/// non-positive effect of a (0, ∞) kind, one outside a truncation), and for an unknown
/// kind. A transform's ETA is itself Gaussian and maps to itself.
pub(crate) fn latent_of_effect(d: &crate::types::EtaDistribution, x: f64) -> f64 {
    if d.form != "density" {
        return x;
    }
    let Some(kind) = DensityKind::from_name(&d.kind) else {
        return f64::NAN;
    };
    let pos = |v: f64| if v > 0.0 { v.ln() } else { f64::NAN };
    match d.truncation {
        Some((Some(lo), Some(hi))) => {
            let u = (x - lo) / (hi - lo);
            if u > 0.0 && u < 1.0 {
                (u / (1.0 - u)).ln()
            } else {
                f64::NAN
            }
        }
        Some((Some(lo), None)) => pos(x - lo),
        Some((None, Some(hi))) => pos(hi - x),
        _ => match kind.map() {
            DomainMap::Identity => x,
            DomainMap::Exp => pos(x),
            DomainMap::InvLogit if x > 0.0 && x < 1.0 => (x / (1.0 - x)).ln(),
            DomainMap::InvLogit => f64::NAN,
        },
    }
}

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
    d: &crate::types::EtaDistribution, omega: f64, shapes: &[f64],
) -> Option<(f64, f64)> {
    if d.form != "density" || d.truncation.is_some() {
        return None;
    }
    let kind = DensityKind::from_name(&d.kind)?;
    kinds::effect_moments(kind, omega, shapes).filter(|(m, v)| m.is_finite() && v.is_finite())
}

/// The values of a distribution's shape θs, read from `theta` by `theta_names`.
fn shapes_by_name(
    d: &crate::types::EtaDistribution, theta_names: &[String], theta: &[f64],
) -> Option<Vec<f64>> {
    d.shape_thetas
        .iter()
        .map(|n| {
            let t = theta_names.iter().position(|x| x == n)?;
            theta.get(t).copied()
        })
        .collect()
}

/// The prior SD of a `density` ETA's effect at variance (or scale²) `omega` and the
/// model's `theta`; `NaN` where it has no closed form (a truncation, a t with ν ≤ 2, …).
pub(crate) fn effect_sd(
    d: &crate::types::EtaDistribution, omega: f64, model: &crate::types::CompiledModel,
    theta: &[f64],
) -> f64 {
    shapes_by_name(d, &model.theta_names, theta)
        .and_then(|s| effect_moments_of(d, omega, &s))
        .filter(|&(_, v)| v > 0.0)
        .map_or(f64::NAN, |(_, v)| v.sqrt())
}

/// Effect-scale shrinkage for every density ETA (#1721 review), written over
/// `shrinkage[k]`. Where the effect's prior mean `μ` and variance `σ²` exist in closed form
/// (`effect_moments_of`) it is `1 − RMS(m(ẑ) − μ) / σ`: the Gaussian formula `1 − RMS(ẑ)/√ω`
/// reads the latent against the effect's variance and assumes a centred latent, neither of
/// which holds for a density. Where they do not (a truncation, a t with ν ≤ 2, Cauchy,
/// Lévy, …) it is the quantile analogue `1 − IQR(m(ẑ)) / IQR(prior)` (#2669), which equals
/// the first for a Gaussian EBE spread and needs no moment. `NaN` with fewer than four
/// subjects, or where neither exists. Gaussian ETAs are left as they are.
pub(crate) fn effect_shrinkage(
    model: &crate::types::CompiledModel, theta: &[f64], omega: &DMatrix<f64>,
    subjects: &[crate::types::SubjectResult], shrinkage: &mut [f64],
) {
    let maps = model_effect_maps(model, theta);
    for (k, d) in maps.iter().enumerate() {
        let Some(d) = d else { continue };
        if k >= shrinkage.len() || k >= omega.nrows() {
            continue;
        }
        let effects: Vec<f64> = subjects.iter().map(|s| effect_of_latent(d, s.eta[k])).collect();
        shrinkage[k] = effect_shrinkage_of(d, omega[(k, k)], model, theta, &effects);
    }
}

/// [`effect_shrinkage`] of one ETA, from its subjects' effects `m(ẑ)`.
pub(crate) fn effect_shrinkage_of(
    d: &crate::types::EtaDistribution, omega: f64, model: &crate::types::CompiledModel,
    theta: &[f64], effects: &[f64],
) -> f64 {
    let n = effects.len();
    if n < 2 || effects.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    let Some(shapes) = shapes_by_name(d, &model.theta_names, theta) else {
        return f64::NAN;
    };
    if let Some((mu, v)) = effect_moments_of(d, omega, &shapes).filter(|&(_, v)| v > 0.0) {
        let ms = effects.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n as f64;
        return 1.0 - ms.sqrt() / v.sqrt();
    }
    if n < 4 {
        return f64::NAN;
    }
    let Some(q) = effect_quantiles_at(d, omega, &shapes, &[0.25, 0.75]) else {
        return f64::NAN;
    };
    let iqr_prior = q[1] - q[0];
    if !(iqr_prior > 0.0) {
        return f64::NAN;
    }
    let mut e = effects.to_vec();
    e.sort_by(f64::total_cmp);
    1.0 - (sample_quantile(&e, 0.75) - sample_quantile(&e, 0.25)) / iqr_prior
}

/// The `p` quantile of sorted `x`, linearly interpolated (type 7, R's default).
fn sample_quantile(x: &[f64], p: f64) -> f64 {
    let h = (x.len() - 1) as f64 * p;
    let (lo, frac) = (h.floor() as usize, h - h.floor());
    let hi = (lo + 1).min(x.len() - 1);
    x[lo] + frac * (x[hi] - x[lo])
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
    d: &crate::types::EtaDistribution, omega: f64, shapes: &[f64],
) -> Option<[f64; 3]> {
    let q = effect_quantiles_at(d, omega, shapes, &[0.05, 0.5, 0.95])?;
    Some([q[0], q[1], q[2]])
}

/// [`effect_quantiles`] at the probabilities `p`, ascending and inside (0, 1).
pub(crate) fn effect_quantiles_at(
    d: &crate::types::EtaDistribution, omega: f64, shapes: &[f64], p: &[f64],
) -> Option<Vec<f64>> {
    let n_p = p.len();
    if n_p == 0 {
        return Some(Vec::new());
    }
    if !(omega.is_finite() && omega > 0.0) {
        return None;
    }
    if d.form == "transform" {
        let kind = crate::parser::eta_shape::ShapeKind::from_name(&d.kind)?;
        let s = shapes.first().copied().unwrap_or(f64::NAN);
        let q: Vec<f64> = p
            .iter()
            .map(|&p| {
                crate::parser::eta_shape::shape_f64(
                    kind,
                    crate::parser::eta_shape::ShapeOut::Value,
                    omega.sqrt() * crate::stats::special::normal_inv_cdf(p),
                    s,
                )
            })
            .collect();
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
    let targets: Vec<f64> = p.iter().map(|p| f_lo + p * (f_hi - f_lo)).collect();
    for _ in 0..60 {
        if cdf(at(-half))? <= targets[0] && cdf(at(half))? >= targets[n_p - 1] {
            break;
        }
        if map != DomainMap::Identity {
            return None;
        }
        half *= 2.0;
    }
    let mut q = vec![0.0; n_p];
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
///
/// Its own factor loop, not `sens::focei_kernel::chol`: a second caller of that function from
/// this module cost the weibull fit 25% (0.40 s against 0.32 s, `ci-test`, medians of 7) — the
/// likely mechanism is that with LTO off it stops being inlined in the likelihood kernel.
fn spd_inverse<T: PkNum>(a: &[Vec<T>]) -> Option<(Vec<Vec<T>>, T)> {
    let n = a.len();
    let z = T::from_f64(0.0);
    let mut l = vec![vec![z; n]; n];
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d = d - l[j][k] * l[j][k];
        }
        if !d.br_cmp_c(B::Gt, 0.0) {
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

/// `Φ⁻¹(u)` over `T`.
fn normal_quantile<T: PkNum>(u: T) -> T {
    u.jet_unary(JetFn::NormInvCdf)
}

/// `c_g(z)` of one group (see the module docs); `+∞` where the parameters leave the
/// kind's domain (a beta variance at or above `μ(1 − μ)`). `eta` is the full latent
/// vector, `oi(a, b)` reads `Ω⁻¹`, `theta(k)` reads θ.
fn group_correction_g<T: PkNum>(
    d: &EtaDensity, eta: &[T], oi: &impl Fn(usize, usize) -> T, theta: &impl Fn(usize) -> T,
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
        if !mass.br_cmp_c(B::Gt, 0.0) {
            return c(f64::INFINITY);
        }
        let lp = kinds::eta_log_pdf(d.kind, x, w, ks) + log_jac - mass.ln();
        if !kinds::finite(lp) {
            return c(f64::INFINITY);
        }
        return c(-2.0) * lp - (q + w.ln() + ln_2pi);
    }
    // Log-normal of median 1, scalar or MvLogNormal: `ln η ~ N(0, Ω)` is the Gaussian prior
    // itself, so the correction is exactly 0.
    if d.kind == DensityKind::LogNormal {
        return c(0.0);
    }
    if d.kind == DensityKind::Copula || (d.kind == DensityKind::LogNormalMean && d.etas.len() > 1) {
        let n = d.etas.len();
        // Ω of the block, from Ω⁻¹'s block (block-diagonal across groups).
        let oi_g: Vec<Vec<T>> =
            (0..n).map(|i| (0..n).map(|j| oi(d.etas[i], d.etas[j])).collect()).collect();
        let Some((om, ld_oi)) = spd_inverse(&oi_g) else {
            return c(f64::INFINITY);
        };
        let ld_om = -ld_oi;
        let gauss = q + ld_om + c(n as f64) * ln_2pi;
        let z: Vec<T> = d.etas.iter().map(|&a| eta[a]).collect();
        if d.kind == DensityKind::LogNormalMean {
            // MvLogNormal, mean 1: ln η ~ N(−diag(S)/2, S), S = ln(1 + Ω) elementwise.
            let sm: Vec<Vec<T>> =
                om.iter().map(|r| r.iter().map(|&v| ln1p_g(v)).collect()).collect();
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
            if !kinds::finite(neg2) {
                return c(f64::INFINITY);
            }
            total = total + neg2;
            // A normal margin's score is its standardised latent: `Φ⁻¹(Φ(z/√ω))` exactly,
            // without the round trip through a CDF that saturates in the tails.
            if kind == DensityKind::Normal {
                qv.push(z[i] / w.sqrt());
                continue;
            }
            let Some(u) = kinds::eta_cdf(kind, x, w, sk) else {
                return c(f64::INFINITY);
            };
            if !(u.br_cmp_c(B::Gt, 0.0) && u.br_cmp_c(B::Lt, 1.0)) {
                return c(f64::INFINITY);
            }
            qv.push(normal_quantile(u));
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
    if !kinds::finite(neg2) {
        return c(f64::INFINITY);
    }
    neg2 - (q + w.ln() + ln_2pi)
}

/// `Σ_g c_g(η)`: what turns the Gaussian `ηᵀΩ⁻¹η + ln|Ω|` into the exact `−2 ln p(η)`.
/// Exactly `0` with no density groups.
pub(crate) fn correction_g<T: PkNum>(
    densities: &[EtaDensity], eta: &[T], oi: impl Fn(usize, usize) -> T, theta: impl Fn(usize) -> T,
) -> T {
    let mut s = T::from_f64(0.0);
    for d in densities {
        s = s + group_correction_g(d, eta, &oi, &theta);
    }
    s
}

/// [`correction_g`] at `f64`, with `Ω⁻¹` from its cached matrix.
pub(crate) fn correction(
    densities: &[EtaDensity], eta: &[f64], omega_inv: &DMatrix<f64>, theta: &[f64],
) -> f64 {
    correction_g(densities, eta, |a, b| omega_inv[(a, b)], |k| theta[k])
}

/// `½c(η)`: the density correction on the individual objective's `½(−2 ln p)` scale, the
/// share an importance-sampling weight's `−ln p(y | η) − ln N(η; 0, Ω)` is missing.
/// `0` without density groups, `+∞` outside a Box-Cox support. `eta` may be a stacked
/// `[η, κ…]` vector (BSV first) with `omega_inv` its block-diagonal prior precision.
pub(crate) fn half_correction(
    densities: &[EtaDensity], eta: &[f64], omega_inv: &DMatrix<f64>, theta: &[f64],
) -> f64 {
    if densities.is_empty() {
        return 0.0;
    }
    0.5 * correction(densities, eta, omega_inv, theta)
}

/// Add `½∇_η c` into `grad`: the density correction's share of the gradient of the
/// individual objective `½(data + prior)`. A no-op without density groups.
pub(crate) fn add_half_grad(
    densities: &[EtaDensity], eta: &[f64], omega_inv: &DMatrix<f64>, theta: &[f64],
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
    densities: &[EtaDensity], eta: &[f64], omega_inv: &DMatrix<f64>, theta: &[f64],
    grad: &mut [f64], hess: &mut DMatrix<f64>,
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
    model: &crate::types::CompiledModel, k_occasions: usize,
) -> std::borrow::Cow<'_, [EtaDensity]> {
    if model.kappa_densities.is_empty() || k_occasions == 0 {
        return std::borrow::Cow::Borrowed(&model.eta_densities);
    }
    let mut out = model.eta_densities.clone();
    for g in 0..k_occasions {
        for d in &model.kappa_densities {
            let mut dg = d.clone();
            dg.etas = d.etas.iter().map(|&j| model.n_eta + g * model.n_kappa + j).collect();
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
    model: &crate::types::CompiledModel, blocks: &[(Vec<usize>, usize)],
) -> Vec<EtaDensity> {
    let mut out = model.eta_densities.clone();
    for (ks, base) in blocks {
        for d in &model.kappa_densities {
            let pos: Option<Vec<usize>> =
                d.etas.iter().map(|j| ks.iter().position(|k| k == j).map(|p| base + p)).collect();
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
    model: &crate::types::CompiledModel, blocks: &[(Vec<usize>, usize)], b: &[f64],
    omega_iov_inv: &DMatrix<f64>, theta: &[f64],
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
                let etas: Option<Vec<usize>> =
                    d.etas.iter().map(|j| ks.iter().position(|k| k == j)).collect();
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
    model: &crate::types::CompiledModel, eta: &[f64], omega_inv: &DMatrix<f64>, theta: &[f64],
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
        densities: &[EtaDensity], eta: &[f64], omega_inv: &DMatrix<f64>, theta: &[f64],
    ) -> Option<Self> {
        if densities.is_empty() {
            return None;
        }
        let (c, _, h) = correction_jets(densities, eta, omega_inv, theta, true);
        Some(Self { c, half_hess: h.expect("requested") * 0.5 })
    }
}

/// `Ω⁻¹`, plus `½∇²c` under a density: the prior's curvature in `H̃`.
pub(crate) fn prior_curvature<'a>(
    omega_inv: &'a DMatrix<f64>, density: Option<&DensityAtMode>,
) -> std::borrow::Cow<'a, DMatrix<f64>> {
    match density {
        None => std::borrow::Cow::Borrowed(omega_inv),
        Some(d) => std::borrow::Cow::Owned(omega_inv + &d.half_hess),
    }
}

/// The correction's value, η-gradient and η-Hessian (full `n_η` layout, zero outside the
/// density groups) over any [`PkNum`] `T`. At `T = f64` these are plain numbers; over the
/// recording type they also carry `∂/∂θ` (ν, λ) and `∂/∂Ω`. Each entry comes from one
/// hyper-dual evaluation of [`correction_g`], seeded on a pair of a group's ETAs, so it is
/// the exact derivative of the one implementation. `want_hessian = false` skips the
/// second-order seeds.
pub(crate) fn correction_jets_g<T: PkNum>(
    densities: &[EtaDensity], eta: &[T], oi: impl Fn(usize, usize) -> T,
    theta: impl Fn(usize) -> T, want_hessian: bool,
) -> (T, Vec<T>, Option<Vec<Vec<T>>>) {
    correction_jets_impl(densities, eta, oi, theta, want_hessian, true)
}

/// [`correction_jets_g`], with `structured = false` taking every group through the
/// generic pair passes (the oracle the separable copula path is tested against).
fn correction_jets_impl<T: PkNum>(
    densities: &[EtaDensity], eta: &[T], oi: impl Fn(usize, usize) -> T,
    theta: impl Fn(usize) -> T, want_hessian: bool, structured: bool,
) -> (T, Vec<T>, Option<Vec<Vec<T>>>) {
    type H<T> = DualT<1, DualT<1, T>>;
    let n = eta.len();
    let (zero, one) = (T::from_f64(0.0), T::from_f64(1.0));
    let lift =
        |v: T| -> H<T> { DualT { v: DualT { v, g: [zero] }, g: [DualT { v: zero, g: [zero] }] } };
    let mut value = zero;
    let mut grad = vec![zero; n];
    let mut hess = want_hessian.then(|| vec![vec![zero; n]; n]);
    for d in densities {
        // A copula is separable in its margins (#2669 review, finding 11): one univariate
        // jet pass per margin and an O(p²) assembly, where the generic pairs below take
        // p(p+1)/2 passes of the whole block. `None` (a margin outside its domain) leaves it
        // to the generic path, which returns the same `+∞`.
        if structured && d.kind == DensityKind::Copula && d.trunc.is_none() {
            if let Some((v, g, h)) = copula_jets(d, eta, &oi, &theta, want_hessian) {
                value = value + v;
                for (i, &a) in d.etas.iter().enumerate() {
                    grad[a] = g[i];
                    if let (Some(hm), Some(hg)) = (hess.as_mut(), h.as_ref()) {
                        for (j, &b) in d.etas.iter().enumerate() {
                            hm[a][b] = hg[i][j];
                        }
                    }
                }
                continue;
            }
        }
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
            let pairs: &[usize] = if want_hessian { &d.etas[..=i] } else { &d.etas[i..=i] };
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

/// The value, η-gradient and η-Hessian of a Gaussian copula group's correction, from its
/// separable form (#2669 review, finding 11). With Ω constant in η,
///
/// ```text
/// c(z) = Σₖ n₂ₖ(zₖ) − Σₖ ln ωₖ − p ln 2π + qᵀAq − zᵀΩ⁻¹z,   A = R⁻¹ − I,
/// ```
///
/// where `n₂ₖ` is margin k's latent `−2 ln p` and `qₖ = Φ⁻¹(Fₖ(m(zₖ)))` its normal score,
/// each a function of `zₖ` alone. So
///
/// ```text
/// ∂c/∂z_a     = n₂ₐ′ + 2(Aq)ₐ qₐ′ − 2(Ω⁻¹z)ₐ
/// ∂²c/∂zₐ∂z_b = δₐ_b (n₂ₐ″ + 2(Aq)ₐ qₐ″) + 2Aₐ_b qₐ′ q_b′ − 2Ω⁻¹ₐ_b,
/// ```
///
/// with each margin's two derivatives from one hyper-dual pass over the same per-margin
/// code the value uses. The value is [`correction_g`]'s own. `None` where a margin leaves
/// its domain.
#[allow(clippy::type_complexity)]
fn copula_jets<T: PkNum>(
    d: &EtaDensity, eta: &[T], oi: &impl Fn(usize, usize) -> T, theta: &impl Fn(usize) -> T,
    want_hessian: bool,
) -> Option<(T, Vec<T>, Option<Vec<Vec<T>>>)> {
    type H<T> = DualT<1, DualT<1, T>>;
    let n = d.etas.len();
    let (zero, one) = (T::from_f64(0.0), T::from_f64(1.0));
    let lift =
        |v: T| -> H<T> { DualT { v: DualT { v, g: [zero] }, g: [DualT { v: zero, g: [zero] }] } };
    let value = correction_g(std::slice::from_ref(d), eta, |i, j| oi(i, j), |k| theta(k));
    if !kinds::finite(value) {
        return None;
    }
    let oi_g: Vec<Vec<T>> =
        (0..n).map(|i| (0..n).map(|j| oi(d.etas[i], d.etas[j])).collect()).collect();
    let (om, _) = spd_inverse(&oi_g)?;
    let shapes: Vec<T> = d.shape_thetas.iter().map(|&k| theta(k)).collect();
    // Per margin: (n₂, n₂′, n₂″) and (q, q′, q″) in its own latent.
    let mut n2 = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    let mut at = 0;
    for (i, &kind) in d.margins.iter().enumerate() {
        let m = kind.shapes().len();
        let sk: Vec<H<T>> = shapes[at..at + m].iter().map(|&v| lift(v)).collect();
        at += m;
        let w = om[i][i];
        let mut z = lift(eta[d.etas[i]]);
        z.v.g[0] = one;
        z.g[0].v = one;
        let wl = lift(w);
        let neg2 = if kind == DensityKind::StudentT {
            H::<T>::from_f64(-2.0) * kinds::eta_log_pdf(kind, z, wl, &sk)
        } else {
            kinds::latent_neg2_logpdf(kind, z, wl, &sk)
        };
        if !kinds::finite(neg2.v.v) {
            return None;
        }
        let qk = if kind == DensityKind::Normal {
            z * (H::<T>::from_f64(1.0) / wl.sqrt())
        } else {
            let x = match kind.map() {
                DomainMap::Identity => z,
                DomainMap::Exp => z.exp(),
                DomainMap::InvLogit => z.inv_logit(),
            };
            let u = kinds::eta_cdf(kind, x, wl, &sk)?;
            if !(u.v.v.br_cmp_c(B::Gt, 0.0) && u.v.v.br_cmp_c(B::Lt, 1.0)) {
                return None;
            }
            normal_quantile(u)
        };
        n2.push((neg2.v.g[0], neg2.g[0].g[0]));
        q.push((qk.v.v, qk.v.g[0], qk.g[0].g[0]));
    }
    // A = R⁻¹ − I, R⁻¹ᵢⱼ = Ω⁻¹ᵢⱼ √(ωᵢ ωⱼ).
    let a = |i: usize, j: usize| {
        let r = oi_g[i][j] * (om[i][i] * om[j][j]).sqrt();
        if i == j {
            r - one
        } else {
            r
        }
    };
    let two = T::from_f64(2.0);
    let aq: Vec<T> = (0..n).map(|i| (0..n).fold(zero, |s, j| s + a(i, j) * q[j].0)).collect();
    let oiz: Vec<T> =
        (0..n).map(|i| (0..n).fold(zero, |s, j| s + oi_g[i][j] * eta[d.etas[j]])).collect();
    let grad: Vec<T> = (0..n).map(|i| n2[i].0 + two * aq[i] * q[i].1 - two * oiz[i]).collect();
    let hess = want_hessian.then(|| {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let mut h = two * a(i, j) * q[i].1 * q[j].1 - two * oi_g[i][j];
                        if i == j {
                            h = h + n2[i].1 + two * aq[i] * q[i].2;
                        }
                        h
                    })
                    .collect()
            })
            .collect()
    });
    Some((value, grad, hess))
}

/// [`correction_jets_g`] at `f64`, with `Ω⁻¹` from its cached matrix.
pub(crate) fn correction_jets(
    densities: &[EtaDensity], eta: &[f64], omega_inv: &DMatrix<f64>, theta: &[f64],
    want_hessian: bool,
) -> (f64, Vec<f64>, Option<DMatrix<f64>>) {
    let (v, g, h) =
        correction_jets_g(densities, eta, |a, b| omega_inv[(a, b)], |k| theta[k], want_hessian);
    let n = eta.len();
    (v, g, h.map(|h| DMatrix::from_fn(n, n, |a, b| h[a][b])))
}

/// A latent draw of a truncated group: `η` by the inverse CDF between `F(lo)` and `F(hi)`
/// (bisection on the kind's CDF, inside a bracket that widens until it holds the target),
/// then the truncation map's inverse.
fn sample_truncated(
    d: &EtaDensity, t: Truncation, w: f64, shapes: &[f64], rng: &mut impl rand::Rng,
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

/// Turn a Gaussian simulation draw into the densities' draw (#1716), in place: `eta` holds
/// `η = L·z` from `N(0, Ω)`, and every density group's entries are replaced by a draw of
/// its latent. A t block rescales its Gaussian draw by `√(ν/W)`, `W ~ χ²_ν`, the
/// multivariate t's scale mixture. A univariate kind draws the random effect from its
/// distribution and stores the latent `m⁻¹(η)`. A Gaussian model draws nothing extra, so
/// its random stream is untouched.
pub(crate) fn redraw_latents(
    densities: &[EtaDensity], omega: &DMatrix<f64>, theta: &[f64], eta: &mut [f64],
    rng: &mut impl rand::Rng,
) {
    use rand_distr::{ChiSquared, Distribution};
    for d in densities {
        let a = d.etas[0];
        let shapes: Vec<f64> = d.shape_thetas.iter().map(|&k| theta[k]).collect();
        if d.kind == DensityKind::LogNormal {
            // Median 1: ln η ~ N(0, Ω), the Gaussian draw itself.
            let n = d.etas.len();
            let sm = DMatrix::from_fn(n, n, |i, j| omega[(d.etas[i], d.etas[j])]);
            let Some(l) = sm.cholesky().map(|c| c.l()) else {
                continue;
            };
            let zs = DVector::from_fn(n, |_, _| {
                rand_distr::Distribution::sample(&rand_distr::StandardNormal, rng)
            });
            let x = l * zs;
            for i in 0..n {
                eta[d.etas[i]] = x[i];
            }
            continue;
        }
        if d.kind == DensityKind::LogNormalMean && d.etas.len() > 1 {
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
            // q ~ N(0, R) from the Gaussian draw (η = L·z already correlated by Ω, so
            // q = D^{-½}η), u = Φ(q), each margin by its inverse CDF.
            let mut at = 0;
            for (i, &kind) in d.margins.iter().enumerate() {
                let m = kind.shapes().len();
                let sk = &shapes[at..at + m];
                at += m;
                let a = d.etas[i];
                let w = omega[(a, a)];
                let u =
                    crate::stats::special::normal_cdf(eta[a] / w.sqrt()).clamp(1e-300, 1.0 - 1e-16);
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

/// The first kind with a kink or a positive-density edge (`DensityKind::has_kink_or_edge`)
/// among a model's density groups, BSV and κ, copula margins included, with the name of the
/// random effect that follows it; `None` when every density is smooth (#2669 review,
/// finding 1). The FOCE/FOCEI/Laplace objective, and the covariance step built on it, read a
/// prior curvature such a kind does not have.
pub(crate) fn kink_kind(model: &crate::types::CompiledModel) -> Option<(DensityKind, String)> {
    let bsv = model.eta_densities.iter().map(|d| (d, &model.eta_names));
    let iov = model.kappa_densities.iter().map(|d| (d, &model.kappa_names));
    for (d, names) in bsv.chain(iov) {
        for (i, &e) in d.etas.iter().enumerate() {
            let k = if d.kind == DensityKind::Copula { d.margins[i] } else { d.kind };
            if k.has_kink_or_edge() {
                return Some((k, names.get(e).cloned().unwrap_or_default()));
            }
        }
    }
    None
}

/// Which summary of its individual parameter each typical-value θ is (#2669), from the
/// model's mu-references and its density groups' log anchors:
///
/// - a Gaussian (or shape-transformed) ETA under `TV·exp(η)` or `inv_logit(logit(TV) + η)`:
///   the **median** (a monotone map of a symmetric η with `h(0) = 0`); under `TV + η` the
///   **mean**;
/// - a positive `density` read as `TV·η` whose effect has mean 1: the **mean**;
/// - a `density` on ℝ read as `TV·exp(η)`: the **median** for a symmetric kind, the
///   **geometric mean** for an asymmetric one (`gumbel`, `gev`, `nig`);
/// - the μ of `beta(μ)` (mean) and `logitnormal(μ)` (median), for the parameter reading it.
///
/// A θ that is the typical value of two parameters, or of a truncated or another kind,
/// gets no entry.
pub(crate) fn typical_values(
    model: &crate::types::CompiledModel,
) -> Vec<crate::types::TypicalValue> {
    use crate::types::{MuTransform, TypicalStatistic as S, TypicalValue};
    let param_of = |eta: &str| {
        model
            .eta_param_info
            .iter()
            .find(|i| i.eta_name == eta)
            .map(|i| i.individual_param_name.clone())
    };
    let mut out: Vec<TypicalValue> = Vec::new();
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut push = |theta: String, parameter: Option<String>, statistic: S| {
        let Some(parameter) = parameter else { return };
        *seen.entry(theta.clone()).or_default() += 1;
        out.push(TypicalValue { theta, parameter, statistic });
    };
    let mut refs: Vec<(&String, &crate::types::MuRef)> = model.mu_refs.iter().collect();
    refs.sort_by(|a, b| a.0.cmp(b.0));
    for (eta, mr) in refs {
        let st = match mr.transform {
            MuTransform::Identity => S::Mean,
            MuTransform::Log | MuTransform::LogitProbability => S::Median,
            // θ is the median on the logit scale, not a value of the parameter.
            _ => continue,
        };
        push(mr.theta_name.clone(), param_of(eta), st);
    }
    for d in &model.eta_densities {
        for (pos, &e) in d.etas.iter().enumerate() {
            let kind = if d.kind == DensityKind::Copula { d.margins[pos] } else { d.kind };
            let eta = &model.eta_names[e];
            if d.trunc.is_none() && matches!(kind, DensityKind::Beta | DensityKind::LogitNormal) {
                if let Some(&t) = d.shape_thetas.first().filter(|_| d.kind != DensityKind::Copula) {
                    let st = if kind == DensityKind::Beta { S::Mean } else { S::Median };
                    push(model.theta_names[t].clone(), param_of(eta), st);
                }
                continue;
            }
            let Some(Some(t)) = d.log_anchors.get(pos) else { continue };
            if d.trunc.is_some() {
                continue;
            }
            let st = match kind.map() {
                DomainMap::Exp => {
                    let w = 1.0 / model.default_params.omega.inv[(e, e)].max(f64::MIN_POSITIVE);
                    let shapes: Vec<f64> =
                        d.shape_thetas.iter().map(|&k| model.default_params.theta[k]).collect();
                    match kinds::effect_moments(kind, w, &shapes) {
                        Some((m, _)) if (m - 1.0).abs() < 1e-12 => S::Mean,
                        _ => continue,
                    }
                }
                DomainMap::Identity => match kind {
                    DensityKind::Gumbel
                    | DensityKind::GeneralizedExtremeValue
                    | DensityKind::NormalInverseGaussian => S::GeometricMean,
                    _ => S::Median,
                },
                DomainMap::InvLogit => continue,
            };
            push(model.theta_names[*t].clone(), param_of(eta), st);
        }
    }
    out.retain(|v| seen.get(&v.theta) == Some(&1));
    out
}

/// What a `density` random effect is not carried beside (#1716), as `fit()`'s refusal: a
/// kink or edge kind (`DensityKind::has_kink_or_edge`) under a FOCE-family or Laplace
/// method, and a κ density under nested IOV levels (which share κs across cells). `None`
/// when the fit can run.
pub(crate) fn fit_refusal(
    model: &crate::types::CompiledModel, options: &crate::types::FitOptions,
    _params: &crate::types::ModelParameters,
) -> Option<String> {
    if model.eta_densities.is_empty() && model.kappa_densities.is_empty() {
        return None;
    }
    // A kink or an edge under a method whose reported objective reads the prior's curvature
    // at the EBE: the FOCE family, Laplace, VI's Laplace objective, and SAEM under
    // `saem_ofv = focei`. IMP, SAEM's other objectives and Bayes score the prior itself.
    use crate::types::EstimationMethod as M;
    if let Some((k, eta)) = kink_kind(model) {
        let curvature = options.method_chain().into_iter().find_map(|m| {
            let label = match m {
                // A pure adaptive-quadrature evaluation integrates the prior itself (its
                // Hessian only places the nodes); Laplace at one node is the curvature.
                M::Laplace if options.agq_eval_only && options.n_agq > 1 => return None,
                M::Foce | M::FoceI | M::FoceGn | M::FoceGnHybrid | M::Laplace => {
                    m.label().to_string()
                }
                M::Vi if options.vi_final_ofv == crate::types::ViFinalOfv::Laplace => {
                    "VI's `vi_final_ofv = laplace`".to_string()
                }
                M::Saem if options.saem_ofv == crate::types::SaemOfv::Focei => {
                    "SAEM's `saem_ofv = focei`".to_string()
                }
                _ => return None,
            };
            Some(label)
        });
        if let Some(m) = curvature {
            return Some(format!(
                "`{eta}` follows `density {}`, which has a kink or an edge where its density \
                 is positive, so {m} would read a prior curvature that does not exist there \
                 and report an objective that may be wrong. Fit it with `method = imp` (its \
                 OFV is then the importance-sampling −2 log L), `saem` (with `saem_ofv = saem`, \
                 `imp` or `agq`) or `bayes`",
                k.name()
            ));
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
