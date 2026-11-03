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
//! | kind | support of η | parameters | latent density |
//! |---|---|---|---|
//! | `t(ν)` | ℝ (block: ℝⁿ) | scale matrix Ω, ν | multivariate t |
//! | `logistic` | ℝ | variance ω | logistic, scale `√(3ω)/π` |
//! | `gamma` | (0, ∞) | mean 1, variance ω | `k z − e^z/ω − lnΓ(k) − k ln ω`, `k = 1/ω` |
//! | `invgamma` | (0, ∞) | mean 1, variance ω | `α ln β − lnΓ(α) − α z − β e^{−z}`, `α = 2 + 1/ω`, `β = α − 1` |
//! | `lognormal` | (0, ∞) | mean 1, variance ω | `z ~ N(−s²/2, s²)`, `s² = ln(1 + ω)` |
//! | `beta(μ)` | (0, 1) | mean μ, variance ω | `a ln σ(z) + b ln σ(−z) − ln B(a, b)`, `a = μn`, `b = (1−μ)n`, `n = μ(1−μ)/ω − 1` |
//! | `logitnormal(μ)` | (0, 1) | location `logit μ`, variance ω | `z ~ N(logit μ, ω)` |
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

/// Which distribution a group of ETAs follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityKind {
    /// Student-t (multivariate over a block), one ν.
    StudentT,
    /// Logistic.
    Logistic,
    /// Gamma, mean 1.
    Gamma,
    /// Inverse gamma, mean 1.
    InvGamma,
    /// Log-normal, mean 1.
    LogNormal,
    /// Beta, mean μ.
    Beta,
    /// Logit-normal, location `logit μ`.
    LogitNormal,
}

/// How the model reads a latent `z`: `η = m(z)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DomainMap {
    /// `η = z`, on ℝ.
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
}

impl DensityKind {
    /// Every kind, for name lookup.
    pub(crate) const ALL: [DensityKind; 7] = [
        DensityKind::StudentT,
        DensityKind::Logistic,
        DensityKind::Gamma,
        DensityKind::InvGamma,
        DensityKind::LogNormal,
        DensityKind::Beta,
        DensityKind::LogitNormal,
    ];

    /// The model-file name after `density`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            DensityKind::StudentT => "t",
            DensityKind::Logistic => "logistic",
            DensityKind::Gamma => "gamma",
            DensityKind::InvGamma => "invgamma",
            DensityKind::LogNormal => "lognormal",
            DensityKind::Beta => "beta",
            DensityKind::LogitNormal => "logitnormal",
        }
    }

    /// The kind a (lower-cased) name spells, if any.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// The domain map the model reads the latent through.
    pub(crate) fn map(self) -> DomainMap {
        match self {
            DensityKind::StudentT | DensityKind::Logistic => DomainMap::Identity,
            DensityKind::Gamma | DensityKind::InvGamma | DensityKind::LogNormal => DomainMap::Exp,
            DensityKind::Beta | DensityKind::LogitNormal => DomainMap::InvLogit,
        }
    }

    /// Whether a block of several ETAs can follow it (only the multivariate t).
    pub(crate) fn allows_block(self) -> bool {
        self == DensityKind::StudentT
    }

    /// The shape θ it takes, if any: `(prefix, init, lower, upper)` of an auto-declared
    /// one (`NU_<ETA>`, `MU_<ETA>`). ν starts at the t-transform's Pharmpy default; μ at ½.
    pub(crate) fn shape_theta(self) -> Option<(&'static str, f64, f64, f64)> {
        match self {
            DensityKind::StudentT => Some(("NU", 80.0, 3.0, 100.0)),
            DensityKind::Beta | DensityKind::LogitNormal => Some(("MU", 0.5, 0.001, 0.999)),
            _ => None,
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
    /// θ indices of the shape: `[ν]` for t, `[μ]` for beta and logit-normal, empty
    /// otherwise.
    pub(crate) shape_thetas: Vec<usize>,
}

/// `ln(1 + x)` over `T`: the value from libm's `ln_1p` (no cancellation), the jets from
/// `ln(1 + x)`, which are `ln_1p`'s. The `l − l.val()` term has value 0 and carries them.
fn ln1p_g<T: PkNum>(x: T) -> T {
    let l = (T::from_f64(1.0) + x).ln();
    (l - T::from_f64(l.val())) + T::from_f64(x.val().ln_1p())
}

/// `ln(1 + eˣ)`, without overflow: `x + ln(1 + e⁻ˣ)` for `x > 0`.
fn softplus_g<T: PkNum>(x: T) -> T {
    if x.val() > 0.0 {
        x + ln1p_g((-x).exp())
    } else {
        ln1p_g(x.exp())
    }
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
    let gauss = q + w.ln() + ln_2pi;
    // `−2 ln p_z(z)` of the kind.
    let neg2 = match d.kind {
        DensityKind::StudentT => unreachable!("handled above"),
        DensityKind::Logistic => {
            // s = √(3ω)/π; ln p = −u − 2 ln(1 + e^{−u}) − ln s, u = z/s.
            let s = (c(3.0) * w).sqrt() * c(1.0 / std::f64::consts::PI);
            let u = z * (c(1.0) / s);
            c(2.0) * u + c(4.0) * softplus_g(-u) + c(2.0) * s.ln()
        }
        DensityKind::Gamma => {
            let k = c(1.0) / w;
            c(-2.0) * (k * z - z.exp() * k - k.ln_gamma() - k * w.ln())
        }
        DensityKind::InvGamma => {
            let alpha = c(2.0) + c(1.0) / w;
            let beta = alpha - c(1.0);
            c(-2.0) * (alpha * beta.ln() - alpha.ln_gamma() - alpha * z - beta * (-z).exp())
        }
        DensityKind::LogNormal => {
            let s2 = ln1p_g(w);
            let r = z + s2 * c(0.5);
            r * r * (c(1.0) / s2) + s2.ln() + ln_2pi
        }
        DensityKind::Beta => {
            let mu = theta(d.shape_thetas[0]);
            let n = mu * (c(1.0) - mu) * (c(1.0) / w) - c(1.0);
            if !(n.val() > 0.0) {
                return c(f64::INFINITY);
            }
            let (pa, pb) = (mu * n, (c(1.0) - mu) * n);
            let ln_beta = pa.ln_gamma() + pb.ln_gamma() - n.ln_gamma();
            // ln σ(z) = −softplus(−z), ln σ(−z) = −softplus(z).
            c(2.0) * (pa * softplus_g(-z) + pb * softplus_g(z) + ln_beta)
        }
        DensityKind::LogitNormal => {
            let mu = theta(d.shape_thetas[0]);
            let m = mu.ln() - (c(1.0) - mu).ln();
            let r = z - m;
            r * r * (c(1.0) / w) + w.ln() + ln_2pi
        }
    };
    neg2 - gauss
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
pub(crate) fn redraw_latents(
    densities: &[EtaDensity],
    omega: &DMatrix<f64>,
    theta: &[f64],
    eta: &mut [f64],
    rng: &mut impl rand::Rng,
) {
    use rand_distr::{Beta, ChiSquared, Distribution, Gamma, StandardNormal};
    for d in densities {
        let a = d.etas[0];
        let w = omega[(a, a)];
        let shape = d.shape_thetas.first().map(|&k| theta[k]);
        match d.kind {
            DensityKind::StudentT => {
                let nu = shape.expect("t has ν");
                let scale = match ChiSquared::new(nu) {
                    Ok(chi) => (nu / chi.sample(rng).max(1e-300)).sqrt(),
                    Err(_) => f64::NAN,
                };
                for &k in &d.etas {
                    eta[k] *= scale;
                }
            }
            DensityKind::Logistic => {
                let s = (3.0 * w).sqrt() / std::f64::consts::PI;
                let u: f64 = rand_distr::Uniform::new(0.0f64, 1.0)
                    .map_or(0.5, |d| d.sample(rng))
                    .max(f64::MIN_POSITIVE);
                eta[a] = s * (u / (1.0 - u)).ln();
            }
            DensityKind::Gamma => {
                eta[a] = Gamma::new(1.0 / w, w).map_or(f64::NAN, |g| g.sample(rng).ln());
            }
            DensityKind::InvGamma => {
                let alpha = 2.0 + 1.0 / w;
                // η = 1/G, G ~ Gamma(α, rate β = α − 1), so z = −ln G.
                eta[a] = Gamma::new(alpha, 1.0 / (alpha - 1.0))
                    .map_or(f64::NAN, |g| -g.sample(rng).ln());
            }
            DensityKind::LogNormal => {
                let s2 = w.ln_1p();
                let n: f64 = StandardNormal.sample(rng);
                eta[a] = -0.5 * s2 + s2.sqrt() * n;
            }
            DensityKind::Beta => {
                let mu = shape.expect("beta has μ");
                let n = mu * (1.0 - mu) / w - 1.0;
                eta[a] = Beta::new(mu * n, (1.0 - mu) * n).map_or(f64::NAN, |b| {
                    let e: f64 = b.sample(rng);
                    (e / (1.0 - e)).ln()
                });
            }
            DensityKind::LogitNormal => {
                let mu = shape.expect("logit-normal has μ");
                let n: f64 = StandardNormal.sample(rng);
                eta[a] = (mu / (1.0 - mu)).ln() + w.sqrt() * n;
            }
        }
    }
}

/// What a `density` random effect is not carried beside yet (#1716), as `fit()`'s refusal:
/// a `[mixture]` (each class draws its own Ω), FREM (its covariate ETAs are Gaussian
/// pseudo-observations), a multivariate t block under Bayes (its ω move is univariate),
/// VI's Adam Ω step (the Gaussian prior's gradient), and FOCE without interaction on a
/// correlated residual (the conditional form FOCE takes under a density is the diagonal
/// one). `None` when the fit can run.
pub(crate) fn fit_refusal(
    model: &crate::types::CompiledModel,
    options: &crate::types::FitOptions,
    params: &crate::types::ModelParameters,
) -> Option<String> {
    use crate::types::EstimationMethod as M;
    if model.eta_densities.is_empty() {
        return None;
    }
    if model.mixture.is_some() {
        return Some("a `density` random effect cannot be combined with a [mixture] yet".into());
    }
    if model.frem_config.is_some() {
        return Some("a `density` random effect cannot be combined with FREM".into());
    }
    let methods = std::iter::once(options.method).chain(options.methods.iter().copied());
    for m in methods {
        if m == M::Vi && options.vi_omega_update != crate::types::ViOmegaUpdate::ClosedForm {
            return Some(
                "`vi_omega_update = adam` steps Ω by the Gaussian prior's gradient; a `density` \
                 random effect needs the default `vi_omega_update = closed_form`, whose density \
                 groups take their own M-step"
                    .into(),
            );
        }
        if m == M::Bayes && model.eta_densities.iter().any(|d| d.etas.len() > 1) {
            return Some(
                "`method = bayes` samples a univariate `density`'s ω, not a multivariate t \
                 block's: declare each ETA on its own `omega` line"
                    .into(),
            );
        }
        let foce = matches!(m, M::Foce | M::FoceGn | M::FoceGnHybrid)
            && !crate::types::interaction_for(m, options.interaction);
        if foce && model.residual_cov_at(params).is_dense() {
            return Some(
                "FOCE without interaction scores a `density` random effect in its conditional \
                 form, which is diagonal: a correlated residual (`block_sigma`, `correlation`) \
                 needs `method = focei`"
                    .into(),
            );
        }
    }
    None
}

#[cfg(test)]
#[path = "eta_density_tests.rs"]
mod tests;
