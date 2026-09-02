//! ELBO value and gradient assembly.
//!
//! Implements the analytic-KL form described in [`super`]:
//!
//! ```text
//! −ELBO = Σᵢ [ E_{η∼qᵢ}( −log p(yᵢ | η, θ, Σ) )  +  KL( qᵢ ‖ N(0, Ω) ) ]
//!              └──── Monte Carlo, `n_mc_samples` draws ────┘  └─ closed form ─┘
//! ```
//!
//! Under `vi_kl = mc` the second term is estimated from the same draws instead, as
//! `E_q[log q(η) − log p(η|Ω)]` — see `mc_kl_draw`, and note that its `φ` gradient
//! is then a path derivative rather than the derivative of the reported value.
//!
//! Everything here works with **`−ELBO`**, so the quantity is a thing to
//! *minimize*, consistent with every other objective in the crate, and can be
//! handed straight to [`super::adam::AdamState::step`].
//!
//! # The additive constant
//!
//! The data term is `obs_nll_subject_grad`'s NLL, which — like `individual_nll`,
//! FOCE's objective, and NONMEM's OFV — omits the `½·n_obs·log 2π` constant. The
//! KL term is exact. So `2·(−ELBO)` is on the same footing as the OFVs reported
//! elsewhere in `ferx-core`: comparable *between VI fits of the same data*, and
//! not a `−2 log L`. See the plan's §4.3 and `vi_final_ofv` for getting a real one.
//!
//! # Common random numbers
//!
//! `ε` is drawn from a seed derived deterministically from
//! `(seed, iteration, subject, sample)` — never from a shared mutable RNG. Two
//! consequences, both wanted: the fit is reproducible regardless of thread
//! scheduling, and at a *fixed* iteration index `−ELBO` is a deterministic
//! function of `(x, φ)`, which is what makes the finite-difference parity tests
//! exact rather than noisy.

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::estimation::fixed_eta_gradient::{
    obs_nll_subject_grad, obs_nll_subject_grad_iov, obs_nll_subject_into_iov,
};
use crate::estimation::inner_optimizer::{
    analytic_eta_nll_gradient, analytic_eta_nll_gradient_iov,
};
use crate::estimation::parameterization::{lower_tri_iter, theta_packs_log, unpack_params};
use crate::pk::EventPkParams;
use crate::stats::likelihood::{iov_occasion_groups, obs_nll_subject_into};
use crate::types::{
    BloqMethod, CompiledModel, ErrorModel, ErrorSpec, ModelParameters, OmegaMatrix, Population,
    Subject,
};

use super::family::VariationalFamily;

/// How `∂/∂η` of the data term is obtained. Re-exported from
/// [`crate::types::ViEtaGrad`], which is where the user-facing option lives.
pub use crate::types::ViEtaGrad as EtaGradMode;

/// How the KL half is evaluated. Re-exported from [`crate::types::ViKl`].
pub use crate::types::ViKl as KlMode;

/// Everything the ELBO evaluation needs that does not change between iterations.
#[derive(Debug, Clone)]
pub struct ElboConfig {
    /// Monte-Carlo draws per subject per iteration. Janssen et al. use 3; their
    /// supplementary Table 2 reports 1 loses no accuracy at ~3× the throughput.
    pub n_mc_samples: usize,
    /// How to differentiate the data term with respect to `η`.
    pub eta_grad: EtaGradMode,
    /// How to evaluate the KL half.
    pub kl: KlMode,
    /// Base seed for the common random numbers.
    pub seed: u64,
}

impl Default for ElboConfig {
    fn default() -> Self {
        Self {
            n_mc_samples: 3,
            eta_grad: EtaGradMode::default(),
            kl: KlMode::default(),
            seed: 0,
        }
    }
}

/// Offsets of the blocks of the packed parameter vector produced by
/// [`crate::estimation::parameterization::pack_params`]:
/// `[θ | chol(Ω) | σ | chol(Ω_iov) | ρ]`.
///
/// The `Ω_iov` block is present exactly when the model declares `kappa`
/// parameters; `n_omega_iov` is zero otherwise, so every offset below collapses to the
/// three-block layout without special-casing.
///
/// The trailing `ρ` block holds the `block_sigma` off-diagonals (#847). VI does not
/// estimate them — the ELBO's σ machinery has no ρ channel — so the block exists here
/// only to keep [`Self::total`] equal to `pack_params().len()`, which is what lets VI's
/// `x` round-trip through `unpack_params`. Its gradient entries stay zero, so Adam
/// leaves ρ at the declared value.
#[derive(Debug, Clone, Copy)]
pub struct PackedLayout {
    pub n_theta: usize,
    pub n_omega: usize,
    pub n_sigma: usize,
    pub n_omega_iov: usize,
    pub n_rho: usize,
}

impl PackedLayout {
    pub fn new(template: &ModelParameters) -> Self {
        let n_eta = template.omega.dim();
        Self {
            n_theta: template.theta.len(),
            n_omega: lower_tri_iter(n_eta, template.omega.diagonal).count(),
            n_sigma: template.sigma.values.len(),
            n_omega_iov: template
                .omega_iov
                .as_ref()
                .map(|iov| lower_tri_iter(iov.dim(), iov.diagonal).count())
                .unwrap_or(0),
            n_rho: template.residual_correlations.len(),
        }
    }

    pub fn total(&self) -> usize {
        self.n_theta + self.n_omega + self.n_sigma + self.n_omega_iov + self.n_rho
    }

    /// Index of the first `Ω` Cholesky coordinate.
    pub fn omega_start(&self) -> usize {
        self.n_theta
    }

    /// Index of the first `σ` coordinate.
    pub fn sigma_start(&self) -> usize {
        self.n_theta + self.n_omega
    }

    /// Index of the first `Ω_iov` Cholesky coordinate. Equal to
    /// [`Self::omega_iov_end`] when the model has no IOV, so the empty range
    /// `omega_iov_start()..omega_iov_end()` is a no-op.
    pub fn omega_iov_start(&self) -> usize {
        self.n_theta + self.n_omega + self.n_sigma
    }

    /// One past the last `Ω_iov` coordinate — **not** [`Self::total`], which now
    /// also counts the trailing ρ block (#847). Every `Ω_iov` range must end here,
    /// or a `block_sigma` model's ρ coordinate gets treated as an IOV Cholesky
    /// entry.
    pub fn omega_iov_end(&self) -> usize {
        self.omega_iov_start() + self.n_omega_iov
    }
}

/// Which variational family serves each subject.
///
/// Without IOV every subject shares one family, and [`Families::Uniform`] borrows it
/// rather than allocating N identical copies. Under IOV a subject's stacked vector is
/// `[η, κ₁ … κ_{K_i}]`, so its dimension depends on how many occasions that subject
/// has — a property of the data, not of the model — and the family becomes per-subject.
///
/// Modelled as an enum rather than always taking a slice so the common case stays free
/// and reads as what it is: one family, every subject.
#[derive(Clone, Copy)]
pub enum Families<'a> {
    /// One family for every subject.
    Uniform(&'a dyn VariationalFamily),
    /// One family per subject, indexed in `population.subjects` order.
    PerSubject(&'a [Box<dyn VariationalFamily>]),
}

impl<'a> Families<'a> {
    /// The family serving subject `i`.
    #[inline]
    pub fn for_subject(&self, i: usize) -> &'a dyn VariationalFamily {
        match self {
            Families::Uniform(f) => *f,
            Families::PerSubject(v) => v[i].as_ref(),
        }
    }
}

impl<'a> From<&'a dyn VariationalFamily> for Families<'a> {
    fn from(f: &'a dyn VariationalFamily) -> Self {
        Families::Uniform(f)
    }
}

/// The prior over one subject's **stacked** random-effects vector
/// `z = [η, κ₁ … κ_K]`: the block diagonal `Σ_b = Ω ⊕ Ω_iov^{⊗K}`.
///
/// This is what lets IOV reuse the variational families untouched. A
/// [`VariationalFamily`] never assumes its dimension means "η" — `kl_to_normal` reads
/// only `omega.inv` / `omega.log_det`, `init` only `omega.chol` — so constructing the
/// family at `d = n_eta + K·n_kappa` and handing it this prior makes
/// `KL(q ‖ N(0, Σ_b))` the same formula it already computes.
///
/// Returns `Ω` unchanged when the model has no IOV (or a subject has no occasions), so
/// the non-IOV path allocates nothing and stays bit-identical.
///
/// The `free_mask` is the direct sum of the two blocks' masks: cross-block entries are
/// structural zeros — `η` and `κ` are independent by construction, as are two distinct
/// occasions — and must never pick up a covariance the model does not declare.
pub fn stacked_prior(
    omega: &OmegaMatrix,
    omega_iov: Option<&OmegaMatrix>,
    k_occasions: usize,
) -> OmegaMatrix {
    let Some(iov) = omega_iov else {
        return omega.clone();
    };
    if k_occasions == 0 || iov.dim() == 0 {
        return omega.clone();
    }

    let n_eta = omega.dim();
    let n_kappa = iov.dim();
    let d = n_eta + k_occasions * n_kappa;

    let mut m = DMatrix::zeros(d, d);
    let mut mask = DMatrix::from_element(d, d, false);
    for i in 0..n_eta {
        for j in 0..n_eta {
            m[(i, j)] = omega.matrix[(i, j)];
            mask[(i, j)] = omega.free_mask[(i, j)];
        }
    }
    for g in 0..k_occasions {
        let off = n_eta + g * n_kappa;
        for i in 0..n_kappa {
            for j in 0..n_kappa {
                m[(off + i, off + j)] = iov.matrix[(i, j)];
                mask[(off + i, off + j)] = iov.free_mask[(i, j)];
            }
        }
    }

    // Names are cosmetic here (nothing indexes the stacked prior by name), but making
    // each occasion's block distinct keeps a debug dump readable.
    let mut names = omega.eta_names.clone();
    for g in 0..k_occasions {
        for k in 0..n_kappa {
            let base = iov
                .eta_names
                .get(k)
                .cloned()
                .unwrap_or_else(|| format!("KAPPA{k}"));
            names.push(format!("{base}@{}", g + 1));
        }
    }

    // Diagonal only when *both* blocks are: a block_omega on either side leaves the
    // stacked matrix non-diagonal, and `lower_tri_iter` must then walk the full triangle.
    let diagonal = omega.diagonal && iov.diagonal;
    OmegaMatrix::from_matrix_with_mask(m, names, diagonal, mask)
}

// ---------------------------------------------------------------------------
// Support scope
// ---------------------------------------------------------------------------

/// Why VI cannot fit this model at all, or `None` when it can.
///
/// This is about the **objective**, not the gradient. A gradient the analytic
/// provider declines can fall back to finite differences and still be right
/// (see [`EtaGradMode`]); a data term that omits part of the likelihood cannot.
/// The two failure modes therefore get different treatment, and conflating them
/// is exactly how a scope gap turns into silently wrong estimates.
pub fn unsupported_data_term_reason(model: &CompiledModel) -> Option<String> {
    // The data term is `obs_nll_subject_grad`'s NLL, and
    // `obs_nll_subject_from_preds` *skips* rows belonging to a non-Gaussian
    // endpoint (#905) — they are scored through `obs_records` instead. Fitting
    // such a model with VI would silently drop those rows from the likelihood, so
    // refuse rather than fall back.
    #[cfg(feature = "survival")]
    if !model.endpoints.is_empty() {
        return Some(
            "VI does not support non-Gaussian endpoints (TTE / categorical): its data term \
             would omit their likelihood contribution. Use method = laplace (with n_agq) \
             or saem for these models."
                .to_string(),
        );
    }
    None
}

/// Whether the analytic `Dual2` `∂/∂η` provider can serve this subject.
///
/// Implemented by **probing** the provider rather than re-deriving its scope.
/// The provider's own preconditions (ODE models, missing analytical PK path,
/// LTBS + `ExpressionScale`, IIV on residual error, some time-varying-covariate
/// configurations) are documented but numerous, and a hand-maintained copy of
/// them would drift silently the moment the provider's scope changed. A probe
/// cannot drift.
pub fn analytic_eta_grad_available(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma: &[f64],
) -> bool {
    let eta = vec![0.0; model.n_eta];
    analytic_eta_nll_gradient(model, subject, theta, &eta, omega, sigma).is_some()
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// One evaluation of `−ELBO` and its gradients.
#[derive(Debug, Clone)]
pub struct ElboEval {
    /// `−ELBO`, the quantity being minimized.
    pub neg_elbo: f64,
    /// The `Σᵢ E_q[−log p(yᵢ|η)]` half, for reporting.
    pub data_term: f64,
    /// The `Σᵢ KL(qᵢ ‖ N(0,Ω))` half, for reporting.
    pub kl_term: f64,
    /// `∂(−ELBO)/∂x` in packed space. The `Ω` block is always filled; a caller
    /// using the closed-form `Ω` update simply ignores it.
    pub grad_x: Vec<f64>,
    /// `∂(−ELBO)/∂φᵢ`, one entry per subject.
    ///
    /// Under `vi_kl = mc` this is the **path-derivative** gradient, which is not the
    /// finite difference of [`Self::neg_elbo`] — see `mc_kl_draw`.
    pub grad_phi: Vec<Vec<f64>>,
    /// Subjects whose `∂/∂η` came from finite differences this evaluation.
    pub n_fd_subjects: usize,
    /// Subjects whose KL was sampled *despite* `vi_kl = analytic`, because the
    /// variational family has no closed form. Always 0 for the families shipped
    /// today; non-zero only for a family added later that lacks one.
    pub n_kl_fallback_subjects: usize,
}

/// Per-subject partial results, folded by the caller.
struct SubjectTerms {
    data_nll: f64,
    kl: f64,
    /// `[log θ | log σ]`, the layout `obs_nll_subject_grad` returns.
    grad_theta_sigma: Vec<f64>,
    grad_phi: Vec<f64>,
    /// `∂KL/∂Ω` as a dense symmetric matrix.
    d_omega: DMatrix<f64>,
    used_fd: bool,
    /// `vi_kl = analytic` was asked for but this family had no closed form, so the
    /// KL was sampled instead.
    kl_fell_back: bool,
}

/// Deterministic `ε` for one `(iteration, subject, sample)` triple.
///
/// Seeding per draw rather than threading one RNG through the population is what
/// makes the result independent of iteration order and thread count. The mixing
/// below is SplitMix64's finalizer, which decorrelates the sequential inputs well
/// enough that adjacent subjects do not get visibly related draws.
fn crn_eps(seed: u64, iter: u64, subject: usize, sample: usize, d: usize) -> Vec<f64> {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(iter.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((subject as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add((sample as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;

    let mut rng = StdRng::seed_from_u64(z);
    (0..d).map(|_| StandardNormal.sample(&mut rng)).collect()
}

/// `∂(−log p(y|η))/∂η` by central finite differences of the same data term the
/// value uses.
#[allow(clippy::too_many_arguments)]
fn fd_eta_data_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    z: &[f64],
    _kappas: &[Vec<f64>],
    k_occasions: usize,
    scratch: &mut EventPkParams,
) -> Vec<f64> {
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let iov = k_occasions > 0 && n_kappa > 0;
    // One evaluator, differenced over every stacked coordinate. Splitting inside the
    // closure rather than at the call site keeps the perturbed `κ` consistent with the
    // perturbed `η` for the same draw.
    let eval = |zz: &[f64], scratch: &mut EventPkParams| -> f64 {
        if iov {
            let (eta, kappas) = split_stacked(zz, n_eta, n_kappa, k_occasions);
            obs_nll_subject_into_iov(model, subject, theta, sigma, eta, &kappas, scratch)
        } else {
            obs_nll_subject_into(model, subject, theta, sigma, zz, scratch)
        }
    };
    let mut g = vec![0.0; z.len()];
    for k in 0..z.len() {
        let h = 1e-5 * (1.0 + z[k].abs());
        let mut ep = z.to_vec();
        let mut em = z.to_vec();
        ep[k] += h;
        em[k] -= h;
        let fp = eval(&ep, scratch);
        let fm = eval(&em, scratch);
        g[k] = (fp - fm) / (2.0 * h);
    }
    g
}

/// `∂(−log p(y|η))/∂η`, analytic where available.
///
/// [`analytic_eta_nll_gradient`] returns the gradient of the **joint** inner NLL
/// `½(ηᵀΩ⁻¹η + log|Ω|) + data`, so the prior part `Ω⁻¹η` is subtracted off: under
/// the analytic KL that term is accounted for exactly, and counting it twice
/// would pull every `μᵢ` toward zero.
#[allow(clippy::too_many_arguments)]
fn eta_data_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    params: &ModelParameters,
    stacked_prior: &OmegaMatrix,
    sigma: &[f64],
    z: &[f64],
    kappas: &[Vec<f64>],
    k_occasions: usize,
    mode: EtaGradMode,
    scratch: &mut EventPkParams,
) -> (Vec<f64>, bool) {
    let n_eta = params.omega.dim();
    let iov = k_occasions > 0 && model.n_kappa > 0;
    if mode != EtaGradMode::Fd {
        // Both providers return the gradient of the **joint** inner NLL, i.e. data plus
        // the `½ zᵀ Σ_b⁻¹ z` prior. Under the analytic KL that prior is accounted for
        // exactly, so subtract it here or every `μ` is pulled toward zero twice. The
        // subtraction uses the *stacked* prior, which is `Ω` itself without IOV.
        let joint = if iov {
            analytic_eta_nll_gradient_iov(
                model,
                subject,
                theta,
                z,
                &params.omega,
                params
                    .omega_iov
                    .as_ref()
                    .expect("an IOV subject implies an Omega_iov"),
                sigma,
                n_eta,
                model.n_kappa,
                k_occasions,
                model.ruv_obs_mult(subject, theta).as_deref(),
            )
        } else {
            analytic_eta_nll_gradient(model, subject, theta, z, &params.omega, sigma)
        };
        if let Some(joint) = joint {
            let z_v = DVector::from_column_slice(z);
            let prior = &stacked_prior.inv * &z_v;
            let g = joint.iter().zip(prior.iter()).map(|(j, p)| j - p).collect();
            return (g, false);
        }
    }
    // `true` whenever the finite-difference route was taken, whether it was asked
    // for or fallen back to — `n_fd_subjects` reports what actually happened.
    // In `Analytic` mode FD is never *requested*, so a `true` there can only mean
    // the provider declined, which is what the caller turns into an error.
    (
        fd_eta_data_grad(
            model,
            subject,
            theta,
            sigma,
            z,
            kappas,
            k_occasions,
            scratch,
        ),
        true,
    )
}

// ---------------------------------------------------------------------------
// Monte-Carlo KL
// ---------------------------------------------------------------------------

/// One draw's contribution to the Monte-Carlo KL estimate, `log q_φ(η) − log p(η|Ω)`,
/// plus the derivatives the caller folds in.
struct McKlDraw {
    /// The integrand itself. Its expectation over `q` is exactly
    /// `KL(q ‖ N(0, Ω))` — both densities are fully normalized, so their
    /// `−(d/2)·log 2π` terms cancel and the estimate is on the same additive
    /// footing as the closed form. Individual draws can be negative; only the
    /// mean is a divergence.
    integrand: f64,
    /// `∂/∂η` of the integrand, `∇_η log q(η) + Ω⁻¹η`.
    ///
    /// Deliberately shaped to be **added to the data term's `∂/∂η`** so one
    /// `chain_to_phi` covers both. Chaining only this pathwise part — and dropping
    /// the direct `∂ log q_φ(η)/∂φ` score term that a total derivative would also
    /// carry — *is* the path-derivative / "sticking the landing" estimator.
    d_eta: Vec<f64>,
    /// `∂/∂Ω` of the integrand: `½(Ω⁻¹ − Ω⁻¹ηηᵀΩ⁻¹)`, from `½log|Ω|` and
    /// `½ηᵀΩ⁻¹η` respectively.
    ///
    /// Unlike the `φ` gradient this one has nothing dropped — `Ω` does not appear
    /// in `q` — so it is the exact derivative of the reported value, and it
    /// averages to the closed form's `∂KL/∂Ω` because `E_q[ηηᵀ] = S + μμᵀ`. The
    /// closed-form `Ω` update therefore stays valid under this route; it is only
    /// noisier.
    d_omega: DMatrix<f64>,
}

/// Evaluate the Monte-Carlo KL integrand and its derivatives at one draw.
///
/// # Why the `φ` gradient is not the derivative of the value
///
/// The path-derivative estimator drops `∂ log q_φ(η)/∂φ` at fixed `η`. That term has
/// zero expectation under `q` (it is a score function), so the estimator stays
/// unbiased, and dropping it *reduces* variance — decisively so near the optimum,
/// where it is the only surviving noise. It also makes the KL gradient vanish
/// identically, draw by draw, when `q = N(0, Ω)`: there `∇_η log q + Ω⁻¹η ≡ 0`.
///
/// The consequence to keep in mind: under this route `∂(−ELBO)/∂φ` is **not** the
/// finite difference of `−ELBO`, and a parity test asserting so would fail by
/// design. The `θ`, `σ` and `Ω` blocks are unaffected and remain exactly FD-checkable
/// — see `mc_kl_grad_x_matches_fd_but_phi_is_path_derivative`.
fn mc_kl_draw(
    family: &dyn VariationalFamily,
    phi: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
) -> McKlDraw {
    let d = eta.len();
    let eta_v = DVector::from_column_slice(eta);
    let oinv_eta = &omega.inv * &eta_v;

    let (log_q, d_log_q) = family.log_density(phi, eta);
    // −log p(η | Ω) = ½(ηᵀΩ⁻¹η + log|Ω| + d·log 2π)
    let neg_log_p =
        0.5 * (eta_v.dot(&oinv_eta) + omega.log_det + (d as f64) * std::f64::consts::TAU.ln());

    let d_eta = (0..d).map(|k| d_log_q[k] + oinv_eta[k]).collect();
    let d_omega = (&omega.inv - &oinv_eta * oinv_eta.transpose()) * 0.5;

    McKlDraw {
        integrand: log_q + neg_log_p,
        d_eta,
        d_omega,
    }
}

/// Packed-space bounds marking FIXed coordinates, in the `[log θ | log σ]` layout
/// `obs_nll_subject_grad` expects. A FIXed coordinate gets `lower == upper`, which
/// makes it contribute zero *and* skips its finite-difference evaluation.
fn theta_sigma_masks(template: &ModelParameters) -> (Vec<bool>, Vec<f64>, Vec<f64>) {
    let n_theta = template.theta.len();
    let n_sigma = template.sigma.values.len();
    let log_mask: Vec<bool> = (0..n_theta)
        .map(|i| theta_packs_log(template.theta_lower[i]))
        .collect();

    let mut lower = Vec::with_capacity(n_theta + n_sigma);
    let mut upper = Vec::with_capacity(n_theta + n_sigma);
    for i in 0..n_theta {
        let fixed = template.theta_fixed.get(i).copied().unwrap_or(false);
        lower.push(if fixed { 0.0 } else { f64::NEG_INFINITY });
        upper.push(if fixed { 0.0 } else { f64::INFINITY });
    }
    for k in 0..n_sigma {
        let fixed = template.sigma_fixed.get(k).copied().unwrap_or(false);
        lower.push(if fixed { 0.0 } else { f64::NEG_INFINITY });
        upper.push(if fixed { 0.0 } else { f64::INFINITY });
    }
    (log_mask, lower, upper)
}

/// Chain `∂KL/∂Ω` into the packed Cholesky coordinates.
///
/// With `Ω = LLᵀ` and `G = ∂KL/∂Ω` symmetric, `∂KL/∂L = 2GL`; the packed diagonal
/// is `log L_ii`, contributing a further factor of `L_ii`. The iteration order
/// matches `pack_params` exactly (and collapses to the diagonal when `Ω` is
/// declared diagonal).
///
/// # Which coordinates are left at zero
///
/// Two kinds of slot are not estimated parameters and so get no gradient, matching
/// [`crate::estimation::parameterization::omega_structural_zero_mask`] and
/// [`crate::estimation::parameterization::packed_fixed_mask`] respectively:
///
/// * **Structural zeros.** In a mixed `block_omega (ETA_CL, ETA_V)` + `omega ETA_KA`
///   Ω, the cross-block entries are not parameters at all — the model says that
///   covariance does not exist. Their `∂KL/∂Ω` is nonzero all the same (perturbing
///   `Ω` there *does* change the KL), so without this skip `vi_omega_update = adam`
///   would happily estimate a covariance the model declared absent.
/// * **FIXed etas**, under the `fi || fj` rule: an off-diagonal is fixed when
///   *either* of its etas is, because a fixed eta pins its whole row and column.
///
/// Leaving the gradient at zero is sufficient to pin the coordinate — Adam's step
/// for a zero gradient is exactly zero — and for a structural zero it holds in `Ω`
/// space too, not just in `L`: the Cholesky factor of a matrix that is
/// block-diagonal under permutation is zero wherever the matrix is (each recursion
/// term needs an index sharing a block with both `i` and `j`), so a slot that
/// starts at `0.0` and never moves reconstructs `Ω[i,j] = Σₖ L[i,k]·L[j,k]` as
/// exactly `0.0`.
///
/// A FIXed *off-diagonal* is only pinned as well as the Cholesky parameterization
/// allows: `Ω[i,j]` also depends on row `j` of `L`, which is free when `ETA_j` is.
/// That is the same limitation FOCE/FOCEI's packed vector has. The default
/// `vi_omega_update = closed_form` route does not inherit it — [`closed_form_omega`]
/// restores FIXed entries in `Ω` space directly, where the restoration is exact.
fn chain_omega_grad(
    d_omega: &DMatrix<f64>,
    omega: &OmegaMatrix,
    tmpl_block: &OmegaMatrix,
    fixed: &[bool],
    out: &mut [f64],
) {
    let n = omega.dim();
    let dl = (d_omega * &omega.chol) * 2.0;
    let is_fixed = |k: usize| fixed.get(k).copied().unwrap_or(false);
    for (slot, (i, j)) in lower_tri_iter(n, tmpl_block.diagonal).enumerate() {
        if !tmpl_block.free_mask[(i, j)] || is_fixed(i) || is_fixed(j) {
            continue;
        }
        let chain = if i == j { omega.chol[(i, i)] } else { 1.0 };
        out[slot] += dl[(i, j)] * chain;
    }
}

/// Number of occasion groups this subject presents, or `0` for a model without IOV.
///
/// This is what makes the stacked dimension a per-subject quantity: `K` comes from the
/// subject's own occasion column, not from the model.
pub(crate) fn subject_k_occasions(model: &CompiledModel, subject: &Subject) -> usize {
    if model.n_kappa == 0 {
        0
    } else {
        iov_occasion_groups(subject).len()
    }
}

/// Split a stacked draw `z = [η, κ₁ … κ_K]` into its BSV head and per-occasion blocks.
///
/// Returns an empty `kappas` for a non-IOV model, so callers can branch on that alone.
fn split_stacked(
    z: &[f64],
    n_eta: usize,
    n_kappa: usize,
    k_occasions: usize,
) -> (&[f64], Vec<Vec<f64>>) {
    if n_kappa == 0 || k_occasions == 0 {
        return (z, Vec::new());
    }
    let kappas = (0..k_occasions)
        .map(|g| {
            let off = n_eta + g * n_kappa;
            z[off..off + n_kappa].to_vec()
        })
        .collect();
    (&z[..n_eta], kappas)
}

/// One subject's contribution to `−ELBO` and its gradients.
#[allow(clippy::too_many_arguments)]
fn subject_terms(
    model: &CompiledModel,
    subject: &Subject,
    subject_idx: usize,
    params: &ModelParameters,
    family: &dyn VariationalFamily,
    phi: &[f64],
    cfg: &ElboConfig,
    iter: u64,
    log_mask: &[bool],
    lower: &[f64],
    upper: &[f64],
    scratch: &mut EventPkParams,
) -> Result<SubjectTerms, String> {
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let n_eta = params.omega.dim();
    let n_kappa = model.n_kappa;
    // `K` is a property of this subject's data, so the stacked dimension — and hence the
    // family serving it — is per-subject. Zero for a non-IOV model, in which case
    // everything below collapses to the plain `η` path.
    let k_occasions = subject_k_occasions(model, subject);
    let iov = k_occasions > 0 && n_kappa > 0;
    let d = family.n_eta();
    debug_assert_eq!(
        d,
        n_eta + k_occasions * n_kappa,
        "family dimension must match this subject's stacked vector"
    );
    let s = cfg.n_mc_samples.max(1);
    let inv_s = 1.0 / s as f64;

    // `Σ_b = Ω ⊕ Ω_iov^{⊗K}`, or `Ω` itself without IOV. This is the prior the KL is
    // taken against and the one subtracted from the joint η-gradient, so building it
    // once here keeps those two consistent by construction.
    let prior = stacked_prior(&params.omega, params.omega_iov.as_ref(), k_occasions);

    let mut data_nll = 0.0;
    let mut grad_theta_sigma = vec![0.0; n_theta + n_sigma];
    let mut grad_phi = vec![0.0; family.n_params()];
    let mut used_fd = false;

    // Resolve the KL route *before* the draw loop: the closed form is a property of
    // the family, not of the draw, so probing it once keeps the loop branch-free and
    // makes "asked for analytic, had to sample" a single decision to report.
    let closed_form = match cfg.kl {
        KlMode::Analytic => family.kl_to_normal(phi, &prior),
        KlMode::Mc => None,
    };
    let kl_fell_back = cfg.kl == KlMode::Analytic && closed_form.is_none();
    let sampling_kl = closed_form.is_none();
    let mut kl_value = 0.0;
    let mut kl_d_omega = DMatrix::<f64>::zeros(d, d);

    for sample in 0..s {
        let eps = crn_eps(cfg.seed, iter, subject_idx, sample, d);
        let z = family.sample(phi, &eps);
        let (eta, kappas) = split_stacked(&z, n_eta, n_kappa, k_occasions);

        // Data term value and its ∂/∂(θ, σ) at this draw.
        let (nll, g_ts) = if iov {
            obs_nll_subject_grad_iov(
                model,
                subject,
                &params.theta,
                &params.sigma.values,
                eta,
                &kappas,
                log_mask,
                lower,
                upper,
                n_theta,
                n_sigma,
                scratch,
            )
        } else {
            obs_nll_subject_grad(
                model,
                subject,
                &params.theta,
                &params.sigma.values,
                eta,
                log_mask,
                lower,
                upper,
                n_theta,
                n_sigma,
                scratch,
            )
        };
        data_nll += inv_s * nll;
        for (acc, gi) in grad_theta_sigma.iter_mut().zip(g_ts.iter()) {
            *acc += inv_s * gi;
        }

        // Data term's ∂/∂η, pushed through the reparameterization path to φ.
        let (g_eta, fd) = eta_data_grad(
            model,
            subject,
            &params.theta,
            params,
            &prior,
            &params.sigma.values,
            &z,
            &kappas,
            k_occasions,
            cfg.eta_grad,
            scratch,
        );
        if fd {
            if cfg.eta_grad == EtaGradMode::Analytic {
                return Err(format!(
                    "subject {}: the analytic ∂/∂η provider declined this subject, but \
                     vi_eta_grad = analytic forbids the finite-difference fallback",
                    subject.id
                ));
            }
            used_fd = true;
        }

        // The KL's `∂/∂η` rides along with the data term's through the *same*
        // reparameterization chain — that is the whole economy of the path-derivative
        // estimator, and why sampling the KL costs one density evaluation per draw
        // rather than a second gradient pass.
        let mut g_total = g_eta;
        if sampling_kl {
            let draw = mc_kl_draw(family, phi, &z, &prior);
            kl_value += inv_s * draw.integrand;
            kl_d_omega += draw.d_omega * inv_s;
            for (acc, g) in g_total.iter_mut().zip(draw.d_eta.iter()) {
                *acc += g;
            }
        }

        let scaled: Vec<f64> = g_total.iter().map(|g| g * inv_s).collect();
        family.chain_to_phi(phi, &eps, &scaled, &mut grad_phi);
    }

    // Closed form: the value and both derivatives come back exactly, with no
    // contribution from the draw loop above.
    if let Some(kl) = closed_form {
        kl_value = kl.value;
        kl_d_omega = kl.d_omega;
        for (acc, g) in grad_phi.iter_mut().zip(kl.d_phi.iter()) {
            *acc += g;
        }
    }

    Ok(SubjectTerms {
        data_nll,
        kl: kl_value,
        grad_theta_sigma,
        grad_phi,
        d_omega: kl_d_omega,
        used_fd,
        kl_fell_back,
    })
}

/// Evaluate `−ELBO` and its gradients over the whole population.
///
/// `x` is the packed population vector (`[θ | chol(Ω) | σ]`), `phis[i]` subject
/// `i`'s variational parameters, and `iter` the iteration index that selects the
/// common random numbers.
///
/// Subjects are evaluated in parallel but folded **serially in subject order**:
/// `f64` addition is not associative, so a parallel reduction would make the
/// objective depend on the rayon worker count (#703).
#[allow(clippy::too_many_arguments)]
pub fn population_neg_elbo(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    families: Families<'_>,
    phis: &[Vec<f64>],
    cfg: &ElboConfig,
    iter: u64,
) -> Result<ElboEval, String> {
    let params = unpack_params(x, template);
    let layout = PackedLayout::new(template);
    let (log_mask, lower, upper) = theta_sigma_masks(template);

    let per_subj: Vec<Result<SubjectTerms, String>> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            subject_terms(
                model,
                subject,
                i,
                &params,
                families.for_subject(i),
                &phis[i],
                cfg,
                iter,
                &log_mask,
                &lower,
                &upper,
                scratch,
            )
        })
        .collect();

    let mut eval = ElboEval {
        neg_elbo: 0.0,
        data_term: 0.0,
        kl_term: 0.0,
        grad_x: vec![0.0; layout.total()],
        grad_phi: Vec::with_capacity(population.subjects.len()),
        n_fd_subjects: 0,
        n_kl_fallback_subjects: 0,
    };

    for terms in per_subj {
        let t = terms?;
        eval.data_term += t.data_nll;
        eval.kl_term += t.kl;
        for i in 0..layout.n_theta {
            eval.grad_x[i] += t.grad_theta_sigma[i];
        }
        for k in 0..layout.n_sigma {
            eval.grad_x[layout.sigma_start() + k] += t.grad_theta_sigma[layout.n_theta + k];
        }
        // `d_omega` is over the *stacked* prior, so split it before chaining. The BSV
        // block is the top-left corner; each occasion's `κ` block is a copy of the same
        // `Ω_iov`, so by the chain rule for a direct sum their derivatives **add**.
        let n_eta = params.omega.dim();
        let d_bsv = t.d_omega.view((0, 0), (n_eta, n_eta)).into_owned();
        chain_omega_grad(
            &d_bsv,
            &params.omega,
            &template.omega,
            &template.omega_fixed,
            &mut eval.grad_x[layout.omega_start()..layout.sigma_start()],
        );
        if let (Some(iov), Some(tmpl_iov)) =
            (params.omega_iov.as_ref(), template.omega_iov.as_ref())
        {
            let n_kappa = iov.dim();
            let k_occ = (t.d_omega.nrows().saturating_sub(n_eta)) / n_kappa.max(1);
            let mut d_iov = DMatrix::<f64>::zeros(n_kappa, n_kappa);
            for g in 0..k_occ {
                let off = n_eta + g * n_kappa;
                d_iov += t.d_omega.view((off, off), (n_kappa, n_kappa));
            }
            chain_omega_grad(
                &d_iov,
                iov,
                tmpl_iov,
                &template.kappa_fixed,
                &mut eval.grad_x[layout.omega_iov_start()..layout.omega_iov_end()],
            );
        }
        eval.grad_phi.push(t.grad_phi);
        if t.used_fd {
            eval.n_fd_subjects += 1;
        }
        if t.kl_fell_back {
            eval.n_kl_fallback_subjects += 1;
        }
    }

    eval.neg_elbo = eval.data_term + eval.kl_term;
    Ok(eval)
}

/// The ELBO-maximizing `Ω` given the current variational posteriors:
/// `Ω* = (1/N) Σᵢ (Sᵢ + μᵢ μᵢᵀ)`.
///
/// Under the analytic KL this is **exact**, not an approximation — it is the
/// stationary point of `Σᵢ KL(qᵢ ‖ N(0,Ω))` in `Ω`, which is the only place `Ω`
/// appears in the objective. Taking it directly removes `Ω` from the stochastic
/// optimization, which is where Janssen et al. report their worst instability.
///
/// Structural zeros (`free_mask`) are respected, so a diagonal or block `Ω` keeps
/// its declared shape rather than picking up sampling correlation in entries the
/// model says do not exist. FIXed diagonals are left at their declared value.
/// Impose a template's structural zeros, FIXed rows/columns and diagonal floor on an
/// accumulated `Σ(S + μμᵀ)`, then wrap it as an `OmegaMatrix`.
///
/// Shared by the BSV and IOV blocks, which differ only in which template block and which
/// FIXed flags they read — the masking rules themselves are identical, and having one
/// copy is what stops them drifting apart.
fn finalize_omega_block(mut acc: DMatrix<f64>, tmpl: &OmegaMatrix, fixed: &[bool]) -> OmegaMatrix {
    let d = tmpl.dim();
    // Structural zeros stay zero: `Σᵢ(Sᵢ + μᵢμᵢᵀ)` is always dense, so a mixed
    // `block_omega` + diagonal Ω would otherwise pick up sampling correlation in
    // cross-block entries the model says do not exist.
    for i in 0..d {
        for j in 0..d {
            if !tmpl.free_mask[(i, j)] {
                acc[(i, j)] = 0.0;
            }
        }
    }
    // Then FIXed entries keep their whole declared row and column, under the same
    // `fi || fj` rule `packed_fixed_mask` uses — a FIXed eta pins its covariances
    // with every other eta, not just its own variance. Restoring in `Ω` space (as
    // SAEM's Ω update does) rather than in Cholesky coordinates makes this exact:
    // holding `L[i,·]` fixed would not hold `Ω[i,j]` fixed, since that also depends
    // on the free row `L[j,·]`. Applied after the structural zeroing, though the
    // order is immaterial — a structurally-absent entry is zero in the template too.
    for i in 0..d {
        for j in 0..d {
            let fi = fixed.get(i).copied().unwrap_or(false);
            let fj = fixed.get(j).copied().unwrap_or(false);
            if fi || fj {
                acc[(i, j)] = tmpl.matrix[(i, j)];
            }
        }
    }
    crate::estimation::saem::floor_omega_diagonal(
        &mut acc,
        fixed,
        crate::estimation::saem::SAEM_OMEGA_DIAG_FLOOR,
    );
    OmegaMatrix::from_matrix_with_mask(
        acc,
        tmpl.eta_names.clone(),
        tmpl.diagonal,
        tmpl.free_mask.clone(),
    )
}

/// The ELBO-maximizing `Ω` — and, under IOV, `Ω_iov` — given the current variational
/// posteriors.
///
/// ```text
///   Ω*     = (1/N)      Σᵢ ( Sᵢ[ηη]  + μᵢ[η]  μᵢ[η]ᵀ  )
///   Ω_iov* = (1/Σᵢ Kᵢ)  Σᵢ Σ_g ( Sᵢ[κ_g] + μᵢ[κ_g] μᵢ[κ_g]ᵀ )
/// ```
///
/// Under the analytic KL these are **exact**, not approximations — they are the
/// stationary point of `Σᵢ KL(qᵢ ‖ N(0, Σ_b))` in each block, which is the only place
/// either appears in the objective. Taking them directly removes both from the
/// stochastic optimization.
///
/// Note the two different denominators. `Ω` averages over **subjects**, while `Ω_iov`
/// pools over **every occasion of every subject** — the same sufficient statistic SAEM
/// accumulates, and the reason a subject with more occasions contributes proportionally
/// more to the IOV variance. It also makes explicit how much less data informs `Ω_iov`:
/// `Σᵢ Kᵢ` occasions, each contributing one `S + μμᵀ`, against `N` subjects for `Ω`.
///
/// `k_occasions[i]` is subject `i`'s occasion count (0 without IOV). Returns `None` for
/// the IOV block when the model declares no `kappa`.
pub fn closed_form_omega(
    families: Families<'_>,
    phis: &[Vec<f64>],
    k_occasions: &[usize],
    template: &ModelParameters,
) -> (OmegaMatrix, Option<OmegaMatrix>) {
    let n_eta = template.omega.dim();
    let mut acc = DMatrix::<f64>::zeros(n_eta, n_eta);

    let iov_tmpl = template.omega_iov.as_ref();
    let n_kappa = iov_tmpl.map(|m| m.dim()).unwrap_or(0);
    let mut acc_iov = DMatrix::<f64>::zeros(n_kappa.max(1), n_kappa.max(1));
    let mut n_occ_total = 0usize;

    for (i, phi) in phis.iter().enumerate() {
        let (mu, s) = families.for_subject(i).moments(phi);
        // BSV block: the leading `n_eta` coordinates of the stacked posterior.
        let mu_e = mu.rows(0, n_eta);
        acc += s.view((0, 0), (n_eta, n_eta)) + &mu_e * mu_e.transpose();

        if n_kappa == 0 {
            continue;
        }
        let k = k_occasions.get(i).copied().unwrap_or(0);
        for g in 0..k {
            let off = n_eta + g * n_kappa;
            let mu_k = mu.rows(off, n_kappa);
            acc_iov += s.view((off, off), (n_kappa, n_kappa)) + &mu_k * mu_k.transpose();
            n_occ_total += 1;
        }
    }

    if !phis.is_empty() {
        acc /= phis.len() as f64;
    }
    let omega = finalize_omega_block(acc, &template.omega, &template.omega_fixed);

    let omega_iov = iov_tmpl.map(|tmpl| {
        if n_occ_total > 0 {
            acc_iov /= n_occ_total as f64;
        } else {
            // No occasions anywhere (an IOV model whose data carries none). Leave the
            // declared value rather than collapsing to the diagonal floor.
            acc_iov = tmpl.matrix.clone();
        }
        finalize_omega_block(acc_iov.clone(), tmpl, &template.kappa_fixed)
    });

    (omega, omega_iov)
}

/// Whether this model's residual error admits the closed-form `σ` M-step, and if
/// so which `σ` it applies to.
///
/// `Ok(k)` means `sigma.values[k]` is the single scalar the closed form solves for.
/// `Err(reason)` is a user-facing sentence explaining the fall back to Adam; it is
/// surfaced through `FitResult::warnings` rather than swallowed, on the same
/// principle as the KL fallback — a scope gap must fail loudly, not silently
/// produce a `σ` from a formula that does not apply to the model in hand.
pub fn closed_form_sigma_support(
    model: &CompiledModel,
    template: &ModelParameters,
) -> Result<usize, String> {
    // The error structure is checked *before* the σ count, so the reason a user is given
    // is the one that explains their model: a combined model necessarily declares two σ,
    // and "combined error has no scalar stationary point" is more use than "you declared
    // two σ". The count check below still catches a declared-but-unused σ.
    match model.error_spec {
        ErrorSpec::Single(ErrorModel::Additive) | ErrorSpec::Single(ErrorModel::Proportional) => {}
        ErrorSpec::Single(ErrorModel::Combined) => {
            return Err(
                "the combined error model couples two σ in one variance, which has no \
                 scalar stationary point"
                    .to_string(),
            )
        }
        // Per-endpoint and covariate-selected error models each carry their own
        // σ set, so the single-scalar derivation does not describe them.
        ErrorSpec::PerCmt(_) => {
            return Err("per-CMT error models declare a σ set per endpoint".to_string())
        }
        ErrorSpec::Selected { .. } => {
            return Err("covariate-selected error models declare a σ set per branch".to_string())
        }
    }
    // One scalar only. Two σ that both scale the same variance are not separately
    // identified by a single stationarity condition.
    if template.sigma.values.len() != 1 {
        return Err(format!(
            "the model declares {} σ parameters and the closed form solves for one",
            template.sigma.values.len()
        ));
    }
    if template.sigma_fixed.first().copied().unwrap_or(false) {
        return Err("σ is declared FIX".to_string());
    }
    if !model.residual_correlations.is_empty() {
        return Err("correlated residuals put σ inside a non-diagonal R".to_string());
    }
    if model.frem_config.is_some() {
        return Err("FREM pseudo-observations carry their own residual variance".to_string());
    }
    if matches!(model.bloq_method, BloqMethod::M3) {
        return Err(
            "M3 BLOQ adds a censored log-CDF term in which σ has no closed form".to_string(),
        );
    }
    if model.residual_error_eta.is_some() {
        return Err("IIV on residual error scales the variance per subject".to_string());
    }
    Ok(0)
}

/// The **exact** maximizer of the ELBO in a single proportional or additive `σ`.
///
/// # The identity this uses
///
/// The data term is `E_q[−log p(y|η)] = ½ Σ_obs E_q[log 2π + log v + (y−f)²/v]`,
/// with `v = (σ·f)²` for proportional error and `v = σ²` for additive. Either way
/// `σ` enters `log v` as `2 log σ` and `(y−f)²/v` as a `σ⁻²` factor, so
///
/// ```text
/// ∂(data term)/∂(log σ) = n_obs − (1/σ²) · Σ_obs E_q[(y − f)² / f²]      (proportional)
///                       = n_obs − (1/σ²) · Σ_obs E_q[(y − f)²]           (additive)
/// ```
///
/// Setting that to zero gives `σ*² = (1/n_obs) Σ E_q[…]`, and rearranging the same
/// identity expresses the sufficient statistic in terms of the gradient already in
/// hand: `Σ E_q[…] = σ²(n_obs − g)`. Hence
///
/// ```text
/// σ* = σ · sqrt(1 − g / n_obs)
/// ```
///
/// So the M-step costs **nothing** beyond the gradient the ELBO computes anyway —
/// no extra prediction pass, unlike a fresh `E_q` evaluation. `g` is the gradient
/// of the **negative** ELBO with respect to `log σ`; the KL term does not involve
/// `σ`, so the data term is the whole of it.
///
/// # Returns
///
/// `None` when the implied statistic is not positive (`g ≥ n_obs`). Exactly, that
/// cannot happen — `Σ E_q[(y−f)²/f²] ≥ 0` bounds `g ≤ n_obs` — so it signals Monte
/// Carlo noise having pushed the estimate past the boundary, and the caller keeps
/// the current `σ` for this iteration rather than taking a `sqrt` of a negative.
pub fn closed_form_sigma(sigma: f64, grad_log_sigma: f64, n_obs: usize) -> Option<f64> {
    if n_obs == 0 || !sigma.is_finite() || sigma <= 0.0 || !grad_log_sigma.is_finite() {
        return None;
    }
    let factor = 1.0 - grad_log_sigma / n_obs as f64;
    if factor <= 0.0 || factor.is_nan() {
        return None;
    }
    let next = sigma * factor.sqrt();
    if next.is_finite() && next > 0.0 {
        Some(next)
    } else {
        None
    }
}

#[cfg(test)]
#[path = "elbo_tests.rs"]
mod tests;

/// Exact-case oracle (VI_PLAN §6 test 2). Kept as its own module because it is the
/// only VI check anchored on an *externally* known answer rather than on internal
/// consistency — see the module docs for why that distinction matters.
#[cfg(test)]
#[path = "elbo_oracle.rs"]
mod oracle_tests;

/// The bound property against AGQ (`VI_VALIDATION.md` Anchor A). Separate from
/// `oracle_tests` because it covers the case that oracle structurally cannot: a
/// posterior the variational family is *unable* to represent, so the bound gap is
/// genuinely nonzero — see the module docs.
#[cfg(test)]
#[path = "elbo_agq_bound.rs"]
mod agq_bound_tests;

// ---------------------------------------------------------------------------
// ELBO tightness diagnostic
// ---------------------------------------------------------------------------

/// How far the sampled data term sits above the same term evaluated at the
/// variational means, against how far it *should* sit.
///
/// # What this catches
///
/// The ELBO is a lower bound on the log likelihood, so `−2·ELBO` overshoots
/// `−2 log L`. A healthy fit overshoots by a little. A fit that has descended
/// into a bad basin can overshoot by orders of magnitude — and nothing in the
/// convergence test notices, because a stuck optimizer produces a flat trace
/// and stable parameters just as a converged one does. Measured on a
/// 60-subject deep compartment model: `−2·ELBO` reported 140 136 where the true
/// `−2 log L` at the same estimate was 2 034, and the fit reported
/// `converged: true` — 973 OFV worse than the same model started sensibly, with
/// the clearance decline understated by 43%.
///
/// The check is a comparison the run can make on its own, without a second
/// estimator. `E_q[−log p(y|η)] − (−log p(y|μ))` is the expected excess of the
/// data term over its value at the posterior mean; to second order that is
/// `½ tr(H·S)`, and when `q` approximates the posterior curvature (`S ≈ H⁻¹`)
/// it reduces to `d/2` per subject. So the *scale* of a healthy excess is set by
/// the stacked dimension and the subject count, both of which are known.
/// Anything far above that means `q` is nowhere near the posterior, the
/// objective is dominated by draws in a region the mean never sees, or both.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ElboTightness {
    /// `data_term − data_term_at_means`.
    pub excess: f64,
    /// `Σᵢ dᵢ/2`, the second-order expectation when `q` matches the posterior.
    pub expected: f64,
}

impl ElboTightness {
    /// Multiple of the expected excess. `1.0` is the ideal; large values mean
    /// the bound is not usable and the estimate should not be trusted.
    pub fn ratio(&self) -> f64 {
        if self.expected > 0.0 {
            self.excess / self.expected
        } else {
            f64::INFINITY
        }
    }

    /// Above this multiple the fit is reported as untrustworthy.
    ///
    /// Measured on the same 60-subject deep compartment model, same data, same
    /// seed, differing only in where the network started:
    ///
    /// | start | `−2 log L` | ratio |
    /// |---|---|---|
    /// | bare `softplus` head (every output at 0.693) | 2033.6 | **306.6** |
    /// | `init = [1.0, 10.0]` | 1061.0 | **0.54** |
    ///
    /// Nearly three orders of magnitude apart, so the threshold sits in a wide
    /// empty gap rather than on a boundary. That is the property worth having:
    /// a diagnostic needing per-model calibration would not survive contact with
    /// models nobody has run yet. It is set at 25 rather than nearer the healthy
    /// value because the `d/2` expectation is exact only for a quadratic
    /// log-likelihood, and a nonlinear PK model with a proportional residual can
    /// legitimately run some multiple above it.
    pub const IMPLAUSIBLE_RATIO: f64 = 25.0;

    pub fn is_implausible(&self) -> bool {
        self.excess.is_finite() && self.ratio() > Self::IMPLAUSIBLE_RATIO
    }
}

/// Evaluate the data term at the variational means and compare it against the
/// sampled one.
///
/// Costs one likelihood evaluation per subject — the same call the ELBO already
/// makes `n_mc_samples` times per subject per iteration — so running it once at
/// the end is free relative to the fit.
pub(crate) fn elbo_tightness(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    sampled_data_term: f64,
    eta_means: &[Vec<f64>],
    kappa_means: &[Vec<Vec<f64>>],
) -> ElboTightness {
    let n_eta = params.omega.dim();
    let mut at_means = 0.0;
    let mut expected = 0.0;
    let mut scratch = EventPkParams::default();

    for (i, subject) in population.subjects.iter().enumerate() {
        let Some(eta) = eta_means.get(i) else {
            continue;
        };
        let kappas = kappa_means.get(i).cloned().unwrap_or_default();
        let iov = !kappas.is_empty() && model.n_kappa > 0;
        at_means += if iov {
            obs_nll_subject_into_iov(
                model,
                subject,
                &params.theta,
                &params.sigma.values,
                eta,
                &kappas,
                &mut scratch,
            )
        } else {
            obs_nll_subject_into(
                model,
                subject,
                &params.theta,
                &params.sigma.values,
                eta,
                &mut scratch,
            )
        };
        // The stacked dimension this subject's `q` actually covers.
        expected += 0.5 * (n_eta + kappas.len() * model.n_kappa) as f64;
    }

    ElboTightness {
        excess: sampled_data_term - at_means,
        expected,
    }
}
