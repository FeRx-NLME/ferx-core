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
//! `E_q[log q(η) − log p(η|Ω)]` — see [`mc_kl_draw`], and note that its `φ` gradient
//! is then a path derivative rather than the derivative of the reported value.
//!
//! Everything here works with **`−ELBO`**, so the quantity is a thing to
//! *minimize*, consistent with every other objective in the crate, and can be
//! handed straight to [`super::adam::AdamState::step`].
//!
//! # The additive constant
//!
//! The data term is [`obs_nll_subject_grad`]'s NLL, which — like `individual_nll`,
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

use crate::estimation::fixed_eta_gradient::obs_nll_subject_grad;
use crate::estimation::inner_optimizer::analytic_eta_nll_gradient;
use crate::estimation::parameterization::{lower_tri_iter, theta_packs_log, unpack_params};
use crate::pk::EventPkParams;
use crate::stats::likelihood::obs_nll_subject_into;
use crate::types::{CompiledModel, ModelParameters, OmegaMatrix, Population, Subject};

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

/// Offsets of the three blocks of the packed parameter vector produced by
/// [`crate::estimation::parameterization::pack_params`]: `[θ | chol(Ω) | σ]`.
///
/// VI v1 rejects IOV, so the trailing `Ω_iov` block that `pack_params` would
/// otherwise append is always absent.
#[derive(Debug, Clone, Copy)]
pub struct PackedLayout {
    pub n_theta: usize,
    pub n_omega: usize,
    pub n_sigma: usize,
}

impl PackedLayout {
    pub fn new(template: &ModelParameters) -> Self {
        let n_eta = template.omega.dim();
        Self {
            n_theta: template.theta.len(),
            n_omega: lower_tri_iter(n_eta, template.omega.diagonal).count(),
            n_sigma: template.sigma.values.len(),
        }
    }

    pub fn total(&self) -> usize {
        self.n_theta + self.n_omega + self.n_sigma
    }

    /// Index of the first `Ω` Cholesky coordinate.
    pub fn omega_start(&self) -> usize {
        self.n_theta
    }

    /// Index of the first `σ` coordinate.
    pub fn sigma_start(&self) -> usize {
        self.n_theta + self.n_omega
    }
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
    if model.n_kappa > 0 {
        return Some(
            "VI does not support IOV (n_kappa > 0): the variational family would have to \
             cover (η, κ) jointly. Use method = saem or focei for IOV models."
                .to_string(),
        );
    }
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
    /// finite difference of [`Self::neg_elbo`] — see [`mc_kl_draw`].
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
fn fd_eta_data_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta: &[f64],
    scratch: &mut EventPkParams,
) -> Vec<f64> {
    let mut g = vec![0.0; eta.len()];
    for k in 0..eta.len() {
        let h = 1e-5 * (1.0 + eta[k].abs());
        let mut ep = eta.to_vec();
        let mut em = eta.to_vec();
        ep[k] += h;
        em[k] -= h;
        let fp = obs_nll_subject_into(model, subject, theta, sigma, &ep, scratch);
        let fm = obs_nll_subject_into(model, subject, theta, sigma, &em, scratch);
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
    omega: &OmegaMatrix,
    sigma: &[f64],
    eta: &[f64],
    mode: EtaGradMode,
    scratch: &mut EventPkParams,
) -> (Vec<f64>, bool) {
    if mode != EtaGradMode::Fd {
        if let Some(joint) = analytic_eta_nll_gradient(model, subject, theta, eta, omega, sigma) {
            let eta_v = DVector::from_column_slice(eta);
            let prior = &omega.inv * &eta_v;
            let g = joint.iter().zip(prior.iter()).map(|(j, p)| j - p).collect();
            return (g, false);
        }
    }
    // `true` whenever the finite-difference route was taken, whether it was asked
    // for or fallen back to — `n_fd_subjects` reports what actually happened.
    // In `Analytic` mode FD is never *requested*, so a `true` there can only mean
    // the provider declined, which is what the caller turns into an error.
    (
        fd_eta_data_grad(model, subject, theta, sigma, eta, scratch),
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
    template: &ModelParameters,
    out: &mut [f64],
) {
    let n_eta = omega.dim();
    let dl = (d_omega * &omega.chol) * 2.0;
    let is_fixed = |k: usize| template.omega_fixed.get(k).copied().unwrap_or(false);
    for (slot, (i, j)) in lower_tri_iter(n_eta, template.omega.diagonal).enumerate() {
        if !template.omega.free_mask[(i, j)] || is_fixed(i) || is_fixed(j) {
            continue;
        }
        let chain = if i == j { omega.chol[(i, i)] } else { 1.0 };
        out[slot] += dl[(i, j)] * chain;
    }
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
    let d = family.n_eta();
    let s = cfg.n_mc_samples.max(1);
    let inv_s = 1.0 / s as f64;

    let mut data_nll = 0.0;
    let mut grad_theta_sigma = vec![0.0; n_theta + n_sigma];
    let mut grad_phi = vec![0.0; family.n_params()];
    let mut used_fd = false;

    // Resolve the KL route *before* the draw loop: the closed form is a property of
    // the family, not of the draw, so probing it once keeps the loop branch-free and
    // makes "asked for analytic, had to sample" a single decision to report.
    let closed_form = match cfg.kl {
        KlMode::Analytic => family.kl_to_normal(phi, &params.omega),
        KlMode::Mc => None,
    };
    let kl_fell_back = cfg.kl == KlMode::Analytic && closed_form.is_none();
    let sampling_kl = closed_form.is_none();
    let mut kl_value = 0.0;
    let mut kl_d_omega = DMatrix::<f64>::zeros(d, d);

    for sample in 0..s {
        let eps = crn_eps(cfg.seed, iter, subject_idx, sample, d);
        let eta = family.sample(phi, &eps);

        // Data term value and its ∂/∂(θ, σ) at this η.
        let (nll, g_ts) = obs_nll_subject_grad(
            model,
            subject,
            &params.theta,
            &params.sigma.values,
            &eta,
            log_mask,
            lower,
            upper,
            n_theta,
            n_sigma,
            scratch,
        );
        data_nll += inv_s * nll;
        for (acc, gi) in grad_theta_sigma.iter_mut().zip(g_ts.iter()) {
            *acc += inv_s * gi;
        }

        // Data term's ∂/∂η, pushed through the reparameterization path to φ.
        let (g_eta, fd) = eta_data_grad(
            model,
            subject,
            &params.theta,
            &params.omega,
            &params.sigma.values,
            &eta,
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
            let draw = mc_kl_draw(family, phi, &eta, &params.omega);
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
    family: &dyn VariationalFamily,
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
                model, subject, i, &params, family, &phis[i], cfg, iter, &log_mask, &lower, &upper,
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
        chain_omega_grad(
            &t.d_omega,
            &params.omega,
            template,
            &mut eval.grad_x[layout.omega_start()..layout.sigma_start()],
        );
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
pub fn closed_form_omega(
    family: &dyn VariationalFamily,
    phis: &[Vec<f64>],
    template: &ModelParameters,
) -> OmegaMatrix {
    let d = template.omega.dim();
    let mut acc = DMatrix::<f64>::zeros(d, d);
    for phi in phis {
        let (mu, s) = family.moments(phi);
        acc += s + &mu * mu.transpose();
    }
    if !phis.is_empty() {
        acc /= phis.len() as f64;
    }

    // Structural zeros stay zero: `Σᵢ(Sᵢ + μᵢμᵢᵀ)` is always dense, so a mixed
    // `block_omega` + diagonal Ω would otherwise pick up sampling correlation in
    // cross-block entries the model says do not exist.
    for i in 0..d {
        for j in 0..d {
            if !template.omega.free_mask[(i, j)] {
                acc[(i, j)] = 0.0;
            }
        }
    }
    // Then FIXed etas keep their whole declared row and column, under the same
    // `fi || fj` rule `packed_fixed_mask` uses — a FIXed eta pins its covariances
    // with every other eta, not just its own variance. Restoring in `Ω` space (as
    // SAEM's Ω update does) rather than in Cholesky coordinates makes this exact:
    // holding `L[i,·]` fixed would not hold `Ω[i,j]` fixed, since that also depends
    // on the free row `L[j,·]`. Applied after the structural zeroing, though the
    // order is immaterial — a structurally-absent entry is zero in the template too.
    for i in 0..d {
        for j in 0..d {
            let fi = template.omega_fixed.get(i).copied().unwrap_or(false);
            let fj = template.omega_fixed.get(j).copied().unwrap_or(false);
            if fi || fj {
                acc[(i, j)] = template.omega.matrix[(i, j)];
            }
        }
    }
    crate::estimation::saem::floor_omega_diagonal(
        &mut acc,
        &template.omega_fixed,
        crate::estimation::saem::SAEM_OMEGA_DIAG_FLOOR,
    );

    OmegaMatrix::from_matrix_with_mask(
        acc,
        template.omega.eta_names.clone(),
        template.omega.diagonal,
        template.omega.free_mask.clone(),
    )
}

#[cfg(test)]
#[path = "elbo_tests.rs"]
mod tests;
