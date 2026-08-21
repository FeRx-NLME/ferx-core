/// SAEM (Stochastic Approximation EM) for NLME population parameter estimation.
///
/// Reference: Delyon, Lavielle, Moulines (1999) Annals of Statistics 94–128.
///            Kuhn & Lavielle (2004) ESAIM: Probability and Statistics 8:115–131.
///
/// Two-phase step-size schedule (Monolix convention):
///   Phase 1 (exploration, k ≤ K1):  γₖ = 1          — rapid basin convergence
///   Phase 2 (convergence, k > K1):  γₖ = 1/(k−K1)   — almost-sure convergence to MLE
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{pop_nll, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::pk::EventPkParams;
use crate::stats::likelihood::{
    individual_nll, individual_nll_into, individual_nll_iov, iov_occasion_groups,
    obs_nll_subject_from_preds, obs_nll_subject_into,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

/// NLopt algorithm used for the SAEM M-step (non-mu-ref thetas + sigma).
///
/// BOBYQA was chosen over the prior SLSQP after the Emax PKPD benchmark
/// showed SLSQP locking onto one side of the Emax-Hill identifiability
/// ridge while BOBYQA's quadratic trust-region exploration landed much
/// closer to truth at ~40% lower wall (no FD-gradient eval per parameter).
/// On simpler PK-only models the two are numerically equivalent
/// (|ΔOFV| < 0.1) and within measurement noise on wall.
///
/// Exposed pub(crate) so the unit test can pin the choice across refactors.
pub(crate) const MSTEP_NLOPT_ALGORITHM: nlopt::Algorithm = nlopt::Algorithm::Bobyqa;

// ---------------------------------------------------------------------------
// SAEM state
// ---------------------------------------------------------------------------

/// Positive-definite floor for free BSV Ω diagonals in the M-step.
///
/// Larger than the IOV floor (1e-8) because the BSV MH proposal scale is
/// `step_scale · chol(Ω)`: if a diagonal is allowed near zero the proposal for
/// that η collapses and the chain can no longer move it, so Ω must stay large
/// enough to keep the random walk alive. 1e-6 keeps a free η explorable while
/// being far below any plausible estimated variance.
pub(crate) const SAEM_OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Target acceptance rate for the componentwise (1-D) eta kernel. The optimal
/// scaling result for single-coordinate random-walk Metropolis is ≈0.44
/// (Roberts & Rosenthal 2001), higher than the block kernel's 0.40 target.
const CW_TARGET_ACCEPT: f64 = 0.44;

/// Combined (block + componentwise) post-burn-in MH acceptance rate below which
/// SAEM appends a "sampler is not mixing" warning to `FitResult.warnings`
/// (issue #895). A genuinely stuck E-step never updates the ETAs, so the M-step
/// runs on degenerate sufficient statistics and the estimates are unreliable.
const SAEM_MH_STUCK_ACCEPT: f64 = 0.01;

/// Maximum growth of a free PK-residual σ during a SAEM run, in natural-log
/// units, when `iiv_on_ruv` is active (issue #895). The IIV-on-RUV
/// parameterization writes the residual variance as `σ²·exp(2·η_RUV)`, so the
/// *marginal* residual is a function of `σ²·exp(2·ω_RUV)`: σ and ω_RUV trade off
/// along a ridge. If the E-step mixes poorly the M-step can ride that ridge until
/// σ hits its e⁵ ceiling. Capping σ's growth to e³ ≈ 20× its initial SD keeps a
/// genuinely ill-posed run bounded near sensible values while leaving ample room
/// for a well-posed fit to correct a modest starting guess. The cap only ever
/// *tightens* the existing σ upper bound and only for the RUV-scaled residual
/// σ(s) — a FREM EPSCOV (always FIX) is untouched.
const SAEM_RUV_SIGMA_LN_GROWTH: f64 = 3.0;

/// Maximum growth of the `iiv_on_ruv` Ω *variance* during a SAEM run, in
/// natural-log units (issue #895). The other half of the σ × ω_RUV ridge: with
/// both free, the residual-error IIV variance can run away symmetrically to σ
/// (observed ω_RUV → ~49 in the original report). Capping ω_RUV's growth to
/// e³ ≈ 20× its starting variance bounds a genuinely ill-posed run near sensible
/// values while never binding on a well-posed fit. Applied as a
/// correlation-preserving rescale of the RUV row/column so a block Ω stays
/// positive-definite; a FIXed RUV Ω is untouched.
const SAEM_RUV_OMEGA_LN_GROWTH: f64 = 3.0;

/// Maximum per-iteration stochastic-approximation step for the Ω sufficient
/// statistic *during the exploration phase*. The θ/σ M-step uses the full γ
/// (1.0 in exploration), but Ω is averaged at no more than this rate so a single
/// un-equilibrated MCMC draw cannot overwrite a correlated Ω and trigger the
/// rank-1 collapse feedback. In the convergence phase the cap is lifted and Ω
/// uses the full decaying γ = 1/(k−k1), the same Robbins-Monro schedule as θ.
const OMEGA_SA_MAX_STEP: f64 = 0.1;

/// Maximum per-iteration stochastic-approximation step for the **numerical θ/σ
/// M-step** during the exploration phase — the θ-side counterpart of
/// [`OMEGA_SA_MAX_STEP`] (issue #1011).
///
/// A log-mu-referenced θ is updated by the closed-form `log θ += γ·mean(η)`,
/// which is an exact Robbins-Monro average and therefore already damped. A θ
/// with **no ETA** has no such channel: it is moved only by NLopt re-maximising
/// the η-frozen conditional observation likelihood against a *single* MCMC draw.
/// Assigning that maximiser outright is `argmax` of one draw, not the SA average
/// of `E[argmax]`, so the estimate carries a Monte-Carlo bias that does not decay
/// with iteration count — precisely the "single un-equilibrated MCMC draw"
/// hazard [`OMEGA_SA_MAX_STEP`] exists to prevent for Ω, which the θ channel was
/// left exposed to.
///
/// Measured on the FREM `iiv_on_ruv` reprex of #1011, whose absorption fraction
/// `TVFRD1` carries no ETA (marginal −2logL optimum ≈ 0.29, NONMEM IMP 0.394):
///
/// | exploration cap | TVFRD1 |
/// |---|---|
/// | undamped (1.0, pre-#1011) | 0.039 |
/// | 0.1 | 0.170 |
/// | **0.03** | **0.290** |
/// | 0.01 | 0.355 |
/// | 0.003 | 0.357 |
///
/// 0.03 lands on the marginal optimum. Averaging the M-step *objective* over the
/// last K draws instead reaches only 0.081 at K = 50, and 0.130 at K = 150 for
/// 15× the wall time, because consecutive SAEM draws are heavily autocorrelated.
/// Damping the iterate accumulates over every iteration rather than a K-window,
/// so it is both the more accurate and the cheaper cure.
///
/// In the convergence phase the cap is lifted and the full decaying
/// `γ = 1/(k−k1)` applies, exactly as for Ω.
const MSTEP_SA_MAX_STEP: f64 = 0.03;

/// Robbins-Monro blend of a numerical M-step result into the running estimate:
/// `cur += γ·(new − cur)`. `γ >= 1.0` reproduces the undamped assignment
/// byte-for-byte, which is what a fit with no free numerical θ still does.
///
/// A pinned dimension (mu-referenced, or FIXed) is unaffected either way — NLopt
/// returns it unchanged, so `new == cur` and the blend is a no-op regardless of γ.
/// Does this fit's numerical θ/σ M-step get the #1011 SA damping?
///
/// Only when NLopt actually has a θ to estimate: a θ that is `FIX`ed, or pinned
/// out by the closed-form mu-ref shift (`pinned`), is returned unchanged, so a
/// fit whose every θ is one of those has nothing to damp — and skipping it there
/// keeps such fits byte-identical to the pre-#1011 behaviour, σ included.
///
/// `is_mixture` vetoes it outright. A `MIXNUM`-switched typical value is
/// estimated by this same numerical M-step (#996 routes it there deliberately,
/// because SAEM's hard class draw makes the per-class η mean a biased
/// statistic), but it is solving a different problem: the class typical values
/// must *separate* from a common start, and the class assignments only stabilise
/// once they have. Damping that excursion stalls the separation — on
/// `tests/nonmem/mixture_iv_saem` a 0.03 exploration cap leaves `TVCL1 = 1.145`
/// against NONMEM's 1.002, and drags the chained IMP marginal with it. Whether
/// the #1011 bias also affects mixture class θ (it plausibly does) needs its own
/// schedule and its own anchor, so it is left alone rather than half-fixed; the
/// no-ETA advisory still fires for those models.
fn damps_numerical_mstep(
    is_mixture: bool,
    n_theta: usize,
    theta_fixed: &[bool],
    pinned: &[usize],
) -> bool {
    if is_mixture {
        return false;
    }
    (0..n_theta).any(|t| !theta_fixed.get(t).copied().unwrap_or(false) && !pinned.contains(&t))
}

/// SA step size for the numerical θ/σ M-step (#1011).
///
/// * No θ for NLopt to estimate → `1.0`, the undamped pre-#1011 assignment, so
///   those fits are byte-identical.
/// * Exploration → capped at [`MSTEP_SA_MAX_STEP`], the θ-side counterpart of
///   the [`OMEGA_SA_MAX_STEP`] cap on Ω.
/// * Convergence → the full decaying `γ = 1/(k−k1)`, same schedule as Ω.
fn mstep_sa_step(numerically_estimated_theta: bool, exploring: bool, gamma: f64, cap: f64) -> f64 {
    // `cap >= 1.0` is the documented "off" switch and must restore the pre-#1011
    // assignment in **both** phases. Returning `gamma` in convergence would still
    // damp at `1/(k−k1)`, so "off" has to short-circuit here rather than rely on
    // the `min` below.
    if !numerically_estimated_theta || cap >= 1.0 {
        return 1.0;
    }
    if exploring {
        gamma.min(cap)
    } else {
        gamma
    }
}

fn damp_mstep(cur: &mut [f64], new: &[f64], gamma: f64) {
    if gamma >= 1.0 {
        cur.copy_from_slice(new);
        return;
    }
    for (c, &n) in cur.iter_mut().zip(new.iter()) {
        *c += gamma * (n - *c);
    }
}

/// Raise every *free* diagonal entry of the BSV Ω that has fallen below `floor`
/// up to `floor`. FIX-ed diagonals (`omega_fixed[i] == true`) are left untouched
/// — they carry the user's declared variance and must not be perturbed.
///
/// Shared source of truth for the SAEM and IMPMAP estimators (`impmap.rs` calls
/// this instead of carrying its own byte-identical copy).
pub(crate) fn floor_omega_diagonal(omega_mat: &mut DMatrix<f64>, omega_fixed: &[bool], floor: f64) {
    for i in 0..omega_mat.nrows() {
        let fixed = omega_fixed.get(i).copied().unwrap_or(false);
        if !fixed && omega_mat[(i, i)] < floor {
            omega_mat[(i, i)] = floor;
        }
    }
}

struct SaemState {
    /// Per-subject current ETAs
    etas: Vec<Vec<f64>>,
    /// Per-subject per-occasion kappa samples. `kappas[i][k]` = kappas for
    /// subject i, occasion k.  Empty outer vecs when `n_kappa == 0`.
    kappas: Vec<Vec<Vec<f64>>>,
    /// Cached individual NLL at current ETAs (and kappas for IOV models)
    nll_cache: Vec<f64>,
    /// Per-subject MH step sizes (for the block eta kernel)
    step_scales: Vec<f64>,
    /// Per-subject, per-eta step sizes for the componentwise eta kernel
    /// (Kuhn-Lavielle kernel 2).  Adapted independently for each coordinate
    /// so that etas with vastly different posterior precision (e.g. FREM
    /// covariate etas vs PK etas) can converge to their individual optima.
    /// Indexed `[subject][eta]`.
    cw_step_scales: Vec<Vec<f64>>,
    /// Per-subject kappa MH step sizes.  Empty when `n_kappa == 0`.
    kappa_step_scales: Vec<f64>,
    /// Per-subject acceptance counts since last adaptation
    accept_counts: Vec<usize>,
    /// Per-subject proposal counts since last adaptation (1 for HMC, n_mh_steps for MH)
    proposal_counts: Vec<usize>,
    /// Per-subject, per-eta componentwise-kernel acceptance counts since last
    /// adaptation.  Indexed `[subject][eta]`.
    cw_accept_counts: Vec<Vec<usize>>,
    /// Per-subject, per-eta componentwise-kernel proposal counts since last
    /// adaptation.  Indexed `[subject][eta]`.
    cw_proposal_counts: Vec<Vec<usize>>,
    /// Per-subject kappa acceptance counts since last adaptation.
    kappa_accept_counts: Vec<usize>,
    /// Per-subject kappa proposal counts since last adaptation.
    kappa_proposal_counts: Vec<usize>,
    /// Steps since last adaptation
    steps_since_adapt: usize,
    /// SA sufficient statistic for Omega: running average of (1/N) Σ ηᵢηᵢᵀ
    s2: DMatrix<f64>,
    /// SA sufficient statistic for Omega_iov: running average of (1/N_occ) Σᵢ Σₖ κᵢₖκᵢₖᵀ.
    /// Zero-sized when `n_kappa == 0`.
    s2_iov: DMatrix<f64>,
    /// Current theta
    theta: Vec<f64>,
    /// Current omega matrix
    omega_mat: DMatrix<f64>,
    /// Current Omega_iov matrix (zero-sized when `n_kappa == 0`).
    omega_iov_mat: DMatrix<f64>,
    /// Current sigma values
    sigma_vals: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Metropolis-Hastings step for one subject
// ---------------------------------------------------------------------------

/// Run `n_steps` symmetric random-walk MH iterations for one subject in-place.
/// Returns (n_accepted, updated_nll).
///
/// `eta` is in deviation (eta_true) space — the same space the model's
/// `pk_param_fn` consumes — so proposals are random walks
/// `eta + step_scale · L · z` from the current position. The acceptance
/// log-ratio is `nll_current − nll_prop`, which is correct because the
/// symmetric proposal density cancels.
///
/// Note: an earlier version centred proposals on `mu_k` during exploration.
/// That was incorrect: `individual_nll` interprets `eta` as the deviation
/// `log(CL_i) − log(TVCL)`, while `mu_k = log(TVCL)`, so the model evaluated
/// `CL = TVCL · exp(log TVCL) = TVCL²` for every accepted exploration step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_steps(
    eta: &mut [f64],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    step_scale: f64,
    // Optional per-coordinate multiplier on the joint proposal (issue #895).
    // `None` (or a row of 1.0) reproduces the plain `chol(Ω)·z` block move. A
    // value < 1 shrinks the joint step for that coordinate — used to damp
    // near-deterministic FREM covariate ETAs (posterior SD ≈ √EPSCOV ≪ √Ω_jj),
    // whose full-scale block move would otherwise be rejected every time and
    // pin the whole joint acceptance at 0%. The multiplier is deterministic and
    // symmetric in η, so detailed balance is preserved. Indexed `[0, n_eta)`.
    eta_block_scale: Option<&[f64]>,
    rng: &mut impl Rng,
    n_steps: usize,
    pk_scratch: &mut EventPkParams,
    // When Some, eta proposals are evaluated with IOV-aware NLL (kappas held fixed).
    // This is required for Gibbs correctness in IOV models: the acceptance ratio
    // must target p(η | κ, θ, data), which includes the per-occasion kappa terms.
    kappas_opt: Option<(&[Vec<f64>], &OmegaMatrix)>,
) -> (usize, f64) {
    let n_eta = eta.len();
    let l = &omega.chol;
    let mut nll = nll_current;
    let mut n_accepted = 0;

    for _ in 0..n_steps {
        let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let perturbation = l * z_vec;

        let eta_prop: Vec<f64> = (0..n_eta)
            .map(|j| {
                let bs = eta_block_scale.map_or(1.0, |s| s[j]);
                eta[j] + step_scale * bs * perturbation[j]
            })
            .collect();

        // For non-IOV models: reuse pk_scratch to avoid per-call allocation
        // (dominant allocator pressure on the SAEM hot loop for TV-cov subjects).
        // For IOV models: individual_nll_iov allocates its own scratch; correctness
        // of the Gibbs conditional p(η | κ, θ, data) requires the per-occasion
        // [eta_prop, kappa_k] predictions, which individual_nll_into does not compute.
        let nll_prop = if let Some((kappas, omega_iov)) = kappas_opt {
            individual_nll_iov(
                model,
                subject,
                theta,
                &eta_prop,
                kappas,
                omega,
                Some(omega_iov),
                sigma_values,
            )
        } else {
            individual_nll_into(
                model,
                subject,
                theta,
                &eta_prop,
                omega,
                sigma_values,
                pk_scratch,
            )
        };

        // Symmetric proposal q(η_prop|η) = q(η|η_prop) cancels in the ratio,
        // so the prior+likelihood difference encoded in `individual_nll` is
        // the full acceptance criterion.
        let log_u: f64 = rng.random::<f64>().ln();
        if log_u < nll - nll_prop {
            eta.copy_from_slice(&eta_prop);
            nll = nll_prop;
            n_accepted += 1;
        }
    }

    (n_accepted, nll)
}

/// Componentwise (single-coordinate) Metropolis-within-Gibbs sweep for one
/// subject — the second kernel of the Kuhn & Lavielle (2004) mixture.
///
/// Each sweep proposes a perturbation to one η coordinate at a time,
/// `η'_j = η_j + step_scale · √Ω_jj · z`, holding the other coordinates fixed,
/// and accepts/rejects with the full conditional NLL (which carries the
/// correlated prior, so detailed balance for p(η | data) is preserved). Returns
/// `(n_accepted, n_proposed, updated_nll)` with `n_proposed = n_sweeps · n_eta`.
///
/// Why this kernel exists: the block kernel `mh_steps` proposes along
/// `chol(Ω)·z`, so once Ω drifts toward a high correlation the proposal can only
/// move η along that near-degenerate direction. The single-draw Ω M-step then
/// feeds the induced correlation back into Ω, and during the γ=1 exploration
/// phase (no SA averaging) this compounds into a runaway collapse toward a
/// rank-1 Ω (every off-diagonal correlation → ±1, one variance → 0). A
/// per-coordinate proposal can always move a single η independently of Ω's
/// off-diagonals, so the sampled draws are not forced collinear and the
/// sufficient statistic recovers the true correlation. See the
/// `saem-block-omega-rank1-collapse` investigation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_steps_componentwise(
    eta: &mut [f64],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    // Per-eta step scales — each coordinate adapts its own scale independently
    // so that etas with vastly different posterior precision (e.g. near-
    // deterministic FREM covariate etas vs broad PK etas) can each reach
    // their optimal acceptance rate.
    step_scales: &[f64],
    // Per-coordinate proposal SD = √(marginal variance), precomputed once per
    // iteration from Ω's diagonal (it is identical across subjects) and floored
    // to match the Ω diagonal floor so a collapsing diagonal can't shrink the
    // decorrelating step to zero. Indexed `[0, n_eta)`.
    cw_sd: &[f64],
    rng: &mut impl Rng,
    n_sweeps: usize,
    pk_scratch: &mut EventPkParams,
    kappas_opt: Option<(&[Vec<f64>], &OmegaMatrix)>,
) -> (Vec<usize>, usize, f64) {
    let n_eta = eta.len();
    let mut nll = nll_current;
    let mut per_eta_accepted = vec![0usize; n_eta];

    for _ in 0..n_sweeps {
        for j in 0..n_eta {
            let z: f64 = rng.sample(StandardNormal);
            let old_j = eta[j];
            eta[j] = old_j + step_scales[j] * cw_sd[j] * z;

            let nll_prop = if let Some((kappas, omega_iov)) = kappas_opt {
                individual_nll_iov(
                    model,
                    subject,
                    theta,
                    eta,
                    kappas,
                    omega,
                    Some(omega_iov),
                    sigma_values,
                )
            } else {
                individual_nll_into(model, subject, theta, eta, omega, sigma_values, pk_scratch)
            };

            // Symmetric scalar proposal cancels, same as the block kernel.
            let log_u: f64 = rng.random::<f64>().ln();
            if log_u < nll - nll_prop {
                nll = nll_prop;
                per_eta_accepted[j] += 1;
            } else {
                eta[j] = old_j; // reject — restore
            }
        }
    }

    (per_eta_accepted, n_eta * n_sweeps, nll)
}

// ---------------------------------------------------------------------------
// Per-occasion kappa MH step for IOV models
// ---------------------------------------------------------------------------

/// Run one symmetric random-walk MH proposal for each occasion's kappa.
///
/// For each occasion k, proposes `κ_k_prop = κ_k + step_scale · L_iov · z` and
/// accepts/rejects using the full IOV individual NLL (includes both the kappa
/// prior and the observation likelihood).  The per-occasion Gibbs structure
/// means proposals are low-dimensional (n_kappa typically 1–3), so the MH
/// acceptance rate stays high even without HMC.
///
/// Returns `(n_accepted, n_proposed, updated_nll)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_kappa_steps(
    kappas: &mut [Vec<f64>],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    eta: &[f64],
    omega_bsv: &OmegaMatrix,
    omega_iov: &OmegaMatrix,
    sigma_values: &[f64],
    step_scale: f64,
    rng: &mut impl Rng,
) -> (usize, usize, f64) {
    let n_kappa = omega_iov.matrix.nrows();
    let l = &omega_iov.chol;
    let mut nll = nll_current;
    let mut n_accepted = 0;
    let n_occ = kappas.len();

    for k in 0..n_occ {
        let z: Vec<f64> = (0..n_kappa).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let perturbation = l * z_vec;

        let kap_prop: Vec<f64> = (0..n_kappa)
            .map(|j| kappas[k][j] + step_scale * perturbation[j])
            .collect();

        // Temporarily substitute kappa_k with the proposal.
        let old_kap = kappas[k].clone();
        kappas[k] = kap_prop;

        let nll_prop = individual_nll_iov(
            model,
            subject,
            theta,
            eta,
            kappas,
            omega_bsv,
            Some(omega_iov),
            sigma_values,
        );

        let log_u: f64 = rng.random::<f64>().ln();
        if log_u < nll - nll_prop {
            // Accept
            nll = nll_prop;
            n_accepted += 1;
        } else {
            // Reject — restore old kappa
            kappas[k] = old_kap;
        }
    }

    (n_accepted, n_occ, nll)
}

// ---------------------------------------------------------------------------
// IOV-aware observation NLL for M-step (no priors, per-occasion predictions)
// ---------------------------------------------------------------------------

/// Compute the observation-only NLL for an IOV subject in the SAEM M-step.
///
/// ETAs and kappas are held fixed (sampled values from the E-step).  For each
/// occasion k the combined `[eta, kappa_k]` vector is used to compute predictions;
/// only the observations belonging to that occasion are scored.  No eta or kappa
/// prior terms are included — those are handled by the SA sufficient-statistic
/// update for Ω_bsv and Ω_iov separately.
fn obs_nll_subject_into_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    _pk_scratch: &mut crate::pk::EventPkParams,
) -> f64 {
    use crate::stats::likelihood::m3_logcdf;
    let m3 = matches!(model.bloq_method, BloqMethod::M3);
    // Continuous per-occasion-aware prediction (issue #104) — same model the
    // E-step (`individual_nll_iov`) and FOCEI use, so E and M steps stay
    // consistent. `_pk_scratch` is retained for signature stability but unused
    // (predict_iov manages its own per-event params).
    let preds = crate::pk::predict_iov(model, subject, theta, eta, kappas);
    // FREM covariate pseudo-observations (FREMTYPE > 0) use the covariate sigma
    // (EPSCOV), not the PK residual error — otherwise their near-zero residuals
    // drag PROP/ADD toward zero. See build_frem_r_override.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma_values,
    );
    // IIV on residual error (#409): scale the PK residual variance by
    // exp(2·η_ruv); FREM rows keep their own variance.
    let ruv_scale = model.residual_var_scale(eta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let mut total_nll = 0.0_f64;
    for j in 0..subject.observations.len() {
        // Floors protect log(0) in the M-step objective. individual_nll_iov
        // (the E-step evaluator) does not floor — see obs_nll_subject_grad_iov
        // for why the asymmetry is intentional.
        let f = preds[j].max(1e-12);
        let v = match frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x) {
            Some(vv) => vv.max(1e-12),
            None => {
                (model.residual_variance_at(err_keys[j], f, sigma_values) * ruv_scale).max(1e-12)
            }
        };
        let cens = subject.cens.get(j).copied().unwrap_or(0);
        if m3 && cens != 0 {
            total_nll += -m3_logcdf(subject.observations[j], f, v.sqrt(), cens);
        } else {
            total_nll += 0.5 * (v.ln() + (subject.observations[j] - f).powi(2) / v);
        }
    }

    // Non-Gaussian data term at raw-NLL weight (1×) to match the true NLL (Gaussian obs
    // already contribute at 0.5·(log v + r²/v)): TTE endpoints plus the discrete
    // (binary/categorical) term. No joint-share on the IOV M-step path, and its analytic TTE
    // term matches the `tte_data_term` this site inlined.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        crate::stats::likelihood::accumulate_non_gaussian_nll(
            model,
            subject,
            theta,
            eta,
            None,
            1.0,
            &mut total_nll,
        );
    }

    total_nll
}

/// Gradient of the IOV observation NLL w.r.t. the SAEM packed vector
/// `[log_theta | log_sigma]` for one subject with ETAs and kappas fixed.
///
/// Sigma gradient is analytical (same formula as the non-IOV path but summed
/// across all occasions' observations).  Theta gradient uses forward-FD of
/// per-occasion predictions, chain-rule'd through the per-observation obs_nll.
#[allow(clippy::too_many_arguments)]
fn obs_nll_subject_grad_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    theta_packs_log_mask: &[bool],
    lower: &[f64],
    upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    pk_scratch: &mut crate::pk::EventPkParams,
) -> (f64, Vec<f64>) {
    let n = n_theta + n_sigma;
    // IOV + block_sigma is rejected up front (E_BLOCK_SIGMA_IOV_UNSUPPORTED), so
    // `residual_correlations` is never set on this IOV path — only M3 (and TTE,
    // under the `survival` feature) need the full-FD fallback here.
    let fd_all = matches!(model.bloq_method, BloqMethod::M3);
    // Fall back to full FD when TTE endpoints are present: the analytic non-M3
    // path is Gaussian-only and would silently zero hazard-parameter gradients.
    #[cfg(feature = "survival")]
    let fd_all = fd_all || !model.endpoints.is_empty();

    if fd_all {
        // M3 / TTE path: forward-FD of obs_nll_subject_into_iov.
        let nll_base =
            obs_nll_subject_into_iov(model, subject, theta, sigma_values, eta, kappas, pk_scratch);
        let mut grad = vec![0.0f64; n];
        let h = 1e-5;
        for i in 0..n {
            if lower[i] == upper[i] {
                continue;
            }
            if i < n_theta {
                let mut theta_p = theta.to_vec();
                let delta = h * (1.0 + theta[i].abs());
                theta_p[i] += delta;
                let nll_p = obs_nll_subject_into_iov(
                    model,
                    subject,
                    &theta_p,
                    sigma_values,
                    eta,
                    kappas,
                    pk_scratch,
                );
                let raw = (nll_p - nll_base) / delta;
                grad[i] = if theta_packs_log_mask[i] {
                    theta[i] * raw
                } else {
                    raw
                };
            } else {
                let k = i - n_theta;
                let mut sigma_p = sigma_values.to_vec();
                let delta = h * (1.0 + sigma_values[k].abs());
                sigma_p[k] += delta;
                let nll_p = obs_nll_subject_into_iov(
                    model, subject, theta, &sigma_p, eta, kappas, pk_scratch,
                );
                grad[i] = sigma_values[k] * (nll_p - nll_base) / delta;
            }
        }
        return (nll_base, grad);
    }

    // Non-M3 path: continuous per-occasion-aware base predictions (issue #104).
    let n_obs = subject.observations.len();
    let preds = crate::pk::predict_iov(model, subject, theta, eta, kappas);
    // FREM covariate rows use EPSCOV, not the PK residual error (see
    // build_frem_r_override); their variance is η-independent so dvar_df = 0.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma_values,
    );
    // IIV on residual error (#409): per-subject `exp(2·η_ruv)` scale on the PK
    // residual variance (FREM rows excluded). η_ruv is a BSV eta, indexed into
    // `eta`.  See the non-IOV `obs_nll_subject_grad` for the score-consistency
    // argument behind scaling V, dV/df, and dV/dlogσ together.
    let ruv_scale = model.residual_var_scale(eta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);

    let mut nll_base = 0.0_f64;
    let mut all_preds_base = vec![0.0f64; n_obs];
    let mut residuals = vec![0.0f64; n_obs];
    let mut variances = vec![0.0f64; n_obs];
    let mut d_nll_d_f = vec![0.0f64; n_obs];
    let mut obs_var_scale = vec![1.0f64; n_obs];

    for j in 0..n_obs {
        let cmt = err_keys[j];
        let f = preds[j].max(1e-12);
        let frem_vj = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
        let s = if frem_vj.is_some() { 1.0 } else { ruv_scale };
        obs_var_scale[j] = s;
        let v = match frem_vj {
            Some(vv) => vv.max(1e-12),
            None => (model.residual_variance_at(cmt, f, sigma_values) * s).max(1e-12),
        };
        let resid = subject.observations[j] - f;
        nll_base += 0.5 * (v.ln() + resid * resid / v);
        all_preds_base[j] = f;
        residuals[j] = resid;
        variances[j] = v;
        let dv_df = if frem_vj.is_some() {
            0.0
        } else {
            model.error_spec.dvar_df(cmt, f, sigma_values) * s
        };
        d_nll_d_f[j] = -resid / v + 0.5 * dv_df * (1.0 / v - resid * resid / (v * v));
    }

    let mut grad = vec![0.0f64; n];

    // Theta gradient: forward-FD of the continuous prediction (one perturbed
    // prediction per theta; κ affects later occasions via carryover so the
    // sensitivity is captured across all rows).
    let h_fd = 1e-5;
    for i in 0..n_theta {
        if lower[i] == upper[i] {
            continue;
        }
        let delta = h_fd * (1.0 + theta[i].abs());
        let mut theta_p = theta.to_vec();
        theta_p[i] += delta;
        let preds_p = crate::pk::predict_iov(model, subject, &theta_p, eta, kappas);
        let mut d_obs_nll = 0.0_f64;
        for j in 0..n_obs {
            d_obs_nll += d_nll_d_f[j] * (preds_p[j] - all_preds_base[j]) / delta;
        }
        grad[i] = if theta_packs_log_mask[i] {
            theta[i] * d_obs_nll
        } else {
            d_obs_nll
        };
    }

    // Sigma gradient: analytical — same formula as non-IOV, summed over all obs.
    for k in 0..n_sigma {
        let i = n_theta + k;
        if lower[i] == upper[i] {
            continue;
        }
        let g: f64 = (0..n_obs)
            .map(|j| {
                let f = all_preds_base[j];
                let v = variances[j];
                let resid = residuals[j];
                // d(v_j)/d(log sigma_k); zero unless sigma_k enters obs j's
                // endpoint, so per-CMT each sigma picks up only its own
                // endpoint's observations.
                let ratio = model
                    .error_spec
                    .dvar_dlogsigma(err_keys[j], k, f, sigma_values)
                    * obs_var_scale[j];
                0.5 * ratio * (1.0 / v - resid * resid / (v * v))
            })
            .sum();
        grad[i] = g;
    }

    (nll_base, grad)
}

// ---------------------------------------------------------------------------
// Gradient of conditional observation NLL w.r.t. log(theta) and log(sigma)
// ---------------------------------------------------------------------------

/// Serially fold per-subject `(nll, grad)` pairs, already collected in subject
/// order, into a single `(nll, grad)` total. Deterministic regardless of the
/// rayon worker count that produced `per_subj` (#703): a parallel `reduce`
/// would combine partials along thread-count-dependent boundaries, and f64
/// addition is non-associative.
fn fold_nll_grad(per_subj: Vec<(f64, Vec<f64>)>, n: usize) -> (f64, Vec<f64>) {
    per_subj
        .into_iter()
        .fold((0.0, vec![0.0f64; n]), |(nll_a, mut ga), (nll_b, gb)| {
            for (a, b) in ga.iter_mut().zip(gb.iter()) {
                *a += b;
            }
            (nll_a + nll_b, ga)
        })
}

/// Lightweight M-step: run NLopt SLSQP for a few iterations in packed
/// space, warm-started from the current packed theta / log-sigma.
///
/// `theta_packs_log_mask[i]` selects per-theta packing: log when true,
/// identity when false. Sigma is always log-packed (sigma > 0 by
/// construction). See the run_saem comment on `theta_packs_log_mask` for
/// motivation — without per-theta packing, any theta with `theta_lower < 0`
/// got pinned at 1e-10 and could never be estimated.
/// Mixture context for the θ/σ M-step (#985): each subject's drawn class plus
/// the per-class held σ overrides.
///
/// The class drives the `MIXNUM` guard (so a class-switched typical value is
/// estimated from its own members) *and* the σ vector each subject is scored
/// under: a `sigma(k)` override is held at its init, so a class-`k` subject's
/// residual variance does not depend on the free base σ at all. Scoring it under
/// the base σ anyway would drag the free base estimate toward the override — the
/// E-step samples η under `class_sigma(c)` while the M-step would optimise a
/// different objective (#987 review).
#[derive(Clone, Copy)]
pub(crate) struct MixMstep<'a> {
    /// Per-subject drawn class (0-based).
    pub classes: &'a [usize],
    /// Per-class held σ overrides, `[class][(sigma_index, held_value)]`.
    pub class_sigma_over: &'a [Vec<(usize, f64)>],
}

/// Substitute a class's held σ overrides into the base σ vector. Returns `None`
/// when the class has no override (the caller then uses the base vector as-is,
/// avoiding a per-subject allocation on the common path).
fn class_sigma_subst(sigma_values: &[f64], over: &[(usize, f64)]) -> Option<Vec<f64>> {
    if over.is_empty() {
        return None;
    }
    let mut v = sigma_values.to_vec();
    for &(s, val) in over {
        if s < v.len() {
            v[s] = val;
        }
    }
    Some(v)
}

#[allow(clippy::too_many_arguments)]
fn theta_sigma_mstep_light(
    model: &CompiledModel,
    population: &Population,
    etas: &[Vec<f64>],
    kappas_opt: Option<&[Vec<Vec<f64>>]>,
    log_theta_init: &[f64],
    log_sigma_init: &[f64],
    log_theta_lower: &[f64],
    log_theta_upper: &[f64],
    log_sigma_lower: &[f64],
    log_sigma_upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    maxiter: u32,
    scale_params: bool,
    theta_packs_log_mask: &[bool],
    // Mixture (#985): per-subject drawn class + held σ overrides. When `Some`,
    // every per-subject observation-likelihood evaluation runs under that
    // subject's `MIXNUM` guard and its class's σ, so a class-switched typical
    // value is estimated from its own class members and a held `sigma(k)`
    // override does not bias the free base σ.
    mix_mstep: Option<MixMstep<'_>>,
) -> (Vec<f64>, Vec<f64>) {
    let n = n_theta + n_sigma;

    let mut x: Vec<f64> = Vec::with_capacity(n);
    x.extend_from_slice(log_theta_init);
    x.extend_from_slice(log_sigma_init);

    let mut lower: Vec<f64> = Vec::with_capacity(n);
    lower.extend_from_slice(log_theta_lower);
    lower.extend_from_slice(log_sigma_lower);
    let mut upper: Vec<f64> = Vec::with_capacity(n);
    upper.extend_from_slice(log_theta_upper);
    upper.extend_from_slice(log_sigma_upper);

    for i in 0..n {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }

    // Unpack a slice of packed theta values into natural-scale theta.
    // Closure (not local fn) so it captures `theta_packs_log_mask`.
    let unpack_thetas = |packed: &[f64]| -> Vec<f64> {
        (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    packed[i].exp()
                } else {
                    packed[i]
                }
            })
            .collect()
    };

    // Objective operating on the unscaled packed parameters.
    //
    // Gradient strategy: single rayon pass over subjects, each computing its
    // own partial gradient via `obs_nll_subject_grad` (analytical sigma,
    // FD-of-predictions for theta). This replaces the old per-parameter
    // forward-FD of `obs_nll_sum` which launched `n_dim` rayon jobs
    // sequentially. Key improvements:
    //  • Sigma gradient is analytical — no extra predict calls per sigma dim.
    //  • Single rayon launch instead of n_dim sequential launches.
    //  • Better cache locality: one subject's data stays in cache while
    //    iterating over all its theta perturbations.
    //  • Pinned dims (lower == upper) are skipped per-subject, saving the
    //    predict calls entirely (same as the old FD guard).
    let obj = |xv: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta]);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();

        if let Some(g) = grad {
            use rayon::prelude::*;
            // Collect in subject order, then fold serially (#703): a parallel
            // `reduce` combines partial (nll, grad) pairs along thread-count-
            // dependent boundaries, and f64 addition is non-associative.
            let (val, grad_vec) = if let Some(kappas) = kappas_opt {
                let per_subj: Vec<(f64, Vec<f64>)> = population
                    .subjects
                    .par_iter()
                    .zip(etas.par_iter())
                    .zip(kappas.par_iter())
                    .enumerate()
                    .map_init(
                        EventPkParams::default,
                        |scratch, (i, ((subject, eta), kaps))| {
                            let cls = mix_mstep.map(|m| m.classes[i]);
                            let _g = cls.map(|c| {
                                crate::parser::model_parser::MixtureClassGuard::enter(c + 1)
                            });
                            let over: &[(usize, f64)] = match (mix_mstep, cls) {
                                (Some(m), Some(c)) => &m.class_sigma_over[c],
                                _ => &[],
                            };
                            let sub_sg = class_sigma_subst(&sg, over);
                            let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(&sg);
                            let (nll, mut grad) = obs_nll_subject_grad_iov(
                                model,
                                subject,
                                &th,
                                sg_i,
                                eta,
                                kaps,
                                &theta_packs_log_mask,
                                &lower,
                                &upper,
                                n_theta,
                                n_sigma,
                                scratch,
                            );
                            // A held σ override carries no information about the
                            // free base σ — zero that subject's contribution.
                            for &(sidx, _) in over {
                                if n_theta + sidx < grad.len() {
                                    grad[n_theta + sidx] = 0.0;
                                }
                            }
                            (nll, grad)
                        },
                    )
                    .collect();
                fold_nll_grad(per_subj, n)
            } else {
                let per_subj: Vec<(f64, Vec<f64>)> = population
                    .subjects
                    .par_iter()
                    .zip(etas.par_iter())
                    .enumerate()
                    .map_init(EventPkParams::default, |scratch, (i, (subject, eta))| {
                        let cls = mix_mstep.map(|m| m.classes[i]);
                        let _g = cls
                            .map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                        let over: &[(usize, f64)] = match (mix_mstep, cls) {
                            (Some(m), Some(c)) => &m.class_sigma_over[c],
                            _ => &[],
                        };
                        let sub_sg = class_sigma_subst(&sg, over);
                        let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(&sg);
                        let (nll, mut grad) = obs_nll_subject_grad(
                            model,
                            subject,
                            &th,
                            sg_i,
                            eta,
                            &theta_packs_log_mask,
                            &lower,
                            &upper,
                            n_theta,
                            n_sigma,
                            scratch,
                        );
                        for &(sidx, _) in over {
                            if n_theta + sidx < grad.len() {
                                grad[n_theta + sidx] = 0.0;
                            }
                        }
                        (nll, grad)
                    })
                    .collect();
                fold_nll_grad(per_subj, n)
            };
            for (gi, &gv) in g.iter_mut().zip(grad_vec.iter()) {
                *gi = if gv.is_finite() { gv } else { 0.0 };
            }
            if val.is_finite() {
                val
            } else {
                1e20
            }
        } else {
            let val = match (mix_mstep, kappas_opt) {
                (Some(mx), Some(kappas)) => {
                    obs_nll_sum_iov_mix(model, population, &th, &sg, etas, kappas, mx)
                }
                (Some(mx), None) => obs_nll_sum_mix(model, population, &th, &sg, etas, mx),
                (None, Some(kappas)) => obs_nll_sum_iov(model, population, &th, &sg, etas, kappas),
                (None, None) => obs_nll_sum(model, population, &th, &sg, etas),
            };
            if val.is_finite() {
                val
            } else {
                1e20
            }
        }
    };

    // Compute per-element scale factors from the initial point.
    let scale: Vec<f64> = if scale_params {
        compute_scale(&x)
    } else {
        vec![1.0; n]
    };

    // Scaled starting point and bounds: xs[i] = x[i] / scale[i].
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();
    let lower_s: Vec<f64> = (0..n).map(|i| lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| upper[i] / scale[i]).collect();

    // Wrapper objective: receives scaled xs, unscales before evaluating obj,
    // then scales the gradient back: d(OFV)/d(xs[i]) = d(OFV)/d(x[i]) * scale[i].
    let obj_s = |xv_s: &[f64], grad: Option<&mut [f64]>, data: &mut ()| -> f64 {
        let xv: Vec<f64> = (0..n).map(|i| xv_s[i] * scale[i]).collect();
        if let Some(g) = grad {
            let mut g_raw = vec![0.0_f64; n];
            let val = obj(&xv, Some(&mut g_raw), data);
            for i in 0..n {
                g[i] = g_raw[i] * scale[i];
            }
            val
        } else {
            obj(&xv, None, data)
        }
    };

    // See `MSTEP_NLOPT_ALGORITHM` for rationale (BOBYQA vs SLSQP).
    let mut opt = nlopt::Nlopt::new(MSTEP_NLOPT_ALGORITHM, n, obj_s, nlopt::Target::Minimize, ());
    opt.set_lower_bounds(&lower_s).unwrap();
    opt.set_upper_bounds(&upper_s).unwrap();
    opt.set_maxeval(maxiter * (n as u32 + 1)).unwrap();
    opt.set_ftol_rel(1e-4).unwrap();

    match opt.optimize(&mut xs) {
        Ok(_) | Err(_) => {}
    }

    // Unscale back to log-space.
    let x_final: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    let log_theta_new = x_final[..n_theta].to_vec();
    let log_sigma_new = x_final[n_theta..].to_vec();
    (log_theta_new, log_sigma_new)
}

/// Gradient of `obs_nll` w.r.t. the SAEM packed parameter vector
/// `[log_theta_0 … log_theta_{P-1} | log_sigma_0 … log_sigma_{Q-1}]`
/// for a single subject with ETAs held fixed.
///
/// For non-M3 models:
/// - Sigma: analytical from the residual-variance formula (no extra predict call).
/// - Theta: forward-FD of `compute_predictions_with_tv_into` + chain rule through
///   obs_nll (one extra predict call per non-pinned theta, not one full-subject
///   NLL call).
///
/// For M3 models (complex Mills-ratio sigma gradient): forward-FD of
/// `obs_nll_subject_into` for all parameters.
///
/// `lower`/`upper` are the packed-space bounds used to detect pinned dimensions
/// (`lower[i] == upper[i]`); pinned dimensions contribute 0 to the gradient and
/// skip their FD call.
#[allow(clippy::too_many_arguments)]
fn obs_nll_subject_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    theta_packs_log_mask: &[bool],
    lower: &[f64],
    upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    pk_scratch: &mut EventPkParams,
) -> (f64, Vec<f64>) {
    let n = n_theta + n_sigma;
    let fd_all =
        matches!(model.bloq_method, BloqMethod::M3) || !model.residual_correlations.is_empty();
    // Fall back to the full-FD path when TTE endpoints are present: the analytic
    // non-M3 path is Gaussian-only and would silently zero hazard-parameter gradients.
    #[cfg(feature = "survival")]
    let fd_all = fd_all || !model.endpoints.is_empty();

    if fd_all {
        // M3 / TTE / dense residual-covariance path: forward-FD of
        // obs_nll_subject_into for all parameters. Predictions are σ-independent,
        // so solve the model once and reuse the base predictions across every σ
        // perturbation — only θ perturbations need a fresh solve (#557).
        let preds_base =
            crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);
        let nll_base =
            obs_nll_subject_from_preds(model, subject, &preds_base, theta, sigma_values, eta);
        let mut grad = vec![0.0f64; n];
        let h = 1e-5;
        for i in 0..n {
            if lower[i] == upper[i] {
                continue;
            }
            if i < n_theta {
                let mut theta_p = theta.to_vec();
                let delta = h * (1.0 + theta[i].abs());
                theta_p[i] += delta;
                let nll_p =
                    obs_nll_subject_into(model, subject, &theta_p, sigma_values, eta, pk_scratch);
                let raw = (nll_p - nll_base) / delta;
                grad[i] = if theta_packs_log_mask[i] {
                    theta[i] * raw
                } else {
                    raw
                };
            } else {
                let k = i - n_theta;
                let mut sigma_p = sigma_values.to_vec();
                let delta = h * (1.0 + sigma_values[k].abs());
                sigma_p[k] += delta;
                let nll_p =
                    obs_nll_subject_from_preds(model, subject, &preds_base, theta, &sigma_p, eta);
                // log-packing for sigma: d/d(log_sigma_k) = sigma_k * d/d(sigma_k)
                grad[i] = sigma_values[k] * (nll_p - nll_base) / delta;
            }
        }
        return (nll_base, grad);
    }

    // Non-M3 path.
    let preds_base =
        crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);

    let mut nll_base = 0.0f64;
    let n_obs = subject.observations.len();

    // FREM covariate rows use EPSCOV, not the PK residual error (see
    // build_frem_r_override); their variance is η-independent so dvar_df = 0.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma_values,
    );

    // IIV on residual error (#409): per-subject scale on the PK residual
    // variance (`exp(2·η_ruv)`). FREM covariate rows are not scaled, so we hold
    // a per-obs scale and apply it consistently to V, dV/df, and dV/dlogσ so the
    // analytical score stays exact.
    let ruv_scale = model.residual_var_scale(eta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);

    // per-obs residual, variance, d(obs_nll)/d(f_j), and the variance scale used.
    let mut residuals = vec![0.0f64; n_obs];
    let mut variances = vec![0.0f64; n_obs];
    let mut d_nll_d_f = vec![0.0f64; n_obs];
    let mut obs_var_scale = vec![1.0f64; n_obs];

    for j in 0..n_obs {
        let cmt = err_keys[j];
        let f = preds_base[j].max(1e-12);
        let frem_vj = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
        let s = if frem_vj.is_some() { 1.0 } else { ruv_scale };
        obs_var_scale[j] = s;
        let v = match frem_vj {
            Some(vv) => vv.max(1e-12),
            None => (model.residual_variance_at(cmt, f, sigma_values) * s).max(1e-12),
        };
        let resid = subject.observations[j] - f;
        nll_base += 0.5 * (v.ln() + resid * resid / v);
        residuals[j] = resid;
        variances[j] = v;
        // d(obs_nll_j)/d(f_j) = -resid/V + 0.5 * (dV/df) * (1/V - resid²/V²)
        let dv_df = if frem_vj.is_some() {
            0.0
        } else {
            model.error_spec.dvar_df(cmt, f, sigma_values) * s
        };
        d_nll_d_f[j] = -resid / v + 0.5 * dv_df * (1.0 / v - resid * resid / (v * v));
    }

    let mut grad = vec![0.0f64; n];

    // Theta gradient: forward-FD of predictions, chain rule through obs_nll.
    let h_fd = 1e-5;
    for i in 0..n_theta {
        if lower[i] == upper[i] {
            continue;
        }
        let delta = h_fd * (1.0 + theta[i].abs());
        let mut theta_p = theta.to_vec();
        theta_p[i] += delta;
        let preds_p =
            crate::pk::compute_predictions_with_tv_into(model, subject, &theta_p, eta, pk_scratch);
        // Difference on raw predictions — do NOT clip before differencing.
        // Clipping both pp and pb at 1e-12 before subtracting would produce a
        // zero difference whenever pb < 1e-12, silently zeroing the gradient.
        let d_obs_nll: f64 = d_nll_d_f
            .iter()
            .zip(preds_p.iter().zip(preds_base.iter()))
            .map(|(&dl, (&pp, &pb))| dl * (pp - pb) / delta)
            .sum();
        grad[i] = if theta_packs_log_mask[i] {
            theta[i] * d_obs_nll
        } else {
            d_obs_nll
        };
    }

    // Sigma gradient: analytical.
    // d(obs_nll)/d(log_sigma_k) = Σ_j 0.5 * ratio_jk * (1/V_j - resid_j²/V_j²)
    // where ratio_jk = sigma_k * dV_j/d_sigma_k.
    for k in 0..n_sigma {
        let i = n_theta + k;
        if lower[i] == upper[i] {
            continue;
        }
        let g: f64 = (0..n_obs)
            .map(|j| {
                let f = preds_base[j].max(1e-12);
                let v = variances[j];
                let resid = residuals[j];
                // ratio = d(V_j)/d(log sigma_k); zero unless sigma_k enters
                // obs j's endpoint (so per-CMT each sigma sums only over its
                // own endpoint's observations).
                let ratio = model
                    .error_spec
                    .dvar_dlogsigma(err_keys[j], k, f, sigma_values)
                    * obs_var_scale[j];
                0.5 * ratio * (1.0 / v - resid * resid / (v * v))
            })
            .sum();
        grad[i] = g;
    }

    (nll_base, grad)
}

/// Sum of observation log-likelihoods with ETAs held fixed.
///
/// Under M3, censored rows contribute the matching normal-tail likelihood
/// instead of the Gaussian residual term. Without this branch, the SAEM M-step
/// would optimize θ/σ as if censored observations were exact Gaussians at the limit,
/// producing silently-biased population estimates.
///
/// Uses rayon's `map_init` so each worker thread allocates one
/// `EventPkParams` scratch on first use and reuses it across every
/// subject the worker handles. With NLopt's central-FD gradient
/// hitting `obs_nll_sum` `1 + 2·n_dim` times per M-step, this cuts
/// per-call `Vec<PkParams>` churn to near-zero on TV-cov data.
fn obs_nll_sum(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
) -> f64 {
    use rayon::prelude::*;
    // Collect in subject order and sum serially so the objective does not
    // depend on the rayon worker count (f64 addition is non-associative and a
    // parallel `.sum()` splits by thread count) — #703.
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            obs_nll_subject_into(model, subject, theta, sigma_values, &etas[i], scratch)
        })
        .collect();
    per_subj.iter().sum()
}

/// IOV variant of `obs_nll_sum`: per-occasion predictions using `[eta, kappa_k]`.
fn obs_nll_sum_iov(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    kappas: &[Vec<Vec<f64>>],
) -> f64 {
    use rayon::prelude::*;
    // Deterministic reduction (collect in subject order, fold serially): a
    // parallel `.sum()` would make the objective depend on the rayon worker
    // count — #703.
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            obs_nll_subject_into_iov(
                model,
                subject,
                theta,
                sigma_values,
                &etas[i],
                &kappas[i],
                scratch,
            )
        })
        .collect();
    per_subj.iter().sum()
}

/// Mixture (#985) variant of [`obs_nll_sum`]: each subject's observation NLL is
/// evaluated under its drawn class's `MIXNUM` guard — so the class-switched
/// typical values (`if MIXNUM == k …`) are seen — and under that class's σ, so a
/// held `sigma(k)` override is honoured rather than replaced by the free base σ.
fn obs_nll_sum_mix(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    mix: MixMstep<'_>,
) -> f64 {
    use rayon::prelude::*;
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            let c = mix.classes[i];
            let _g = crate::parser::model_parser::MixtureClassGuard::enter(c + 1);
            let sub_sg = class_sigma_subst(sigma_values, &mix.class_sigma_over[c]);
            let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(sigma_values);
            obs_nll_subject_into(model, subject, theta, sg_i, &etas[i], scratch)
        })
        .collect();
    per_subj.iter().sum()
}

/// Mixture (#985) + IOV variant of [`obs_nll_sum_iov`], class-guarded per subject.
fn obs_nll_sum_iov_mix(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
    kappas: &[Vec<Vec<f64>>],
    mix: MixMstep<'_>,
) -> f64 {
    use rayon::prelude::*;
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            let c = mix.classes[i];
            let _g = crate::parser::model_parser::MixtureClassGuard::enter(c + 1);
            let sub_sg = class_sigma_subst(sigma_values, &mix.class_sigma_over[c]);
            let sg_i: &[f64] = sub_sg.as_deref().unwrap_or(sigma_values);
            obs_nll_subject_into_iov(model, subject, theta, sg_i, &etas[i], &kappas[i], scratch)
        })
        .collect();
    per_subj.iter().sum()
}

/// True when a free (non-`FIX`) additive component of a `Combined` endpoint has
/// collapsed onto its optimizer lower bound.
///
/// Sigma is optimized in log space with a lower bound of `exp(-8) ≈ 3.35e-4`
/// (see `parameterization.rs`) and is carried here on the standard-deviation
/// scale. `SIGMA_FLOOR_NEAR = 1e-3` is the detection band just above that hard
/// bound: a value at or below it means the additive term pinned to the floor
/// rather than identifying a genuine non-zero additive error.
fn combined_additive_sigma_at_floor(model: &CompiledModel, params: &ModelParameters) -> bool {
    const SIGMA_FLOOR_NEAR: f64 = 1.0e-3;
    model
        .error_spec
        .combined_additive_sigma_indices()
        .into_iter()
        .any(|idx| {
            !params.sigma_fixed.get(idx).copied().unwrap_or(false)
                && params
                    .sigma
                    .values
                    .get(idx)
                    .copied()
                    .unwrap_or(f64::INFINITY)
                    <= SIGMA_FLOOR_NEAR
        })
}

/// Per-σ growth ceiling (log scale) for iiv_on_ruv SAEM runs (issue #895).
///
/// The IIV-on-RUV parameterization writes the residual variance as
/// `σ²·exp(2·η_RUV)`, so σ and ω_RUV trade off along a ridge; a poorly-mixing
/// E-step can let the M-step ride σ up to its e⁵ ceiling. This returns, per σ,
/// `Some(cap)` = `log σ₀ + SAEM_RUV_SIGMA_LN_GROWTH` for each *free* RUV-scaled
/// residual σ **whose growth cap is stricter than its own NLopt upper bound**,
/// and `None` for any σ that must not carry a growth cap: every σ when the model
/// has no `iiv_on_ruv` (`has_ruv_eta == false`), a FIXed σ, the FREM covariate σ
/// (EPSCOV — always FIX and independent of the RUV scaling), or a σ whose
/// user-set upper bound is already tighter than the growth cap (the NLopt bound
/// governs there, so a σ converging to its own bound must not be mis-flagged as a
/// bound RUV growth cap — #903 review). The cap is enforced as a post-M-step
/// clamp, never as an NLopt bound, so a well-posed fit whose σ stays below the cap
/// is unaffected.
fn compute_ruv_sigma_caps(
    has_ruv_eta: bool,
    frem_cov_sigma: Option<usize>,
    log_sigma_init: &[f64],
    log_sigma_upper: &[f64],
    sigma_fixed: &[bool],
) -> Vec<Option<f64>> {
    let n = log_sigma_init.len();
    if !has_ruv_eta {
        return vec![None; n];
    }
    (0..n)
        .map(|i| {
            if sigma_fixed.get(i).copied().unwrap_or(false) || Some(i) == frem_cov_sigma {
                return None;
            }
            let upper = log_sigma_upper.get(i).copied().unwrap_or(f64::INFINITY);
            let growth_cap = log_sigma_init[i] + SAEM_RUV_SIGMA_LN_GROWTH;
            // If the user's own upper bound is at least as tight, NLopt already
            // enforces it and our clamp/warning would just misattribute that bound
            // to the RUV safeguard — so carry no growth cap here.
            if growth_cap < upper {
                Some(growth_cap)
            } else {
                None
            }
        })
        .collect()
}

/// Whether the `iiv_on_ruv` η may be re-centered while exactly preserving each
/// subject's residual variance (issue #904 correctness gate).
///
/// Re-centering shifts η_RUV by `−mean`, multiplying the *whole* residual variance
/// `R = Σ σ_c²` by `exp(−2·mean)`, and compensates by scaling the free residual σ
/// by `exp(mean)`. The compensation is exact only if *every* non-FREM residual σ
/// component absorbs the shift, i.e. is free. If any RUV-scaled σ component is
/// FIXed (e.g. a fixed additive term of a combined error), scaling only the free
/// component leaves the fixed part uncompensated and silently perturbs `R`, so
/// re-centering must be skipped (the σ/ω_RUV growth caps then act as the
/// backstop; a fixed component also partially anchors the ridge). The FREM EPSCOV
/// (always FIX and not on a real-observation row) is exempt.
fn ruv_recenter_allowed(
    has_ruv_eta: bool,
    frem_cov_sigma: Option<usize>,
    sigma_fixed: &[bool],
) -> bool {
    has_ruv_eta && (0..sigma_fixed.len()).all(|i| Some(i) == frem_cov_sigma || !sigma_fixed[i])
}

/// Re-center the `iiv_on_ruv` η to zero mean, absorbing the shift into the
/// RUV-scaled residual σ (issue #904). Returns the mean that was removed.
///
/// `Y = f + EPS·exp(η_RUV)` has no typical-value θ, so — unlike a mu-referenced
/// structural η — η_RUV's mean is otherwise never absorbed and drifts along the
/// degenerate direction (`σ²·exp(2η)` is invariant to η→η−c, σ→σ·exp(c)),
/// polluting `ω_RUV = mean(η²)` with a spurious `mean²`. Shifting η_RUV by
/// `−mean` and scaling every absorbing σ by `exp(mean)` leaves each subject's
/// residual variance exactly unchanged while restoring `E[η_RUV] = 0`. `σ` is the
/// residual-scale typical value that plays the role of the structural TVP here.
///
/// `absorb_sigma[k]` marks the free, non-FREM, RUV-scaled σ that may take the
/// shift; when none absorb (every RUV σ FIXed) this is a no-op and the mean is
/// left in η_RUV, since a FIXed σ already pins the mean (no degeneracy).
fn recenter_ruv_eta(
    etas: &mut [Vec<f64>],
    kr: usize,
    log_sigma: &mut [f64],
    sigma_vals: &mut [f64],
    absorb_sigma: &[bool],
) -> f64 {
    let n = etas.len();
    if n == 0 || !absorb_sigma.iter().any(|&a| a) {
        return 0.0;
    }
    let mean = etas.iter().map(|e| e[kr]).sum::<f64>() / n as f64;
    for e in etas.iter_mut() {
        e[kr] -= mean;
    }
    for k_s in 0..log_sigma.len() {
        if absorb_sigma.get(k_s).copied().unwrap_or(false) {
            log_sigma[k_s] += mean;
            sigma_vals[k_s] = log_sigma[k_s].exp();
        }
    }
    mean
}

/// Log-variance ceiling for the `iiv_on_ruv` Ω diagonal in a SAEM run (#895).
///
/// Returns `Some(cap)` = `log(ω₀) + SAEM_RUV_OMEGA_LN_GROWTH` for the RUV eta's
/// variance when the model has an `iiv_on_ruv` eta whose Ω is free and in range,
/// else `None` (no `iiv_on_ruv`, out of range, or a FIXed RUV Ω). The cap is
/// enforced as a correlation-preserving rescale after the Ω M-step, so a
/// well-posed fit whose ω_RUV stays below it is unaffected.
fn compute_ruv_omega_cap(
    residual_error_eta: Option<usize>,
    n_eta: usize,
    omega_init: &DMatrix<f64>,
    omega_fixed: &[bool],
) -> Option<f64> {
    let k = residual_error_eta?;
    if k >= n_eta || omega_fixed.get(k).copied().unwrap_or(false) {
        return None;
    }
    let v0 = omega_init[(k, k)];
    if !(v0 > 0.0) {
        return None;
    }
    Some(v0.ln() + SAEM_RUV_OMEGA_LN_GROWTH)
}

/// Cap the `iiv_on_ruv` Ω diagonal at `exp(log_cap)` in place, rescaling the RUV
/// row/column covariances by `√(v_cap/v_old)` so every correlation with the RUV
/// eta is preserved and the matrix stays positive-definite (#895). A no-op when
/// the diagonal is already at or below the cap. Returns `true` when it clamped.
///
/// A FIXed off-diagonal partner `j` (`omega_fixed[j]`) is left untouched: its
/// covariance with the RUV eta is a user-declared constant that the Ω M-step
/// restores verbatim each iteration, so rescaling it would silently mutate a
/// FIXed entry (#903 review). Skipping it can only *lower* the correlation the
/// cap preserves, never break positive-definiteness (the diagonal still shrinks).
fn apply_ruv_omega_cap(
    omega_mat: &mut DMatrix<f64>,
    k: usize,
    log_cap: f64,
    omega_fixed: &[bool],
) -> bool {
    let v_old = omega_mat[(k, k)];
    let v_cap = log_cap.exp();
    if !(v_old > v_cap) {
        return false;
    }
    let s = (v_cap / v_old).sqrt();
    let n = omega_mat.nrows();
    for j in 0..n {
        if j != k && !omega_fixed.get(j).copied().unwrap_or(false) {
            omega_mat[(k, j)] *= s;
            omega_mat[(j, k)] *= s;
        }
    }
    omega_mat[(k, k)] = v_cap;
    true
}

/// Re-anchor the per-σ iiv_on_ruv growth caps to the data-informed σ reached by
/// the end of the exploration phase (#903 review). Each cap is loosened (never
/// tightened) to `max(existing, min(log σ_now + growth, upper))`, so a well-posed
/// fit started from a σ guess far below the truth is not spuriously clamped and
/// falsely flagged, while a genuine post-exploration runaway is still bounded.
fn reanchor_ruv_sigma_caps(caps: &mut [Option<f64>], log_sigma: &[f64], log_sigma_upper: &[f64]) {
    for (i, cap) in caps.iter_mut().enumerate() {
        if let Some(c) = cap {
            let upper = log_sigma_upper.get(i).copied().unwrap_or(f64::INFINITY);
            let settled = (log_sigma[i] + SAEM_RUV_SIGMA_LN_GROWTH).min(upper);
            *c = c.max(settled);
        }
    }
}

/// Re-anchor the iiv_on_ruv Ω growth cap to the ω_RUV variance reached by the end
/// of exploration (#903 review), loosening only. `None` (no cap) stays `None`;
/// a non-positive variance leaves the cap unchanged.
fn reanchor_ruv_omega_cap(cap: Option<f64>, omega_ruv_var: f64) -> Option<f64> {
    cap.map(|c| {
        if omega_ruv_var > 0.0 {
            c.max(omega_ruv_var.ln() + SAEM_RUV_OMEGA_LN_GROWTH)
        } else {
            c
        }
    })
}

/// Decide whether SAEM should warn that its E-step never mixed (issue #895).
///
/// `cum_acc` / `cum_prop` are the run-cumulative combined (block + componentwise)
/// MH accept / proposal counts over the post-burn-in iterations. Returns the
/// warning string when the acceptance rate is below `SAEM_MH_STUCK_ACCEPT` (and
/// at least one proposal was made), else `None`. A near-zero rate means the
/// sampled ETAs barely moved, so the M-step ran on degenerate sufficient
/// statistics and Ω/σ are unreliable.
fn saem_mixing_warning(cum_acc: u64, cum_prop: u64) -> Option<String> {
    if cum_prop == 0 {
        return None;
    }
    let rate = cum_acc as f64 / cum_prop as f64;
    if rate < SAEM_MH_STUCK_ACCEPT {
        Some(format!(
            "SAEM Metropolis-Hastings acceptance was {:.2}% over the post-burn-in \
             iterations — the E-step is not mixing, so Ω/σ estimates are unreliable. \
             Check for extreme Ω-diagonal scale differences (e.g. FREM covariate ETAs) \
             or a mis-scaled initial Ω.",
            rate * 100.0
        ))
    } else {
        None
    }
}

/// Build (theta_idx, eta_idx) pairs for log-transformed mu-references only.
///
/// Only `log_transformed = true` mu-refs (patterns `THETA*exp(ETA)` and
/// `exp(log(THETA)+ETA)`) participate in the gradient-step M-step.  For these
/// the chain rule gives `d/d_log(theta) = -Σᵢ d/d_eta`, which matches the
/// update applied in the SAEM loop.  Additive mu-refs (`THETA + ETA`,
/// `log_transformed = false`) require the extra factor of `theta` from the
/// log-space chain rule and are deliberately excluded — they fall through to
/// the regular NLopt M-step.
pub(crate) fn get_mu_ref_pairs(model: &CompiledModel) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if !mu_ref.log_transformed {
                continue;
            }
            if let Some(theta_idx) = model
                .theta_names
                .iter()
                .position(|n| n == &mu_ref.theta_name)
            {
                pairs.push((theta_idx, eta_idx));
            }
        }
    }
    pairs
}

/// A class-aware (`MIXNUM`-switched) log-mu-ref pair: one eta paired with one
/// anchor theta **per class** (#996).
///
/// A class-shared typical value (`V = TVV * exp(ETA_V)` in a mixture model) is
/// represented as the same theta index repeated `n_classes` times, so both the
/// switched and the shared case run through one update rule — and the
/// all-classes-share-one-theta case reduces exactly to the classical pooled
/// `log θ += γ · mean(η)` shift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MixtureMuRefPair {
    /// Index into `model.eta_names`.
    pub eta_idx: usize,
    /// Anchor theta index per class; `theta_idx[c]` serves class `c` (0-based).
    /// Length is always the mixture's `n_classes`.
    pub theta_idx: Vec<usize>,
}

/// Build the class-aware log-mu-ref pairs for a mixture model (#996).
///
/// Returns empty for a non-mixture model — use [`get_mu_ref_pairs`] there.
/// Each eta contributes at most one pair: the class-aware anchor set detected
/// by the parser (`MixtureSpec::mu_refs`) when the typical value is
/// `MIXNUM`-switched, otherwise the classical single-theta mu-ref broadcast
/// across all classes. Additive (`THETA + ETA`) mu-refs are excluded for the
/// same reason as in [`get_mu_ref_pairs`]: the closed-form shift is only valid
/// on the log scale.
///
/// A theta is claimed by **at most one** pair. Two etas anchored to the same
/// typical value (`CL = TVP*exp(ETA_CL)` and `V = TVP*exp(ETA_V)`) have no
/// well-defined joint closed form — applying both shifts would move that θ twice
/// in one iteration — so the first eta to claim a θ keeps it and any later pair
/// that reuses it is dropped, leaving those θ to the numerical / weighted M-step
/// (#996 review).
pub(crate) fn get_mixture_mu_ref_pairs(model: &CompiledModel) -> Vec<MixtureMuRefPair> {
    let Some(spec) = model.mixture.as_ref() else {
        return Vec::new();
    };
    let k = spec.n_classes;
    let idx_of = |name: &str| model.theta_names.iter().position(|n| n == name);
    let mut out: Vec<MixtureMuRefPair> = Vec::new();
    let mut claimed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut push_pair = |out: &mut Vec<MixtureMuRefPair>, pair: MixtureMuRefPair| {
        if pair.theta_idx.iter().any(|t| claimed.contains(t)) {
            return;
        }
        claimed.extend(pair.theta_idx.iter().copied());
        out.push(pair);
    };
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(m) = spec.mu_refs.iter().find(|m| &m.eta_name == eta_name) {
            if !m.log_transformed || m.theta_names.len() != k {
                continue;
            }
            let Some(theta_idx) = m
                .theta_names
                .iter()
                .map(|n| idx_of(n))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            push_pair(&mut out, MixtureMuRefPair { eta_idx, theta_idx });
        } else if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if !mu_ref.log_transformed {
                continue;
            }
            let Some(t) = idx_of(&mu_ref.theta_name) else {
                continue;
            };
            push_pair(
                &mut out,
                MixtureMuRefPair {
                    eta_idx,
                    theta_idx: vec![t; k],
                },
            );
        }
    }
    out
}

/// Responsibility-weighted per-class mu-ref mean shift (#996).
///
/// For each anchor theta `t`, returns `(Σ_i Σ_{c: θ_c = t} r_ic · η̄_ic) /
/// (Σ_i Σ_{c: θ_c = t} r_ic)` — the EM-optimal `Δ log θ_t` for the complete-data
/// log-likelihood of `log P_i = log θ_{c_i} + η_i` restricted to the members of
/// the classes that theta serves. `resp[i][c]` is subject `i`'s weight in class
/// `c`: a hard 0/1 indicator under SAEM (which draws one class per subject) and
/// the responsibility `PMIX_ic` under IMP/IMPMAP (which importance-samples
/// within every class). `eta_at(i, c)` supplies subject `i`'s η mean under
/// class `c` for the eta this pair carries.
///
/// The result is `None` for any theta no class weight reached — the label-switch
/// case where a class won zero subjects this iteration. The caller holds that
/// θ_k and lets the next iteration move it, rather than dividing by zero or
/// falling back to a pooled mean that would drag the class toward its neighbours.
///
/// When every class maps to the *same* theta the weights sum to one per subject
/// and this collapses to the pooled `mean_i(η_i)` of the classical mu-ref
/// update, in the same accumulation order — which is what makes the degenerate
/// single-theta mixture reproduce the non-mixture path exactly.
pub(crate) fn mixture_mu_ref_means(
    n_theta: usize,
    theta_idx_per_class: &[usize],
    resp: &[Vec<f64>],
    eta_at: impl Fn(usize, usize) -> f64,
) -> Vec<Option<f64>> {
    let mut numer = vec![0.0f64; n_theta];
    let mut denom = vec![0.0f64; n_theta];
    for (i, r_i) in resp.iter().enumerate() {
        for (c, &t) in theta_idx_per_class.iter().enumerate() {
            if t >= n_theta {
                continue;
            }
            let r = r_i.get(c).copied().unwrap_or(0.0);
            if r <= 0.0 {
                continue;
            }
            numer[t] += r * eta_at(i, c);
            denom[t] += r;
        }
    }
    (0..n_theta)
        .map(|t| (denom[t] > 0.0).then(|| numer[t] / denom[t]))
        .collect()
}

/// One-line description of the SAEM E-step sampler kernel, for the startup
/// banner. SAEM's estimation is sampling-based (not gradient-driven), so the
/// banner reports the kernel here instead of a gradient route. HMC is used
/// only when `saem_n_leapfrog > 0` on an analytical PK model (its η-gradient is
/// the analytic Dual2 gradient) — the same gate as [`run_saem`]; this mirrors
/// that condition so the banner reflects what will actually run.
pub(crate) fn saem_sampler_summary(model: &CompiledModel, options: &FitOptions) -> String {
    let n_leapfrog = options.saem_n_leapfrog;
    // HMC is BSV-only (`hmc_step` and the AD NLL/gradient are kappa-unaware), so
    // it is disabled for IOV models (`n_kappa > 0`); those subjects use the MH
    // kernels, whose acceptance targets the IOV conditional p(η | κ, θ, data).
    // IIV on residual error (#409) also disables HMC: the Dual2 gradient kernel
    // carries no `exp(2·η_ruv)` variance-scaling rule, so these models fall back
    // to MH (same gate as [`run_saem`]).
    let using_hmc = n_leapfrog > 0
        && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && model.n_kappa == 0
        && model.residual_error_eta.is_none();
    if using_hmc {
        format!("HMC ({n_leapfrog} leapfrog steps, Dual2 analytic gradients)")
    } else if n_leapfrog > 0 {
        "Metropolis-Hastings random walk \
         (HMC requested but unavailable — needs an analytical PK model, no IOV)"
            .to_string()
    } else {
        "Metropolis-Hastings random walk".to_string()
    }
}

/// Assemble a `ModelParameters` snapshot from the current SAEM `state`. Shared
/// by the final post-loop parameter build and by the periodic resume checkpoint
/// (#755) so the two never drift. `n_kappa > 0` preserves the IOV Ω structural
/// free-mask (see the inline note at the call site).
fn saem_state_to_params(
    state: &SaemState,
    init_params: &ModelParameters,
    n_kappa: usize,
) -> ModelParameters {
    let omega = OmegaMatrix::from_matrix(
        state.omega_mat.clone(),
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    ModelParameters {
        theta: state.theta.clone(),
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: state.sigma_vals.clone(),
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        omega_iov: if n_kappa > 0 {
            // Use from_matrix_with_mask so the structural free_mask is preserved
            // when this snapshot is handed to a chained estimator (e.g.
            // [saem, foce]); from_matrix would infer the mask from nonzeros and
            // could mark a legitimately-zero off-diagonal as structurally fixed.
            init_params.omega_iov.as_ref().map(|iov_ref| {
                OmegaMatrix::from_matrix_with_mask(
                    state.omega_iov_mat.clone(),
                    iov_ref.eta_names.clone(),
                    iov_ref.diagonal,
                    iov_ref.free_mask.clone(),
                )
            })
        } else {
            init_params.omega_iov.clone()
        },
        kappa_fixed: init_params.kappa_fixed.clone(),
        mixture: None,
    }
}

// ---------------------------------------------------------------------------
// Main SAEM loop
// ---------------------------------------------------------------------------

/// Progress line printed once SAEM's final OFV is known, *before* the covariance
/// step runs (#893). SAEM learns its OFV only at the very end (the final FOCE
/// approximation), and the covariance step is often the most expensive part of
/// the run, so reporting the OFV first lets a CLI user interrupt (Ctrl-C) on a
/// bad OFV before paying for the covariance matrix. Kept as a pure function so
/// the message is unit-testable.
fn saem_final_ofv_report(ofv: f64) -> String {
    format!("SAEM completed. Final OFV = {:.4}", ofv)
}

pub fn run_saem(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k1 = options.saem_n_exploration;
    let k2 = options.saem_n_convergence;
    let n_iter = k1 + k2;
    // Suppress the Ω M-step for the first `omega_burnin` iterations so the MH
    // chain warms up at the initial Ω before any variance component is
    // estimated. Clamped to the exploration length — burning in past K1 would
    // freeze Ω into the convergence phase. See `FitOptions::saem_omega_burnin`.
    let omega_burnin = options.saem_omega_burnin.min(k1);
    let n_mh_steps = options.saem_n_mh_steps;
    // Componentwise sweeps per iteration (Kuhn-Lavielle kernel 2). Each sweep is
    // `n_eta` single-coordinate proposals, so sizing it `n_mh_steps / n_eta`
    // keeps the kernel's NLL-eval cost roughly on par with the block kernel.
    // Skipped entirely for single-η models, where there is no off-diagonal to
    // decorrelate and the kernel would duplicate the block move.
    let n_cw_sweeps = if n_eta >= 2 {
        (n_mh_steps / n_eta).max(2)
    } else {
        0
    };
    let adapt_interval = options.saem_adapt_interval;
    let verbose = options.verbose;
    let n_leapfrog = options.saem_n_leapfrog;
    // HMC is BSV-only (kappa-unaware); disable it for IOV models so eta sampling
    // uses the MH kernels that target the IOV conditional p(η | κ, θ, data).
    // Without this guard, an IOV model with an analytical PK path and
    // `n_leapfrog > 0` would propose eta against the kappa-free posterior and
    // hand a BSV-only NLL to the componentwise kernel as its (mismatched)
    // acceptance baseline.
    //
    // IIV on residual error (#409): the Dual2 NLL/gradient kernels build the
    // residual variance from σ alone and carry no `exp(2·η_ruv)` scaling rule,
    // so an HMC E-step would sample η against the unscaled conditional — η_ruv
    // sees no data curvature and collapses toward the prior. Disable HMC for
    // these models so the (correctly-scaled) MH kernels run instead.
    let using_hmc: bool = n_leapfrog > 0
        && model.ode_spec.is_none()
        && model.tv_fn.is_some()
        && n_kappa == 0
        && model.residual_error_eta.is_none()
        // Mixture models (#985): the E-step must sample the class indicator and
        // run η-MCMC within the drawn class (per-subject `MIXNUM` guard). The HMC
        // gradient kernel is class-unaware, so disable it and use the MH kernels,
        // which honour the class thread-local set in the E-step closure.
        && model.mixture.is_none();

    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();

    // Master RNG
    let master_seed = options.saem_seed.unwrap_or(12345);

    if verbose {
        eprintln!(
            "SAEM: {} subjects, {} ETAs, {} total iter ({} explore + {} converge)",
            n_subjects, n_eta, n_iter, k1, k2
        );
    }

    let mut warnings = Vec::new();
    if n_leapfrog > 0 && !using_hmc {
        // Keep the substring "HMC is unavailable" in both arms — `classify_warning`
        // keys on it to tag this as an Info/gradient_fallback warning.
        let reason = if n_kappa > 0 {
            "HMC is unavailable for IOV models (it is kappa-unaware)"
        } else if model.residual_error_eta.is_some() {
            "HMC is unavailable with IIV on residual error (iiv_on_ruv) — the Dual2 \
             gradient kernel has no exp(2·η_ruv) variance-scaling rule"
        } else {
            "HMC is unavailable (requires an analytical PK model the Dual2 gradient supports)"
        };
        warnings.push(format!(
            "saem_n_leapfrog > 0 but {reason}; falling back to Metropolis-Hastings"
        ));
    }
    let target_accept_rate = if using_hmc { 0.65_f64 } else { 0.40_f64 };

    // Initialize state
    let theta_cur = init_params.theta.clone();
    let omega_cur = init_params.omega.matrix.clone();
    let sigma_cur = init_params.sigma.values.clone();
    let s2 = omega_cur.clone();

    let etas: Vec<Vec<f64>> = (0..n_subjects)
        .map(|si| {
            let mut eta = get_eta_init(n_eta, None, None);
            // For FREM models, initialise covariate etas at their
            // conditional mode: eta_j = DV_cov - theta_k.  The posterior
            // for these etas is extremely peaked (EPSCOV ≈ 1e-6), so
            // starting at 0 leaves the chain far from the mode and
            // virtually every MH proposal gets rejected.
            if let Some(ref fc) = model.frem_config {
                let subj = &population.subjects[si];
                if !subj.fremtype.is_empty() {
                    for (&ft, &(theta_idx, eta_idx)) in &fc.fremtype_to_indices {
                        // Find the first observation with this FREMTYPE
                        if let Some(pos) = subj.fremtype.iter().position(|&f| f == ft) {
                            let dv = subj.observations[pos];
                            let tv = theta_cur[theta_idx];
                            eta[eta_idx] = dv - tv;
                        }
                    }
                }
            }
            eta
        })
        .collect();
    let step_scales = vec![0.3; n_subjects];
    // Componentwise kernel scales η'_j by √Ω_jj (a marginal SD), so a multiplier
    // near 1 is already a sensible 1-D step; start higher than the block kernel
    // and let adaptation climb toward the ~2.4 optimum.
    //
    // For FREM covariate etas the posterior is near-deterministic (EPSCOV
    // ≈ 1e-6) so the optimal CW step is orders of magnitude below the
    // prior SD.  Pre-compute: step_scale_j ≈ √(EPSCOV) / √(Ω_jj) so
    // that `step_scale_j · √Ω_jj ≈ √EPSCOV`.  This avoids thousands of
    // adaptation iterations to shrink from 1.0 down to ~1e-5.
    let cw_init = {
        let mut v = vec![1.0_f64; n_eta];
        if let Some(ref fc) = model.frem_config {
            let epscov = init_params.sigma.values[fc.covariate_sigma_index];
            for &(_theta_idx, eta_idx) in fc.fremtype_to_indices.values() {
                if eta_idx < n_eta {
                    let omega_jj = init_params.omega.matrix[(eta_idx, eta_idx)].max(1e-10);
                    // Target proposal SD = √EPSCOV; CW multiplies by √Ω_jj,
                    // so step_scale = √EPSCOV / √Ω_jj.  Floor at 1e-6.
                    v[eta_idx] = (epscov.sqrt() / omega_jj.sqrt()).max(1e-6);
                }
            }
        }
        v
    };
    let cw_step_scales = vec![cw_init; n_subjects];

    // Guard: the parser must guarantee omega_iov is present whenever kappas
    // are declared; if this fires, the caller wired up a broken ModelParameters.
    debug_assert!(
        n_kappa == 0 || init_params.omega_iov.is_some(),
        "n_kappa > 0 but init_params.omega_iov is None — model is misconfigured"
    );

    // Initialize IOV kappa state
    let (kappas_init, omega_iov_init, s2_iov_init): (
        Vec<Vec<Vec<f64>>>,
        DMatrix<f64>,
        DMatrix<f64>,
    ) = if n_kappa > 0 {
        let kaps: Vec<Vec<Vec<f64>>> = population
            .subjects
            .iter()
            .map(|s| {
                let n_occ = iov_occasion_groups(s).len();
                vec![vec![0.0f64; n_kappa]; n_occ]
            })
            .collect();
        let iov_mat = init_params
            .omega_iov
            .as_ref()
            .map(|iov| iov.matrix.clone())
            .unwrap_or_else(|| DMatrix::identity(n_kappa, n_kappa));
        (kaps, iov_mat.clone(), iov_mat)
    } else {
        (
            vec![vec![]; n_subjects],
            DMatrix::zeros(0, 0),
            DMatrix::zeros(0, 0),
        )
    };
    let kappa_step_scales = vec![0.3; n_subjects];

    // Initial NLL cache — use IOV-aware NLL when kappas are present
    let omega_iov_init_om = if n_kappa > 0 {
        init_params.omega_iov.clone()
    } else {
        None
    };
    let nll_cache: Vec<f64> = population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            if n_kappa > 0 {
                individual_nll_iov(
                    model,
                    subject,
                    &theta_cur,
                    &etas[i],
                    &kappas_init[i],
                    &init_params.omega,
                    omega_iov_init_om.as_ref(),
                    &sigma_cur,
                )
            } else {
                individual_nll(
                    model,
                    subject,
                    &theta_cur,
                    &etas[i],
                    &init_params.omega,
                    &sigma_cur,
                )
            }
        })
        .collect();

    // Per-theta packing flag: log for `theta_lower >= 0` (CL/V/KA…),
    // identity when `theta_lower < 0` (covariate exponents like
    // THETA_AGE_CL = -0.01 or THETA_CL_GAMMA = -0.8). Same convention
    // as `parameterization.rs::pack_params`. Without this, every theta
    // with a negative lower bound got clamped to 1e-10 by the old
    // `t.max(1e-10).ln()` packing and could never be estimated —
    // visible regression: SAD_SCEN4 SAEM left γ_CL stuck at 0 (truth
    // -0.8), letting the rest of the fit drift to compensate.
    let theta_packs_log_mask: Vec<bool> = init_params
        .theta_lower
        .iter()
        .map(|&lo| crate::estimation::parameterization::theta_packs_log(lo))
        .collect();
    let pack_theta = |i: usize, t: f64| -> f64 {
        if theta_packs_log_mask[i] {
            t.max(1e-10).ln()
        } else {
            t
        }
    };
    let unpack_theta = |i: usize, packed: f64| -> f64 {
        if theta_packs_log_mask[i] {
            packed.exp()
        } else {
            packed
        }
    };

    // Pack initial theta (per-mask) and sigma (always log).
    let mut log_theta: Vec<f64> = (0..n_theta).map(|i| pack_theta(i, theta_cur[i])).collect();
    let mut log_sigma: Vec<f64> = sigma_cur.iter().map(|&s| s.max(1e-10).ln()).collect();

    // Bounds in packed space — log when log-packed, identity otherwise.
    let mut log_theta_lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_lower[i].max(1e-10).ln()
            } else {
                init_params.theta_lower[i]
            }
        })
        .collect();
    let mut log_theta_upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_upper[i].min(1e9).ln()
            } else {
                init_params.theta_upper[i]
            }
        })
        .collect();
    let mut log_sigma_lower = vec![-8.0f64; n_sigma];
    let mut log_sigma_upper = vec![5.0f64; n_sigma];

    // Pin FIX parameters: set lower == upper == packed_value so the inner
    // NLopt M-step treats them as constants. Matches the FOCE/FOCEI treatment.
    for i in 0..n_theta {
        if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
            log_theta_lower[i] = log_theta[i];
            log_theta_upper[i] = log_theta[i];
        }
    }
    for i in 0..n_sigma {
        if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
            log_sigma_lower[i] = log_sigma[i];
            log_sigma_upper[i] = log_sigma[i];
        }
    }

    // #895: iiv_on_ruv creates a σ × ω_RUV ridge (residual var = σ²·exp(2·η_RUV)),
    // so a poorly-mixing E-step can let the M-step ride σ up to its e⁵ ceiling.
    // Cap each *free* RUV-scaled residual σ's growth to a generous multiple of its
    // starting value (`SAEM_RUV_SIGMA_LN_GROWTH`) so a genuinely ill-posed run
    // stays bounded near sensible SDs. A FREM EPSCOV (always FIX, and independent
    // of the RUV scaling) is skipped, as is any FIXed σ.
    //
    // The cap is applied as a post-M-step clamp on `log_sigma`, NOT by tightening
    // the NLopt upper bound: a well-posed fit whose σ never approaches the cap
    // must reproduce the un-capped trajectory bit-for-bit (changing the bound
    // handed to NLopt perturbs its search path even when the optimum is interior).
    // `None` means "no cap for this σ"; a `Some(cap)` records the log-σ ceiling so
    // the update can be clamped and a run that ends pinned against it flagged.
    let mut ruv_sigma_caps: Vec<Option<f64>> = compute_ruv_sigma_caps(
        model.residual_error_eta.is_some(),
        model
            .frem_config
            .as_ref()
            .map(|fc| fc.covariate_sigma_index),
        &log_sigma,
        &log_sigma_upper,
        &init_params.sigma_fixed,
    );

    // #904: which residual σ absorb the iiv_on_ruv η-mean during re-centering —
    // exactly the free, non-FREM, RUV-scaled σ. Computed directly (not from
    // `ruv_sigma_caps`, which drops a free σ whose growth cap is looser than a
    // tight user upper bound — that σ must still absorb the shift). A FREM EPSCOV
    // or a FIXed σ is never scaled.
    let frem_cov_sigma_idx = model
        .frem_config
        .as_ref()
        .map(|fc| fc.covariate_sigma_index);
    let ruv_sigma_absorb: Vec<bool> = (0..n_sigma)
        .map(|i| {
            !init_params.sigma_fixed.get(i).copied().unwrap_or(false)
                && Some(i) != frem_cov_sigma_idx
        })
        .collect();

    // Re-centering scales η_RUV by −mean, which multiplies the *whole* residual
    // variance R = Σ σ_c² (all components, e.g. additive + proportional) by
    // exp(−2·mean). It preserves each subject's residual variance exactly only if
    // *every* non-FREM residual σ component absorbs the shift (each scaled by
    // exp(mean), so R → R·exp(2·mean)). If any RUV-scaled σ component is FIXed,
    // scaling only the free ones leaves the fixed part uncompensated and silently
    // perturbs R — so re-centering is disabled for that config and the σ/ω_RUV
    // growth caps carry the load instead (a FIXed component also partially anchors
    // the σ × η_RUV ridge). The FREM EPSCOV (always FIX, not on a real-obs row) is
    // exempt from this check.
    let ruv_recenter_ok = ruv_recenter_allowed(
        model.residual_error_eta.is_some(),
        model
            .frem_config
            .as_ref()
            .map(|fc| fc.covariate_sigma_index),
        &init_params.sigma_fixed,
    );

    // #895: log-variance ceiling for the iiv_on_ruv Ω diagonal — the ω_RUV half
    // of the σ × ω_RUV ridge backstop (see `compute_ruv_omega_cap`).
    let mut ruv_omega_cap: Option<f64> = compute_ruv_omega_cap(
        model.residual_error_eta,
        n_eta,
        &init_params.omega.matrix,
        &init_params.omega_fixed,
    );

    let mut state = SaemState {
        etas,
        kappas: kappas_init,
        nll_cache,
        step_scales,
        cw_step_scales,
        kappa_step_scales,
        accept_counts: vec![0; n_subjects],
        proposal_counts: vec![0; n_subjects],
        cw_accept_counts: vec![vec![0usize; n_eta]; n_subjects],
        cw_proposal_counts: vec![vec![0usize; n_eta]; n_subjects],
        kappa_accept_counts: vec![0; n_subjects],
        kappa_proposal_counts: vec![0; n_subjects],
        steps_since_adapt: 0,
        s2,
        s2_iov: s2_iov_init,
        theta: theta_cur,
        omega_mat: omega_cur,
        omega_iov_mat: omega_iov_init,
        sigma_vals: sigma_cur,
    };

    // Mu-referencing pairs for the closed-form M-step: (theta_idx, eta_idx).
    // Only log-mu-ref pairs are returned (`get_mu_ref_pairs` filters out
    // additive ones), since the closed-form `log_theta += γ · mean(η)` only
    // applies to log-mu-referenced thetas.
    let mu_ref_pairs: Vec<(usize, usize)> = get_mu_ref_pairs(model);
    // Mixture: a MIXNUM-switched typical value pairs one η with several class
    // thetas, so the *pooled* `log_theta += γ·mean(η)` update above (one theta
    // per η) does not apply. It is well-posed per class though — SAEM draws a
    // hard class per subject, so `log θ_k += γ·mean_{i : c_i = k}(η_i)` — and
    // `get_mixture_mu_ref_pairs` supplies that class-resolved anchor set (#996).
    // Any class θ the parser could not resolve to a mu-ref pattern still routes
    // through the full NLopt θ/σ M-step, which — run under each subject's class
    // guard — estimates every class's thetas from its own members (#985).
    let mut saem_mix: Option<crate::estimation::saem_mixture::SaemMixture> =
        model.mixture.as_ref().map(|_| {
            crate::estimation::saem_mixture::SaemMixture::build(model, init_params, population)
        });
    if let Some(mix) = saem_mix.as_ref() {
        if mix.has_sigma_override() {
            warnings.push(
                "SAEM does not yet re-estimate per-class σ overrides (sigma(k)); they are held \
                 at their initial values. Class-switched θ, Ω overrides, and the mixing \
                 coefficients are estimated. Route to FOCEI if the σ overrides must be fit (#985)."
                    .to_string(),
            );
        }
        // Reject a theta that drives both the mixing expression and a structural
        // typical value: SAEM's separated M-step would estimate it from the
        // residual likelihood and then discard that for the mixing fit, silently
        // mis-fitting. FOCEI's joint marginal handles the shared parameter, so
        // route there instead of guessing (#987 review).
        let overlap = crate::estimation::saem_mixture::mixing_structural_overlap(
            model,
            init_params,
            population,
            &mix.mixing_theta_idx,
        );
        if !overlap.is_empty() {
            let names: Vec<String> = overlap
                .iter()
                .map(|&j| {
                    init_params
                        .theta_names
                        .get(j)
                        .cloned()
                        .unwrap_or_else(|| format!("theta[{j}]"))
                })
                .collect();
            return Err(format!(
                "SAEM cannot fit a mixture where a mixing-coefficient theta also drives the \
                 structural model: {} appear(s) in both the [mixture] mixing expression and an \
                 [individual_parameters] typical value. SAEM estimates the mixing coefficients \
                 and the structural thetas in separate M-steps, so a shared parameter would be \
                 double-owned. Split it into two thetas (one for structure, one for mixing), or \
                 fit with FOCE/FOCEI (whose joint marginal handles the shared parameter) (#985).",
                names.join(", ")
            ));
        }
    }
    // Per-class held σ overrides, hoisted out of the loop: they are constant
    // under SAEM (held at their inits) and the θ/σ M-step substitutes them per
    // subject so the free base σ is not dragged by override-class members.
    let mix_sigma_over: Vec<Vec<(usize, f64)>> = saem_mix
        .as_ref()
        .map(|m| m.class_sigma_overrides())
        .unwrap_or_default();
    // #996 open question, confirmed: an identity-packed θ (`theta_lower < 0`,
    // see `theta_packs_log`) reaches the closed-form loop today and would be
    // updated as `θ += mean(η)` instead of `θ *= exp(mean(η))` — the shift is
    // only the EM optimum on the log scale. Such a θ is dropped from the
    // closed-form channel and estimated by the numerical M-step instead.
    let identity_packed = |pairs: &[(usize, usize)]| -> Vec<usize> {
        let mut v: Vec<usize> = pairs
            .iter()
            .filter(|&&(t, _e)| !theta_packs_log_mask[t])
            .map(|&(t, _e)| t)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let dropped_identity = identity_packed(&mu_ref_pairs);
    let mu_ref_pairs: Vec<(usize, usize)> = mu_ref_pairs
        .into_iter()
        .filter(|&(t, _e)| theta_packs_log_mask[t])
        .collect();

    // Class-aware mu-ref pairs for a mixture (#996). Empty for a non-mixture
    // model, which uses `mu_ref_pairs` above.
    //
    // SAEM takes only the **class-shared** anchors — an η whose typical value is
    // the same theta in every class (`V = TVV * exp(ETA_V)` inside a mixture).
    // For those the per-class update collapses to the classical pooled
    // `log θ += γ·mean(η)` (every class maps to one theta, so the weights sum to
    // N), so this is a strict extension of the single-population closed form:
    // before #996 a mixture disabled mu-referencing wholesale and even a
    // class-shared typical value went through the numerical M-step.
    //
    // A genuinely `MIXNUM`-switched typical value is deliberately *not* taken
    // here, even though `get_mixture_mu_ref_pairs` can express it. SAEM draws a
    // hard class per subject, and the per-class mean `mean_{i : c_i = k}(η_i)`
    // is then a classification-EM statistic: the class boundary is re-drawn from
    // the θ_k it just moved, and the two feed back. Measured on the two-class
    // anchor (`tests/nonmem/mixture_iv.csv`, seed 20250818) the class-aware
    // variant lands at `TVCL2 = 3.02` with OFV 305.0 against `2.75` / 302.2 for
    // the numerical M-step (NONMEM SAEM: 2.735; FOCEI optimum: 2.842) — worse on
    // every coordinate — and a Rao-Blackwellised soft-responsibility variant
    // converges to the same wrong point, so the bias is in the hard-class
    // sufficient statistic, not the weighting. IMP/IMPMAP, which importance-
    // samples within *every* class and weights by the responsibilities, does not
    // have this failure mode and does use the switched anchors.
    //
    // All of this is conditional on `mu_referencing`: with it off every θ goes
    // through the numerical M-step by construction, so telling the user to switch
    // estimator to get a closed-form shift they turned off is noise (#996 review).
    let mut mix_switched_skipped: Vec<usize> = Vec::new();
    let mix_mu_ref_pairs: Vec<MixtureMuRefPair> = if saem_mix.is_some() && options.mu_referencing {
        get_mixture_mu_ref_pairs(model)
            .into_iter()
            .filter(|p| p.theta_idx.iter().all(|&t| theta_packs_log_mask[t]))
            .filter(|p| {
                let shared = p.theta_idx.windows(2).all(|w| w[0] == w[1]);
                if !shared {
                    mix_switched_skipped.extend(p.theta_idx.iter().copied());
                }
                shared
            })
            .collect()
    } else {
        Vec::new()
    };
    if !mix_switched_skipped.is_empty() {
        mix_switched_skipped.sort_unstable();
        mix_switched_skipped.dedup();
        let names: Vec<&str> = mix_switched_skipped
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        warnings.push(format!(
            "SAEM: the MIXNUM-switched typical value(s) {} are estimated by the numerical \
             M-step, not the closed-form mu-referencing shift — SAEM's hard per-subject class \
             draw makes the per-class η mean a biased (classification-EM) statistic. Use \
             IMP/IMPMAP if the class-aware mu-ref shift is wanted (#996).",
            names.join(", ")
        ));
    }
    let mut dropped_identity = dropped_identity;
    if saem_mix.is_some() && options.mu_referencing {
        for p in get_mixture_mu_ref_pairs(model) {
            if p.theta_idx.iter().any(|&t| !theta_packs_log_mask[t]) {
                dropped_identity.extend(p.theta_idx.iter().copied());
            }
        }
        dropped_identity.sort_unstable();
        dropped_identity.dedup();
    }
    // Same gate: the identity-packing advisory is about a closed-form update that
    // does not run at all under `mu_referencing = false` (#996 review).
    if !dropped_identity.is_empty() && options.mu_referencing {
        let names: Vec<&str> = dropped_identity
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        warnings.push(format!(
            "SAEM: typical value(s) {} are log-mu-referenced but declared with a negative \
             lower bound, so they are packed on the identity scale; the closed-form \
             `log θ += γ·mean(η)` update does not apply and they are estimated by the \
             numerical M-step instead. Give them a non-negative lower bound to use the \
             closed-form update (#996).",
            names.join(", ")
        ));
    }

    let use_closed_form_mstep = options.mu_referencing
        && if saem_mix.is_some() {
            // Mixture: the class-aware shift replaces the per-θ numerical
            // M-step for every log-mu-ref θ (#996).
            !mix_mu_ref_pairs.is_empty()
        } else {
            !mu_ref_pairs.is_empty()
        };

    // STRONG advisory: an estimated θ with **no associated ETA** is not
    // mu-referenced, so it never receives the γ-damped closed-form
    // `log θ += γ·mean(η)` shift. It is moved only by the η-frozen numerical
    // M-step, which re-maximises the conditional observation likelihood against
    // a *single* MCMC η draw with no stochastic-approximation damping — the
    // update is a random walk on a noisy surface, not an SA-averaged statistic,
    // and it can drift a long way from the marginal optimum.
    //
    // Measured on the FREM `iiv_on_ruv` reprex (475 subjects, 12 ETAs, the same
    // `FRD1` absorption fraction that motivated the IMP/IMPMAP advisory in
    // #406): SAEM drove `TVFRD1` 0.383 → 0.039 while IMP (0.311), IMPMAP
    // (0.318) and NONMEM IMP (0.394) agree, dragging `TVV` +6% and `TVMAT` +9%
    // with it. The drift is not a start artifact — restarting SAEM *at* the
    // IMPMAP solution still walked it down to 0.065 — and it is removed
    // entirely by attaching an ETA (`FRD1 = TVFRD1*exp(ETA_FRD1)`, ω² = 0.01 →
    // SAEM recovers 0.313) or by holding the parameter FIX (every other θ then
    // lands within 3% of NONMEM).
    let class_mu_ref_thetas: Vec<usize> = if use_closed_form_mstep && saem_mix.is_some() {
        mix_mu_ref_pairs
            .iter()
            .flat_map(|p| p.theta_idx.iter().copied())
            .collect()
    } else {
        Vec::new()
    };
    let thetas_without_eta = crate::estimation::impmap::non_fixed_thetas_without_eta(
        model,
        &init_params.theta_fixed,
        &class_mu_ref_thetas,
    );
    if !thetas_without_eta.is_empty() {
        warnings.push(format!(
            "SAEM: estimated parameter(s) [{}] have NO associated ETA, so they are not \
             mu-referenced and are moved only by the η-frozen numerical M-step. That channel \
             re-maximises against a single MCMC η draw, so it carries a Monte-Carlo bias that \
             SA damping (#1011) reduces but does not remove, and it can still settle away from \
             the marginal optimum. Prefer an ETA on each (e.g. `P = TVP * exp(ETA_P)` with a \
             small, optionally FIX, omega — ferx applies mu-referencing automatically), or hold \
             the parameter FIX, or use FOCEI/IMPMAP.",
            thetas_without_eta.join(", ")
        ));
    }

    // #1011: is the numerical M-step actually estimating a θ this fit? A θ that
    // is FIXed, or pinned out by the closed-form mu-ref shift, is returned
    // unchanged by NLopt, so a fit whose every θ is one of those has nothing for
    // the SA damping below to act on — and skipping it there keeps such fits
    // byte-identical to the pre-#1011 behaviour (σ included).
    //
    // **Mixtures are excluded.** A `MIXNUM`-switched typical value is estimated
    // by this same numerical M-step (#996 routes it there deliberately, because
    // SAEM's hard class draw makes the per-class η mean a biased statistic), but
    // it is solving a different problem: the class typical values must *separate*
    // from a common start, and the class assignments only stabilise once they
    // have. Damping that excursion stalls the separation — measured on
    // `tests/nonmem/mixture_iv_saem`, a 0.03 exploration cap leaves
    // `TVCL1 = 1.145` against NONMEM's 1.002, and drags the chained IMP marginal
    // with it. Whether the #1011 bias also affects mixture class θ (it plausibly
    // does) needs its own schedule and its own anchor, so it is left alone here
    // rather than half-fixed; the advisory above still fires for them.
    let numerically_estimated_theta = {
        let pinned: Vec<usize> = if !use_closed_form_mstep {
            Vec::new()
        } else if saem_mix.is_some() {
            class_mu_ref_thetas.clone()
        } else {
            mu_ref_pairs.iter().map(|&(t, _e)| t).collect()
        };
        damps_numerical_mstep(
            saem_mix.is_some(),
            n_theta,
            &init_params.theta_fixed,
            &pinned,
        )
    };

    // #1011: exploration-phase cap on the numerical M-step's SA step. `None`
    // takes the calibrated default; `1.0` disables the damping entirely.
    let mstep_damping_cap = options.saem_mstep_damping.unwrap_or(MSTEP_SA_MAX_STEP);
    // A setting the fit cannot act on is worse than no setting: say so rather
    // than accepting it silently.
    if options.saem_mstep_damping.is_some() && !numerically_estimated_theta {
        let reason = if saem_mix.is_some() {
            "this is a mixture model, whose class typical values must separate from a common \
             start before the class assignments settle — damping that excursion stalls it"
        } else {
            "every theta in this model is mu-referenced or FIX, so the numerical M-step has no \
             theta to damp"
        };
        warnings.push(format!(
            "SAEM: `mstep_damping` was set but has no effect — {reason} (#1011)."
        ));
    }

    // Accumulator for the `obs_nll_sum` (population OFV) evaluations skipped
    // by pinning mu-ref dims out of NLopt's central-FD gradient.  Each pinned
    // dim costs `2 * mstep_maxiter` `obs_nll_sum` calls inside NLopt — that's
    // the value we add per M-step that takes the closed-form branch.
    let mut mstep_grad_step_evals_saved: u64 = 0;

    // Per-subject flag: did this subject successfully use HMC at least once?
    // Only meaningful when `using_hmc = true`; stays all-false otherwise.
    let mut hmc_subjects = vec![false; n_subjects];

    // Run-cumulative combined (block + componentwise) MH accept / proposal counts
    // over the post-burn-in iterations, for the end-of-run "sampler not mixing"
    // warning (#895). Kept separate from `state.*_counts`, which the adaptation
    // step resets every `adapt_interval`.
    let mut cum_mh_acc: u64 = 0;
    let mut cum_mh_prop: u64 = 0;

    // Main loop
    for k in 1..=n_iter {
        // Per-iteration combined (block + componentwise) accept / proposal
        // tallies — the honest E-step mixing rate reported to the trace (#895).
        let mut iter_acc: usize = 0;
        let mut iter_prop: usize = 0;
        if crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("SAEM: cancelled at iteration {}", k);
            }
            break;
        }
        let gamma = if k <= k1 { 1.0 } else { 1.0 / (k - k1) as f64 };
        // Damped SA step for the Ω sufficient statistic during exploration only.
        // With the full γ=1 used for θ, an undamped Ω would be overwritten each
        // exploration iteration by a single (warm-started, not-yet-equilibrated)
        // MCMC draw; for a correlated block that snapshot is biased toward the
        // chain's current correlation, and the bias feeds back through chol(Ω)
        // into the next proposal — a runaway toward a near rank-1 Ω. Capping the
        // Ω learning rate during exploration averages those draws (Robbins-Monro)
        // and breaks the feedback, while θ keeps moving at full γ. In the
        // convergence phase the cap is lifted: Ω uses the full decaying
        // γ = 1/(k−k1), the same schedule as θ, so the SA estimate settles
        // correctly (the chain is equilibrated by then, so the single-draw
        // overwrite risk that motivated the cap no longer applies).
        // #1011: SA step for the numerical θ/σ M-step result, mirroring
        // `gamma_omega`. Capped during exploration so a single un-equilibrated
        // MCMC draw cannot carry an un-mu-referenced θ away; full decaying γ in
        // the convergence phase. Left at 1.0 (undamped, pre-#1011) when NLopt has
        // no θ to estimate, so those fits are unchanged.
        let gamma_theta = mstep_sa_step(
            numerically_estimated_theta,
            k <= k1,
            gamma,
            mstep_damping_cap,
        );
        let gamma_omega = if k <= k1 {
            gamma.min(OMEGA_SA_MAX_STEP)
        } else {
            gamma
        };
        // Rebuild omega for this iteration
        let omega_k = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );

        // Rebuild omega_iov for this iteration.  Using from_matrix_with_mask
        // (not from_matrix) preserves the structural free_mask so that an
        // off-diagonal entry that converges to zero is not mistakenly treated
        // as a structural zero in the Cholesky proposal distribution.
        // Used in both the eta MH (Bug 2 fix) and the kappa MH (Step 1b).
        let omega_iov_cur_opt: Option<OmegaMatrix> = if n_kappa > 0 {
            init_params.omega_iov.as_ref().map(|iov_ref| {
                OmegaMatrix::from_matrix_with_mask(
                    state.omega_iov_mat.clone(),
                    iov_ref.eta_names.clone(),
                    iov_ref.diagonal,
                    iov_ref.free_mask.clone(),
                )
            })
        } else {
            None
        };

        // Mixture (#985): refresh every class's Ω/σ from the just-rebuilt base
        // plus the per-class overrides, then precompute the per-class proposal
        // Ω, σ, and componentwise SDs the E-step indexes by each subject's drawn
        // class. `class_omegas[0]` is the base (== omega_k) for a shared-Ω model.
        let (class_omegas, class_sigmas, class_cw_sd): (
            Vec<OmegaMatrix>,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
        ) = if let Some(mix) = saem_mix.as_mut() {
            mix.sync_base(&omega_k, &state.sigma_vals);
            let n_c = mix.n_classes;
            let omg: Vec<OmegaMatrix> = (0..n_c).map(|c| mix.class_omega(c).clone()).collect();
            let sig: Vec<Vec<f64>> = (0..n_c).map(|c| mix.class_sigma(c).to_vec()).collect();
            let cw: Vec<Vec<f64>> = omg
                .iter()
                .map(|o| {
                    (0..n_eta)
                        .map(|j| o.matrix[(j, j)].max(SAEM_OMEGA_DIAG_FLOOR).sqrt())
                        .collect()
                })
                .collect();
            (omg, sig, cw)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // ---- Step 1: MH simulation (parallelized) ----
        // Symmetric random-walk MH in eta_true space, identical schedule
        // throughout exploration and convergence — the only thing that
        // changes between phases is the SA step size `gamma`.
        //
        // Two kernels run per subject per iteration (Kuhn & Lavielle 2004
        // mixture): (1) the primary block kernel — HMC when available, else a
        // `chol(Ω)`-preconditioned block RW; then (2) a componentwise sweep
        // (`mh_steps_componentwise`) that perturbs one η at a time. Kernel (2)
        // is what keeps a block Ω from collapsing to rank-1 — see that fn's
        // docstring.
        {
            use crate::parser::model_parser::MixtureClassGuard;
            use rayon::prelude::*;
            let theta_ref = &state.theta;
            let sigma_ref = &state.sigma_vals;
            let omega_ref = &omega_k;
            // Immutable view of the mixture state for the per-subject class draw
            // (the `sync_base` mutable borrow above has ended). The per-class
            // proposal Ω/σ/CW-SDs precomputed above are indexed by the draw.
            let mix_ref = saem_mix.as_ref();
            let class_omegas_ref = &class_omegas;
            let class_sigmas_ref = &class_sigmas;
            let class_cw_sd_ref = &class_cw_sd;
            // Per-coordinate componentwise proposal SDs — computed once here (Ω's
            // diagonal is shared across subjects) rather than per subject inside
            // the parallel kernel. Floored to match the Ω diagonal floor.
            let cw_sd: Vec<f64> = (0..n_eta)
                .map(|j| omega_k.matrix[(j, j)].max(SAEM_OMEGA_DIAG_FLOOR).sqrt())
                .collect();
            let cw_sd_ref = &cw_sd;
            // Per-coordinate multiplier for the block kernel (issue #895). 1.0 for
            // every ordinary ETA — so non-FREM models get the exact `chol(Ω)·z`
            // move, byte-for-byte (×1.0). FREM covariate ETAs are near-
            // deterministic (posterior SD ≈ √EPSCOV ≪ √Ω_jj), so a full-scale
            // joint proposal for them is rejected every time and pins the whole
            // block acceptance at 0%; damp their coordinate to ≈ √EPSCOV/√Ω_jj so
            // the joint move can still explore the correlated PK block. The
            // componentwise kernel handles the covariate ETAs themselves.
            let blk_eta_scale: Option<Vec<f64>> = model.frem_config.as_ref().map(|fc| {
                let epscov = state.sigma_vals[fc.covariate_sigma_index];
                let mut v = vec![1.0_f64; n_eta];
                for &(_theta_idx, eta_idx) in fc.fremtype_to_indices.values() {
                    if eta_idx < n_eta {
                        v[eta_idx] = (epscov.sqrt() / cw_sd[eta_idx]).clamp(1e-6, 1.0);
                    }
                }
                v
            });
            let blk_eta_scale_ref = blk_eta_scale.as_deref();
            // For IOV models, eta proposals must target p(η | κ, θ, data):
            // the per-occasion [eta_prop, kappa_k] predictions determine
            // which etas are accepted.  Pass omega_iov to mh_steps so it
            // can call individual_nll_iov with kappas held fixed.
            let omega_iov_for_eta_mh: Option<&OmegaMatrix> = omega_iov_cur_opt.as_ref();

            // Returns (eta_new, nll_after, n_acc_primary, n_prop_primary,
            //          per_eta_acc_cw, n_sweeps_cw, used_hmc, mix_class)
            #[allow(clippy::type_complexity)]
            let results: Vec<(
                Vec<f64>,
                f64,
                usize,
                usize,
                Vec<usize>,
                usize,
                bool,
                usize,
            )> = state
                .etas
                .par_iter()
                .zip(state.nll_cache.par_iter())
                .zip(state.step_scales.par_iter())
                .zip(state.cw_step_scales.par_iter())
                .zip(state.kappas.par_iter())
                .enumerate()
                // Per-rayon-worker `EventPkParams` scratch: allocated
                // once per worker per outer iteration, reused across
                // every subject the worker handles. Without `map_init`
                // the scratch was allocated per subject per outer
                // iter (5937 × N_iter on the cefepime SAEM bench);
                // with it, n_workers × N_iter ≈ 10 × N_iter.
                .map_init(
                    EventPkParams::default,
                    |pk_scratch, (i, ((((eta, &nll), &scale), cw_sc_i), kappas_i))| {
                        let subject = &population.subjects[i];
                        let mut rng = StdRng::seed_from_u64(
                            master_seed
                                .wrapping_add(k as u64 * 100_000)
                                .wrapping_add(i as u64),
                        );
                        let kappas_mh_opt =
                            omega_iov_for_eta_mh.map(|iov| (kappas_i.as_slice(), iov));

                        // ---- Mixture E-step: draw the latent class ----
                        // Sample z_i ~ Categorical(PMIX_i) from the current
                        // posterior at this subject's η (and κ), then run the η
                        // moves *within* the drawn class: set the MIXNUM guard and
                        // point the proposal Ω/σ/CW-SDs at that class (#985).
                        // Draw the latent class; the per-subject posterior is
                        // recomputed at the converged params after the loop (via
                        // `mixture_ofv`), so only the sampled class is carried out.
                        let mix_class: usize = if let Some(mix) = mix_ref {
                            crate::estimation::saem_mixture::draw_class(
                                model,
                                subject,
                                theta_ref,
                                eta,
                                kappas_i,
                                mix,
                                omega_iov_for_eta_mh,
                                pk_scratch,
                                &mut rng,
                            )
                        } else {
                            0
                        };
                        let _class_guard = mix_ref.map(|_| MixtureClassGuard::enter(mix_class + 1));
                        let (omega_ref, sigma_ref, cw_sd_ref): (&OmegaMatrix, &[f64], &[f64]) =
                            if mix_ref.is_some() {
                                (
                                    &class_omegas_ref[mix_class],
                                    &class_sigmas_ref[mix_class],
                                    &class_cw_sd_ref[mix_class],
                                )
                            } else {
                                (omega_ref, sigma_ref.as_slice(), cw_sd_ref.as_slice())
                            };

                        let mut eta_work = eta.clone();

                        // ---- Kernel 1: primary block move ----
                        // Baseline for the first MH acceptance ratio is the cached
                        // NLL (as in the non-mixture path). For a mixture that value
                        // was computed under the previous iteration's drawn class, so
                        // on a class flip the very first proposal is scored against a
                        // slightly mismatched baseline — but the sweep recomputes the
                        // NLL under the current class from the next step on, so the
                        // effect is a single self-correcting proposal. Re-deriving the
                        // baseline at the drawn class each iteration was tried and
                        // measurably *worsened* the fit (it lets a wrong class draw
                        // drag η, inflating Monte-Carlo variance), so the cached
                        // baseline is deliberate (#987 review).
                        let mut nll_cur = nll;
                        let mut n_acc_primary = 0_usize;
                        let mut n_prop_primary = 0_usize;
                        // HMC path: one gradient-guided proposal per SAEM iteration.
                        // hmc_step returns None if HMC is unavailable for this subject
                        // (e.g. TV-cov subject with unsupported PK model); fall through
                        // to the block MH kernel. `did_hmc` doubles as the `used_hmc`
                        // flag reported back for diagnostics.
                        let did_hmc = if using_hmc {
                            if let Some((new_eta, new_nll, accepted, _divergent)) =
                                crate::estimation::hmc::hmc_step(
                                    subject, &eta_work, nll, model, theta_ref, omega_ref,
                                    sigma_ref, scale, n_leapfrog, &mut rng,
                                )
                            {
                                eta_work = new_eta;
                                nll_cur = new_nll;
                                n_acc_primary = accepted as usize;
                                n_prop_primary = 1;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        if !did_hmc {
                            let (n_acc, nll_new) = mh_steps(
                                &mut eta_work,
                                nll_cur,
                                subject,
                                model,
                                theta_ref,
                                omega_ref,
                                sigma_ref,
                                scale,
                                blk_eta_scale_ref,
                                &mut rng,
                                n_mh_steps,
                                pk_scratch,
                                kappas_mh_opt,
                            );
                            nll_cur = nll_new;
                            n_acc_primary = n_acc;
                            n_prop_primary = n_mh_steps;
                        }

                        // ---- Kernel 2: componentwise decorrelating sweep ----
                        let (per_eta_acc_cw, n_prop_cw, nll_cw) = mh_steps_componentwise(
                            &mut eta_work,
                            nll_cur,
                            subject,
                            model,
                            theta_ref,
                            omega_ref,
                            sigma_ref,
                            cw_sc_i,
                            cw_sd_ref,
                            &mut rng,
                            n_cw_sweeps,
                            pk_scratch,
                            kappas_mh_opt,
                        );

                        (
                            eta_work,
                            nll_cw,
                            n_acc_primary,
                            n_prop_primary,
                            per_eta_acc_cw,
                            n_prop_cw,
                            did_hmc,
                            mix_class,
                        )
                    },
                )
                .collect();

            // Collect the drawn classes for the mixture M-step. The per-subject
            // posterior is recomputed at the converged parameters after the loop
            // (via `mixture_ofv`), matching what the FOCEI mixture path reports.
            let mut drawn_classes: Vec<usize> = vec![0; n_subjects];
            for (
                i,
                (eta_new, nll_new, n_acc, n_prop, per_eta_acc_cw, n_prop_cw, used_hmc, mix_class),
            ) in results.into_iter().enumerate()
            {
                drawn_classes[i] = mix_class;
                state.etas[i] = eta_new;
                state.nll_cache[i] = nll_new;
                state.accept_counts[i] += n_acc;
                state.proposal_counts[i] += n_prop;
                // Accumulate per-eta CW acceptance counts
                let cw_acc_i: usize = per_eta_acc_cw.iter().sum();
                for j in 0..n_eta {
                    state.cw_accept_counts[i][j] += per_eta_acc_cw[j];
                    state.cw_proposal_counts[i][j] += n_cw_sweeps;
                }
                hmc_subjects[i] |= used_hmc;
                // Combined (block + componentwise) acceptance for this iteration —
                // the honest E-step mixing metric (#895). The block kernel alone
                // reads 0% for FREM-scale Ω even when the componentwise sweep is
                // mixing fine, so a block-only rate is misleading.
                iter_acc += n_acc + cw_acc_i;
                iter_prop += n_prop + n_prop_cw;
            }

            // ---- Mixture bookkeeping (#985) ----
            // Record the drawn classes, SA-update the responsibility average and
            // the Ω-override statistics, and accumulate the posterior for the
            // final output (convergence phase only, once θ/Ω have settled).
            if let Some(mix) = saem_mix.as_mut() {
                mix.classes = drawn_classes;
                mix.update_rbar(gamma);
            }
        }

        // Fold this iteration's combined tallies into the post-burn-in totals
        // that back the end-of-run mixing warning (#895).
        if k > omega_burnin {
            cum_mh_acc += iter_acc as u64;
            cum_mh_prop += iter_prop as u64;
        }

        // ---- Step 1b: Per-occasion kappa MH (IOV models only) ----
        // For each subject, propose one new kappa per occasion and accept/reject
        // using the full IOV individual NLL (kappa prior + observation likelihood).
        // This is a sequential per-subject loop (non-parallel) because the kappa
        // MH is cheap (low-dimensional, analytical PK) and share-free.
        if n_kappa > 0 {
            if let Some(omega_iov_cur) = omega_iov_cur_opt.as_ref() {
                for i in 0..n_subjects {
                    let subject = &population.subjects[i];
                    // Mixture (#985): κ must be sampled inside the subject's drawn
                    // class — under that class's `MIXNUM` branch and its Ω/σ.
                    // Without the guard every subject's κ would be proposed against
                    // the class-1 typical values and class-1 Ω/σ, corrupting
                    // `state.kappas` (hence `s2_iov`, Ω_IOV and the θ/σ M-step)
                    // for every class-2+ subject (#987 review).
                    let cls = saem_mix.as_ref().map(|m| m.classes[i]);
                    let _class_guard =
                        cls.map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                    let (omega_i, sigma_i): (&OmegaMatrix, &[f64]) = match cls {
                        Some(c) => (&class_omegas[c], class_sigmas[c].as_slice()),
                        None => (&omega_k, state.sigma_vals.as_slice()),
                    };
                    let mut rng = StdRng::seed_from_u64(
                        master_seed
                            .wrapping_add(k as u64 * 100_000)
                            .wrapping_add(i as u64)
                            .wrapping_add(999_999),
                    );
                    // Recompute NLL under the IOV-consistent function before
                    // proposing kappa.  After the eta MH block, nll_cache[i]
                    // may have been set by mh_steps via individual_nll_iov
                    // (with kappas fixed) — but to be safe we always recompute
                    // with the current kappas so detailed balance is guaranteed:
                    // both nll_kappa_ref and nll_prop are evaluated by the same
                    // individual_nll_iov, giving the correct acceptance ratio for
                    // p(κ | η, θ, data).
                    let nll_kappa_ref = individual_nll_iov(
                        model,
                        subject,
                        &state.theta,
                        &state.etas[i],
                        &state.kappas[i],
                        omega_i,
                        Some(omega_iov_cur),
                        sigma_i,
                    );
                    let (n_acc, n_prop, nll_new) = mh_kappa_steps(
                        &mut state.kappas[i],
                        nll_kappa_ref,
                        subject,
                        model,
                        &state.theta,
                        &state.etas[i],
                        omega_i,
                        omega_iov_cur,
                        sigma_i,
                        state.kappa_step_scales[i],
                        &mut rng,
                    );
                    state.nll_cache[i] = nll_new;
                    state.kappa_accept_counts[i] += n_acc;
                    state.kappa_proposal_counts[i] += n_prop;
                }
            }
        }

        state.steps_since_adapt += 1;

        // ---- Step 2: SA update of sufficient statistic for Omega ----
        // #904: re-center the iiv_on_ruv η to zero mean, absorbing the shift into
        // the residual σ. `Y = f + EPS·exp(η_RUV)` has no typical-value θ, so —
        // unlike a mu-referenced structural η (CL/V…), whose mean is folded into
        // its TVP each iteration — η_RUV's mean is otherwise never absorbed. It
        // then drifts along the degenerate direction (`σ²·exp(2η)` is invariant to
        // η→η−c, σ→σ·exp(c)), and the drift pollutes ω_RUV = mean(η²) with a
        // spurious mean² term that pumps the σ × ω_RUV runaway. σ plays the role
        // of the residual-scale typical value here: shifting η_RUV by −mean and
        // scaling every RUV-scaled σ by exp(mean) leaves each subject's residual
        // variance exactly unchanged while restoring E[η_RUV] = 0, so the next Ω
        // M-step sees the true variance. Only done when every non-FREM residual σ
        // is free to absorb the shift (`ruv_recenter_ok`); a FIXed component would
        // leave R only partially rescaled and break the exact-invariance guarantee
        // (it also already partially pins the mean — no full degeneracy).
        if let (true, Some(kr)) = (ruv_recenter_ok, model.residual_error_eta) {
            recenter_ruv_eta(
                &mut state.etas,
                kr,
                &mut log_sigma,
                &mut state.sigma_vals,
                &ruv_sigma_absorb,
            );
        }

        // Mixture with `omega(k)` overrides (#987 review): the base Ω statistic
        // must be built from the subjects that actually *share* the base entry —
        // pooling an override class's members biases the base toward the
        // mixture-wide spread. Entries with no base members keep their previous
        // value. Without overrides this reduces to the pooled statistic below.
        let base_partitioned = saem_mix
            .as_ref()
            .filter(|m| !m.mp.omega_override_addr.is_empty())
            .map(|m| m.base_eta_outer(&state.etas, n_eta));

        if let Some((mean, counts)) = base_partitioned {
            for j in 0..n_eta {
                for l in 0..n_eta {
                    if counts[(j, l)] > 0 {
                        state.s2[(j, l)] =
                            (1.0 - gamma_omega) * state.s2[(j, l)] + gamma_omega * mean[(j, l)];
                    }
                }
            }
        } else {
            let mut eta_outer = DMatrix::zeros(n_eta, n_eta);
            for eta in &state.etas {
                let ev = DVector::from_column_slice(eta);
                eta_outer += &ev * ev.transpose();
            }
            eta_outer /= n_subjects as f64;

            state.s2 = (1.0 - gamma_omega) * &state.s2 + gamma_omega * &eta_outer;
        }

        // Per-class Ω-override statistic: SA-updated on the *same* schedule as
        // the base `s2` above — every iteration, including burn-in, so the first
        // post-burn-in override update reflects the warmed-up chain rather than a
        // single-iteration class mean. `omega_stat_active` is what gates whether
        // `sync_base` *applies* it, mirroring the base Ω's burn-in gate below.
        if let Some(mix) = saem_mix.as_mut() {
            mix.mstep_omega_overrides(&state.etas, gamma_omega);
            mix.omega_stat_active = k > omega_burnin;
        }

        // ---- Step 2b: SA update for Omega_iov (IOV only) ----
        // s2_iov = (1 - γ) s2_iov + γ · (1/N_occ) Σᵢ Σₖ κᵢₖ κᵢₖᵀ
        if n_kappa > 0 {
            let mut kappa_outer = DMatrix::zeros(n_kappa, n_kappa);
            let mut n_total_occ = 0_usize;
            for kappas_i in &state.kappas {
                for kap in kappas_i {
                    let kv = DVector::from_column_slice(kap);
                    kappa_outer += &kv * kv.transpose();
                    n_total_occ += 1;
                }
            }
            if n_total_occ > 0 {
                kappa_outer /= n_total_occ as f64;
            }
            state.s2_iov = (1.0 - gamma_omega) * &state.s2_iov + gamma_omega * &kappa_outer;
        }

        // ---- Step 3: M-step Omega (BSV + IOV) ----
        // Gated by the burn-in: while `k <= omega_burnin` Ω (and Ω_iov) are held
        // at their initial values so the MH chain can warm up before any
        // variance component is estimated. Step 2 still refreshes the SA
        // statistic `s2` each burn-in iteration (damped at `gamma_omega`, so it
        // is a running average of the warming chain rather than the latest
        // snapshot), so the first Ω update after burn-in reflects the warmed-up
        // chain, not the cold-start spread.
        if k > omega_burnin {
            // ---- Step 3a: Omega_bsv (closed form) ----
            // Restore FIX-ed rows / columns from the template. An eta flagged FIX
            // keeps its initial variance AND its initial off-diagonal couplings
            // (zero for a diagonal declaration, block cov for a FIX-ed block).
            // Letting the sufficient statistic bleed into row/col of a fixed eta
            // breaks positive-definiteness once the free-block diagonals shrink
            // during the exploration phase.
            state.omega_mat = state.s2.clone();
            // Zero structurally-absent off-diagonals. `s2 = (1/N) Σ ηη^T` always
            // produces a dense matrix; entries that aren't free parameters
            // (standalone etas, or etas from different `block_omega` declarations)
            // must be zeroed so they don't feed sampling correlations back into
            // the next iteration's Cholesky proposal. Without this the chain drives
            // Ω toward a rank-deficient state, log|Ω| → -∞, and the M-step pushes
            // thetas to bounds to compensate.
            for i in 0..n_eta {
                for j in 0..n_eta {
                    if !init_params.omega.free_mask[(i, j)] {
                        state.omega_mat[(i, j)] = 0.0;
                    }
                }
            }
            // Restore FIX-ed rows / columns from the template.
            for i in 0..n_eta {
                for j in 0..n_eta {
                    let fi = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                    let fj = init_params.omega_fixed.get(j).copied().unwrap_or(false);
                    if fi || fj {
                        state.omega_mat[(i, j)] = init_params.omega.matrix[(i, j)];
                    }
                }
            }
            // Floor the free diagonal to keep Ω positive-definite, mirroring the
            // IOV Ω floor below. On sparse data (few obs/subject) a free η can
            // sample a near-zero spread early — once that feeds back into the
            // Cholesky MH proposal the scale collapses and the chain can never
            // re-inflate Ω, dumping between-subject variability into residual
            // error. FIX-ed entries were just restored from the template and are
            // left exactly as declared.
            floor_omega_diagonal(
                &mut state.omega_mat,
                &init_params.omega_fixed,
                SAEM_OMEGA_DIAG_FLOOR,
            );

            // #895: cap the iiv_on_ruv Ω variance so a runaway ridge can't inflate
            // ω_RUV without bound (the original report saw ~49). No-op for a
            // well-posed fit; a correlation-preserving rescale keeps Ω PD.
            if let (Some(k), Some(log_cap)) = (model.residual_error_eta, ruv_omega_cap) {
                apply_ruv_omega_cap(&mut state.omega_mat, k, log_cap, &init_params.omega_fixed);
            }

            // ---- Step 3b: Omega_iov (analytic, IOV only) ----
            // Apply the SA sufficient statistic, zeroing structural off-diagonals
            // and restoring FIX-ed kappa entries, mirroring the BSV omega treatment.
            if n_kappa > 0 {
                if let Some(omega_iov_ref) = init_params.omega_iov.as_ref() {
                    state.omega_iov_mat = state.s2_iov.clone();
                    // Zero structurally-absent off-diagonals.
                    for i in 0..n_kappa {
                        for j in 0..n_kappa {
                            if !omega_iov_ref.free_mask[(i, j)] {
                                state.omega_iov_mat[(i, j)] = 0.0;
                            }
                        }
                    }
                    // Restore FIX-ed kappa rows/columns from the template.
                    for i in 0..n_kappa {
                        for j in 0..n_kappa {
                            let fi = init_params.kappa_fixed.get(i).copied().unwrap_or(false);
                            let fj = init_params.kappa_fixed.get(j).copied().unwrap_or(false);
                            if fi || fj {
                                state.omega_iov_mat[(i, j)] = omega_iov_ref.matrix[(i, j)];
                            }
                        }
                    }
                    // Floor diagonal to stay positive-definite.
                    for i in 0..n_kappa {
                        if state.omega_iov_mat[(i, i)] < 1e-8 {
                            state.omega_iov_mat[(i, i)] = 1e-8;
                        }
                    }
                }
            }
        }

        // ---- Step 4: M-step theta, sigma (lightweight NLopt, warm-started) ----
        // Only run every few iterations during exploration to save time
        let run_mstep = k <= 5 || k % 3 == 0 || k > k1;
        let kappas_for_mstep = if n_kappa > 0 {
            Some(state.kappas.as_slice())
        } else {
            None
        };
        if run_mstep {
            let mstep_maxiter = if k <= k1 { 3 } else { 5 }; // more precise in convergence phase

            if use_closed_form_mstep && saem_mix.is_none() {
                // Closed-form EM M-step for log-mu-referenced thetas.
                //
                // Model: log(P_i) = log(TVP) + η_i, η_i ~ N(0, ω²).
                // The complete-data log-likelihood is maximised at
                //     log(TVP)_new = log(TVP)_old + mean_i(η_i)
                // and SAEM applies the stochastic-approximation step size γ:
                //     log(TVP)_new = log(TVP)_old + γ · mean_i(η_i)
                // After the update, η_i is re-centred by `mean(η)` so the
                // sufficient statistic for ω is taken from zero-mean residuals
                // (ω is updated from `s2` *after* the next MH step, but
                // re-centring keeps `state.etas` consistent with the new TVP
                // for the rest of this iteration's NLL cache refresh).
                let n_subj = state.etas.len() as f64;
                let mut temp_theta_lower = log_theta_lower.clone();
                let mut temp_theta_upper = log_theta_upper.clone();
                let mut n_pinned: u64 = 0;
                for &(theta_idx, eta_idx) in &mu_ref_pairs {
                    if init_params
                        .theta_fixed
                        .get(theta_idx)
                        .copied()
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    let mean_eta: f64 = state.etas.iter().map(|e| e[eta_idx]).sum::<f64>() / n_subj;
                    let log_theta_before = log_theta[theta_idx];
                    log_theta[theta_idx] = (log_theta_before + gamma * mean_eta)
                        .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                    // Re-centre etas by the *actual* shift applied to log_theta,
                    // not by `gamma * mean_eta` directly: when the update is
                    // clamped at a bound the realised delta is smaller, and
                    // shifting etas by the unclamped quantity would break
                    // log(P_i) = log(TVP) + η_i until the next MH refresh.
                    let delta = log_theta[theta_idx] - log_theta_before;
                    for e in state.etas.iter_mut() {
                        e[eta_idx] -= delta;
                    }
                    // Pin so NLopt leaves the closed-form value unchanged.
                    temp_theta_lower[theta_idx] = log_theta[theta_idx];
                    temp_theta_upper[theta_idx] = log_theta[theta_idx];
                    n_pinned += 1;
                }
                // Each pinned mu-ref dim avoids 2 obs_nll_sum calls per NLopt
                // gradient request, capped at `mstep_maxiter` requests. FIXed
                // thetas are not pinned by the closed form (NLopt sees them as
                // FIXed via the regular bounds path) so they aren't counted.
                mstep_grad_step_evals_saved += 2 * mstep_maxiter as u64 * n_pinned;

                // NLopt for non-mu-ref thetas (pinned) and sigma.
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &state.etas,
                    kappas_for_mstep,
                    &log_theta,
                    &log_sigma,
                    &temp_theta_lower,
                    &temp_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                    &theta_packs_log_mask,
                    // Closed-form branch is never taken for a mixture (disabled
                    // above), so no class guard is needed here.
                    None,
                );
                damp_mstep(&mut log_theta, &theta_new, gamma_theta);
                damp_mstep(&mut log_sigma, &sigma_new, gamma_theta);
            } else if use_closed_form_mstep {
                // ---- Closed-form EM M-step under a mixture (#996) ----
                //
                // Runs the general class-resolved update
                //     log theta_k += gamma * mean_{i : c_i = k}(eta_i)
                // over `mix_mu_ref_pairs`, weighting each subject by a one-hot
                // indicator of the class SAEM drew for it this E-step. That pair
                // set is filtered to **class-shared** anchors upstream (see its
                // construction for the measurement that rules out the switched
                // case here), so in practice every class maps to one theta and
                // this reduces to the pooled `mean_i(eta_i)` of the
                // single-population branch above -- same accumulation order, same
                // result. The class-resolved form is kept because it is what makes
                // that reduction exact rather than a second copy of the formula.
                let mix = saem_mix.as_ref().expect("mixture arm without saem_mix");
                let resp: Vec<Vec<f64>> = mix
                    .classes
                    .iter()
                    .map(|&c| {
                        let mut r = vec![0.0f64; mix.n_classes];
                        if c < r.len() {
                            r[c] = 1.0;
                        }
                        r
                    })
                    .collect();
                let mut temp_theta_lower = log_theta_lower.clone();
                let mut temp_theta_upper = log_theta_upper.clone();
                let mut n_pinned: u64 = 0;
                for pair in &mix_mu_ref_pairs {
                    let eta_idx = pair.eta_idx;
                    let means = mixture_mu_ref_means(n_theta, &pair.theta_idx, &resp, |i, _c| {
                        state.etas[i][eta_idx]
                    });
                    // Realised per-theta shift, so the eta re-centering below uses
                    // the *clamped* delta (see the single-population branch).
                    let mut delta = vec![0.0f64; n_theta];
                    for (theta_idx, mean_eta) in means.iter().enumerate() {
                        // `None` means no subject was drawn into any class this
                        // theta serves (label switching, or a class that won nobody
                        // this iteration): hold theta_k, let the next E-step move it.
                        let Some(mean_eta) = mean_eta else { continue };
                        if init_params
                            .theta_fixed
                            .get(theta_idx)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let before = log_theta[theta_idx];
                        log_theta[theta_idx] = (before + gamma * mean_eta)
                            .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                        delta[theta_idx] = log_theta[theta_idx] - before;
                        // Pin so NLopt leaves the closed-form value unchanged.
                        temp_theta_lower[theta_idx] = log_theta[theta_idx];
                        temp_theta_upper[theta_idx] = log_theta[theta_idx];
                        n_pinned += 1;
                    }
                    // Re-centre each subject by the shift applied to *its* class's
                    // theta, so `log(P_i) = log(TVP_{c_i}) + eta_i` still holds for
                    // the rest of this iteration's NLL cache refresh.
                    for (i, e) in state.etas.iter_mut().enumerate() {
                        let c = mix.classes[i];
                        if let Some(&t) = pair.theta_idx.get(c) {
                            e[eta_idx] -= delta[t];
                        }
                    }
                }
                mstep_grad_step_evals_saved += 2 * mstep_maxiter as u64 * n_pinned;

                // NLopt for the remaining (non-mu-ref) thetas and sigma, still
                // class-guarded so any class-switched theta outside the mu-ref set
                // is estimated from its own members (#985).
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &state.etas,
                    kappas_for_mstep,
                    &log_theta,
                    &log_sigma,
                    &temp_theta_lower,
                    &temp_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                    &theta_packs_log_mask,
                    Some(MixMstep {
                        classes: mix.classes.as_slice(),
                        class_sigma_over: &mix_sigma_over,
                    }),
                );
                damp_mstep(&mut log_theta, &theta_new, gamma_theta);
                damp_mstep(&mut log_sigma, &sigma_new, gamma_theta);
            } else {
                // mu_referencing = false (or a mixture whose class thetas could not
                // be class-aware mu-referenced): full NLopt M-step for all thetas
                // + sigma. For a mixture, `mstep_classes` guards each subject so
                // class-switched thetas are estimated per class (#985).
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &state.etas,
                    kappas_for_mstep,
                    &log_theta,
                    &log_sigma,
                    &log_theta_lower,
                    &log_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                    &theta_packs_log_mask,
                    saem_mix.as_ref().map(|m| MixMstep {
                        classes: m.classes.as_slice(),
                        class_sigma_over: &mix_sigma_over,
                    }),
                );
                damp_mstep(&mut log_theta, &theta_new, gamma_theta);
                damp_mstep(&mut log_sigma, &sigma_new, gamma_theta);
            }

            // #895: clamp any RUV-scaled residual σ that the M-step pushed past its
            // growth cap. A no-op for a well-posed fit (σ stays well below the cap),
            // so the un-capped trajectory is reproduced bit-for-bit; only a runaway
            // riding the σ × ω_RUV ridge is pulled back.
            for (i, cap) in ruv_sigma_caps.iter().enumerate() {
                if let Some(cap) = cap {
                    if log_sigma[i] > *cap {
                        log_sigma[i] = *cap;
                    }
                }
            }

            state.theta = (0..n_theta)
                .map(|i| unpack_theta(i, log_theta[i]))
                .collect();
            state.sigma_vals = log_sigma.iter().map(|&v| v.exp()).collect();

            // ---- Step 4b: M-step for the mixing coefficients (#985) ----
            // The mixing thetas do not enter the residual/η likelihood, so the
            // θ/σ M-step above leaves them untouched. Update them from the SA
            // responsibility average `r̄_ik` (constant closed form or covariate
            // logistic fit), then re-sync `log_theta` for those indices.
            if let Some(mix) = saem_mix.as_ref() {
                crate::estimation::saem_mixture::mstep_mixing(
                    model,
                    population,
                    mix,
                    &mut state.theta,
                    &init_params.theta_lower,
                    &init_params.theta_upper,
                    mstep_maxiter,
                );
                for &j in &mix.mixing_theta_idx {
                    log_theta[j] = if theta_packs_log_mask[j] {
                        state.theta[j].max(1e-12).ln()
                    } else {
                        state.theta[j]
                    };
                }
            }
        }

        // ---- Update NLL cache (parallelized, needed for MH acceptance ratios) ----
        let omega_upd = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        // Mixture (#985): refresh each class's Ω/σ from the just-updated base and
        // evaluate every subject's cached NLL under its drawn class (guard + class
        // Ω/σ), so the next iteration's MH acceptance baseline is class-consistent.
        let mix_classes: Option<Vec<usize>> = saem_mix.as_ref().map(|m| m.classes.clone());
        let (mix_omegas, mix_sigmas): (Vec<OmegaMatrix>, Vec<Vec<f64>>) =
            if let Some(mix) = saem_mix.as_mut() {
                mix.sync_base(&omega_upd, &state.sigma_vals);
                (
                    (0..mix.n_classes)
                        .map(|c| mix.class_omega(c).clone())
                        .collect(),
                    (0..mix.n_classes)
                        .map(|c| mix.class_sigma(c).to_vec())
                        .collect(),
                )
            } else {
                (Vec::new(), Vec::new())
            };
        if n_kappa > 0 {
            // IOV NLL cache refresh — sequential rather than rayon-parallel.
            // individual_nll_iov is cheap (analytical PK, few occasions) and
            // the sequential loop avoids a second rayon scatter/gather.
            // Parallelise here if profiling shows a bottleneck.
            let omega_iov_upd = init_params.omega_iov.as_ref().map(|iov_ref| {
                OmegaMatrix::from_matrix_with_mask(
                    state.omega_iov_mat.clone(),
                    iov_ref.eta_names.clone(),
                    iov_ref.diagonal,
                    iov_ref.free_mask.clone(),
                )
            });
            let new_nlls: Vec<f64> = (0..n_subjects)
                .map(|i| {
                    let cls = mix_classes.as_ref().map(|c| c[i]);
                    let _g =
                        cls.map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                    let (omega_i, sigma_i): (&OmegaMatrix, &[f64]) = match cls {
                        Some(c) => (&mix_omegas[c], &mix_sigmas[c]),
                        None => (&omega_upd, state.sigma_vals.as_slice()),
                    };
                    individual_nll_iov(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        &state.etas[i],
                        &state.kappas[i],
                        omega_i,
                        omega_iov_upd.as_ref(),
                        sigma_i,
                    )
                })
                .collect();
            state.nll_cache = new_nlls;
        } else {
            use rayon::prelude::*;
            // map_init lets each rayon worker keep one `EventPkParams`
            // scratch alive across every subject it handles, the same
            // pattern as the MH step above. Without it, the per-iter
            // refresh was allocating n_subj scratch buffers per outer
            // iter on TV-cov data.
            let mix_omegas_ref = &mix_omegas;
            let mix_sigmas_ref = &mix_sigmas;
            let mix_classes_ref = mix_classes.as_ref();
            let new_nlls: Vec<f64> = state
                .etas
                .par_iter()
                .enumerate()
                .map_init(EventPkParams::default, |scratch, (i, eta)| {
                    let cls = mix_classes_ref.map(|c| c[i]);
                    let _g =
                        cls.map(|c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
                    let (omega_i, sigma_i): (&OmegaMatrix, &[f64]) = match cls {
                        Some(c) => (&mix_omegas_ref[c], &mix_sigmas_ref[c]),
                        None => (&omega_upd, state.sigma_vals.as_slice()),
                    };
                    individual_nll_into(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        eta,
                        omega_i,
                        sigma_i,
                        scratch,
                    )
                })
                .collect();
            state.nll_cache = new_nlls;
        }

        // #903 review: re-anchor the iiv_on_ruv growth caps to the data-informed
        // σ/ω_RUV reached by the end of the exploration phase, taking the *looser*
        // of the init-based and settled-value-based ceilings. The caps are pure
        // runaway backstops (~20× a sensible scale); anchoring them to the raw
        // user *init* alone would spuriously clamp — and falsely warn about — a
        // well-posed fit started from a σ/ω guess many-fold below the truth. Only
        // ever loosens (`max`), so a genuine post-exploration runaway is still
        // bounded, and it preserves the bit-for-bit trajectory of a fit that never
        // approaches either ceiling.
        if k == k1 {
            reanchor_ruv_sigma_caps(&mut ruv_sigma_caps, &log_sigma, &log_sigma_upper);
            if let Some(kr) = model.residual_error_eta {
                ruv_omega_cap = reanchor_ruv_omega_cap(ruv_omega_cap, state.omega_mat[(kr, kr)]);
            }
        }

        // ---- Adapt MH step sizes ----
        if state.steps_since_adapt >= adapt_interval {
            for i in 0..n_subjects {
                // Use the actual per-subject proposal count as the denominator so
                // that MH-fallback subjects in HMC mode (which run n_mh_steps
                // proposals) are not scaled by the HMC denominator of 1.
                let total_proposals = state.proposal_counts[i].max(1);
                let rate = state.accept_counts[i] as f64 / total_proposals as f64;
                if rate > target_accept_rate {
                    state.step_scales[i] = (state.step_scales[i] * 1.1).min(5.0);
                } else {
                    state.step_scales[i] = (state.step_scales[i] * 0.9).max(0.01);
                }
                state.accept_counts[i] = 0;
                state.proposal_counts[i] = 0;
                // Adapt per-eta componentwise kernel scales toward the 1-D
                // optimum (~0.44 acceptance, Roberts & Rosenthal 2001).
                // Each eta adapts independently so that etas with very
                // different posterior precision (e.g. FREM covariate etas
                // with near-deterministic data vs broad PK etas) can each
                // reach their optimal step size.  The floor is 1e-6 (not
                // 0.01) to accommodate near-deterministic etas whose
                // posterior SD may be orders of magnitude below √Ω_jj.
                if n_cw_sweeps > 0 {
                    for j in 0..n_eta {
                        let cw_total = state.cw_proposal_counts[i][j].max(1);
                        let cw_rate = state.cw_accept_counts[i][j] as f64 / cw_total as f64;
                        if cw_rate > CW_TARGET_ACCEPT {
                            state.cw_step_scales[i][j] =
                                (state.cw_step_scales[i][j] * 1.1).min(5.0);
                        } else {
                            state.cw_step_scales[i][j] =
                                (state.cw_step_scales[i][j] * 0.9).max(1e-6);
                        }
                        state.cw_accept_counts[i][j] = 0;
                        state.cw_proposal_counts[i][j] = 0;
                    }
                }
                // Adapt kappa step sizes (target 40% for MH on kappas).
                if n_kappa > 0 {
                    let kappa_total = state.kappa_proposal_counts[i].max(1);
                    let kappa_rate = state.kappa_accept_counts[i] as f64 / kappa_total as f64;
                    if kappa_rate > 0.40 {
                        state.kappa_step_scales[i] = (state.kappa_step_scales[i] * 1.1).min(5.0);
                    } else {
                        state.kappa_step_scales[i] = (state.kappa_step_scales[i] * 0.9).max(0.01);
                    }
                    state.kappa_accept_counts[i] = 0;
                    state.kappa_proposal_counts[i] = 0;
                }
            }
            state.steps_since_adapt = 0;
        }

        // ---- Verbose output + optimizer trace ----
        {
            let phase = if k <= k1 { "explore" } else { "converge" };
            let cond_nll: f64 = state.nll_cache.iter().sum();
            // Combined (block + componentwise) acceptance for this iteration (#895).
            // Reporting the block kernel alone reads a misleading 0% for FREM-scale
            // Ω — the near-deterministic covariate ETAs reject every joint move —
            // even though the componentwise sweep is mixing the chain fine. The
            // combined rate is the honest E-step mixing diagnostic.
            let mh_accept_rate: f64 = iter_acc as f64 / iter_prop.max(1) as f64;

            if verbose && (k == 1 || k % 50 == 0 || k == n_iter) {
                eprintln!(
                    "  SAEM iter {:>4}/{} [{}] γ={:.3}  condNLL={:.3}",
                    k, n_iter, phase, gamma, cond_nll
                );
            }

            // Per-coordinate estimates (natural scale) for the trace's `val:*`
            // columns (#640). SAEM has no OFV gradient, so `grad:*` are NA.
            // Guard the per-iteration Vec build on `is_active` so it costs
            // nothing on the default (trace-off) path across the many SAEM
            // iterations (#640 review).
            if crate::estimation::trace::is_active() {
                let iov = init_params
                    .omega_iov
                    .as_ref()
                    .map(|m| (&state.omega_iov_mat, m.diagonal));
                let values = crate::estimation::parameterization::coordinate_values_raw(
                    &state.theta,
                    &state.omega_mat,
                    init_params.omega.diagonal,
                    &state.sigma_vals,
                    iov,
                );
                crate::estimation::trace::write_saem(
                    k,
                    phase,
                    cond_nll,
                    gamma,
                    mh_accept_rate,
                    &values,
                );
            }

            // Checkpoint (#755): snapshot the current SAEM estimates periodically
            // so an interrupted run resumes near here. `cond_nll` stands in for
            // the OFV (the FOCE OFV is not yet available mid-SAEM).
            if crate::io::checkpoint::is_due() {
                let snap = saem_state_to_params(&state, init_params, n_kappa);
                let packed = crate::estimation::parameterization::pack_params(&snap);
                crate::io::checkpoint::maybe_write(k, cond_nll, &packed);
            }
        }
    }

    // If the user cancelled mid-run the loop broke early; skip the final
    // EBE/OFV computation (which iterates over every subject) and abort.
    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    if verbose {
        eprintln!("SAEM iterations complete. Computing final EBEs and OFV...");
    }

    // ---- Post-SAEM: build final parameters ----
    let mut final_params = saem_state_to_params(&state, init_params, n_kappa);
    // Mixture (#985): carry the estimated per-class Ω/σ overrides so the final
    // OFV, EBEs, and covariance step see the mixture structure. `mp`'s base was
    // synced from the final Ω/σ in the last NLL-cache refresh.
    if let Some(mix) = saem_mix.as_ref() {
        let mut mp = mix.mp.clone();
        // SAEM holds the per-class σ overrides at their inits (warned above), so
        // they are FIX as far as this fit is concerned. Marking them keeps the
        // covariance step from building an FD-Hessian row for a coordinate SAEM
        // never optimised — evaluated off its stationary point that row can push
        // an otherwise-PD Hessian into the eigen-floor/SIR fallback and degrade
        // every other SE (#987 review). Their SEs report as 0, matching FIX.
        mp.sigma_override_fixed = vec![true; mp.sigma_override_addr.len()];
        final_params.mixture = Some(mp);
    }

    if combined_additive_sigma_at_floor(model, &final_params) {
        warnings
            .push("SAEM combined-error additive sigma collapsed to its lower bound.".to_string());
    }

    // #895: warn when the E-step never mixed. A combined (block + componentwise)
    // post-burn-in acceptance rate near zero means the sampled ETAs barely moved
    // from their starting values, so the M-step ran on degenerate sufficient
    // statistics and the Ω/σ estimates are unreliable (the classic FREM-scale
    // "0% acceptance" failure).
    if let Some(w) = saem_mixing_warning(cum_mh_acc, cum_mh_prop) {
        warnings.push(w);
    }

    // #895: warn when a free RUV-scaled residual σ ended pinned against the
    // iiv_on_ruv growth cap. That signals the σ × ω_RUV ridge is poorly
    // identified from the data alone; the cap kept σ bounded but the split
    // between σ and ω_RUV should not be trusted — fix one of them (e.g. σ at a
    // known value) or drop the IIV on the residual error.
    for (i, cap) in ruv_sigma_caps.iter().enumerate() {
        if let Some(cap) = cap {
            if final_params.sigma.values[i].max(1e-300).ln() >= cap - 1e-6 {
                warnings.push(format!(
                    "SAEM residual sigma '{}' hit its iiv_on_ruv growth cap (≈{:.4}); the \
                     sigma × omega_RUV ridge is weakly identified and the estimates are not \
                     trustworthy. The most reliable fix is to FIX sigma at a known value (e.g. \
                     from an IMP/IMPMAP run) and re-fit — omega_RUV then recovers; alternatively \
                     FIX the RUV omega or remove iiv_on_ruv.",
                    final_params
                        .sigma
                        .names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("sigma[{i}]")),
                    cap.exp()
                ));
            }
        }
    }

    // #895: warn when the iiv_on_ruv Ω variance ended pinned against its growth
    // cap — the ω_RUV half of the same ridge instability. Same guidance: fix σ (or
    // the RUV Ω) and re-fit.
    if let (Some(k), Some(log_cap)) = (model.residual_error_eta, ruv_omega_cap) {
        if final_params.omega.matrix[(k, k)].max(1e-300).ln() >= log_cap - 1e-6 {
            warnings.push(format!(
                "SAEM iiv_on_ruv omega '{}' hit its growth cap (≈{:.4}); the sigma × omega_RUV \
                 ridge is weakly identified and the estimates are not trustworthy. FIX sigma at \
                 a known value (e.g. from an IMP/IMPMAP run) and re-fit — omega_RUV then recovers.",
                final_params
                    .omega
                    .eta_names
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| format!("eta[{k}]")),
                log_cap.exp()
            ));
        }
    }

    // ---- Final EBEs + OFV ----
    // For a mixture (#985), the population objective is the K-fold marginal
    // `−2 Σ_i log Σ_k p_ik e^{−nll_ik}`, and the reported EBEs / κ̂ are the
    // MIXEST class's — both produced by `mixture_ofv` (the same routine the
    // FOCEI mixture path uses), so SAEM and FOCEI report comparable OFVs and
    // per-subject posteriors. Otherwise the single-population FOCE approximation.
    let (eta_hats, h_matrices, final_kappas, ofv, mixture_posteriors) =
        if final_params.mixture.is_some() {
            let meval = crate::estimation::mixture::mixture_ofv(
                model,
                population,
                &final_params,
                options,
                None,
            );
            let posteriors = crate::estimation::outer_optimizer::MixturePosteriors {
                pmix: meval.pmix.clone(),
                mixest: meval.mixest.clone(),
            };
            (
                meval.mixest_etas,
                meval.mixest_h_mats,
                meval.mixest_kappas,
                meval.ofv,
                Some(posteriors),
            )
        } else {
            let warm_etas: Vec<DVector<f64>> = state
                .etas
                .iter()
                .map(|e| DVector::from_column_slice(e))
                .collect();
            let saem_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
            let (eta_hats, h_matrices, _, final_kappas) = run_inner_loop_warm(
                model,
                population,
                &final_params,
                options.inner_maxiter,
                options.inner_tol,
                Some(&warm_etas),
                Some(&saem_final_mu_k),
                0, // SAEM: no EBE convergence tracking
                0, // SAEM final EBE is warm-started; no inner multi-start
            );
            let ofv = 2.0
                * pop_nll(
                    model,
                    population,
                    &final_params,
                    &eta_hats,
                    &h_matrices,
                    &final_kappas,
                    options.interaction,
                );
            (eta_hats, h_matrices, final_kappas, ofv, None)
        };

    // ---- Report OFV *before* the covariance step (#893) ----
    // SAEM only learns its OFV here (the final FOCE approximation); the covariance
    // step that follows can be the most expensive part of the run. Print the OFV
    // first so a CLI user can judge the fit and interrupt (Ctrl-C) before paying
    // for the covariance matrix when the OFV already rules the run out.
    if verbose {
        eprintln!("{}", saem_final_ofv_report(ofv));
    }

    // ---- Covariance step ----
    let packed = pack_params(&final_params);
    let out = crate::estimation::covariance::run_covariance_step(
        &packed,
        &final_params,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &final_kappas,
        options,
        verbose.then_some("Running covariance step..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
    } = out;
    warnings.extend(cov_warnings);

    let saem_mu_ref_m_step_evals_saved = if use_closed_form_mstep {
        Some(mstep_grad_step_evals_saved)
    } else {
        None
    };

    let saem_n_subjects_hmc = if using_hmc {
        Some(hmc_subjects.iter().filter(|&&b| b).count())
    } else {
        None
    };

    // ---- Post-fit conditional-distribution pass (opt-in, #257) ----
    // Characterise each subject's p(η_i | y_i; θ̂) by MCMC at the fixed
    // population parameters, warm-started from the EBE mode (`eta_hats`).
    // The conditional-distribution pass samples p(η_i | y_i) at fixed θ̂ with a
    // single-population target; it is not class-aware, so skip it for a mixture
    // (#985) rather than characterise the posterior under the wrong class.
    if options.saem_conddist && final_params.mixture.is_some() {
        warnings.push(
            "SAEM conditional-distribution pass (conddist) is not yet mixture-aware; skipping \
             it for this mixture fit (#985)."
                .to_string(),
        );
    }
    let cond_dist = if options.saem_conddist && final_params.mixture.is_none() {
        if verbose {
            eprintln!(
                "Running SAEM conditional-distribution pass ({} samples/subject, {} burn-in)...",
                options.saem_conddist_nsamp, options.saem_conddist_burnin
            );
        }
        Some(
            crate::estimation::saem_conddist::run_conditional_distribution(
                model,
                population,
                &final_params,
                &eta_hats,
                &final_kappas,
                options,
            ),
        )
    } else {
        None
    };

    Ok(OuterResult {
        params: final_params,
        ofv,
        // A finite-but-enormous OFV is the bounded blowup of a runaway, not a
        // converged fit — guard against it the same way IMP/IMPMAP does, since
        // SAEM is commonly the first phase of a SAEM→IMP chain (issue #528).
        converged: crate::estimation::impmap::objective_converged(ofv),
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
        kappas: final_kappas,
        covariance_matrix,
        covariance_wall_time_secs,
        warnings,
        saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        final_gradient: None,
        sir_fallback_proposal,
        impmap_trace: None,
        bayes: None,
        cond_dist,
        packed_estimate: None,
        mixture_posteriors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── #996 end-to-end helpers: a tiny 2-class mixture dataset ──────────
    //
    // Two weight groups with well-separated clearances, so the class draw is
    // stable and a handful of iterations is enough to exercise the M-step.

    /// Small 1-cpt IV dataset, `n_per` subjects at each of two weights.
    fn mix996_csv(n_per: usize) -> String {
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        let mut sid = 0;
        for (g, &wt) in [60.0_f64, 90.0].iter().enumerate() {
            let cl = if g == 0 { 1.0 } else { 3.0 };
            for _ in 0..n_per {
                sid += 1;
                s.push_str(&format!("{sid},0,0,100,1,1,{wt}\n"));
                for (ti, t) in [0.5_f64, 1.0, 2.0, 4.0].iter().enumerate() {
                    let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                    let dv = c * (1.0 + 0.02 * ((sid + ti) as f64).sin());
                    s.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{wt}\n"));
                }
            }
        }
        s
    }

    fn mix996_pop(n_per: usize) -> Population {
        use std::io::Write;
        let csv = mix996_csv(n_per);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    /// Two-class mixture. `v_expr` picks whether V carries a class-shared
    /// mu-referenced ETA (`TVV * exp(ETA_V)`) or none (`TVV`).
    fn mix996_model(v_expr: &str) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V ~ 0.04 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = {v_expr}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("mixture model parses")
    }

    /// Single-population 1-cpt IV model whose `V` either carries a
    /// mu-referenceable ETA or is a bare fixed-effect-only θ — the shape that
    /// makes SAEM's η-frozen numerical M-step the *only* channel that can move
    /// it (the FREM `FRD1` absorption fraction of #406).
    fn noeta_model(v_expr: &str) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V ~ 0.04 FIX
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = {v_expr}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("model parses")
    }

    fn saem_noeta_warnings(v_expr: &str) -> Vec<String> {
        let model = noeta_model(v_expr);
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(406);
        opts.run_covariance_step = false;
        crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("SAEM Ok")
            .warnings
    }

    /// An estimated θ with no ETA is not mu-referenced, so it never gets the
    /// γ-damped closed-form shift and is left to the η-frozen numerical M-step.
    /// SAEM must name it, the way IMP/IMPMAP already do (#406) — on the FREM
    /// `iiv_on_ruv` reprex that channel drove `TVFRD1` 0.383 → 0.039 against
    /// IMP 0.311 / IMPMAP 0.318 / NONMEM 0.394, and attaching an ETA recovered
    /// 0.313.
    #[test]
    fn saem_warns_when_an_estimated_theta_has_no_eta() {
        let ws = saem_noeta_warnings("TVV");
        let hit = ws
            .iter()
            .find(|w| w.contains("NO associated ETA"))
            .unwrap_or_else(|| panic!("expected the no-ETA advisory, got {ws:?}"));
        assert!(
            hit.starts_with("SAEM:") && hit.contains("TVV"),
            "advisory must be SAEM-labelled and name TVV: {hit}"
        );
        assert!(
            !hit.contains("TVCL"),
            "TVCL is mu-referenced and must not be named: {hit}"
        );
    }

    /// The undamped assignment must be reproduced byte-for-byte at `γ >= 1.0`,
    /// and a pinned dimension (where NLopt returns `new == cur`) must be a no-op
    /// at every γ — those two properties are what keep a fit with no free
    /// numerical θ identical to its pre-#1011 result.
    #[test]
    fn damp_mstep_is_a_robbins_monro_blend() {
        let new = [1.0_f64, -2.0, 0.5];

        // γ = 1 (and above) → straight assignment, bit-for-bit.
        for g in [1.0_f64, 1.5] {
            let mut cur = [10.0_f64, 10.0, 10.0];
            damp_mstep(&mut cur, &new, g);
            assert_eq!(cur, new, "γ = {g} must assign outright");
        }

        // 0 < γ < 1 → cur += γ·(new − cur).
        let mut cur = [0.0_f64, 0.0, 0.0];
        damp_mstep(&mut cur, &new, 0.25);
        for (c, n) in cur.iter().zip(new.iter()) {
            assert!(
                (c - 0.25 * n).abs() < 1e-15,
                "expected {}, got {c}",
                0.25 * n
            );
        }

        // γ = 0 → frozen.
        let mut cur = [3.0_f64, 4.0, 5.0];
        let before = cur;
        damp_mstep(&mut cur, &new, 0.0);
        assert_eq!(cur, before);

        // A pinned dim (new == cur) is untouched at any γ — no drift from the
        // closed-form mu-ref value the M-step was told to leave alone.
        for g in [0.0_f64, 0.03, 0.5, 1.0] {
            let mut cur = [7.0_f64; 3];
            damp_mstep(&mut cur, &[7.0; 3], g);
            assert_eq!(cur, [7.0; 3], "pinned dim moved at γ = {g}");
        }
    }

    /// The SA schedule: off entirely when NLopt has no θ to estimate (so those
    /// fits keep their pre-#1011 numbers), capped in exploration, full decaying
    /// γ in convergence.
    #[test]
    fn mstep_sa_step_caps_exploration_and_frees_convergence() {
        let d = MSTEP_SA_MAX_STEP;

        // No numerically-estimated θ → undamped, in both phases.
        assert_eq!(mstep_sa_step(false, true, 1.0, d), 1.0);
        assert_eq!(mstep_sa_step(false, false, 0.002, d), 1.0);

        // Exploration (γ = 1.0) is capped.
        assert_eq!(mstep_sa_step(true, true, 1.0, d), d);
        // ...but a γ already below the cap is not raised to it.
        assert_eq!(mstep_sa_step(true, true, 0.001, d), 0.001);
        // Convergence uses the full decaying γ, uncapped.
        assert_eq!(mstep_sa_step(true, false, 0.5, d), 0.5);
        assert_eq!(mstep_sa_step(true, false, 0.002, d), 0.002);

        // `mstep_damping` overrides the cap.
        assert_eq!(mstep_sa_step(true, true, 1.0, 0.005), 0.005);

        // Its documented "off" value of 1.0 must restore the undamped pre-#1011
        // assignment in BOTH phases — convergence included, where the schedule
        // would otherwise still damp at γ = 1/(k−k1). Regression: `cap = 1.0`
        // first only lifted the exploration cap, which left the reprex at
        // TVFRD1 0.047 instead of its true undamped 0.039.
        assert_eq!(mstep_sa_step(true, true, 1.0, 1.0), 1.0);
        assert_eq!(mstep_sa_step(true, false, 0.002, 1.0), 1.0);
        assert_eq!(mstep_sa_step(true, false, 0.5, 1.0), 1.0);

        // The θ cap must be at least as tight as Ω's: the θ channel has no
        // closed-form averaged alternative, so it is the more exposed of the two.
        assert!(MSTEP_SA_MAX_STEP <= OMEGA_SA_MAX_STEP);
    }

    /// The damping gate (#1011). A mixture is vetoed outright; otherwise the
    /// damping runs exactly when NLopt has a θ left to estimate, so an
    /// all-mu-referenced or all-`FIX` fit keeps its pre-#1011 numbers.
    #[test]
    fn damps_numerical_mstep_only_when_nlopt_has_a_theta() {
        let free = [false, false, false];

        // One free, unpinned θ (the #1011 shape: a theta with no ETA) → damp.
        assert!(damps_numerical_mstep(false, 3, &free, &[0, 1]));
        // Every θ pinned by the closed-form mu-ref shift → nothing to damp.
        assert!(!damps_numerical_mstep(false, 3, &free, &[0, 1, 2]));
        // Every θ FIXed → nothing to damp.
        assert!(!damps_numerical_mstep(false, 3, &[true, true, true], &[]));
        // Mixed: the only unpinned θ is also FIXed → nothing to damp.
        assert!(!damps_numerical_mstep(
            false,
            3,
            &[false, false, true],
            &[0, 1]
        ));
        // No mu-referencing at all → every θ is numerical.
        assert!(damps_numerical_mstep(false, 3, &free, &[]));
        // No thetas at all (σ-only M-step) → undamped, as before #1011.
        assert!(!damps_numerical_mstep(false, 0, &[], &[]));

        // A mixture is vetoed even with a free unpinned θ — its class typical
        // values must separate from a common start before the class assignments
        // settle, and damping that stalls it.
        assert!(!damps_numerical_mstep(true, 3, &free, &[]));
        assert!(!damps_numerical_mstep(true, 3, &free, &[0, 1]));
    }

    /// ...and stays quiet once every estimated θ carries one, so the advisory
    /// keeps its signal.
    #[test]
    fn saem_no_eta_advisory_silent_when_every_theta_is_mu_referenced() {
        let ws = saem_noeta_warnings("TVV * exp(ETA_V)");
        assert!(
            !ws.iter().any(|w| w.contains("NO associated ETA")),
            "no advisory when every θ is mu-referenced, got {ws:?}"
        );
    }

    #[test]
    fn saem_mixture_uses_closed_form_for_class_shared_mu_ref() {
        // Before #996 a mixture disabled mu-referencing wholesale, so even a
        // class-*shared* typical value (V = TVV * exp(ETA_V)) went through the
        // numerical M-step. It now takes the pooled closed-form shift, which the
        // saved-evaluation counter reports.
        let model = mix996_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert!(
            res.saem_mu_ref_m_step_evals_saved.unwrap_or(0) > 0,
            "class-shared mu-ref must take the closed-form M-step: {:?}",
            res.saem_mu_ref_m_step_evals_saved
        );
        // ...and the MIXNUM-switched clearances must be named as staying numerical.
        assert!(
            res.warnings.iter().any(|w| {
                w.contains("MIXNUM-switched typical value")
                    && w.contains("TVCL1")
                    && w.contains("TVCL2")
            }),
            "expected the #996 switched-theta warning, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn saem_mixture_mu_referencing_off_suppresses_the_muref_advisories() {
        // With mu_referencing off every θ goes through the numerical M-step by
        // construction, so advising the user to switch estimator to get a
        // closed-form shift they turned off is noise (#996 review).
        let model = mix996_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(996);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("MIXNUM-switched typical value")),
            "no switched-theta advisory with mu_referencing off, got {:?}",
            res.warnings
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("packed on the identity scale")),
            "no identity-pack advisory with mu_referencing off, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn saem_mixture_without_shared_mu_ref_stays_on_numerical_mstep() {
        // Only the class-switched CL is mu-ref-shaped; V carries no ETA. SAEM
        // takes no closed-form pair, so the whole θ/σ M-step stays numerical.
        let model = mix996_model("TVV");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Saem;
        opts.saem_n_exploration = 4;
        opts.saem_n_convergence = 2;
        opts.saem_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("SAEM Ok");
        assert_eq!(res.saem_mu_ref_m_step_evals_saved, None);
    }
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, MuRef};

    #[test]
    fn saem_final_ofv_report_formats_ofv_to_four_decimals() {
        // #893: the pre-covariance progress line reports the OFV to 4 dp so a
        // CLI user can judge the fit before the covariance step runs.
        assert_eq!(
            saem_final_ofv_report(1234.56789),
            "SAEM completed. Final OFV = 1234.5679"
        );
        assert_eq!(
            saem_final_ofv_report(-42.0),
            "SAEM completed. Final OFV = -42.0000"
        );
    }

    #[test]
    fn class_sigma_subst_is_none_without_overrides() {
        assert!(class_sigma_subst(&[0.2, 0.3], &[]).is_none());
    }

    #[test]
    fn class_sigma_subst_replaces_only_overridden_indices() {
        let out = class_sigma_subst(&[0.2, 0.3, 0.4], &[(2, 0.9)]).unwrap();
        assert_eq!(out, vec![0.2, 0.3, 0.9]);
        // Out-of-range indices are ignored rather than panicking.
        let out = class_sigma_subst(&[0.2], &[(7, 0.9)]).unwrap();
        assert_eq!(out, vec![0.2]);
    }

    /// The mixture M-step objective must actually reach the `MIXNUM` branch: the
    /// same subjects scored as class 1 vs class 2 must give different objective
    /// values, since the model's typical clearance switches on `MIXNUM` (#987
    /// review — this path was previously only exercised by slow-tests).
    #[test]
    fn obs_nll_sum_mix_class_guard_reaches_mixnum() {
        const MIX: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(5.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  sigma(2) EPS ~ 0.25

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(MIX).unwrap();
        let mut csv = String::from(
            "ID,TIME,DV,AMT,EVID,CMT
",
        );
        for sid in 1..=2 {
            csv.push_str(&format!(
                "{sid},0,0,100,1,1
"
            ));
            for t in [0.5_f64, 1.0, 2.0, 4.0] {
                csv.push_str(&format!(
                    "{sid},{t},{:.5},0,0,1
",
                    10.0 * (-0.1 * t).exp()
                ));
            }
        }
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, csv.as_bytes()).unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(f.path(), None, None).unwrap();

        let params = &model.default_params;
        let etas = vec![vec![0.0], vec![0.0]];
        let sigma = &params.sigma.values;
        let no_over: Vec<Vec<(usize, f64)>> = vec![Vec::new(), Vec::new()];

        let as_class1 = obs_nll_sum_mix(
            &model,
            &pop,
            &params.theta,
            sigma,
            &etas,
            MixMstep {
                classes: &[0, 0],
                class_sigma_over: &no_over,
            },
        );
        let as_class2 = obs_nll_sum_mix(
            &model,
            &pop,
            &params.theta,
            sigma,
            &etas,
            MixMstep {
                classes: &[1, 1],
                class_sigma_over: &no_over,
            },
        );
        assert!(as_class1.is_finite() && as_class2.is_finite());
        assert!(
            (as_class1 - as_class2).abs() > 1e-6,
            "class guard must switch TVCL1/TVCL2: {as_class1} vs {as_class2}"
        );

        // A held `sigma(2)` override must be what a class-2 subject is scored
        // under — not the free base σ the optimizer is moving.
        let mix = crate::estimation::saem_mixture::SaemMixture::build(&model, params, &pop);
        let over = mix.class_sigma_overrides();
        assert_eq!(over[1].len(), 1, "sigma(2) override present");
        let with_override = obs_nll_sum_mix(
            &model,
            &pop,
            &params.theta,
            sigma,
            &etas,
            MixMstep {
                classes: &[1, 1],
                class_sigma_over: &over,
            },
        );
        assert!(
            (with_override - as_class2).abs() > 1e-6,
            "class-2 σ override must change the M-step objective"
        );
    }

    #[test]
    fn fold_nll_grad_sums_nll_and_grad_elementwise_in_input_order() {
        let per_subj = vec![
            (1.0, vec![1.0, 10.0]),
            (2.0, vec![2.0, 20.0]),
            (3.0, vec![3.0, 30.0]),
        ];
        let (nll, grad) = fold_nll_grad(per_subj, 2);
        assert_eq!(nll, 6.0);
        assert_eq!(grad, vec![6.0, 60.0]);
    }

    #[test]
    fn fold_nll_grad_of_empty_input_is_zero() {
        let (nll, grad) = fold_nll_grad(vec![], 3);
        assert_eq!(nll, 0.0);
        assert_eq!(grad, vec![0.0, 0.0, 0.0]);
    }

    /// Pin the SAEM M-step optimizer choice.
    ///
    /// BOBYQA (derivative-free trust-region) was chosen over the prior SLSQP
    /// after the Emax PKPD benchmark surfaced an Emax-Hill identifiability
    /// failure mode where SLSQP locks population thetas onto one side of the
    /// ridge (EMAX under-estimated by ~40%, OFV virtually identical to the
    /// nlmixr2-matching basin). BOBYQA's quadratic trust-region exploration
    /// lands much closer to truth at ~40% lower wall on that benchmark.
    /// Simpler PK-only models are numerically equivalent across the two
    /// algorithms (|ΔOFV| < 0.1).
    ///
    /// If a future change switches to a different algorithm — particularly
    /// any gradient-based one (LBFGS, SLSQP, MMA) — re-run the Emax PKPD
    /// regression in the experiment repo and confirm EMAX/EC50 recovery
    /// before merging. The OFV alone is NOT a sufficient regression signal
    /// here because the Hill ridge produces near-identical OFV at very
    /// different parameter values.
    #[test]
    fn mstep_uses_bobyqa_optimizer() {
        assert!(
            matches!(MSTEP_NLOPT_ALGORITHM, nlopt::Algorithm::Bobyqa),
            "MSTEP_NLOPT_ALGORITHM changed — see comment above this test \
             for the Emax-Hill identifiability rationale before adjusting."
        );
    }

    /// `combined_additive_sigma_at_floor` flags only a free additive component
    /// (sigma index 1) sitting at/below the near-floor band, and ignores
    /// non-combined specs and FIXed sigmas.
    #[test]
    fn combined_additive_sigma_at_floor_detects_collapsed_free_additive() {
        let mut model = analytical_model(GradientMethod::Fd);
        model.error_spec = ErrorSpec::Single(ErrorModel::Combined);

        let mut params = model.default_params.clone();
        params.sigma = SigmaVector {
            values: vec![0.1, 0.5],
            names: vec!["PROP".into(), "ADD".into()],
        };
        params.sigma_fixed = vec![false, false];

        // Healthy additive term well above the floor band.
        assert!(!combined_additive_sigma_at_floor(&model, &params));

        // Additive term collapsed onto the floor → flagged.
        params.sigma.values[1] = 5.0e-4;
        assert!(combined_additive_sigma_at_floor(&model, &params));

        // A FIXed additive at the floor is intentional, not a collapse.
        params.sigma_fixed[1] = true;
        assert!(!combined_additive_sigma_at_floor(&model, &params));

        // Non-combined specs never flag, even with a tiny second sigma.
        params.sigma_fixed[1] = false;
        model.error_spec = ErrorSpec::Single(ErrorModel::Proportional);
        assert!(!combined_additive_sigma_at_floor(&model, &params));
    }

    #[test]
    fn saem_sampler_summary_defaults_to_metropolis_hastings() {
        // Default options (saem_n_leapfrog = 0) → MH random walk in every build.
        let model = analytical_model(GradientMethod::Auto);
        let opts = crate::types::FitOptions::default();
        let s = saem_sampler_summary(&model, &opts);
        assert!(
            s.starts_with("Metropolis-Hastings"),
            "default SAEM kernel should be MH, got: {s}"
        );
        // Requesting leapfrog steps without HMC support must say so, not claim HMC.
        let mut hmc_opts = crate::types::FitOptions::default();
        hmc_opts.saem_n_leapfrog = 10;
        let s2 = saem_sampler_summary(&model, &hmc_opts);
        assert!(
            s2.starts_with("HMC"),
            "analytical model + leapfrog steps should use HMC (Dual2 gradient), got: {s2}"
        );
    }

    fn model_with_mu_refs(
        theta_names: &[&str],
        eta_names: &[&str],
        mu_refs: &[(&str, &str, bool)],
    ) -> CompiledModel {
        let mut m = analytical_model(GradientMethod::Auto);
        m.theta_names = theta_names.iter().map(|s| (*s).to_string()).collect();
        m.eta_names = eta_names.iter().map(|s| (*s).to_string()).collect();
        m.n_theta = theta_names.len();
        m.n_eta = eta_names.len();
        m.mu_refs = mu_refs
            .iter()
            .map(|(eta, theta, log_t)| {
                (
                    (*eta).to_string(),
                    MuRef {
                        theta_name: (*theta).to_string(),
                        log_transformed: *log_t,
                    },
                )
            })
            .collect();
        m
    }

    // ── Class-aware mu-referencing for mixtures (#996) ──────────────────

    /// `model_with_mu_refs` plus a `MixtureSpec` carrying `n_classes` and the
    /// given class-aware anchors (`(eta, [theta per class])`).
    fn model_with_mixture_mu_refs(
        theta_names: &[&str],
        eta_names: &[&str],
        mu_refs: &[(&str, &str, bool)],
        n_classes: usize,
        class_refs: &[(&str, Vec<&str>)],
    ) -> CompiledModel {
        let mut m = model_with_mu_refs(theta_names, eta_names, mu_refs);
        m.mixture = Some(crate::types::MixtureSpec {
            n_classes,
            mixing: Vec::new(),
            logit_covariates: Vec::new(),
            omega_overrides: Vec::new(),
            sigma_overrides: Vec::new(),
            mu_refs: class_refs
                .iter()
                .map(|(eta, thetas)| crate::types::MixtureMuRef {
                    eta_name: (*eta).to_string(),
                    theta_names: thetas.iter().map(|t| (*t).to_string()).collect(),
                    log_transformed: true,
                })
                .collect(),
        });
        m
    }

    #[test]
    fn mixture_mu_ref_pairs_empty_for_non_mixture() {
        let m = model_with_mu_refs(&["TVCL"], &["ETA_CL"], &[("ETA_CL", "TVCL", true)]);
        assert!(get_mixture_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn mixture_mu_ref_pairs_map_class_switched_anchors() {
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV"],
            &["ETA_CL"],
            &[],
            2,
            &[("ETA_CL", vec!["TVCL1", "TVCL2"])],
        );
        assert_eq!(
            get_mixture_mu_ref_pairs(&m),
            vec![MixtureMuRefPair {
                eta_idx: 0,
                theta_idx: vec![0, 1]
            }]
        );
    }

    #[test]
    fn mixture_mu_ref_pairs_broadcast_class_shared_theta() {
        // A plain (non-switched) mu-ref in a mixture model becomes the same
        // theta in every class slot, so one update rule covers both shapes.
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_V", "TVV", true)],
            3,
            &[("ETA_CL", vec!["TVCL1", "TVCL2", "TVCL2"])],
        );
        assert_eq!(
            get_mixture_mu_ref_pairs(&m),
            vec![
                MixtureMuRefPair {
                    eta_idx: 0,
                    theta_idx: vec![0, 1, 1]
                },
                MixtureMuRefPair {
                    eta_idx: 1,
                    theta_idx: vec![2, 2, 2]
                },
            ]
        );
    }

    #[test]
    fn mixture_mu_ref_pairs_drop_a_theta_claimed_by_a_second_eta() {
        // Two etas anchored to the same typical value: the shift loops apply one
        // update per pair, so keeping both would move TVCL twice in a single
        // iteration. The first eta keeps it; the second pair is dropped (#996
        // review).
        let m = model_with_mixture_mu_refs(
            &["TVCL", "TVV"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "TVCL", true), ("ETA_V", "TVCL", true)],
            2,
            &[],
        );
        let pairs = get_mixture_mu_ref_pairs(&m);
        assert_eq!(
            pairs,
            vec![MixtureMuRefPair {
                eta_idx: 0,
                theta_idx: vec![0, 0]
            }]
        );
        // Every theta is claimed at most once across the whole pair set.
        let mut all: Vec<usize> = pairs
            .iter()
            .flat_map(|p| p.theta_idx.iter().copied())
            .collect();
        all.sort_unstable();
        let n = all.len();
        all.dedup();
        assert_eq!(all.len(), 1, "theta claimed once, got {n} slots");
    }

    #[test]
    fn mixture_mu_ref_pairs_drop_a_partially_overlapping_class_anchor_set() {
        // ETA_CL claims TVCL1/TVCL2; ETA_V's class anchors reuse TVCL2, so its
        // whole pair is dropped rather than double-shifting that theta.
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV1"],
            &["ETA_CL", "ETA_V"],
            &[],
            2,
            &[
                ("ETA_CL", vec!["TVCL1", "TVCL2"]),
                ("ETA_V", vec!["TVV1", "TVCL2"]),
            ],
        );
        assert_eq!(
            get_mixture_mu_ref_pairs(&m),
            vec![MixtureMuRefPair {
                eta_idx: 0,
                theta_idx: vec![0, 1]
            }]
        );
    }

    #[test]
    fn mixture_mu_ref_pairs_exclude_additive_and_unknown_thetas() {
        let m = model_with_mixture_mu_refs(
            &["TVCL1", "TVCL2", "TVV"],
            &["ETA_CL", "ETA_V"],
            // additive: excluded, like the single-population `get_mu_ref_pairs`
            &[("ETA_V", "TVV", false)],
            2,
            &[("ETA_CL", vec!["TVCL1", "MISSING"])],
        );
        assert!(get_mixture_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn mixture_mu_ref_means_reduce_to_pooled_mean_when_theta_shared() {
        // Degenerate oracle: every class anchors on theta 0, so the weighted
        // per-class update must equal the classical pooled mean(η) exactly —
        // including bit-for-bit, since the accumulation order is the same.
        let etas = [0.3f64, -0.1, 0.7, -0.4];
        let resp = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let means = mixture_mu_ref_means(2, &[0, 0], &resp, |i, _c| etas[i]);
        let pooled = etas.iter().sum::<f64>() / etas.len() as f64;
        assert_eq!(means[0], Some(pooled));
        assert_eq!(means[1], None, "theta 1 is served by no class");
    }

    #[test]
    fn mixture_mu_ref_means_reduce_to_pooled_mean_under_soft_responsibilities() {
        // The same degeneracy under IMP-style fractional responsibilities:
        // every subject's weights sum to 1, so the shared-theta update is the
        // responsibility-weighted average of the per-class η means.
        let eta = |i: usize, c: usize| [[0.2f64, 0.4], [-0.6, -0.2]][i][c];
        let resp = vec![vec![0.75, 0.25], vec![0.4, 0.6]];
        let means = mixture_mu_ref_means(1, &[0, 0], &resp, eta);
        let expect = (0.75 * 0.2 + 0.25 * 0.4 + 0.4 * -0.6 + 0.6 * -0.2) / 2.0;
        assert!((means[0].unwrap() - expect).abs() < 1e-15);
    }

    #[test]
    fn mixture_mu_ref_means_split_by_class_membership() {
        // Hard classes: subjects 0,2 in class 1 (theta 0); 1,3 in class 2
        // (theta 1). Each theta moves by its own members' mean only.
        let etas = [0.3f64, -0.1, 0.7, -0.5];
        let resp = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let means = mixture_mu_ref_means(2, &[0, 1], &resp, |i, _c| etas[i]);
        assert!((means[0].unwrap() - 0.5).abs() < 1e-15);
        assert!((means[1].unwrap() - (-0.3)).abs() < 1e-15);
    }

    #[test]
    fn mixture_mu_ref_means_hold_theta_for_an_empty_class() {
        // Label switching: class 2 won zero subjects this iteration, so its
        // theta has no η mean and must be held rather than dragged.
        let etas = [0.2f64, 0.4];
        let resp = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let means = mixture_mu_ref_means(2, &[0, 1], &resp, |i, _c| etas[i]);
        assert!((means[0].unwrap() - 0.3).abs() < 1e-15);
        assert_eq!(means[1], None);
    }

    #[test]
    fn mixture_mu_ref_means_weight_classes_by_responsibility() {
        // Soft responsibilities with distinct class thetas: each theta gets the
        // responsibility-weighted mean of its own class's per-class η means.
        let eta = |i: usize, c: usize| [[0.1f64, 0.9], [0.3, 0.5]][i][c];
        let resp = vec![vec![0.8, 0.2], vec![0.25, 0.75]];
        let means = mixture_mu_ref_means(2, &[0, 1], &resp, eta);
        let e0 = (0.8 * 0.1 + 0.25 * 0.3) / (0.8 + 0.25);
        let e1 = (0.2 * 0.9 + 0.75 * 0.5) / (0.2 + 0.75);
        assert!((means[0].unwrap() - e0).abs() < 1e-15);
        assert!((means[1].unwrap() - e1).abs() < 1e-15);
    }

    #[test]
    fn floor_omega_diagonal_floors_free_entries_only() {
        // Three etas: a free near-zero diagonal (should be floored), a free
        // healthy diagonal (untouched), and a FIX-ed near-zero diagonal (kept).
        let mut omega = DMatrix::<f64>::zeros(3, 3);
        omega[(0, 0)] = 1e-9; // free, below floor → raised
        omega[(1, 1)] = 0.2; // free, above floor → unchanged
        omega[(2, 2)] = 1e-9; // FIX-ed, below floor → preserved
                              // an off-diagonal that must not be touched by the diagonal floor
        omega[(0, 1)] = 0.01;
        omega[(1, 0)] = 0.01;

        let omega_fixed = vec![false, false, true];
        floor_omega_diagonal(&mut omega, &omega_fixed, 1e-6);

        assert_eq!(
            omega[(0, 0)],
            1e-6,
            "free near-zero diagonal must be floored"
        );
        assert_eq!(
            omega[(1, 1)],
            0.2,
            "healthy free diagonal must be unchanged"
        );
        assert_eq!(
            omega[(2, 2)],
            1e-9,
            "FIX-ed diagonal must be left exactly as declared"
        );
        assert_eq!(omega[(0, 1)], 0.01, "off-diagonals must not be touched");
    }

    #[test]
    fn floor_omega_diagonal_treats_missing_fixed_flags_as_free() {
        // `omega_fixed` shorter than the matrix: missing entries default to free.
        let mut omega = DMatrix::<f64>::zeros(2, 2);
        omega[(0, 0)] = 1e-9;
        omega[(1, 1)] = 1e-9;
        floor_omega_diagonal(&mut omega, &[], 1e-6);
        assert_eq!(omega[(0, 0)], 1e-6);
        assert_eq!(omega[(1, 1)], 1e-6);
    }

    #[test]
    fn get_mu_ref_pairs_empty_when_no_mu_refs() {
        let m = analytical_model(GradientMethod::Auto);
        assert!(get_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn get_mu_ref_pairs_returns_log_transformed_pair() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", true)],
        );
        let mut pairs = get_mu_ref_pairs(&m);
        pairs.sort();
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn get_mu_ref_pairs_excludes_additive_mu_refs() {
        // ETA_CL is lognormal (THETA*exp(ETA)) — included.
        // ETA_V is additive (THETA+ETA) — excluded because the gradient-step
        // chain rule used in run_saem assumes log-transformed parameters.
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", false)],
        );
        assert_eq!(get_mu_ref_pairs(&m), vec![(0, 0)]);
    }

    #[test]
    fn get_mu_ref_pairs_skips_orphaned_theta() {
        // mu_ref points at a theta name that doesn't exist — silently skipped.
        let m = model_with_mu_refs(&["CL"], &["ETA_CL"], &[("ETA_CL", "MISSING", true)]);
        assert!(get_mu_ref_pairs(&m).is_empty());
    }

    // ---- Regression tests for the three SAEM correctness bugs ----

    /// Bug 1 (diagonal): `from_diagonal` produces a free_mask that marks only
    /// diagonal entries free. The SAEM M-step uses this mask to zero
    /// SA-accumulated off-diagonals, preventing the rank-deficient Ω failure.
    #[test]
    fn diagonal_omega_free_mask_has_no_off_diagonals() {
        let omega = OmegaMatrix::from_diagonal(&[0.1, 0.2], vec!["ETA_CL".into(), "ETA_V".into()]);
        assert!(omega.free_mask[(0, 0)]);
        assert!(omega.free_mask[(1, 1)]);
        assert!(!omega.free_mask[(0, 1)]);
        assert!(!omega.free_mask[(1, 0)]);
    }

    /// Bug 1 (mixed structure): `from_matrix_with_mask` preserves an explicit
    /// mask that marks cross-block entries as structural zeros. This is the
    /// case that the `diagonal` flag alone cannot express (one standalone eta
    /// + one block_omega pair → diagonal=false, but cross entries are zero).
    #[test]
    fn mixed_omega_free_mask_zeros_cross_block_entries() {
        // Three etas: ETA_CL(0) and ETA_V(1) in a block; ETA_KA(2) standalone.
        let mut matrix = nalgebra::DMatrix::zeros(3, 3);
        matrix[(0, 0)] = 0.1;
        matrix[(1, 1)] = 0.2;
        matrix[(2, 2)] = 0.1;
        matrix[(0, 1)] = 0.01;
        matrix[(1, 0)] = 0.01;

        let mut free_mask = nalgebra::DMatrix::from_element(3, 3, false);
        free_mask[(0, 0)] = true;
        free_mask[(1, 1)] = true;
        free_mask[(2, 2)] = true;
        free_mask[(0, 1)] = true; // within CL-V block
        free_mask[(1, 0)] = true;

        let names = vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()];
        let omega = OmegaMatrix::from_matrix_with_mask(matrix, names, false, free_mask);

        assert!(omega.free_mask[(0, 1)]);
        assert!(omega.free_mask[(1, 0)]);
        assert!(!omega.free_mask[(2, 0)]);
        assert!(!omega.free_mask[(0, 2)]);
        assert!(!omega.free_mask[(2, 1)]);
        assert!(!omega.free_mask[(1, 2)]);
    }

    /// Bug 2: `mh_steps` is a symmetric random walk — proposals are
    /// `eta_prop = eta + step·perturbation`, not `mu_k + step·perturbation`.
    ///
    /// Discriminator: with `step_scale = 0` the new kernel proposes exactly
    /// the current eta, so the chain cannot move regardless of the data.
    /// The pre-fix `mu_k`-centred kernel proposed exactly `mu_k` (= log TVCL),
    /// so a starting eta far from `mu_k` would either jump to `mu_k`
    /// whenever the proposal looked better, or oscillate. We pick a starting
    /// eta of 5.0 with TVCL=1 (mu_k=0): the simulated observation lives near
    /// the data-generating eta=0 region, so individual_nll(eta=0) is much
    /// lower than individual_nll(eta=5), meaning the broken kernel would
    /// accept the eta=0 proposal with probability ≈1 on the first step.
    /// The new kernel must leave eta at exactly 5.0.
    #[test]
    fn mh_steps_random_walk_uses_current_eta_not_mu_k() {
        use crate::stats::likelihood::individual_nll;
        use crate::types::{DoseEvent, SigmaVector};
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let omega = OmegaMatrix::from_diagonal(&[1.0], vec!["ETA_CL".into()]);
        let sigma = SigmaVector {
            values: vec![1.0],
            names: vec!["PROP".into()],
        };
        let theta = vec![1.0]; // mu_k = log(1) = 0
        let mut eta = vec![5.0_f64]; // far from mu_k
        let nll_start = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
        let mut rng = StdRng::seed_from_u64(42);

        let mut pk_scratch = EventPkParams::with_capacity_for(&subj);
        mh_steps(
            &mut eta,
            nll_start,
            &subj,
            &model,
            &theta,
            &omega,
            &sigma.values,
            0.0, // zero perturbation: random walk MUST stay put exactly
            None,
            &mut rng,
            100,
            &mut pk_scratch,
            None,
        );

        // Random walk with step=0: every proposal == current eta, accepted as
        // identity. The pre-fix kernel would have proposed mu_k=0 every step
        // and accepted it (lower nll than eta=5), driving eta to 0.
        assert_eq!(
            eta[0], 5.0,
            "eta moved despite step_scale=0 — proposals were re-centred on mu_k"
        );
    }

    /// Bug 3 / closed-form M-step: a synthetic SAEM run with mu_referencing=true
    /// and mean(eta) ≠ 0 must move log_theta in the right direction *without*
    /// pinning at the bound. We exercise the closed-form formula directly:
    /// `log_theta_new = log_theta_old + γ · mean(eta)`.
    #[test]
    fn closed_form_mu_ref_mstep_is_bounded_and_signed_correctly() {
        // Simulate post-MH state: 5 subjects, eta_mean = +0.4 (population CL
        // is higher than current TVCL), gamma = 1.0 (exploration step).
        let etas: Vec<Vec<f64>> = vec![vec![0.5], vec![0.3], vec![0.4], vec![0.6], vec![0.2]];
        let n = etas.len() as f64;
        let mean_eta: f64 = etas.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!((mean_eta - 0.4).abs() < 1e-12);

        let gamma = 1.0;
        let log_theta_old = 0.0_f64; // TVCL = 1.0
        let log_theta_new = log_theta_old + gamma * mean_eta;
        // log_theta moved by exactly mean(eta), independent of N.  This is the
        // property that the broken gradient step (γ · Σ ∂obs_nll/∂eta) lacked:
        // its update scaled with N and pinned thetas at bounds for moderate N.
        assert!((log_theta_new - 0.4).abs() < 1e-12);

        // After re-centring etas by gamma*mean, mean(eta) = 0.
        let mut etas_recentered = etas.clone();
        for e in etas_recentered.iter_mut() {
            e[0] -= gamma * mean_eta;
        }
        let new_mean: f64 = etas_recentered.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!(new_mean.abs() < 1e-12);
    }

    /// Bug 3 follow-up: the broken gradient step (γ · Σᵢ ∂obs_nll/∂eta) is no
    /// longer in the code path. The closed-form `log_theta += γ · mean(η)` is
    /// what runs when mu_referencing=true. Pair detection is unchanged.
    #[test]
    fn mu_ref_pair_detection_drives_closed_form_branch() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", true)],
        );
        let pairs = get_mu_ref_pairs(&m);
        assert_eq!(pairs.len(), 2);
        // The closed-form branch is taken iff `options.mu_referencing` AND
        // `!pairs.is_empty()`.  Both conditions are tested via the public API
        // in api::iov_integration::test_iov_foce_mu_referencing_on; this unit
        // test pins the precondition (pair detection still produces work).
    }

    /// A pre-cancelled `CancelFlag` makes the SAEM main loop break at the
    /// first iteration and `run_saem` must return `Err("cancelled by user")`
    /// without entering the post-loop "Computing final EBEs and OFV..." block
    /// (which iterates over every subject and is what makes a cancelled run
    /// feel like it isn't aborting).
    #[test]
    fn cancelled_run_returns_err_and_skips_final_ebe() {
        use crate::cancel::CancelFlag;
        use crate::types::{DoseEvent, FitOptions, Population};
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0, 0.5],
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subj],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let flag = CancelFlag::new();
        flag.cancel(); // pre-cancel: loop breaks at iteration 1

        let mut opts = FitOptions::default();
        opts.verbose = false;
        opts.run_covariance_step = false;
        opts.cancel = Some(flag);

        match run_saem(&model, &population, &model.default_params, &opts) {
            Err(msg) => assert!(
                msg.contains("cancelled by user"),
                "unexpected error message: {msg}"
            ),
            Ok(_) => panic!("pre-cancelled SAEM must return Err, not Ok"),
        }
    }

    /// Per-theta packing must round-trip values identically for both log-packed
    /// (`theta_lower >= 0`) and identity-packed (`theta_lower < 0`) thetas. SAEM
    /// uses its own pack/unpack closures inside the M-step, so this exercises
    /// the same math the closures rely on (`theta_packs_log` from
    /// parameterization plus the `if mask[i] { ln/exp } else { identity }`
    /// branches in `theta_sigma_mstep_light`).
    #[test]
    fn saem_pack_unpack_handles_negative_lower_bound() {
        use crate::estimation::parameterization::theta_packs_log;

        // Mix: CL (lower=0), V (lower=0.001), THETA_AGE_CL (lower=-1).
        let lowers: [f64; 3] = [0.0, 0.001, -1.0];
        let values: [f64; 3] = [5.0, 20.0, -0.01];
        let mask: Vec<bool> = lowers.iter().map(|&lo| theta_packs_log(lo)).collect();
        assert_eq!(mask, vec![true, true, false]);

        // Forward: simulate the SAEM init-pack construction (lines ~444–451 of
        // run_saem: log when log-packed, identity when identity-packed).
        let packed: Vec<f64> = values
            .iter()
            .zip(mask.iter())
            .map(|(&v, &log_pack)| if log_pack { v.max(1e-10).ln() } else { v })
            .collect();

        // Reverse: the M-step `unpack_thetas` closure.
        let unpacked: Vec<f64> = packed
            .iter()
            .zip(mask.iter())
            .map(|(&p, &log_pack)| if log_pack { p.exp() } else { p })
            .collect();

        for (orig, round) in values.iter().zip(unpacked.iter()) {
            assert!(
                (orig - round).abs() < 1e-12,
                "saem pack/unpack should round-trip: {orig} != {round}"
            );
        }
        // The identity-packed theta carries a negative value through —
        // pre-fix, this was clamped to 1e-10 by the log path.
        assert!(unpacked[2] < 0.0);
    }

    /// `obs_nll_subject_grad` summed over subjects must match the reference
    /// forward-FD of `obs_nll_sum` to within 1e-4 relative tolerance for all
    /// non-pinned packed parameters (theta + sigma).
    #[test]
    fn obs_nll_subject_grad_matches_obs_nll_sum_fd() {
        use crate::types::{DoseEvent, Population};
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);

        let make_subj = |id: &str, obs: f64| Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![obs, obs * 0.6, obs * 0.3],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            fremtype: Vec::new(),
            obs_records: vec![],
        };

        let population = Population {
            subjects: vec![
                make_subj("1", 8.0),
                make_subj("2", 5.0),
                make_subj("3", 11.0),
            ],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let theta = vec![1.5f64, 20.0]; // CL, V
        let sigma_values = vec![0.2f64]; // proportional
        let etas: Vec<Vec<f64>> = vec![vec![0.0], vec![0.1], vec![-0.1]];
        let n_theta = 2;
        let n_sigma = 1;
        let n = n_theta + n_sigma;

        // Compute reference gradient via forward-FD of obs_nll_sum.
        let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas);
        let h = 1e-5;
        let mut ref_grad = vec![0.0f64; n];
        // Theta perturbations (in natural scale).
        for i in 0..n_theta {
            let mut theta_p = theta.clone();
            theta_p[i] += h;
            let fp = obs_nll_sum(&model, &population, &theta_p, &sigma_values, &etas);
            // FD in natural scale; convert to log-packed space (d/d_log = theta * d/d_theta)
            ref_grad[i] = theta[i] * (fp - f0) / h;
        }
        // Sigma perturbation (in natural scale; convert to log-packed).
        {
            let mut sigma_p = sigma_values.clone();
            sigma_p[0] += h;
            let fp = obs_nll_sum(&model, &population, &theta, &sigma_p, &etas);
            ref_grad[n_theta] = sigma_values[0] * (fp - f0) / h;
        }

        // Compute gradient via obs_nll_subject_grad summed over subjects.
        let mask: Vec<bool> = theta.iter().map(|_| true).collect(); // all log-packed
        let lo = vec![-1e30f64; n];
        let hi = vec![1e30f64; n];
        let mut total_nll = 0.0f64;
        let mut total_grad = vec![0.0f64; n];
        let mut scratch = EventPkParams::default();
        for (i, subject) in population.subjects.iter().enumerate() {
            let (nll_i, grad_i) = obs_nll_subject_grad(
                &model,
                subject,
                &theta,
                &sigma_values,
                &etas[i],
                &mask,
                &lo,
                &hi,
                n_theta,
                n_sigma,
                &mut scratch,
            );
            total_nll += nll_i;
            for (g, gi) in total_grad.iter_mut().zip(grad_i.iter()) {
                *g += gi;
            }
        }

        assert!(
            (total_nll - f0).abs() < 1e-10,
            "nll mismatch: {} vs {}",
            total_nll,
            f0
        );

        for j in 0..n {
            let rel = if ref_grad[j].abs() > 1e-10 {
                (total_grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
            } else {
                (total_grad[j] - ref_grad[j]).abs()
            };
            assert!(
                rel < 1e-4,
                "grad[{j}]: obs_nll_subject_grad={:.6e}, ref={:.6e}, rel={:.2e}",
                total_grad[j],
                ref_grad[j],
                rel
            );
        }
    }

    /// IOV M-step gradient (`obs_nll_subject_grad_iov`) must match the forward-FD
    /// of `obs_nll_subject_into_iov` in log-packed space. This guards the
    /// analytical gradient that the gradient-based M-step would use — it is not
    /// exercised by the default BOBYQA M-step (derivative-free), so without this
    /// direct test the function is untested. Single subject, 2 occasions, κ on CL.
    #[test]
    fn obs_nll_subject_grad_iov_matches_fd() {
        use crate::types::{
            BloqMethod, CompiledModel, DoseEvent, ErrorModel, ErrorSpec, GradientMethod,
            ModelParameters, OmegaMatrix, PkModel, PkParams, ScalingSpec, SigmaVector, Subject,
        };
        use std::collections::HashMap;

        // Minimal IOV model: CL = TVCL·exp(ETA_CL + KAPPA_CL), V = TVV.
        let model = CompiledModel {
            name: "iov_grad_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(
                |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                    p.values[0] = theta[0] * (eta[0] + kappa).exp();
                    p.values[1] = theta[1];
                    p
                },
            ),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params: ModelParameters {
                theta: vec![5.0, 50.0],
                theta_names: vec!["TVCL".into(), "TVV".into()],
                theta_lower: vec![0.1, 5.0],
                theta_upper: vec![50.0, 500.0],
                theta_fixed: vec![false; 2],
                omega: OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]),
                omega_fixed: vec![false],
                sigma: SigmaVector {
                    values: vec![0.05],
                    names: vec!["PROP_ERR".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: Some(OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()])),
                kappa_fixed: vec![false],
                mixture: None,
            },
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: vec![false],
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: Vec::new(),
            output_columns: Vec::new(),
            #[cfg(feature = "survival")]
            endpoints: HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
            absorption_ode_equivalent: None,
            mixture: None,
        };

        // One subject, 2 occasions (times 1–3 occ 1, 4–6 occ 2), one dose each.
        let subject = Subject {
            id: "S1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            obs_raw_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            observations: vec![36.0, 28.0, 21.0, 34.0, 26.0, 19.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            obs_records: Vec::new(),
        };

        let theta = vec![5.0f64, 50.0];
        let sigma = vec![0.05f64];
        let eta = vec![0.1f64];
        let kappas: Vec<Vec<f64>> = vec![vec![0.05], vec![-0.05]]; // one per occasion
        let n_theta = 2;
        let n_sigma = 1;
        let n = n_theta + n_sigma;

        let mut scratch = EventPkParams::default();
        let (nll, grad) = obs_nll_subject_grad_iov(
            &model,
            &subject,
            &theta,
            &sigma,
            &eta,
            &kappas,
            &[true, true, true],
            &[-1e30; 3],
            &[1e30; 3],
            n_theta,
            n_sigma,
            &mut scratch,
        );

        // Reference: forward-FD of obs_nll_subject_into_iov in log-packed space.
        let f0 = obs_nll_subject_into_iov(
            &model,
            &subject,
            &theta,
            &sigma,
            &eta,
            &kappas,
            &mut scratch,
        );
        assert!((nll - f0).abs() < 1e-10, "nll mismatch: {nll} vs {f0}");

        let h = 1e-6;
        let mut ref_grad = vec![0.0f64; n];
        for i in 0..n_theta {
            let mut tp = theta.clone();
            tp[i] += h;
            let fp = obs_nll_subject_into_iov(
                &model,
                &subject,
                &tp,
                &sigma,
                &eta,
                &kappas,
                &mut scratch,
            );
            ref_grad[i] = theta[i] * (fp - f0) / h; // d/d_log = theta · d/d_theta
        }
        {
            let mut sp = sigma.clone();
            sp[0] += h;
            let fp = obs_nll_subject_into_iov(
                &model,
                &subject,
                &theta,
                &sp,
                &eta,
                &kappas,
                &mut scratch,
            );
            ref_grad[n_theta] = sigma[0] * (fp - f0) / h;
        }

        for j in 0..n {
            let rel = if ref_grad[j].abs() > 1e-8 {
                (grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
            } else {
                (grad[j] - ref_grad[j]).abs()
            };
            assert!(
                rel < 1e-4,
                "grad[{j}]: analytical={:.6e}, fd={:.6e}, rel={:.2e}",
                grad[j],
                ref_grad[j],
                rel
            );
        }
    }

    /// Per-CMT (multi-endpoint) M-step gradient must match the forward-FD of
    /// `obs_nll_sum` — the correctness gate for the per-CMT `dvar_df` /
    /// `dvar_dlogsigma` score terms. Two endpoints with *different* error
    /// models (proportional PK on CMT=1, additive PD on CMT=2) so a single
    /// error model would give the wrong Jacobian for one endpoint.
    #[test]
    fn obs_nll_subject_grad_per_cmt_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{DoseEvent, Population};
        use std::collections::HashMap;

        let model = parse_model_string(
            r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVKE0(0.5, 0.05, 5.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 0.50 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KE0 = TVKE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  KE0 * (central/V - effect)

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_PK)
  CMT=2: DV ~ additive(ADD_ERR_PD)
",
        )
        .expect("per-CMT ODE model parses");

        // obs at CMT=1 (PK) and CMT=2 (PD), interleaved.
        let make_subj = |id: &str, scale: f64| Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
            obs_raw_times: Vec::new(),
            observations: vec![
                8.0 * scale,
                2.0 * scale,
                6.0 * scale,
                3.0 * scale,
                4.0 * scale,
                3.5 * scale,
            ],
            obs_cmts: vec![1, 2, 1, 2, 1, 2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![make_subj("1", 1.0), make_subj("2", 1.1)],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let theta = vec![1.0f64, 10.0, 0.5];
        let sigma_values = vec![0.10f64, 0.50];
        let etas: Vec<Vec<f64>> = vec![vec![0.0], vec![0.05]];
        let n_theta = 3;
        let n_sigma = 2;
        let n = n_theta + n_sigma;

        // Reference gradient: forward-FD of obs_nll_sum, in log-packed space.
        let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas);
        let h = 1e-6;
        let mut ref_grad = vec![0.0f64; n];
        for i in 0..n_theta {
            let mut tp = theta.clone();
            tp[i] += h;
            let fp = obs_nll_sum(&model, &population, &tp, &sigma_values, &etas);
            ref_grad[i] = theta[i] * (fp - f0) / h;
        }
        for k in 0..n_sigma {
            let mut sp = sigma_values.clone();
            sp[k] += h;
            let fp = obs_nll_sum(&model, &population, &theta, &sp, &etas);
            ref_grad[n_theta + k] = sigma_values[k] * (fp - f0) / h;
        }

        // Analytical gradient: sum of per-subject obs_nll_subject_grad.
        let mask = vec![true; n_theta];
        let lo = vec![-1e30f64; n];
        let hi = vec![1e30f64; n];
        let mut total_nll = 0.0f64;
        let mut total_grad = vec![0.0f64; n];
        let mut scratch = EventPkParams::default();
        for (i, subject) in population.subjects.iter().enumerate() {
            let (nll_i, grad_i) = obs_nll_subject_grad(
                &model,
                subject,
                &theta,
                &sigma_values,
                &etas[i],
                &mask,
                &lo,
                &hi,
                n_theta,
                n_sigma,
                &mut scratch,
            );
            total_nll += nll_i;
            for (g, gi) in total_grad.iter_mut().zip(grad_i.iter()) {
                *g += gi;
            }
        }

        assert!(
            (total_nll - f0).abs() < 1e-8,
            "nll mismatch: {total_nll} vs {f0}"
        );
        for j in 0..n {
            let rel = if ref_grad[j].abs() > 1e-8 {
                (total_grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
            } else {
                (total_grad[j] - ref_grad[j]).abs()
            };
            assert!(
                rel < 1e-3,
                "per-CMT grad[{j}]: analytical={:.6e}, fd={:.6e}, rel={:.2e}",
                total_grad[j],
                ref_grad[j],
                rel
            );
        }
    }

    /// Dense residual-covariance M-step gradient must match FD of the same
    /// dense observation NLL. This exercises the `block_sigma` SAEM path, which
    /// deliberately routes through full FD because the analytic scalar-RUV score
    /// terms do not apply to off-diagonal R blocks.
    #[test]
    fn obs_nll_subject_grad_block_sigma_cross_endpoint_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{DoseEvent, Population};
        use std::collections::HashMap;

        let model = parse_model_string(
            r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR_UNBOUND, PROP_ERR_TOTAL) = [
    0.04,
    0.01, 0.09
  ]

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y[CMT=1] = 2.0 * central / V
  y[CMT=2] = central / V

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_TOTAL)
  CMT=2: DV ~ proportional(PROP_ERR_UNBOUND)
",
        )
        .expect("cross-endpoint block_sigma ODE model parses");

        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 1.0, 2.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: vec![17.0, 8.0, 15.0, 7.0],
            obs_cmts: vec![1, 2, 1, 2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject.clone()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let theta = vec![1.0f64, 10.0];
        let sigma_values = vec![0.20f64, 0.30];
        let etas: Vec<Vec<f64>> = vec![vec![0.05]];
        let n_theta = 2;
        let n_sigma = 2;
        let n = n_theta + n_sigma;

        let f0 = obs_nll_sum(&model, &population, &theta, &sigma_values, &etas);
        let h = 1e-6;
        let mut ref_grad = vec![0.0f64; n];
        for i in 0..n_theta {
            let mut tp = theta.clone();
            tp[i] += h;
            let fp = obs_nll_sum(&model, &population, &tp, &sigma_values, &etas);
            ref_grad[i] = theta[i] * (fp - f0) / h;
        }
        for k in 0..n_sigma {
            let mut sp = sigma_values.clone();
            sp[k] += h;
            let fp = obs_nll_sum(&model, &population, &theta, &sp, &etas);
            ref_grad[n_theta + k] = sigma_values[k] * (fp - f0) / h;
        }

        let mask = vec![true; n_theta];
        let lo = vec![-1e30f64; n];
        let hi = vec![1e30f64; n];
        let mut scratch = EventPkParams::default();
        let (nll, grad) = obs_nll_subject_grad(
            &model,
            &subject,
            &theta,
            &sigma_values,
            &etas[0],
            &mask,
            &lo,
            &hi,
            n_theta,
            n_sigma,
            &mut scratch,
        );

        assert!((nll - f0).abs() < 1e-8, "nll mismatch: {nll} vs {f0}");
        for j in 0..n {
            let rel = if ref_grad[j].abs() > 1e-8 {
                (grad[j] - ref_grad[j]).abs() / ref_grad[j].abs()
            } else {
                (grad[j] - ref_grad[j]).abs()
            };
            assert!(
                rel < 1e-4,
                "block_sigma grad[{j}]: fd-path={:.6e}, ref={:.6e}, rel={:.2e}",
                grad[j],
                ref_grad[j],
                rel
            );
        }
    }

    // ── IOV kappa MH: rejection restores kappa ─────────────────────────────

    /// With `step_scale = 0` the proposal is always identical to the current
    /// kappa, so ΔH = 0 and every step is accepted.  The kappa values must
    /// not change (proposal == current).
    #[test]
    fn mh_kappa_zero_step_always_accepts_and_preserves_kappa() {
        use crate::types::test_helpers::analytical_model;
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);

        // One subject with 2 occasions (occasions = [1,1,2,2]).
        let subject = Subject {
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0],
            obs_raw_times: Vec::new(),
            observations: vec![50.0, 40.0, 35.0, 28.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1u32, 1, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: vec![1u32],
            fremtype: Vec::new(),
            obs_records: vec![],
        };

        let omega_bsv = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let sigma = vec![0.1];
        // Two occasions, each with one kappa.
        let mut kappas = vec![vec![0.2_f64], vec![-0.1_f64]];
        let kappas_before = kappas.clone();

        let nll0 = individual_nll_iov(
            &model,
            &subject,
            &theta,
            &eta,
            &kappas,
            &omega_bsv,
            Some(&omega_iov),
            &sigma,
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (n_acc, n_prop, nll_after) = mh_kappa_steps(
            &mut kappas,
            nll0,
            &subject,
            &model,
            &theta,
            &eta,
            &omega_bsv,
            &omega_iov,
            &sigma,
            0.0, // step_scale = 0 → proposal == current → always accepted
            &mut rng,
        );

        // With step_scale=0 every occasion proposal is accepted (2 occasions).
        assert_eq!(n_prop, 2, "expected 2 proposals (one per occasion)");
        assert_eq!(n_acc, 2, "step_scale=0: all proposals must be accepted");
        // Kappa values must be unchanged (proposal == current point).
        assert_eq!(
            kappas, kappas_before,
            "kappas must not change with step_scale=0"
        );
        // NLL must not change either.
        assert!(
            (nll_after - nll0).abs() < 1e-10,
            "NLL must not change with step_scale=0"
        );
    }

    // ── IOV omega analytic update formula ──────────────────────────────────

    /// The analytic update `(1/N_occ) Σᵢ Σₖ κᵢₖ κᵢₖᵀ` for a 1-dimensional
    /// omega_iov with two subjects, two occasions each, and known kappas must
    /// match the hand-computed value exactly.
    #[test]
    fn iov_omega_analytic_update_matches_hand_computation() {
        // Subject 1: occ1 = [0.2], occ2 = [-0.1]
        // Subject 2: occ1 = [0.3], occ2 = [-0.2]
        // Hand sum = 0.2² + 0.1² + 0.3² + 0.2² = 0.04 + 0.01 + 0.09 + 0.04 = 0.18
        // Divided by 4 occasions → 0.045
        let kappas: Vec<Vec<Vec<f64>>> =
            vec![vec![vec![0.2], vec![-0.1]], vec![vec![0.3], vec![-0.2]]];
        let n_kappa = 1_usize;
        let mut kappa_outer = DMatrix::zeros(n_kappa, n_kappa);
        let mut n_total_occ = 0_usize;
        for kappas_i in &kappas {
            for kap in kappas_i {
                let kv = DVector::from_column_slice(kap);
                kappa_outer += &kv * kv.transpose();
                n_total_occ += 1;
            }
        }
        kappa_outer /= n_total_occ as f64;
        let expected = (0.04 + 0.01 + 0.09 + 0.04) / 4.0;
        assert!(
            (kappa_outer[(0, 0)] - expected).abs() < 1e-12,
            "IOV omega analytic update: got {:.6e}, expected {:.6e}",
            kappa_outer[(0, 0)],
            expected
        );
    }

    /// SAEM with the optimizer trace active must emit the per-parameter `val:*`
    /// columns and write `NA` for every `grad:*` column (SAEM has no OFV
    /// gradient). This is the fast test that registers PR coverage for the SAEM
    /// trace call site (#640), which is otherwise reached only by slow fits.
    #[test]
    fn saem_trace_emits_value_columns_with_na_grads() {
        use crate::types::{DoseEvent, FitOptions, Population};
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0, 0.5],
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subj],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let coord_names =
            crate::estimation::parameterization::coordinate_names(&model.default_params);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = format!(
            "/tmp/ferx_trace_saem_param_{}_{}.csv",
            std::process::id(),
            nanos
        );
        crate::estimation::trace::init(path.clone(), &coord_names).unwrap();

        let opts = FitOptions {
            saem_n_exploration: 2,
            saem_n_convergence: 0,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let _ = run_saem(&model, &population, &model.default_params, &opts);
        crate::estimation::trace::finish();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        let header: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
        assert!(header.iter().any(|c| c.starts_with("val:")));
        let n_coords = coord_names.len();
        let fixed = 17;
        let mut rows = 0;
        for row in lines {
            let c: Vec<String> = row.split(',').map(String::from).collect();
            assert_eq!(c.len(), fixed + 2 * n_coords, "SAEM row column count");
            // Every value column finite; every gradient column NA.
            for i in fixed..(fixed + n_coords) {
                assert!(c[i].parse::<f64>().is_ok(), "val column must be finite");
            }
            for i in (fixed + n_coords)..(fixed + 2 * n_coords) {
                assert_eq!(c[i], "NA", "SAEM grad column must be NA");
            }
            rows += 1;
        }
        assert!(rows >= 1, "expected at least one SAEM trace row");
        std::fs::remove_file(&path).ok();
    }

    // ---- #895: iiv_on_ruv sigma-ridge cap ----

    #[test]
    fn ruv_sigma_caps_none_without_iiv_on_ruv() {
        // A model with no iiv_on_ruv carries no cap on any sigma, so the M-step
        // sigma trajectory is left exactly as the un-capped code produced it.
        let caps = compute_ruv_sigma_caps(
            false,
            None,
            &[(-1.75_f64), 0.0],
            &[5.0, 5.0],
            &[false, false],
        );
        assert_eq!(caps, vec![None, None]);
    }

    #[test]
    fn ruv_sigma_caps_bounds_free_residual_sigma_growth() {
        // iiv_on_ruv active: the free residual sigma gets a cap of
        // log(σ₀) + SAEM_RUV_SIGMA_LN_GROWTH (well below the e⁵ ceiling).
        let ln_s0 = 0.1738_f64.ln(); // ≈ -1.75
        let caps = compute_ruv_sigma_caps(true, None, &[ln_s0], &[5.0], &[false]);
        assert_eq!(caps.len(), 1);
        let cap = caps[0].expect("free residual sigma must carry a cap");
        assert!((cap - (ln_s0 + SAEM_RUV_SIGMA_LN_GROWTH)).abs() < 1e-12);
        // e³ ≈ 20× growth in SD, and comfortably under the e⁵ ceiling.
        assert!(cap < 5.0);
        assert!((cap.exp() / 0.1738 - SAEM_RUV_SIGMA_LN_GROWTH.exp()).abs() < 1e-9);
    }

    #[test]
    fn ruv_sigma_caps_skip_fixed_and_frem_covariate_sigma() {
        // sigma[0] = free PK residual (capped), sigma[1] = FREM EPSCOV (skipped),
        // sigma[2] = FIXed (skipped).
        let caps = compute_ruv_sigma_caps(
            true,
            Some(1),
            &[0.0, -6.0, -1.0],
            &[5.0, 5.0, 5.0],
            &[false, true, true],
        );
        assert!(caps[0].is_some());
        assert_eq!(caps[1], None, "FREM EPSCOV must not be capped");
        assert_eq!(caps[2], None, "FIXed sigma must not be capped");
    }

    #[test]
    fn ruv_sigma_caps_defer_to_tighter_user_upper_bound() {
        // If the model's own upper bound is tighter than log σ₀ + growth, NLopt
        // already enforces it, so no RUV growth cap is carried — a σ converging to
        // its own bound must not be mis-flagged as hitting the iiv_on_ruv cap (#903).
        let caps = compute_ruv_sigma_caps(true, None, &[0.0], &[1.0], &[false]);
        assert_eq!(caps[0], None);
        // A growth cap strictly below the upper bound is still carried.
        let caps = compute_ruv_sigma_caps(true, None, &[0.0], &[5.0], &[false]);
        assert_eq!(caps[0], Some(SAEM_RUV_SIGMA_LN_GROWTH));
    }

    // ---- #904: iiv_on_ruv eta re-centering ----

    #[test]
    fn recenter_ruv_eta_zeroes_mean_and_preserves_residual_variance() {
        // η_RUV at index 1, with a non-zero mean; two etas per subject.
        let mut etas = vec![vec![0.3, 0.8], vec![-0.1, -0.4], vec![0.2, 0.2]];
        let kr = 1;
        // Per-subject residual variance before: σ²·exp(2·η_RUV) (σ = sd = 0.2).
        let sd0 = 0.2_f64;
        let var_before: Vec<f64> = etas
            .iter()
            .map(|e| sd0 * sd0 * (2.0_f64 * e[kr]).exp())
            .collect();
        let mean_in = (0.8 - 0.4 + 0.2) / 3.0;

        let mut log_sigma = vec![sd0.ln()];
        let mut sigma_vals = vec![sd0];
        let removed = recenter_ruv_eta(&mut etas, kr, &mut log_sigma, &mut sigma_vals, &[true]);

        // Removed exactly the mean; η_RUV now zero-mean.
        assert!((removed - mean_in).abs() < 1e-12);
        let mean_after = etas.iter().map(|e| e[kr]).sum::<f64>() / 3.0;
        assert!(mean_after.abs() < 1e-12);
        // σ scaled by exp(mean).
        assert!((sigma_vals[0] - sd0 * mean_in.exp()).abs() < 1e-12);
        assert!((log_sigma[0] - (sd0.ln() + mean_in)).abs() < 1e-12);
        // Each subject's residual variance is exactly unchanged (the invariance).
        for (e, &v0) in etas.iter().zip(var_before.iter()) {
            let v_after = sigma_vals[0] * sigma_vals[0] * (2.0_f64 * e[kr]).exp();
            assert!(
                (v_after - v0).abs() < 1e-12,
                "residual variance moved: {v_after} vs {v0}"
            );
        }
        // Non-RUV coordinate untouched.
        assert_eq!(etas[0][0], 0.3);
    }

    #[test]
    fn ruv_recenter_allowed_only_when_all_residual_sigma_free() {
        // No iiv_on_ruv → never re-center.
        assert!(!ruv_recenter_allowed(false, None, &[false]));
        // Single free residual σ → OK.
        assert!(ruv_recenter_allowed(true, None, &[false]));
        // Combined error, additive FIXed (index 1) → NOT OK: scaling only the free
        // proportional σ would leave the fixed additive part uncompensated and
        // break the residual-variance invariance (the #903 review finding).
        assert!(!ruv_recenter_allowed(true, None, &[false, true]));
        // FREM EPSCOV (index 1, always FIX) is exempt — a free PK residual σ at 0
        // still re-centers.
        assert!(ruv_recenter_allowed(true, Some(1), &[false, true]));
        // FREM EPSCOV exempt, but a second *real* residual σ FIXed (index 2) blocks.
        assert!(!ruv_recenter_allowed(true, Some(1), &[false, true, true]));
    }

    #[test]
    fn recenter_ruv_eta_noop_when_no_sigma_absorbs() {
        // Every RUV σ FIXed (none absorbs) → η mean is left in place, σ untouched.
        let mut etas = vec![vec![0.5], vec![-0.1]];
        let mut log_sigma = vec![0.2_f64.ln()];
        let mut sigma_vals = vec![0.2];
        let removed = recenter_ruv_eta(&mut etas, 0, &mut log_sigma, &mut sigma_vals, &[false]);
        assert_eq!(removed, 0.0);
        assert_eq!(etas, vec![vec![0.5], vec![-0.1]]);
        assert_eq!(sigma_vals, vec![0.2]);
    }

    // ---- #895: iiv_on_ruv omega-ridge cap ----

    #[test]
    fn ruv_omega_cap_none_without_ruv_or_when_fixed() {
        let om = DMatrix::from_diagonal(&DVector::from_vec(vec![0.3, 0.2]));
        // No iiv_on_ruv eta.
        assert_eq!(compute_ruv_omega_cap(None, 2, &om, &[false, false]), None);
        // RUV eta index out of range.
        assert_eq!(
            compute_ruv_omega_cap(Some(5), 2, &om, &[false, false]),
            None
        );
        // RUV omega FIXed.
        assert_eq!(compute_ruv_omega_cap(Some(0), 2, &om, &[true, false]), None);
    }

    #[test]
    fn ruv_omega_cap_is_log_variance_plus_growth() {
        let om = DMatrix::from_diagonal(&DVector::from_vec(vec![0.2977886, 0.2]));
        let cap = compute_ruv_omega_cap(Some(0), 2, &om, &[false, false]).unwrap();
        assert!((cap - (0.2977886_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH)).abs() < 1e-12);
        // ≈ 20× the starting variance.
        assert!((cap.exp() / 0.2977886 - SAEM_RUV_OMEGA_LN_GROWTH.exp()).abs() < 1e-9);
    }

    #[test]
    fn apply_ruv_omega_cap_noop_below_cap() {
        let mut om = DMatrix::from_row_slice(2, 2, &[0.3, 0.05, 0.05, 0.2]);
        let before = om.clone();
        // cap = exp(log(0.3)+3) ≈ 6.0, well above 0.3 → no change.
        let clamped = apply_ruv_omega_cap(
            &mut om,
            0,
            0.3_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH,
            &[false, false],
        );
        assert!(!clamped);
        assert_eq!(om, before);
    }

    #[test]
    fn apply_ruv_omega_cap_leaves_fixed_offdiagonal_untouched() {
        // ω_RUV (index 0) runs away to 9.0 with a covariance to a FIXed eta (index
        // 1). The cap must pull the diagonal down but leave the user-declared FIXed
        // covariance Ω[0,1] exactly as-is (#903 review), never silently rescale it.
        let cov = 0.5_f64;
        let mut om = DMatrix::from_row_slice(2, 2, &[9.0, cov, cov, 0.2]);
        let log_cap = 0.2977886_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH;
        let clamped = apply_ruv_omega_cap(&mut om, 0, log_cap, &[false, true]);
        assert!(clamped);
        assert!((om[(0, 0)] - log_cap.exp()).abs() < 1e-9);
        // FIXed off-diagonal unchanged (both symmetric entries).
        assert_eq!(om[(0, 1)], cov);
        assert_eq!(om[(1, 0)], cov);
        // Diagonal shrank while the off-diagonal held, so it stays PD here
        // (det = 5.98·0.2 − 0.25 > 0).
        let det = om[(0, 0)] * om[(1, 1)] - om[(0, 1)] * om[(1, 0)];
        assert!(det > 0.0);
    }

    #[test]
    fn apply_ruv_omega_cap_preserves_correlation_and_pd() {
        // ω_RUV runaway to 9.0, correlated with a second eta (var 0.2).
        let cov = 0.9_f64; // corr = 0.9/sqrt(9*0.2) ≈ 0.671
        let mut om = DMatrix::from_row_slice(2, 2, &[9.0, cov, cov, 0.2]);
        let corr_before = om[(0, 1)] / (om[(0, 0)] * om[(1, 1)]).sqrt();
        let log_cap = 0.2977886_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH; // ≈ log(5.98)
        let clamped = apply_ruv_omega_cap(&mut om, 0, log_cap, &[false, false]);
        assert!(clamped);
        // Diagonal pulled down to the cap.
        assert!((om[(0, 0)] - log_cap.exp()).abs() < 1e-9);
        // Correlation with the other eta preserved.
        let corr_after = om[(0, 1)] / (om[(0, 0)] * om[(1, 1)]).sqrt();
        assert!((corr_after - corr_before).abs() < 1e-9);
        // Still symmetric and positive-definite (2×2: det > 0, diag > 0).
        assert_eq!(om[(0, 1)], om[(1, 0)]);
        let det = om[(0, 0)] * om[(1, 1)] - om[(0, 1)] * om[(1, 0)];
        assert!(det > 0.0 && om[(0, 0)] > 0.0);
    }

    // ---- #903 review: end-of-exploration cap re-anchoring ----

    #[test]
    fn reanchor_ruv_sigma_caps_loosens_for_low_init_only() {
        // σ init 0.01 (log ≈ -4.6): init cap = -4.6 + 3 = -1.6 (≈ 0.2). If the fit
        // legitimately settled near σ = 0.3 (log ≈ -1.2) by end of exploration, the
        // cap must loosen to -1.2 + 3 = 1.8, not clamp the well-posed fit at 0.2.
        let ln_init = 0.01_f64.ln();
        let mut caps = vec![Some(ln_init + SAEM_RUV_SIGMA_LN_GROWTH)];
        let log_sigma_settled = vec![0.3_f64.ln()];
        reanchor_ruv_sigma_caps(&mut caps, &log_sigma_settled, &[5.0]);
        assert!((caps[0].unwrap() - (0.3_f64.ln() + SAEM_RUV_SIGMA_LN_GROWTH)).abs() < 1e-12);

        // If σ never moved above its init, the cap is unchanged (only loosens).
        let mut caps = vec![Some(ln_init + SAEM_RUV_SIGMA_LN_GROWTH)];
        reanchor_ruv_sigma_caps(&mut caps, &[ln_init], &[5.0]);
        assert!((caps[0].unwrap() - (ln_init + SAEM_RUV_SIGMA_LN_GROWTH)).abs() < 1e-12);

        // Never loosens past the user's own upper bound.
        let mut caps = vec![Some(ln_init + SAEM_RUV_SIGMA_LN_GROWTH)];
        reanchor_ruv_sigma_caps(&mut caps, &[10.0], &[2.0]);
        assert_eq!(caps[0], Some(2.0));

        // `None` (uncapped σ) stays `None`.
        let mut caps = vec![None];
        reanchor_ruv_sigma_caps(&mut caps, &[0.0], &[5.0]);
        assert_eq!(caps[0], None);
    }

    #[test]
    fn reanchor_ruv_omega_cap_loosens_only() {
        let ln_init = 0.01_f64.ln();
        let cap = Some(ln_init + SAEM_RUV_OMEGA_LN_GROWTH);
        // Settled ω_RUV = 0.3 → loosen to log(0.3) + growth.
        let out = reanchor_ruv_omega_cap(cap, 0.3);
        assert!((out.unwrap() - (0.3_f64.ln() + SAEM_RUV_OMEGA_LN_GROWTH)).abs() < 1e-12);
        // Settled below init → unchanged.
        assert_eq!(reanchor_ruv_omega_cap(cap, 0.01), cap);
        // No cap stays none; non-positive variance is a no-op.
        assert_eq!(reanchor_ruv_omega_cap(None, 0.3), None);
        assert_eq!(reanchor_ruv_omega_cap(cap, 0.0), cap);
    }

    // ---- #895: MH mixing warning ----

    #[test]
    fn saem_mixing_warning_fires_only_when_stuck() {
        // 0% acceptance over the post-burn-in window → warn.
        assert!(saem_mixing_warning(0, 100_000).is_some());
        // Just below the 1% threshold → warn.
        assert!(saem_mixing_warning(50, 100_000).is_some());
        // Healthy acceptance → silent.
        assert!(saem_mixing_warning(8_000, 100_000).is_none());
        // No proposals accumulated (e.g. burn-in ≥ n_iter) → silent, no div-by-0.
        assert!(saem_mixing_warning(0, 0).is_none());
    }

    // ---- #895: block-kernel per-coordinate scaling ----

    fn one_obs_subject() -> Subject {
        use crate::types::DoseEvent;
        use std::collections::HashMap;
        Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: vec![],
            obs_l2: Vec::new(),
            dose_occasions: vec![],
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    }

    #[test]
    fn mh_steps_block_scale_none_equals_all_ones() {
        // `None` must reproduce the plain chol(Ω)·z block move bit-for-bit, and an
        // explicit all-ones scale must be identical to it — the multiplier is a
        // no-op at 1.0, so non-FREM models are unaffected.
        use crate::stats::likelihood::individual_nll;
        use crate::types::SigmaVector;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let model = analytical_model(GradientMethod::Auto);
        let subj = one_obs_subject();
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let sigma = SigmaVector {
            values: vec![0.1],
            names: vec!["PROP".into()],
        };
        let theta = vec![1.0, 8.0];

        let run = |scale: Option<&[f64]>| {
            let mut eta = vec![0.2_f64, -0.1];
            let nll0 = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
            let mut rng = StdRng::seed_from_u64(7);
            let mut scratch = EventPkParams::with_capacity_for(&subj);
            let (acc, nll) = mh_steps(
                &mut eta,
                nll0,
                &subj,
                &model,
                &theta,
                &omega,
                &sigma.values,
                0.5,
                scale,
                &mut rng,
                50,
                &mut scratch,
                None,
            );
            (eta, acc, nll)
        };

        let (eta_none, acc_none, nll_none) = run(None);
        let (eta_ones, acc_ones, nll_ones) = run(Some(&[1.0, 1.0]));
        assert_eq!(eta_none, eta_ones);
        assert_eq!(acc_none, acc_ones);
        assert_eq!(nll_none, nll_ones);
    }

    #[test]
    fn mh_steps_block_scale_freezes_damped_coordinate() {
        // A block scale of 0 on coordinate 1 means the joint proposal never moves
        // that coordinate (its perturbation is multiplied by 0), while coordinate
        // 0 is free to move — the mechanism that damps near-deterministic FREM
        // covariate ETAs so the block kernel can still explore the PK block.
        use crate::stats::likelihood::individual_nll;
        use crate::types::SigmaVector;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let model = analytical_model(GradientMethod::Auto);
        let subj = one_obs_subject();
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let sigma = SigmaVector {
            values: vec![0.1],
            names: vec!["PROP".into()],
        };
        let theta = vec![1.0, 8.0];
        let eta0 = vec![0.2_f64, -0.1];

        let mut eta = eta0.clone();
        let nll0 = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
        let mut rng = StdRng::seed_from_u64(11);
        let mut scratch = EventPkParams::with_capacity_for(&subj);
        mh_steps(
            &mut eta,
            nll0,
            &subj,
            &model,
            &theta,
            &omega,
            &sigma.values,
            0.8,
            Some(&[1.0, 0.0]),
            &mut rng,
            100,
            &mut scratch,
            None,
        );
        // Coordinate 1 is frozen at its start; coordinate 0 has moved.
        assert_eq!(
            eta[1], eta0[1],
            "coordinate with block scale 0 must not move"
        );
        assert_ne!(eta[0], eta0[0], "unclamped coordinate should have moved");
    }
}
