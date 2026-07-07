/// SAEM (Stochastic Approximation EM) for NLME population parameter estimation.
///
/// Reference: Delyon, Lavielle, Moulines (1999) Annals of Statistics 94–128.
///            Kuhn & Lavielle (2004) ESAIM: Probability and Statistics 8:115–131.
///
/// Two-phase step-size schedule (Monolix convention):
///   Phase 1 (exploration, k ≤ K1):  γₖ = 1          — rapid basin convergence
///   Phase 2 (convergence, k > K1):  γₖ = 1/(k−K1)   — almost-sure convergence to MLE
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{
    compute_covariance, pop_nll, CovarianceStepResult, OuterResult,
};
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

/// Maximum per-iteration stochastic-approximation step for the Ω sufficient
/// statistic *during the exploration phase*. The θ/σ M-step uses the full γ
/// (1.0 in exploration), but Ω is averaged at no more than this rate so a single
/// un-equilibrated MCMC draw cannot overwrite a correlated Ω and trigger the
/// rank-1 collapse feedback. In the convergence phase the cap is lifted and Ω
/// uses the full decaying γ = 1/(k−k1), the same Robbins-Monro schedule as θ.
const OMEGA_SA_MAX_STEP: f64 = 0.1;

/// Raise every *free* diagonal entry of the BSV Ω that has fallen below `floor`
/// up to `floor`. FIX-ed diagonals (`omega_fixed[i] == true`) are left untouched
/// — they carry the user's declared variance and must not be perturbed.
fn floor_omega_diagonal(omega_mat: &mut DMatrix<f64>, omega_fixed: &[bool], floor: f64) {
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
            .map(|j| eta[j] + step_scale * perturbation[j])
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

    // TTE term: same convention as obs_nll_subject_into (weight 1.0 to match
    // the true NLL, since Gaussian obs already contribute at 0.5*(log v + r²/v)).
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        use crate::survival::tte_data_term;
        use crate::types::EndpointLikelihood;
        for (cmt, endpoint) in &model.endpoints {
            if let EndpointLikelihood::Tte { hazard } = endpoint {
                let records_for_cmt: Vec<crate::types::ObsRecord> = subject
                    .obs_records
                    .iter()
                    .filter(
                        |r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt),
                    )
                    .cloned()
                    .collect();
                if records_for_cmt.is_empty() {
                    continue;
                }
                total_nll +=
                    tte_data_term(&records_for_cmt, hazard, theta, eta, &subject.covariates);
            }
        }
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
                    .map_init(EventPkParams::default, |scratch, ((subject, eta), kaps)| {
                        obs_nll_subject_grad_iov(
                            model,
                            subject,
                            &th,
                            &sg,
                            eta,
                            kaps,
                            &theta_packs_log_mask,
                            &lower,
                            &upper,
                            n_theta,
                            n_sigma,
                            scratch,
                        )
                    })
                    .collect();
                fold_nll_grad(per_subj, n)
            } else {
                let per_subj: Vec<(f64, Vec<f64>)> = population
                    .subjects
                    .par_iter()
                    .zip(etas.par_iter())
                    .map_init(EventPkParams::default, |scratch, (subject, eta)| {
                        obs_nll_subject_grad(
                            model,
                            subject,
                            &th,
                            &sg,
                            eta,
                            &theta_packs_log_mask,
                            &lower,
                            &upper,
                            n_theta,
                            n_sigma,
                            scratch,
                        )
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
            let val = if let Some(kappas) = kappas_opt {
                obs_nll_sum_iov(model, population, &th, &sg, etas, kappas)
            } else {
                obs_nll_sum(model, population, &th, &sg, etas)
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

/// Build (theta_idx, eta_idx) pairs for log-transformed mu-references only.
///
/// Only `log_transformed = true` mu-refs (patterns `THETA*exp(ETA)` and
/// `exp(log(THETA)+ETA)`) participate in the gradient-step M-step.  For these
/// the chain rule gives `d/d_log(theta) = -Σᵢ d/d_eta`, which matches the
/// update applied in the SAEM loop.  Additive mu-refs (`THETA + ETA`,
/// `log_transformed = false`) require the extra factor of `theta` from the
/// log-space chain rule and are deliberately excluded — they fall through to
/// the regular NLopt M-step.
fn get_mu_ref_pairs(model: &CompiledModel) -> Vec<(usize, usize)> {
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

// ---------------------------------------------------------------------------
// Main SAEM loop
// ---------------------------------------------------------------------------

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
        && model.residual_error_eta.is_none();

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
    let use_closed_form_mstep = options.mu_referencing && !mu_ref_pairs.is_empty();
    // Accumulator for the `obs_nll_sum` (population OFV) evaluations skipped
    // by pinning mu-ref dims out of NLopt's central-FD gradient.  Each pinned
    // dim costs `2 * mstep_maxiter` `obs_nll_sum` calls inside NLopt — that's
    // the value we add per M-step that takes the closed-form branch.
    let mut mstep_grad_step_evals_saved: u64 = 0;

    // Per-subject flag: did this subject successfully use HMC at least once?
    // Only meaningful when `using_hmc = true`; stays all-false otherwise.
    let mut hmc_subjects = vec![false; n_subjects];

    // Main loop
    for k in 1..=n_iter {
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
            use rayon::prelude::*;
            let theta_ref = &state.theta;
            let sigma_ref = &state.sigma_vals;
            let omega_ref = &omega_k;
            // Per-coordinate componentwise proposal SDs — computed once here (Ω's
            // diagonal is shared across subjects) rather than per subject inside
            // the parallel kernel. Floored to match the Ω diagonal floor.
            let cw_sd: Vec<f64> = (0..n_eta)
                .map(|j| omega_k.matrix[(j, j)].max(SAEM_OMEGA_DIAG_FLOOR).sqrt())
                .collect();
            let cw_sd_ref = &cw_sd;
            // For IOV models, eta proposals must target p(η | κ, θ, data):
            // the per-occasion [eta_prop, kappa_k] predictions determine
            // which etas are accepted.  Pass omega_iov to mh_steps so it
            // can call individual_nll_iov with kappas held fixed.
            let omega_iov_for_eta_mh: Option<&OmegaMatrix> = omega_iov_cur_opt.as_ref();

            // Returns (eta_new, nll_after, n_acc_primary, n_prop_primary,
            //          per_eta_acc_cw, n_sweeps_cw, used_hmc)
            let results: Vec<(Vec<f64>, f64, usize, usize, Vec<usize>, usize, bool)> = state
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
                        let mut eta_work = eta.clone();

                        // ---- Kernel 1: primary block move ----
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
                        )
                    },
                )
                .collect();

            for (i, (eta_new, nll_new, n_acc, n_prop, per_eta_acc_cw, _n_prop_cw, used_hmc)) in
                results.into_iter().enumerate()
            {
                state.etas[i] = eta_new;
                state.nll_cache[i] = nll_new;
                state.accept_counts[i] += n_acc;
                state.proposal_counts[i] += n_prop;
                // Accumulate per-eta CW acceptance counts
                for j in 0..n_eta {
                    state.cw_accept_counts[i][j] += per_eta_acc_cw[j];
                    state.cw_proposal_counts[i][j] += n_cw_sweeps;
                }
                hmc_subjects[i] |= used_hmc;
            }
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
                        &omega_k,
                        Some(omega_iov_cur),
                        &state.sigma_vals,
                    );
                    let (n_acc, n_prop, nll_new) = mh_kappa_steps(
                        &mut state.kappas[i],
                        nll_kappa_ref,
                        subject,
                        model,
                        &state.theta,
                        &state.etas[i],
                        &omega_k,
                        omega_iov_cur,
                        &state.sigma_vals,
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
        let mut eta_outer = DMatrix::zeros(n_eta, n_eta);
        for eta in &state.etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= n_subjects as f64;

        state.s2 = (1.0 - gamma_omega) * &state.s2 + gamma_omega * &eta_outer;

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

            if use_closed_form_mstep {
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
                );
                log_theta = theta_new;
                log_sigma = sigma_new;
            } else {
                // mu_referencing = false: full NLopt M-step for all thetas + sigma (unchanged)
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
                );
                log_theta = theta_new;
                log_sigma = sigma_new;
            }

            state.theta = (0..n_theta)
                .map(|i| unpack_theta(i, log_theta[i]))
                .collect();
            state.sigma_vals = log_sigma.iter().map(|&v| v.exp()).collect();
        }

        // ---- Update NLL cache (parallelized, needed for MH acceptance ratios) ----
        let omega_upd = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
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
                    individual_nll_iov(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        &state.etas[i],
                        &state.kappas[i],
                        &omega_upd,
                        omega_iov_upd.as_ref(),
                        &state.sigma_vals,
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
            let new_nlls: Vec<f64> = state
                .etas
                .par_iter()
                .enumerate()
                .map_init(EventPkParams::default, |scratch, (i, eta)| {
                    individual_nll_into(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        eta,
                        &omega_upd,
                        &state.sigma_vals,
                        scratch,
                    )
                })
                .collect();
            state.nll_cache = new_nlls;
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
            // Rolling accept rate since the last adapt reset (per-subject proposal counts
            // as denominator so mixed HMC/MH runs report a meaningful rate).
            let total_proposals: usize = state.proposal_counts.iter().sum();
            let mh_accept_rate: f64 =
                state.accept_counts.iter().sum::<usize>() as f64 / total_proposals.max(1) as f64;

            if verbose && (k == 1 || k % 50 == 0 || k == n_iter) {
                eprintln!(
                    "  SAEM iter {:>4}/{} [{}] γ={:.3}  condNLL={:.3}",
                    k, n_iter, phase, gamma, cond_nll
                );
            }

            crate::estimation::trace::write_saem(k, phase, cond_nll, gamma, mh_accept_rate);
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
    let final_omega = OmegaMatrix::from_matrix(
        state.omega_mat.clone(),
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    let final_params = ModelParameters {
        theta: state.theta.clone(),
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: final_omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: state.sigma_vals.clone(),
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        omega_iov: if n_kappa > 0 {
            // Use from_matrix_with_mask so structural free_mask is preserved
            // when this OuterResult is handed to a chained estimator (e.g.
            // [saem, foce]); from_matrix would infer the mask from nonzeros
            // and could mark a legitimately-zero off-diagonal as structurally
            // fixed, corrupting the next estimator's parameterisation.
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
    };

    if combined_additive_sigma_at_floor(model, &final_params) {
        warnings
            .push("SAEM combined-error additive sigma collapsed to its lower bound.".to_string());
    }

    // ---- Final EBEs via inner loop (warm-started from SAEM etas) ----
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
    );

    // ---- Final OFV via FOCE approximation (for AIC/BIC comparability) ----
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

    // ---- Covariance step ----
    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    let (covariance_matrix, covariance_wall_time_secs) =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("Running covariance step...");
            }
            let cov_timer = std::time::Instant::now();
            let packed = pack_params(&final_params);
            let cm = match compute_covariance(
                &packed,
                &final_params,
                model,
                population,
                &eta_hats,
                &h_matrices,
                &final_kappas,
                options,
            ) {
                CovarianceStepResult::Success(out) => {
                    warnings.extend(out.warnings);
                    Some(out.matrix)
                }
                CovarianceStepResult::Unusable(msg) => {
                    warnings.push(msg);
                    None
                }
                CovarianceStepResult::FailedNonPd {
                    reason,
                    fallback_proposal,
                } => {
                    warnings.push(reason);
                    sir_fallback_proposal = Some(fallback_proposal);
                    None
                }
            };
            (cm, cov_timer.elapsed().as_secs_f64())
        } else {
            (None, 0.0)
        };

    if verbose {
        eprintln!("SAEM completed. Final OFV = {:.4}", ofv);
    }

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
    let cond_dist = if options.saem_conddist {
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, MuRef};

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
            dose_occasions: vec![],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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
            dose_occasions: vec![],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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
            dose_occasions: vec![],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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
            transit_ode_equivalent: None,
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
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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
            dose_occasions: vec![],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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
            dose_occasions: vec![1u32],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
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
}
