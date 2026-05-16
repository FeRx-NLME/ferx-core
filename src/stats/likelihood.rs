use crate::pk;
use crate::stats::residual_error::{compute_r_diag, residual_variance};
use crate::stats::special::log_normal_cdf;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Route predictions through analytical PK or ODE solver depending on model,
/// honouring per-event PK parameters when the subject has time-varying
/// covariates. The TV-aware dispatcher in `pk::compute_predictions_with_tv`
/// handles the analytical / ODE / event-driven branching.
///
/// This is the canonical predictions entry point for FOCE/FOCEI inner-loop
/// objectives. Callers must pass the same `(theta, eta)` they use elsewhere
/// in the NLL — `pk_param_fn` is invoked internally once per event (TV path)
/// or once per subject (no-TV path).
///
/// Allocates a fresh `EventPkParams` scratch on each call. Hot loops should
/// use [`model_predictions_into`] with a reused scratch buffer instead.
#[inline]
fn model_predictions(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    pk::compute_predictions_with_tv(model, subject, theta, eta)
}

/// Caller-owned-scratch variant of [`model_predictions`] that also
/// accepts an optional pre-built
/// [`pk::event_driven::EventSchedule`]. Used by FOCE inner-loop callers
/// (BFGS line search, post-convergence eval) that build the schedule
/// once per `find_ebe` call and reuse it across many `(theta, eta)`
/// evaluations of the same subject. SAEM and other callers pass `None`
/// — the no-TV fast path doesn't consume the schedule, and the
/// dispatcher falls back to building one on demand on the TV path.
#[inline]
fn model_predictions_into_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> Vec<f64> {
    pk::compute_predictions_with_tv_into_with_schedule(
        model, subject, theta, eta, scratch, schedule,
    )
}

/// True when observation `j` of `subject` is censored AND the model requests M3.
fn is_m3_bloq(model: &CompiledModel, subject: &Subject, j: usize) -> bool {
    matches!(model.bloq_method, BloqMethod::M3) && subject.cens.get(j).copied().unwrap_or(0) != 0
}

/// Compute individual negative log-likelihood for EBE estimation (inner loop objective).
///
/// NLL(eta | subject) = 0.5 * [eta'*Omega_inv*eta + log|Omega|
///                             + sum_j( term_j )]
/// where term_j is:
///   - `(y_j - f_j)² / V_j + log(V_j)` for quantified observations, or
///   - `-2·log Φ((LLOQ_j - f_j)/√V_j)` for M3-censored observations (CENS=1)
///     with LLOQ_j carried in `observations[j]`.
pub fn individual_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
) -> f64 {
    // Allocate-on-each-call wrapper — see `individual_nll_into` for
    // the scratch-aware version used by SAEM's MH loop.
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    individual_nll_into(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        &mut scratch,
    )
}

/// Same as [`individual_nll`] but uses a caller-owned scratch buffer.
/// The hot-path entry point for SAEM's MH proposals: a single buffer
/// allocated outside the per-subject MH loop is reused across all
/// proposed `eta`s, eliminating the per-call `Vec<PkParams>` churn
/// that previously dominated SAEM allocator pressure on TV-cov data.
pub fn individual_nll_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    scratch: &mut pk::EventPkParams,
) -> f64 {
    individual_nll_into_with_schedule(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        scratch,
        None,
    )
}

/// Hot-path variant that additionally threads through a pre-built
/// [`pk::event_driven::EventSchedule`]. The FOCE inner-loop obj closure
/// and Jacobian build the schedule once per `find_ebe` call and reuse
/// it across all BFGS iterations.
pub fn individual_nll_into_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> f64 {
    // Compute Omega inverse and log-determinant via Cholesky
    let omega_inv = match omega.matrix.clone().cholesky() {
        Some(chol) => chol.inverse(),
        None => return 1e20,
    };
    let log_det_omega = omega_log_det(omega);

    // Eta prior: eta' * Omega_inv * eta
    let eta_vec = DVector::from_column_slice(eta);
    let eta_prior = eta_vec.dot(&(&omega_inv * &eta_vec));

    // Compute individual predictions using the caller's scratch buffer
    // for per-event PK params (only consumed on the TV-cov path; ignored
    // on the no-TV fast path).
    let preds = model_predictions_into_with_schedule(model, subject, theta, eta, scratch, schedule);
    // For SDE models, compute per-observation EKF process-noise variance and
    // add it to the residual variance to form V_total.
    let p_obs = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta, sigma_values)
    } else {
        Vec::new()
    };
    let mut data_ll = 0.0;
    for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
        let v_resid = residual_variance(model.error_model, f_pred, sigma_values);
        let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
        if is_m3_bloq(model, subject, j) {
            // y carries LLOQ on CENS=1 rows.
            let z = (y - f_pred) / v.sqrt();
            data_ll += -2.0 * log_normal_cdf(z);
        } else {
            let resid = y - f_pred;
            data_ll += resid * resid / v + v.ln();
        }
    }

    0.5 * (eta_prior + log_det_omega + data_ll)
}

/// Compute per-observation EKF process-noise variance (p_obs) for an SDE model.
///
/// Returns an empty vec when `model.is_sde()` is false — callers should check
/// `model.is_sde()` before calling this to avoid an unnecessary ODE pass.
fn ekf_p_obs(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    sigma_values: &[f64],
) -> Vec<f64> {
    let (start, state_indices) = match model.diffusion_theta_start {
        Some(s) => (s, &model.diffusion_state_indices),
        None => return Vec::new(),
    };
    let ode = match model.ode_spec.as_ref() {
        Some(o) => o,
        None => return Vec::new(),
    };

    // Build current diffusion_var from the live theta slice — this is what
    // changes each outer iteration as the optimizer updates diffusion thetas.
    let mut diffusion_var = vec![0.0f64; ode.n_states];
    for (k, &state_idx) in state_indices.iter().enumerate() {
        let theta_idx = start + k;
        if theta_idx < theta.len() && state_idx < ode.n_states {
            diffusion_var[state_idx] = theta[theta_idx].max(0.0);
        }
    }

    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);
    let error_model = model.error_model;

    // Temporarily shadow ode_spec with updated diffusion_var for this call.
    // We cannot mutate model.ode_spec, so we pass diffusion_var separately
    // via a local OdeSpec-like struct. Since solve_ekf takes rhs + n_states
    // + obs_cmt_idx + diffusion_var as separate args, we call it directly.
    // TODO: unify EKF ipred with likelihood ipred to avoid double ODE evaluation
    let (_, p_obs) = crate::ode::ode_predictions_ekf_with_diffusion(
        ode,
        &pk.values,
        subject,
        &diffusion_var,
        |f_pred| crate::stats::residual_error::residual_variance(error_model, f_pred, sigma_values),
    );
    p_obs
}

/// Log-determinant of Omega via Cholesky: log|Omega| = 2 * sum(log(L_ii))
fn omega_log_det(omega: &OmegaMatrix) -> f64 {
    let n = omega.chol.nrows();
    let mut ld = 0.0;
    for i in 0..n {
        let lii = omega.chol[(i, i)];
        if lii > 0.0 {
            ld += lii.ln();
        } else {
            return 1e20;
        }
    }
    2.0 * ld
}

/// FOCE per-subject negative log-likelihood.
///
/// Non-interaction (standard FOCE):
///   NLL_i = 0.5 * [(y - f0)' * R_tilde_inv * (y - f0) + log|R_tilde|]
///   where f0 = f(eta_hat) - H * eta_hat  (linearized population prediction)
///         R_tilde = H * Omega * H' + R(f0)
///
/// When M3 BLOQ is active and the subject has any CENS=1 row, we route through
/// the interaction path: mixing a linearized Gaussian term with a non-linearized
/// `log Φ(·)` BLOQ term produces inconsistent OFVs near the LLOQ boundary, so we
/// promote the whole subject to FOCEI — which is what NONMEM LAPLACE+M3 does in
/// practice.
pub fn foce_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    // Individual predictions at eta_hat (per-event PK when subject has TV covariates).
    let ipreds = model_predictions(model, subject, theta, eta_hat.as_slice());

    // For SDE models, inflate R with the EKF process-noise variance.
    let p_obs = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta_hat.as_slice(), sigma_values)
    } else {
        Vec::new()
    };

    let m3_active = matches!(model.bloq_method, BloqMethod::M3) && subject.has_bloq();

    if interaction || m3_active {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            model.error_model,
            model.bloq_method,
            &p_obs,
        )
    } else {
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            model.error_model,
            model.bloq_method,
            &p_obs,
        )
    }
}

/// Standard FOCE (no interaction). When any CENS rows are present AND
/// `bloq_method == M3`, the dispatcher has already routed to the interaction
/// path — so inside this function the only case we need to handle is
/// `bloq_method == Drop` (treat CENS rows as ordinary obs) or no CENS at all.
pub fn foce_subject_nll_standard(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
    _bloq_method: BloqMethod,
    p_obs: &[f64],
) -> f64 {
    let n_obs = subject.observations.len();

    // f0 = ipred - H * eta_hat (linearized population prediction)
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // R diagonal at f0; inflate with EKF process-noise variance for SDE models.
    let mut r_diag = compute_r_diag(error_model, &f0, sigma_values);
    for (j, r) in r_diag.iter_mut().enumerate() {
        *r += p_obs.get(j).copied().unwrap_or(0.0);
    }

    // R_tilde = H * Omega * H' + diag(R)
    let r_tilde = compute_r_tilde(h_matrix, &omega.matrix, &r_diag);

    // Cholesky of R_tilde
    let chol = match r_tilde.clone().cholesky() {
        Some(c) => c,
        None => return 1e20,
    };

    // Residuals: y - f0
    let residuals: DVector<f64> = DVector::from_iterator(
        n_obs,
        subject
            .observations
            .iter()
            .zip(f0.iter())
            .map(|(&y, &f)| y - f),
    );

    // (y - f0)' * R_tilde_inv * (y - f0)
    let solved = chol.solve(&residuals);
    let quad_form = residuals.dot(&solved);

    // log|R_tilde|
    let log_det_r = chol_log_det(&chol.l());

    0.5 * (quad_form + log_det_r)
}

/// FOCEI per-subject NLL.
///
/// Same Sheiner–Beal linearised marginal form as the standard FOCE path
/// (`(y - f₀)' R̃⁻¹ (y - f₀) + log|R̃|`), but with R evaluated at η̂
/// (the "interaction" piece) — this is what NONMEM's `METHOD=1 INTER`
/// reports.
///
/// The previous implementation decomposed via the Laplace identity
/// (`(y - f)' diag(R)⁻¹ (y - f) + η̂' Ω⁻¹ η̂ + log|R̃|`). For *linear*
/// models that decomposition is exactly equal to the Sheiner–Beal form
/// at the EBE, but for nonlinear models the linearised residual `y - f₀`
/// pulled through `R̃⁻¹` is not the same as the per-row `(y - f)/R(η̂)`
/// quadratic, and the OFV drifts away from NONMEM by a few units per
/// subject. The standard form below stays in lockstep with NONMEM at
/// matched parameter values for both linear and nonlinear PK.
///
/// With `bloq_method == M3`, BLOQ observations are dropped from the
/// Gaussian residual sum and the R̃ Cholesky, and instead contribute
/// `-2·log Φ((LLOQ - f)/√V)` evaluated at η̂.
pub fn foce_subject_nll_interaction(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
    bloq_method: BloqMethod,
    p_obs: &[f64],
) -> f64 {
    let n_obs = subject.observations.len();

    // Linearisation point: f₀ = ipred - H · η̂. (Same construction as
    // the no-interaction path; the only FOCEI difference is that R is
    // evaluated at η̂/ipred instead of at f₀.)
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // Partition observation indices into quantified vs BLOQ (M3 only).
    let (quant_idx, bloq_idx): (Vec<usize>, Vec<usize>) = (0..n_obs).partition(|&j| {
        !(matches!(bloq_method, BloqMethod::M3) && subject.cens.get(j).copied().unwrap_or(0) != 0)
    });

    let n_quant = quant_idx.len();
    let n_eta = eta_hat.len();
    let h_quant = DMatrix::from_fn(n_quant, n_eta, |r, c| h_matrix[(quant_idx[r], c)]);
    let ipreds_quant: Vec<f64> = quant_idx.iter().map(|&j| ipreds[j]).collect();

    // R diagonal at η̂/ipred (interaction); inflate with EKF variance for SDE.
    let mut r_diag_quant = compute_r_diag(error_model, &ipreds_quant, sigma_values);
    for (qi, &orig_j) in quant_idx.iter().enumerate() {
        r_diag_quant[qi] += p_obs.get(orig_j).copied().unwrap_or(0.0);
    }

    // R̃ over quantified rows: H_q · Ω · H_qᵀ + diag(R_q(η̂))
    let r_tilde = compute_r_tilde(&h_quant, &omega.matrix, &r_diag_quant);

    let (quad_form, log_det) = if n_quant > 0 {
        let chol = match r_tilde.clone().cholesky() {
            Some(c) => c,
            None => return 1e20,
        };
        let resid_quant: DVector<f64> = DVector::from_iterator(
            n_quant,
            quant_idx.iter().map(|&j| subject.observations[j] - f0[j]),
        );
        let solved = chol.solve(&resid_quant);
        let quad = resid_quant.dot(&solved);
        let ld = chol_log_det(&chol.l());
        (quad, ld)
    } else {
        (0.0, 0.0)
    };

    // BLOQ contributions: -2·log Φ((lloq - f)/√V) at η̂ (ipred-based variance).
    let mut bloq_term = 0.0;
    for &j in &bloq_idx {
        let lloq = subject.observations[j];
        let f = ipreds[j];
        let v = residual_variance(error_model, f, sigma_values);
        let z = (lloq - f) / v.sqrt();
        bloq_term += -2.0 * log_normal_cdf(z);
    }

    0.5 * (quad_form + log_det + bloq_term)
}

/// R_tilde = H * Omega * H' + diag(r_diag)
fn compute_r_tilde(h: &DMatrix<f64>, omega: &DMatrix<f64>, r_diag: &[f64]) -> DMatrix<f64> {
    let n_obs = h.nrows();
    let h_omega = h * omega;
    let mut r_tilde = &h_omega * h.transpose();
    for j in 0..n_obs {
        r_tilde[(j, j)] += r_diag[j];
    }
    r_tilde
}

/// log-determinant from Cholesky factor L: 2 * sum(log(L_ii))
fn chol_log_det(l: &DMatrix<f64>) -> f64 {
    let n = l.nrows();
    let mut ld = 0.0;
    for i in 0..n {
        let lii = l[(i, i)];
        if lii > 0.0 {
            ld += lii.ln();
        } else {
            return 1e20;
        }
    }
    2.0 * ld
}

/// IOV-aware FOCE per-subject NLL.
///
/// Computes per-occasion predictions using combined `[bsv_eta, kappa_k]`,
/// linearises with the BSV-only H-matrix, and adds explicit kappa priors so
/// the outer optimiser receives a gradient w.r.t. `omega_iov`.
///
/// `kappas[k]` is the EBE kappa vector for occasion k (same order as
/// `split_obs_by_occasion`).  When `kappas` is empty, falls through to the
/// non-IOV path (no overhead for non-IOV subjects or models).
pub fn foce_subject_nll_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega_bsv: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
    kappas: &[DVector<f64>],
    omega_iov: &OmegaMatrix,
) -> f64 {
    if kappas.is_empty() {
        return foce_subject_nll(
            model,
            subject,
            theta,
            eta_hat,
            h_matrix,
            omega_bsv,
            sigma_values,
            interaction,
        );
    }

    // Build per-occasion ipreds: obs j in occasion k uses combined=[bsv_eta, kappa_k].
    let occ_groups = split_obs_by_occasion(subject);
    let n_obs = subject.obs_times.len();
    let mut ipreds = vec![0.0_f64; n_obs];
    for (k, (_occ_id, obs_indices)) in occ_groups.iter().enumerate() {
        let kap: &[f64] = if k < kappas.len() {
            kappas[k].as_slice()
        } else {
            &[]
        };
        let combined: Vec<f64> = eta_hat.iter().copied().chain(kap.iter().copied()).collect();
        let all_preds = model_predictions(model, subject, theta, &combined);
        for &j in obs_indices {
            ipreds[j] = all_preds[j];
        }
    }

    let m3_active = matches!(model.bloq_method, BloqMethod::M3) && subject.has_bloq();
    let p_obs_iov = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta_hat.as_slice(), sigma_values)
    } else {
        Vec::new()
    };
    let foce_term = if interaction || m3_active {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega_bsv,
            sigma_values,
            model.error_model,
            model.bloq_method,
            &p_obs_iov,
        )
    } else {
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega_bsv,
            sigma_values,
            model.error_model,
            model.bloq_method,
            &p_obs_iov,
        )
    };

    // Kappa prior: 0.5 * [sum_k kappa_k' Omega_iov^{-1} kappa_k + K * log|Omega_iov|]
    let iov_inv = match omega_iov.matrix.clone().cholesky() {
        Some(chol) => chol.inverse(),
        None => return 1e20,
    };
    let log_det_iov = omega_log_det(omega_iov);
    let mut kappa_quad = 0.0;
    for kap in kappas {
        kappa_quad += kap.dot(&(&iov_inv * kap));
    }
    let k_occ = kappas.len() as f64;

    foce_term + 0.5 * (kappa_quad + k_occ * log_det_iov)
}

/// Population FOCE objective with IOV: sum over all subjects using
/// `foce_subject_nll_iov`.  `kappas_per_subject[i]` holds the per-occasion
/// kappa EBEs for subject i (empty slice = no IOV for that subject).
pub fn foce_population_nll_iov(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas_per_subject: &[Vec<DVector<f64>>],
    omega_bsv: &OmegaMatrix,
    omega_iov: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let kappas = if i < kappas_per_subject.len() {
                kappas_per_subject[i].as_slice()
            } else {
                &[]
            };
            foce_subject_nll_iov(
                model,
                subject,
                theta,
                &eta_hats[i],
                &h_matrices[i],
                omega_bsv,
                sigma_values,
                interaction,
                kappas,
                omega_iov,
            )
        })
        .sum::<f64>()
}

/// Population FOCE objective: sum over all subjects
pub fn foce_population_nll(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            foce_subject_nll(
                model,
                subject,
                theta,
                &eta_hats[i],
                &h_matrices[i],
                omega,
                sigma_values,
                interaction,
            )
        })
        .sum::<f64>()
}

/// Compute CWRES (Conditional Weighted Residuals) for a subject.
/// BLOQ observations get `NaN` since a weighted Gaussian residual is undefined
/// when the observed value is censored.
pub fn compute_cwres(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
) -> Vec<f64> {
    let n_obs = subject.observations.len();

    // f0 = ipred - H * eta_hat
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // R_tilde
    let r_diag = compute_r_diag(error_model, &f0, sigma_values);
    let r_tilde = compute_r_tilde(h_matrix, &omega.matrix, &r_diag);

    // CWRES_j = (y_j - f0_j) / sqrt(R_tilde_jj), or NaN if censored.
    (0..n_obs)
        .map(|j| {
            if subject.cens.get(j).copied().unwrap_or(0) != 0 {
                f64::NAN
            } else {
                let resid = subject.observations[j] - f0[j];
                let var = r_tilde[(j, j)].max(1e-12);
                resid / var.sqrt()
            }
        })
        .collect()
}

/// Group observation indices by occasion (preserving first-seen order of occasions).
/// Returns `Vec<(occ_id, Vec<obs_index>)>` sorted by first appearance of the occasion.
pub fn split_obs_by_occasion(subject: &Subject) -> Vec<(u32, Vec<usize>)> {
    let mut occ_order: Vec<u32> = Vec::new();
    let mut occ_map: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
    for (j, &occ) in subject.occasions.iter().enumerate() {
        if !occ_map.contains_key(&occ) {
            occ_order.push(occ);
            occ_map.insert(occ, Vec::new());
        }
        occ_map.get_mut(&occ).unwrap().push(j);
    }
    occ_order
        .into_iter()
        .map(|occ| (occ, occ_map.remove(&occ).unwrap()))
        .collect()
}

/// Build a block-diagonal omega from BSV omega and K copies of IOV omega.
/// Used for the extended H-matrix in the FOCE outer loop with IOV.
pub fn build_block_diag_omega(
    omega_bsv: &DMatrix<f64>,
    omega_iov: &DMatrix<f64>,
    n_occasions: usize,
) -> DMatrix<f64> {
    let n_bsv = omega_bsv.nrows();
    let n_iov = omega_iov.nrows();
    let n_total = n_bsv + n_occasions * n_iov;
    let mut m = DMatrix::zeros(n_total, n_total);
    // BSV block
    for i in 0..n_bsv {
        for j in 0..n_bsv {
            m[(i, j)] = omega_bsv[(i, j)];
        }
    }
    // K copies of IOV block
    for k in 0..n_occasions {
        let offset = n_bsv + k * n_iov;
        for i in 0..n_iov {
            for j in 0..n_iov {
                m[(offset + i, offset + j)] = omega_iov[(i, j)];
            }
        }
    }
    m
}

/// IOV-aware individual NLL: uses per-occasion kappas.
///
/// `kappas[k]` is the kappa vector for the k-th unique occasion (in the order
/// returned by `split_obs_by_occasion`).  When `kappas` is empty, falls back
/// to the standard (no-IOV) `individual_nll` path.
///
/// The PK parameters for occasion k are computed from:
///   `combined_eta_k = [eta[0..n_eta], kappas[k][0..n_kappa]]`
/// Predictions for occasion-k observations use those PK params with the full
/// subject dose history.
///
/// **Option A simplification — cross-occasion dose carryover.**
/// Each occasion's predictions are computed with that occasion's pk_params
/// against the *entire* dose history of the subject; only the obs rows
/// belonging to that occasion are then scored. So a dose given in occasion
/// `j` contributes to an occasion-`k` observation (`k > j`) using
/// occasion-`k`'s CL/V/etc., not occasion-`j`'s. NONMEM's strict per-dose
/// occasion accounting (each dose's contribution computed with its own
/// occasion's parameters across the intervals it dominates) is not modeled
/// here; for typical IOV designs (sparse PK with non-overlapping occasion
/// windows) the difference is small, but for densely sampled designs with
/// significant cross-occasion carryover the bias can matter. The
/// FD Jacobian in `compute_jacobian_fd_iov` shares this convention so
/// gradients and NLL values are internally consistent.
pub fn individual_nll_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    omega: &OmegaMatrix,
    omega_iov: Option<&OmegaMatrix>,
    sigma_values: &[f64],
) -> f64 {
    if kappas.is_empty() {
        return individual_nll(model, subject, theta, eta, omega, sigma_values);
    }

    // BSV eta prior
    let omega_inv = match omega.matrix.clone().cholesky() {
        Some(chol) => chol.inverse(),
        None => return 1e20,
    };
    let log_det_omega = omega_log_det(omega);
    let eta_vec = DVector::from_column_slice(eta);
    let eta_prior = eta_vec.dot(&(&omega_inv * &eta_vec));

    // Kappa priors and IOV log-det
    let (iov_inv, log_det_iov) = if let Some(iov) = omega_iov {
        let inv = match iov.matrix.clone().cholesky() {
            Some(chol) => chol.inverse(),
            None => return 1e20,
        };
        (inv, omega_log_det(iov))
    } else {
        (DMatrix::identity(1, 1), 0.0) // unreachable when kappas non-empty
    };

    let mut kappa_prior = 0.0;
    for kap in kappas {
        let kap_vec = DVector::from_column_slice(kap);
        kappa_prior += kap_vec.dot(&(&iov_inv * &kap_vec));
    }
    let k_occasions = kappas.len();

    // Data NLL — per-occasion predictions
    let occ_groups = split_obs_by_occasion(subject);
    let mut data_ll = 0.0;

    for (k, (_occ_id, obs_indices)) in occ_groups.iter().enumerate() {
        if k >= kappas.len() {
            break; // guard against mismatch
        }
        // Build combined eta for this occasion
        let combined: Vec<f64> = eta.iter().chain(kappas[k].iter()).copied().collect();
        let all_preds = model_predictions(model, subject, theta, &combined);

        for &j in obs_indices {
            let y = subject.observations[j];
            let f_pred = all_preds[j];
            let v = residual_variance(model.error_model, f_pred, sigma_values);
            if is_m3_bloq(model, subject, j) {
                let z = (y - f_pred) / v.sqrt();
                data_ll += -2.0 * log_normal_cdf(z);
            } else {
                let resid = y - f_pred;
                data_ll += resid * resid / v + v.ln();
            }
        }
    }

    0.5 * (eta_prior + log_det_omega + kappa_prior + (k_occasions as f64) * log_det_iov + data_ll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BloqMethod, DoseEvent, ErrorModel, GradientMethod, PkModel, PkParams};
    use std::collections::HashMap;

    fn make_simple_subject() -> Subject {
        Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            observations: vec![50.0, 40.0, 30.0, 45.0, 35.0, 25.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: Vec::new(),
        }
    }

    fn make_omega(var: f64) -> OmegaMatrix {
        OmegaMatrix::from_diagonal(&[var], vec!["ETA_CL".into()])
    }

    fn make_model() -> CompiledModel {
        CompiledModel {
            name: "test".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp(); // CL uses combined eta[0]
                p.values[1] = theta[1]; // V
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            default_params: crate::types::ModelParameters {
                theta: vec![5.0, 50.0],
                theta_names: vec!["TVCL".into(), "TVV".into()],
                theta_lower: vec![0.01, 1.0],
                theta_upper: vec![100.0, 500.0],
                theta_fixed: vec![false; 2],
                omega: make_omega(0.09),
                omega_fixed: vec![false],
                sigma: crate::types::SigmaVector {
                    values: vec![0.05],
                    names: vec!["PROP_ERR".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::default(),
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            n_kappa: 0,
            kappa_names: Vec::new(),
            kappa_mu_refs: HashMap::new(),
            indiv_param_names: vec!["CL".into(), "V".into()],
        }
    }

    #[test]
    fn test_split_obs_by_occasion_two_occ() {
        let subj = make_simple_subject();
        let groups = split_obs_by_occasion(&subj);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, 1);
        assert_eq!(groups[0].1, vec![0, 1, 2]);
        assert_eq!(groups[1].0, 2);
        assert_eq!(groups[1].1, vec![3, 4, 5]);
    }

    #[test]
    fn test_split_obs_by_occasion_empty() {
        let mut subj = make_simple_subject();
        subj.occasions = Vec::new();
        subj.obs_times = Vec::new();
        subj.observations = Vec::new();
        subj.obs_cmts = Vec::new();
        subj.cens = Vec::new();
        let groups = split_obs_by_occasion(&subj);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_individual_nll_iov_no_kappa_same_as_base() {
        let model = make_model();
        let subj = make_simple_subject();
        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let omega = make_omega(0.09);
        let sigma = vec![0.05];

        let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        let iov = individual_nll_iov(&model, &subj, &theta, &eta, &[], &omega, None, &sigma);
        approx::assert_relative_eq!(base, iov, epsilon = 1e-10);
    }

    #[test]
    fn test_individual_nll_iov_with_kappa_adds_prior() {
        let model = make_model();
        let subj = make_simple_subject();
        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let omega = make_omega(0.09);
        let omega_iov = make_omega(0.01);
        let sigma = vec![0.05];

        let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        // Non-zero kappas add a kappa prior ≥ 0, so IOV NLL ≥ base NLL.
        let kappas = vec![vec![0.1], vec![-0.1]];
        let iov = individual_nll_iov(
            &model,
            &subj,
            &theta,
            &eta,
            &kappas,
            &omega,
            Some(&omega_iov),
            &sigma,
        );
        // Kappa prior is positive → IOV NLL should differ from base
        assert!(
            (iov - base).abs() > 1e-6,
            "IOV NLL={}, base NLL={}",
            iov,
            base
        );
    }

    #[test]
    fn test_build_block_diag_omega_structure() {
        let bsv = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.09, 0.04]));
        let iov = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.01]));
        let combined = build_block_diag_omega(&bsv, &iov, 2);
        // 2 BSV + 2*1 IOV = 4x4
        assert_eq!(combined.nrows(), 4);
        assert_eq!(combined.ncols(), 4);
        assert_eq!(combined[(0, 0)], 0.09);
        assert_eq!(combined[(1, 1)], 0.04);
        assert_eq!(combined[(2, 2)], 0.01); // occ 1 kappa
        assert_eq!(combined[(3, 3)], 0.01); // occ 2 kappa
        assert_eq!(combined[(0, 2)], 0.0); // off-block must be zero
    }

    /// Regression: FOCEI must produce the same Sheiner–Beal linearised
    /// marginal as FOCE, only differing in *where R is evaluated*.
    /// Specifically, with an additive-only error model R doesn't depend
    /// on f, so R(f0) == R(η̂) and FOCEI must equal FOCE exactly. This
    /// catches the bug where FOCEI used a Laplace decomposition that
    /// drifted from FOCE by a few OFV units per nonlinear subject.
    #[test]
    fn test_focei_matches_foce_when_r_is_eta_independent() {
        let subj = make_simple_subject();
        let mut model = make_model();
        // Switch to additive error so R is constant w.r.t. eta.
        model.error_model = ErrorModel::Additive;

        let theta = vec![5.0, 50.0];
        let eta_hat = nalgebra::DVector::from_vec(vec![0.05]);
        let omega = make_omega(0.09);
        let sigma = vec![1.0];

        // ipreds at eta_hat
        let ipreds = pk::compute_predictions_with_tv(&model, &subj, &theta, eta_hat.as_slice());

        // Build a finite-difference H matrix d(ipred)/d(eta) at eta_hat
        // — same approach used inside find_ebe.
        let n_obs = subj.obs_times.len();
        let eps = 1e-6;
        let mut h = DMatrix::zeros(n_obs, 1);
        let h_step = eps * (1.0 + eta_hat[0].abs());
        let eta_plus = vec![eta_hat[0] + h_step];
        let eta_minus = vec![eta_hat[0] - h_step];
        let preds_plus = pk::compute_predictions_with_tv(&model, &subj, &theta, &eta_plus);
        let preds_minus = pk::compute_predictions_with_tv(&model, &subj, &theta, &eta_minus);
        for i in 0..n_obs {
            h[(i, 0)] = (preds_plus[i] - preds_minus[i]) / (2.0 * h_step);
        }

        let foce = foce_subject_nll_standard(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            ErrorModel::Additive,
            BloqMethod::Drop,
            &[],
        );
        let focei = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            ErrorModel::Additive,
            BloqMethod::Drop,
            &[],
        );

        // For an η-independent residual variance, FOCEI is mathematically
        // identical to FOCE. Pre-fix this test would fail by several OFV
        // units because the Laplace-style decomposition added an extra
        // η̂'Ω⁻¹η̂ term and used (y - ipred) instead of (y - f₀).
        assert!(
            (focei - foce).abs() < 1e-9,
            "FOCEI ({}) and FOCE ({}) must agree when R is eta-independent (additive error)",
            focei,
            foce,
        );
    }
}
