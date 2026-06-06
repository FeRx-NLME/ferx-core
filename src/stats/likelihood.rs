use crate::pk;
use crate::stats::residual_error::compute_r_diag;
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
    // Ω⁻¹ and log|Ω| are pre-computed in `OmegaMatrix::from_matrix_*`.
    // Hot-path users (FOCE inner BFGS, SAEM MH) call this 100s–1000s of
    // times per subject per outer iter — recomputing Cholesky+inverse
    // here used to dominate small-omega problems.
    if !omega.log_det.is_finite() {
        return 1e20;
    }
    let omega_inv = &omega.inv;
    let log_det_omega = omega.log_det;

    // Eta prior: eta' * Omega_inv * eta
    let eta_vec = DVector::from_column_slice(eta);
    let eta_prior = eta_vec.dot(&(omega_inv * &eta_vec));

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
        let v_resid = model.residual_variance_at(subject.obs_cmts[j], f_pred, sigma_values);
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

    let nll = 0.5 * (eta_prior + log_det_omega + data_ll);
    // Guard a non-finite prediction the same way we guard a non-finite Ω above:
    // an ODE integration can blow up to NaN/inf when the EBE search pushes eta
    // into an extreme region, which would otherwise poison the inner optimizer
    // (e.g. the Nelder-Mead simplex sort). Return the large finite sentinel so
    // the bad point sorts as worst and gets reflected away. See issue #97.
    if nll.is_finite() {
        nll
    } else {
        1e20
    }
}

/// Observation-only NLL for a single subject with ETAs held fixed.
///
/// Returns the data term `−log p(y_i | η, θ, σ)` (no prior, no |Ω| term) — the
/// piece that participates in the SAEM M-step gradient and the IS-LL numerator.
///
/// Under M3, CENS=1 rows contribute `−log Φ((LLOQ − f)/√V)` instead of the
/// Gaussian residual term.
pub(crate) fn obs_nll_subject_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    pk_scratch: &mut pk::EventPkParams,
) -> f64 {
    let m3 = matches!(model.bloq_method, BloqMethod::M3);
    let preds = pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);
    let mut nll = 0.0;
    for (j, (&y, &f)) in subject.observations.iter().zip(preds.iter()).enumerate() {
        let f = f.max(1e-12);
        let v = model
            .residual_variance_at(subject.obs_cmts[j], f, sigma_values)
            .max(1e-12);
        if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
            let z = (y - f) / v.sqrt();
            nll += -crate::stats::special::log_normal_cdf(z);
        } else {
            nll += 0.5 * (v.ln() + (y - f).powi(2) / v);
        }
    }
    nll
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
    // EKF process-noise variance uses a single error model. This is sound: a
    // per-CMT (multi-endpoint) error model needs a Form C `y[CMT=N]` readout to
    // observe multiple compartments, and the parser rejects Form C on SDE
    // models — so an SDE model is always `ErrorSpec::Single` and the
    // representative `model.error_model` is exact here.
    debug_assert!(
        matches!(model.error_spec, ErrorSpec::Single(_)),
        "EKF path reached with a non-Single error spec (per-CMT + SDE should be unreachable)"
    );
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
            &model.error_spec,
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
            &model.error_spec,
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
    error_spec: &ErrorSpec,
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
    let mut r_diag = compute_r_diag(error_spec, &f0, &subject.obs_cmts, sigma_values);
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

/// FOCEI INTER per-subject −2·log marginal — Almquist 2015 Laplace form.
///
/// Per-subject objective (without the `N·log(2π)` data-side constant that
/// NONMEM and nlmixr2 also drop):
///
/// ```text
///   data_ll(η̂) + η̂'·Ω⁻¹·η̂ + log|Ω| + log|H̃|
/// ```
///
/// where
///   `data_ll(η̂) = Σⱼ [(yⱼ − fⱼ)² / Rⱼ + log Rⱼ]`     (R evaluated at η̂)
///   `H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + Ω⁻¹`            (Almquist 2015 eq. 15)
///   `a_{j,k} = ∂fⱼ/∂η_k`                              (rows of `h_matrix`)
///   `c̃_{j,k} = (∂Rⱼ/∂η_k) / Rⱼ = (∂Rⱼ/∂fⱼ)·a_{j,k} / Rⱼ`
///                                                     (chain rule;
///                                                      `∂R/∂f` from
///                                                      `ErrorSpec::dvar_df`)
///
/// The `½·c̃'·c̃` piece is the **INTER correction**: it captures the
/// η-dependence of the residual variance in the conditional Hessian. It
/// vanishes for additive (η-independent R) error, in which case `H̃`
/// reduces to the FOCE-non-interaction `a'·diag(1/R)·a + Ω⁻¹`.
///
/// This matches NONMEM's `METHOD=1 INTER` and nlmixr2's `est="focei"` —
/// independently verified on the jasmine peds vanco dataset: at NONMEM's
/// converged params, NM reports OFV 66 539, nlmixr2 66 727, and ferx's
/// Almquist Laplace agrees to within FD-vs-analytical-sensitivity noise.
/// The Python reconstruction of NM's per-subject OBJ from its own (η̂, ETC,
/// IPRED) using this exact formula reproduces NM's reported OFV to within
/// 0.013 out of 66 539 — confirming Almquist 2015 first-order is what
/// NONMEM computes.
///
/// The previous implementation used the Sheiner–Beal linearised marginal
/// `(y − f₀)' R̃⁻¹ (y − f₀) + log|R̃|` with `R̃ = HΩH' + R(η̂)`. For
/// nonlinear PK with INTER, that form diverges from the Laplace value at
/// large |η|, and the outer optimiser can exploit the gap to drive `σ_add`
/// small (the negative-EPS-shrinkage symptom on jasmine peds vanco). See
/// `[[focei-laplace-not-sheiner-beal]]` memory.
///
/// With `bloq_method == M3`, BLOQ observations are dropped from the
/// Gaussian residual sum and the H̃ accumulation, and instead contribute
/// `−2·log Φ((LLOQ − f)/√V)` evaluated at η̂.
pub fn foce_subject_nll_interaction(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    bloq_method: BloqMethod,
    p_obs: &[f64],
) -> f64 {
    let n_obs = subject.observations.len();
    let n_eta = eta_hat.len();

    // Partition observation indices into quantified vs BLOQ (M3 only).
    let (quant_idx, bloq_idx): (Vec<usize>, Vec<usize>) = (0..n_obs).partition(|&j| {
        !(matches!(bloq_method, BloqMethod::M3) && subject.cens.get(j).copied().unwrap_or(0) != 0)
    });

    // Accumulate data_ll at η̂ and the conditional Hessian pieces over the
    // quantified rows. For SDE the EKF process-noise variance `p_obs` inflates
    // R additively, treated as η-independent here (its η-dependence enters
    // via the same a path; EKF-vs-FOCEI cross terms are dropped under
    // Almquist's first-order convention).
    let mut data_ll = 0.0_f64;
    let mut hrh = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut ctc = DMatrix::<f64>::zeros(n_eta, n_eta);
    for &j in &quant_idx {
        let f = ipreds[j];
        let v_resid = error_spec.variance_at(subject.obs_cmts[j], f, sigma_values);
        let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
        if !(v.is_finite() && v > 0.0) {
            return 1e20;
        }
        let r = subject.observations[j] - f;
        data_ll += r * r / v + v.ln();

        // a_j = row j of H (∂f_j/∂η). c̃_j = (∂R_j/∂f_j) · a_j / R_j.
        let aj = h_matrix.row(j);
        let dvar_df = error_spec.dvar_df(subject.obs_cmts[j], f, sigma_values);
        let c_scale = dvar_df / v; // c̃_j = c_scale · a_j

        // hrh += a_j' · a_j / v;  ctc += c̃_j' · c̃_j = c_scale² · a_j' · a_j.
        let inv_v = 1.0 / v;
        let cs2 = c_scale * c_scale;
        for a in 0..n_eta {
            let aa = aj[a];
            for b in 0..n_eta {
                let ab = aj[b];
                let outer = aa * ab;
                hrh[(a, b)] += outer * inv_v;
                ctc[(a, b)] += outer * cs2;
            }
        }
    }

    // η̂'Ω⁻¹η̂  +  log|Ω|  (both cached on OmegaMatrix).
    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat));
    let log_det_omega = omega.log_det;

    // H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + Ω⁻¹.  log|H̃| via Cholesky; the 1e20
    // sentinel handles extreme-η points where H̃ is not PD — the inner-loop
    // optimiser falls back via Nelder–Mead.
    let htilde = hrh + 0.5 * ctc + &omega.inv;
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => return 1e20,
    };

    // BLOQ contributions: −2·log Φ((lloq − f)/√V) at η̂ (ipred-based variance).
    let mut bloq_term = 0.0;
    for &j in &bloq_idx {
        let lloq = subject.observations[j];
        let f = ipreds[j];
        let v = error_spec.variance_at(subject.obs_cmts[j], f, sigma_values);
        let z = (lloq - f) / v.sqrt();
        bloq_term += -2.0 * log_normal_cdf(z);
    }

    0.5 * (data_ll + eta_prior + log_det_omega + log_det_htilde + bloq_term)
}

/// R_tilde = H * Omega * H' + diag(r_diag)
pub(crate) fn compute_r_tilde(
    h: &DMatrix<f64>,
    omega: &DMatrix<f64>,
    r_diag: &[f64],
) -> DMatrix<f64> {
    let n_obs = h.nrows();
    let h_omega = h * omega;
    let mut r_tilde = &h_omega * h.transpose();
    for j in 0..n_obs {
        r_tilde[(j, j)] += r_diag[j];
    }
    r_tilde
}

/// log-determinant from Cholesky factor L: 2 * sum(log(L_ii))
pub(crate) fn chol_log_det(l: &DMatrix<f64>) -> f64 {
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

/// IOV-aware FOCE per-subject NLL — a *proper* linearised marginal over the
/// full random-effect vector `b = [η, κ₁, …, κ_K]`.
///
/// The per-occasion κ draws are integrated out by the same Sheiner–Beal
/// marginal that handles the BSV η: we assemble the augmented sensitivity
/// matrix `H_full = [∂f/∂η │ ∂f/∂κ₁ │ … │ ∂f/∂κ_K]` and the block-diagonal
/// prior covariance `Σ_b = blkdiag(Ω_bsv, Ω_iov, …, Ω_iov)` (K copies), then
/// evaluate the ordinary FOCE/FOCEI form `0.5·[(y−f₀)ᵀ R̃⁻¹ (y−f₀) + log|R̃|]`
/// with `R̃ = H_full Σ_b H_fullᵀ + R`.
///
/// Because `∂f/∂κ_k` is non-zero only on occasion-k's observation rows (κ_k
/// enters only that occasion's predictions, under the cross-occasion
/// dose-carryover convention of `individual_nll_iov`), the κ columns are
/// block-structured and the κ blocks of `Σ_b` couple only same-occasion rows
/// — independent occasions stay independent in `R̃`.
///
/// This replaces the earlier shortcut (BSV-only linearisation plus an explicit
/// `0.5·Σ_k[κᵀΩ_iov⁻¹κ + log|Ω_iov|]` MAP penalty). That penalty omitted the
/// κ-block Laplace determinant `log|H_κᵀR⁻¹H_κ + Ω_iov⁻¹|`; in a correct
/// marginal `log|Ω| + log|J|` combine into the bounded `log|R̃/R|`, so dropping
/// `log|J|` left a bare `+0.5·K·log|Ω_iov|` that → −∞ as Ω_iov → 0, leaving
/// `omega_iov` unidentified and the FOCE OFV not comparable to NONMEM / SAEM.
/// See issue #101. With the augmented form, no separate κ prior is added (it
/// is already folded into `R̃`), and the K=0 case reduces exactly to
/// [`foce_subject_nll`].
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

    let occ_groups = split_obs_by_occasion(subject);
    let n_obs = subject.obs_times.len();
    let n_eta = eta_hat.len();
    let n_iov = omega_iov.matrix.nrows();
    // Defensive: the EBE pipeline always yields exactly one κ vector per
    // occasion group, each of width n_iov. A mismatch would silently leave the
    // unmatched occasions' ipreds (and H columns) at 0.0 and score the
    // augmented marginal against wrong predictions, so bail with the large
    // finite sentinel — mirroring the guards in `individual_nll_iov`.
    if kappas.len() != occ_groups.len() || kappas.iter().any(|k| k.len() != n_iov) {
        return 1e20;
    }
    let k_occ = occ_groups.len();
    let n_b = n_eta + k_occ * n_iov;

    // ipreds at the joint EBE via the continuous, per-occasion-aware prediction
    // (proper cross-occasion carryover; issue #104), plus the augmented
    // H-matrix. BSV columns come from the passed-in H (FD of the same prediction
    // w.r.t. η, in `compute_jacobian_fd_iov`); the κ columns are FD here.
    // Because κ_k changes occasion-k's clearance, it affects occasion-k's
    // observations AND the carryover into later occasions — so a κ column is
    // dense across rows, exactly what FD of the continuous prediction captures
    // (the old Option-A version wrote κ_k only on occasion-k's rows).
    let kappa_slices: Vec<Vec<f64>> = kappas.iter().map(|k| k.as_slice().to_vec()).collect();
    let ipreds = pk::predict_iov(model, subject, theta, eta_hat.as_slice(), &kappa_slices);

    let mut h_full = DMatrix::zeros(n_obs, n_b);
    for j in 0..n_obs {
        for c in 0..n_eta {
            h_full[(j, c)] = h_matrix[(j, c)];
        }
    }
    // Reused κ buffer: perturb one element in place and restore it, rather than
    // cloning all occasions' κ twice per FD step.
    let mut kpert = kappa_slices.clone();
    const EPS: f64 = 1e-6;
    for k in 0..k_occ {
        let col_base = n_eta + k * n_iov;
        for ki in 0..n_iov {
            let orig = kpert[k][ki];
            let step = EPS * (1.0 + orig.abs());
            kpert[k][ki] = orig + step;
            let preds_plus = pk::predict_iov(model, subject, theta, eta_hat.as_slice(), &kpert);
            kpert[k][ki] = orig - step;
            let preds_minus = pk::predict_iov(model, subject, theta, eta_hat.as_slice(), &kpert);
            kpert[k][ki] = orig;
            let inv_2step = 1.0 / (2.0 * step);
            for j in 0..n_obs {
                h_full[(j, col_base + ki)] = (preds_plus[j] - preds_minus[j]) * inv_2step;
            }
        }
    }

    // Joint EBE vector b̂ = [η̂, κ̂₁, …, κ̂_K].
    let mut b_hat = DVector::zeros(n_b);
    for i in 0..n_eta {
        b_hat[i] = eta_hat[i];
    }
    for (k, kap) in kappas.iter().enumerate() {
        for ki in 0..n_iov {
            b_hat[n_eta + k * n_iov + ki] = kap[ki];
        }
    }

    // Block-diagonal prior covariance Σ_b = blkdiag(Ω_bsv, Ω_iov × K).
    // `from_matrix` regularises if a sub-block is not PD, matching the
    // robustness of the non-IOV OmegaMatrix path; the standard/interaction
    // FOCE routines below read only `Σ_b.matrix`.
    let sigma_b_mat = build_block_diag_omega(&omega_bsv.matrix, &omega_iov.matrix, k_occ);
    let sigma_b = OmegaMatrix::from_matrix(sigma_b_mat, Vec::new(), false);

    // The augmented system is now an ordinary FOCE/FOCEI marginal: κ is
    // integrated out through R̃ exactly like η, so no separate κ prior is
    // added (doing so would double-count the random-effect penalty).
    let m3_active = matches!(model.bloq_method, BloqMethod::M3) && subject.has_bloq();
    let p_obs_iov = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta_hat.as_slice(), sigma_values)
    } else {
        Vec::new()
    };
    if interaction || m3_active {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            &b_hat,
            &h_full,
            &sigma_b,
            sigma_values,
            &model.error_spec,
            model.bloq_method,
            &p_obs_iov,
        )
    } else {
        foce_subject_nll_standard(
            subject,
            &ipreds,
            &b_hat,
            &h_full,
            &sigma_b,
            sigma_values,
            &model.error_spec,
            model.bloq_method,
            &p_obs_iov,
        )
    }
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
    error_spec: &ErrorSpec,
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
    let r_diag = compute_r_diag(error_spec, &f0, &subject.obs_cmts, sigma_values);
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
/// **Cross-occasion dose carryover (issue #104).** Predictions are computed by
/// [`pk::predict_iov`], which builds per-event PK parameters carrying each
/// event's occasion kappa and propagates the compartment amounts continuously
/// across occasion boundaries (via the event-driven solver). A dose given in an
/// earlier occasion therefore decays through a later occasion with the *later*
/// occasion's clearance — matching NONMEM's integration model. This replaced
/// the earlier "Option A" superposition, which scored each occasion against the
/// whole dose history with a single clearance and biased the likelihood on
/// no-washout designs. The FD Jacobian (`compute_jacobian_fd_iov`) and the
/// augmented marginal (`foce_subject_nll_iov`) use the same prediction, so NLL
/// and gradients stay consistent.
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

    // Data NLL — single continuous prediction with per-event occasion kappa
    // (proper cross-occasion carryover; issue #104).
    let preds = pk::predict_iov(model, subject, theta, eta, kappas);
    let mut data_ll = 0.0;
    for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
        let v = model.residual_variance_at(subject.obs_cmts[j], f_pred, sigma_values);
        if is_m3_bloq(model, subject, j) {
            let z = (y - f_pred) / v.sqrt();
            data_ll += -2.0 * log_normal_cdf(z);
        } else {
            let resid = y - f_pred;
            data_ll += resid * resid / v + v.ln();
        }
    }

    0.5 * (eta_prior + log_det_omega + kappa_prior + (k_occasions as f64) * log_det_iov + data_ll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BloqMethod, DoseEvent, ErrorModel, ErrorSpec, GradientMethod, PkModel, PkParams,
    };
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
            reset_times: Vec::new(),
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
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
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
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
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
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
        }
    }

    #[test]
    fn test_ltbs_individual_nll_matches_additive_on_log_scale() {
        // Under LTBS the inner-loop NLL must score the (already-log-scale)
        // observations against log(prediction) with additive variance σ². This
        // checks the prediction sink's log-wrap flows through `individual_nll`.
        let mut model = make_model();
        model.error_model = ErrorModel::Additive;
        model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
        model.log_transform = true;

        let theta = vec![5.0, 50.0];
        let eta = vec![0.0]; // eta_prior = 0
        let omega = make_omega(0.09);
        let sigma = vec![0.3]; // additive SD on the log scale

        // Observations on the log scale (what `fit()` produces for case 2).
        let mut subj = make_simple_subject();
        for v in &mut subj.observations {
            *v = v.ln();
        }

        // Manual reference: log(natural prediction), additive variance σ².
        let mut natural_model = make_model();
        natural_model.log_transform = false;
        let natural = pk::compute_predictions_with_tv(&natural_model, &subj, &theta, &eta);
        let var = sigma[0] * sigma[0];
        let mut data_ll = 0.0;
        for (j, &f_nat) in natural.iter().enumerate() {
            let log_f = f_nat.max(pk::LTBS_FLOOR).ln();
            let resid = subj.observations[j] - log_f;
            data_ll += resid * resid / var + var.ln();
        }
        let expected = 0.5 * (omega.log_det + data_ll);

        let got = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        approx::assert_relative_eq!(got, expected, epsilon = 1e-9);
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
    fn test_individual_nll_finite_sentinel_on_nonfinite_eta() {
        // Regression for issue #97: when the EBE search wanders into an extreme
        // region (here a non-finite eta, standing in for an ODE blow-up), the
        // NLL must return the large finite sentinel, never a non-finite value.
        // A NaN/inf leaking out poisons the inner Nelder-Mead simplex sort and
        // aborts the fit; this guard mirrors the existing non-finite Ω guard.
        //
        // Note the analytical PK path scrubs NaN via `.max()`/`.min()`
        // (`NaN.max(1e-30) == 1e-30`), so the non-finiteness here enters through
        // the eta-prior term `η'Ω⁻¹η`, which is exactly the quantity the inner
        // optimizer drives.
        let model = make_model();
        let subj = make_simple_subject();
        let omega = make_omega(0.09);
        let nll = individual_nll(
            &model,
            &subj,
            &[5.0, 50.0],
            &[f64::INFINITY],
            &omega,
            &[0.05],
        );
        assert!(nll.is_finite(), "NLL must stay finite, got {nll}");
        assert_eq!(
            nll, 1e20,
            "a non-finite NLL should map to the 1e20 sentinel"
        );
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

    /// A model whose CL depends on both the BSV eta and the per-occasion
    /// kappa (`combined[1]`), so the kappa block genuinely enters the
    /// augmented R̃. The kappa read is defensive so the BSV-only
    /// `foce_subject_nll` path (which passes a length-1 eta) doesn't panic.
    fn make_iov_kappa_model() -> CompiledModel {
        let mut model = make_model();
        model.pk_param_fn = Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
            let mut p = PkParams::default();
            let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
            p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL
            p.values[1] = theta[1]; // V
            p
        });
        model
    }

    /// Issue #101: `foce_subject_nll_iov` must be a proper augmented marginal,
    /// not a BSV FOCE term plus an additive kappa MAP penalty.
    #[test]
    fn test_foce_subject_nll_iov_is_proper_marginal() {
        let model = make_iov_kappa_model();
        let subj = make_simple_subject(); // occasions [1,1,1,2,2,2]
        let theta = vec![5.0, 50.0];
        let eta_hat = DVector::from_vec(vec![0.1]);
        let omega_bsv = make_omega(0.09);
        let sigma = vec![0.05];

        // BSV-only H via central FD of predictions w.r.t. eta[0] at kappa = 0.
        let n_obs = subj.observations.len();
        let mut h_bsv = DMatrix::zeros(n_obs, 1);
        let eps = 1e-6;
        let pp = model_predictions(&model, &subj, &theta, &[0.1 + eps]);
        let pm = model_predictions(&model, &subj, &theta, &[0.1 - eps]);
        for j in 0..n_obs {
            h_bsv[(j, 0)] = (pp[j] - pm[j]) / (2.0 * eps);
        }

        // (1) Reduction: zero kappas + Ω_iov → 0 collapses to the BSV-only
        //     marginal. The OLD code added 0.5·K·log|Ω_iov| = log(1e-12) ≈ -27.6,
        //     so this assertion fails without the proper-marginal fix.
        let base = foce_subject_nll(
            &model, &subj, &theta, &eta_hat, &h_bsv, &omega_bsv, &sigma, false,
        );
        let zero_kappas = vec![DVector::zeros(1), DVector::zeros(1)];
        let reduced = foce_subject_nll_iov(
            &model,
            &subj,
            &theta,
            &eta_hat,
            &h_bsv,
            &omega_bsv,
            &sigma,
            false,
            &zero_kappas,
            &make_omega(1e-12),
        );
        // max_relative (not epsilon): these are O(1e5), and the residual κ-block
        // contribution at Ω_iov = 1e-12 is ~1e-11 relative. The old additive
        // penalty would shift by ~27.6 absolute (≈1.5e-4 relative) and fail.
        approx::assert_relative_eq!(reduced, base, max_relative = 1e-9);

        // (2) The marginal responds to Ω_iov through R̃ (the determinant term
        //     the old penalty was missing): with non-zero kappas, two different
        //     Ω_iov give materially different, finite OFVs.
        let kappas = vec![
            DVector::from_vec(vec![0.08]),
            DVector::from_vec(vec![-0.05]),
        ];
        let nll = |iov_var: f64| {
            foce_subject_nll_iov(
                &model,
                &subj,
                &theta,
                &eta_hat,
                &h_bsv,
                &omega_bsv,
                &sigma,
                false,
                &kappas,
                &make_omega(iov_var),
            )
        };
        let small = nll(0.005);
        let large = nll(0.5);
        assert!(small.is_finite() && large.is_finite());
        assert!(
            (small - large).abs() > 1e-6,
            "Ω_iov must change the marginal OFV (small={small}, large={large})"
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

    /// Algebraic identity check: under additive (η-independent R) error, the
    /// Almquist `½·c̃'·c̃` INTER correction is identically zero, so
    /// `H̃ = a'·diag(1/R)·a + Ω⁻¹`. We hand-compute the closed-form Laplace
    /// value from the same H, R, η̂ and assert bit-for-bit agreement with
    /// `foce_subject_nll_interaction`.
    ///
    /// (Replaces the previous `test_focei_matches_foce_when_r_is_eta_independent`,
    /// which asserted FOCEI INTER == FOCE non-INTER exactly under additive
    /// error. That identity only holds for the Sheiner–Beal form; the
    /// Almquist Laplace form NONMEM/nlmixr2 use does *not* satisfy it because
    /// the two forms approximate the same true marginal differently for
    /// nonlinear models. See `[[focei-laplace-not-sheiner-beal]]`.)
    #[test]
    fn test_focei_laplace_additive_matches_handcomputed_hessian() {
        let subj = make_simple_subject();
        let mut model = make_model();
        model.error_model = ErrorModel::Additive;
        model.error_spec = ErrorSpec::Single(ErrorModel::Additive);

        let theta = vec![5.0, 50.0];
        let eta_hat = nalgebra::DVector::from_vec(vec![0.05]);
        let omega = make_omega(0.09);
        let sigma = vec![1.0];

        let ipreds = pk::compute_predictions_with_tv(&model, &subj, &theta, eta_hat.as_slice());
        let n_obs = subj.obs_times.len();
        let eps = 1e-6;
        let mut h = DMatrix::zeros(n_obs, 1);
        let h_step = eps * (1.0 + eta_hat[0].abs());
        let preds_p =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] + h_step]);
        let preds_m =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] - h_step]);
        for i in 0..n_obs {
            h[(i, 0)] = (preds_p[i] - preds_m[i]) / (2.0 * h_step);
        }

        let espec = ErrorSpec::Single(ErrorModel::Additive);
        let focei = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &espec,
            BloqMethod::Drop,
            &[],
        );

        // Hand-compute the Laplace value with c̃ ≡ 0 (additive R).
        let r = sigma[0] * sigma[0]; // ErrorModel::Additive: R = σ²
        let mut data_ll = 0.0;
        for j in 0..n_obs {
            let res = subj.observations[j] - ipreds[j];
            data_ll += res * res / r + r.ln();
        }
        let eta_prior = eta_hat.dot(&(&omega.inv * &eta_hat));
        let mut htilde_scalar = omega.inv[(0, 0)];
        for j in 0..n_obs {
            htilde_scalar += h[(j, 0)] * h[(j, 0)] / r;
        }
        let log_det_htilde = htilde_scalar.ln(); // 1×1 case
        let expected = 0.5 * (data_ll + eta_prior + omega.log_det + log_det_htilde);

        assert!(
            (focei - expected).abs() < 1e-9,
            "FOCEI Laplace ({}) must equal hand-computed value ({}) under \
             additive error; diff = {}",
            focei,
            expected,
            focei - expected,
        );
    }

    /// Confirms the Almquist `½·c̃'·c̃` INTER correction is actually wired:
    /// switching from additive (c̃ ≡ 0) to combined error (`dvar_df ≠ 0`,
    /// hence c̃ ≠ 0) must change the Laplace H̃ and therefore the per-subject
    /// OFV by a non-trivial amount, even when the proportional-component
    /// magnitude is small enough that the *data* term barely shifts.
    ///
    /// Catches a regression where the c̃'c̃ accumulator is dropped or zeroed —
    /// in that case the additive and combined Laplace values would coincide
    /// up to the tiny `(prop·f)² → R` difference in the data quadratic.
    #[test]
    fn test_focei_laplace_combined_uses_inter_correction() {
        let subj = make_simple_subject();
        let mut model = make_model();
        model.error_model = ErrorModel::Combined;
        model.error_spec = ErrorSpec::Single(ErrorModel::Combined);

        let theta = vec![5.0, 50.0];
        let eta_hat = nalgebra::DVector::from_vec(vec![0.1]);
        let omega = make_omega(0.09);
        // Two sigmas for Combined: (prop, add). The proportional component is
        // intentionally small relative to additive so the data-side R hardly
        // changes — any noticeable OFV shift is then due to the c̃'c̃ piece.
        let sigma_combined = vec![0.02, 1.0];
        let sigma_additive_only = vec![1.0]; // matches Combined R when prop ≈ 0

        let ipreds = pk::compute_predictions_with_tv(&model, &subj, &theta, eta_hat.as_slice());
        let n_obs = subj.obs_times.len();
        let eps = 1e-6;
        let mut h = DMatrix::zeros(n_obs, 1);
        let h_step = eps * (1.0 + eta_hat[0].abs());
        let preds_p =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] + h_step]);
        let preds_m =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] - h_step]);
        for i in 0..n_obs {
            h[(i, 0)] = (preds_p[i] - preds_m[i]) / (2.0 * h_step);
        }

        let espec_combined = ErrorSpec::Single(ErrorModel::Combined);
        let espec_additive = ErrorSpec::Single(ErrorModel::Additive);
        let focei_combined = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma_combined,
            &espec_combined,
            BloqMethod::Drop,
            &[],
        );
        let focei_additive = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma_additive_only,
            &espec_additive,
            BloqMethod::Drop,
            &[],
        );
        let gap = focei_combined - focei_additive;
        assert!(
            gap.abs() > 1e-3,
            "FOCEI Laplace must respond to the Almquist `½·c̃'·c̃` INTER \
             correction; combined ({}) and additive ({}) gave gap = {} — \
             too small, c̃'c̃ likely not being accumulated.",
            focei_combined,
            focei_additive,
            gap,
        );
    }
}
