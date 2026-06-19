//! HMC (Hamiltonian Monte Carlo) E-step proposals for SAEM.
//!
//! Uses standard HMC with identity mass matrix (M = I):
//!   - Momentum p ~ N(0, I)
//!   - Kinetic energy K(p) = ½ ‖p‖²
//!   - Hamiltonian H(η, p) = NLL(η) + K(p)
//!   - Velocity Störmer-Verlet (leapfrog) integrates the dynamics
//!   - Metropolis accept/reject on ΔH
//!
//! `leapfrog` has no autodiff dependency and can be unit-tested without the
//! Enzyme toolchain.  `hmc_step` is gated on `#[cfg(feature = "autodiff")]`
//! because the gradient uses `compute_nll_gradient_ad`.

#[cfg(feature = "autodiff")]
use rand::Rng;
#[cfg(feature = "autodiff")]
use rand_distr::StandardNormal;

/// Leapfrog energy-error magnitude above which an HMC transition is flagged
/// divergent (matches Stan's `Δ_max`).
#[cfg(feature = "autodiff")]
const HMC_DIVERGENCE_THRESHOLD: f64 = 1000.0;

// ---------------------------------------------------------------------------
// Leapfrog integrator (no autodiff dependency)
// ---------------------------------------------------------------------------

/// Standard velocity Störmer-Verlet (leapfrog) integrator.
///
/// Integrates the Hamiltonian H(q, p) = NLL(q) + ½‖p‖² with identity mass
/// matrix.  `nll_grad_eta` must return ∂NLL/∂η at the supplied η.
///
/// Algorithm (L+1 gradient evaluations for L full steps):
///
/// 1. `p ← p − (ε/2) · g(q)`                  [initial half-step for p]
/// 2. for _ in 0..L:
///    a. `q ← q + ε · p`                       [full position step]
///    b. `g = grad(q)`; `p ← p − ε · g`        [full momentum step]
/// 3. `p ← p + (ε/2) · g`                      [undo the last half-overshoot]
///
/// Step 3 adds back half the gradient because step 2's last iteration applied a
/// full ε·g where only ε/2·g should be applied.  After steps 2–3, the
/// momentum carries exactly the correct final half-step.
///
/// When `n_steps == 0` both q and p are returned unchanged (the initial
/// half-step and the correction cancel exactly) — the proposal equals the
/// current state, ΔH = 0, and the step is trivially accepted.
pub fn leapfrog(
    eta: &[f64],
    momentum: &[f64],
    nll_grad_eta: &dyn Fn(&[f64]) -> Vec<f64>,
    step_size: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = eta.len();
    let mut q = eta.to_vec();
    let mut p = momentum.to_vec();

    // Initial half-step for momentum.
    let mut g = nll_grad_eta(&q);
    for k in 0..n {
        p[k] -= 0.5 * step_size * g[k];
    }

    // L full steps: full position update, full momentum update.
    // The final iteration overcounts by ε/2 (corrected below).
    for _ in 0..n_steps {
        for k in 0..n {
            q[k] += step_size * p[k];
        }
        g = nll_grad_eta(&q);
        for k in 0..n {
            p[k] -= step_size * g[k];
        }
    }

    // Correct the overshoot: the last full step applied ε·g but only ε/2·g
    // is needed for the final half-step.  Add ε/2·g back.
    // `g` here is grad(q_final), computed in the last loop iteration.
    // When n_steps == 0, g = grad(q_0) and this exactly cancels the initial
    // half-step, leaving both q and p unchanged.
    for k in 0..n {
        p[k] += 0.5 * step_size * g[k];
    }

    (q, p)
}

// ---------------------------------------------------------------------------
// HMC step (requires autodiff)
// ---------------------------------------------------------------------------

/// One HMC proposal for a single SAEM subject.
///
/// Builds all AD helpers internally from the current SAEM state
/// `(model, theta, omega, sigma_values)`, matching the interface of
/// `mh_steps`.
///
/// Returns `Some((new_eta, new_nll, accepted, divergent))` when the AD gradient path is
/// available.  Returns `None` when HMC cannot be applied — caller must fall
/// back to `mh_steps` in that case.  HMC is unavailable when:
///   - the model uses an ODE (`model.ode_spec.is_some()`)
///   - the model has no analytical PK path (`model.tv_fn.is_none()`)
///   - `omega.log_det` is non-finite (degenerate variance matrix)
///   - [`crate::estimation::inner_optimizer::resolve_gradient_method`] resolves
///     the subject to `Fd` (no AD path consistent with the analytical
///     objective): SS doses,
///     an oral model with a zero-order infusion dose, eta-dependent lagtime, or
///     a TV-covariate / reset subject on a PK model the event-driven AD path
///     doesn't support.
///
/// Reset (EVID=3/4), TV-covariate, and lagtime subjects take the event-driven
/// AD path; plain subjects take the single-snapshot path — the same routing as
/// the FOCEI inner loop.
#[cfg(feature = "autodiff")]
#[allow(clippy::too_many_arguments)]
pub fn hmc_step(
    subject: &crate::types::Subject,
    eta: &[f64],
    nll_current: f64,
    model: &crate::types::CompiledModel,
    theta: &[f64],
    omega: &crate::types::OmegaMatrix,
    sigma_values: &[f64],
    step_size: f64,
    n_leapfrog: usize,
    rng: &mut impl Rng,
) -> Option<(Vec<f64>, f64, bool, bool)> {
    use crate::ad::ad_gradients::{compute_nll_gradient_ad, FlatDoseData};
    use crate::ad::event_driven_ad;
    use crate::types::BloqMethod;
    use std::cell::Cell;

    // AD requires an analytical PK path.
    if model.ode_spec.is_some() || model.tv_fn.is_none() {
        return None;
    }
    if !omega.log_det.is_finite() {
        return None;
    }

    let n_eta = eta.len();

    // Flatten Ω⁻¹ (row-major) for the AD kernel.
    let omega_inv = &omega.inv;
    let mut omega_inv_flat = Vec::with_capacity(n_eta * n_eta);
    for i in 0..n_eta {
        for j in 0..n_eta {
            omega_inv_flat.push(omega_inv[(i, j)]);
        }
    }
    let log_det_omega = omega.log_det;

    // Censoring flags for M3.
    let cens_f64: Vec<f64> = if matches!(model.bloq_method, BloqMethod::M3) {
        subject.cens.iter().map(|&c| c as f64).collect()
    } else {
        vec![0.0; subject.observations.len()]
    };

    // `last_nll` is written by the grad closure on every call so that after
    // leapfrog, `last_nll.get()` == NLL(η_proposal) without an extra AD call.
    let last_nll = Cell::new(nll_current);

    // Route by the SAME policy as the FOCEI inner loop
    // (`inner_optimizer::resolve_gradient_method`) so the two estimators stay
    // consistent: reset / TV-covariate / lagtime subjects take the event-driven
    // AD path, and subjects where AD would be inconsistent with the analytical
    // objective (SS doses, oral + zero-order infusion, eta-dependent lagtime,
    // unsupported models) resolve to `Fd` — for which HMC has no path, so we
    // return `None` and the caller falls back to its non-gradient sampler.
    use crate::estimation::inner_optimizer::{resolve_gradient_method, InnerGradientMethod};

    // Build the gradient closure for leapfrog, dispatching on the resolved route.
    let (p_init, eta_prop, p_prop, nll_prop) = match resolve_gradient_method(model, subject) {
        InnerGradientMethod::Fd => return None,
        InnerGradientMethod::AdEventDriven => {
            // Per-dose lagtimes (eta-independent part) baked into the event
            // timeline — same source as `find_ebe`. Empty for non-lagtime models.
            let dose_lagtimes: Vec<f64> = if model.has_lagtime() {
                let zeros = vec![0.0; n_eta];
                crate::pk::compute_event_pk_params(model, subject, theta, &zeros)
                    .dose
                    .iter()
                    .map(|p| p.lagtime())
                    .collect()
            } else {
                Vec::new()
            };
            let event_data = event_driven_ad::FlatEventData::from_subject(subject, &dose_lagtimes);
            let tv_per_event =
                event_driven_ad::FlatEventTv::from_subject(model, subject, theta, &dose_lagtimes);
            let obs = subject.observations.clone();

            let grad_fn = |q: &[f64]| -> Vec<f64> {
                // Per-event scale array — q is the eta proposal, matches the
                // pattern in `inner_optimizer::find_ebe`.
                let event_scale =
                    crate::estimation::inner_optimizer::build_event_scale_array_for_ad(
                        model,
                        subject,
                        &event_data,
                        theta,
                        q,
                    );
                let (nll, g) = event_driven_ad::compute_nll_gradient_event_driven_ad(
                    q,
                    &tv_per_event,
                    &omega_inv_flat,
                    log_det_omega,
                    sigma_values,
                    &event_data,
                    &obs,
                    &cens_f64,
                    model.pk_model,
                    model.error_model,
                    &model.pk_idx_f64,
                    &model.sel_flat,
                    &event_scale,
                    model.log_transform,
                );
                last_nll.set(nll);
                g
            };

            let p_init: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
            let (qp, pp) = leapfrog(eta, &p_init, &grad_fn, step_size, n_leapfrog);
            let nll_p = last_nll.get();
            (p_init, qp, pp, nll_p)
        }
        InnerGradientMethod::AdSingleSnapshot => {
            // Single-snapshot AD path — no TV covariates, no resets.
            let tv_fn = model.tv_fn.as_ref().unwrap();
            let tv_adjusted = tv_fn(theta, &subject.covariates);
            let dose_data = FlatDoseData::from_subject(subject);
            let obs = subject.observations.clone();
            let obs_times = subject.obs_times.clone();

            let grad_fn = |q: &[f64]| -> Vec<f64> {
                // Per-observation scale array (single-snapshot AD path).
                let obs_scale = crate::estimation::inner_optimizer::build_scale_array_for_ad(
                    model, subject, theta, q,
                );
                let (nll, g) = compute_nll_gradient_ad(
                    q,
                    &tv_adjusted,
                    &omega_inv_flat,
                    log_det_omega,
                    sigma_values,
                    &dose_data,
                    &obs_times,
                    &obs,
                    &cens_f64,
                    model.pk_model,
                    model.error_model,
                    &model.pk_idx_f64,
                    &model.sel_flat,
                    &obs_scale,
                    model.log_transform,
                );
                last_nll.set(nll);
                g
            };

            let p_init: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
            let (qp, pp) = leapfrog(eta, &p_init, &grad_fn, step_size, n_leapfrog);
            let nll_p = last_nll.get();
            (p_init, qp, pp, nll_p)
        }
    };

    // Metropolis accept/reject on ΔH = H_curr − H_prop.
    // H = NLL(η) + ½‖p‖²  (identity mass matrix).
    let kinetic_curr = 0.5 * p_init.iter().map(|&x| x * x).sum::<f64>();
    let kinetic_prop = 0.5 * p_prop.iter().map(|&x| x * x).sum::<f64>();
    let delta_h = nll_current + kinetic_curr - nll_prop - kinetic_prop;

    // A divergence: the leapfrog energy error blew up (the integrator could not
    // follow the Hamiltonian flow — typically sharp posterior curvature). A
    // non-finite or large |ΔH| flags it; many divergences mean the chain is
    // failing to explore part of the posterior, so they are surfaced as a
    // diagnostic. Threshold matches Stan's Δ_max.
    let divergent = !delta_h.is_finite() || delta_h.abs() > HMC_DIVERGENCE_THRESHOLD;

    let log_u: f64 = rng.gen::<f64>().ln();
    if log_u < delta_h {
        Some((eta_prop, nll_prop, true, divergent)) // accepted: advance to proposal
    } else {
        Some((eta.to_vec(), nll_current, false, divergent)) // rejected: stay put
    }
}

// ---------------------------------------------------------------------------
// Unit tests (no autodiff dependency)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-D harmonic oscillator: NLL(q) = ½q², grad = q.
    /// H(q, p) = ½q² + ½p² is analytically conserved.
    /// Symplectic integrators have O(ε²) *bounded* global energy error —
    /// it does not grow with L.  With ε = 0.1 and H₀ ≈ 0.5 the expected
    /// |ΔH| is well below 0.01.
    #[test]
    fn test_leapfrog_energy_conservation() {
        let grad_fn = |q: &[f64]| vec![q[0]]; // ∂(½q²)/∂q = q

        let q0 = vec![1.0f64];
        let p0 = vec![0.0f64];
        let h0 = 0.5 * q0[0] * q0[0] + 0.5 * p0[0] * p0[0];

        let (q1, p1) = leapfrog(&q0, &p0, &grad_fn, 0.1, 10);
        let h1 = 0.5 * q1[0] * q1[0] + 0.5 * p1[0] * p1[0];

        assert!(
            (h1 - h0).abs() < 0.01,
            "Hamiltonian not conserved: H0={h0:.6}, H1={h1:.6}, |ΔH|={:.6}",
            (h1 - h0).abs()
        );
    }

    /// With n_steps = 0, initial half-step and correction cancel:
    /// both q and p are returned unchanged.
    #[test]
    fn test_leapfrog_zero_steps_identity() {
        let grad_fn = |q: &[f64]| vec![q[0], 2.0 * q[1]]; // arbitrary 2-D

        let q0 = vec![1.5f64, -0.3f64];
        let p0 = vec![0.7f64, -1.2f64];
        let (q1, p1) = leapfrog(&q0, &p0, &grad_fn, 0.05, 0);

        for k in 0..2 {
            assert!(
                (q1[k] - q0[k]).abs() < 1e-14,
                "q[{k}] changed with n_steps=0: {:.6} → {:.6}",
                q0[k],
                q1[k]
            );
            assert!(
                (p1[k] - p0[k]).abs() < 1e-14,
                "p[{k}] changed with n_steps=0: {:.6} → {:.6}",
                p0[k],
                p1[k]
            );
        }
    }

    /// With step_size = 0, leapfrog applies zero-sized steps:
    /// all updates multiply by 0, so both q and p are returned unchanged.
    #[test]
    fn test_leapfrog_zero_step_size_identity() {
        let grad_fn = |q: &[f64]| vec![q[0]];

        let q0 = vec![2.0f64];
        let p0 = vec![1.0f64];
        let (q1, p1) = leapfrog(&q0, &p0, &grad_fn, 0.0, 5);

        assert!((q1[0] - q0[0]).abs() < 1e-14, "q changed with step_size=0");
        assert!((p1[0] - p0[0]).abs() < 1e-14, "p changed with step_size=0");
    }

    /// Quadratic NLL: f(q) = ½ aᵢ qᵢ².  Leapfrog must decrease the quadratic
    /// model from the initial gradient step (it moves in the right direction).
    #[test]
    fn test_leapfrog_decreases_quadratic_nll() {
        // NLL = ½(4q₀² + q₁²), grad = [4q₀, q₁]
        let grad_fn = |q: &[f64]| vec![4.0 * q[0], q[1]];
        let nll = |q: &[f64]| 2.0 * q[0] * q[0] + 0.5 * q[1] * q[1];

        let q0 = vec![1.0f64, 0.5f64];
        let p0 = vec![0.0f64, 0.0f64]; // start at rest so proposal moves down-gradient
        let nll0 = nll(&q0);

        let (q1, _) = leapfrog(&q0, &p0, &grad_fn, 0.05, 5);
        let nll1 = nll(&q1);

        assert!(
            nll1 < nll0,
            "leapfrog did not reduce NLL: nll0={nll0:.6}, nll1={nll1:.6}"
        );
    }
}
