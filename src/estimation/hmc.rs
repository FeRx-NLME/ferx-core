//! HMC (Hamiltonian Monte Carlo) E-step proposals for SAEM.
//!
//! Uses standard HMC with identity mass matrix (M = I):
//!   - Momentum p ~ N(0, I)
//!   - Kinetic energy K(p) = ½ ‖p‖²
//!   - Hamiltonian H(η, p) = NLL(η) + K(p)
//!   - Velocity Störmer-Verlet (leapfrog) integrates the dynamics
//!   - Metropolis accept/reject on ΔH
//!
//! The η-gradient comes from the hand-rolled `Dual2` analytic sensitivities
//! ([`crate::estimation::inner_optimizer::analytic_eta_nll_gradient`]) — the same
//! exact gradient the FOCEI inner loop uses — so HMC needs no autodiff.

use rand::Rng;
use rand_distr::StandardNormal;

/// Leapfrog energy-error magnitude above which an HMC transition is flagged
/// divergent (matches Stan's `Δ_max`).
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
// HMC step
// ---------------------------------------------------------------------------

/// One HMC proposal for a single SAEM/Bayes subject, matching the interface of
/// `mh_steps`. The leapfrog gradient is the exact `Dual2` analytic `∂NLL/∂η`
/// ([`crate::estimation::inner_optimizer::analytic_eta_nll_gradient`]) — no autodiff.
///
/// Returns `Some((new_eta, new_nll, accepted, divergent))`, or `None` when HMC
/// cannot be applied (caller falls back to `mh_steps`):
///   - the model uses an ODE (`model.ode_spec.is_some()`),
///   - it has no analytical PK path (`model.tv_fn.is_none()`),
///   - `omega.log_det` is non-finite (degenerate variance matrix), or
///   - the Dual2 light provider can't differentiate the subject (time-varying
///     covariates, oral infusion, SS+reset, expression scaling) — then there is no
///     gradient consistent with the analytical objective.
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
    use crate::estimation::inner_optimizer::analytic_eta_nll_gradient;
    use crate::stats::likelihood::individual_nll;
    use std::cell::Cell;

    if model.ode_spec.is_some() || model.tv_fn.is_none() || !omega.log_det.is_finite() {
        return None;
    }
    // HMC needs the exact `∂NLL/∂η`. The Dual2 light provider supplies it for the
    // analytical models in scope; for anything it can't differentiate (TV
    // covariates, oral infusion, SS+reset, expression scaling) there is no
    // consistent gradient, so return `None` and let the caller fall back to its
    // gradient-free MH sampler. Scope is model-level, so one probe at `eta`
    // settles it for the whole trajectory.
    analytic_eta_nll_gradient(model, subject, theta, eta, omega, sigma_values)?;

    let n_eta = eta.len();

    // `last_nll` carries NLL(η_proposal) out of the leapfrog without a second
    // evaluation. NLL is computed by the same `individual_nll` the caller used for
    // `nll_current`, so the Metropolis ratio is exact.
    let last_nll = Cell::new(nll_current);
    // The model-level scope probe above only settles `eta`; a leapfrog step can
    // still reach a point the provider can't differentiate (e.g. a residual
    // variance `v <= 0`). If that happens mid-trajectory, a zero gradient would
    // turn the step into momentum-only free-flight and the frozen proposal could
    // be accepted on a finite ΔH — so flag it and treat the whole transition as a
    // failed/divergent move (reject, stay put) rather than silently biasing.
    let grad_failed = Cell::new(false);
    let grad_fn = |q: &[f64]| -> Vec<f64> {
        last_nll.set(individual_nll(
            model,
            subject,
            theta,
            q,
            omega,
            sigma_values,
        ));
        match analytic_eta_nll_gradient(model, subject, theta, q, omega, sigma_values) {
            Some(g) => g,
            None => {
                grad_failed.set(true);
                vec![0.0; n_eta]
            }
        }
    };

    let p_init: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
    let (eta_prop, p_prop) = leapfrog(eta, &p_init, &grad_fn, step_size, n_leapfrog);
    if grad_failed.get() {
        // Trajectory left the differentiable region — reject and flag divergent.
        return Some((eta.to_vec(), nll_current, false, true));
    }
    let nll_prop = last_nll.get();

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
