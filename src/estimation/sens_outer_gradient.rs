//! The exact FOCE / FOCEI outer gradient — the public entry points over the adjoint.
//!
//! Every per-subject gradient here is `dFᵢ/dx` in the packed optimizer space (log-θ /
//! Cholesky-Ω / log-σ / Fisher-z ρ), EBE response included, from the adjoint identity
//! `Ψ = F − wᵀ∇_η l`, `w = H⁻¹∇_η F` (`sens::ode_provider::adjoint`). FOCEI differentiates
//! the Laplace marginal `l + ½log|H̃|`; FOCE, ferx's Sheiner–Beal marginal with its
//! frozen-variance EBE. IOV subjects take the stacked `[η, κ₁ … κ_K]` with the block prior
//! `Ω ⊕ (I_K ⊗ Ω_IOV)`. A subject no recorder serves returns `None`, and the caller fills it
//! with its own per-subject salvage.

use crate::estimation::parameterization::theta_packs_log;
use crate::sens::ode_provider::adjoint::{Objective, PackedPoint};
use crate::types::{CompiledModel, ModelParameters, Population, Subject};
use nalgebra::DVector;
use rayon::prelude::*;

/// The exact per-subject FOCEI gradient `dFᵢ/dx` in the **packed** optimizer space, or
/// `None` outside the adjoint's scope. `eta_hat` must be the EBE for `unpack_params(x)`.
pub fn subject_packed_gradient(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    subject_gradient(model, subject, template, x, eta_hat, true)
}

/// [`subject_packed_gradient`] for an **IOV** subject, over the stacked EBE
/// `[η_bsv, κ₁ … κ_K]`.
pub fn subject_packed_gradient_iov(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    stacked_eta_hat: &[f64],
) -> Option<Vec<f64>> {
    subject_gradient(model, subject, template, x, stacked_eta_hat, true)
}

/// The exact per-subject **FOCE** (Sheiner–Beal, no interaction) gradient `dFᵢ/dx` in the
/// packed space, at FOCE's frozen-variance EBE.
pub fn subject_packed_gradient_foce(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
) -> Option<Vec<f64>> {
    subject_gradient(model, subject, template, x, eta_hat, false)
}

/// [`subject_packed_gradient_foce`] for an **IOV** subject, over the stacked EBE.
pub fn subject_packed_gradient_foce_iov(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    stacked_eta_hat: &[f64],
) -> Option<Vec<f64>> {
    subject_gradient(model, subject, template, x, stacked_eta_hat, false)
}

fn subject_gradient(
    model: &CompiledModel,
    subject: &Subject,
    template: &ModelParameters,
    x: &[f64],
    eta_hat: &[f64],
    interaction: bool,
) -> Option<Vec<f64>> {
    if subject.observations.is_empty() && subject.obs_records.is_empty() {
        return Some(vec![0.0; x.len()]);
    }
    let pt = PackedPoint::new(template, x)?;
    subject_gradient_at(model, subject, &pt, eta_hat, interaction)
}

/// [`subject_gradient`] at a [`PackedPoint`] the population loop unpacked once — the fit's
/// per-subject entry (`outer_optimizer`'s population gradients), so `x` is unpacked once per
/// evaluation rather than once per subject. `eta_hat` is the stacked `[η, κ₁ … κ_K]` under
/// IOV, as for the per-subject wrappers above.
pub(crate) fn subject_gradient_at(
    model: &CompiledModel,
    subject: &Subject,
    pt: &PackedPoint<'_>,
    eta_hat: &[f64],
    interaction: bool,
) -> Option<Vec<f64>> {
    // No data at all contributes nothing; a subject with only non-Gaussian records still
    // carries its term (the adjoint folds it in).
    if subject.observations.is_empty() && subject.obs_records.is_empty() {
        return Some(vec![0.0; pt.x.len()]);
    }
    crate::sens::ode_provider::adjoint::subject_adjoint_gradient_at(
        model,
        subject,
        pt,
        eta_hat,
        if interaction {
            Objective::Focei
        } else {
            Objective::Foce
        },
    )
}

/// Per-subject packed gradients `d(nllᵢ)/dx` (FOCEI when `interaction`, FOCE otherwise),
/// `None` for a subject no recorder serves, so the caller keeps the exact gradient for
/// the rest and salvages only those (#1529). The caller scales by 2 and zeroes fixed
/// coordinates.
///
/// `mixture_class` (1-based) is `Some(k)` for the mixture outer gradient (#977): the
/// `MIXTURE_CLASS` thread-local is set **inside** each rayon worker, so `MIXNUM` resolves
/// to class `k` there (the calling thread's guard does not reach the workers).
pub fn per_subject_packed_gradients(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
    interaction: bool,
    mixture_class: Option<usize>,
) -> Vec<Option<Vec<f64>>> {
    // One unpack for the population (every subject is at the same packed point).
    let pt = PackedPoint::new(template, x);
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let _guard = mixture_class.map(crate::parser::model_parser::MixtureClassGuard::enter);
            subject_gradient_at(
                model,
                subject,
                pt.as_ref()?,
                eta_hats[i].as_slice(),
                interaction,
            )
        })
        .collect()
}

/// [`per_subject_packed_gradients`] for an **IOV** model: `eta_hats[i]` the BSV EBEs and
/// `kappas[i]` the per-occasion κ̂, stacked into `[η_bsv, κ₁ … κ_K]` per subject.
pub fn per_subject_packed_gradients_iov(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<Option<Vec<f64>>> {
    let pt = PackedPoint::new(template, x);
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let mut stacked: Vec<f64> = eta_hats[i].iter().copied().collect();
            for kap in &kappas[i] {
                stacked.extend(kap.iter().copied());
            }
            subject_gradient_at(model, subject, pt.as_ref()?, &stacked, interaction)
        })
        .collect()
}

/// θ→packed chain rule `∂θ/∂x`: `θ` when the parameter packs in log-space, else `1.0`.
#[inline]
pub(crate) fn theta_dx_chain(template: &ModelParameters, theta: &[f64], m: usize) -> f64 {
    if theta_packs_log(template.theta_lower[m]) {
        theta[m]
    } else {
        1.0
    }
}

/// Per-subject `dη̂/dx` Jacobians (one `n_eta`-vector per packed coordinate) for the whole
/// population, or `None` if any subject is unsupported: the exact mode response
/// (`adjoint::subject_adjoint_eta_dx`) of the fit's own inner problem — FOCEI's conditional
/// mode when `interaction`, FOCE's frozen-variance mode otherwise (#2093 review).
pub fn population_eta_dx(
    model: &CompiledModel,
    population: &Population,
    template: &ModelParameters,
    x: &[f64],
    eta_hats: &[DVector<f64>],
    interaction: bool,
) -> Option<Vec<Vec<DVector<f64>>>> {
    let pt = PackedPoint::new(template, x)?;
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, s)| {
            crate::sens::ode_provider::adjoint::subject_adjoint_eta_dx_at(
                model,
                s,
                &pt,
                eta_hats[i].as_slice(),
                interaction,
            )
        })
        .collect()
}

/// [`predict_warm_etas`] with the step capped in the prior's own metric: a subject's
/// predicted step `δ` is shortened to `‖δ‖_Ω = √(δᵀΩ⁻¹δ) ≤ 2`, two prior standard deviations,
/// so the cap means the same thing whatever the scale of each random effect (#2549). A
/// vector whose length is not `Ω`'s (or an `Ω` that does not invert) keeps the unscaled cap.
pub(crate) fn predict_warm_etas_in_prior_metric(
    prev_etas: &[DVector<f64>],
    jacs: &[Vec<DVector<f64>>],
    x_prev: &[f64],
    x_new: &[f64],
    omega: &nalgebra::DMatrix<f64>,
) -> Vec<DVector<f64>> {
    const MAX_PRIOR_SDS: f64 = 2.0;
    let omega_inv = omega.clone().try_inverse();
    prev_etas
        .iter()
        .zip(jacs.iter())
        .map(|(eta, jac)| {
            let mut step = DVector::zeros(eta.len());
            for (k, jk) in jac.iter().enumerate() {
                let dx = x_new[k] - x_prev[k];
                if dx != 0.0 {
                    step += jk * dx;
                }
            }
            let norm = match omega_inv.as_ref().filter(|oi| oi.nrows() == step.len()) {
                Some(oi) => step.dot(&(oi * &step)).max(0.0).sqrt(),
                None => step.norm(),
            };
            if norm > MAX_PRIOR_SDS {
                step *= MAX_PRIOR_SDS / norm;
            }
            eta + step
        })
        .collect()
}

/// Almquist Eq. 48 warm-start: `η⁰ᵢ = η̂ᵢ + Σₖ (dη̂ᵢ/dx_k)·(x_new−x_prev)_k`.
pub fn predict_warm_etas(
    prev_etas: &[DVector<f64>],
    jacs: &[Vec<DVector<f64>>],
    x_prev: &[f64],
    x_new: &[f64],
) -> Vec<DVector<f64>> {
    // Cap on the L2 norm of a single predicted η warm-start step. The inner solve
    // re-refines from the warm start, so this only needs to keep it inside a sane
    // region: on a large or ill-conditioned outer step the linear Eq.48
    // extrapolation can overshoot the basin, and if the inner BFGS then hits
    // max_iter it can land at a different mode, perturbing the reported OFV.
    // η live on the O(1) random-effects scale, so ~2 (a few IIV SDs) rarely binds
    // on a normal step but blocks a runaway one. PR #381 review finding #8.
    const MAX_PREDICT_STEP_NORM: f64 = 2.0;
    prev_etas
        .iter()
        .zip(jacs.iter())
        .map(|(eta, jac)| {
            let mut step = DVector::zeros(eta.len());
            for (k, jk) in jac.iter().enumerate() {
                let dx = x_new[k] - x_prev[k];
                if dx != 0.0 {
                    step += jk * dx;
                }
            }
            let norm = step.norm();
            if norm > MAX_PREDICT_STEP_NORM {
                step *= MAX_PREDICT_STEP_NORM / norm;
            }
            eta + step
        })
        .collect()
}

#[cfg(test)]
mod prior_metric_cap_tests {
    use super::predict_warm_etas_in_prior_metric;
    use nalgebra::{DMatrix, DVector};

    /// The cap is two prior SDs in Ω's metric, not 2 in raw η units (#2549 review): with
    /// SDs 0.1 and 2, a step along the narrow effect is cut at 0.2 and one along the wide
    /// effect at 4, and a step inside the region passes unchanged.
    #[test]
    fn the_step_is_capped_at_two_prior_sds_per_effect() {
        let omega = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 4.0]));
        let eta = vec![DVector::from_vec(vec![0.0, 0.0])];
        let along = |k: usize, size: f64| {
            let mut col = vec![DVector::zeros(2)];
            col[0][k] = size;
            let p = predict_warm_etas_in_prior_metric(&eta, &[col], &[0.0], &[1.0], &omega);
            p[0].clone()
        };
        let narrow = along(0, 1.0);
        assert!((narrow[0] - 0.2).abs() < 1e-12, "{narrow}");
        let wide = along(1, 10.0);
        assert!((wide[1] - 4.0).abs() < 1e-12, "{wide}");
        let inside = along(1, 3.0);
        assert_eq!(inside[1], 3.0, "a step inside two prior SDs is taken whole");
    }
}
