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
    {
        let __r0 = &population.subjects;
        crate::par::map(__r0.len(), |i| {
            let subject = &__r0[i];
            let _guard = mixture_class.map(crate::parser::model_parser::MixtureClassGuard::enter);
            subject_gradient_at(
                model,
                subject,
                pt.as_ref()?,
                eta_hats[i].as_slice(),
                interaction,
            )
        })
    }
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
    {
        let __r0 = &population.subjects;
        crate::par::map(__r0.len(), |i| {
            let subject = &__r0[i];
            let mut stacked: Vec<f64> = eta_hats[i].iter().copied().collect();
            for kap in &kappas[i] {
                stacked.extend(kap.iter().copied());
            }
            subject_gradient_at(model, subject, pt.as_ref()?, &stacked, interaction)
        })
    }
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

/// What an outer gradient evaluation hands the EBE warm start about each subject's mode
/// (#2549): the exact mode responses `dη̂ᵢ/dx`, one `n_eta`-vector per packed coordinate (a
/// subject the response does not reach is predicted not to move), and, under the
/// quadrature's exact anchor, each subject's exact inner Hessian at its mode — the seed of
/// the next warm inner solve there (`None` for a subject without one).
#[derive(Debug, Default)]
pub(crate) struct ModeResponses {
    /// Form the mode responses (the predictor serves the fit).
    pub(crate) want_dx: bool,
    pub(crate) eta_dx: Option<Vec<Vec<DVector<f64>>>>,
    pub(crate) hessians: Option<Vec<Option<nalgebra::DMatrix<f64>>>>,
    /// Under the quadrature, each subject's exact packed score — the rows the gradient sums —
    /// kept for the BHHH decrement at a tolerance stop (`quadrature_predicted_remaining_gain`).
    pub(crate) scores: Option<Vec<Vec<f64>>>,
}

/// The EBE warm start at `x_new` from the modes at `x_prev`: the mode map's first-order
/// expansion `η̂ + (dη̂/dx)·(x_new − x_prev)` (Almquist Eq. 48), taken whole (#2549).
///
/// `dη̂/dx` is the exact derivative of the mode (implicit function theorem), so the start
/// is accurate to second order in the outer step and the inner solve corrects the rest.
/// Nothing is capped: every bound measured on the benchmark set lost iterations somewhere —
/// a cap of two prior SDs (and, before #2549, an L2 cap of 2 on this function), or damping
/// by the Newton decrement (which measures how far the mode moves, not how wrong its
/// prediction is) — and choosing per evaluation between this start and the previous mode
/// broke the start's smooth dependence on `x`, which is what makes the objective the outer
/// optimizer sees smooth (`rtte` 18 → 35, `mm` 53 → 255 outer iterations). A start the
/// inner solve then fails from is solved again from the mode by the fit's inner pass, a
/// rescue that leaves every working prediction alone. A non-finite step leaves the mode, and
/// a Jacobian column of the wrong length is skipped.
pub fn predict_warm_etas(
    prev_etas: &[DVector<f64>],
    jacs: &[Vec<DVector<f64>>],
    x_prev: &[f64],
    x_new: &[f64],
) -> Vec<DVector<f64>> {
    prev_etas
        .iter()
        .zip(jacs.iter())
        .map(|(eta, jac)| {
            let mut start = eta.clone();
            for (k, jk) in jac.iter().enumerate() {
                let dx = x_new[k] - x_prev[k];
                if dx != 0.0 && jk.len() == start.len() {
                    start.axpy(dx, jk, 1.0);
                }
            }
            if start.iter().all(|v| v.is_finite()) {
                start
            } else {
                eta.clone()
            }
        })
        .collect()
}

#[cfg(test)]
mod first_order_prediction_tests {
    use super::predict_warm_etas;
    use nalgebra::DVector;

    /// The start is the mode map's first-order expansion, whole however long the step: a
    /// mode that moves linearly in `x` is predicted exactly, at any distance.
    #[test]
    fn the_start_is_the_whole_first_order_expansion() {
        let eta = vec![DVector::from_vec(vec![0.3, -0.1])];
        let jac = vec![vec![
            DVector::from_vec(vec![2.0, 0.0]),
            DVector::from_vec(vec![0.5, -1.0]),
        ]];
        for scale in [1e-3, 1.0, 50.0] {
            let p = predict_warm_etas(&eta, &jac, &[1.0, 2.0], &[1.0 + scale, 2.0 + scale]);
            let want = [0.3 + 2.5 * scale, -0.1 - scale];
            assert!(
                (p[0][0] - want[0]).abs() < 1e-12 * (1.0 + want[0].abs()),
                "{}",
                p[0]
            );
            assert!(
                (p[0][1] - want[1]).abs() < 1e-12 * (1.0 + want[1].abs()),
                "{}",
                p[0]
            );
        }
    }

    /// A non-finite response leaves the mode rather than starting the solve at `NaN`.
    #[test]
    fn a_non_finite_step_leaves_the_mode() {
        let eta = vec![DVector::from_vec(vec![0.3, -0.1])];
        let jac = vec![vec![DVector::from_vec(vec![f64::NAN, 1.0])]];
        let p = predict_warm_etas(&eta, &jac, &[0.0], &[1.0]);
        assert_eq!(p[0].as_slice(), &[0.3, -0.1]);
    }
}
