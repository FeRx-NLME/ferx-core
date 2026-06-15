//! Helpers shared by the EM-family estimators (SAEM and IMPMAP).
//!
//! Both run a Monte-Carlo EM loop with the same Ω M-step invariants and the
//! same log-mu-referencing closed-form θ shift, so the leaf helpers that encode
//! those invariants live here and are used from both [`saem`](super::saem) and
//! [`impmap`](super::impmap). Keeping a single copy means a fix to the Ω-floor
//! rule or the mu-ref-pair selection cannot silently diverge between the two
//! estimators (which is exactly how the IMPMAP duplicates were introduced).

use crate::estimation::parameterization::theta_packs_log;
use crate::types::{CompiledModel, ModelParameters};
use nalgebra::DMatrix;

/// Positive-definite floor for free BSV Ω diagonals in the M-step.
///
/// Larger than the IOV floor (1e-8) because the SAEM BSV MH proposal scale is
/// `step_scale · chol(Ω)`: if a diagonal is allowed near zero the proposal for
/// that η collapses and the chain can no longer move it, so Ω must stay large
/// enough to keep the random walk alive. 1e-6 keeps a free η explorable while
/// being far below any plausible estimated variance. IMPMAP reuses the same
/// floor to keep its IS proposal `Σᵢ = Hᵢ⁻¹` (and the prior Ω⁻¹) well-conditioned.
pub(crate) const OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Raise every *free* diagonal entry of the BSV Ω that has fallen below `floor`
/// up to `floor`. FIX-ed diagonals (`omega_fixed[i] == true`) are left untouched
/// — they carry the user's declared variance and must not be perturbed. Missing
/// `omega_fixed` entries (slice shorter than the matrix) default to free.
pub(crate) fn floor_omega_diagonal(omega_mat: &mut DMatrix<f64>, omega_fixed: &[bool], floor: f64) {
    for i in 0..omega_mat.nrows() {
        let fixed = omega_fixed.get(i).copied().unwrap_or(false);
        if !fixed && omega_mat[(i, i)] < floor {
            omega_mat[(i, i)] = floor;
        }
    }
}

/// Log-transformed mu-referencing pairs `(theta_idx, eta_idx)`.
///
/// Only `log_transformed = true` mu-refs (patterns `THETA*exp(ETA)` and
/// `exp(log(THETA)+ETA)`) are returned. For these the typical value satisfies
/// `log(P_i) = log(θ) + η_i`, so the EM M-step can shift `log(θ) += mean(η)` in
/// closed form (the chain rule gives `d/d_log(theta) = -Σᵢ d/d_eta`). Additive
/// mu-refs (`THETA + ETA`, `log_transformed = false`) require the extra factor
/// of `theta` from the log-space chain rule and are deliberately excluded — they
/// fall through to the regular NLopt M-step.
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

/// Unpack a packed θ vector back to natural space, per the packing mask.
///
/// `mask[i] == true` means θ_i was log-packed (so unpack with `exp`); `false`
/// means identity packing (covariate exponents with a negative lower bound,
/// which must not be forced positive). The SAEM and IMPMAP θ/σ M-steps share
/// this so the unpack can never drift from the pack in [`PackedThetaSigma`].
pub(crate) fn unpack_thetas(packed: &[f64], mask: &[bool]) -> Vec<f64> {
    packed
        .iter()
        .zip(mask)
        .map(|(&p, &is_log)| if is_log { p.exp() } else { p })
        .collect()
}

/// Initial θ/σ values and bounds in the packed (log/identity) optimizer space
/// shared by the SAEM and IMPMAP M-steps.
///
/// θ uses log-packing when its lower bound is ≥ 0 (`theta_packs_log`), identity
/// otherwise — so a covariate exponent with a negative lower bound is not pinned
/// near 0 by `ln`. σ is always log-packed with fixed `[-8, 5]` bounds. FIX-ed
/// parameters get `lower == upper == packed value`, so the inner NLopt M-step
/// treats them as constants (matching the FOCE/FOCEI treatment). Built once from
/// `init_params`; both estimators then mutate the packed `log_theta`/`log_sigma`
/// in place across iterations and unpack via [`unpack_thetas`].
pub(crate) struct PackedThetaSigma {
    /// Per-θ packing selector: `true` → log, `false` → identity.
    pub theta_packs_log_mask: Vec<bool>,
    /// Initial θ in packed space.
    pub log_theta: Vec<f64>,
    /// Initial σ in packed space (always log).
    pub log_sigma: Vec<f64>,
    pub log_theta_lower: Vec<f64>,
    pub log_theta_upper: Vec<f64>,
    pub log_sigma_lower: Vec<f64>,
    pub log_sigma_upper: Vec<f64>,
}

impl PackedThetaSigma {
    pub(crate) fn new(init: &ModelParameters) -> Self {
        let n_theta = init.theta.len();
        let n_sigma = init.sigma.values.len();

        let theta_packs_log_mask: Vec<bool> = init
            .theta_lower
            .iter()
            .map(|&lo| theta_packs_log(lo))
            .collect();
        let pack_theta = |i: usize, t: f64| -> f64 {
            if theta_packs_log_mask[i] {
                t.max(1e-10).ln()
            } else {
                t
            }
        };

        // Pack initial θ (per-mask) and σ (always log).
        let log_theta: Vec<f64> = (0..n_theta).map(|i| pack_theta(i, init.theta[i])).collect();
        let log_sigma: Vec<f64> = init
            .sigma
            .values
            .iter()
            .map(|&s| s.max(1e-10).ln())
            .collect();

        // Bounds in packed space — log when log-packed, identity otherwise.
        let mut log_theta_lower: Vec<f64> = (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    init.theta_lower[i].max(1e-10).ln()
                } else {
                    init.theta_lower[i]
                }
            })
            .collect();
        let mut log_theta_upper: Vec<f64> = (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    init.theta_upper[i].min(1e9).ln()
                } else {
                    init.theta_upper[i]
                }
            })
            .collect();
        let mut log_sigma_lower = vec![-8.0f64; n_sigma];
        let mut log_sigma_upper = vec![5.0f64; n_sigma];

        // Pin FIX parameters: lower == upper == packed value.
        for i in 0..n_theta {
            if init.theta_fixed.get(i).copied().unwrap_or(false) {
                log_theta_lower[i] = log_theta[i];
                log_theta_upper[i] = log_theta[i];
            }
        }
        for i in 0..n_sigma {
            if init.sigma_fixed.get(i).copied().unwrap_or(false) {
                log_sigma_lower[i] = log_sigma[i];
                log_sigma_upper[i] = log_sigma[i];
            }
        }

        PackedThetaSigma {
            theta_packs_log_mask,
            log_theta,
            log_sigma,
            log_theta_lower,
            log_theta_upper,
            log_sigma_lower,
            log_sigma_upper,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::{analytical_model, model_with_mu_refs};
    use crate::types::GradientMethod;

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
        // chain rule used in the EM M-step assumes log-transformed parameters.
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

    #[test]
    fn unpack_thetas_applies_mask_per_index() {
        // mask[0]=log → exp; mask[1]=identity → passthrough (negative-bound exponent).
        let out = unpack_thetas(&[1.0, -0.8], &[true, false]);
        assert!((out[0] - 1.0_f64.exp()).abs() < 1e-12);
        assert_eq!(
            out[1], -0.8,
            "identity-packed theta must pass through unchanged"
        );
    }

    /// Build a minimal ModelParameters with the given theta bounds/fixed flags
    /// and a single sigma, for exercising PackedThetaSigma.
    fn params(
        theta: &[f64],
        theta_lower: &[f64],
        theta_upper: &[f64],
        theta_fixed: &[bool],
        sigma: &[f64],
        sigma_fixed: &[bool],
    ) -> ModelParameters {
        use crate::types::{OmegaMatrix, SigmaVector};
        ModelParameters {
            theta: theta.to_vec(),
            theta_names: (0..theta.len()).map(|i| format!("T{i}")).collect(),
            theta_lower: theta_lower.to_vec(),
            theta_upper: theta_upper.to_vec(),
            theta_fixed: theta_fixed.to_vec(),
            omega: OmegaMatrix::from_diagonal(&[0.1], vec!["ETA".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: sigma.to_vec(),
                names: (0..sigma.len()).map(|i| format!("E{i}")).collect(),
            },
            sigma_fixed: sigma_fixed.to_vec(),
            omega_iov: None,
            kappa_fixed: Vec::new(),
        }
    }

    #[test]
    fn packed_theta_sigma_log_packs_only_nonneg_lower_bound_theta() {
        // T0: lower 0 → log-packed; T1: lower -1 (covariate exponent) → identity.
        let p = params(
            &[2.0, -0.5],
            &[0.0, -1.0],
            &[100.0, 5.0],
            &[false, false],
            &[0.3],
            &[false],
        );
        let packed = PackedThetaSigma::new(&p);
        assert_eq!(packed.theta_packs_log_mask, vec![true, false]);
        assert!((packed.log_theta[0] - 2.0_f64.ln()).abs() < 1e-12);
        assert_eq!(packed.log_theta[1], -0.5, "identity theta packed as-is");
        // Sigma is always log-packed with fixed [-8, 5] bounds.
        assert!((packed.log_sigma[0] - 0.3_f64.ln()).abs() < 1e-12);
        assert_eq!(packed.log_sigma_lower, vec![-8.0]);
        assert_eq!(packed.log_sigma_upper, vec![5.0]);
        // Round-trips back through unpack_thetas.
        let nat = unpack_thetas(&packed.log_theta, &packed.theta_packs_log_mask);
        assert!((nat[0] - 2.0).abs() < 1e-12);
        assert!((nat[1] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn packed_theta_sigma_pins_fixed_params() {
        // T1 fixed and sigma fixed → lower == upper == packed value.
        let p = params(
            &[2.0, 7.0],
            &[0.0, 0.0],
            &[100.0, 100.0],
            &[false, true],
            &[0.3],
            &[true],
        );
        let packed = PackedThetaSigma::new(&p);
        assert_eq!(packed.log_theta_lower[1], packed.log_theta[1]);
        assert_eq!(packed.log_theta_upper[1], packed.log_theta[1]);
        assert_eq!(packed.log_sigma_lower[0], packed.log_sigma[0]);
        assert_eq!(packed.log_sigma_upper[0], packed.log_sigma[0]);
        // The free theta keeps its real bounds, not pinned.
        assert!(packed.log_theta_lower[0] < packed.log_theta_upper[0]);
    }
}
