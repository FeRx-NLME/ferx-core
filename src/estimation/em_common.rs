//! Helpers shared by the EM-family estimators (SAEM and IMPMAP).
//!
//! Both run a Monte-Carlo EM loop with the same Ω M-step invariants and the
//! same log-mu-referencing closed-form θ shift, so the leaf helpers that encode
//! those invariants live here and are used from both [`saem`](super::saem) and
//! [`impmap`](super::impmap). Keeping a single copy means a fix to the Ω-floor
//! rule or the mu-ref-pair selection cannot silently diverge between the two
//! estimators (which is exactly how the IMPMAP duplicates were introduced).

use crate::types::CompiledModel;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, MuRef};

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
}
