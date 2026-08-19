use super::*;
// Cov helpers moved to `estimation::covariance` (refactor T4); the unit tests
// below still live here and reach them across the module boundary.
use crate::estimation::covariance::{
    assemble_score_cross_product, build_non_pd_fallback_proposal, combine_covariance,
    cov_progress_eta, cov_progress_should_print, cov_progress_step, extract_eigenvalues,
    format_non_pd_warning, invert_psd_with_floor, packed_param_label, select_fd_step,
    FALLBACK_PROPOSAL_MAX_COND,
};
use crate::estimation::parameterization::{compute_bounds, pack_params};

/// The guard-rejected objective (`guard_penalty_value`) must **integrate** the
/// center-push gradient `g[i] = 100·(xs[i] − c[i])` the closures return alongside it —
/// otherwise NLopt's More-Thuente line search cannot reconcile `f` and `∇f` and fails on
/// a first-step overshoot (#486, the ODE + `iiv_on_ruv` LBFGS failure). Pin the
/// consistency (central FD of the value matches the center-push) and the wall height.
#[test]
fn guard_penalty_value_integrates_the_center_push_gradient() {
    let lower = vec![-2.0, -1.0, 0.0, -5.0];
    let upper = vec![2.0, 3.0, 4.0, 1.0];
    let xs = vec![0.5, -0.3, 1.7, -2.1];
    let n = xs.len();
    let centers: Vec<f64> = (0..n).map(|i| (lower[i] + upper[i]) / 2.0).collect();
    for i in 0..n {
        // The center-push gradient the closures pair with the penalty value.
        let analytic = 100.0 * (xs[i] - centers[i]);
        // Central FD of the exactly-quadratic penalty is exact at any step; use a large
        // `h` so the penalty change clears the f64 ULP floor of the `1e12` base
        // (a tiny `h` would lose the derivative to catastrophic cancellation).
        let h = 0.25;
        let mut xp = xs.clone();
        xp[i] += h;
        let mut xm = xs.clone();
        xm[i] -= h;
        let fd = (guard_penalty_value(&xp, &lower, &upper)
            - guard_penalty_value(&xm, &lower, &upper))
            / (2.0 * h);
        approx::assert_relative_eq!(analytic, fd, max_relative = 1e-3, epsilon = 1e-2);
    }
    // A wall: far above any feasible OFV, so a guarded point is never mistaken for best.
    assert!(guard_penalty_value(&xs, &lower, &upper) > 1e11);
    // Minimised at the bound midpoint (the center-push fixed point).
    assert!(
        guard_penalty_value(&centers, &lower, &upper) < guard_penalty_value(&xs, &lower, &upper)
    );
}

/// #603 review #1/#2/#8: the centralised EBE guard. A hard reject forces rejection
/// unconditionally; otherwise the fraction trigger behaves as before.
#[test]
fn ebe_guard_rejects_on_hard_reject_and_fraction() {
    let stats = |n_unconverged, n_start_rejected| InnerLoopStats {
        n_unconverged,
        n_fallback: 0,
        n_start_rejected,
    };

    // A single hard reject forces rejection even with a finite OFV, zero unconverged
    // fraction, and the fraction trigger disabled (negative threshold).
    assert!(ebe_guard_rejects(&stats(0, 1), 100, 1234.5, -1.0));

    // No hard reject, fraction below threshold → not rejected.
    assert!(!ebe_guard_rejects(&stats(5, 0), 100, 1234.5, 0.1)); // 0.05 <= 0.10

    // No hard reject, fraction above threshold with finite OFV → rejected.
    assert!(ebe_guard_rejects(&stats(20, 0), 100, 1234.5, 0.1)); // 0.20 > 0.10

    // Fraction trigger needs a finite OFV (non-finite is handled by the caller's
    // `!raw.is_finite()` branch), and a negative threshold disables it.
    assert!(!ebe_guard_rejects(&stats(20, 0), 100, f64::NAN, 0.1));
    assert!(!ebe_guard_rejects(&stats(99, 0), 100, 1234.5, -1.0));
}

/// `outer_ftol` auto-selection (#469): pure-TTE tightens to 1e-8; ODE/PK/LTBS
/// and TTE-on-ODE keep 1e-6; an explicit override always wins.
#[test]
fn resolve_outer_ftol_auto_and_override() {
    // Pure-TTE (exact objective) → tighten.
    assert_eq!(resolve_outer_ftol(true, false, None), 1e-8);
    // TTE carried on an ODE disposition (noisy) → safe floor.
    assert_eq!(resolve_outer_ftol(true, true, None), 1e-6);
    // Non-TTE analytical/LTBS and ODE-PK → safe floor.
    assert_eq!(resolve_outer_ftol(false, false, None), 1e-6);
    assert_eq!(resolve_outer_ftol(false, true, None), 1e-6);
    // Explicit override wins regardless of model shape.
    assert_eq!(resolve_outer_ftol(true, false, Some(1e-5)), 1e-5);
    assert_eq!(resolve_outer_ftol(false, false, Some(1e-10)), 1e-10);
}

/// Covariance progress reporter math (the pure pieces behind `cov_progress`).
/// Stride caps output at ~20 lines but never zero; the print predicate fires
/// every `step` items plus the final one; the ETA extrapolates wall-clock
/// throughput and degrades to 0 (not Inf/NaN) before any item/elapsed.
#[test]
fn test_cov_progress_math() {
    // Stride: total/20, floored at 1 for small loops.
    assert_eq!(cov_progress_step(40), 2);
    assert_eq!(cov_progress_step(100), 5);
    assert_eq!(cov_progress_step(5), 1); // < 20 → every item
    assert_eq!(cov_progress_step(0), 1); // never zero (no modulo-by-zero)

    // Print predicate: every `step`, plus always the final item.
    let step = cov_progress_step(40); // 2
    assert!(cov_progress_should_print(2, 40, step));
    assert!(!cov_progress_should_print(3, 40, step));
    assert!(cov_progress_should_print(40, 40, step)); // final, even off-stride
    assert!(cov_progress_should_print(39, 39, 2)); // final == total wins

    // ETA = elapsed · (total − n) / n. Halfway through 100 items after 10 s
    // ⇒ ~10 s remaining.
    assert!((cov_progress_eta(100, 50, 10.0) - 10.0).abs() < 1e-9);
    // Near the end the estimate shrinks.
    assert!((cov_progress_eta(100, 99, 9.9) - 0.1).abs() < 1e-9);
    // Degenerate inputs return 0, never Inf/NaN.
    assert_eq!(cov_progress_eta(100, 0, 5.0), 0.0); // no item done yet
    assert_eq!(cov_progress_eta(100, 10, 0.0), 0.0); // no wall-clock yet
    assert_eq!(cov_progress_eta(40, 40, 8.0), 0.0); // done → 0 remaining
}

/// `freeze_flat_thetas` freezes a genuinely-unmapped theta (`TVFLAT`, declared
/// but never used) — the perturbation probe confirms moving it leaves the
/// objective unchanged — while leaving the mapped, identifiable thetas free.
/// This exercises the probe machinery (#826 follow-up) at the lib level; the
/// complementary "identifiable-but-flat-at-init is NOT frozen" case is pinned
/// end-to-end by the nightly `joint_pktte_sse_recovers_pk_and_omega`.
#[test]
fn freeze_flat_thetas_freezes_only_the_unmapped_theta() {
    use crate::parser::model_parser::parse_model_file;
    use crate::{read_nonmem_csv, EstimationMethod, FitOptions};
    use std::path::Path;

    let model = parse_model_file(Path::new("examples/flat_theta_warfarin.ferx"))
        .expect("flat_theta_warfarin parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        ..FitOptions::default()
    };

    let (frozen, warnings) = freeze_flat_thetas(&model, &pop, &model.default_params, &opts)
        .expect("the unmapped TVFLAT must be detected and frozen");
    let idx = |name: &str| {
        model
            .default_params
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("theta {name} not found"))
    };
    assert!(frozen.theta_fixed[idx("TVFLAT")], "TVFLAT must be frozen");
    for name in ["TVCL", "TVV", "TVKA"] {
        assert!(
            !frozen.theta_fixed[idx(name)],
            "identifiable {name} must stay free"
        );
    }
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("TVFLAT") && w.contains("no effect")),
        "a flat-theta warning must name TVFLAT: {warnings:?}"
    );
}

/// `resolve_scaling` maps `Auto` to `Abs` (magnitude scaling) for the
/// gradient-based optimizers (incl. `Slsqp`) and to `None` for the
/// derivative-free `Bobyqa` default (and `Mma`/`TrustRegion`); explicit
/// non-`Auto` values pass through unchanged. Guards the gradient-optimizer
/// preconditioner routing (Rescale2 → Abs, the fix that recovers warfarin /
/// tvcov / two_cpt_oral_cov convergence while preserving #335).
#[test]
fn resolve_scaling_routes_auto_by_optimizer() {
    use crate::types::ParameterScaling::{Abs, Auto, None as PsNone, Rescale2};
    for opt in [
        Optimizer::Bfgs,
        Optimizer::Lbfgs,
        Optimizer::NloptLbfgs,
        Optimizer::Slsqp,
    ] {
        assert_eq!(
            resolve_scaling(Auto, opt),
            Abs,
            "{opt:?} should be Abs under Auto"
        );
    }
    for opt in [Optimizer::Bobyqa, Optimizer::Mma, Optimizer::TrustRegion] {
        assert_eq!(
            resolve_scaling(Auto, opt),
            PsNone,
            "{opt:?} should be unscaled under Auto"
        );
    }
    // Explicit values pass through regardless of optimizer.
    assert_eq!(resolve_scaling(Rescale2, Optimizer::Bobyqa), Rescale2);
    assert_eq!(resolve_scaling(PsNone, Optimizer::Bfgs), PsNone);
    assert_eq!(resolve_scaling(Abs, Optimizer::Slsqp), Abs);
}

// ── invert_psd_with_floor: regularised PD inversion ──────────────────────

/// A genuinely PD matrix is inverted unchanged (n_clipped == 0) and the
/// result satisfies `H · H⁻¹ ≈ I` to high precision.
#[test]
fn test_invert_psd_with_floor_pd_matrix_unchanged() {
    // 3×3 SPD: build as Lᵀ·L with L lower-triangular so eigenvalues are O(1).
    let l = DMatrix::from_row_slice(3, 3, &[2.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.3, 0.2, 1.2]);
    let h = l.transpose() * &l;
    let r = invert_psd_with_floor(&h).expect("PD input inverts");
    assert_eq!(r.n_clipped, 0, "PD input should not trigger clipping");
    assert!(r.min_eigenvalue > 0.0);

    let prod = &h * &r.inverse;
    let eye = DMatrix::<f64>::identity(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (prod[(i, j)] - eye[(i, j)]).abs() < 1e-9,
                "H·H⁻¹ deviates at ({i},{j}): {:.3e}",
                (prod[(i, j)] - eye[(i, j)]).abs()
            );
        }
    }
}

/// Helper: assert two matrices agree element-wise.
#[cfg(test)]
fn assert_mat_close(a: &DMatrix<f64>, b: &DMatrix<f64>, tol: f64, ctx: &str) {
    assert_eq!(a.shape(), b.shape(), "{ctx}: shape mismatch");
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!(
                (a[(i, j)] - b[(i, j)]).abs() < tol,
                "{ctx}: ({i},{j}) {:.6e} vs {:.6e}",
                a[(i, j)],
                b[(i, j)]
            );
        }
    }
}

/// Information-matrix equality: when `S = R`, all three estimators collapse
/// to the model-based `R⁻¹` (`R⁻¹SR⁻¹ = R⁻¹RR⁻¹ = R⁻¹`, and `S⁻¹ = R⁻¹`). This
/// is the asymptotic behaviour at the MLE of a correctly-specified model.
#[test]
fn test_combine_covariance_collapses_when_s_equals_r() {
    let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.5, 1.3]);
    let r = l.transpose() * &l; // SPD
    let r_inv = invert_psd_with_floor(&r).expect("R PD").inverse;
    for m in [
        CovarianceMethod::Hessian,
        CovarianceMethod::CrossProduct,
        CovarianceMethod::Sandwich,
    ] {
        let cov = combine_covariance(m, r_inv.clone(), &r)
            .unwrap_or_else(|| panic!("{m:?} should produce a covariance"));
        assert_mat_close(&cov, &r_inv, 1e-9, &format!("{m:?} with S=R"));
    }
}

/// With `S ≠ R`, the sandwich is exactly `R⁻¹ S R⁻¹`.
#[test]
fn test_combine_covariance_sandwich_matches_explicit_product() {
    let l = DMatrix::from_row_slice(2, 2, &[1.7, 0.0, 0.3, 1.1]);
    let r = l.transpose() * &l;
    let r_inv = invert_psd_with_floor(&r).expect("R PD").inverse;
    let s = DMatrix::from_row_slice(2, 2, &[3.0, 0.4, 0.4, 2.0]);
    let sandwich =
        combine_covariance(CovarianceMethod::Sandwich, r_inv.clone(), &s).expect("sandwich");
    let expected = &r_inv * &s * &r_inv;
    assert_mat_close(&sandwich, &expected, 1e-12, "sandwich = R⁻¹SR⁻¹");
    // Sandwich must stay symmetric (S and R⁻¹ are symmetric).
    assert_mat_close(
        &sandwich,
        &sandwich.transpose(),
        1e-12,
        "sandwich symmetric",
    );
}

/// A rank-deficient `S` (here a single score's outer product) is singular, so
/// `S⁻¹` (cross-product) is unavailable — but the sandwich, which never
/// inverts `S`, is still defined.
#[test]
fn test_combine_covariance_singular_s() {
    let l = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.2, 1.0]);
    let r = l.transpose() * &l;
    let r_inv = invert_psd_with_floor(&r).expect("R PD").inverse;
    let g = DVector::from_column_slice(&[1.0, 2.0]);
    let s_rank1 = &g * g.transpose(); // rank-1, singular 2×2
    assert!(
        combine_covariance(CovarianceMethod::CrossProduct, r_inv.clone(), &s_rank1).is_none(),
        "S⁻¹ must report singular S"
    );
    assert!(
        combine_covariance(CovarianceMethod::Sandwich, r_inv, &s_rank1).is_some(),
        "sandwich must tolerate rank-deficient S"
    );
}

/// A near-singular symmetric matrix with one tiny-negative eigenvalue
/// (the exact failure mode reported in issue #129) is regularised: the
/// helper flags the clip and returns a PD inverse with positive
/// diagonals — what the old code rejected as "negative diagonal".
#[test]
fn test_invert_psd_with_floor_clips_negative_eigenvalue() {
    // H = Q diag(λ) Qᵀ with λ = [1.0, 0.5, -1e-9]. The tiny-negative
    // eigenvalue is the kind of FD noise the issue calls out.
    let q = {
        // Any orthogonal 3×3 will do. Use a Householder reflector built
        // from v = (1, 1, 1)/√3:  Q = I - 2 v vᵀ.
        let v = DVector::from_column_slice(&[1.0, 1.0, 1.0]) / (3.0_f64).sqrt();
        let mut m = DMatrix::<f64>::identity(3, 3);
        m -= 2.0 * &v * v.transpose();
        m
    };
    let lambdas = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 0.5, -1e-9]));
    let h = &q * lambdas * q.transpose();

    let r = invert_psd_with_floor(&h).expect("near-PD input must regularise");
    assert_eq!(r.n_clipped, 1, "exactly one eigenvalue should clip");
    assert!(
        r.min_eigenvalue < 0.0 && r.min_eigenvalue.abs() < 1e-6,
        "min_eigenvalue should record the raw (pre-clip) value: {:.3e}",
        r.min_eigenvalue
    );

    // Inverse is PD ⇒ all diagonal entries positive. This is the assertion
    // the old neg-diag check used to fail on for the warfarin FD Hessian.
    for i in 0..3 {
        assert!(
            r.inverse[(i, i)] > 0.0,
            "regularised inverse diag[{i}] = {:.3e} should be positive",
            r.inverse[(i, i)]
        );
    }
    // Inverse is also numerically symmetric.
    for i in 0..3 {
        for j in i + 1..3 {
            assert!(
                (r.inverse[(i, j)] - r.inverse[(j, i)]).abs() < 1e-12,
                "inverse not symmetric at ({i},{j})",
            );
        }
    }
}

/// Hopelessly indefinite input (all eigenvalues ≤ 0) returns None. Because
/// the eigenvalues are finite, `compute_covariance` surfaces this as
/// `CovarianceStepResult::FailedNonPd` — carrying the eigenvalue-list warning
/// and a usable fallback proposal — rather than `Unusable`.
#[test]
fn test_invert_psd_with_floor_rejects_negative_definite() {
    let h = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
    assert!(invert_psd_with_floor(&h).is_none());
    // The eigenvalues are finite, so the fallback path is taken (not Unusable):
    // extract_eigenvalues succeeds and the proposal is all-finite and PD.
    let eigs = extract_eigenvalues(&h).expect("finite eigenvalues");
    assert!(eigs.iter().all(|e| e.is_finite()));
    let proposal = build_non_pd_fallback_proposal(&h, &[0, 1], 2, 4.0);
    assert!(
        proposal.iter().all(|v| v.is_finite()),
        "fallback proposal must be finite for a finite non-PD Hessian"
    );
}

/// extract_eigenvalues returns eigenvalues sorted descending and returns None
/// for inputs with non-finite entries.
#[test]
fn test_extract_eigenvalues_sorts_descending() {
    let h = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
    let ev = extract_eigenvalues(&h).expect("finite eigenvalues for this input");
    assert_eq!(ev.len(), 2);
    assert!(ev[0] >= ev[1], "eigenvalues must be sorted descending");
    assert!(
        ev.iter().all(|&e| e < 0.0),
        "both eigenvalues must be negative"
    );
}

/// format_non_pd_warning produces a message with the expected structure and
/// includes the eigenvalue list.
#[test]
fn test_format_non_pd_warning_structure() {
    let ev = vec![8.4, 2.1, 0.3, -0.01];
    let msg = format_non_pd_warning(&ev);
    assert!(
        msg.contains("Hessian is not positive definite"),
        "message must flag non-PD Hessian"
    );
    assert!(
        msg.contains("Eigenvalues:"),
        "message must include eigenvalue list"
    );
    assert!(
        msg.contains("SE estimates not available"),
        "message must indicate SEs are unavailable"
    );
    // Most-negative eigenvalue appears in the output.
    assert!(
        msg.contains("-0.0100"),
        "negative eigenvalue must appear: {msg}"
    );
}

/// extract_eigenvalues returns None when the matrix contains a NaN entry.
#[test]
fn test_extract_eigenvalues_none_on_nan() {
    let mut h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    h[(0, 0)] = f64::NAN;
    assert!(
        extract_eigenvalues(&h).is_none(),
        "NaN entry must cause None return"
    );
}

/// packed_param_label decodes the lower-triangular block-omega index correctly.
/// Packing order (column-major lower triangle): (0,0), (1,0), (1,1).
/// With n_theta=2, packed_idx=3 → omega_idx=1 → (row=1, col=0).
#[test]
fn test_packed_param_label_block_omega() {
    use crate::types::{OmegaMatrix, SigmaVector};
    let mut mat = DMatrix::zeros(2, 2);
    mat[(0, 0)] = 0.04;
    mat[(1, 1)] = 0.04;
    mat[(0, 1)] = 0.01;
    mat[(1, 0)] = 0.01;
    let free_mask = DMatrix::from_element(2, 2, true);
    let omega = OmegaMatrix::from_matrix_with_mask(
        mat,
        vec!["ETA_CL".into(), "ETA_V".into()],
        false,
        free_mask,
    );
    let template = ModelParameters {
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.1, 5.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![false, false, false],
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
        mixture: None,
    };

    // n_theta=2, so: idx=2 → omega[ETA_CL, ETA_CL], idx=3 → omega[ETA_V, ETA_CL] (off-diag),
    // idx=4 → omega[ETA_V, ETA_V].
    let label_diag = packed_param_label(2, &template);
    assert_eq!(label_diag, "omega[ETA_CL, ETA_CL]", "diagonal 0,0");

    let label_off = packed_param_label(3, &template);
    assert_eq!(label_off, "omega[ETA_V, ETA_CL]", "off-diagonal 1,0");

    let label_diag2 = packed_param_label(4, &template);
    assert_eq!(label_diag2, "omega[ETA_V, ETA_V]", "diagonal 1,1");
}

/// `format_non_pd_warning` is a pure formatting function: it embeds whatever
/// eigenvalue list it receives, regardless of sign. This test exercises the
/// fixed-4 and scientific-3 branches of `fmt_eig` without needing an actual
/// non-PD Hessian.
///
/// Note: in production `format_non_pd_warning` is only reached when
/// `invert_psd_with_floor` returns `None` (all eigenvalues ≤ 0), so the
/// "all-positive" input below is a formatter unit test, not a semantic one.
#[test]
fn test_format_non_pd_warning_all_positive() {
    let ev = vec![5.0, 0.01, 1e-9];
    let msg = format_non_pd_warning(&ev);
    assert!(
        msg.contains("Hessian is not positive definite"),
        "message must flag non-PD Hessian even for all-positive inputs: {msg}"
    );
    assert!(
        msg.contains("5.0000"),
        "largest eigenvalue in output: {msg}"
    );
    assert!(msg.contains("0.0100"), "medium eigenvalue in output: {msg}");
    // Tiny eigenvalue formatted in scientific notation.
    assert!(msg.contains("e-"), "tiny eigenvalue in scientific: {msg}");
}

/// packed_param_label — sigma[1] and sigma[2] paths (1-indexed by convention).
#[test]
fn test_packed_param_label_sigma() {
    use crate::types::SigmaVector;
    // n_theta=1 (diagonal omega), n_omega=1 (diagonal), n_sigma=2
    let template = ModelParameters {
        theta: vec![5.0],
        theta_names: vec!["CL".into()],
        theta_lower: vec![0.1],
        theta_upper: vec![50.0],
        theta_fixed: vec![false],
        omega: crate::types::OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.1, 0.2],
            names: vec!["ADD".into(), "PROP".into()],
        },
        sigma_fixed: vec![false, false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
        mixture: None,
    };
    // packed layout: [theta(0), omega(1), sigma(2), sigma(3)]
    assert_eq!(packed_param_label(2, &template), "sigma[1]");
    assert_eq!(packed_param_label(3, &template), "sigma[2]");
}

/// packed_param_label — kappa[1] path (IOV diagonal omega).
#[test]
fn test_packed_param_label_kappa() {
    use crate::types::SigmaVector;
    let template = ModelParameters {
        theta: vec![5.0],
        theta_names: vec!["CL".into()],
        theta_lower: vec![0.1],
        theta_upper: vec![50.0],
        theta_fixed: vec![false],
        omega: crate::types::OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: Some(crate::types::OmegaMatrix::from_diagonal(
            &[0.02],
            vec!["KAPPA_CL".into()],
        )),
        kappa_fixed: vec![false],
        mixture: None,
    };
    // packed layout: [theta(0), omega(1), sigma(2), kappa(3)]
    assert_eq!(packed_param_label(3, &template), "kappa[1]");
}

/// invert_psd_with_floor severity thresholds: 1-of-3 clipped → pct=33 → "minor".
#[test]
fn test_regularization_severity_minor() {
    // Build a matrix with exactly one eigenvalue near-zero so exactly 1 of 3 is clipped.
    // Diagonal 3×3: eigenvalues are the diagonal entries.
    let h = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
        1.0, 1.0, 1e-20, // this one will be clipped
    ]));
    let r = invert_psd_with_floor(&h).expect("should succeed");
    assert_eq!(r.n_clipped, 1, "exactly one eigenvalue should be clipped");
    // pct = 1*100/3 = 33 → "minor" threshold
    let pct = r.n_clipped * 100 / 3;
    assert_eq!(pct, 33, "33% → minor severity bucket");
}

/// fd_hessian_step = 0.0 triggers Unusable early return in compute_covariance.
#[test]
fn test_compute_covariance_invalid_eps() {
    use crate::types::{FitOptions, OmegaMatrix, SigmaVector};
    let model = make_model();
    let population = make_population(1);
    let template = ModelParameters {
        theta: vec![5.0, 50.0],
        theta_names: vec!["CL".into(), "V".into()],
        theta_lower: vec![0.1, 1.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false, false],
        omega: OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: vec![],
        mixture: None,
    };
    let x_hat: Vec<f64> = vec![
        5.0_f64.ln(),
        50.0_f64.ln(),
        0.04_f64.sqrt().ln(),
        0.1_f64.ln(),
    ];
    let eta_hats = vec![nalgebra::DVector::zeros(1)];
    let h_mats = vec![DMatrix::zeros(1, 1)];
    let kappas = vec![vec![]];
    let mut opts = FitOptions::default();
    opts.fd_hessian_step = 0.0;

    let result = compute_covariance(
        &x_hat,
        &template,
        &model,
        &population,
        &eta_hats,
        &h_mats,
        &kappas,
        &opts,
    );
    assert!(
        matches!(result, CovarianceStepResult::Unusable(_)),
        "eps=0.0 must return Unusable"
    );
    if let CovarianceStepResult::Unusable(msg) = result {
        assert!(
            msg.contains("fd_hessian_step"),
            "message names the option: {msg}"
        );
    }
}

/// A cancel flag set during the covariance step short-circuits the
/// finite-difference Hessian loop and returns `Unusable` (cooperative abort)
/// instead of running the perturbed-point sweep to completion. `verbose` is
/// on so the drained points also exercise the progress reporter's closure.
#[test]
fn test_compute_covariance_cancelled() {
    use crate::cancel::CancelFlag;
    use crate::types::FitOptions;
    let model = make_model();
    // Same near-optimum synthetic data as the reconverged-FD test, so the
    // base OFV is finite and the function reaches the (short-circuited)
    // Hessian loop rather than failing earlier.
    let mut population = make_population(8);
    for s in &mut population.subjects {
        s.observations = vec![1.80967, 1.34064, 0.89866];
    }
    let mut template = model.default_params.clone();
    template.omega_fixed = vec![true];
    template.sigma_fixed = vec![true];
    let x = pack_params(&template);

    let n_subj = 8;
    let n_eta = 1;
    let n_obs = 3;
    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    let h_matrices: Vec<DMatrix<f64>> = (0..n_subj)
        .map(|_| DMatrix::from_element(n_obs, n_eta, 0.1))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

    let flag = CancelFlag::new();
    flag.cancel(); // pre-cancel: every perturbed point short-circuits

    let mut options = FitOptions::default();
    options.interaction = true; // FOCEI → analytical FD Hessian path
    options.verbose = true; // also drive the progress reporter closure
    options.cancel = Some(flag);

    let result = compute_covariance(
        &x,
        &template,
        &model,
        &population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &options,
    );
    // A cancelled step must be `Unusable` and name the cancellation — never
    // `Success`/`FailedNonPd`. A single `matches!` assertion keeps the
    // not-supposed-to-happen variants from becoming dead (uncoverable) arms.
    assert!(
        matches!(&result, CovarianceStepResult::Unusable(msg) if msg.contains("cancelled")),
        "cancelled covariance must be Unusable(cancelled)"
    );
}

/// `assemble_score_cross_product` honours the cancel flag: each subject's
/// score short-circuits to a zero vector, so the assembled S-matrix is
/// all-zero and finite (no panic). The caller discards it via the
/// post-assembly cancel bail in `compute_covariance`.
#[test]
fn test_assemble_score_cross_product_cancelled() {
    use crate::cancel::CancelFlag;
    use crate::types::FitOptions;
    let model = make_model();
    let population = make_population(4);
    let template = model.default_params.clone();
    let x = pack_params(&template);
    let bounds = compute_bounds(&template);

    let n_subj = 4;
    let n_eta = 1;
    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    let h_matrices: Vec<DMatrix<f64>> = (0..n_subj)
        .map(|_| DMatrix::identity(n_eta, n_eta))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];
    let free_idx: Vec<usize> = (0..x.len()).collect();

    let flag = CancelFlag::new();
    flag.cancel();
    let mut options = FitOptions::default();
    options.cancel = Some(flag);

    let s = assemble_score_cross_product(
        &x,
        &template,
        &model,
        &population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &bounds,
        &options,
        &free_idx,
    );
    assert!(
        s.iter().all(|v| v.is_finite()),
        "cancelled S must be finite"
    );
    assert!(
        s.iter().all(|v| *v == 0.0),
        "cancelled S must be all-zero (per-subject scores short-circuited)"
    );
}

/// Empty matrix is a valid zero-dimensional input (all-FIX parameter
/// case). Returns a 0×0 inverse without clipping.
#[test]
fn test_invert_psd_with_floor_empty() {
    let r = invert_psd_with_floor(&DMatrix::<f64>::zeros(0, 0)).expect("0×0 input must succeed");
    assert_eq!(r.inverse.nrows(), 0);
    assert_eq!(r.n_clipped, 0);
}

// ── detect_stagnation: NLopt stagnation-guard unit tests ─────────────────

/// Build a minimal `NloptState` for `detect_stagnation` unit tests.
/// The non-stagnation fields (`cached_etas`, `cached_h_mats`, `prev_x`)
/// are left empty because `detect_stagnation` reads only the
/// `*_evals` / `*_improvement` / `stagnation_stopped` fields.
fn fresh_state() -> NloptState {
    NloptState {
        cached_etas: Vec::new(),
        cached_h_mats: Vec::new(),
        cached_etas_by_class: Vec::new(),
        best_ofv: 0.0,
        n_evals: 0,
        n_grad_evals: 0,
        prev_x: Vec::new(),
        last_improvement_eval: 0,
        best_at_last_improvement: f64::INFINITY,
        stagnation_stopped: false,
    }
}

/// `detect_stagnation(enabled=false)` is a no-op: it never latches, never
/// fires, even after a window of zero improvement.  Replaces the
/// end-to-end `stagnation_guard_toggle_runs_to_natural_termination` test
/// (removed from `tests/new_optimizers.rs`), which became unreliable
/// after SLSQP's own xtol fires before the guard window elapses on the
/// warfarin example (both guard-on and guard-off now exit at exactly
/// 100 evals via NLopt `XtolReached`, so the e2e toggle comparison no
/// longer discriminates).
#[test]
fn test_detect_stagnation_disabled_never_fires() {
    let mut state = fresh_state();
    state.best_ofv = -100.0;
    state.best_at_last_improvement = -100.0;
    // Far past the stagnation window (n=7 → window = max(3*8, 50) = 50)
    // with zero improvement: still must not fire when disabled.
    for n_evals in 0..200 {
        state.n_evals = n_evals;
        assert!(
            !detect_stagnation(&mut state, 7, false),
            "enabled=false must never fire (n_evals={n_evals})"
        );
    }
    assert!(
        !state.stagnation_stopped,
        "disabled path must not latch `stagnation_stopped`"
    );
}

/// `detect_stagnation(enabled=true)` fires once `n_evals - last_improvement`
/// reaches the stagnation window and latches sticky thereafter.
#[test]
fn test_detect_stagnation_enabled_fires_at_window_and_latches() {
    let mut state = fresh_state();
    state.best_ofv = -100.0;
    state.best_at_last_improvement = -100.0; // identical → no improvement
    state.last_improvement_eval = 0;

    let n = 7usize;
    let window = (3 * (n + 1)).max(50); // = 50

    // Within the window, no firing.
    for n_evals in 1..window {
        state.n_evals = n_evals;
        assert!(
            !detect_stagnation(&mut state, n, true),
            "must not fire inside window (n_evals={n_evals}, window={window})"
        );
        assert!(!state.stagnation_stopped);
    }

    // At the window, fires and latches.
    state.n_evals = window;
    assert!(
        detect_stagnation(&mut state, n, true),
        "must fire at window (n_evals={window})"
    );
    assert!(
        state.stagnation_stopped,
        "first firing must latch `stagnation_stopped`"
    );

    // Latched: subsequent calls keep returning `true` without re-checking
    // the window arithmetic.  Drop n_evals well below the window to prove
    // the short-circuit is on `stagnation_stopped`, not on the counter.
    state.n_evals = 1;
    assert!(
        detect_stagnation(&mut state, n, true),
        "latched state must stay sticky-true regardless of n_evals"
    );
}

/// `detect_stagnation` resets the improvement counter when OFV moves down
/// by more than the 1e-3 threshold — so a long run of fruitful descent
/// never triggers the guard.
#[test]
fn test_detect_stagnation_resets_on_improvement() {
    let mut state = fresh_state();
    state.best_ofv = -100.0;
    state.best_at_last_improvement = -100.0;
    state.last_improvement_eval = 0;

    let n = 7usize;
    // Walk almost up to the window with zero improvement…
    state.n_evals = 49;
    assert!(!detect_stagnation(&mut state, n, true));

    // …then improve OFV by > 1e-3.  Improvement must reset the
    // last-improvement counter so the next 50 evals start fresh.
    state.best_ofv = -100.5;
    state.n_evals = 50;
    assert!(
        !detect_stagnation(&mut state, n, true),
        "improvement must reset the counter"
    );
    assert_eq!(
        state.last_improvement_eval, 50,
        "last_improvement_eval must advance to the improving eval"
    );
    assert_eq!(
        state.best_at_last_improvement, -100.5,
        "best_at_last_improvement must update to the new best"
    );

    // Now we need another full window of zero improvement before firing.
    state.n_evals = 99;
    assert!(!detect_stagnation(&mut state, n, true));
    state.n_evals = 100;
    assert!(detect_stagnation(&mut state, n, true));
}

/// Improvement *below* the 1e-3 threshold counts as stagnation — the
/// guard is deliberately noise-tolerant.  Without this, OFV noise of a
/// few ULPs would constantly reset the counter and the guard would
/// never fire.
#[test]
fn test_detect_stagnation_subthreshold_improvement_does_not_reset() {
    let mut state = fresh_state();
    state.best_ofv = -100.0;
    state.best_at_last_improvement = -100.0;
    state.last_improvement_eval = 0;

    let n = 7usize;
    // Improve OFV by 5e-4 — below the 1e-3 threshold.  Counter must
    // NOT reset.
    state.best_ofv = -100.0005;
    state.n_evals = 25;
    assert!(!detect_stagnation(&mut state, n, true));
    assert_eq!(
        state.last_improvement_eval, 0,
        "sub-threshold improvement must not advance the counter"
    );

    // 50 evals after the original last_improvement_eval (= 0), it fires.
    state.n_evals = 50;
    assert!(detect_stagnation(&mut state, n, true));
}

#[test]
fn test_reconverge_this_eval_schedule() {
    let mut opts = FitOptions::default();

    // Interval 0 (the default): never reconverge, and never a
    // divide-by-zero from the modulo (the `!= 0` guard short-circuits).
    opts.reconverge_gradient_interval = 0;
    for idx in 0..7 {
        assert!(!reconverge_this_eval(&opts, idx), "idx {idx}");
    }

    // Interval 1: every eval reconverges (the always-on case).
    opts.reconverge_gradient_interval = 1;
    for idx in 0..7 {
        assert!(reconverge_this_eval(&opts, idx), "idx {idx}");
    }

    // Interval 5: reconverge only on 0, 5, 10, …
    opts.reconverge_gradient_interval = 5;
    let got: Vec<usize> = (0..12)
        .filter(|&i| reconverge_this_eval(&opts, i))
        .collect();
    assert_eq!(got, vec![0, 5, 10]);
}

use crate::types::{
    BloqMethod, CompiledModel, DoseEvent, ErrorModel, FitOptions, GradientMethod, ModelParameters,
    OmegaMatrix, PkModel, PkParams, Population, SigmaVector, Subject,
};
use nalgebra::DVector;
use std::collections::HashMap;

fn make_model() -> CompiledModel {
    let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
    let default_params = ModelParameters {
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.1, 5.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
        mixture: None,
    };
    CompiledModel {
        name: "outer_test".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
                p
            },
        ),
        n_theta: 2,
        n_eta: 1,
        n_epsilon: 1,
        n_kappa: 0,
        kappa_names: Vec::new(),
        theta_names: vec!["TVCL".into(), "TVV".into()],
        eta_names: vec!["ETA_CL".into()],
        indiv_param_names: vec!["CL".into(), "V".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        default_params,
        omega_init_as_sd: vec![false],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: Vec::new(),
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
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    }
}

fn make_population(n_subj: usize) -> Population {
    let subjects = (0..n_subj)
        .map(|_| Subject {
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![25.0, 15.0, 9.0],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![1, 1, 1],
            obs_l2: Vec::new(),
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            obs_records: vec![],
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn check_gradient(model: &CompiledModel, population: &Population, n_eta: usize) {
    let template = &model.default_params;
    let n_subj = population.subjects.len();
    let n_obs = population.subjects[0].observations.len();

    let x = pack_params(template);
    let bounds = compute_bounds(template);
    let n = x.len();
    // FOCE non-interaction: the AD/analytical population-gradient path is the
    // Sheiner–Beal-derived closed form for the SB NLL. FOCEI INTER now uses
    // the Almquist Laplace NLL, whose gradient takes the FD fallback in
    // `subject_nll_pop_grad`; under INTER this test would be vacuous
    // (FD-vs-FD). Set interaction=false to exercise the analytical path.
    let mut options = FitOptions::default();
    options.interaction = false;

    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    // Use a non-zero H-matrix so r_tilde = R + H·Ω·Hᵀ depends on Ω and
    // the omega/off-diagonal Cholesky gradients are non-trivially exercised.
    let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
        .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

    let ad_grad = ad_population_gradient(
        &x,
        n_subj,
        template,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &bounds,
        &options,
    );

    let ofv_at = |xp: &[f64]| -> f64 {
        let p = unpack_params(xp, template);
        2.0 * pop_nll(
            model,
            population,
            &p,
            &eta_hats,
            &h_matrices,
            &kappas,
            options.interaction,
        )
    };
    let eps = 1e-4;
    let fd_grad: Vec<f64> = (0..n)
        .map(|j| {
            let h = eps * (1.0 + x[j].abs());
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            (ofv_at(&xp) - ofv_at(&xm)) / (2.0 * h)
        })
        .collect();

    for j in 0..n {
        let tol = 1e-4 * (1.0 + fd_grad[j].abs());
        assert!(
            (ad_grad[j] - fd_grad[j]).abs() < tol,
            "grad[{j}]: AD={:.6e}, FD={:.6e}",
            ad_grad[j],
            fd_grad[j],
        );
    }
}

/// IIV (diagonal omega, 1 ETA): analytical path.
#[test]
fn test_outer_ad_gradient_iiv() {
    check_gradient(&make_model(), &make_population(3), 1);
}

/// Pre-flight flat-theta guard (#826): a model whose second theta (TVV) is
/// structurally flat — V no longer reads it — is detected and frozen, while the
/// active TVCL is left free. A single unmapped theta must not kill the fit.
#[test]
fn preflight_freezes_flat_theta() {
    let mut model = make_model();
    // Drop theta[1] from the structural model: V becomes a constant, so TVV
    // never reaches the objective (identically-zero outer gradient).
    model.pk_param_fn = Box::new(
        |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
            let mut p = PkParams::default();
            p.values[0] = theta[0] * eta[0].exp();
            p.values[1] = 50.0;
            p
        },
    );
    let pop = make_population(3);
    let mut options = FitOptions::default();
    options.interaction = false;

    let (frozen, warnings) = freeze_flat_thetas(&model, &pop, &model.default_params, &options)
        .expect("flat TVV must be detected");
    assert!(!frozen.theta_fixed[0], "active TVCL stays free");
    assert!(frozen.theta_fixed[1], "flat TVV is frozen (FIX)");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("TVV") && w.contains("no effect")),
        "warning must name the flat param: {warnings:?}"
    );
}

/// Negative case: every theta of the baseline model reaches the objective, so
/// the pre-flight freezes nothing (returns `None`, leaving `init_params` alone).
#[test]
fn preflight_no_freeze_when_all_thetas_active() {
    let model = make_model();
    let pop = make_population(3);
    let mut options = FitOptions::default();
    options.interaction = false;
    assert!(
        freeze_flat_thetas(&model, &pop, &model.default_params, &options).is_none(),
        "no theta is flat — nothing to freeze"
    );
}

/// An out-of-bounds initial value for a flat theta must be frozen (and reported)
/// at the *clamped* value the gradient was evaluated at — never pinned outside its
/// declared bounds, and never reported as the raw out-of-bounds init.
#[test]
fn preflight_freezes_out_of_bounds_flat_theta_at_clamped_value() {
    let mut model = make_model();
    model.pk_param_fn = Box::new(
        |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
            let mut p = PkParams::default();
            p.values[0] = theta[0] * eta[0].exp();
            p.values[1] = 50.0; // TVV (theta[1]) is flat
            p
        },
    );
    // Push TVV's initial well above its declared upper bound (500).
    model.default_params.theta[1] = 1000.0;
    let upper = model.default_params.theta_upper[1];
    let pop = make_population(3);
    let mut options = FitOptions::default();
    options.interaction = false;

    let (frozen, warnings) = freeze_flat_thetas(&model, &pop, &model.default_params, &options)
        .expect("flat TVV must be detected");
    assert!(frozen.theta_fixed[1], "flat TVV is frozen");
    assert!(
        frozen.theta[1].is_finite() && frozen.theta[1] <= upper + 1e-6,
        "frozen value {} must sit within the declared upper bound {upper}",
        frozen.theta[1]
    );
    assert!(
        frozen.theta[1] < 1000.0,
        "must not pin at the out-of-bounds init (1000): {}",
        frozen.theta[1]
    );
    assert!(
        !warnings.iter().any(|w| w.contains("1000")),
        "warning must report the clamped value, not the out-of-bounds init: {warnings:?}"
    );
}

/// Block omega (2×2 with off-diagonal): tests Cholesky-param gradient.
#[test]
fn test_outer_ad_gradient_block_omega() {
    use crate::types::{OmegaMatrix, PkParams};
    // 2-ETA model: CL and V both random with correlation.
    // Build 2×2 omega with variance 0.04 on diagonal and covariance 0.01.
    let mut mat = nalgebra::DMatrix::zeros(2, 2);
    mat[(0, 0)] = 0.04;
    mat[(1, 1)] = 0.04;
    mat[(0, 1)] = 0.01;
    mat[(1, 0)] = 0.01;
    let free_mask = nalgebra::DMatrix::from_element(2, 2, true);
    let omega = OmegaMatrix::from_matrix_with_mask(
        mat,
        vec!["ETA_CL".into(), "ETA_V".into()],
        false,
        free_mask,
    );
    let default_params = ModelParameters {
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.1, 5.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![false, false, false],
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
        mixture: None,
    };
    let model = CompiledModel {
        name: "block_test".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1] * eta[1].exp();
                p
            },
        ),
        n_theta: 2,
        n_eta: 2,
        n_epsilon: 1,
        n_kappa: 0,
        kappa_names: Vec::new(),
        theta_names: vec!["TVCL".into(), "TVV".into()],
        eta_names: vec!["ETA_CL".into(), "ETA_V".into()],
        indiv_param_names: vec!["CL".into(), "V".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        default_params,
        omega_init_as_sd: vec![false; 2],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: Vec::new(),
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        tv_fn: None,
        pk_indices: vec![0, 1],
        eta_map: vec![0, 1],
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
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    };
    check_gradient(&model, &make_population(3), 2);
}

/// With omega_iov=None but non-empty kappas, subject_nll_pop_grad falls
/// through to central FD (no analytical IOV path without omega_iov).
/// The population-sum must still match the population-level FD reference.
///
/// Note: with omega_iov=None the IOV NLL formula is not exercised here —
/// this test covers the FD *code path*, not IOV NLL correctness.
#[test]
fn test_outer_ad_gradient_fd_fallback_path() {
    // `subject_nll_at` only enters the IOV NLL branch when kappas is
    // non-empty AND omega_iov is Some. With omega_iov=None and non-empty
    // kappas the function falls through to standard FOCE; without
    // omega_iov the dispatch in subject_nll_pop_grad also falls through to
    // central FD — the code path this test exercises.
    let model = make_model();

    let template = &model.default_params;
    let n_subj = 3;
    let n_eta = 1;
    let n_obs = 3;
    let population = make_population(n_subj);

    let x = pack_params(template);
    let bounds = compute_bounds(template);
    let n = x.len();
    let options = FitOptions::default();

    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
        .map(|_| nalgebra::DMatrix::zeros(n_obs, n_eta))
        .collect();
    // Non-empty kappas trigger the FD fallback path.
    let kappas: Vec<Vec<DVector<f64>>> = (0..n_subj).map(|_| vec![DVector::zeros(1)]).collect();

    let ad_grad = ad_population_gradient(
        &x,
        n_subj,
        template,
        &model,
        &population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &bounds,
        &options,
    );

    let ofv_at = |xp: &[f64]| -> f64 {
        let p = unpack_params(xp, template);
        2.0 * pop_nll(
            &model,
            &population,
            &p,
            &eta_hats,
            &h_matrices,
            &kappas,
            options.interaction,
        )
    };
    let eps = 1e-4;
    let fd_grad: Vec<f64> = (0..n)
        .map(|j| {
            let h = eps * (1.0 + x[j].abs());
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            (ofv_at(&xp) - ofv_at(&xm)) / (2.0 * h)
        })
        .collect();

    for j in 0..n {
        let tol = 1e-3 * (1.0 + fd_grad[j].abs());
        assert!(
            (ad_grad[j] - fd_grad[j]).abs() < tol,
            "IOV grad[{j}]: AD={:.6e}, FD={:.6e}",
            ad_grad[j],
            fd_grad[j],
        );
    }
}

// ── covariance_gradient (issue #209 / #243) ─────────────────────────────

/// `covariance_gradient` (FOCE path, interaction=false) must match FD of
/// `ofv_fixed = 2·pop_nll`. The Sheiner–Beal marginal already carries the Ω
/// penalty via R̃ = HΩHᵀ + R, so there is no separate omega-prior add-back
/// (issue #243 — adding one double-counted Ω and under-stated the FOCE
/// omega SEs).
#[test]
fn test_covariance_gradient_foce_matches_fd_ofv_fixed() {
    let model = make_model();
    let template = &model.default_params;
    let population = make_population(3);
    let n_subj = 3;
    let n_obs = 3;
    let n_eta = 1;

    let x = pack_params(template);
    let bounds = compute_bounds(template);
    let n = x.len();
    let mut options = FitOptions::default();
    options.interaction = false; // FOCE: Sheiner–Beal marginal

    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
        .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

    // FOCE ofv_fixed = 2·pop_nll (Ω penalty already inside the SB marginal).
    let ofv_fixed = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, template);
        2.0 * pop_nll(
            &model,
            &population,
            &p,
            &eta_hats,
            &h_matrices,
            &kappas,
            false, // FOCE
        )
    };

    let grad = covariance_gradient(
        &x,
        template,
        &model,
        &population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &bounds,
        &options,
    );

    let eps = 1e-4;
    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += h;
        xm[j] -= h;
        let fd = (ofv_fixed(&xp) - ofv_fixed(&xm)) / (2.0 * h);
        let tol = 1e-3 * (1.0 + fd.abs());
        assert!(
            (grad[j] - fd).abs() < tol,
            "covariance_gradient FOCE [{j}]: grad={:.6e}, FD_ofv={:.6e}",
            grad[j],
            fd,
        );
    }
}

/// `covariance_gradient` (FOCEI path, interaction=true) must match FD of
/// `2·pop_nll` only — pop_nll already contains ηᵀΩ⁻¹η + log|Ω| per subject.
#[test]
fn test_covariance_gradient_focei_matches_fd_2pop_nll() {
    let model = make_model();
    let template = &model.default_params;
    let population = make_population(3);
    let n_subj = 3;
    let n_obs = 3;
    let n_eta = 1;

    let x = pack_params(template);
    let bounds = compute_bounds(template);
    let n = x.len();
    let mut options = FitOptions::default();
    options.interaction = true; // FOCEI: omega prior inside pop_nll

    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
        .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

    // FOCEI ofv_fixed = 2·pop_nll (omega prior already inside)
    let ofv_fixed_focei = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, template);
        2.0 * pop_nll(
            &model,
            &population,
            &p,
            &eta_hats,
            &h_matrices,
            &kappas,
            true, // FOCEI
        )
    };

    let grad = covariance_gradient(
        &x,
        template,
        &model,
        &population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &bounds,
        &options,
    );

    let eps = 1e-4;
    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += h;
        xm[j] -= h;
        let fd = (ofv_fixed_focei(&xp) - ofv_fixed_focei(&xm)) / (2.0 * h);
        let tol = 1e-3 * (1.0 + fd.abs());
        assert!(
            (grad[j] - fd).abs() < tol,
            "covariance_gradient FOCEI [{j}]: grad={:.6e}, FD_ofv={:.6e}",
            grad[j],
            fd,
        );
    }
}

/// End-to-end guard for `compute_covariance` (#209 factor-of-2, #256 flatten,
/// #274 Δ correction): the reconverging point-flatten gradient-FD covariance
/// must (a) compute without regularization on a well-conditioned surface,
/// (b) be positive-definite, and (c) equal `2·H⁻¹` of an *independently*
/// reconverged scalar-FD Hessian of the same FOCEI objective. Because the
/// model has proportional error, the reference (a second difference of the
/// reconverged OFV) carries the `log|H̃|` EBE-response curvature `Δ`; the
/// gradient-FD path only matches it because the #274 correction adds `Δ` back —
/// so this also guards the Δ correction. A missing factor of two would be ~29%
/// off (caught by the 15% band); a broken reconvergence would diverge wildly.
#[test]
fn test_compute_covariance_reconverged_matches_scalar_fd_with_factor_two() {
    let model = make_model();
    // Put the model at a near-optimum: set observations to the η=0
    // predictions of the 1-cpt IV model (CL=5, V=50, dose=100):
    // conc(t) = (100/50)·exp(−(5/50)·t) at t = 1, 4, 8.
    let mut population = make_population(8);
    for s in &mut population.subjects {
        s.observations = vec![1.80967, 1.34064, 0.89866];
    }
    // Fix ω and σ so the free block is the θ Hessian, which is positive
    // definite at this near-optimum (ω/σ would otherwise be pulled toward
    // their boundaries by the noise-free residuals, an artefact of the
    // synthetic data, not of the covariance code).
    let mut template = model.default_params.clone();
    template.omega_fixed = vec![true];
    template.sigma_fixed = vec![true];
    let template = &template;

    let n_subj = 8;
    let n_eta = 1;
    let n_obs = 3;
    let x = pack_params(template);
    let n = x.len();
    let mut options = FitOptions::default();
    options.interaction = true;

    // Warm-start EBEs (compute_covariance reconverges from these; the passed
    // h_matrices are intentionally ignored and recomputed).
    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    let h_matrices: Vec<DMatrix<f64>> = (0..n_subj)
        .map(|_| DMatrix::from_element(n_obs, n_eta, 0.1))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

    let out = match compute_covariance(
        &x,
        template,
        &model,
        &population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &options,
    ) {
        CovarianceStepResult::Success(out) => out,
        CovarianceStepResult::Unusable(msg) => {
            panic!("covariance must compute on the synthetic 1-cpt model: {msg}")
        }
        CovarianceStepResult::FailedNonPd { reason, .. } => {
            panic!("covariance must be PD on synthetic 1-cpt model: {reason}")
        }
    };

    // (a) No eigenvalue clipping on this well-conditioned surface.
    assert!(
        out.warnings.is_empty(),
        "unexpected covariance regularization: {:?}",
        out.warnings
    );

    let fixed = packed_fixed_mask(template);
    let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();

    // (b) Positive-definite: every free diagonal is positive and finite.
    for &i in &free_idx {
        let v = out.matrix[(i, i)];
        assert!(
            v.is_finite() && v > 0.0,
            "covariance diagonal [{i}] = {v} is not positive-finite"
        );
    }

    // (c) Independent reference: 2·inv(reconverged scalar-FD Hessian).
    // Mirrors the production `ofv` closure (interaction=true → 2·pop_nll,
    // reconverging EBEs from the same warm start).
    let ofv = |xv: &[f64]| -> f64 {
        let params = unpack_params(xv, template);
        let mu_k = compute_mu_k(&model, &params.theta, options.mu_referencing);
        let (ehs, hms, _s, kaps) = run_inner_loop_warm(
            &model,
            &population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&eta_hats),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );
        2.0 * pop_nll(&model, &population, &params, &ehs, &hms, &kaps, true)
    };

    let eps = 1e-2;
    let f0 = ofv(&x);
    let nf = free_idx.len();
    let mut h = DMatrix::zeros(nf, nf);
    let mut x_ij = x.clone();
    for (a, &i) in free_idx.iter().enumerate() {
        let hi = eps * (1.0 + x[i].abs());
        x_ij[i] = x[i] + hi;
        let fp = ofv(&x_ij);
        x_ij[i] = x[i] - hi;
        let fm = ofv(&x_ij);
        x_ij[i] = x[i];
        h[(a, a)] = (fp - 2.0 * f0 + fm) / (hi * hi);
        for (b, &j) in free_idx.iter().enumerate() {
            if j <= i {
                continue;
            }
            let hj = eps * (1.0 + x[j].abs());
            x_ij[i] = x[i] + hi;
            x_ij[j] = x[j] + hj;
            let fpp = ofv(&x_ij);
            x_ij[j] = x[j] - hj;
            let fpm = ofv(&x_ij);
            x_ij[i] = x[i] - hi;
            let fmm = ofv(&x_ij);
            x_ij[j] = x[j] + hj;
            let fmp = ofv(&x_ij);
            x_ij[i] = x[i];
            x_ij[j] = x[j];
            let v = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            h[(a, b)] = v;
            h[(b, a)] = v;
        }
    }
    let h_sym = (&h + h.transpose()) * 0.5;
    let ref_cov = invert_psd_with_floor(&h_sym)
        .expect("reference Hessian inverts")
        .inverse
        * 2.0;

    // SE (sqrt of diagonal) must agree within 15%: catches a missing factor
    // of two (~29%) and any reconvergence/scale break, while tolerating the
    // gradient-FD-vs-scalar-FD truncation difference at eps=1e-2.
    for (a, &i) in free_idx.iter().enumerate() {
        let se_prod = out.matrix[(i, i)].sqrt();
        let se_ref = ref_cov[(a, a)].sqrt();
        let rel = (se_prod - se_ref).abs() / se_ref;
        assert!(
                rel < 0.15,
                "SE[{i}]: compute_covariance {se_prod:.6e} vs scalar-FD reference {se_ref:.6e} (rel {:.1}%)",
                rel * 100.0
            );
    }
}

/// Coverage + smoke guard for the **IOV** covariance branch (#256 flatten +
/// the #298 perturbation-spec memory rewrite): an IOV model routes through
/// the scalar-`OFV`-2nd-difference `serial_ofv` stencil — subjects
/// reconverged via the shared `reconverge_point`, points built from the
/// lightweight `Pert` specs rather than materialised x-vectors. ω/κ/σ are
/// fixed so the free block is the θ Hessian (positive-definite at the
/// near-optimum where observations equal the η=κ=0 predictions); the test
/// asserts the branch runs and returns positive-finite θ SEs.
#[test]
fn test_compute_covariance_iov_runs_and_is_pd() {
    // 1-cpt IV, CL = θ₀·exp(η); IOV κ on CL. Predictions at η=κ=0:
    // conc(t) = (100/50)·exp(−(5/50)·t) = 2·exp(−0.1·t).
    let preds: Vec<f64> = (1..=6).map(|t| 2.0 * (-0.1 * t as f64).exp()).collect();

    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
    let default_params = ModelParameters {
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.1, 5.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![true], // fix ω/κ/σ → free block is the θ Hessian
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![true],
        omega_iov: Some(omega_iov),
        kappa_fixed: vec![true],
        mixture: None,
    };
    let model = CompiledModel {
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
        name: "iov_cov_test".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
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
        default_params,
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
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
    };

    let n_subj = 6;
    let subjects = (0..n_subj)
        .map(|_| Subject {
            fremtype: Vec::new(),
            id: "S".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            obs_raw_times: Vec::new(),
            observations: preds.clone(),
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
            dose_occasions: vec![1],
            obs_records: vec![],
        })
        .collect();
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let template = &model.default_params;
    let x = pack_params(template);
    let n = x.len();
    let mut options = FitOptions::default();
    options.interaction = true;

    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(1)).collect();
    let h_matrices: Vec<DMatrix<f64>> = (0..n_subj)
        .map(|_| DMatrix::from_element(6, 1, 0.1))
        .collect();
    // Non-empty per-occasion kappas → is_iov = true → exercises the IOV
    // scalar-FD stencil (serial_ofv + Pert specs + reconverge_point kaps).
    let kappas: Vec<Vec<DVector<f64>>> = (0..n_subj)
        .map(|_| vec![DVector::zeros(1), DVector::zeros(1)])
        .collect();

    let out = match compute_covariance(
        &x,
        template,
        &model,
        &population,
        &eta_hats,
        &h_matrices,
        &kappas,
        &options,
    ) {
        CovarianceStepResult::Success(out) => out,
        CovarianceStepResult::Unusable(msg) => panic!("IOV covariance unusable: {msg}"),
        CovarianceStepResult::FailedNonPd { reason, .. } => {
            panic!("IOV covariance not PD: {reason}")
        }
    };

    let fixed = packed_fixed_mask(template);
    let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();
    assert!(!free_idx.is_empty(), "θ block must be free");
    for &i in &free_idx {
        let v = out.matrix[(i, i)];
        assert!(
            v.is_finite() && v > 0.0,
            "IOV covariance diagonal [{i}] = {v} is not positive-finite"
        );
    }
}

// ── SLSQP overshoot guard tests (issue #55) ────────────────────────────
//
// NLopt LD_SLSQP starts every fit with its quasi-Newton Hessian set to
// identity; the QP's unconstrained first step is therefore d = -∇f. The
// AD/analytical FOCE gradient introduced in PR #48 has inf-norm ≈ 10²–10³
// on standard PK models, while the scaled bound width is ≈ 3–9, so the
// projected step lands at a corner of the box and the OFV explodes. The
// `cap_scaled_gradient` helper rescales `g` by a single scalar so the
// would-be Newton step fits inside the box on every dimension.

/// Cap fires when the gradient inf-norm exceeds the per-dimension
/// step budget, and the cap is a uniform rescale (preserves direction
/// and relative magnitudes between components).
#[test]
fn test_cap_scaled_gradient_uniformly_rescales_when_huge() {
    // Bounds chosen so each dimension's budget = clamp(half-width, 0.1, 1.0).
    //   i=0: width=2.0 → budget = clamp(1.0, …) = 1.0
    //   i=1: width=4.0 → budget = clamp(2.0, …) = 1.0 (clamped to 1.0)
    //   i=2: width=0.2 → budget = clamp(0.1, …) = 0.1 (clamped to 0.1)
    let lower = vec![-1.0, -2.0, -0.1];
    let upper = vec![1.0, 2.0, 0.1];

    // Gradient with inf-norm 200 at the third component → worst_ratio = 200/0.1 = 2000.
    let mut g = vec![10.0, 100.0, 200.0];
    let g_before = g.clone();
    let fired = cap_scaled_gradient(&mut g, &lower, &upper);
    assert!(fired, "cap should have fired for huge gradient");

    // Direction preserved: g[i] / g_before[i] is the same scalar across i.
    let scalar0 = g[0] / g_before[0];
    let scalar1 = g[1] / g_before[1];
    let scalar2 = g[2] / g_before[2];
    assert!(
        (scalar0 - scalar1).abs() < 1e-12 && (scalar1 - scalar2).abs() < 1e-12,
        "cap should be a uniform rescale: scalars {scalar0}, {scalar1}, {scalar2}",
    );

    // Inf-norm relative to per-dim budget should be exactly 1.0 after capping
    // (the dimension that drove the rescale is now at its budget).
    let after_inf_ratio = (g[0].abs() / 1.0)
        .max(g[1].abs() / 1.0)
        .max(g[2].abs() / 0.1);
    assert!(
        (after_inf_ratio - 1.0).abs() < 1e-12,
        "post-cap inf-norm ratio should equal 1.0, got {after_inf_ratio}",
    );
}

/// Cap is a no-op when the gradient is already within budget — preserves
/// SLSQP convergence behaviour once it's in the basin of the optimum.
#[test]
fn test_cap_scaled_gradient_noop_when_within_budget() {
    let lower = vec![-1.0, -2.0];
    let upper = vec![1.0, 2.0];
    // Per-dim budgets are both clamped to 1.0; gradient inf-norm = 0.5 < 1.0.
    let mut g = vec![0.5, -0.3];
    let g_before = g.clone();
    let fired = cap_scaled_gradient(&mut g, &lower, &upper);
    assert!(!fired, "cap should not fire for in-budget gradient");
    assert_eq!(g, g_before, "in-budget gradient must be untouched");
}

/// Even when one dimension has very wide bounds (a typical pattern in
/// log-Cholesky omega/sigma packing, where bounds span 10+ units), the
/// budget is clamped to 1.0 so the cap still fires.
#[test]
fn test_cap_scaled_gradient_clamps_wide_bounds_to_unit_budget() {
    // Wide bounds: half-width = 5 → budget clamped to 1.0.
    let lower = vec![-10.0, -10.0];
    let upper = vec![10.0, 10.0];
    let mut g = vec![5.0, 0.0];
    let fired = cap_scaled_gradient(&mut g, &lower, &upper);
    assert!(fired, "cap should fire: budget clamped to 1.0, |g_max| = 5");
    // Worst ratio = 5/1 = 5 → divide all by 5 → g[0] becomes 1.0.
    assert!(
        (g[0] - 1.0).abs() < 1e-12,
        "g[0] post-cap should be 1.0, got {}",
        g[0]
    );
    assert_eq!(g[1], 0.0);
}

/// [`should_cap_gradient`] gates the overshoot cap per-algorithm (#960, #751):
///   - SLSQP: cap every eval (QP re-solves from the current Hessian).
///   - L-BFGS: cap the first gradient eval always, and later evals only on the
///     stall-retry pass (`hold_cap_at_init`), which `optimize_nlopt` runs when
///     the default attempt never left its initial estimates. Holding the cap on
///     by default would corrupt the `(s, y)` curvature pairs L-BFGS builds from
///     gradient differences — the regression that blocked a uniform fix.
///   - MMA / BOBYQA: never capped here.
#[test]
fn test_should_cap_gradient_per_algo_gating() {
    use crate::estimation::outer_optimizer::should_cap_gradient;
    use nlopt::Algorithm;

    // SLSQP: every eval, including well past the first, moved or not.
    assert!(should_cap_gradient(Algorithm::Slsqp, 1, true));
    assert!(should_cap_gradient(Algorithm::Slsqp, 5, false));

    // L-BFGS: the first gradient eval is always capped. `n_grad_evals == 1` is
    // the first eval because `population_gradient` increments before returning.
    assert!(should_cap_gradient(Algorithm::Lbfgs, 1, false));
    assert!(should_cap_gradient(Algorithm::Lbfgs, 1, true));

    // Later evals: capped only on the retry pass. The default pass leaves every
    // later `(s, y)` pair built from uncapped gradients (#960); the retry keeps
    // the identity-Hessian step tame for a fit that already stalled (#751).
    assert!(!should_cap_gradient(Algorithm::Lbfgs, 2, false));
    assert!(!should_cap_gradient(Algorithm::Lbfgs, 50, false));
    assert!(should_cap_gradient(Algorithm::Lbfgs, 2, true));
    assert!(should_cap_gradient(Algorithm::Lbfgs, 50, true));

    // Derivative-free / self-safeguarding methods are never capped here.
    assert!(!should_cap_gradient(Algorithm::Mma, 1, true));
    assert!(!should_cap_gradient(Algorithm::Bobyqa, 1, true));
}

/// [`resolve_stall_retry`] decides whether a finished fit is re-run with the
/// identity-Hessian cap held on, and which of the two attempts is reported
/// (#751). Driven here with `T = f64` (the OFV itself) so every branch is
/// exercised without running two NLopt fits.
#[test]
fn test_stall_retry_only_fires_for_a_stalled_lbfgs_fit() {
    use crate::estimation::outer_optimizer::resolve_stall_retry;

    let ofv = |x: &f64| *x;
    // Retry must never run for a fit that left its initial estimates.
    let out = resolve_stall_retry(
        Optimizer::NloptLbfgs,
        false,
        (-286.0, true),
        || panic!("retry must not run for a fit that left init"),
        ofv,
    );
    assert_eq!(out, -286.0);

    // ...nor for the other optimizers: SLSQP is capped on every eval already,
    // and BOBYQA never takes an identity-Hessian step.
    for optimizer in [Optimizer::Slsqp, Optimizer::Bobyqa, Optimizer::Mma] {
        let out = resolve_stall_retry(
            optimizer,
            false,
            (-250.87, false),
            || panic!("retry is L-BFGS-only"),
            ofv,
        );
        assert_eq!(out, -250.87);
    }
}

#[test]
fn test_stall_retry_keeps_the_better_attempt() {
    use crate::estimation::outer_optimizer::resolve_stall_retry;

    let ofv = |x: &f64| *x;
    // The #751 case: the default attempt stalled at -250.87, the held-cap retry
    // escaped and reached the true optimum. Report the retry. `verbose = true`
    // also covers the reporting line.
    assert_eq!(
        resolve_stall_retry(
            Optimizer::NloptLbfgs,
            true,
            (-250.87, false),
            || (-286.0042, true),
            ofv
        ),
        -286.0042
    );

    // Retry escaped but landed *worse*: keep the first result.
    assert_eq!(
        resolve_stall_retry(
            Optimizer::NloptLbfgs,
            false,
            (-286.0042, false),
            || (-250.87, true),
            ofv
        ),
        -286.0042
    );

    // Retry stalled too, even at a nominally lower OFV: keep the first result,
    // so a genuinely stuck fit reports as stuck instead of as a second stall.
    assert_eq!(
        resolve_stall_retry(
            Optimizer::NloptLbfgs,
            false,
            (-250.87, false),
            || (-260.0, false),
            ofv
        ),
        -250.87
    );

    // Ties do not displace the first attempt.
    assert_eq!(
        resolve_stall_retry(
            Optimizer::NloptLbfgs,
            false,
            (-250.87, false),
            || (-250.87, true),
            ofv
        ),
        -250.87
    );
}

/// [`max_scaled_deviation`] is the L∞ "how far has the fit moved?" measure both
/// the cap gate and the plateau verdict key off.
#[test]
fn test_max_scaled_deviation_is_l_infinity() {
    use crate::estimation::outer_optimizer::max_scaled_deviation;

    assert_eq!(max_scaled_deviation(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    // Largest single-coordinate move wins, sign-independent.
    assert_eq!(max_scaled_deviation(&[1.0, 2.0], &[1.5, -3.0]), 5.0);
    // The stalled user-ODE fit's displacement (~1e-4) is below the escape step;
    // a real first step (O(0.1)) is above it.
    use crate::estimation::outer_optimizer::INIT_ESCAPE_STEP_S;
    assert!(max_scaled_deviation(&[1.0], &[1.000_14]) < INIT_ESCAPE_STEP_S);
    assert!(max_scaled_deviation(&[1.0], &[1.2]) >= INIT_ESCAPE_STEP_S);
}

/// Regression test for the original issue #55 symptom: SLSQP optimizing
/// a multi-theta mu-referenced FOCEI fit terminated with theta byte-
/// identical to init. The cap doesn't restore SLSQP to LBFGS's optimum
/// (the QP is still less aggressive than a line-search method on this
/// objective), but it does guarantee meaningful movement and a real OFV
/// improvement — the failure mode of "looks converged, didn't run".
///
/// Gated under `slow-tests` because it calls fit() to convergence.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn test_slsqp_moves_on_mu_referenced_two_cpt_oral_cov() {
    use crate::api::fit_from_files;
    use crate::types::{EstimationMethod, FitOptions, Optimizer};

    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        optimizer: Optimizer::Slsqp,
        outer_maxiter: 200,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let model_path = "examples/two_cpt_oral_cov.ferx";
    let data_path = "data/two_cpt_oral_cov.csv";
    let result =
        fit_from_files(model_path, Some(data_path), None, Some(opts)).expect("fit should succeed");

    // Initial theta from the .ferx file: [4.0, 40.0, 8.0, 80.0, 1.0, 0.6, 0.3].
    let init = [4.0, 40.0, 8.0, 80.0, 1.0, 0.6, 0.3];
    let max_rel_delta = result
        .theta
        .iter()
        .zip(init.iter())
        .map(|(t, i)| ((t - i) / i).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_rel_delta > 0.01,
        "SLSQP didn't move (max relative theta change = {:.4e}); \
             this is the issue #55 byte-identical-theta regression.\n\
             theta = {:?}\ninit  = {:?}",
        max_rel_delta,
        result.theta,
        init,
    );

    // OFV at init on this model + data is around -1040; LBFGS finds
    // ≈ -1198. SLSQP-with-cap reaches ≈ -1182. Assert at least a
    // 100-unit OFV improvement so we catch silent regressions where
    // SLSQP only moves by a hair.
    assert!(
        result.ofv < -1140.0,
        "SLSQP OFV = {:.2} is too close to init (-1040); cap may be \
             overly aggressive and throttling convergence.",
        result.ofv,
    );
}

// ── built-in BFGS optimizer (non-NLopt path) ─────────────────────────────

/// Drives the `Optimizer::Bfgs` branch of `optimize_population` for a few
/// outer iterations. Exercises `optimize_bfgs`, `bfgs_update`, and
/// `backtracking_line_search_warm` end-to-end on a tiny 1-cpt IV problem
/// without running to convergence (fast, Tier-1).
#[test]
fn bfgs_optimizer_runs_and_improves_ofv() {
    use crate::types::{EstimationMethod, FitOptions, Optimizer};
    let model = make_model();
    let population = make_population(4);
    let opts = FitOptions {
        method: EstimationMethod::Foce,
        optimizer: Optimizer::Bfgs,
        outer_maxiter: 5,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };

    // OFV at the initial point, for an improvement comparison.
    let init_ofv = {
        let init = optimize_population(
            &model,
            &population,
            &model.default_params,
            &FitOptions {
                optimizer: Optimizer::Bfgs,
                outer_maxiter: 0,
                run_covariance_step: false,
                ..opts.clone()
            },
        );
        init.ofv
    };

    let result = optimize_population(&model, &population, &model.default_params, &opts);
    assert!(result.ofv.is_finite(), "BFGS produced non-finite OFV");
    assert_eq!(result.eta_hats.len(), population.subjects.len());
    // Built-in BFGS does not export a final gradient (NLopt-only field).
    assert!(result.final_gradient.is_none());
    // A handful of iterations should not make the OFV worse.
    assert!(
        result.ofv <= init_ofv + 1e-6,
        "BFGS worsened OFV: init={init_ofv:.4} final={:.4}",
        result.ofv
    );
}

/// Built-in BFGS with the optimizer trace active must emit the per-parameter
/// `val:*` / `grad:*` columns (#640) via the second `write_foce` call site,
/// and the gradient columns must reconstruct the `grad_norm` column.
#[test]
fn bfgs_trace_emits_per_param_columns() {
    use crate::types::{EstimationMethod, FitOptions, Optimizer};
    let model = make_model();
    let population = make_population(4);
    let coord_names = crate::estimation::parameterization::coordinate_names(&model.default_params);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = format!(
        "/tmp/ferx_trace_bfgs_param_{}_{}.csv",
        std::process::id(),
        nanos
    );
    crate::estimation::trace::init(path.clone(), &coord_names).unwrap();
    let opts = FitOptions {
        method: EstimationMethod::Foce,
        optimizer: Optimizer::Bfgs,
        outer_maxiter: 3,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let _ = optimize_population(&model, &population, &model.default_params, &opts);
    crate::estimation::trace::finish();

    let contents = std::fs::read_to_string(&path).unwrap();
    let mut lines = contents.lines();
    let header = lines.next().unwrap();
    assert!(
        header.contains("val:"),
        "header missing val columns: {header}"
    );
    assert!(header.contains("grad:"), "header missing grad columns");
    let n_coords = coord_names.len();
    let fixed = 17;
    // Find a data row with a finite grad_norm and check the invariant.
    let cols_of = |l: &str| l.split(',').map(String::from).collect::<Vec<_>>();
    let hdr = cols_of(header);
    let gn_idx = hdr.iter().position(|c| c == "grad_norm").unwrap();
    let mut checked = false;
    for row in lines {
        let c = cols_of(row);
        assert_eq!(c.len(), fixed + 2 * n_coords, "row column count");
        if c[gn_idx] == "NA" {
            continue;
        }
        let gn: f64 = c[gn_idx].parse().unwrap();
        let mut sq = 0.0;
        for i in (fixed + n_coords)..(fixed + 2 * n_coords) {
            if let Ok(v) = c[i].parse::<f64>() {
                sq += v * v;
            }
        }
        assert!(
            (sq.sqrt() - gn).abs() <= 1e-6 * gn.abs().max(1.0),
            "grad cols must reconstruct grad_norm: recon={} grad_norm={}",
            sq.sqrt(),
            gn
        );
        checked = true;
        break;
    }
    assert!(checked, "expected at least one row with a finite grad_norm");
    std::fs::remove_file(&path).ok();
}

/// Gradient-mode NLopt (SLSQP) with the trace active must emit the
/// per-parameter columns via the *nlopt* `write_foce` call site — the branch
/// that snapshots `grad_vec_for_trace` (now guarded by `is_active`). The
/// built-in-BFGS test above covers a different site; SLSQP is otherwise only
/// reached by slow fits, so this registers PR coverage for the nlopt path.
#[test]
fn nlopt_gradient_trace_emits_per_param_columns() {
    use crate::types::{EstimationMethod, FitOptions, Optimizer};
    let model = make_model();
    let population = make_population(4);
    let coord_names = crate::estimation::parameterization::coordinate_names(&model.default_params);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = format!(
        "/tmp/ferx_trace_slsqp_param_{}_{}.csv",
        std::process::id(),
        nanos
    );
    crate::estimation::trace::init(path.clone(), &coord_names).unwrap();
    let opts = FitOptions {
        method: EstimationMethod::Foce,
        optimizer: Optimizer::Slsqp,
        outer_maxiter: 3,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let _ = optimize_population(&model, &population, &model.default_params, &opts);
    crate::estimation::trace::finish();

    let contents = std::fs::read_to_string(&path).unwrap();
    let mut lines = contents.lines();
    let header: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
    assert!(header.iter().any(|c| c.starts_with("val:")));
    assert!(header.iter().any(|c| c.starts_with("grad:")));
    let n_coords = coord_names.len();
    let fixed = 17;
    let gn_idx = header.iter().position(|c| c == "grad_norm").unwrap();
    let mut checked = false;
    for row in lines {
        let c: Vec<String> = row.split(',').map(String::from).collect();
        assert_eq!(c.len(), fixed + 2 * n_coords, "row column count");
        if c[gn_idx] == "NA" {
            continue;
        }
        let gn: f64 = c[gn_idx].parse().unwrap();
        let mut sq = 0.0;
        for i in (fixed + n_coords)..(fixed + 2 * n_coords) {
            if let Ok(v) = c[i].parse::<f64>() {
                sq += v * v;
            }
        }
        assert!(
            (sq.sqrt() - gn).abs() <= 1e-6 * gn.abs().max(1.0),
            "grad cols must reconstruct grad_norm: recon={} grad_norm={}",
            sq.sqrt(),
            gn
        );
        checked = true;
        break;
    }
    assert!(checked, "expected at least one row with a finite grad_norm");
    std::fs::remove_file(&path).ok();
}

// ── global pre-search (CRS2-LM) ──────────────────────────────────────────

/// Exercises the `run_global_presearch` branch. CRS2-LM may be absent from
/// the linked NLopt build; in that case the pre-search returns `Err` and a
/// `global_search disabled` warning is recorded. Either way the run must
/// finish with a finite OFV — this test just ensures the branch is taken
/// and handled gracefully (small `global_maxeval` keeps it fast).
#[test]
fn global_presearch_branch_runs_or_warns() {
    use crate::types::{EstimationMethod, FitOptions, Optimizer};
    let model = make_model();
    let population = make_population(4);
    let opts = FitOptions {
        method: EstimationMethod::Foce,
        optimizer: Optimizer::Bobyqa,
        outer_maxiter: 3,
        global_search: true,
        global_maxeval: 8,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let result = optimize_population(&model, &population, &model.default_params, &opts);
    assert!(
        result.ofv.is_finite(),
        "presearch run produced non-finite OFV"
    );
    // #713: with `run_covariance_step = false` the covariance block never
    // runs, so its timer must report exactly 0.0, not a stray near-zero
    // `Instant::now()` reading.
    assert_eq!(result.covariance_wall_time_secs, 0.0);
}

// ── Covariance Hessian throughput benchmark (issue #209) ─────────────────
//
// Run with:  cargo test --lib --no-default-features --features ci \
//              bench_cov_hessian -- --ignored --nocapture

/// Measures wall time for the gradient-FD Hessian (new path, issue #209) vs
/// the legacy scalar-FD Hessian (reconstructed inline) on the same setup.
///
/// n_free = 4 (2 theta + 1 omega + 1 sigma).
/// Old: ~2·n_free² = 32 OFV evaluations.
/// New: n_free+1 = 5 gradient evaluations.
#[test]
#[ignore = "benchmark: run with -- --ignored --nocapture"]
fn bench_cov_hessian_throughput() {
    use std::time::Instant;

    let model = make_model();
    let template = &model.default_params;
    let population = make_population(30);
    let n_subj = 30;
    let n_obs = 3;
    let n_eta = 1;
    let x = pack_params(template);
    let bounds = compute_bounds(template);
    let n = x.len();
    let options = FitOptions::default();

    let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
    let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
        .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

    let fixed_mask = packed_fixed_mask(template);
    let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed_mask[i]).collect();
    let eps = 1e-2;

    // ── Scalar-FD Hessian (old path, reconstructed inline) ────────────
    let ofv_at = |xv: &[f64]| -> f64 {
        let p = unpack_params(xv, template);
        let foce = pop_nll(
            &model,
            &population,
            &p,
            &eta_hats,
            &h_matrices,
            &kappas,
            options.interaction,
        );
        let omega_inv = p.omega.matrix.clone().cholesky().unwrap().inverse();
        let n_e = p.omega.dim();
        let log_det = 2.0 * (0..n_e).map(|i| p.omega.chol[(i, i)].ln()).sum::<f64>();
        let om_terms: f64 = eta_hats
            .iter()
            .map(|eta| eta.dot(&(&omega_inv * eta)) + log_det)
            .sum();
        2.0 * foce + om_terms
    };
    let f0 = ofv_at(&x);

    const REPS: u32 = 20;
    let t0 = Instant::now();
    for _ in 0..REPS {
        let mut hess = DMatrix::zeros(n, n);
        let mut xij = x.clone();
        for &i in &free_idx {
            let hi = eps * (1.0 + x[i].abs());
            xij[i] = x[i] + hi;
            let fp = ofv_at(&xij);
            xij[i] = x[i] - hi;
            let fm = ofv_at(&xij);
            xij[i] = x[i];
            if ((fp - 2.0 * f0 + fm) / (hi * hi)).is_finite() {
                hess[(i, i)] = (fp - 2.0 * f0 + fm) / (hi * hi);
            }
            for &j in &free_idx {
                if j <= i {
                    continue;
                }
                let hj = eps * (1.0 + x[j].abs());
                xij[i] = x[i] + hi;
                xij[j] = x[j] + hj;
                let fpp = ofv_at(&xij);
                xij[j] = x[j] - hj;
                let fpm = ofv_at(&xij);
                xij[i] = x[i] - hi;
                let fmm = ofv_at(&xij);
                xij[j] = x[j] + hj;
                let fmp = ofv_at(&xij);
                xij[i] = x[i];
                xij[j] = x[j];
                let v = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                if v.is_finite() {
                    hess[(i, j)] = v;
                    hess[(j, i)] = v;
                }
            }
        }
        std::hint::black_box(hess);
    }
    let scalar_ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;

    // ── Gradient-FD Hessian (new path) ───────────────────────────────
    let t1 = Instant::now();
    for _ in 0..REPS {
        let mut hess = DMatrix::zeros(n, n);
        let g0 = covariance_gradient(
            &x,
            template,
            &model,
            &population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &bounds,
            &options,
        );
        for &k in &free_idx {
            let hk = eps * (1.0 + x[k].abs());
            let mut xp = x.clone();
            xp[k] += hk;
            let gp = covariance_gradient(
                &xp,
                template,
                &model,
                &population,
                &eta_hats,
                &h_matrices,
                &kappas,
                &bounds,
                &options,
            );
            for &j in &free_idx {
                let v = (gp[j] - g0[j]) / hk;
                if v.is_finite() {
                    hess[(j, k)] = v;
                }
            }
        }
        std::hint::black_box(hess);
    }
    let grad_ms = t1.elapsed().as_secs_f64() * 1000.0 / REPS as f64;

    println!(
        "\n── Covariance Hessian throughput (n_free={}, n_subj={}) ──────────",
        free_idx.len(),
        n_subj
    );
    println!("  scalar-FD (old): {:.2}ms/Hessian", scalar_ms);
    println!("  gradient-FD (new): {:.2}ms/Hessian", grad_ms);
    println!("  speedup: {:.1}×", scalar_ms / grad_ms);
}

// ── build_non_pd_fallback_proposal ───────────────────────────────────────

/// Diagonal 2×2 Hessian with one negative eigenvalue (-2) and one positive
/// (4). The proposal covariance should have eigenvalues inflation / |λ_i|,
/// inflated by factor 4: so 4/2 = 2.0 and 4/4 = 1.0.
#[test]
fn build_fallback_proposal_is_pd_and_inflated() {
    let hess = DMatrix::from_row_slice(2, 2, &[-2.0_f64, 0.0, 0.0, 4.0]);
    let free_idx = [0usize, 1];
    let proposal = build_non_pd_fallback_proposal(&hess, &free_idx, 2, 4.0);
    // Result must be symmetric PD.
    assert!(proposal[(0, 0)] > 0.0, "diagonal must be positive");
    assert!(proposal[(1, 1)] > 0.0, "diagonal must be positive");
    assert!(
        (proposal[(0, 1)] - proposal[(1, 0)]).abs() < 1e-12,
        "must be symmetric"
    );
    // Eigenvalues of the proposal should be inflation / |original eigenvalue|.
    let eig = SymmetricEigen::new(proposal.clone());
    let mut evs: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Expected: [4/4, 4/2] = [1.0, 2.0]
    assert!(
        (evs[0] - 1.0).abs() < 1e-10,
        "smaller eigenvalue should be 1.0: {:?}",
        evs
    );
    assert!(
        (evs[1] - 2.0).abs() < 1e-10,
        "larger eigenvalue should be 2.0: {:?}",
        evs
    );
}

/// Fixed parameter (index 2 absent from free_idx) stays zero in full matrix.
#[test]
fn build_fallback_proposal_zeros_fixed_params() {
    let hess = DMatrix::from_row_slice(1, 1, &[2.0_f64]);
    let free_idx = [0usize];
    let proposal = build_non_pd_fallback_proposal(&hess, &free_idx, 3, 4.0);
    assert_eq!(proposal.nrows(), 3);
    assert_eq!(proposal.ncols(), 3);
    assert!(
        proposal[(0, 0)] > 0.0,
        "free param row/col must be non-zero"
    );
    assert_eq!(proposal[(1, 1)], 0.0, "fixed param row/col must be zero");
    assert_eq!(proposal[(2, 2)], 0.0, "fixed param row/col must be zero");
}

/// A near-zero eigenvalue must be floored *relative* to the largest, capping
/// the proposal's condition number at `FALLBACK_PROPOSAL_MAX_COND`. With the
/// old absolute `1e-10` floor a 1e-9 eigenvalue would give a variance of
/// 4/1e-9 = 4e9 (and a 1e12 condition number) — far enough to scatter every
/// SIR draw out of bounds. The relative floor caps it at 4/(λ_max/1e8).
#[test]
fn build_fallback_proposal_caps_condition_number() {
    // diag(1000, 1e-9): one well-determined direction, one near-flat.
    let hess = DMatrix::from_row_slice(2, 2, &[1000.0_f64, 0.0, 0.0, 1e-9]);
    let free_idx = [0usize, 1];
    let proposal = build_non_pd_fallback_proposal(&hess, &free_idx, 2, 4.0);
    let eig = SymmetricEigen::new(proposal.clone());
    let max_var = eig.eigenvalues.iter().cloned().fold(f64::MIN, f64::max);
    let min_var = eig.eigenvalues.iter().cloned().fold(f64::MAX, f64::min);
    // floor = 1000 / 1e8 = 1e-5 ⇒ largest variance = 4 / 1e-5 = 4e5,
    // well below the un-floored 4e9.
    assert!(
        max_var < 1e6,
        "near-zero direction variance must be capped by the relative floor, got {max_var:e}"
    );
    // Condition number of the proposal must not exceed the cap (allow a
    // little slack for the inflation/eigen round-trip).
    assert!(
        max_var / min_var <= FALLBACK_PROPOSAL_MAX_COND * 1.01,
        "proposal condition number {} exceeds cap {}",
        max_var / min_var,
        FALLBACK_PROPOSAL_MAX_COND
    );
}

// ── select_fd_step ───────────────────────────────────────────────────────

/// When all stencils are finite from the start, no halvings occur and the
/// initial step is returned unchanged.
#[test]
fn select_fd_step_no_halving_needed() {
    let ofv = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let x_hat = [1.0f64, 2.0];
    let free_idx = [0usize, 1];
    let f0 = ofv(&x_hat);
    let (eps, halvings) = select_fd_step(&x_hat, &free_idx, 0.01, f0, &ofv);
    assert_eq!(eps, 0.01, "step should be unchanged");
    assert_eq!(halvings, 0, "no halvings expected");
}

/// When the initial step causes overflow (NaN stencils), the function halves
/// until stencils are finite and returns the reduced step.
#[test]
fn select_fd_step_halves_on_overflow() {
    // Returns NaN whenever |x[0]| >= 0.5 — simulates model overflow.
    let ofv = |x: &[f64]| {
        if x[0].abs() >= 0.5 {
            f64::NAN
        } else {
            x[0] * x[0]
        }
    };
    let x_hat = [0.0f64];
    let free_idx = [0usize];
    let f0 = 0.0f64;
    // initial_eps=1.0 → hi=1.0 → x=1.0 ≥ 0.5 → NaN → halve
    // eps=0.5 → hi=0.5 → x=0.5 ≥ 0.5 → NaN → halve
    // eps=0.25 → hi=0.25 → x=0.25 < 0.5 → 0.0625 → OK
    let (eps, halvings) = select_fd_step(&x_hat, &free_idx, 1.0, f0, &ofv);
    assert_eq!(eps, 0.25, "should have halved twice");
    assert_eq!(halvings, 2);
    // Verify the chosen step actually produces finite stencils.
    let hi = eps * (1.0 + x_hat[0].abs());
    let fp = ofv(&[x_hat[0] + hi]);
    let fm = ofv(&[x_hat[0] - hi]);
    assert!(fp.is_finite() && fm.is_finite());
}

/// Empty free_idx — vacuously all OK, returns initial eps without halvings.
#[test]
fn select_fd_step_empty_free_idx() {
    let ofv = |_x: &[f64]| f64::NAN; // would fail any real stencil
    let (eps, halvings) = select_fd_step(&[1.0], &[], 0.01, 0.0, &ofv);
    assert_eq!(eps, 0.01);
    assert_eq!(halvings, 0);
}

/// Regression: a stencil whose *numerator* (fp − 2·f0 + fm) is finite but
/// whose *quotient* (÷ hi²) overflows must not be accepted at halvings == 0.
/// The old numerator-only check declared this step usable, then the FD loop —
/// which divides — rejected the diagonal and the covariance step failed
/// without ever halving. select_fd_step now applies the same quotient the FD
/// loop does, so it recognises the step as unusable (and exhausts its
/// halvings rather than falsely reporting success on the first try).
#[test]
fn select_fd_step_rejects_finite_numerator_infinite_quotient() {
    // f0 = 0; any non-zero perturbation returns 1e200, so the numerator is a
    // finite 2e200 but hi² is ~1e-200, making the quotient overflow to +inf.
    let ofv = |x: &[f64]| if x[0] == 0.0 { 0.0 } else { 1e200 };
    let x_hat = [0.0f64];
    let free_idx = [0usize];
    let f0 = 0.0f64;
    let initial_eps = 1e-100;
    // Numerator is finite (the old check would accept immediately) …
    let hi = initial_eps * (1.0 + x_hat[0].abs());
    let numerator = ofv(&[hi]) - 2.0 * f0 + ofv(&[-hi]);
    assert!(
        numerator.is_finite(),
        "test setup: numerator must be finite"
    );
    assert!(
        !(numerator / (hi * hi)).is_finite(),
        "test setup: quotient must overflow"
    );
    // … but the quotient overflows, so the step is not accepted on the first
    // pass: halvings > 0 (smaller steps can't rescue this pathological case,
    // so it exhausts the budget — the point is it did not return 0).
    let (_eps, halvings) = select_fd_step(&x_hat, &free_idx, initial_eps, f0, &ofv);
    assert!(
        halvings > 0,
        "finite-numerator/infinite-quotient step must not be accepted at halvings == 0"
    );
}

/// `outer_maxiter == 0` is evaluation-only (NONMEM `MAXEVAL=0`): every
/// optimizer must report the *initial*-point objective with zero iterations,
/// and the value must be identical across optimizers. Regression for #562:
/// the gradient NLopt path set `set_maxeval(0)` — which NLopt reads as "no
/// limit" — so `NloptLbfgs`/`Slsqp`/`Mma` silently ran a full fit on a
/// `maxiter = 0` request, making the reported "init" OFV optimizer- and
/// platform-dependent.
#[test]
fn outer_maxiter_zero_is_eval_only_and_optimizer_independent() {
    let model = make_model();
    let population = make_population(3);
    let template = &model.default_params;

    // Reference init objective, computed through the same pack/unpack/clamp
    // round-trip the optimizer entry uses, so the comparison is exact.
    let mut x = pack_params(template);
    clamp_to_bounds(&mut x, &compute_bounds(template));
    let init_params = unpack_params(&x, template);
    let mu_k = compute_mu_k(&model, &init_params.theta, true);
    let cold: Vec<DVector<f64>> = (0..population.subjects.len())
        .map(|_| DVector::zeros(model.n_eta))
        .collect();
    let (ehs, hms, _, kappas) = run_inner_loop_warm(
        &model,
        &population,
        &init_params,
        200,
        1e-6,
        Some(&cold),
        Some(&mu_k),
        0,
        0,
    );
    let init_ofv = 2.0 * pop_nll(&model, &population, &init_params, &ehs, &hms, &kappas, true);
    assert!(init_ofv.is_finite(), "test setup: init OFV must be finite");

    let base = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        outer_maxiter: 0,
        run_covariance_step: false,
        mu_referencing: true,
        ..FitOptions::default()
    };

    // The buggy path was `NloptLbfgs`; check it alongside the others (incl.
    // `Auto`, which resolves per-model) — all must agree and not iterate.
    for opt in [
        Optimizer::NloptLbfgs,
        Optimizer::Slsqp,
        Optimizer::Bobyqa,
        Optimizer::Bfgs,
        Optimizer::Auto,
    ] {
        let o = FitOptions {
            optimizer: opt,
            ..base.clone()
        };
        let r = optimize_population(&model, &population, template, &o);
        assert_eq!(r.n_iterations, 0, "maxiter=0 must not iterate ({opt:?})");
        assert!(
            !r.converged,
            "maxiter=0 must not claim convergence ({opt:?})"
        );
        assert!(
            (r.ofv - init_ofv).abs() < 1e-9,
            "maxiter=0 OFV must equal the init objective for {opt:?}: got {} vs {init_ofv}",
            r.ofv
        );
    }

    // Counter-check: given a real budget, `NloptLbfgs` actually moves — so the
    // eval-only result above is a genuine skip, not a coincidence.
    let run = FitOptions {
        optimizer: Optimizer::NloptLbfgs,
        outer_maxiter: 50,
        ..base.clone()
    };
    let r = optimize_population(&model, &population, template, &run);
    assert!(
        r.ofv < init_ofv - 1e-6,
        "with a budget the optimizer should improve the OFV: {} vs init {init_ofv}",
        r.ofv
    );
}

// The end-to-end #960 convergence fix (analytic-gradient L-BFGS leaving init on
// warfarin FOCEI / the SS-oral fit) is guarded by the re-enabled slow tests
// (`warfarin_covariance_nonmem`, `ss_fit_smoke`, `covariance_method_sandwich`).
// It is deliberately *not* reproduced as a fast unit test: the synthetic
// `make_model()` is structurally FD-only (`indiv_param_partials::empty()` ⇒
// `analytic_outer_gradient_available` is false even under `GradientMethod::Auto`),
// and its scaled first-step gradient never overshoots, so the cap does not change
// its outcome — a unit "convergence" test here would pass with or without the fix
// and guard nothing. The cap's *mechanism* is covered fast instead by
// `should_cap_gradient` (per-algorithm gating) and the `cap_scaled_gradient_*`
// rescale tests above; the L-BFGS call site is exercised by the non-SLSQP fit in
// `non_convergence_reports_directly_without_second_optimization` below.

/// A non-SLSQP gradient primary that stops **non-converged** runs exactly one
/// outer optimization — there is no automatic SLSQP retry from the stop point
/// (issue #657, which removed `should_run_slsqp_fallback` and the second
/// `nlopt::Nlopt` run). Two invariants, both on the non-converged shape the
/// pre-#657 fallback actually fired on:
///   1. Non-convergence surfaces the "did not converge" warning directly.
///   2. Only one optimization runs; a re-added fallback would launch a *second*
///      full nlopt run from the endpoint, ~doubling the eval count.
///
/// Non-convergence is forced fix-independently: `inner_maxiter = 1` with
/// `max_unconverged_frac = 0.0` leaves at least one subject's EBEs unconverged
/// every eval, so the EBE guard rejects every step, the fit never forms a flat
/// plateau, and it terminates at the `maxeval` ceiling reported non-converged.
/// (Pre-#960 the original of this test relied on the L-BFGS first-step stall to
/// be non-converged; that stall is now fixed, so the shape is driven by the
/// starved inner loop instead — otherwise the fix would have silently converted
/// this into a converged run and dropped the coverage.)
#[test]
fn non_convergence_reports_directly_without_second_optimization() {
    let model = make_model();
    let population = make_population(3);
    let template = &model.default_params;
    let n = pack_params(template).len();

    let outer_maxiter = 8;
    let o = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        optimizer: Optimizer::NloptLbfgs,
        outer_maxiter,
        // Starve the inner loop so the EBE guard rejects every outer step and the
        // fit cannot converge (independent of the #960 first-step cap).
        inner_maxiter: 1,
        max_unconverged_frac: 0.0,
        run_covariance_step: false,
        mu_referencing: true,
        ..FitOptions::default()
    };
    let r = optimize_population(&model, &population, template, &o);

    assert!(
        !r.converged,
        "test setup: a starved-inner-loop fit must not converge (OFV = {})",
        r.ofv
    );
    assert!(
        r.warnings.iter().any(|w| w.contains("did not converge")),
        "non-convergence must surface the warning directly; got {:?}",
        r.warnings
    );
    // One optimization only. NLopt LD_LBFGS checks `maxeval` only between line
    // searches, so a single run slightly overruns `outer_maxiter * (n + 1)` (≈45
    // on this model, budget 40); a re-added fallback would add a *second* full
    // nlopt run from the endpoint, pushing the total toward ~2× that. Bound at
    // `2 * budget` cleanly separates the single run (~45) from a doubling (~90).
    let single_budget = outer_maxiter as usize * (n + 1);
    assert!(
        r.n_iterations < 2 * single_budget,
        "expected a single optimization (< {} evals), got {} — a second \
         optimization (SLSQP fallback) appears to have run",
        2 * single_budget,
        r.n_iterations
    );
}

// ── #751: reclassifying a bare NLopt `Failure` at a plateaued optimum ──────────
//
// `failure_is_converged_plateau` is the pure decision behind treating a generic
// `NLOPT_FAILURE`/`FORCED_STOP` (returned by the analytic-gradient L-BFGS default
// once its line search can no longer beat an already-flat OFV) as convergence.
// It must accept a genuine plateau and reject a real early stall.

// Signature: (feasible_evals, last_sig_feasible_eval, best_seen, final,
// left_init).
// Every eval count/index is over *feasible* (unguarded) evals only, 1-based;
// `last_sig_feasible_eval == 1` means the last significant improvement was the
// first feasible eval (the baseline), i.e. the fit never descended.

#[test]
fn plateau_with_flat_tail_and_consistent_ofv_is_converged() {
    // Long flat tail (npde/schnider shape): last significant improvement was many
    // feasible evals before termination, and the cold-restart final OFV
    // reproduces best-seen.
    assert!(failure_is_converged_plateau(
        44, // feasible evals
        36, // last significant improvement → flat tail of 8 (≥ 5)
        Some(-286.004247),
        -286.004205, // ties best-seen to ~4e-5
        true,        // best point is far from init
    ));
}

#[test]
fn short_descending_tail_is_not_converged() {
    // SS-oral shape: the fit quits after ~5 evals still plunging — the last
    // improvement is the final feasible eval, so there is no flat tail at all.
    assert!(!failure_is_converged_plateau(
        5,
        5,
        Some(83.26),
        83.26,
        true
    ));
    // Even one eval short of the minimum flat tail must stay unconverged.
    assert!(!failure_is_converged_plateau(
        10,
        10 - (PLATEAU_MIN_FLAT_EVALS - 1),
        Some(-100.0),
        -100.0,
        true,
    ));
}

#[test]
fn plateau_but_inconsistent_cold_restart_is_not_converged() {
    // SS-oral warm-start artifact: the OFV trace could look flat, yet the cold
    // inner-loop restart lands far worse (best-seen 83.3 vs final 121.4) — the
    // "optimum" was an EBE warm-start artifact, so it is rejected.
    assert!(!failure_is_converged_plateau(
        50,
        40,
        Some(83.26),
        121.36,
        true
    ));
}

#[test]
fn plateau_with_better_cold_restart_is_converged() {
    // A cold restart that ties or *improves* on best-seen is a valid minimum —
    // only the materially-worse direction signals an artifact.
    assert!(failure_is_converged_plateau(
        50,
        40,
        Some(-286.0),
        -286.5,
        true
    ));
}

#[test]
fn plateau_check_reaches_min_flat_tail_boundary() {
    // Exactly `PLATEAU_MIN_FLAT_EVALS` flat feasible evals is enough (inclusive).
    assert!(failure_is_converged_plateau(
        20,
        20 - PLATEAU_MIN_FLAT_EVALS,
        Some(1.0),
        1.0,
        true,
    ));
}

#[test]
fn plateau_pinned_at_init_is_not_converged() {
    // The user-ODE warfarin twin (#751): its line search died at feasible eval 4
    // having improved the OFV by 0.028 — enough to clear `PLATEAU_OFV_THRESHOLD`
    // and register as "progress" — then went flat for the rest of the budget.
    // Progress + plateau + consistency all hold, yet the fit never left its
    // initial estimates, so it must not be reported converged (it would publish
    // standard errors for the initial point).
    assert!(!failure_is_converged_plateau(
        15,
        4,
        Some(-250.866184),
        -250.866178,
        false, // never left init
    ));
    // Same trace, but the fit did move: that is a genuine plateau.
    assert!(failure_is_converged_plateau(
        15,
        4,
        Some(-250.866184),
        -250.866178,
        true,
    ));
}

#[test]
fn plateau_check_handles_missing_best_seen() {
    // No best-seen point recorded → consistency cannot fail; the plateau length
    // alone decides.
    assert!(failure_is_converged_plateau(30, 10, None, -50.0, true));
    assert!(!failure_is_converged_plateau(3, 3, None, -50.0, true));
}

#[test]
fn stuck_at_initial_estimate_is_not_converged() {
    // NLopt L-BFGS whose first step overshoots and whose line search fails
    // (warfarin FOCEI): the only significant improvement is the first feasible
    // eval registering OFV₀, so `last_sig_feasible_eval == 1`. The objective is
    // then flat for the remaining line-search probes (a long flat tail) and
    // self-consistent (the fit never left the initial point), but it never
    // descended — it must NOT be reported as converged.
    assert!(!failure_is_converged_plateau(
        12,
        1,
        Some(-250.838),
        -250.838,
        false
    ));
    // No feasible eval at all (`feasible_evals == 0`): not converged.
    assert!(!failure_is_converged_plateau(0, 0, None, -250.838, false));
}

#[test]
fn guard_rejected_first_eval_does_not_fake_progress() {
    // #751 regression: initial estimates marginally violate the EBE guard, so the
    // early evals are guard-penalised and never counted. The first *feasible*
    // point is feasible-eval 1 (the baseline) regardless of how many guarded evals
    // preceded it, so `last_sig_feasible_eval == 1`. A long flat tail and a
    // consistent cold restart follow, but the fit never descended past a real
    // objective. Counting over feasible evals keeps this `converged = false`; the
    // earlier total-eval basis let the first feasible point land at index ≥ 2 and
    // wrongly satisfy `>= 2`.
    assert!(!failure_is_converged_plateau(
        20,
        1,
        Some(83.26),
        83.26,
        true
    ));
}

#[test]
fn guarded_tail_does_not_pad_plateau() {
    // Copilot review: a tail of guard-rejected boundary probes must not inflate
    // the plateau length. Because the classifier counts feasible evals only, a run
    // with a real improvement at feasible-eval 3 and then only 1 further feasible
    // eval (feasible_evals = 4) has a flat tail of 1 — NOT converged — even if
    // dozens of guarded evals followed. The guarded tail is invisible here by
    // construction (it never advances `feasible_evals`).
    assert!(!failure_is_converged_plateau(
        4,
        3,
        Some(-100.0),
        -100.0,
        true
    ));
    // The same real improvement followed by ≥ 5 *feasible* flat evals does plateau.
    assert!(failure_is_converged_plateau(
        3 + PLATEAU_MIN_FLAT_EVALS,
        3,
        Some(-100.0),
        -100.0,
        true
    ));
}

#[test]
fn genuine_progress_after_guarded_start_is_converged() {
    // Guard-rejected early evals (invisible to the feasible counter), but the fit
    // then genuinely descends: a significant improvement lands at feasible-eval 5,
    // after the baseline, followed by a flat tail of 15. Real progress-then-plateau
    // — must be accepted.
    assert!(failure_is_converged_plateau(
        20,
        5,
        Some(-286.0),
        -286.0,
        true
    ));
}
