//! The covariance step's per-subject score (#2974): each row of `S = Σᵢ gᵢgᵢᵀ` is the
//! subject's **total** derivative `dnllᵢ/dx`, the mode re-estimated at every `x` — NONMEM's
//! `MATRIX=S` convention.
//!
//! The oracle is independent of the adjoint: a central difference of the subject's own
//! objective with its mode re-solved (warm-started) at every stencil point
//! (`subject_reconverged_fd_gradient[_iov]`), per model class. The fixed-mode score that
//! was the row before #2974 (`subject_nll_pop_grad`, the mode held at x̂) misses that oracle
//! by far more than the bound, which each test asserts too, so the bound discriminates.

use super::{assemble_score_cross_product, pack_with_bounds, subject_total_score, PackedStart};
use crate::estimation::inner_optimizer::find_ebe_for_stage;
use crate::estimation::outer_optimizer::{
    joint_warm_start, subject_reconverged_fd_gradient, subject_reconverged_fd_gradient_iov,
};
use crate::estimation::parameterization::compute_mu_k;
use crate::sens::ode_provider::adjoint::PackedPoint;
use crate::types::{CompiledModel, FitOptions, Population};
use nalgebra::{DMatrix, DVector};

type Modes = (Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<DVector<f64>>>);

/// Every subject's mode `(η̂, H, κ̂)` at the model's initial values, at `tol`.
fn modes(model: &CompiledModel, pop: &Population, opts: &FitOptions, tol: f64) -> Modes {
    let p = &model.default_params;
    let mu_k = compute_mu_k(model, &p.theta, opts.mu_referencing);
    let mut out: Modes = (Vec::new(), Vec::new(), Vec::new());
    for s in &pop.subjects {
        let ebe = find_ebe_for_stage(model, s, p, 1000, tol, None, Some(&mu_k), 0, opts);
        out.0.push(ebe.eta);
        out.1.push(ebe.h_matrix);
        out.2.push(ebe.kappas);
    }
    out
}

/// Every subject's score against the reconverged central difference, on every free
/// coordinate, as `|g − fd| / max(|fd|, 1)`; and the same for the fixed-mode score. Each bound
/// is about 10× the measured total-score error and at least 4× below the fixed-mode miss.
/// Returns `(worst total, worst fixed-mode)`.
fn score_vs_oracle(
    model: &CompiledModel,
    pop: &Population,
    opts: &FitOptions,
    label: &str,
) -> (f64, f64) {
    let template = &model.default_params;
    let PackedStart {
        packed: x,
        bounds,
        fixed,
        ..
    } = pack_with_bounds(template);
    let (etas, hs, kappas) = modes(model, pop, opts, 1e-12);
    let pt = PackedPoint::new(template, &x);
    let (mut worst, mut worst_held) = (0.0f64, 0.0f64);
    for i in 0..pop.subjects.len() {
        let subject = &pop.subjects[i];
        let g = subject_total_score(
            &x,
            template,
            model,
            pop,
            i,
            &etas[i],
            &kappas[i],
            pt.as_ref(),
            &bounds,
            opts,
        );
        let fd = if model.n_kappa > 0 {
            let warm = joint_warm_start(&etas[i], &kappas[i]);
            subject_reconverged_fd_gradient_iov(&x, template, model, subject, &warm, &bounds, opts)
        } else {
            subject_reconverged_fd_gradient(&x, template, model, subject, &etas[i], &bounds, opts)
        };
        let (_, held) = crate::estimation::gauss_newton::subject_nll_pop_grad(
            &x, template, model, pop, i, &etas[i], &hs[i], &kappas[i], &bounds, opts,
        );
        for j in (0..x.len()).filter(|&j| !fixed[j]) {
            assert!(
                g[j].is_finite() && fd[j].is_finite(),
                "{label}: subject {i} coord {j}: score {} / FD {}",
                g[j],
                fd[j]
            );
            let err = (g[j] - fd[j]).abs() / fd[j].abs().max(1.0);
            let err_held = (held[j] - fd[j]).abs() / fd[j].abs().max(1.0);
            worst = worst.max(err);
            worst_held = worst_held.max(err_held);
        }
    }
    eprintln!(
        "{label}: worst total score vs oracle {worst:.3e}; fixed-mode score {worst_held:.3e}"
    );
    (worst, worst_held)
}

/// `examples/warfarin.ferx` on its first `n_subj` subjects, FOCEI or FOCE.
fn warfarin(n_subj: usize, interaction: bool) -> (CompiledModel, Population, FitOptions) {
    let (model, pop, opts) = crate::estimation::inner_optimizer::test_helpers::load_iov_fixture(
        include_str!("../../examples/warfarin.ferx"),
        "data/warfarin.csv",
        n_subj,
    );
    let opts = FitOptions {
        interaction,
        method: if interaction {
            crate::types::EstimationMethod::FoceI
        } else {
            crate::types::EstimationMethod::Foce
        },
        inner_tol: 1e-10,
        ..opts
    };
    (model, pop, opts)
}

// Measured (Windows x86_64): total score vs oracle 1.2e-7, fixed-mode score 6.5e-1.
#[test]
fn focei_score_is_the_reconverged_total_derivative() {
    let (model, pop, opts) = warfarin(4, true);
    let (total, held) = score_vs_oracle(&model, &pop, &opts, "warfarin FOCEI");
    assert!(total < 1e-6, "FOCEI total score vs oracle: {total:.3e}");
    assert!(
        held > 1e-2,
        "straddle: the fixed-mode score must miss the oracle: {held:.3e}"
    );
}

// Measured (Windows x86_64): total score vs oracle 5.1e-7, fixed-mode score 4.4e-3.
#[test]
fn foce_score_is_the_reconverged_total_derivative() {
    let (model, pop, opts) = warfarin(4, false);
    let (total, held) = score_vs_oracle(&model, &pop, &opts, "warfarin FOCE");
    assert!(total < 5e-6, "FOCE total score vs oracle: {total:.3e}");
    assert!(
        held > 1e-3,
        "straddle: the fixed-mode score must miss the oracle: {held:.3e}"
    );
}

// Measured (Windows x86_64): total score vs oracle 4.6e-7, fixed-mode score 2.3e-1.
#[test]
fn iov_score_is_the_reconverged_total_derivative() {
    let (model, pop, opts) =
        crate::estimation::inner_optimizer::test_helpers::warfarin_iov_fixture();
    let opts = FitOptions {
        inner_tol: 1e-10,
        ..opts
    };
    let (total, held) = score_vs_oracle(&model, &pop, &opts, "warfarin IOV");
    assert!(total < 5e-6, "IOV total score vs oracle: {total:.3e}");
    assert!(
        held > 1e-2,
        "straddle: the fixed-mode score must miss the oracle: {held:.3e}"
    );
}

#[cfg(feature = "survival")]
// Measured (Windows x86_64): total score vs oracle 2.3e-8, fixed-mode score 1.1e-2.
#[test]
fn tte_score_is_the_reconverged_total_derivative() {
    let (model, pop, opts) = crate::estimation::inner_optimizer::test_helpers::load_iov_fixture(
        include_str!("../../examples/tte_exponential.ferx"),
        "data/tte_exponential.csv",
        6,
    );
    let opts = FitOptions {
        interaction: true,
        method: crate::types::EstimationMethod::FoceI,
        inner_tol: 1e-10,
        ..opts
    };
    let (total, held) = score_vs_oracle(&model, &pop, &opts, "TTE exponential");
    assert!(total < 1e-6, "TTE total score vs oracle: {total:.3e}");
    assert!(
        held > 1e-3,
        "straddle: the fixed-mode score must miss the oracle: {held:.3e}"
    );
}

/// The κ wiring (#1817): an IOV subject's row is the adjoint's total derivative over the
/// stacked `[η̂, κ̂₁ … κ̂_K]`, bit for bit. Dropping the subject's κ̂ (`kap_i → &[]` in
/// `assemble_score_cross_product`) sends every IOV subject to the reconverged-FD fallback,
/// whose rows differ from the adjoint's past the 6th significant figure, so this equality
/// fails on that mutation.
#[test]
fn iov_s_matrix_is_the_adjoint_rows_over_the_stacked_mode() {
    let (model, pop, opts) =
        crate::estimation::inner_optimizer::test_helpers::warfarin_iov_fixture();
    let template = &model.default_params;
    let PackedStart {
        packed: x,
        bounds,
        fixed,
        ..
    } = pack_with_bounds(template);
    let free: Vec<usize> = (0..x.len()).filter(|&j| !fixed[j]).collect();
    let (etas, _, kappas) = modes(&model, &pop, &opts, 1e-10);
    assert!(
        kappas.iter().any(|k| !k.is_empty()),
        "fixture: some subject carries κ̂"
    );
    let s = assemble_score_cross_product(
        &x, template, &model, &pop, &etas, &kappas, &bounds, &opts, &free,
    )
    .expect("S");
    let pt = PackedPoint::new(template, &x).expect("packed point");
    let objective = if opts.interaction {
        crate::sens::ode_provider::adjoint::Objective::Focei
    } else {
        crate::sens::ode_provider::adjoint::Objective::Foce
    };
    let mut want = DMatrix::<f64>::zeros(free.len(), free.len());
    for (i, subject) in pop.subjects.iter().enumerate() {
        let b = super::stacked_mode(&model, subject, &etas[i], Some(&kappas[i])).expect("mode");
        let g = crate::sens::ode_provider::adjoint::subject_adjoint_gradient_at(
            &model, subject, &pt, &b, objective,
        )
        .expect("the adjoint serves warfarin_iov");
        let gf = DVector::from_iterator(free.len(), free.iter().map(|&k| g[k]));
        want.ger(1.0, &gf, &gf, 1.0);
    }
    assert_eq!(s, want, "S is the adjoint rows' cross-product");
}
