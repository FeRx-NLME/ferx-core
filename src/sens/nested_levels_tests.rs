//! #1082: the adjoint's recordings — compartment-free, closed form and ODE — on a model
//! with nested levels, against central finite differences of the production `predict_iov`
//! and `individual_nll_iov` over the **compact** vector.
//!
//! A recording reads the per-cell stacked vector and records over the compact axes it
//! expands from (`ode_adjoint::expand_nested`: each cell's κ is its group's axis leaf), so
//! two arms of one study land their study-κ derivative on one axis. FD perturbs the compact
//! vector and expands it, which is what the prediction the objective scores sees: a leaf on
//! the wrong axis, or one per cell instead of per group, is a disagreement here, and the
//! failure names the axis — `level STUDY group 1 κ K_A` — so a mutation of one level's axes
//! names that level. The second-order check runs through the exact inner Hessian, whose
//! prior is the compact one `stacked_omega` places per group of each level.
//!
//! The θ side and the assembled packed gradient are checked end to end against the
//! reconverged-FD oracle (`nested_packed_gradients_match_the_reconverged_fd_gradient`).
//!
//! The fixture (`stats::nested_levels::test_helpers`) keeps every side live: two levels
//! with two and five groups, a weight that differs on every arm, an off-diagonal study
//! block, and doses whose carryover crosses an arm and a study boundary with drug still
//! present. The non-degeneracy of each of those is asserted, not assumed.

use crate::pk::predict_iov;
use crate::sens::ode_provider::adjoint::{
    exact_eta_jacobian, exact_inner_hessian, exact_obs_grads, fixed_eta_theta_adjoint,
};
use crate::stats::nested_levels::test_helpers as fx;
use crate::stats::nested_levels::NestedLayout;
use crate::types::{CompiledModel, Subject};

#[derive(Clone, Copy, Debug)]
enum Walk {
    Algebraic,
    ClosedForm,
    Ode,
}

fn fixture(walk: Walk) -> (CompiledModel, Subject) {
    let src = match walk {
        Walk::Algebraic => fx::algebraic(),
        Walk::ClosedForm => fx::closed_form(),
        Walk::Ode => fx::ode(),
    };
    let (model, mut pop) = fx::prepared(&src, 1);
    assert!(model.has_nested_levels(), "{walk:?}: the fixture must nest");
    // Which engine the recording runs — asserted, so a fixture that slid onto another one
    // (or out of the recorders' scope) cannot pass for this one.
    let engine = match walk {
        Walk::Algebraic => model.is_algebraic(),
        Walk::ClosedForm => {
            model.ode_spec.is_none()
                && !model.is_algebraic()
                && model.absorption_ode_equivalent.is_none()
        }
        Walk::Ode => model.ode_spec.is_some(),
    };
    assert!(engine, "{walk:?}: the fixture is not on its engine");
    assert!(
        crate::sens::ode_provider::adjoint::model_in_scope(&model),
        "{walk:?}"
    );
    (model, pop.subjects.remove(0))
}

/// `|got − want| / (eps + rel·|want|)`: below 1 passes. Non-finite values fail outright,
/// so a `NaN` jet cannot hide in a fold.
fn margin(got: f64, want: f64, rel: f64, eps: f64) -> f64 {
    assert!(got.is_finite() && want.is_finite(), "{got} vs {want}");
    (got - want).abs() / (eps + rel * want.abs())
}

/// Central FD of the prediction over the compact vector (and θ), computed once per
/// perturbation for every observation.
struct Fd<'a> {
    model: &'a CompiledModel,
    subject: &'a Subject,
    layout: &'a NestedLayout,
}

impl Fd<'_> {
    fn pred(&self, b: &[f64], theta: &[f64]) -> Vec<f64> {
        predict_iov(
            self.model,
            self.subject,
            theta,
            &b[..self.model.n_eta],
            &self.layout.expand(b),
        )
    }

    fn shifted(v: &[f64], moves: &[(usize, f64)]) -> Vec<f64> {
        let mut out = v.to_vec();
        for &(i, h) in moves {
            out[i] += h;
        }
        out
    }

    /// `∂f/∂b_k` for every observation.
    fn grad(&self, b: &[f64], theta: &[f64], k: usize) -> Vec<f64> {
        let h = 1e-6;
        let p = self.pred(&Self::shifted(b, &[(k, h)]), theta);
        let m = self.pred(&Self::shifted(b, &[(k, -h)]), theta);
        p.iter().zip(&m).map(|(a, c)| (a - c) / (2.0 * h)).collect()
    }
}

/// The compact layout, the evaluation point and the stacked vector the recordings read.
fn point(model: &CompiledModel, subject: &Subject) -> (NestedLayout, Vec<f64>, Vec<f64>) {
    let layout = NestedLayout::of(model, subject).expect("prepared");
    let b = fx::point(&layout, 1.0);
    let stacked = fx::stacked(&layout, &b);
    // The straddle that makes the compact axes observable at all: the stacked vector the
    // recordings read is wider than the compact one they differentiate over.
    assert!(layout.n_compact() < stacked.len());
    (layout, b, stacked)
}

/// Every compact axis is live, and the fixture's structure is live on it: the study κs
/// collect a derivative from two cells, and (PK) the drug carried across an arm boundary
/// and across the study boundary moves with the earlier arm's κ.
fn assert_non_degenerate(walk: Walk, model: &CompiledModel, layout: &NestedLayout, d: &[Vec<f64>]) {
    let n_obs = d.len();
    let scale = d.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    for i in 0..layout.n_compact() {
        let live = (0..n_obs).any(|j| d[j][i].abs() > 1e-4 * scale);
        assert!(
            live,
            "{walk:?}: {} carries no derivative",
            fx::axis_label(model, layout, i)
        );
    }
    let k_a = model.kappa_names.iter().position(|n| n == "K_A").unwrap();
    let k_arm = model.kappa_names.iter().position(|n| n == "K_ARM").unwrap();
    // Observations 0..3 are arm 0 (study 1, arm 1), 3..6 arm 1 (study 1, arm 2).
    let study1 = layout.compact_index(0, k_a);
    assert_eq!(
        study1,
        layout.compact_index(1, k_a),
        "two arms, one study axis"
    );
    assert!(d[0][study1] != 0.0 && d[4][study1] != 0.0, "{walk:?}");
    if matches!(walk, Walk::Algebraic) {
        return;
    }
    // t = 25: arm 1 of study 1 has begun, its dose (t = 30) has not — the drug is dose 1's,
    // eliminated through arm 0's V until t = 24. t = 50: study 2 arm 1, no dose of its own;
    // its drug was dosed in study 1 arm 2.
    let arm0 = layout.compact_index(0, k_arm);
    let arm1 = layout.compact_index(1, k_arm);
    assert!(
        d[3][arm0].abs() > 1e-6 * scale,
        "{walk:?}: no carryover into arm 2"
    );
    assert!(
        d[6][arm1].abs() > 1e-6 * scale,
        "{walk:?}: no carryover into study 2"
    );
}

/// The first-order recording — value and `∂f/∂b` — against FD of the prediction.
fn check_jacobian(walk: Walk) {
    let (model, subject) = fixture(walk);
    let theta = model.default_params.theta.clone();
    let (layout, b, stacked) = point(&model, &subject);
    let n = layout.n_compact();
    let jac = exact_eta_jacobian(&model, &subject, &theta, &stacked)
        .unwrap_or_else(|| panic!("{walk:?}: the recording declined"));
    let rows = exact_obs_grads(&model, &subject, &theta, &stacked)
        .unwrap_or_else(|| panic!("{walk:?}: the recording declined"));
    let fd = Fd {
        model: &model,
        subject: &subject,
        layout: &layout,
    };
    let f0 = fd.pred(&b, &theta);
    assert_eq!(jac.nrows(), f0.len());
    assert_eq!(
        jac.ncols(),
        n,
        "{walk:?}: the jet is not over the compact axes"
    );
    let mut worst = [0.0_f64; 2];
    for (j, o) in rows.iter().enumerate() {
        let e = margin(o.f, f0[j], 1e-9, 1e-12);
        worst[0] = worst[0].max(e);
        assert!(e < 1.0, "{walk:?} obs {j}: value {} vs {}", o.f, f0[j]);
        assert_eq!(o.df_deta.len(), n, "{walk:?}");
    }
    for k in 0..n {
        let g = fd.grad(&b, &theta, k);
        for j in 0..f0.len() {
            // The two first-order entry points read one recording.
            assert_eq!(
                jac[(j, k)].to_bits(),
                rows[j].df_deta[k].to_bits(),
                "{walk:?}"
            );
            let e = margin(jac[(j, k)], g[j], 2e-4, 1e-7);
            worst[1] = worst[1].max(e);
            assert!(
                e < 1.0,
                "{walk:?} obs {j}: ∂f/∂[{}] {} vs FD {}",
                fx::axis_label(&model, &layout, k),
                jac[(j, k)],
                g[j]
            );
        }
    }
    let d: Vec<Vec<f64>> = rows.iter().map(|o| o.df_deta.clone()).collect();
    assert_non_degenerate(walk, &model, &layout, &d);
    eprintln!(
        "MEASURE {walk:?} jacobian worst margin (1 fails): value {:.3e} ∂b {:.3e}",
        worst[0], worst[1]
    );
}

/// The second-order recording through the exact inner Hessian `∇²_b l` — the data term's
/// `∂²f/∂b²` and the compact prior's precision — against a central second difference of
/// the conditional objective over the compact vector.
fn check_inner_hessian(walk: Walk) {
    let (model, subject) = fixture(walk);
    let params = model.default_params.clone();
    let omega_iov = params.omega_iov.as_ref().expect("omega_iov");
    let (layout, b, stacked) = point(&model, &subject);
    let n = layout.n_compact();
    let h = exact_inner_hessian(&model, &subject, &params, &stacked)
        .unwrap_or_else(|| panic!("{walk:?}: the recording declined"));
    assert_eq!((h.nrows(), h.ncols()), (n, n), "{walk:?}");
    let nll = |bb: &[f64]| -> f64 {
        crate::stats::likelihood::individual_nll_iov(
            &model,
            &subject,
            &params.theta,
            &bb[..model.n_eta],
            &layout.expand(bb),
            &params.omega,
            Some(omega_iov),
            &params.sigma.values,
        )
    };
    let step = 1e-4;
    let at = |moves: &[(usize, f64)]| nll(&Fd::shifted(&b, moves));
    let mut worst = 0.0_f64;
    for k in 0..n {
        for l in k..n {
            let fd = if k == l {
                (at(&[(k, step)]) - 2.0 * at(&[]) + at(&[(k, -step)])) / (step * step)
            } else {
                (at(&[(k, step), (l, step)])
                    - at(&[(k, step), (l, -step)])
                    - at(&[(k, -step), (l, step)])
                    + at(&[(k, -step), (l, -step)]))
                    / (4.0 * step * step)
            };
            for (p, q) in [(k, l), (l, k)] {
                let e = margin(h[(p, q)], fd, 1e-4, 1e-5);
                worst = worst.max(e);
                assert!(
                    e < 1.0,
                    "{walk:?}: ∂²l/∂[{}]∂[{}] {} vs FD {fd}",
                    fx::axis_label(&model, &layout, p),
                    fx::axis_label(&model, &layout, q),
                    h[(p, q)]
                );
            }
        }
    }
    eprintln!("MEASURE {walk:?} inner Hessian worst margin (1 fails): {worst:.3e}");
}

#[test]
fn nested_algebraic_recording_matches_fd_over_the_compact_vector() {
    check_jacobian(Walk::Algebraic);
    check_inner_hessian(Walk::Algebraic);
}

#[test]
fn nested_closed_form_recording_matches_fd_over_the_compact_vector() {
    check_jacobian(Walk::ClosedForm);
    check_inner_hessian(Walk::ClosedForm);
}

#[test]
fn nested_ode_recording_matches_fd_over_the_compact_vector() {
    check_jacobian(Walk::Ode);
    check_inner_hessian(Walk::Ode);
}

/// #2468 review: the fixed-η θ adjoint (SAEM's M-step, VI's score) on a nested-level subject
/// reads the per-cell stacked vector its callers hold, and records over the compact axes it
/// expands from (`resolve`), not per cell: `∂f_j/∂θ` against central differences of the
/// production `predict_iov` at that point. Handed the stack without that conversion, a cell
/// read another cell's κ and every prediction after it moved.
fn check_fixed_eta_theta(walk: Walk) {
    let (model, subject) = fixture(walk);
    let (layout, b, stacked) = point(&model, &subject);
    let fd = Fd {
        model: &model,
        subject: &subject,
        layout: &layout,
    };
    let theta = model.default_params.theta.clone();
    let n_obs = subject.observations.len();
    let x = fixed_eta_theta_adjoint(&model, &subject, &theta, &stacked, true, &vec![1.0; n_obs])
        .unwrap_or_else(|| panic!("{walk:?}: the nested subject is in the recorders' scope"));
    let jac = x.jac.expect("asked for");
    let n_theta = theta.len();
    let base = fd.pred(&b, &theta);
    let scale = base.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    let mut live = false;
    for m in 0..n_theta {
        let h = 1e-6 * theta[m].abs().max(1.0);
        let mut tp = theta.clone();
        tp[m] += h;
        let p = fd.pred(&b, &tp);
        tp[m] -= 2.0 * h;
        let q = fd.pred(&b, &tp);
        for j in 0..n_obs {
            let want = (p[j] - q[j]) / (2.0 * h);
            live |= want.abs() > 1e-6 * scale;
            worst = worst.max(margin(jac[j * n_theta + m], want, 1e-5, 1e-8 * scale));
        }
    }
    assert!(live, "{walk:?}: no θ moves a prediction");
    assert!(
        worst < 1.0,
        "{walk:?}: fixed-η θ Jacobian vs FD, margin {worst:.2e}"
    );
}

#[test]
fn nested_fixed_eta_theta_adjoint_matches_fd_over_theta() {
    for walk in [Walk::Algebraic, Walk::ClosedForm, Walk::Ode] {
        check_fixed_eta_theta(walk);
    }
}
