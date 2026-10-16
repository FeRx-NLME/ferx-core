//! #1082 PR3: the six sensitivity walks — compartment-free, closed form and ODE, each
//! outer `Dual2` and inner `Dual1` — on a model with nested levels, against central
//! finite differences of the production `predict_iov` over the **compact** vector.
//!
//! The walks read the per-cell stacked vector and seed each κ on its compact axis
//! (`StackedLayout::kappa_axis`), so two arms of one study land their study-κ derivative on
//! one axis. FD perturbs the compact vector and expands it, which is what the prediction
//! the objective scores sees: a seed on the wrong axis, or one per cell instead of per
//! group, is a disagreement here, and the failure names the axis — `level STUDY group 1 κ
//! K_A` — so a mutation of one level's axes names that level.
//!
//! The fixture (`stats::nested_levels::test_helpers`) keeps every side live: two levels
//! with two and five groups, a weight that differs on every arm, an off-diagonal study
//! block, and doses whose carryover crosses an arm and a study boundary with drug still
//! present. The non-degeneracy of each of those is asserted, not assumed.

use crate::pk::predict_iov;
use crate::sens::provider::{subject_eta_grad_iov, subject_sensitivities_iov};
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
    // Which engine the jets come from — asserted, so a fixture that slid onto another walk
    // (or onto FD) cannot pass for this one.
    let engine = match walk {
        Walk::Algebraic => model.is_algebraic() && crate::sens::algebraic::supported_iov(&model),
        Walk::ClosedForm => {
            model.ode_spec.is_none()
                && !model.is_algebraic()
                && model.absorption_ode_equivalent.is_none()
                && crate::sens::provider::iov_analytical_supported(&model)
        }
        Walk::Ode => {
            model.ode_spec.is_some() && crate::sens::ode_provider::ode_iov_supported(&model)
        }
    };
    assert!(engine, "{walk:?}: the fixture is not on its walk");
    assert!(
        crate::sens::provider::iov_sens_supported(&model),
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

    /// `∂²f/∂b_k∂b_l` for every observation.
    fn hess(&self, b: &[f64], theta: &[f64], k: usize, l: usize) -> Vec<f64> {
        let h = 1e-4;
        let at = |sk: f64, sl: f64| self.pred(&Self::shifted(b, &[(k, sk), (l, sl)]), theta);
        let (pp, pm, mp, mm) = (at(h, h), at(h, -h), at(-h, h), at(-h, -h));
        (0..pp.len())
            .map(|j| (pp[j] - pm[j] - mp[j] + mm[j]) / (4.0 * h * h))
            .collect()
    }

    /// `∂f/∂θ_m` and `∂²f/∂b_k∂θ_m` for every observation.
    fn theta(&self, b: &[f64], theta: &[f64], m: usize, k: Option<usize>) -> Vec<f64> {
        let (h, s) = match k {
            None => (0.0, 1e-6 * (1.0 + theta[m].abs())),
            Some(_) => (1e-4, 1e-4 * (1.0 + theta[m].abs())),
        };
        let at = |sb: f64, st: f64| {
            let bb = match k {
                Some(k) => Self::shifted(b, &[(k, sb)]),
                None => b.to_vec(),
            };
            self.pred(&bb, &Self::shifted(theta, &[(m, st)]))
        };
        match k {
            None => {
                let (p, q) = (at(0.0, s), at(0.0, -s));
                p.iter().zip(&q).map(|(a, c)| (a - c) / (2.0 * s)).collect()
            }
            Some(_) => {
                let (pp, pm, mp, mm) = (at(h, s), at(h, -s), at(-h, s), at(-h, -s));
                (0..pp.len())
                    .map(|j| (pp[j] - pm[j] - mp[j] + mm[j]) / (4.0 * h * s))
                    .collect()
            }
        }
    }
}

/// The compact layout, the evaluation point and the stacked vector the walks read.
fn point(model: &CompiledModel, subject: &Subject) -> (NestedLayout, Vec<f64>, Vec<f64>) {
    let layout = NestedLayout::of(model, subject).expect("prepared");
    let b = fx::point(model, &layout, 1.0);
    let stacked = fx::stacked(&layout, &b);
    // The straddle that makes the compact axes observable at all: the stacked vector the
    // walks read is wider than the compact one they differentiate over.
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

/// The outer `Dual2` jet — value, `∂f/∂b`, `∂²f/∂b²`, `∂f/∂θ`, `∂²f/∂b∂θ` — against FD.
fn check_outer(walk: Walk) {
    let (model, subject) = fixture(walk);
    let theta = model.default_params.theta.clone();
    let (layout, b, stacked) = point(&model, &subject);
    let n = layout.n_compact();
    let sens = subject_sensitivities_iov(&model, &subject, &theta, &stacked)
        .unwrap_or_else(|| panic!("{walk:?}: the outer walk declined"));
    let fd = Fd {
        model: &model,
        subject: &subject,
        layout: &layout,
    };
    let f0 = fd.pred(&b, &theta);
    assert_eq!(sens.obs.len(), f0.len());
    let label = |i: usize| fx::axis_label(&model, &layout, i);
    let mut worst = [0.0_f64; 5];
    for (j, o) in sens.obs.iter().enumerate() {
        assert_eq!(
            o.df_deta.len(),
            n,
            "{walk:?}: the jet is not over the compact axes"
        );
        worst[0] = worst[0].max(margin(o.f, f0[j], 1e-9, 1e-12));
        assert!(
            worst[0] < 1.0,
            "{walk:?} obs {j}: value {} vs {}",
            o.f,
            f0[j]
        );
    }
    let grads: Vec<Vec<f64>> = (0..n).map(|k| fd.grad(&b, &theta, k)).collect();
    for k in 0..n {
        for (j, o) in sens.obs.iter().enumerate() {
            let e = margin(o.df_deta[k], grads[k][j], 2e-4, 1e-7);
            worst[1] = worst[1].max(e);
            assert!(
                e < 1.0,
                "{walk:?} outer obs {j}: ∂f/∂[{}] {} vs FD {}",
                label(k),
                o.df_deta[k],
                grads[k][j]
            );
        }
        for l in k..n {
            let h = fd.hess(&b, &theta, k, l);
            for (j, o) in sens.obs.iter().enumerate() {
                for (p, q) in [(k, l), (l, k)] {
                    let e = margin(o.d2f_deta2[p * n + q], h[j], 3e-3, 1e-5);
                    worst[2] = worst[2].max(e);
                    assert!(
                        e < 1.0,
                        "{walk:?} outer obs {j}: ∂²f/∂[{}]∂[{}] {} vs FD {}",
                        label(p),
                        label(q),
                        o.d2f_deta2[p * n + q],
                        h[j]
                    );
                }
            }
        }
    }
    let n_theta = theta.len();
    for m in 0..n_theta {
        let g = fd.theta(&b, &theta, m, None);
        for (j, o) in sens.obs.iter().enumerate() {
            let e = margin(o.df_dtheta[m], g[j], 2e-4, 1e-7);
            worst[3] = worst[3].max(e);
            assert!(
                e < 1.0,
                "{walk:?} obs {j}: ∂f/∂θ{m} {} vs {}",
                o.df_dtheta[m],
                g[j]
            );
        }
        for k in 0..n {
            let h = fd.theta(&b, &theta, m, Some(k));
            for (j, o) in sens.obs.iter().enumerate() {
                let e = margin(o.d2f_deta_dtheta[k * n_theta + m], h[j], 3e-3, 1e-5);
                worst[4] = worst[4].max(e);
                assert!(
                    e < 1.0,
                    "{walk:?} obs {j}: ∂²f/∂[{}]∂θ{m} {} vs FD {}",
                    label(k),
                    o.d2f_deta_dtheta[k * n_theta + m],
                    h[j]
                );
            }
        }
    }
    let d: Vec<Vec<f64>> = sens.obs.iter().map(|o| o.df_deta.clone()).collect();
    assert_non_degenerate(walk, &model, &layout, &d);
    eprintln!(
        "MEASURE {walk:?} outer worst margin (1 fails): value {:.3e} ∂b {:.3e} ∂²b {:.3e} \
         ∂θ {:.3e} ∂b∂θ {:.3e}",
        worst[0], worst[1], worst[2], worst[3], worst[4]
    );
}

/// The inner `Dual1` jet `∂f/∂b` against FD.
fn check_inner(walk: Walk) {
    let (model, subject) = fixture(walk);
    let theta = model.default_params.theta.clone();
    let (layout, b, stacked) = point(&model, &subject);
    let n = layout.n_compact();
    let grads = subject_eta_grad_iov(&model, &subject, &theta, &stacked)
        .unwrap_or_else(|| panic!("{walk:?}: the inner walk declined"));
    let fd = Fd {
        model: &model,
        subject: &subject,
        layout: &layout,
    };
    let f0 = fd.pred(&b, &theta);
    let mut worst = 0.0_f64;
    for k in 0..n {
        let g = fd.grad(&b, &theta, k);
        for (j, o) in grads.iter().enumerate() {
            assert_eq!(
                o.df_deta.len(),
                n,
                "{walk:?}: the jet is not over the compact axes"
            );
            assert!(
                margin(o.f, f0[j], 1e-9, 1e-12) < 1.0,
                "{walk:?} obs {j}: value"
            );
            let e = margin(o.df_deta[k], g[j], 2e-4, 1e-7);
            worst = worst.max(e);
            assert!(
                e < 1.0,
                "{walk:?} inner obs {j}: ∂f/∂[{}] {} vs FD {}",
                fx::axis_label(&model, &layout, k),
                o.df_deta[k],
                g[j]
            );
        }
    }
    let d: Vec<Vec<f64>> = grads.iter().map(|o| o.df_deta.clone()).collect();
    assert_non_degenerate(walk, &model, &layout, &d);
    eprintln!("MEASURE {walk:?} inner worst margin (1 fails): ∂b {worst:.3e}");
}

#[test]
fn nested_algebraic_outer_walk_matches_fd_over_the_compact_vector() {
    check_outer(Walk::Algebraic);
}

#[test]
fn nested_algebraic_inner_walk_matches_fd_over_the_compact_vector() {
    check_inner(Walk::Algebraic);
}

#[test]
fn nested_closed_form_outer_walk_matches_fd_over_the_compact_vector() {
    check_outer(Walk::ClosedForm);
}

#[test]
fn nested_closed_form_inner_walk_matches_fd_over_the_compact_vector() {
    check_inner(Walk::ClosedForm);
}

#[test]
fn nested_ode_outer_walk_matches_fd_over_the_compact_vector() {
    check_outer(Walk::Ode);
}

#[test]
fn nested_ode_inner_walk_matches_fd_over_the_compact_vector() {
    check_inner(Walk::Ode);
}
