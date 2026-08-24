//! `Dual2`/`Dual1`-vs-FD parity for the compartment-free sensitivity walks (#811).
//!
//! Both walks must agree with central finite differences of the **production f64
//! predictor** (`pk::compute_predictions_with_tv`), because there is no second copy
//! of the readout chain to disagree with them — the repo rule for any new analytic
//! sensitivity path. The routing is pinned in both directions too: a model this
//! module cannot serve must decline (→ FD) rather than return a wrong gradient
//! while the fit reports "analytic".

use super::*;
use crate::parser::model_parser::parse_full_model;
use crate::pk::compute_predictions_with_tv;
use crate::types::Population;
use std::collections::HashMap;

/// Emax time-course with IIV on the baseline, the shape of an MBMA structural
/// model: `y = E0·exp(η) − EMAX·t/(ET50 + t)`. `extra_params` / `extra_indiv` /
/// `equation` let each test vary one axis of the model.
fn model_src(extra_params: &str, extra_indiv: &str, equation: &str) -> String {
    format!(
        "[parameters]\n\
        \x20 theta TVE0(10.0, 0.1, 100.0)\n\
        \x20 theta TVEMAX(6.0, 0.1, 100.0)\n\
        \x20 theta TVET50(2.0, 0.01, 100.0)\n\
        {extra_params}\
        \x20 omega ETA_E0 ~ 0.09\n\
        \x20 sigma PROP ~ 0.04 (sd)\n\n\
        [individual_parameters]\n\
        \x20 E0   = TVE0 * exp(ETA_E0)\n\
        \x20 EMAX = TVEMAX\n\
        \x20 ET50 = TVET50\n\
        {extra_indiv}\n\
        [structural_model]\n\
        {equation}\n\
        [error_model]\n\
        \x20 DV ~ proportional(PROP)\n"
    )
}

fn compile(src: &str) -> crate::types::CompiledModel {
    parse_full_model(src).expect("model parses").model
}

/// One subject, no doses — observations across the rise and the plateau.
/// `covariates` seeds the subject-level snapshot; `obs_covariates` (when
/// non-empty) makes them time-varying, which switches the walks to per-observation
/// re-seeding.
fn subject(covariates: Vec<(&str, f64)>, obs_covariates: Vec<Vec<(&str, f64)>>) -> Subject {
    let obs_times = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0];
    let n = obs_times.len();
    let to_map = |v: Vec<(&str, f64)>| -> HashMap<String, f64> {
        v.into_iter().map(|(k, x)| (k.to_string(), x)).collect()
    };
    Subject {
        id: "1".into(),
        doses: vec![],
        obs_times,
        obs_raw_times: vec![],
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: to_map(covariates),
        dose_covariates: vec![],
        obs_covariates: obs_covariates.into_iter().map(to_map).collect(),
        pk_only_times: vec![],
        pk_only_covariates: vec![],
        reset_times: vec![],
        cens: vec![0; n],
        occasions: vec![],
        obs_l2: Vec::new(),
        dose_occasions: vec![],
        fremtype: vec![],
        obs_records: vec![],
    }
}

/// Every block of the outer `Dual2` jet against central FDs of the production
/// predictor: value, `∂f/∂η`, `∂²f/∂η²`, `∂f/∂θ`, `∂²f/∂η∂θ`.
fn check_outer_vs_fd(model: &crate::types::CompiledModel, subject: &Subject) {
    let theta: Vec<f64> = model.default_params.theta.clone();
    let eta = vec![0.3_f64; model.n_eta];
    let sens = subject_sensitivities(model, subject, &theta, &eta)
        .expect("compartment-free model is in analytic scope");

    let pred = |e: &[f64], th: &[f64], j: usize| -> f64 {
        compute_predictions_with_tv(model, subject, th, e)[j]
    };
    let (he, ht, heh) = (1e-6_f64, 1e-6_f64, 1e-4_f64);
    let (n_eta, n_theta) = (model.n_eta, theta.len());

    for (j, obs) in sens.obs.iter().enumerate() {
        approx::assert_relative_eq!(obs.f, pred(&eta, &theta, j), max_relative = 1e-12);

        for k in 0..n_eta {
            let (mut ep, mut em) = (eta.clone(), eta.clone());
            ep[k] += he;
            em[k] -= he;
            let g = (pred(&ep, &theta, j) - pred(&em, &theta, j)) / (2.0 * he);
            approx::assert_relative_eq!(obs.df_deta[k], g, max_relative = 2e-4, epsilon = 1e-7);

            for l in 0..n_eta {
                let at = |dk: f64, dl: f64| {
                    let mut e = eta.clone();
                    e[k] += dk;
                    e[l] += dl;
                    pred(&e, &theta, j)
                };
                let hh = (at(heh, heh) - at(heh, -heh) - at(-heh, heh) + at(-heh, -heh))
                    / (4.0 * heh * heh);
                approx::assert_relative_eq!(
                    obs.d2f_deta2[k * n_eta + l],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }

        for m in 0..n_theta {
            let step = ht * (1.0 + theta[m].abs());
            let (mut tp, mut tm) = (theta.clone(), theta.clone());
            tp[m] += step;
            tm[m] -= step;
            let g = (pred(&eta, &tp, j) - pred(&eta, &tm, j)) / (2.0 * step);
            approx::assert_relative_eq!(obs.df_dtheta[m], g, max_relative = 2e-4, epsilon = 1e-7);
        }

        for k in 0..n_eta {
            for m in 0..n_theta {
                let s = heh * (1.0 + theta[m].abs());
                let at = |dk: f64, dm: f64| {
                    let (mut e, mut th) = (eta.clone(), theta.clone());
                    e[k] += dk;
                    th[m] += dm;
                    pred(&e, &th, j)
                };
                let hh = (at(heh, s) - at(heh, -s) - at(-heh, s) + at(-heh, -s)) / (4.0 * heh * s);
                approx::assert_relative_eq!(
                    obs.d2f_deta_dtheta[k * n_theta + m],
                    hh,
                    max_relative = 3e-3,
                    epsilon = 1e-5
                );
            }
        }
    }
}

/// The inner `Dual1` walk must agree with the outer jet's own `∂f/∂η` — the
/// property that keeps the inner EBE loop and the outer gradient on one route.
fn check_inner_matches_outer(model: &crate::types::CompiledModel, subject: &Subject) {
    let theta: Vec<f64> = model.default_params.theta.clone();
    let eta = vec![0.3_f64; model.n_eta];
    let outer = subject_sensitivities(model, subject, &theta, &eta).expect("outer in scope");
    let inner = subject_eta_grad(model, subject, &theta, &eta).expect("inner in scope");
    assert_eq!(outer.obs.len(), inner.len());
    for (o, i) in outer.obs.iter().zip(&inner) {
        approx::assert_relative_eq!(o.f, i.f, max_relative = 1e-14);
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(o.df_deta[k], i.df_deta[k], max_relative = 1e-14);
        }
    }
}

#[test]
fn algebraic_outer_jet_matches_fd() {
    let model = compile(&model_src(
        "",
        "",
        "  y = E0 - EMAX * TIME / (ET50 + TIME)\n",
    ));
    check_outer_vs_fd(&model, &subject(vec![], vec![]));
}

/// Named intermediates are inlined at parse, so the jet must be identical to the
/// hand-inlined form — the readout program cannot tell the two apart.
#[test]
fn algebraic_outer_jet_matches_fd_with_intermediates() {
    let model = compile(&model_src(
        "",
        "",
        "  EFF = EMAX * TIME / (ET50 + TIME)\n  y   = E0 - EFF\n",
    ));
    check_outer_vs_fd(&model, &subject(vec![], vec![]));
}

/// A covariate on the equation — how an MBMA model reaches its per-row study /
/// arm data. Subject-static here, so both walks take the single-snapshot path.
#[test]
fn algebraic_outer_jet_matches_fd_with_a_covariate() {
    let model = compile(&model_src(
        "  theta TVSLOPE(0.4, -5.0, 5.0)\n",
        "  SLOPE = TVSLOPE\n",
        "  y = E0 - EMAX * TIME / (ET50 + TIME) + SLOPE * DOSE\n",
    ));
    check_outer_vs_fd(&model, &subject(vec![("DOSE", 25.0)], vec![]));
}

/// Time-varying covariates switch both walks to per-observation re-seeding. The
/// prediction and its gradient must still linearise about the same point — the
/// failure mode this cadence exists to prevent.
#[test]
fn algebraic_outer_jet_matches_fd_with_time_varying_covariates() {
    let model = compile(&model_src(
        "  theta TVSLOPE(0.4, -5.0, 5.0)\n",
        "  SLOPE = TVSLOPE\n",
        "  y = E0 - EMAX * TIME / (ET50 + TIME) + SLOPE * DOSE\n",
    ));
    let per_obs: Vec<Vec<(&str, f64)>> =
        (0..6).map(|i| vec![("DOSE", 10.0 * (i as f64))]).collect();
    let subj = subject(vec![("DOSE", 0.0)], per_obs);
    assert!(
        subj.has_tv_covariates(),
        "the test must exercise the per-obs path"
    );
    check_outer_vs_fd(&model, &subj);
}

/// A nonlinear (`exp`/`ln`) readout, so the second-order blocks are not trivially
/// zero — a linear equation would pass an η-η Hessian check by accident.
#[test]
fn algebraic_outer_jet_matches_fd_on_a_nonlinear_readout() {
    let model = compile(&model_src(
        "",
        "",
        "  y = ln(E0 + exp(EMAX * TIME / (ET50 + TIME)))\n",
    ));
    check_outer_vs_fd(&model, &subject(vec![], vec![]));
}

#[test]
fn algebraic_inner_grad_matches_the_outer_jet() {
    for equation in [
        "  y = E0 - EMAX * TIME / (ET50 + TIME)\n",
        "  y = ln(E0 + exp(EMAX * TIME / (ET50 + TIME)))\n",
    ] {
        let model = compile(&model_src("", "", equation));
        check_inner_matches_outer(&model, &subject(vec![], vec![]));
    }
}

/// A model with `log_transform` (LTBS) wraps the readout in `ln`; the jet must
/// carry that wrap, not the raw readout.
#[test]
fn algebraic_outer_jet_matches_fd_under_ltbs() {
    let src = "[parameters]\n\
        \x20 theta TVE0(10.0, 0.1, 100.0)\n\
        \x20 theta TVEMAX(6.0, 0.1, 100.0)\n\
        \x20 theta TVET50(2.0, 0.01, 100.0)\n\
        \x20 omega ETA_E0 ~ 0.09\n\
        \x20 sigma ADD ~ 0.04 (sd)\n\n\
        [individual_parameters]\n\
        \x20 E0   = TVE0 * exp(ETA_E0)\n\
        \x20 EMAX = TVEMAX\n\
        \x20 ET50 = TVET50\n\n\
        [structural_model]\n\
        \x20 y = E0 - EMAX * TIME / (ET50 + TIME)\n\n\
        [error_model]\n\
        \x20 log(DV) ~ additive(ADD)\n";
    let model = compile(src);
    assert!(model.log_transform, "the test must exercise the LTBS wrap");
    check_outer_vs_fd(&model, &subject(vec![], vec![]));
}

// ── Routing: a scope gap must fail loudly to FD, never silently ──────────────

/// The happy path reports analytic on *both* loops, so the fit's reported
/// gradient method matches the route it takes.
#[test]
fn algebraic_model_reports_analytic_on_both_loops() {
    let model = compile(&model_src(
        "",
        "",
        "  y = E0 - EMAX * TIME / (ET50 + TIME)\n",
    ));
    assert!(supported(&model));
    assert!(crate::sens::provider::sens_supported(&model));
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        &model
    ));
    assert!(crate::estimation::inner_optimizer::inner_reports_analytic_model(&model));
}

/// The placeholder `pk_model` must never reach the closed-form provider: it names
/// a real one-compartment IV model, whose walks would evaluate a solution this
/// model does not have against doses it does not carry.
#[test]
fn algebraic_model_declines_the_closed_form_provider() {
    let model = compile(&model_src(
        "",
        "",
        "  y = E0 - EMAX * TIME / (ET50 + TIME)\n",
    ));
    assert!(
        !crate::sens::provider::analytical_supported(&model),
        "the closed-form provider must decline a model with no closed form"
    );
}

/// IOV is out of scope for now (the readout would need per-occasion seeding).
/// It must decline here *and* stop reporting analytic, so the fit falls back to
/// FD — which differentiates the per-occasion f64 predictor and is correct.
#[test]
fn algebraic_model_with_iov_declines_to_fd() {
    let src = "[parameters]\n\
        \x20 theta TVE0(10.0, 0.1, 100.0)\n\
        \x20 theta TVEMAX(6.0, 0.1, 100.0)\n\
        \x20 theta TVET50(2.0, 0.01, 100.0)\n\
        \x20 omega ETA_E0 ~ 0.09\n\
        \x20 kappa KAPPA_E0 ~ 0.04\n\
        \x20 sigma PROP ~ 0.04 (sd)\n\n\
        [individual_parameters]\n\
        \x20 E0   = TVE0 * exp(ETA_E0 + KAPPA_E0)\n\
        \x20 EMAX = TVEMAX\n\
        \x20 ET50 = TVET50\n\n\
        [structural_model]\n\
        \x20 y = E0 - EMAX * TIME / (ET50 + TIME)\n\n\
        [error_model]\n\
        \x20 DV ~ proportional(PROP)\n";
    let model = compile(src);
    assert!(model.is_algebraic());
    assert!(model.n_kappa > 0, "the test must exercise the IOV path");
    assert!(!supported(&model), "IOV is out of scope for now");
    let subj = subject(vec![], vec![]);
    let theta = model.default_params.theta.clone();
    let eta = vec![0.1; model.n_eta];
    assert!(
        subject_sensitivities(&model, &subj, &theta, &eta).is_none(),
        "an out-of-scope model must decline, not return a gradient"
    );
    assert!(subject_eta_grad(&model, &subj, &theta, &eta).is_none());
}

/// A compartment model must not be claimed by this provider.
#[test]
fn compartment_models_are_not_claimed() {
    let src = "[parameters]\n\
        \x20 theta TVCL(3.0, 0.01, 100.0)\n\
        \x20 theta TVV(20.0, 1.0, 500.0)\n\
        \x20 omega ETA_CL ~ 0.09\n\
        \x20 sigma PROP ~ 0.04 (sd)\n\n\
        [individual_parameters]\n\
        \x20 CL = TVCL * exp(ETA_CL)\n\
        \x20 V  = TVV\n\n\
        [structural_model]\n\
        \x20 pk one_cpt_iv(cl=CL, v=V)\n\n\
        [error_model]\n\
        \x20 DV ~ proportional(PROP)\n";
    let model = compile(src);
    assert!(!model.is_algebraic());
    assert!(!supported(&model));
}

/// The dispatch ladders and the scope predicate are keyed on the same cap, so an
/// in-scope model can never fall through `_ => None` to FD while the fit reports
/// an analytic gradient. Exercises every arm the outer ladder enumerates.
#[test]
fn every_axis_count_up_to_the_cap_dispatches() {
    let subj = subject(vec![], vec![]);
    // 1 η is fixed by the template, so `n_theta` runs to `cap - 1`.
    for n_extra in 0..(MAX_ALGEBRAIC_AXES - 4) {
        let mut params = String::new();
        let mut indiv = String::new();
        let mut terms = String::new();
        for i in 0..n_extra {
            params.push_str(&format!("  theta TVX{i}(0.5, -5.0, 5.0)\n"));
            indiv.push_str(&format!("  X{i} = TVX{i}\n"));
            terms.push_str(&format!(" + X{i}"));
        }
        let model = compile(&model_src(
            &params,
            &indiv,
            &format!("  y = E0 - EMAX * TIME / (ET50 + TIME){terms}\n"),
        ));
        let axes = model.n_theta + model.n_eta;
        assert!(axes <= MAX_ALGEBRAIC_AXES);
        assert!(supported(&model), "in scope at {axes} axes");
        let theta = model.default_params.theta.clone();
        let eta = vec![0.2; model.n_eta];
        assert!(
            subject_sensitivities(&model, &subj, &theta, &eta).is_some(),
            "the {axes}-axis dispatch arm must be wired"
        );
        assert!(subject_eta_grad(&model, &subj, &theta, &eta).is_some());
    }
}

/// Past the cap the model must decline at the *gate*, so the fit reports FD and
/// takes FD — rather than passing the gate and hitting the ladder's `_ => None`.
#[test]
fn past_the_axis_cap_declines_at_the_gate() {
    let mut params = String::new();
    let mut indiv = String::new();
    let mut terms = String::new();
    for i in 0..MAX_ALGEBRAIC_AXES {
        params.push_str(&format!("  theta TVX{i}(0.5, -5.0, 5.0)\n"));
        indiv.push_str(&format!("  X{i} = TVX{i}\n"));
        terms.push_str(&format!(" + X{i}"));
    }
    let model = compile(&model_src(
        &params,
        &indiv,
        &format!("  y = E0 - EMAX * TIME / (ET50 + TIME){terms}\n"),
    ));
    assert!(model.n_theta + model.n_eta > MAX_ALGEBRAIC_AXES);
    assert!(!supported(&model), "past the cap the gate must decline");
    assert!(!crate::sens::provider::sens_supported(&model));
    assert!(!crate::estimation::inner_optimizer::inner_reports_analytic_model(&model));
}

/// The fit must still be correct past the cap — FD of the readout-aware predictor
/// is the fallback, and it has to produce a finite objective.
#[test]
fn a_population_fit_past_the_cap_still_runs() {
    let subj = subject(vec![], vec![]);
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subj],
    };
    let model = compile(&model_src(
        "",
        "",
        "  y = E0 - EMAX * TIME / (ET50 + TIME)\n",
    ));
    let preds = crate::predict(&model, &pop, &model.default_params);
    assert!(preds.iter().all(|p| p.pred.is_finite()));
}
