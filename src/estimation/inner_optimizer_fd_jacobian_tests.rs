//! #2701: the FD Jacobian `H` of an integrated model, which the FOCE/FOCEI determinant term
//! reads. Fixtures are a small Friberg-style myelosuppression chain with feedback, nonlinear
//! enough that the integrator's step sequence moves with η.

use super::*;
use crate::ode::solver::{arm_ode_solver_override, OdeSolverOverride};
use crate::ode::OdeMethod;
use crate::parser::model_parser::parse_model_string;
use crate::sens::ode_provider::adjoint::exact_eta_jacobian;
use crate::types::Population;
use std::io::Write;
use std::path::Path;

const CHAIN: &str = "
[individual_parameters]
  SLOPE = TSLOPE * exp(ETA_SLOPE)
  MTT   = TMTT * exp(ETA_MTT KAPPA_TERM)
  CIRC0 = TC0 * exp(ETA_C)
  KTR   = 4.0 / MTT

[structural_model]
  ode(states=[chemo, sm, mt, pm, rb])

[odes]
  init(sm) = CIRC0
  init(mt) = CIRC0
  init(pm) = CIRC0
  init(rb) = CIRC0
  d/dt(chemo) = -0.1 * chemo
  d/dt(sm) = KTR * sm * (1.0 - SLOPE * chemo / 50.0) * (0.5 * (CIRC0 / rb + 1.0 + abs(CIRC0 / rb - 1.0)))^0.16 - KTR * sm
  d/dt(mt) = KTR * (sm - mt)
  d/dt(pm) = KTR * (mt - pm)
  d/dt(rb) = KTR * pm - KTR * rb

[scaling]
  y[CMT=5] = log(rb + 0.00001)

[error_model]
  DV ~ additive(S)
";

const PARAMETERS: &str = "
[parameters]
  theta TSLOPE(15.0, 0.01, 1000.0)
  theta TMTT(120.0, 10.0, 500.0)
  theta TC0(5.0, 0.1, 50.0)
  omega ETA_SLOPE ~ 0.3
  omega ETA_MTT ~ 0.05
  omega ETA_C ~ 0.1
  sigma S ~ 0.3 (sd)
";

fn chain_model() -> CompiledModel {
    parse_model_string(&format!("{PARAMETERS}{}", CHAIN.replace(" KAPPA_TERM", "")))
        .expect("the chain model parses")
}

fn chain_iov_model() -> CompiledModel {
    parse_model_string(&format!(
        "{}{}\n[fit_options]\n  iov_column = OCC\n",
        PARAMETERS.replace("  sigma S", "  kappa KAPPA_MTT ~ 0.01\n  sigma S"),
        CHAIN.replace(" KAPPA_TERM", " + KAPPA_MTT")
    ))
    .expect("the chain IOV model parses")
}

/// `n` subjects on two chemo cycles, sampled through the nadir and recovery of each. Nothing
/// here reads the DV column.
fn chain_population(n: usize) -> Population {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    writeln!(f, "ID,TIME,DV,EVID,AMT,CMT,MDV").unwrap();
    for id in 1..=n {
        for start in [0.0, 504.0] {
            writeln!(f, "{id},{start},0,1,100,1,1").unwrap();
            for t in [72.0, 168.0, 240.0, 336.0, 480.0] {
                writeln!(f, "{id},{},0,0,0,5,0", start + t).unwrap();
            }
        }
    }
    f.flush().unwrap();
    let mut pop = crate::read_nonmem_csv(f.path(), None, None).expect("the chain data loads");
    for s in &mut pop.subjects {
        s.occasions = (0..s.obs_times.len()).map(|k| 1 + (k / 5) as u32).collect();
        s.dose_occasions = vec![1, 2];
    }
    pop
}

fn eta_of(i: usize) -> [f64; 3] {
    let u = i as f64 + 1.0;
    [
        0.5 * (1.7 * u).sin(),
        0.2 * (2.3 * u).cos(),
        0.3 * (0.9 * u).sin(),
    ]
}

fn kappas_of(i: usize) -> Vec<Vec<f64>> {
    let u = i as f64 + 1.0;
    vec![vec![0.05 * u.sin()], vec![0.05 * u.cos()]]
}

/// The worst element of `h` against `truth`, relative to `1 + |truth|`.
fn worst_gap(h: &DMatrix<f64>, truth: &DMatrix<f64>) -> f64 {
    assert_eq!(h.shape(), truth.shape());
    assert!(h.iter().chain(truth.iter()).all(|v| v.is_finite()));
    h.iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
        .fold(0.0, f64::max) // nan-ok: both operands asserted finite above
}

/// `log|Ω⁻¹ + HᵀH/σ²|`, the term of the FOCE/FOCEI objective `H` reaches, at this fixture's
/// diagonal Ω and σ.
fn log_det_term(h: &DMatrix<f64>) -> f64 {
    let omega_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0 / 0.3, 1.0 / 0.05, 10.0]));
    let m = omega_inv + h.transpose() * h / 0.09;
    m.determinant().ln()
}

/// Solves run at a tolerance far past both the fit's and the Jacobian's, so the exact Jacobian
/// here is the oracle: at the fit's own tolerance it is itself off by up to 0.62 on this
/// fixture.
fn oracle_scope() -> crate::ode::solver::OdeSolverOverrideGuard {
    arm_ode_solver_override(OdeSolverOverride {
        reltol: Some(1e-11),
        abstol: Some(1e-13),
        max_steps: Some(10_000_000),
        ..Default::default()
    })
}

/// The rule #2701 replaced: a roundoff-sized step at the fit's tolerance.
fn roundoff_step_jacobian(
    n_obs: usize,
    eta: &[f64],
    predict: &mut dyn FnMut(&[f64]) -> Vec<f64>,
) -> DMatrix<f64> {
    let mut h = DMatrix::zeros(n_obs, eta.len());
    let mut pert = eta.to_vec();
    for j in 0..eta.len() {
        let step = FD_JACOBIAN_STEP * (1.0 + eta[j].abs());
        pert[j] = eta[j] + step;
        let plus = predict(&pert);
        pert[j] = eta[j] - step;
        let minus = predict(&pert);
        pert[j] = eta[j];
        for i in 0..n_obs {
            h[(i, j)] = (plus[i] - minus[i]) / (2.0 * step);
        }
    }
    h
}

/// The worst gaps to the oracle over a population: the FD Jacobian, the replaced rule, and the
/// largest shift each puts on the determinant term.
#[derive(Default)]
struct Gaps {
    h: f64,
    h_replaced: f64,
    log_det: f64,
    log_det_replaced: f64,
}

impl Gaps {
    fn add(&mut self, h: &DMatrix<f64>, replaced: &DMatrix<f64>, truth: &DMatrix<f64>) {
        let shift = |m: &DMatrix<f64>| {
            let d = (log_det_term(m) - log_det_term(truth)).abs();
            assert!(d.is_finite());
            d
        };
        self.h = self.h.max(worst_gap(h, truth)); // nan-ok: `worst_gap` asserts finite
        self.h_replaced = self.h_replaced.max(worst_gap(replaced, truth)); // nan-ok: same
        self.log_det = self.log_det.max(shift(h)); // nan-ok: asserted finite in `shift`
        self.log_det_replaced = self.log_det_replaced.max(shift(replaced)); // nan-ok: same
    }

    /// The bounds of both tests. Measured on macOS arm64: `H` 1.7e-4 and 5.9e-4 (IOV), log
    /// determinant 3.6e-4 and 8.6e-4, against 1.0 and 1.6, and 0.66 and 1.1, for the replaced
    /// rule. That rule must stay outside the bounds, or the fixture has stopped exposing
    /// integration noise. A `1e-6` step at the Jacobian's tolerance measured 4.2e-3 and 3.3e-3
    /// under IOV, so the bounds see the step as well as the tolerance.
    fn assert_resolved(&self, h_bound: f64, log_det_bound: f64) {
        assert!(
            self.h < h_bound && self.log_det < log_det_bound,
            "FD H gap {:.3e}, log-determinant shift {:.3e}",
            self.h,
            self.log_det
        );
        assert!(
            self.h_replaced > 10.0 * h_bound && self.log_det_replaced > 10.0 * log_det_bound,
            "the fixture no longer exposes the noise: replaced rule's H gap {:.3e}, shift {:.3e}",
            self.h_replaced,
            self.log_det_replaced
        );
    }
}

/// #2701: at the default tolerance the FD Jacobian is the exact Jacobian, and so is the
/// determinant term it feeds, where the rule it replaced was off by order one.
#[test]
fn fd_h_of_an_integrated_model_resolves_at_the_default_tolerance() {
    let model = chain_model();
    let pop = chain_population(12);
    let theta = &model.default_params.theta;
    let mut scratch = pk::EventPkParams::default();
    let mut gaps = Gaps::default();
    for (i, s) in pop.subjects.iter().enumerate() {
        let eta = eta_of(i);
        let truth = {
            let _oracle = oracle_scope();
            exact_eta_jacobian(&model, s, theta, &eta).expect("the adjoint serves the chain")
        };
        let h = compute_jacobian_fd(&model, s, theta, &eta, &mut scratch, None);
        let replaced = roundoff_step_jacobian(s.obs_times.len(), &eta, &mut |e| {
            pk::compute_predictions_with_tv_into_with_schedule(
                &model,
                s,
                theta,
                e,
                &mut scratch,
                None,
            )
        });
        gaps.add(&h, &replaced, &truth);
    }
    gaps.assert_resolved(2e-3, 2.5e-3);
}

/// #2701: the IOV twin, whose differences run through `predict_iov`.
#[test]
fn fd_h_under_iov_resolves_at_the_default_tolerance() {
    let model = chain_iov_model();
    let pop = chain_population(12);
    let theta = &model.default_params.theta;
    let mut gaps = Gaps::default();
    for (i, s) in pop.subjects.iter().enumerate() {
        let eta = eta_of(i);
        let kappas = kappas_of(i);
        let stacked: Vec<f64> = eta.iter().chain(kappas.iter().flatten()).copied().collect();
        let truth = {
            let _oracle = oracle_scope();
            exact_eta_jacobian(&model, s, theta, &stacked)
                .expect("the adjoint serves the chain")
                .columns(0, eta.len())
                .into_owned()
        };
        let h = compute_jacobian_fd_iov(&model, s, theta, &eta, &kappas);
        let replaced = roundoff_step_jacobian(s.obs_times.len(), &eta, &mut |e| {
            pk::predict_iov(&model, s, theta, e, &kappas)
        });
        gaps.add(&h, &replaced, &truth);
    }
    gaps.assert_resolved(2e-3, 2.5e-3);
}

fn pinned_chain(reltol: f64, abstol: f64, max_steps: usize) -> CompiledModel {
    let mut model = chain_model();
    let opts = &mut model.ode_spec.as_mut().unwrap().solver_opts;
    opts.reltol = reltol;
    opts.abstol = abstol;
    opts.max_steps = max_steps;
    model
}

/// The tolerance is the Jacobian's own: a default-tolerance model's `H` is bit for bit that of
/// a model pinned there, a tighter model keeps its tolerance, and nothing outlives the call.
#[test]
fn fd_h_runs_at_the_jacobian_tolerance_and_leaves_the_fits_in_force() {
    let default = chain_model();
    let baked = default.ode_spec.as_ref().unwrap().effective_solver_opts();
    let pinned = pinned_chain(
        ODE_FD_JACOBIAN_RELTOL,
        ODE_FD_JACOBIAN_ABSTOL,
        baked.max_steps * ODE_FD_JACOBIAN_MAX_STEPS_FACTOR,
    );
    let tighter = pinned_chain(1e-10, 1e-12, baked.max_steps * 100);
    assert!(fd_jacobian_ode_override(&default).is_some());
    assert_eq!(fd_jacobian_ode_override(&pinned), None);
    assert_eq!(fd_jacobian_ode_override(&tighter), None);

    let pop = chain_population(1);
    let s = &pop.subjects[0];
    let theta = default.default_params.theta.clone();
    let eta = eta_of(0);
    let mut scratch = pk::EventPkParams::default();
    let mut h_of = |m: &CompiledModel| compute_jacobian_fd(m, s, &theta, &eta, &mut scratch, None);
    let (d, p, t) = (h_of(&default), h_of(&pinned), h_of(&tighter));
    assert_eq!(
        d, p,
        "the default model's H must integrate at the pinned values"
    );
    assert_ne!(
        d, t,
        "a tighter model's H must integrate at its own tolerance"
    );

    let after = default.ode_spec.as_ref().unwrap().effective_solver_opts();
    assert_eq!(
        (after.reltol, after.abstol, after.max_steps),
        (baked.reltol, baked.abstol, baked.max_steps),
        "the Jacobian's tolerance must not outlive the Jacobian"
    );
}

/// Each tolerance tightens on its own, and the fit's other solver choices come along.
#[test]
fn fd_jacobian_override_tightens_each_tolerance_and_keeps_the_fits_solver() {
    let model = pinned_chain(1e-10, 1e-6, 5000);
    let _fit = arm_ode_solver_override(OdeSolverOverride {
        method: Some(OdeMethod::Rosenbrock23),
        stiff_abort_after: Some(Some(7)),
        auto_switch: Some(false),
        ..Default::default()
    });
    assert_eq!(
        fd_jacobian_ode_override(&model),
        Some(OdeSolverOverride {
            reltol: Some(1e-10),
            abstol: Some(ODE_FD_JACOBIAN_ABSTOL),
            max_steps: Some(5000 * ODE_FD_JACOBIAN_MAX_STEPS_FACTOR),
            method: Some(OdeMethod::Rosenbrock23),
            stiff_abort_after: Some(Some(7)),
            auto_switch: Some(false),
        })
    );
    let looser_abstol_only = pinned_chain(1e-4, 1e-12, 5000);
    let tight = fd_jacobian_ode_override(&looser_abstol_only).unwrap();
    assert_eq!(
        (tight.reltol, tight.abstol),
        (Some(ODE_FD_JACOBIAN_RELTOL), Some(1e-12))
    );
}

/// A model with no integrator keeps its roundoff-sized step, bit for bit, and a closed form
/// that can reroute to an absorption twin counts as integrated.
#[test]
fn fd_h_of_a_closed_form_is_unchanged() {
    let model = crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
        .expect("warfarin parses");
    let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    let s = &pop.subjects[0];
    let theta = &model.default_params.theta;
    let eta = vec![0.1; model.n_eta];
    let mut scratch = pk::EventPkParams::default();
    assert_eq!(fd_jacobian_ode_override(&model), None);
    let h = compute_jacobian_fd(&model, s, theta, &eta, &mut scratch, None);
    let by_hand = roundoff_step_jacobian(s.obs_times.len(), &eta, &mut |e| {
        pk::compute_predictions_with_tv_into_with_schedule(&model, s, theta, e, &mut scratch, None)
    });
    assert_eq!(h, by_hand);

    let twin =
        crate::parser::model_parser::parse_model_file(Path::new("examples/one_cpt_transit.ferx"))
            .expect("the transit model parses");
    assert!(twin.ode_spec.is_none() && twin.integrator_spec().is_some());
    assert!(fd_jacobian_ode_override(&twin).is_some());
}

/// A tight pass that is non-finite is redone at the fit's tolerance with the same step, and
/// counted; a clean one is neither.
#[test]
fn a_non_finite_tight_jacobian_is_redone_at_the_fits_tolerance() {
    let model = chain_model();
    let spec = model.ode_spec.as_ref().unwrap();
    let fit_reltol = spec.effective_solver_opts().reltol;
    let redone_before = FD_H_REDONE.with(|c| c.get());

    let mut calls = Vec::new();
    let h = with_fd_jacobian_integration(&model, |step| {
        let reltol = spec.effective_solver_opts().reltol;
        calls.push((step, reltol));
        let v = if reltol < fit_reltol { f64::NAN } else { 1.0 };
        DMatrix::from_element(1, 1, v)
    });
    assert_eq!(h[(0, 0)], 1.0);
    assert_eq!(
        calls,
        vec![
            (ODE_FD_JACOBIAN_STEP, ODE_FD_JACOBIAN_RELTOL),
            (ODE_FD_JACOBIAN_STEP, fit_reltol),
        ]
    );
    assert_eq!(FD_H_REDONE.with(|c| c.get()), redone_before + 1);

    let mut calls = 0;
    with_fd_jacobian_integration(&model, |_| {
        calls += 1;
        DMatrix::from_element(1, 1, 1.0)
    });
    assert_eq!(calls, 1);
    assert_eq!(FD_H_REDONE.with(|c| c.get()), redone_before + 1);
}

/// #2701 review: a tight pass truncated by its step budget stays **finite**, freeze-padded, so
/// finiteness alone would accept it. The solves here are real ones, cut to 3 steps a segment
/// while the Jacobian's tolerance is armed.
#[test]
fn a_truncated_but_finite_tight_jacobian_is_redone() {
    let model = chain_model();
    let spec = model.ode_spec.as_ref().unwrap();
    let fit_reltol = spec.effective_solver_opts().reltol;
    let pop = chain_population(1);
    let s = &pop.subjects[0];
    let theta = &model.default_params.theta;
    let eta = eta_of(0);
    let mut scratch = pk::EventPkParams::default();

    let mut tight_pass = None;
    let h = with_fd_jacobian_integration(&model, |step| {
        let tight = spec.effective_solver_opts().reltol < fit_reltol;
        let _cut = tight.then(|| {
            arm_ode_solver_override(OdeSolverOverride {
                max_steps: Some(3),
                ..Default::default()
            })
        });
        let preds = pk::compute_predictions_with_tv_into_with_schedule(
            &model,
            s,
            theta,
            &eta,
            &mut scratch,
            None,
        );
        let m = DMatrix::from_vec(preds.len(), 1, preds) * step;
        if tight {
            tight_pass = Some(m.clone());
        }
        m
    });
    let truncated = tight_pass.expect("the tight pass ran");
    assert!(
        truncated.iter().all(|v| v.is_finite()),
        "this fixture needs a finite truncated pass"
    );
    assert_ne!(h, truncated, "the truncated pass must not be returned");
}

/// A chain model pinned where its fit-tolerance solves fit the step budget and the Jacobian's
/// tighter ones do not, on a stepper whose steps grow fastest with tolerance.
fn budget_starved_chain() -> CompiledModel {
    let mut model = pinned_chain(1e-1, 1e-1, STARVED_MAX_STEPS);
    model.ode_spec.as_mut().unwrap().solver_opts.method = OdeMethod::Rosenbrock23;
    model
}

/// Measured on this fixture at `1e-1` / `1e-1`: the fit's tolerance finishes from 25 steps a
/// segment, and the Jacobian's, at 32 times this, leaves every subject unfinished up to 40,
/// under IOV and at η = 0 alike.
const STARVED_MAX_STEPS: usize = 30;

/// #2701 review: the report. On real solves, the fit's tolerance finishes and the Jacobian's
/// does not, so every subject is counted and the fit carries the warning; with the budget
/// raised, no subject is.
#[test]
fn a_redone_final_h_is_reported_by_the_fit() {
    let pop = chain_population(4);
    let etas: Vec<DVector<f64>> = (0..4)
        .map(|i| DVector::from_column_slice(&eta_of(i)))
        .collect();

    let mut model = budget_starved_chain();
    model.gradient_method = GradientMethod::Fd;
    let params = model.default_params.clone();

    // The straddle the fixture needs: finished at the fit's tolerance.
    let scope = crate::ode::solver::SolverStatsScope::enter();
    let mut scratch = pk::EventPkParams::default();
    for (s, eta) in pop.subjects.iter().zip(&etas) {
        pk::compute_predictions_with_tv_into_with_schedule(
            &model,
            s,
            &params.theta,
            eta.as_slice(),
            &mut scratch,
            None,
        );
    }
    assert_eq!(scope.collected().unfinished_segments, 0);
    drop(scope);

    assert_eq!(
        fd_h_subjects_redone(&model, &pop, &params, &etas, &[], None),
        4
    );
    // A subject with no reported EBE is skipped rather than read out of bounds.
    assert_eq!(
        fd_h_subjects_redone(&model, &pop, &params, &etas[..1], &[], None),
        1
    );

    let mut roomy = pinned_chain(1e-1, 1e-1, 100_000);
    roomy.ode_spec.as_mut().unwrap().solver_opts.method = OdeMethod::Rosenbrock23;
    roomy.gradient_method = GradientMethod::Fd;
    assert_eq!(
        fd_h_subjects_redone(&roomy, &pop, &params, &etas, &[], None),
        0
    );

    // Already at the Jacobian's tolerance: nothing to redo, and nothing is built.
    let mut tight = pinned_chain(1e-9, 1e-11, STARVED_MAX_STEPS);
    tight.gradient_method = GradientMethod::Fd;
    assert_eq!(
        fd_h_subjects_redone(&tight, &pop, &params, &etas, &[], None),
        0
    );
}

/// Under IOV the count runs on the IOV `H`, and a subject with no κ̂ is skipped.
#[test]
fn a_redone_final_h_is_counted_under_iov() {
    let mut model = chain_iov_model();
    let opts = &mut model.ode_spec.as_mut().unwrap().solver_opts;
    opts.reltol = 1e-1;
    opts.abstol = 1e-1;
    opts.max_steps = STARVED_MAX_STEPS;
    opts.method = OdeMethod::Rosenbrock23;
    model.gradient_method = GradientMethod::Fd;
    let pop = chain_population(2);
    let params = model.default_params.clone();
    let etas: Vec<DVector<f64>> = (0..2)
        .map(|i| DVector::from_column_slice(&eta_of(i)))
        .collect();
    let kappas: Vec<Vec<DVector<f64>>> = (0..2)
        .map(|i| {
            kappas_of(i)
                .iter()
                .map(|k| DVector::from_column_slice(k))
                .collect()
        })
        .collect();
    assert_eq!(
        fd_h_subjects_redone(&model, &pop, &params, &etas, &kappas, None),
        2
    );
    assert_eq!(
        fd_h_subjects_redone(&model, &pop, &params, &etas, &kappas[..1], None),
        1
    );
}

/// A subject on the exact Jacobian takes no FD pass, so it cannot be counted, whatever the
/// budget. The mutation this catches is a report that forgets the route.
#[test]
fn an_exact_h_is_never_counted_as_redone() {
    let model = budget_starved_chain();
    assert!(!inner_analytic_opted_out(&model));
    let pop = chain_population(2);
    let params = model.default_params.clone();
    let etas: Vec<DVector<f64>> = (0..2)
        .map(|i| DVector::from_column_slice(&eta_of(i)))
        .collect();
    assert_eq!(
        fd_h_subjects_redone(&model, &pop, &params, &etas, &[], None),
        0
    );
}

/// End to end: `fit()` carries the warning, typed, in both its channels, and each sentence of
/// it says what it should.
#[test]
fn the_fit_warns_when_its_final_h_ran_at_the_fits_tolerance() {
    use crate::types::{FitOptions, WarningCode, WarningSeverity};
    let mut pop = chain_population(4);
    for s in &mut pop.subjects {
        s.occasions = Vec::new();
        s.dose_occasions = Vec::new();
        s.observations = vec![1.0; s.obs_times.len()];
    }
    let mut model = budget_starved_chain();
    model.gradient_method = GradientMethod::Fd;
    let params = model.default_params.clone();
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        interaction: true,
        ..FitOptions::default()
    };
    let result = crate::fit(&model, &pop, &params, &options).expect("the evaluation runs");
    let msg = result
        .warnings
        .iter()
        .find(|w| w.starts_with("W_FD_JACOBIAN_NOT_TIGHTENED:"))
        .unwrap_or_else(|| panic!("no FD Jacobian warning in {:?}", result.warnings));
    for clause in [
        "4 of 4 subjects' finite-difference Jacobian",
        "ode_reltol = 1e-8 / ode_abstol = 1e-10",
        "ran at the fit's own ODE tolerance",
        "so the OFV may be too high",
        "Raise ode_max_steps or ode_stiff_abort_after, or name a stiff ode_method",
    ] {
        assert!(msg.contains(clause), "missing {clause:?} in {msg}");
    }
    let classified = crate::types::classify_warning(msg);
    assert_eq!(classified.category, WarningCode::OdeSolver);
    assert_eq!(classified.severity, WarningSeverity::Warning);
}
