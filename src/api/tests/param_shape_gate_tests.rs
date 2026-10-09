//! Tier-1 tests for the Ω / σ / Ω_IOV shape gate (#1764), `E_PARAM_SHAPE`, and the θ-length
//! gate on `fit()`'s initial parameters.
//!
//! # What was measured on `main` (`cfc84253`) before this
//!
//! One-θ-short, one-θ-long, σ of length 0 and 2 against a model with 1, Ω of 0×0 and 2×2
//! against 1 η, and Ω_IOV of 0×0 / 2×2 / absent against 1 κ (or present against none):
//!
//! | Entry | θ | σ short | σ long | Ω short/long | Ω_IOV mis-sized | Ω_IOV absent (IOV model) | Ω_IOV on a κ-free model |
//! |---|---|---|---|---|---|---|---|
//! | `fit` (`init_params`) | short: `Ok`, OFV 2.6e14; long: panic | panic | `Ok`, σ carried unused | panic | panic | panic | panic |
//! | `compute_npde_npd` | gated (#1615) | panic | `Ok` | panic | panic | `Ok`, κ = 0 — documented fallback, kept | `Ok` |
//! | `simulate*` | gated (#1614) | panic | `Ok` | panic | panic | `Err` (#1019) | `Ok` |
//! | `simulate_with_uncertainty` (fit's blocks) | gated | panic | `Ok` | panic | `Err` naming the covariance | `Ok`, ipred ≈ 1e-10 | — |
//! | `run_sir` | `Err` (no code) | `Err` naming the covariance | same | `Err` (n_eta) | `Err` naming the covariance | `Ok`, no κ | — |
//! | `run_covariance` | `Err` (no code) | panic | `Ok` | `Err` (n_eta) | panic | `Ok`, no κ | — |
//! | `predict` / `predict_diag` | gated | `Ok`, unread | `Ok`, unread | `Ok`, unread | `Ok`, unread | `Ok`, unread | `Ok`, unread |
//!
//! `predict` runs at η = 0 and reads none of the three, so it is not gated. The adaptive
//! entries are pinned in `adaptive_sim_tests.rs`, which owns their fixtures.
//!
//! # Engine
//!
//! Analytic one-compartment IV and a one-state ODE with one κ; no gradient path is asserted.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Population, Subject};

const ONE_CPT_IV: &str = "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  \
    theta TVV(20.0, 0.001, 500.0)\n  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  \
    pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n";

const ODE_IOV: &str = "[parameters]\n  theta TVCL(5.0, 0.1, 50.0)\n  \
    theta TVV(50.0, 1.0, 500.0)\n  omega ETA_CL ~ 0.09\n  kappa KAPPA_CL ~ 0.09\n  \
    sigma PROP ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  \
    V  = TVV\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
    d/dt(central) = -(CL / V) * central\n[error_model]\n  DV ~ proportional(PROP)\n";

/// The reason clause of a mis-sized Ω or Ω_IOV.
pub(super) const ETA_BY_POSITION: &str = "the model reads it by position, so its values would \
    be read against the wrong random effects";
/// The reason clause of a mis-sized σ.
pub(super) const SIGMA_BY_POSITION: &str = "the model reads it by position, so its values \
    would be read against the wrong residual errors";
/// The reason clause of an absent Ω_IOV on a model with κ.
pub(super) const NO_IOV: &str = "without it every occasion would get the same parameters";
/// The reason clause of an Ω_IOV on a model without κ.
pub(super) const EXTRA_IOV: &str = "the inter-occasion variability it describes would be dropped";

/// Four subjects, two doses, two occasions when `iov`.
fn population(iov: bool) -> Population {
    let subjects = (0..4)
        .map(|i| Subject {
            id: format!("{}", i + 1),
            obs_times: vec![1.0, 4.0, 12.0, 25.0, 28.0, 36.0],
            observations: vec![4.0, 3.0, 1.0, 5.0, 3.5, 1.5],
            obs_cmts: vec![1; 6],
            cens: vec![0; 6],
            occasions: if iov {
                vec![1, 1, 1, 2, 2, 2]
            } else {
                vec![1; 6]
            },
            dose_occasions: if iov { vec![1, 2] } else { vec![1, 1] },
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            ..Default::default()
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn diag(n: usize, v: f64, prefix: &str) -> OmegaMatrix {
    OmegaMatrix::from_diagonal(
        &vec![v; n],
        (0..n).map(|i| format!("{prefix}{i}")).collect(),
    )
}

/// One mis-shaped parameter set: its label, the parameters, and the message it must get.
pub(super) struct Cell {
    pub label: &'static str,
    pub params: ModelParameters,
    pub message: String,
}

/// Every block mis-shaped one at a time against `p`, each FIX vector resized with it so the
/// shape is the only thing wrong. On a model with κ the Ω_IOV cells are mis-sized and absent;
/// without κ, one present Ω_IOV. Assumes one η, one σ and at most one κ, as every fixture here.
pub(super) fn shape_cells(p: &ModelParameters) -> Vec<Cell> {
    assert_eq!((p.omega.dim(), p.sigma.values.len()), (1, 1));
    let mut cells = Vec::new();
    let mut push = |label, params, lead: &str, reason: &str| {
        cells.push(Cell {
            label,
            params,
            message: format!("{lead}; {reason}"),
        })
    };
    for n in [0usize, 2] {
        let mut q = p.clone();
        q.sigma.values.resize(n, 0.2);
        q.sigma.names.resize(n, "EXTRA".into());
        q.sigma_fixed.resize(n, false);
        let lead = format!("the supplied sigma has {n} values but this model has 1");
        push(
            if n == 0 { "σ short" } else { "σ long" },
            q,
            &lead,
            SIGMA_BY_POSITION,
        );
    }
    for n in [0usize, 2] {
        let mut q = p.clone();
        q.omega = diag(n, 0.09, "E");
        q.omega_fixed.resize(n, false);
        let lead = format!("the supplied omega is {n}×{n} but this model has 1 eta");
        push(
            if n == 0 { "Ω short" } else { "Ω long" },
            q,
            &lead,
            ETA_BY_POSITION,
        );
    }
    if p.omega_iov.is_some() {
        for n in [0usize, 2] {
            let mut q = p.clone();
            q.omega_iov = Some(diag(n, 0.04, "K"));
            q.kappa_fixed.resize(n, false);
            let lead = format!("the supplied omega_iov is {n}×{n} but this model has 1 kappa");
            push(
                if n == 0 {
                    "Ω_IOV short"
                } else {
                    "Ω_IOV long"
                },
                q,
                &lead,
                ETA_BY_POSITION,
            );
        }
        let mut q = p.clone();
        q.omega_iov = None;
        q.kappa_fixed.clear();
        let lead = "the supplied parameters carry no omega_iov but this model has 1 kappa";
        push("Ω_IOV absent", q, lead, NO_IOV);
    } else {
        let mut q = p.clone();
        q.omega_iov = Some(diag(1, 0.04, "K"));
        q.kappa_fixed = vec![false];
        let lead = "the supplied parameters carry a 1×1 omega_iov but this model has no kappa";
        push("Ω_IOV on a κ-free model", q, lead, EXTRA_IOV);
    }
    cells
}

/// `e` is `E_PARAM_SHAPE` with exactly `cell`'s message — lead and reason both, so deleting
/// either half of any variant's text reddens the cell that produces it.
pub(super) fn assert_refused(e: &crate::diagnostics::EngineError, cell: &Cell, entry: &str) {
    assert_eq!(
        e.code(),
        Some("E_PARAM_SHAPE"),
        "{entry}, {}: wrong refusal: {e}",
        cell.label
    );
    assert_eq!(e.message(), cell.message, "{entry}, {}", cell.label);
}

fn short_fit() -> FitOptions {
    FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        ..Default::default()
    }
}

/// The fit built from `params`, its Ω_IOV included (`synthetic_fit` leaves it `None`).
fn fit_of(params: &ModelParameters) -> FitResult {
    let mut f = super::simulate_with_uncertainty_tests::synthetic_fit(params);
    f.omega_iov = params.omega_iov.as_ref().map(|m| m.matrix.clone());
    f
}

/// Every entry point that reads Ω, σ or Ω_IOV refuses each mis-shaped block with
/// `E_PARAM_SHAPE`, on a κ-free model and on a model with one κ.
///
/// Regressions this catches: the panics and silent `Ok`s in the module table. Mutation —
/// delete the `check_param_shape` call from any one site and that site's arms go red, each
/// naming its entry: `fit_unstamped` (`fit`, every cell); `compute_npde_npd`;
/// `check_simulate_preconditions` (`simulate`, `simulate_with_seed`,
/// `simulate_with_options_diag`); the up-front call in `simulate_with_uncertainty_diag`
/// (`simulate_with_uncertainty`, zero draws, so the per-draw chokepoint never runs).
///
/// On `simulate*` an absent Ω_IOV keeps #1019's message: `validate_iov_simulatable` runs
/// first, and the arm asserts so — move the shape check above it and that arm reddens. On
/// `compute_npde_npd` it is the documented κ = 0 fallback and stays `Ok`.
#[test]
fn every_reader_refuses_a_mis_shaped_omega_sigma_or_omega_iov() {
    for (text, iov) in [(ONE_CPT_IV, false), (ODE_IOV, true)] {
        let model = parse_model_string(text).expect("parse");
        assert_eq!(model.n_kappa, usize::from(iov));
        let pop = population(iov);
        for cell in shape_cells(&model.default_params) {
            let p = &cell.params;
            let e = fit(&model, &pop, p, &short_fit()).expect_err("fit");
            assert_eq!(e, cell.message, "fit, {}", cell.label);
            let npde = crate::stats::npde::compute_npde_npd(&model, &pop, p, 20, Some(4));
            if cell.label == "Ω_IOV absent" {
                // Documented κ = 0 fallback (#1019), pinned numerically in `stats::npde`.
                npde.expect("compute_npde_npd, absent Ω_IOV falls back to κ = 0");
            } else {
                assert_refused(
                    &npde.expect_err("compute_npde_npd"),
                    &cell,
                    "compute_npde_npd",
                );
            }

            let opts = SimulateOptions {
                seed: Some(5),
                ..Default::default()
            };
            let sims = [
                ("simulate", simulate(&model, &pop, p, 1).err()),
                (
                    "simulate_with_seed",
                    simulate_with_seed(&model, &pop, p, 1, 5).err(),
                ),
                (
                    "simulate_with_options_diag",
                    simulate_with_options_diag(&model, &pop, p, 1, &opts).err(),
                ),
            ];
            for (entry, e) in sims {
                let e = e.unwrap_or_else(|| panic!("{entry} accepted {}", cell.label));
                if cell.label == "Ω_IOV absent" {
                    assert_eq!(e.code(), None, "{entry}: #1019's refusal, not the shape's");
                    assert!(e
                        .to_string()
                        .contains("carry no omega_iov; simulation draws"));
                } else {
                    assert_refused(&e, &cell, entry);
                }
            }

            let uo = SimulateUncertaintyOptions {
                n_uncertainty_draws: 0,
                n_sim_per_draw: 1,
                seed: Some(3),
                ..Default::default()
            };
            let e = simulate_with_uncertainty(&model, &pop, &fit_of(p), &uo)
                .expect_err("simulate_with_uncertainty");
            assert_refused(&e, &cell, "simulate_with_uncertainty");
        }
    }
}

/// `fit()` refuses initial parameters whose θ is not the model's length, with
/// `E_THETA_LENGTH`'s message (`fit` returns a `String`, so the message is all it carries).
///
/// Measured before: one θ short fitted to `Ok` with OFV 2.6e14 against a 2-θ model; one long
/// panicked in the outer gradient. Mutation — delete the `check_theta_length` call in
/// `fit_unstamped` and the short arm returns `Ok`, the long arm panics.
#[test]
fn fit_refuses_initial_theta_of_the_wrong_length() {
    let model = parse_model_string(ONE_CPT_IV).expect("parse");
    let pop = population(false);
    for n in [1usize, 3] {
        let p = super::theta_length_gate_tests::with_theta_len(&model.default_params, n);
        let e = fit(&model, &pop, &p, &short_fit()).expect_err("fit");
        assert_eq!(
            e,
            format!(
                "the supplied theta has {n} values but this model has 2; the model reads \
                 theta by position, so these values would be read against the wrong parameters"
            )
        );
    }
}

/// `run_sir` and `run_covariance` refuse a fit whose σ or Ω_IOV is not the model's.
///
/// Measured before on this IOV fit: an absent Ω_IOV ran both to `Ok` with the model's
/// initial Ω_IOV in its place; on
/// `run_covariance` a short σ and a mis-sized Ω_IOV panicked and a long σ ran to `Ok`;
/// on `run_sir` the mis-sized ones were an `Err` about the covariance matrix. Ω is not
/// in this list: both already refuse it by `n_eta`. Mutation — delete the
/// `check_param_shape` call in `resolve_fit_inputs` and both arms of every cell go red.
#[test]
fn run_sir_and_run_covariance_refuse_a_fit_with_a_mis_shaped_sigma_or_omega_iov() {
    let model = parse_model_string(ODE_IOV).expect("parse");
    let pop = population(true);
    let opts = FitOptions {
        outer_maxiter: 3,
        run_covariance_step: true,
        sir_samples: 50,
        sir_resamples: 20,
        ..Default::default()
    };
    let fitted = fit(&model, &pop, &model.default_params, &opts).expect("fit");
    assert!(fitted.covariance_matrix.is_some() && fitted.omega_iov.is_some());
    let mut n = 0;
    for cell in shape_cells(&model.default_params) {
        if cell.label.starts_with('Ω') && !cell.label.starts_with("Ω_IOV") {
            continue;
        }
        n += 1;
        let mut f = fitted.clone();
        f.sigma = cell.params.sigma.values.clone();
        f.sigma_names = cell.params.sigma.names.clone();
        f.sigma_fixed = cell.params.sigma_fixed.clone();
        f.omega_iov = cell.params.omega_iov.as_ref().map(|m| m.matrix.clone());
        let e = crate::run_sir(&f, Some(&model), Some(&pop), &opts).expect_err("run_sir");
        assert_refused(&e, &cell, "run_sir");
        let e =
            crate::run_covariance(&f, Some(&model), Some(&pop), &opts).expect_err("run_covariance");
        assert_refused(&e, &cell, "run_covariance");
    }
    assert_eq!(n, 5, "σ short/long and the three Ω_IOV cells");
}

/// `fitted_params_from_result` refuses a fit whose Ω_IOV is not the model's (#1789), with
/// `E_PARAM_SHAPE`, on a model with κ and on one without.
///
/// Measured before on `d43afca9`: an absent Ω_IOV on the κ model rebuilt `Ok` with the
/// model's *initial* Ω_IOV in its place; a present one on the κ-free model rebuilt `Ok` with
/// it dropped; a mis-sized one was copied in unchecked. The in-core callers are gated
/// upstream (`resolve_fit_inputs`, `simulate_with_uncertainty_diag`), so only a direct call
/// reaches this. Mutation — delete the `check_param_shape` call and all four cells go `Ok`;
/// restore the `unwrap_or_else` as well and the absent cell is `Ok` again with the initial
/// Ω_IOV, the defect itself.
///
/// The own-shape arm pins the other side: a fitted Ω_IOV that differs from the model's
/// initial one is carried through bit for bit. Mutation — read `iov_template.matrix`
/// instead of the fit's and it reddens, since the fixture asserts the two differ.
#[test]
fn fitted_params_from_result_refuses_a_fit_with_a_mis_shaped_omega_iov() {
    use crate::estimation::uncertainty_samples::fitted_params_from_result;
    let mut n = 0;
    for (text, iov) in [(ONE_CPT_IV, false), (ODE_IOV, true)] {
        let model = parse_model_string(text).expect("parse");
        for cell in shape_cells(&model.default_params) {
            if !cell.label.starts_with("Ω_IOV") {
                continue;
            }
            n += 1;
            let e = fitted_params_from_result(&fit_of(&cell.params), &model)
                .expect_err("fitted_params_from_result");
            assert_refused(&e, &cell, "fitted_params_from_result");
        }

        let mut own = model.default_params.clone();
        if let Some(m) = own.omega_iov.as_mut() {
            m.matrix[(0, 0)] = 0.25;
        }
        let rebuilt = fitted_params_from_result(&fit_of(&own), &model)
            .unwrap_or_else(|e| panic!("own shape refused, iov = {iov}: {e}"));
        match (&model.default_params.omega_iov, &rebuilt.omega_iov) {
            (None, None) => assert!(!iov),
            (Some(template), Some(got)) => {
                assert!(iov);
                assert_ne!(template.matrix[(0, 0)], 0.25, "fixture must differ");
                assert_eq!(got.matrix[(0, 0)].to_bits(), 0.25f64.to_bits());
            }
            (t, g) => panic!("iov = {iov}: template {t:?}, rebuilt {g:?}"),
        }
    }
    assert_eq!(
        n, 4,
        "the κ model's short / long / absent and the κ-free model's present"
    );
}

/// The other side of every gate: the gate admits parameters of the model's own shape — the
/// model's `default_params` and a copy whose Ω, σ and Ω_IOV are rebuilt from their values —
/// on `fit`, `compute_npde_npd`, `simulate_with_seed` and `run_covariance`. The live
/// assertions are the `expect`s: a gate comparing against the wrong count (the `dim + 1`
/// mutation) refuses here. The `to_bits` pairs compare two runs of this head, so they pin
/// only that the gate reads shapes and never the values.
#[test]
fn the_gate_admits_parameters_of_the_models_own_shape() {
    for (text, iov) in [(ONE_CPT_IV, false), (ODE_IOV, true)] {
        let model = parse_model_string(text).expect("parse");
        let pop = population(iov);
        let a = model.default_params.clone();
        let mut b = a.clone();
        b.omega = OmegaMatrix::from_diagonal(&[a.omega.matrix[(0, 0)]], a.omega.eta_names.clone());
        b.sigma = SigmaVector {
            values: a.sigma.values.clone(),
            names: a.sigma.names.clone(),
        };
        b.omega_iov = a
            .omega_iov
            .as_ref()
            .map(|m| OmegaMatrix::from_diagonal(&[m.matrix[(0, 0)]], m.eta_names.clone()));

        let opts = FitOptions {
            outer_maxiter: 2,
            run_covariance_step: false,
            ..Default::default()
        };
        let fa = fit(&model, &pop, &a, &opts).expect("fit, own shape");
        let fb = fit(&model, &pop, &b, &opts).expect("fit, rebuilt");
        assert!(fa.ofv.is_finite(), "{}", fa.ofv);
        assert_eq!(fa.ofv.to_bits(), fb.ofv.to_bits(), "fit, iov = {iov}");

        let na = crate::stats::npde::compute_npde_npd(&model, &pop, &a, 20, Some(4)).expect("ok");
        let nb = crate::stats::npde::compute_npde_npd(&model, &pop, &b, 20, Some(4)).expect("ok");
        for (x, y) in na.iter().zip(&nb) {
            assert_eq!(x.npd.len(), 6);
            for (p, q) in x.npd.iter().zip(&y.npd) {
                assert!(p.is_finite(), "{p}");
                assert_eq!(p.to_bits(), q.to_bits(), "npd, iov = {iov}");
            }
        }

        let sa = simulate_with_seed(&model, &pop, &a, 2, 9).expect("ok");
        let sb = simulate_with_seed(&model, &pop, &b, 2, 9).expect("ok");
        assert_eq!(sa.len(), 48);
        for (x, y) in sa.iter().zip(&sb) {
            assert!(x.ipred.is_finite() && x.ipred > 0.0, "{}", x.ipred);
            assert_eq!(
                x.ipred.to_bits(),
                y.ipred.to_bits(),
                "simulate, iov = {iov}"
            );
        }

        // The post-hoc gate reads the fit's blocks: a fit of the model's own shape passes.
        crate::run_covariance(&fa, Some(&model), Some(&pop), &opts).unwrap_or_else(|e| {
            panic!("run_covariance refused an own-shape fit, iov = {iov}: {e}")
        });
    }
}
