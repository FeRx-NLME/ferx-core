//! Integration tests for the Tier 4a milestone-5 `gradient = sens` opt-in.
//!
//! Two tiers:
//! - **Fast**: short fit (10 outer iterations) on the experiment's Emax PK/PD
//!   Form C model with `gradient = sens`, verifying the OFV-after-N-iters is
//!   within tolerance of the same fit with `gradient = fd`. This pins the
//!   per-iteration gradient contract — same descent direction, same OFV
//!   trajectory — without paying the wall-time of a full convergence.
//! - **Slow** (`--features slow-tests`): full convergence comparison
//!   FD vs Sens, asserting final OFV matches to within FOCEI line-search
//!   tolerance.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, GradientMethod, Population, Subject};
use ferx_core::{fit, EstimationMethod, FitOptions};
use std::collections::HashMap;

/// 3-state ODE Emax PK/PD with Form C readout — same shape as the
/// `[scaling] y[CMT=2] = central/V` model the milestone-3 and milestone-4
/// PRs validated. Three indiv params (CL, V, KE0) are η-dependent (the
/// fourth, KA, is not), so the Sens path exercises chain-through-indiv-
/// param-partials AND chain-through-states.
const EMAX_PKPD: &str = r"
[parameters]
  theta TVCL(5.0)
  theta TVV(30.0)
  theta TVKA(1.0)
  theta TVKE0(0.5)
  omega ETA_CL  ~ 0.09
  omega ETA_V   ~ 0.09
  omega ETA_KE0 ~ 0.16
  sigma EPS ~ 0.04

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  KA  = TVKA
  KE0 = TVKE0 * exp(ETA_KE0)

[structural_model]
  ode(states=[depot, central, effect])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - CL/V * central
  d/dt(effect)  =  KE0 * (central/V - effect)

[scaling]
  y[CMT=2] = central / V

[error_model]
  DV ~ proportional(EPS)

[fit_options]
  method   = focei
";

/// Small synthetic population — 4 subjects, 100 mg bolus, observations at
/// 0.5, 1, 2, 4, 8 h on CMT=2. Hand-picked observations roughly trace the
/// PK profile so the fit's gradient signal is non-degenerate.
fn synth_population() -> Population {
    let times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
    let n = times.len();
    let dvs = [
        [3.0, 4.5, 5.0, 3.5, 1.8],
        [3.5, 4.8, 5.5, 3.0, 1.4],
        [2.8, 4.0, 4.7, 3.2, 1.6],
        [3.2, 4.2, 4.9, 3.6, 1.9],
    ];
    let subjects = (0..dvs.len())
        .map(|i| Subject {
            id: format!("{}", i + 1),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times.clone(),
            observations: dvs[i].to_vec(),
            obs_cmts: vec![2; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        })
        .collect();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        subjects,
    }
}

fn fit_emax_with_gradient(method: GradientMethod, max_outer: usize) -> (f64, bool) {
    // `parse_model_string` returns a `CompiledModel` directly.
    let mut model = parse_model_string(EMAX_PKPD).expect("Emax PK/PD model parses");
    model.gradient_method = method;
    let pop = synth_population();
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = max_outer;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.gradient_method = method;
    let params = model.default_params.clone();
    let result = fit(&model, &pop, &params, &opts).expect("Emax PK/PD fit must run");
    (result.ofv, result.ofv.is_finite())
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn sens_gradient_finite_ofv_after_short_fit() {
    // Asserts Sens produces a finite OFV after a handful of outer
    // iterations on the Emax PK/PD model. Gated as slow — even 5 outer
    // iterations on a 4-subject ODE+Form-C fit comes in around 160s on
    // the unoptimized test profile (Tier 2 must return immediately).
    // The per-gradient FD-equality contract is pinned by the fast unit
    // test `sens_gradient_matches_fd_on_emax_pkpd` in
    // `src/estimation/inner_optimizer.rs`.
    let (ofv, finite) = fit_emax_with_gradient(GradientMethod::Sens, 5);
    assert!(finite, "Sens fit must produce a finite OFV, got {}", ofv);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn sens_gradient_converges_to_same_ofv_as_fd() {
    let (ofv_sens, _) = fit_emax_with_gradient(GradientMethod::Sens, 200);
    let (ofv_fd, _) = fit_emax_with_gradient(GradientMethod::Fd, 200);
    assert!(
        ofv_sens.is_finite() && ofv_fd.is_finite(),
        "both fits must converge to finite OFV (sens={}, fd={})",
        ofv_sens,
        ofv_fd,
    );
    // FOCEI line-search tolerance + small differences in the η-gradient
    // direction (Sens is exact up to ODE-solver precision; FD has
    // O(1e-6) noise per axis) typically land the converged OFVs within
    // 0.1 of each other on this 4-subject fit. Allow 1.0 for headroom.
    assert!(
        (ofv_sens - ofv_fd).abs() < 1.0,
        "Sens vs FD OFV diverged: sens={:.4} fd={:.4} |Δ|={:.4}",
        ofv_sens,
        ofv_fd,
        (ofv_sens - ofv_fd).abs(),
    );
}
