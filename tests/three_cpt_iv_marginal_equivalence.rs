//! FOCE **marginal** OFV equivalence of the analytical and ODE forms of the
//! 3-compartment IV proportional-error model (issue #378).
//!
//! `tests/analytical_ode_equivalence.rs` guards the *structural* (eta=0) side —
//! pointwise PRED and the fixed-eta population NLL. That is insufficient: it
//! never exercises the per-subject inner EBE solve, so it could not catch the
//! FOCE **marginal** divergence #378 reported (ODE vs analytical diverged by up
//! to ~18 OFV units on the stiff 3-cpt IV proportional pair, platform-sensitive).
//!
//! Root cause (#378): on a subject whose individual objective is multimodal
//! (here, id 14 — 6 IIV etas conditioned on 10 points), the closed-form inner
//! BFGS *finds* the better mode but stalls at a gradient norm just above the
//! tightened `inner_tol` (1e-5, #330). The exact-path recovery then discarded
//! that near-global partial for a worse cold Nelder-Mead basin, while the ODE
//! form (which reaches `gnorm < tol`) kept the good mode — so the two forms'
//! marginals diverged. The fix keeps the lower-objective of {BFGS partial, NM}
//! on the non-FREM exact path, matching the ODE path's existing guard (#555).
//!
//! This is a `maxiter = 0` fit (inner EBE solve + Sheiner–Beal marginal at the
//! shared initial theta), so it is a fast, deterministic check of the marginal
//! surface — no convergence loop. It reproduces ferx-r's
//! `test-ode-analytical-equivalence.R` comparison inside ferx-core.

use ferx_core::run_model_with_data;
use std::fs;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/three_cpt_iv.csv");

// Common [fit_options] for a marginal-OFV-only evaluation.
const FIT_OPTS: &str = "\n[fit_options]\n  method     = foce\n  maxiter    = 0\n  covariance = false\n  checkpoint = false\n  inner_tol  = 1e-5\n";

fn analytical_model() -> String {
    format!(
        "[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(10.0, 0.1, 500.0)
  theta TVQ2(2.0, 0.01, 50.0)
  theta TVV2(20.0, 0.1, 1000.0)
  theta TVQ3(1.5, 0.01, 50.0)
  theta TVV3(30.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_Q2 ~ 0.04
  omega ETA_V2 ~ 0.04
  omega ETA_Q3 ~ 0.04
  omega ETA_V3 ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2 * exp(ETA_Q2)
  V2 = TVV2 * exp(ETA_V2)
  Q3 = TVQ3 * exp(ETA_Q3)
  V3 = TVV3 * exp(ETA_V3)
[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)
[error_model]
  DV ~ proportional(PROP_ERR){FIT_OPTS}"
    )
}

fn ode_model() -> String {
    format!(
        "[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(10.0, 0.1, 500.0)
  theta TVQ2(2.0, 0.01, 50.0)
  theta TVV2(20.0, 0.1, 1000.0)
  theta TVQ3(1.5, 0.01, 50.0)
  theta TVV3(30.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.04
  omega ETA_Q2 ~ 0.04
  omega ETA_V2 ~ 0.04
  omega ETA_Q3 ~ 0.04
  omega ETA_V3 ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q2 = TVQ2 * exp(ETA_Q2)
  V2 = TVV2 * exp(ETA_V2)
  Q3 = TVQ3 * exp(ETA_Q3)
  V3 = TVV3 * exp(ETA_V3)
[structural_model]
  ode(states=[central, peripheral1, peripheral2])
[odes]
  d/dt(central)     = -(CL/V1)*central - (Q2/V1)*central + (Q2/V2)*peripheral1 - (Q3/V1)*central + (Q3/V3)*peripheral2
  d/dt(peripheral1) =  (Q2/V1)*central - (Q2/V2)*peripheral1
  d/dt(peripheral2) =  (Q3/V1)*central - (Q3/V3)*peripheral2
[scaling]
  y = central / V1
[error_model]
  DV ~ proportional(PROP_ERR){FIT_OPTS}  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n"
    )
}

#[test]
fn three_cpt_iv_ode_matches_analytical_foce_marginal_ofv() {
    let dir = env!("CARGO_TARGET_TMPDIR");
    let an_path = format!("{dir}/r378_three_cpt_iv_an.ferx");
    let ode_path = format!("{dir}/r378_three_cpt_iv_ode.ferx");
    fs::write(&an_path, analytical_model()).unwrap();
    fs::write(&ode_path, ode_model()).unwrap();

    let (an, _) = run_model_with_data(&an_path, Some(DATA)).expect("analytical fit");
    let (ode, _) = run_model_with_data(&ode_path, Some(DATA)).expect("ODE fit");

    let dofv = (an.ofv - ode.ofv).abs();
    // The two forms compute the *same* SB-FOCE marginal with analytic
    // sensitivities (closed-form Dual2 vs the RK45-integrated Dual1 walk), so at
    // the tight ode_reltol they must agree to well within solver tolerance. Pre-fix
    // this diverged by ~0.13 here (macOS) / ~18 (Linux CI). 1e-3 catches that
    // regression with a >100x margin while tolerating RK45 round-off.
    assert!(
        dofv < 1e-3,
        "ODE vs analytical FOCE marginal OFV diverged (#378): an = {:.6}, ode = {:.6}, |dOFV| = {:.6}",
        an.ofv,
        ode.ofv,
        dofv
    );
}
