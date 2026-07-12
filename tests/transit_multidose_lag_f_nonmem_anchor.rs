//! NONMEM FOCEI cross-check for **transit absorption under multiple doses + lagtime + F** (#719
//! close-out: "lag/F/covariate are also anchored vs NONMEM, incl. multiple dose"). This is the
//! ODE-twin-path counterpart to `transit_multidose_cov_nonmem_anchor.rs` (the analytic path).
//!
//! ferx's `pk one_cpt_transit(..., lagtime=LAG, f=FB)` routes each subject to its `transit()` ODE
//! twin (#735, because a lagtime/F mapping is present) and fits **multiple overlapping doses**
//! (100 mg q6h × 5) with:
//!   * an **estimated absorption lagtime** (`TVLAG`, truth 0.5 h — NONMEM recovers 0.501);
//!   * a **fixed bioavailability** `F = 0.7` (structurally non-identifiable on single-route oral —
//!     both engines FIX it; the anchor checks that ferx *applies* F identically to NONMEM);
//!   * a fixed allometric `CL = TVCL·(WT/70)^0.75` covariate composing on the twin path.
//!
//! The matching NONMEM control (`nonmem_anchor/transit_multidose_lag_f.ctl`) is the same explicit
//! integer-N (n = 3 → 4 transit compartments) Erlang chain, now with the NONMEM-native
//! compartment-1 dose attributes `ALAG1` (lag) and `F1` (bioavailability) — so lag / F / multi-dose
//! are all NONMEM-native (no hand-coded `$DES` time shift). Data simulated FROM ferx
//! (`tests/gen_transit_multidose_lag_f_anchor.rs`); NONMEM 7.6.0 FOCEI MINIMIZATION SUCCESSFUL,
//! #OBJV = −2463.833, recovering TVLAG = 0.501 (truth 0.5) and F FIXed at 0.70.
//!
//! The transit **ODE twin** free fit at `TOL=9`-equivalent tolerances is expensive (transit
//! multi-dose fit-speed is tracked in #812), so this pins ferx's FOCEI **objective at NONMEM's
//! optimum** (every parameter FIXed at NONMEM's MLE) — the same evaluate-at-optimum check the
//! `igd`/`weibull`/`transit_iov` anchors use, which isolates the objective agreement from any
//! optimiser-path difference. ferx observes CMT 1 (its single disposition compartment); the
//! committed data file re-keys the obs CMT from the NONMEM copy's CMT 5 — likelihood-identical.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.6.0 FOCEI MLE (nonmem_anchor/results/transit_multidose_lag_f.ext, MINIMIZATION
// SUCCESSFUL). omega/kappa are variances; sigma is the SD ferx reports (sqrt(0.00977627)).
const NM_TVCL: f64 = 8.95560;
const NM_TVV: f64 = 53.9985;
const NM_TVMTT: f64 = 3.01014;
const NM_TVLAG: f64 = 0.501415;
const NM_F: f64 = 0.70; // FIXed in both engines
const NM_OMEGA_CL: f64 = 0.0669717;
const NM_OMEGA_V: f64 = 0.0729285;
const NM_SIGMA_PROP_SD: f64 = 0.0988750; // sqrt(0.00977627)
const NM_OFV_NO_CONST: f64 = -2463.8329;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: NONMEM transit multidose+lag+F anchor, opt in with --features slow-tests"
)]
fn transit_multidose_lag_f_objective_matches_nonmem() {
    // `pk one_cpt_transit(..., lagtime, f)` with every parameter FIXed at NONMEM's MLE, so `fit()`
    // reduces to an objective evaluation at NONMEM's optimum (only the inner EBEs move).
    let fixed = format!(
        r"
[parameters]
  theta TVCL({NM_TVCL}, FIX)
  theta TVV({NM_TVV}, FIX)
  theta TVMTT({NM_TVMTT}, FIX)
  theta TVN(3.0, FIX)
  theta TVLAG({NM_TVLAG}, FIX)
  theta FBIO({NM_F}, FIX)
  omega ETA_CL ~ {NM_OMEGA_CL} FIX
  omega ETA_V  ~ {NM_OMEGA_V} FIX
  sigma PROP_ERR ~ {prop} (sd) FIX

[individual_parameters]
  CL  = TVCL * (WT/70.0)^0.75 * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN
  LAG = TVLAG
  FB  = FBIO

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT, lagtime=LAG, f=FB)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-9
",
        prop = NM_SIGMA_PROP_SD,
    );

    let model = parse_model_string(&fixed).expect("fixed-param transit lag+F model parses");
    let pop = read_nonmem_csv(
        Path::new("data/transit_multidose_lag_f.csv"),
        Some(&["WT"]),
        None,
    )
    .expect("transit_multidose_lag_f data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true; // match NONMEM METHOD=1 INTER
    opts.run_covariance_step = false;
    opts.ode_reltol = 1e-9; // match NONMEM ADVAN13 TOL=9
    opts.ode_abstol = 1e-9;
    opts.inner_tol = 1e-6;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fixed-param transit lag+F objective evaluation must run");

    let diff = (NM_OFV_NO_CONST - result.ofv).abs();
    println!(
        "FERX transit multidose+lag+F @ NONMEM optimum = {:.4}; NONMEM #OBJV = {:.4}; |gap| = {:.4}",
        result.ofv, NM_OFV_NO_CONST, diff
    );
    // ODE-twin path, no IOV (so the #810 ODE-IOV-marginal gap does not apply); this is the same
    // evaluate-at-optimum check the igd/weibull anchors pass to ~0.01-0.02. A gap here would signal
    // a wrong lagtime / bioavailability / multi-dose superposition on the transit ODE twin.
    assert!(
        result.ofv.is_finite() && diff < 0.5,
        "ferx FOCEI transit multidose+lag+F at NONMEM's optimum = {:.4}; NONMEM #OBJV = {:.4}; \
         |gap| {:.4} exceeds the agreement band (0.5 OFV units) — a wrong lagtime / F / multi-dose \
         superposition on the transit ODE twin",
        result.ofv,
        NM_OFV_NO_CONST,
        diff
    );
}
