//! NONMEM FOCEI cross-check for **analytic transit absorption under multiple doses + a TV
//! covariate** (#719 close-out: "lag/F/covariate are also anchored vs NONMEM, incl. multiple
//! dose"). This is the analytic-path counterpart to `transit_multidose_lag_f_nonmem_anchor.rs`
//! (the ODE-twin path).
//!
//! ferx's `pk one_cpt_transit(cl, v, n, mtt)` with `CL = TVCL·(WT/70)^dWTCL·exp(η)` (an estimated
//! allometric covariate) fits **multiple overlapping doses** (100 mg q6h × 5) on its analytic
//! exponential-tilting closed form — no lagtime/F, so no ODE twin. The matching NONMEM control
//! (`nonmem_anchor/transit_multidose_cov.ctl`) delivers the identical model as an explicit
//! integer-N (n = 3 → 4 transit compartments) Erlang chain, whose NONMEM-native multi-dose
//! bookkeeping superposes the five doses exactly (for integer n the Erlang chain's central input
//! *is* the Savic density). The data (`transit_multidose_cov.csv`, 24 subjects) were simulated
//! FROM ferx (`tests/gen_transit_multidose_cov_anchor.rs`); NONMEM 7.6.0 FOCEI MINIMIZATION
//! SUCCESSFUL, #OBJV = −2055.172. The WT range (42–134 kg) and small CL IIV (ω² = 0.01) make
//! `dWTCL` identifiable: it recovers to 0.509 (not the simulated 0.75) on this 24-subject
//! realization — an accidental η_CL–WT sample correlation, NOT a ferx artifact (NONMEM
//! independently lands at the same 0.509).
//!
//! This pins ferx's FOCEI **objective at NONMEM's optimum** (every parameter FIXed at NONMEM's
//! MLE) — the igd/weibull/transit_iov evaluate-at-optimum pattern, which isolates objective
//! agreement from any optimiser-path difference and stays fast/deterministic (the free fit at
//! `TOL=9`-equivalent tolerances is expensive and, on the TVCL↔dWTCL ridge, optimiser-path
//! sensitive — the fit-speed of transit multi-dose models is tracked in #812). ferx observes CMT 1
//! (its single disposition compartment); the committed data file
//! re-keys the obs CMT from the NONMEM copy's CMT 5 (the chain's central) — likelihood-identical.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.6.0 FOCEI MLE (nonmem_anchor/results/transit_multidose_cov.ext, MINIMIZATION
// SUCCESSFUL). omega are variances; sigma is the SD ferx reports (sqrt(0.0110428)).
const NM_TVCL: f64 = 9.31957;
const NM_TVV: f64 = 61.3411;
const NM_TVMTT: f64 = 2.98576;
const NM_DWTCL: f64 = 0.509006;
const NM_OMEGA_CL: f64 = 0.0137998;
const NM_OMEGA_V: f64 = 0.0667750;
const NM_SIGMA_PROP_SD: f64 = 0.1050847; // sqrt(0.0110428)
const NM_OFV_NO_CONST: f64 = -2055.1716;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: NONMEM transit multidose+covariate anchor, opt in with --features slow-tests"
)]
fn transit_multidose_cov_objective_matches_nonmem() {
    // `pk one_cpt_transit` + WT-on-CL covariate, every parameter FIXed at NONMEM's MLE, so `fit()`
    // reduces to an objective evaluation at NONMEM's optimum (only the inner EBEs move).
    let fixed = format!(
        r"
[parameters]
  theta TVCL({NM_TVCL}, FIX)
  theta TVV({NM_TVV}, FIX)
  theta TVMTT({NM_TVMTT}, FIX)
  theta TVN(3.0, FIX)
  theta DWTCL({NM_DWTCL}, FIX)
  omega ETA_CL ~ {NM_OMEGA_CL} FIX
  omega ETA_V  ~ {NM_OMEGA_V} FIX
  sigma PROP_ERR ~ {prop} (sd) FIX

[individual_parameters]
  CL  = TVCL * (WT/70.0)^DWTCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  MTT = TVMTT
  NTR = TVN

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-9
",
        prop = NM_SIGMA_PROP_SD,
    );

    let model = parse_model_string(&fixed).expect("fixed-param transit multidose+cov model parses");
    let pop = read_nonmem_csv(
        Path::new("data/transit_multidose_cov.csv"),
        Some(&["WT"]),
        None,
    )
    .expect("transit_multidose_cov data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true; // match NONMEM METHOD=1 INTER
    opts.run_covariance_step = false;
    opts.ode_reltol = 1e-9; // match NONMEM ADVAN13 TOL=9
    opts.ode_abstol = 1e-9;
    opts.inner_tol = 1e-6;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fixed-param transit multidose+cov objective evaluation must run");

    let diff = (NM_OFV_NO_CONST - result.ofv).abs();
    println!(
        "FERX transit multidose+cov @ NONMEM optimum = {:.4}; NONMEM #OBJV = {:.4}; |gap| = {:.4}",
        result.ofv, NM_OFV_NO_CONST, diff
    );
    // Analytic path, no IOV/twin — the closed form's multi-dose superposition and the WT-on-CL
    // covariate reproduce NONMEM's explicit integer-N chain *exactly* at the shared optimum
    // (observed |gap| = 0.0000, i.e. below print precision). The band is therefore a tight
    // regression guard, not a loose agreement tolerance: 0.1 OFV units sits far above any
    // platform floating-point drift on a single fixed-point objective evaluation, yet catches a
    // genuine multi-dose-superposition or covariate-application bug (which moves a −2055 objective
    // by ≥ 0.1). Tightened from the initial 0.5 (which was 25–50× the actual agreement).
    assert!(
        result.ofv.is_finite() && diff < 0.1,
        "ferx FOCEI transit multidose+cov at NONMEM's optimum = {:.4}; NONMEM #OBJV = {:.4}; \
         |gap| {:.4} exceeds the agreement band (0.1 OFV units) — a wrong multi-dose superposition \
         or covariate application on the analytic transit closed form",
        result.ofv,
        NM_OFV_NO_CONST,
        diff
    );
}
