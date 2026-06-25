//! NONMEM cross-check for the built-in Weibull absorption function
//! `weibull(td, beta)` (issue FeRx-NLME/ferx-core#322 Phase 2, follow-up #503).
//!
//! The model, data, and NONMEM control stream are the anchor kit in the
//! repository's `nonmem_anchor/` directory:
//!
//!   - `nonmem_anchor/weibull_absorption.ctl` is the NONMEM 7.x `ADVAN13 $DES
//!     TOL=9` FOCEI control; its final estimates are in
//!     `nonmem_anchor/results/weibull_absorption.{ext,lst}`.
//!   - `data/igd_oral.csv` is the 20-subject / 240-observation single-dose
//!     dataset shared with the Savic transit and IG anchors, re-keyed to a
//!     1-compartment layout (every record on CMT 1) so the dose feeds the
//!     `weibull()` compartment directly. This is likelihood-equivalent to the
//!     NONMEM control's inert depot (CMT 1, `F1=0`) + central (CMT 2) with `PODO`
//!     driving `R_in` into central — same `R_in`, same predictions, same OFV.
//!
//! ## What this anchors — the likelihood, at the shared optimum
//!
//! The data were simulated from a Savic *transit* model, so the Weibull fit is
//! mildly **mis-specified** (the Weibull density approximates the transit
//! delayed-absorption shape). As with the `igd` anchor, the check is therefore
//! **NONMEM-weibull ≈ ferx-weibull at the same parameters**, not parameter
//! recovery: the two engines must agree on the **objective at the optimum**, not
//! on the optimiser path (on the flat mis-specified ridge NONMEM's gradient FOCEI
//! and ferx's derivative-free BOBYQA stall at slightly different points — here
//! only ~1.5 OFV units apart, milder than `igd`'s ~18). This test evaluates
//! ferx's full FOCEI marginal objective (inner EBE optimisation + Laplace
//! approximation + ODE integration of the `weibull` forcing, with dose routing /
//! bolus suppression / superposition) **at NONMEM's reported optimum** and asserts
//! it matches NONMEM's `#OBJV`. This isolates the `weibull` likelihood from
//! outer-optimiser path-dependence and is a *stronger* check than a full-fit OFV
//! match (which a likelihood bug could mask by landing elsewhere).
//!
//! From `nonmem_anchor/results/weibull_absorption.ext` (NONMEM, FOCEI,
//! MINIMIZATION SUCCESSFUL): `#OBJV = −943.833`, `CL 5.39758`, `V 63.0166`,
//! `TD 1.65572`, `BETA 3.47905`, `ω²(CL) 0.0506751`, `ω²(V) 0.0420421`,
//! `σ²(prop) 0.0479645`. Evaluating ferx's FOCEI objective at exactly those
//! values gives **−943.845** — agreement to 0.01 units, confirming the `weibull`
//! density and its ODE machinery reproduce the NONMEM `$DES` Weibull input.
//! (Unlike the stiffer transit/IG forcings, the smooth Weibull density matches to
//! 0.01 even at default ODE tolerances; the tight `1e-9` below matches `TOL=9`.)

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// 1-cpt Weibull-absorption model, amount form + `[scaling] y = central/V` (the
/// NONMEM `A(2)` amount / `IPRED = A(2)/V` convention). Initial estimates are
/// **NONMEM's final values** (`nonmem_anchor/results/weibull_absorption.ext`): the
/// test evaluates ferx's objective here without taking outer steps. The
/// proportional `sigma` is given as the SD `√0.0479645 = 0.2190080`.
const MODEL_AT_NONMEM_OPTIMUM: &str = r"
[parameters]
  theta TVCL(5.39758,  0.1, 100.0)
  theta TVV(63.0166,   5.0, 500.0)
  theta TVTD(1.65572, 0.05,  24.0)
  theta TVBETA(3.47905, 0.1, 10.0)

  omega ETA_CL ~ 0.0506751
  omega ETA_V  ~ 0.0420421

  sigma PROP_ERR ~ 0.2190080 (sd)

[individual_parameters]
  CL   = TVCL  * exp(ETA_CL)
  V    = TVV   * exp(ETA_V)
  TD   = TVTD
  BETA = TVBETA

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// ferx's FOCEI marginal objective for the `weibull` model, evaluated at NONMEM's
/// optimum, must equal NONMEM's `#OBJV` — the NONMEM-weibull ≈ ferx-weibull check.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored Weibull absorption (#322/#503) acceptance: opt in with --features slow-tests"
)]
fn weibull_marginal_ofv_matches_nonmem_at_their_optimum() {
    // NONMEM #OBJV is −943.833 (nonmem_anchor/results/weibull_absorption.ext,
    // MINIMIZATION SUCCESSFUL). ferx evaluates −943.845 at the same parameters on
    // the FD path CI exercises, with the tight ODE tolerances below; the ±1 band
    // excludes any wrong normalisation constant / off-by-units in the Weibull
    // density (which would shift the objective by many units).
    const NONMEM_OFV: f64 = -943.833;
    const TOLERANCE: f64 = 1.0;

    let model = parse_full_model(MODEL_AT_NONMEM_OPTIMUM)
        .expect("Weibull model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/igd_oral.csv"), None, None)
        .expect("igd_oral data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    // Evaluate the marginal objective at NONMEM's optimum without taking outer
    // steps (see module docs: the outer-optimiser path on this flat mis-specified
    // ridge is not the implementation check; the objective at the optimum is).
    opts.outer_maxiter = 0;
    opts.run_covariance_step = false;
    opts.verbose = false;
    // NONMEM-equivalent ODE accuracy (TOL=9).
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-9;
    opts.inner_tol = 1e-6;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("Weibull fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {:.3} at NONMEM's optimum is outside the NONMEM-anchored \
         band {:.2} ± {:.1} (NONMEM #OBJV −943.833, nonmem_anchor/results/weibull_absorption.ext); \
         a multi-unit gap signals a wrong Weibull density / normalisation",
        result.ofv,
        NONMEM_OFV,
        TOLERANCE,
    );
}
