//! NONMEM cross-check for the built-in Savic transit-compartment absorption
//! function `transit(n, mtt)` (PR FeRx-NLME/ferx-core#343, issue #322).
//!
//! The model, data, and NONMEM control stream are the anchor kit in the
//! repository's `nonmem_anchor/` directory:
//!
//!   - `nonmem_anchor/savic_transit.ctl` is the NONMEM 7.x `ADVAN13 TOL=9`
//!     FOCEI control; its final estimates are in `nonmem_anchor/results/`.
//!   - `data/transit_oral.csv` is the 20-subject / 240-observation single-dose
//!     oral dataset, simulated from the Savic model (truths `TVCL=5`, `TVV=50`,
//!     `TVKA=1`, `TVMTT=1`, `TVN=3`; IIV on CL & V ω²=0.09; prop SD 0.15).
//!
//! ## NONMEM cross-check
//!
//! From `nonmem_anchor/results/savic_transit.ext` (NONMEM, FOCEI):
//!
//! | Quantity | NONMEM | ferx (tight ODE tols) |
//! |----------|--------|-----------------------|
//! | OFV      | −1077.13 | −1076.67 |
//! | TVCL     | 5.386  | 5.453 |
//! | TVV      | 56.169 | 55.759 |
//! | TVKA     | 0.952  | 0.950 |
//! | TVMTT    | 0.965  | 0.966 |
//! | TVN      | 3.133  | 3.128 |
//!
//! ferx and NONMEM agree only when the ODE solver runs at NONMEM-equivalent
//! accuracy (`ode_reltol = ode_abstol = 1e-9`, matching `TOL=9`). The loose
//! defaults (`1e-4`/`1e-6`) recover the fixed effects but inject integration
//! noise into the FOCEI Hessian that inflates the ω² estimates (see
//! `docs/model-file/absorption.qmd` "Verification against NONMEM"). This
//! test therefore sets the tight tolerances explicitly.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// `nonmem_anchor/transit_savic_fit.ferx` minus the `[fit_options]` block, which
/// is set programmatically below so the test pins the ODE tolerances.
const MODEL_SRC: &str = r"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVKA(1.0,  0.05,  24.0)
  theta TVMTT(1.0, 0.05,  24.0)
  theta TVN(3.0,    0.1,  30.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09

  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  KA  = TVKA
  MTT = TVMTT
  NTR = TVN

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// FOCEI fit of the Savic transit model, at NONMEM-equivalent ODE accuracy, must
/// land on the NONMEM objective.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored Savic transit (#343/#322) acceptance: opt in with --features slow-tests"
)]
fn savic_transit_matches_nonmem_ofv() {
    // NONMEM #OBJV is −1077.13 (nonmem_anchor/results/savic_transit.ext). ferx
    // lands at −1076.67 on the FD path CI exercises, with the tight tolerances
    // set below; the ±2 band excludes the loose-tolerance regime (−1069.07,
    // ω² inflated ~60–120%) this test guards against.
    const EXPECTED_OFV: f64 = -1076.67;
    const TOLERANCE: f64 = 2.0;

    let model = parse_full_model(MODEL_SRC)
        .expect("Savic transit model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/transit_oral.csv"), None, None)
        .expect("transit_oral data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 500;
    opts.run_covariance_step = false;
    opts.verbose = false;
    // NONMEM-equivalent ODE accuracy (TOL=9). Without this the ω² inflate and
    // the OFV drifts ~8 units high — the regression this test pins.
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-9;
    opts.inner_tol = 1e-6;

    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("Savic transit fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - EXPECTED_OFV).abs() < TOLERANCE,
        "OFV {:.3} is outside the NONMEM-anchored band {:.2} ± {:.1} (NONMEM #OBJV \
         −1077.13, nonmem_anchor/results/savic_transit.ext); a value near −1069 \
         signals the loose-ODE-tolerance regression (inflated ω²) this guards",
        result.ofv,
        EXPECTED_OFV,
        TOLERANCE,
    );
}
