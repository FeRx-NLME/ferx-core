//! NONMEM cross-check for the built-in Freijer & Post inverse-Gaussian
//! absorption function `igd(mat, cv2)` (issue FeRx-NLME/ferx-core#347, #322).
//!
//! The model, data, and NONMEM control stream are the anchor kit in the
//! repository's `nonmem_anchor/` directory:
//!
//!   - `nonmem_anchor/freijer_ig.ctl` is the NONMEM 7.x `ADVAN13 $DES TOL=9`
//!     FOCEI control; its final estimates are in `nonmem_anchor/results/`.
//!   - `data/igd_oral.csv` is the 20-subject / 240-observation single-dose
//!     dataset shared with the Savic transit anchor, re-keyed to a 1-compartment
//!     layout (every record on CMT 1) so the dose feeds the `igd()` compartment
//!     directly. This is likelihood-equivalent to the NONMEM control's inert
//!     depot (CMT 1, `F1=0`) + central (CMT 2) with `PODO` driving `R_in` into
//!     central — same `R_in`, same predictions, same OFV.
//!
//! ## What this anchors — the likelihood, at the shared optimum
//!
//! The data were simulated from a Savic *transit* model, so the inverse-Gaussian
//! fit is mildly **mis-specified** (the IG approximates the transit
//! delayed-absorption shape). On that mis-specified objective the likelihood
//! surface has a long, flat ridge — NONMEM's gradient FOCEI climbs it from
//! `MAT≈2` to the optimum `MAT≈6.07` over ~30 iterations, whereas ferx's default
//! derivative-free outer optimiser (BOBYQA) takes small steps and stalls partway
//! up the ridge (≈ −881). Both are legitimate optimiser *paths* on a flat
//! surface; the path is not the implementation check. (This stall is the ODE
//! analogue of the fixed-EBE gradient bias that the analytic FOCE/FOCEI gradient
//! work — #367 / #381 — removes for analytical models; an exact ODE gradient via
//! sensitivity equations is the path to converging such fits.)
//!
//! The implementation check is **NONMEM-igd ≈ ferx-igd**: the two engines must
//! agree on the **objective at the same parameters**. This test therefore
//! evaluates ferx's full FOCEI marginal objective (inner EBE optimisation +
//! Laplace approximation + ODE integration of the `igd` forcing, with dose
//! routing / bolus suppression / superposition) **at NONMEM's reported optimum**
//! and asserts it matches NONMEM's `#OBJV`. This isolates the `igd` likelihood
//! from outer-optimiser path-dependence and is a *stronger* check than a
//! full-fit OFV match (which a likelihood bug could mask by landing elsewhere).
//!
//! From `nonmem_anchor/results/freijer_ig.ext` (NONMEM, FOCEI, MINIMIZATION
//! SUCCESSFUL): `#OBJV = −899.38`, `CL 5.612`, `V 33.95`, `MAT 6.071`,
//! `CV2 1.868`, `ω²(CL) 0.0401`, `ω²(V) 0.0484`, `σ²(prop) 0.0583`. Evaluating
//! ferx's FOCEI objective at exactly those values gives **−899.39** — agreement
//! to 0.02 units, confirming the `igd` density and its ODE machinery reproduce
//! the NONMEM `$DES` inverse-Gaussian input. (NONMEM-equivalent ODE accuracy
//! `ode_reltol = ode_abstol = 1e-9`, matching `TOL=9`, is required.)

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// 1-cpt inverse-Gaussian model, amount form + `[scaling] y = central/V` (the
/// NONMEM `A(2)` amount / `IPRED = A(2)/V` convention). Initial estimates are
/// **NONMEM's final values** (`nonmem_anchor/results/freijer_ig.ext`): the test
/// evaluates ferx's objective here without taking outer steps. The proportional
/// `sigma` is given as the SD `√0.0582643 = 0.2413795`.
const MODEL_AT_NONMEM_OPTIMUM: &str = r"
[parameters]
  theta TVCL(5.6119,   0.1, 100.0)
  theta TVV(33.9549,   5.0, 500.0)
  theta TVMAT(6.07130, 0.05,  24.0)
  theta TVCV2(1.86793, 0.001, 10.0)

  omega ETA_CL ~ 0.0400919
  omega ETA_V  ~ 0.0484205

  sigma PROP_ERR ~ 0.2413795 (sd)

[individual_parameters]
  CL  = TVCL  * exp(ETA_CL)
  V   = TVV   * exp(ETA_V)
  MAT = TVMAT
  CV2 = TVCV2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// ferx's FOCEI marginal objective for the `igd` model, evaluated at NONMEM's
/// optimum, must equal NONMEM's `#OBJV` — the NONMEM-igd ≈ ferx-igd check.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored Freijer IG (#347/#322) acceptance: opt in with --features slow-tests"
)]
fn igd_marginal_ofv_matches_nonmem_at_their_optimum() {
    // NONMEM #OBJV is −899.38 (nonmem_anchor/results/freijer_ig.ext, MINIMIZATION
    // SUCCESSFUL). ferx evaluates −899.39 at the same parameters on the FD path
    // CI exercises, with the tight ODE tolerances below; the ±1 band excludes any
    // wrong normalisation constant / off-by-units in the igd density (which would
    // shift the objective by many units).
    const NONMEM_OFV: f64 = -899.38;
    const TOLERANCE: f64 = 1.0;

    let model = parse_full_model(MODEL_AT_NONMEM_OPTIMUM)
        .expect("inverse-Gaussian model must parse")
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

    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("inverse-Gaussian fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {:.3} at NONMEM's optimum is outside the NONMEM-anchored \
         band {:.2} ± {:.1} (NONMEM #OBJV −899.38, nonmem_anchor/results/freijer_ig.ext); \
         a multi-unit gap signals a wrong igd density / normalisation",
        result.ofv,
        NONMEM_OFV,
        TOLERANCE,
    );
}
