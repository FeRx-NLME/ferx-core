//! NONMEM cross-check for the **biphasic** inverse-Gaussian absorption — ferx's
//! pathway-fraction mechanism `FR1*igd(...) + FR2*igd(...)` (issue
//! FeRx-NLME/ferx-core#388, parent #322).
//!
//! The model, data, and NONMEM control stream are the anchor kit in the
//! repository's `nonmem_anchor/` directory:
//!
//!   - `nonmem_anchor/freijer_biphasic_ig.ctl` is the NONMEM 7.x `ADVAN13 $DES
//!     TOL=9` FOCEI control whose `$DES` sums two inverse-Gaussian pathways split
//!     by `FR1`/`(1-FR1)`; its final estimates are in `nonmem_anchor/results/`.
//!   - `data/biphasic_ig_oral.csv` is the 20-subject / 240-observation single-dose
//!     dataset, re-keyed to a 1-compartment layout (every record on CMT 1) so the
//!     dose feeds the absorption compartment directly. It is likelihood-equivalent
//!     to the NONMEM control's inert depot (CMT 1, `F1=0`) + central (CMT 2) with
//!     `PODO` driving the IG sum into central — same `R_in`, same predictions,
//!     same OFV.
//!
//! ## What this anchors — the likelihood, at the shared optimum
//!
//! Unlike the single-IG / Weibull anchors (which reuse transit-truth data and are
//! therefore *mis-specified*), this dataset is **simulated from the biphasic model
//! itself** (`nonmem_anchor/simulate_biphasic_ig_data.py`, same truths), so the fit
//! is well-specified: NONMEM both recovers the data-generating values
//! (`MINIMIZATION SUCCESSFUL`: `FR1 0.644`, fast `MAT1 0.528`, slow `MAT2 4.124`,
//! with the `MAT1 < MAT2` bounds breaking the pathway-label symmetry) **and** agrees
//! with ferx on the objective.
//!
//! The implementation check is **NONMEM-biphasic ≈ ferx-biphasic**: the two engines
//! must agree on the **objective at the same parameters**. This test evaluates
//! ferx's full FOCEI marginal objective (inner EBE optimisation + Laplace + ODE
//! integration of the *summed, fraction-weighted* IG forcing) **at NONMEM's reported
//! optimum** and asserts it matches NONMEM's `#OBJV` — isolating the fraction
//! mechanism + IG density from outer-optimiser path-dependence.
//!
//! From `nonmem_anchor/results/freijer_biphasic_ig.ext` (NONMEM, FOCEI,
//! MINIMIZATION SUCCESSFUL): `#OBJV = −754.211`, `CL 5.366`, `V 56.94`,
//! `FR1 0.6435`, `MAT1 0.5281`, `MAT2 4.124`, `CV2_1 0.2188`, `CV2_2 0.3453`,
//! `ω²(CL) 0.0480`, `ω²(V) 0.0429`, `σ²(prop) 0.02493`. Evaluating ferx's FOCEI
//! objective at exactly those values gives **−754.2113** — agreement to ~1e-5,
//! confirming the pathway-fraction superposition and the IG density reproduce the
//! NONMEM `$DES` biphasic input. (NONMEM-equivalent ODE accuracy
//! `ode_reltol = ode_abstol = 1e-9`, matching `TOL=9`, is required.)

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// 1-cpt biphasic inverse-Gaussian model, amount form + `[scaling] y = central/V`
/// (the NONMEM `A(2)` amount / `IPRED = A(2)/V` convention). Initial estimates are
/// **NONMEM's final values** (`nonmem_anchor/results/freijer_biphasic_ig.ext`): the
/// test evaluates ferx's objective here without taking outer steps. The
/// proportional `sigma` is given as the SD `√0.0249290 = 0.1578892`. `FR2 = 1 - FR1`
/// is the declared-complement pattern; the `MAT1 < MAT2` bounds match the control's
/// pathway-label convention.
const MODEL_AT_NONMEM_OPTIMUM: &str = r"
[parameters]
  theta TVCL(5.36634,    0.1, 100.0)
  theta TVV(56.9366,     5.0, 500.0)
  theta TVFR1(0.643546, 0.001, 0.999)
  theta TVMAT1(0.528110, 0.05,  2.0)
  theta TVMAT2(4.12396,   2.0, 24.0)
  theta TVCV2_1(0.218819, 0.001, 10.0)
  theta TVCV2_2(0.345333, 0.001, 10.0)

  omega ETA_CL ~ 0.0480105
  omega ETA_V  ~ 0.0429064

  sigma PROP_ERR ~ 0.1578892 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV  * exp(ETA_V)
  FR1   = TVFR1
  FR2   = 1 - TVFR1
  MAT1  = TVMAT1
  MAT2  = TVMAT2
  CV2_1 = TVCV2_1
  CV2_2 = TVCV2_2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) + FR2*igd(mat=MAT2, cv2=CV2_2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// ferx's FOCEI marginal objective for the biphasic `igd` model, evaluated at
/// NONMEM's optimum, must equal NONMEM's `#OBJV` — the NONMEM ≈ ferx check for the
/// pathway-fraction mechanism (#388).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored biphasic IG (#388/#322) acceptance: opt in with --features slow-tests"
)]
fn biphasic_igd_marginal_ofv_matches_nonmem_at_their_optimum() {
    // NONMEM #OBJV is −754.211 (nonmem_anchor/results/freijer_biphasic_ig.ext,
    // MINIMIZATION SUCCESSFUL). ferx evaluates −754.2113 at the same parameters with
    // the tight ODE tolerances below — agreement to ~1e-5. The ±0.5 band is many
    // orders above that, excluding any wrong fraction weighting / IG normalisation
    // (which would shift the objective by tens of units).
    const NONMEM_OFV: f64 = -754.211;
    const TOLERANCE: f64 = 0.5;

    let model = parse_full_model(MODEL_AT_NONMEM_OPTIMUM)
        .expect("biphasic inverse-Gaussian model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/biphasic_ig_oral.csv"), None, None)
        .expect("biphasic_ig_oral data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    // Evaluate the marginal objective at NONMEM's optimum without taking outer steps
    // (the implementation check is the objective at the optimum, not the path).
    opts.outer_maxiter = 0;
    opts.run_covariance_step = false;
    opts.verbose = false;
    // NONMEM-equivalent ODE accuracy (TOL=9).
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-9;
    opts.inner_tol = 1e-6;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("biphasic inverse-Gaussian fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {:.4} at NONMEM's optimum is outside the NONMEM-anchored \
         band {:.3} ± {:.1} (NONMEM #OBJV −754.211, nonmem_anchor/results/freijer_biphasic_ig.ext); \
         a multi-unit gap signals a wrong pathway-fraction weighting / igd density",
        result.ofv,
        NONMEM_OFV,
        TOLERANCE,
    );
}
