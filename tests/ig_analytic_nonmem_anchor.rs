//! NONMEM cross-check for the **analytic** inverse-Gaussian absorption closed form
//! — the structural model `pk one_cpt_ig(cl, v, mat, cv2)` (#790).
//!
//! Unlike `tests/igd_nonmem_anchor.rs` (which anchors the numerical `igd()` ODE
//! forcing on transit-truth data, whose IG optimum sits in the *flip-flop* regime
//! where the closed form would reroute to its ODE twin), this test uses an
//! **IG-truth, in-domain** dataset so the exponential-tilting **closed form itself
//! actually runs** and is anchored directly against NONMEM.
//!
//! The model, data, and NONMEM control are the anchor kit under `nonmem_anchor/`:
//!
//!   - `nonmem_anchor/ig_indomain.ctl` is the NONMEM 7.x `ADVAN13 $DES TOL=9`
//!     FOCEI control; `nonmem_anchor/ig_indomain_sim.ctl` simulates the data from
//!     the same `$DES` IG density with in-domain truth (`CL 5, V 50, MAT 2,
//!     CV2 0.3`, so `ke = CL/V = 0.10 < 1/(2·MAT·CV²) = 0.833`). Final estimates are
//!     in `nonmem_anchor/results/ig_indomain.*`.
//!   - `data/ig_indomain_oral.csv` is the 20-subject / 240-observation dataset
//!     (dose CMT 1 into the depot, observations CMT 2 = central) — the same layout
//!     the analytic `pk one_cpt_ig` uses ([depot, central], central = CMT 2).
//!
//! ## What this anchors — the likelihood of the closed form, at the shared optimum
//!
//! The data are simulated from the IG model itself, so the fit is well-specified and
//! lands **inside** the tilting convergence domain: at NONMEM's optimum
//! `MAT 1.993, CV2 0.298` the abscissa `1/(2·MAT·CV²) = 0.841` exceeds
//! `ke = CL/V = 5.312/48.19 = 0.110`, so `pk one_cpt_ig` evaluates the analytic
//! exponential-tilting closed form (no ODE, no flip-flop reroute). Evaluating ferx's
//! full FOCEI marginal objective for that closed form **at NONMEM's reported
//! optimum** must match NONMEM's `#OBJV` — the direct NONMEM-IG ≈ ferx-analytic-IG
//! check, isolating the closed-form likelihood from outer-optimiser path-dependence.
//!
//! From `nonmem_anchor/results/ig_indomain.ext` (NONMEM, FOCEI, MINIMIZATION
//! SUCCESSFUL): `#OBJV = −1303.639` (WITHOUT CONSTANT — ferx's convention, per the
//! sibling `igd_nonmem_anchor`), `CL 5.31216`, `V 48.1911`, `MAT 1.99315`,
//! `CV2 0.298123`, `ω²(CL) 0.0323524`, `ω²(V) 0.124`, `σ²(prop) 0.0240620`. Evaluating
//! ferx's analytic-IG FOCEI objective at exactly those values gives **−1303.528** —
//! agreement to ~0.11 units (the exact closed form vs NONMEM's numerically-integrated
//! `$DES`, plus the two engines' FOCEI-Laplace inner solves), confirming the IG
//! closed-form density and normalisation reproduce the NONMEM `$DES` result.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// The `[parameters]` + `[individual_parameters]` preamble at **NONMEM's final values**
/// (`nonmem_anchor/results/ig_indomain.ext`), shared verbatim by the analytic and ODE
/// forms below so the "same optimum" invariant the triangle relies on can't drift. The
/// proportional `sigma` is the SD `√0.0240620 = 0.1551193`. Observations are CMT 2
/// (central), matching the `pk one_cpt_ig` [depot, central] layout.
const PREAMBLE: &str = r"
[parameters]
  theta TVCL(5.31216,  0.1, 100.0)
  theta TVV(48.1911,   5.0, 500.0)
  theta TVMAT(1.99315, 0.05, 24.0)
  theta TVCV2(0.298123, 0.001, 10.0)

  omega ETA_CL ~ 0.0323524
  omega ETA_V  ~ 0.124

  sigma PROP_ERR ~ 0.1551193 (sd)

[individual_parameters]
  CL  = TVCL  * exp(ETA_CL)
  V   = TVV   * exp(ETA_V)
  MAT = TVMAT
  CV2 = TVCV2
";

const ERROR_MODEL: &str = "\n[error_model]\n  DV ~ proportional(PROP_ERR)\n";

/// Analytic 1-cpt IG model (the closed form) at NONMEM's optimum.
fn analytic_model() -> String {
    format!(
        "{PREAMBLE}\n[structural_model]\n  pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)\n{ERROR_MODEL}"
    )
}

/// The **ODE `igd()` forcing** form at the *same* optimum, so the triangle
/// NONMEM ≈ ferx-ODE ≈ ferx-analytic is pinned on **one** in-domain dataset (the sibling
/// `igd_nonmem_anchor` anchors the ODE on a *different*, flip-flop dataset).
/// `ode(obs_cmt=central, …)` reads central regardless of the CMT column.
fn ode_model() -> String {
    format!(
        "{PREAMBLE}\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n\n\
         [odes]\n  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central\n\n\
         [scaling]\n  obs_scale = V\n{ERROR_MODEL}"
    )
}

/// ferx's FOCEI marginal objective for the analytic `pk one_cpt_ig` closed form,
/// evaluated at NONMEM's in-domain optimum, must equal NONMEM's `#OBJV`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored analytic IG (#790) acceptance: opt in with --features slow-tests"
)]
fn ig_analytic_marginal_ofv_matches_nonmem_at_their_optimum() {
    // NONMEM #OBJV (WITHOUT CONSTANT) is −1303.639 (MINIMIZATION SUCCESSFUL). The
    // ±1 band excludes any wrong normalisation / off-by-units in the IG closed form
    // (which would shift the objective by many units), while allowing the small
    // FOCEI-Laplace difference between the two engines' inner solves.
    const NONMEM_OFV: f64 = -1303.639;
    const TOLERANCE: f64 = 1.0;

    let model = parse_full_model(&analytic_model())
        .expect("analytic inverse-Gaussian model must parse")
        .model;
    // The model must be the closed form (not silently rerouted to an ODE at parse).
    assert!(
        model.ode_spec.is_none(),
        "pk one_cpt_ig must be the analytic closed form, not an ODE"
    );
    let pop = read_nonmem_csv(Path::new("data/ig_indomain_oral.csv"), None, None)
        .expect("ig_indomain data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    // Evaluate the marginal objective at NONMEM's optimum without outer steps.
    opts.outer_maxiter = 0;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.inner_tol = 1e-6;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("analytic IG fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {:.3} for the analytic `pk one_cpt_ig` closed form at NONMEM's \
         in-domain optimum is outside the NONMEM-anchored band {:.2} ± {:.1} (NONMEM #OBJV \
         −1303.639, nonmem_anchor/results/ig_indomain.ext); a multi-unit gap signals a wrong \
         IG closed-form density / normalisation",
        result.ofv,
        NONMEM_OFV,
        TOLERANCE,
    );
}

/// Triangle closure: the ferx **ODE `igd()`** form, at the same optimum on the same
/// in-domain data, must also match NONMEM's `#OBJV`. Together with the analytic test
/// above (and the closed-form ≡ ODE equivalence suite), this pins NONMEM ≈ ferx-ODE ≈
/// ferx-analytic on a single dataset where the closed form is in-domain.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored IG ODE triangle (#790) acceptance: opt in with --features slow-tests"
)]
fn ig_ode_marginal_ofv_matches_nonmem_at_their_optimum() {
    const NONMEM_OFV: f64 = -1303.639;
    const TOLERANCE: f64 = 1.0;

    let model = parse_full_model(&ode_model())
        .expect("ODE inverse-Gaussian model must parse")
        .model;
    assert!(model.ode_spec.is_some(), "must be the ODE `igd()` form");
    let pop = read_nonmem_csv(Path::new("data/ig_indomain_oral.csv"), None, None)
        .expect("ig_indomain data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 0;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.inner_tol = 1e-6;
    // NONMEM-equivalent ODE accuracy (TOL=9).
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-9;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("ODE IG fit must run");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {:.3} for the ODE `igd()` form at NONMEM's in-domain optimum is \
         outside the NONMEM-anchored band {:.2} ± {:.1}",
        result.ofv,
        NONMEM_OFV,
        TOLERANCE,
    );
}
