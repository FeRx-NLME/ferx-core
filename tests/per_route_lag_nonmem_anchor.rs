//! NONMEM anchor for **per-route absorption lag** — the `lag=` argument on an
//! input-rate function giving each pathway its own onset delay.
//!
//! A licensed NONMEM `ADVAN13 TOL=9` `$DES` run (committed under `nonmem_anchor/`)
//! fits an immediate-release + delayed-release model — two first-order inputs where
//! the DR pathway's `$DES` input is shifted by a per-route lag `LAG2` (the same
//! F1=0 + PODO trick as the parallel anchor, with `TAD2 = TAD - LAG2`). The dataset
//! `per_route_lag_oral.csv` is simulated from that model (so NONMEM recovers the
//! truths: it lands `LAG2 = 3.074` from a truth of 3.0, `FR1 = 0.501`, `KA1 = 1.221`,
//! `KA2 = 0.811`), and the fit is matched / well-specified.
//!
//! The check evaluates ferx's FOCEI marginal objective **at NONMEM's optimum** (no
//! outer steps, NONMEM-equivalent ODE accuracy) against NONMEM's `#OBJV` — a
//! path-independent implementation check that isolates the per-route-lag mechanism.
//! (Since #859 a `first_order` per-route lag is fit on the exact analytic path, but the
//! objective here is an *evaluation* — the predictions are exact regardless of the
//! gradient route, so this pins the same absorption mechanism the analytic parallel/mixed
//! anchors do.)
//! Final NONMEM values are read from `nonmem_anchor/results/per_route_lag.ext`.
//!
//! This same static model is what the **closed-form modified-release** path (#860) serves for
//! `predict()` / `simulate()` — a superposition of shifted single-route Bateman terms, no
//! integration. That closed form is anchored to NONMEM *transitively*: the reduction-to-ODE
//! tests (`pk::modified_release::tests::reduces_to_ode_1cpt_per_route_lag`) pin it bit-for-bit
//! (to solver tolerance) against this same integrated `[odes]` twin, and this test pins the
//! twin against NONMEM's `#OBJV` — so closed form ≡ twin ≡ NONMEM.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// 1-cpt IR + delayed-release (per-route lag) model at **NONMEM's final values**
/// (`nonmem_anchor/results/per_route_lag.ext`). `FR2 = 1 - FR1` is the declared
/// complement; the DR route carries its own `lag=LAG2`. `KA1 > KA2` bounds and the
/// `LAG2 > 0` lag match the control's IR/DR convention. The proportional `sigma` is
/// the SD `√0.0231509 = 0.152154` (NONMEM stores the variance).
const PER_ROUTE_LAG_AT_NONMEM_OPTIMUM: &str = r"
[parameters]
  theta TVCL(4.73922,    0.1, 100.0)
  theta TVV(49.2335,     5.0, 500.0)
  theta TVFR1(0.500693, 0.001, 0.999)
  theta TVKA1(1.22062,   0.3,  24.0)
  theta TVKA2(0.811426, 0.05,   1.1)
  theta TVLAG2(3.07375, 0.001, 12.0)

  omega ETA_CL ~ 0.0671036
  omega ETA_V  ~ 0.147961

  sigma PROP_ERR ~ 0.152154 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  FR1  = TVFR1
  FR2  = 1 - TVFR1
  KA1  = TVKA1
  KA2  = TVKA2
  LAG2 = TVLAG2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2, lag=LAG2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Evaluate ferx's FOCEI marginal objective at NONMEM's optimum (no outer steps,
/// NONMEM-equivalent ODE accuracy) on the ferx-keyed matched dataset.
fn ferx_ofv_at_optimum(model_src: &str, data: &str) -> f64 {
    let model = parse_full_model(model_src)
        .expect("anchor model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new(data), None, None).expect("anchor data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 0; // evaluate at the optimum, don't take outer steps
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.ode_reltol = 1e-9; // NONMEM TOL=9
    opts.ode_abstol = 1e-9;
    opts.inner_tol = 1e-6;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("anchor fit must run");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    result.ofv
}

/// Per-route lag: ferx's FOCEI objective at NONMEM's optimum must equal NONMEM's
/// `#OBJV = −882.357` (minimization terminated on rounding error ERROR=134, benign —
/// all parameters recovered, gradients small). ferx evaluates −882.3568 at the same
/// parameters — agreement ~1e-6, the smooth-`first_order` precision of the parallel
/// anchor. The ±0.5 band is many orders above that, excluding any wrong lag placement
/// (a mis-shifted DR onset would move the objective by tens of units) or fraction
/// weighting.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored per-route absorption lag: opt in with --features slow-tests"
)]
fn per_route_lag_marginal_ofv_matches_nonmem_at_their_optimum() {
    const NONMEM_OFV: f64 = -882.357;
    const TOLERANCE: f64 = 0.5;
    let ofv = ferx_ofv_at_optimum(
        PER_ROUTE_LAG_AT_NONMEM_OPTIMUM,
        "data/per_route_lag_oral.csv",
    );
    assert!(
        (ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {ofv:.4} at NONMEM's optimum is outside the NONMEM-anchored \
         band {NONMEM_OFV} ± {TOLERANCE} (per-route absorption lag)"
    );
}
