//! NONMEM anchor for **parallel / mixed dual-pathway absorption** (#505) — the
//! `first_order(ka)` composition and the zero-order-channel fraction.
//!
//! Two licensed NONMEM `ADVAN13 TOL=9` `$DES` runs (committed under
//! `nonmem_anchor/`):
//!   - `nonmem_anchor/parallel_first_order.ctl` — `$DES` sums two first-order
//!     pathways `FR1*KA1*e^{-KA1*t} + FR2*KA2*e^{-KA2*t}` (PODO split, F1=0).
//!   - `nonmem_anchor/mixed_zero_first.ctl` — `$DES` sums a first-order pathway and
//!     a zero-order rectangle `FZO1*KA*e^{-KA*t} + FZO*PODO/DUR·1{t≤DUR}`.
//! Each runs on its own **matched** dataset (`{parallel,mixed}_oral.csv`, simulated
//! from the model itself), so NONMEM recovers the truths and the fit is
//! well-specified. The committed outputs are in `nonmem_anchor/results/`.
//!
//! Like the biphasic-IG anchor (`biphasic_igd_nonmem_anchor.rs`), the check is the
//! ferx FOCEI **marginal objective evaluated at NONMEM's optimum** (no outer steps)
//! against NONMEM's `#OBJV` — a path-independent implementation check that isolates
//! the absorption mechanism. Final NONMEM values are read from
//! `nonmem_anchor/results/{parallel_first_order,mixed_zero_first}.ext`.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// 1-cpt parallel dual-first-order model at **NONMEM's final values**
/// (`nonmem_anchor/results/parallel_first_order.ext`, MINIMIZATION SUCCESSFUL).
/// `FR2 = 1 - FR1` is the declared complement; `KA1 > KA2` bounds match the
/// control's pathway-label convention. The proportional `sigma` is the SD
/// `√0.0252118 = 0.158782` (NONMEM stores the variance).
const PARALLEL_AT_NONMEM_OPTIMUM: &str = r"
[parameters]
  theta TVCL(5.39863,    0.1, 100.0)
  theta TVV(55.3038,     5.0, 500.0)
  theta TVFR1(0.639398, 0.001, 0.999)
  theta TVKA1(1.36719,   0.5,  24.0)
  theta TVKA2(0.255311, 0.01,   0.5)

  omega ETA_CL ~ 0.047579
  omega ETA_V  ~ 0.043025

  sigma PROP_ERR ~ 0.158782 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  FR1 = TVFR1
  FR2 = 1 - TVFR1
  KA1 = TVKA1
  KA2 = TVKA2

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// 1-cpt mixed zero+first-order model at **NONMEM's final values**
/// (`nonmem_anchor/results/mixed_zero_first.ext`, MINIMIZATION SUCCESSFUL —
/// covariance step flagged, see the run's `.lst`). `FZO1 = 1 - FZO` is the declared
/// complement. The proportional `sigma` is the SD `√0.0252075 = 0.158769`.
const MIXED_AT_NONMEM_OPTIMUM: &str = r"
[parameters]
  theta TVCL(5.38838,    0.1, 100.0)
  theta TVV(55.6332,     5.0, 500.0)
  theta TVFZO(0.292718, 0.001, 0.999)
  theta TVKA(0.863735,  0.05,  24.0)
  theta TVDUR(3.04388,  0.05,  24.0)

  omega ETA_CL ~ 0.047828
  omega ETA_V  ~ 0.042666

  sigma PROP_ERR ~ 0.158769 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  FZO  = TVFZO
  FZO1 = 1 - TVFZO
  KA   = TVKA
  DUR  = TVDUR

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR) - CL/V*central

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

/// Parallel (dual first-order): ferx's FOCEI objective at NONMEM's optimum must
/// equal NONMEM's `#OBJV = −688.019` (MINIMIZATION SUCCESSFUL). ferx evaluates
/// −688.0194 at the same parameters — agreement ~1e-5. The ±0.5 band is many orders
/// above that, excluding any wrong fraction weighting / first-order normalisation
/// (which would shift the objective by tens of units, as the off-channel −661 of an
/// earlier mis-keyed eval showed).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored parallel absorption (#505/#322): opt in with --features slow-tests"
)]
fn parallel_marginal_ofv_matches_nonmem_at_their_optimum() {
    const NONMEM_OFV: f64 = -688.019;
    const TOLERANCE: f64 = 0.5;
    let ofv = ferx_ofv_at_optimum(PARALLEL_AT_NONMEM_OPTIMUM, "data/parallel_oral.csv");
    assert!(
        (ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {ofv:.4} at NONMEM's optimum is outside the NONMEM-anchored \
         band {NONMEM_OFV} ± {TOLERANCE} (parallel dual first-order, #505)"
    );
}

/// Mixed (zero + first-order): ferx's FOCEI objective at NONMEM's optimum must equal
/// NONMEM's `#OBJV = −698.966`. ferx evaluates −698.9662 — agreement ~1e-4 (looser
/// than the smooth parallel/biphasic anchors, as expected: the zero-order pathway's
/// hard `tad ≤ DUR` cutoff is integrated slightly differently by ferx's per-segment
/// break vs NONMEM's adaptive `$DES`, and NONMEM flagged its covariance step). Still
/// far inside ±0.5.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored mixed absorption (#505/#322): opt in with --features slow-tests"
)]
fn mixed_marginal_ofv_matches_nonmem_at_their_optimum() {
    const NONMEM_OFV: f64 = -698.966;
    const TOLERANCE: f64 = 0.5;
    let ofv = ferx_ofv_at_optimum(MIXED_AT_NONMEM_OPTIMUM, "data/mixed_oral.csv");
    assert!(
        (ofv - NONMEM_OFV).abs() < TOLERANCE,
        "ferx FOCEI objective {ofv:.4} at NONMEM's optimum is outside the NONMEM-anchored \
         band {NONMEM_OFV} ± {TOLERANCE} (mixed zero+first-order, #505)"
    );
}
