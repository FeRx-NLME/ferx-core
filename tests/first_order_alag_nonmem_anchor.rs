//! NONMEM cross-check for the built-in `first_order(ka)` absorption forcing
//! combined with an **estimated lagtime** (`ALAG1`) on the event-driven ODE
//! sensitivity walk — PR2 of the #486 analytic-gradient-completion plan.
//!
//! ## Why `first_order`, not `igd`/`transit`/`weibull`
//!
//! `first_order`'s onset `R_in(0⁺) = dose·ka` is *always finite and nonzero*,
//! unlike `igd` (whose onset vanishes identically — an essential singularity
//! dominates for every valid `(mat, cv2)`) or `transit` at `n > 0` (also zero).
//! The new rate-on **saltation** the event-driven walk injects at a dose's lagged
//! arrival — mirroring the existing lagged-infusion rate-on injection, but with
//! `Δr = R_in(0⁺)` instead of a constant infusion rate — only has a *nonzero*
//! jump to get right for `first_order` (and the degenerate `transit(n=0)` case).
//! This is therefore the sharpest available NONMEM cross-check for that new
//! code path.
//!
//! ## Why a plain `ADVAN2 TRANS2` control, not `ADVAN13 $DES`
//!
//! ferx's `d/dt(central) = first_order(ka=KA) - CL/V*central` (the dose feeds the
//! forcing directly, no separate depot state) is mathematically **identical** to
//! the classic depot→central first-order absorption: the "input into central"
//! from an exponentially-decaying depot IS exactly `R_in(tad) = dose·ka·e^{-ka·tad}`.
//! So this anchors against a plain, standard `ADVAN2 TRANS2 + ALAG1` control
//! stream — no `$DES`/`PODO`/`TDOS` bookkeeping, no ODE solver at all on the
//! NONMEM side. (An `ADVAN13 $DES` control combining `igd()` with a manually
//! shifted `TDOS` was tried first and abandoned: NONMEM's `LSODA` integrator
//! choked on the resulting hand-rolled discontinuity even after forcing a
//! segment break via a declared `ALAG1`. `ADVAN2` sidesteps the issue entirely by
//! using NONMEM's own closed-form solution.)
//!
//! ## The anchor kit (`nonmem_anchor/`)
//!
//! - `simulate_first_order_alag.py` — deterministic simulator (pure stdlib,
//!   seed 486): 30 subjects, single 100 mg dose, IIV on `CL`/`KA`, proportional
//!   residual error, observations restricted to `t > TVLAG` (a `t < lag`
//!   observation has *exactly* zero predicted concentration, which is a genuine
//!   singularity for a purely proportional error model on either engine).
//! - `first_order_alag.ctl` — the NONMEM `ADVAN2 TRANS2` control (`ALAG1 =
//!   THETA(4)`); final estimates in `results/first_order_alag.{ext,lst}`.
//! - `first_order_alag_nm.csv` — NONMEM's copy (dose `CMT=1` = depot, obs
//!   `CMT=2` = central).
//! - `data/first_order_alag.csv` — the same simulation re-keyed to `CMT=1` for
//!   both dose and observations (ferx's single `central` state fed directly by
//!   the forcing) — likelihood-equivalent to the NONMEM layout.
//! - `first_order_alag_fit.ferx` — the matching ferx model.
//!
//! ## Result — independently converged (not evaluated at a shared point)
//!
//! Unlike the `igd`/`transit`/`weibull` anchors (mis-specified vs their shared
//! transit-simulated dataset, hence evaluated at NONMEM's optimum to sidestep a
//! flat-ridge optimiser-path difference), this dataset is simulated from the
//! *matched* model — both engines converge cleanly from their own defaults.
//! NONMEM (`MINIMIZATION SUCCESSFUL`): `OBJ = -1046.6853`, `CL = 5.17325`,
//! `V = 49.71960`, `KA = 1.02497`, `ALAG1 = 0.49929`, `ω²(CL) = 0.066478`,
//! `ω²(KA) = 0.024357`, `σ²(prop) = 0.009762`. ferx (this test, FOCEI, default
//! BOBYQA outer): `OFV = -1046.6853`, `CL = 5.173217`, `V = 49.719630`,
//! `KA = 1.024977`, `ALAG1 = 0.499291`, `ω²(CL) = 0.066477`, `ω²(KA) = 0.024362`,
//! `σ²(prop) = 0.009761`. Agreement to ~1e-4 relative on every parameter and to
//! 8 decimal places on the OFV.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const MODEL: &str = r"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0,  0.05, 20.0)
  theta TVLAG(0.5, 0.01, 5.0)

  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.04

  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA * exp(ETA_KA)
  ALAG1 = TVLAG

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - CL/V*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// ferx's independently-converged FOCEI fit for `first_order(ka)` + an
/// estimated compartment-indexed lag must match NONMEM's independently-
/// converged `ADVAN2 TRANS2 + ALAG1` fit on the same (matched-truth) data.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored first_order()+ALAG1 (#486 PR2) acceptance: opt in with --features slow-tests"
)]
fn first_order_alag_focei_matches_nonmem() {
    const NONMEM_OFV: f64 = -1046.6853;
    const OFV_TOLERANCE: f64 = 0.5;

    let model = parse_full_model(MODEL)
        .expect("first_order()+ALAG1 model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new("data/first_order_alag.csv"), None, None)
        .expect("first_order_alag data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.run_covariance_step = false;
    opts.verbose = false;
    // NONMEM-equivalent ODE accuracy.
    opts.ode_reltol = 1e-9;
    opts.ode_abstol = 1e-11;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("first_order()+ALAG1 fit must converge");

    assert!(
        (result.ofv - NONMEM_OFV).abs() < OFV_TOLERANCE,
        "ferx FOCEI OFV {:.4} vs NONMEM {:.4} (band ±{OFV_TOLERANCE}) — a gap here signals a \
         wrong lagtime + input-rate gradient (or value) on the event-driven ODE walk (#486 PR2)",
        result.ofv,
        NONMEM_OFV,
    );

    // NONMEM theta order: CL, V, KA, ALAG1.
    let nm_theta = [5.17325, 49.71960, 1.02497, 0.49929];
    let names = ["TVCL", "TVV", "TVKA", "TVLAG"];
    for (i, (&nm, &name)) in nm_theta.iter().zip(names.iter()).enumerate() {
        let ferx_val = result.theta[i];
        let rel_err = (ferx_val - nm).abs() / nm.abs();
        assert!(
            rel_err < 0.02,
            "{name}: ferx {ferx_val:.6} vs NONMEM {nm:.6} — relative error {rel_err:.4} exceeds 2%"
        );
    }
}
