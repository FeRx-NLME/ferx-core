//! NONMEM cross-check for **per-compartment bioavailability and lag**
//! (`F1`/`F2`, `ALAG2`) on a model dosed into **two** compartments
//! (issue #369 — the follow-up anchor flagged in PR #371's open questions).
//!
//! The existing `bioavailability_ode_nonmem` anchor only exercises a single
//! dose route (`F1` on the depot). It cannot catch the failure mode this PR
//! fixes: before the `DoseAttrMap`, a dose into a *second* compartment silently
//! reused compartment 1's `F`/lag. This file pins the new routing by dosing one
//! subject **both** orally into the depot (CMT=1, bioavailability `F1`) **and**
//! as an IV bolus into the central compartment (CMT=2, bioavailability `F2` with
//! absorption lag `ALAG2`), then checks the central concentration.
//!
//! ## Reference
//!
//! `REFERENCE` below is the **exact closed-form** solution of this mixed-route
//! one-compartment model — a superposition of the oral Bateman term and the
//! lagged IV bolus term — computed independently of ferx's ODE integrator (and
//! checked against it by `reference_equals_closed_form`). It was **confirmed
//! against a real NONMEM 7.5.1 run** (`nmfe75`) of the control file committed at
//! `tests/nonmem/two_cmt_dose.ctl` — ADVAN2 TRANS2, `$ESTIMATION MAXEVAL=0`
//! (fixed θ, η = 0, `S2 = V`): NONMEM's `PRED` (sdtab1, transcribed below)
//! matches `REFERENCE` to all of NONMEM's printed digits. So the ferx ODE
//! engine, NONMEM, and the analytic solution all agree.
//!
//! ### `two_cmt_dose.ctl` — ADVAN2 TRANS2; CL=5 V=50 KA=1.5 F1=0.70 F2=0.40 ALAG2=2.0
//! ```text
//! $SUBROUTINES ADVAN2 TRANS2
//! $PK
//!   CL=THETA(1) V=THETA(2) KA=THETA(3)
//!   F1=THETA(4) F2=THETA(5) ALAG2=THETA(6) S2=V
//! $ERROR
//!   IPRED=F  Y=IPRED*(1+EPS(1))
//! $THETA 5 FIX 50 FIX 1.5 FIX 0.70 FIX 0.40 FIX 2.0 FIX
//! $OMEGA 0 FIX  $SIGMA 0.01 FIX
//! $ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
//! ```
//! Subject dosed `AMT=100, CMT=1` (oral) **and** `AMT=50, CMT=2` (IV), both at
//! t=0; observations on the central compartment (CMT=2). NONMEM `PRED` (sdtab1):
//! 0.71829, 1.0226, 1.1537, 1.5134, 1.3293, 0.89351, 0.59894, 0.32871, 0.18040
//! at t = 0.5, 1, 1.9, 2.5, 4, 8, 12, 18, 24 — matching `REFERENCE` to 5 sig figs.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::path::Path;

/// Two-compartment-*dose* ODE model: amount-based central state with the
/// concentration read-out provided by Form-C scaling (`y = central / V`), so a
/// direct IV bolus into `central` (CMT=2) adds an amount, not a concentration.
/// `F1`/`F2` and `ALAG2` are fixed thetas (η = 0) → individual parameters keyed
/// by dose compartment via the `DoseAttrMap` under test.
const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1,  50.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.5, 0.05,  20.0)
  theta THETA_F1(0.70, 0.001, 0.999)
  theta THETA_F2(0.40, 0.001, 0.999)
  theta THETA_LAG2(2.0, 0.0, 24.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  KA    = TVKA
  F1    = THETA_F1
  F2    = THETA_F2
  ALAG2 = THETA_LAG2

[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// Closed-form central concentration keyed by observation time.
// Pre-lag (t < ALAG2 = 2): oral term only. Post-lag: oral + IV bolus.
const REFERENCE: &[(f64, f64)] = &[
    (0.5, 0.718294),  // oral-only
    (1.0, 1.022561),  // oral-only
    (1.9, 1.153672),  // oral-only (just before the cmt-2 dose lands)
    (2.5, 1.513416),  // oral + IV (just after)
    (4.0, 1.329254),  // oral + IV
    (8.0, 0.893509),  // oral + IV
    (12.0, 0.598943), // oral + IV
    (18.0, 0.328707), // oral + IV
    (24.0, 0.180398), // oral + IV
];

/// Independent analytical solution: oral Bateman into central + lagged IV bolus.
/// CL=5, V=50, KA=1.5 (k = CL/V = 0.1); D1 = F1·100 = 70, D2 = F2·50 = 20,
/// lag = ALAG2 = 2.
fn closed_form(t: f64) -> f64 {
    let (k, ka, v) = (0.1_f64, 1.5_f64, 50.0_f64);
    let (d1, d2, lag) = (70.0_f64, 20.0_f64, 2.0_f64);
    let oral = d1 * ka / (ka - k) * ((-k * t).exp() - (-ka * t).exp());
    let iv = if t >= lag {
        d2 * (-k * (t - lag)).exp()
    } else {
        0.0
    };
    (oral + iv) / v
}

#[test]
fn reference_equals_closed_form() {
    // The hardcoded REFERENCE is the exact analytical solution (so a transcription
    // slip can't masquerade as a NONMEM mismatch). This is what NONMEM ADVAN2
    // reproduces to machine precision; see two_cmt_dose.ctl.
    for &(t, expected) in REFERENCE {
        let cf = closed_form(t);
        let rel = (cf - expected).abs() / expected;
        assert!(
            rel < 1e-5,
            "t={t}: REFERENCE {expected:.6} vs closed form {cf:.6} (rel {rel:.2e})"
        );
    }
}

#[test]
fn ode_two_cmt_dose_matches_nonmem() {
    let model = parse_full_model(MODEL).expect("model parses").model;
    let population =
        read_nonmem_csv(Path::new("data/two_cmt_dose_ref.csv"), None, None).expect("dataset loads");

    let preds = predict(&model, &population, &model.default_params);
    assert_eq!(preds.len(), REFERENCE.len());

    for (p, &(t, expected)) in preds.iter().zip(REFERENCE) {
        assert!(
            (p.time - t).abs() < 1e-9,
            "prediction time {} != expected {t}",
            p.time
        );
        let rel = (p.pred - expected).abs() / expected;
        assert!(
            rel < 1e-4,
            "t={t}: ferx ODE PRED {:.6} vs NONMEM/closed-form {expected:.6} (rel err {rel:.2e})",
            p.pred
        );
    }
}
