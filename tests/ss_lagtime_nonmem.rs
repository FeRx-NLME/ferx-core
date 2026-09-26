//! End-to-end NONMEM cross-check for steady-state (`SS=1`) dosing combined
//! with an absorption lagtime (`ALAG1`) — the acceptance test for issue #15.
//!
//! Exercises the public path parser → NONMEM CSV reader → `predict()` →
//! analytical 1-cpt oral SS closed form, and asserts the population
//! predictions (eta = 0) match NONMEM 7.5.1 to 1e-4 relative. The per-path
//! coverage (analytical / event-driven / ODE, including the previous-interval
//! tail for samples earlier than the lagtime) lives in the unit tests:
//!   - `src/pk/mod.rs::test_ss_oral_with_lagtime_matches_nonmem` (analytical)
//!   - `src/pk/event_driven.rs::event_driven_ss_iv_bolus_with_lagtime_matches_nonmem`
//!   - `src/ode/predictions.rs::ode_ss_iv_bolus_with_lagtime_matches_nonmem`
//!
//! ## Reproducing the NONMEM reference
//!
//! The reference PRED values baked into `data/ss_oral_lagtime.csv` (the `DV`
//! column) were produced with NONMEM 7.5.1, `$ESTIMATION MAXEVAL=0` (no
//! estimation — pure evaluation at the fixed thetas), from this control file:
//!
//! ```text
//! $PROBLEM 1-cpt oral SS + ALAG1 reference for ferx-core issue #15
//! $DATA data.csv IGNORE=@
//! $INPUT ID TIME DV EVID AMT CMT RATE MDV II SS
//! $SUBROUTINES ADVAN2 TRANS2
//! $PK
//!   CL    = THETA(1)
//!   V     = THETA(2)
//!   KA    = THETA(3)
//!   ALAG1 = THETA(4)
//!   S2    = V
//! $ERROR
//!   IPRED = F
//!   Y     = IPRED*(1+EPS(1))
//! $THETA  2.0 FIX  20.0 FIX  1.5 FIX  1.5 FIX   ; CL V KA ALAG1
//! $OMEGA 0 FIX
//! $SIGMA 0.01 FIX
//! $ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
//! $TABLE ID TIME EVID PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
//! ```
//!
//! over a single subject dosed `SS=1, II=24, AMT=100` into the depot with
//! observations at TIME = 0.5, 1, 2, 4, 8, 12, 18, 23, 25, 30. The two
//! earliest samples (TIME < ALAG1 = 1.5) land in the *previous* dosing
//! interval and NONMEM reports the steady-state tail there (~0.59 / ~0.56),
//! not 0 — the case that regressed before issue #15 was fixed.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};

const SS_ORAL_LAGTIME_MODEL: &str = r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(1.5, 0.05, 20.0)
  theta TVLAG(1.5, 0.001, 10.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KA      = TVKA
  LAGTIME = TVLAG

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn predict_matches_nonmem_ss_oral_with_lagtime() {
    let parsed = parse_full_model(SS_ORAL_LAGTIME_MODEL).expect("model parses");
    let model = parsed.model;

    let population = read_nonmem_csv(std::path::Path::new("data/ss_oral_lagtime.csv"), None, None)
        .expect("dataset loads");
    assert!(
        population.subjects.iter().any(|s| s.has_ss_doses()),
        "dataset should contain SS=1 doses"
    );

    let preds = predict(&model, &population, &model.default_params).unwrap();

    // NONMEM PRED keyed by observation time (the DV column of the CSV is the
    // same reference, but we list it here so the assertion is explicit).
    let nonmem: &[(f64, f64)] = &[
        (0.5, 0.59069),
        (1.0, 0.56188),
        (2.0, 3.07370),
        (4.0, 4.46240),
        (8.0, 3.07540),
        (12.0, 2.06170),
        (18.0, 1.13150),
        (23.0, 0.68628),
        (25.0, 0.56188),
        (30.0, 0.34080),
    ];
    assert_eq!(preds.len(), nonmem.len());

    for (p, &(t, expected)) in preds.iter().zip(nonmem) {
        assert!(
            (p.time - t).abs() < 1e-9,
            "prediction time {} != expected {}",
            p.time,
            t
        );
        let rel = (p.pred - expected).abs() / expected;
        assert!(
            rel < 1e-4,
            "t={}: ferx PRED {:.5} vs NONMEM {:.5} (rel err {:.2e})",
            t,
            p.pred,
            expected,
            rel
        );
    }
}
