//! End-to-end NONMEM cross-check for **bioavailability in an ODE model**
//! (issue #122).
//!
//! NONMEM applies bioavailability `F1` when the dose enters the compartment:
//! the depot is loaded with `F1·AMT`, and the absorption flux is the plain
//! `KA·depot`. Before issue #122, ferx's ODE engine added the *full* dose to
//! the depot and required the user to bake `F` into the absorption flux — a
//! convention that disagreed with NONMEM and with ferx's own analytical path.
//!
//! This test asserts that, after the fix, a plain-flux ODE oral model with an
//! `F` individual parameter reproduces NONMEM's `PRED` (population, η = 0) to
//! 1e-4 relative, and that the ODE and analytical (`one_cpt_oral(…, f=F)`)
//! forms of the same model agree.
//!
//! ## Reproducing the NONMEM reference
//!
//! The `PRED` values below were produced with NONMEM 7.5.1,
//! `$ESTIMATION MAXEVAL=0` (pure evaluation at the fixed thetas), from this
//! control file (CL=5, V=50, KA=1.5, F1=0.70; single subject dosed
//! `AMT=100, CMT=1` at t=0, observations on CMT=2):
//!
//! ```text
//! $PROBLEM 1-cpt oral with bioavailability F1 reference for ferx-core issue #122
//! $DATA data.csv IGNORE=@
//! $INPUT ID TIME DV EVID AMT CMT RATE MDV
//! $SUBROUTINES ADVAN2 TRANS2
//! $PK
//!   CL    = THETA(1)
//!   V     = THETA(2)
//!   KA    = THETA(3)
//!   F1    = THETA(4)
//!   S2    = V
//! $ERROR
//!   IPRED = F
//!   Y     = IPRED*(1+EPS(1))
//! $THETA  5.0 FIX  50.0 FIX  1.5 FIX  0.70 FIX   ; CL V KA F1
//! $OMEGA 0 FIX
//! $SIGMA 0.01 FIX
//! $ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
//! $TABLE ID TIME EVID PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
//! ```

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::path::Path;

/// ODE form: depot loaded with `F·AMT` by the engine; plain `KA·depot` flux.
/// `F = THETA_F` directly (η = 0, fixed) so PRED matches NONMEM's F1 = 0.70.
const ODE_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1,  50.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.5, 0.05,  20.0)
  theta THETA_F(0.70, 0.001, 0.999)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = THETA_F

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// Analytical form of the identical model: bioavailability via `f=F`.
const ANALYTICAL_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1,  50.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.5, 0.05,  20.0)
  theta THETA_F(0.70, 0.001, 0.999)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = THETA_F

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// NONMEM PRED keyed by observation time (sdtab1).
const NONMEM: &[(f64, f64)] = &[
    (0.5, 0.71829),
    (1.0, 1.0226),
    (2.0, 1.1534),
    (4.0, 1.0018),
    (8.0, 0.67398),
    (12.0, 0.45179),
    (18.0, 0.24795),
    (24.0, 0.13608),
];

#[test]
fn ode_bioavailability_matches_nonmem() {
    let model = parse_full_model(ODE_MODEL).expect("ODE model parses").model;
    let population = read_nonmem_csv(Path::new("data/bioavailability_ode_ref.csv"), None, None)
        .expect("dataset loads");

    let preds = predict(&model, &population, &model.default_params).unwrap();
    assert_eq!(preds.len(), NONMEM.len());

    for (p, &(t, expected)) in preds.iter().zip(NONMEM) {
        assert!(
            (p.time - t).abs() < 1e-9,
            "prediction time {} != expected {}",
            p.time,
            t
        );
        let rel = (p.pred - expected).abs() / expected;
        assert!(
            rel < 1e-4,
            "t={}: ferx ODE PRED {:.5} vs NONMEM {:.5} (rel err {:.2e})",
            t,
            p.pred,
            expected,
            rel
        );
    }
}

#[test]
fn ode_bioavailability_matches_analytical() {
    // The ODE and analytical forms of the same oral model with F = 0.70 must
    // agree — both now apply F at dose entry (NONMEM convention).
    let ode = parse_full_model(ODE_MODEL).expect("ODE model parses").model;
    let analytical = parse_full_model(ANALYTICAL_MODEL)
        .expect("analytical model parses")
        .model;
    let population = read_nonmem_csv(Path::new("data/bioavailability_ode_ref.csv"), None, None)
        .expect("dataset loads");

    let ode_preds = predict(&ode, &population, &ode.default_params).unwrap();
    let an_preds = predict(&analytical, &population, &analytical.default_params).unwrap();
    assert_eq!(ode_preds.len(), an_preds.len());

    for (o, a) in ode_preds.iter().zip(an_preds.iter()) {
        let rel = (o.pred - a.pred).abs() / a.pred.abs().max(1e-12);
        assert!(
            rel < 1e-4,
            "t={}: ODE PRED {:.6} vs analytical PRED {:.6} (rel err {:.2e})",
            o.time,
            o.pred,
            a.pred,
            rel
        );
    }
}
